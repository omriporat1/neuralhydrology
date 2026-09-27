"""Registry lock/read/append primitives for the RD1 P20-P24 fixed chain.

Extracted from the retired self-propagating RD1 continuation controller
(``.scratch_local/rd1_continuation_chain_20260925T123812Z/controller/
rd1_continuation_controller.py`` on Moriah, never a tracked/maintained file)
into a small, tracked, independently tested module. The registry
lock/contiguity/backup/verified-append contract that module implemented was
not itself the cause of the P20 failure -- the failure was architectural
(a Slurm job submitting further Slurm jobs after contacting W&B, inheriting a
stale local wandb-core service handle) -- so that contract is preserved
here essentially unchanged rather than redesigned.

Every write is: lock -> read -> verify expected prior contiguous state ->
refuse a duplicate append for an already-present order -> backup -> atomic
replace -> reread -> verify every prior row is byte-identical and the new
row is exactly as intended -> release. Any violation raises
:class:`RegistryError` and leaves the on-disk registry untouched (the backup
copy is written before the replace, never after).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

try:
    import fcntl  # POSIX-only; the RD1 chain runs exclusively on Moriah/Catfish Linux nodes.
except ImportError:  # pragma: no cover -- exercised only on Windows dev machines
    fcntl = None

__all__ = [
    "RegistryError",
    "RegistryLock",
    "read_registry",
    "registry_orders",
    "backup_registry",
    "append_row_verified",
]

DEFAULT_LOCK_TIMEOUT_SEC = 60


class RegistryError(Exception):
    """Any registry-contract violation. Carries a machine-readable reason
    code as the first argument so callers can branch on it without string
    parsing."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RegistryLock:
    """Exclusive (or shared, for a read-only preflight) ``flock`` on a
    dedicated lock file next to the registry -- never the registry payload
    file itself, so a lock holder crashing never leaves the registry
    mid-write with a stale lock on the payload.

    POSIX-only (``fcntl``); the RD1 chain runs exclusively on Moriah/Catfish
    Linux nodes, so this is not a portability gap for this module's actual
    use.
    """

    def __init__(self, lock_path: "str | Path", *, exclusive: bool, timeout_sec: float = DEFAULT_LOCK_TIMEOUT_SEC):
        self._lock_path = Path(lock_path)
        self._exclusive = exclusive
        self._timeout_sec = timeout_sec
        self._fd = None

    def __enter__(self) -> "RegistryLock":
        import signal

        if fcntl is None:
            raise RegistryError("fcntl_unavailable", "RegistryLock requires a POSIX platform (Moriah/Catfish)")

        self._lock_path.touch(exist_ok=True)
        self._fd = open(self._lock_path, "a+")
        mode = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH

        def _on_alarm(signum, frame):
            raise TimeoutError("registry lock timeout")

        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(self._timeout_sec))
        try:
            fcntl.flock(self._fd, mode)
        except TimeoutError:
            self._fd.close()
            self._fd = None
            raise RegistryError("registry_lock_timeout", f"{self._timeout_sec}s")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None
        return False


def read_registry(registry_path: "str | Path") -> dict:
    with open(registry_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def registry_orders(registry: Mapping) -> list[int]:
    return sorted(int(r["order"]) for r in registry.get("runs", []))


def backup_registry(registry_path: "str | Path", backup_dir: "str | Path") -> Path:
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"{Path(registry_path).name}.backup_{_now_utc().replace(':', '')}.json"
    shutil.copy2(registry_path, dest)
    return dest


def append_row_verified(
    registry_path: "str | Path",
    row: Mapping,
    *,
    expected_prior_orders: "list[int]",
    backup_dir: "str | Path",
) -> dict:
    """Append exactly one new row under the caller's own exclusive
    :class:`RegistryLock`, or raise :class:`RegistryError` and leave the
    on-disk registry byte-identical to before the call.

    Preconditions enforced (in order), each a distinct fail-closed reason:

    1. ``row["order"]`` must not already be present in the on-disk registry
       (refuses a duplicate append outright, before touching the file).
    2. The on-disk registry's orders must equal ``expected_prior_orders``
       exactly (contiguity/identity precondition -- catches a registry that
       drifted since the caller's own preflight read).
    3. After the atomic replace, a fresh read must show every prior row
       byte-identical to before, the new row present and exactly equal to
       ``row``, and the full order set contiguous
       ``expected_prior_orders + [row["order"]]``.

    Does not itself acquire the lock -- callers hold a :class:`RegistryLock`
    (exclusive) around the whole read-check-append-verify sequence, matching
    the retired controller's convention and this project's registry-lock
    preservation requirement.
    """
    registry_path = Path(registry_path)
    order = int(row["order"])

    registry = read_registry(registry_path)
    current_orders = registry_orders(registry)

    if order in current_orders:
        raise RegistryError("duplicate_append_refused", f"order {order} already present")

    if current_orders != list(expected_prior_orders):
        raise RegistryError(
            "registry_not_at_expected_prior_state",
            f"orders={current_orders} expected_prior={list(expected_prior_orders)}",
        )

    backup_path = backup_registry(registry_path, backup_dir)

    pre_rows = {r["order"]: r for r in registry["runs"]}
    registry["runs"].append(dict(row))
    tmp_path = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2)
    os.replace(tmp_path, registry_path)

    reread = read_registry(registry_path)
    re_orders = registry_orders(reread)
    expected_final = list(expected_prior_orders) + [order]
    if re_orders != expected_final:
        raise RegistryError("post_append_orders_not_contiguous", str(re_orders))
    for prior_order, prior_row in pre_rows.items():
        match = next((r for r in reread["runs"] if r["order"] == prior_order), None)
        if match != prior_row:
            raise RegistryError("post_append_prior_row_mutated", str(prior_order))
    new_row = next((r for r in reread["runs"] if r["order"] == order), None)
    if new_row != dict(row):
        raise RegistryError("post_append_new_row_mismatch", str(new_row))

    return {"registry_backup_path": str(backup_path), "orders_after": re_orders}
