#!/usr/bin/env python
"""Per-proposal driver for the RD1 P20-P24 fixed Slurm chain.

Replaces the retired self-propagating controller
(``.scratch_local/rd1_continuation_chain_20260925T123812Z/controller/
rd1_continuation_controller.py`` on Moriah). The structural fix is "no job
submits another job": this driver runs once, inside one already-submitted
Slurm job, and never calls ``sbatch``. The fixed ``afterok`` dependency
chain across P20-P24 is created once, up front, by
``scripts/submit_rd1_p20_p24_chain.sh`` from a clean login shell.

This module is split into pure, host-independent logic (importable and
tested without Slurm, W&B, or a POSIX filesystem) and a thin ``main()`` CLI
that performs the actual subprocess/W&B/registry I/O. The pure functions
carry every behavior this replacement was specifically required to get
right (contiguity/hard-stop gating, socket-failure retry eligibility
wiring); the CLI wiring is intentionally thin.

Correction (Phase A.1 review, 2026-09-27): ``run_agent_with_retry`` below was
present and fully tested but never wired to any CLI subcommand -- the actual
sbatch script ran its own, separate, weaker bash retry loop instead (it
never captured a "before" run-ids snapshot, so it could not detect that a
retried attempt might have already created a run). This file now adds the
one ``run-agent-with-retry`` subcommand that is the actual execution path's
only retry-eligibility implementation, delegating entirely to the
already-tested ``run_agent_with_retry``/``is_socket_failure_retry_eligible``
functions below. ``rd1_p20_p24_job.sbatch`` calls this subcommand instead of
running its own loop.

Reuses rather than reimplements:

* ``scripts/run_sweep_v2_six_axis_wandb_agent_moriah.sbatch`` for manifest
  validation, WANDB_PROJECT/ENTITY resolution, ``.netrc``-based
  ``WANDB_API_KEY`` resolution, and the single ``wandb agent --count 1``
  invocation (invoked here as a plain subprocess, never via ``sbatch``);
* ``scripts/create_sweep_v2_six_axis_wandb_bridge_production_sweep.py
  build-manifest`` for building each of P21-P24's own manifest, after its
  predecessor has already passed (enforced by the Slurm ``afterok``
  dependency itself -- this driver never runs early);
* :mod:`src.baseline.rd1_p20_p24_registry` for the lock/append/verify
  contract;
* :mod:`src.baseline.rd1_p20_p24_env` for environment sanitization;
* :mod:`src.baseline.rd1_p20_p24_retry` for bounded backoff and the
  socket-failure retry-eligibility gate.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline.rd1_p20_p24_env import sanitize_environment  # noqa: E402
from src.baseline.rd1_p20_p24_registry import (  # noqa: E402
    RegistryError,
    RegistryLock,
    append_row_verified,
    read_registry,
    registry_orders,
)
from src.baseline.rd1_p20_p24_retry import (  # noqa: E402
    RetryAuditError,
    fetch_with_backoff,
    is_socket_failure_retry_eligible,
    no_new_run_appeared,
)

__all__ = [
    "FIRST_PROPOSAL_ORDER",
    "HARD_STOP_ORDER",
    "PreflightError",
    "ManifestIdentityError",
    "AgentAttemptResult",
    "compute_expected_prior_orders",
    "check_preflight",
    "verify_pinned_manifest_checksum",
    "run_agent_with_retry",
    "main",
]

FIRST_PROPOSAL_ORDER = 1
HARD_STOP_ORDER = 24


class PreflightError(Exception):
    """Any reason a proposal's job must refuse to proceed before touching
    W&B or the registry. Carries a machine-readable reason code."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class ManifestIdentityError(Exception):
    """P20's manifest failed the byte-for-byte checksum-pinned reuse check."""


def compute_expected_prior_orders(order: int) -> list[int]:
    """The full contiguous order set that must already be registered
    before ``order`` may run: every order from :data:`FIRST_PROPOSAL_ORDER`
    up to (not including) ``order``."""
    return list(range(FIRST_PROPOSAL_ORDER, order))


def check_preflight(
    order: int,
    current_orders: Iterable[int],
    *,
    hard_stop_order: int = HARD_STOP_ORDER,
) -> None:
    """Raise :class:`PreflightError` unless ``order`` is exactly the next
    proposal the registry is ready for. Refuses (in order): any order past
    the hard stop, a non-positive order, an order already registered, and
    an order whose full prior contiguous history is not exactly present
    (catches both "too early" and any drifted/ambiguous registry state).
    Never mutates anything -- callers hold a shared/exclusive
    :class:`~src.baseline.rd1_p20_p24_registry.RegistryLock` around the
    read that produced ``current_orders``.
    """
    if order > hard_stop_order:
        raise PreflightError("hard_stop_exceeded", f"order {order} > hard stop {hard_stop_order}")
    if order < FIRST_PROPOSAL_ORDER:
        raise PreflightError("invalid_order", str(order))

    current = sorted(int(o) for o in current_orders)
    if order in current:
        raise PreflightError("duplicate_order_already_registered", str(order))

    expected_prior = compute_expected_prior_orders(order)
    if current != expected_prior:
        raise PreflightError(
            "registry_not_contiguous_for_order",
            f"current={current} expected_prior={expected_prior}",
        )


def verify_pinned_manifest_checksum(manifest_path: "str | Path", expected_sha256: str) -> None:
    """P20 reuses its already-validated manifest byte-for-byte (design
    requirement: no rebuild). Raise :class:`ManifestIdentityError` if the
    file's sha256 does not exactly match ``expected_sha256``; never
    silently proceeds with a manifest that merely looks similar."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ManifestIdentityError(f"pinned manifest not found: {manifest_path}")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ManifestIdentityError(
            f"pinned manifest checksum mismatch: got {digest} expected {expected_sha256}"
        )


@dataclasses.dataclass(frozen=True)
class AgentAttemptResult:
    success: bool
    returncode: int
    output_text: str
    run_id: "str | None"


def run_agent_with_retry(
    *,
    attempt_fn: "Callable[[], AgentAttemptResult]",
    audit_run_ids_fn: "Callable[[], Iterable[str]]",
    max_attempts: int = 2,
) -> AgentAttemptResult:
    """Run ``attempt_fn`` (one ``wandb agent --count 1`` invocation) and, on
    failure, retry at most once -- and only for the exact pre-run W&B
    service-socket failure signature, only after a bounded read-only W&B
    audit independently confirms the failed attempt created no run. Any
    other failure is returned as-is (fail-closed: no retry).

    ``audit_run_ids_fn`` is a read-only sweep-runs listing call, wrapped by
    the caller-supplied function itself in nothing extra -- backoff is
    applied here via :func:`~src.baseline.rd1_p20_p24_retry.fetch_with_backoff`.
    If every backoff attempt raises, :class:`~src.baseline.rd1_p20_p24_retry.
    RetryAuditError` propagates to the caller uncaught: an unavailable or
    ambiguous audit must fail closed (no retry decision made at all), never
    be silently treated as "no new run".
    """
    run_ids_before = list(fetch_with_backoff(audit_run_ids_fn))
    attempt_index = 0
    result: "AgentAttemptResult | None" = None
    while True:
        attempt_index += 1
        result = attempt_fn()
        if result.success or attempt_index >= max_attempts:
            return result

        run_ids_after = list(fetch_with_backoff(audit_run_ids_fn))
        no_new = no_new_run_appeared(run_ids_before, run_ids_after)
        eligible, _reason = is_socket_failure_retry_eligible(
            attempt_index=attempt_index,
            max_attempts=max_attempts,
            attempt_output_text=result.output_text,
            run_id_resolved=result.run_id,
            no_new_run_confirmed=no_new,
        )
        if not eligible:
            return result
        run_ids_before = run_ids_after


def _subprocess_attempt_fn(agent_launcher_path: Path, env: dict) -> AgentAttemptResult:
    """One ``bash <agent_launcher_path>`` subprocess attempt, run with
    exactly the caller-supplied ``env`` (no inheritance from this process'
    own environment -- callers pass a narrow, explicit allowlist, mirroring
    the ``env -i`` pattern previously used directly in the sbatch script).

    Never parses a run id out of launcher stdout/stderr: run-id resolution
    is a separate, explicit read-only W&B audit performed by the CLI wiring
    (see ``_wandb_resolve_latest_run_id``), not text-scraped from the
    launcher's own output.
    """
    proc = subprocess.run(
        ["bash", str(agent_launcher_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output_text = (proc.stdout or "") + (proc.stderr or "")
    return AgentAttemptResult(
        success=(proc.returncode == 0),
        returncode=proc.returncode,
        output_text=output_text,
        run_id=None,
    )


def _wandb_sweep_run_ids(sweep_id: str) -> list[str]:
    """Read-only listing of every run id currently in ``sweep_id``. Imports
    ``wandb`` lazily so importing this module for the pure-function unit
    tests never requires the ``wandb`` package to be installed."""
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    return sorted(run.id for run in sweep.runs)


def _wandb_resolve_latest_run_id(sweep_id: str) -> str:
    """Read-only resolution of the most-recently-created run in
    ``sweep_id``, called exactly once after ``run_agent_with_retry`` reports
    overall success. Raises ``RuntimeError`` if the sweep has no runs at
    all (a successful agent exit with no resolvable run is itself a fatal,
    non-retryable condition -- the caller must not append a registry row
    without a real run id)."""
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    runs = sorted(sweep.runs, key=lambda run: run.created_at)
    if not runs:
        raise RuntimeError("no runs found in sweep after a successful agent exit")
    return runs[-1].id


def _cmd_preflight(args: argparse.Namespace) -> int:
    with RegistryLock(args.lock_path, exclusive=False):
        registry = read_registry(args.registry_path)
    current_orders = registry_orders(registry)
    try:
        check_preflight(args.order, current_orders, hard_stop_order=args.hard_stop_order)
    except PreflightError as exc:
        print(f"PREFLIGHT_REFUSED reason={exc.reason} detail={exc.detail}", file=sys.stderr)
        return 1
    print(f"PREFLIGHT_OK order={args.order} expected_prior_orders={current_orders}")
    return 0


def _cmd_append_registry_row(args: argparse.Namespace) -> int:
    row = {
        "order": args.order,
        "wandb_run_id": args.run_id,
        "recorded_at_utc": args.recorded_at_utc,
    }
    with RegistryLock(args.lock_path, exclusive=True):
        try:
            result = append_row_verified(
                args.registry_path,
                row,
                expected_prior_orders=compute_expected_prior_orders(args.order),
                backup_dir=args.backup_dir,
            )
        except RegistryError as exc:
            print(f"APPEND_REFUSED reason={exc.reason} detail={exc.detail}", file=sys.stderr)
            return 1
    print(f"APPEND_OK order={args.order} backup={result['registry_backup_path']}")
    return 0


def _cmd_verify_pinned_manifest(args: argparse.Namespace) -> int:
    try:
        verify_pinned_manifest_checksum(args.manifest_path, args.expected_sha256)
    except ManifestIdentityError as exc:
        print(f"MANIFEST_IDENTITY_REFUSED {exc}", file=sys.stderr)
        return 1
    print(f"MANIFEST_IDENTITY_OK {args.manifest_path}")
    return 0


def _cmd_sanitize_env_probe(args: argparse.Namespace) -> int:
    import os

    cleaned = sanitize_environment(os.environ)
    leaked = sorted(name for name in cleaned if name == "WANDB" or name.startswith("WANDB_"))
    if leaked:
        print(f"SANITIZE_FAILED leaked={leaked}", file=sys.stderr)
        return 1
    print("SANITIZE_OK no WANDB_* variable present after sanitization")
    return 0


def _cmd_run_agent_with_retry(args: argparse.Namespace) -> int:
    """The one live retry-eligibility execution path. Builds the exact
    narrow env allowlist the agent launcher runs under, then delegates the
    entire attempt/retry decision to the tested ``run_agent_with_retry`` /
    ``is_socket_failure_retry_eligible`` functions above -- this function
    adds no additional retry logic of its own.

    On overall success, resolves the created run id via one more read-only
    W&B call and prints it for the sbatch wrapper to pass to
    ``append-registry-row``. Never appends to the registry itself (kept as
    a separate, existing subcommand/step).
    """
    attempt_env = {
        "PATH": args.path,
        "HOME": args.home,
        "FLASHNH_SWEEP_V2_PRODUCTION_MANIFEST": args.manifest_path,
        "WANDB_SWEEP_ID": args.wandb_sweep_id,
    }
    agent_launcher_path = Path(args.agent_launcher_path)

    def attempt_fn() -> AgentAttemptResult:
        return _subprocess_attempt_fn(agent_launcher_path, dict(attempt_env))

    def audit_run_ids_fn() -> list[str]:
        return _wandb_sweep_run_ids(args.wandb_sweep_id)

    try:
        result = run_agent_with_retry(
            attempt_fn=attempt_fn,
            audit_run_ids_fn=audit_run_ids_fn,
            max_attempts=args.max_attempts,
        )
    except RetryAuditError as exc:
        print(f"AGENT_FAILED reason=retry_audit_unavailable detail={exc}", file=sys.stderr)
        return 1

    if not result.success:
        print(
            f"AGENT_FAILED reason=agent_attempts_exhausted returncode={result.returncode}",
            file=sys.stderr,
        )
        return 1

    try:
        run_id = _wandb_resolve_latest_run_id(args.wandb_sweep_id)
    except Exception as exc:  # noqa: BLE001 -- any resolution failure is fatal, never silently skipped
        print(f"AGENT_FAILED reason=run_id_resolution_failed detail={exc}", file=sys.stderr)
        return 1

    print(f"AGENT_OK run_id={run_id}")
    return 0


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_preflight = sub.add_parser("preflight", help="Refuse unless the registry is exactly ready for --order.")
    p_preflight.add_argument("--order", type=int, required=True)
    p_preflight.add_argument("--registry-path", required=True)
    p_preflight.add_argument("--lock-path", required=True)
    p_preflight.add_argument("--hard-stop-order", type=int, default=HARD_STOP_ORDER)
    p_preflight.set_defaults(func=_cmd_preflight)

    p_append = sub.add_parser("append-registry-row", help="Verified append of one completed proposal's row.")
    p_append.add_argument("--order", type=int, required=True)
    p_append.add_argument("--run-id", required=True)
    p_append.add_argument("--recorded-at-utc", required=True)
    p_append.add_argument("--registry-path", required=True)
    p_append.add_argument("--lock-path", required=True)
    p_append.add_argument("--backup-dir", required=True)
    p_append.set_defaults(func=_cmd_append_registry_row)

    p_manifest = sub.add_parser("verify-pinned-manifest", help="P20-only: checksum-pin the reused manifest.")
    p_manifest.add_argument("--manifest-path", required=True)
    p_manifest.add_argument("--expected-sha256", required=True)
    p_manifest.set_defaults(func=_cmd_verify_pinned_manifest)

    p_sanitize = sub.add_parser("sanitize-env-probe", help="Fail if any WANDB_* var survives sanitization.")
    p_sanitize.set_defaults(func=_cmd_sanitize_env_probe)

    p_run_agent = sub.add_parser(
        "run-agent-with-retry",
        help="The one live retry-eligibility execution path: run the agent launcher, "
        "retrying at most once under the tested socket-failure eligibility gate.",
    )
    p_run_agent.add_argument("--agent-launcher-path", required=True)
    p_run_agent.add_argument("--manifest-path", required=True)
    p_run_agent.add_argument("--wandb-sweep-id", required=True)
    p_run_agent.add_argument("--path", required=True, help="PATH value for the narrow attempt-env allowlist.")
    p_run_agent.add_argument("--home", required=True, help="HOME value for the narrow attempt-env allowlist.")
    p_run_agent.add_argument("--max-attempts", type=int, default=2)
    p_run_agent.set_defaults(func=_cmd_run_agent_with_retry)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
