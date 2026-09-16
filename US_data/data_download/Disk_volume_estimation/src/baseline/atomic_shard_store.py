"""Generic atomic shard + receipt store for sharded scientific products.

RD1-C4-D1 section J. A Stage-1 v2 review product is produced one shard at a
time, usually one shard per trial, usually on Moriah, usually under an array
task that can die at any instant. Every such product needs the same four
properties, and none of them are specific to what the shard contains:

1. **Atomic publication.** A shard is either completely present or entirely
   absent. There is no state in which a reducer can see half of one.
2. **A receipt written last.** A shard directory without a valid, matching
   receipt is an incomplete attempt, not a result, and is invisible to
   reduction.
3. **Exact identity.** A receipt pins every input identity the producer was
   given, so a later run can prove that an existing shard was produced from
   exactly the same inputs before reusing it.
4. **Non-destructive resume.** An existing completed shard is never
   overwritten and never silently recomputed; an identity conflict is
   reported, field by field, and refused.

This module implements exactly those four properties and nothing else. It
has no opinion about what a shard contains, so the same helpers serve the
RD1-C4-D1 observation diagnostic today and are the intended mechanism for
the later formal RD1-C4 executor's shard families:

``nse_kge/``
    Per-trial frozen-support NSE/KGE metric shards.
``objective_receipts/``
    Per-trial exact official-objective reproduction receipts.
``observation_provenance/``
    Per-trial package-versus-pickle observation evidence (what D1 writes).
``q98/``
    Per-trial Q98 high-flow configuration diagnostics.

**Extension point.** A new family needs no change here. A producer:

* picks a ``family`` string (the directory name above);
* builds an identity mapping of every input fact that determines the shard's
  content (see :meth:`AtomicShardStore.begin_attempt`);
* writes its own files into the attempt directory the store hands it;
* calls :meth:`AtomicShardStore.publish`.

The store hashes every component, publishes atomically, and writes the
receipt last. A consumer calls :meth:`AtomicShardStore.completed_receipt`
and, if it needs byte-level proof, :meth:`AtomicShardStore.verify`.

**Deliberate non-goals.** This module never deletes anything, never prunes
stale attempt directories, and never publishes scientific interpretation.
Abandoned attempt directories are left on disk for inspection: an operator
removes them, not this code.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = [
    "ShardStoreError",
    "SHARD_RECEIPT_SCHEMA_NAME",
    "SHARD_RECEIPT_SCHEMA_VERSION",
    "ShardComponent",
    "ShardReceipt",
    "AtomicShardStore",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "atomic_write_bytes",
    "sha256_path",
]


class ShardStoreError(RuntimeError):
    """Raised for a shard-store protocol violation: publishing over an
    existing completed shard, publishing an empty attempt, reusing a shard
    whose identity disagrees, or reading a shard whose bytes no longer match
    its receipt."""


SHARD_RECEIPT_SCHEMA_NAME = "flashnh_atomic_shard_receipt"
SHARD_RECEIPT_SCHEMA_VERSION = 1

_SHARDS_DIR = "shards"
_RECEIPTS_DIR = "receipts"
_ATTEMPTS_DIR = "_attempts"


def canonical_json_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 JSON bytes: sorted keys, no insignificant
    whitespace, ASCII-escaped, and NaN/Infinity refused (they are not JSON
    and would silently produce an unparseable receipt)."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_path(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_dir(path: Path) -> None:
    """Flush a directory entry to stable storage where the platform supports
    it. Windows has no directory fd to fsync, so this is a documented no-op
    there rather than a silent failure."""
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path, payload: bytes) -> str:
    """Write ``payload`` to ``path`` atomically, returning its SHA-256.

    Writes to a sibling temporary file, flushes and fsyncs it, then
    ``os.replace``s it into place, so a reader never observes a partially
    written file even if the process dies mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ShardComponent:
    """One published file inside a shard, addressed by its path relative to
    the shard root (always POSIX-separated, so a receipt written on Linux
    and read on Windows compares equal)."""

    relative_path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @staticmethod
    def from_dict(payload: Mapping) -> "ShardComponent":
        return ShardComponent(
            relative_path=str(payload["relative_path"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
        )


@dataclass(frozen=True)
class ShardReceipt:
    """Proof that one shard is complete, plus the identity it was produced
    from.

    ``content_sha256`` is the hash of the deterministic component table, so
    two shards with identical file contents at identical relative paths have
    identical ``content_sha256`` regardless of when or where they were
    written. ``identity_sha256`` is the hash of the producer-supplied
    identity mapping, so identity agreement is a single string comparison
    before any field-by-field diff is needed.
    """

    schema_name: str
    schema_version: int
    family: str
    shard_id: str
    created_at_utc: str
    identity: Mapping[str, Any]
    identity_sha256: str
    components: Sequence[ShardComponent]
    content_sha256: str

    def as_dict(self) -> dict:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "family": self.family,
            "shard_id": self.shard_id,
            "created_at_utc": self.created_at_utc,
            "identity": dict(self.identity),
            "identity_sha256": self.identity_sha256,
            "components": [component.as_dict() for component in self.components],
            "content_sha256": self.content_sha256,
        }

    @staticmethod
    def from_dict(payload: Mapping) -> "ShardReceipt":
        return ShardReceipt(
            schema_name=str(payload["schema_name"]),
            schema_version=int(payload["schema_version"]),
            family=str(payload["family"]),
            shard_id=str(payload["shard_id"]),
            created_at_utc=str(payload["created_at_utc"]),
            identity=dict(payload["identity"]),
            identity_sha256=str(payload["identity_sha256"]),
            components=tuple(ShardComponent.from_dict(item) for item in payload["components"]),
            content_sha256=str(payload["content_sha256"]),
        )

    def component_by_path(self, relative_path: str) -> Optional[ShardComponent]:
        for component in self.components:
            if component.relative_path == relative_path:
                return component
        return None


def _identity_mismatches(expected: Mapping, actual: Mapping) -> list:
    """Every disagreeing identity field, reported -- never just the first."""
    mismatches = []
    for key in sorted(set(expected) | set(actual)):
        if key not in expected:
            mismatches.append({"field": key, "expected": None, "actual": actual[key], "reason": "unexpected"})
        elif key not in actual:
            mismatches.append({"field": key, "expected": expected[key], "actual": None, "reason": "missing"})
        elif expected[key] != actual[key]:
            mismatches.append(
                {"field": key, "expected": expected[key], "actual": actual[key], "reason": "differs"}
            )
    return mismatches


class AtomicShardStore:
    """Family-scoped atomic shard store rooted at ``root/<family>/``.

    Layout::

        <root>/<family>/shards/<shard_id>/...        published shard content
        <root>/<family>/receipts/<shard_id>.json     receipt, written last
        <root>/<family>/_attempts/<shard_id>__<tok>/ in-progress attempt

    A shard is "completed" iff its receipt exists, parses, names this family
    and shard id, and every component it lists is present. The shard
    directory alone never means completed.
    """

    def __init__(self, root, family: str) -> None:
        if not family or "/" in family or "\\" in family or family.startswith("."):
            raise ShardStoreError(f"invalid shard family name: {family!r}")
        self.root = Path(root)
        self.family = family

    # -- paths ------------------------------------------------------------ #

    @property
    def family_dir(self) -> Path:
        return self.root / self.family

    def shard_dir(self, shard_id: str) -> Path:
        return self.family_dir / _SHARDS_DIR / self._safe_id(shard_id)

    def receipt_path(self, shard_id: str) -> Path:
        return self.family_dir / _RECEIPTS_DIR / f"{self._safe_id(shard_id)}.json"

    def attempts_dir(self) -> Path:
        return self.family_dir / _ATTEMPTS_DIR

    @staticmethod
    def _safe_id(shard_id: str) -> str:
        if not shard_id or "/" in shard_id or "\\" in shard_id or shard_id in {".", ".."}:
            raise ShardStoreError(f"invalid shard id: {shard_id!r}")
        return shard_id

    # -- attempts --------------------------------------------------------- #

    def begin_attempt(self, shard_id: str, *, attempt_token: str) -> Path:
        """Create and return a unique, attempt-specific working directory.

        ``attempt_token`` must make the directory unique to this attempt (a
        Slurm job/array id, or a UUID locally) so two concurrent attempts at
        the same shard can never write into each other's files. Refuses if
        the shard is already completed -- a completed shard is never
        recomputed by accident.
        """
        if self.completed_receipt(shard_id) is not None:
            raise ShardStoreError(
                f"{self.family}/{shard_id}: already completed -- refusing to start a new attempt "
                "(reuse it, or publish under a different shard id)"
            )
        attempt = self.attempts_dir() / f"{self._safe_id(shard_id)}__{attempt_token}"
        if attempt.exists():
            raise ShardStoreError(
                f"attempt directory already exists: {attempt} -- attempt tokens must be unique per attempt"
            )
        attempt.mkdir(parents=True)
        return attempt

    # -- publication ------------------------------------------------------ #

    def publish(
        self,
        shard_id: str,
        *,
        attempt_dir,
        identity: Mapping[str, Any],
        created_at_utc: Optional[str] = None,
    ) -> ShardReceipt:
        """Atomically publish an attempt directory as a completed shard.

        Order of operations (each step must complete before the next):

        1. every file under ``attempt_dir`` is flushed to stable storage and
           hashed, producing the deterministic component table;
        2. the attempt directory is moved into place with a single
           ``os.replace`` -- the shard becomes visible in one step;
        3. the receipt is written atomically, last.

        A crash before step 2 leaves only an attempt directory; a crash
        between 2 and 3 leaves a shard with no receipt. Both are incomplete
        and invisible to reduction, and neither is ever repaired silently.

        Refuses to publish over an existing shard directory or receipt.
        """
        attempt_dir = Path(attempt_dir)
        if not attempt_dir.is_dir():
            raise ShardStoreError(f"attempt directory does not exist: {attempt_dir}")
        shard_dir = self.shard_dir(shard_id)
        receipt_path = self.receipt_path(shard_id)
        if receipt_path.exists():
            raise ShardStoreError(
                f"{self.family}/{shard_id}: a receipt already exists at {receipt_path} -- refusing to "
                "overwrite a completed shard"
            )
        if shard_dir.exists():
            raise ShardStoreError(
                f"{self.family}/{shard_id}: shard directory already exists at {shard_dir} (without a receipt) "
                "-- refusing to overwrite; inspect and remove it explicitly if it is a dead attempt"
            )

        components = self._hash_attempt(attempt_dir)
        if not components:
            raise ShardStoreError(f"{self.family}/{shard_id}: refusing to publish an empty shard")

        identity = dict(identity)
        content_sha256 = canonical_json_sha256([component.as_dict() for component in components])
        receipt = ShardReceipt(
            schema_name=SHARD_RECEIPT_SCHEMA_NAME,
            schema_version=SHARD_RECEIPT_SCHEMA_VERSION,
            family=self.family,
            shard_id=shard_id,
            created_at_utc=created_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            identity=identity,
            identity_sha256=canonical_json_sha256(identity),
            components=components,
            content_sha256=content_sha256,
        )

        shard_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(attempt_dir, shard_dir)
        _fsync_dir(shard_dir.parent)
        atomic_write_bytes(receipt_path, canonical_json_bytes(receipt.as_dict()))
        return receipt

    def _hash_attempt(self, attempt_dir: Path) -> tuple:
        components = []
        for path in sorted(p for p in attempt_dir.rglob("*") if p.is_file()):
            relative = path.relative_to(attempt_dir).as_posix()
            components.append(
                ShardComponent(
                    relative_path=relative,
                    sha256=sha256_path(path),
                    size_bytes=path.stat().st_size,
                )
            )
        components.sort(key=lambda component: component.relative_path)
        return tuple(components)

    # -- reading ---------------------------------------------------------- #

    def read_receipt(self, shard_id: str) -> Optional[ShardReceipt]:
        """Parse this shard's receipt, or ``None`` if there is none.

        A receipt that exists but is unparseable, or that names a different
        family/shard, is a protocol violation and raises -- it is never
        treated as "no receipt".
        """
        path = self.receipt_path(shard_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            receipt = ShardReceipt.from_dict(payload)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ShardStoreError(f"{path}: unreadable shard receipt: {exc}") from exc
        if receipt.schema_name != SHARD_RECEIPT_SCHEMA_NAME:
            raise ShardStoreError(f"{path}: unexpected receipt schema {receipt.schema_name!r}")
        if receipt.schema_version != SHARD_RECEIPT_SCHEMA_VERSION:
            raise ShardStoreError(
                f"{path}: receipt schema version {receipt.schema_version} != {SHARD_RECEIPT_SCHEMA_VERSION}"
            )
        if receipt.family != self.family or receipt.shard_id != shard_id:
            raise ShardStoreError(
                f"{path}: receipt is for {receipt.family}/{receipt.shard_id}, not {self.family}/{shard_id}"
            )
        return receipt

    def completed_receipt(self, shard_id: str) -> Optional[ShardReceipt]:
        """The receipt of a completed shard, or ``None``.

        Completeness requires that every component the receipt names is
        present at the expected size. Content hashes are NOT re-read here
        (that is :meth:`verify`), so this stays cheap enough to call for
        every shard at resume time.
        """
        receipt = self.read_receipt(shard_id)
        if receipt is None:
            return None
        shard_dir = self.shard_dir(shard_id)
        for component in receipt.components:
            path = shard_dir / component.relative_path
            if not path.is_file() or path.stat().st_size != component.size_bytes:
                raise ShardStoreError(
                    f"{self.family}/{shard_id}: receipt names component {component.relative_path!r} that is "
                    "missing or truncated -- the shard is damaged, not merely incomplete; refusing to treat "
                    "it as absent"
                )
        return receipt

    def verify(self, shard_id: str) -> ShardReceipt:
        """Re-hash every component and prove the shard still matches its
        receipt. Raises :class:`ShardStoreError` on any disagreement."""
        receipt = self.completed_receipt(shard_id)
        if receipt is None:
            raise ShardStoreError(f"{self.family}/{shard_id}: no completed shard to verify")
        shard_dir = self.shard_dir(shard_id)
        for component in receipt.components:
            actual = sha256_path(shard_dir / component.relative_path)
            if actual != component.sha256:
                raise ShardStoreError(
                    f"{self.family}/{shard_id}: component {component.relative_path!r} has sha256 {actual} but "
                    f"the receipt records {component.sha256}"
                )
        recomputed = canonical_json_sha256([component.as_dict() for component in receipt.components])
        if recomputed != receipt.content_sha256:
            raise ShardStoreError(
                f"{self.family}/{shard_id}: recomputed content_sha256 {recomputed} != receipt "
                f"{receipt.content_sha256}"
            )
        return receipt

    def list_completed(self) -> list:
        """Every completed shard id in this family, sorted. Attempt
        directories and receipt-less shard directories are not listed."""
        receipts_dir = self.family_dir / _RECEIPTS_DIR
        if not receipts_dir.is_dir():
            return []
        shard_ids = sorted(path.stem for path in receipts_dir.glob("*.json"))
        return [shard_id for shard_id in shard_ids if self.completed_receipt(shard_id) is not None]

    # -- resume ----------------------------------------------------------- #

    def reuse_if_identical(self, shard_id: str, identity: Mapping[str, Any]) -> tuple:
        """Decide whether an existing completed shard may be reused.

        Returns ``(receipt_or_None, mismatches)``:

        * no completed shard -> ``(None, [])``: the caller should compute it;
        * completed with an identical identity -> ``(receipt, [])``: reuse;
        * completed with a different identity -> raises
          :class:`ShardStoreError` naming EVERY disagreeing field.

        A conflicting shard is never overwritten, never deleted, and never
        silently recomputed: a shard id that means one thing cannot quietly
        start meaning another.
        """
        receipt = self.completed_receipt(shard_id)
        if receipt is None:
            return None, []
        identity = dict(identity)
        if receipt.identity_sha256 == canonical_json_sha256(identity):
            return receipt, []
        mismatches = _identity_mismatches(identity, dict(receipt.identity))
        raise ShardStoreError(
            f"{self.family}/{shard_id}: an existing completed shard was produced from a DIFFERENT identity -- "
            f"refusing to reuse, overwrite, or recompute it. Disagreeing fields: "
            f"{json.dumps(mismatches, sort_keys=True, default=str)}"
        )


def copy_into_attempt(attempt_dir, source, relative_path: str) -> Path:
    """Copy an already-written file into an attempt directory under a chosen
    relative path (convenience for producers that build a large artifact
    elsewhere before publication)."""
    destination = Path(attempt_dir) / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def iter_families(root) -> Iterable[str]:
    """Family names present under ``root`` -- the extension point's
    directory listing, used by reducers and operators, never by producers."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())
