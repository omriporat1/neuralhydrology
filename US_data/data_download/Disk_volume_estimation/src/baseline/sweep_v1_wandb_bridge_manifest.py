"""Strict, versioned, checksummed launch manifest for the Sweep-v1
FRESH-PROPOSAL W&B-agent bridge (``scripts/run_sweep_v1_wandb_bridge.py``).

This is the fresh-proposal sibling of
:mod:`src.baseline.sweep_v1_launch_manifest` (built for the exact-retry
bridge). It deliberately reuses that module's ``MODE_PRODUCTION``/
``MODE_REHEARSAL``/``PRODUCTION_WANDB_SWEEP_ID`` constants and its
credential-shape defense-in-depth helpers by direct import -- never a second,
independently-maintained copy of the production sweep id or the
credential-scan logic -- because the two bridges' manifests differ only in
which durable inputs a controller-driven fresh proposal needs (no
``frozen_proposal_record_path``/``expected_identity``, since the five axes
are unknown until a real W&B agent hands them to ``run.config`` at runtime).

Proposal-order ledger
----------------------
Section C of the fresh-bridge qualification task explicitly forbids
hardcoding "proposal_order > 1" as a universal rule: the production sweep's
next legal order and a disposable rehearsal sweep's own order live in
different namespaces and must never be confused. Rather than a separate
mutable ledger file (overkill for a single production sweep with one
proposal consumed so far), the ledger is this module's own small, reviewed,
committed constant below -- update it only via a reviewed commit when a new
production proposal is durably consumed and its outcome recorded (see
docs/decision_log.md). Rehearsal never reads or affects this constant: it is
pinned to its own explicit, reserved, out-of-band order/generation numbers
that can never equal a legal production value or imitate a specific historic
attempt (see ``REHEARSAL_RESERVED_PROPOSAL_ORDER``/
``REHEARSAL_RESERVED_EXECUTION_GENERATION`` below).

Never put credentials in a launch manifest -- see
``sweep_v1_launch_manifest``'s module docstring for the full rationale; the
same defense-in-depth scan is reused here unchanged.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .sweep_v1_launch_manifest import (
    LaunchManifestError,
    MODE_PRODUCTION,
    MODE_REHEARSAL,
    PRODUCTION_WANDB_SWEEP_ID,
    _reject_credential_shaped_fields,
)

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "MODE_PRODUCTION",
    "MODE_REHEARSAL",
    "PRODUCTION_WANDB_SWEEP_ID",
    "PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER",
    "REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR",
    "REHEARSAL_RESERVED_PROPOSAL_ORDER",
    "REHEARSAL_RESERVED_EXECUTION_GENERATION",
    "WandbBridgeManifestError",
    "compute_manifest_checksum",
    "build_wandb_bridge_manifest",
    "write_wandb_bridge_manifest",
    "load_wandb_bridge_manifest",
]

MANIFEST_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Proposal-order ledger (see module docstring). Proposal 1 (order=1) is
# closed and countable via attempt005/ardib08c (objective 0.391678449944578).
# No order=2 proposal has been consumed. Update this constant -- and only
# this constant -- by a reviewed commit once a new production proposal is
# durably recorded.
# ---------------------------------------------------------------------------
PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER = 2

# Any real production proposal_order will always be a small positive integer
# assigned sequentially by the campaign ledger above. Reserving a namespace
# far above any plausible campaign size (36 total planned proposals) makes a
# rehearsal identity structurally incapable of colliding with, or being
# mistaken for, any real production proposal_order/execution_generation --
# including the two specifically forbidden identities in this task
# (proposal_order=2 / execution_generation=6, i.e. "attempt006").
REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR = 900_000
REHEARSAL_RESERVED_PROPOSAL_ORDER = 900_001
REHEARSAL_RESERVED_EXECUTION_GENERATION = 900_001

_REQUIRED_FIELDS = (
    "manifest_label",
    "created_at_utc",
    "mode",
    "expected_commit",
    "expected_runtime_python",
    "package_root",
    "screening_basin_ids_path",
    "output_root",
    "baseline_policy_path",
    "base_pilot_policy_path",
    "wandb_project",
    "wandb_sweep_id",
    "proposal_order",
    "execution_generation",
    "stop_before_training",
)
_OPTIONAL_FIELDS = ("wandb_entity",)
_COMPUTED_FIELDS = ("schema_version", "manifest_sha256")
_ALLOWED_FIELDS = frozenset(_REQUIRED_FIELDS) | frozenset(_OPTIONAL_FIELDS) | frozenset(_COMPUTED_FIELDS)

_VALID_MODES = (MODE_PRODUCTION, MODE_REHEARSAL)


class WandbBridgeManifestError(ValueError):
    """A fresh-proposal-bridge launch manifest failed schema, checksum, or
    safety validation."""


def _validate_schema(fields: Mapping[str, Any]) -> None:
    try:
        _reject_credential_shaped_fields(fields)
    except LaunchManifestError as exc:
        # Reused unchanged from sweep_v1_launch_manifest (never a second,
        # independently-maintained credential scan) -- but this module's own
        # public API always raises WandbBridgeManifestError, never its
        # sibling's exception type.
        raise WandbBridgeManifestError(str(exc)) from exc

    unknown = set(fields) - _ALLOWED_FIELDS
    if unknown:
        raise WandbBridgeManifestError(f"launch manifest contains unknown field(s): {sorted(unknown)}")

    missing = [key for key in _REQUIRED_FIELDS if fields.get(key) is None]
    if missing:
        raise WandbBridgeManifestError(f"launch manifest is missing required field(s): {missing}")

    schema_version = fields.get("schema_version", MANIFEST_SCHEMA_VERSION)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise WandbBridgeManifestError(f"unsupported launch manifest schema_version: {schema_version!r}")

    mode = fields["mode"]
    if mode not in _VALID_MODES:
        raise WandbBridgeManifestError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}")

    if not isinstance(fields["stop_before_training"], bool):
        raise WandbBridgeManifestError("stop_before_training must be a boolean")

    proposal_order = fields["proposal_order"]
    if isinstance(proposal_order, bool) or not isinstance(proposal_order, int) or proposal_order <= 0:
        raise WandbBridgeManifestError("proposal_order must be a positive plain integer")

    execution_generation = fields["execution_generation"]
    if (isinstance(execution_generation, bool) or not isinstance(execution_generation, int)
            or execution_generation <= 0):
        raise WandbBridgeManifestError("execution_generation must be a positive plain integer")

    sweep_id = fields["wandb_sweep_id"]
    if mode == MODE_PRODUCTION:
        if sweep_id != PRODUCTION_WANDB_SWEEP_ID:
            raise WandbBridgeManifestError(
                f"mode={MODE_PRODUCTION!r} requires wandb_sweep_id == {PRODUCTION_WANDB_SWEEP_ID!r}, "
                f"got {sweep_id!r}"
            )
        if fields["stop_before_training"] is not False:
            raise WandbBridgeManifestError(f"mode={MODE_PRODUCTION!r} requires stop_before_training=False")
        if proposal_order != PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER:
            raise WandbBridgeManifestError(
                f"mode={MODE_PRODUCTION!r} requires proposal_order == "
                f"{PRODUCTION_NEXT_PERMISSIBLE_PROPOSAL_ORDER!r} (the campaign ledger's next permissible "
                f"production proposal order), got {proposal_order!r}. A disposable rehearsal's own order "
                "must never be used to infer or override this value."
            )
        if execution_generation != 1:
            raise WandbBridgeManifestError(
                f"mode={MODE_PRODUCTION!r} requires execution_generation == 1 (a fresh controller-assigned "
                "proposal is always a first attempt; retrying an already-attempted proposal belongs to the "
                "exact-retry bridge, not this one)"
            )
    else:
        if sweep_id == PRODUCTION_WANDB_SWEEP_ID:
            raise WandbBridgeManifestError(
                f"mode={MODE_REHEARSAL!r} must never target the production sweep "
                f"({PRODUCTION_WANDB_SWEEP_ID!r}); refusing"
            )
        if fields["stop_before_training"] is not True:
            raise WandbBridgeManifestError(f"mode={MODE_REHEARSAL!r} requires stop_before_training=True")
        if proposal_order < REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR:
            raise WandbBridgeManifestError(
                f"mode={MODE_REHEARSAL!r} requires proposal_order >= "
                f"{REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR!r} (the explicit rehearsal namespace); "
                f"got {proposal_order!r}. A rehearsal identity must never be able to equal or imitate a real "
                "production proposal order."
            )
        if execution_generation < REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR:
            raise WandbBridgeManifestError(
                f"mode={MODE_REHEARSAL!r} requires execution_generation >= "
                f"{REHEARSAL_PROPOSAL_ORDER_NAMESPACE_FLOOR!r} (the explicit rehearsal namespace); "
                f"got {execution_generation!r}. A rehearsal identity must never be able to equal or imitate "
                "a real production attempt generation (e.g. attempt006)."
            )


def _canonical_bytes(fields: Mapping[str, Any]) -> bytes:
    payload = {key: value for key, value in fields.items() if key != "manifest_sha256"}
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def compute_manifest_checksum(fields: Mapping[str, Any]) -> str:
    import hashlib
    return hashlib.sha256(_canonical_bytes(fields)).hexdigest()


def build_wandb_bridge_manifest(**fields: Any) -> dict[str, Any]:
    """Validate and canonically checksum a fresh-proposal-bridge manifest
    without writing it."""
    fields = dict(fields)
    fields.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    for key in _OPTIONAL_FIELDS:
        fields.setdefault(key, None)
    if "manifest_sha256" in fields:
        raise WandbBridgeManifestError("manifest_sha256 is computed, do not supply it")
    _validate_schema(fields)
    checksum = compute_manifest_checksum(fields)
    return {**fields, "manifest_sha256": checksum}


def write_wandb_bridge_manifest(path: "str | Path", **fields: Any) -> dict[str, Any]:
    """Write a new fresh-proposal-bridge launch manifest. Strict
    no-overwrite: refuses if ``path`` exists."""
    path = Path(path)
    if path.exists():
        raise WandbBridgeManifestError(f"REFUSING to overwrite an existing launch manifest: {path}")

    manifest = build_wandb_bridge_manifest(**fields)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True))
            handle.write("\n")
        if path.exists():
            raise WandbBridgeManifestError(f"REFUSING to overwrite an existing launch manifest: {path}")
        os.replace(tmp_name, str(path))
        tmp_name = None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.remove(tmp_name)
    return manifest


def load_wandb_bridge_manifest(path: "str | Path") -> dict[str, Any]:
    """Load and fully validate a fresh-proposal-bridge launch manifest:
    checksum, schema, and safety."""
    path = Path(path)
    if not path.is_file():
        raise WandbBridgeManifestError(f"launch manifest not found: {path}")

    try:
        fields = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WandbBridgeManifestError(f"launch manifest is not valid JSON: {path} ({exc})") from exc

    if not isinstance(fields, dict):
        raise WandbBridgeManifestError(f"launch manifest must be a JSON object: {path}")

    stored_checksum = fields.get("manifest_sha256")
    if not stored_checksum:
        raise WandbBridgeManifestError(f"launch manifest is missing manifest_sha256: {path}")

    recomputed = compute_manifest_checksum(fields)
    if recomputed != stored_checksum:
        raise WandbBridgeManifestError(
            f"launch manifest checksum mismatch (tampered or corrupted?): {path} "
            f"stored={stored_checksum!r} recomputed={recomputed!r}"
        )

    _validate_schema(fields)
    return fields
