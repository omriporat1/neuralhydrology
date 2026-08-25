"""Strict, versioned, checksummed launch manifest for Sweep-v1 exact-retry launches.

Replaces the long ``sbatch --export=ALL,VAR=value,...`` interface with one
project-local JSON file passed as a single positional argument to the shared
launcher scripts. Production and disposable rehearsal launches share this
exact loader/schema (see docs/decision_log.md, the accepted Sweep-v1
exact-retry startup rehearsal design).

Never put credentials in a launch manifest. Authentication continues to come
from the standard on-host credential store (``~/.netrc``); this module
actively refuses field names or bare hex-token-shaped values that look
credential-shaped, as defense in depth, but that check is not a substitute
for keeping secrets out in the first place.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

MANIFEST_SCHEMA_VERSION = 1

# The real, frozen Sweep-v1 production W&B sweep identity
# (stage1_phase_b_sweep_v1_original_domain_v001). Not a secret: this sweep id
# already appears throughout this repository's committed tests and docs.
PRODUCTION_WANDB_SWEEP_ID = "4x3btz2s"

MODE_PRODUCTION = "production"
MODE_REHEARSAL = "rehearsal"
_VALID_MODES = (MODE_PRODUCTION, MODE_REHEARSAL)

_REQUIRED_FIELDS = (
    "manifest_label",
    "created_at_utc",
    "mode",
    "expected_commit",
    "expected_runtime_python",
    "frozen_proposal_record_path",
    "expected_identity",
    "execution_generation",
    "package_root",
    "screening_basin_ids_path",
    "output_root",
    "baseline_policy_path",
    "base_pilot_policy_path",
    "wandb_project",
    "wandb_sweep_id",
    "stop_before_training",
)
_OPTIONAL_FIELDS = ("wandb_entity", "prior_attempts_path", "retry_of_trial_id_expected")
_COMPUTED_FIELDS = ("schema_version", "manifest_sha256")
_ALLOWED_FIELDS = frozenset(_REQUIRED_FIELDS) | frozenset(_OPTIONAL_FIELDS) | frozenset(_COMPUTED_FIELDS)

# Field names exempted from the bare-hex-token defense-in-depth scan below
# because they are legitimately expected to hold long hex strings that are
# NOT secrets (a git commit, or the manifest's own checksum).
_HEX_EXEMPT_FIELDS = frozenset({"expected_commit", "manifest_sha256"})

_CREDENTIAL_KEY_MARKERS = ("key", "token", "secret", "password", "credential", "netrc", "auth")


class LaunchManifestError(ValueError):
    """A launch manifest failed schema, checksum, or safety validation."""


def _looks_like_bare_hex_token(value: str) -> bool:
    if len(value) not in (32, 40, 64):
        return False
    return all(c in "0123456789abcdefABCDEF" for c in value)


def _reject_credential_shaped_fields(fields: Mapping[str, Any]) -> None:
    for key, value in fields.items():
        lowered = key.lower()
        if any(marker in lowered for marker in _CREDENTIAL_KEY_MARKERS):
            raise LaunchManifestError(
                f"launch manifest field name looks credential-shaped, refusing to accept it: {key!r}"
            )
        if key in _HEX_EXEMPT_FIELDS:
            continue
        if isinstance(value, str) and _looks_like_bare_hex_token(value):
            raise LaunchManifestError(
                f"launch manifest field {key!r} holds a bare hex-token-shaped value; "
                "this looks like an accidentally-embedded secret, refusing"
            )


def _validate_schema(fields: Mapping[str, Any]) -> None:
    # Credential-shaped field names are rejected before anything else -- even
    # before the unknown-field check -- so a caller gets the specific,
    # security-relevant reason rather than a generic "unknown field" message.
    _reject_credential_shaped_fields(fields)

    unknown = set(fields) - _ALLOWED_FIELDS
    if unknown:
        raise LaunchManifestError(f"launch manifest contains unknown field(s): {sorted(unknown)}")

    missing = [key for key in _REQUIRED_FIELDS if fields.get(key) is None]
    if missing:
        raise LaunchManifestError(f"launch manifest is missing required field(s): {missing}")

    schema_version = fields.get("schema_version", MANIFEST_SCHEMA_VERSION)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise LaunchManifestError(f"unsupported launch manifest schema_version: {schema_version!r}")

    mode = fields["mode"]
    if mode not in _VALID_MODES:
        raise LaunchManifestError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}")

    execution_generation = fields["execution_generation"]
    if isinstance(execution_generation, bool) or not isinstance(execution_generation, int):
        raise LaunchManifestError("execution_generation must be a plain integer")

    if not isinstance(fields["stop_before_training"], bool):
        raise LaunchManifestError("stop_before_training must be a boolean")

    if not isinstance(fields["expected_identity"], dict):
        raise LaunchManifestError("expected_identity must be a JSON object")

    sweep_id = fields["wandb_sweep_id"]
    if mode == MODE_PRODUCTION:
        if sweep_id != PRODUCTION_WANDB_SWEEP_ID:
            raise LaunchManifestError(
                f"mode={MODE_PRODUCTION!r} requires wandb_sweep_id == {PRODUCTION_WANDB_SWEEP_ID!r}, got {sweep_id!r}"
            )
        if fields["stop_before_training"] is not False:
            raise LaunchManifestError(f"mode={MODE_PRODUCTION!r} requires stop_before_training=False")
    else:
        if sweep_id == PRODUCTION_WANDB_SWEEP_ID:
            raise LaunchManifestError(
                f"mode={MODE_REHEARSAL!r} must never target the production sweep "
                f"({PRODUCTION_WANDB_SWEEP_ID!r}); refusing"
            )
        if fields["stop_before_training"] is not True:
            raise LaunchManifestError(f"mode={MODE_REHEARSAL!r} requires stop_before_training=True")


def _canonical_bytes(fields: Mapping[str, Any]) -> bytes:
    payload = {key: value for key, value in fields.items() if key != "manifest_sha256"}
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def compute_manifest_checksum(fields: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(fields)).hexdigest()


def build_launch_manifest(**fields: Any) -> dict[str, Any]:
    """Validate and canonically checksum a manifest without writing it."""
    fields = dict(fields)
    fields.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
    for key in _OPTIONAL_FIELDS:
        fields.setdefault(key, None)
    if "manifest_sha256" in fields:
        raise LaunchManifestError("manifest_sha256 is computed, do not supply it")
    _validate_schema(fields)
    checksum = compute_manifest_checksum(fields)
    return {**fields, "manifest_sha256": checksum}


def write_launch_manifest(path: "str | Path", **fields: Any) -> dict[str, Any]:
    """Write a new launch manifest. Strict no-overwrite: refuses if ``path`` exists."""
    path = Path(path)
    if path.exists():
        raise LaunchManifestError(f"REFUSING to overwrite an existing launch manifest: {path}")

    manifest = build_launch_manifest(**fields)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True))
            handle.write("\n")
        if path.exists():
            raise LaunchManifestError(f"REFUSING to overwrite an existing launch manifest: {path}")
        os.replace(tmp_name, str(path))
        tmp_name = None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.remove(tmp_name)
    return manifest


def load_launch_manifest(path: "str | Path") -> dict[str, Any]:
    """Load and fully validate a launch manifest: checksum, schema, and safety."""
    path = Path(path)
    if not path.is_file():
        raise LaunchManifestError(f"launch manifest not found: {path}")

    try:
        fields = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LaunchManifestError(f"launch manifest is not valid JSON: {path} ({exc})") from exc

    if not isinstance(fields, dict):
        raise LaunchManifestError(f"launch manifest must be a JSON object: {path}")

    stored_checksum = fields.get("manifest_sha256")
    if not stored_checksum:
        raise LaunchManifestError(f"launch manifest is missing manifest_sha256: {path}")

    recomputed = compute_manifest_checksum(fields)
    if recomputed != stored_checksum:
        raise LaunchManifestError(
            f"launch manifest checksum mismatch (tampered or corrupted?): {path} "
            f"stored={stored_checksum!r} recomputed={recomputed!r}"
        )

    _validate_schema(fields)
    return fields
