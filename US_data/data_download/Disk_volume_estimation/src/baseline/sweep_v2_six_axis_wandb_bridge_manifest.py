"""Strict non-secret manifest for the single-agent v2 fresh-proposal bridge."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .sweep_v1_launch_manifest import MODE_PRODUCTION, MODE_REHEARSAL, _reject_credential_shaped_fields
from .sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2, CONFIGURATION_CANONICALIZATION_VERSION_V2, DOMAIN_VERSION_V2, FORBIDDEN_V1_SWEEP_ID, OBJECTIVE_ID_V2

MANIFEST_SCHEMA_VERSION = 1
_REQUIRED = frozenset({"manifest_label", "created_at_utc", "mode", "expected_commit", "repository_root", "expected_runtime_python", "wandb_project", "wandb_sweep_id", "output_root", "package_root", "screening_basin_ids_path", "screening_basin_ids_sha256", "fixed_support_contract_path", "fixed_support_contract_version", "fixed_support_contract_sha256", "baseline_policy_path", "policy_overlay_path", "base_pilot_policy_path", "proposal_order", "execution_generation", "stop_before_training", "max_agents", "campaign_id", "domain_version", "canonicalization_version", "objective_id", "manifest_sha256"})
_OPTIONAL = frozenset({"wandb_entity"})
_ALLOWED = _REQUIRED | _OPTIONAL | {"schema_version"}

class SweepV2BridgeManifestError(ValueError):
    pass

def _canonical_bytes(data: Mapping[str, Any]) -> bytes:
    return json.dumps({k: v for k, v in data.items() if k != "manifest_sha256"}, sort_keys=True, separators=(",", ":")).encode()

def compute_manifest_checksum(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(data)).hexdigest()

def _validate(data: Mapping[str, Any]) -> None:
    try:
        # The v1 helper intentionally exempts only the checksum field names
        # known to its own schema.  These two v2 artifact checksums are
        # identities, not credentials, and are separately shape-checked.
        _reject_credential_shaped_fields({k: v for k, v in data.items() if k not in {"screening_basin_ids_sha256", "fixed_support_contract_sha256"}})
    except Exception as exc:
        raise SweepV2BridgeManifestError(str(exc)) from exc
    unknown, missing = set(data) - _ALLOWED, _REQUIRED - set(data)
    if unknown or missing:
        raise SweepV2BridgeManifestError(f"unknown={sorted(unknown)} missing={sorted(missing)}")
    if data.get("schema_version", MANIFEST_SCHEMA_VERSION) != MANIFEST_SCHEMA_VERSION:
        raise SweepV2BridgeManifestError("unsupported schema_version")
    if data["campaign_id"] != CAMPAIGN_ID_V2 or data["domain_version"] != DOMAIN_VERSION_V2 or data["canonicalization_version"] != CONFIGURATION_CANONICALIZATION_VERSION_V2 or data["objective_id"] != OBJECTIVE_ID_V2:
        raise SweepV2BridgeManifestError("v2 campaign/domain/canonicalization/objective identity mismatch")
    if data["wandb_sweep_id"] == FORBIDDEN_V1_SWEEP_ID:
        raise SweepV2BridgeManifestError("v1 production sweep is forbidden")
    if data["mode"] not in (MODE_PRODUCTION, MODE_REHEARSAL):
        raise SweepV2BridgeManifestError("mode must be production or rehearsal")
    if data["mode"] == MODE_REHEARSAL and data["stop_before_training"] is not True:
        raise SweepV2BridgeManifestError("rehearsal requires stop_before_training=True")
    if data["mode"] == MODE_PRODUCTION and data["stop_before_training"] is not False:
        raise SweepV2BridgeManifestError("production requires stop_before_training=False")
    if data["max_agents"] != 1:
        raise SweepV2BridgeManifestError("v2 bridge authorizes exactly one agent")
    if not isinstance(data["proposal_order"], int) or isinstance(data["proposal_order"], bool) or data["proposal_order"] < 1:
        raise SweepV2BridgeManifestError("proposal_order must be positive integer")
    if not isinstance(data["execution_generation"], int) or isinstance(data["execution_generation"], bool) or data["execution_generation"] < 1:
        raise SweepV2BridgeManifestError("execution_generation must be positive integer")
    for field in ("screening_basin_ids_sha256", "fixed_support_contract_sha256"):
        value = data[field]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise SweepV2BridgeManifestError(f"{field} must be a SHA-256")

def build_v2_wandb_bridge_manifest(**fields: Any) -> dict[str, Any]:
    data = {**fields, "schema_version": fields.get("schema_version", MANIFEST_SCHEMA_VERSION), "wandb_entity": fields.get("wandb_entity")}
    if "manifest_sha256" in fields:
        raise SweepV2BridgeManifestError("manifest_sha256 is computed")
    data["manifest_sha256"] = "pending"
    _validate(data)
    data["manifest_sha256"] = compute_manifest_checksum(data)
    return data

def write_v2_wandb_bridge_manifest(path: "str | Path", **fields: Any) -> dict[str, Any]:
    """Write the manifest via a genuinely atomic no-clobber publication: the
    complete payload is written to a same-directory temp file, then
    published with ``os.link`` (never ``os.replace``), which fails with
    ``FileExistsError`` -- without touching the destination -- if a
    concurrent writer created ``path`` first. The temp file is always
    removed, on both success and failure, and the destination's bytes are
    never altered by a losing writer."""
    path = Path(path)
    if path.exists(): raise SweepV2BridgeManifestError("refusing manifest overwrite")
    data = build_v2_wandb_bridge_manifest(**fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle: handle.write(json.dumps(data, sort_keys=True, indent=2) + "\n")
        try:
            os.link(tmp, path)
        except FileExistsError as exc:
            raise SweepV2BridgeManifestError(
                f"refusing manifest overwrite: destination appeared during publication: {path}"
            ) from exc
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return data

def load_v2_wandb_bridge_manifest(path: "str | Path") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file(): raise SweepV2BridgeManifestError("manifest not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("manifest_sha256") != compute_manifest_checksum(data):
        raise SweepV2BridgeManifestError("manifest checksum mismatch")
    _validate(data)
    return data
