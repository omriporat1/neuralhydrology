"""Prepare one frozen Sweep-v1 Bayesian proposal without W&B or training.

The public core accepts only the five scientific coordinates plus bounded
proposal provenance.  It deliberately delegates all candidate mathematics and
the configuration hash to the committed Sweep-v1 and pilot helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import sweep_v1_campaign as sweep
from .nh_config_generation import GeneratedConfigBundle, write_generated_config
from .pilot_lead06_config import (
    SCREENING_VALIDATION_POPULATION_ROLE,
    build_pilot_bundle_with_validation_scope,
    load_screening_basin_ids,
)
from .nh_config_generation import read_package_manifest, validate_full_population_basin_membership

__all__ = [
    "SweepV1PreparationError", "PreparationPaths", "PreparedSweepV1Proposal",
    "canonicalize_wandb_proposal", "prepare_bayesian_proposal", "prepare_random_control_row",
    "write_prepared_proposal",
]


class SweepV1PreparationError(ValueError):
    """Raised when a proposed candidate could drift outside the frozen wave."""


@dataclass(frozen=True)
class PreparationPaths:
    baseline_policy_path: Path
    package_root: Path
    splits_dir: Path
    screening_basin_ids_path: Path


@dataclass(frozen=True)
class PreparedSweepV1Proposal:
    proposal: dict[str, Any]
    configuration_id: str
    proposal_id: str
    trial_id: str
    execution_generation: int
    bundle: GeneratedConfigBundle
    evidence: dict[str, Any]


_AXES = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")
_PROVENANCE = {"proposal_order", "execution_generation", "wandb_sweep_id", "wandb_run_id",
               "campaign_id", "domain_version", "search_arm"}
_FORBIDDEN = {"seed", "model_seed", "epochs", "target_epoch", "max_updates_per_epoch", "package_identity",
              "screening_artifact_sha256", "screening_path", "evaluation_scope", "sealed_scope",
              "performance_early_stopping_enabled", "qualification_kind", "launch_contract_qualification"}

# Recovered from the accepted Gate-4 source inventory retained in
# reports/fullpop_moriah_readiness_gate_v001/.../part2_source_inventory and
# independently corroborated by four transferred Moriah rendering manifests.
# The manifest binds package schema, population/order, dynamic/target/static
# contracts, provenance, and every basin NetCDF/gap checksum.  The separately
# pinned checksum table binds attributes, basin list, and all authoritative
# package payload files; the tiny provenance record binds builder/schema state.
PACKAGE_MANIFEST_SHA256 = "6c52fb1b81f6a5f730b805d0c273e9d00cbf5bb93d1cd0da58452f5a0e5bcc4a"
PACKAGE_FILE_CHECKSUMS_SHA256 = "83b47374725d418b130a8e28dcf1cb118cee88f99624907238e25ee2a9067d13"
PACKAGE_RUN_PROVENANCE_SHA256 = "030de2f9458aa40deba74d84910904f02468adb9eb1786ee3a71556bfcb11a8b"
DEVELOPMENT_SPLIT_SHA256 = "397ab432564c18c3abc5158a47ada2b28840bbf6f0c213d2475444fded33858f"
SPATIAL_HOLDOUT_SPLIT_SHA256 = "76d1c546e703b1b5aa8f4a3ead971327de0151dae4fcce0c90b1272da0f587b7"


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise SweepV1PreparationError(f"required frozen artifact is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_artifact_identities(paths: PreparationPaths) -> dict[str, str]:
    """Verify production package and split bytes before any config preparation."""
    package_root = Path(paths.package_root)
    artifacts = {
        "package_manifest_sha256": (package_root / "manifests" / "package_manifest.json", PACKAGE_MANIFEST_SHA256),
        "package_file_checksums_sha256": (package_root / "manifests" / "file_checksums.csv", PACKAGE_FILE_CHECKSUMS_SHA256),
        "package_run_provenance_sha256": (package_root / "run_provenance.json", PACKAGE_RUN_PROVENANCE_SHA256),
        "development_split_sha256": (Path(paths.splits_dir) / "development_train.txt", DEVELOPMENT_SPLIT_SHA256),
        "spatial_holdout_split_sha256": (Path(paths.splits_dir) / "spatial_holdout_nonca.txt", SPATIAL_HOLDOUT_SPLIT_SHA256),
    }
    verified = {}
    for name, (path, expected) in artifacts.items():
        actual = _sha256(path)
        if actual != expected:
            raise SweepV1PreparationError(f"frozen artifact identity mismatch for {name}: {actual} != {expected}")
        verified[name] = actual
    return verified


def canonicalize_wandb_proposal(config: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract a W&B-like mapping at the boundary, without importing W&B."""
    merged = dict(config)
    if metadata:
        overlap = set(merged) & set(metadata)
        if overlap:
            raise SweepV1PreparationError(f"W&B config/metadata duplicate keys: {sorted(overlap)}")
        merged.update(metadata)
    return merged


def _validate_proposal(proposal: Mapping[str, Any], *, expected_arm: str) -> tuple[dict[str, Any], int, int]:
    keys = set(proposal)
    forbidden = keys & _FORBIDDEN
    if forbidden:
        raise SweepV1PreparationError(f"scientific override/qualification metadata forbidden: {sorted(forbidden)}")
    unknown = keys - set(_AXES) - _PROVENANCE
    if unknown:
        raise SweepV1PreparationError(f"unexpected scientific/provenance fields: {sorted(unknown)}")
    missing = set(_AXES) - keys
    if missing:
        raise SweepV1PreparationError(f"missing frozen search axes: {sorted(missing)}")
    if proposal.get("campaign_id", sweep.CAMPAIGN_ID) != sweep.CAMPAIGN_ID or proposal.get("domain_version", sweep.DOMAIN_VERSION) != sweep.DOMAIN_VERSION:
        raise SweepV1PreparationError("proposal campaign/domain does not match frozen original wave")
    if proposal.get("search_arm", expected_arm) != expected_arm:
        raise SweepV1PreparationError(f"preparation expects {expected_arm!r} provenance")
    order = proposal.get("proposal_order")
    generation = proposal.get("execution_generation", 1)
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise SweepV1PreparationError("proposal_order must be a positive integer")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise SweepV1PreparationError("execution_generation must be a positive integer")
    try:
        axes = sweep.canonical_hyperparameters({key: proposal[key] for key in _AXES})
    except ValueError as exc:
        raise SweepV1PreparationError(str(exc)) from exc
    return axes, order, generation


def _audit_generated_config(bundle: GeneratedConfigBundle, axes: Mapping[str, Any]) -> None:
    cfg = bundle.config_mapping
    expected = {
        "learning_rate": float(axes["learning_rate"]), "hidden_size": axes["hidden_size"],
        "output_dropout": float(axes["output_dropout"]), "batch_size": axes["batch_size"],
        "optimizer": "Adam", "seed": sweep.MODEL_SEED_A, "seq_length": 72,
        "epochs": sweep.TARGET_EPOCH, "save_weights_every": 1,
        "max_updates_per_epoch": sweep.MAX_UPDATES_PER_EPOCH,
        "dynamic_inputs": sweep.FROZEN_FIXED_CONFIGURATION["dynamic_inputs"],
        "target_variables": ["qobs_mm_per_h_lead06"],
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise SweepV1PreparationError(f"generated config drift: {key}={cfg.get(key)!r}, expected {value!r}")
    emb = cfg.get("statics_embedding")
    if not isinstance(emb, dict) or emb.get("hiddens") != [128, 32] or emb.get("activation") != "tanh" or emb.get("dropout") != float(axes["embedding_dropout"]):
        raise SweepV1PreparationError("generated config drifted from frozen static-embedding contract")
    if (cfg.get("train_start_date"), cfg.get("train_end_date")) != ("14/10/2020", "31/12/2023"):
        raise SweepV1PreparationError("generated config does not preserve frozen development-training dates")
    if cfg.get("validation_start_date") != "01/01/2024" or cfg.get("validation_end_date") != "31/12/2024":
        raise SweepV1PreparationError("generated config does not preserve 2024 development-validation screening")
    if bundle.population_role != SCREENING_VALIDATION_POPULATION_ROLE or len(bundle.validation_basin_ids or []) != 400:
        raise SweepV1PreparationError("generated bundle does not preserve the pinned screening population")


def _prepare_proposal(*, proposal: Mapping[str, Any], paths: PreparationPaths, expected_arm: str) -> PreparedSweepV1Proposal:
    axes, order, generation = _validate_proposal(proposal, expected_arm=expected_arm)
    artifact_identities = _verify_artifact_identities(paths)
    package_manifest = read_package_manifest(paths.package_root)
    membership = validate_full_population_basin_membership(package_manifest, paths.splits_dir)
    screening = load_screening_basin_ids(paths.screening_basin_ids_path, development_basins=membership.development_basins,
                                         expected_count=400, expected_sha256=sweep.SCREENING_ARTIFACT_SHA256)
    bundle = build_pilot_bundle_with_validation_scope(
        baseline_policy_path=paths.baseline_policy_path, package_root=paths.package_root, splits_dir=paths.splits_dir,
        lead_hours=6, seq_length=72, run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        validation_basin_ids=screening, population_role=SCREENING_VALIDATION_POPULATION_ROLE,
        package_type=sweep.PACKAGE_IDENTITY, max_updates_per_epoch=sweep.MAX_UPDATES_PER_EPOCH,
        learning_rate=float(axes["learning_rate"]), hidden_size=axes["hidden_size"],
        embedding_dropout=float(axes["embedding_dropout"]), output_dropout=float(axes["output_dropout"]),
        batch_size=axes["batch_size"], dynamic_inputs=list(sweep.FROZEN_FIXED_CONFIGURATION["dynamic_inputs"]),
    )
    bundle = replace(bundle, config_mapping={**bundle.config_mapping, "epochs": sweep.TARGET_EPOCH})
    _audit_generated_config(bundle, axes)
    config_id = sweep.configuration_id(axes)
    pid = sweep.proposal_id(expected_arm, order)
    tid = sweep.trial_id(config_id, execution_generation=generation)
    evidence = {
        "prepare_status": "PASS", "prepare_only": True, "objective_score": None,
        "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION, "search_arm": expected_arm,
        "proposal_id": pid, "proposal_order": order, "configuration_id": config_id, "trial_id": tid,
        "execution_generation": generation, "hyperparameters": axes, "model_seed": sweep.MODEL_SEED_A,
        "fidelity_id": "mf12x50000", "target_epoch": sweep.TARGET_EPOCH,
        "max_updates_per_epoch": sweep.MAX_UPDATES_PER_EPOCH, "save_weights_every": 1,
        "authoritative_screening_epochs": list(range(1, 13)), "performance_early_stopping_enabled": False,
        "package_identity": sweep.PACKAGE_IDENTITY, "screening_artifact_sha256": sweep.SCREENING_ARTIFACT_SHA256,
        "artifact_identity_status": "PASS", **artifact_identities,
        "screening_policy_identity": sweep.SCREENING_POLICY_IDENTITY, "evaluation_scope": "development_validation_2024_only",
        "sealed_scope": False, "wandb_sweep_id": proposal.get("wandb_sweep_id"), "wandb_run_id": proposal.get("wandb_run_id"),
    }
    return PreparedSweepV1Proposal(dict(proposal), config_id, pid, tid, generation, bundle, evidence)


def prepare_bayesian_proposal(*, proposal: Mapping[str, Any], paths: PreparationPaths) -> PreparedSweepV1Proposal:
    """Prepare one Bayesian proposal; W&B values are telemetry-only provenance."""
    return _prepare_proposal(proposal=proposal, paths=paths, expected_arm="bayesian")


def prepare_random_control_row(*, row: Mapping[str, Any], manifest_path: Path, paths: PreparationPaths,
                               execution_generation: int = 1) -> PreparedSweepV1Proposal:
    """Prepare one immutable committed random-control row without regenerating it."""
    manifest_path = Path(manifest_path)
    if _sha256(manifest_path) != sweep.RANDOM_CONTROL_MANIFEST_SHA256:
        raise SweepV1PreparationError("random-control manifest SHA-256 does not match frozen bytes")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or dict(row) not in rows:
        raise SweepV1PreparationError("requested random row is not an exact committed manifest row")
    supplied = dict(row)
    supplied["execution_generation"] = execution_generation
    supplied.pop("configuration_id", None)
    supplied.pop("proposal_id", None)
    supplied.pop("manifest_rng_seed", None)
    supplied.pop("manifest_index", None)
    return _prepare_proposal(proposal=supplied, paths=paths, expected_arm="random_control")


_LAYER_B_PROVENANCE_FILENAME = "execution_provenance.json"


def write_prepared_proposal(
    prepared: PreparedSweepV1Proposal, output_dir: Path, *, allow_layer_b_provenance: bool = False
) -> dict[str, Any]:
    """Write the exact NH config and finalize compact prepare-only evidence.

    By default ``output_dir`` must be empty (or absent) -- the same strict
    no-overwrite behavior ``write_generated_config`` has always had. Every
    caller other than the Sweep-v1 W&B bridge should leave
    ``allow_layer_b_provenance`` at its default ``False``.

    ``allow_layer_b_provenance=True`` is the single narrow exception: it
    tolerates ``output_dir`` already containing exactly one pre-existing file,
    ``execution_provenance.json`` -- the durable Layer-B record written by
    ``sweep_v1_execution.write_proposal_intake_provenance`` before this call
    -- and nothing else (``write_generated_config`` itself refuses to ever
    accept one of its own protected generated-target filenames into an
    allowlist, and refuses an allowlisted name that exists but is not a
    regular file, so this can never be abused to recreate the old unsafe
    ``force=True`` overwrite behavior). Before writing anything, if that file
    is present its recorded ``trial_id`` must be present and exactly equal to
    ``prepared.trial_id`` -- unlike ``sweep.trial_identity_conflicts`` (which
    treats a missing/``None`` id on either side as "no conflict", appropriate
    for progressive enrichment), a missing, ``null``, or merely different
    ``trial_id`` here is always a hard failure, since this boundary is
    specifically deciding whether to tolerate a pre-existing file, not
    enriching one. Any failure raises ``SweepV1PreparationError`` and writes
    no generated artifact -- a stale/foreign/malformed provenance file must
    never be silently coexisted with. The provenance file itself is never
    read for any other purpose, never rewritten, deleted, or moved -- any
    other pre-existing entry (including an already-generated ``config.yaml``
    or basin file) remains a hard error raised before any write.
    """
    if allow_layer_b_provenance:
        provenance_path = Path(output_dir) / _LAYER_B_PROVENANCE_FILENAME
        if provenance_path.exists():
            existing_trial_id = json.loads(provenance_path.read_text(encoding="utf-8")).get("trial_id")
            if existing_trial_id != prepared.trial_id:
                raise SweepV1PreparationError(
                    f"existing {_LAYER_B_PROVENANCE_FILENAME} trial_id={existing_trial_id!r} does not "
                    f"exactly match the trial being prepared ({prepared.trial_id!r}); "
                    "refusing to write generated config"
                )
        written = write_generated_config(
            prepared.bundle, output_dir, experiment_name=prepared.trial_id,
            allowed_existing_files=frozenset({_LAYER_B_PROVENANCE_FILENAME}),
        )
    else:
        written = write_generated_config(prepared.bundle, output_dir, experiment_name=prepared.trial_id)
    config_sha = hashlib.sha256(Path(written["config.yaml"]).read_bytes()).hexdigest()
    return {**prepared.evidence, "generated_nh_config_path": str(written["config.yaml"]),
            "generated_nh_config_sha256": config_sha, "generation_manifest_path": str(written["generation_manifest.json"]),
            "expected_output_dir": str(Path(output_dir))}
