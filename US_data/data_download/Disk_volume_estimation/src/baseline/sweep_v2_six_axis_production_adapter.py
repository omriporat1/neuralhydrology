"""Prepare one frozen Sweep-v2 six-axis Bayesian proposal without W&B or
training (Section D, additive six-axis campaign foundation).

Strictly additive sibling of :mod:`sweep_v1_production_adapter`: it imports
and reuses that module's package/split artifact-identity verification and
screening-population loading helpers unmodified (package and split identity
are literally v1's frozen contract, unchanged for v2), and it never edits
``sweep_v1_production_adapter.py``. The only scientific difference from v1's
adapter is the sixth axis (``seq_length``, threaded through the existing
``seq_length`` parameter of ``build_pilot_bundle_with_validation_scope``
rather than hardcoded to ``72``), the v2 policy overlay (legalizing
48-120h in 12h steps), and binding of the frozen fixed-support contract
identity into the v2 configuration identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from . import sweep_v1_campaign as sweep
from .fixed_support_contract_v2 import load_fixed_support_contract
from .nh_config_generation import (
    GeneratedConfigBundle,
    read_package_manifest,
    validate_full_population_basin_membership,
    write_generated_config,
)
from .pilot_lead06_config import (
    SCREENING_VALIDATION_POPULATION_ROLE,
    build_pilot_bundle_with_validation_scope,
    load_screening_basin_ids,
)
from .policy_v2_six_axis import load_stage1_baseline_policy_v2_six_axis
from . import sweep_v2_six_axis_random_control as rc
from .sweep_v1_production_adapter import (
    PreparationPaths as PreparationPathsV1,
    _sha256,
    _verify_artifact_identities,
)
from .sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2,
    CONFIGURATION_CANONICALIZATION_VERSION_V2,
    DOMAIN_VERSION_V2,
    FIDELITY_ID_V2,
    FROZEN_FIXED_CONFIGURATION_V2,
    OBJECTIVE_ID_V2,
    _AXES_V2,
    canonical_hyperparameters_v2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
    validate_v2_proposal_shape,
)

__all__ = [
    "SweepV2PreparationError",
    "PreparationPathsV2",
    "PreparedSweepV2Proposal",
    "prepare_bayesian_proposal_v2",
    "prepare_random_control_proposal_v2",
    "write_prepared_proposal_v2",
]


class SweepV2PreparationError(ValueError):
    """Raised when a v2 six-axis proposal or its generated config drift
    outside the frozen v2 six-axis contract."""


@dataclass(frozen=True)
class PreparationPathsV2:
    baseline_policy_path: Path
    policy_overlay_path: Path
    package_root: Path
    splits_dir: Path
    screening_basin_ids_path: Path
    fixed_support_contract_path: Path


@dataclass(frozen=True)
class PreparedSweepV2Proposal:
    proposal: dict[str, Any]
    configuration_id: str
    proposal_id: str
    trial_id: str
    execution_generation: int
    bundle: GeneratedConfigBundle
    evidence: dict[str, Any]


def _validate_provenance_ints(proposal: Mapping[str, Any]) -> tuple[int, int]:
    order = proposal.get("proposal_order")
    generation = proposal.get("execution_generation", 1)
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise SweepV2PreparationError("proposal_order must be a positive integer")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation < 1:
        raise SweepV2PreparationError("execution_generation must be a positive integer")
    return order, generation


def _audit_generated_config_v2(bundle: GeneratedConfigBundle, axes: Mapping[str, Any]) -> None:
    cfg = bundle.config_mapping
    expected = {
        "learning_rate": float(axes["learning_rate"]), "hidden_size": axes["hidden_size"],
        "output_dropout": float(axes["output_dropout"]), "batch_size": axes["batch_size"],
        "optimizer": "Adam", "seed": sweep.MODEL_SEED_A, "seq_length": axes["seq_length"],
        "epochs": sweep.TARGET_EPOCH, "save_weights_every": 1,
        "max_updates_per_epoch": sweep.MAX_UPDATES_PER_EPOCH,
        "dynamic_inputs": FROZEN_FIXED_CONFIGURATION_V2["dynamic_inputs"],
        "target_variables": ["qobs_mm_per_h_lead06"],
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise SweepV2PreparationError(f"generated config drift: {key}={cfg.get(key)!r}, expected {value!r}")
    emb = cfg.get("statics_embedding")
    if not isinstance(emb, dict) or emb.get("hiddens") != [128, 32] or emb.get("activation") != "tanh" or emb.get("dropout") != float(axes["embedding_dropout"]):
        raise SweepV2PreparationError("generated config drifted from frozen static-embedding contract")
    if (cfg.get("train_start_date"), cfg.get("train_end_date")) != ("14/10/2020", "31/12/2023"):
        raise SweepV2PreparationError("generated config does not preserve frozen development-training dates")
    if cfg.get("validation_start_date") != "01/01/2024" or cfg.get("validation_end_date") != "31/12/2024":
        raise SweepV2PreparationError("generated config does not preserve 2024 development-validation screening")
    if bundle.population_role != SCREENING_VALIDATION_POPULATION_ROLE or len(bundle.validation_basin_ids or []) != 400:
        raise SweepV2PreparationError("generated bundle does not preserve the pinned screening population")


def _prepare_proposal_v2(*, proposal: Mapping[str, Any], paths: PreparationPathsV2, expected_arm: str) -> PreparedSweepV2Proposal:
    validate_v2_proposal_shape(proposal, expected_arm=expected_arm)
    order, generation = _validate_provenance_ints(proposal)
    axes = canonical_hyperparameters_v2({key: proposal[key] for key in _AXES_V2})

    v1_paths = PreparationPathsV1(
        baseline_policy_path=paths.baseline_policy_path, package_root=paths.package_root,
        splits_dir=paths.splits_dir, screening_basin_ids_path=paths.screening_basin_ids_path,
    )
    artifact_identities = _verify_artifact_identities(v1_paths)
    package_manifest = read_package_manifest(paths.package_root)
    membership = validate_full_population_basin_membership(package_manifest, paths.splits_dir)
    screening = load_screening_basin_ids(paths.screening_basin_ids_path, development_basins=membership.development_basins,
                                         expected_count=400, expected_sha256=sweep.SCREENING_ARTIFACT_SHA256)

    policy_v2 = load_stage1_baseline_policy_v2_six_axis(paths.baseline_policy_path, paths.policy_overlay_path)
    contract = load_fixed_support_contract(paths.fixed_support_contract_path)
    for key in (
        "package_manifest_sha256", "package_file_checksums_sha256", "package_run_provenance_sha256",
        "development_split_sha256", "spatial_holdout_split_sha256",
    ):
        if contract[key] != artifact_identities[key]:
            raise SweepV2PreparationError(f"fixed-support contract {key} does not match the verified package identity")
    support_contract_version = contract["contract_id"]
    support_contract_sha256 = contract["checksum_sha256"]

    bundle = build_pilot_bundle_with_validation_scope(
        baseline_policy_path=paths.baseline_policy_path, package_root=paths.package_root, splits_dir=paths.splits_dir,
        lead_hours=6, seq_length=axes["seq_length"], run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        validation_basin_ids=screening, population_role=SCREENING_VALIDATION_POPULATION_ROLE,
        package_type=sweep.PACKAGE_IDENTITY, max_updates_per_epoch=sweep.MAX_UPDATES_PER_EPOCH,
        learning_rate=float(axes["learning_rate"]), hidden_size=axes["hidden_size"],
        embedding_dropout=float(axes["embedding_dropout"]), output_dropout=float(axes["output_dropout"]),
        batch_size=axes["batch_size"], dynamic_inputs=list(FROZEN_FIXED_CONFIGURATION_V2["dynamic_inputs"]),
        policy_override=policy_v2,
    )
    bundle = replace(bundle, config_mapping={**bundle.config_mapping, "epochs": sweep.TARGET_EPOCH})
    _audit_generated_config_v2(bundle, axes)

    config_id = configuration_id_v2(axes, support_contract_version=support_contract_version,
                                     support_contract_sha256=support_contract_sha256)
    pid = proposal_id_v2(expected_arm, order)
    tid = trial_id_v2(config_id, pid, execution_generation=generation)

    evidence = {
        "prepare_status": "PASS", "prepare_only": True, "objective_score": None,
        "campaign_id": CAMPAIGN_ID_V2, "domain_version": DOMAIN_VERSION_V2,
        "canonicalization_version": CONFIGURATION_CANONICALIZATION_VERSION_V2,
        "objective_id": OBJECTIVE_ID_V2, "search_arm": expected_arm,
        "proposal_id": pid, "proposal_order": order, "configuration_id": config_id, "trial_id": tid,
        "execution_generation": generation, "hyperparameters": axes,
        "seq_length_raw": proposal["seq_length"], "seq_length_normalized": axes["seq_length"],
        "model_seed": sweep.MODEL_SEED_A, "fidelity_id": FIDELITY_ID_V2, "target_epoch": sweep.TARGET_EPOCH,
        "max_updates_per_epoch": sweep.MAX_UPDATES_PER_EPOCH, "save_weights_every": 1,
        "authoritative_screening_epochs": list(range(1, 13)), "performance_early_stopping_enabled": False,
        "package_identity": sweep.PACKAGE_IDENTITY, "screening_artifact_sha256": sweep.SCREENING_ARTIFACT_SHA256,
        "artifact_identity_status": "PASS", **artifact_identities,
        "screening_policy_identity": sweep.SCREENING_POLICY_IDENTITY, "evaluation_scope": "development_validation_2024_only",
        "sealed_scope": False, "support_contract_version": support_contract_version,
        "support_contract_sha256": support_contract_sha256,
        "wandb_sweep_id": proposal.get("wandb_sweep_id"), "wandb_run_id": proposal.get("wandb_run_id"),
    }
    return PreparedSweepV2Proposal(dict(proposal), config_id, pid, tid, generation, bundle, evidence)


def prepare_bayesian_proposal_v2(*, proposal: Mapping[str, Any], paths: PreparationPathsV2) -> PreparedSweepV2Proposal:
    """Prepare one v2 six-axis Bayesian proposal; W&B values are
    telemetry-only provenance. Mirrors
    :func:`sweep_v1_production_adapter.prepare_bayesian_proposal` exactly in
    shape; ``expected_arm`` is pinned to ``"bayesian"`` here, so the
    scientifically independent ``random_control`` arm can never be prepared
    through this Bayesian front door (see
    :data:`sweep_v2_six_axis_campaign.SEARCH_ARMS_V2`)."""
    return _prepare_proposal_v2(proposal=proposal, paths=paths, expected_arm="bayesian")


def prepare_random_control_proposal_v2(*, row: Mapping[str, Any], manifest_path: Path,
                                       paths: PreparationPathsV2,
                                       execution_generation: int = 1) -> PreparedSweepV2Proposal:
    """Prepare one immutable committed v2 six-axis random-control row without
    regenerating it.

    Mirrors :func:`sweep_v1_production_adapter.prepare_random_control_row`
    exactly: verify the manifest's frozen SHA-256, confirm the requested row
    is an exact committed manifest row (never recomputed), strip the
    manifest-only identity/provenance fields, then run the identical
    :func:`_prepare_proposal_v2` path the Bayesian arm uses with
    ``expected_arm="random_control"``. No W&B, no Bayesian-controller
    contact: random-control rows are a fixed committed manifest, not a live
    search."""
    manifest_path = Path(manifest_path)
    if _sha256(manifest_path) != rc.RANDOM_CONTROL_MANIFEST_SHA256_V2:
        raise SweepV2PreparationError("v2 random-control manifest SHA-256 does not match frozen bytes")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list) or dict(row) not in rows:
        raise SweepV2PreparationError("requested random row is not an exact committed manifest row")
    supplied = dict(row)
    supplied["execution_generation"] = execution_generation
    for manifest_only in ("configuration_id", "proposal_id", "trial_id_attempt001",
                          "manifest_rng_seed", "manifest_index"):
        supplied.pop(manifest_only, None)
    return _prepare_proposal_v2(proposal=supplied, paths=paths, expected_arm="random_control")


_LAYER_B_PROVENANCE_FILENAME = "execution_provenance.json"


def write_prepared_proposal_v2(
    prepared: PreparedSweepV2Proposal, output_dir: Path, *, allow_layer_b_provenance: bool = False
) -> dict[str, Any]:
    """Write the exact NH config and finalize compact v2 prepare-only
    evidence. Mirrors :func:`sweep_v1_production_adapter.write_prepared_proposal`
    exactly in shape and no-overwrite/Layer-B-tolerance semantics -- see that
    function's docstring for the full contract. The only difference is the
    error type (:class:`SweepV2PreparationError`) and that the tolerated
    pre-existing Layer-B file's ``trial_id`` is checked against a v2
    ``trial_id`` (embeds ``proposal_id``, per :func:`trial_id_v2`)."""
    if allow_layer_b_provenance:
        provenance_path = Path(output_dir) / _LAYER_B_PROVENANCE_FILENAME
        if provenance_path.exists():
            existing_trial_id = json.loads(provenance_path.read_text(encoding="utf-8")).get("trial_id")
            if existing_trial_id != prepared.trial_id:
                raise SweepV2PreparationError(
                    f"existing {_LAYER_B_PROVENANCE_FILENAME} trial_id={existing_trial_id!r} does not "
                    f"exactly match the v2 trial being prepared ({prepared.trial_id!r}); "
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
