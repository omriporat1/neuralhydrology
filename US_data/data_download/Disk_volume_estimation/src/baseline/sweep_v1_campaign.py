"""Offline static contract for the frozen Phase-B Sweep-v1 original wave.

This module deliberately contains no launcher, NeuralHydrology, or W&B
integration.  It makes the frozen campaign inputs, identity rules,
trajectory diagnostics, and evidence-table schemas available to later bounded
implementation steps.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "CAMPAIGN_ID", "DOMAIN_VERSION", "MANIFEST_RNG_SEED", "MODEL_SEED_A", "FROZEN_FIXED_CONFIGURATION",
    "GENERATOR_VERSION", "GENERATOR_RNG_IMPLEMENTATION", "GENERATOR_DRAW_ORDER",
    "CONFIGURATION_CANONICALIZATION_VERSION",
    "RANDOM_CONTROL_MANIFEST_SHA256", "SEARCH_ARMS", "SEARCH_DOMAIN",
    "TRIAL_SUMMARY_FIELDS", "EPOCH_TRAJECTORY_FIELDS", "PROPOSAL_RECORD_FIELDS",
    "OPERATIONS_RECORD_FIELDS", "canonical_hyperparameters", "configuration_id",
    "trial_id", "proposal_id", "trial_identity_conflicts", "generate_random_control_rows", "manifest_payload",
    "render_manifest", "sha256_bytes", "derive_trajectory_diagnostics",
    "validate_review_record", "validate_manifest_rows",
]

CAMPAIGN_ID = "stage1_phase_b_sweep_v1_original_domain_v001"
DOMAIN_VERSION = "original_domain_v001"
MODEL_SEED_A = 967139
TARGET_EPOCH = 12
MAX_UPDATES_PER_EPOCH = 50_000
MANIFEST_RNG_SEED = 20260822
RANDOM_CONTROL_COUNT = 12
SEARCH_ARMS = frozenset({"bayesian", "random_control"})

# The package name and screening checksum are immutable campaign provenance,
# not paths to data and not an invitation to open any sealed scope.
PACKAGE_IDENTITY = "stage1_scientific_package_v002"
SCREENING_POLICY_IDENTITY = "stage1_provisional_operational_screening_subset_v001"
SCREENING_ARTIFACT_SHA256 = "d4395d93ebc567cf09e149c0121463d75cf4f7ecc02c07a7c4a7999763baa372"
OBJECTIVE_ID = "best_eligible_authoritative_median_per_basin_raw_space_nse_epochs_1_to_12"
GENERATOR_ALGORITHM = "python_random_mt19937_iid_log_uniform_and_uniform_categorical_v001"
# These source/audit identifiers intentionally do not alter the already
# accepted manifest payload or bytes.  They document exactly how the frozen
# rows were obtained and how their configuration hashes are canonicalized.
GENERATOR_VERSION = "sweep_v1_iid_random_manifest_v1"
GENERATOR_RNG_IMPLEMENTATION = "python random.Random (MT19937)"
GENERATOR_DRAW_ORDER = (
    "learning_rate", "embedding_dropout", "output_dropout", "hidden_size", "batch_size",
)
CONFIGURATION_CANONICALIZATION_VERSION = "sweep_v1_five_axis_canonical_json_v001"
FROZEN_FIXED_CONFIGURATION = {
    "optimizer": "Adam", "dynamic_input_family": "PT",
    "dynamic_inputs": ["mrms_qpe_1h_mm", "rtma_2t_K"], "seq_length": 72,
    "static_embedding": {"hiddens": [128, 32], "activation": "tanh"}, "lead_hours": 6,
    "save_weights_every": 1, "performance_early_stopping_enabled": False,
    "authoritative_screening_epochs": list(range(1, TARGET_EPOCH + 1)),
    "evaluation_scope": "development_validation_2024_only",
}

SEARCH_DOMAIN = {
    "learning_rate": {"kind": "continuous", "distribution": "log_uniform", "lower": 1e-4, "upper": 1e-3,
                      "lower_boundary": "expandable", "upper_boundary": "expandable"},
    "hidden_size": {"kind": "categorical", "distribution": "uniform", "values": [64, 128, 256],
                    "lower_boundary": "expandable", "upper_boundary": "expandable"},
    "embedding_dropout": {"kind": "continuous", "distribution": "uniform", "lower": 0.0, "upper": 0.4,
                          "lower_boundary": "natural", "upper_boundary": "expandable"},
    "output_dropout": {"kind": "continuous", "distribution": "uniform", "lower": 0.0, "upper": 0.4,
                       "lower_boundary": "natural", "upper_boundary": "expandable"},
    "batch_size": {"kind": "categorical", "distribution": "uniform", "values": [128, 256, 512],
                   "lower_boundary": "expandable", "upper_boundary": "expandable"},
}

_HYPERPARAMETER_FIELDS = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")

# Durable, plotting-independent records.  Fields are intentionally plain so
# Flash-NH evidence remains authoritative even if W&B is unavailable.
TRIAL_SUMMARY_FIELDS = frozenset({
    "campaign_id", "domain_version", "search_arm", "proposal_id", "configuration_id", "trial_id",
    "workflow_status", "objective_score", "best_epoch", "best_score", "final_epoch_score", "best_minus_final",
    "best_score_10", "best_score_12", "late_gain_10_to_12", "late_best", *_HYPERPARAMETER_FIELDS,
    "runtime_seconds", "gpu_hours", "execution_generation", "retry_of_trial_id", "failure_category",
})
EPOCH_TRAJECTORY_FIELDS = frozenset({
    "campaign_id", "domain_version", "configuration_id", "trial_id", "search_arm", "epoch",
    "median_raw_space_nse", "evaluation_status",
})
PROPOSAL_RECORD_FIELDS = frozenset({
    "campaign_id", "domain_version", "search_arm", "proposal_id", "proposal_order", "configuration_id",
    *_HYPERPARAMETER_FIELDS, "valid_result_order", "boundary_review_checkpoint", "wave_id",
})
OPERATIONS_RECORD_FIELDS = frozenset({
    "campaign_id", "domain_version", "configuration_id", "trial_id", "search_arm", "execution_generation",
    "slurm_job_id", "slurm_state", "runtime_seconds", "gpu_hours", "retry_of_trial_id", "failure_category",
})

# Populated only after the committed JSON is generated.  It pins exact bytes,
# while the manifest's internal payload hash pins its semantic content.
RANDOM_CONTROL_MANIFEST_SHA256 = "180d94feefa2cfc9686a29609b91711c7586ddfe903ea9c4ef2fcaef002346e3"


def _normal_float(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"expected finite numeric value, got {value!r}")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected finite numeric value, got {value!r}") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"expected finite numeric value, got {value!r}")
    return format(normalized, ".17g")


def canonical_hyperparameters(hyperparameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the versioned five-axis representation used by configuration_id.

    The returned insertion order is fixed by the literal below; continuous
    values use finite IEEE-754 ``.17g`` serialization and categoricals remain
    integer values.  No campaign, arm, proposal, or execution metadata enters
    this scientific-coordinate representation.
    """
    if set(hyperparameters) != set(_HYPERPARAMETER_FIELDS):
        raise ValueError(f"hyperparameters must contain exactly {_HYPERPARAMETER_FIELDS}")
    for key in ("hidden_size", "batch_size"):
        value = hyperparameters[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer categorical value")
    result = {
        "learning_rate": _normal_float(hyperparameters["learning_rate"]),
        "hidden_size": hyperparameters["hidden_size"],
        "embedding_dropout": _normal_float(hyperparameters["embedding_dropout"]),
        "output_dropout": _normal_float(hyperparameters["output_dropout"]),
        "batch_size": hyperparameters["batch_size"],
    }
    _validate_hyperparameters(result)
    return result


def _validate_hyperparameters(hyperparameters: Mapping[str, Any]) -> None:
    lr = float(hyperparameters["learning_rate"])
    if not SEARCH_DOMAIN["learning_rate"]["lower"] <= lr <= SEARCH_DOMAIN["learning_rate"]["upper"]:
        raise ValueError("learning_rate outside frozen Sweep-v1 domain")
    for key in ("embedding_dropout", "output_dropout"):
        value = float(hyperparameters[key]); domain = SEARCH_DOMAIN[key]
        if not domain["lower"] <= value <= domain["upper"]:
            raise ValueError(f"{key} outside frozen Sweep-v1 domain")
    for key in ("hidden_size", "batch_size"):
        if hyperparameters[key] not in SEARCH_DOMAIN[key]["values"]:
            raise ValueError(f"{key} outside frozen Sweep-v1 domain")


def configuration_id(hyperparameters: Mapping[str, Any]) -> str:
    """Hash the fixed-order, versioned five-axis canonical JSON representation."""
    canonical = canonical_hyperparameters(hyperparameters)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sweep_v1_cfg_" + hashlib.sha256(encoded).hexdigest()[:20]


def trial_id(configuration: Mapping[str, Any] | str, *, execution_generation: int = 1) -> str:
    """Bind one executable attempt to a frozen configuration and fidelity."""
    if not isinstance(execution_generation, int) or execution_generation < 1:
        raise ValueError("execution_generation must be a positive integer")
    config_id = configuration if isinstance(configuration, str) else configuration_id(configuration)
    return f"{CAMPAIGN_ID}__{config_id}__mf12x50000__seedA{MODEL_SEED_A}__attempt{execution_generation:03d}"


def proposal_id(search_arm: str, proposal_order: int) -> str:
    if search_arm not in SEARCH_ARMS:
        raise ValueError(f"unknown search arm: {search_arm!r}")
    if not isinstance(proposal_order, int) or proposal_order < 1:
        raise ValueError("proposal_order must be a positive integer")
    return f"{CAMPAIGN_ID}__{search_arm}__proposal{proposal_order:03d}"


def trial_identity_conflicts(existing_trial_id: "str | None", expected_trial_id: "str | None") -> bool:
    """True iff both a recorded and an expected ``trial_id`` are present and
    disagree -- the single reused Layer-B provenance identity-consistency
    check (``sweep_v1_execution.enrich_layer_b_provenance`` and
    ``sweep_v1_production_adapter.write_prepared_proposal`` both call this
    rather than each re-implementing the comparison)."""
    return existing_trial_id is not None and expected_trial_id is not None and existing_trial_id != expected_trial_id


def generate_random_control_rows() -> list[dict[str, Any]]:
    """Generate the frozen 12-row IID control exactly once from its seed.

    The fixed :data:`GENERATOR_DRAW_ORDER` is LR in log geometry, embedding
    dropout, output dropout, hidden-size category, then batch-size category.
    No filtering, stratification, or duplicate rejection occurs.
    """
    rng = random.Random(MANIFEST_RNG_SEED)
    rows: list[dict[str, Any]] = []
    log_lower = math.log10(SEARCH_DOMAIN["learning_rate"]["lower"])
    log_upper = math.log10(SEARCH_DOMAIN["learning_rate"]["upper"])
    for index in range(1, RANDOM_CONTROL_COUNT + 1):
        hyperparameters = {
            "learning_rate": 10 ** rng.uniform(log_lower, log_upper),
            "embedding_dropout": rng.uniform(0.0, 0.4),
            "output_dropout": rng.uniform(0.0, 0.4),
            "hidden_size": rng.choice(SEARCH_DOMAIN["hidden_size"]["values"]),
            "batch_size": rng.choice(SEARCH_DOMAIN["batch_size"]["values"]),
        }
        canonical = canonical_hyperparameters(hyperparameters)
        rows.append({
            "manifest_index": index, "search_arm": "random_control",
            "proposal_id": proposal_id("random_control", index),
            "proposal_order": index, "configuration_id": configuration_id(canonical),
            **canonical, "manifest_rng_seed": MANIFEST_RNG_SEED,
            "campaign_id": CAMPAIGN_ID, "domain_version": DOMAIN_VERSION,
        })
    return rows


def manifest_payload(rows: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(generate_random_control_rows() if rows is None else rows)
    return {
        "schema_version": "stage1_phase_b_sweep_v1_random_control_manifest_v001",
        "campaign_id": CAMPAIGN_ID, "domain_version": DOMAIN_VERSION,
        "generator_algorithm": GENERATOR_ALGORITHM, "manifest_rng_seed": MANIFEST_RNG_SEED,
        "random_control_count": RANDOM_CONTROL_COUNT, "model_seed_a": MODEL_SEED_A,
        "target_epoch": TARGET_EPOCH, "max_updates_per_epoch": MAX_UPDATES_PER_EPOCH,
        "fixed_configuration": FROZEN_FIXED_CONFIGURATION,
        "package_identity": PACKAGE_IDENTITY, "screening_policy_identity": SCREENING_POLICY_IDENTITY,
        "screening_artifact_sha256": SCREENING_ARTIFACT_SHA256, "objective_id": OBJECTIVE_ID,
        "search_domain": SEARCH_DOMAIN, "rows": rows,
    }


def render_manifest(rows: Iterable[Mapping[str, Any]] | None = None) -> bytes:
    return (json.dumps(manifest_payload(rows), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_manifest_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if len(rows) != RANDOM_CONTROL_COUNT:
        raise ValueError(f"expected exactly {RANDOM_CONTROL_COUNT} random-control rows")
    for index, row in enumerate(rows, start=1):
        if row.get("manifest_index") != index or row.get("proposal_order") != index:
            raise ValueError("manifest rows must preserve the canonical sequential index/order")
        if row.get("search_arm") != "random_control" or row.get("manifest_rng_seed") != MANIFEST_RNG_SEED:
            raise ValueError("invalid frozen random-control provenance")
        hyperparameters = {key: row[key] for key in _HYPERPARAMETER_FIELDS}
        if row.get("configuration_id") != configuration_id(hyperparameters):
            raise ValueError("configuration_id does not match normalized scientific hyperparameters")


def derive_trajectory_diagnostics(epoch_objective: Mapping[int, float]) -> dict[str, Any]:
    """Derive descriptive, non-decisional diagnostics for a complete 12-epoch run."""
    expected_epochs = set(range(1, TARGET_EPOCH + 1))
    if set(epoch_objective) != expected_epochs:
        raise ValueError("trajectory must contain exactly authoritative epochs 1..12")
    values = {epoch: float(epoch_objective[epoch]) for epoch in sorted(epoch_objective)}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("trajectory objective values must be finite")
    best_score = max(values.values())
    best_epoch = next(epoch for epoch, value in values.items() if value == best_score)
    best_score_10 = max(values[epoch] for epoch in range(1, 11))
    best_score_12 = max(values.values())
    final_score = values[TARGET_EPOCH]
    return {
        "best_epoch": best_epoch, "best_score": best_score, "final_epoch_score": final_score,
        "best_minus_final": best_score - final_score, "best_score_10": best_score_10,
        "best_score_12": best_score_12, "late_gain_10_to_12": best_score_12 - best_score_10,
        "late_best": best_epoch >= 11,
    }


def validate_review_record(record_type: str, record: Mapping[str, Any]) -> None:
    """Reject incomplete records and sealed/W&B-derived schema expansion."""
    schemas = {
        "trial_summary": TRIAL_SUMMARY_FIELDS, "epoch_trajectory": EPOCH_TRAJECTORY_FIELDS,
        "proposal": PROPOSAL_RECORD_FIELDS, "operations": OPERATIONS_RECORD_FIELDS,
    }
    if record_type not in schemas:
        raise ValueError(f"unknown record type: {record_type!r}")
    missing = schemas[record_type] - set(record)
    if missing:
        raise ValueError(f"{record_type} missing required fields: {sorted(missing)}")
    forbidden_markers = ("sealed", "temporal_test", "spatial_holdout", "california")
    forbidden = {key for key in record if any(marker in key.lower() for marker in forbidden_markers)
                 or key.lower().startswith("wandb_")}
    if forbidden:
        raise ValueError(f"{record_type} contains forbidden non-authoritative fields: {sorted(forbidden)}")
    if record["campaign_id"] != CAMPAIGN_ID or record["domain_version"] != DOMAIN_VERSION:
        raise ValueError("record campaign/domain identity does not match frozen original wave")
    if record["search_arm"] not in SEARCH_ARMS:
        raise ValueError("record has unknown search arm")
