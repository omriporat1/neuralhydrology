"""Offline contract tests for the frozen Phase-B Sweep-v1 static foundation."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from src.baseline import sweep_v1_campaign as sweep


MANIFEST_PATH = Path(__file__).parents[1] / "config" / "stage1_phase_b_sweep_v1_original_domain_v001_random_control_manifest.json"
MANIFEST_SHA_PATH = MANIFEST_PATH.with_suffix(".sha256")


def _hyperparameters():
    return {"learning_rate": 3e-4, "hidden_size": 128, "embedding_dropout": 0.1,
            "output_dropout": 0.25, "batch_size": 256}


def test_frozen_campaign_domain_and_identity_are_exact():
    assert (sweep.CAMPAIGN_ID, sweep.DOMAIN_VERSION, sweep.MODEL_SEED_A, sweep.TARGET_EPOCH,
            sweep.MAX_UPDATES_PER_EPOCH) == (
        "stage1_phase_b_sweep_v1_original_domain_v001", "original_domain_v001", 967139, 12, 50_000)
    assert sweep.SEARCH_DOMAIN["learning_rate"] == {
        "kind": "continuous", "distribution": "log_uniform", "lower": 1e-4, "upper": 1e-3,
        "lower_boundary": "expandable", "upper_boundary": "expandable"}
    assert sweep.SEARCH_DOMAIN["hidden_size"]["values"] == [64, 128, 256]
    assert sweep.SEARCH_DOMAIN["batch_size"]["values"] == [128, 256, 512]
    assert sweep.SEARCH_DOMAIN["embedding_dropout"]["lower_boundary"] == "natural"
    assert sweep.SEARCH_DOMAIN["output_dropout"]["lower_boundary"] == "natural"
    assert sweep.FROZEN_FIXED_CONFIGURATION == {
        "optimizer": "Adam", "dynamic_input_family": "PT",
        "dynamic_inputs": ["mrms_qpe_1h_mm", "rtma_2t_K"], "seq_length": 72,
        "static_embedding": {"hiddens": [128, 32], "activation": "tanh"}, "lead_hours": 6,
        "save_weights_every": 1, "performance_early_stopping_enabled": False,
        "authoritative_screening_epochs": list(range(1, 13)),
        "evaluation_scope": "development_validation_2024_only",
    }


def test_random_manifest_is_exactly_reproducible_and_checksum_pinned():
    committed = MANIFEST_PATH.read_bytes()
    assert committed == sweep.render_manifest()
    assert sweep.sha256_bytes(committed) == sweep.RANDOM_CONTROL_MANIFEST_SHA256
    assert MANIFEST_SHA_PATH.read_text(encoding="utf-8").split()[0] == sweep.RANDOM_CONTROL_MANIFEST_SHA256
    payload = json.loads(committed)
    rows = sweep.generate_random_control_rows()
    assert payload["manifest_rng_seed"] == sweep.MANIFEST_RNG_SEED == 20260822
    assert payload["generator_algorithm"] == sweep.GENERATOR_ALGORITHM
    assert sweep.GENERATOR_VERSION == "sweep_v1_iid_random_manifest_v1"
    assert sweep.GENERATOR_RNG_IMPLEMENTATION == "python random.Random (MT19937)"
    assert sweep.GENERATOR_DRAW_ORDER == (
        "learning_rate", "embedding_dropout", "output_dropout", "hidden_size", "batch_size")
    assert payload["rows"] == rows
    assert len(rows) == 12
    sweep.validate_manifest_rows(rows)


def test_random_manifest_uses_frozen_iid_draw_order_and_legal_domain():
    rows = sweep.generate_random_control_rows()
    rng = random.Random(20260822)
    expected_lr = 10 ** rng.uniform(math.log10(1e-4), math.log10(1e-3))
    expected_embedding = rng.uniform(0.0, 0.4)
    expected_output = rng.uniform(0.0, 0.4)
    expected_hidden = rng.choice([64, 128, 256])
    expected_batch = rng.choice([128, 256, 512])
    first = rows[0]
    assert float(first["learning_rate"]) == expected_lr
    assert float(first["embedding_dropout"]) == expected_embedding
    assert float(first["output_dropout"]) == expected_output
    assert (first["hidden_size"], first["batch_size"]) == (expected_hidden, expected_batch)
    assert all(1e-4 <= float(row["learning_rate"]) <= 1e-3 for row in rows)
    assert all(0.0 <= float(row[axis]) <= 0.4 for row in rows for axis in ("embedding_dropout", "output_dropout"))
    assert {row["hidden_size"] for row in rows} <= {64, 128, 256}
    assert {row["batch_size"] for row in rows} <= {128, 256, 512}
    assert [row["manifest_index"] for row in rows] == list(range(1, 13))


def test_configuration_identity_is_arm_and_order_independent_and_trial_attempts_differ():
    hyperparameters = _hyperparameters()
    config_id = sweep.configuration_id(hyperparameters)
    assert sweep.CONFIGURATION_CANONICALIZATION_VERSION == "sweep_v1_five_axis_canonical_json_v001"
    assert config_id == "sweep_v1_cfg_e12a5afd4468833c7bfd"
    assert config_id == sweep.configuration_id(dict(reversed(list(hyperparameters.items()))))
    assert config_id == sweep.configuration_id({**hyperparameters})
    assert sweep.proposal_id("bayesian", 1) != sweep.proposal_id("random_control", 1)
    assert sweep.trial_id(config_id, execution_generation=1) != sweep.trial_id(config_id, execution_generation=2)
    assert "wandb" not in config_id.lower() and "slurm" not in config_id.lower()
    # A future identical proposal from either arm keeps one scientific config identity.
    random_row = {**hyperparameters, "search_arm": "random_control", "proposal_order": 1}
    bayesian_row = {**hyperparameters, "search_arm": "bayesian", "proposal_order": 99}
    assert sweep.configuration_id({k: random_row[k] for k in _hyperparameters()}) == \
        sweep.configuration_id({k: bayesian_row[k] for k in _hyperparameters()})


def test_no_manifest_filtering_or_deduplication_is_hidden_in_validation():
    rows = sweep.generate_random_control_rows()
    rows[1] = {**rows[1], **{key: rows[0][key] for key in _hyperparameters()},
               "configuration_id": sweep.configuration_id({key: rows[0][key] for key in _hyperparameters()})}
    # Duplicate scientific configurations are structurally legal; only canonical
    # generation/order/provenance are validated, never uniqueness.
    sweep.validate_manifest_rows(rows)


def test_non_monotonic_trajectory_diagnostics_are_descriptive_only():
    trajectory = {epoch: 0.10 + epoch / 100 for epoch in range(1, 13)}
    trajectory.update({9: 0.43, 10: 0.42, 11: 0.45, 12: 0.40})
    diagnostic = sweep.derive_trajectory_diagnostics(trajectory)
    assert diagnostic == {
        "best_epoch": 11, "best_score": 0.45, "final_epoch_score": 0.40,
        "best_minus_final": pytest.approx(0.05), "best_score_10": 0.43,
        "best_score_12": 0.45, "late_gain_10_to_12": pytest.approx(0.02), "late_best": True,
    }
    with pytest.raises(ValueError, match="exactly authoritative epochs"):
        sweep.derive_trajectory_diagnostics({epoch: 0.1 for epoch in range(1, 12)})


def test_review_schemas_are_complete_authoritative_and_sealed_wandb_free():
    required = {"trial_summary": sweep.TRIAL_SUMMARY_FIELDS, "epoch_trajectory": sweep.EPOCH_TRAJECTORY_FIELDS,
                "proposal": sweep.PROPOSAL_RECORD_FIELDS, "operations": sweep.OPERATIONS_RECORD_FIELDS}
    for fields in required.values():
        assert not any("sealed" in field or field.startswith("wandb_") for field in fields)
        record = {field: "placeholder" for field in fields}
        record.update(campaign_id=sweep.CAMPAIGN_ID, domain_version=sweep.DOMAIN_VERSION, search_arm="random_control")
        sweep.validate_review_record(next(name for name, candidate in required.items() if candidate == fields), record)
    missing = {field: "placeholder" for field in sweep.EPOCH_TRAJECTORY_FIELDS - {"epoch"}}
    missing.update(campaign_id=sweep.CAMPAIGN_ID, domain_version=sweep.DOMAIN_VERSION, search_arm="random_control")
    with pytest.raises(ValueError, match="missing required"):
        sweep.validate_review_record("epoch_trajectory", missing)
    forbidden = {field: "placeholder" for field in sweep.EPOCH_TRAJECTORY_FIELDS}
    forbidden.update(campaign_id=sweep.CAMPAIGN_ID, domain_version=sweep.DOMAIN_VERSION, search_arm="random_control",
                     sealed_score=0.0)
    with pytest.raises(ValueError, match="forbidden"):
        sweep.validate_review_record("epoch_trajectory", forbidden)


def test_static_foundation_has_no_wandb_dependency():
    source = Path(sweep.__file__).read_text(encoding="utf-8")
    assert "import wandb" not in source and "from wandb" not in source
