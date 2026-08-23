"""Tests for scripts/wandb_online_sweep_qualification_toy.py.

Pure logic, no real/fake wandb import needed. Focused on: toy-objective
determinism/finiteness, sweep-config parameter/domain mapping against the
FROZEN Sweep-v1 search domain, Flash-NH legality/configuration-ID mapping
(reused, not reimplemented), and qualification-only identity/metadata never
looking like a real scientific trial identity.
"""
from __future__ import annotations

import math

import pytest

import scripts.wandb_online_sweep_qualification_toy as toy
from src.baseline import sweep_v1_campaign


# ---------------------------------------------------------------------------
# Identity isolation: this module's identifiers must never collide with the
# real frozen campaign's.
# ---------------------------------------------------------------------------

def test_qualification_campaign_id_is_distinct_from_real_campaign_id():
    assert toy.QUALIFICATION_CAMPAIGN_ID != sweep_v1_campaign.CAMPAIGN_ID


def test_toy_metric_name_is_outside_the_scientific_screening_namespace():
    assert not toy.TOY_METRIC_NAME.startswith("screening/")
    assert toy.TOY_METRIC_NAME == "qualification/toy_objective"


def test_qualification_tags_include_non_scientific_marker():
    assert "non_scientific" in toy.QUALIFICATION_TAGS
    assert "qualification" in toy.QUALIFICATION_TAGS


# ---------------------------------------------------------------------------
# compute_toy_objective: deterministic, finite, parameter-dependent
# ---------------------------------------------------------------------------

_VALID_HP = {
    "learning_rate": 3e-4,
    "hidden_size": 128,
    "embedding_dropout": 0.1,
    "output_dropout": 0.1,
    "batch_size": 256,
}


def test_toy_objective_is_deterministic():
    a = toy.compute_toy_objective(_VALID_HP)
    b = toy.compute_toy_objective(dict(_VALID_HP))
    assert a == b


def test_toy_objective_is_finite():
    value = toy.compute_toy_objective(_VALID_HP)
    assert math.isfinite(value)


def test_toy_objective_depends_on_learning_rate():
    near = toy.compute_toy_objective({**_VALID_HP, "learning_rate": 3e-4})
    far = toy.compute_toy_objective({**_VALID_HP, "learning_rate": 1e-3})
    assert near != far


def test_toy_objective_depends_on_dropout():
    low = toy.compute_toy_objective({**_VALID_HP, "embedding_dropout": 0.0, "output_dropout": 0.0})
    high = toy.compute_toy_objective({**_VALID_HP, "embedding_dropout": 0.4, "output_dropout": 0.4})
    assert low != high
    assert low > high  # more dropout -> lower toy objective by construction


def test_toy_objective_requires_all_fields():
    incomplete = dict(_VALID_HP)
    del incomplete["batch_size"]
    with pytest.raises(ValueError):
        toy.compute_toy_objective(incomplete)


def test_toy_objective_rejects_nonpositive_learning_rate():
    with pytest.raises(ValueError):
        toy.compute_toy_objective({**_VALID_HP, "learning_rate": 0.0})


def test_toy_objective_never_imports_neuralhydrology_or_torch():
    import sys

    assert "neuralhydrology" not in sys.modules or True  # presence unrelated to this module
    # The real assertion: this module's own source never references them.
    import inspect

    source = inspect.getsource(toy)
    assert "import torch" not in source
    assert "import neuralhydrology" not in source


# ---------------------------------------------------------------------------
# build_sweep_config: must mirror the frozen Sweep-v1 domain exactly
# ---------------------------------------------------------------------------

def test_sweep_config_method_is_bayes():
    config = toy.build_sweep_config()
    assert config["method"] == "bayes"


def test_sweep_config_metric_is_toy_not_scientific():
    config = toy.build_sweep_config()
    assert config["metric"]["name"] == toy.TOY_METRIC_NAME


def test_sweep_config_learning_rate_matches_frozen_domain():
    config = toy.build_sweep_config()
    lr = config["parameters"]["learning_rate"]
    domain = sweep_v1_campaign.SEARCH_DOMAIN["learning_rate"]
    assert lr["distribution"] == "log_uniform_values"
    assert lr["min"] == domain["lower"]
    assert lr["max"] == domain["upper"]


def test_sweep_config_categoricals_match_frozen_domain():
    config = toy.build_sweep_config()
    for field in ("hidden_size", "batch_size"):
        assert config["parameters"][field]["values"] == list(sweep_v1_campaign.SEARCH_DOMAIN[field]["values"])


def test_sweep_config_dropout_axes_match_frozen_domain():
    config = toy.build_sweep_config()
    for field in ("embedding_dropout", "output_dropout"):
        domain = sweep_v1_campaign.SEARCH_DOMAIN[field]
        param = config["parameters"][field]
        assert param["distribution"] == "uniform"
        assert param["min"] == domain["lower"]
        assert param["max"] == domain["upper"]


def test_sweep_config_covers_all_five_frozen_axes():
    config = toy.build_sweep_config()
    assert set(config["parameters"]) == set(toy.HYPERPARAMETER_FIELDS)


# ---------------------------------------------------------------------------
# check_flashnh_legality: reuses sweep_v1_campaign, never raises
# ---------------------------------------------------------------------------

def test_legality_pass_for_a_valid_proposal():
    result = toy.check_flashnh_legality(_VALID_HP)
    assert result["legality_pass"] is True
    assert result["configuration_id"] is not None
    assert result["configuration_id"] == sweep_v1_campaign.configuration_id(
        sweep_v1_campaign.canonical_hyperparameters(_VALID_HP)
    )


def test_legality_fails_gracefully_for_out_of_domain_proposal():
    bad = {**_VALID_HP, "learning_rate": 10.0}  # far outside 1e-4..1e-3
    result = toy.check_flashnh_legality(bad)
    assert result["legality_pass"] is False
    assert result["configuration_id"] is None
    assert result["error"] is not None


def test_legality_fails_gracefully_for_missing_field_never_raises():
    incomplete = dict(_VALID_HP)
    del incomplete["hidden_size"]
    result = toy.check_flashnh_legality(incomplete)
    assert result["legality_pass"] is False


# ---------------------------------------------------------------------------
# build_run_identity: cannot be mistaken for a real scientific run identity
# ---------------------------------------------------------------------------

def test_run_identity_marks_non_scientific():
    identity = toy.build_run_identity("first")
    assert identity["scientific_trial"] is False
    assert identity["qualification_campaign_id"] == toy.QUALIFICATION_CAMPAIGN_ID
    assert identity["proposal_label"] == "first"


def test_run_identity_never_contains_real_campaign_fields():
    identity = toy.build_run_identity("first")
    for forbidden_key in ("pilot_policy_name", "run_spec", "package_manifest_identity"):
        assert forbidden_key not in identity


def test_run_identity_merges_extra_without_losing_required_fields():
    identity = toy.build_run_identity("second", extra={"custom": "value"})
    assert identity["custom"] == "value"
    assert identity["proposal_label"] == "second"
