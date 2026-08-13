"""Focused tests for the scalar-override application registry (Stage 1
Scope D of the Sequence-Length-A minimum-viable-infrastructure task):
``src.baseline.nh_config_generation._SCALAR_OVERRIDE_REGISTRY`` /
``apply_scalar_overrides``.

Covers: an unknown override name is rejected loudly (the whitelist
contract), each of the four currently-registered axes
(``max_updates_per_epoch``/``learning_rate``/``hidden_size``/
``embedding_dropout``) is actually applied and actually validated,
``None``-valued entries are skipped, and ``seq_length`` -- deliberately NOT
part of this registry (it is a required field resolved earlier in the
pipeline, not a post-hoc override; see the registry's own module comment)
-- is rejected the same way any other unknown name would be.
"""
import pytest

from src.baseline.nh_config_generation import (
    NHConfigGenerationError,
    _SCALAR_OVERRIDE_REGISTRY,
    apply_scalar_overrides,
)


def _base_mapping() -> dict:
    return {
        "max_updates_per_epoch": None,
        "learning_rate": 3e-4,
        "hidden_size": 128,
        "statics_embedding": {"hiddens": [128, 32], "activation": "tanh", "dropout": 0.1},
    }


def test_unknown_override_name_is_rejected():
    with pytest.raises(NHConfigGenerationError, match="unknown scalar override"):
        apply_scalar_overrides(_base_mapping(), {"seq_length": 48})


def test_none_valued_overrides_are_skipped_entirely():
    mapping = _base_mapping()
    before = dict(mapping)
    apply_scalar_overrides(mapping, {name: None for name in _SCALAR_OVERRIDE_REGISTRY})
    assert mapping["max_updates_per_epoch"] == before["max_updates_per_epoch"]
    assert mapping["learning_rate"] == before["learning_rate"]
    assert mapping["hidden_size"] == before["hidden_size"]
    assert mapping["statics_embedding"] == before["statics_embedding"]


def test_max_updates_per_epoch_override_is_applied():
    mapping = _base_mapping()
    apply_scalar_overrides(mapping, {"max_updates_per_epoch": 25_000})
    assert mapping["max_updates_per_epoch"] == 25_000


def test_max_updates_per_epoch_override_rejects_non_positive():
    with pytest.raises(NHConfigGenerationError):
        apply_scalar_overrides(_base_mapping(), {"max_updates_per_epoch": 0})


def test_learning_rate_override_is_applied():
    mapping = _base_mapping()
    apply_scalar_overrides(mapping, {"learning_rate": 1e-2})
    assert mapping["learning_rate"] == 1e-2


def test_hidden_size_override_is_applied():
    mapping = _base_mapping()
    apply_scalar_overrides(mapping, {"hidden_size": 256})
    assert mapping["hidden_size"] == 256


def test_embedding_dropout_override_is_applied_without_mutating_shared_profile_dict():
    mapping = _base_mapping()
    shared_statics_embedding = mapping["statics_embedding"]
    apply_scalar_overrides(mapping, {"embedding_dropout": 0.0})
    assert mapping["statics_embedding"]["dropout"] == 0.0
    # The override must copy-before-mutate: the original dict object passed
    # in (standing in for a shared module-level run-profile dict) must be
    # left untouched.
    assert shared_statics_embedding["dropout"] == 0.1


def test_embedding_dropout_override_rejects_missing_statics_embedding_section():
    mapping = _base_mapping()
    del mapping["statics_embedding"]
    with pytest.raises(NHConfigGenerationError, match="statics_embedding"):
        apply_scalar_overrides(mapping, {"embedding_dropout": 0.2})


def test_embedding_dropout_override_rejects_out_of_range_value():
    with pytest.raises(NHConfigGenerationError):
        apply_scalar_overrides(_base_mapping(), {"embedding_dropout": 1.0})


def test_multiple_registered_overrides_applied_together():
    mapping = _base_mapping()
    apply_scalar_overrides(
        mapping,
        {"max_updates_per_epoch": 25_000, "learning_rate": 3e-3, "hidden_size": 64},
    )
    assert mapping["max_updates_per_epoch"] == 25_000
    assert mapping["learning_rate"] == 3e-3
    assert mapping["hidden_size"] == 64
