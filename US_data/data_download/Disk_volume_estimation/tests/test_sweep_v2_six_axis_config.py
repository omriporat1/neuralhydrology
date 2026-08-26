"""Local, no-W&B tests for the v2 six-axis sweep config builders (Section
C, additive six-axis campaign foundation).

``src.baseline.sweep_v2_six_axis_config`` was previously exercised only by
its own import-time self-check (``_assert_five_axes_match_v1``); no test
file called ``build_production_sweep_config_v2`` or
``build_wandb_bridge_rehearsal_sweep_config_v2`` directly, so the binding
``q_uniform`` wire-representation mandate (min=48, max=120, q=12) had no
dedicated regression coverage. These tests close that gap.
"""
from __future__ import annotations

import pytest

from src.baseline import sweep_v1_execution as v1_execution
from src.baseline import sweep_v2_six_axis_config as config_v2
from src.baseline.sweep_v2_six_axis_campaign import (
    FORBIDDEN_V1_SWEEP_ID,
    OBJECTIVE_ID_V2,
    SEQ_LENGTH_DOMAIN_V2,
    SEQ_LENGTH_MAX,
    SEQ_LENGTH_MIN,
    SEQ_LENGTH_STEP,
)

_FIVE_AXES = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")


def test_v2_metric_name_embeds_the_v2_objective_id_and_differs_from_v1():
    assert config_v2.V2_METRIC_NAME == f"flashnh/{OBJECTIVE_ID_V2}"
    v1_config = v1_execution.build_production_sweep_config(program="p")
    assert config_v2.V2_METRIC_NAME != v1_config["metric"]["name"]


def test_seq_length_axis_is_q_uniform_with_the_mandated_bounds():
    cfg = config_v2.build_production_sweep_config_v2(program="p")
    seq_length_param = cfg["parameters"]["seq_length"]
    assert seq_length_param == {
        "distribution": "q_uniform",
        "min": 48,
        "max": 120,
        "q": 12,
    }
    # Cross-checked against the normalization contract's own constants, not
    # re-hardcoded, so the wire config can never silently drift from them.
    assert seq_length_param["min"] == SEQ_LENGTH_MIN
    assert seq_length_param["max"] == SEQ_LENGTH_MAX
    assert seq_length_param["q"] == SEQ_LENGTH_STEP
    lo, hi, step = seq_length_param["min"], seq_length_param["max"], seq_length_param["q"]
    grid = tuple(range(lo, hi + 1, step))
    assert grid == SEQ_LENGTH_DOMAIN_V2


def test_five_existing_axes_are_byte_identical_to_v1_production_config():
    v1_config = v1_execution.build_production_sweep_config(program="p")
    v2_config = config_v2.build_production_sweep_config_v2(program="p")
    for axis in _FIVE_AXES:
        assert v2_config["parameters"][axis] == v1_config["parameters"][axis]


def test_v2_config_has_exactly_six_parameters():
    cfg = config_v2.build_production_sweep_config_v2(program="p")
    assert set(cfg["parameters"]) == set(_FIVE_AXES) | {"seq_length"}


def test_method_and_command_shape_match_v1_production_config():
    v1_config = v1_execution.build_production_sweep_config(program="prog")
    v2_config = config_v2.build_production_sweep_config_v2(program="prog")
    assert v2_config["method"] == v1_config["method"] == "bayes"
    assert v2_config["command"] == v1_config["command"] == ["${interpreter}", "${program}"]
    assert v2_config["program"] == v1_config["program"] == "prog"


def test_v2_production_config_never_mentions_the_real_v1_sweep_id():
    cfg = config_v2.build_production_sweep_config_v2(program="p")
    assert FORBIDDEN_V1_SWEEP_ID not in repr(cfg)


def test_rehearsal_config_reuses_the_six_axis_parameters_unchanged():
    production = config_v2.build_production_sweep_config_v2(program="p")
    rehearsal = config_v2.build_wandb_bridge_rehearsal_sweep_config_v2(
        program="p", manifest_path="/abs/path/manifest.json"
    )
    assert rehearsal["parameters"] == production["parameters"]
    assert rehearsal["method"] == production["method"]


def test_rehearsal_config_uses_a_disposable_placeholder_metric():
    rehearsal = config_v2.build_wandb_bridge_rehearsal_sweep_config_v2(
        program="p", manifest_path="/abs/path/manifest.json"
    )
    assert rehearsal["metric"]["name"] == "qualification/rehearsal_placeholder_metric_v2"
    assert rehearsal["metric"]["name"] != config_v2.V2_METRIC_NAME


def test_rehearsal_config_command_embeds_the_manifest_path_as_a_third_token():
    rehearsal = config_v2.build_wandb_bridge_rehearsal_sweep_config_v2(
        program="p", manifest_path="/abs/path/manifest.json"
    )
    assert rehearsal["command"] == ["${interpreter}", "${program}", "/abs/path/manifest.json"]


def test_rehearsal_config_never_mentions_the_real_v1_sweep_id():
    rehearsal = config_v2.build_wandb_bridge_rehearsal_sweep_config_v2(
        program="p", manifest_path="/abs/path/manifest.json"
    )
    assert FORBIDDEN_V1_SWEEP_ID not in repr(rehearsal)


def test_builders_are_pure_and_side_effect_free_across_repeated_calls():
    first = config_v2.build_production_sweep_config_v2(program="p")
    second = config_v2.build_production_sweep_config_v2(program="p")
    assert first == second
    first["parameters"]["seq_length"]["min"] = 999
    # Mutating the first result must never leak into a freshly-built config.
    third = config_v2.build_production_sweep_config_v2(program="p")
    assert third["parameters"]["seq_length"]["min"] == 48


def test_module_never_imports_wandb():
    import sys

    assert "wandb" not in sys.modules or not any(
        name.startswith("wandb") for name in dir(config_v2) if not name.startswith("_")
    )


@pytest.mark.parametrize("bad_program", ["", "another_program.py"])
def test_program_field_is_threaded_through_unchanged(bad_program):
    cfg = config_v2.build_production_sweep_config_v2(program=bad_program)
    assert cfg["program"] == bad_program
