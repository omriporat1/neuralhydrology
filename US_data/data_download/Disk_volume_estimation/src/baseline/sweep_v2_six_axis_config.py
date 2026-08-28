"""v2 six-axis W&B sweep configuration builders (Section C, additive
six-axis campaign foundation).

Mirrors :func:`src.baseline.sweep_v1_execution.build_production_sweep_config`
/ :func:`build_wandb_bridge_rehearsal_sweep_config` exactly in shape and
intent -- a pure, side-effect-free dict builder, never a W&B API call. No
sweep is created, registered, or contacted by importing or calling anything
in this module.

Reuses the five existing (learning_rate/hidden_size/embedding_dropout/
output_dropout/batch_size) parameter distributions byte-for-byte from the
v1 production config (v1's own builder is called and only extended, so the
five existing domains can never silently drift from v1's). Adds the sixth
axis, ``seq_length``, using the user's binding ``q_uniform`` wire
representation (min=48, max=120, q=12 -- see
:mod:`src.baseline.sweep_v2_six_axis_campaign`'s
``SEQ_LENGTH_MIN``/``SEQ_LENGTH_MAX``/``SEQ_LENGTH_STEP``, so the wire
config can never drift from the normalization contract's own bounds).

Uses a new v2 objective metric name and never references v1's production
sweep id (``4x3btz2s``) anywhere -- there is no sweep-id parameter on either
builder here at all.
"""
from __future__ import annotations

from typing import Any

from .sweep_v1_execution import build_production_sweep_config
from .sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2, SEQ_LENGTH_MAX, SEQ_LENGTH_MIN, SEQ_LENGTH_STEP

__all__ = [
    "V2_METRIC_NAME",
    "V2_REHEARSAL_PLACEHOLDER_METRIC_NAME",
    "build_production_sweep_config_v2",
    "build_wandb_bridge_rehearsal_sweep_config_v2",
]

V2_METRIC_NAME = f"flashnh/{OBJECTIVE_ID_V2}"
V2_REHEARSAL_PLACEHOLDER_METRIC_NAME = "qualification/rehearsal_placeholder_metric_v2"


def build_production_sweep_config_v2(*, program: str) -> dict[str, Any]:
    """Six-axis production W&B sweep config. Starts from the exact v1
    production config (same ``method``, same ``command`` shape, same five
    existing parameter distributions -- see
    :func:`src.baseline.sweep_v1_execution.build_production_sweep_config`),
    then: (1) replaces ``metric`` with the new v2 objective name
    (:data:`V2_METRIC_NAME`, distinct from v1's ``flashnh/best_score`` --
    the v2 primary objective is the fixed-support raw-space NSE, a
    different scientific quantity, and must never be confused with v1's
    natural-support objective downstream in W&B); (2) adds the sixth axis,
    ``seq_length``, as ``q_uniform`` per the user's binding six-axis wire
    representation mandate. The five existing axes' ``parameters`` entries
    are untouched dict values copied from v1's own builder output --
    verified identical by :func:`_assert_five_axes_match_v1` at import time
    below, not merely by construction.
    """
    config = build_production_sweep_config(program=program)
    config["metric"] = {"name": V2_METRIC_NAME, "goal": "maximize"}
    config["parameters"] = dict(config["parameters"])
    config["parameters"]["seq_length"] = {
        "distribution": "q_uniform",
        "min": SEQ_LENGTH_MIN,
        "max": SEQ_LENGTH_MAX,
        "q": SEQ_LENGTH_STEP,
    }
    return config


def build_wandb_bridge_rehearsal_sweep_config_v2(*, program: str, manifest_path: str) -> dict[str, Any]:
    """Disposable-sweep sibling of :func:`build_production_sweep_config_v2`,
    mirroring :func:`src.baseline.sweep_v1_execution.build_wandb_bridge_rehearsal_sweep_config`'s
    two-field divergence exactly: a disposable, non-scientific placeholder
    ``metric`` (this config's run stops before any objective could be
    computed) and a ``command`` carrying one literal, static extra
    positional argument -- the absolute path to a pre-built v2 rehearsal
    launch manifest. Does not create or contact a real W&B sweep."""
    config = build_production_sweep_config_v2(program=program)
    config["metric"] = {"name": V2_REHEARSAL_PLACEHOLDER_METRIC_NAME, "goal": "maximize"}
    config["command"] = ["${interpreter}", "${program}", manifest_path]
    return config


def _assert_five_axes_match_v1() -> None:
    v1_config = build_production_sweep_config(program="__contract_check__")
    v2_config = build_production_sweep_config_v2(program="__contract_check__")
    for axis in ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size"):
        if v2_config["parameters"][axis] != v1_config["parameters"][axis]:
            raise AssertionError(
                f"v2 sweep config axis {axis!r} diverged from v1's own production config: "
                f"v1={v1_config['parameters'][axis]!r} v2={v2_config['parameters'][axis]!r}"
            )
    if "seq_length" in v1_config["parameters"]:
        raise AssertionError("v1 production sweep config must never contain a seq_length axis")
    if v2_config["method"] != v1_config["method"]:
        raise AssertionError("v2 sweep method must match v1's (bayes)")


_assert_five_axes_match_v1()
