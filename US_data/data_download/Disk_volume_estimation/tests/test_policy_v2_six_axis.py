"""Local tests for the v2 six-axis seq_length policy overlay loader/
validator (Section I, additive six-axis campaign foundation).

``src.baseline.policy_v2_six_axis`` had no dedicated test file: its happy
path (``load_stage1_baseline_policy_v2_six_axis`` against the real
committed base policy + overlay) is exercised transitively by
``tests/test_sweep_v2_six_axis_production_adapter.py`` via
``prepare_bayesian_proposal_v2``, but every negative/validation path of
``validate_v2_six_axis_policy_overlay`` and the cross-check logic in
``load_stage1_baseline_policy_v2_six_axis`` had zero coverage. These tests
close that gap using the real committed
``config/stage1_scientific_baseline_v001.yaml`` and
``config/stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml`` files
(never manufactured base-policy fixtures for the positive-path tests), plus
small in-memory / tmp-path variants for the negative paths.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.baseline.policy_v2_six_axis import (
    PolicyOverlayError,
    load_stage1_baseline_policy_v2_six_axis,
    validate_v2_six_axis_policy_overlay,
)
from src.baseline.sweep_v2_six_axis_campaign import SEQ_LENGTH_DOMAIN_V2
from tests._pilot_support import BASELINE_POLICY_PATH

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REAL_OVERLAY_PATH = _REPO_ROOT / "config" / "stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml"


def _real_overlay_data() -> dict:
    return yaml.safe_load(_REAL_OVERLAY_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Positive path, against the real committed files
# ---------------------------------------------------------------------------

def test_real_overlay_file_exists_and_validates():
    validate_v2_six_axis_policy_overlay(_real_overlay_data())


def test_load_against_the_real_committed_base_policy_and_overlay():
    merged = load_stage1_baseline_policy_v2_six_axis(BASELINE_POLICY_PATH, _REAL_OVERLAY_PATH)
    assert tuple(merged["seq_lengths_hours"]) == SEQ_LENGTH_DOMAIN_V2 == (48, 60, 72, 84, 96, 108, 120)
    assert merged["policy_overlay"] == {
        "overlay_name": "stage1_scientific_baseline_v2_six_axis_overlay_v001",
        "overlay_version": 1,
        "base_policy_name": "stage1_scientific_baseline_v001",
        "base_policy_version": 2,
    }
    # Every other v1-validated key survives untouched.
    assert merged["policy_name"] == "stage1_scientific_baseline_v001"
    assert merged["policy_version"] == 2


def test_load_never_mutates_the_v1_committed_overlay_or_base_policy_files():
    before_overlay = _REAL_OVERLAY_PATH.read_text(encoding="utf-8")
    before_base = BASELINE_POLICY_PATH.read_text(encoding="utf-8")
    load_stage1_baseline_policy_v2_six_axis(BASELINE_POLICY_PATH, _REAL_OVERLAY_PATH)
    assert _REAL_OVERLAY_PATH.read_text(encoding="utf-8") == before_overlay
    assert BASELINE_POLICY_PATH.read_text(encoding="utf-8") == before_base


def test_returned_dict_is_a_copy_not_an_alias_of_a_cached_base_policy():
    first = load_stage1_baseline_policy_v2_six_axis(BASELINE_POLICY_PATH, _REAL_OVERLAY_PATH)
    first["seq_lengths_hours"].append(999)
    second = load_stage1_baseline_policy_v2_six_axis(BASELINE_POLICY_PATH, _REAL_OVERLAY_PATH)
    assert tuple(second["seq_lengths_hours"]) == SEQ_LENGTH_DOMAIN_V2


# ---------------------------------------------------------------------------
# validate_v2_six_axis_policy_overlay: negative paths (in-memory, no filesystem)
# ---------------------------------------------------------------------------

def test_rejects_non_dict_top_level():
    with pytest.raises(PolicyOverlayError, match="top level must be a mapping"):
        validate_v2_six_axis_policy_overlay(["not", "a", "dict"])


@pytest.mark.parametrize(
    "key", ["overlay_name", "overlay_version", "base_policy_name", "base_policy_version", "seq_lengths_hours"]
)
def test_rejects_missing_required_key(key):
    data = _real_overlay_data()
    del data[key]
    with pytest.raises(PolicyOverlayError):
        validate_v2_six_axis_policy_overlay(data)


def test_rejects_wrong_overlay_name():
    data = _real_overlay_data()
    data["overlay_name"] = "some_other_overlay_v001"
    with pytest.raises(PolicyOverlayError, match="overlay_name"):
        validate_v2_six_axis_policy_overlay(data)


def test_rejects_wrong_overlay_version():
    data = _real_overlay_data()
    data["overlay_version"] = 2
    with pytest.raises(PolicyOverlayError, match="overlay_version"):
        validate_v2_six_axis_policy_overlay(data)


def test_rejects_wrong_base_policy_name():
    data = _real_overlay_data()
    data["base_policy_name"] = "stage1_scientific_baseline_v999"
    with pytest.raises(PolicyOverlayError, match="base_policy_name"):
        validate_v2_six_axis_policy_overlay(data)


def test_rejects_wrong_base_policy_version():
    data = _real_overlay_data()
    data["base_policy_version"] = 1
    with pytest.raises(PolicyOverlayError, match="base_policy_version"):
        validate_v2_six_axis_policy_overlay(data)


@pytest.mark.parametrize(
    "bad_seq_lengths",
    [
        [48, 60, 72, 84, 96, 108],  # missing 120
        [48, 60, 72, 84, 96, 108, 120, 132],  # extra out-of-range value
        [60, 48, 72, 84, 96, 108, 120],  # wrong order
        [12, 24, 48, 72],  # this is v1's own domain, must never be silently accepted here
        "not-a-list",
        None,
    ],
)
def test_rejects_seq_lengths_hours_that_do_not_exactly_equal_v2_domain(bad_seq_lengths):
    data = _real_overlay_data()
    data["seq_lengths_hours"] = bad_seq_lengths
    with pytest.raises(PolicyOverlayError, match="seq_lengths_hours"):
        validate_v2_six_axis_policy_overlay(data)


def test_rejects_seq_lengths_hours_containing_a_bool():
    data = _real_overlay_data()
    data["seq_lengths_hours"] = [48, 60, 72, 84, 96, 108, True]
    with pytest.raises(PolicyOverlayError, match="seq_lengths_hours"):
        validate_v2_six_axis_policy_overlay(data)


def test_accepts_the_unmodified_real_overlay_data_verbatim():
    # Guards against the negative-path tests above accidentally mutating a
    # shared dict: a freshly re-read copy must still validate cleanly.
    validate_v2_six_axis_policy_overlay(_real_overlay_data())


# ---------------------------------------------------------------------------
# load_stage1_baseline_policy_v2_six_axis: filesystem-facing negative paths
# ---------------------------------------------------------------------------

def test_rejects_missing_overlay_file(tmp_path):
    with pytest.raises(PolicyOverlayError, match="not found"):
        load_stage1_baseline_policy_v2_six_axis(BASELINE_POLICY_PATH, tmp_path / "does_not_exist.yaml")


def test_rejects_empty_overlay_file(tmp_path):
    empty = tmp_path / "empty_overlay.yaml"
    empty.write_text("   \n", encoding="utf-8")
    with pytest.raises(PolicyOverlayError, match="empty"):
        load_stage1_baseline_policy_v2_six_axis(BASELINE_POLICY_PATH, empty)


def test_rejects_missing_base_policy_file(tmp_path):
    with pytest.raises(PolicyOverlayError, match="base v1 policy failed validation"):
        load_stage1_baseline_policy_v2_six_axis(tmp_path / "does_not_exist.yaml", _REAL_OVERLAY_PATH)


def test_rejects_base_policy_that_fails_v1_validation(tmp_path):
    bad_base = tmp_path / "bad_base_policy.yaml"
    bad_base.write_text("policy_name: not_the_real_policy\n", encoding="utf-8")
    with pytest.raises(PolicyOverlayError, match="base v1 policy failed validation"):
        load_stage1_baseline_policy_v2_six_axis(bad_base, _REAL_OVERLAY_PATH)


def test_load_rejects_an_overlay_with_a_wrong_declared_base_policy_version(tmp_path):
    """Exercises the same base_policy_version pin as
    ``test_rejects_wrong_base_policy_version`` but through the full
    filesystem-facing ``load_stage1_baseline_policy_v2_six_axis`` path
    (real base policy file + tmp-path overlay), not just the in-memory
    validator directly."""
    data = _real_overlay_data()
    data["base_policy_version"] = 999
    overlay = tmp_path / "mismatched_overlay.yaml"
    overlay.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(PolicyOverlayError, match="base_policy_version"):
        load_stage1_baseline_policy_v2_six_axis(BASELINE_POLICY_PATH, overlay)


def test_overlay_never_authorizes_a_seq_length_outside_its_own_frozen_domain():
    merged = load_stage1_baseline_policy_v2_six_axis(BASELINE_POLICY_PATH, _REAL_OVERLAY_PATH)
    assert 144 not in merged["seq_lengths_hours"]
    assert 36 not in merged["seq_lengths_hours"]
