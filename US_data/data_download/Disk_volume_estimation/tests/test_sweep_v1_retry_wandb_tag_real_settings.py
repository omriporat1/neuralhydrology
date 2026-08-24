"""Real, installed-``wandb``-package validation of the exact-retry tag
contract, complementing the fake-module tests in ``tests/test_sweep_v1_retry.py``
and ``tests/test_sweep_v1_exact_retry_bridge.py``.

The attempt002/job 45939764 incident was a REAL rejection inside
``wandb.init()``'s own ``Settings`` pydantic validation (a 125-character
``retry_of_<trial_id>`` tag), not something a fake ``wandb`` module can prove
one way or the other. These tests instantiate the real, installed
``wandb.sdk.wandb_settings.Settings`` model directly -- no ``wandb.init()``
call, no network, no run creation -- and assert:

  1. the real production bounded tag set for the real attempt001 -> attempt003
     retry identity is ACCEPTED;
  2. boundary lengths 63 and 64 characters are ACCEPTED;
  3. boundary length 65 characters is REJECTED (matching the real failure
     category from job 45939764, just below the length that actually failed);
  4. the real historical offending tag (``retry_of_<attempt001 trial_id>``,
     125 characters) is REJECTED.

Skipped entirely (via ``pytest.importorskip``) in any environment without a
real ``wandb`` install -- e.g. the local dev environment used for this repo's
day-to-day test runs. Must be exercised for real on Moriah (wandb 0.28.1,
per the qualification precedent in
``scripts/wandb_exact_retry_join_qualification.py``) before attempt003 is
submitted.
"""
from __future__ import annotations

import pytest

from src.baseline.sweep_v1_retry import MAX_WANDB_TAG_LENGTH, build_bounded_wandb_tags, validate_wandb_tags

wandb = pytest.importorskip("wandb")

_REAL_ATTEMPT001_TRIAL_ID = (
    "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc__mf12x50000__seedA967139__attempt001"
)
_REAL_OFFENDING_ATTEMPT002_TAG = f"retry_of_{_REAL_ATTEMPT001_TRIAL_ID}"

_REAL_PRODUCTION_ATTEMPT003_TAGS = build_bounded_wandb_tags(
    proposal_order=1, execution_generation=3, configuration_id="sweep_v1_cfg_5731e180d1bf9d582afc",
)


def _settings_accepts(tags: "list[str]") -> None:
    """Instantiate the REAL wandb Settings model with run_tags=tags -- the
    exact assignment path that raised inside wandb.init() during job
    45939764. No wandb.init(), no network."""
    settings = wandb.Settings()
    settings.run_tags = tuple(tags)


def test_real_wandb_settings_accepts_the_production_bounded_tag_set_for_real_attempt003_identity():
    assert len(_REAL_OFFENDING_ATTEMPT002_TAG) > MAX_WANDB_TAG_LENGTH  # sanity: this is really the incident's shape
    validate_wandb_tags(_REAL_PRODUCTION_ATTEMPT003_TAGS)  # our own guard agrees
    _settings_accepts(_REAL_PRODUCTION_ATTEMPT003_TAGS)  # the real wandb package agrees


def test_real_wandb_settings_accepts_boundary_63_and_64_character_tags():
    tag_63 = "x" * 63
    tag_64 = "x" * 64
    validate_wandb_tags([tag_63, tag_64])
    _settings_accepts([tag_63, tag_64])


def test_real_wandb_settings_rejects_boundary_65_character_tag():
    tag_65 = "x" * 65
    with pytest.raises(Exception):
        _settings_accepts([tag_65])


def test_real_wandb_settings_rejects_the_real_historical_offending_attempt002_tag():
    with pytest.raises(Exception):
        _settings_accepts([_REAL_OFFENDING_ATTEMPT002_TAG])
