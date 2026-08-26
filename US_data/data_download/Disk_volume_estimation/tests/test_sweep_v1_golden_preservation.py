"""Golden v1 five-axis preservation tests (additive six-axis campaign
foundation, Section A).

Unlike every other test file touched this session -- which uses small,
freshly hand-built trial tables -- these tests pin the REAL recorded
production identity of Sweep-v1's sole accepted result (Proposal 1 /
``ardib08c``, sweep ``omri-porat1-huji/flashnh-stage1/4x3btz2s``, accepted
objective ``0.391678449944578``) against v1's identity functions as they
exist in the tree TODAY. The hyperparameters and identity strings below are
copied verbatim from the pre-existing, already-checksummed local evidence
bundle at
``.scratch_local/sweep_v1_completion_audits/attempt005_completion_audit_v001/
remote_copies/attempt005_completion_audit_bundle.json`` (lines 770-825) --
not re-derived, not manufactured, and not altered in any way by this task.

If any assertion here ever fails, it means v1's frozen identity math
(``canonical_hyperparameters``/``configuration_id``/``trial_id``/
``proposal_id``) has silently drifted from what actually produced the real,
already-accepted Sweep-v1 result -- exactly the kind of five-axis-system
corruption the binding six-axis task's preservation contract forbids. This
file makes no six-axis assertion of its own; it exists purely to catch v1
regression, including regression accidentally introduced by any v2 sibling
module reusing or extending v1 code.
"""
from __future__ import annotations

from src.baseline import sweep_v1_campaign as sweep
from src.baseline import sweep_v2_six_axis_campaign as sweep_v2

# Real recorded five-axis hyperparameters of Sweep-v1 Proposal 1
# (seq_length=72h was v1's frozen fixed configuration, not a swept axis).
_PROPOSAL_1_HYPERPARAMETERS = {
    "learning_rate": float("0.00024474898782657741"),
    "hidden_size": 64,
    "embedding_dropout": float("0.08637149503762416"),
    "output_dropout": float("0.20096018948154892"),
    "batch_size": 512,
}

# Real recorded identity strings for the same proposal/configuration, copied
# verbatim from the evidence bundle referenced in the module docstring.
_REAL_CONFIGURATION_ID = "sweep_v1_cfg_5731e180d1bf9d582afc"
_REAL_PROPOSAL_ID = "stage1_phase_b_sweep_v1_original_domain_v001__bayesian__proposal001"
_REAL_TRIAL_ID_ATTEMPT001 = (
    "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc"
    "__mf12x50000__seedA967139__attempt001"
)
_REAL_TRIAL_ID_ATTEMPT005 = (
    "stage1_phase_b_sweep_v1_original_domain_v001__sweep_v1_cfg_5731e180d1bf9d582afc"
    "__mf12x50000__seedA967139__attempt005"
)
_REAL_WANDB_SWEEP_ID = "4x3btz2s"
_REAL_ACCEPTED_RUN_ID = "ardib08c"
_REAL_ACCEPTED_OBJECTIVE = 0.391678449944578


def test_configuration_id_reproduces_the_real_accepted_proposal_hash():
    assert sweep.configuration_id(_PROPOSAL_1_HYPERPARAMETERS) == _REAL_CONFIGURATION_ID


def test_proposal_id_reproduces_the_real_accepted_proposal_string():
    assert sweep.proposal_id("bayesian", 1) == _REAL_PROPOSAL_ID


def test_trial_id_reproduces_the_real_attempt001_and_attempt005_strings():
    assert sweep.trial_id(_REAL_CONFIGURATION_ID, execution_generation=1) == _REAL_TRIAL_ID_ATTEMPT001
    assert sweep.trial_id(_REAL_CONFIGURATION_ID, execution_generation=5) == _REAL_TRIAL_ID_ATTEMPT005
    # Same result whether trial_id is given the raw hyperparameters or the
    # already-hashed configuration_id -- both paths must agree.
    assert sweep.trial_id(_PROPOSAL_1_HYPERPARAMETERS, execution_generation=5) == _REAL_TRIAL_ID_ATTEMPT005


def test_frozen_identity_constants_match_the_real_accepted_run():
    assert sweep.MODEL_SEED_A == 967139
    assert sweep.CAMPAIGN_ID == "stage1_phase_b_sweep_v1_original_domain_v001"
    assert sweep.DOMAIN_VERSION == "original_domain_v001"
    assert f"seedA{sweep.MODEL_SEED_A}" in _REAL_TRIAL_ID_ATTEMPT005
    assert sweep.CAMPAIGN_ID in _REAL_TRIAL_ID_ATTEMPT005
    assert sweep.CAMPAIGN_ID in _REAL_PROPOSAL_ID


def test_real_proposal_1_hyperparameters_remain_inside_the_frozen_v1_domain():
    """Confirms v1's SEARCH_DOMAIN bounds were not narrowed/widened in a way
    that would now reject the real, already-accepted proposal."""
    canonical = sweep.canonical_hyperparameters(_PROPOSAL_1_HYPERPARAMETERS)
    assert canonical["hidden_size"] == 64
    assert canonical["batch_size"] == 512


def test_v2_forbidden_sweep_id_matches_the_real_production_sweep():
    assert sweep_v2.FORBIDDEN_V1_SWEEP_ID == _REAL_WANDB_SWEEP_ID


def test_v2_contamination_guard_refuses_the_real_v1_sweep_campaign_and_domain():
    import pytest
    from src.baseline.sweep_v2_six_axis_campaign import SweepV2CampaignError

    with pytest.raises(SweepV2CampaignError, match=_REAL_WANDB_SWEEP_ID):
        sweep_v2.assert_no_v1_contamination(wandb_sweep_id=_REAL_WANDB_SWEEP_ID)
    with pytest.raises(SweepV2CampaignError):
        sweep_v2.assert_no_v1_contamination(campaign_id=sweep.CAMPAIGN_ID)
    with pytest.raises(SweepV2CampaignError):
        sweep_v2.assert_no_v1_contamination(domain_version=sweep.DOMAIN_VERSION)
    # A genuinely foreign (v2) identity must never be refused.
    sweep_v2.assert_no_v1_contamination(
        wandb_sweep_id="some_other_sweep_id",
        campaign_id=sweep_v2.CAMPAIGN_ID_V2,
        domain_version=sweep_v2.DOMAIN_VERSION_V2,
    )


def test_v2_configuration_id_never_collides_with_the_real_v1_configuration_id():
    """Same five axes (plus seq_length=72, v1's frozen fixed value) run
    through the v2 six-axis identity path must NEVER reproduce v1's real
    production configuration_id -- confirms the two hash namespaces
    (``sweep_v1_cfg_`` vs ``sweep_v2_cfg_``, different payload shape) never
    accidentally collide even at the exact real historical coordinate."""
    six_axis = {**_PROPOSAL_1_HYPERPARAMETERS, "seq_length": 72}
    v2_id = sweep_v2.configuration_id_v2(
        six_axis, support_contract_version="common120_raw_space_nse_v001", support_contract_sha256="a" * 64
    )
    assert v2_id != _REAL_CONFIGURATION_ID
    assert v2_id.startswith("sweep_v2_cfg_")
    assert _REAL_CONFIGURATION_ID.startswith("sweep_v1_cfg_")


def test_real_accepted_run_identity_values_are_stable_string_literals():
    """No computation -- just guards these three literal identifiers (the
    ones every other golden/contamination check in this file and in
    Section G/D's test suites is built around) against an accidental typo
    or silent edit inside this test file itself, since they are re-typed
    by hand in several other test modules across this session's work."""
    assert _REAL_ACCEPTED_RUN_ID == "ardib08c"
    assert _REAL_ACCEPTED_OBJECTIVE == 0.391678449944578
    assert _REAL_WANDB_SWEEP_ID == "4x3btz2s"
