"""Focused tests for the minimal execution-facing campaign declaration
(``src/baseline/campaign_spec.py``, Stage 1 Scope C of the Sequence-Length-A
minimum-viable-infrastructure task).

Covers: structural validation (empty candidates, key/run_id mismatch,
non-positive max_target_epoch, comparator/candidate overlap), and that
construction both succeeds for a well-formed spec and reserves its run_ids
in ``campaign_registry`` (collision detection is exercised end-to-end here,
in addition to ``campaign_registry``'s own direct tests). Each test uses a
unique, test-local ``name``/run_id namespace so tests do not collide with
each other or with the real ``"Sequence-Length-A"`` CampaignSpec constructed
at import time by ``scripts/run_stage1_seq_length_range_seedA_closure.py``.
"""
import pytest

from src.baseline.campaign_registry import CampaignRegistryError, reserved_run_id_index
from src.baseline.campaign_spec import CampaignSpec, CampaignSpecError
from src.baseline.pilot_lead06_config import PilotRunSpec


def _make_run_spec(run_id: str, **overrides) -> PilotRunSpec:
    return PilotRunSpec(
        run_id=run_id,
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=967139,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        **overrides,
    )


def test_well_formed_spec_constructs_and_reserves_run_ids():
    run_id = "test_campaign_spec_wellformed_run_a"
    spec = CampaignSpec(
        name="TestCampaignSpec-WellFormed-001",
        version="v001",
        varied_axis="seq_length",
        candidates={run_id: _make_run_spec(run_id, seq_length=48)},
        max_target_epoch=6,
    )
    assert spec.candidates[run_id].seq_length == 48
    assert reserved_run_id_index()[run_id] == "TestCampaignSpec-WellFormed-001"


def test_empty_candidates_raises():
    with pytest.raises(CampaignSpecError, match="no candidates"):
        CampaignSpec(
            name="TestCampaignSpec-Empty-001",
            version="v001",
            varied_axis="seq_length",
            candidates={},
            max_target_epoch=6,
        )


def test_candidates_key_run_id_mismatch_raises():
    run_id = "test_campaign_spec_mismatch_actual"
    with pytest.raises(CampaignSpecError, match="does not match"):
        CampaignSpec(
            name="TestCampaignSpec-Mismatch-001",
            version="v001",
            varied_axis="seq_length",
            candidates={"test_campaign_spec_mismatch_key": _make_run_spec(run_id, seq_length=24)},
            max_target_epoch=6,
        )


def test_non_pilot_run_spec_candidate_raises():
    with pytest.raises(CampaignSpecError, match="not a PilotRunSpec"):
        CampaignSpec(
            name="TestCampaignSpec-WrongType-001",
            version="v001",
            varied_axis="seq_length",
            candidates={"test_campaign_spec_wrongtype_run": {"seq_length": 24}},
            max_target_epoch=6,
        )


@pytest.mark.parametrize("bad_epoch", [0, -1, 6.0, True])
def test_non_positive_or_non_int_max_target_epoch_raises(bad_epoch):
    run_id = f"test_campaign_spec_badepoch_run_{bad_epoch}".replace(".", "_").replace("-", "neg")
    with pytest.raises(CampaignSpecError, match="max_target_epoch"):
        CampaignSpec(
            name=f"TestCampaignSpec-BadEpoch-{bad_epoch}",
            version="v001",
            varied_axis="seq_length",
            candidates={run_id: _make_run_spec(run_id, seq_length=24)},
            max_target_epoch=bad_epoch,
        )


def test_comparator_overlapping_candidates_raises():
    run_id = "test_campaign_spec_overlap_run"
    with pytest.raises(CampaignSpecError, match="overlaps with candidates"):
        CampaignSpec(
            name="TestCampaignSpec-Overlap-001",
            version="v001",
            varied_axis="seq_length",
            candidates={run_id: _make_run_spec(run_id, seq_length=24)},
            max_target_epoch=6,
            comparator_run_ids=(run_id,),
        )


def test_run_id_collision_with_another_campaign_spec_raises():
    run_id = "test_campaign_spec_collision_shared_run"
    CampaignSpec(
        name="TestCampaignSpec-CollisionFirst-001",
        version="v001",
        varied_axis="seq_length",
        candidates={run_id: _make_run_spec(run_id, seq_length=24)},
        max_target_epoch=6,
    )
    with pytest.raises(CampaignRegistryError, match="collides with"):
        CampaignSpec(
            name="TestCampaignSpec-CollisionSecond-001",
            version="v001",
            varied_axis="seq_length",
            candidates={run_id: _make_run_spec(run_id, seq_length=48)},
            max_target_epoch=6,
        )


def test_comparator_run_ids_are_not_registered_as_candidates():
    candidate_run_id = "test_campaign_spec_comparator_candidate_run"
    comparator_run_id = "test_campaign_spec_comparator_reference_run"
    CampaignSpec(
        name="TestCampaignSpec-ComparatorNotRegistered-001",
        version="v001",
        varied_axis="seq_length",
        candidates={candidate_run_id: _make_run_spec(candidate_run_id, seq_length=24)},
        max_target_epoch=6,
        comparator_run_ids=(comparator_run_id,),
    )
    assert comparator_run_id not in reserved_run_id_index()
