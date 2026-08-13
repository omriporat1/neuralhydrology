"""Focused tests for the central reserved-run-ID registry
(``src/baseline/campaign_registry.py``, Stage 1 Scope B of the Sequence-
Length-A minimum-viable-infrastructure task).

Covers: historical-group internal consistency, collision detection against
both historical and prospective groups, idempotent re-registration, same-
source/different-run_ids rejection, and within-group duplicate rejection.
Each test that registers a prospective group uses a unique, test-local
``source`` label so tests do not collide with each other or with the real
``"Sequence-Length-A"`` registration performed at import time by
``scripts/run_stage1_seq_length_range_seedA_closure.py`` (module-level
process-lifetime state, per the module's own docstring).
"""
import pytest

from src.baseline.campaign_registry import (
    CampaignRegistryError,
    HISTORICAL_RESERVED_RUN_ID_GROUPS,
    reserved_run_id_index,
    register_prospective_campaign_run_ids,
)


def test_historical_groups_have_no_internal_collisions():
    index = reserved_run_id_index()
    n_historical_run_ids = sum(len(g.run_ids) for g in HISTORICAL_RESERVED_RUN_ID_GROUPS)
    assert len(index) >= n_historical_run_ids
    for group in HISTORICAL_RESERVED_RUN_ID_GROUPS:
        assert group.status == "historical"
        for run_id in group.run_ids:
            assert index[run_id] == group.source


def test_register_new_prospective_campaign_succeeds():
    group = register_prospective_campaign_run_ids(
        "TestCampaign-Fresh-001", ("test_fresh_run_a", "test_fresh_run_b")
    )
    assert group.status == "prospective"
    assert group.run_ids == ("test_fresh_run_a", "test_fresh_run_b")
    index = reserved_run_id_index()
    assert index["test_fresh_run_a"] == "TestCampaign-Fresh-001"
    assert index["test_fresh_run_b"] == "TestCampaign-Fresh-001"


def test_register_colliding_with_historical_run_id_raises():
    historical_run_id = HISTORICAL_RESERVED_RUN_ID_GROUPS[0].run_ids[0]
    with pytest.raises(CampaignRegistryError, match="collides with"):
        register_prospective_campaign_run_ids(
            "TestCampaign-CollidesHistorical-001", (historical_run_id,)
        )


def test_register_colliding_with_other_prospective_group_raises():
    register_prospective_campaign_run_ids(
        "TestCampaign-Prospective-First-001", ("test_shared_run_id_001",)
    )
    with pytest.raises(CampaignRegistryError, match="collides with"):
        register_prospective_campaign_run_ids(
            "TestCampaign-Prospective-Second-001", ("test_shared_run_id_001",)
        )


def test_register_duplicate_run_id_within_same_call_raises():
    with pytest.raises(CampaignRegistryError, match="duplicate run_id"):
        register_prospective_campaign_run_ids(
            "TestCampaign-InternalDup-001", ("test_dup_run", "test_dup_run")
        )


def test_reregistering_identical_source_and_run_ids_is_idempotent():
    first = register_prospective_campaign_run_ids(
        "TestCampaign-Idempotent-001", ("test_idem_run_a",)
    )
    second = register_prospective_campaign_run_ids(
        "TestCampaign-Idempotent-001", ("test_idem_run_a",)
    )
    assert first == second


def test_reregistering_same_source_with_different_run_ids_raises():
    register_prospective_campaign_run_ids(
        "TestCampaign-Contradicts-001", ("test_orig_run",)
    )
    with pytest.raises(CampaignRegistryError, match="already registered"):
        register_prospective_campaign_run_ids(
            "TestCampaign-Contradicts-001", ("test_different_run",)
        )


def test_reserved_run_id_index_reflects_registrations_across_calls():
    before = set(reserved_run_id_index())
    register_prospective_campaign_run_ids(
        "TestCampaign-IndexGrowth-001", ("test_index_growth_run",)
    )
    after = set(reserved_run_id_index())
    assert after - before == {"test_index_growth_run"}
