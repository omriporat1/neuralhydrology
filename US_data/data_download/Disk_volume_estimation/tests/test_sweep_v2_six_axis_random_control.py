"""Offline contract tests for the frozen Phase-B Sweep-v2 six-axis IID
random-control manifest.

Scientific framing under test: the manifest was **frozen after Bayesian
observation 1**, drawn only from the exact committed v2 six-axis priors, and
is scientifically independent of Proposal 1. These tests protect the seed
derivation, one-shot deterministic realisation, exact size/ordering,
canonical checksum, six-axis domain legality, the ``q_uniform`` seq_length
semantics, the absence of any redraw/dedup/rebalance, the stable
configuration/proposal/trial identity grammar, separation from the live
Bayesian controller and every forbidden production sweep id, the exact v2
fidelity and Common-120 identity, and collision-free concurrent launch
construction.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest

from src.baseline import sweep_v1_campaign as sweep_v1
from src.baseline import sweep_v2_six_axis_random_control as rc
from src.baseline.sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2,
    DOMAIN_VERSION_V2,
    FIDELITY_ID_V2,
    FORBIDDEN_PRODUCTION_SWEEP_IDS,
    OBJECTIVE_ID_V2,
    SEQ_LENGTH_DOMAIN_V2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
)

MANIFEST_PATH = Path(__file__).parents[1] / "config" / (
    "stage1_phase_b_sweep_v2_six_axis_random_control_v001_random_control_manifest.json"
)
MANIFEST_SHA_PATH = MANIFEST_PATH.with_suffix(".sha256")

# The live production Bayesian controller run -- this arm must never name or
# contact it.
_BAYESIAN_CONTROLLER_RUN_ID = "wta85z3b"


def _reference_rows() -> list[dict]:
    """Re-derive the 12 rows independently, in the pinned per-row draw order
    (learning_rate, hidden_size, embedding_dropout, output_dropout,
    batch_size, seq_length), without touching the module's generator."""
    rng = random.Random(158987734)
    log_lower, log_upper = math.log10(1e-4), math.log10(1e-3)
    rows = []
    for _ in range(12):
        lr = 10 ** rng.uniform(log_lower, log_upper)
        hidden = rng.choice([64, 128, 256])
        emb = rng.uniform(0.0, 0.4)
        out = rng.uniform(0.0, 0.4)
        batch = rng.choice([128, 256, 512])
        raw_seq = rng.uniform(48, 120)
        seq = int(round(raw_seq / 12) * 12)
        rows.append({"learning_rate": lr, "hidden_size": hidden, "embedding_dropout": emb,
                     "output_dropout": out, "batch_size": batch, "seq_length": seq})
    return rows


def test_seed_derivation_is_exact_and_distinct_from_reserved_seeds():
    derived = rc.derive_manifest_rng_seed()
    assert derived["namespace"] == "stage1_phase_b_sweep_v2_six_axis_random_control_v001"
    assert derived["digest"] == "0979f5d60aa60db35d6f0b5c248bfdf73ac24b734b1b5fcb9753db8517299ea2"
    assert derived["seed_hex_prefix8"] == "0979f5d6"
    assert derived["seed"] == int("0979f5d6", 16) == 158987734 == rc.MANIFEST_RNG_SEED_V2
    # Independent recomputation from raw bytes.
    import hashlib
    assert hashlib.sha256(derived["namespace"].encode("utf-8")).hexdigest() == derived["digest"]
    # Distinct from model Seed A and the v1 five-axis manifest seed.
    assert rc.MANIFEST_RNG_SEED_V2 != sweep_v1.MODEL_SEED_A == 967139
    assert rc.MANIFEST_RNG_SEED_V2 != sweep_v1.MANIFEST_RNG_SEED == 20260822


def test_seed_collision_with_a_reserved_seed_is_a_hard_stop_with_no_fallback(monkeypatch):
    # If the derived seed ever equalled a reserved seed, derivation must raise
    # and never fall back to another seed.
    monkeypatch.setattr(rc, "_MODEL_SEED_A", rc.MANIFEST_RNG_SEED_V2)
    with pytest.raises(rc.SweepV2RandomControlError, match="no fallback seed is permitted"):
        rc.derive_manifest_rng_seed()


def test_manifest_is_exactly_reproducible_and_checksum_pinned():
    committed = MANIFEST_PATH.read_bytes()
    assert committed == rc.render_manifest_v2()
    # Regeneration is stable across repeated calls (no RNG-state leakage).
    assert rc.render_manifest_v2() == rc.render_manifest_v2()
    assert rc.sha256_bytes(committed) == rc.RANDOM_CONTROL_MANIFEST_SHA256_V2
    assert MANIFEST_SHA_PATH.read_text(encoding="utf-8").split()[0] == rc.RANDOM_CONTROL_MANIFEST_SHA256_V2
    payload = json.loads(committed)
    assert payload["manifest_rng_seed"] == 158987734
    assert payload["frozen_after_bayesian_observation"] == 1
    assert payload["prospective_pre_outcome_frozen"] is False
    assert payload["per_row_draw_order"] == [
        "learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size", "seq_length"]
    assert payload["seq_length_sampler"] == "int(round(uniform(min, max) / q) * q)"
    rows = rc.generate_random_control_rows_v2()
    assert payload["rows"] == rows
    rc.validate_manifest_rows_v2(rows)


def test_manifest_is_exactly_twelve_rows_in_frozen_order():
    rows = rc.generate_random_control_rows_v2()
    assert len(rows) == 12
    assert [row["manifest_index"] for row in rows] == list(range(1, 13))
    assert [row["proposal_order"] for row in rows] == list(range(1, 13))
    assert all(row["search_arm"] == "random_control" for row in rows)
    assert all(row["manifest_rng_seed"] == 158987734 for row in rows)


def test_rows_match_independent_iid_draw_in_pinned_order_and_legal_domain():
    rows = rc.generate_random_control_rows_v2()
    reference = _reference_rows()
    for row, ref in zip(rows, reference):
        assert float(row["learning_rate"]) == pytest.approx(ref["learning_rate"], rel=0, abs=0)
        assert row["hidden_size"] == ref["hidden_size"]
        assert float(row["embedding_dropout"]) == pytest.approx(ref["embedding_dropout"], rel=0, abs=0)
        assert float(row["output_dropout"]) == pytest.approx(ref["output_dropout"], rel=0, abs=0)
        assert row["batch_size"] == ref["batch_size"]
        assert row["seq_length"] == ref["seq_length"]
    assert all(1e-4 <= float(row["learning_rate"]) <= 1e-3 for row in rows)
    assert all(0.0 <= float(row[axis]) <= 0.4 for row in rows for axis in ("embedding_dropout", "output_dropout"))
    assert {row["hidden_size"] for row in rows} <= {64, 128, 256}
    assert {row["batch_size"] for row in rows} <= {128, 256, 512}
    assert {row["seq_length"] for row in rows} <= set(SEQ_LENGTH_DOMAIN_V2) == {48, 60, 72, 84, 96, 108, 120}


def test_seq_length_uses_q_uniform_semantics_not_a_categorical_prior():
    # q_uniform((48,120),12): X ~ uniform(48,120) then round(X/12)*12. Endpoints
    # 48 and 120 carry half the mass of an interior grid value -- structurally
    # different from a categorical 1/7-each draw.
    assert rc.SEQ_LENGTH_SAMPLER_V2 == "int(round(uniform(min, max) / q) * q)"
    probe = random.Random(999)
    expected = random.Random(999)
    for _ in range(50):
        drawn = rc.draw_seq_length_q_uniform(probe)
        assert drawn == int(round(expected.uniform(48, 120) / 12) * 12)
        assert drawn in SEQ_LENGTH_DOMAIN_V2
    # A large Monte-Carlo estimate of the endpoint deficit relative to interior.
    sampler = random.Random(12345)
    counts = {value: 0 for value in SEQ_LENGTH_DOMAIN_V2}
    for _ in range(120_000):
        counts[int(round(sampler.uniform(48, 120) / 12) * 12)] += 1
    interior = sum(counts[v] for v in (60, 72, 84, 96, 108)) / 5
    assert counts[48] < 0.75 * interior and counts[120] < 0.75 * interior


def test_no_redraw_dedup_or_rebalancing_is_hidden_in_validation():
    rows = rc.generate_random_control_rows_v2()
    axes = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size", "seq_length")
    forced = {key: rows[0][key] for key in axes}
    config_id = configuration_id_v2(
        forced, support_contract_version=rc.SUPPORT_CONTRACT_VERSION_V2,
        support_contract_sha256=rc.SUPPORT_CONTRACT_SHA256_V2)
    pid = proposal_id_v2("random_control", 2)
    rows[1] = {**rows[1], **forced, "configuration_id": config_id, "proposal_id": pid,
               "trial_id_attempt001": trial_id_v2(config_id, pid, execution_generation=1)}
    # A natural duplicate scientific configuration is structurally legal; only
    # canonical generation/order/identity/domain are validated, never uniqueness.
    rc.validate_manifest_rows_v2(rows)


def test_configuration_identity_is_arm_and_order_independent_and_trial_attempts_differ():
    rows = rc.generate_random_control_rows_v2()
    axes = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size", "seq_length")
    hp = {key: rows[3][key] for key in axes}
    kwargs = dict(support_contract_version=rc.SUPPORT_CONTRACT_VERSION_V2,
                  support_contract_sha256=rc.SUPPORT_CONTRACT_SHA256_V2)
    config_id = configuration_id_v2(hp, **kwargs)
    assert config_id == rows[3]["configuration_id"]
    assert config_id == configuration_id_v2(dict(reversed(list(hp.items()))), **kwargs)
    # arm/order independent
    assert proposal_id_v2("bayesian", 4) != proposal_id_v2("random_control", 4)
    pid = rows[3]["proposal_id"]
    assert trial_id_v2(config_id, pid, execution_generation=1) != trial_id_v2(config_id, pid, execution_generation=2)
    assert rows[3]["trial_id_attempt001"].endswith("__attempt001")
    assert "wandb" not in config_id.lower() and "slurm" not in config_id.lower()


def test_infrastructure_retry_keeps_configuration_and_proposal_identity():
    rows = rc.generate_random_control_rows_v2()
    row = rows[5]
    config_id, pid = row["configuration_id"], row["proposal_id"]
    a1 = trial_id_v2(config_id, pid, execution_generation=1)
    a2 = trial_id_v2(config_id, pid, execution_generation=2)
    # Only the trailing attempt token advances; the scientific configuration
    # and controller-proposal identity are byte-identical between attempts.
    assert a1.rsplit("__attempt", 1)[0] == a2.rsplit("__attempt", 1)[0]
    assert a1.endswith("__attempt001") and a2.endswith("__attempt002")


def test_exact_v2_fidelity_and_common120_identity():
    payload = json.loads(MANIFEST_PATH.read_bytes())
    assert payload["campaign_id"] == CAMPAIGN_ID_V2 == "stage1_phase_b_sweep_v2_six_axis_common120_v001"
    assert payload["domain_version"] == DOMAIN_VERSION_V2 == "six_axis_q12_48_120_v001"
    assert payload["objective_id"] == OBJECTIVE_ID_V2 == "common120_raw_space_nse_v001"
    assert payload["fidelity_id"] == FIDELITY_ID_V2 == "mf12x50000"
    assert payload["model_seed_a"] == 967139
    assert payload["target_epoch"] == 12
    assert payload["max_updates_per_epoch"] == 50_000
    assert payload["save_weights_every"] == 1
    assert payload["support_contract_version"] == "common120_raw_space_nse_v001"
    assert payload["support_contract_sha256"] == (
        "cb4ebe86afa501ef3d5929ead5b455f8df06e7d38b58ebf4148f8545fe6851ef")


def test_no_bayesian_controller_or_forbidden_sweep_ids_anywhere_in_this_arm():
    for module_file in (rc.__file__,):
        source = Path(module_file).read_text(encoding="utf-8")
        assert "import wandb" not in source and "from wandb" not in source
        assert _BAYESIAN_CONTROLLER_RUN_ID not in source
        for forbidden in FORBIDDEN_PRODUCTION_SWEEP_IDS:
            assert forbidden not in source
    committed = MANIFEST_PATH.read_text(encoding="utf-8")
    assert _BAYESIAN_CONTROLLER_RUN_ID not in committed
    for forbidden in FORBIDDEN_PRODUCTION_SWEEP_IDS:
        assert forbidden not in committed
    assert "wandb_sweep_id" not in committed and "wandb_run_id" not in committed


def test_concurrent_launch_output_roots_never_collide():
    rows = rc.generate_random_control_rows_v2()
    trial_ids = [row["trial_id_attempt001"] for row in rows]
    assert len(set(trial_ids)) == 12
    # Three waves of four concurrent jobs sharing one output root: each job
    # writes to output_root / <trial_id>, so the concurrent roots are disjoint.
    output_root = Path("/scratch/out")
    for wave in ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]):
        dirs = [output_root / trial_ids[i] for i in wave]
        assert len(set(dirs)) == 4


def test_generator_and_manifest_have_no_wandb_dependency():
    source = Path(rc.__file__).read_text(encoding="utf-8")
    assert "wandb" not in source.lower()
