"""Preparation-only structural comparison for the Sequence-Length-A
sequence-length range-characterization campaign
(``seq_length_range_seedA_25k_v001``).

For each of the four NEW, approved Sequence-Length-A candidates
(``SEQ_LENGTH_A_RUN_SPECS`` in
scripts/run_stage1_seq_length_range_seedA_closure.py), calls the REAL,
unmocked ``prepare_pilot_run_only`` -- against a real synthetic package
covering the actual full 2,557-basin development/spatial-holdout union
(``tests._pilot_support.build_full_union_package``) and the real committed
pilot/baseline policies and split files -- then inspects the generated
``config.yaml``/``generation_manifest.json`` to confirm:

  * architecture/target/lead/embedding-shape/embedding-activation/output-
    dropout/seed/cap/hidden_size/learning_rate/embedding_dropout invariants
    match the frozen Sequence-Length-A contract (hidden_size, learning_rate,
    and embedding_dropout held FIXED at 128/3e-4/profile-default(0.1) across
    every candidate -- NOT varied);
  * each candidate's ``seq_length`` is exactly its assigned value, with
    explicit provenance in its run identity
    (``seq_length_override``/``resolved_seq_length``);
  * no candidate's generated config or manifest differs from its siblings in
    anything other than ``seq_length`` plus unavoidable identity/path
    metadata (``experiment_name``, the three basin-list file paths,
    ``run_dir``) -- proven both at the config-mapping level and, separately,
    by comparing the basin-list FILE CONTENTS byte-for-byte (basin
    membership does not depend on seq_length: no basin is dropped from any
    split because of its assigned seq_length);
  * every candidate's identity (``experiment_name``, ``run_dir``,
    ``run_identity``) is unique;
  * the fresh seq24 candidate's generated identity is distinct from the
    historical Hidden-size-A H=128 comparator's (``REFERENCE_RUN_ID``),
    despite both resolving to ``seq_length == 24``;
  * no training/evaluation/W&B backend was started by any of this.

No Slurm job, real NH training call, or real checkpoint evaluation happens
anywhere in this file.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from src.baseline.pilot_lead06_config import PilotRunSpec

from tests._pilot_support import (
    BASELINE_POLICY_PATH,
    PILOT_POLICY_PATH,
    SPLITS_DIR,
    build_full_union_package,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_seq_length_range_seedA_closure.py"

_SEEDA = 967139
_CAP25K = 25_000
_LR3EM4 = 3e-4
_H128 = 128
_EXPECTED_SEQ_LENGTHS = {
    "emb128x32_seedA_seq12_h128_lr3em4_cap25k_cal": 12,
    "emb128x32_seedA_seq24_h128_lr3em4_cap25k_cal": 24,
    "emb128x32_seedA_seq48_h128_lr3em4_cap25k_cal": 48,
    "emb128x32_seedA_seq72_h128_lr3em4_cap25k_cal": 72,
}

# Identity/path-derived keys that MUST differ across sibling candidates
# sharing the same policy/profile -- everything else in config.yaml must be
# identical except seq_length itself. hidden_size, learning_rate, and
# statics_embedding are NOT in this set: they must be identical across every
# candidate (held fixed).
_ALLOWED_DIFFERING_CONFIG_KEYS = {
    "experiment_name", "train_basin_file", "validation_basin_file",
    "test_basin_file", "run_dir", "seq_length",
}

# Identity/path/generation-time-derived generation-manifest keys allowed to
# differ across sibling candidates: run_id/experiment_name are identity by
# construction; package_type is derived as
# f"stage1_lead06_pilot_{run_id}" (also identity, not a scientific field);
# generated_at_utc is a wall-clock timestamp; artifact_sha256 necessarily
# changes whenever seq_length/run_id do; seq_length is the intended axis.
_ALLOWED_DIFFERING_MANIFEST_KEYS = {
    "seq_length", "artifact_sha256", "run_id", "experiment_name",
    "package_type", "generated_at_utc",
}

# A fifth, locally-synthesized run_id used ONLY to validate the local
# config-generation code path -- never the literal historical reference
# identity, and never added to the real committed policy.
_STRUCTURAL_COMPARISON_RUN_ID = "emb128x32_seedA_seq36_structural_comparison_only"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_stage1_seq_length_range_seedA_closure_preparation_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_cli_module()


@pytest.fixture(scope="module")
def seq_length_a_policy(cli_module):
    from src.baseline.pilot_lead06_config import load_pilot_policy

    real_policy = load_pilot_policy(PILOT_POLICY_PATH)
    policy = cli_module._build_seq_length_a_policy(real_policy)
    policy = cli_module._resolve_policy_relative_paths(policy)
    return policy


@pytest.fixture(scope="module")
def package_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("seq_length_a_prep_package") / "package"
    build_full_union_package(root)
    return root


def _prepare(cli_module, seq_length_a_policy, package_root, run_id, out_root):
    config_out_dir = out_root / run_id / "config"
    preparation_out_dir = out_root / run_id / "preparation"
    result = cli_module.prepare_pilot_run_only(
        pilot_policy=seq_length_a_policy,
        run_id=run_id,
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
        config_out_dir=config_out_dir,
        preparation_out_dir=preparation_out_dir,
    )
    return result, config_out_dir


@pytest.fixture(scope="module")
def prepared_candidates(cli_module, seq_length_a_policy, package_root, tmp_path_factory):
    out_root = tmp_path_factory.mktemp("seq_length_a_prep_runs")
    prepared = {}
    for run_id in sorted(cli_module.SEQ_LENGTH_A_RUN_SPECS):
        result, config_out_dir = _prepare(cli_module, seq_length_a_policy, package_root, run_id, out_root)
        config_yaml = yaml.safe_load((config_out_dir / "config.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((config_out_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        prepared[run_id] = {
            "result": result,
            "config_out_dir": config_out_dir,
            "config": config_yaml,
            "manifest": manifest,
        }
    return prepared


# --- per-candidate invariants (frozen Sequence-Length-A contract) ---------


def test_all_four_candidates_prepare_cleanly_with_no_training_or_tracking(prepared_candidates):
    for run_id, entry in prepared_candidates.items():
        result = entry["result"]
        assert result["status"] == "PREPARED_ONLY"
        assert result["run_id"] == run_id
        assert result["training_started"] is False
        assert result["evaluation_started"] is False
        assert result["wandb_backend_initialized"] is False


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_SEQ_LENGTHS))
def test_candidate_config_matches_frozen_seq_length_a_contract(prepared_candidates, run_id):
    config = prepared_candidates[run_id]["config"]
    assert config["output_dropout"] == pytest.approx(0.25)
    assert config["optimizer"] == "Adam"
    assert config["loss"] == "NSE"
    assert "lr_scheduler" not in config
    assert config["target_variables"] == ["qobs_mm_per_h_lead06"]
    assert config["epochs"] == 6
    assert config["max_updates_per_epoch"] == _CAP25K
    assert config["seed"] == _SEEDA

    embedding = config["statics_embedding"]
    assert embedding["hiddens"] == [128, 32]
    assert embedding["activation"] == "tanh"
    # embedding_dropout is deliberately NOT overridden by this campaign --
    # held at the Embedding-Dropout-A-characterized provisional anchor.
    assert embedding["dropout"] == pytest.approx(0.1)

    # hidden_size and learning_rate are held FIXED at the Hidden-size-A/LR-A
    # provisional anchors across every seq_length candidate -- not varied
    # per-candidate.
    assert config["hidden_size"] == _H128
    assert config["learning_rate"] == pytest.approx(_LR3EM4)

    expected_seq_length = _EXPECTED_SEQ_LENGTHS[run_id]
    assert config["seq_length"] == expected_seq_length


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_SEQ_LENGTHS))
def test_candidate_manifest_records_explicit_seq_length_provenance(prepared_candidates, run_id):
    manifest = prepared_candidates[run_id]["manifest"]
    result = prepared_candidates[run_id]["result"]
    expected_seq_length = _EXPECTED_SEQ_LENGTHS[run_id]
    assert manifest["seq_length"] == expected_seq_length
    assert manifest["hidden_size_override"] == _H128
    assert manifest["resolved_hidden_size"] == _H128
    assert manifest["learning_rate_override"] == pytest.approx(_LR3EM4)
    assert manifest["resolved_learning_rate"] == pytest.approx(_LR3EM4)
    assert manifest["max_updates_per_epoch"] == _CAP25K
    assert manifest["embedding_dropout_override"] is None
    assert manifest["resolved_embedding_dropout"] == pytest.approx(0.1)

    run_identity = result["run_identity"]
    assert run_identity["seq_length_override"] == expected_seq_length
    assert run_identity["resolved_seq_length"] == expected_seq_length
    assert run_identity["seq_length"] == expected_seq_length
    assert run_identity["hidden_size_override"] == _H128
    assert run_identity["resolved_hidden_size"] == _H128
    assert run_identity["learning_rate_override"] == pytest.approx(_LR3EM4)
    assert run_identity["resolved_learning_rate"] == pytest.approx(_LR3EM4)
    assert run_identity["embedding_dropout_override"] is None
    assert run_identity["resolved_embedding_dropout"] == pytest.approx(0.1)


# --- cross-candidate: only seq_length (+ identity/path) may differ --------


def test_configs_differ_only_in_seq_length_and_identity_metadata(prepared_candidates):
    run_ids = sorted(prepared_candidates)
    reference_run_id = run_ids[0]
    reference_config = prepared_candidates[reference_run_id]["config"]
    for run_id in run_ids[1:]:
        other_config = prepared_candidates[run_id]["config"]
        all_keys = set(reference_config) | set(other_config)
        differing = {
            k for k in all_keys
            if reference_config.get(k) != other_config.get(k)
        }
        unexpected = differing - _ALLOWED_DIFFERING_CONFIG_KEYS
        assert not unexpected, (
            f"{reference_run_id} vs {run_id}: unexpected differing config keys {unexpected}"
        )
        # data_dir (shared package_root) must be byte-identical.
        assert reference_config["data_dir"] == other_config["data_dir"]
        # hidden_size/learning_rate/statics_embedding must NOT differ -- held
        # fixed across every candidate.
        assert reference_config["hidden_size"] == other_config["hidden_size"]
        assert reference_config["learning_rate"] == other_config["learning_rate"]
        assert reference_config["statics_embedding"] == other_config["statics_embedding"]


def test_manifests_differ_only_in_seq_length_and_identity_metadata(prepared_candidates):
    run_ids = sorted(prepared_candidates)
    reference_run_id = run_ids[0]
    reference_manifest = prepared_candidates[reference_run_id]["manifest"]
    for run_id in run_ids[1:]:
        other_manifest = prepared_candidates[run_id]["manifest"]
        all_keys = set(reference_manifest) | set(other_manifest)
        differing = {
            k for k in all_keys
            if reference_manifest.get(k) != other_manifest.get(k)
        }
        unexpected = differing - _ALLOWED_DIFFERING_MANIFEST_KEYS
        assert not unexpected, (
            f"{reference_run_id} vs {run_id}: unexpected differing manifest keys {unexpected}"
        )


def test_basin_list_file_contents_are_identical_across_all_candidates(prepared_candidates):
    """Basin membership must not depend on seq_length -- no basin is
    silently dropped from any split because of its assigned seq_length at
    config-generation time (a cheap history-length audit against the real
    time-series is a separate concern; see docs/decision_log.md)."""
    run_ids = sorted(prepared_candidates)
    reference_run_id = run_ids[0]
    reference_config = prepared_candidates[reference_run_id]["config"]
    reference_contents = {
        key: Path(reference_config[key]).read_text(encoding="utf-8")
        for key in ("train_basin_file", "validation_basin_file", "test_basin_file")
    }
    for run_id in run_ids[1:]:
        other_config = prepared_candidates[run_id]["config"]
        for key, expected_text in reference_contents.items():
            other_path = Path(other_config[key])
            assert other_path.read_text(encoding="utf-8") == expected_text, (
                f"{reference_run_id} vs {run_id}: {key} contents differ"
            )


def test_candidate_identities_are_all_unique(prepared_candidates):
    experiment_names = [entry["result"]["experiment_name"] for entry in prepared_candidates.values()]
    assert len(experiment_names) == len(set(experiment_names))

    run_dirs = [entry["config"]["run_dir"] for entry in prepared_candidates.values()]
    assert len(run_dirs) == len(set(run_dirs))

    wandb_run_ids = [
        entry["result"]["run_identity"]["wandb_run_id"] for entry in prepared_candidates.values()
    ]
    assert len(wandb_run_ids) == len(set(wandb_run_ids))


# --- fresh seq24 vs. historical Hidden-size-A H=128 comparator ------------


def test_fresh_seq24_candidate_identity_is_distinct_from_historical_comparator(
    cli_module, seq_length_a_policy, package_root, tmp_path_factory
):
    """Both the fresh seq24 candidate and the historical Hidden-size-A H=128
    comparator resolve to seq_length == 24, but they must never be treated
    as the same run identity -- the comparator is read-only and reachable
    only via --status-only, never prepared/trained here."""
    out_root = tmp_path_factory.mktemp("seq_length_a_seq24_vs_reference")
    seq24_result, seq24_config_out_dir = _prepare(
        cli_module, seq_length_a_policy, package_root,
        "emb128x32_seedA_seq24_h128_lr3em4_cap25k_cal", out_root,
    )
    seq24_config = yaml.safe_load((seq24_config_out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert seq24_config["seq_length"] == 24
    assert seq24_result["run_id"] != cli_module.REFERENCE_RUN_ID
    assert seq24_result["experiment_name"] != f"stage1_lead06_pilot_{cli_module.REFERENCE_RUN_ID}_v001"


def test_reference_run_id_is_not_a_key_of_seq_length_a_run_specs(cli_module):
    assert cli_module.REFERENCE_RUN_ID not in cli_module.SEQ_LENGTH_A_RUN_SPECS


def test_reference_run_id_cannot_be_prepared_through_prepare_pilot_run_only(
    cli_module, seq_length_a_policy, package_root, tmp_path_factory
):
    out_root = tmp_path_factory.mktemp("seq_length_a_reference_guard")
    with pytest.raises(Exception):
        _prepare(cli_module, seq_length_a_policy, package_root, cli_module.REFERENCE_RUN_ID, out_root)


# --- structural-comparison-only fifth candidate (never the real reference) -


def test_locally_synthesized_seq36_candidate_is_rejected_by_validate_seq_length(
    cli_module, seq_length_a_policy, package_root, tmp_path_factory
):
    """Unlike Embedding-Dropout-A's continuous dropout axis, seq_length is a
    closed {12, 24, 48, 72} set (nh_config_generation.validate_seq_length) --
    there is no valid "structural-comparison-only" seq_length outside that
    set. This test instead confirms the closed-set guard rejects an
    off-set value end-to-end (real code, not mocked), rather than silently
    accepting it."""
    assert _STRUCTURAL_COMPARISON_RUN_ID != cli_module.REFERENCE_RUN_ID
    assert _STRUCTURAL_COMPARISON_RUN_ID not in cli_module.SEQ_LENGTH_A_RUN_SPECS

    structural_spec = PilotRunSpec(
        run_id=_STRUCTURAL_COMPARISON_RUN_ID,
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=_LR3EM4,
        hidden_size=_H128,
        seq_length=36,
    )
    augmented_runs = dict(seq_length_a_policy.runs)
    assert structural_spec.run_id not in augmented_runs
    augmented_runs[structural_spec.run_id] = structural_spec
    augmented_policy = dataclasses.replace(seq_length_a_policy, runs=augmented_runs)

    out_root = tmp_path_factory.mktemp("seq_length_a_structural_comparison")
    with pytest.raises(Exception):
        _prepare(cli_module, augmented_policy, package_root, structural_spec.run_id, out_root)
