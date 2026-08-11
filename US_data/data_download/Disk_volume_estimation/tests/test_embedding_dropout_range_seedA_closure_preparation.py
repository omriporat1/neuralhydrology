"""Preparation-only structural comparison for the Embedding-Dropout-A
embedding-dropout range-characterization campaign
(``embedding_dropout_range_seedA_25k_v001``).

For each of the five NEW, approved Embedding-Dropout-A candidates
(``EMBEDDING_DROPOUT_A_RUN_SPECS`` in
scripts/run_stage1_embedding_dropout_range_seedA_closure.py), calls the REAL,
unmocked ``prepare_pilot_run_only`` -- against a real synthetic package
covering the actual full 2,557-basin development/spatial-holdout union
(``tests._pilot_support.build_full_union_package``) and the real committed
pilot/baseline policies and split files -- then inspects the generated
``config.yaml``/``generation_manifest.json`` to confirm:

  * architecture/target/lead/seq_length/embedding-shape/embedding-activation/
    output-dropout/seed/cap/hidden_size/learning_rate invariants match the
    frozen Embedding-Dropout-A contract (hidden_size and learning_rate held
    FIXED at 128/3e-4 across every candidate -- NOT varied);
  * each candidate's ``statics_embedding.dropout`` is exactly its assigned
    value (including the drop00 candidate's explicit ``0.0``, never lost or
    confused with "no override"), with explicit provenance in its generation
    manifest (``embedding_dropout_override``/``resolved_embedding_dropout``);
  * no candidate's generated config or manifest differs from its siblings in
    anything other than ``statics_embedding`` (dropout) plus unavoidable
    identity/path metadata (``experiment_name``, the three basin-list file
    paths, ``run_dir``) -- proven both at the config-mapping level and,
    separately, by comparing the basin-list FILE CONTENTS byte-for-byte;
  * every candidate's identity (``experiment_name``, ``run_dir``,
    ``run_identity``) is unique;
  * the fresh drop10 candidate's generated identity is distinct from the
    historical Hidden-size-A H=128 comparator's (``REFERENCE_RUN_ID``),
    despite both resolving to ``statics_embedding.dropout == 0.1``;
  * no training/evaluation/W&B backend was started by any of this.

Also builds one locally-synthesized, clearly-labeled sixth
"structural-comparison-only" PilotRunSpec (embedding_dropout=0.15, a run_id
distinct from and never equal to REFERENCE_RUN_ID or any of the five real
candidates) through the exact same real generation pathway, solely to
confirm the code path injects embedding_dropout as the only scientific
difference between sibling candidates.

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
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_embedding_dropout_range_seedA_closure.py"

_SEEDA = 967139
_CAP25K = 25_000
_LR3EM4 = 3e-4
_H128 = 128
_EXPECTED_EMBEDDING_DROPOUTS = {
    "emb128x32_seedA_drop00_h128_lr3em4_cap25k_cal": 0.00,
    "emb128x32_seedA_drop05_h128_lr3em4_cap25k_cal": 0.05,
    "emb128x32_seedA_drop10_h128_lr3em4_cap25k_cal": 0.10,
    "emb128x32_seedA_drop20_h128_lr3em4_cap25k_cal": 0.20,
    "emb128x32_seedA_drop40_h128_lr3em4_cap25k_cal": 0.40,
}

# Identity/path-derived keys that MUST differ across sibling candidates
# sharing the same policy/profile -- everything else in config.yaml must be
# identical except statics_embedding (dropout) itself. hidden_size and
# learning_rate are NOT in this set: they must be identical across every
# candidate (held fixed).
_ALLOWED_DIFFERING_CONFIG_KEYS = {
    "experiment_name", "train_basin_file", "validation_basin_file",
    "test_basin_file", "run_dir", "statics_embedding",
}

# Identity/path/generation-time-derived generation-manifest keys allowed to
# differ across sibling candidates, beyond the embedding-dropout-provenance
# fields themselves: run_id and experiment_name are identity by
# construction; package_type is derived as
# f"stage1_lead06_pilot_{run_id}" (also identity, not a scientific field);
# generated_at_utc is a wall-clock timestamp; artifact_sha256 necessarily
# changes whenever embedding_dropout/run_id do.
_ALLOWED_DIFFERING_MANIFEST_KEYS = {
    "embedding_dropout_override", "resolved_embedding_dropout", "artifact_sha256",
    "run_id", "experiment_name", "package_type", "generated_at_utc",
}

# A sixth, locally-synthesized run_id used ONLY to validate the local
# config-generation code path -- never the literal historical reference
# identity, and never added to the real committed policy.
_STRUCTURAL_COMPARISON_RUN_ID = "emb128x32_seedA_drop15_structural_comparison_only"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_stage1_embedding_dropout_range_seedA_closure_preparation_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_cli_module()


@pytest.fixture(scope="module")
def embedding_dropout_a_policy(cli_module):
    from src.baseline.pilot_lead06_config import load_pilot_policy

    real_policy = load_pilot_policy(PILOT_POLICY_PATH)
    policy = cli_module._build_embedding_dropout_a_policy(real_policy)
    policy = cli_module._resolve_policy_relative_paths(policy)
    return policy


@pytest.fixture(scope="module")
def package_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("embedding_dropout_a_prep_package") / "package"
    build_full_union_package(root)
    return root


def _prepare(cli_module, embedding_dropout_a_policy, package_root, run_id, out_root):
    config_out_dir = out_root / run_id / "config"
    preparation_out_dir = out_root / run_id / "preparation"
    result = cli_module.prepare_pilot_run_only(
        pilot_policy=embedding_dropout_a_policy,
        run_id=run_id,
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
        config_out_dir=config_out_dir,
        preparation_out_dir=preparation_out_dir,
    )
    return result, config_out_dir


@pytest.fixture(scope="module")
def prepared_candidates(cli_module, embedding_dropout_a_policy, package_root, tmp_path_factory):
    out_root = tmp_path_factory.mktemp("embedding_dropout_a_prep_runs")
    prepared = {}
    for run_id in sorted(cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS):
        result, config_out_dir = _prepare(cli_module, embedding_dropout_a_policy, package_root, run_id, out_root)
        config_yaml = yaml.safe_load((config_out_dir / "config.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((config_out_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        prepared[run_id] = {
            "result": result,
            "config_out_dir": config_out_dir,
            "config": config_yaml,
            "manifest": manifest,
        }
    return prepared


# --- per-candidate invariants (frozen Embedding-Dropout-A contract) --------


def test_all_five_candidates_prepare_cleanly_with_no_training_or_tracking(prepared_candidates):
    for run_id, entry in prepared_candidates.items():
        result = entry["result"]
        assert result["status"] == "PREPARED_ONLY"
        assert result["run_id"] == run_id
        assert result["training_started"] is False
        assert result["evaluation_started"] is False
        assert result["wandb_backend_initialized"] is False


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_EMBEDDING_DROPOUTS))
def test_candidate_config_matches_frozen_embedding_dropout_a_contract(prepared_candidates, run_id):
    config = prepared_candidates[run_id]["config"]
    assert config["output_dropout"] == pytest.approx(0.25)
    assert config["optimizer"] == "Adam"
    assert config["loss"] == "NSE"
    assert "lr_scheduler" not in config
    assert config["seq_length"] == 24
    assert config["target_variables"] == ["qobs_mm_per_h_lead06"]
    assert config["epochs"] == 6
    assert config["max_updates_per_epoch"] == _CAP25K
    assert config["seed"] == _SEEDA

    embedding = config["statics_embedding"]
    assert embedding["hiddens"] == [128, 32]
    assert embedding["activation"] == "tanh"

    # hidden_size and learning_rate are held FIXED at the Hidden-size-A/LR-A
    # provisional anchors across every embedding-dropout candidate -- not
    # varied per-candidate.
    assert config["hidden_size"] == _H128
    assert config["learning_rate"] == pytest.approx(_LR3EM4)

    expected_dropout = _EXPECTED_EMBEDDING_DROPOUTS[run_id]
    assert embedding["dropout"] == pytest.approx(expected_dropout)


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_EMBEDDING_DROPOUTS))
def test_candidate_manifest_records_explicit_embedding_dropout_provenance(prepared_candidates, run_id):
    manifest = prepared_candidates[run_id]["manifest"]
    expected_dropout = _EXPECTED_EMBEDDING_DROPOUTS[run_id]
    assert manifest["embedding_dropout_override"] == pytest.approx(expected_dropout)
    assert manifest["resolved_embedding_dropout"] == pytest.approx(expected_dropout)
    assert manifest["hidden_size_override"] == _H128
    assert manifest["resolved_hidden_size"] == _H128
    assert manifest["learning_rate_override"] == pytest.approx(_LR3EM4)
    assert manifest["resolved_learning_rate"] == pytest.approx(_LR3EM4)
    assert manifest["max_updates_per_epoch"] == _CAP25K


def test_drop00_candidate_dropout_is_explicit_zero_not_none(prepared_candidates):
    """The drop00 candidate's manifest/config values must be the real float
    0.0, never None/absent -- this is the central "is not None, never
    truthiness" safety property under real end-to-end code, not a mock."""
    entry = prepared_candidates["emb128x32_seedA_drop00_h128_lr3em4_cap25k_cal"]
    assert entry["config"]["statics_embedding"]["dropout"] == 0.0
    assert entry["config"]["statics_embedding"]["dropout"] is not None
    assert entry["manifest"]["embedding_dropout_override"] == 0.0
    assert entry["manifest"]["embedding_dropout_override"] is not None
    assert entry["manifest"]["resolved_embedding_dropout"] == 0.0
    assert entry["manifest"]["resolved_embedding_dropout"] is not None


# --- cross-candidate: only statics_embedding (+ identity/path) may differ --


def test_configs_differ_only_in_statics_embedding_and_identity_metadata(prepared_candidates):
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
        # hidden_size/learning_rate must NOT differ -- held fixed across
        # every candidate.
        assert reference_config["hidden_size"] == other_config["hidden_size"]
        assert reference_config["learning_rate"] == other_config["learning_rate"]
        # within statics_embedding, only dropout may differ.
        ref_emb = reference_config["statics_embedding"]
        other_emb = other_config["statics_embedding"]
        emb_keys = set(ref_emb) | set(other_emb)
        emb_differing = {k for k in emb_keys if ref_emb.get(k) != other_emb.get(k)}
        assert emb_differing <= {"dropout"}, (
            f"{reference_run_id} vs {run_id}: unexpected differing statics_embedding keys {emb_differing}"
        )


def test_manifests_differ_only_in_embedding_dropout_provenance_and_identity_metadata(prepared_candidates):
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


# --- fresh drop10 vs. historical Hidden-size-A H=128 comparator ------------


def test_fresh_drop10_candidate_identity_is_distinct_from_historical_comparator(
    cli_module, embedding_dropout_a_policy, package_root, tmp_path_factory
):
    """Both the fresh drop10 candidate and the historical Hidden-size-A H=128
    comparator resolve to statics_embedding.dropout == 0.1, but they must
    never be treated as the same run identity -- the comparator is read-only
    and reachable only via --status-only, never prepared/trained here."""
    out_root = tmp_path_factory.mktemp("embedding_dropout_a_drop10_vs_reference")
    drop10_result, drop10_config_out_dir = _prepare(
        cli_module, embedding_dropout_a_policy, package_root,
        "emb128x32_seedA_drop10_h128_lr3em4_cap25k_cal", out_root,
    )
    drop10_config = yaml.safe_load((drop10_config_out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert drop10_config["statics_embedding"]["dropout"] == pytest.approx(0.1)
    assert drop10_result["run_id"] != cli_module.REFERENCE_RUN_ID
    assert drop10_result["experiment_name"] != f"stage1_lead06_pilot_{cli_module.REFERENCE_RUN_ID}_v001"


def test_reference_run_id_is_not_a_key_of_embedding_dropout_a_run_specs(cli_module):
    assert cli_module.REFERENCE_RUN_ID not in cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS


def test_reference_run_id_cannot_be_prepared_through_prepare_pilot_run_only(
    cli_module, embedding_dropout_a_policy, package_root, tmp_path_factory
):
    out_root = tmp_path_factory.mktemp("embedding_dropout_a_reference_guard")
    with pytest.raises(Exception):
        _prepare(cli_module, embedding_dropout_a_policy, package_root, cli_module.REFERENCE_RUN_ID, out_root)


# --- structural-comparison-only sixth candidate (never the real reference) -


def test_locally_synthesized_drop15_candidate_differs_from_siblings_only_in_embedding_dropout(
    cli_module, embedding_dropout_a_policy, package_root, tmp_path_factory
):
    assert _STRUCTURAL_COMPARISON_RUN_ID != cli_module.REFERENCE_RUN_ID
    assert _STRUCTURAL_COMPARISON_RUN_ID not in cli_module.EMBEDDING_DROPOUT_A_RUN_SPECS

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
        embedding_dropout=0.15,
    )
    augmented_runs = dict(embedding_dropout_a_policy.runs)
    assert structural_spec.run_id not in augmented_runs
    augmented_runs[structural_spec.run_id] = structural_spec
    augmented_policy = dataclasses.replace(embedding_dropout_a_policy, runs=augmented_runs)

    out_root = tmp_path_factory.mktemp("embedding_dropout_a_structural_comparison")
    result, config_out_dir = _prepare(
        cli_module, augmented_policy, package_root, structural_spec.run_id, out_root
    )
    assert result["status"] == "PREPARED_ONLY"
    config = yaml.safe_load((config_out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["statics_embedding"]["dropout"] == pytest.approx(0.15)
    assert config["hidden_size"] == _H128
    assert config["learning_rate"] == pytest.approx(_LR3EM4)
    assert config["statics_embedding"]["hiddens"] == [128, 32]
    assert config["seed"] == _SEEDA
    assert config["max_updates_per_epoch"] == _CAP25K

    another_run_id = sorted(_EXPECTED_EMBEDDING_DROPOUTS)[0]
    another_result, another_config_out_dir = _prepare(
        cli_module, embedding_dropout_a_policy, package_root, another_run_id, out_root
    )
    another_config = yaml.safe_load(
        (another_config_out_dir / "config.yaml").read_text(encoding="utf-8")
    )
    all_keys = set(config) | set(another_config)
    differing = {k for k in all_keys if config.get(k) != another_config.get(k)}
    unexpected = differing - _ALLOWED_DIFFERING_CONFIG_KEYS
    assert not unexpected, f"unexpected differing config keys {unexpected}"
