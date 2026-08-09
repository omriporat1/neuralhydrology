"""Preparation-only structural comparison for the Hidden-size-A hidden-size
range-characterization campaign (``hidden_size_range_seedA_25k_v001``).

For each of the four NEW, approved Hidden-size-A candidates
(``HIDDEN_SIZE_A_RUN_SPECS`` in
scripts/run_stage1_hidden_size_range_seedA_closure.py), calls the REAL,
unmocked ``prepare_pilot_run_only`` -- against a real synthetic package
covering the actual full 2,557-basin development/spatial-holdout union
(``tests._pilot_support.build_full_union_package``) and the real committed
pilot/baseline policies and split files -- then inspects the generated
``config.yaml``/``generation_manifest.json`` to confirm:

  * architecture/target/lead/seq_length/embedding-shape/dropout/seed/cap/
    learning_rate invariants match the frozen Hidden-size-A contract;
  * each candidate's ``hidden_size`` is exactly its assigned value, with
    explicit provenance in its generation manifest
    (``hidden_size_override``/``resolved_hidden_size``);
  * ``learning_rate`` is held fixed at 3e-4 across every candidate (NOT
    varied -- this is the LR-A-characterized provisional anchor, deliberately
    not re-tuned per hidden size);
  * no candidate's generated config or manifest differs from its siblings
    in anything other than ``hidden_size`` plus unavoidable identity/path
    metadata (``experiment_name``, the three basin-list file paths,
    ``run_dir``) -- proven both at the config-mapping level and, separately,
    by comparing the basin-list FILE CONTENTS byte-for-byte;
  * every candidate's identity (``experiment_name``, ``run_dir``,
    ``run_identity``) is unique;
  * no training/evaluation/W&B backend was started by any of this.

Also builds one locally-synthesized, clearly-labeled fifth
"structural-comparison-only" PilotRunSpec (hidden_size=32, a run_id distinct
from and never equal to REFERENCE_RUN_ID) through the exact same real
generation pathway, solely to confirm the code path injects hidden_size as
the only scientific difference between sibling candidates. This makes no
claim about reproducing the real, historical, Moriah-trained
emb128x32_seedA_lr3em4_cap25k_cal (LR-A H=128) reference itself -- that
reference's own PilotRunSpec/generation manifest was never committed to this
repository and is not accessible from this local/text-only session; the
fresh-vs-historical reproducibility comparison itself is explicitly deferred
until after the fresh H=128 candidate completes (see
docs/decision_log.md's 2026-08-09 Hidden-size-A design-freeze entry).

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
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_hidden_size_range_seedA_closure.py"

_SEEDA = 967139
_CAP25K = 25_000
_LR3EM4 = 3e-4
_EXPECTED_HIDDEN_SIZE_A_HIDDEN_SIZES = {
    "emb128x32_seedA_h64_lr3em4_cap25k_cal": 64,
    "emb128x32_seedA_h128_lr3em4_cap25k_cal": 128,
    "emb128x32_seedA_h256_lr3em4_cap25k_cal": 256,
    "emb128x32_seedA_h512_lr3em4_cap25k_cal": 512,
}

# Identity/path-derived keys that MUST differ across sibling candidates
# sharing the same policy/profile -- everything else in config.yaml must be
# identical except hidden_size itself.
_ALLOWED_DIFFERING_CONFIG_KEYS = {
    "experiment_name", "train_basin_file", "validation_basin_file",
    "test_basin_file", "run_dir", "hidden_size",
}

# Identity/path/generation-time-derived generation-manifest keys allowed to
# differ across sibling candidates, beyond the hidden-size-provenance fields
# themselves: run_id and experiment_name are identity by construction;
# package_type is derived as f"stage1_lead06_pilot_{run_id}" (also identity,
# not a scientific field); generated_at_utc is a wall-clock timestamp;
# artifact_sha256 necessarily changes whenever hidden_size/run_id do.
_ALLOWED_DIFFERING_MANIFEST_KEYS = {
    "hidden_size_override", "resolved_hidden_size", "artifact_sha256",
    "run_id", "experiment_name", "package_type", "generated_at_utc",
}

# A fifth, locally-synthesized run_id used ONLY to validate the local
# config-generation code path -- never the literal historical reference
# identity, and never added to the real committed policy.
_STRUCTURAL_COMPARISON_RUN_ID = "emb128x32_seedA_h32_structural_comparison_only"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_stage1_hidden_size_range_seedA_closure_preparation_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_cli_module()


@pytest.fixture(scope="module")
def hidden_size_a_policy(cli_module):
    from src.baseline.pilot_lead06_config import load_pilot_policy

    real_policy = load_pilot_policy(PILOT_POLICY_PATH)
    policy = cli_module._build_hidden_size_a_policy(real_policy)
    policy = cli_module._resolve_policy_relative_paths(policy)
    return policy


@pytest.fixture(scope="module")
def package_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("hidden_size_a_prep_package") / "package"
    build_full_union_package(root)
    return root


def _prepare(cli_module, hidden_size_a_policy, package_root, run_id, out_root):
    config_out_dir = out_root / run_id / "config"
    preparation_out_dir = out_root / run_id / "preparation"
    result = cli_module.prepare_pilot_run_only(
        pilot_policy=hidden_size_a_policy,
        run_id=run_id,
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
        config_out_dir=config_out_dir,
        preparation_out_dir=preparation_out_dir,
    )
    return result, config_out_dir


@pytest.fixture(scope="module")
def prepared_candidates(cli_module, hidden_size_a_policy, package_root, tmp_path_factory):
    out_root = tmp_path_factory.mktemp("hidden_size_a_prep_runs")
    prepared = {}
    for run_id in sorted(cli_module.HIDDEN_SIZE_A_RUN_SPECS):
        result, config_out_dir = _prepare(cli_module, hidden_size_a_policy, package_root, run_id, out_root)
        config_yaml = yaml.safe_load((config_out_dir / "config.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((config_out_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        prepared[run_id] = {
            "result": result,
            "config_out_dir": config_out_dir,
            "config": config_yaml,
            "manifest": manifest,
        }
    return prepared


# --- per-candidate invariants (frozen Hidden-size-A contract) ---------------


def test_all_four_candidates_prepare_cleanly_with_no_training_or_tracking(prepared_candidates):
    for run_id, entry in prepared_candidates.items():
        result = entry["result"]
        assert result["status"] == "PREPARED_ONLY"
        assert result["run_id"] == run_id
        assert result["training_started"] is False
        assert result["evaluation_started"] is False
        assert result["wandb_backend_initialized"] is False


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_HIDDEN_SIZE_A_HIDDEN_SIZES))
def test_candidate_config_matches_frozen_hidden_size_a_contract(prepared_candidates, run_id):
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
    assert embedding["dropout"] == pytest.approx(0.1)

    # learning_rate is held FIXED at the LR-A-characterized provisional
    # anchor across every hidden-size candidate -- not varied per-candidate.
    assert config["learning_rate"] == pytest.approx(_LR3EM4)

    assert config["hidden_size"] == _EXPECTED_HIDDEN_SIZE_A_HIDDEN_SIZES[run_id]


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_HIDDEN_SIZE_A_HIDDEN_SIZES))
def test_candidate_manifest_records_explicit_hidden_size_provenance(prepared_candidates, run_id):
    manifest = prepared_candidates[run_id]["manifest"]
    expected_hidden_size = _EXPECTED_HIDDEN_SIZE_A_HIDDEN_SIZES[run_id]
    assert manifest["hidden_size_override"] == expected_hidden_size
    assert manifest["resolved_hidden_size"] == expected_hidden_size
    assert manifest["learning_rate_override"] == pytest.approx(_LR3EM4)
    assert manifest["resolved_learning_rate"] == pytest.approx(_LR3EM4)
    assert manifest["max_updates_per_epoch"] == _CAP25K


# --- cross-candidate: only hidden_size (+ identity/path) may differ --------


def test_configs_differ_only_in_hidden_size_and_identity_metadata(prepared_candidates):
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
        # learning_rate must NOT differ -- held fixed across every candidate.
        assert reference_config["learning_rate"] == other_config["learning_rate"]


def test_manifests_differ_only_in_hidden_size_provenance_and_identity_metadata(prepared_candidates):
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


# --- structural-comparison-only fifth candidate (never the real reference) -


def test_locally_synthesized_h32_candidate_differs_from_siblings_only_in_hidden_size(
    cli_module, hidden_size_a_policy, package_root, tmp_path_factory
):
    assert _STRUCTURAL_COMPARISON_RUN_ID != cli_module.REFERENCE_RUN_ID

    structural_spec = PilotRunSpec(
        run_id=_STRUCTURAL_COMPARISON_RUN_ID,
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=_LR3EM4,
        hidden_size=32,
    )
    augmented_runs = dict(hidden_size_a_policy.runs)
    assert structural_spec.run_id not in augmented_runs
    augmented_runs[structural_spec.run_id] = structural_spec
    augmented_policy = dataclasses.replace(hidden_size_a_policy, runs=augmented_runs)

    out_root = tmp_path_factory.mktemp("hidden_size_a_structural_comparison")
    result, config_out_dir = _prepare(
        cli_module, augmented_policy, package_root, structural_spec.run_id, out_root
    )
    assert result["status"] == "PREPARED_ONLY"
    config = yaml.safe_load((config_out_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config["hidden_size"] == 32
    assert config["learning_rate"] == pytest.approx(_LR3EM4)
    assert config["statics_embedding"]["hiddens"] == [128, 32]
    assert config["seed"] == _SEEDA
    assert config["max_updates_per_epoch"] == _CAP25K

    another_run_id = sorted(_EXPECTED_HIDDEN_SIZE_A_HIDDEN_SIZES)[0]
    another_result, another_config_out_dir = _prepare(
        cli_module, hidden_size_a_policy, package_root, another_run_id, out_root
    )
    another_config = yaml.safe_load(
        (another_config_out_dir / "config.yaml").read_text(encoding="utf-8")
    )
    all_keys = set(config) | set(another_config)
    differing = {k for k in all_keys if config.get(k) != another_config.get(k)}
    unexpected = differing - _ALLOWED_DIFFERING_CONFIG_KEYS
    assert not unexpected, f"unexpected differing config keys {unexpected}"


# --- reused reference stays unreachable through this preparation path ------


def test_reference_run_id_cannot_be_prepared_through_prepare_pilot_run_only(
    cli_module, hidden_size_a_policy, package_root, tmp_path_factory
):
    out_root = tmp_path_factory.mktemp("hidden_size_a_reference_guard")
    with pytest.raises(Exception):
        _prepare(cli_module, hidden_size_a_policy, package_root, cli_module.REFERENCE_RUN_ID, out_root)
