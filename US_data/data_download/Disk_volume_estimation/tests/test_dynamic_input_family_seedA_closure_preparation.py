"""Preparation-only structural comparison for the Dynamic-Input-Family-A
dynamic-input-family range-characterization campaign
(``dynamic_input_family_seedA_25k_v001``).

For each of the four approved Dynamic-Input-Family-A candidates
(``DYNAMIC_INPUT_FAMILY_A_RUN_SPECS`` in
scripts/run_stage1_dynamic_input_family_seedA_closure.py), calls the REAL,
unmocked ``prepare_pilot_run_only`` -- against a real synthetic package
covering the actual full 2,557-basin development/spatial-holdout union
(``tests._pilot_support.build_full_union_package``) and the real committed
pilot/baseline policies and split files -- then inspects the generated
``config.yaml``/``generation_manifest.json`` to confirm:

  * architecture/target/lead/embedding-shape/embedding-activation/output-
    dropout/seed/cap/hidden_size/learning_rate/embedding_dropout/seq_length
    invariants match the frozen Dynamic-Input-Family-A contract (hidden_size,
    learning_rate, seq_length, and embedding_dropout held FIXED at
    128/3e-4/72/profile-default(0.1) across every candidate -- NOT varied);
  * each candidate's resolved ``dynamic_inputs`` is EXACTLY its assigned
    family (P/PT/PTM/PTMW), in the exact frozen order, with explicit
    provenance in its run identity (``dynamic_inputs_override``/
    ``resolved_dynamic_inputs``);
  * neither gap-QC variable (``mrms_qpe_1h_mm_gap``, ``rtma_gap``) nor the
    dewpoint variable (``rtma_2d_K``) is present in any of the four
    candidates' resolved ``dynamic_inputs`` -- intentional, not a package
    omission (the package itself still advertises and contains them; see
    docs/decision_log.md);
  * no candidate's generated config or manifest differs from its siblings in
    anything other than ``dynamic_inputs`` plus unavoidable identity/path
    metadata (``experiment_name``, the three basin-list file paths,
    ``run_dir``) -- proven both at the config-mapping level and, separately,
    by comparing the basin-list FILE CONTENTS byte-for-byte (basin
    membership does not depend on dynamic_inputs);
  * every candidate's identity (``experiment_name``, ``run_dir``,
    ``run_identity``) is unique;
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
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_stage1_dynamic_input_family_seedA_closure.py"

_SEEDA = 967139
_CAP25K = 25_000
_LR3EM4 = 3e-4
_H128 = 128
_SEQ72 = 72

_FAMILY_P = ("mrms_qpe_1h_mm",)
_FAMILY_PT = _FAMILY_P + ("rtma_2t_K",)
_FAMILY_PTM = _FAMILY_PT + ("rtma_2sh_kgkg",)
_FAMILY_PTMW = _FAMILY_PTM + ("rtma_10u_ms", "rtma_10v_ms")

_EXPECTED_DYNAMIC_INPUTS = {
    "emb128x32_seedA_dynP_seq72_h128_lr3em4_cap25k_cal": _FAMILY_P,
    "emb128x32_seedA_dynPT_seq72_h128_lr3em4_cap25k_cal": _FAMILY_PT,
    "emb128x32_seedA_dynPTM_seq72_h128_lr3em4_cap25k_cal": _FAMILY_PTM,
    "emb128x32_seedA_dynPTMW_seq72_h128_lr3em4_cap25k_cal": _FAMILY_PTMW,
}

_FORBIDDEN_VARS = {"mrms_qpe_1h_mm_gap", "rtma_gap", "rtma_2d_K"}

# Identity/path-derived keys that MUST differ across sibling candidates
# sharing the same policy/profile -- everything else in config.yaml must be
# identical except dynamic_inputs itself. hidden_size, learning_rate,
# seq_length, and statics_embedding are NOT in this set: they must be
# identical across every candidate (held fixed).
_ALLOWED_DIFFERING_CONFIG_KEYS = {
    "experiment_name", "train_basin_file", "validation_basin_file",
    "test_basin_file", "run_dir", "dynamic_inputs",
}

# Identity/path/generation-time-derived generation-manifest keys allowed to
# differ across sibling candidates. generation_manifest.json has no
# "run_id"/"experiment_name" keys of its own (unlike config.yaml and
# run_identity). "package_type" is derived as f"stage1_lead06_pilot_{run_id}"
# (see pilot_lead06_config.build_pilot_bundle) -- an identity label, not a
# scientific field -- so it varies with run_id exactly like experiment_name
# does in config.yaml.
_ALLOWED_DIFFERING_MANIFEST_KEYS = {
    "dynamic_inputs", "artifact_sha256", "generated_at_utc", "package_type",
}


def _load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "run_stage1_dynamic_input_family_seedA_closure_preparation_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cli_module():
    return _load_cli_module()


@pytest.fixture(scope="module")
def dynamic_input_family_a_policy(cli_module):
    from src.baseline.pilot_lead06_config import load_pilot_policy

    real_policy = load_pilot_policy(PILOT_POLICY_PATH)
    policy = cli_module._build_dynamic_input_family_a_policy(real_policy)
    policy = cli_module._resolve_policy_relative_paths(policy)
    return policy


@pytest.fixture(scope="module")
def package_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("dynamic_input_family_a_prep_package") / "package"
    build_full_union_package(root)
    return root


def _prepare(cli_module, policy, package_root, run_id, out_root):
    config_out_dir = out_root / run_id / "config"
    preparation_out_dir = out_root / run_id / "preparation"
    result = cli_module.prepare_pilot_run_only(
        pilot_policy=policy,
        run_id=run_id,
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
        config_out_dir=config_out_dir,
        preparation_out_dir=preparation_out_dir,
    )
    return result, config_out_dir


@pytest.fixture(scope="module")
def prepared_candidates(cli_module, dynamic_input_family_a_policy, package_root, tmp_path_factory):
    out_root = tmp_path_factory.mktemp("dynamic_input_family_a_prep_runs")
    prepared = {}
    for run_id in sorted(cli_module.DYNAMIC_INPUT_FAMILY_A_RUN_SPECS):
        result, config_out_dir = _prepare(
            cli_module, dynamic_input_family_a_policy, package_root, run_id, out_root
        )
        config_yaml = yaml.safe_load((config_out_dir / "config.yaml").read_text(encoding="utf-8"))
        manifest = json.loads((config_out_dir / "generation_manifest.json").read_text(encoding="utf-8"))
        prepared[run_id] = {
            "result": result,
            "config_out_dir": config_out_dir,
            "config": config_yaml,
            "manifest": manifest,
        }
    return prepared


# --- per-candidate invariants (frozen Dynamic-Input-Family-A contract) ----


def test_all_four_candidates_prepare_cleanly_with_no_training_or_tracking(prepared_candidates):
    for run_id, entry in prepared_candidates.items():
        result = entry["result"]
        assert result["status"] == "PREPARED_ONLY"
        assert result["run_id"] == run_id
        assert result["training_started"] is False
        assert result["evaluation_started"] is False
        assert result["wandb_backend_initialized"] is False


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_DYNAMIC_INPUTS))
def test_candidate_config_matches_frozen_dynamic_input_family_a_contract(prepared_candidates, run_id):
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
    assert embedding["dropout"] == pytest.approx(0.1)

    # hidden_size, learning_rate, and seq_length are held FIXED at the
    # Hidden-size-A/LR-A/Sequence-Length-A provisional anchors across every
    # dynamic-input-family candidate -- not varied per-candidate.
    assert config["hidden_size"] == _H128
    assert config["learning_rate"] == pytest.approx(_LR3EM4)
    assert config["seq_length"] == _SEQ72

    expected_dynamic_inputs = list(_EXPECTED_DYNAMIC_INPUTS[run_id])
    assert config["dynamic_inputs"] == expected_dynamic_inputs


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_DYNAMIC_INPUTS))
def test_candidate_dynamic_inputs_never_includes_gap_flags_or_dewpoint(prepared_candidates, run_id):
    config = prepared_candidates[run_id]["config"]
    resolved = set(config["dynamic_inputs"])
    assert not (resolved & _FORBIDDEN_VARS), (
        f"{run_id}: resolved dynamic_inputs unexpectedly includes forbidden variable(s) "
        f"{resolved & _FORBIDDEN_VARS}"
    )


@pytest.mark.parametrize("run_id", sorted(_EXPECTED_DYNAMIC_INPUTS))
def test_candidate_manifest_records_explicit_dynamic_inputs_provenance(prepared_candidates, run_id):
    manifest = prepared_candidates[run_id]["manifest"]
    result = prepared_candidates[run_id]["result"]
    expected_dynamic_inputs = list(_EXPECTED_DYNAMIC_INPUTS[run_id])
    assert manifest["dynamic_inputs"] == expected_dynamic_inputs
    assert manifest["hidden_size_override"] == _H128
    assert manifest["resolved_hidden_size"] == _H128
    assert manifest["learning_rate_override"] == pytest.approx(_LR3EM4)
    assert manifest["resolved_learning_rate"] == pytest.approx(_LR3EM4)
    # generation_manifest.json (unlike run_identity) only carries the single
    # resolved "seq_length" key for this axis -- there is no
    # "seq_length_override"/"resolved_seq_length" pair here.
    assert manifest["seq_length"] == _SEQ72
    assert manifest["max_updates_per_epoch"] == _CAP25K
    assert manifest["embedding_dropout_override"] is None
    assert manifest["resolved_embedding_dropout"] == pytest.approx(0.1)

    run_identity = result["run_identity"]
    assert run_identity["dynamic_inputs_override"] == expected_dynamic_inputs
    assert run_identity["resolved_dynamic_inputs"] == expected_dynamic_inputs
    assert run_identity["hidden_size_override"] == _H128
    assert run_identity["resolved_hidden_size"] == _H128
    assert run_identity["learning_rate_override"] == pytest.approx(_LR3EM4)
    assert run_identity["resolved_learning_rate"] == pytest.approx(_LR3EM4)
    assert run_identity["seq_length_override"] == _SEQ72
    assert run_identity["resolved_seq_length"] == _SEQ72
    assert run_identity["embedding_dropout_override"] is None
    assert run_identity["resolved_embedding_dropout"] == pytest.approx(0.1)


# --- pairwise family exact membership/order --------------------------------


def test_family_matrix_is_exact_order_preserving_nested_prefix():
    p_run = "emb128x32_seedA_dynP_seq72_h128_lr3em4_cap25k_cal"
    pt_run = "emb128x32_seedA_dynPT_seq72_h128_lr3em4_cap25k_cal"
    ptm_run = "emb128x32_seedA_dynPTM_seq72_h128_lr3em4_cap25k_cal"
    ptmw_run = "emb128x32_seedA_dynPTMW_seq72_h128_lr3em4_cap25k_cal"
    p = list(_EXPECTED_DYNAMIC_INPUTS[p_run])
    pt = list(_EXPECTED_DYNAMIC_INPUTS[pt_run])
    ptm = list(_EXPECTED_DYNAMIC_INPUTS[ptm_run])
    ptmw = list(_EXPECTED_DYNAMIC_INPUTS[ptmw_run])
    assert p == ["mrms_qpe_1h_mm"]
    assert pt == p + ["rtma_2t_K"]
    assert ptm == pt + ["rtma_2sh_kgkg"]
    assert ptmw == ptm + ["rtma_10u_ms", "rtma_10v_ms"]


# --- cross-candidate: only dynamic_inputs (+ identity/path) may differ -----


def test_configs_differ_only_in_dynamic_inputs_and_identity_metadata(prepared_candidates):
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
        # hidden_size/learning_rate/seq_length/statics_embedding must NOT
        # differ -- held fixed across every candidate.
        assert reference_config["hidden_size"] == other_config["hidden_size"]
        assert reference_config["learning_rate"] == other_config["learning_rate"]
        assert reference_config["seq_length"] == other_config["seq_length"]
        assert reference_config["statics_embedding"] == other_config["statics_embedding"]


def test_manifests_differ_only_in_dynamic_inputs_and_identity_metadata(prepared_candidates):
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
    """Basin membership must not depend on dynamic_inputs -- no basin is
    silently dropped from any split because of its assigned dynamic-input
    family at config-generation time."""
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


# --- continuation-identity: dynamic_inputs is frozen per run directory -----


def test_reordered_dynamic_inputs_is_rejected_as_a_continuation_of_the_same_candidate(
    cli_module, dynamic_input_family_a_policy, package_root, tmp_path_factory
):
    """A resolved dynamic_inputs list that differs only in ORDER from an
    already-used run directory's frozen identity must still be treated as a
    contradiction (see enforce_pilot_dynamic_inputs_identity's docstring:
    order is part of the identity)."""
    from src.baseline.pilot_orchestration import enforce_pilot_dynamic_inputs_identity

    out_root = tmp_path_factory.mktemp("dynamic_input_family_a_reorder_guard")
    p_run_id = "emb128x32_seedA_dynP_seq72_h128_lr3em4_cap25k_cal"
    result, config_out_dir = _prepare(
        cli_module, dynamic_input_family_a_policy, package_root, p_run_id, out_root
    )
    nh_run_dir = config_out_dir  # a real on-disk directory suffices for the identity-file location

    # First call persists the real identity (order: ["mrms_qpe_1h_mm"]).
    enforce_pilot_dynamic_inputs_identity(run_identity=result["run_identity"], nh_run_dir=nh_run_dir)

    reordered_identity = dict(result["run_identity"])
    reordered_identity["resolved_dynamic_inputs"] = list(reversed(result["run_identity"]["resolved_dynamic_inputs"]))
    if reordered_identity["resolved_dynamic_inputs"] == result["run_identity"]["resolved_dynamic_inputs"]:
        pytest.skip("single-element family has no distinct reordering to test")
    with pytest.raises(Exception):
        enforce_pilot_dynamic_inputs_identity(run_identity=reordered_identity, nh_run_dir=nh_run_dir)


def test_pt_family_dynamic_inputs_rejected_as_continuation_of_p_family_run_dir(
    cli_module, dynamic_input_family_a_policy, package_root, tmp_path_factory
):
    """A PT-resolved identity must never be accepted as a continuation of a
    run directory whose frozen identity was P (and vice versa) -- family
    membership is part of the training identity, exactly like every other
    scalar-identity guard."""
    from src.baseline.pilot_orchestration import enforce_pilot_dynamic_inputs_identity

    out_root = tmp_path_factory.mktemp("dynamic_input_family_a_cross_family_guard")
    p_run_id = "emb128x32_seedA_dynP_seq72_h128_lr3em4_cap25k_cal"
    pt_run_id = "emb128x32_seedA_dynPT_seq72_h128_lr3em4_cap25k_cal"
    p_result, p_config_out_dir = _prepare(
        cli_module, dynamic_input_family_a_policy, package_root, p_run_id, out_root
    )
    pt_result, _ = _prepare(
        cli_module, dynamic_input_family_a_policy, package_root, pt_run_id, out_root
    )

    enforce_pilot_dynamic_inputs_identity(run_identity=p_result["run_identity"], nh_run_dir=p_config_out_dir)
    with pytest.raises(Exception):
        enforce_pilot_dynamic_inputs_identity(run_identity=pt_result["run_identity"], nh_run_dir=p_config_out_dir)


# --- structural-comparison-only fifth candidate (never a real campaign member)


def test_locally_synthesized_gap_flag_dynamic_inputs_is_rejected_at_campaign_layer(
    cli_module, dynamic_input_family_a_policy
):
    """The campaign-definition-layer defense-in-depth check
    (_assert_no_forbidden_dynamic_inputs) must reject a family literal that
    accidentally includes a gap-QC or dewpoint variable -- exercised directly
    here since the frozen P/PT/PTM/PTMW literals themselves never include
    one (see the module-level assertions already run at import time)."""
    with pytest.raises(RuntimeError, match="forbidden"):
        cli_module._assert_no_forbidden_dynamic_inputs(
            "structural_comparison_only", ("mrms_qpe_1h_mm", "mrms_qpe_1h_mm_gap")
        )
    with pytest.raises(RuntimeError, match="forbidden"):
        cli_module._assert_no_forbidden_dynamic_inputs(
            "structural_comparison_only", ("mrms_qpe_1h_mm", "rtma_2t_K", "rtma_2d_K")
        )


def test_locally_synthesized_unknown_variable_is_rejected_by_validate_dynamic_inputs_override(
    cli_module, dynamic_input_family_a_policy, package_root, tmp_path_factory
):
    """A structural-comparison-only run_id requesting a variable the
    package does not advertise must be rejected end-to-end (real code, not
    mocked) -- never silently accepted."""
    structural_run_id = "emb128x32_seedA_dyn_structural_comparison_only"
    structural_spec = PilotRunSpec(
        run_id=structural_run_id,
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=_LR3EM4,
        hidden_size=_H128,
        seq_length=_SEQ72,
        dynamic_inputs=("mrms_qpe_1h_mm", "not_a_real_package_variable"),
    )
    augmented_runs = dict(dynamic_input_family_a_policy.runs)
    assert structural_spec.run_id not in augmented_runs
    augmented_runs[structural_spec.run_id] = structural_spec
    augmented_policy = dataclasses.replace(dynamic_input_family_a_policy, runs=augmented_runs)

    out_root = tmp_path_factory.mktemp("dynamic_input_family_a_unknown_var_guard")
    with pytest.raises(Exception):
        _prepare(cli_module, augmented_policy, package_root, structural_spec.run_id, out_root)


def test_duplicate_dynamic_inputs_is_rejected(
    cli_module, dynamic_input_family_a_policy, package_root, tmp_path_factory
):
    structural_run_id = "emb128x32_seedA_dyn_dup_structural_comparison_only"
    structural_spec = PilotRunSpec(
        run_id=structural_run_id,
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=_LR3EM4,
        hidden_size=_H128,
        seq_length=_SEQ72,
        dynamic_inputs=("mrms_qpe_1h_mm", "mrms_qpe_1h_mm"),
    )
    augmented_runs = dict(dynamic_input_family_a_policy.runs)
    augmented_runs[structural_spec.run_id] = structural_spec
    augmented_policy = dataclasses.replace(dynamic_input_family_a_policy, runs=augmented_runs)

    out_root = tmp_path_factory.mktemp("dynamic_input_family_a_dup_guard")
    with pytest.raises(Exception):
        _prepare(cli_module, augmented_policy, package_root, structural_spec.run_id, out_root)


def test_empty_dynamic_inputs_is_rejected(
    cli_module, dynamic_input_family_a_policy, package_root, tmp_path_factory
):
    structural_run_id = "emb128x32_seedA_dyn_empty_structural_comparison_only"
    structural_spec = PilotRunSpec(
        run_id=structural_run_id,
        static_pathway="learned_fc_embedding",
        embedding_hiddens=[128, 32],
        seed_name="seed_a",
        seed=_SEEDA,
        run_profile_name="pilot_lead06_emb128x32_seedA_v001",
        max_updates_per_epoch=_CAP25K,
        learning_rate=_LR3EM4,
        hidden_size=_H128,
        seq_length=_SEQ72,
        dynamic_inputs=(),
    )
    augmented_runs = dict(dynamic_input_family_a_policy.runs)
    augmented_runs[structural_spec.run_id] = structural_spec
    augmented_policy = dataclasses.replace(dynamic_input_family_a_policy, runs=augmented_runs)

    out_root = tmp_path_factory.mktemp("dynamic_input_family_a_empty_guard")
    with pytest.raises(Exception):
        _prepare(cli_module, augmented_policy, package_root, structural_spec.run_id, out_root)


# --- backward compatibility: a historical/no-override run still generates --
# --- exactly as before -------------------------------------------------------


def test_historical_no_override_run_still_resolves_to_the_binding_policy_default(
    dynamic_input_family_a_policy, package_root, tmp_path_factory
):
    """A run_id with no dynamic_inputs override (e.g. any of the six real
    committed run_ids) must still resolve to the policy's own binding
    8-variable dynamic_inputs list, exactly as before this campaign's
    machinery was added."""
    from src.baseline.pilot_lead06_config import build_pilot_bundle

    real_run_id = "emb128x64_seedA"
    assert real_run_id in dynamic_input_family_a_policy.runs
    assert dynamic_input_family_a_policy.runs[real_run_id].dynamic_inputs is None

    bundle = build_pilot_bundle(
        pilot_policy=dynamic_input_family_a_policy,
        run_id=real_run_id,
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
    )

    baseline_policy_raw = yaml.safe_load(BASELINE_POLICY_PATH.read_text(encoding="utf-8"))
    assert bundle.dynamic_inputs == list(baseline_policy_raw["dynamic_inputs"])
