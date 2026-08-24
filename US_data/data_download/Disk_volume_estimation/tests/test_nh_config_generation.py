"""Tests for src.baseline.nh_config_generation (local implementation increment).

Uses the real, committed Stage 1 scientific baseline policy
(config/stage1_scientific_baseline_v001.yaml) and the real canonical split
files (config/stage1_baseline_splits_v001/) -- both already accepted
artifacts in this repository -- paired with a lightweight fake package
fixture (manifests/package_manifest.json + attributes/attributes.csv only;
no NetCDF time-series, which the structural-preflight tests cover
separately). No h2o/Moriah access, no training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.baseline.nh_config_generation import (
    COMPACT_SMOKE_RUN_PROFILE_NAME,
    EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME,
    KNOWN_RUN_PROFILE_NAMES,
    PILOT_LEAD06_EMB64X32_SEEDA_PROFILE_NAME,
    PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME,
    PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME,
    PILOT_LEAD06_EMB256X64_SEEDA_PROFILE_NAME,
    PROTECTED_GENERATED_TARGET_BASENAMES,
    NHConfigGenerationError,
    build_nh_config_mapping,
    generate_stage1_nh_config,
    get_run_profile_mapping,
    read_package_attribute_columns,
    validate_basin_membership,
    validate_dynamic_inputs,
    validate_lead_hours,
    validate_hidden_size_override,
    validate_embedding_dropout_override,
    validate_learning_rate_override,
    validate_max_updates_per_epoch,
    validate_seq_length,
    validate_static_attribute_contract,
    validate_statics_embedding_spec,
    validate_target_variables,
    write_generated_config,
)
from src.baseline.policy import load_stage1_baseline_policy
from src.baseline.splits import load_eligible_basins, sha256_of

REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"
SPLITS_DIR = REPO_ROOT / "config" / "stage1_baseline_splits_v001"

POLICY = load_stage1_baseline_policy(POLICY_PATH)
REAL_DYNAMIC_INPUTS = list(POLICY["dynamic_inputs"])
STATIC_COUNT = POLICY["static_attributes"]["expected_model_input_columns"]


def _pick_basins(n: int = 32) -> list:
    dev = load_eligible_basins(SPLITS_DIR / "development_train.txt")
    assert len(dev) >= n
    return dev[:n]


def _static_columns(n: int = STATIC_COUNT) -> list:
    return [f"col_{i:04d}" for i in range(n)]


def _build_fake_package(
    root: Path,
    basin_ids,
    *,
    static_columns=None,
    dynamic_variables=None,
    attrs_columns=None,
    extra_manifest_fields=None,
) -> Path:
    static_columns = list(static_columns if static_columns is not None else _static_columns())
    dynamic_variables = list(dynamic_variables if dynamic_variables is not None else REAL_DYNAMIC_INPUTS)
    attrs_columns = list(attrs_columns if attrs_columns is not None else static_columns)

    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    columns_sha256 = hashlib.sha256("\n".join(static_columns).encode("utf-8")).hexdigest()
    manifest = {
        "schema_name": "stage1_compact_package_manifest",
        "schema_version": 1,
        "package_role": "compact_scientific_package",
        "basin_count": len(basin_ids),
        "basin_ids": list(basin_ids),
        "dynamic_variables": dynamic_variables,
        "static_model_input_columns": static_columns,
        "static_model_input_columns_sha256": columns_sha256,
    }
    if extra_manifest_fields:
        manifest.update(extra_manifest_fields)
    (manifests_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    attrs_dir = root / "attributes"
    attrs_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"gauge_id": b, **{c: 0.0 for c in attrs_columns}} for b in basin_ids]
    df = pd.DataFrame(rows, columns=["gauge_id"] + attrs_columns)
    df.to_csv(attrs_dir / "attributes.csv", index=False)
    return root


# ---------------------------------------------------------------------------
# 1. exact lead06/seq24 generation, end to end
# ---------------------------------------------------------------------------

def test_generate_lead06_seq24_end_to_end(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)

    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    assert bundle.target_variable == "qobs_mm_per_h_lead06"
    assert bundle.seq_length == 24
    assert bundle.dynamic_inputs == REAL_DYNAMIC_INPUTS
    assert bundle.static_attribute_result.count == STATIC_COUNT
    assert set(bundle.basin_ids) == set(basins)
    assert len(bundle.basin_ids) == 32

    out_dir = tmp_path / "out"
    written = write_generated_config(bundle, out_dir, experiment_name="test_exp")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))

    assert cfg["dataset"] == "flashnh"
    assert cfg["seq_length"] == 24
    assert cfg["target_variables"] == ["qobs_mm_per_h_lead06"]
    assert cfg["dynamic_inputs"] == REAL_DYNAMIC_INPUTS
    assert len(cfg["static_attributes"]) == STATIC_COUNT
    assert cfg["predict_last_n"] == 1
    assert "nan_handling_method" not in cfg

    for label, path in (
        ("train", written["train_basins.txt"]),
        ("validation", written["validation_basins.txt"]),
        ("test", written["test_basins.txt"]),
    ):
        lines = path.read_text(encoding="utf-8").split()
        assert set(lines) == set(basins), f"{label} basin list mismatch"


# ---------------------------------------------------------------------------
# 5. exact dates / date format (DD/MM/YYYY per policy nh.date_format)
# ---------------------------------------------------------------------------

def test_generated_config_dates_exact(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    written = write_generated_config(bundle, tmp_path / "out")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))

    assert cfg["train_start_date"] == "14/10/2020"
    assert cfg["train_end_date"] == "31/12/2023"
    assert cfg["validation_start_date"] == "01/01/2024"
    assert cfg["validation_end_date"] == "31/12/2024"
    assert cfg["test_start_date"] == "01/01/2025"
    assert cfg["test_end_date"] == "31/12/2025"


# ---------------------------------------------------------------------------
# 2 & 3. target-variable rejection: multi-target and raw source variable
# ---------------------------------------------------------------------------

def test_validate_target_variables_rejects_multi_target():
    with pytest.raises(NHConfigGenerationError):
        validate_target_variables(["qobs_mm_per_h_lead06", "qobs_mm_per_h_lead12"], POLICY)


def test_validate_target_variables_rejects_raw_source_variable():
    with pytest.raises(NHConfigGenerationError):
        validate_target_variables(["qobs_m3s"], POLICY)


def test_validate_target_variables_accepts_single_lead_shifted_variable():
    validate_target_variables(["qobs_mm_per_h_lead06"], POLICY)  # must not raise


# ---------------------------------------------------------------------------
# 4. invalid lead / seq_length rejection
# ---------------------------------------------------------------------------

def test_validate_seq_length_rejects_forbidden_and_unknown_values():
    with pytest.raises(NHConfigGenerationError):
        validate_seq_length(168, POLICY)  # policy-forbidden (Stage 2)
    with pytest.raises(NHConfigGenerationError):
        validate_seq_length(36, POLICY)  # not in the approved set at all


def test_validate_lead_hours_rejects_unapproved_value():
    with pytest.raises(NHConfigGenerationError):
        validate_lead_hours(9, POLICY)


def test_generate_end_to_end_rejects_invalid_seq_length(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    with pytest.raises(NHConfigGenerationError):
        generate_stage1_nh_config(
            policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
            lead_hours=6, seq_length=168,
        )


# ---------------------------------------------------------------------------
# 6. exact dynamic-input order
# ---------------------------------------------------------------------------

def test_validate_dynamic_inputs_accepts_exact_match():
    validate_dynamic_inputs(list(REAL_DYNAMIC_INPUTS), POLICY)  # must not raise


def test_validate_dynamic_inputs_rejects_reordering():
    with pytest.raises(NHConfigGenerationError):
        validate_dynamic_inputs(list(reversed(REAL_DYNAMIC_INPUTS)), POLICY)


def test_validate_dynamic_inputs_rejects_subset():
    with pytest.raises(NHConfigGenerationError):
        validate_dynamic_inputs(REAL_DYNAMIC_INPUTS[:-1], POLICY)


# ---------------------------------------------------------------------------
# 7. exact 473-column static equality (count mismatch rejected)
# ---------------------------------------------------------------------------

def test_static_attribute_contract_accepts_exact_expected_count(tmp_path):
    package_root = tmp_path / "package"
    basins = _pick_basins(4)
    _build_fake_package(package_root, basins)
    manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    attrs_cols = read_package_attribute_columns(package_root)
    result = validate_static_attribute_contract(POLICY, manifest, attrs_cols)
    assert result.count == STATIC_COUNT


def test_static_attribute_contract_rejects_wrong_count(tmp_path):
    package_root = tmp_path / "package"
    basins = _pick_basins(4)
    cols = _static_columns(STATIC_COUNT - 1)
    _build_fake_package(package_root, basins, static_columns=cols)
    manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    attrs_cols = read_package_attribute_columns(package_root)
    with pytest.raises(NHConfigGenerationError):
        validate_static_attribute_contract(POLICY, manifest, attrs_cols)


# ---------------------------------------------------------------------------
# 8. static order-mismatch rejection
# ---------------------------------------------------------------------------

def test_static_attribute_contract_rejects_order_mismatch(tmp_path):
    package_root = tmp_path / "package"
    basins = _pick_basins(4)
    cols = _static_columns(10)
    _build_fake_package(package_root, basins, static_columns=cols, attrs_columns=list(reversed(cols)))
    manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    attrs_cols = read_package_attribute_columns(package_root)
    with pytest.raises(NHConfigGenerationError):
        validate_static_attribute_contract(POLICY, manifest, attrs_cols)


# ---------------------------------------------------------------------------
# 9. forbidden-static-field rejection
# ---------------------------------------------------------------------------

def test_static_attribute_contract_rejects_forbidden_field(tmp_path):
    package_root = tmp_path / "package"
    basins = _pick_basins(4)
    cols = _static_columns(STATIC_COUNT - 1) + ["STATE"]
    _build_fake_package(package_root, basins, static_columns=cols)
    manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    attrs_cols = read_package_attribute_columns(package_root)
    with pytest.raises(NHConfigGenerationError):
        validate_static_attribute_contract(POLICY, manifest, attrs_cols)


# ---------------------------------------------------------------------------
# 10. 8/9/15-char STAID handling in basin-membership validation
# ---------------------------------------------------------------------------

def test_validate_basin_membership_zero_pads_short_ids():
    dev = load_eligible_basins(SPLITS_DIR / "development_train.txt")
    candidates = [b for b in dev if b.startswith("0")]
    assert candidates, "expected at least one zero-padded development_train basin id"
    basin = candidates[0]
    short_form = basin.lstrip("0")
    assert len(short_form) < 8
    result = validate_basin_membership({"basin_ids": [short_form]}, SPLITS_DIR)
    assert result == [basin]


def test_validate_basin_membership_preserves_15_char_ids():
    dev = load_eligible_basins(SPLITS_DIR / "development_train.txt")
    long_ids = [b for b in dev if len(b) == 15]
    assert long_ids, "expected at least one 15-char development_train basin id"
    result = validate_basin_membership({"basin_ids": long_ids}, SPLITS_DIR)
    assert set(result) == set(long_ids)


def test_validate_basin_membership_preserves_8_char_ids():
    dev = load_eligible_basins(SPLITS_DIR / "development_train.txt")
    eight_char = [b for b in dev if len(b) == 8][:3]
    assert eight_char
    result = validate_basin_membership({"basin_ids": eight_char}, SPLITS_DIR)
    assert set(result) == set(eight_char)


# ---------------------------------------------------------------------------
# 11 & 12. exact compact basin membership + spatial-holdout/California rejection
# ---------------------------------------------------------------------------

def test_validate_basin_membership_rejects_non_development_train_id():
    with pytest.raises(NHConfigGenerationError):
        validate_basin_membership({"basin_ids": ["99999999"]}, SPLITS_DIR)


def test_validate_basin_membership_rejects_spatial_holdout_basin():
    holdout = load_eligible_basins(SPLITS_DIR / "spatial_holdout_nonca.txt")
    assert holdout
    with pytest.raises(NHConfigGenerationError):
        validate_basin_membership({"basin_ids": [holdout[0]]}, SPLITS_DIR)


def test_validate_basin_membership_rejects_california_basin():
    ca = load_eligible_basins(SPLITS_DIR / "california_all.txt")
    assert ca
    with pytest.raises(NHConfigGenerationError):
        validate_basin_membership({"basin_ids": [ca[0]]}, SPLITS_DIR)


def test_validate_basin_membership_rejects_duplicate_ids():
    basins = _pick_basins(2)
    with pytest.raises(NHConfigGenerationError):
        validate_basin_membership({"basin_ids": [basins[0], basins[0]]}, SPLITS_DIR)


# ---------------------------------------------------------------------------
# 13. output-directory safety
# ---------------------------------------------------------------------------

def test_write_generated_config_rejects_nonempty_out_dir_without_force(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "preexisting.txt").write_text("x", encoding="utf-8")

    with pytest.raises(NHConfigGenerationError):
        write_generated_config(bundle, out_dir)

    written = write_generated_config(bundle, out_dir, force=True)
    assert written["config.yaml"].is_file()


def _prepared_bundle(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    return generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )


@pytest.mark.parametrize("protected_name", sorted(PROTECTED_GENERATED_TARGET_BASENAMES))
def test_write_generated_config_rejects_protected_target_in_allowlist(tmp_path, protected_name):
    bundle = _prepared_bundle(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sentinel = f"sentinel bytes for {protected_name}".encode()
    (out_dir / protected_name).write_bytes(sentinel)
    (out_dir / "execution_provenance.json").write_bytes(b'{"trial_id": "t"}')

    with pytest.raises(NHConfigGenerationError):
        write_generated_config(bundle, out_dir, allowed_existing_files=frozenset({protected_name}))

    assert (out_dir / protected_name).read_bytes() == sentinel
    assert (out_dir / "execution_provenance.json").read_bytes() == b'{"trial_id": "t"}'
    for other_name in PROTECTED_GENERATED_TARGET_BASENAMES - {protected_name}:
        assert not (out_dir / other_name).exists()


def test_write_generated_config_allows_named_auxiliary_file_to_coexist(tmp_path):
    bundle = _prepared_bundle(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    aux = out_dir / "execution_provenance.json"
    aux_bytes = b'{"trial_id": "t", "provenance_stage": "prepared"}'
    aux.write_bytes(aux_bytes)

    written = write_generated_config(bundle, out_dir, allowed_existing_files=frozenset({"execution_provenance.json"}))

    assert written["config.yaml"].is_file()
    assert aux.read_bytes() == aux_bytes


def test_write_generated_config_rejects_allowlisted_name_that_is_a_directory(tmp_path):
    bundle = _prepared_bundle(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "execution_provenance.json").mkdir()

    with pytest.raises(NHConfigGenerationError):
        write_generated_config(bundle, out_dir, allowed_existing_files=frozenset({"execution_provenance.json"}))

    assert (out_dir / "execution_provenance.json").is_dir()
    assert not (out_dir / "config.yaml").exists()


# ---------------------------------------------------------------------------
# 14. nan_handling_method must be absent everywhere in the generated output
# ---------------------------------------------------------------------------

def test_generated_config_never_sets_nan_handling_method(tmp_path):
    basins = _pick_basins(4)
    package_root = _build_fake_package(tmp_path / "package", basins)
    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    assert "nan_handling_method" not in bundle.config_mapping
    written = write_generated_config(bundle, tmp_path / "out")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))
    assert "nan_handling_method" not in cfg


# ---------------------------------------------------------------------------
# 21. generated manifest checksums / identities
# ---------------------------------------------------------------------------

def test_generation_manifest_checksums_and_identity(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)
    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    written = write_generated_config(bundle, tmp_path / "out")
    manifest = json.loads(written["generation_manifest.json"].read_text(encoding="utf-8"))

    for name, path in written.items():
        if name == "generation_manifest.json":
            continue
        assert manifest["artifact_sha256"][name] == sha256_of(path), f"checksum mismatch for {name}"

    package_manifest = json.loads((package_root / "manifests" / "package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["package_manifest_identity"]["basin_count"] == package_manifest["basin_count"]
    assert (
        manifest["package_manifest_identity"]["static_model_input_columns_sha256"]
        == package_manifest["static_model_input_columns_sha256"]
    )
    assert manifest["nan_handling_method"] is None
    assert manifest["compact_smoke_run_profile"] is True
    assert manifest["nh_runtime_dataset_key"] == "flashnh"


# ---------------------------------------------------------------------------
# 22. Part I (section 13): embedded-static CudaLSTM pilot profile
# ---------------------------------------------------------------------------

def test_embedded_static_profile_is_registered_and_design_smoke_only():
    assert EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME in KNOWN_RUN_PROFILE_NAMES
    assert EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME == "embedded_static_cudalstm_pilot"


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "fc", "hiddens": [128, 64], "activation": "tanh", "dropout": 0.1},
        {"type": "fc", "hiddens": [32], "activation": "sigmoid", "dropout": 0.0},
        {"type": "fc", "hiddens": [16], "activation": "linear", "dropout": 0.999},
    ],
)
def test_validate_statics_embedding_spec_accepts_valid_specs(spec):
    validate_statics_embedding_spec(spec)  # must not raise


@pytest.mark.parametrize(
    "spec",
    [
        "not-a-dict",
        {"type": "lstm", "hiddens": [64], "activation": "tanh", "dropout": 0.1},
        {"type": "fc", "hiddens": [], "activation": "tanh", "dropout": 0.1},
        {"type": "fc", "hiddens": "64", "activation": "tanh", "dropout": 0.1},
        {"type": "fc", "hiddens": [0], "activation": "tanh", "dropout": 0.1},
        {"type": "fc", "hiddens": [-1], "activation": "tanh", "dropout": 0.1},
        {"type": "fc", "hiddens": [64.0], "activation": "tanh", "dropout": 0.1},
        {"type": "fc", "hiddens": [64], "activation": "relu", "dropout": 0.1},
        {"type": "fc", "hiddens": [64], "activation": "tanh", "dropout": 1.0},
        {"type": "fc", "hiddens": [64], "activation": "tanh", "dropout": -0.1},
        {"type": "fc", "hiddens": [64], "activation": "tanh", "dropout": "0.1"},
    ],
)
def test_validate_statics_embedding_spec_rejects_invalid_specs(spec):
    with pytest.raises(NHConfigGenerationError):
        validate_statics_embedding_spec(spec)


def test_generate_embedded_static_cudalstm_pilot_end_to_end(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)

    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
        run_profile_name=EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME,
    )
    assert bundle.run_profile_name == EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME
    assert bundle.config_mapping["statics_embedding"]["type"] == "fc"

    written = write_generated_config(bundle, tmp_path / "out", experiment_name="test_embedded_static_pilot")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))
    assert cfg["statics_embedding"] == {
        "type": "fc",
        "hiddens": [128, 64],
        "activation": "tanh",
        "dropout": 0.1,
    }
    assert cfg["model"] == "cudalstm"
    # This is a structural-smoke config only -- not the full-population seed
    # profile's epoch/hidden_size/batch_size scale (section 13 scope).
    assert cfg["epochs"] == 2

    manifest = json.loads(written["generation_manifest.json"].read_text(encoding="utf-8"))
    assert manifest["run_profile_name"] == EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE_NAME
    assert "compact_smoke_run_profile" not in manifest


def test_generate_rejects_invalid_statics_embedding_in_profile(tmp_path, monkeypatch):
    import src.baseline.nh_config_generation as mod

    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)

    bad_profile_name = "test_bad_statics_embedding_profile"
    monkeypatch.setitem(mod._RUN_PROFILES, bad_profile_name, {
        "model": "cudalstm",
        "hidden_size": 64,
        "epochs": 2,
        "statics_embedding": {"type": "fc", "hiddens": [], "activation": "tanh", "dropout": 0.1},
    })
    monkeypatch.setitem(mod._RUN_PROFILE_NOTES, bad_profile_name, "test fixture")

    with pytest.raises(NHConfigGenerationError):
        generate_stage1_nh_config(
            policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
            lead_hours=6, seq_length=24, run_profile_name=bad_profile_name,
        )


# ---------------------------------------------------------------------------
# 23. lead-6 embedding-shape neighborhood profiles ([64,32]/[128,32]/[256,64])
# canonical registry additions for the next approved 25k embedding-shape
# batch (runtime-only calibration cannot invent new _RUN_PROFILES entries).
# ---------------------------------------------------------------------------

_LEAD06_NEIGHBORHOOD_PROFILES = [
    (PILOT_LEAD06_EMB64X32_SEEDA_PROFILE_NAME, [64, 32]),
    (PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME, [128, 32]),
    (PILOT_LEAD06_EMB256X64_SEEDA_PROFILE_NAME, [256, 64]),
]

# Fields that must match the [128,64] reference profile exactly (everything
# except profile identity, statics_embedding.hiddens, and the manifest note).
_LEAD06_REFERENCE_MATCH_KEYS = (
    "seed", "model", "hidden_size", "output_dropout", "batch_size",
    "optimizer", "learning_rate", "loss", "epochs", "device",
)


def test_lead06_embedding_neighborhood_profile_names_are_registered():
    for profile_name, _hiddens in _LEAD06_NEIGHBORHOOD_PROFILES:
        assert profile_name in KNOWN_RUN_PROFILE_NAMES


@pytest.mark.parametrize("profile_name,hiddens", _LEAD06_NEIGHBORHOOD_PROFILES)
def test_lead06_embedding_neighborhood_profile_mapping_matches_reference(profile_name, hiddens):
    reference = get_run_profile_mapping(PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME)
    candidate = get_run_profile_mapping(profile_name)

    for key in _LEAD06_REFERENCE_MATCH_KEYS:
        assert candidate[key] == reference[key], f"field {key!r} diverged from [128,64] reference"

    assert candidate["statics_embedding"]["hiddens"] == hiddens
    assert candidate["statics_embedding"]["activation"] == reference["statics_embedding"]["activation"] == "tanh"
    assert candidate["statics_embedding"]["dropout"] == reference["statics_embedding"]["dropout"] == 0.1
    assert candidate["statics_embedding"]["type"] == reference["statics_embedding"]["type"] == "fc"

    # No canonical profile encodes a per-epoch training-batch cap or a
    # permanent pilot-policy run entry; those remain runtime-only.
    assert "max_updates_per_epoch" not in candidate

    # Differ from the reference only in embedding hiddens (name/note are
    # registry keys, not mapping fields, so the mapping itself differs only
    # in statics_embedding).
    assert set(candidate) == set(reference)
    diverging_keys = {k for k in candidate if candidate[k] != reference[k]}
    assert diverging_keys == {"statics_embedding"}


@pytest.mark.parametrize("profile_name,hiddens", _LEAD06_NEIGHBORHOOD_PROFILES)
def test_generate_lead06_embedding_neighborhood_profile_end_to_end(tmp_path, profile_name, hiddens):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)

    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24, run_profile_name=profile_name,
    )
    assert bundle.run_profile_name == profile_name
    assert bundle.target_variable == "qobs_mm_per_h_lead06"
    assert bundle.seq_length == 24
    assert bundle.dynamic_inputs == REAL_DYNAMIC_INPUTS
    assert bundle.config_mapping["statics_embedding"]["hiddens"] == hiddens
    assert bundle.config_mapping["seed"] == 967139
    assert bundle.config_mapping["hidden_size"] == 128
    assert bundle.config_mapping["batch_size"] == 256
    assert bundle.config_mapping["optimizer"] == "Adam"
    assert bundle.config_mapping["learning_rate"] == 0.001
    assert bundle.config_mapping["loss"] == "NSE"
    assert bundle.config_mapping["output_dropout"] == 0.25

    written = write_generated_config(bundle, tmp_path / "out", experiment_name=f"test_{profile_name}")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))
    assert cfg["statics_embedding"] == {
        "type": "fc", "hiddens": hiddens, "activation": "tanh", "dropout": 0.1,
    }
    assert cfg["target_variables"] == ["qobs_mm_per_h_lead06"]
    assert cfg["seed"] == 967139

    manifest = json.loads(written["generation_manifest.json"].read_text(encoding="utf-8"))
    assert manifest["run_profile_name"] == profile_name
    assert manifest["run_profile_note"]  # non-empty, records the correct profile identity


def test_lead06_embedding_neighborhood_profiles_do_not_alter_existing_emb128x64_seedA():
    reference = get_run_profile_mapping(PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME)
    assert reference["statics_embedding"]["hiddens"] == [128, 64]
    assert reference["seed"] == 967139
    assert reference["model"] == "cudalstm"
    assert reference["hidden_size"] == 128
    assert reference["output_dropout"] == 0.25
    assert reference["batch_size"] == 256
    assert reference["optimizer"] == "Adam"
    assert reference["learning_rate"] == 0.001
    assert reference["loss"] == "NSE"


def test_unknown_lead06_neighborhood_profile_name_still_rejected():
    with pytest.raises(NHConfigGenerationError):
        get_run_profile_mapping("pilot_lead06_emb999x999_seedA_v001")


def test_unknown_run_profile_name_rejected_at_generation(tmp_path):
    basins = _pick_basins(4)
    package_root = _build_fake_package(tmp_path / "package", basins)
    with pytest.raises(NHConfigGenerationError):
        generate_stage1_nh_config(
            policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
            lead_hours=6, seq_length=24, run_profile_name="pilot_lead06_emb999x999_seedA_v001",
        )


# ---------------------------------------------------------------------------
# max_updates_per_epoch: optional per-epoch NH training-batch cap for cheap
# early-fidelity screening (efficiency feature, not a numerical-cap decision;
# uncapped/None remains the default for every pre-existing caller).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1, 5, 100])
def test_validate_max_updates_per_epoch_accepts_positive_ints(value):
    validate_max_updates_per_epoch(value)  # must not raise


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, -100, 1.5, "5", None, [], {}],
)
def test_validate_max_updates_per_epoch_rejects_invalid_values(value):
    with pytest.raises(NHConfigGenerationError):
        validate_max_updates_per_epoch(value)


# ---------------------------------------------------------------------------
# learning_rate: LR-A range-characterization campaign's per-candidate
# learning-rate override (see docs/decision_log.md's LR-A design-freeze
# entry and validate_learning_rate_override's own docstring). Pre-commit
# review found this validator implemented but untested; this block closes
# that gap. No implementation change.
#
# validate_learning_rate_override(value) itself only ever receives a
# non-None override (its docstring: "None ... is never passed to this
# function -- callers only call it once they already know an override was
# requested") -- exactly the same design already established by
# validate_max_updates_per_epoch above, which likewise rejects None when
# called directly. The LR-override *feature*'s tolerance of "no override"
# is therefore exercised at build_nh_config_mapping's level (see
# test_build_nh_config_mapping_omits_learning_rate_key_when_no_override
# below), not by asserting validate_learning_rate_override(None) succeeds.
# ---------------------------------------------------------------------------

_LR_A_CANDIDATE_LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


@pytest.mark.parametrize("value", _LR_A_CANDIDATE_LEARNING_RATES)
def test_validate_learning_rate_override_accepts_lr_a_candidate_values(value):
    validate_learning_rate_override(value)  # must not raise


@pytest.mark.parametrize("value", [1, 5, 0.5])
def test_validate_learning_rate_override_accepts_other_positive_reals(value):
    validate_learning_rate_override(value)  # must not raise


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, -0.001, "0.001", [], {}, float("nan"), float("inf"), float("-inf")],
)
def test_validate_learning_rate_override_rejects_invalid_values(value):
    with pytest.raises(NHConfigGenerationError, match="learning_rate override"):
        validate_learning_rate_override(value)


def test_validate_learning_rate_override_rejects_none_directly_by_design():
    # By design (see this test block's header comment): the validator itself
    # never accepts None -- only build_nh_config_mapping's None-means-
    # "no override" short-circuit does, without ever calling this function.
    with pytest.raises(NHConfigGenerationError, match="learning_rate override"):
        validate_learning_rate_override(None)


def _build_mapping_kwargs(**overrides):
    kwargs = dict(
        policy=POLICY,
        target_variable="qobs_mm_per_h_lead06",
        seq_length=24,
        dynamic_inputs=REAL_DYNAMIC_INPUTS,
        static_attributes=_static_columns(),
    )
    kwargs.update(overrides)
    return kwargs


def test_build_nh_config_mapping_omits_key_when_uncapped():
    mapping = build_nh_config_mapping(**_build_mapping_kwargs())
    assert "max_updates_per_epoch" not in mapping


def test_build_nh_config_mapping_includes_exact_int_when_capped():
    mapping = build_nh_config_mapping(**_build_mapping_kwargs(max_updates_per_epoch=25))
    assert mapping["max_updates_per_epoch"] == 25


def test_build_nh_config_mapping_rejects_invalid_cap():
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(max_updates_per_epoch=True))
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(max_updates_per_epoch=0))


def test_build_nh_config_mapping_omits_learning_rate_key_when_no_override():
    # "Accepted: None" at the LR-override *feature* level (see the
    # validate_learning_rate_override test block above): no override means
    # the profile's own learning_rate is left completely untouched, and the
    # validator is never invoked at all.
    mapping = build_nh_config_mapping(**_build_mapping_kwargs())
    default_mapping = build_nh_config_mapping(**_build_mapping_kwargs(learning_rate=None))
    assert mapping == default_mapping
    assert "learning_rate" in mapping  # the profile's own default value


@pytest.mark.parametrize("value", _LR_A_CANDIDATE_LEARNING_RATES)
def test_build_nh_config_mapping_applies_learning_rate_override(value):
    mapping = build_nh_config_mapping(**_build_mapping_kwargs(learning_rate=value))
    assert mapping["learning_rate"] == pytest.approx(value)


def test_build_nh_config_mapping_rejects_invalid_learning_rate_override():
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(learning_rate=True))
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(learning_rate=0))
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(learning_rate=-1e-3))
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(learning_rate=float("nan")))


# ---------------------------------------------------------------------------
# hidden_size: Hidden-size-A range-characterization campaign's per-candidate
# hidden-size override (see docs/decision_log.md's 2026-08-09 Hidden-size-A
# design-freeze entry and validate_hidden_size_override's own docstring).
# Mirrors the learning_rate test block above field-for-field.
#
# validate_hidden_size_override(value) itself only ever receives a non-None
# override (its docstring: same "None ... is never passed to this function"
# design already established by validate_learning_rate_override/
# validate_max_updates_per_epoch above). The hidden-size-override *feature*'s
# tolerance of "no override" is therefore exercised at
# build_nh_config_mapping's level (see
# test_build_nh_config_mapping_omits_hidden_size_key_when_no_override
# below), not by asserting validate_hidden_size_override(None) succeeds.
# ---------------------------------------------------------------------------

_HIDDEN_SIZE_A_CANDIDATE_HIDDEN_SIZES = [64, 128, 256, 512]


@pytest.mark.parametrize("value", _HIDDEN_SIZE_A_CANDIDATE_HIDDEN_SIZES)
def test_validate_hidden_size_override_accepts_hidden_size_a_candidate_values(value):
    validate_hidden_size_override(value)  # must not raise


@pytest.mark.parametrize("value", [1, 16, 1024])
def test_validate_hidden_size_override_accepts_other_positive_ints(value):
    validate_hidden_size_override(value)  # must not raise


@pytest.mark.parametrize(
    "value",
    [True, False, 0, -1, -128, 1.5, "128", [], {}, None],
)
def test_validate_hidden_size_override_rejects_invalid_values(value):
    with pytest.raises(NHConfigGenerationError, match="hidden_size override"):
        validate_hidden_size_override(value)


def test_build_nh_config_mapping_omits_hidden_size_key_when_no_override():
    # "Accepted: None" at the hidden-size-override *feature* level (see the
    # validate_hidden_size_override test block above): no override means the
    # profile's own hidden_size is left completely untouched, and the
    # validator is never invoked at all.
    mapping = build_nh_config_mapping(**_build_mapping_kwargs())
    default_mapping = build_nh_config_mapping(**_build_mapping_kwargs(hidden_size=None))
    assert mapping == default_mapping
    assert "hidden_size" in mapping  # the profile's own default value


@pytest.mark.parametrize("value", _HIDDEN_SIZE_A_CANDIDATE_HIDDEN_SIZES)
def test_build_nh_config_mapping_applies_hidden_size_override(value):
    mapping = build_nh_config_mapping(**_build_mapping_kwargs(hidden_size=value))
    assert mapping["hidden_size"] == value


def test_build_nh_config_mapping_rejects_invalid_hidden_size_override():
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(hidden_size=True))
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(hidden_size=0))
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(hidden_size=-64))
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(**_build_mapping_kwargs(hidden_size=64.0))


# ---------------------------------------------------------------------------
# embedding_dropout: Embedding-Dropout-A range-characterization campaign's
# per-candidate embedding-dropout override (see docs/decision_log.md's
# Embedding-Dropout-A design-freeze entry and
# validate_embedding_dropout_override's own docstring). Mirrors the
# hidden_size test block above field-for-field, plus explicit 0.0-vs-None
# coverage (the drop00 candidate) and the raw-pathway rejection this override
# is unique in needing (a hidden_size/learning_rate override applies
# regardless of pathway; embedding_dropout only makes sense when the
# profile's mapping actually has a statics_embedding section).
#
# validate_embedding_dropout_override(value) itself only ever receives a
# non-None override -- same "None is never passed to this function" design
# already established by validate_hidden_size_override/
# validate_learning_rate_override above. The embedding-dropout-override
# *feature*'s tolerance of "no override" is therefore exercised at
# build_nh_config_mapping's level (see
# test_build_nh_config_mapping_omits_embedding_dropout_key_when_no_override
# below), not by asserting validate_embedding_dropout_override(None)
# succeeds.
# ---------------------------------------------------------------------------

_EMBEDDING_DROPOUT_A_CANDIDATE_VALUES = [0.00, 0.05, 0.10, 0.20, 0.40]


@pytest.mark.parametrize("value", _EMBEDDING_DROPOUT_A_CANDIDATE_VALUES)
def test_validate_embedding_dropout_override_accepts_embedding_dropout_a_candidate_values(value):
    validate_embedding_dropout_override(value)  # must not raise


def test_validate_embedding_dropout_override_accepts_explicit_zero_not_confused_with_falsy():
    # 0.0 (the drop00 candidate) must be accepted like any other in-range
    # value -- this is a plain numeric-range check, never a truthiness check.
    validate_embedding_dropout_override(0.0)  # must not raise


@pytest.mark.parametrize("value", [0.15, 0.3, 0.999, 1e-9])
def test_validate_embedding_dropout_override_accepts_other_in_range_reals(value):
    validate_embedding_dropout_override(value)  # must not raise


@pytest.mark.parametrize(
    "value",
    [True, False, 1.0, 1.5, -0.001, -1, "0.1", [], {}, float("nan"), float("inf"), float("-inf")],
)
def test_validate_embedding_dropout_override_rejects_invalid_values(value):
    with pytest.raises(NHConfigGenerationError, match="embedding_dropout override"):
        validate_embedding_dropout_override(value)


def test_validate_embedding_dropout_override_rejects_none_directly_by_design():
    # By design (see this test block's header comment): the validator itself
    # never accepts None -- only build_nh_config_mapping's None-means-
    # "no override" short-circuit does, without ever calling this function.
    with pytest.raises(NHConfigGenerationError, match="embedding_dropout override"):
        validate_embedding_dropout_override(None)


def test_build_nh_config_mapping_omits_embedding_dropout_key_when_no_override():
    # "Accepted: None" at the embedding-dropout-override *feature* level (see
    # the validate_embedding_dropout_override test block above): no override
    # means the profile's own statics_embedding.dropout is left completely
    # untouched, and the validator is never invoked at all.
    kwargs = _build_mapping_kwargs(run_profile_name=PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME)
    mapping = build_nh_config_mapping(**kwargs)
    default_mapping = build_nh_config_mapping(**{**kwargs, "embedding_dropout": None})
    assert mapping == default_mapping
    assert mapping["statics_embedding"]["dropout"] == 0.1  # the profile's own default value


@pytest.mark.parametrize("value", _EMBEDDING_DROPOUT_A_CANDIDATE_VALUES)
def test_build_nh_config_mapping_applies_embedding_dropout_override(value):
    mapping = build_nh_config_mapping(
        **_build_mapping_kwargs(
            run_profile_name=PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME, embedding_dropout=value
        )
    )
    assert mapping["statics_embedding"]["dropout"] == pytest.approx(value)


def test_build_nh_config_mapping_applies_embedding_dropout_override_leaves_shape_and_activation_untouched():
    mapping = build_nh_config_mapping(
        **_build_mapping_kwargs(
            run_profile_name=PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME, embedding_dropout=0.40
        )
    )
    assert mapping["statics_embedding"]["hiddens"] == [128, 32]
    assert mapping["statics_embedding"]["activation"] == "tanh"


def test_build_nh_config_mapping_embedding_dropout_override_does_not_mutate_shared_profile_registry():
    # Copy-before-mutate safety: applying an override must never permanently
    # corrupt the module-level _RUN_PROFILES registry for later callers that
    # reuse the same run_profile_name (see build_nh_config_mapping's own
    # copy-before-mutate comment).
    from src.baseline import nh_config_generation as _nh_config_generation_module

    before = dict(
        _nh_config_generation_module._RUN_PROFILES[PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME]["statics_embedding"]
    )
    build_nh_config_mapping(
        **_build_mapping_kwargs(
            run_profile_name=PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME, embedding_dropout=0.40
        )
    )
    after = _nh_config_generation_module._RUN_PROFILES[PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME]["statics_embedding"]
    assert after == before
    assert after["dropout"] == 0.1


def test_build_nh_config_mapping_rejects_invalid_embedding_dropout_override():
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(
            **_build_mapping_kwargs(
                run_profile_name=PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME, embedding_dropout=True
            )
        )
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(
            **_build_mapping_kwargs(
                run_profile_name=PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME, embedding_dropout=1.0
            )
        )
    with pytest.raises(NHConfigGenerationError):
        build_nh_config_mapping(
            **_build_mapping_kwargs(
                run_profile_name=PILOT_LEAD06_EMB128X32_SEEDA_PROFILE_NAME, embedding_dropout=-0.1
            )
        )


def test_build_nh_config_mapping_rejects_embedding_dropout_override_on_raw_pathway_profile():
    # embedding_dropout only makes sense when the profile's mapping actually
    # has a statics_embedding section to override -- unlike hidden_size/
    # learning_rate, which apply regardless of static pathway. Uses the
    # default (raw, no statics_embedding) run_profile_name.
    with pytest.raises(NHConfigGenerationError, match="statics_embedding"):
        build_nh_config_mapping(**_build_mapping_kwargs(embedding_dropout=0.10))


def test_generate_stage1_nh_config_uncapped_default_omits_key_everywhere(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)

    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24,
    )
    assert bundle.max_updates_per_epoch is None
    assert "max_updates_per_epoch" not in bundle.config_mapping

    written = write_generated_config(bundle, tmp_path / "out")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))
    assert "max_updates_per_epoch" not in cfg

    manifest = json.loads(written["generation_manifest.json"].read_text(encoding="utf-8"))
    assert manifest["max_updates_per_epoch"] is None


def test_generate_stage1_nh_config_capped_threads_exact_value(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)

    bundle = generate_stage1_nh_config(
        policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
        lead_hours=6, seq_length=24, max_updates_per_epoch=10,
    )
    assert bundle.max_updates_per_epoch == 10
    assert bundle.config_mapping["max_updates_per_epoch"] == 10

    written = write_generated_config(bundle, tmp_path / "out")
    cfg = yaml.safe_load(written["config.yaml"].read_text(encoding="utf-8"))
    assert cfg["max_updates_per_epoch"] == 10

    manifest = json.loads(written["generation_manifest.json"].read_text(encoding="utf-8"))
    assert manifest["max_updates_per_epoch"] == 10


def test_generate_stage1_nh_config_rejects_invalid_cap(tmp_path):
    basins = _pick_basins(32)
    package_root = _build_fake_package(tmp_path / "package", basins)

    with pytest.raises(NHConfigGenerationError):
        generate_stage1_nh_config(
            policy_path=POLICY_PATH, package_root=package_root, splits_dir=SPLITS_DIR,
            lead_hours=6, seq_length=24, max_updates_per_epoch=-5,
        )
