"""Focused tests for src/baseline/pilot_lead06_config.py (task item 10).

Covers: the real committed policy loads and matches the closed 6-run
matrix; Seed A/B resolution; rejection of a declared run-id set that omits
or adds to the 6-run matrix; run_profile_name cross-check;
load_screening_basin_ids's subset/count-range/no-duplicate/not-equal-to-
full-population checks; build_pilot_bundle's sealed-population rejection;
build_all_pilot_bundles's full 6-bundle construction.

Uses only the real committed policy/config files plus a synthetic fake
package (tests/_pilot_support.py) -- no Moriah/GPU/W&B/network required.
"""
from __future__ import annotations

import copy
import dataclasses

import pytest
import yaml

import src.baseline.nh_config_generation as nh_config_generation
from src.baseline.nh_config_generation import PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME
from src.baseline.pilot_lead06_config import (
    PilotConfigError,
    build_all_pilot_bundles,
    build_pilot_bundle,
    load_pilot_policy,
    load_screening_basin_ids,
    pilot_run_ids,
    resolve_pilot_run_spec,
    SCREENING_VALIDATION_POPULATION_ROLE,
)
from src.baseline.splits import sha256_of

from tests._pilot_support import (
    BASELINE_POLICY_PATH,
    PILOT_POLICY_PATH,
    REAL_DEVELOPMENT,
    REPO_ROOT,
    SPLITS_DIR,
    build_full_union_package,
    pick_development_basins,
    write_screening_basin_ids_file,
)

EXPECTED_RUN_IDS = {
    "raw_seedA", "raw_seedB", "emb128x64_seedA", "emb128x64_seedB",
    "emb64_seedA", "emb128_seedA",
}

# The accepted provisional 400-basin screening realization (Part D of the
# Stage 1 validation/optimization foundation report, selection_v001/,
# seed=42) -- independently confirmed via `sha256sum` and cross-checked
# against selection_manifest.json's own recorded artifact_sha256.
REAL_SCREENING_BASIN_IDS_PATH = (
    REPO_ROOT
    / "reports/stage1_validation_optimization_foundation_v001/part_d_screening_subset"
    / "selection_v001/screening_subset_basin_ids.txt"
)
REAL_SCREENING_EXPECTED_COUNT = 400
REAL_SCREENING_EXPECTED_SHA256 = "d4395d93ebc567cf09e149c0121463d75cf4f7ecc02c07a7c4a7999763baa372"


def _raw_policy_dict() -> dict:
    with open(PILOT_POLICY_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_policy(tmp_path, raw: dict, name="pilot_policy.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


# --- real committed policy: closed 6-run matrix ------------------------------

def test_real_pilot_policy_declares_exactly_the_closed_six_run_matrix():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    assert set(pilot_run_ids(policy)) == EXPECTED_RUN_IDS
    assert len(policy.runs) == 6


def test_real_pilot_policy_seed_a_and_seed_b_values():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    assert policy.seeds["seed_a"] == 967139
    assert policy.seeds["seed_b"] == 1729
    assert policy.seeds["seed_a"] != policy.seeds["seed_b"]


def test_real_pilot_policy_run_seed_resolution():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    assert resolve_pilot_run_spec(policy, "raw_seedA").seed == 967139
    assert resolve_pilot_run_spec(policy, "raw_seedB").seed == 1729
    assert resolve_pilot_run_spec(policy, "emb128x64_seedA").seed == 967139
    assert resolve_pilot_run_spec(policy, "emb128x64_seedB").seed == 1729
    assert resolve_pilot_run_spec(policy, "emb64_seedA").seed == 967139
    assert resolve_pilot_run_spec(policy, "emb128_seedA").seed == 967139


def test_real_pilot_policy_embedding_hiddens_per_run():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    assert resolve_pilot_run_spec(policy, "raw_seedA").embedding_hiddens is None
    assert resolve_pilot_run_spec(policy, "raw_seedB").embedding_hiddens is None
    assert resolve_pilot_run_spec(policy, "emb128x64_seedA").embedding_hiddens == [128, 64]
    assert resolve_pilot_run_spec(policy, "emb128x64_seedB").embedding_hiddens == [128, 64]
    assert resolve_pilot_run_spec(policy, "emb64_seedA").embedding_hiddens == [64]
    assert resolve_pilot_run_spec(policy, "emb128_seedA").embedding_hiddens == [128]


def test_real_pilot_policy_run_profile_names_match_registry():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    for run_id, spec in policy.runs.items():
        assert spec.run_profile_name == PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME[run_id]


def test_real_pilot_policy_workflow_qualification_run_is_emb128x64_seedA():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    assert policy.workflow_qualification_run_id == "emb128x64_seedA"


def test_real_pilot_policy_screening_pin_is_exact_and_matches_artifact_on_disk():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    assert policy.screening_expected_count == REAL_SCREENING_EXPECTED_COUNT
    assert policy.screening_expected_sha256 == REAL_SCREENING_EXPECTED_SHA256
    artifact_path = REPO_ROOT / policy.screening_basin_ids_path
    assert sha256_of(artifact_path) == REAL_SCREENING_EXPECTED_SHA256


def test_real_pilot_policy_epoch_budget_and_cadence():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    assert policy.pilot_max_epoch_budget == 36
    assert policy.screening_validation_every_n_epochs == 3
    assert policy.diagnostic_only_epoch == 3
    assert policy.stopping_eligible_from_epoch == 6


def test_resolve_pilot_run_spec_rejects_unknown_run_id():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    with pytest.raises(PilotConfigError):
        resolve_pilot_run_spec(policy, "does_not_exist")


# --- rejection: run_id set must exactly equal the closed 6-run matrix -------

def test_load_pilot_policy_rejects_incomplete_run_matrix(tmp_path):
    raw = _raw_policy_dict()
    raw["runs"] = [r for r in raw["runs"] if r["run_id"] != "emb128_seedA"]
    raw["workflow_qualification_run_id"] = "raw_seedA"
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_unintended_seventh_run(tmp_path):
    raw = _raw_policy_dict()
    extra = copy.deepcopy(raw["runs"][0])
    extra["run_id"] = "raw_seedC_unintended"
    extra["seed_name"] = "seed_a"
    raw["runs"].append(extra)
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_run_id_with_no_registry_entry(tmp_path):
    raw = _raw_policy_dict()
    raw["runs"][0]["run_id"] = "totally_unknown_run_id"
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_mismatched_run_profile_name(tmp_path):
    raw = _raw_policy_dict()
    raw["runs"][0]["run_profile_name"] = "some_other_profile_name"
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_duplicate_run_id(tmp_path):
    raw = _raw_policy_dict()
    dup = copy.deepcopy(raw["runs"][0])
    raw["runs"].append(dup)
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_equal_seeds(tmp_path):
    raw = _raw_policy_dict()
    raw["seeds"]["seed_b"] = raw["seeds"]["seed_a"]
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_unknown_workflow_qualification_run_id(tmp_path):
    raw = _raw_policy_dict()
    raw["workflow_qualification_run_id"] = "not_a_declared_run_id"
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_missing_top_level_key(tmp_path):
    raw = _raw_policy_dict()
    del raw["screening"]
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_missing_file_raises(tmp_path):
    with pytest.raises(PilotConfigError):
        load_pilot_policy(tmp_path / "does_not_exist.yaml")


# --- six-run semantic contract mutation tests (task item 5) ----------------

def test_load_pilot_policy_rejects_raw_run_falsely_declaring_an_embedding(tmp_path):
    raw = _raw_policy_dict()
    entry = next(r for r in raw["runs"] if r["run_id"] == "raw_seedA")
    entry["static_pathway"] = "learned_fc_embedding"
    entry["embedding_hiddens"] = [128, 64]
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_embedded_run_falsely_declaring_identity(tmp_path):
    raw = _raw_policy_dict()
    entry = next(r for r in raw["runs"] if r["run_id"] == "emb128x64_seedA")
    entry["static_pathway"] = "raw_identity_concatenation"
    entry["embedding_hiddens"] = None
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_wrong_embedding_shape(tmp_path):
    raw = _raw_policy_dict()
    entry = next(r for r in raw["runs"] if r["run_id"] == "emb128x64_seedA")
    entry["embedding_hiddens"] = [64]
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_wrong_seed_assignment(tmp_path):
    raw = _raw_policy_dict()
    entry = next(r for r in raw["runs"] if r["run_id"] == "emb64_seedA")
    entry["seed_name"] = "seed_b"
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_wrong_top_level_embedding_activation(tmp_path):
    raw = _raw_policy_dict()
    raw["embedding_activation"] = "relu"
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_wrong_top_level_embedding_dropout(tmp_path):
    raw = _raw_policy_dict()
    raw["embedding_dropout"] = 0.5
    path = _write_policy(tmp_path, raw)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(path)


def test_load_pilot_policy_rejects_run_profile_registry_mismatch_via_monkeypatch(monkeypatch):
    # The declared policy YAML is untouched and internally consistent -- this
    # proves the NEW cross-check against the ACTUAL _RUN_PROFILES registry
    # entry (get_run_profile_mapping) catches a mismatch that pure
    # declared-vs-declared YAML validation could never see: here the raw
    # run's registered profile is mutated to secretly define an embedding.
    profile_name = PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME["raw_seedA"]
    mutated_profile = nh_config_generation.get_run_profile_mapping(profile_name)
    mutated_profile["statics_embedding"] = {"hiddens": [64], "activation": "tanh", "dropout": 0.1}
    monkeypatch.setitem(nh_config_generation._RUN_PROFILES, profile_name, mutated_profile)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(PILOT_POLICY_PATH)


def test_load_pilot_policy_rejects_embedded_run_profile_secretly_dropping_embedding(monkeypatch):
    profile_name = PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME["emb128x64_seedA"]
    mutated_profile = nh_config_generation.get_run_profile_mapping(profile_name)
    del mutated_profile["statics_embedding"]
    monkeypatch.setitem(nh_config_generation._RUN_PROFILES, profile_name, mutated_profile)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(PILOT_POLICY_PATH)


def test_load_pilot_policy_rejects_profile_embedding_shape_mismatch_via_monkeypatch(monkeypatch):
    profile_name = PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME["emb128x64_seedA"]
    mutated_profile = nh_config_generation.get_run_profile_mapping(profile_name)
    # get_run_profile_mapping only shallow-copies -- "statics_embedding" is
    # still the SAME nested dict object as the real registry entry, so it
    # must be copied again before mutating a field on it, or this leaks a
    # permanent corruption of module state into every later test.
    mutated_profile["statics_embedding"] = dict(mutated_profile["statics_embedding"])
    mutated_profile["statics_embedding"]["hiddens"] = [256]
    monkeypatch.setitem(nh_config_generation._RUN_PROFILES, profile_name, mutated_profile)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(PILOT_POLICY_PATH)


def test_load_pilot_policy_rejects_profile_embedding_activation_mismatch_via_monkeypatch(monkeypatch):
    profile_name = PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME["emb128x64_seedA"]
    mutated_profile = nh_config_generation.get_run_profile_mapping(profile_name)
    mutated_profile["statics_embedding"] = dict(mutated_profile["statics_embedding"])
    mutated_profile["statics_embedding"]["activation"] = "relu"
    monkeypatch.setitem(nh_config_generation._RUN_PROFILES, profile_name, mutated_profile)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(PILOT_POLICY_PATH)


def test_load_pilot_policy_rejects_profile_embedding_dropout_mismatch_via_monkeypatch(monkeypatch):
    profile_name = PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME["emb128x64_seedA"]
    mutated_profile = nh_config_generation.get_run_profile_mapping(profile_name)
    mutated_profile["statics_embedding"] = dict(mutated_profile["statics_embedding"])
    mutated_profile["statics_embedding"]["dropout"] = 0.5
    monkeypatch.setitem(nh_config_generation._RUN_PROFILES, profile_name, mutated_profile)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(PILOT_POLICY_PATH)


def test_load_pilot_policy_rejects_profile_seed_mismatch_via_monkeypatch(monkeypatch):
    profile_name = PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME["raw_seedA"]
    mutated_profile = nh_config_generation.get_run_profile_mapping(profile_name)
    mutated_profile["seed"] = 1
    monkeypatch.setitem(nh_config_generation._RUN_PROFILES, profile_name, mutated_profile)
    with pytest.raises(PilotConfigError):
        load_pilot_policy(PILOT_POLICY_PATH)


# --- load_screening_basin_ids contract: exact-pin (task item 3) ------------

def test_load_screening_basin_ids_accepts_the_exact_accepted_screening_realization():
    result = load_screening_basin_ids(
        REAL_SCREENING_BASIN_IDS_PATH,
        development_basins=REAL_DEVELOPMENT,
        expected_count=REAL_SCREENING_EXPECTED_COUNT,
        expected_sha256=REAL_SCREENING_EXPECTED_SHA256,
    )
    assert len(result) == REAL_SCREENING_EXPECTED_COUNT


def test_load_screening_basin_ids_accepts_valid_proper_subset_when_pin_matches(tmp_path):
    # Not the real accepted realization -- a synthetic subset whose exact
    # count/sha256 pin is passed in matching, to isolate this function's
    # general subset-acceptance logic from the specific accepted-artifact pin.
    development = REAL_DEVELOPMENT[:350]
    path = write_screening_basin_ids_file(tmp_path / "screening.txt", development)
    result = load_screening_basin_ids(
        path, development_basins=REAL_DEVELOPMENT,
        expected_count=350, expected_sha256=sha256_of(path),
    )
    assert result == development


def test_load_screening_basin_ids_rejects_one_substituted_basin_same_count(tmp_path):
    real_ids = load_screening_basin_ids(
        REAL_SCREENING_BASIN_IDS_PATH,
        development_basins=REAL_DEVELOPMENT,
        expected_count=REAL_SCREENING_EXPECTED_COUNT,
        expected_sha256=REAL_SCREENING_EXPECTED_SHA256,
    )
    substitute = next(b for b in REAL_DEVELOPMENT if b not in real_ids)
    tampered = list(real_ids[:-1]) + [substitute]
    assert len(tampered) == REAL_SCREENING_EXPECTED_COUNT
    path = write_screening_basin_ids_file(tmp_path / "tampered.txt", tampered)
    with pytest.raises(PilotConfigError):
        load_screening_basin_ids(
            path, development_basins=REAL_DEVELOPMENT,
            expected_count=REAL_SCREENING_EXPECTED_COUNT,
            expected_sha256=REAL_SCREENING_EXPECTED_SHA256,
        )


def test_load_screening_basin_ids_rejects_399_basins(tmp_path):
    real_ids = load_screening_basin_ids(
        REAL_SCREENING_BASIN_IDS_PATH,
        development_basins=REAL_DEVELOPMENT,
        expected_count=REAL_SCREENING_EXPECTED_COUNT,
        expected_sha256=REAL_SCREENING_EXPECTED_SHA256,
    )
    path = write_screening_basin_ids_file(tmp_path / "short.txt", real_ids[:-1])
    with pytest.raises(PilotConfigError):
        load_screening_basin_ids(
            path, development_basins=REAL_DEVELOPMENT,
            expected_count=REAL_SCREENING_EXPECTED_COUNT,
            expected_sha256=REAL_SCREENING_EXPECTED_SHA256,
        )


def test_load_screening_basin_ids_rejects_401_basins(tmp_path):
    real_ids = load_screening_basin_ids(
        REAL_SCREENING_BASIN_IDS_PATH,
        development_basins=REAL_DEVELOPMENT,
        expected_count=REAL_SCREENING_EXPECTED_COUNT,
        expected_sha256=REAL_SCREENING_EXPECTED_SHA256,
    )
    extra = next(b for b in REAL_DEVELOPMENT if b not in real_ids)
    path = write_screening_basin_ids_file(tmp_path / "long.txt", list(real_ids) + [extra])
    with pytest.raises(PilotConfigError):
        load_screening_basin_ids(
            path, development_basins=REAL_DEVELOPMENT,
            expected_count=REAL_SCREENING_EXPECTED_COUNT,
            expected_sha256=REAL_SCREENING_EXPECTED_SHA256,
        )


def test_load_screening_basin_ids_rejects_basin_outside_development(tmp_path):
    ids = REAL_DEVELOPMENT[:350] + ["99999999"]
    path = write_screening_basin_ids_file(tmp_path / "screening.txt", ids)
    with pytest.raises(PilotConfigError):
        load_screening_basin_ids(
            path, development_basins=REAL_DEVELOPMENT,
            expected_count=len(ids), expected_sha256=sha256_of(path),
        )


def test_load_screening_basin_ids_rejects_spatial_holdout_membership(tmp_path):
    from tests._pilot_support import REAL_SPATIAL_HOLDOUT
    ids = REAL_DEVELOPMENT[:350] + [REAL_SPATIAL_HOLDOUT[0]]
    path = write_screening_basin_ids_file(tmp_path / "screening.txt", ids)
    with pytest.raises(PilotConfigError):
        load_screening_basin_ids(
            path, development_basins=REAL_DEVELOPMENT,
            expected_count=len(ids), expected_sha256=sha256_of(path),
        )


def test_load_screening_basin_ids_rejects_full_population_equality():
    path_ids = REAL_DEVELOPMENT
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        path = write_screening_basin_ids_file(Path(tmp) / "screening.txt", path_ids)
        with pytest.raises(PilotConfigError):
            load_screening_basin_ids(
                path, development_basins=REAL_DEVELOPMENT,
                expected_count=len(path_ids), expected_sha256=sha256_of(path),
            )


# --- build_pilot_bundle: sealed-population rejection + full bundle build ----

def test_build_pilot_bundle_rejects_screening_ids_outside_development(tmp_path, monkeypatch):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)

    bad_screening_path = tmp_path / "bad_screening.txt"
    # includes a spatial-holdout basin, which must never be reachable as a
    # pilot validation target (sealed-data-access risk).
    from tests._pilot_support import REAL_SPATIAL_HOLDOUT
    write_screening_basin_ids_file(
        bad_screening_path, REAL_DEVELOPMENT[:350] + [REAL_SPATIAL_HOLDOUT[0]],
    )

    policy = load_pilot_policy(PILOT_POLICY_PATH)
    policy = _with_screening_path(policy, bad_screening_path, expected_count=len(REAL_DEVELOPMENT[:350]) + 1)

    with pytest.raises(PilotConfigError):
        build_pilot_bundle(
            pilot_policy=policy,
            run_id="raw_seedA",
            baseline_policy_path=BASELINE_POLICY_PATH,
            package_root=package_root,
            splits_dir=SPLITS_DIR,
        )


def _with_screening_path(policy, path, *, expected_count=None, expected_sha256=None):
    resolved_count = len(path.read_text(encoding="utf-8").split()) if expected_count is None else expected_count
    resolved_sha256 = sha256_of(path) if expected_sha256 is None else expected_sha256
    return dataclasses.replace(
        policy,
        screening_basin_ids_path=str(path),
        screening_expected_count=resolved_count,
        screening_expected_sha256=resolved_sha256,
    )


def test_build_pilot_bundle_train_is_full_development_validation_is_screening(tmp_path):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)

    screening = REAL_DEVELOPMENT[:350]
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", screening)
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    policy = _with_screening_path(policy, screening_path)

    bundle = build_pilot_bundle(
        pilot_policy=policy,
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
    )
    assert sorted(bundle.train_basin_ids) == sorted(REAL_DEVELOPMENT)
    assert sorted(bundle.validation_basin_ids) == sorted(screening)
    assert sorted(bundle.test_basin_ids) == sorted(REAL_DEVELOPMENT)
    assert bundle.population_role == SCREENING_VALIDATION_POPULATION_ROLE
    assert bundle.run_profile_name == PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME["raw_seedA"]
    # validation is a proper subset of train, never equal to it
    assert set(bundle.validation_basin_ids) < set(bundle.train_basin_ids)


def test_build_all_pilot_bundles_builds_exactly_six(tmp_path):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)

    screening = REAL_DEVELOPMENT[:350]
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", screening)
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    policy = _with_screening_path(policy, screening_path)

    bundles = build_all_pilot_bundles(
        pilot_policy=policy,
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
    )
    assert set(bundles) == EXPECTED_RUN_IDS
    for run_id, bundle in bundles.items():
        assert bundle.run_profile_name == PILOT_LEAD06_RUN_ID_TO_PROFILE_NAME[run_id]
        assert sorted(bundle.validation_basin_ids) == sorted(screening)


# ---------------------------------------------------------------------------
# max_updates_per_epoch: optional per-epoch NH training-batch cap for cheap
# early-fidelity screening (efficiency feature; uncapped/None remains the
# default for every declared run unless a policy entry explicitly opts in).
# ---------------------------------------------------------------------------

def test_pilot_run_spec_max_updates_per_epoch_defaults_to_none_for_every_real_run():
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    for run_id in EXPECTED_RUN_IDS:
        assert resolve_pilot_run_spec(policy, run_id).max_updates_per_epoch is None


def test_load_pilot_policy_parses_valid_max_updates_per_epoch_for_one_run(tmp_path):
    raw = _raw_policy_dict()
    entry = next(r for r in raw["runs"] if r["run_id"] == "raw_seedA")
    entry["max_updates_per_epoch"] = 10
    path = _write_policy(tmp_path, raw)

    policy = load_pilot_policy(path)
    assert resolve_pilot_run_spec(policy, "raw_seedA").max_updates_per_epoch == 10
    # no cross-adoption onto sibling runs left undeclared in the YAML
    assert resolve_pilot_run_spec(policy, "raw_seedB").max_updates_per_epoch is None
    assert resolve_pilot_run_spec(policy, "emb128x64_seedA").max_updates_per_epoch is None


@pytest.mark.parametrize("bad_value", [True, False, 0, -1, -100, 1.5, "5"])
def test_load_pilot_policy_rejects_invalid_max_updates_per_epoch(tmp_path, bad_value):
    raw = _raw_policy_dict()
    entry = next(r for r in raw["runs"] if r["run_id"] == "raw_seedA")
    entry["max_updates_per_epoch"] = bad_value
    path = _write_policy(tmp_path, raw)
    with pytest.raises(nh_config_generation.NHConfigGenerationError):
        load_pilot_policy(path)


def test_build_pilot_bundle_uncapped_default_omits_key_from_mapping(tmp_path):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)

    screening = REAL_DEVELOPMENT[:350]
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", screening)
    policy = load_pilot_policy(PILOT_POLICY_PATH)
    policy = _with_screening_path(policy, screening_path)

    bundle = build_pilot_bundle(
        pilot_policy=policy,
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
    )
    assert bundle.max_updates_per_epoch is None
    assert "max_updates_per_epoch" not in bundle.config_mapping


def test_build_pilot_bundle_threads_declared_cap_without_cross_adoption(tmp_path):
    package_root = tmp_path / "package"
    build_full_union_package(package_root)

    screening = REAL_DEVELOPMENT[:350]
    screening_path = write_screening_basin_ids_file(tmp_path / "screening.txt", screening)

    raw = _raw_policy_dict()
    entry = next(r for r in raw["runs"] if r["run_id"] == "raw_seedA")
    entry["max_updates_per_epoch"] = 7
    policy_path = _write_policy(tmp_path, raw)
    policy = load_pilot_policy(policy_path)
    policy = _with_screening_path(policy, screening_path)

    capped_bundle = build_pilot_bundle(
        pilot_policy=policy,
        run_id="raw_seedA",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
    )
    assert capped_bundle.max_updates_per_epoch == 7
    assert capped_bundle.config_mapping["max_updates_per_epoch"] == 7

    uncapped_bundle = build_pilot_bundle(
        pilot_policy=policy,
        run_id="raw_seedB",
        baseline_policy_path=BASELINE_POLICY_PATH,
        package_root=package_root,
        splits_dir=SPLITS_DIR,
    )
    assert uncapped_bundle.max_updates_per_epoch is None
    assert "max_updates_per_epoch" not in uncapped_bundle.config_mapping
