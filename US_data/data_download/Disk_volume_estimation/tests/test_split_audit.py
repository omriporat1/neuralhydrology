"""Tests for src/baseline/split_audit.py (Milestone 2K-G-I, sub-task I-A3).

The fixtures below build a small synthetic population, run it through the
REAL generator (src.baseline.splits.build_split_assignment /
write_split_artifacts) to produce a golden candidate directory, then feed
that candidate directory to the INDEPENDENT auditor (run_audit). The
synthetic population is deliberately designed (see the module docstring in
split_audit.py for the independence rationale) to exercise every route the
auditor must recompute: a large direct stratum, two HUC02-pooled
sparse_pool_sample groups, a sparse_pool_forced_training group in HUC02 "09"
(mirroring the real v001 zero-holdout special case), and the five known
v001 missing-aridity basins. The aggregate non-CA/CA holdout fractions are
tuned to land inside the policy's [8%, 12%] band even though several small
groups round their own local holdout count to zero.

Mutation tests each apply one targeted corruption to a copy of the golden
candidate directory and assert the auditor's independent recomputation
catches it (report.status == "FAIL", with the expected check_id present).
"""
from __future__ import annotations

import copy
import csv
import json

import pandas as pd
import pytest
import yaml

from src.baseline.split_audit import (
    SplitAuditError,
    V001_EXCLUDED_TARGET_STAIDS,
    V001_KNOWN_MISSING_ARIDITY_STAIDS,
    fit_terciles,
    reconstruct_population,
    run_audit,
    tercile_label,
    write_audit_outputs,
)
from src.baseline.splits import (
    build_split_assignment,
    join_eligible_with_matrix,
    load_eligible_basins,
    load_matrix_for_splits,
    sha256_of,
    write_split_artifacts,
)

MISSING_ARIDITY_STAIDS = sorted(V001_KNOWN_MISSING_ARIDITY_STAIDS)
NONSTANDARD_9DIGIT_STAID = "103366092"

_MIN_STRATUM_SIZE = 3
_NONCA_HOLDOUT_FRACTION = 0.1
_CA_HOLDOUT_FRACTION = 0.1
_SEED = 42


def _nonca_staid(i: int) -> str:
    return f"{10000000 + i:08d}"


def _ca_staid(i: int) -> str:
    return f"{50000000 + i:08d}"


def _build_population_rows():
    """Synthetic population exercising every route the auditor must recompute:
    direct_stratum_sample (large HUC02 "02" stratum), sparse_pool_sample (two
    HUC02-pooled groups), sparse_pool_forced_training (HUC02 "09" singleton,
    mirroring the real v001 zero-holdout special case), and
    missing_hydroatlas_stratifier (the five known v001 basins).
    """
    rows = []
    i = 0

    def add(state, huc02, area, hydro, n):
        nonlocal i
        for _ in range(n):
            i += 1
            staid = _ca_staid(i) if state == "CA" else _nonca_staid(i)
            rows.append(
                {
                    "STAID": staid,
                    "STATE": state,
                    "HUC02": huc02,
                    "DRAIN_SQKM": area,
                    "ari_ix_uav": hydro,
                }
            )

    # non-CA: large direct stratum (HUC02 02)
    add("OH", "02", 1.0, 1.0, 81)
    # non-CA: two initially-sparse strata sharing HUC02 03 -> pooled (size 4) -> sampled
    add("OH", "03", 1000.0, 1.0, 2)
    add("OH", "03", 1000.0, 2_000_000.0, 2)
    # non-CA: HUC02 09 singleton -> sparse, pool of 1 -> forced training, zero holdout
    add("OH", "09", 2_000_000.0, 1000.0, 1)
    # non-CA: two initially-sparse strata sharing HUC02 04 -> pooled (size 3) -> sampled
    add("OH", "04", 2_000_000.0, 2_000_000.0, 1)
    add("OH", "04", 2_000_000.0, 1.0, 2)

    # non-CA basin with a non-8-character STAID (known v001 fact)
    rows.append(
        {
            "STAID": NONSTANDARD_9DIGIT_STAID,
            "STATE": "OH",
            "HUC02": "02",
            "DRAIN_SQKM": 1.0,
            "ari_ix_uav": 1.0,
        }
    )

    # non-CA basins missing ari_ix_uav (exactly the 5 known v001 STAIDs)
    for staid in MISSING_ARIDITY_STAIDS:
        rows.append(
            {
                "STAID": staid,
                "STATE": "OH",
                "HUC02": "06",
                "DRAIN_SQKM": 5.0,
                "ari_ix_uav": None,
            }
        )

    # CA: direct stratum
    add("CA", "18", 1.0, 1.0, 18)
    # CA: sparse statewide pool of 2 -> forced training
    add("CA", "18", 1000.0, 1000.0, 1)
    add("CA", "18", 2_000_000.0, 2_000_000.0, 1)

    return rows


def _build_inputs(tmp_path, rows=None):
    """Write matrix/eligible/policy input files and return everything needed
    to (re)generate split artifact directories against them."""
    rows = rows if rows is not None else _build_population_rows()
    df = pd.DataFrame(rows)

    matrix_path = tmp_path / "matrix.parquet"
    df.rename(columns={"STAID": "gauge_id"}).to_parquet(matrix_path, index=False)

    eligible_path = tmp_path / "eligible_basins_v001.txt"
    eligible_path.write_text("\n".join(sorted(df["STAID"])) + "\n", encoding="utf-8")

    eligible = load_eligible_basins(eligible_path)
    matrix_df = load_matrix_for_splits(matrix_path, ["STATE", "HUC02", "DRAIN_SQKM", "ari_ix_uav"])
    joined = join_eligible_with_matrix(matrix_df, eligible)

    matrix_sha256 = sha256_of(matrix_path)
    eligible_sha256 = sha256_of(eligible_path)

    nonca_n = int((joined["STATE"] != "CA").sum())
    ca_n = int((joined["STATE"] == "CA").sum())

    policy = {
        "basin_universe": {
            "expected_eligible_count": len(joined),
            "expected_nonca_count": nonca_n,
            "expected_ca_count": ca_n,
            "excluded_staids": sorted(V001_EXCLUDED_TARGET_STAIDS),
        },
        "spatial_split": {
            "seed": _SEED,
            "min_composite_stratum_size": _MIN_STRATUM_SIZE,
            "nonca_holdout_fraction": _NONCA_HOLDOUT_FRACTION,
            "california_finetune_fraction": 1.0 - _CA_HOLDOUT_FRACTION,
        },
        "static_attributes": {"sha256": matrix_sha256},
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")

    return {
        "policy": policy,
        "policy_path": policy_path,
        "matrix_path": matrix_path,
        "eligible_path": eligible_path,
        "eligible": eligible,
        "joined": joined,
        "matrix_sha256": matrix_sha256,
        "eligible_sha256": eligible_sha256,
    }


def _generate_split_dir(inputs, out_dir):
    """Run the REAL generator against the shared inputs, writing a candidate
    artifact bundle to out_dir. Calling this twice with the same `inputs`
    simulates the real repeat-generation evidence (byte-identical outputs)."""
    assignment_df, manifest_pieces = build_split_assignment(
        inputs["joined"],
        seed=_SEED,
        nonca_holdout_fraction=_NONCA_HOLDOUT_FRACTION,
        ca_holdout_fraction=_CA_HOLDOUT_FRACTION,
        min_stratum_size=_MIN_STRATUM_SIZE,
    )
    manifest = {
        "created_by": "test fixture (mirrors scripts/generate_stage1_baseline_splits.py)",
        "status": "candidate_subject_to_machine_and_human_qc",
        "policy_path": str(inputs["policy_path"]),
        "policy_sha256": sha256_of(inputs["policy_path"]),
        "attributes_parquet_path": str(inputs["matrix_path"]),
        "attributes_parquet_sha256": inputs["matrix_sha256"],
        "eligible_basins_path": str(inputs["eligible_path"]),
        "eligible_basins_sha256": inputs["eligible_sha256"],
        "eligible_basins_count": len(inputs["eligible"]),
        "temporal_lists_identical_by_design": True,
        **manifest_pieces,
    }
    write_split_artifacts(out_dir, inputs["eligible"], assignment_df, manifest, force=True)
    return out_dir


def _run(inputs, candidate_dir, repeat_dir=None, forcing_basins=None, policy=None):
    return run_audit(
        policy=policy if policy is not None else inputs["policy"],
        policy_path=inputs["policy_path"],
        candidate_dir=candidate_dir,
        repeat_dir=repeat_dir,
        attributes_parquet=inputs["matrix_path"],
        eligible_basins=inputs["eligible_path"],
        forcing_basins=forcing_basins,
    )


def _error_ids(report):
    return {r.check_id for r in report.records if r.severity == "ERROR"}


def _read_lines(path):
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rewrite_assignment_csv(path, mutate_rows_fn):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fieldnames = list(rows[0].keys())
    mutate_rows_fn(rows)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_manifest(cand):
    return json.loads((cand / "split_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(cand, manifest):
    (cand / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Low-level independent-recomputation unit tests
# ---------------------------------------------------------------------------


def test_fit_terciles_matches_expected_quantiles():
    values = pd.Series(range(1, 10), dtype=float)
    e1, e2 = fit_terciles(values)
    assert e1 == pytest.approx(3.6667, abs=1e-3)
    assert e2 == pytest.approx(6.3333, abs=1e-3)


def test_tercile_label_boundaries():
    edges = (3.0, 6.0)
    assert tercile_label(3.0, edges) == "low"
    assert tercile_label(3.0001, edges) == "middle"
    assert tercile_label(6.0, edges) == "middle"
    assert tercile_label(6.0001, edges) == "high"


def test_reconstruct_population_rejects_missing_basin():
    matrix = pd.DataFrame({"STATE": ["OH"]}, index=pd.Index(["10000001"], name="gauge_id"))
    with pytest.raises(SplitAuditError):
        reconstruct_population(matrix, ["10000001", "10000002"])


# ---------------------------------------------------------------------------
# End-to-end PASS fixture
# ---------------------------------------------------------------------------


def test_valid_candidate_passes(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")

    report, diagnostics = _run(inputs, cand)

    assert report.error_count == 0, report.failed_messages()
    assert report.status == "PASS"
    fractions = diagnostics["recomputed_fractions"]
    assert 0.08 <= fractions["nonca_holdout_of_nonca"] <= 0.12
    assert 0.08 <= fractions["ca_holdout_of_ca"] <= 0.12


def test_write_audit_outputs_creates_expected_files(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    report, diagnostics = _run(inputs, cand)

    out_dir = tmp_path / "audit_out"
    paths = write_audit_outputs(out_dir, report, diagnostics)

    assert set(paths) == {"audit_summary.json", "audit_checks.csv", "audit_summary.md"}
    for p in paths.values():
        assert p.is_file()
    summary = json.loads((out_dir / "audit_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"


def test_repeat_directory_byte_identical_passes(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    repeat = _generate_split_dir(inputs, tmp_path / "repeat")

    report, diagnostics = _run(inputs, cand, repeat_dir=repeat)

    assert report.status == "PASS", report.failed_messages()
    assert diagnostics["repeat_comparison"]["all_match"] is True


def test_forcing_basins_agreement_passes(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")

    report, _ = _run(inputs, cand, forcing_basins=inputs["eligible_path"])

    assert report.status == "PASS", report.failed_messages()


# ---------------------------------------------------------------------------
# Mutation / failure-detection tests
# ---------------------------------------------------------------------------


def test_detects_omitted_basin_from_list(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    dev_path = cand / "development_train.txt"
    lines = _read_lines(dev_path)
    lines.pop(0)
    _write_lines(dev_path, lines)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "dev_holdout_union_is_nonca" in _error_ids(report)


def test_detects_extra_basin_in_list(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    dev_path = cand / "development_train.txt"
    lines = _read_lines(dev_path)
    lines.append("99999999")
    _write_lines(dev_path, lines)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "dev_holdout_union_is_nonca" in _error_ids(report)


def test_detects_duplicate_assignment_across_role_lists(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    dev_lines = _read_lines(cand / "development_train.txt")
    holdout_path = cand / "spatial_holdout_nonca.txt"
    holdout_lines = _read_lines(holdout_path)
    holdout_lines.append(dev_lines[0])
    _write_lines(holdout_path, holdout_lines)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "dev_holdout_disjoint" in _error_ids(report)


def test_detects_california_leakage_into_development_train(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    ca_basin = _read_lines(cand / "california_all.txt")[0]
    dev_path = cand / "development_train.txt"
    lines = _read_lines(dev_path)
    lines.append(ca_basin)
    _write_lines(dev_path, lines)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "stage1_3_no_california_leakage" in _error_ids(report)


def test_detects_nonca_leakage_into_california_all(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    nonca_basin = _read_lines(cand / "development_train.txt")[0]
    ca_all_path = cand / "california_all.txt"
    lines = _read_lines(ca_all_path)
    lines.append(nonca_basin)
    _write_lines(ca_all_path, lines)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    errs = _error_ids(report)
    assert "no_nonca_leakage_into_ca_lists" in errs or "ca_all_matches_independent_ca_population" in errs


def test_detects_table_role_inconsistent_with_lists(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")

    def mutate(rows):
        for row in rows:
            if row["split_role"] == "development_train":
                row["split_role"] = "spatial_holdout_nonca"
                break

    _rewrite_assignment_csv(cand / "split_assignment.csv", mutate)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "table_role_matches_list_membership" in _error_ids(report)


def test_detects_altered_nonstandard_staid(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    dev_path = cand / "development_train.txt"
    lines = _read_lines(dev_path)
    lines = ["19999999" if ln == NONSTANDARD_9DIGIT_STAID else ln for ln in lines]
    _write_lines(dev_path, lines)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "dev_holdout_union_is_nonca" in _error_ids(report)


def test_detects_excluded_target_gauge_present(tmp_path):
    excluded_staid = sorted(V001_EXCLUDED_TARGET_STAIDS)[0]
    rows = _build_population_rows()
    rows.append(
        {"STAID": excluded_staid, "STATE": "OH", "HUC02": "02", "DRAIN_SQKM": 1.0, "ari_ix_uav": 1.0}
    )
    inputs = _build_inputs(tmp_path, rows=rows)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    errs = _error_ids(report)
    assert "excluded_target_gauges_absent" in errs
    assert "v001_excluded_target_gauges_absent" in errs


def test_detects_missing_stratifier_basin_moved_to_holdout(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    victim = MISSING_ARIDITY_STAIDS[0]

    dev_path = cand / "development_train.txt"
    holdout_path = cand / "spatial_holdout_nonca.txt"
    _write_lines(dev_path, [ln for ln in _read_lines(dev_path) if ln != victim])
    _write_lines(holdout_path, _read_lines(holdout_path) + [victim])

    def mutate(rows):
        for row in rows:
            if row["STAID"] == victim:
                row["split_role"] = "spatial_holdout_nonca"

    _rewrite_assignment_csv(cand / "split_assignment.csv", mutate)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    errs = _error_ids(report)
    assert "role_within_route_allowed_roles" in errs
    assert "sparse_and_missing_never_in_holdout" in errs


def test_detects_wrong_missing_stratifier_reason(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    victim = MISSING_ARIDITY_STAIDS[0]

    def mutate(rows):
        for row in rows:
            if row["STAID"] == victim:
                row["assignment_reason"] = "direct_stratum_sample"

    _rewrite_assignment_csv(cand / "split_assignment.csv", mutate)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "assignment_reason_matches_recomputed_route" in _error_ids(report)


def test_detects_incorrect_direct_reason_on_sparse_basin(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")

    def mutate(rows):
        for row in rows:
            if row["assignment_reason"] == "sparse_pool_sample":
                row["assignment_reason"] = "direct_stratum_sample"
                break

    _rewrite_assignment_csv(cand / "split_assignment.csv", mutate)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "assignment_reason_matches_recomputed_route" in _error_ids(report)


def test_detects_incorrect_forced_training_reason_on_direct_basin(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")

    def mutate(rows):
        for row in rows:
            if row["assignment_reason"] == "direct_stratum_sample" and row["split_role"] == "development_train":
                row["assignment_reason"] = "sparse_pool_forced_training"
                break

    _rewrite_assignment_csv(cand / "split_assignment.csv", mutate)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "assignment_reason_matches_recomputed_route" in _error_ids(report)


def test_detects_wrong_sparse_pool_membership_in_manifest(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    manifest = _read_manifest(cand)
    assert manifest["fallback_log"], "fixture must contain at least one forced-training basin"
    manifest["fallback_log"][0]["pool_size"] = 999
    _write_manifest(cand, manifest)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "manifest_fallback_log_matches_recomputed_sparse_forced" in _error_ids(report)


def test_detects_wrong_tercile_edge_in_manifest(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    manifest = _read_manifest(cand)
    manifest["tercile_edges"]["nonca_area"] = [-999.0, -998.0]
    _write_manifest(cand, manifest)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "manifest_tercile_edges_match_recomputed" in _error_ids(report)


def test_detects_holdout_fraction_outside_band(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    holdout_path = cand / "spatial_holdout_nonca.txt"
    dev_path = cand / "development_train.txt"
    holdout_lines = _read_lines(holdout_path)
    dev_lines = _read_lines(dev_path)
    _write_lines(dev_path, dev_lines + holdout_lines)
    _write_lines(holdout_path, [])

    def mutate(rows):
        moved = set(holdout_lines)
        for row in rows:
            if row["STAID"] in moved:
                row["split_role"] = "development_train"

    _rewrite_assignment_csv(cand / "split_assignment.csv", mutate)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "nonca_holdout_fraction_in_band" in _error_ids(report)


def test_detects_incorrect_huc02_role_count_in_manifest(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    manifest = _read_manifest(cand)
    manifest["huc02_role_counts"]["02"]["development_train"] += 5
    _write_manifest(cand, manifest)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "manifest_huc02_role_counts_match_recomputed" in _error_ids(report)


def test_detects_huc02_09_basin_moved_to_holdout(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")

    assignment_path = cand / "split_assignment.csv"
    with open(assignment_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    victim = next(row["STAID"] for row in rows if row["HUC02"] == "09")

    dev_path = cand / "development_train.txt"
    holdout_path = cand / "spatial_holdout_nonca.txt"
    _write_lines(dev_path, [ln for ln in _read_lines(dev_path) if ln != victim])
    _write_lines(holdout_path, _read_lines(holdout_path) + [victim])

    def mutate(rows):
        for row in rows:
            if row["STAID"] == victim:
                row["split_role"] = "spatial_holdout_nonca"

    _rewrite_assignment_csv(assignment_path, mutate)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    errs = _error_ids(report)
    assert "huc02_09_zero_holdout" in errs
    assert "role_within_route_allowed_roles" in errs


def test_detects_manifest_count_mismatch(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    manifest = _read_manifest(cand)
    manifest["counts"]["development_train"] += 1
    _write_manifest(cand, manifest)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "manifest_counts_match_recomputed" in _error_ids(report)


def test_detects_policy_checksum_mismatch_in_manifest(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    manifest = _read_manifest(cand)
    manifest["policy_sha256"] = "0" * 64
    _write_manifest(cand, manifest)

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "manifest_policy_sha256" in _error_ids(report)


def test_detects_static_matrix_checksum_mismatch_vs_policy(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    bad_policy = copy.deepcopy(inputs["policy"])
    bad_policy["static_attributes"]["sha256"] = "1" * 64

    report, _ = _run(inputs, cand, policy=bad_policy)

    assert report.status == "FAIL"
    assert "static_matrix_checksum_vs_policy" in _error_ids(report)


def test_detects_candidate_artifact_checksum_mismatch(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    with open(cand / "development_train.txt", "a", encoding="utf-8") as fh:
        fh.write("\n")

    report, _ = _run(inputs, cand)

    assert report.status == "FAIL"
    assert "candidate_artifact_checksums_match_manifest" in _error_ids(report)


def test_detects_candidate_repeat_byte_mismatch(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    repeat = _generate_split_dir(inputs, tmp_path / "repeat")
    with open(repeat / "development_train.txt", "a", encoding="utf-8") as fh:
        fh.write("00000001\n")

    report, _ = _run(inputs, cand, repeat_dir=repeat)

    assert report.status == "FAIL"
    assert "repeat_candidate_byte_identical" in _error_ids(report)


def test_detects_missing_repeat_artifact(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    repeat = _generate_split_dir(inputs, tmp_path / "repeat")
    (repeat / "california_holdout.txt").unlink()

    report, _ = _run(inputs, cand, repeat_dir=repeat)

    assert report.status == "FAIL"
    assert "repeat_artifact_inventory_complete" in _error_ids(report)


def test_detects_volatile_repeat_manifest_field_change(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    repeat = _generate_split_dir(inputs, tmp_path / "repeat")
    manifest = json.loads((repeat / "split_manifest.json").read_text(encoding="utf-8"))
    manifest["created_by"] = "something else entirely"
    (repeat / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report, _ = _run(inputs, cand, repeat_dir=repeat)

    assert report.status == "FAIL"
    assert "repeat_manifest_byte_identical" in _error_ids(report)


def test_detects_forcing_basins_mismatch(tmp_path):
    inputs = _build_inputs(tmp_path)
    cand = _generate_split_dir(inputs, tmp_path / "candidate")
    forcing_path = tmp_path / "forcing_basins_v001.txt"
    lines = _read_lines(inputs["eligible_path"])
    lines[0] = "77777777"
    _write_lines(forcing_path, lines)

    report, _ = _run(inputs, cand, forcing_basins=forcing_path)

    assert report.status == "FAIL"
    assert "eligible_forcing_agreement" in _error_ids(report)
