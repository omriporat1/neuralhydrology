#!/usr/bin/env python
"""Generate a CANDIDATE Stage 1 spatial split (Milestone 2K-G-I I-A2).

Reads the checksum-verified static-attribute matrix and eligible-basin list,
runs the seeded random stratified split (src/baseline/splits.py), and writes
a candidate artifact bundle to --out-dir. This is a CANDIDATE ONLY: it is not
promoted into config/stage1_baseline_splits_v001/ by this script -- promotion
requires the independent auditor (I-A3), human QC review (I-A4), and explicit
user approval recorded in docs/decision_log.md (I-A5).

Usage:
    python scripts/generate_stage1_baseline_splits.py \\
        --attributes-parquet <path to stage1_static_attributes_v001.parquet> \\
        --eligible-basins <path to eligible_basins_v001.txt> \\
        --policy config/stage1_scientific_baseline_v001.yaml \\
        --out-dir tmp/stage1_baseline_splits_v001_candidate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.policy import Stage1BaselinePolicyError, load_stage1_baseline_policy
from src.baseline.splits import (
    REASON_MISSING_STRATIFIER,
    SplitGenerationError,
    build_split_assignment,
    join_eligible_with_matrix,
    load_eligible_basins,
    load_matrix_for_splits,
    sha256_of,
    write_split_artifacts,
)

_REQUIRED_MATRIX_COLUMNS = ["STATE", "HUC02", "DRAIN_SQKM", "ari_ix_uav"]


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--attributes-parquet", required=True,
                    help="Path to stage1_static_attributes_v001.parquet")
    p.add_argument("--eligible-basins", required=True,
                    help="Path to eligible_basins_v001.txt")
    p.add_argument("--policy", default=str(REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"),
                    help="Path to the Stage 1 scientific baseline policy YAML")
    p.add_argument("--out-dir", required=True, help="Candidate output directory")
    p.add_argument("--force", action="store_true",
                    help="Allow writing into a non-empty --out-dir")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        policy = load_stage1_baseline_policy(args.policy)
    except Stage1BaselinePolicyError as exc:
        _fail(f"policy load/validate failed: {exc}")

    policy_sha256 = sha256_of(args.policy)

    attributes_path = Path(args.attributes_parquet)
    eligible_path = Path(args.eligible_basins)
    if not attributes_path.is_file():
        _fail(f"attributes parquet not found: {attributes_path}")
    if not eligible_path.is_file():
        _fail(f"eligible basins file not found: {eligible_path}")

    matrix_sha256 = sha256_of(attributes_path)
    expected_matrix_sha256 = policy["static_attributes"]["sha256"]
    if matrix_sha256 != expected_matrix_sha256:
        _fail(
            "attributes parquet checksum mismatch vs policy: "
            f"got {matrix_sha256}, expected {expected_matrix_sha256}"
        )

    try:
        eligible = load_eligible_basins(eligible_path)
    except SplitGenerationError as exc:
        _fail(str(exc))
    expected_eligible_count = policy["basin_universe"]["expected_eligible_count"]
    if len(eligible) != expected_eligible_count:
        _fail(f"eligible count {len(eligible)} != policy expected {expected_eligible_count}")

    eligible_sha256 = sha256_of(eligible_path)

    try:
        matrix_df = load_matrix_for_splits(attributes_path, _REQUIRED_MATRIX_COLUMNS)
        joined = join_eligible_with_matrix(matrix_df, eligible)
    except SplitGenerationError as exc:
        _fail(str(exc))

    seed = policy["spatial_split"]["seed"]
    min_stratum_size = policy["spatial_split"]["min_composite_stratum_size"]
    nonca_holdout_fraction = policy["spatial_split"]["nonca_holdout_fraction"]
    ca_holdout_fraction = 1.0 - policy["spatial_split"]["california_finetune_fraction"]

    try:
        assignment_df, manifest_pieces = build_split_assignment(
            joined,
            seed=seed,
            nonca_holdout_fraction=nonca_holdout_fraction,
            ca_holdout_fraction=ca_holdout_fraction,
            min_stratum_size=min_stratum_size,
        )
    except SplitGenerationError as exc:
        _fail(f"split assignment failed: {exc}")

    # ---- post-hoc acceptance checks (belt-and-braces on top of build_split_assignment) ----
    role_counts = assignment_df.groupby("STAID").size()
    if (role_counts != 1).any():
        _fail(f"basin(s) with != 1 role: {role_counts[role_counts != 1].index.tolist()[:10]}")

    ca_rows = assignment_df.loc[assignment_df["STATE"].astype(str) == "CA"]
    bad_ca_roles = set(ca_rows["split_role"]) - {"california_finetune_train", "california_holdout"}
    if bad_ca_roles:
        _fail(f"California basin(s) assigned non-California role(s): {bad_ca_roles}")

    nonca_rows = assignment_df.loc[assignment_df["STATE"].astype(str) != "CA"]
    bad_nonca_roles = set(nonca_rows["split_role"]) - {"development_train", "spatial_holdout_nonca"}
    if bad_nonca_roles:
        _fail(f"non-California basin(s) assigned California role(s): {bad_nonca_roles}")

    expected_nonca_count = policy["basin_universe"]["expected_nonca_count"]
    expected_ca_count = policy["basin_universe"]["expected_ca_count"]
    if len(nonca_rows) != expected_nonca_count:
        _fail(f"non-CA basin count {len(nonca_rows)} != policy expected {expected_nonca_count}")
    if len(ca_rows) != expected_ca_count:
        _fail(f"CA basin count {len(ca_rows)} != policy expected {expected_ca_count}")

    missing_stratifier_staids = set(
        assignment_df.loc[assignment_df["assignment_reason"] == REASON_MISSING_STRATIFIER, "STAID"]
    )
    holdout_roles = {"spatial_holdout_nonca", "california_holdout"}
    bad_holdout = missing_stratifier_staids & set(
        assignment_df.loc[assignment_df["split_role"].isin(holdout_roles), "STAID"]
    )
    if bad_holdout:
        _fail(f"missing-stratifier basin(s) entered a holdout: {sorted(bad_holdout)}")

    out_dir = Path(args.out_dir)
    try:
        manifest = {
            "created_by": "scripts/generate_stage1_baseline_splits.py",
            "status": "candidate_subject_to_machine_and_human_qc",
            "policy_path": str(Path(args.policy)),
            "policy_sha256": policy_sha256,
            "attributes_parquet_path": str(attributes_path),
            "attributes_parquet_sha256": matrix_sha256,
            "eligible_basins_path": str(eligible_path),
            "eligible_basins_sha256": eligible_sha256,
            "eligible_basins_count": len(eligible),
            "temporal_lists_identical_by_design": True,
            **manifest_pieces,
        }
        paths = write_split_artifacts(out_dir, eligible, assignment_df, manifest, force=args.force)
    except SplitGenerationError as exc:
        _fail(str(exc))

    dev_sha = sha256_of(paths["development_train.txt"])
    val_sha = sha256_of(paths["validation.txt"])
    test_sha = sha256_of(paths["temporal_test.txt"])
    if not (dev_sha == val_sha == test_sha):
        _fail("development_train/validation/temporal_test are not byte-identical")

    print(json.dumps({
        "out_dir": str(out_dir),
        "counts": manifest_pieces["counts"],
        "resulting_fractions": manifest_pieces["resulting_fractions"],
        "temporal_lists_sha256": dev_sha,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
