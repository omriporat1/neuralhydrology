#!/usr/bin/env python
"""Seed-sensitivity check for the Stage 1 screening-subset selection (Part D).

Section 8.3 asks for "a small sensitivity check over 3-5 alternative seeds
total -- do not search dozens of seeds to find an artificially favorable
subset." This script builds the screening-subset selection once per seed
(the policy's default seed plus a small fixed list of alternates), then
reports, for each alternate seed vs. the default: selected-set overlap
(Jaccard and raw intersection size), and the max marginal balance
statistic. It does NOT search over seeds for a "best" result and does NOT
choose a seed based on this output -- seed 42 (the policy default) remains
the adopted screening-subset selection regardless of these results.

This reads only already-generated inputs (split assignment, one epoch's
per-basin NSE table, the RBI table) and writes one small JSON report --
it does not build NH packages, train models, or touch sealed test data.

Usage:
    python scripts/analyze_stage1_screening_subset_seed_sensitivity.py \\
        --split-assignment config/stage1_baseline_splits_v001/split_assignment.csv \\
        --per-basin-nse reports/seed_validation_review_v001/per_basin/epoch009/epoch009_per_basin_metrics.csv \\
        --flow-variability reports/flashnh_usgs_rbi_screening_wy2024_v001/usgs_rbi_screening_results.csv \\
        --policy config/stage1_screening_subset_v001.yaml \\
        --alternate-seeds 43 44 45 46 \\
        --out-json reports/stage1_validation_optimization_foundation_v001/part_d_screening_subset/seed_sensitivity/seed_sensitivity_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.compact_selection import load_split_assignment, select_universe
from src.baseline.screening_subset_selection import (
    SelectionError,
    build_screening_subset_selection,
    load_flow_variability,
    load_per_basin_nse,
    load_screening_subset_policy,
    sha256_of,
)

_REQUIRED_ASSIGNMENT_COLUMNS = ["STAID", "split_role", "STATE", "HUC02", "area_class", "hydro_class"]


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-assignment", required=True)
    p.add_argument("--per-basin-nse", required=True)
    p.add_argument("--flow-variability", required=True)
    p.add_argument("--policy", default=str(REPO_ROOT / "config" / "stage1_screening_subset_v001.yaml"))
    p.add_argument("--alternate-seeds", type=int, nargs="+", default=[43, 44, 45, 46],
                   help="Alternate seeds to compare against the policy's default seed (3-5 total per section 8.3)")
    p.add_argument("--out-json", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not (3 <= len(args.alternate_seeds) <= 5):
        _fail(f"--alternate-seeds must supply 3-5 values per section 8.3, got {len(args.alternate_seeds)}")

    try:
        base_policy = load_screening_subset_policy(args.policy)
        assignment_df = load_split_assignment(args.split_assignment, _REQUIRED_ASSIGNMENT_COLUMNS)
        universe = select_universe(assignment_df, base_policy)
        nse_series = load_per_basin_nse(args.per_basin_nse)
        flow_var_series = load_flow_variability(
            args.flow_variability,
            staid_column=base_policy["flow_variability"]["source_staid_column"],
            value_column=base_policy["flow_variability"]["source_value_column"],
        )
    except SelectionError as exc:
        _fail(str(exc))

    default_seed = base_policy["seed"]
    all_seeds = [default_seed] + list(args.alternate_seeds)

    selections = {}
    balance = {}
    for seed in all_seeds:
        policy = {**base_policy, "seed": seed}
        sel_df, manifest_pieces = build_screening_subset_selection(universe, nse_series, flow_var_series, policy)
        selections[seed] = set(sel_df["gauge_id"])
        balance[seed] = manifest_pieces["max_abs_marginal_frac_diff"]

    default_set = selections[default_seed]
    comparisons = []
    for seed in args.alternate_seeds:
        alt_set = selections[seed]
        intersection = default_set & alt_set
        union = default_set | alt_set
        comparisons.append({
            "alternate_seed": seed,
            "n_selected": len(alt_set),
            "n_intersection_with_default_seed": len(intersection),
            "jaccard_vs_default_seed": len(intersection) / len(union) if union else None,
            "max_abs_marginal_frac_diff": balance[seed],
        })

    report = {
        "created_by": "scripts/analyze_stage1_screening_subset_seed_sensitivity.py",
        "purpose": (
            "Section 8.3 seed-sensitivity check: 3-5 alternate seeds compared against the "
            "policy default seed. This does NOT search for a favorable seed -- the default "
            "seed remains adopted regardless of this report's contents."
        ),
        "default_seed": default_seed,
        "alternate_seeds": args.alternate_seeds,
        "n_selected_default_seed": len(default_set),
        "max_abs_marginal_frac_diff_default_seed": balance[default_seed],
        "comparisons": comparisons,
        "input_provenance": {
            "policy_path": str(Path(args.policy)),
            "policy_sha256": sha256_of(args.policy),
            "split_assignment_path": str(Path(args.split_assignment)),
            "split_assignment_sha256": sha256_of(args.split_assignment),
            "per_basin_nse_path": str(Path(args.per_basin_nse)),
            "per_basin_nse_sha256": sha256_of(args.per_basin_nse),
            "flow_variability_path": str(Path(args.flow_variability)),
            "flow_variability_sha256": sha256_of(args.flow_variability),
        },
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
