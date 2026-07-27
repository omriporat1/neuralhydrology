#!/usr/bin/env python
"""Section 8.3: compare the screening subset against the full development
population across the seed run's per-epoch raw-space validation NSE.

Reads only already-computed per-basin metric CSVs (one per epoch, produced
by scripts/dump_per_basin_table.py) plus the Part D screening-subset basin
list -- never touches temporal-test/spatial-holdout data, never reruns NH
inference or training. For each epoch directory found under
--per-basin-dir, computes subset-vs-full: median NSE, p25, p75, and
negative-NSE fraction. Across the epochs that are present, also reports
Spearman and Kendall rank correlation of per-epoch median NSE (subset
ranking of epochs vs. full-population ranking of epochs), top-epoch
agreement, and top-3 overlap.

If fewer than 11 epochs are present, the script still runs and clearly
marks the report as PARTIAL (n_epochs_found < 11) rather than failing --
by design, so it can be re-run as more epochs' per-basin CSVs become
available without needing code changes.

Usage:
    python scripts/analyze_stage1_screening_subset_validation_comparison.py \\
        --per-basin-dir reports/seed_validation_review_v001/per_basin \\
        --extra-per-basin-dir tmp/stage1_screening_subset_per_basin_dump_v001 \\
        --screening-subset-ids reports/stage1_validation_optimization_foundation_v001/part_d_screening_subset/selection_v001/screening_subset_basin_ids.txt \\
        --out-json reports/stage1_validation_optimization_foundation_v001/part_d_screening_subset/validation_comparison/validation_comparison_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_EPOCH_DIR_RE = re.compile(r"^epoch(\d{3})$")


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-basin-dir", required=True, action="append",
                   help="Directory containing epochNNN/epochNNN_per_basin_metrics.csv subdirs "
                        "(repeatable -- pass once per source directory)")
    p.add_argument("--screening-subset-ids", required=True,
                   help="Path to screening_subset_basin_ids.txt (one basin id per line)")
    p.add_argument("--out-json", required=True)
    return p.parse_args(argv)


def _discover_epoch_csvs(per_basin_dirs: list[Path]) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for base_dir in per_basin_dirs:
        if not base_dir.is_dir():
            continue
        for child in sorted(base_dir.iterdir()):
            m = _EPOCH_DIR_RE.match(child.name)
            if not m:
                continue
            epoch = int(m.group(1))
            csv_path = child / f"{child.name}_per_basin_metrics.csv"
            if csv_path.is_file():
                found[epoch] = csv_path
    return found


def _stats_for(df: pd.DataFrame) -> dict:
    nse = df["nse"].to_numpy(dtype=float)
    finite = nse[np.isfinite(nse)]
    return {
        "n_basins": int(len(finite)),
        "median_nse": float(np.median(finite)) if len(finite) else None,
        "p25_nse": float(np.percentile(finite, 25)) if len(finite) else None,
        "p75_nse": float(np.percentile(finite, 75)) if len(finite) else None,
        "frac_nse_lt_0": float(np.mean(finite < 0)) if len(finite) else None,
    }


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    return float(rx.corr(ry, method="pearson"))


def _kendall(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    n = len(x)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            sign = dx * dy
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    if total_pairs == 0:
        return None
    return (concordant - discordant) / total_pairs


def main(argv=None) -> int:
    args = parse_args(argv)

    per_basin_dirs = [Path(d) for d in args.per_basin_dir]
    epoch_csvs = _discover_epoch_csvs(per_basin_dirs)
    if not epoch_csvs:
        _fail(f"no epochNNN_per_basin_metrics.csv files found under {per_basin_dirs}")

    subset_ids_path = Path(args.screening_subset_ids)
    if not subset_ids_path.is_file():
        _fail(f"missing screening-subset id list: {subset_ids_path}")
    subset_ids = {
        line.strip() for line in subset_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()
    }

    per_epoch = {}
    for epoch in sorted(epoch_csvs):
        df = pd.read_csv(epoch_csvs[epoch], dtype={"basin_id": str})
        df["basin_id"] = df["basin_id"].str.strip()
        full_stats = _stats_for(df)
        subset_df = df[df["basin_id"].isin(subset_ids)]
        subset_stats = _stats_for(subset_df)
        per_epoch[epoch] = {
            "csv_path": str(epoch_csvs[epoch]),
            "full": full_stats,
            "subset": subset_stats,
            "n_subset_ids_matched": int(len(subset_df)),
            "n_subset_ids_total": int(len(subset_ids)),
            "abs_median_nse_diff": (
                abs(full_stats["median_nse"] - subset_stats["median_nse"])
                if full_stats["median_nse"] is not None and subset_stats["median_nse"] is not None
                else None
            ),
        }

    epochs_sorted = sorted(per_epoch)
    full_medians = [per_epoch[e]["full"]["median_nse"] for e in epochs_sorted]
    subset_medians = [per_epoch[e]["subset"]["median_nse"] for e in epochs_sorted]

    full_ranked = sorted(epochs_sorted, key=lambda e: per_epoch[e]["full"]["median_nse"], reverse=True)
    subset_ranked = sorted(epochs_sorted, key=lambda e: per_epoch[e]["subset"]["median_nse"], reverse=True)

    top_epoch_agreement = bool(full_ranked and subset_ranked and full_ranked[0] == subset_ranked[0])
    top3_full = set(full_ranked[:3])
    top3_subset = set(subset_ranked[:3])
    top3_overlap = len(top3_full & top3_subset)

    report = {
        "created_by": "scripts/analyze_stage1_screening_subset_validation_comparison.py",
        "status": "PARTIAL" if len(epochs_sorted) < 11 else "COMPLETE",
        "n_epochs_found": len(epochs_sorted),
        "epochs_found": epochs_sorted,
        "epochs_missing": sorted(set(range(1, 12)) - set(epochs_sorted)),
        "per_epoch": per_epoch,
        "spearman_rank_correlation_epoch_medians": _spearman(full_medians, subset_medians),
        "kendall_rank_correlation_epoch_medians": _kendall(full_medians, subset_medians),
        "top_epoch_full": full_ranked[0] if full_ranked else None,
        "top_epoch_subset": subset_ranked[0] if subset_ranked else None,
        "top_epoch_agreement": top_epoch_agreement,
        "top3_epochs_full": full_ranked[:3],
        "top3_epochs_subset": subset_ranked[:3],
        "top3_overlap_count": top3_overlap,
        "max_abs_median_nse_diff_across_epochs": (
            max(v["abs_median_nse_diff"] for v in per_epoch.values() if v["abs_median_nse_diff"] is not None)
            if per_epoch else None
        ),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_epoch"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
