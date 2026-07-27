#!/usr/bin/env python
"""Stage 1 validation and optimization foundation, Part A: seed percentile
diagnostic closure.

Fills the gap identified against the existing seed checkpoint-comparison
evidence (``reports/seed_validation_review_v001/aggregate/
seed_ckpt_comparison_report_epochs1to11.json``), which reports only
median/mean/q25/q75 per epoch. This script reuses
``src.baseline.nh_seed_evaluation.raw_space_metrics_for_run_period`` (the
existing, certified per-basin computation, unmodified) to obtain per-basin
raw-space NSE for each requested (epoch, run_dir) pair -- reading only the
already-computed ``validation_results.p`` pickle, never rerunning NH
inference -- then computes, via ``src.baseline.percentile_diagnostics``:

  - the full percentile grid (p1/p5/p10/p25/p50/p75/p90/p95/p99) + sign
    fractions + finite-basin count + min/max, per epoch;
  - epoch-over-epoch percentile deltas + monotonic-direction classification
    (numeric only -- no hydrological interpretation);
  - the optional basin-consistency diagnostic (rank stability / quartile
    persistence between consecutive epochs).

Writes three artifacts under ``--out-dir``: ``percentile_table.csv``,
``percentile_diagnostics.json`` (full structured report, including the
change table and consistency diagnostic), and
``percentile_diagnostics.md`` (human-readable summary).

--period is hardcoded to "validation" (no flag exposed) -- never reads
temporal-test or spatial-holdout data. Diagnostic-only: does not select or
freeze a checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, raw_space_metrics_for_run_period
from src.baseline.percentile_diagnostics import (
    PERCENTILES,
    basin_consistency_diagnostic,
    build_epoch_percentile_tables,
    percentile_change_table,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--epoch-run-dir",
        action="append",
        required=True,
        help="EPOCH=RUN_DIR, repeatable, e.g. --epoch-run-dir 1=/path/to/base_run_dir",
    )
    p.add_argument("--package-root", required=True)
    p.add_argument("--target-variable", default="qobs_mm_per_h_lead06")
    p.add_argument("--lead-hours", type=int, default=6)
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def _per_epoch_basin_nse(*, run_dir, epoch, package_root, target_variable, lead_hours):
    """Return (nse_array, {basin_id: nse}) for one epoch, reusing the
    existing certified per-basin computation unmodified."""
    result = raw_space_metrics_for_run_period(
        run_dir=run_dir,
        period="validation",
        epoch=epoch,
        package_root=package_root,
        target_variable=target_variable,
        lead_hours=lead_hours,
    )
    per_basin = result["per_basin"]
    nse_by_basin = {row["basin_id"]: row["nse"] for row in per_basin}
    nse_array = np.array([row["nse"] for row in per_basin], dtype=np.float64)
    return nse_array, nse_by_basin, result["n_basins_evaluated"], result["n_basins_area_excluded"]


def _write_csv(out_path: Path, epoch_tables: dict) -> None:
    columns = ["epoch", "n_total_basins", "n_finite_basins"] + [f"p{p}" for p in PERCENTILES] + [
        "min", "max", "frac_gt_0", "frac_gt_0p5", "frac_lt_0"
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for epoch in sorted(epoch_tables.keys()):
            row = {"epoch": epoch, **epoch_tables[epoch]}
            writer.writerow({k: row.get(k, "") for k in columns})


def _render_markdown(epoch_tables: dict, change_rows: list, consistency_rows: list) -> str:
    lines = [
        "# Stage 1 seed percentile diagnostic closure (Part A)",
        "",
        "Diagnostic-only. Raw-space (m^3/s) per-basin NSE, development "
        "validation period. Does not select or freeze a checkpoint.",
        "",
        "## Per-epoch percentile table",
        "",
        "| epoch | n_finite | " + " | ".join(f"p{p}" for p in PERCENTILES) + " | min | max | frac<0 | frac>0 | frac>0.5 |",
        "|---" * (4 + len(PERCENTILES)) + "|",
    ]
    for epoch in sorted(epoch_tables.keys()):
        t = epoch_tables[epoch]
        cells = [str(epoch), str(t["n_finite_basins"])]
        cells += [f"{t[f'p{p}']:.4f}" for p in PERCENTILES]
        cells += [f"{t['min']:.4f}", f"{t['max']:.4f}", f"{t['frac_lt_0']:.4f}", f"{t['frac_gt_0']:.4f}", f"{t['frac_gt_0p5']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Percentile-change table (epoch-over-epoch, numeric only)", ""]
    lines += ["| percentile | epochs | values | monotonic |", "|---|---|---|---|"]
    for row in change_rows:
        values_str = ", ".join(f"{v:.4f}" if np.isfinite(v) else "nan" for v in row["values"])
        lines.append(f"| {row['percentile']} | {row['epochs']} | {values_str} | {row['monotonic']} |")

    lines += ["", "## Basin-consistency diagnostic (optional)", ""]
    lines += ["| epoch_from | epoch_to | n_common_finite_basins | frac_same_quartile | spearman_r |", "|---|---|---|---|---|"]
    for row in consistency_rows:
        fsq = f"{row['frac_same_quartile']:.4f}" if np.isfinite(row["frac_same_quartile"]) else "nan"
        sr = f"{row['spearman_r']:.4f}" if np.isfinite(row["spearman_r"]) else "nan"
        lines.append(f"| {row['epoch_from']} | {row['epoch_to']} | {row['n_common_finite_basins']} | {fsq} | {sr} |")

    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    epoch_run_dirs = []
    for item in args.epoch_run_dir:
        epoch_str, run_dir = item.split("=", 1)
        epoch_run_dirs.append((int(epoch_str), run_dir))
    epoch_run_dirs.sort(key=lambda x: x[0])

    per_epoch_nse_arrays = {}
    per_epoch_nse_by_basin = {}
    per_epoch_counts = {}
    for epoch, run_dir in epoch_run_dirs:
        try:
            nse_array, nse_by_basin, n_evaluated, n_area_excluded = _per_epoch_basin_nse(
                run_dir=run_dir,
                epoch=epoch,
                package_root=args.package_root,
                target_variable=args.target_variable,
                lead_hours=args.lead_hours,
            )
        except NHSeedEvaluationError as exc:
            print(f"WARNING: epoch {epoch} skipped: {exc}", file=sys.stderr)
            continue
        per_epoch_nse_arrays[epoch] = nse_array
        per_epoch_nse_by_basin[epoch] = nse_by_basin
        per_epoch_counts[epoch] = {"n_basins_evaluated": n_evaluated, "n_basins_area_excluded": n_area_excluded}

    epoch_tables = build_epoch_percentile_tables(per_epoch_nse_arrays, metric_name="nse")
    change_rows = percentile_change_table(epoch_tables)
    consistency_rows = basin_consistency_diagnostic(per_epoch_nse_by_basin)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "percentile_table.csv", epoch_tables)

    artifact = {
        "schema_name": "stage1_seed_percentile_diagnostic_closure",
        "schema_version": 1,
        "note": (
            "Diagnostic-only Part A percentile closure (Stage 1 validation "
            "and optimization foundation). Does NOT freeze a checkpoint "
            "selection and does NOT read temporal-test or spatial-holdout data."
        ),
        "package_root": args.package_root,
        "target_variable": args.target_variable,
        "lead_hours": args.lead_hours,
        "percentiles_grid": list(PERCENTILES),
        "per_epoch_counts": per_epoch_counts,
        "per_epoch_percentile_table": epoch_tables,
        "percentile_change_table": change_rows,
        "basin_consistency_diagnostic": consistency_rows,
    }
    with open(out_dir / "percentile_diagnostics.json", "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)

    with open(out_dir / "percentile_diagnostics.md", "w", encoding="utf-8") as fh:
        fh.write(_render_markdown(epoch_tables, change_rows, consistency_rows))

    print(f"wrote percentile diagnostics for {len(epoch_tables)} epoch(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
