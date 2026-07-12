#!/usr/bin/env python3
"""Stage 1 window/sample feasibility estimator -- geometry + optional gap exclusion.

Phase A scaffold for Milestone 2K-G-G (Target Scaling + Gap Policy +
Lead-Time Feasibility Report, see
`docs/stage1_target_scaling_gap_leadtime_feasibility.md`).

Computes, WITHOUT importing or requiring NeuralHydrology, how many training
windows are geometrically possible for each `seq_length` x `lead_time`
combination over a given period, and -- if a gap inventory CSV is supplied
-- how many of those windows would additionally be excluded under an
MRMS-gap / RTMA-gap / either-gap window-exclusion policy. Optionally layers
on a target (`qobs`) availability CSV as well.

No NeuralHydrology import. No training. No NH package generation.

ASSUMED WINDOWING CONVENTION (geometry-only, NOT verified against the
installed NeuralHydrology 1.13 code -- confirming or correcting this is
exactly what the Moriah inspection
(`scripts/inspect_neuralhydrology_stage1_mechanics.py`) and Phase B analysis
are for):
  A window starting at absolute hourly index i uses `seq_length` (L) input
  steps [i, i+L-1] and predicts a target at absolute index i+L-1+lead_time
  (lead_time=1 -> the hour immediately following the input window's last
  step; lead_time=H -> H hours after the last input step). A window is
  geometrically valid iff both the input span and the target index fall
  inside [0, total_hours-1].

Every number this script produces is labeled with an explicit
"*_estimate_type" field so results are never mistaken for verified NH
behavior.

Usage (geometry-only):
  python scripts/analyze_stage1_window_feasibility.py \\
      --period-start 2020-10-14 --period-end 2025-12-31 \\
      --seq-lengths 12,24,48,72 --lead-times 1,3,6,12 \\
      --out-dir tmp/stage1_window_feasibility_<TS>

Usage (with a gap inventory CSV):
  python scripts/analyze_stage1_window_feasibility.py \\
      --period-start 2020-10-14 --period-end 2025-12-31 \\
      --gap-inventory-csv /path/to/gap_inventory.csv \\
      --out-dir tmp/stage1_window_feasibility_<TS>

Gap inventory CSV -- two schemas are supported (auto-detected from columns):

  1. timestamp-rows schema (long format, one row per known gap hour):
       timestamp  (required; parseable by pandas.to_datetime)
       product    (optional; rows without this column, or with an
                  unrecognized value, only count toward the "either" pool)
       basin      (optional; if present, gap exclusion is computed per basin
                  and aggregated as mean/min/max across basins; if absent,
                  the gap pattern is treated as uniform across all basins --
                  appropriate for archive-level "not_in_s3" gaps, which are
                  the same absolute hours for every basin)

  2. gap-run-intervals schema (Flash-NH forcing gap-run audit tables, e.g.
     `fullperiod_gap_inventory.csv`), one row per contiguous gap run:
       gap_start_utc  (required; parseable by pandas.to_datetime)
       gap_end_utc    (required; parseable by pandas.to_datetime)
       product        (optional; see mapping rule below)
       basin          (optional; same semantics as schema 1)
     Other columns such as `gap_length_hours`, `gap_type`, `reason`,
     `product_synchronized` are accepted and ignored. Each gap run is
     expanded inclusively into one row per hourly timestamp (both the
     start hour and end hour, floored to the hour, are included) before
     the same counting logic as schema 1 is applied.

  In both schemas, `product` values are mapped robustly rather than
  matched exactly: any value containing the substring "mrms" (case
  insensitive) maps to the "mrms" pool (e.g. `mrms_qpe_1h_pass1`); any
  value containing "rtma" maps to the "rtma" pool (e.g.
  `rtma_conus_aws_2p5km`); anything else maps to "unspecified" and only
  counts toward the "either" pool.

Target availability CSV schema (optional, long format):
  timestamp                (required)
  basin                    (optional)
  qobs_valid OR qobs_nan   (exactly one required; 1/0 flag for that hour)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--period-start", required=True, help="e.g. 2020-10-14")
    p.add_argument("--period-end", required=True,
                   help="e.g. 2025-12-31 (inclusive date; last hour included is 23:00)")
    p.add_argument("--seq-lengths", default="12,24,48,72",
                   help="Comma-separated hours (default: Stage 1 candidates 12,24,48,72)")
    p.add_argument("--lead-times", default="1,3,6,12",
                   help="Comma-separated hours (default: 1,3,6,12)")
    p.add_argument("--gap-inventory-csv", default=None,
                   help="Optional CSV of known gap hours (see module docstring for schema)")
    p.add_argument("--target-availability-csv", default=None,
                   help="Optional CSV of target (qobs) validity (see module docstring for schema)")
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def _int_list(csv: str) -> list[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _build_hourly_index(start: str, end: str) -> pd.DatetimeIndex:
    end_ts = pd.Timestamp(end) + pd.Timedelta(hours=23)
    return pd.date_range(start=start, end=end_ts, freq="h")


def _geometry_valid_windows(total_hours: int, seq_length: int, lead_time: int) -> int:
    n = total_hours - seq_length - lead_time + 1
    return max(0, n)


def _count_valid_with_mask(total_hours: int, seq_length: int, lead_time: int,
                            bad_mask: np.ndarray) -> int:
    """Count windows that are geometrically valid AND touch no True hour in
    bad_mask, across either the input span [i, i+L-1] or the target index."""
    n_start = total_hours - seq_length - lead_time + 1
    if n_start <= 0:
        return 0
    cum = np.concatenate([[0], np.cumsum(bad_mask.astype(np.int64))])
    starts = np.arange(n_start)
    input_bad = cum[starts + seq_length] - cum[starts]
    target_idx = starts + seq_length - 1 + lead_time
    target_bad = bad_mask[target_idx]
    valid = (input_bad == 0) & (~target_bad)
    return int(valid.sum())


# ---------------------------------------------------------------------------
# Optional CSV loaders
# ---------------------------------------------------------------------------


def _normalize_product_name(raw: object) -> str:
    """Map arbitrary product labels to one of "mrms"/"rtma"/"unspecified".

    Substring match, case-insensitive, so real Flash-NH product names like
    `mrms_qpe_1h_pass1` and `rtma_conus_aws_2p5km` map correctly without
    requiring the CSV to spell out the exact short labels "mrms"/"rtma".
    """
    s = str(raw).strip().lower()
    if "mrms" in s:
        return "mrms"
    if "rtma" in s:
        return "rtma"
    return "unspecified"


def _expand_gap_runs_to_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Expand gap_start_utc/gap_end_utc interval rows into one row per
    hourly timestamp, inclusive of both endpoints (each floored to the
    hour). Preserves `product` (and `basin`, if present)."""
    starts = pd.to_datetime(df["gap_start_utc"]).dt.floor("h")
    ends = pd.to_datetime(df["gap_end_utc"]).dt.floor("h")
    has_basin = "basin" in df.columns
    has_product = "product" in df.columns

    records = []
    for i in range(len(df)):
        run_index = pd.date_range(start=starts.iloc[i], end=ends.iloc[i], freq="h")
        product = df["product"].iloc[i] if has_product else "unspecified"
        basin = df["basin"].iloc[i] if has_basin else None
        for ts in run_index:
            rec = {"timestamp": ts, "product": product}
            if has_basin:
                rec["basin"] = basin
            records.append(rec)
    return pd.DataFrame.from_records(records, columns=(
        ["timestamp", "product", "basin"] if has_basin else ["timestamp", "product"]
    ))


def _load_gap_masks(
    csv_path: str, index: pd.DatetimeIndex
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Returns ({basin_key: {"mrms": bool[], "rtma": bool[], "either": bool[]}}, meta).

    basin_key is "__uniform__" if the CSV has no `basin` column. Supports
    both the timestamp-rows schema and the Flash-NH gap-run-intervals
    schema (see module docstring for both schemas and the product-name
    mapping rule).
    """
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        schema = "timestamp_rows"
        expanded = df.copy()
        expanded["timestamp"] = pd.to_datetime(expanded["timestamp"])
        if "product" not in expanded.columns:
            expanded["product"] = "unspecified"
    elif {"gap_start_utc", "gap_end_utc"}.issubset(df.columns):
        schema = "gap_run_intervals"
        expanded = _expand_gap_runs_to_timestamps(df)
    else:
        raise ValueError(
            "gap inventory CSV must have either a 'timestamp' column "
            "(timestamp-rows schema) or both 'gap_start_utc' and "
            "'gap_end_utc' columns (Flash-NH gap-run-intervals schema): "
            f"{csv_path}"
        )

    expanded["product"] = expanded["product"].map(_normalize_product_name)

    has_basin = "basin" in expanded.columns
    basins = sorted(expanded["basin"].astype(str).unique()) if has_basin else ["__uniform__"]

    n = len(index)
    pos = pd.Series(np.arange(n), index=index)

    result: dict[str, dict[str, np.ndarray]] = {}
    for basin in basins:
        sub = expanded[expanded["basin"].astype(str) == basin] if has_basin else expanded
        mrms = np.zeros(n, dtype=bool)
        rtma = np.zeros(n, dtype=bool)
        either = np.zeros(n, dtype=bool)
        matched = sub[sub["timestamp"].isin(pos.index)]
        idx = pos.loc[matched["timestamp"]].to_numpy()
        either[idx] = True
        product = matched["product"]
        mrms[idx[(product == "mrms").to_numpy()]] = True
        rtma[idx[(product == "rtma").to_numpy()]] = True
        result[basin] = {"mrms": mrms, "rtma": rtma, "either": either}

    meta = {
        "detected_gap_schema": schema,
        "basin_mode": "per_basin" if basins != ["__uniform__"] else "uniform",
        "n_basin_keys": len(result),
        "expanded_gap_rows_total": int(len(expanded)),
        "expanded_gap_hours_by_product": {
            "mrms": int((expanded["product"] == "mrms").sum()),
            "rtma": int((expanded["product"] == "rtma").sum()),
            "unspecified": int((expanded["product"] == "unspecified").sum()),
        },
    }
    return result, meta


def _load_target_missing_masks(csv_path: str, index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Returns {basin_key: bool[]} where True = target missing (NaN) at that hour."""
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"target availability CSV missing required 'timestamp' column: {csv_path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "qobs_nan" in df.columns:
        missing_col, missing_true = "qobs_nan", 1
    elif "qobs_valid" in df.columns:
        missing_col, missing_true = "qobs_valid", 0
    else:
        raise ValueError(
            f"target availability CSV must have a 'qobs_nan' or 'qobs_valid' column: {csv_path}"
        )

    has_basin = "basin" in df.columns
    basins = sorted(df["basin"].astype(str).unique()) if has_basin else ["__uniform__"]

    n = len(index)
    pos = pd.Series(np.arange(n), index=index)
    result: dict[str, np.ndarray] = {}
    for basin in basins:
        sub = df[df["basin"].astype(str) == basin] if has_basin else df
        matched = sub[sub["timestamp"].isin(pos.index)]
        idx = pos.loc[matched["timestamp"]].to_numpy()
        missing = np.zeros(n, dtype=bool)
        missing[idx[(matched[missing_col] == missing_true).to_numpy()]] = True
        result[basin] = missing
    return result


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([header, sep] + rows)


def _build_markdown(args: argparse.Namespace, total_hours: int, df: pd.DataFrame,
                     has_gap: bool, has_target: bool) -> str:
    lines = [
        "# Stage 1 window/sample feasibility (Phase A, Milestone 2K-G-G)",
        "",
        f"Period: `{args.period_start}` to `{args.period_end}` inclusive "
        f"({total_hours} hourly steps).",
        "",
        "## Estimate-type labels -- read before using any number below",
        "",
        "- `geometry_estimate_type=exact_given_assumed_convention`: the arithmetic is exact "
        "for the ASSUMED windowing convention documented in this script's module docstring. "
        "That convention (how `seq_length` and lead time combine into a window) has **not** "
        "been verified against the installed NeuralHydrology 1.13 code -- see "
        "`docs/stage1_target_scaling_gap_leadtime_feasibility.md` for the Moriah evidence "
        "that must confirm or correct it.",
    ]
    if has_gap:
        lines.append(
            "- gap-exclusion columns (`*_gap_valid_windows_*`, `*_gap_loss_fraction_*`) are "
            "exact given the supplied `--gap-inventory-csv` and the same assumed convention. "
            "They do **not** confirm whether NeuralHydrology natively supports this exclusion "
            "policy at sample time, or whether it requires a custom sampler / package-level "
            "sample mask -- that is a Phase B question."
        )
    else:
        lines.append("- No `--gap-inventory-csv` was supplied: gap-exclusion columns are not computed.")
    if has_target:
        lines.append(
            "- target-availability columns (`target_available_*`) are exact given the supplied "
            "`--target-availability-csv` and the same assumed convention."
        )
    else:
        lines.append(
            "- No `--target-availability-csv` was supplied: target-availability columns are not computed."
        )
    lines += [
        "",
        "## Results",
        "",
        _df_to_markdown_table(df),
        "",
        "Full machine-readable data: `window_feasibility.csv`, `window_feasibility_summary.json`.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_lengths = _int_list(args.seq_lengths)
    lead_times = _int_list(args.lead_times)
    index = _build_hourly_index(args.period_start, args.period_end)
    total_hours = len(index)

    print("=" * 70)
    print("Flash-NH Stage 1 — window/sample feasibility (geometry-only unless CSVs supplied)")
    print(f"Period: {args.period_start} -> {args.period_end} ({total_hours} hourly steps)")
    print(f"seq_lengths={seq_lengths}  lead_times={lead_times}")
    print("=" * 70)

    gap_masks: dict[str, dict[str, np.ndarray]] | None = None
    gap_meta: dict | None = None
    if args.gap_inventory_csv:
        gap_path = Path(args.gap_inventory_csv)
        if not gap_path.exists():
            print(f"WARNING: --gap-inventory-csv not found, skipping gap analysis: {gap_path}")
        else:
            gap_masks, gap_meta = _load_gap_masks(str(gap_path), index)
            print(f"Loaded gap inventory: schema={gap_meta['detected_gap_schema']} "
                  f"{len(gap_masks)} basin key(s) ({gap_meta['basin_mode']}); "
                  f"expanded gap hours by product={gap_meta['expanded_gap_hours_by_product']}")

    target_missing: dict[str, np.ndarray] | None = None
    if args.target_availability_csv:
        tgt_path = Path(args.target_availability_csv)
        if not tgt_path.exists():
            print(f"WARNING: --target-availability-csv not found, skipping target analysis: {tgt_path}")
        else:
            target_missing = _load_target_missing_masks(str(tgt_path), index)
            print(f"Loaded target availability: {len(target_missing)} basin key(s)")

    rows = []
    for seq_length in seq_lengths:
        for lead_time in lead_times:
            geom_valid = _geometry_valid_windows(total_hours, seq_length, lead_time)
            row = {
                "seq_length": seq_length,
                "lead_time": lead_time,
                "total_hours": total_hours,
                "geometry_valid_windows": geom_valid,
                "geometry_loss_windows": max(0, total_hours - geom_valid),
                "geometry_loss_fraction": (
                    round((total_hours - geom_valid) / total_hours, 6) if total_hours else None
                ),
                "geometry_estimate_type": "exact_given_assumed_convention",
            }

            if gap_masks is not None:
                for product in ("mrms", "rtma", "either"):
                    vals = np.array([
                        _count_valid_with_mask(total_hours, seq_length, lead_time, masks[product])
                        for masks in gap_masks.values()
                    ])
                    row[f"{product}_gap_valid_windows_mean"] = float(vals.mean())
                    row[f"{product}_gap_valid_windows_min"] = int(vals.min())
                    row[f"{product}_gap_valid_windows_max"] = int(vals.max())
                    row[f"{product}_gap_loss_fraction_mean"] = (
                        round(1 - (vals.mean() / geom_valid), 6) if geom_valid else None
                    )
                row["gap_estimate_type"] = (
                    "exact_given_gap_inventory_and_assumed_convention"
                    if list(gap_masks) == ["__uniform__"]
                    else "per_basin_aggregated_given_gap_inventory_and_assumed_convention"
                )
            else:
                row["gap_estimate_type"] = "not_computed_no_gap_inventory_supplied"

            if target_missing is not None:
                vals = np.array([
                    _count_valid_with_mask(total_hours, seq_length, lead_time, missing)
                    for missing in target_missing.values()
                ])
                row["target_available_valid_windows_mean"] = float(vals.mean())
                row["target_available_valid_windows_min"] = int(vals.min())
                row["target_available_valid_windows_max"] = int(vals.max())
                row["target_available_estimate_type"] = (
                    "exact_given_target_availability_csv_and_assumed_convention"
                )
            else:
                row["target_available_estimate_type"] = "not_computed_no_target_availability_csv_supplied"

            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "window_feasibility.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "period_start": args.period_start,
        "period_end": args.period_end,
        "total_hours": total_hours,
        "seq_lengths": seq_lengths,
        "lead_times": lead_times,
        "gap_inventory_csv": args.gap_inventory_csv,
        "gap_inventory_meta": gap_meta,
        "target_availability_csv": args.target_availability_csv,
        "assumed_windowing_convention": (
            "window at absolute index i uses input [i, i+seq_length-1] and predicts the "
            "target at i+seq_length-1+lead_time; UNVERIFIED against installed "
            "NeuralHydrology code -- geometry-only assumption pending Moriah inspection "
            "(see docs/stage1_target_scaling_gap_leadtime_feasibility.md)"
        ),
        "rows": rows,
    }
    (out_dir / "window_feasibility_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "window_feasibility_summary.md").write_text(
        _build_markdown(args, total_hours, df, gap_masks is not None, target_missing is not None)
    )

    print("=" * 70)
    print(f"Wrote window_feasibility.csv ({len(df)} rows), "
          f"window_feasibility_summary.md, window_feasibility_summary.json to {out_dir}")
    print("=" * 70)
    sys.exit(0)


if __name__ == "__main__":
    main()