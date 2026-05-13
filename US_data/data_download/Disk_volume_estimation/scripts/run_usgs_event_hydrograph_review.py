#!/usr/bin/env python3
"""Event-centered hydrograph review for WY2024 USGS RBI screening results.

This script selects a stratified sample across RBI bands, (optionally) retrieves
hourly WY2024 discharge for selected basins using the existing USGS IV helper
logic, identifies top event times (largest hourly jumps and peaks), and
writes event-centered and full-year hydrograph plots for visual QC.

Hard constraints:
- This script may fetch USGS IV data only for the selected sample basins.
- Do not change RBI formula or candidate universe.
- Do not save raw USGS payloads unless --debug-raw is passed.
- Keep run bounded and resume-safe; support --max-basins smoke mode.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "reports/flashnh_usgs_rbi_screening_wy2024_v001"
DEFAULT_OUTPUT_DIR = ROOT / "reports/flashnh_usgs_event_hydrograph_review_v001"
WY_START = pd.Timestamp("2023-10-01T00:00:00Z")
WY_END = pd.Timestamp("2024-09-30T23:00:00Z")
EXPECTED_HOURLY_INDEX = pd.date_range(start=WY_START, end=WY_END, freq="1h")

# RBI bins definition
RBI_BINS = {
    "rbi_lt_0p05": (None, 0.05),
    "rbi_0p05_0p10": (0.05, 0.10),
    "rbi_0p10_0p20": (0.10, 0.20),
    "rbi_0p20_0p50": (0.20, 0.50),
    "rbi_ge_0p50": (0.50, None),
}

from scripts.usgs_rbi_screening_scale import fetch_iv_json  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run event-centered hydrograph review for RBI sample")
    p.add_argument("--max-basins", type=int, default=None, help="Limit sample across all bins (smoke mode)")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--per-bin", type=int, default=20, help="Max basins per RBI bin")
    p.add_argument("--sleep-seconds", type=float, default=0.35)
    p.add_argument("--timeout-seconds", type=int, default=90)
    p.add_argument("--debug-raw", action="store_true", help="Save raw USGS payloads (not recommended)")
    return p.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    log_path = output_dir / "logs" / "event_review.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("usgs_event_review")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def find_results_csv(results_dir: Path) -> Path:
    candidates = list(results_dir.glob("usgs_rbi_screening_results.*"))
    if not candidates:
        raise FileNotFoundError(f"No screening results found under {results_dir}")
    # prefer CSV if present
    for c in candidates:
        if c.suffix.lower() == ".csv":
            return c
    return candidates[0]


def load_screening_results(results_path: Path) -> pd.DataFrame:
    if results_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(results_path)
    else:
        df = pd.read_csv(results_path, dtype={"STAID": str})
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    return df


# Reuse the existing request/resample/unit logic from the screening workflow
from scripts.usgs_rbi_screening_scale import fetch_iv_json, flatten_iv_payload, convert_units, build_hourly_series  # type: ignore


def sample_by_bins(df: pd.DataFrame, per_bin: int, max_basins: Optional[int], logger: logging.Logger) -> pd.DataFrame:
    selected = []
    rng = np.random.default_rng(42)
    for bin_name, (lo, hi) in RBI_BINS.items():
        mask = df["rbi"].notna()
        if lo is None:
            mask &= df["rbi"] < hi
        elif hi is None:
            mask &= df["rbi"] >= lo
        else:
            mask &= (df["rbi"] >= lo) & (df["rbi"] < hi)
        bin_df = df[mask & (df["screening_status"] == "RBI_READY")].copy()
        if bin_df.empty:
            logger.info("No RBI_READY basins in bin %s", bin_name)
            continue
        # keep highest-RBI candidates
        top = bin_df.sort_values("rbi", ascending=False)
        top_n = top.head(max(5, per_bin//4))
        remaining = top.drop(top_n.index)
        # sample randomly from remaining to fill up to per_bin
        need = max(0, per_bin - len(top_n))
        if len(remaining) > 0 and need > 0:
            sampled = remaining.sample(n=min(need, len(remaining)), random_state=42)
            sel = pd.concat([top_n, sampled], ignore_index=True)
        else:
            sel = top_n
        sel = sel.head(per_bin).copy()
        sel["rbi_bin"] = bin_name
        # stratify by area_bin where possible - simple approach: ensure coverage if possible
        # (already sampled highest and random mix)
        selected.append(sel)
    final = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if max_basins is not None and len(final) > max_basins:
        final = final.sample(n=max_basins, random_state=42)
    # deduplicate
    final = final.drop_duplicates(subset=["STAID"]).reset_index(drop=True)
    return final


def retrieve_hourly_series(session: requests.Session, staid: str, logger: logging.Logger, sleep_seconds: float, timeout_seconds: int, debug_raw: bool, output_raw_dir: Path, max_retries: int = 5) -> pd.Series:
    payload, status, size, url = fetch_iv_json(
        session=session,
        site_no=staid,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        logger=logger,
    )
    frame, meta = flatten_iv_payload(payload)
    frame["dateTime"] = pd.to_datetime(frame["dateTime"], utc=True, errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["dateTime", "value"]) if not frame.empty else frame
    if debug_raw and not output_raw_dir.exists():
        output_raw_dir.mkdir(parents=True, exist_ok=True)
        (output_raw_dir / f"{staid}_payload.json").write_text(json.dumps(payload))
    if frame.empty:
        return pd.Series(dtype=float)
    raw_series = frame.set_index("dateTime")["value"].groupby(level=0).mean().sort_index()
    converted, units_out, converted_flag = convert_units(raw_series, meta.get("unit_code"))
    hourly = build_hourly_series(converted)
    return hourly


def detect_events(hourly: pd.Series) -> dict:
    hourly = hourly.dropna()
    if hourly.empty:
        return {"top_jumps": [], "top_peaks": []}
    diffs = hourly.diff().abs()
    diffs = diffs.dropna()
    top_jumps = diffs.sort_values(ascending=False).head(10)
    top_peaks = hourly.sort_values(ascending=False).head(10)
    # convert to timestamps
    jumps = list(top_jumps.index)
    peaks = list(top_peaks.index)
    # choose top 5 each
    return {"top_jumps": jumps[:5], "top_peaks": peaks[:5], "max_jump_val": float(top_jumps.iloc[0]) if not top_jumps.empty else None}


def merge_event_windows(event_times: List[pd.Timestamp], window: pd.Timedelta) -> List[pd.Timestamp]:
    if not event_times:
        return []
    sorted_times = sorted(event_times)
    groups = [sorted_times[0]]
    for t in sorted_times[1:]:
        if t - groups[-1] <= window:
            # merge by replacing last time with later one (keeps center near later event)
            groups[-1] = t
        else:
            groups.append(t)
    return groups


def make_event_plot(hourly: pd.Series, event_time: pd.Timestamp, outpath: Path, title: str, event_marker: Optional[pd.Timestamp] = None) -> None:
    window_start = event_time - pd.Timedelta(days=7)
    window_end = event_time + pd.Timedelta(days=7)
    sub = hourly.reindex(EXPECTED_HOURLY_INDEX)[window_start:window_end]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sub.index, sub.values, color="#2b8cbe", linewidth=1.2)
    if event_marker is not None:
        ax.axvline(event_marker, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Discharge (m3/s)")
    ax.set_title(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def make_full_wy_plot(hourly: pd.Series, event_times: List[pd.Timestamp], outpath: Path, title: str) -> None:
    sub = hourly.reindex(EXPECTED_HOURLY_INDEX)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sub.index, sub.values, color="#2b8cbe", linewidth=0.8)
    for t in event_times:
        ax.axvline(t, color="red", linestyle="--", linewidth=0.6)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Discharge (m3/s)")
    ax.set_title(title)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def compute_metrics(hourly: pd.Series) -> dict:
    h = hourly.reindex(EXPECTED_HOURLY_INDEX)
    valid = h.dropna()
    completeness = 100.0 * valid.size / len(h)
    zero_frac = float((valid == 0).sum()) / valid.size if valid.size else np.nan
    diffs = valid.diff().abs().dropna()
    max_jump = float(diffs.max()) if not diffs.empty else np.nan
    max_jump_time = str(diffs.idxmax()) if not diffs.empty else None
    q95 = float(valid.quantile(0.95)) if not valid.empty else np.nan
    q99 = float(valid.quantile(0.99)) if not valid.empty else np.nan
    return {
        "hourly_completeness_pct": round(float(completeness), 3),
        "zero_flow_fraction": zero_frac,
        "max_hourly_jump": max_jump,
        "max_hourly_jump_time": max_jump_time,
        "q95": q95,
        "q99": q99,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "hourly_dir": output_dir / "hourly_series",
        "plots_dir": output_dir / "plots",
        "tables_dir": output_dir / "tables",
        "review_bundle": output_dir / "review_bundle",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)

    results_path = find_results_csv(Path(args.results_dir))
    logger.info("Loading screening results from %s", results_path)
    df = load_screening_results(results_path)

    # Select RBI_READY basins by default
    df_ready = df[df["screening_status"] == "RBI_READY"].copy()
    if df_ready.empty:
        logger.error("No RBI_READY basins found; aborting")
        return

    sample = sample_by_bins(df_ready, per_bin=args.per_bin, max_basins=args.max_basins, logger=logger)
    logger.info("Selected %s basins for event review", len(sample))

    session = requests.Session()
    session.headers.update({"User-Agent": "Flash-NH event hydrograph review script"})

    review_rows = []
    total_plots = 0
    failed_sites = []

    for _, row in sample.iterrows():
        staid = row["STAID"]
        rbi = row.get("rbi")
        area_bin = row.get("area_bin")
        huc02 = row.get("HUC02")
        state = row.get("STATE")
        bfi = row.get("BFI_AVE")
        drain = row.get("DRAIN_SQKM")
        logger.info("Processing basin %s", staid)

        try:
            hourly = retrieve_hourly_series(session, staid, logger, args.sleep_seconds, args.timeout_seconds, args.debug_raw, paths["hourly_dir"]) 
            if hourly.empty or hourly.dropna().empty:
                logger.warning("No hourly values for %s", staid)
                failed_sites.append(staid)
                continue

            # save hourly lightweight CSV
            hourly_df = hourly.reset_index()
            hourly_df.columns = ["dateTime", "discharge"]
            hourly_df.to_csv(paths["hourly_dir"] / f"{staid}_hourly.csv", index=False)

            metrics = compute_metrics(hourly)
            events = detect_events(hourly)
            # merge top event times
            merged_jumps = merge_event_windows(events.get("top_jumps", []), pd.Timedelta(days=7))
            merged_peaks = merge_event_windows(events.get("top_peaks", []), pd.Timedelta(days=7))
            event_times = sorted(set(merged_jumps + merged_peaks))

            # create plots per event (limit per basin)
            basin_plot_dir = paths["plots_dir"] / (row.get("rbi_bin") or "unknown") / staid
            num_event_plots = 0
            for i, et in enumerate(event_times[:5]):
                title = f"{staid} | RBI={rbi:.3f} | BFI={bfi} | area={drain} km2 | {area_bin} | {huc02}/{state} | completeness={metrics['hourly_completeness_pct']:.1f}% | event_rank={i+1}"
                outpath = basin_plot_dir / f"event_{i+1}_{et.date().isoformat()}.png"
                make_event_plot(hourly, pd.to_datetime(et), outpath, title, event_marker=pd.to_datetime(et))
                num_event_plots += 1
                total_plots += 1

            # full WY plot
            full_title = f"{staid} full WY | RBI={rbi:.3f} | completeness={metrics['hourly_completeness_pct']:.1f}%"
            full_out = basin_plot_dir / f"full_wy_{staid}.png"
            top_times = event_times[:10]
            make_full_wy_plot(hourly, top_times, full_out, full_title)
            total_plots += 1

            # auto flags
            auto_flags = {
                "rbi_ge_1p0": bool(rbi is not None and float(rbi) >= 1.0),
                "low_completeness": bool(metrics["hourly_completeness_pct"] < 90),
                "zero_flow_fraction_high": bool(metrics["zero_flow_fraction"] is not None and metrics["zero_flow_fraction"] > 0.25),
                "max_jump_adjacent_to_gap": False,
                "repeated_isolated_spikes": False,
                "flatline_detected": False,
            }
            # simple tests for adjacency-to-gap and flatline
            valid = hourly.dropna()
            if not valid.empty:
                diffs = valid.diff().abs()
                mx_idx = diffs.idxmax() if not diffs.empty else None
                if mx_idx is not None:
                    # check neighbors for NaN
                    prev = mx_idx - pd.Timedelta(hours=1)
                    nxt = mx_idx + pd.Timedelta(hours=1)
                    if prev not in hourly.index or pd.isna(hourly.get(prev)) or nxt not in hourly.index or pd.isna(hourly.get(nxt)):
                        auto_flags["max_jump_adjacent_to_gap"] = True
                # flatline: many equal values in a row
                runs = (valid == valid.shift()).astype(int).groupby((valid != valid.shift()).cumsum()).sum()
                if runs.max() >= 24:
                    auto_flags["flatline_detected"] = True

            review_row = {
                "STAID": staid,
                "RBI": rbi,
                "RBI_bin": row.get("rbi_bin"),
                "BFI_AVE": bfi,
                "DRAIN_SQKM": drain,
                "area_bin": area_bin,
                "HUC02": huc02,
                "STATE": state,
                "hourly_completeness_pct": metrics["hourly_completeness_pct"],
                "zero_flow_fraction": metrics["zero_flow_fraction"],
                "max_hourly_jump": metrics["max_hourly_jump"],
                "max_hourly_jump_time": metrics["max_hourly_jump_time"],
                "q95": metrics["q95"],
                "q99": metrics["q99"],
                "top_event_times": json.dumps([str(t) for t in event_times]),
                "number_of_event_plots": num_event_plots,
                "auto_flags": json.dumps(auto_flags),
            }
            review_rows.append(review_row)
        except Exception as exc:
            logger.exception("Failed processing %s: %s", staid, exc)
            failed_sites.append(staid)

    # write outputs
    sample_csv = output_dir / "event_hydrograph_review_sample.csv"
    pd.DataFrame(sample[["STAID", "rbi", "rbi_bin"]].copy()).to_csv(sample_csv, index=False)

    metrics_df = pd.DataFrame(review_rows)
    metrics_df.to_csv(output_dir / "event_hydrograph_review_metrics.csv", index=False)

    summary = {
        "selected_basins": int(len(sample)),
        "successful_basins": int(len(metrics_df)),
        "failed_basins": failed_sites,
        "total_event_plots": int(total_plots),
        "top_reviewed_basins": metrics_df.sort_values("RBI", ascending=False).head(20)[["STAID", "RBI"]].to_dict(orient="records"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "event_hydrograph_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # markdown summary
    md_lines = [
        "# Event Hydrograph Review Summary",
        "",
        f"Selected basins: {summary['selected_basins']}",
        f"Successful basins: {summary['successful_basins']}",
        f"Failed basins: {len(summary['failed_basins'])}",
        f"Total event plots generated: {summary['total_event_plots']}",
        "",
        "## Top 20 highest-RBI reviewed basins",
        "",
    ]
    for rec in summary["top_reviewed_basins"]:
        md_lines.append(f"- {rec['STAID']}: RBI={rec['RBI']}")
    (output_dir / "event_hydrograph_review_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    # review bundle manifestation: pick up to 5 basins per bin and include their fullwy+1 event plot
    bundle_dir = paths["review_bundle"]
    bundle_dir.mkdir(parents=True, exist_ok=True)
    selected_for_bundle = []
    by_bin = metrics_df.groupby("RBI_bin")
    for bin_name, group in by_bin:
        picks = group.sort_values("RBI", ascending=False).head(5)
        selected_for_bundle.extend(picks["STAID"].astype(str).tolist())
    # copy files
    manifest = {"files": [], "generated_at": datetime.now(timezone.utc).isoformat()}
    for staid in selected_for_bundle:
        # find plot files
        for p in paths["plots_dir"].rglob(f"*/{staid}/*.png"):
            rel = p.relative_to(output_dir)
            dst = bundle_dir / p.name
            try:
                dst.write_bytes(p.read_bytes())
                manifest["files"].append(str(rel).replace('\\', '/'))
            except Exception:
                continue
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Event hydrograph review outputs written to {output_dir}")
    print(f"Selected: {len(sample)}; Successful: {len(metrics_df)}; Plots: {total_plots}")


if __name__ == "__main__":
    main()
