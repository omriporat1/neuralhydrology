#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
import re

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.lines import Line2D  # NEW

IV_CSV_DEFAULT = r"C:\PhD\Python\neuralhydrology\US_data\iv_scan_results.csv"
ATTR_DIR_DEFAULT = r"C:\PhD\Python\neuralhydrology\US_data\attributes"
OUT_FIG_DEFAULT = r"C:\PhD\Python\neuralhydrology\US_data\basin_climate_histograms.png"

ID_CANDIDATES = ["site_id", "STAID", "staid", "gage_id", "usgs_id", "site_no", "site_no_txt", "gageid", "station_id"]

# Attribute specs: name used in plot, plus candidate column names to search across files
# Synonyms expanded to better match GAGES-II files.
ATTR_SPECS = [
    # climate (original 9)
    ("p-mean",          ["p_mean", "p-mean", "pmean", "p_mean_mm_per_day", "pre_mm_syr/365", "pre_mm_syr_per_day"]),
    ("pet-mean",        ["pet_mean", "pet-mean", "petmean", "pet_mean_mm_per_day"]),
    ("aridity-index",   ["aridity_index", "aridity-index", "ari_ix_sav", "moisture_index_inv", "ai"]),
    ("p-seasonality",   ["p_seasonality", "p-seasonality", "seasonality", "precip_seasonality", "prec_seasonality"]),
    ("frac-snow",       ["frac_snow", "frac-snow", "fraction_snow", "snow_fraction"]),
    ("high-prec-freq",  ["high_prec_freq", "high-prec-freq", "high_precip_freq", "high_prec_frequency"]),
    ("high-prec-dur",   ["high_prec_dur", "high-prec-dur", "high_precip_dur", "high_prec_duration"]),
    ("low-prec-freq",   ["low_prec_freq", "low-prec-freq", "low_precip_freq", "low_prec_frequency"]),
    ("low-prec-dur",    ["low_prec_dur", "low-prec-dur", "low_precip_dur", "low_prec_duration"]),
    # extra basin/climate
    ("ele_mt_sav",      ["ele_mt_sav", "ele_mt_smn", "ele_mt_smx", "elev_mean", "elev_mean_m", "elevation_mean_m"]),
    ("dis_m3_pmn",      ["dis_m3_pmn", "dis_m3_pyr", "q_mean", "mean_discharge", "discharge_mean", "flow_mean", "q_mean_cms", "streamflow_mean"]),
    ("urb_pc_sse",      ["urb_pc_sse", "pct_urban", "percent_urban", "urban_percent", "pcturb2006", "pct_urb_2006", "pcturb"]),
    ("slp_dg_sav",      ["slp_dg_sav", "slope_deg_mean", "slope_mean_deg", "slope_mean"]),
    ("tmp_dc_syr",      ["tmp_dc_syr", "tmean_c", "tmp_mean_c", "temperature_mean_c", "temp_mean_c", "tmp_dc_smy"]),  # fixed: use *_syr
    ("pre_mm_syr",      ["pre_mm_syr", "ppt_mm_syr", "precip_mm_yr", "p_mm_syr", "precip_mm_year"]),
    ("forest_pc",       ["for_pc_sse", "forest_pc", "pct_forest", "pct_for", "percent_forest"]),
    ("crop_pc",         ["crp_pc_sse", "crop_pc", "pct_crop", "pct_cultivated", "pct_cult", "percent_cultivated"]),
]

def detect_id_column(cols: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for cand in ID_CANDIDATES:
        if cand.lower() in low:
            return low[cand.lower()]
    return None

def load_iv_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"site_id": str})
    # Normalize ID
    if "site_id" not in df.columns:
        id_col = detect_id_column(df.columns)
        if not id_col:
            raise SystemExit("Could not detect site_id column in IV CSV.")
        df["site_id"] = df[id_col].astype(str).str.zfill(8)
    else:
        df["site_id"] = df["site_id"].astype(str).str.zfill(8)

    # Parse times and numeric fields
    for col in ["iv_start", "iv_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
    if "median_dt_min" in df.columns:
        df["median_dt_min"] = pd.to_numeric(df["median_dt_min"], errors="coerce")
    if "drainage_km2" in df.columns:
        df["drainage_km2"] = pd.to_numeric(df["drainage_km2"], errors="coerce")
    return df

def ask_area_bounds(args) -> (float, float):
    if args.min_area is not None and args.max_area is not None:
        return float(args.min_area), float(args.max_area)
    try:
        min_a = float(input("Enter minimum drainage area (km^2): ").strip())
        max_a = float(input("Enter maximum drainage area (km^2): ").strip())
    except Exception:
        raise SystemExit("Invalid numeric input for area bounds.")
    return min_a, max_a

def filter_basins(iv_df: pd.DataFrame, min_area: float, max_area: float) -> pd.DataFrame:
    cond = (
        (iv_df["median_dt_min"] == 15)
        & (iv_df["iv_start"].dt.year <= 2014)
        & (iv_df["iv_end"].dt.year == 2025)
        & (iv_df["drainage_km2"] >= min_area)
        & (iv_df["drainage_km2"] <= max_area)
    )
    out = iv_df.loc[cond].copy()
    out = out.dropna(subset=["site_id"])
    return out

def load_attribute_tables(attr_dir: str) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}
    for p in Path(attr_dir).glob("*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        id_col = detect_id_column(df.columns)
        if not id_col:
            # keep for suggestions even if we can't use it
            continue
        df["site_id"] = df[id_col].astype(str).str.zfill(8)
        df.columns = [c if c == "site_id" else c.lower() for c in df.columns]
        dfs[p.name] = df
    if not dfs:
        raise SystemExit(f"No attribute CSVs with a recognizable ID column found in {attr_dir}")
    return dfs

def _suggest_cols(all_cols: Dict[str, List[str]], target: str, k: int = 5) -> List[str]:
    # simple substring/regex-based suggestions from available columns across files
    pat = re.sub(r"[_\-]+", ".*", target.lower())
    hits = []
    for fname, cols in all_cols.items():
        for c in cols:
            if re.search(pat, c):
                hits.append(f"{c} [{fname}]")
    # return up to k unique
    seen, out = set(), []
    for h in hits:
        c = h.split(" [", 1)[0]
        if c in seen:
            continue
        seen.add(c)
        out.append(h)
        if len(out) >= k:
            break
    return out

def pick_attribute_series(dfs: Dict[str, pd.DataFrame], candidates: List[str]) -> Optional[pd.DataFrame]:
    cand_low = [c.lower() for c in candidates]
    for name, df in dfs.items():
        cols = set(df.columns)
        for c in cand_low:
            if c in cols:
                s = df[["site_id", c]].copy()
                s.columns = ["site_id", "value"]
                s["value"] = pd.to_numeric(s["value"], errors="coerce")
                return s
    return None

def build_attributes_for_sites(sites: List[str], dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = pd.DataFrame({"site_id": [str(s).zfill(8) for s in sites]})

    # collect all columns (for suggestions)
    all_cols = {name: [c for c in df.columns if c != "site_id"] for name, df in dfs.items()}

    for label, cand in ATTR_SPECS:
        ser = pick_attribute_series(dfs, cand)
        if ser is None:
            # helpful suggestion
            hints = []
            for c in cand:
                hints.extend(_suggest_cols(all_cols, c))
            if hints:
                print(f"[WARN] attribute '{label}' not found; examples of similar columns: {', '.join(hints[:6])}")
            else:
                print(f"[WARN] attribute '{label}' not found in any CSV; skipping.")
            continue
        base = base.merge(ser, on="site_id", how="left")
        base.rename(columns={"value": label}, inplace=True)
    return base

def _grid_dims(n: int) -> tuple[int, int]:
    # NEW: choose a near-square layout
    ncols = max(3, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)
    return nrows, ncols

def plot_histograms(attr_df: pd.DataFrame, out_path: str, n_samples: int, min_area: float, max_area: float):
    # labels present from specs
    labels = [lab for lab, _ in ATTR_SPECS if lab in attr_df.columns]
    # include area if available (comes from IV table merge)
    if "area_km2" in attr_df.columns and "area_km2" not in labels:
        labels = ["area_km2"] + labels
    if not labels:
        raise SystemExit("No attributes available to plot.")

    n = len(labels)
    nrows, ncols = _grid_dims(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows))
    axes = np.array(axes).ravel()

    for i, lab in enumerate(labels):
        ax = axes[i]
        vals = pd.to_numeric(attr_df[lab], errors="coerce").dropna()
        ax.hist(vals, bins=20, color="#4C78A8", edgecolor="white")

        if len(vals) > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
            vmean, vmed = float(vals.mean()), float(vals.median())
            ax.axvline(vmean, color="tomato", lw=1.2, ls="--")  # mean (orange dashed)
            ax.axvline(vmed, color="seagreen", lw=1.2, ls="-")  # median (green solid)
            ax.text(
                0.02, 0.98,
                f"n={len(vals)}\nmin={vmin:.2f}\nmax={vmax:.2f}\nmean={vmean:.2f}\nmedian={vmed:.2f}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.7)
            )

        ax.set_title(lab)
        ax.grid(True, alpha=0.2)

    for j in range(len(labels), nrows * ncols):
        axes[j].axis("off")

    # Global legend explaining line styles
    fig.legend(
        handles=[
            Line2D([0], [0], color="tomato", ls="--", lw=1.5, label="Mean (orange, dashed)"),
            Line2D([0], [0], color="seagreen", ls="-", lw=1.5, label="Median (green, solid)"),
        ],
        loc="upper right", frameon=True
    )

    fig.suptitle(
        f"Climate and basin attributes — filtered basins "
        f"(N={n_samples}, area {min_area:g}-{max_area:g} km^2) — 20-bin histograms"
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    print(f"[saved] {out_path}")
    try:
        plt.show()
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv-csv", default=IV_CSV_DEFAULT)
    ap.add_argument("--attr-dir", default=ATTR_DIR_DEFAULT)
    ap.add_argument("--out-fig", default=OUT_FIG_DEFAULT)
    ap.add_argument("--min-area", type=float)
    ap.add_argument("--max-area", type=float)
    args = ap.parse_args()

    if not os.path.exists(args.iv_csv):
        raise SystemExit(f"IV CSV not found: {args.iv_csv}")
    if not os.path.isdir(args.attr_dir):
        raise SystemExit(f"Attribute folder not found: {args.attr_dir}")

    iv_df = load_iv_table(args.iv_csv)
    min_a, max_a = ask_area_bounds(args)

    filt = filter_basins(iv_df, min_a, max_a)
    print(f"[filter] {len(filt)} basins matched "
          f"(area {min_a}-{max_a} km^2, median_dt_min=15, iv_start<=2014, iv_end==2025)")

    if filt.empty:
        print("No basins matched the filter.")
        return

    dfs = load_attribute_tables(args.attr_dir)
    attrs = build_attributes_for_sites(filt["site_id"].tolist(), dfs)

    # Join drainage area for context and print basic stats
    attrs = attrs.merge(filt[["site_id", "drainage_km2"]], on="site_id", how="left")
    attrs.rename(columns={"drainage_km2": "area_km2"}, inplace=True)
    print("[area] min/median/mean/max:",
          attrs["area_km2"].min(), attrs["area_km2"].median(),
          attrs["area_km2"].mean(), attrs["area_km2"].max())

    plot_histograms(attrs, args.out_fig, n_samples=len(filt), min_area=min_a, max_area=max_a)

if __name__ == "__main__":
    main()