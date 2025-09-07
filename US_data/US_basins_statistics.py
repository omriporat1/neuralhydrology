#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
import re
import difflib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.lines import Line2D

IV_CSV_DEFAULT = r"C:\PhD\Python\neuralhydrology\US_data\iv_scan_results.csv"
ATTR_DIR_DEFAULT = r"C:\PhD\Python\neuralhydrology\US_data\attributes"
OUT_FIG_DEFAULT = r"C:\PhD\Python\neuralhydrology\US_data\basin_climate_histograms.png"
OUT_DESC_DEFAULT = r"C:\PhD\Python\neuralhydrology\US_data\basin_attribute_descriptions.csv"  # NEW

ID_CANDIDATES = ["site_id", "STAID", "staid", "gage_id", "usgs_id", "site_no", "site_no_txt", "gageid", "station_id"]

# Attribute specs: name used in plot, plus candidate column names to search across files
# Synonyms target GAGES-II column names (files like attributes_gageii_*.csv). All matches are case-insensitive.
ATTR_SPECS = [
    # Core climate (as in the figure)
    ("p-mean",          ["p_mean", "p-mean", "pmean", "p_mean_mm_per_day", "pre_mm_syr/365", "pre_mm_syr_per_day", "pptavg_basin/365"]),
    ("pet-mean",        ["pet_mean", "pet-mean", "petmean", "pet_mean_mm_per_day", "pet/365", "pet_mm/365"]),
    ("aridity-index",   ["aridity_index", "aridity-index", "ari_ix_sav", "moisture_index_inv", "ai"]),
    ("frac-snow",       ["frac_snow", "frac-snow", "snow_pct_precip", "fraction_snow", "snow_fraction"]),
    ("high-prec-freq",  ["high_prec_freq", "high-prec-freq", "high_precip_freq", "high_prec_frequency"]),
    ("high-prec-dur",   ["high_prec_dur", "high-prec-dur", "high_precip_dur", "high_prec_duration"]),
    ("low-prec-freq",   ["low_prec_freq", "low-prec-freq", "low_precip_freq", "low_prec_frequency"]),
    ("low-prec-dur",    ["low_prec_dur", "low-prec-dur", "low_precip_dur", "low_prec_duration"]),
    # Elevation (GAGES-II Topo)
    ("elev_mean_m_basin", ["elev_mean_m_basin", "elev_mean_m", "elev_mean", "ele_mt_sav"]),
    # Morphology, topography, climate (GAGES-II)
    ("bas_compactness", ["bas_compactness", "compactness", "compactness_ratio"]),
    ("slope_pct",       ["slope_pct", "slope_pct_basin", "slope_mean_percent", "slope_percent", "slp_dg_sav"]),  # includes degree-based alias
    ("pptavg_basin",    ["pptavg_basin", "ppt_avg_basin", "pre_mm_syr", "p_mm_syr", "precip_mm_year", "ppt_mm_syr"]),
    ("t_avg_basin",     ["t_avg_basin", "tavg_basin", "tmp_dc_syr", "tmean_c", "temp_mean_c"]),
    ("pet_mm",          ["pet_mm", "pet", "pet_total_mm", "pet_mean_mm_per_year"]),
    ("snow_pct_precip", ["snow_pct_precip", "frac_snow", "snow_fraction"]),
    ("precip_seas_ind", ["precip_seas_ind", "p_seasonality", "precip_seasonality"]),

    # Land cover (replace pct_urban/pct_forest/pct_agri with NLCD06 fields)
    ("DEVNLCD06",       ["devnlcd06"]),
    ("FORESTNLCD06",    ["forestnlcd06"]),
    ("PLANTNLCD06",     ["plantnlcd06"]),

    # Soils/permeability (replace AWC_AVG with PERMAVE, and CLAY_PCT with CLAYAVE)
    ("PERMAVE",         ["permave"]),                             # new requested field
    ("CLAYAVE",         ["clayave"]),                             # new requested field


    # Hydrologic disturbance
    ("hydro_disturb_indx", ["hydro_disturb_indx", "hydromod_indx", "hydromod_index", "disturbance_index"]),
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
    # RETURN: (df, source_file, source_col) or (None, None, None)
    cand_low = [c.lower() for c in candidates]
    for name, df in dfs.items():
        cols = set(df.columns)
        for c in cand_low:
            if c in cols:
                s = df[["site_id", c]].copy()
                s.columns = ["site_id", "value"]
                s["value"] = pd.to_numeric(s["value"], errors="coerce")
                return s, name, c
    return None, None, None

def build_attributes_for_sites(sites: List[str], dfs: Dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    base = pd.DataFrame({"site_id": [str(s).zfill(8) for s in sites]})
    sources: Dict[str, Dict[str, str]] = {}

    # collect all columns (for suggestions)
    all_cols = {name: [c for c in df.columns if c != "site_id"] for name, df in dfs.items()}

    for label, cand in ATTR_SPECS:
        ser, src_file, src_col = pick_attribute_series(dfs, cand)
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
        sources[label] = {"file": src_file, "column": src_col}
    return base, sources

def _labels_for_plot(attr_sel: pd.DataFrame, attr_all: pd.DataFrame) -> List[str]:
    labels = [lab for lab, _ in ATTR_SPECS if lab in attr_sel.columns or lab in attr_all.columns]
    if "area_km2" in attr_sel.columns or "area_km2" in attr_all.columns:
        if "area_km2" not in labels:
            labels = ["area_km2"] + labels
    return labels

def load_var_descriptions(attr_dir: str) -> Dict[str, str]:
    """Load VARIABLE_NAME -> DESCRIPTION from Var description_gageii.xlsx (all sheets)."""
    xls_candidates = (
        list(Path(attr_dir).glob("Var description_gageii.xlsx")) +
        list(Path(attr_dir).glob("*Var*description*gageii*.xlsx"))
    )
    if not xls_candidates:
        print("[WARN] Var description_gageii.xlsx not found; descriptions will be empty.")
        return {}

    xls_path = str(xls_candidates[0])
    try:
        # read all sheets to be safe
        sheets = pd.read_excel(xls_path, sheet_name=None, engine="openpyxl")
    except Exception:
        # try default engine if openpyxl is not available
        sheets = pd.read_excel(xls_path, sheet_name=None)

    desc_map: Dict[str, str] = {}

    def norm(s: str) -> str:
        return str(s).strip().lower()

    for _, df in sheets.items():
        # normalize headers
        df = df.rename(columns={str(c): str(c).strip() for c in df.columns})
        cols_lower = {c.lower(): c for c in df.columns}

        # look specifically for VARIABLE_NAME and DESCRIPTION
        var_col = cols_lower.get("variable_name") or cols_lower.get("variable") or cols_lower.get("name")
        desc_col = cols_lower.get("description")
        if var_col is None or desc_col is None:
            continue

        sub = df[[var_col, desc_col]].dropna()
        for _, row in sub.iterrows():
            key = norm(row[var_col])
            val = str(row[desc_col]).strip()
            if key and key != "nan" and val:
                desc_map[key] = val

    if not desc_map:
        print(f"[WARN] No VARIABLE_NAME/DESCRIPTION pairs found in {xls_path}.")
    else:
        print(f"[info] Loaded {len(desc_map)} variable descriptions from {Path(xls_path).name}")
    return desc_map

def write_description_csv(labels: List[str],
                          sources: Dict[str, Dict[str, str]],
                          desc_map: Dict[str, str],
                          out_csv: str) -> None:
    def norm(s: str) -> str:
        return str(s).strip().lower()

    # helper to get description with fuzzy fallback
    def lookup_desc(key: str) -> str:
        k = norm(key)
        if k in desc_map:
            return desc_map[k]
        # try simple variants
        variants = {
            k,
            k.replace("_", ""),
            k.replace("_", " "),
            k.replace("-", "_")
        }
        for v in variants:
            if v in desc_map:
                return desc_map[v]
        # fuzzy fallback
        candidates = difflib.get_close_matches(k, list(desc_map.keys()), n=1, cutoff=0.8)
        if candidates:
            return desc_map[candidates[0]]
        return ""

    rows = []
    for lab in labels:
        if lab == "area_km2":
            rows.append({
                "plot_name": lab,
                "source_file": "iv_scan_results.csv",
                "source_column": "drainage_km2",
                "description": "Drainage area (km^2) from CAMELSH IV scan results."
            })
            continue

        src = sources.get(lab, {})
        src_col = (src.get("column") or "")
        src_file = src.get("file") or ""
        desc = lookup_desc(src_col)

        rows.append({
            "plot_name": lab,
            "source_file": src_file,
            "source_column": src_col,
            "description": desc or "(description not found)"
        })

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[saved] descriptions table -> {out_csv}")

def plot_histograms(
    attr_sel: pd.DataFrame,
    attr_all: pd.DataFrame,
    out_path: str,
    n_samples_sel: int,
    min_area: float,
    max_area: float,
    n_samples_all: int,
):
    # labels present from specs
    labels = _labels_for_plot(attr_sel, attr_all)
    if not labels:
        raise SystemExit("No attributes available to plot.")

    # styles: color distinguishes dataset; style distinguishes statistic
    FILT_COLOR = "#1f77b4"   # filtered (bars + lines)
    ALL_COLOR  = "#666666"   # all (lines), bars use a lighter gray
    FILT_BAR   = FILT_COLOR
    ALL_BAR    = "#BBBBBB"
    MEAN_LS    = "-"         # mean = solid
    MEDIAN_LS  = "--"        # median = dashed

    n = len(labels)
    nrows, ncols = _grid_dims(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.2*nrows))
    axes = np.array(axes).ravel()

    for i, lab in enumerate(labels):
        ax = axes[i]

        s_vals = pd.to_numeric(attr_sel.get(lab), errors="coerce").dropna()
        a_vals = pd.to_numeric(attr_all.get(lab), errors="coerce").dropna()

        # common bin edges
        if len(s_vals) and len(a_vals):
            vmin = float(min(s_vals.min(), a_vals.min()))
            vmax = float(max(s_vals.max(), a_vals.max()))
        elif len(a_vals):
            vmin, vmax = float(a_vals.min()), float(a_vals.max())
        elif len(s_vals):
            vmin, vmax = float(s_vals.min()), float(s_vals.max())
        else:
            ax.set_title(lab + " (no data)")
            ax.axis("off")
            continue
        if vmin == vmax:
            delta = max(1e-6, abs(vmin) * 1e-3)
            vmin -= delta
            vmax += delta
        edges = np.linspace(vmin, vmax, 21)

        # histograms (all first as background)
        if len(a_vals):
            ax.hist(a_vals, bins=edges, color=ALL_BAR, edgecolor="white", alpha=0.5, label="All")
        if len(s_vals):
            ax.hist(s_vals, bins=edges, color=FILT_BAR, edgecolor="white", alpha=0.8, label="Filtered")

        # lines and stats — Filtered
        if len(s_vals):
            s_min, s_max = float(s_vals.min()), float(s_vals.max())
            s_mean, s_med = float(s_vals.mean()), float(s_vals.median())
            ax.axvline(s_mean, color=FILT_COLOR, lw=1.4, ls=MEAN_LS)
            ax.axvline(s_med,  color=FILT_COLOR, lw=1.2, ls=MEDIAN_LS)
        # lines and stats — All
        if len(a_vals):
            a_min, a_max = float(a_vals.min()), float(a_vals.max())
            a_mean, a_med = float(a_vals.mean()), float(a_vals.median())
            ax.axvline(a_mean, color=ALL_COLOR, lw=1.4, ls=MEAN_LS)
            ax.axvline(a_med,  color=ALL_COLOR, lw=1.2, ls=MEDIAN_LS)

        # stats box
        lines = []
        if len(s_vals):
            lines.append(f"F: min={s_min:.2f} max={s_max:.2f} mean={s_mean:.2f} med={s_med:.2f}")
        if len(a_vals):
            lines.append(f"A: min={a_min:.2f} max={a_max:.2f} mean={a_mean:.2f} med={a_med:.2f}")
        if lines:
            ax.text(0.02, 0.98, "\n".join(lines),
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.75))

        ax.set_title(lab)
        ax.grid(True, alpha=0.2)

    for j in range(len(labels), nrows * ncols):
        axes[j].axis("off")

    # Legend below plots (outside), avoids covering axes
    fig.legend(
        handles=[
            Line2D([0], [0], color=ALL_BAR, lw=6, label="All distribution"),
            Line2D([0], [0], color=FILT_BAR, lw=6, label="Filtered distribution"),
            Line2D([0], [0], color=ALL_COLOR, ls=MEAN_LS, lw=1.8, label="All mean"),
            Line2D([0], [0], color=ALL_COLOR, ls=MEDIAN_LS, lw=1.8, label="All median"),
            Line2D([0], [0], color=FILT_COLOR, ls=MEAN_LS, lw=1.8, label="Filtered mean"),
            Line2D([0], [0], color=FILT_COLOR, ls=MEDIAN_LS, lw=1.8, label="Filtered median"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=True
    )

    fig.suptitle(
        f"Attributes — Filtered (N={n_samples_sel}, area {min_area:g}-{max_area:g} km^2) "
        f"vs All (N={n_samples_all}) — 20-bin histograms"
    )
    # leave space at top and bottom for title and legend
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    fig.savefig(out_path, dpi=200)
    print(f"[saved] {out_path}")
    try:
        plt.show()
    except Exception:
        pass
    return labels  # NEW: return the labels used

def list_available_columns(dfs: Dict[str, pd.DataFrame]) -> None:
    print("Available columns per attributes CSV (lower-cased):")
    for name, df in sorted(dfs.items()):
        cols = [c for c in df.columns if c != "site_id"]
        print(f"- {name}:")
        # print in wrapped lines
        line, buf = 0, []
        for c in sorted(cols):
            buf.append(c)
            if len(buf) >= 8:
                print("  ", ", ".join(buf))
                buf = []
                line += 1
        if buf:
            print("  ", ", ".join(buf))

def _grid_dims(n: int):
    """Choose near-square subplot grid with at least 3 columns."""
    if n <= 0:
        return 1, 1
    ncols = max(3, int(math.ceil(math.sqrt(n))))
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iv-csv", default=IV_CSV_DEFAULT)
    ap.add_argument("--attr-dir", default=ATTR_DIR_DEFAULT)
    ap.add_argument("--out-fig", default=OUT_FIG_DEFAULT)
    ap.add_argument("--out-desc", default=OUT_DESC_DEFAULT)  # NEW
    ap.add_argument("--min-area", type=float)
    ap.add_argument("--max-area", type=float)
    ap.add_argument("--list-cols", action="store_true", help="List all available attribute columns and exit")
    args = ap.parse_args()

    if not os.path.exists(args.iv_csv):
        raise SystemExit(f"IV CSV not found: {args.iv_csv}")
    if not os.path.isdir(args.attr_dir):
        raise SystemExit(f"Attribute folder not found: {args.attr_dir}")

    iv_df = load_iv_table(args.iv_csv)
    dfs = load_attribute_tables(args.attr_dir)

    if args.list_cols:
        list_available_columns(dfs)
        return

    min_a, max_a = ask_area_bounds(args)

    filt = filter_basins(iv_df, min_a, max_a)
    print(f"[filter] {len(filt)} basins matched "
          f"(area {min_a}-{max_a} km^2, median_dt_min=15, iv_start<=2014, iv_end==2025)")
    if filt.empty:
        print("No basins matched the filter.")
        return

    # filtered attributes
    attrs_sel, src_sel = build_attributes_for_sites(filt["site_id"].tolist(), dfs)
    attrs_sel = attrs_sel.merge(filt[["site_id", "drainage_km2"]], on="site_id", how="left")
    attrs_sel.rename(columns={"drainage_km2": "area_km2"}, inplace=True)

    # ALL attributes (unfiltered)
    all_ids = iv_df["site_id"].astype(str).str.zfill(8).unique().tolist()
    attrs_all, src_all = build_attributes_for_sites(all_ids, dfs)
    attrs_all = attrs_all.merge(iv_df[["site_id", "drainage_km2"]], on="site_id", how="left")
    attrs_all.rename(columns={"drainage_km2": "area_km2"}, inplace=True)

    print("[area filtered] min/median/mean/max:",
          attrs_sel["area_km2"].min(), attrs_sel["area_km2"].median(),
          attrs_sel["area_km2"].mean(), attrs_sel["area_km2"].max())

    labels_used = plot_histograms(
        attrs_sel, attrs_all, args.out_fig,
        n_samples_sel=len(filt),
        min_area=min_a, max_area=max_a,
        n_samples_all=len(all_ids),
    )

    # Build and save descriptions CSV
    desc_map = load_var_descriptions(args.attr_dir)
    # unify source maps (prefer sel, fallback to all)
    src_all_combined = {**src_all, **src_sel, **src_all}
    write_description_csv(labels_used, src_all_combined, desc_map, args.out_desc)

if __name__ == "__main__":
    main()