"""
analyze_manual_review_filter_rules.py

Analyzes manual hydrograph review labels and proposes basin-quality risk rules
for the Flash-NH hydrological modeling project.

PRELIMINARY ANALYSIS ONLY -- DO NOT USE OUTPUT STATUS AS FINAL TRAINING DECISIONS.
"""

import argparse
import json
import traceback
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent
ATTR_DIR     = Path("C:/PhD/Python/neuralhydrology/US_data/attributes")
REV_DIR      = REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v004_main_training_candidate"
LABELS_CSV   = REV_DIR / "manual_review_labels.csv"
TEMPLATE_CSV = REV_DIR / "tables/human_review_template.csv"
METRICS_CSV  = REPO_ROOT / "reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv"
OUT_DIR      = REPO_ROOT / "reports/flashnh_manual_review_rule_analysis_v001"

ATTR_FILES = {
    "BasinID":        "attributes_gageii_BasinID.csv",
    "Bas_Classif":    "attributes_gageii_Bas_Classif.csv",
    "Bas_Morph":      "attributes_gageii_Bas_Morph.csv",
    "Bound_QA":       "attributes_gageii_Bound_QA.csv",
    "Climate":        "attributes_gageii_Climate.csv",
    "FlowRec":        "attributes_gageii_FlowRec.csv",
    "Geology":        "attributes_gageii_Geology.csv",
    "Hydro":          "attributes_gageii_Hydro.csv",
    "HydroMod_Dams":  "attributes_gageii_HydroMod_Dams.csv",
    "HydroMod_Other": "attributes_gageii_HydroMod_Other.csv",
    "LC06_Basin":     "attributes_gageii_LC06_Basin.csv",
    "hydroATLAS":     "attributes_hydroATLAS.csv",
    "nldas2":         "attributes_nldas2_climate.csv",
}

# Proxy substitution notes
PROXY_NOTES = {
    "WATERNLCD06":    "LC06_Basin: open water pct (proxy for lentic; HIRES_LENTIC_PCT not in GAGES-II)",
    "CANALS_PCT":     "HydroMod_Other: canals pct (proxy for artificial path; ARTIFPATH_PCT not in GAGES-II)",
    "dor_pc_pva":     "hydroATLAS: degree of regulation (proxy for regulated storage)",
    "lka_pc_use":     "hydroATLAS: lake area pct (additional lentic proxy)",
}

# Decision color map
DECISION_COLORS = {
    "KEEP":                "green",
    "EXCLUDE":             "red",
    "KEEP_LOW_CONFIDENCE": "orange",
    "UNSURE":              "gray",
}

# ---------------------------------------------------------------------------
# RUN LOG
# ---------------------------------------------------------------------------
run_log: list[str] = []


def log(msg: str) -> None:
    """Print and record a log message (ASCII-safe)."""
    run_log.append(msg)
    print(msg)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def normalize_staid(series: pd.Series) -> pd.Series:
    """Normalize a STAID column to a consistent string key for joining.

    USGS station IDs are nominally 8-digit zero-padded strings, but some data
    sources (notably HydroATLAS) store them as plain integers without leading
    zeros (e.g. '3144816' instead of '03144816').  This function:
      1. Strips surrounding whitespace.
      2. Zero-pads to 8 characters IF the value is shorter than 8 chars
         (str.zfill(8) is a no-op for strings already >= 8 characters, so
         9-char sites like '103366092' and 15-char coordinate-based IDs are
         left unchanged).
      3. Never truncates.

    Apply to EVERY dataframe that participates in a STAID join before joining.
    """
    return series.astype(str).str.strip().str.zfill(8)


def safe_read_csv(path: Path, dtype_staid: bool = True, **kwargs) -> pd.DataFrame | None:
    """Read CSV with STAID as string; return None on failure."""
    try:
        dt = {"STAID": str} if dtype_staid else {}
        return pd.read_csv(path, dtype=dt, **kwargs)
    except Exception as exc:
        log(f"  WARNING: Could not read {path}: {exc}")
        return None


def pct_fn(df: pd.DataFrame, col: str, q: float) -> float | None:
    """Return quantile q for col in df, or None if col missing."""
    if col in df.columns:
        return float(df[col].quantile(q))
    return None


def safe_col(df: pd.DataFrame, col: str) -> pd.Series | None:
    """Return a column Series or None."""
    return df[col] if col in df.columns else None


def flag_series(df: pd.DataFrame, mask: pd.Series, fill: bool = False) -> pd.Series:
    """Convert boolean mask to a bool Series aligned to df.index, filling NaN with fill."""
    s = mask.reindex(df.index).fillna(fill)
    return s.astype(bool)


def rule_stats(reviewed: pd.DataFrame, flag_col: str) -> dict:
    """Compute per-decision stats for a boolean rule column on the reviewed subset."""
    flagged = reviewed[reviewed[flag_col] == True]
    n = len(flagged)
    if n == 0:
        return {
            "reviewed_flagged_n": 0,
            "reviewed_exclude_n": 0,
            "reviewed_unsure_n": 0,
            "reviewed_keep_n": 0,
            "reviewed_keep_low_confidence_n": 0,
            "reviewed_exclude_or_unsure_rate": float("nan"),
            "reviewed_false_positive_keep_rate": float("nan"),
        }
    dec = flagged["human_decision"] if "human_decision" in flagged.columns else pd.Series([], dtype=str)
    vc = dec.value_counts()
    excl  = int(vc.get("EXCLUDE", 0))
    uns   = int(vc.get("UNSURE", 0))
    keep  = int(vc.get("KEEP", 0))
    klc   = int(vc.get("KEEP_LOW_CONFIDENCE", 0))
    return {
        "reviewed_flagged_n": n,
        "reviewed_exclude_n": excl,
        "reviewed_unsure_n": uns,
        "reviewed_keep_n": keep,
        "reviewed_keep_low_confidence_n": klc,
        "reviewed_exclude_or_unsure_rate": round((excl + uns) / n, 4),
        "reviewed_false_positive_keep_rate": round(keep / n, 4),
    }


def full_stats(full: pd.DataFrame, flag_col: str) -> dict:
    """Compute full candidate flagging stats."""
    n_total = len(full)
    n_flagged = int(full[flag_col].sum()) if flag_col in full.columns else 0
    return {
        "full_candidate_flagged_n": n_flagged,
        "full_candidate_flagged_pct": round(n_flagged / n_total * 100, 2) if n_total > 0 else 0.0,
    }


def stacked_hbar(
    df_cross: pd.DataFrame,
    title: str,
    out_path: Path,
    color_map: dict | None = None,
) -> None:
    """Draw a stacked horizontal bar chart from a cross-tab DataFrame."""
    if color_map is None:
        color_map = {}
    cols = list(df_cross.columns)
    colors = [color_map.get(c, None) for c in cols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_cross) * 0.55 + 1.5)))
    left = np.zeros(len(df_cross))
    bars = df_cross.values
    for i, col in enumerate(cols):
        ax.barh(df_cross.index, bars[:, i], left=left,
                color=colors[i], label=col, edgecolor="white", linewidth=0.4)
        left += bars[:, i]
    ax.set_xlabel("Count")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def attr_boxplot(
    df: pd.DataFrame,
    var: str,
    label_col: str,
    out_path: Path,
    log_scale: bool = False,
) -> None:
    """Violin + strip plot for a variable by label group."""
    if var not in df.columns:
        return
    decisions = [d for d in DECISION_COLORS if d in df[label_col].values]
    data_groups = []
    labels_used = []
    for d in decisions:
        sub = df.loc[df[label_col] == d, var].dropna()
        if len(sub) > 0:
            data_groups.append(sub.values)
            labels_used.append(d)
    if not data_groups:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(labels_used) * 1.6 + 1), 5))
    positions = range(1, len(labels_used) + 1)
    parts = ax.violinplot(data_groups, positions=positions, showmedians=True, showextrema=True)
    for i, (pc, lbl) in enumerate(zip(parts["bodies"], labels_used)):
        pc.set_facecolor(DECISION_COLORS.get(lbl, "steelblue"))
        pc.set_alpha(0.6)
    # Overlay jitter
    for i, (grp, lbl) in enumerate(zip(data_groups, labels_used)):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(grp))
        ax.scatter(np.full(len(grp), i + 1) + jitter, grp,
                   s=18, alpha=0.7, color=DECISION_COLORS.get(lbl, "steelblue"), zorder=3)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels_used, fontsize=9)
    scale_note = " (log scale)" if log_scale else ""
    ax.set_title(f"{var} by human_decision (N={len(df)}){scale_note}", fontsize=10)
    ax.set_ylabel(var, fontsize=9)
    if log_scale:
        ax.set_yscale("log")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze manual hydrograph review labels and propose basin-quality risk rules."
    )
    parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Setup output directories
    # ------------------------------------------------------------------
    for sub in ("tables", "plots", "summaries", "logs"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {OUT_DIR}")

    # ------------------------------------------------------------------
    # 2. Load inputs
    # ------------------------------------------------------------------
    log("Loading inputs ...")

    labels_df = safe_read_csv(LABELS_CSV)
    if labels_df is None:
        raise FileNotFoundError(f"Cannot read LABELS_CSV: {LABELS_CSV}")
    log(f"  manual_review_labels.csv: {len(labels_df)} rows")

    template_df = safe_read_csv(TEMPLATE_CSV)
    if template_df is None:
        raise FileNotFoundError(f"Cannot read TEMPLATE_CSV: {TEMPLATE_CSV}")
    log(f"  human_review_template.csv: {len(template_df)} rows")

    metrics_df = safe_read_csv(METRICS_CSV)
    if metrics_df is None:
        raise FileNotFoundError(f"Cannot read METRICS_CSV: {METRICS_CSV}")
    log(f"  wy2024_metrics_with_site_metadata.csv: {len(metrics_df)} rows")

    mc_col = metrics_df["main_training_candidate"]
    if mc_col.dtype == object:
        mc_mask = mc_col.astype(str).str.strip().str.upper().isin({"TRUE", "1", "YES"})
    else:
        mc_mask = mc_col.astype(bool)
    full_cands = metrics_df[mc_mask].copy()
    log(f"  main_training_candidate == True: {len(full_cands)} rows")

    # Normalize STAID on all primary dataframes before any joins.
    # normalize_staid() applies str.strip() + str.zfill(8) so that sources like
    # HydroATLAS (which drops leading zeros) join correctly with USGS 8-digit IDs.
    for df in (labels_df, template_df, full_cands, metrics_df):
        if "STAID" in df.columns:
            df["STAID"] = normalize_staid(df["STAID"])

    # ------------------------------------------------------------------
    # 3. Load attribute files
    # ------------------------------------------------------------------
    attr_dfs: dict[str, pd.DataFrame] = {}
    join_audit_rows: list[dict] = []

    for name, fname in ATTR_FILES.items():
        fpath = ATTR_DIR / fname
        exists = fpath.exists()
        notes = ""
        row_count = None
        matched = None
        unmatched = None
        match_pct = None

        if not exists:
            log(f"  SKIP attribute file (not found): {fname}")
            notes = "File not found"
        else:
            df_attr = safe_read_csv(fpath)
            if df_attr is None:
                notes = "Read error"
            else:
                raw_ids_before = df_attr["STAID"].astype(str).str.strip().nunique() if "STAID" in df_attr.columns else 0
                df_attr["STAID"] = normalize_staid(df_attr["STAID"])
                row_count = len(df_attr)
                unique_norm_ids = int(df_attr["STAID"].nunique())
                dup_norm_ids = int(row_count - unique_norm_ids)

                merged_check = full_cands[["STAID"]].merge(
                    df_attr[["STAID"]], on="STAID", how="left", indicator=True
                )
                matched = int((merged_check["_merge"] == "both").sum())
                unmatched = int((merged_check["_merge"] == "left_only").sum())
                match_pct = round(matched / len(full_cands) * 100, 2) if len(full_cands) > 0 else 0.0

                # Compute first-10 unmatched candidate STAIDs
                unmatched_staids = list(
                    merged_check.loc[merged_check["_merge"] == "left_only", "STAID"].head(10)
                )
                # Compute first-10 attr IDs not used (present in attr but absent from candidates)
                candidate_set = set(full_cands["STAID"])
                attr_not_used = [
                    sid for sid in df_attr["STAID"].unique()
                    if sid not in candidate_set
                ][:10]

                attr_dfs[name] = df_attr
                log(f"  Loaded {name}: {row_count} rows, {matched}/{len(full_cands)} matched ({match_pct}%)")

                # Add proxy notes
                for col, note in PROXY_NOTES.items():
                    if col in df_attr.columns:
                        if notes:
                            notes += "; "
                        notes += f"proxy: {note}"

                join_audit_rows.append({
                    "source_name": name,
                    "file_path": str(fpath),
                    "exists": exists,
                    "row_count": row_count,
                    "unique_raw_ids": raw_ids_before,
                    "unique_normalized_ids": unique_norm_ids,
                    "duplicate_normalized_ids": dup_norm_ids,
                    "join_key_column": "STAID",
                    "matched_main_training_candidate_count": matched,
                    "match_pct": match_pct,
                    "unmatched_candidate_count": unmatched,
                    "first_10_unmatched_candidate_staids": "; ".join(unmatched_staids),
                    "first_10_attr_ids_not_used": "; ".join(attr_not_used),
                    "notes": notes if notes else "OK",
                })
                continue  # skip the append below

        join_audit_rows.append({
            "source_name": name,
            "file_path": str(fpath),
            "exists": exists,
            "row_count": row_count,
            "unique_raw_ids": None,
            "unique_normalized_ids": None,
            "duplicate_normalized_ids": None,
            "join_key_column": "STAID",
            "matched_main_training_candidate_count": matched,
            "match_pct": match_pct,
            "unmatched_candidate_count": unmatched,
            "first_10_unmatched_candidate_staids": "",
            "first_10_attr_ids_not_used": "",
            "notes": notes if notes else "OK",
        })

    # ------------------------------------------------------------------
    # 4. Build master joined table for full candidates
    # ------------------------------------------------------------------
    log("Building master joined table ...")

    master = full_cands.copy()
    for name, df_attr in attr_dfs.items():
        # Avoid duplicate columns (keep left if clash, suffix _attr)
        overlap = [c for c in df_attr.columns if c != "STAID" and c in master.columns]
        suffix = f"_{name}" if overlap else ""
        master = master.merge(
            df_attr,
            on="STAID",
            how="left",
            suffixes=("", suffix),
        )
        # Drop duplicate suffixed columns that are exact copies; keep original
        for c in overlap:
            dup_col = f"{c}{suffix}"
            if dup_col in master.columns:
                # If original already has values (non-NaN), prefer it; drop dup
                if master[c].notna().sum() >= master[dup_col].notna().sum():
                    master.drop(columns=[dup_col], inplace=True)
                else:
                    master.drop(columns=[c], inplace=True)
                    master.rename(columns={dup_col: c}, inplace=True)

    log(f"  Master joined table: {len(master)} rows, {len(master.columns)} columns")

    # ------------------------------------------------------------------
    # Build reviewed subset (73 basins)
    # ------------------------------------------------------------------
    # Prefer labels_df as source of truth for decisions (overrides template cols)
    # template_df has the metrics; labels_df has the decisions
    review_cols_from_labels = ["STAID", "human_decision", "hydrograph_behavior", "artifact_type",
                               "confidence", "reviewer_notes"]
    review_cols_present = [c for c in review_cols_from_labels if c in labels_df.columns]

    # Merge labels with template (template has all the numeric cols + candidate_class etc.)
    reviewed = template_df.copy()
    # Drop duplicate decision columns from template if they exist (use labels_df as authority)
    for col in ["human_decision", "hydrograph_behavior", "artifact_type", "confidence", "reviewer_notes"]:
        if col in reviewed.columns:
            reviewed.drop(columns=[col], inplace=True)
    reviewed = reviewed.merge(labels_df[review_cols_present], on="STAID", how="left")

    # Now join to master attribute table to pull in all attribute columns
    # Get columns from master that are NOT already in reviewed
    master_extra_cols = ["STAID"] + [
        c for c in master.columns
        if c not in reviewed.columns and c != "STAID"
    ]
    reviewed = reviewed.merge(master[master_extra_cols], on="STAID", how="left")

    # Normalize column names: template uses lowercase q50/q95/q99; metrics uses Q50/Q95/Q99
    for lc, uc in [("q50", "Q50"), ("q95", "Q95"), ("q99", "Q99")]:
        if lc in reviewed.columns and uc not in reviewed.columns:
            reviewed.rename(columns={lc: uc}, inplace=True)
        elif uc in reviewed.columns and lc in reviewed.columns:
            # Prefer uppercase (from metrics); drop lowercase
            reviewed.drop(columns=[lc], inplace=True)

    # Same for full_cands (already uppercase from metrics)
    # Ensure consistency
    for lc, uc in [("q50", "Q50"), ("q95", "Q95"), ("q99", "Q99")]:
        if lc in master.columns and uc not in master.columns:
            master.rename(columns={lc: uc}, inplace=True)

    log(f"  Reviewed subset: {len(reviewed)} rows, {len(reviewed.columns)} columns")

    # Recompute area_bin from DRAIN_SQKM
    def make_area_bin(s):
        if pd.isna(s):
            return "unknown"
        if s < 10:
            return "1-10 km2"
        if s < 100:
            return "10-100 km2"
        return "100-1000 km2"

    for df in (reviewed, master, full_cands):
        if "DRAIN_SQKM" in df.columns:
            df["area_bin"] = df["DRAIN_SQKM"].apply(make_area_bin)

    # ------------------------------------------------------------------
    # 4b. Write join audit table
    # ------------------------------------------------------------------
    audit_df = pd.DataFrame(join_audit_rows)
    audit_df.to_csv(OUT_DIR / "tables/input_join_audit.csv", index=False)
    log("  Wrote tables/input_join_audit.csv")

    # ------------------------------------------------------------------
    # 5. Label summary
    # ------------------------------------------------------------------
    log("Writing label summary ...")
    summary_rows = []

    for field in ["human_decision", "confidence", "hydrograph_behavior"]:
        if field not in labels_df.columns:
            continue
        vc = labels_df[field].fillna("(blank)").value_counts()
        total = vc.sum()
        for val, cnt in vc.items():
            summary_rows.append({
                "field": field, "value": val,
                "count": int(cnt), "pct": round(cnt / total * 100, 1)
            })

    # artifact_type: multi-value (split on ; or ,)
    if "artifact_type" in labels_df.columns:
        exploded = (
            labels_df["artifact_type"]
            .fillna("")
            .str.replace(",", ";", regex=False)
            .str.split(";")
            .explode()
            .str.strip()
            .replace("", "(blank)")
        )
        vc = exploded.value_counts()
        total = len(labels_df)
        for val, cnt in vc.items():
            summary_rows.append({
                "field": "artifact_type", "value": val,
                "count": int(cnt), "pct": round(cnt / total * 100, 1)
            })

    label_summary_df = pd.DataFrame(summary_rows)
    label_summary_df.to_csv(OUT_DIR / "tables/manual_review_label_summary.csv", index=False)
    log("  Wrote tables/manual_review_label_summary.csv")

    # ------------------------------------------------------------------
    # 6. Cross-tab by candidate_class
    # ------------------------------------------------------------------
    log("Writing cross-tab tables ...")

    def crosstab_with_pct(df, row_col, col_col):
        if row_col not in df.columns or col_col not in df.columns:
            return pd.DataFrame()
        ct = pd.crosstab(df[row_col], df[col_col])
        ct["row_pct_total"] = (ct.sum(axis=1) / ct.sum(axis=1).sum() * 100).round(1)
        # row pct for each decision col
        row_totals = ct.drop(columns=["row_pct_total"]).sum(axis=1)
        for c in ct.drop(columns=["row_pct_total"]).columns:
            ct[f"pct_{c}"] = (ct[c] / row_totals * 100).round(1)
        return ct

    ct_class = crosstab_with_pct(reviewed, "candidate_class", "human_decision")
    ct_class.to_csv(OUT_DIR / "tables/manual_review_by_candidate_class.csv")
    log("  Wrote tables/manual_review_by_candidate_class.csv")

    ct_group = crosstab_with_pct(reviewed, "review_group", "human_decision")
    ct_group.to_csv(OUT_DIR / "tables/manual_review_by_review_group.csv")
    log("  Wrote tables/manual_review_by_review_group.csv")

    # ------------------------------------------------------------------
    # 7 & 8. By area_bin, BFI_bin, HUC02, STATE
    # ------------------------------------------------------------------
    area_bfi_rows = []
    decisions_order = ["KEEP", "EXCLUDE", "KEEP_LOW_CONFIDENCE", "UNSURE"]

    for field in ["area_bin", "BFI_bin", "HUC02", "STATE"]:
        if field not in reviewed.columns:
            continue
        grp = reviewed.groupby(field)["human_decision"].value_counts().unstack(fill_value=0)
        for col in decisions_order:
            if col not in grp.columns:
                grp[col] = 0
        grp = grp[[c for c in decisions_order if c in grp.columns]]
        grp.index = [f"{field}={v}" for v in grp.index]
        area_bfi_rows.append(grp)

    if area_bfi_rows:
        area_bfi_df = pd.concat(area_bfi_rows)
        area_bfi_df.to_csv(OUT_DIR / "tables/manual_review_by_area_bfi_huc.csv")
        log("  Wrote tables/manual_review_by_area_bfi_huc.csv")

    # ------------------------------------------------------------------
    # 9. Attribute summary (per decision stats)
    # ------------------------------------------------------------------
    log("Writing attribute summary ...")

    ATTR_VARS = [
        "DRAIN_SQKM", "BFI_AVE", "RBI", "hourly_completeness_pct", "zero_flow_fraction",
        "Q50", "Q95", "Q99", "q95_q50_ratio", "q99_q50_ratio",
        "max_hourly_rise", "max_hourly_rise_per_km2", "max_abs_hourly_jump_over_Q50", "max_hourly_fall",
        "event_count_q95",
        "HYDRO_DISTURB_INDX", "BASIN_BOUNDARY_CONFIDENCE", "PCT_DIFF_NWIS",
        "WATERNLCD06", "CANALS_PCT", "lka_pc_use", "dor_pc_pva", "ria_ha_usu", "rev_mc_usu",
        "aridity_index", "low_prec_freq",
        "NDAMS_2009", "DDENS_2009", "STOR_NID_2009",
    ]

    attr_summary_rows = []
    for var in ATTR_VARS:
        if var not in reviewed.columns:
            continue
        col_data = pd.to_numeric(reviewed[var], errors="coerce")
        if col_data.isna().all():
            continue
        for dec in reviewed["human_decision"].dropna().unique():
            sub = col_data[reviewed["human_decision"] == dec]
            sub_valid = sub.dropna()
            n = len(sub_valid)
            miss = int(sub.isna().sum())
            row = {
                "variable": var,
                "human_decision": dec,
                "n": n,
                "mean": round(sub_valid.mean(), 6) if n > 0 else None,
                "median": round(sub_valid.median(), 6) if n > 0 else None,
                "p10": round(sub_valid.quantile(0.10), 6) if n > 0 else None,
                "p25": round(sub_valid.quantile(0.25), 6) if n > 0 else None,
                "p75": round(sub_valid.quantile(0.75), 6) if n > 0 else None,
                "p90": round(sub_valid.quantile(0.90), 6) if n > 0 else None,
                "min": round(sub_valid.min(), 6) if n > 0 else None,
                "max": round(sub_valid.max(), 6) if n > 0 else None,
                "missing_count": miss,
            }
            attr_summary_rows.append(row)

    attr_summary_df = pd.DataFrame(attr_summary_rows)
    attr_summary_df.to_csv(OUT_DIR / "tables/manual_review_attribute_summary.csv", index=False)
    log("  Wrote tables/manual_review_attribute_summary.csv")

    # ------------------------------------------------------------------
    # 10. Notes keyword flags
    # ------------------------------------------------------------------
    log("Writing notes keyword flags ...")

    KEYWORD_GROUPS = {
        "regulated_or_managed": [
            "regulated", "regulation", "dam", "reservoir", "managed",
            "release", "daily pattern", "hydropeaking", "lake"
        ],
        "tidal_or_backwater": ["tidal", "tide", "backwater"],
        "mostly_zero_or_ephemeral": ["zero", "dry", "ephemeral", "intermittent", "mostly zero"],
        "artifact_or_sensor": [
            "artifact", "sensor", "noise", "suspicious", "impossible", "bad data", "negative"
        ],
        "slow_or_large_basin": ["slow", "large", "sluggish", "broad"],
        "good_reference": ["good", "natural", "valid", "clean", "looks good"],
    }

    notes_rows = []
    for _, row in reviewed.iterrows():
        note_text = str(row.get("reviewer_notes", "") or "").lower()
        rec = {
            "STAID": row.get("STAID", ""),
            "human_decision": row.get("human_decision", ""),
            "reviewer_notes": row.get("reviewer_notes", ""),
        }
        for grp, keywords in KEYWORD_GROUPS.items():
            rec[grp] = any(kw in note_text for kw in keywords)
        notes_rows.append(rec)

    notes_df = pd.DataFrame(notes_rows)
    notes_df.to_csv(OUT_DIR / "tables/manual_review_notes_keyword_flags.csv", index=False)
    log("  Wrote tables/manual_review_notes_keyword_flags.csv")

    # ------------------------------------------------------------------
    # 11. Compute candidate rules
    # ------------------------------------------------------------------
    log("Computing candidate rules ...")

    # Helper: quantile on master (full candidate universe with attributes joined in).
    # Must use master here, NOT full_cands -- attribute columns (WATERNLCD06, CANALS_PCT,
    # HYDRO_DISTURB_INDX, etc.) only exist in master after the attribute joins; they are
    # absent from full_cands (metrics-only). Computing thresholds on full_cands would
    # produce None for all attribute-derived rules, silently disabling them.
    def pct(col, q):
        return pct_fn(master, col, q)

    # Precompute thresholds
    thresholds = {
        "WATERNLCD06_p95":               pct("WATERNLCD06", 0.95),
        "lka_pc_use_p95":                pct("lka_pc_use", 0.95),
        "CANALS_PCT_p95":                pct("CANALS_PCT", 0.95),
        "dor_pc_pva_p95":                pct("dor_pc_pva", 0.95),
        "HYDRO_DISTURB_INDX_p95":        pct("HYDRO_DISTURB_INDX", 0.95),
        "DRAIN_SQKM_p90":                pct("DRAIN_SQKM", 0.90),
        "RBI_p25":                       pct("RBI", 0.25),
        "max_hourly_rise_per_km2_p25":   pct("max_hourly_rise_per_km2", 0.25),
        "zero_flow_fraction_p95":        pct("zero_flow_fraction", 0.95),
        "PCT_DIFF_NWIS_p95":             pct("PCT_DIFF_NWIS", 0.95),
        "max_abs_hourly_jump_over_Q50_p99": pct("max_abs_hourly_jump_over_Q50", 0.99),
        "max_hourly_rise_per_km2_p99":   pct("max_hourly_rise_per_km2", 0.99),
    }

    log("  Threshold values:")
    for k, v in thresholds.items():
        log(f"    {k}: {v}")

    # Set of reviewed EXCLUDE STAIDs
    reviewed_exclude_staids = set(
        reviewed.loc[reviewed["human_decision"] == "EXCLUDE", "STAID"].astype(str)
    )
    reviewed_unsure_staids = set(
        reviewed.loc[reviewed["human_decision"] == "UNSURE", "STAID"].astype(str)
    )
    reviewed_klc_staids = set(
        reviewed.loc[reviewed["human_decision"] == "KEEP_LOW_CONFIDENCE", "STAID"].astype(str)
    )

    # ---- Rule A: known_manual_exclude ----
    def apply_rule_A(df):
        return df["STAID"].astype(str).isin(reviewed_exclude_staids)

    # ---- Rule B: metadata_hard_exclude ----
    def apply_rule_B(df):
        if "metadata_policy_bucket" not in df.columns:
            return pd.Series(False, index=df.index)
        return df["metadata_policy_bucket"] == "HARD_EXCLUDE"

    # ---- Rule C: high_lentic_open_water ----
    def apply_rule_C(df):
        t_water = thresholds["WATERNLCD06_p95"]
        t_lka   = thresholds["lka_pc_use_p95"]
        flag = pd.Series(False, index=df.index)
        if t_water is not None and "WATERNLCD06" in df.columns:
            flag = flag | (pd.to_numeric(df["WATERNLCD06"], errors="coerce") >= t_water)
        if t_lka is not None and "lka_pc_use" in df.columns:
            flag = flag | (pd.to_numeric(df["lka_pc_use"], errors="coerce") >= t_lka)
        return flag.fillna(False)

    # ---- Rule D: high_canals_regulated ----
    def apply_rule_D(df):
        t_canal = thresholds["CANALS_PCT_p95"]
        t_dor   = thresholds["dor_pc_pva_p95"]
        flag = pd.Series(False, index=df.index)
        if t_canal is not None and "CANALS_PCT" in df.columns:
            flag = flag | (pd.to_numeric(df["CANALS_PCT"], errors="coerce") >= t_canal)
        if t_dor is not None and "dor_pc_pva" in df.columns:
            flag = flag | (pd.to_numeric(df["dor_pc_pva"], errors="coerce") >= t_dor)
        return flag.fillna(False)

    # ---- Rule E: high_hydro_disturbance ----
    def apply_rule_E(df):
        t = thresholds["HYDRO_DISTURB_INDX_p95"]
        if t is None or "HYDRO_DISTURB_INDX" not in df.columns:
            return pd.Series(False, index=df.index)
        return (pd.to_numeric(df["HYDRO_DISTURB_INDX"], errors="coerce") >= t).fillna(False)

    # ---- Rule F: large_slow_low_flashiness ----
    def apply_rule_F(df):
        t_drain = thresholds["DRAIN_SQKM_p90"]
        t_rbi   = thresholds["RBI_p25"]
        t_rise  = thresholds["max_hourly_rise_per_km2_p25"]
        flag = pd.Series(True, index=df.index)
        if t_drain is not None and "DRAIN_SQKM" in df.columns:
            flag = flag & (pd.to_numeric(df["DRAIN_SQKM"], errors="coerce") >= t_drain)
        else:
            flag = pd.Series(False, index=df.index)
        if t_rbi is not None and "RBI" in df.columns:
            flag = flag & (pd.to_numeric(df["RBI"], errors="coerce") <= t_rbi)
        if t_rise is not None and "max_hourly_rise_per_km2" in df.columns:
            flag = flag & (pd.to_numeric(df["max_hourly_rise_per_km2"], errors="coerce") <= t_rise)
        return flag.fillna(False)

    # ---- Rule G: mostly_zero_suspicious ----
    def apply_rule_G(df):
        t_zff = thresholds["zero_flow_fraction_p95"]
        flag = pd.Series(False, index=df.index)
        if t_zff is not None and "zero_flow_fraction" in df.columns:
            zff = pd.to_numeric(df["zero_flow_fraction"], errors="coerce")
            flag = flag | (zff >= t_zff)
        if "Q50" in df.columns:
            q50 = pd.to_numeric(df["Q50"], errors="coerce")
            flag = flag & (q50 <= 0.001)
        return flag.fillna(False)

    # ---- Rule H: possible_measurement_artifact ----
    def apply_rule_H(df):
        t_jump = thresholds["max_abs_hourly_jump_over_Q50_p99"]
        t_rise = thresholds["max_hourly_rise_per_km2_p99"]
        flag = pd.Series(False, index=df.index)
        if t_jump is not None and "max_abs_hourly_jump_over_Q50" in df.columns:
            flag = flag | (pd.to_numeric(df["max_abs_hourly_jump_over_Q50"], errors="coerce") >= t_jump)
        if t_rise is not None and "max_hourly_rise_per_km2" in df.columns:
            flag = flag | (pd.to_numeric(df["max_hourly_rise_per_km2"], errors="coerce") >= t_rise)
        return flag.fillna(False)

    # ---- Rule I: boundary_uncertain ----
    def apply_rule_I(df):
        t_pct = thresholds["PCT_DIFF_NWIS_p95"]
        flag = pd.Series(False, index=df.index)
        if "BASIN_BOUNDARY_CONFIDENCE" in df.columns:
            flag = flag | (df["BASIN_BOUNDARY_CONFIDENCE"].astype(str).str.strip().str.lower() == "low")
        if t_pct is not None and "PCT_DIFF_NWIS" in df.columns:
            flag = flag | (pd.to_numeric(df["PCT_DIFF_NWIS"], errors="coerce") >= t_pct)
        return flag.fillna(False)

    RULE_APPLY_FNS = {
        "rule_A": apply_rule_A,
        "rule_B": apply_rule_B,
        "rule_C": apply_rule_C,
        "rule_D": apply_rule_D,
        "rule_E": apply_rule_E,
        "rule_F": apply_rule_F,
        "rule_G": apply_rule_G,
        "rule_H": apply_rule_H,
        "rule_I": apply_rule_I,
    }

    # Apply all rules to reviewed and full_cands
    for rname, fn in RULE_APPLY_FNS.items():
        try:
            reviewed[rname] = fn(reviewed).values
            master[rname] = fn(master).values
        except Exception as exc:
            log(f"  WARNING: Error applying {rname}: {exc}")
            reviewed[rname] = False
            master[rname] = False

    # Rule J: compound_risk_score (rules C through I)
    compound_rules = ["rule_C", "rule_D", "rule_E", "rule_F", "rule_G", "rule_H", "rule_I"]
    reviewed["compound_risk_count"] = reviewed[[r for r in compound_rules if r in reviewed.columns]].astype(int).sum(axis=1)
    master["compound_risk_count"] = master[[r for r in compound_rules if r in master.columns]].astype(int).sum(axis=1)

    reviewed["rule_J"] = reviewed["compound_risk_count"] >= 2
    master["rule_J"] = master["compound_risk_count"] >= 2

    log("  Rules applied to reviewed and full candidate tables.")

    # ------------------------------------------------------------------
    # Rule metadata definitions
    # ------------------------------------------------------------------
    def fmt_thresh(keys):
        """Format threshold dict as JSON string for the specified keys."""
        return json.dumps({k: thresholds.get(k) for k in keys}, default=str)

    rules_meta = [
        {
            "rule_id": "rule_A",
            "plain_language_description": "Basin is in the manual review EXCLUDE set",
            "exact_boolean_expression": "STAID in reviewed_exclude_staids",
            "threshold_values_used": "{}",
            "recommendation": "EXCLUDE_TRAINING_PRELIM",
            "caution_notes": "Only applies to the 73 reviewed basins; all other basins are False",
        },
        {
            "rule_id": "rule_B",
            "plain_language_description": "metadata_policy_bucket == HARD_EXCLUDE",
            "exact_boolean_expression": "metadata_policy_bucket == 'HARD_EXCLUDE'",
            "threshold_values_used": "{}",
            "recommendation": "EXCLUDE_TRAINING_PRELIM",
            "caution_notes": "Should be 0 in main_training_candidate universe (all are ACCEPT)",
        },
        {
            "rule_id": "rule_C",
            "plain_language_description": "High open-water or lake area (lentic proxy)",
            "exact_boolean_expression": (
                "WATERNLCD06 >= p95(WATERNLCD06) OR lka_pc_use >= p95(lka_pc_use)"
            ),
            "threshold_values_used": fmt_thresh(["WATERNLCD06_p95", "lka_pc_use_p95"]),
            "recommendation": "HOLDOUT_REVIEW_CANDIDATE",
            "caution_notes": (
                "Proxy for lentic; HIRES_LENTIC_PCT not available in GAGES-II. "
                "Valid natural lakes included. Use WATERNLCD06 (LC06_Basin) and lka_pc_use (hydroATLAS)."
            ),
        },
        {
            "rule_id": "rule_D",
            "plain_language_description": "High canals or degree of regulation (regulated proxy)",
            "exact_boolean_expression": (
                "CANALS_PCT >= p95(CANALS_PCT) OR dor_pc_pva >= p95(dor_pc_pva)"
            ),
            "threshold_values_used": fmt_thresh(["CANALS_PCT_p95", "dor_pc_pva_p95"]),
            "recommendation": "HOLDOUT_REVIEW_CANDIDATE",
            "caution_notes": (
                "CANALS_PCT is proxy for ARTIFPATH_PCT (not in GAGES-II). "
                "dor_pc_pva from hydroATLAS. Check against manual labels before excluding."
            ),
        },
        {
            "rule_id": "rule_E",
            "plain_language_description": "High hydrologic disturbance index (composite)",
            "exact_boolean_expression": "HYDRO_DISTURB_INDX >= p95(HYDRO_DISTURB_INDX)",
            "threshold_values_used": fmt_thresh(["HYDRO_DISTURB_INDX_p95"]),
            "recommendation": "HOLDOUT_REVIEW_CANDIDATE",
            "caution_notes": (
                "HYDRO_DISTURB_INDX is a composite index; high values likely indicate regulation or disturbance."
            ),
        },
        {
            "rule_id": "rule_F",
            "plain_language_description": "Large, slow, low-flashiness basin",
            "exact_boolean_expression": (
                "DRAIN_SQKM >= p90(DRAIN_SQKM) AND RBI <= p25(RBI) AND "
                "max_hourly_rise_per_km2 <= p25(max_hourly_rise_per_km2)"
            ),
            "threshold_values_used": fmt_thresh([
                "DRAIN_SQKM_p90", "RBI_p25", "max_hourly_rise_per_km2_p25"
            ]),
            "recommendation": "SOFT_RISK_FLAG",
            "caution_notes": (
                "Large slow basins may be perfectly valid for training. "
                "Do not auto-exclude; use as a soft flag only."
            ),
        },
        {
            "rule_id": "rule_G",
            "plain_language_description": "High zero-flow fraction and near-zero median flow",
            "exact_boolean_expression": (
                "zero_flow_fraction >= p95(zero_flow_fraction) AND Q50 <= 0.001"
            ),
            "threshold_values_used": fmt_thresh(["zero_flow_fraction_p95"]),
            "recommendation": "HOLDOUT_REVIEW_CANDIDATE",
            "caution_notes": "Intermittent and ephemeral streams can be valid; check region and basin characteristics.",
        },
        {
            "rule_id": "rule_H",
            "plain_language_description": "Possible measurement artifact (extreme hourly jump or rise)",
            "exact_boolean_expression": (
                "max_abs_hourly_jump_over_Q50 >= p99(max_abs_hourly_jump_over_Q50) OR "
                "max_hourly_rise_per_km2 >= p99(max_hourly_rise_per_km2)"
            ),
            "threshold_values_used": fmt_thresh([
                "max_abs_hourly_jump_over_Q50_p99", "max_hourly_rise_per_km2_p99"
            ]),
            "recommendation": "HOLDOUT_REVIEW_CANDIDATE",
            "caution_notes": (
                "Extreme flash events can be real. "
                "Cross-check with spike flags and manual review labels before excluding."
            ),
        },
        {
            "rule_id": "rule_I",
            "plain_language_description": "Uncertain basin boundary (low confidence or large NWIS discrepancy)",
            "exact_boolean_expression": (
                "BASIN_BOUNDARY_CONFIDENCE == 'Low' OR PCT_DIFF_NWIS >= p95(PCT_DIFF_NWIS)"
            ),
            "threshold_values_used": fmt_thresh(["PCT_DIFF_NWIS_p95"]),
            "recommendation": "SOFT_RISK_FLAG",
            "caution_notes": "Boundary uncertainty alone is not an exclusion criterion.",
        },
        {
            "rule_id": "rule_J",
            "plain_language_description": "Compound risk: 2 or more moderate rules (C through I) trigger",
            "exact_boolean_expression": "count(rule_C through rule_I that trigger) >= 2",
            "threshold_values_used": "{}",
            "recommendation": "HOLDOUT_REVIEW_CANDIDATE",
            "caution_notes": "Compound risk; requires manual verification before exclusion.",
        },
    ]

    # ------------------------------------------------------------------
    # 12. Rule screening table
    # ------------------------------------------------------------------
    log("Writing rule screening table ...")

    rule_rows = []
    for meta in rules_meta:
        rid = meta["rule_id"]
        rs = rule_stats(reviewed, rid)
        fs = full_stats(master, rid)
        row = {**meta, **rs, **fs}
        rule_rows.append(row)

    rule_df = pd.DataFrame(rule_rows)
    # Reorder columns
    col_order = [
        "rule_id", "plain_language_description", "exact_boolean_expression",
        "threshold_values_used", "recommendation", "caution_notes",
        "reviewed_flagged_n", "reviewed_exclude_n", "reviewed_unsure_n",
        "reviewed_keep_n", "reviewed_keep_low_confidence_n",
        "reviewed_exclude_or_unsure_rate", "reviewed_false_positive_keep_rate",
        "full_candidate_flagged_n", "full_candidate_flagged_pct",
    ]
    rule_df = rule_df[[c for c in col_order if c in rule_df.columns]]
    rule_df.to_csv(OUT_DIR / "tables/candidate_rule_screening.csv", index=False)
    log("  Wrote tables/candidate_rule_screening.csv")

    # ------------------------------------------------------------------
    # 13. Reviewed basin rule matrix
    # ------------------------------------------------------------------
    log("Writing reviewed basin rule matrix ...")

    rule_cols = [f"rule_{x}" for x in list("ABCDEFGHIJ")]
    base_cols = [
        "STAID", "human_decision", "hydrograph_behavior", "artifact_type",
        "confidence", "reviewer_notes",
        "candidate_class", "review_group",
    ]
    metric_cols = [
        "DRAIN_SQKM", "BFI_AVE", "RBI", "hourly_completeness_pct", "zero_flow_fraction",
        "Q50", "Q95", "Q99", "q95_q50_ratio", "max_hourly_rise_per_km2",
        "max_abs_hourly_jump_over_Q50",
    ]
    optional_cols = [
        "HYDRO_DISTURB_INDX", "WATERNLCD06", "CANALS_PCT",
        "lka_pc_use", "dor_pc_pva", "BASIN_BOUNDARY_CONFIDENCE", "PCT_DIFF_NWIS",
    ]
    all_wanted = (
        base_cols
        + [r for r in rule_cols if r in reviewed.columns]
        + ["compound_risk_count"]
        + [c for c in metric_cols if c in reviewed.columns]
        + [c for c in optional_cols if c in reviewed.columns]
    )
    rev_matrix = reviewed[[c for c in all_wanted if c in reviewed.columns]].copy()
    rev_matrix.to_csv(OUT_DIR / "tables/reviewed_basin_rule_matrix.csv", index=False)
    log("  Wrote tables/reviewed_basin_rule_matrix.csv")

    # ------------------------------------------------------------------
    # 14. Full candidate rule matrix + preliminary status
    # ------------------------------------------------------------------
    log("Writing full candidate rule matrix ...")

    def assign_prelim_status(row):
        """Assign preliminary training status with priority: EXCLUDE > HOLDOUT > SOFT_KEEP > CORE."""
        # Rule A: reviewed EXCLUDE
        if row.get("rule_A", False):
            return "EXCLUDE_TRAINING_PRELIM"
        # Rule B: hard exclude in metadata (should be 0 in this universe)
        if row.get("rule_B", False):
            return "EXCLUDE_TRAINING_PRELIM"
        # Reviewed UNSURE -> HOLDOUT
        if str(row.get("STAID", "")) in reviewed_unsure_staids:
            return "HOLDOUT_REVIEW_PRELIM"
        # Compound risk >= 2 or rule_G or rule_H
        if row.get("compound_risk_count", 0) >= 2:
            return "HOLDOUT_REVIEW_PRELIM"
        if row.get("rule_G", False) or row.get("rule_H", False):
            return "HOLDOUT_REVIEW_PRELIM"
        # Reviewed KEEP_LOW_CONFIDENCE or single moderate risk
        if str(row.get("STAID", "")) in reviewed_klc_staids:
            return "TRAIN_SOFT_KEEP_PRELIM"
        for r in ["rule_C", "rule_D", "rule_E", "rule_F", "rule_I"]:
            if row.get(r, False):
                return "TRAIN_SOFT_KEEP_PRELIM"
        return "TRAIN_CORE_PRELIM"

    master["preliminary_training_status"] = master.apply(assign_prelim_status, axis=1)
    master["status_is_preliminary"] = True

    log("  Preliminary status value counts:")
    for val, cnt in master["preliminary_training_status"].value_counts().items():
        log(f"    {val}: {cnt}")

    full_matrix_base = [
        "STAID", "candidate_class", "metadata_policy_bucket", "main_training_candidate",
    ]
    full_matrix_cols = (
        full_matrix_base
        + [r for r in rule_cols if r in master.columns]
        + ["compound_risk_count", "preliminary_training_status", "status_is_preliminary"]
        + [c for c in metric_cols if c in master.columns]
        + [c for c in optional_cols if c in master.columns]
        + ["LAT_GAGE", "LNG_GAGE", "STATE", "HUC02", "area_bin"]
    )
    full_matrix_cols_present = [c for c in full_matrix_cols if c in master.columns]
    # Deduplicate while preserving order
    seen = set()
    full_matrix_cols_dedup = []
    for c in full_matrix_cols_present:
        if c not in seen:
            seen.add(c)
            full_matrix_cols_dedup.append(c)

    full_matrix = master[full_matrix_cols_dedup].copy()
    full_matrix.to_csv(OUT_DIR / "tables/full_candidate_rule_matrix.csv", index=False)
    log("  Wrote tables/full_candidate_rule_matrix.csv")

    # ------------------------------------------------------------------
    # 15. PLOTS
    # ------------------------------------------------------------------
    log("Generating plots ...")
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 120,
    })

    # ---- 15.1 Manual decision counts bar chart ----
    try:
        dec_counts = reviewed["human_decision"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(
            dec_counts.index,
            dec_counts.values,
            color=[DECISION_COLORS.get(d, "steelblue") for d in dec_counts.index],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, dec_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Human Decision")
        ax.set_ylabel("Count")
        ax.set_title(f"Manual Review Decision Counts (N={len(reviewed)})")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "plots/manual_decision_counts.png", dpi=120)
        plt.close(fig)
        log("  Wrote plots/manual_decision_counts.png")
    except Exception as exc:
        log(f"  WARNING: plot manual_decision_counts failed: {exc}")

    # ---- 15.2 By candidate_class (stacked hbar) ----
    try:
        if "candidate_class" in reviewed.columns:
            ct = (
                reviewed.groupby("candidate_class")["human_decision"]
                .value_counts()
                .unstack(fill_value=0)
            )
            # Keep only decision columns
            dec_cols = [d for d in DECISION_COLORS if d in ct.columns]
            ct = ct[dec_cols]
            stacked_hbar(
                ct,
                title=f"Manual Decision by Candidate Class (N={len(reviewed)})",
                out_path=OUT_DIR / "plots/manual_decision_by_candidate_class.png",
                color_map=DECISION_COLORS,
            )
            log("  Wrote plots/manual_decision_by_candidate_class.png")
    except Exception as exc:
        log(f"  WARNING: plot by_candidate_class failed: {exc}")

    # ---- 15.3 By review_group (stacked hbar) ----
    try:
        if "review_group" in reviewed.columns:
            ct = (
                reviewed.groupby("review_group")["human_decision"]
                .value_counts()
                .unstack(fill_value=0)
            )
            dec_cols = [d for d in DECISION_COLORS if d in ct.columns]
            ct = ct[dec_cols]
            stacked_hbar(
                ct,
                title=f"Manual Decision by Review Group (N={len(reviewed)})",
                out_path=OUT_DIR / "plots/manual_decision_by_review_group.png",
                color_map=DECISION_COLORS,
            )
            log("  Wrote plots/manual_decision_by_review_group.png")
    except Exception as exc:
        log(f"  WARNING: plot by_review_group failed: {exc}")

    # ---- 15.4 Attribute distributions by decision ----
    LOG_SCALE_VARS = {
        "DRAIN_SQKM", "max_hourly_rise_per_km2", "max_abs_hourly_jump_over_Q50",
        "HYDRO_DISTURB_INDX",
    }
    attr_plot_vars = [
        "DRAIN_SQKM", "BFI_AVE", "RBI", "zero_flow_fraction",
        "max_hourly_rise_per_km2", "max_abs_hourly_jump_over_Q50",
        "HYDRO_DISTURB_INDX", "WATERNLCD06",
    ]
    for var in attr_plot_vars:
        try:
            if var not in reviewed.columns:
                continue
            use_log = var in LOG_SCALE_VARS
            safe_var = var.replace("/", "_").replace(" ", "_")
            attr_boxplot(
                reviewed,
                var=var,
                label_col="human_decision",
                out_path=OUT_DIR / f"plots/attribute_distributions_by_decision_{safe_var}.png",
                log_scale=use_log,
            )
            log(f"  Wrote plots/attribute_distributions_by_decision_{safe_var}.png")
        except Exception as exc:
            log(f"  WARNING: attr plot {var} failed: {exc}")

    # ---- 15.5 Rule capture summary (flagged reviewed basins by decision) ----
    try:
        all_rule_ids = [f"rule_{x}" for x in list("ABCDEFGHIJ")]
        rule_ids_present = [r for r in all_rule_ids if r in reviewed.columns]

        rows_r = []
        for rid in rule_ids_present:
            flagged = reviewed[reviewed[rid] == True]
            vc = flagged["human_decision"].value_counts() if len(flagged) > 0 else pd.Series(dtype=int)
            row = {d: int(vc.get(d, 0)) for d in DECISION_COLORS}
            row["rule_id"] = rid
            rows_r.append(row)

        if rows_r:
            rc_df = pd.DataFrame(rows_r).set_index("rule_id")
            dec_cols = [d for d in DECISION_COLORS if d in rc_df.columns]
            rc_df = rc_df[dec_cols]
            stacked_hbar(
                rc_df,
                title="Rule Capture: Flagged Reviewed Basins by Decision",
                out_path=OUT_DIR / "plots/rule_capture_summary.png",
                color_map=DECISION_COLORS,
            )
            log("  Wrote plots/rule_capture_summary.png")
    except Exception as exc:
        log(f"  WARNING: rule_capture_summary plot failed: {exc}")

    # ---- 15.6 Full candidate rule counts ----
    try:
        rule_ids_master = [r for r in all_rule_ids if r in master.columns]
        flagged_counts = {r: int(master[r].sum()) for r in rule_ids_master}
        n_total = len(master)

        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(
            list(flagged_counts.keys()),
            list(flagged_counts.values()),
            color="steelblue", edgecolor="white",
        )
        for bar, (rid, cnt) in zip(bars, flagged_counts.items()):
            pct_val = cnt / n_total * 100 if n_total > 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{cnt}\n({pct_val:.1f}%)", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Rule ID")
        ax.set_ylabel("Count of Flagged Basins")
        ax.set_title(f"Full Candidate Basins Flagged per Rule (N_total={n_total})")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "plots/full_candidate_rule_counts.png", dpi=120)
        plt.close(fig)
        log("  Wrote plots/full_candidate_rule_counts.png")
    except Exception as exc:
        log(f"  WARNING: full_candidate_rule_counts plot failed: {exc}")

    # ---- 15.7 Preliminary training status counts ----
    try:
        if "preliminary_training_status" in master.columns:
            status_counts = master["preliminary_training_status"].value_counts()
            STATUS_COLORS = {
                "TRAIN_CORE_PRELIM":      "green",
                "TRAIN_SOFT_KEEP_PRELIM": "orange",
                "HOLDOUT_REVIEW_PRELIM":  "steelblue",
                "EXCLUDE_TRAINING_PRELIM": "red",
            }
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(
                status_counts.index,
                status_counts.values,
                color=[STATUS_COLORS.get(s, "gray") for s in status_counts.index],
                edgecolor="white",
            )
            for bar, val in zip(bars, status_counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        str(val), ha="center", va="bottom", fontsize=9)
            ax.set_xlabel("Preliminary Training Status")
            ax.set_ylabel("Count")
            ax.set_title(f"Preliminary Training Status (N={len(master)}) -- PRELIM ONLY")
            ax.tick_params(axis="x", labelsize=8)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            fig.savefig(OUT_DIR / "plots/preliminary_training_status_counts.png", dpi=120)
            plt.close(fig)
            log("  Wrote plots/preliminary_training_status_counts.png")
    except Exception as exc:
        log(f"  WARNING: prelim status plot failed: {exc}")

    # ---- 15.8 Map by preliminary training status ----
    try:
        if "LAT_GAGE" in master.columns and "LNG_GAGE" in master.columns:
            STATUS_COLORS_MAP = {
                "TRAIN_CORE_PRELIM":      "green",
                "TRAIN_SOFT_KEEP_PRELIM": "orange",
                "HOLDOUT_REVIEW_PRELIM":  "steelblue",
                "EXCLUDE_TRAINING_PRELIM": "red",
            }
            fig, ax = plt.subplots(figsize=(13, 6))
            for status, color in STATUS_COLORS_MAP.items():
                sub = master[master["preliminary_training_status"] == status]
                if len(sub) == 0:
                    continue
                ax.scatter(
                    pd.to_numeric(sub["LNG_GAGE"], errors="coerce"),
                    pd.to_numeric(sub["LAT_GAGE"], errors="coerce"),
                    c=color, s=10, alpha=0.55, label=f"{status} (n={len(sub)})",
                    linewidths=0,
                )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"Preliminary Training Status Map (N={len(master)}) -- PRELIM ONLY")
            ax.legend(fontsize=7, markerscale=2, loc="lower left")
            # Approximate CONUS bounds
            ax.set_xlim(-128, -65)
            ax.set_ylim(24, 50)
            plt.tight_layout()
            fig.savefig(OUT_DIR / "plots/map_preliminary_training_status.png", dpi=120)
            plt.close(fig)
            log("  Wrote plots/map_preliminary_training_status.png")
    except Exception as exc:
        log(f"  WARNING: map plot failed: {exc}")

    # ------------------------------------------------------------------
    # 16. Summary markdown
    # ------------------------------------------------------------------
    log("Writing summaries ...")

    # Pull key numbers
    n_reviewed   = len(reviewed)
    n_full_cands = len(master)
    n_total_raw  = len(metrics_df)
    dec_vc       = reviewed["human_decision"].value_counts().to_dict()
    n_keep       = dec_vc.get("KEEP", 0)
    n_excl       = dec_vc.get("EXCLUDE", 0)
    n_klc        = dec_vc.get("KEEP_LOW_CONFIDENCE", 0)
    n_uns        = dec_vc.get("UNSURE", 0)

    status_vc_md = master["preliminary_training_status"].value_counts().to_dict() if "preliminary_training_status" in master.columns else {}

    # Build markdown rule table
    rule_table_lines = [
        "| rule_id | description | reviewed_flagged | excl_or_unsure_rate | full_flagged | recommendation |",
        "|---------|-------------|-----------------|---------------------|--------------|----------------|",
    ]
    for row in rule_rows:
        eou = row.get("reviewed_exclude_or_unsure_rate", "")
        eou_str = f"{eou:.2f}" if isinstance(eou, float) and not (eou != eou) else "N/A"
        rule_table_lines.append(
            f"| {row['rule_id']} | {row['plain_language_description'][:60]} | "
            f"{row['reviewed_flagged_n']} | {eou_str} | "
            f"{row['full_candidate_flagged_n']} ({row['full_candidate_flagged_pct']}%) | "
            f"{row['recommendation']} |"
        )
    rule_table_md = "\n".join(rule_table_lines)

    attr_assoc_lines = []
    if len(attr_summary_df) > 0:
        keep_means = attr_summary_df[attr_summary_df["human_decision"] == "KEEP"].set_index("variable")["mean"]
        excl_means = attr_summary_df[attr_summary_df["human_decision"] == "EXCLUDE"].set_index("variable")["mean"]
        common = keep_means.index.intersection(excl_means.index)
        diffs = []
        for v in common:
            km = keep_means[v]
            em = excl_means[v]
            if pd.notna(km) and pd.notna(em) and km != 0:
                diffs.append((v, km, em, abs(em - km) / (abs(km) + 1e-12)))
        diffs.sort(key=lambda x: x[3], reverse=True)
        for v, km, em, ratio in diffs[:8]:
            attr_assoc_lines.append(
                f"- {v}: KEEP mean={km:.4g}, EXCLUDE mean={em:.4g} (relative diff={ratio:.2f})"
            )

    md_content = f"""# Manual Review Rule Analysis -- Flash-NH Basin Screening

> PRELIMINARY ANALYSIS ONLY. Output statuses (EXCLUDE_TRAINING_PRELIM, etc.) are risk flags
> for human review, NOT final training decisions.

## 1. Executive Summary

- Total USGS candidate basins in metrics file: {n_total_raw}
- Basins with main_training_candidate == True: {n_full_cands}
- Basins reviewed manually: {n_reviewed}
- Manual review decisions: KEEP={n_keep}, EXCLUDE={n_excl}, KEEP_LOW_CONFIDENCE={n_klc}, UNSURE={n_uns}
- Preliminary status across {n_full_cands} candidates:
"""
    for s, c in status_vc_md.items():
        md_content += f"  - {s}: {c} ({c / n_full_cands * 100:.1f}%)\n"

    md_content += f"""
## 2. Input Lineage and Row-Count Audit

See `tables/input_join_audit.csv` for full details.

- LABELS_CSV: {LABELS_CSV}
- TEMPLATE_CSV: {TEMPLATE_CSV}
- METRICS_CSV: {METRICS_CSV}
- ATTR_DIR: {ATTR_DIR}

Proxy substitutions used (GAGES-II does not include ARTIFPATH_PCT, HIRES_LENTIC_PCT):
- WATERNLCD06 (open water % in basin, LC06_Basin file) -> lentic proxy
- CANALS_PCT (canals %, HydroMod_Other) -> artificial path proxy
- dor_pc_pva (degree of regulation, hydroATLAS) -> regulated storage proxy
- lka_pc_use (lake area %, hydroATLAS) -> additional lentic proxy

## 3. Manual Review Findings

Decision breakdown (N={n_reviewed}):
- KEEP: {n_keep} ({n_keep/n_reviewed*100:.1f}%)
- EXCLUDE: {n_excl} ({n_excl/n_reviewed*100:.1f}%)
- KEEP_LOW_CONFIDENCE: {n_klc} ({n_klc/n_reviewed*100:.1f}%)
- UNSURE: {n_uns} ({n_uns/n_reviewed*100:.1f}%)

See `tables/manual_review_label_summary.csv` for full breakdowns by
hydrograph_behavior, artifact_type, and confidence.

Key concerns from reviewer notes (see `tables/manual_review_notes_keyword_flags.csv`):
- Regulation/dams/lake patterns identified in notes
- Artifact/sensor noise flags
- Mostly-zero or ephemeral behavior

## 4. Attribute Associations

Variables most different between KEEP and EXCLUDE groups:

{chr(10).join(attr_assoc_lines) if attr_assoc_lines else "- See tables/manual_review_attribute_summary.csv"}

See `tables/manual_review_attribute_summary.csv` for full per-decision statistics.

## 5. Candidate Rules

{rule_table_md}

Full rule details in `tables/candidate_rule_screening.csv`.

## 6. Recommended Policy

Conservative approach:
1. Hard exclusion: Only rule_A (confirmed EXCLUDE from manual review) or rule_B (hard metadata flag)
   should trigger EXCLUDE_TRAINING_PRELIM. These are high-confidence exclusions.
2. Holdout/review: Rules C, D, E, G, H, J flag basins for a second pass of human review
   before a final decision is made. These are risk signals, not automated exclusions.
3. Soft keep: Rules F and I (large/slow basin; boundary uncertainty) are weak signals only.
   Basins flagged by these alone should NOT be excluded.
4. This analysis is NOT a classifier. All EXCLUDE/HOLDOUT flags must be verified by a human
   before removing basins from training data.

## 7. Limitations

- Only {n_reviewed} basins reviewed; rules generalise with high uncertainty.
- Quantile thresholds computed on {n_full_cands} candidates; may shift if data is updated.
- Proxy variables (WATERNLCD06, CANALS_PCT, dor_pc_pva) are imperfect substitutes.
- Compound risk score (rule_J) counts rules, not severity; a basin flagged by two weak rules
  receives the same score as one flagged by two strong rules.
- BASIN_BOUNDARY_CONFIDENCE is categorical; parsing depends on exact string values in GAGES-II.

## 8. Next Steps

1. Review all HOLDOUT_REVIEW_PRELIM basins (N={status_vc_md.get('HOLDOUT_REVIEW_PRELIM', '?')}) with a human.
2. Expand manual review sample to reduce rule uncertainty.
3. Consider region-specific thresholds (e.g., arid western basins have naturally high zero-flow).
4. Validate proxy variables against direct measurements where available.
5. Do not use preliminary_training_status as a final label without human sign-off.

---
Generated by: analyze_manual_review_filter_rules.py
Analysis date: 2026-05-26
Status: PRELIMINARY ONLY -- DO NOT USE AS FINAL TRAINING DECISIONS
"""
    with open(OUT_DIR / "summaries/manual_review_rule_analysis_summary.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    log("  Wrote summaries/manual_review_rule_analysis_summary.md")

    # ------------------------------------------------------------------
    # 17. Summary JSON
    # ------------------------------------------------------------------
    summary_json = {
        "analysis_note": "PRELIMINARY ONLY -- DO NOT USE AS FINAL TRAINING DECISIONS",
        "input_counts": {
            "total_metrics_rows": n_total_raw,
            "main_training_candidate_true": n_full_cands,
            "reviewed_basins": n_reviewed,
        },
        "manual_review_decisions": dec_vc,
        "preliminary_status_counts": status_vc_md,
        "rule_results": [
            {
                "rule_id": r["rule_id"],
                "recommendation": r["recommendation"],
                "reviewed_flagged_n": r["reviewed_flagged_n"],
                "reviewed_exclude_or_unsure_rate": (
                    r["reviewed_exclude_or_unsure_rate"]
                    if not (isinstance(r["reviewed_exclude_or_unsure_rate"], float)
                            and r["reviewed_exclude_or_unsure_rate"] != r["reviewed_exclude_or_unsure_rate"])
                    else None
                ),
                "full_candidate_flagged_n": r["full_candidate_flagged_n"],
                "full_candidate_flagged_pct": r["full_candidate_flagged_pct"],
            }
            for r in rule_rows
        ],
        "thresholds": {k: (v if v is not None else None) for k, v in thresholds.items()},
        "proxy_substitutions": PROXY_NOTES,
        "outputs": {
            "tables": [
                "input_join_audit.csv",
                "manual_review_label_summary.csv",
                "manual_review_by_candidate_class.csv",
                "manual_review_by_review_group.csv",
                "manual_review_by_area_bfi_huc.csv",
                "manual_review_attribute_summary.csv",
                "manual_review_notes_keyword_flags.csv",
                "candidate_rule_screening.csv",
                "reviewed_basin_rule_matrix.csv",
                "full_candidate_rule_matrix.csv",
            ],
            "plots": [
                "manual_decision_counts.png",
                "manual_decision_by_candidate_class.png",
                "manual_decision_by_review_group.png",
                "attribute_distributions_by_decision_*.png",
                "rule_capture_summary.png",
                "full_candidate_rule_counts.png",
                "preliminary_training_status_counts.png",
                "map_preliminary_training_status.png",
            ],
            "summaries": [
                "manual_review_rule_analysis_summary.md",
                "manual_review_rule_analysis_summary.json",
            ],
        },
    }
    with open(OUT_DIR / "summaries/manual_review_rule_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, default=str)
    log("  Wrote summaries/manual_review_rule_analysis_summary.json")

    # ------------------------------------------------------------------
    # 18. Write run log
    # ------------------------------------------------------------------
    log(f"All outputs written to: {OUT_DIR}")
    log("DONE. PRELIMINARY ANALYSIS ONLY -- DO NOT USE STATUS COLUMNS AS FINAL TRAINING DECISIONS.")

    log_path = OUT_DIR / "logs/run_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(run_log))
    print(f"Run log written to: {log_path}")


if __name__ == "__main__":
    main()
