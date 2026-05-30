#!/usr/bin/env python3
"""
Build the final pre-training basin selection / status table for Flash-NH.

Encodes the accepted policy from the combined manual review analysis
(flashnh_combined_manual_review_analysis_v001) into a versioned, reproducible
basin status table.

Inputs
------
  reports/flashnh_combined_manual_review_analysis_v001/tables/preliminary_final_policy_candidates.csv
  reports/flashnh_combined_manual_review_analysis_v001/tables/combined_manual_labels.csv
  reports/flashnh_manual_review_rule_analysis_v001/tables/full_candidate_rule_matrix.csv
  reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv

Output folder
-------------
  reports/flashnh_final_basin_selection_v001/

Usage
-----
    python scripts/build_final_basin_training_status.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

POLICY_CANDIDATES_CSV = (
    REPO_ROOT / "reports/flashnh_combined_manual_review_analysis_v001"
    / "tables/preliminary_final_policy_candidates.csv"
)
COMBINED_LABELS_CSV = (
    REPO_ROOT / "reports/flashnh_combined_manual_review_analysis_v001"
    / "tables/combined_manual_labels.csv"
)
RULE_MATRIX_CSV = (
    REPO_ROOT / "reports/flashnh_manual_review_rule_analysis_v001"
    / "tables/full_candidate_rule_matrix.csv"
)
METRICS_META_CSV = (
    REPO_ROOT / "reports/flashnh_usgs_site_metadata_v001"
    / "tables/wy2024_metrics_with_site_metadata.csv"
)
ANALYSIS_SUMMARY_MD = (
    REPO_ROOT / "reports/flashnh_combined_manual_review_analysis_v001"
    / "summaries/combined_manual_review_analysis_summary.md"
)

OUT_DIR     = REPO_ROOT / "reports/flashnh_final_basin_selection_v001"
TABLES_DIR  = OUT_DIR / "tables"
PLOTS_DIR   = OUT_DIR / "plots"
SUMMARY_DIR = OUT_DIR / "summaries"

for d in (TABLES_DIR, PLOTS_DIR, SUMMARY_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RULE_COLS   = [f"rule_{x}" for x in "ABCDEFGHIJ"]
CDEJ_COLS   = ["rule_C", "rule_D", "rule_E", "rule_J"]

FINAL_STATUS_ORDER = [
    "TRAIN_CORE",
    "TRAIN_SOFT_KEEP",
    "HOLDOUT_REVIEW",
    "EXCLUDE_TRAINING",
]
STATUS_COLORS = {
    "TRAIN_CORE":        "#2ca02c",
    "TRAIN_SOFT_KEEP":   "#1f77b4",
    "HOLDOUT_REVIEW":    "#ff7f0e",
    "EXCLUDE_TRAINING":  "#d62728",
}
STATUS_FULL = {
    "TRAIN_CORE":        "TRAIN_CORE",
    "TRAIN_SOFT_KEEP":   "TRAIN_SOFT_KEEP",
    "HOLDOUT_REVIEW":    "HOLDOUT_REVIEW",
    "EXCLUDE_TRAINING":  "EXCLUDE_TRAINING",
}

AREA_BIN_ORDER = ["<10 km2", "10-100 km2", "100-1000 km2", ">=1000 km2"]
BFI_BIN_ORDER  = ["<=20", "20-30", "30-40", "40-50", ">50"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_true(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, float):
        return False if np.isnan(v) else bool(v)
    if isinstance(v, int):
        return bool(v)
    return str(v).strip().upper() in ("TRUE", "1", "YES", "T")


def _area_bin(a) -> str:
    try:
        a = float(a)
        if a < 10:    return "<10 km2"
        if a < 100:   return "10-100 km2"
        if a < 1000:  return "100-1000 km2"
        return ">=1000 km2"
    except Exception:
        return "unknown"


def _bfi_bin(b) -> str:
    try:
        b = float(b)
        if b <= 20: return "<=20"
        if b <= 30: return "20-30"
        if b <= 40: return "30-40"
        if b <= 50: return "40-50"
        return ">50"
    except Exception:
        return "unknown"


def _save(fig, fname):
    path = PLOTS_DIR / fname
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved plot: {fname}")


def _cross_tab(df, col, status_col="final_training_status"):
    """Pivot: index=col, columns=status, values=count."""
    ct = (df.groupby([col, status_col]).size()
            .reset_index(name="n")
            .pivot(index=col, columns=status_col, values="n")
            .fillna(0).astype(int))
    for s in FINAL_STATUS_ORDER:
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[FINAL_STATUS_ORDER]
    ct["total"] = ct.sum(axis=1)
    return ct.reset_index()


def _stacked_bar(ax, ct_df, group_col, title, x_order=None):
    groups = x_order if x_order is not None else ct_df[group_col].tolist()
    ct_idx = ct_df.set_index(group_col)
    x = np.arange(len(groups))
    bot = np.zeros(len(groups))
    for st in FINAL_STATUS_ORDER:
        vals = np.array([ct_idx.loc[g, st] if g in ct_idx.index else 0 for g in groups])
        ax.bar(x, vals, bottom=bot, color=STATUS_COLORS[st], label=st, width=0.7)
        bot += vals
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Basin count")
    ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# 1. Load inputs
# ---------------------------------------------------------------------------
print("\n-- 1. Loading inputs ---------------------------------------------------")

pc = pd.read_csv(POLICY_CANDIDATES_CSV, dtype={"STAID": str})
cl = pd.read_csv(COMBINED_LABELS_CSV,   dtype={"STAID": str})
rm = pd.read_csv(RULE_MATRIX_CSV,       dtype={"STAID": str})
mm = pd.read_csv(METRICS_META_CSV,      dtype={"STAID": str})

print(f"  Policy candidates:   {len(pc)} rows")
print(f"  Combined labels:     {len(cl)} rows (148 reviewed)")
print(f"  Rule matrix:         {len(rm)} rows")
print(f"  Metrics + metadata:  {len(mm)} rows")

# Validate entry count
assert len(pc) == 3034, f"Expected 3034 policy candidate rows, got {len(pc)}"
print(f"  Confirmed: policy_candidates == 3,034")

# Filter metrics to main_training_candidate only (for validation)
mm_mtc = mm[mm["main_training_candidate"].astype(str).str.strip().str.upper()
              .isin({"TRUE", "1", "YES"})].copy()
print(f"  Metrics main_training_candidate == True: {len(mm_mtc)} rows")

# ---------------------------------------------------------------------------
# 2. Build attribute join
# ---------------------------------------------------------------------------
print("\n-- 2. Joining attributes -----------------------------------------------")

# Rule-matrix attributes missing from policy candidates
RM_ATTR_COLS = [
    "STAID", "BFI_AVE", "zero_flow_fraction", "q95_q50_ratio",
    "max_hourly_rise_per_km2", "max_abs_hourly_jump_over_Q50",
    "HYDRO_DISTURB_INDX", "WATERNLCD06", "CANALS_PCT",
    "lka_pc_use", "dor_pc_pva",
]
rm_attrs = rm[[c for c in RM_ATTR_COLS if c in rm.columns]].copy()

# Metrics-only attributes
MM_ATTR_COLS = [
    "STAID", "site_type_code", "site_type_name",
]
mm_attrs = mm[[c for c in MM_ATTR_COLS if c in mm.columns]].copy()

# Start from policy candidates
df = pc.copy()

# Drop any rm cols already in df (keep rule flags from policy candidates)
rm_drop = [c for c in rm_attrs.columns if c != "STAID" and c in df.columns]
df = df.drop(columns=rm_drop)
df = df.merge(rm_attrs, on="STAID", how="left")

# Drop any mm cols already in df
mm_drop = [c for c in mm_attrs.columns if c != "STAID" and c in df.columns]
df = df.drop(columns=mm_drop)
df = df.merge(mm_attrs, on="STAID", how="left")

# Compute area_bin and BFI_bin fresh (avoid special chars from external files)
df["area_bin"] = df["DRAIN_SQKM"].apply(_area_bin)
df["bfi_bin"]  = df["BFI_AVE"].apply(_bfi_bin)

print(f"  Join result: {len(df)} rows, {len(df.columns)} columns")

# ---------------------------------------------------------------------------
# 3. Apply final policy
# ---------------------------------------------------------------------------
print("\n-- 3. Applying final policy ---------------------------------------------")

final_statuses      = []
excl_reasons        = []
holdout_reasons     = []
soft_keep_reasons   = []

for _, row in df.iterrows():
    prop  = str(row.get("proposed_training_status_candidate", ""))
    hd    = str(row.get("human_decision",     "")).strip()
    meta  = str(row.get("metadata_policy_bucket", "")).upper()
    ra    = _is_true(row.get("rule_A", False))
    rb    = _is_true(row.get("rule_B", False))
    cdej  = int(row.get("cdej_active_count", 0) or 0)
    cdej_names = str(row.get("cdej_active_rules", ""))

    # --- EXCLUDE_TRAINING ---
    if meta == "HARD_EXCLUDE":
        final_statuses.append("EXCLUDE_TRAINING")
        excl_reasons.append("metadata_policy_bucket=HARD_EXCLUDE")
        holdout_reasons.append("")
        soft_keep_reasons.append("")
        continue

    if rb:
        final_statuses.append("EXCLUDE_TRAINING")
        excl_reasons.append("rule_B (hard streamflow QC fail)")
        holdout_reasons.append("")
        soft_keep_reasons.append("")
        continue

    if hd == "EXCLUDE" or prop == "EXCLUDE_TRAINING_CANDIDATE":
        excl_r = ""
        if hd == "EXCLUDE":
            excl_r = f"manual review {row.get('review_pass','')}: human_decision==EXCLUDE"
        elif ra:
            excl_r = "rule_A (previous manual EXCLUDE designation)"
        else:
            excl_r = str(row.get("proposed_exclusion_reason", "EXCLUDE_TRAINING_CANDIDATE"))
        final_statuses.append("EXCLUDE_TRAINING")
        excl_reasons.append(excl_r)
        holdout_reasons.append("")
        soft_keep_reasons.append("")
        continue

    # --- HOLDOUT_REVIEW ---
    if hd == "UNSURE" or prop == "HOLDOUT_REVIEW_CANDIDATE":
        hold_r = ""
        if hd == "UNSURE":
            hold_r = f"manual review {row.get('review_pass','')}: human_decision==UNSURE"
        elif cdej >= 2 and cdej_names:
            hold_r = f"CDEJ compound risk >= 2 rules: {cdej_names}"
        else:
            hold_r = str(row.get("proposed_holdout_reason", "HOLDOUT_REVIEW_CANDIDATE"))
        final_statuses.append("HOLDOUT_REVIEW")
        excl_reasons.append("")
        holdout_reasons.append(hold_r)
        soft_keep_reasons.append("")
        continue

    # --- TRAIN_SOFT_KEEP ---
    if hd == "KEEP_LOW_CONFIDENCE" or prop == "TRAIN_SOFT_KEEP_CANDIDATE":
        sk_r = ""
        if hd == "KEEP_LOW_CONFIDENCE":
            sk_r = f"manual review {row.get('review_pass','')}: human_decision==KEEP_LOW_CONFIDENCE"
        else:
            # Identify which rule(s) triggered soft keep
            rule_flags = [r for r in ["rule_C","rule_D","rule_E","rule_F","rule_G",
                                       "rule_H","rule_I","rule_J"]
                          if _is_true(row.get(r, False))]
            sk_r = f"rule flag(s): {'+'.join(rule_flags)}" if rule_flags else "TRAIN_SOFT_KEEP_CANDIDATE"
        final_statuses.append("TRAIN_SOFT_KEEP")
        excl_reasons.append("")
        holdout_reasons.append("")
        soft_keep_reasons.append(sk_r)
        continue

    # --- TRAIN_CORE ---
    final_statuses.append("TRAIN_CORE")
    excl_reasons.append("")
    holdout_reasons.append("")
    soft_keep_reasons.append("")

df["final_training_status"]  = final_statuses
df["final_exclusion_reason"] = excl_reasons
df["final_holdout_reason"]   = holdout_reasons
df["final_soft_keep_reason"] = soft_keep_reasons

counts = df.final_training_status.value_counts()
print(f"  Final status distribution:")
for s in FINAL_STATUS_ORDER:
    n = counts.get(s, 0)
    print(f"    {s}: {n}  ({n/len(df)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Validation assertions
# ---------------------------------------------------------------------------
print("\n-- 4. Running validation assertions -----------------------------------")

assert len(df) == 3034, f"Expected 3034 rows, got {len(df)}"
print(f"  [PASS] Total rows == 3,034")

assert counts.sum() == 3034, f"Status counts sum to {counts.sum()}, not 3034"
print(f"  [PASS] All rows have a final_training_status (sum == 3,034)")

approx_checks = [
    ("EXCLUDE_TRAINING",  35,  10),
    ("HOLDOUT_REVIEW",   156,  40),
    ("TRAIN_SOFT_KEEP",  627,  80),
    ("TRAIN_CORE",      2216, 100),
]
for status, expected, tol in approx_checks:
    n = counts.get(status, 0)
    assert abs(n - expected) <= tol, (
        f"{status}: got {n}, expected ~{expected} (tolerance ±{tol})"
    )
    print(f"  [PASS] {status}: {n} (expected ~{expected}, tolerance ±{tol})")

# All HARD_EXCLUDE metadata basins must be excluded
hard_excl = df[df.metadata_policy_bucket.astype(str).str.upper() == "HARD_EXCLUDE"]
if len(hard_excl) > 0:
    assert (hard_excl.final_training_status == "EXCLUDE_TRAINING").all(), \
        "Some HARD_EXCLUDE basins not in EXCLUDE_TRAINING"
    print(f"  [PASS] All {len(hard_excl)} HARD_EXCLUDE metadata basins are EXCLUDE_TRAINING")
else:
    print(f"  [INFO] No HARD_EXCLUDE metadata basins in main_training_candidate pool")

# All manual EXCLUDE basins must be EXCLUDE_TRAINING
reviewed = df[df.is_reviewed.apply(_is_true)]
man_excl = reviewed[reviewed.human_decision == "EXCLUDE"]
assert (man_excl.final_training_status == "EXCLUDE_TRAINING").all(), \
    f"Some manual EXCLUDE basins not in EXCLUDE_TRAINING: " \
    f"{man_excl[man_excl.final_training_status != 'EXCLUDE_TRAINING'].STAID.tolist()}"
print(f"  [PASS] All {len(man_excl)} manual EXCLUDE basins are EXCLUDE_TRAINING")

# All manual UNSURE basins must be HOLDOUT_REVIEW (or EXCLUDE_TRAINING if overridden)
man_unsure = reviewed[reviewed.human_decision == "UNSURE"]
bad_unsure = man_unsure[~man_unsure.final_training_status.isin(
    ["HOLDOUT_REVIEW", "EXCLUDE_TRAINING"])]
assert len(bad_unsure) == 0, \
    f"Some manual UNSURE basins not in HOLDOUT/EXCLUDE: {bad_unsure.STAID.tolist()}"
print(f"  [PASS] All {len(man_unsure)} manual UNSURE basins are HOLDOUT_REVIEW or EXCLUDE_TRAINING")

# Initial training set
train_df = df[df.final_training_status.isin(["TRAIN_CORE", "TRAIN_SOFT_KEEP"])]
expected_train = counts.get("TRAIN_CORE", 0) + counts.get("TRAIN_SOFT_KEEP", 0)
assert len(train_df) == expected_train
print(f"  [PASS] Initial training set: {len(train_df)} basins (TRAIN_CORE + TRAIN_SOFT_KEEP)")

print(f"\n  All validation assertions passed.")

# ---------------------------------------------------------------------------
# 5. Write output tables
# ---------------------------------------------------------------------------
print("\n-- 5. Writing output tables --------------------------------------------")

# Define output column order for the main table
MAIN_COLS = [
    "STAID",
    "final_training_status",
    "current_preliminary_training_status",
    "proposed_training_status_candidate",
    "human_decision",
    "review_pass",
    "is_reviewed",
    "final_exclusion_reason",
    "final_holdout_reason",
    "final_soft_keep_reason",
    "rule_A", "rule_C", "rule_D", "rule_E", "rule_F",
    "rule_G", "rule_H", "rule_I", "rule_J",
    "cdej_active_count",
    "cdej_active_rules",
    "compound_risk_count",
    "candidate_class",
    "metadata_policy_bucket",
    "site_type_code",
    "site_type_name",
    "HUC02",
    "STATE",
    "DRAIN_SQKM",
    "area_bin",
    "BFI_AVE",
    "bfi_bin",
    "RBI",
    "zero_flow_fraction",
    "q95_q50_ratio",
    "max_hourly_rise_per_km2",
    "max_abs_hourly_jump_over_Q50",
    "HYDRO_DISTURB_INDX",
    "WATERNLCD06",
    "CANALS_PCT",
    "lka_pc_use",
    "dor_pc_pva",
    "LAT_GAGE",
    "LNG_GAGE",
]
out_cols = [c for c in MAIN_COLS if c in df.columns]

# Table 1: main status table
df[out_cols].to_csv(TABLES_DIR / "final_basin_training_status.csv", index=False)
print(f"  final_basin_training_status.csv: {len(df)} rows, {len(out_cols)} cols")

# Table 2: status counts
status_counts = pd.DataFrame([
    {"final_training_status": s,
     "n": counts.get(s, 0),
     "pct_of_3034": round(counts.get(s, 0) / len(df) * 100, 2)}
    for s in FINAL_STATUS_ORDER
])
status_counts.to_csv(TABLES_DIR / "final_status_counts.csv", index=False)
print(f"  final_status_counts.csv: {len(status_counts)} rows")

# Table 3: by candidate class
ct_class = _cross_tab(df, "candidate_class")
ct_class.to_csv(TABLES_DIR / "final_status_by_candidate_class.csv", index=False)
print(f"  final_status_by_candidate_class.csv: {len(ct_class)} rows")

# Table 4: by HUC02
ct_huc = _cross_tab(df, "HUC02")
ct_huc.to_csv(TABLES_DIR / "final_status_by_huc02.csv", index=False)
print(f"  final_status_by_huc02.csv: {len(ct_huc)} rows")

# Table 5: by area_bin
ct_area = _cross_tab(df, "area_bin")
ct_area.to_csv(TABLES_DIR / "final_status_by_area_bin.csv", index=False)
print(f"  final_status_by_area_bin.csv: {len(ct_area)} rows")

# Table 6: by BFI_bin
ct_bfi = _cross_tab(df, "bfi_bin")
ct_bfi.to_csv(TABLES_DIR / "final_status_by_bfi_bin.csv", index=False)
print(f"  final_status_by_bfi_bin.csv: {len(ct_bfi)} rows")

# Tables 7-10: per-status detail tables
def _status_detail(status, fname):
    sub = df[df.final_training_status == status][out_cols].copy()
    sub.to_csv(TABLES_DIR / fname, index=False)
    print(f"  {fname}: {len(sub)} rows")

_status_detail("EXCLUDE_TRAINING",  "excluded_basins.csv")
_status_detail("HOLDOUT_REVIEW",    "holdout_review_basins.csv")
_status_detail("TRAIN_CORE",        "train_core_basins.csv")
_status_detail("TRAIN_SOFT_KEEP",   "train_soft_keep_basins.csv")

# Tables 11-13: STAID-only lists
df[df.final_training_status.isin(["TRAIN_CORE", "TRAIN_SOFT_KEEP"])][["STAID", "final_training_status"]]\
    .to_csv(TABLES_DIR / "training_basin_list_initial.csv", index=False)
print(f"  training_basin_list_initial.csv: "
      f"{(df.final_training_status.isin(['TRAIN_CORE','TRAIN_SOFT_KEEP'])).sum()} rows")

df[df.final_training_status == "HOLDOUT_REVIEW"][["STAID"]]\
    .to_csv(TABLES_DIR / "holdout_basin_list.csv", index=False)
print(f"  holdout_basin_list.csv: "
      f"{(df.final_training_status == 'HOLDOUT_REVIEW').sum()} rows")

df[df.final_training_status == "EXCLUDE_TRAINING"][["STAID"]]\
    .to_csv(TABLES_DIR / "excluded_basin_list.csv", index=False)
print(f"  excluded_basin_list.csv: "
      f"{(df.final_training_status == 'EXCLUDE_TRAINING').sum()} rows")

# ---------------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------------
print("\n-- 6. Generating plots -------------------------------------------------")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.titlesize": 11})

# Plot 1: overall counts
fig, ax = plt.subplots(figsize=(7, 4))
vals = [counts.get(s, 0) for s in FINAL_STATUS_ORDER]
bars = ax.bar(FINAL_STATUS_ORDER, vals,
              color=[STATUS_COLORS[s] for s in FINAL_STATUS_ORDER])
for b, v in zip(bars, vals):
    pct = v / len(df) * 100
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
            f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_title(f"Final basin training status — all {len(df)} main-training-candidate basins")
ax.set_ylabel("Basin count")
ax.set_ylim(0, max(vals) * 1.18)
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
_save(fig, "final_status_counts.png")

# Plot 2: by candidate class
fig, ax = plt.subplots(figsize=(9, 5))
cls_order = ["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE",
             "MANUAL_REVIEW_CONTEXT", "LOW_FLASHINESS_CONTROL"]
_stacked_bar(ax, ct_class, "candidate_class",
             "Final training status by candidate class", x_order=cls_order)
fig.tight_layout()
_save(fig, "final_status_by_candidate_class.png")

# Plot 3: by HUC02 (sorted by total)
huc_order = ct_huc.sort_values("total", ascending=False)["HUC02"].tolist()
fig, ax = plt.subplots(figsize=(13, 5))
_stacked_bar(ax, ct_huc, "HUC02", "Final training status by HUC02", x_order=huc_order)
fig.tight_layout()
_save(fig, "final_status_by_huc02.png")

# Plot 4: map
lat_ok = df.LAT_GAGE.notna() & df.LNG_GAGE.notna()
if lat_ok.sum() > 0:
    fig, ax = plt.subplots(figsize=(12, 7))
    for st in reversed(FINAL_STATUS_ORDER):
        sub = df[lat_ok & (df.final_training_status == st)]
        ax.scatter(sub.LNG_GAGE, sub.LAT_GAGE,
                   c=STATUS_COLORS[st], s=7, alpha=0.6,
                   label=f"{st} (n={len(sub)})", linewidths=0)
    # Ring for reviewed basins
    rev_ok = lat_ok & df.is_reviewed.apply(_is_true)
    ax.scatter(df.loc[rev_ok, "LNG_GAGE"], df.loc[rev_ok, "LAT_GAGE"],
               c="none", edgecolors="black", s=22, linewidths=0.6,
               label=f"Reviewed (n={rev_ok.sum()})", zorder=5)
    ax.set_xlim(-130, -60)
    ax.set_ylim(24, 52)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Final basin training status — CONUS")
    ax.legend(loc="lower left", fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    _save(fig, "map_final_training_status.png")

# Plot 5: by RBI (histogram per status, overlapping)
fig, ax = plt.subplots(figsize=(9, 5))
for st in FINAL_STATUS_ORDER:
    sub = df[(df.final_training_status == st) & df.RBI.notna()]
    if len(sub) > 0:
        ax.hist(sub.RBI.clip(0, 1), bins=40, alpha=0.55,
                color=STATUS_COLORS[st], label=f"{st} (n={len(sub)})", density=True)
ax.set_xlabel("RBI (Richards-Baker Flashiness Index)")
ax.set_ylabel("Density")
ax.set_title("RBI distribution by final training status")
ax.legend(fontsize=8)
fig.tight_layout()
_save(fig, "final_status_by_rbi.png")

# Plot 6: by area_bin
fig, ax = plt.subplots(figsize=(8, 5))
_stacked_bar(ax, ct_area, "area_bin",
             "Final training status by drainage area bin",
             x_order=[b for b in AREA_BIN_ORDER if b in ct_area["area_bin"].values])
fig.tight_layout()
_save(fig, "final_status_by_area_bin.png")

# Plot 7: by BFI_bin
fig, ax = plt.subplots(figsize=(8, 5))
_stacked_bar(ax, ct_bfi, "bfi_bin",
             "Final training status by BFI bin",
             x_order=[b for b in BFI_BIN_ORDER if b in ct_bfi["bfi_bin"].values])
fig.tight_layout()
_save(fig, "final_status_by_bfi_bin.png")

# ---------------------------------------------------------------------------
# 7. Summaries
# ---------------------------------------------------------------------------
print("\n-- 7. Writing summaries ------------------------------------------------")

n_core     = counts.get("TRAIN_CORE",       0)
n_soft     = counts.get("TRAIN_SOFT_KEEP",  0)
n_holdout  = counts.get("HOLDOUT_REVIEW",   0)
n_excl     = counts.get("EXCLUDE_TRAINING", 0)
n_train    = n_core + n_soft
n_total    = len(df)

reviewed_df = df[df.is_reviewed.apply(_is_true)]
n_reviewed  = len(reviewed_df)
n_man_excl  = (reviewed_df.human_decision == "EXCLUDE").sum()
n_man_unsr  = (reviewed_df.human_decision == "UNSURE").sum()

# Rule-driven breakdown
rule_a_excl = (df.rule_A.apply(_is_true) & (df.final_training_status == "EXCLUDE_TRAINING")).sum()
cdej_holdout = ((df.cdej_active_count >= 2) & (df.final_training_status == "HOLDOUT_REVIEW")).sum()

# JSON
summary_json = {
    "output_version":  "flashnh_final_basin_selection_v001",
    "n_main_training_candidate": n_total,
    "final_status_counts": {
        "TRAIN_CORE":       int(n_core),
        "TRAIN_SOFT_KEEP":  int(n_soft),
        "HOLDOUT_REVIEW":   int(n_holdout),
        "EXCLUDE_TRAINING": int(n_excl),
    },
    "initial_training_set_size": int(n_train),
    "n_reviewed":   int(n_reviewed),
    "n_man_exclude": int(n_man_excl),
    "n_man_unsure":  int(n_man_unsr),
    "n_rule_A_exclusions":   int(rule_a_excl),
    "n_cdej_ge2_holdouts":   int(cdej_holdout),
    "inputs": {
        "policy_candidates": str(POLICY_CANDIDATES_CSV.name),
        "combined_labels":   str(COMBINED_LABELS_CSV.name),
        "rule_matrix":       str(RULE_MATRIX_CSV.name),
        "metrics_meta":      str(METRICS_META_CSV.name),
    },
    "policy_rules": {
        "EXCLUDE_TRAINING": [
            "metadata_policy_bucket == HARD_EXCLUDE",
            "rule_B (hard streamflow QC fail)",
            "human_decision == EXCLUDE (manual review)",
            "proposed_training_status_candidate == EXCLUDE_TRAINING_CANDIDATE",
        ],
        "HOLDOUT_REVIEW": [
            "human_decision == UNSURE (manual review)",
            "proposed_training_status_candidate == HOLDOUT_REVIEW_CANDIDATE",
            "  (which includes: CDEJ active count >= 2)",
        ],
        "TRAIN_SOFT_KEEP": [
            "human_decision == KEEP_LOW_CONFIDENCE (manual review)",
            "proposed_training_status_candidate == TRAIN_SOFT_KEEP_CANDIDATE",
            "  (which includes: single CDEJ rule, rule_G, rule_H, rule_F, rule_I)",
        ],
        "TRAIN_CORE": [
            "No exclusion, holdout, or soft-keep trigger applies",
        ],
    },
    "status_is_preliminary": True,
}

with open(SUMMARY_DIR / "final_basin_selection_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)
print("  Wrote final_basin_selection_summary.json")

# Markdown
md = f"""# Final Basin Training Status — Flash-NH WY2024 Pre-Training Selection

**Version**: flashnh_final_basin_selection_v001
**Status**: PRELIMINARY — no basins have been permanently removed; all decisions are
  reversible and should be validated with post-training residual analysis.

---

## Executive Summary

This document records the accepted pre-training basin selection policy for the
Flash-NH WY2024 streamflow modeling dataset. Of the **{n_total} main-training-candidate
basins**, the final assignment is:

| Status | Count | % |
|--------|-------|---|
| TRAIN_CORE | {n_core} | {n_core/n_total*100:.1f}% |
| TRAIN_SOFT_KEEP | {n_soft} | {n_soft/n_total*100:.1f}% |
| HOLDOUT_REVIEW | {n_holdout} | {n_holdout/n_total*100:.1f}% |
| EXCLUDE_TRAINING | {n_excl} | {n_excl/n_total*100:.1f}% |

**Initial training set**: {n_train} basins (TRAIN_CORE + TRAIN_SOFT_KEEP).
**Held out for review**: {n_holdout} basins.
**Excluded**: {n_excl} basins.

---

## Input Lineage

| Input | Description |
|-------|-------------|
| `{POLICY_CANDIDATES_CSV.name}` | Combined analysis proposed status per basin (3,034 rows) |
| `{COMBINED_LABELS_CSV.name}` | Manually reviewed labels, both passes (148 rows) |
| `{RULE_MATRIX_CSV.name}` | Rule flags and attributes for all 3,034 candidates |
| `{METRICS_META_CSV.name}` | WY2024 streamflow metrics + USGS site metadata |

---

## Final Status Definitions

### TRAIN_CORE ({n_core} basins, {n_core/n_total*100:.1f}%)
No risk flags trigger. Includes manually reviewed basins labeled **KEEP** with no
compound rule co-occurrence. These basins are expected to have clean, natural
streamflow records suitable for training the Flash-NH neural network model.

### TRAIN_SOFT_KEEP ({n_soft} basins, {n_soft/n_total*100:.1f}%)
One or more soft risk flags are present, OR the basin was manually labeled
**KEEP_LOW_CONFIDENCE**. These basins are **included in the initial training run**
but are tracked as a risk stratum. Post-training residual analysis should check
whether any TRAIN_SOFT_KEEP basins have systematically elevated errors. If so,
they can be moved to HOLDOUT_REVIEW or EXCLUDE_TRAINING with data-driven justification.

Soft flags that trigger TRAIN_SOFT_KEEP (when appearing alone):
- rule_C (single): high HYDRO_DISTURB_INDX, but below compound-risk threshold
- rule_D (single): high lake/reservoir area %, but below compound-risk threshold
- rule_E (single): high degree of regulation, but below compound-risk threshold
- rule_J (single): high canal % or DOR, but below compound-risk threshold
- rule_G: mostly-zero / suspicious ephemeral — real intermittent streams in arid regions
- rule_H: extreme hourly jumps — confirmed to reflect real flashy hydrology in most cases
- rule_F: large, slow, low-flashiness basin — important control for model generalization
- rule_I: high HYDRO_DISTURB_INDX (secondary threshold)

### HOLDOUT_REVIEW ({n_holdout} basins, {n_holdout/n_total*100:.1f}%)
Strong evidence of regulation, lentic influence, or artificial flow control, OR
the basin was manually labeled **UNSURE**.

These basins are **withheld from the initial training run** but are **NOT
permanently excluded**. They should be revisited:
1. After post-training residual analysis (if the model performs well on
   TRAIN_CORE+SOFT_KEEP, HOLDOUT basins may be progressively added).
2. After targeted secondary manual review (especially CDEJ≥2 basins with
   reviewer notes mentioning regulation or lake influence).

Primary HOLDOUT trigger: **CDEJ compound risk count ≥ 2** ({cdej_holdout} basins),
where CDEJ = rule_C (disturbance index) + rule_D (lake/reservoir %) +
rule_E (degree of regulation) + rule_J (canals/DOR). This combination has an
exclude-or-unsure rate of 0.45–0.69 in the reviewed sample (see combined analysis).

### EXCLUDE_TRAINING ({n_excl} basins, {n_excl/n_total*100:.1f}%)
Hard exclusions based on manual review label or prior automated screening:
- **{n_man_excl} basins**: manually labeled EXCLUDE across both review passes.
- **{rule_a_excl} basins**: rule_A flag (manual EXCLUDE designation from pass-1 analysis).
  (Note: these 16 rule_A basins are a subset of the 35 manual EXCLUDEs, not additive.)
- **0 basins**: HARD_EXCLUDE metadata (all 3,034 main_training_candidates passed
  metadata screening; HARD_EXCLUDE basins were already removed before this stage).

---

## Why Manual EXCLUDE Is a Hard Exclusion

Basins manually labeled EXCLUDE were individually inspected and found to have
streamflow records that would mislead model training — typically due to:
- Strong regulation by upstream dams or reservoirs (daily/weekly operational patterns)
- Lake or wetland-dominated catchments with no meaningful event response
- Sensor artifacts or data quality issues making the record unreliable

These are firm recommendations from the reviewer. They are not reversed by any
automated rule because automated rules are calibrated at population thresholds,
not individual basin quality.

---

## Why HOLDOUT_REVIEW Is Not a Permanent Exclusion

The HOLDOUT_REVIEW designation reflects **elevated risk** but not **confirmed
disqualification**. Several reasons to retain these basins for potential future
inclusion:

1. The CDEJ≥2 threshold flags basins with compound regulation/lentic signals, but
   the reviewed exclude-or-unsure rate in this group is ~50–70%, meaning
   30–50% of these basins appear acceptable on visual inspection.
2. Removing 156 additional basins (beyond the 35 hard exclusions) reduces the
   training set by ~5.5% and may reduce geographic and hydrological diversity.
3. Post-training residual analysis provides a cleaner, data-driven basis for
   further exclusions: train on the initial set, identify stations with
   systematically poor model fit, then review those specifically.

---

## Why TRAIN_SOFT_KEEP Is Retained

rule_G (mostly-zero / ephemeral) and rule_H (extreme jumps) were the two rules
most debated during the manual review. The combined analysis found:
- rule_H: exclude-or-unsure rate **0.15** (n=27) — consistent with real flashy hydrology.
- rule_G: exclude-or-unsure rate **0.22** (n=18) — consistent with real intermittent streams.

Excluding all rule_G or rule_H basins would remove legitimate hydrological signal,
including precisely the kinds of fast-response, low-baseflow catchments that the
Flash-NH model is designed to handle. These basins are retained in training but
tracked as a risk stratum for post-training validation.

---

## Evidence Summary from Combined Manual Review

- **148 basins** reviewed across two passes (73 + 75).
- Combined EXCLUDE rate: 23.6%; UNSURE rate: 10.1%; total concern rate: 33.8%.
- **Regulation/lentic risk** is the dominant reason for EXCLUDE and UNSURE:
  reviewer notes mentioning "regulated/managed" (n=28, eu_rate=0.82) and
  "dam/reservoir" (n=26, eu_rate=0.77) strongly predict exclusion decisions.
- **rule_C + rule_D** co-occurrence: exclude-or-unsure rate = 0.60 (n=20).
- **rule_C + rule_D + rule_E + rule_J** co-occurrence: eu_rate = 0.69 (n=13).
- **rule_H alone**: eu_rate = 0.15 — confirmed NOT a reliable exclusion signal.
- **zero/ephemeral keywords**: eu_rate = 0.04 — confirms KEEP for ephemeral regimes.

---

## Exact Policy Rules Applied

```
Priority order: EXCLUDE > HOLDOUT > SOFT_KEEP > CORE

EXCLUDE_TRAINING if:
  metadata_policy_bucket == HARD_EXCLUDE         (hard metadata exclusion)
  OR rule_B == True                              (hard streamflow QC fail)
  OR human_decision == EXCLUDE                   (manual review label)
  OR proposed_training_status_candidate          (= human EXCLUDE or rule_A)
     == EXCLUDE_TRAINING_CANDIDATE

HOLDOUT_REVIEW if not EXCLUDE and:
  human_decision == UNSURE                       (manual review label)
  OR proposed_training_status_candidate          (= CDEJ active count >= 2)
     == HOLDOUT_REVIEW_CANDIDATE

TRAIN_SOFT_KEEP if not EXCLUDE/HOLDOUT and:
  human_decision == KEEP_LOW_CONFIDENCE          (manual review label)
  OR proposed_training_status_candidate          (= single CDEJ/G/H/F/I rule)
     == TRAIN_SOFT_KEEP_CANDIDATE

TRAIN_CORE otherwise
```

---

## Recommended Initial Training Set

**{n_train} basins** (TRAIN_CORE + TRAIN_SOFT_KEEP).

STAID list: `tables/training_basin_list_initial.csv`

This set spans the full range of Flash-NH target hydrology:
- Flashy core and moderate basins that are the primary training targets
- Low-flashiness controls for model generalization
- Intermittent and ephemeral basins (rule_G, rule_H) that represent real
  hydrological diversity

---

## Remaining Caveats

1. **HOLDOUT_REVIEW basins**: {n_holdout} basins are withheld but not excluded.
   They represent ~5.1% of the candidate pool. If post-training analysis shows
   the initial model is robust, these may be progressively included.

2. **TRAIN_SOFT_KEEP basins**: {n_soft} basins are in training but flagged.
   Rule_H and rule_G basins in this group should be monitored for elevated
   residuals during post-training evaluation.

3. **Sampling uncertainty**: The 148 reviewed basins are an enriched sample,
   not a random sample. Exclude/unsure rates from the reviewed sample should
   not be directly extrapolated to the full 3,034 pool.

4. **Rule threshold sensitivity**: Thresholds (p95/p99) were set before reviewing
   pass 2. Alternative thresholds might move some TRAIN_SOFT_KEEP basins to
   HOLDOUT or vice versa. This is an opportunity for a sensitivity analysis
   after the initial training run.

---

## Next Steps

1. **Run NeuralHydrology forcing preprocessing** on the {n_train}-basin initial
   training list (`tables/training_basin_list_initial.csv`).

2. **Initial model training**: train Flash-NH with default hyperparameters on the
   initial set. Record per-basin NSE, KGE, and RBI-weighted scores on the
   validation period.

3. **Post-training residual analysis**: rank the {n_train} training basins by
   validation-period NSE. Investigate the bottom decile for systematic issues.
   Cross-check the bottom decile with TRAIN_SOFT_KEEP flag columns.

4. **HOLDOUT_REVIEW secondary review**: schedule targeted review of the {n_holdout}
   HOLDOUT basins, prioritizing CDEJ≥2 basins with reviewer notes mentioning
   regulation. Remove confirmed problematic basins; add confirmed clean basins
   to training.

5. **Version final exclusion**: once post-training analysis and HOLDOUT review
   are complete, create `flashnh_final_basin_selection_v002` with the finalized
   permanent exclusion list.

---

*Generated by `scripts/build_final_basin_training_status.py`.
All status labels are PRELIMINARY until confirmed by post-training validation.*
"""

with open(SUMMARY_DIR / "final_basin_selection_summary.md", "w", encoding="utf-8") as f:
    f.write(md)
print("  Wrote final_basin_selection_summary.md")

# ---------------------------------------------------------------------------
# 8. Final report
# ---------------------------------------------------------------------------
print("\n-- 8. Output report ----------------------------------------------------")
print(f"\n  Output directory: {OUT_DIR}")
print(f"\n  Tables:")
for f in sorted(TABLES_DIR.glob("*.csv")):
    n_rows = sum(1 for _ in open(f, encoding="utf-8")) - 1
    print(f"    {f.name}: {n_rows} rows")
print(f"\n  Plots:")
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(f"    {f.name}")
print(f"\n  Summaries:")
for f in sorted(SUMMARY_DIR.glob("*")):
    print(f"    {f.name}")

print(f"\n  Final training set: {n_train} basins (TRAIN_CORE + TRAIN_SOFT_KEEP)")
print(f"  Holdout set:        {n_holdout} basins")
print(f"  Excluded:           {n_excl} basins")
print("\n  DONE.")
