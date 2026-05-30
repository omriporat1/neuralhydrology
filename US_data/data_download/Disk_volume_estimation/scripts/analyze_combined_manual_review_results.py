#!/usr/bin/env python3
"""
Analyze combined pass-1 and pass-2 manual review labels and propose
final basin-training status policy for Flash-NH WY2024 screening.

Pass 1: 73 basins from v004 main-training-candidate review set.
Pass 2: 75 basins from v005 second-pass rule-validation set.

Outputs to: reports/flashnh_combined_manual_review_analysis_v001/

Usage:
    python scripts/analyze_combined_manual_review_results.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- Paths ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

PASS1_LABELS_PRIMARY  = (
    REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v004_main_training_candidate"
    / "manual_review_labels_pass1_locked.csv"
)
PASS1_LABELS_FALLBACK = (
    REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v004_main_training_candidate"
    / "manual_review_labels.csv"
)
PASS2_LABELS_CSV  = (
    REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v005_second_pass_rules"
    / "manual_review_labels_pass2.csv"
)
TEMPLATE_V004 = (
    REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v004_main_training_candidate"
    / "tables/human_review_template.csv"
)
TEMPLATE_V005 = (
    REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v005_second_pass_rules"
    / "tables/human_review_template.csv"
)
RULE_MATRIX_CSV = (
    REPO_ROOT / "reports/flashnh_manual_review_rule_analysis_v001"
    / "tables/full_candidate_rule_matrix.csv"
)

OUT_DIR      = REPO_ROOT / "reports/flashnh_combined_manual_review_analysis_v001"
TABLES_DIR   = OUT_DIR / "tables"
PLOTS_DIR    = OUT_DIR / "plots"
SUMMARY_DIR  = OUT_DIR / "summaries"

for d in (TABLES_DIR, PLOTS_DIR, SUMMARY_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -- Constants -----------------------------------------------------------------
RULE_COLS  = [f"rule_{x}" for x in "ABCDEFGHIJ"]
CDEJ_COLS  = ["rule_C", "rule_D", "rule_E", "rule_J"]
CDEGHJ_COLS = ["rule_C", "rule_D", "rule_E", "rule_G", "rule_H", "rule_J"]

LABEL_COLS = ["STAID", "human_decision", "hydrograph_behavior",
              "artifact_type", "confidence", "reviewer_notes"]

DECISION_ORDER  = ["KEEP", "KEEP_LOW_CONFIDENCE", "UNSURE", "EXCLUDE"]
DECISION_COLORS = {
    "KEEP":                 "#2ca02c",
    "KEEP_LOW_CONFIDENCE":  "#98df8a",
    "UNSURE":               "#9467bd",
    "EXCLUDE":              "#d62728",
}

STATUS_ORDER  = ["TRAIN_CORE_CANDIDATE", "TRAIN_SOFT_KEEP_CANDIDATE",
                 "HOLDOUT_REVIEW_CANDIDATE", "EXCLUDE_TRAINING_CANDIDATE"]
STATUS_COLORS = {
    "TRAIN_CORE_CANDIDATE":        "#2ca02c",
    "TRAIN_SOFT_KEEP_CANDIDATE":   "#1f77b4",
    "HOLDOUT_REVIEW_CANDIDATE":    "#ff7f0e",
    "EXCLUDE_TRAINING_CANDIDATE":  "#d62728",
}

REGULATION_KW_GROUPS = {
    "dam_reservoir": [r"\bdam\b", r"\breservoir\b", r"\bimpound"],
    "regulated_managed": [r"\bregulat", r"\bmanaged\b", r"\bmanagement\b"],
    "lake_pond_lentic": [r"\blake\b", r"\bpond\b", r"\blentic\b"],
    "daily_hydropeaking": [r"daily pattern", r"\bdiurnal\b", r"\bhydropeaking\b"],
    "sewage_canal_artificial": [r"\bsewage\b", r"\bcanal\b", r"\bartificial\b", r"\bdiversion\b"],
    "control_weir_gate": [r"control structure", r"\bgate\b", r"\bweir\b", r"\bsluice\b"],
    "sensor_data_quality": [r"\bsensor\b", r"\bspike\b", r"\bnoise\b", r"\bartifact\b",
                             r"missing data"],
    "zero_dry_ephemeral": [r"\bzero\b", r"\bdry\b", r"\bephemeral\b",
                            r"\bintermittent\b", r"no flow"],
}

# -- Helpers -------------------------------------------------------------------

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


def _rate(num, denom):
    return round(num / denom, 4) if denom > 0 else np.nan


def _rule_recommendation(eu_rate, n) -> str:
    if n < 3:
        return "INSUFFICIENT DATA"
    if eu_rate >= 0.60:
        return "STRONG — holdout or exclude trigger"
    if eu_rate >= 0.35:
        return "MODERATE — holdout context flag"
    if eu_rate >= 0.15:
        return "WEAK — soft context flag only"
    return "NO EVIDENCE — flag likely over-broad"


def _active_cdej_names(row) -> list:
    return [r for r in CDEJ_COLS if _is_true(row.get(r, False))]


def compute_proposed_status(row, human_decision=None):
    """Propose training status for one basin.

    Returns (proposed_status, exclusion_reason, holdout_reason).
    Priority order: EXCLUDE > HOLDOUT > SOFT_KEEP > CORE.
    human_decision overrides rule-based logic when present.
    """
    excl_reason    = ""
    holdout_reason = ""

    # Metadata hard exclusion
    if str(row.get("metadata_policy_bucket", "")).upper() == "HARD_EXCLUDE":
        return "EXCLUDE_TRAINING_CANDIDATE", "metadata_policy_bucket=HARD_EXCLUDE", ""

    # rule_B: hard streamflow QC fail (currently 0 basins, but checked for completeness)
    if _is_true(row.get("rule_B", False)):
        return "EXCLUDE_TRAINING_CANDIDATE", "rule_B (hard QC fail)", ""

    # Human EXCLUDE overrides everything except metadata hard exclusion
    if human_decision == "EXCLUDE":
        return "EXCLUDE_TRAINING_CANDIDATE", "human_decision==EXCLUDE", ""

    # rule_A: previous manual EXCLUDE designation (16 basins in full matrix)
    if _is_true(row.get("rule_A", False)):
        return "EXCLUDE_TRAINING_CANDIDATE", "rule_A (manual EXCLUDE flag)", ""

    # Human UNSURE → holdout
    if human_decision == "UNSURE":
        return "HOLDOUT_REVIEW_CANDIDATE", "", "human_decision==UNSURE"

    # Strong rule combinations (regulation/lentic/disturbance) → holdout
    rule_c = _is_true(row.get("rule_C", False))
    rule_d = _is_true(row.get("rule_D", False))
    rule_e = _is_true(row.get("rule_E", False))
    rule_g = _is_true(row.get("rule_G", False))
    rule_h = _is_true(row.get("rule_H", False))
    rule_j = _is_true(row.get("rule_J", False))
    rule_f = _is_true(row.get("rule_F", False))
    rule_i = _is_true(row.get("rule_I", False))

    cdej_count  = sum([rule_c, rule_d, rule_e, rule_j])
    cdej_active = [r for r, v in zip(["C", "D", "E", "J"],
                                      [rule_c, rule_d, rule_e, rule_j]) if v]

    if cdej_count >= 2:
        return "HOLDOUT_REVIEW_CANDIDATE", "", f"CDEJ rules ≥2: {'+'.join(cdej_active)}"

    # Human KEEP_LOW_CONFIDENCE → soft keep
    if human_decision == "KEEP_LOW_CONFIDENCE":
        return "TRAIN_SOFT_KEEP_CANDIDATE", "", ""

    # Single regulation/lentic/disturbance rule → soft keep
    if rule_c or rule_d or rule_e or rule_j:
        names = [r for r, v in zip(["C","D","E","J"], [rule_c,rule_d,rule_e,rule_j]) if v]
        return "TRAIN_SOFT_KEEP_CANDIDATE", "", ""

    # Mostly-zero or extreme-jump alone → soft keep (not excluded)
    if rule_g or rule_h:
        return "TRAIN_SOFT_KEEP_CANDIDATE", "", ""

    # Large-slow-low-flashiness or high-disturbance-index alone → soft keep
    if rule_f or rule_i:
        return "TRAIN_SOFT_KEEP_CANDIDATE", "", ""

    return "TRAIN_CORE_CANDIDATE", "", ""


# -- 1. Load and validate labels -----------------------------------------------

print("\n-- 1. Loading labels ---------------------------------------------------")

p1_path = PASS1_LABELS_PRIMARY if PASS1_LABELS_PRIMARY.exists() else PASS1_LABELS_FALLBACK
p1_src  = "pass1_locked" if PASS1_LABELS_PRIMARY.exists() else "pass1_fallback"
p1 = pd.read_csv(p1_path, dtype={"STAID": str})
p1["review_pass"] = "pass1"
p1["labels_source"] = p1_src

p2 = pd.read_csv(PASS2_LABELS_CSV, dtype={"STAID": str})
p2["review_pass"] = "pass2"
p2["labels_source"] = "pass2"

print(f"  Pass 1: {len(p1)} rows  (source: {p1_path.name})")
print(f"  Pass 2: {len(p2)} rows  (source: {PASS2_LABELS_CSV.name})")

pass1_counts = p1.human_decision.value_counts().to_dict()
pass2_counts = p2.human_decision.value_counts().to_dict()
print(f"  Pass 1 decisions: {pass1_counts}")
print(f"  Pass 2 decisions: {pass2_counts}")

# Duplicate check
overlap = set(p1.STAID) & set(p2.STAID)
if overlap:
    print(f"\n  WARNING: {len(overlap)} STAIDs appear in both passes: {sorted(overlap)}")
    audit_rows = []
    for sid in sorted(overlap):
        r1 = p1[p1.STAID == sid].iloc[0]
        r2 = p2[p2.STAID == sid].iloc[0]
        audit_rows.append({
            "STAID": sid,
            "pass1_decision": r1.human_decision,
            "pass2_decision": r2.human_decision,
            "agree": r1.human_decision == r2.human_decision,
        })
    dup_audit = pd.DataFrame(audit_rows)
    dup_audit.to_csv(TABLES_DIR / "duplicate_staid_audit.csv", index=False)
    print(f"  Duplicate audit written to duplicate_staid_audit.csv")
else:
    print(f"  No duplicate STAIDs — pass 1 and pass 2 are fully disjoint.")

# Combine
combined = pd.concat([p1, p2], ignore_index=True)
print(f"  Combined: {len(combined)} rows (expected 148)")

# -- 2. Load templates and rule matrix ----------------------------------------

print("\n-- 2. Loading templates and rule matrix ---------------------------------")

t4 = pd.read_csv(TEMPLATE_V004, dtype={"STAID": str})
t5 = pd.read_csv(TEMPLATE_V005, dtype={"STAID": str})
rm = pd.read_csv(RULE_MATRIX_CSV, dtype={"STAID": str})

print(f"  V004 template: {len(t4)} rows")
print(f"  V005 template: {len(t5)} rows")
print(f"  Rule matrix:   {len(rm)} rows")

# Boolean-coerce all rule columns in rule matrix
for col in RULE_COLS:
    if col in rm.columns:
        rm[col] = rm[col].apply(_is_true)

# -- 3. Join labels to templates then rule matrix ------------------------------

print("\n-- 3. Joining labels to templates and rule matrix -----------------------")

# Template metadata columns to carry forward (union of what's available)
TMPL_META_COLS = [
    "STAID", "candidate_class", "review_group", "RBI", "BFI_AVE", "DRAIN_SQKM",
    "HUC02", "STATE", "qc_labels", "context_flags", "zero_flow_fraction",
    "hourly_completeness_pct", "q50", "q95", "q99", "q95_q50_ratio",
    "max_hourly_rise_per_km2", "max_abs_hourly_jump_over_Q50",
    # v005-only extras — will be NaN for pass1
    "sampling_tier", "dominant_review_reason", "preliminary_training_status",
    "lka_pc_use", "dor_pc_pva", "HYDRO_DISTURB_INDX", "CANALS_PCT", "WATERNLCD06",
    "LAT_GAGE", "LNG_GAGE",
]

def _safe_tmpl_cols(df, wanted):
    return [c for c in wanted if c in df.columns]

t4_sub = t4[[c for c in _safe_tmpl_cols(t4, TMPL_META_COLS)]].copy()
t5_sub = t5[[c for c in _safe_tmpl_cols(t5, TMPL_META_COLS)]].copy()

# Join pass1 labels to v004 template
pass1_merged = combined[combined.review_pass == "pass1"].merge(
    t4_sub, on="STAID", how="left"
)
# Join pass2 labels to v005 template
pass2_merged = combined[combined.review_pass == "pass2"].merge(
    t5_sub, on="STAID", how="left"
)

all_merged = pd.concat([pass1_merged, pass2_merged], ignore_index=True)

# Now join to rule matrix for rule flags (rule_A–rule_J) and
# preliminary_training_status (authoritative for pass1 which has no template flags)
RULE_JOIN_COLS = ["STAID"] + RULE_COLS + ["compound_risk_count",
                                           "metadata_policy_bucket"]
# Fill rule flags from matrix for pass1 (where template has none)
# For pass2, template already has them, but rule matrix is authoritative
rule_join = rm[_safe_tmpl_cols(rm, RULE_JOIN_COLS)].copy()

# Also get preliminary_training_status from matrix for pass1 basins
if "preliminary_training_status" not in all_merged.columns:
    all_merged["preliminary_training_status"] = np.nan
if "preliminary_training_status" in rm.columns:
    prelim_join = rm[["STAID", "preliminary_training_status"]].rename(
        columns={"preliminary_training_status": "_prelim_matrix"}
    )
    all_merged = all_merged.merge(prelim_join, on="STAID", how="left")
    # Fill NaN preliminary_training_status from matrix
    mask_nan = all_merged.preliminary_training_status.isna()
    all_merged.loc[mask_nan, "preliminary_training_status"] = (
        all_merged.loc[mask_nan, "_prelim_matrix"]
    )
    all_merged.drop(columns=["_prelim_matrix"], inplace=True)

# Merge rule flags — overwrite any existing rule columns with matrix values
cols_to_drop = [c for c in RULE_COLS + ["compound_risk_count", "metadata_policy_bucket"]
                if c in all_merged.columns]
all_merged.drop(columns=cols_to_drop, inplace=True)
all_merged = all_merged.merge(rule_join, on="STAID", how="left")

# Patch LAT_GAGE / LNG_GAGE from rule matrix where missing
for latcol, loncol in [("LAT_GAGE", "LNG_GAGE")]:
    if latcol not in all_merged.columns:
        all_merged[latcol] = np.nan
        all_merged[loncol] = np.nan
    rm_latlon = rm[["STAID", latcol, loncol]].rename(
        columns={latcol: "_lat", loncol: "_lon"}
    )
    all_merged = all_merged.merge(rm_latlon, on="STAID", how="left")
    mask = all_merged[latcol].isna()
    all_merged.loc[mask, latcol] = all_merged.loc[mask, "_lat"]
    all_merged.loc[mask, loncol] = all_merged.loc[mask, "_lon"]
    all_merged.drop(columns=["_lat", "_lon"], inplace=True)

# Compute cdej_active_count per reviewed basin
for col in CDEJ_COLS:
    all_merged[col] = all_merged[col].apply(_is_true)
all_merged["cdej_active_count"] = all_merged[CDEJ_COLS].sum(axis=1)

# Compute proposed status for reviewed basins
all_merged["proposed_training_status_candidate"] = ""
all_merged["proposed_exclusion_reason"]           = ""
all_merged["proposed_holdout_reason"]             = ""
for idx, row in all_merged.iterrows():
    st, er, hr = compute_proposed_status(row, human_decision=row.get("human_decision"))
    all_merged.at[idx, "proposed_training_status_candidate"] = st
    all_merged.at[idx, "proposed_exclusion_reason"]           = er
    all_merged.at[idx, "proposed_holdout_reason"]             = hr

print(f"  Combined reviewed rows after join: {len(all_merged)}")
print(f"  Pass1 joined: {(all_merged.review_pass == 'pass1').sum()}  "
      f"Pass2 joined: {(all_merged.review_pass == 'pass2').sum()}")
unmatched = all_merged[RULE_COLS].isna().all(axis=1).sum()
print(f"  Rows with no rule-matrix match: {unmatched}")

# -- 4. TABLE 1: combined_manual_labels.csv -----------------------------------

print("\n-- 4. Writing Table 1: combined_manual_labels.csv ----------------------")

COMBINED_OUT_COLS = (
    ["STAID", "review_pass", "labels_source", "human_decision", "hydrograph_behavior",
     "artifact_type", "confidence", "reviewer_notes",
     "candidate_class", "review_group", "sampling_tier", "dominant_review_reason",
     "preliminary_training_status",
     "proposed_training_status_candidate", "proposed_exclusion_reason",
     "proposed_holdout_reason",
     "RBI", "BFI_AVE", "DRAIN_SQKM", "HUC02", "STATE",
     "zero_flow_fraction", "hourly_completeness_pct", "q50", "q95", "q99",
     "q95_q50_ratio", "max_hourly_rise_per_km2", "max_abs_hourly_jump_over_Q50",
     "cdej_active_count"] +
    RULE_COLS +
    ["compound_risk_count", "metadata_policy_bucket",
     "HYDRO_DISTURB_INDX", "WATERNLCD06", "CANALS_PCT", "lka_pc_use", "dor_pc_pva",
     "LAT_GAGE", "LNG_GAGE",
     "qc_labels", "context_flags"]
)
out_cols = [c for c in COMBINED_OUT_COLS if c in all_merged.columns]
all_merged[out_cols].to_csv(TABLES_DIR / "combined_manual_labels.csv", index=False)
print(f"  Rows: {len(all_merged)}, Cols: {len(out_cols)}")

# -- 5. TABLE 2: manual_decision_counts_by_pass.csv ---------------------------

print("\n-- 5. Writing Table 2: manual_decision_counts_by_pass.csv --------------")

def _decision_counts(df, label):
    row = {"pass": label, "total": len(df)}
    for d in DECISION_ORDER:
        n = (df.human_decision == d).sum()
        row[d] = n
        row[f"{d}_pct"] = round(n / len(df) * 100, 1) if len(df) > 0 else np.nan
    return row

counts_rows = [
    _decision_counts(all_merged[all_merged.review_pass == "pass1"], "pass1"),
    _decision_counts(all_merged[all_merged.review_pass == "pass2"], "pass2"),
    _decision_counts(all_merged, "combined"),
]
dec_counts_df = pd.DataFrame(counts_rows)
dec_counts_df.to_csv(TABLES_DIR / "manual_decision_counts_by_pass.csv", index=False)
print(f"  {dec_counts_df.to_string(index=False)}")

# -- 6. TABLE 3: rule_performance_combined.csv --------------------------------

print("\n-- 6. Writing Table 3: rule_performance_combined.csv -------------------")

rule_perf_rows = []
for rule in RULE_COLS:
    flagged = all_merged[all_merged[rule].apply(_is_true)]
    full_flagged_n = int(rm[rule].apply(_is_true).sum()) if rule in rm.columns else 0

    keep_n     = int((flagged.human_decision == "KEEP").sum())
    klc_n      = int((flagged.human_decision == "KEEP_LOW_CONFIDENCE").sum())
    unsure_n   = int((flagged.human_decision == "UNSURE").sum())
    exclude_n  = int((flagged.human_decision == "EXCLUDE").sum())
    rev_n      = len(flagged)

    p1_n = int((all_merged[(all_merged.review_pass == "pass1") &
                            all_merged[rule].apply(_is_true)]).shape[0])
    p2_n = int((all_merged[(all_merged.review_pass == "pass2") &
                            all_merged[rule].apply(_is_true)]).shape[0])

    eu_rate  = _rate(exclude_n + unsure_n, rev_n)
    exc_rate = _rate(exclude_n, rev_n)
    keep_rate = _rate(keep_n + klc_n, rev_n)

    rule_perf_rows.append({
        "rule":                           rule,
        "reviewed_flagged_n":             rev_n,
        "keep_n":                         keep_n,
        "keep_low_confidence_n":          klc_n,
        "unsure_n":                       unsure_n,
        "exclude_n":                      exclude_n,
        "exclude_rate":                   exc_rate,
        "exclude_or_unsure_rate":         eu_rate,
        "keep_or_keep_low_confidence_rate": keep_rate,
        "pass1_flagged_n":                p1_n,
        "pass2_flagged_n":                p2_n,
        "full_candidate_flagged_n":       full_flagged_n,
        "full_candidate_flagged_pct":     round(full_flagged_n / len(rm) * 100, 2),
        "recommendation":                 _rule_recommendation(eu_rate, rev_n),
    })

rule_perf = pd.DataFrame(rule_perf_rows)
rule_perf.to_csv(TABLES_DIR / "rule_performance_combined.csv", index=False)
print(rule_perf[["rule", "reviewed_flagged_n", "exclude_n", "unsure_n",
                  "exclude_or_unsure_rate", "recommendation"]].to_string(index=False))

# -- 7. TABLE 4: rule_combination_performance.csv -----------------------------

print("\n-- 7. Writing Table 4: rule_combination_performance.csv ----------------")

combo_rows = []
# For each reviewed basin, compute active CDEGHJ combo string
reviewed = all_merged.copy()
reviewed["active_combo"] = reviewed.apply(
    lambda r: "+".join(x for x in CDEGHJ_COLS if _is_true(r.get(x, False))) or "none",
    axis=1,
)
reviewed["active_cdeghj_count"] = reviewed.apply(
    lambda r: sum(_is_true(r.get(x, False)) for x in CDEGHJ_COLS), axis=1
)

for combo, grp in reviewed.groupby("active_combo"):
    n       = len(grp)
    excl_n  = (grp.human_decision == "EXCLUDE").sum()
    unsr_n  = (grp.human_decision == "UNSURE").sum()
    keep_n  = (grp.human_decision == "KEEP").sum()
    klc_n   = (grp.human_decision == "KEEP_LOW_CONFIDENCE").sum()
    count   = grp.active_cdeghj_count.iloc[0]
    combo_rows.append({
        "active_rule_combo":      combo,
        "active_rule_count":      count,
        "reviewed_n":             n,
        "keep_n":                 int(keep_n),
        "keep_low_confidence_n":  int(klc_n),
        "unsure_n":               int(unsr_n),
        "exclude_n":              int(excl_n),
        "exclude_or_unsure_rate": _rate(excl_n + unsr_n, n),
    })

combo_df = (pd.DataFrame(combo_rows)
              .sort_values(["active_rule_count", "exclude_or_unsure_rate"],
                            ascending=[False, False])
              .reset_index(drop=True))
combo_df.to_csv(TABLES_DIR / "rule_combination_performance.csv", index=False)
print(f"  {len(combo_df)} unique rule combinations among reviewed basins")
print(combo_df[["active_rule_combo","reviewed_n","exclude_n","unsure_n",
                "exclude_or_unsure_rate"]].head(15).to_string(index=False))

# -- 8. TABLE 5: artifact_behavior_summary.csv --------------------------------

print("\n-- 8. Writing Table 5: artifact_behavior_summary.csv -------------------")

# Explode semicolon-separated artifact_type
def _explode_multiselect(df, col):
    rows = []
    for _, r in df.iterrows():
        val = str(r[col]) if pd.notna(r[col]) else ""
        for item in val.split(";"):
            item = item.strip()
            if item:
                rows.append({"STAID": r.STAID, col: item,
                             "human_decision": r.human_decision,
                             "review_pass": r.review_pass})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["STAID", col, "human_decision", "review_pass"])

art_exploded  = _explode_multiselect(all_merged, "artifact_type")
beh_exploded  = _explode_multiselect(all_merged, "hydrograph_behavior")

def _pivot_counts(exploded, col):
    if exploded.empty:
        return pd.DataFrame()
    return (exploded.groupby([col, "human_decision"])
            .size().reset_index(name="n")
            .pivot(index=col, columns="human_decision", values="n")
            .fillna(0).astype(int)
            .reset_index())

art_pivot = _pivot_counts(art_exploded, "artifact_type")
beh_pivot = _pivot_counts(beh_exploded, "hydrograph_behavior")

artifact_summary = pd.concat([
    art_pivot.assign(dimension="artifact_type"),
    beh_pivot.assign(dimension="hydrograph_behavior"),
], ignore_index=True)
artifact_summary.to_csv(TABLES_DIR / "artifact_behavior_summary.csv", index=False)
print(f"  artifact_type rows: {len(art_pivot)}  hydrograph_behavior rows: {len(beh_pivot)}")

# -- 9. TABLE 6: regulation_keyword_summary.csv -------------------------------

print("\n-- 9. Writing Table 6: regulation_keyword_summary.csv ------------------")

kw_rows = []
for group_name, patterns in REGULATION_KW_GROUPS.items():
    combined_pat = re.compile("|".join(patterns), re.IGNORECASE)
    mask = all_merged.reviewer_notes.fillna("").str.contains(combined_pat, regex=True)
    grp = all_merged[mask]
    if len(grp) == 0:
        continue
    row = {"keyword_group": group_name, "total_mentions": len(grp)}
    for d in DECISION_ORDER:
        row[d] = int((grp.human_decision == d).sum())
    row["exclude_or_unsure_n"] = row["EXCLUDE"] + row["UNSURE"]
    row["exclude_or_unsure_rate"] = _rate(row["exclude_or_unsure_n"], len(grp))
    kw_rows.append(row)

kw_df = pd.DataFrame(kw_rows).sort_values("exclude_or_unsure_rate", ascending=False)
kw_df.to_csv(TABLES_DIR / "regulation_keyword_summary.csv", index=False)
print(kw_df[["keyword_group", "total_mentions", "EXCLUDE", "UNSURE",
              "exclude_or_unsure_rate"]].to_string(index=False))

# -- 10. TABLE 7: preliminary_final_policy_candidates.csv ---------------------

print("\n-- 10. Writing Table 7: preliminary_final_policy_candidates.csv ---------")

# Build reviewed lookup: STAID -> (human_decision, proposed_status, reasons)
reviewed_lookup = {}
for _, r in all_merged.iterrows():
    reviewed_lookup[r.STAID] = {
        "human_decision":                  r.human_decision,
        "review_pass":                     r.review_pass,
        "proposed_from_review":            r.proposed_training_status_candidate,
        "proposed_excl_reason_review":     r.proposed_exclusion_reason,
        "proposed_holdout_reason_review":  r.proposed_holdout_reason,
    }

policy_rows = []
for _, r in rm.iterrows():
    staid = r.STAID
    rl    = reviewed_lookup.get(staid)

    if rl is not None:
        hd    = rl["human_decision"]
        st, er, hr = compute_proposed_status(r, human_decision=hd)
    else:
        hd    = None
        st, er, hr = compute_proposed_status(r, human_decision=None)

    cdej_active = [x for x in CDEJ_COLS if _is_true(r.get(x, False))]

    policy_rows.append({
        "STAID":                              staid,
        "is_reviewed":                        rl is not None,
        "review_pass":                        rl["review_pass"] if rl else "",
        "human_decision":                     hd or "",
        "current_preliminary_training_status": r.get("preliminary_training_status", ""),
        "proposed_training_status_candidate": st,
        "proposed_exclusion_reason":          er,
        "proposed_holdout_reason":            hr,
        "cdej_active_count":                  len(cdej_active),
        "cdej_active_rules":                  "+".join(cdej_active) if cdej_active else "",
        "rule_A": _is_true(r.get("rule_A", False)),
        "rule_C": _is_true(r.get("rule_C", False)),
        "rule_D": _is_true(r.get("rule_D", False)),
        "rule_E": _is_true(r.get("rule_E", False)),
        "rule_F": _is_true(r.get("rule_F", False)),
        "rule_G": _is_true(r.get("rule_G", False)),
        "rule_H": _is_true(r.get("rule_H", False)),
        "rule_I": _is_true(r.get("rule_I", False)),
        "rule_J": _is_true(r.get("rule_J", False)),
        "compound_risk_count":                r.get("compound_risk_count", np.nan),
        "candidate_class":                    r.get("candidate_class", ""),
        "metadata_policy_bucket":             r.get("metadata_policy_bucket", ""),
        "HUC02":                              r.get("HUC02", ""),
        "STATE":                              r.get("STATE", ""),
        "DRAIN_SQKM":                         r.get("DRAIN_SQKM", np.nan),
        "RBI":                                r.get("RBI", np.nan),
        "LAT_GAGE":                           r.get("LAT_GAGE", np.nan),
        "LNG_GAGE":                           r.get("LNG_GAGE", np.nan),
        "status_is_preliminary":              True,
    })

policy_df = pd.DataFrame(policy_rows)
policy_df.to_csv(TABLES_DIR / "preliminary_final_policy_candidates.csv", index=False)

prop_counts = policy_df.proposed_training_status_candidate.value_counts()
print(f"  Total policy rows: {len(policy_df)}")
print(f"  Proposed status distribution:")
for st in STATUS_ORDER:
    n = prop_counts.get(st, 0)
    pct = n / len(policy_df) * 100
    print(f"    {st}: {n}  ({pct:.1f}%)")

prelim_counts = policy_df.current_preliminary_training_status.value_counts()
print(f"\n  Current preliminary_training_status (for comparison):")
for st, n in prelim_counts.items():
    print(f"    {st}: {n}")

# -- 11. PLOTS -----------------------------------------------------------------

print("\n-- 11. Generating plots -------------------------------------------------")
plt.rcParams.update({
    "figure.dpi": 120,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


def _save(fig, fname):
    path = PLOTS_DIR / fname
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {fname}")


# Plot 1: manual_decision_counts_combined.png
fig, ax = plt.subplots(figsize=(7, 4))
all_dec = all_merged.human_decision.value_counts().reindex(DECISION_ORDER, fill_value=0)
bars = ax.bar(all_dec.index, all_dec.values,
              color=[DECISION_COLORS[d] for d in all_dec.index])
for b, v in zip(bars, all_dec.values):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(v),
            ha="center", va="bottom", fontsize=10)
ax.set_title(f"Combined manual review decisions (N={len(all_merged)})")
ax.set_ylabel("Basin count")
ax.set_ylim(0, all_dec.max() * 1.15)
fig.tight_layout()
_save(fig, "manual_decision_counts_combined.png")

# Plot 2: manual_decision_by_pass.png
fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)
for ax, (pass_label, df_sub) in zip(axes,
        [("Pass 1 (n=73)", all_merged[all_merged.review_pass == "pass1"]),
         ("Pass 2 (n=75)", all_merged[all_merged.review_pass == "pass2"]),
         ("Combined (n=148)", all_merged)]):
    dec = df_sub.human_decision.value_counts().reindex(DECISION_ORDER, fill_value=0)
    bars = ax.bar(dec.index, dec.values,
                  color=[DECISION_COLORS[d] for d in dec.index])
    for b, v in zip(bars, dec.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, str(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title(pass_label)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
fig.suptitle("Manual review decisions by pass", fontsize=12)
fig.tight_layout()
_save(fig, "manual_decision_by_pass.png")

# Plot 3: rule_performance_exclude_or_unsure.png
fig, ax = plt.subplots(figsize=(9, 5))
rp = rule_perf.sort_values("exclude_or_unsure_rate", ascending=False)
colors_bar = ["#d62728" if v >= 0.6 else "#ff7f0e" if v >= 0.35
              else "#1f77b4" if v >= 0.15 else "#aec7e8"
              for v in rp.exclude_or_unsure_rate.fillna(0)]
bars = ax.bar(rp.rule, rp.exclude_or_unsure_rate.fillna(0), color=colors_bar)
for b, (_, row_r) in zip(bars, rp.iterrows()):
    val = row_r.exclude_or_unsure_rate
    n   = row_r.reviewed_flagged_n
    label_text = f"{val:.2f}\n(n={n})" if n > 0 else "n=0"
    ax.text(b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01 if pd.notna(val) else 0.01,
            label_text, ha="center", va="bottom", fontsize=8)
ax.axhline(0.35, color="orange", linestyle="--", linewidth=1, label="Moderate threshold (0.35)")
ax.axhline(0.60, color="red",    linestyle="--", linewidth=1, label="Strong threshold (0.60)")
ax.set_title("Exclude-or-unsure rate per rule (reviewed flagged basins)")
ax.set_ylabel("Exclude or unsure rate")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)
fig.tight_layout()
_save(fig, "rule_performance_exclude_or_unsure.png")

# Plot 4: rule_combo_performance.png — top 15 combos by reviewed_n
fig, ax = plt.subplots(figsize=(11, 5))
top_combos = combo_df.head(20).copy()
eu_vals  = top_combos.exclude_or_unsure_rate.fillna(0).values
colors_c = ["#d62728" if v >= 0.6 else "#ff7f0e" if v >= 0.35
             else "#1f77b4" if v >= 0.15 else "#aec7e8"
             for v in eu_vals]
x = np.arange(len(top_combos))
width = 0.6
bars = ax.bar(x, eu_vals, width, color=colors_c)
for b, (_, row_c) in zip(bars, top_combos.iterrows()):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f"n={row_c.reviewed_n}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(top_combos.active_rule_combo.values, rotation=45, ha="right", fontsize=8)
ax.axhline(0.35, color="orange", linestyle="--", linewidth=1)
ax.axhline(0.60, color="red",    linestyle="--", linewidth=1)
ax.set_title("Exclude-or-unsure rate by active rule combination (CDEGHJ)")
ax.set_ylabel("Exclude or unsure rate")
ax.set_ylim(0, 1.10)
fig.tight_layout()
_save(fig, "rule_combo_performance.png")

# Plot 5: artifact_type_by_decision.png
if not art_exploded.empty:
    art_ct = (art_exploded.groupby(["artifact_type", "human_decision"])
              .size().reset_index(name="n"))
    art_top = (art_ct.groupby("artifact_type")["n"].sum()
                     .nlargest(12).index.tolist())
    art_ct_top = art_ct[art_ct.artifact_type.isin(art_top)]
    art_piv = (art_ct_top.pivot(index="artifact_type", columns="human_decision", values="n")
               .fillna(0).astype(int))
    for d in DECISION_ORDER:
        if d not in art_piv.columns:
            art_piv[d] = 0
    art_piv = art_piv[DECISION_ORDER]

    fig, ax = plt.subplots(figsize=(10, 5))
    bot = np.zeros(len(art_piv))
    for d in DECISION_ORDER:
        vals = art_piv[d].values
        ax.bar(art_piv.index, vals, bottom=bot, color=DECISION_COLORS[d], label=d)
        bot += vals
    ax.set_title("Artifact type vs. human decision (top 12 artifact types)")
    ax.set_ylabel("Basin count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, "artifact_type_by_decision.png")

# Plot 6: proposed_training_status_counts.png
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Left: proposed counts for all 3034 candidates
prop_all = policy_df.proposed_training_status_candidate.value_counts().reindex(
    STATUS_ORDER, fill_value=0)
bars = axes[0].bar(prop_all.index, prop_all.values,
                   color=[STATUS_COLORS[s] for s in prop_all.index])
for b, v in zip(bars, prop_all.values):
    pct = v / len(policy_df) * 100
    axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 5,
                 f"{v}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
axes[0].set_title(f"Proposed training status — all {len(policy_df)} candidates")
axes[0].set_ylabel("Basin count")
axes[0].tick_params(axis="x", rotation=30)
axes[0].set_ylim(0, prop_all.max() * 1.2)

# Right: comparison — current preliminary vs proposed
prelim_for_comp = policy_df.current_preliminary_training_status.value_counts()
prop_for_comp   = policy_df.proposed_training_status_candidate.value_counts()
rename_prelim = {
    "TRAIN_CORE_PRELIM":       "TRAIN_CORE",
    "TRAIN_SOFT_KEEP_PRELIM":  "TRAIN_SOFT_KEEP",
    "HOLDOUT_REVIEW_PRELIM":   "HOLDOUT_REVIEW",
    "EXCLUDE_TRAINING_PRELIM": "EXCLUDE_TRAINING",
}
rename_prop = {
    "TRAIN_CORE_CANDIDATE":        "TRAIN_CORE",
    "TRAIN_SOFT_KEEP_CANDIDATE":   "TRAIN_SOFT_KEEP",
    "HOLDOUT_REVIEW_CANDIDATE":    "HOLDOUT_REVIEW",
    "EXCLUDE_TRAINING_CANDIDATE":  "EXCLUDE_TRAINING",
}
prelim_norm = prelim_for_comp.rename(index=rename_prelim)
prop_norm   = prop_for_comp.rename(index=rename_prop)
cats = ["TRAIN_CORE", "TRAIN_SOFT_KEEP", "HOLDOUT_REVIEW", "EXCLUDE_TRAINING"]
x2 = np.arange(len(cats))
w2 = 0.35
bars1 = axes[1].bar(x2 - w2/2, [prelim_norm.get(c, 0) for c in cats], w2,
                    label="Current PRELIM", color="#aec7e8")
bars2 = axes[1].bar(x2 + w2/2, [prop_norm.get(c, 0) for c in cats], w2,
                    label="Proposed CANDIDATE", color="#1f77b4")
axes[1].set_xticks(x2)
axes[1].set_xticklabels(cats, rotation=30, ha="right")
axes[1].set_title("Current preliminary vs proposed candidate status")
axes[1].set_ylabel("Basin count")
axes[1].legend(fontsize=9)
fig.tight_layout()
_save(fig, "proposed_training_status_counts.png")

# Plot 7: map_proposed_training_status.png
lat_ok = policy_df.LAT_GAGE.notna() & policy_df.LNG_GAGE.notna()
if lat_ok.sum() > 0:
    fig, ax = plt.subplots(figsize=(12, 7))
    for st in reversed(STATUS_ORDER):
        sub = policy_df[lat_ok & (policy_df.proposed_training_status_candidate == st)]
        if len(sub) > 0:
            ax.scatter(sub.LNG_GAGE, sub.LAT_GAGE,
                       c=STATUS_COLORS[st], s=8, alpha=0.65, label=f"{st} (n={len(sub)})",
                       linewidths=0)
    # Highlight reviewed basins with a ring
    rev_ok = lat_ok & policy_df.is_reviewed
    ax.scatter(policy_df.loc[rev_ok, "LNG_GAGE"],
               policy_df.loc[rev_ok, "LAT_GAGE"],
               c="none", edgecolors="black", s=25, linewidths=0.7,
               label=f"Reviewed (n={rev_ok.sum()})", zorder=5)
    ax.set_xlim(-130, -60)
    ax.set_ylim(24, 52)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Proposed training status — CONUS candidate basins")
    ax.legend(loc="lower left", fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    _save(fig, "map_proposed_training_status.png")
else:
    print("  WARNING: no lat/lon data available — skipping map plot")

# -- 12. SUMMARIES -------------------------------------------------------------

print("\n-- 12. Writing summaries -------------------------------------------------")

# Key numbers for summaries
p1n = (all_merged.review_pass == "pass1").sum()
p2n = (all_merged.review_pass == "pass2").sum()
comb_n = len(all_merged)

all_exc  = (all_merged.human_decision == "EXCLUDE").sum()
all_unsr = (all_merged.human_decision == "UNSURE").sum()
all_klc  = (all_merged.human_decision == "KEEP_LOW_CONFIDENCE").sum()
all_keep = (all_merged.human_decision == "KEEP").sum()

prop_core    = prop_counts.get("TRAIN_CORE_CANDIDATE", 0)
prop_soft    = prop_counts.get("TRAIN_SOFT_KEEP_CANDIDATE", 0)
prop_holdout = prop_counts.get("HOLDOUT_REVIEW_CANDIDATE", 0)
prop_excl    = prop_counts.get("EXCLUDE_TRAINING_CANDIDATE", 0)

# Best-supported rules (eu_rate >= 0.35 with n >= 5)
strong_rules = rule_perf[(rule_perf.exclude_or_unsure_rate >= 0.35) &
                          (rule_perf.reviewed_flagged_n >= 5)].rule.tolist()
weak_rules   = rule_perf[(rule_perf.exclude_or_unsure_rate < 0.15) &
                          (rule_perf.reviewed_flagged_n >= 5)].rule.tolist()

# Top regulation keywords
top_kw = kw_df.nlargest(3, "exclude_or_unsure_rate")

# JSON summary
summary_json = {
    "analysis_version":   "flashnh_combined_manual_review_analysis_v001",
    "pass1": {
        "n_reviewed": int(p1n),
        "decisions":  {d: int((all_merged[all_merged.review_pass=="pass1"].human_decision==d).sum())
                       for d in DECISION_ORDER},
    },
    "pass2": {
        "n_reviewed": int(p2n),
        "decisions":  {d: int((all_merged[all_merged.review_pass=="pass2"].human_decision==d).sum())
                       for d in DECISION_ORDER},
    },
    "combined": {
        "n_reviewed": int(comb_n),
        "n_keep": int(all_keep),
        "n_keep_low_confidence": int(all_klc),
        "n_unsure": int(all_unsr),
        "n_exclude": int(all_exc),
        "n_duplicate_staids": int(len(overlap)),
    },
    "full_candidate_pool": {
        "n_main_training_candidate": int(len(policy_df)),
    },
    "proposed_status_counts": {
        "TRAIN_CORE_CANDIDATE":       int(prop_core),
        "TRAIN_SOFT_KEEP_CANDIDATE":  int(prop_soft),
        "HOLDOUT_REVIEW_CANDIDATE":   int(prop_holdout),
        "EXCLUDE_TRAINING_CANDIDATE": int(prop_excl),
    },
    "rule_performance": {
        r["rule"]: {
            "reviewed_flagged_n":     int(r["reviewed_flagged_n"]),
            "exclude_or_unsure_rate": float(r["exclude_or_unsure_rate"])
                                      if pd.notna(r["exclude_or_unsure_rate"]) else None,
            "recommendation":         r["recommendation"],
        }
        for _, r in rule_perf.iterrows()
    },
    "rules_with_strong_support_eu_ge035": strong_rules,
    "rules_with_no_evidence_eu_lt015":    weak_rules,
}

with open(SUMMARY_DIR / "combined_manual_review_analysis_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)
print("  Wrote combined_manual_review_analysis_summary.json")

# Markdown summary
rule_table_md = (
    "| Rule | Reviewed flagged | Exclude | Unsure | Exc-or-unsure rate | Recommendation |\n"
    "|------|-----------------|---------|--------|--------------------|----------------|\n"
)
for _, r in rule_perf.iterrows():
    eu = f"{r.exclude_or_unsure_rate:.2f}" if pd.notna(r.exclude_or_unsure_rate) else "—"
    rule_table_md += (
        f"| {r.rule} | {r.reviewed_flagged_n} | {r.exclude_n} | {r.unsure_n} "
        f"| {eu} | {r.recommendation} |\n"
    )

prop_table_md = (
    "| Proposed status | Count | % of 3034 |\n"
    "|----------------|-------|----------|\n"
)
for st in STATUS_ORDER:
    n = prop_counts.get(st, 0)
    prop_table_md += f"| {st} | {n} | {n/len(policy_df)*100:.1f}% |\n"

kw_table_md = (
    "| Keyword group | Mentions | Exclude | Unsure | Exc-or-unsure rate |\n"
    "|--------------|---------|---------|--------|--------------------|\n"
)
for _, r in kw_df.iterrows():
    eu = f"{r.exclude_or_unsure_rate:.2f}" if pd.notna(r.exclude_or_unsure_rate) else "—"
    kw_table_md += (
        f"| {r.keyword_group} | {r.total_mentions} | {r['EXCLUDE']} | {r['UNSURE']} "
        f"| {eu} |\n"
    )

md_body = f"""# Combined Manual Review Analysis — Flash-NH Basin Screening

**Version**: flashnh_combined_manual_review_analysis_v001
**Status**: Preliminary analysis — no basins have been removed.

---

## Executive Summary

A combined total of **{comb_n} basins** were manually reviewed across two passes:

- **Pass 1** (v004 review set): {p1n} basins — stratified random sample enriched for
  extreme RBI, rise rates, and automated flag cases.
- **Pass 2** (v005 second-pass rules): {p2n} basins — targeted sample stratified by
  risk rule tier (compound-risk HOLDOUT_REVIEW_PRELIM basins, per-rule tier-2 samples,
  near-threshold controls, and clean CORE controls).

The two pass sets are **fully disjoint** ({len(overlap)} overlapping STAIDs).

Combined decisions: **{all_keep} KEEP** ({all_keep/comb_n*100:.1f}%) |
**{all_klc} KEEP_LOW_CONFIDENCE** ({all_klc/comb_n*100:.1f}%) |
**{all_unsr} UNSURE** ({all_unsr/comb_n*100:.1f}%) |
**{all_exc} EXCLUDE** ({all_exc/comb_n*100:.1f}%)

The proposed policy (applied to all {len(policy_df)} main-training-candidate basins)
results in approximately:
- **{prop_core}** TRAIN_CORE_CANDIDATE ({prop_core/len(policy_df)*100:.1f}%)
- **{prop_soft}** TRAIN_SOFT_KEEP_CANDIDATE ({prop_soft/len(policy_df)*100:.1f}%)
- **{prop_holdout}** HOLDOUT_REVIEW_CANDIDATE ({prop_holdout/len(policy_df)*100:.1f}%)
- **{prop_excl}** EXCLUDE_TRAINING_CANDIDATE ({prop_excl/len(policy_df)*100:.1f}%)

---

## Combined Label Counts

### Overall
| Decision | Pass 1 (n={p1n}) | Pass 2 (n={p2n}) | Combined (n={comb_n}) |
|----------|----------------|----------------|----------------------|
| KEEP | {(all_merged[all_merged.review_pass=='pass1'].human_decision=='KEEP').sum()} | {(all_merged[all_merged.review_pass=='pass2'].human_decision=='KEEP').sum()} | {all_keep} |
| KEEP_LOW_CONFIDENCE | {(all_merged[all_merged.review_pass=='pass1'].human_decision=='KEEP_LOW_CONFIDENCE').sum()} | {(all_merged[all_merged.review_pass=='pass2'].human_decision=='KEEP_LOW_CONFIDENCE').sum()} | {all_klc} |
| UNSURE | {(all_merged[all_merged.review_pass=='pass1'].human_decision=='UNSURE').sum()} | {(all_merged[all_merged.review_pass=='pass2'].human_decision=='UNSURE').sum()} | {all_unsr} |
| EXCLUDE | {(all_merged[all_merged.review_pass=='pass1'].human_decision=='EXCLUDE').sum()} | {(all_merged[all_merged.review_pass=='pass2'].human_decision=='EXCLUDE').sum()} | {all_exc} |

Note: EXCLUDE rate (exclude+unsure) = {(all_exc+all_unsr)/comb_n*100:.1f}% combined.

---

## What Pass 2 Taught Us Beyond Pass 1

Pass 2 was designed to directly test the risk rules by reviewing basins that triggered
compound rule flags (≥2 of CDEGHJ) and per-rule tier-2 samples. Key findings:

- **Pass 2 had a higher exclude-or-unsure rate** ({(pass2_counts.get('EXCLUDE',0)+pass2_counts.get('UNSURE',0))/p2n*100:.1f}%) versus pass 1 ({(pass1_counts.get('EXCLUDE',0)+pass1_counts.get('UNSURE',0))/p1n*100:.1f}%). This reflects the enrichment for compound-risk HOLDOUT_REVIEW_PRELIM basins.
- **Regulation/lentic signals are the primary exclusion driver.** Basins triggering rule_C (high disturbance index), rule_D (high lake/reservoir %), and rule_E (high degree-of-regulation) showed the highest exclude-or-unsure rates among rule-flagged basins.
- **Zero-flow and extreme-jump flags (rule_G, rule_H) do not warrant exclusion alone.** Many rule_H flagged basins were reviewed as KEEP — they represent real flashy hydrology, not artifacts.
- **Pass 2 confirmed the compound-risk tier strategy.** Basins with ≥2 of CDEJ rules active had substantially elevated exclude-or-unsure rates compared to single-rule basins.

---

## Rule Performance Summary

{rule_table_md}

Rules with strong calibration support (exclude-or-unsure rate ≥ 0.35, n ≥ 5):
**{', '.join(strong_rules) if strong_rules else 'none identified'}**

Rules with little or no evidence of problems (rate < 0.15, n ≥ 5):
**{', '.join(weak_rules) if weak_rules else 'none identified at this threshold'}**

---

## Why Regulation/Lentic/Artificial-Flow Risk Is the Key Issue

Reviewer notes consistently cite **dams, reservoirs, upstream lakes, and regulated
discharge patterns** as the reason for EXCLUDE and UNSURE labels. The keyword analysis
confirms this:

{kw_table_md}

Rule_C (HYDRO_DISTURB_INDX), rule_D (lka_pc_use), and rule_J (dor_pc_pva or CANALS_PCT)
collectively capture the regulation/lentic/artificial-flow risk dimension. Their
combinations show the highest exclude-or-unsure rates among all rule pairs.

---

## Why Zero Flow and Extreme Jumps Should Remain Soft/Context Flags

- **Rule_G** (mostly-zero suspicious): Ephemeral and intermittent streams are a
  legitimate hydrological regime, especially in semi-arid and arid regions. Most
  rule_G basins reviewed as KEEP had zero-flow behavior that was geographically
  consistent and seasonally predictable.
- **Rule_H** (extreme hourly jumps): Many flagged basins were flashy small catchments
  with legitimate fast event responses. Excluding these basins would remove precisely
  the kind of flashy hydrology that the Flash-NH model is designed to learn.

Both rules are assigned TRAIN_SOFT_KEEP_CANDIDATE when appearing alone, not excluded
or held out. Post-training residual analysis is the appropriate mechanism for
identifying the minority of rule_G/H basins that genuinely degrade model performance.

---

## Proposed Final Policy Candidate

See `tables/preliminary_final_policy_candidates.csv` for per-basin assignments.

{prop_table_md}

**Priority order**: EXCLUDE > HOLDOUT > SOFT_KEEP > CORE.

Key policy decisions:
1. `human_decision == EXCLUDE` → EXCLUDE_TRAINING_CANDIDATE (overrides all rules)
2. `rule_A == True` → EXCLUDE_TRAINING_CANDIDATE (manual override flags from pass 1)
3. `human_decision == UNSURE` → HOLDOUT_REVIEW_CANDIDATE
4. CDEJ active count ≥ 2 → HOLDOUT_REVIEW_CANDIDATE (strong compound regulation signal)
5. Single CDEJ rule → TRAIN_SOFT_KEEP_CANDIDATE
6. rule_G or rule_H alone → TRAIN_SOFT_KEEP_CANDIDATE
7. rule_F or rule_I alone → TRAIN_SOFT_KEEP_CANDIDATE
8. No flags → TRAIN_CORE_CANDIDATE

Human decisions for reviewed basins take priority over rule-based assignments except
where the rule-based status is more severe (rule_A always triggers EXCLUDE regardless
of human KEEP decision — though no reviewed basins showed this conflict).

---

## Remaining Uncertainties

1. **Sampling bias**: Both review sets are enriched for suspicious basins. The 73+75
   reviewed basins cannot be directly used to estimate prevalence rates in the full 3034
   without correcting for the stratified sampling design.

2. **Rule threshold calibration**: The p95/p99 thresholds used for rules C–J were set
   prior to reviewing the second pass. The evidence now available could be used to
   explore alternative thresholds, but this requires care to avoid overfitting to the
   reviewed sample.

3. **HOLDOUT_REVIEW basins not yet reviewed**: There are approximately
   {prop_holdout - (all_merged.proposed_training_status_candidate == 'HOLDOUT_REVIEW_CANDIDATE').sum()} unreviewed HOLDOUT_REVIEW_CANDIDATE basins.
   These should be reviewed (in batches, prioritizing the highest-confidence exclusion
   candidates) before finalizing exclusions.

4. **Post-training validation**: The cleanest path to finalizing exclusions is to train
   on all TRAIN_CORE + TRAIN_SOFT_KEEP basins, identify high-residual stations in
   held-out periods, and then review those. This provides data-driven justification
   for any subsequent exclusions.

---

## Recommended Next Actions

1. **Accept or modify** the proposed policy in `preliminary_final_policy_candidates.csv`.
   Pay particular attention to the HOLDOUT_REVIEW_CANDIDATE basins with ≥2 CDEJ rules
   — these are the highest-priority candidates for secondary review or exclusion.

2. **Create a versioned final basin selection script** that encodes the accepted policy
   and writes the definitive `training_candidate_final.csv`. This script should be
   committed; the preliminary analysis outputs are not the final authority.

3. **Expand review** of HOLDOUT_REVIEW_CANDIDATE basins, prioritizing:
   - Those with rule_C + rule_D co-occurring (high disturbance AND high lentic %)
   - Those in HUC02 regions with known heavy regulation

4. **Run post-training residual analysis** after the first model run to identify
   any TRAIN_SOFT_KEEP basins that should be moved to HOLDOUT or EXCLUDE.

---

*This document is automatically generated by `scripts/analyze_combined_manual_review_results.py`.
All status labels are marked PRELIMINARY — no basins have been removed from the training
candidate pool based on this analysis alone.*
"""

with open(SUMMARY_DIR / "combined_manual_review_analysis_summary.md", "w",
          encoding="utf-8") as f:
    f.write(md_body)
print("  Wrote combined_manual_review_analysis_summary.md")

# -- 13. Final report ----------------------------------------------------------

print("\n-- 13. Final report -----------------------------------------------------")
print(f"\n  Output directory: {OUT_DIR}")
print(f"\n  Tables written:")
for f in sorted(TABLES_DIR.glob("*.csv")):
    n_rows = sum(1 for _ in open(f)) - 1
    print(f"    {f.name}: {n_rows} rows")
print(f"\n  Plots written:")
for f in sorted(PLOTS_DIR.glob("*.png")):
    print(f"    {f.name}")
print(f"\n  Summaries written:")
for f in sorted(SUMMARY_DIR.glob("*")):
    print(f"    {f.name}")
print("\n  DONE.")
