"""
Consolidated basin-filtering funnel report for Flash-NH.

Reads only existing local files. Does not download data or rerun processing.
Outputs to reports/flashnh_basin_filtering_consolidation_v001/.
"""

import ast
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "flashnh_basin_filtering_consolidation_v001"

SUMMARIES_DIR = OUT_DIR / "summaries"
TABLES_DIR = OUT_DIR / "tables"
PLOTS_DIR = OUT_DIR / "plots"

for d in [SUMMARIES_DIR, TABLES_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MISSING = []  # accumulated warnings about missing optional files


def _warn(msg: str) -> None:
    MISSING.append(msg)
    print(f"  [WARN] {msg}", file=sys.stderr)


def _load_json(path: Path):
    if not path.exists():
        _warn(f"Missing optional file: {path.relative_to(REPO_ROOT)}")
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(path: Path, **kw) -> pd.DataFrame | None:
    if not path.exists():
        _warn(f"Missing optional file: {path.relative_to(REPO_ROOT)}")
        return None
    return pd.read_csv(path, **kw)


def _parse_flag_list(s) -> list[str]:
    """Parse a stringified Python list or JSON array into a list of strings."""
    if pd.isna(s) or s in ("", "[]", "nan"):
        return []
    try:
        val = ast.literal_eval(str(s))
        return [str(x) for x in val] if isinstance(val, list) else []
    except Exception:
        return []


def _pct(n, d):
    return round(100 * n / d, 1) if d else None


def _fmt_pct(n, d):
    p = _pct(n, d)
    return f"{p}%" if p is not None else "—"


# ---------------------------------------------------------------------------
# Load source files
# ---------------------------------------------------------------------------

print("Loading source files ...")

screening_json = _load_json(
    REPO_ROOT / "reports/flashnh_basin_screening_v001/candidate_basin_screening_summary.json"
)
eligibility_json = _load_json(
    REPO_ROOT / "reports/flashnh_usgs_coverage_eligibility_v001/usgs_coverage_eligibility_summary.json"
)
rbi_json = _load_json(
    REPO_ROOT / "reports/flashnh_usgs_rbi_screening_wy2024_v001/usgs_rbi_screening_summary.json"
)
metrics_json = _load_json(
    REPO_ROOT / "reports/flashnh_wy2024_streamflow_metrics_v002/summaries/wy2024_streamflow_metrics_summary.json"
)
pilot_json = _load_json(
    REPO_ROOT / "reports/flashnh_wy2024_pilot_selection_v001/summaries/pilot_selection_summary.json"
)
review_json = _load_json(
    REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v003_diverse/summaries/review_set_summary.json"
)
metrics_df = _load_csv(
    REPO_ROOT / "reports/flashnh_wy2024_streamflow_metrics_v002/tables/wy2024_streamflow_metrics.csv",
    low_memory=False,
)
review_template_df = _load_csv(
    REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v003_diverse/tables/human_review_template.csv",
    low_memory=False,
)
manual_labels_df = _load_csv(
    REPO_ROOT / "reports/flashnh_hydrograph_review_cards_v003_diverse/manual_review_labels.csv"
)

print("  Done.\n")

# ---------------------------------------------------------------------------
# Extract key counts (hard-coded fallbacks from decision memo where files missing)
# ---------------------------------------------------------------------------

N_TOTAL = 9008
N_AREA_FILTERED = 5836
N_BFI_EXPLORATORY = 2130  # BFI<=40 exploratory subset
N_ELIGIBLE_SCREENING_WY = 3647
N_WY2024_PROCESSED = 3324
N_HARD_QC_EXCLUDED = 279
N_USABLE = 3045

if screening_json:
    N_TOTAL = screening_json.get("total_basins", N_TOTAL)
    N_AREA_FILTERED = screening_json.get("basins_after_area_filter", N_AREA_FILTERED)
    N_BFI_EXPLORATORY = screening_json.get("basins_after_area_bfi_filter", N_BFI_EXPLORATORY)

if eligibility_json:
    N_ELIGIBLE_SCREENING_WY = eligibility_json.get("basin_count_eligible_screening_wy", N_ELIGIBLE_SCREENING_WY)

if rbi_json:
    sc = rbi_json.get("status_counts", {})
    rbi_ready = sc.get("RBI_READY", 3092)
    partial = sc.get("PARTIAL_USABLE", 232)
    N_WY2024_PROCESSED = metrics_json.get("total_basins", rbi_ready + partial) if metrics_json else rbi_ready + partial

if metrics_json:
    N_WY2024_PROCESSED = metrics_json.get("total_basins", N_WY2024_PROCESSED)
    N_HARD_QC_EXCLUDED = metrics_json.get("hard_qc_exclusions_count", N_HARD_QC_EXCLUDED)
    N_USABLE = N_WY2024_PROCESSED - N_HARD_QC_EXCLUDED

if pilot_json:
    ub = pilot_json.get("usable_basins", {})
    N_USABLE = ub.get("total", N_USABLE)

N_REVIEW_CARDS = review_json.get("total_basins", 80) if review_json else 80

# Candidate class counts (usable universe)
CLASS_COUNTS: dict[str, int] = {}
if pilot_json:
    CLASS_COUNTS = pilot_json.get("usable_basins", {}).get("by_class", {})
elif metrics_json:
    for row in metrics_json.get("candidate_class_counts", []):
        if row["candidate_class"] != "EXCLUDE_HARD_QC":
            CLASS_COUNTS[row["candidate_class"]] = row["count"]

# Hard QC flag counts
HARD_QC_FLAGS: dict[str, int] = {}
if metrics_json:
    HARD_QC_FLAGS = metrics_json.get("hard_qc_flag_counts", {})

# Context flag counts
CONTEXT_FLAGS: dict[str, int] = {}
if metrics_json:
    CONTEXT_FLAGS = metrics_json.get("context_flag_counts", {})

# Universe distributions from pilot_json
UNIVERSE = pilot_json.get("universe_summary", {}) if pilot_json else {}
BY_AREA = UNIVERSE.get("by_area_bin", {})
BY_BFI = UNIVERSE.get("by_bfi_bin", {})
BY_STATE = UNIVERSE.get("by_state", {})
BY_HUC02 = UNIVERSE.get("by_huc02", {})

# RBI per class
RBI_BY_CLASS = {
    k.replace("rbi_", ""): v
    for k, v in (pilot_json or {}).get("universe_summary", {}).items()
    if k.startswith("rbi_") and isinstance(v, dict)
}

# ---------------------------------------------------------------------------
# 1. Filtering funnel table
# ---------------------------------------------------------------------------

print("Building filtering funnel table ...")

funnel_rows = [
    dict(stage_order=0,  stage_name="Original GAGES-II/CAMELSH universe",
         count=N_TOTAL,              filter_type="context",
         notes="Full USGS-monitored basin catalog (GAGES-II)"),
    dict(stage_order=1,  stage_name="Area-filtered (1–1,000 km²)",
         count=N_AREA_FILTERED,      filter_type="hard",
         notes="Excludes tiny hillslopes (<1 km²) and large routed rivers (>1,000 km²)"),
    dict(stage_order=2,  stage_name="BFI ≤ 40 exploratory subset",
         count=N_BFI_EXPLORATORY,    filter_type="exploratory",
         notes="Exploratory only. Not a final hard filter. USGS audit ran on all area-filtered basins."),
    dict(stage_order=3,  stage_name="USGS coverage eligible (ELIGIBLE_SCREENING_WY)",
         count=N_ELIGIBLE_SCREENING_WY, filter_type="derived",
         notes="Area-filtered basins with USGS param 00060 overlapping screening WY2024 window"),
    dict(stage_order=4,  stage_name="WY2024 streamflow metrics processed",
         count=N_WY2024_PROCESSED,   filter_type="derived",
         notes="Successful hourly discharge retrieval and RBI computation for screening WY"),
    dict(stage_order=5,  stage_name="Hard-QC excluded",
         count=N_HARD_QC_EXCLUDED,   filter_type="hard",
         notes="Completeness <90%, severe negative flow, Q50≈0, or RBI uncomputable"),
    dict(stage_order=6,  stage_name="Hard-QC-passing usable universe",
         count=N_USABLE,             filter_type="hard",
         notes="Recommended carry-forward set for model training and evaluation"),
    dict(stage_order=7,  stage_name="Manual review card set",
         count=N_REVIEW_CARDS,       filter_type="manual",
         notes="Diverse sample for human hydrograph inspection; subset of usable universe"),
]

funnel_df = pd.DataFrame(funnel_rows)
funnel_df["percent_of_original"] = funnel_df["count"].apply(
    lambda n: round(100 * n / N_TOTAL, 1)
)

pct_prev = [None]
prev = N_TOTAL
for _, row in funnel_df.iterrows():
    if row["stage_order"] in (0, 2, 5, 7):
        pct_prev.append(None)  # placeholder; fill below
    else:
        pct_prev.append(round(100 * row["count"] / prev, 1))
    if row["stage_order"] not in (2, 5, 7):  # these are subsets/exclusions, don't update prev
        prev = row["count"]
pct_prev.pop(0)

# Re-do percent_of_previous_stage properly with explicit logic
def _pct_of_prev(stage_order, count):
    mapping = {
        0: None,
        1: (count, N_TOTAL),
        2: (count, N_AREA_FILTERED),
        3: (count, N_AREA_FILTERED),
        4: (count, N_ELIGIBLE_SCREENING_WY),
        5: (count, N_WY2024_PROCESSED),
        6: (count, N_WY2024_PROCESSED),
        7: (count, N_USABLE),
    }
    pair = mapping.get(stage_order)
    if pair is None:
        return None
    n, d = pair
    return round(100 * n / d, 1) if d else None

funnel_df["percent_of_previous_stage"] = funnel_df.apply(
    lambda r: _pct_of_prev(r["stage_order"], r["count"]), axis=1
)

funnel_df = funnel_df[["stage_order", "stage_name", "count", "percent_of_original",
                         "percent_of_previous_stage", "filter_type", "notes"]]
funnel_df.to_csv(TABLES_DIR / "filtering_funnel_counts.csv", index=False)
print(f"  Wrote filtering_funnel_counts.csv ({len(funnel_df)} rows)")

# ---------------------------------------------------------------------------
# 2. Stage definitions table
# ---------------------------------------------------------------------------

stage_defs = [
    dict(stage_order=0, stage_name="Original GAGES-II/CAMELSH universe",
         input_from="—",
         filter_description="No filter applied; full basin catalog",
         filter_type="context",
         count=N_TOTAL,
         notes="Starting point for all downstream filtering"),
    dict(stage_order=1, stage_name="Area filter",
         input_from="Stage 0",
         filter_description="Drainage area 1–1,000 km²",
         filter_type="hard",
         count=N_AREA_FILTERED,
         notes="Headwater to mid-size basins; excludes hillslopes and large rivers"),
    dict(stage_order=2, stage_name="BFI exploratory subset",
         input_from="Stage 1 (parallel branch)",
         filter_description="BFI_AVE ≤ 40 (lower quartile of area-filtered basins)",
         filter_type="exploratory",
         count=N_BFI_EXPLORATORY,
         notes="Used to scope early USGS queries; NOT a hard exclusion criterion. "
               "USGS eligibility audit covers all area-filtered basins."),
    dict(stage_order=3, stage_name="USGS coverage eligibility",
         input_from="Stage 1",
         filter_description="Metadata confirms USGS param 00060 overlaps WY2024 screening window",
         filter_type="derived",
         count=N_ELIGIBLE_SCREENING_WY,
         notes="ELIGIBLE_SCREENING_WY class; excludes HISTORICAL_ONLY, NO_00060, INVALID_SITE"),
    dict(stage_order=4, stage_name="WY2024 streamflow metrics",
         input_from="Stage 3",
         filter_description="Successful hourly IV retrieval; RBI and 40+ metrics computed",
         filter_type="derived",
         count=N_WY2024_PROCESSED,
         notes="RBI_READY and PARTIAL_USABLE basins; screening window Oct 2023–Sep 2024"),
    dict(stage_order=5, stage_name="Hard-QC exclusions",
         input_from="Stage 4 (removed subset)",
         filter_description="Completeness <90%, severe negative flow, Q50≈0, or RBI uncomputable",
         filter_type="hard",
         count=N_HARD_QC_EXCLUDED,
         notes="Permanent removal from downstream analysis; not reversible for WY2024"),
    dict(stage_order=6, stage_name="Hard-QC-passing usable universe",
         input_from="Stage 4 minus Stage 5",
         filter_description="All basins with no hard-QC failure flags",
         filter_type="hard",
         count=N_USABLE,
         notes="Recommended training+evaluation set; candidate classes retained as metadata"),
    dict(stage_order=7, stage_name="Manual review card set",
         input_from="Stage 6 (diverse sample)",
         filter_description="Stratified 80-basin sample for human hydrograph inspection",
         filter_type="manual",
         count=N_REVIEW_CARDS,
         notes="Includes all candidate classes; not a hard filter; calibration exercise"),
]
pd.DataFrame(stage_defs).to_csv(TABLES_DIR / "stage_definitions.csv", index=False)
print(f"  Wrote stage_definitions.csv")

# ---------------------------------------------------------------------------
# 3. Candidate class counts
# ---------------------------------------------------------------------------

if CLASS_COUNTS:
    class_df = pd.DataFrame([
        {"candidate_class": k, "count": v,
         "percent_of_usable": round(100 * v / N_USABLE, 1),
         "percent_of_processed": round(100 * v / N_WY2024_PROCESSED, 1)}
        for k, v in CLASS_COUNTS.items()
    ]).sort_values("count", ascending=False)
    # add EXCLUDE_HARD_QC row
    excl_row = pd.DataFrame([{
        "candidate_class": "EXCLUDE_HARD_QC",
        "count": N_HARD_QC_EXCLUDED,
        "percent_of_usable": None,
        "percent_of_processed": round(100 * N_HARD_QC_EXCLUDED / N_WY2024_PROCESSED, 1),
    }])
    class_df = pd.concat([class_df, excl_row], ignore_index=True)
    class_df.to_csv(TABLES_DIR / "candidate_class_counts.csv", index=False)
    print(f"  Wrote candidate_class_counts.csv")

# ---------------------------------------------------------------------------
# 4–6. RBI bin, area bin, BFI bin from metrics CSV or pilot summary
# ---------------------------------------------------------------------------

# RBI bins from review cards summary (best granularity available)
if review_json:
    rbi_bin_data = review_json.get("by_rbi_bin", {})
    if rbi_bin_data:
        rbi_bin_df = pd.DataFrame([
            {"rbi_bin": k, "count_in_review_set": v,
             "note": "counts from 80-basin review set, not full universe"}
            for k, v in rbi_bin_data.items()
        ])
        rbi_bin_df.to_csv(TABLES_DIR / "rbi_bin_counts.csv", index=False)
        print(f"  Wrote rbi_bin_counts.csv (review set; full-universe bins from metrics CSV)")

# Area bins
if BY_AREA:
    area_df = pd.DataFrame([
        {"area_bin": k, "count": v,
         "percent_of_processed": round(100 * v / N_WY2024_PROCESSED, 1)}
        for k, v in BY_AREA.items()
    ])
    area_df.to_csv(TABLES_DIR / "area_bin_counts.csv", index=False)
    print(f"  Wrote area_bin_counts.csv")

# BFI bins
if BY_BFI:
    bfi_df = pd.DataFrame([
        {"bfi_bin": k, "count": v,
         "percent_of_processed": round(100 * v / N_WY2024_PROCESSED, 1)}
        for k, v in BY_BFI.items()
    ])
    bfi_df.to_csv(TABLES_DIR / "bfi_bin_counts.csv", index=False)
    print(f"  Wrote bfi_bin_counts.csv")

# HUC02
if BY_HUC02:
    huc_df = pd.DataFrame([
        {"huc02": k, "count": v,
         "percent_of_processed": round(100 * v / N_WY2024_PROCESSED, 1)}
        for k, v in sorted(BY_HUC02.items(), key=lambda x: -x[1])
    ])
    huc_df.to_csv(TABLES_DIR / "huc02_counts.csv", index=False)
    print(f"  Wrote huc02_counts.csv")

# State
if BY_STATE:
    state_df = pd.DataFrame([
        {"state": k, "count": v,
         "percent_of_processed": round(100 * v / N_WY2024_PROCESSED, 1)}
        for k, v in sorted(BY_STATE.items(), key=lambda x: -x[1])
    ])
    state_df.to_csv(TABLES_DIR / "state_counts.csv", index=False)
    print(f"  Wrote state_counts.csv")

# Context flags
if CONTEXT_FLAGS:
    ctx_df = pd.DataFrame([
        {"context_flag": k, "count": v,
         "percent_of_usable": round(100 * v / N_USABLE, 1)}
        for k, v in sorted(CONTEXT_FLAGS.items(), key=lambda x: -x[1])
    ])
    ctx_df.to_csv(TABLES_DIR / "context_flag_counts.csv", index=False)
    print(f"  Wrote context_flag_counts.csv")

# Hard QC flags
if HARD_QC_FLAGS:
    hqc_df = pd.DataFrame([
        {"hard_qc_flag": k, "count": v,
         "percent_of_excluded": round(100 * v / N_HARD_QC_EXCLUDED, 1) if N_HARD_QC_EXCLUDED else None}
        for k, v in sorted(HARD_QC_FLAGS.items(), key=lambda x: -x[1])
    ])
    hqc_df.to_csv(TABLES_DIR / "hard_qc_flag_counts.csv", index=False)
    print(f"  Wrote hard_qc_flag_counts.csv")

# Manual review progress
if manual_labels_df is not None:
    n_reviewed = len(manual_labels_df)
    decision_counts = manual_labels_df["human_decision"].value_counts().to_dict() if "human_decision" in manual_labels_df.columns else {}
    mr_rows = [{"metric": "total_review_cards", "value": N_REVIEW_CARDS},
               {"metric": "basins_reviewed", "value": n_reviewed},
               {"metric": "percent_reviewed", "value": round(100 * n_reviewed / N_REVIEW_CARDS, 1)}]
    for dec, cnt in decision_counts.items():
        mr_rows.append({"metric": f"decision_{dec}", "value": cnt})
    pd.DataFrame(mr_rows).to_csv(TABLES_DIR / "manual_review_progress.csv", index=False)
    print(f"  Wrote manual_review_progress.csv ({n_reviewed}/{N_REVIEW_CARDS} reviewed)")
else:
    _warn("manual_review_labels.csv not found; manual_review_progress.csv skipped")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

STYLE = dict(edgecolor="white", linewidth=0.5)
PALETTE = {
    "FLASHY_CORE":          "#d62728",
    "FLASHY_MODERATE":      "#ff7f0e",
    "FLASHY_POSSIBLE":      "#2ca02c",
    "LOW_FLASHINESS_CONTROL": "#1f77b4",
    "MANUAL_REVIEW_CONTEXT": "#9467bd",
    "EXCLUDE_HARD_QC":      "#7f7f7f",
}

print("\nGenerating plots ...")

# ------------------------------------------------------------------
# Plot 1: Filtering funnel bar chart
# ------------------------------------------------------------------

# Select main funnel stages (exclude BFI exploratory branch and hard-QC exclusion row;
# show those as annotations instead)
main_stages = funnel_df[funnel_df["stage_order"].isin([0, 1, 3, 4, 6, 7])].copy()
labels = main_stages["stage_name"].tolist()
counts = main_stages["count"].tolist()
colors = ["#aec7e8", "#4393c3", "#2166ac", "#053061", "#1a9850", "#e6550d"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(labels)), counts, color=colors[:len(labels)], **STYLE)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels([f"{l}\n(stage {s})" for l, s in
                    zip(labels, main_stages["stage_order"])], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Basin count", fontsize=10)
ax.set_title("Flash-NH Basin Filtering Funnel\n"
             f"Total N = {N_TOTAL:,} → Usable universe N = {N_USABLE:,}",
             fontsize=11, fontweight="bold")
for bar, cnt, pct in zip(bars, counts,
                          main_stages["percent_of_original"].tolist()):
    ax.text(cnt + N_TOTAL * 0.005, bar.get_y() + bar.get_height() / 2,
            f"  {cnt:,}  ({pct}% of orig)",
            va="center", fontsize=8)
ax.set_xlim(0, N_TOTAL * 1.25)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
# Add BFI note as text
ax.text(N_TOTAL * 0.5, len(labels) - 0.2,
        f"Note: BFI ≤ 40 exploratory subset (n={N_BFI_EXPLORATORY:,}, {_pct(N_BFI_EXPLORATORY, N_AREA_FILTERED)}% of area-filtered)"
        " is NOT on this funnel; it was exploratory only.",
        fontsize=7.5, color="#555555", style="italic")
plt.tight_layout()
fig.savefig(PLOTS_DIR / "filtering_funnel_bar.png", dpi=150)
plt.close(fig)
print("  Saved filtering_funnel_bar.png")

# ------------------------------------------------------------------
# Plot 2: Retention percentage by stage
# ------------------------------------------------------------------

ret_stages = funnel_df[funnel_df["stage_order"].isin([0, 1, 3, 4, 6])].copy()
ret_pcts = (ret_stages["count"] / N_TOTAL * 100).tolist()
ret_labels = [f"Stage {s}: {n}" for s, n in
              zip(ret_stages["stage_order"], ret_stages["stage_name"])]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(len(ret_pcts)), ret_pcts, marker="o", color="#2166ac", linewidth=2)
for i, (pct, cnt) in enumerate(zip(ret_pcts, ret_stages["count"])):
    ax.annotate(f"{cnt:,}\n({pct:.1f}%)", (i, pct),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8)
ax.set_xticks(range(len(ret_labels)))
ax.set_xticklabels(ret_labels, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("% of original 9,008-basin universe", fontsize=10)
ax.set_title(f"Flash-NH Filtering Funnel — Retention by Stage\n(N_original = {N_TOTAL:,})",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "filtering_funnel_retention.png", dpi=150)
plt.close(fig)
print("  Saved filtering_funnel_retention.png")

# ------------------------------------------------------------------
# Plot 3: Candidate class counts (usable universe)
# ------------------------------------------------------------------

if CLASS_COUNTS:
    cls_order = ["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE",
                 "MANUAL_REVIEW_CONTEXT", "LOW_FLASHINESS_CONTROL"]
    cls_vals = [CLASS_COUNTS.get(c, 0) for c in cls_order]
    cls_colors = [PALETTE.get(c, "#999") for c in cls_order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(cls_order, cls_vals, color=cls_colors, **STYLE)
    for bar, val in zip(bars, cls_vals):
        pct = round(100 * val / N_USABLE, 1)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val:,}\n({pct}%)", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(cls_order)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cls_order], fontsize=9)
    ax.set_ylabel("Basin count", fontsize=10)
    ax.set_title(f"Candidate Class Distribution — Usable Universe\n(N = {N_USABLE:,} hard-QC-passing basins)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(cls_vals) * 1.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "candidate_class_counts.png", dpi=150)
    plt.close(fig)
    print("  Saved candidate_class_counts.png")

# ------------------------------------------------------------------
# Plot 4: RBI distribution by candidate class (log-scale)
# ------------------------------------------------------------------

if metrics_df is not None and "RBI" in metrics_df.columns and "candidate_class" in metrics_df.columns:
    usable_mask = ~metrics_df["candidate_class"].isin(["EXCLUDE_HARD_QC"]) & metrics_df["RBI"].notna()
    df_plot = metrics_df[usable_mask].copy()
    cls_order_rbi = ["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE",
                     "MANUAL_REVIEW_CONTEXT", "LOW_FLASHINESS_CONTROL"]
    groups = [df_plot.loc[df_plot["candidate_class"] == c, "RBI"].dropna().values
              for c in cls_order_rbi]
    n_per = [len(g) for g in groups]

    fig, ax = plt.subplots(figsize=(10, 5))
    bplot = ax.boxplot(
        groups,
        tick_labels=[f"{c.replace('_', chr(10))}\nn={n:,}" for c, n in zip(cls_order_rbi, n_per)],
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, cls in zip(bplot["boxes"], cls_order_rbi):
        patch.set_facecolor(PALETTE.get(cls, "#aaa"))
        patch.set_alpha(0.75)
    ax.set_yscale("log")
    ax.set_ylabel("RBI (log scale)", fontsize=10)
    ax.set_title(f"RBI Distribution by Candidate Class — Usable Universe\n"
                 f"(N = {df_plot['RBI'].notna().sum():,} valid RBI values; "
                 f"outliers hidden; log y-axis)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "rbi_distribution_by_class.png", dpi=150)
    plt.close(fig)
    print("  Saved rbi_distribution_by_class.png")
elif RBI_BY_CLASS:
    # Fallback: bar chart of median RBI per class from JSON
    cls_list = ["FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE",
                "MANUAL_REVIEW_CONTEXT", "LOW_FLASHINESS_CONTROL"]
    medians = [RBI_BY_CLASS.get(c, {}).get("median", None) for c in cls_list]
    ns = [RBI_BY_CLASS.get(c, {}).get("count", 0) for c in cls_list]
    valid = [(c, m, n) for c, m, n in zip(cls_list, medians, ns) if m is not None]
    if valid:
        cv, mv, nv = zip(*valid)
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(cv, mv, color=[PALETTE.get(c, "#aaa") for c in cv], **STYLE)
        for bar, n, m in zip(bars, nv, mv):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.001,
                    f"median={m:.3f}\nn={n:,}", ha="center", va="bottom", fontsize=8)
        ax.set_yscale("log")
        ax.set_xticklabels([c.replace("_", "\n") for c in cv], fontsize=9)
        ax.set_ylabel("Median RBI (log scale)", fontsize=10)
        ax.set_title("Median RBI by Candidate Class (log scale)\n"
                     f"N_usable = {N_USABLE:,}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "rbi_distribution_by_class.png", dpi=150)
        plt.close(fig)
        print("  Saved rbi_distribution_by_class.png (median bars; no full CSV)")

# ------------------------------------------------------------------
# Plot 5: Area bin × BFI bin class summary
# ------------------------------------------------------------------

if metrics_df is not None and "area_bin" in metrics_df.columns and "BFI_bin" in metrics_df.columns:
    df_usable = metrics_df[metrics_df["candidate_class"] != "EXCLUDE_HARD_QC"].copy()
    area_order = ["1-10 km²", "10-100 km²", "100-1000 km²"]
    bfi_order = ["<=20", "20-30", "30-40", "40-50", ">50"]
    area_order = [a for a in area_order if a in df_usable["area_bin"].values]
    bfi_order = [b for b in bfi_order if b in df_usable["BFI_bin"].values]

    pivot = df_usable.pivot_table(index="area_bin", columns="BFI_bin",
                                  values="STAID", aggfunc="count")
    pivot = pivot.reindex(index=area_order, columns=bfi_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(bfi_order)))
    ax.set_xticklabels(bfi_order, fontsize=9)
    ax.set_yticks(range(len(area_order)))
    ax.set_yticklabels(area_order, fontsize=9)
    ax.set_xlabel("BFI bin", fontsize=10)
    ax.set_ylabel("Drainage area bin", fontsize=10)
    ax.set_title(f"Basin Count by Area Bin × BFI Bin — Usable Universe\n"
                 f"(N = {len(df_usable):,})",
                 fontsize=11, fontweight="bold")
    for i in range(len(area_order)):
        for j in range(len(bfi_order)):
            val = int(pivot.values[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=9, color="black" if val < pivot.values.max() * 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Basin count")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "area_bfi_class_summary.png", dpi=150)
    plt.close(fig)
    print("  Saved area_bfi_class_summary.png")
elif BY_AREA and BY_BFI:
    # Fallback: side-by-side bar
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, data, title, xlabel in [
        (axes[0], BY_AREA, "Basin Count by Drainage Area Bin", "Area bin"),
        (axes[1], BY_BFI, "Basin Count by BFI Bin", "BFI_AVE bin"),
    ]:
        ks = list(data.keys())
        vs = list(data.values())
        bars = ax.bar(ks, vs, color="#4393c3", **STYLE)
        for bar, v in zip(bars, vs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 5,
                    f"{v:,}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Basin count", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.suptitle(f"Area and BFI Distributions — Processed Universe (N={N_WY2024_PROCESSED:,})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "area_bfi_class_summary.png", dpi=150)
    plt.close(fig)
    print("  Saved area_bfi_class_summary.png (bar fallback; no CSV for cross-tab)")

# ------------------------------------------------------------------
# Plot 6: Geographic map (lat/lon from metrics CSV)
# ------------------------------------------------------------------

_map_saved = False
if metrics_df is not None and "LAT_GAGE" in metrics_df.columns and "LNG_GAGE" in metrics_df.columns:
    df_map = metrics_df[metrics_df["candidate_class"] != "EXCLUDE_HARD_QC"].dropna(
        subset=["LAT_GAGE", "LNG_GAGE"]
    )
    if len(df_map) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        cls_order_map = ["FLASHY_POSSIBLE", "FLASHY_MODERATE", "FLASHY_CORE",
                         "MANUAL_REVIEW_CONTEXT", "LOW_FLASHINESS_CONTROL"]
        for cls in cls_order_map:
            sub = df_map[df_map["candidate_class"] == cls]
            if len(sub) == 0:
                continue
            ax.scatter(sub["LNG_GAGE"], sub["LAT_GAGE"],
                       c=PALETTE.get(cls, "#aaa"), s=4, alpha=0.5,
                       label=f"{cls} (n={len(sub):,})", linewidths=0)
        ax.set_xlim(-130, -60)
        ax.set_ylim(23, 52)
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.set_title(f"Geographic Distribution of Flash-NH Usable Basins by Candidate Class\n"
                     f"(N = {len(df_map):,} basins with lat/lon)",
                     fontsize=11, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8, markerscale=3)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "map_candidate_classes.png", dpi=150)
        plt.close(fig)
        print("  Saved map_candidate_classes.png")
        _map_saved = True
if not _map_saved:
    _warn("map_candidate_classes.png skipped: no lat/lon data available")

# ------------------------------------------------------------------
# Plot 7: Context flag counts
# ------------------------------------------------------------------

if CONTEXT_FLAGS:
    sorted_ctx = sorted(CONTEXT_FLAGS.items(), key=lambda x: -x[1])
    cf_keys = [k.replace("CONTEXT_", "").replace("_", "\n") for k, _ in sorted_ctx]
    cf_vals = [v for _, v in sorted_ctx]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(cf_keys, cf_vals, color="#9ecae1", **STYLE)
    for bar, val in zip(bars, cf_vals):
        pct = round(100 * val / N_USABLE, 1)
        ax.text(bar.get_x() + bar.get_width() / 2, val + 5,
                f"{val:,}\n({pct}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Basin count", fontsize=10)
    ax.set_title(f"Context Flag Counts — All Processed Basins\n"
                 f"(N_usable={N_USABLE:,}; basins can carry multiple flags; "
                 f"flags are informational, not exclusions)",
                 fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "context_flag_counts.png", dpi=150)
    plt.close(fig)
    print("  Saved context_flag_counts.png")

# ------------------------------------------------------------------
# Plot 8: Hard QC flag counts
# ------------------------------------------------------------------

if HARD_QC_FLAGS:
    sorted_hqc = sorted(HARD_QC_FLAGS.items(), key=lambda x: -x[1])
    hq_keys = [k.replace("HARD_", "").replace("_", "\n") for k, _ in sorted_hqc]
    hq_vals = [v for _, v in sorted_hqc]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(hq_keys, hq_vals, color="#fc8d59", **STYLE)
    for bar, val in zip(bars, hq_vals):
        pct = round(100 * val / N_HARD_QC_EXCLUDED, 1)
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:,}\n({pct}% of excl.)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Basin count", fontsize=10)
    ax.set_title(f"Hard QC Exclusion Flags\n"
                 f"(N_excluded={N_HARD_QC_EXCLUDED:,}, {_fmt_pct(N_HARD_QC_EXCLUDED, N_WY2024_PROCESSED)} "
                 f"of processed; basins can carry multiple flags)",
                 fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "hard_qc_flag_counts.png", dpi=150)
    plt.close(fig)
    print("  Saved hard_qc_flag_counts.png")

# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

print("\nWriting summaries ...")

# --- Build funnel rows text for markdown ---
funnel_md_rows = []
for _, r in funnel_df.iterrows():
    pct_prev_str = f"{r['percent_of_previous_stage']}%" if pd.notna(r["percent_of_previous_stage"]) else "—"
    funnel_md_rows.append(
        f"| {int(r['stage_order'])} | {r['stage_name']} | {int(r['count']):,} | "
        f"{r['percent_of_original']}% | {pct_prev_str} | {r['filter_type']} |"
    )
funnel_md_table = "\n".join(funnel_md_rows)

# candidate class table text
class_md_rows = []
for cls, cnt in sorted(CLASS_COUNTS.items(), key=lambda x: -x[1]):
    pct = round(100 * cnt / N_USABLE, 1)
    rbi_info = RBI_BY_CLASS.get(cls, {})
    med_rbi = rbi_info.get("median", None)
    rbi_str = f"{med_rbi:.3f}" if med_rbi else "—"
    class_md_rows.append(f"| {cls} | {cnt:,} | {pct}% | {rbi_str} |")
class_md_table = "\n".join(class_md_rows)

# context flags text
ctx_md_rows = [f"| {k} | {v:,} | {round(100*v/N_USABLE,1)}% |"
               for k, v in sorted(CONTEXT_FLAGS.items(), key=lambda x: -x[1])]
ctx_md_table = "\n".join(ctx_md_rows)

hard_qc_md_rows = [f"| {k} | {v:,} | {round(100*v/N_HARD_QC_EXCLUDED,1)}% |"
                   for k, v in sorted(HARD_QC_FLAGS.items(), key=lambda x: -x[1])]
hard_qc_md_table = "\n".join(hard_qc_md_rows)

manual_status_str = (
    f"{len(manual_labels_df)}/{N_REVIEW_CARDS} basins reviewed so far "
    f"({round(100*len(manual_labels_df)/N_REVIEW_CARDS,1)}% of review set)."
    if manual_labels_df is not None else
    "No labels recorded yet (manual_review_labels.csv not found)."
)

warnings_section = ""
if MISSING:
    warnings_section = "\n\n## Warnings (missing optional files)\n\n" + "\n".join(
        f"- {w}" for w in MISSING
    )

md_content = f"""# Flash-NH Basin Filtering Funnel — Consolidated Report

**Generated**: {pd.Timestamp.now(tz="UTC").isoformat()}
**Output directory**: `reports/flashnh_basin_filtering_consolidation_v001/`

---

## Executive Summary

Flash-NH screening started from the full 9,008-basin GAGES-II/CAMELSH catalog and applied a
multi-stage filtering funnel to identify a scientifically defensible universe for model training
and evaluation.

**The recommended carry-forward set is all {N_USABLE:,} hard-QC-passing basins.**

Key numbers:
- **Original universe**: {N_TOTAL:,} GAGES-II basins
- **Area-filtered**: {N_AREA_FILTERED:,} (1–1,000 km²; 65% of total)
- **USGS coverage eligible**: {N_ELIGIBLE_SCREENING_WY:,} (screening WY2024; 62.5% of area-filtered)
- **WY2024 processed**: {N_WY2024_PROCESSED:,} (91.2% of eligible)
- **Hard-QC excluded**: {N_HARD_QC_EXCLUDED:,} (8.4% of processed — permanent removal)
- **Usable universe**: **{N_USABLE:,}** (91.6% of processed; 33.8% of original)
- **Manual review set**: {N_REVIEW_CARDS:,} basins sampled for human hydrograph inspection

---

## Filtering Funnel

| Stage | Stage Name | Count | % of Orig | % of Prev | Filter Type |
|---|---|---|---|---|---|
{funnel_md_table}

**Filter type key**:
- `hard` — permanent exclusion or inclusion criterion (non-negotiable data-quality threshold or scope rule)
- `exploratory` — scoping exercise; NOT a hard exclusion from the main analysis path
- `derived` — computed from upstream step, not a subjective filter
- `context` — informational annotation; no basins removed
- `manual` — human review sample; not a final filter

### Important note on BFI ≤ 40

The BFI ≤ 40 subset (n={N_BFI_EXPLORATORY:,}) was an **exploratory step** used to scope early USGS queries.
It is **not** a hard filter on the main funnel. The USGS coverage eligibility audit and all subsequent
analysis were run on all {N_AREA_FILTERED:,} area-filtered basins, not only the BFI ≤ 40 subset.
BFI_AVE is retained as a stratification attribute in the usable universe.

---

## Hard vs. Soft/Context Filtering

### Hard exclusions (permanent removal; n={N_HARD_QC_EXCLUDED:,})

| Flag | Count | % of excluded |
|---|---|---|
{hard_qc_md_table}

These failures reflect objective data-quality deficiencies:
- **HARD_LOW_COMPLETENESS_LT90**: fewer than 90% of expected hourly values available in WY2024
- **HARD_NEGATIVE_FLOW_SEVERE**: implausible negative discharge values (>1% of record)
- **HARD_Q50_ZERO_OR_NEAR_ZERO**: median hourly flow at zero with zero-flow dominating the record; RBI undefined
- **HARD_NO_RBI**: RBI could not be computed (zero total flow or missing denominators)

Hard exclusions are not reversible for WY2024 without downloading a different water year.

### Context flags (informational; basins remain in usable universe)

| Flag | Count | % of usable |
|---|---|---|
{ctx_md_table}

Context flags are **diagnostic annotations only**. A context-flagged basin that passes hard QC
is still in the usable universe. A single basin can carry multiple flags; counts above are not unique-basin totals.

Key interpretation points:
- **CONTEXT_SUSPICIOUS_SPIKE_SEVERE** and **CONTEXT_HIGH_NORMALIZED_JUMP**: potential data artifacts
  or extreme events. Not auto-excluded because small flashy basins legitimately exhibit high
  hourly variability. Manual review recommended for the most extreme cases.
- **CONTEXT_ZERO_FLOW_SOME** / **CONTEXT_INTERMITTENT_LIKE**: zero-flow or seasonal dry periods are
  **real hydrology** in arid/semi-arid regions, not automatically a data problem.
- **CONTEXT_HIGH_BFI**: baseflow-dominated character; included as context, not exclusion.

---

## Current Active Basin Universe

### By candidate class (usable universe, n={N_USABLE:,})

| Candidate Class | Count | % of usable | Median RBI |
|---|---|---|---|
{class_md_table}

**Candidate class interpretation**:
- Classes are **derived stratification strata**, not natural hydrological clusters.
  RBI thresholds (CORE ≥ 0.10, MODERATE 0.05–0.10, POSSIBLE 0.001–0.05) are operationally
  defined for this project; they do not represent universal flashiness thresholds from the literature.
- **FLASHY_POSSIBLE** (68% of usable basins, RBI 0.001–0.05): the dominant class.
  These are moderate-response basins with real hydrological value; **do not exclude them**.
- **FLASHY_CORE** (13%, RBI ≥ 0.10): highest flashiness; 397 basins.
- **FLASHY_MODERATE** (17%, RBI 0.05–0.10): 503 basins.
- **LOW_FLASHINESS_CONTROL** (5 basins): reference basins for model performance benchmarking.
- **MANUAL_REVIEW_CONTEXT** (58 basins): flagged for inspection due to context flags or
  ambiguous classification; **included in usable universe** pending review.

### Recommendation for Stage 1

**Use all {N_USABLE:,} hard-QC-passing basins as the training and evaluation universe.**

- Candidate classes should be used as **stratification metadata** and **sampling weights**,
  not as strict inclusion filters.
- **Pilot-100** (n=100, 92% FLASHY_CORE) is a debugging set for rapid pipeline iteration.
  It is **not representative** of the usable universe (68% FLASHY_POSSIBLE; 13% FLASHY_CORE).
  Scientific model evaluation requires the full {N_USABLE:,}-basin universe.
- Meteorological preprocessing should begin immediately for all {N_USABLE:,} basins.

---

## Manual Review Status and Purpose

{N_REVIEW_CARDS:,} basins were selected for human hydrograph review across all candidate classes
and several extreme-metric review groups (top RBI, top rise/km², top Q95/Q50, zero-flow basins, etc.).

**Review status**: {manual_status_str}

Manual review is a **calibration exercise**, not a prerequisite for proceeding. The purpose is:
1. Validate that high-RBI and spike-flagged basins represent real hydrology (or identify true artifacts)
2. Spot-check data quality across diverse basin types
3. Inform confidence in the usable universe before finalizing model results

**Not all {N_REVIEW_CARDS:,} review basins need manual decisions before proceeding.**
Tier 1 priority (~30–50 extreme cases) should be reviewed first.
The remaining ~900 context-flagged basins outside the review set are deferred to post-training
residual analysis.

---

## Remaining Concerns

1. **Metric skew**: max_abs_hourly_jump/Q50 has median ≈ 11 but max ≈ 25,000,000.
   Log scale is mandatory for any visualization of this metric.
   Q95/Q50 and Q99/Q50 are similarly right-skewed.
2. **Regulated and artifact basins**: CONTEXT_SUSPICIOUS_SPIKE_SEVERE (n=1,239) includes both
   genuine extreme events and potential sensor artifacts or regulation spikes.
   Top ~30 by max_abs_jump/Q50 warrant manual inspection before final model evaluation.
3. **Small basins** (<10 km², n=79 in processed set): sparse forcing grids may limit input quality.
4. **Zero-flow basins**: HARD_Q50_ZERO_OR_NEAR_ZERO exclusion removes only extreme cases.
   Basins with seasonal zero-flow are in the usable universe and should be treated as valid.
5. **HISTORICAL_ONLY basins** (n={eligibility_json.get('basin_count_historical_only', 1993):,} of area-filtered):
   not available for WY2024 screening. May be retrievable for earlier water years if broader
   temporal coverage is needed in the future.

---

## Recommended Next Steps

1. **Tier 1 manual review** (~30–50 basins): top extreme-metric cases and MANUAL_REVIEW_CONTEXT
2. **Meteorological preprocessing** for all {N_USABLE:,} usable basins (main workstream; start now)
3. **Stage 1 model training** on full {N_USABLE:,}-basin universe with stratified evaluation
4. **Post-training residual analysis** to surface any remaining artifact basins{warnings_section}

---

*Sources*: filtering funnel reconstructed from screening and metrics summaries in
`reports/flashnh_basin_screening_v001/`, `reports/flashnh_usgs_coverage_eligibility_v001/`,
`reports/flashnh_usgs_rbi_screening_wy2024_v001/`, `reports/flashnh_wy2024_streamflow_metrics_v002/`,
`reports/flashnh_wy2024_pilot_selection_v001/`, and `reports/flashnh_hydrograph_review_cards_v003_diverse/`.
"""

(SUMMARIES_DIR / "basin_filtering_consolidation.md").write_text(md_content, encoding="utf-8")
print(f"  Wrote basin_filtering_consolidation.md")

# --- JSON summary ---
json_summary = {
    "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    "funnel": {
        "n_original": N_TOTAL,
        "n_area_filtered": N_AREA_FILTERED,
        "n_bfi_exploratory_subset": N_BFI_EXPLORATORY,
        "n_eligible_screening_wy": N_ELIGIBLE_SCREENING_WY,
        "n_wy2024_processed": N_WY2024_PROCESSED,
        "n_hard_qc_excluded": N_HARD_QC_EXCLUDED,
        "n_usable": N_USABLE,
        "n_manual_review_set": N_REVIEW_CARDS,
        "pct_usable_of_original": _pct(N_USABLE, N_TOTAL),
        "pct_usable_of_processed": _pct(N_USABLE, N_WY2024_PROCESSED),
        "pct_hard_qc_excluded": _pct(N_HARD_QC_EXCLUDED, N_WY2024_PROCESSED),
    },
    "candidate_classes": CLASS_COUNTS,
    "hard_qc_flags": HARD_QC_FLAGS,
    "context_flags": CONTEXT_FLAGS,
    "area_bins": BY_AREA,
    "bfi_bins": BY_BFI,
    "states_top10": dict(list(sorted(BY_STATE.items(), key=lambda x: -x[1])[:10])) if BY_STATE else {},
    "huc02_top10": dict(list(sorted(BY_HUC02.items(), key=lambda x: -x[1])[:10])) if BY_HUC02 else {},
    "manual_review": {
        "review_set_size": N_REVIEW_CARDS,
        "labels_recorded": len(manual_labels_df) if manual_labels_df is not None else 0,
        "pct_reviewed": round(100 * len(manual_labels_df) / N_REVIEW_CARDS, 1)
        if manual_labels_df is not None else 0.0,
    },
    "warnings": MISSING,
    "recommendation": (
        f"Carry forward all {N_USABLE:,} hard-QC-passing basins as the Flash-NH usable basin universe. "
        "Use candidate classes as stratification metadata, not inclusion filters. "
        "Begin meteorological preprocessing immediately."
    ),
}

with open(SUMMARIES_DIR / "basin_filtering_consolidation.json", "w") as f:
    json.dump(json_summary, f, indent=2)
print(f"  Wrote basin_filtering_consolidation.json")

# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print(f"\nOutput directory : {OUT_DIR}")
print(f"\nKey funnel counts:")
print(f"  Original universe      : {N_TOTAL:,}")
print(f"  Area-filtered          : {N_AREA_FILTERED:,}  ({_fmt_pct(N_AREA_FILTERED, N_TOTAL)} of orig)")
print(f"  USGS eligible          : {N_ELIGIBLE_SCREENING_WY:,}  ({_fmt_pct(N_ELIGIBLE_SCREENING_WY, N_AREA_FILTERED)} of area-filt)")
print(f"  WY2024 processed       : {N_WY2024_PROCESSED:,}  ({_fmt_pct(N_WY2024_PROCESSED, N_ELIGIBLE_SCREENING_WY)} of eligible)")
print(f"  Hard-QC excluded       : {N_HARD_QC_EXCLUDED:,}  ({_fmt_pct(N_HARD_QC_EXCLUDED, N_WY2024_PROCESSED)} of processed)")
print(f"  Usable universe        : {N_USABLE:,}  ({_fmt_pct(N_USABLE, N_WY2024_PROCESSED)} of processed)")
print(f"  Manual review set      : {N_REVIEW_CARDS:,}  (diverse sample)")

if MISSING:
    print(f"\nWarnings ({len(MISSING)} missing optional files):")
    for w in MISSING:
        print(f"  {w}")
else:
    print("\nNo warnings.")
