#!/usr/bin/env python3
"""
USGS site-type metadata audit for Flash-NH WY2024 basin screening.

Retrieves USGS monitoring-location metadata for all stations in the WY2024
streamflow metrics table, assigns conservative metadata policy buckets,
and writes audit outputs.

Usage:
    python scripts/build_usgs_site_metadata_audit.py
    python scripts/build_usgs_site_metadata_audit.py --max-sites 20
    python scripts/build_usgs_site_metadata_audit.py --force-refresh
    python scripts/build_usgs_site_metadata_audit.py --no-plots

Reads:
    reports/flashnh_wy2024_streamflow_metrics_v002/tables/wy2024_streamflow_metrics.csv

Writes:
    reports/flashnh_usgs_site_metadata_v001/
        tables/usgs_site_metadata.csv
        tables/wy2024_metrics_with_site_metadata.csv
        tables/metadata_policy_counts.csv
        tables/site_type_counts.csv
        tables/metadata_hard_exclusions.csv
        tables/metadata_review_candidates.csv
        tables/static_metadata_attributes_candidates.csv
        summaries/usgs_site_metadata_audit_summary.md
        summaries/usgs_site_metadata_audit_summary.json
        plots/metadata_policy_counts.png
        plots/site_type_counts.png
        plots/metadata_policy_by_candidate_class.png
        plots/map_metadata_policy.png
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

METRICS_CSV = (
    REPO_ROOT / "reports" / "flashnh_wy2024_streamflow_metrics_v002"
    / "tables" / "wy2024_streamflow_metrics.csv"
)
OUT_DIR       = REPO_ROOT / "reports" / "flashnh_usgs_site_metadata_v001"
TABLES_DIR    = OUT_DIR / "tables"
SUMMARIES_DIR = OUT_DIR / "summaries"
PLOTS_DIR     = OUT_DIR / "plots"
CACHE_CSV     = TABLES_DIR / "usgs_site_metadata.csv"

EXPECTED_ROWS       = 3324
NWIS_SITE_URL       = "https://waterservices.usgs.gov/nwis/site/"
BATCH_SIZE          = 150
REQUEST_DELAY       = 0.5   # seconds between batches
SQ_MI_TO_KM2        = 2.58999
AREA_DISC_THRESHOLD = 0.10  # 10% difference flags area discrepancy

# ---------------------------------------------------------------------------
# Site-type policy tables
# ---------------------------------------------------------------------------

SITE_TYPE_NAMES: dict = {
    "ST":      "Stream",
    "ST-TS":   "Tidal stream",
    "ST-CA":   "Canal",
    "ST-DCH":  "Ditch",
    "ES":      "Estuary",
    "LK":      "Lake, Reservoir, Impoundment",
    "OC":      "Ocean",
    "OC-CO":   "Coastal",
    "WE":      "Wetland",
    "GW":      "Well",
    "GW-EX":   "Excavation",
    "GW-IW":   "Infiltration Well",
    "GW-MW":   "Multiple Wells",
    "GW-TH":   "Test Hole not completed as well",
    "SP":      "Spring",
    "AT":      "Atmospheric/Meteorological",
    "AG":      "Aggregate groundwater use",
    "AS":      "Aggregate surface-water use",
    "AW":      "Aggregate water use",
    "SB":      "Seep/subsurface",
    "SB-GWD":  "Groundwater drain",
    "SB-TSP":  "Terrace spring",
    "SB-UZ":   "Unsaturated zone",
    "SB-CV":   "Cave opening",
    "LA":      "Land",
    "LA-PLY":  "Playa",
    "LA-SNK":  "Sinkhole",
    "LA-VOL":  "Volcanic opening",
    "LA-SR":   "Shore",
    "GL":      "Glacier",
    "FA":      "Facility",
    "FA-CI":   "Canal intake",
    "FA-DV":   "Diversion",
    "FA-LF":   "Landfill",
    "FA-OF":   "Outfall",
    "FA-PV":   "Pavement",
    "FA-QC":   "Quality control site",
    "FA-SEW":  "Sewer",
    "FA-SPS":  "Septic system",
    "FA-STS":  "Storm sewer",
    "FA-TEP":  "Treatment plant",
    "FA-WDS":  "Water distribution system",
    "FA-WIW":  "Water injection well",
    "FA-WTP":  "Water treatment plant",
    "FA-WWTP": "Wastewater treatment plant",
    "FA-WWD":  "Wastewater disposal",
    "FA-HP":   "Hydroelectric plant",
    "FA-AWL":  "Auger well",
    "FA-CS":   "Combined sewer",
    "FA-WU":   "Water-use monitoring",
}

ACCEPT_TYPES = {"ST"}

REVIEW_TYPES = {"ST-CA", "ST-DCH", "SP", "LA-SNK", "LA-PLY", "LA-SR"}

HARD_EXCLUDE_EXACT = {
    "ST-TS", "ES", "OC", "OC-CO", "LK",
    "AT", "GL", "WE",
    "AG", "AS", "AW",
    "FA-DV", "FA-OF", "FA-CS", "FA-SEW", "FA-STS", "FA-WWTP", "FA-WWD",
    "FA-WIW", "FA-AWL", "FA-CI", "FA-WDS", "FA-WTP", "FA-WU", "FA-TEP",
    "FA-HP", "FA-QC",
}

EXCLUSION_REASONS: dict = {
    "ST-TS": "tidal_stream",
    "ES":    "estuary",
    "OC":    "ocean",
    "OC-CO": "ocean_coastal",
    "LK":    "lake_reservoir",
    "AT":    "atmospheric",
    "GL":    "glacier",
    "WE":    "wetland",
    "AG":    "aggregate_water_use",
    "AS":    "aggregate_water_use",
    "AW":    "aggregate_water_use",
}

GROUP_MAP: dict = {
    "ST":     "stream",
    "ST-TS":  "tidal_stream",
    "ST-CA":  "canal",
    "ST-DCH": "ditch",
    "ES":     "estuary",
    "OC":     "ocean_coastal",
    "OC-CO":  "ocean_coastal",
    "LK":     "lake_reservoir",
    "SP":     "spring",
    "GL":     "land",
    "WE":     "wetland",
    "AT":     "atmospheric",
    "AG":     "facility",
    "AS":     "facility",
    "AW":     "facility",
    "LA":     "land",
    "LA-PLY": "land",
    "LA-SNK": "land",
    "LA-SR":  "land",
}

BUCKET_COLORS = {
    "ACCEPT":           "#2ca02c",
    "REVIEW":           "#ff7f0e",
    "HARD_EXCLUDE":     "#d62728",
    "MISSING_METADATA": "#7f7f7f",
}

CANDIDATE_CLASS_ORDER = [
    "FLASHY_CORE", "FLASHY_MODERATE", "FLASHY_POSSIBLE",
    "LOW_FLASHINESS_CONTROL", "MANUAL_REVIEW_CONTEXT", "EXCLUDE_HARD_QC",
]


# ---------------------------------------------------------------------------
# Policy classification
# ---------------------------------------------------------------------------

def _type_group(code: str) -> str:
    if code in GROUP_MAP:
        return GROUP_MAP[code]
    if code.startswith("GW"):
        return "groundwater"
    if code.startswith("SB"):
        return "subsurface"
    if code.startswith("FA"):
        return "facility"
    if code.startswith("LA"):
        return "land"
    return "unknown"


def classify(code) -> tuple:
    """Return (policy_bucket, exclusion_reason, site_type_group)."""
    if code is None or (isinstance(code, float) and pd.isna(code)) or str(code).strip() == "":
        return ("MISSING_METADATA", "missing_site_type", "unknown")

    c = str(code).strip().upper()

    if c in ACCEPT_TYPES:
        return ("ACCEPT", "", "stream")

    if c in REVIEW_TYPES:
        return ("REVIEW", c.lower().replace("-", "_"), _type_group(c))

    if c in HARD_EXCLUDE_EXACT:
        reason = EXCLUSION_REASONS.get(c, c.lower().replace("-", "_"))
        return ("HARD_EXCLUDE", reason, _type_group(c))

    # Prefix-based exclusions (REVIEW exceptions already caught above)
    if c.startswith("GW"):
        return ("HARD_EXCLUDE", "groundwater", "groundwater")
    if c.startswith("SB"):
        return ("HARD_EXCLUDE", "subsurface", "subsurface")
    if c.startswith("FA"):
        return ("HARD_EXCLUDE", f"facility_{c.lower().replace('-', '_')}", "facility")
    if c.startswith("LA"):
        # LA-SNK, LA-PLY, LA-SR are in REVIEW_TYPES and caught above; remainder -> HARD_EXCLUDE
        return ("HARD_EXCLUDE", "land", "land")

    # Unknown code — treat as REVIEW rather than silently excluding
    return ("REVIEW", f"unknown_site_type_{c.lower()}", "unknown")


# ---------------------------------------------------------------------------
# USGS NWIS fetch
# ---------------------------------------------------------------------------

def _parse_rdb(text: str) -> list:
    """Parse USGS NWIS RDB tab-separated response into list of dicts."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 2:
        return []
    headers = lines[0].split("\t")
    rows = []
    for line in lines[2:]:          # skip the type-width line (lines[1])
        if not line.strip():
            continue
        vals = line.split("\t")
        if len(vals) < len(headers):
            vals += [""] * (len(headers) - len(vals))
        rows.append(dict(zip(headers, vals[:len(headers)])))
    return rows


def fetch_nwis(site_nos: list, verbose: bool = True) -> pd.DataFrame:
    """Batch-fetch USGS site metadata from NWIS; return raw RDB as DataFrame."""
    all_rows: list = []
    n_total   = len(site_nos)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE

    for bi, start in enumerate(range(0, n_total, BATCH_SIZE), 1):
        batch = site_nos[start:start + BATCH_SIZE]
        params = {"sites": ",".join(batch), "siteOutput": "expanded", "format": "rdb"}
        try:
            resp = requests.get(NWIS_SITE_URL, params=params, timeout=60)
            resp.raise_for_status()
            rows = _parse_rdb(resp.text)
            all_rows.extend(rows)
            if verbose:
                print(f"  Batch {bi}/{n_batches}: {len(batch)} requested, {len(rows)} returned")
        except Exception as exc:
            print(f"  Batch {bi}/{n_batches} ERROR: {exc}", file=sys.stderr)
        if start + BATCH_SIZE < n_total:
            time.sleep(REQUEST_DELAY)

    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)
    if "site_no" in raw.columns:
        raw["STAID"] = raw["site_no"].astype(str).str.strip()
    return raw


def normalise_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename and type-convert NWIS RDB columns to standard metadata schema."""
    col_map = {
        "site_no":               "nwis_site_no",
        "station_nm":            "monitoring_location_name",
        "agency_cd":             "agency_code",
        "site_tp_cd":            "site_type_code",
        "dec_lat_va":            "latitude",
        "dec_long_va":           "longitude",
        "lat_va":                "lat_dms",
        "long_va":               "lon_dms",
        "state_cd":              "state_code_fips",
        "county_cd":             "county_code_fips",
        "huc_cd":                "huc8_code",
        "drain_area_va":         "usgs_drain_area_sqmi",
        "contrib_drain_area_va": "usgs_contrib_area_sqmi",
        "alt_va":                "altitude_ft",
        "alt_datum_cd":          "altitude_datum",
    }

    out = pd.DataFrame()
    out["STAID"] = raw["STAID"].astype(str) if "STAID" in raw.columns else raw.get(
        "site_no", pd.Series(dtype=str)).astype(str)

    for src, dst in col_map.items():
        out[dst] = raw[src].replace("", pd.NA) if src in raw.columns else pd.NA

    for num_col in ["latitude", "longitude", "usgs_drain_area_sqmi",
                    "usgs_contrib_area_sqmi", "altitude_ft"]:
        if num_col in out.columns:
            out[num_col] = pd.to_numeric(out[num_col], errors="coerce")

    # Deduplicate on STAID: some sites are returned once per agency when queried
    # through a shared NWIS record. Prefer the USGS agency row; otherwise keep first.
    if out["STAID"].duplicated().any():
        usgs_mask = out.get("agency_code", pd.Series(dtype=str)).str.strip().str.upper() == "USGS"
        out["_is_usgs"] = usgs_mask
        out = (out.sort_values("_is_usgs", ascending=False)
               .drop_duplicates(subset="STAID", keep="first")
               .drop(columns=["_is_usgs"])
               .reset_index(drop=True))

    out["huc02_from_nwis"] = out["huc8_code"].apply(
        lambda x: str(x).strip()[:2] if pd.notna(x) and str(x).strip() else pd.NA)

    out["usgs_drainage_area_km2"] = out["usgs_drain_area_sqmi"] * SQ_MI_TO_KM2
    out["usgs_contributing_area_km2"] = out["usgs_contrib_area_sqmi"] * SQ_MI_TO_KM2

    def _ratio(row):
        d = row["usgs_drainage_area_km2"]
        c = row["usgs_contributing_area_km2"]
        return float(c) / float(d) if pd.notna(d) and pd.notna(c) and float(d) > 0 else pd.NA

    out["contributing_to_total_area_ratio"] = out.apply(_ratio, axis=1)
    out["has_contributing_area_mismatch"] = out["contributing_to_total_area_ratio"].apply(
        lambda x: bool(pd.notna(x) and abs(float(x) - 1.0) > AREA_DISC_THRESHOLD))

    return out


# ---------------------------------------------------------------------------
# Policy and derived columns
# ---------------------------------------------------------------------------

def add_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Add metadata policy bucket, site-type group, and training candidate flags."""
    trios = df["site_type_code"].apply(classify)
    df["metadata_policy_bucket"]    = [t[0] for t in trios]
    df["metadata_exclusion_reason"] = [t[1] for t in trios]
    df["site_type_group"]           = [t[2] for t in trios]

    df["site_type_name"] = df["site_type_code"].apply(
        lambda c: SITE_TYPE_NAMES.get(str(c).strip().upper(), str(c))
        if pd.notna(c) and str(c).strip() else "")

    # Hard-QC pass: prefer candidate_class column (authoritative)
    if "candidate_class" in df.columns:
        df["hard_qc_pass"] = df["candidate_class"] != "EXCLUDE_HARD_QC"
    elif "hard_flags" in df.columns:
        df["hard_qc_pass"] = df["hard_flags"].apply(
            lambda x: "HARD_" not in str(x) if pd.notna(x) else True)
    else:
        df["hard_qc_pass"] = True

    df["main_training_candidate"] = (
        df["hard_qc_pass"] & (df["metadata_policy_bucket"] == "ACCEPT"))
    df["inclusive_training_candidate"] = (
        df["hard_qc_pass"] & df["metadata_policy_bucket"].isin(["ACCEPT", "REVIEW"]))

    return df


def add_area_discrepancy(df: pd.DataFrame) -> pd.DataFrame:
    """Flag basins where USGS drainage area disagrees with DRAIN_SQKM by >10%."""
    if "DRAIN_SQKM" not in df.columns or "usgs_drainage_area_km2" not in df.columns:
        df["area_source_discrepancy"] = pd.NA
        return df

    def _disc(row):
        existing = pd.to_numeric(row.get("DRAIN_SQKM"), errors="coerce")
        usgs     = row.get("usgs_drainage_area_km2")
        if pd.isna(existing) or pd.isna(usgs) or float(existing) <= 0:
            return pd.NA
        return bool(abs(float(usgs) - float(existing)) / float(existing) > AREA_DISC_THRESHOLD)

    df["area_source_discrepancy"] = df.apply(_disc, axis=1)
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_policy_counts(df: pd.DataFrame, out_path: Path) -> None:
    counts = df["metadata_policy_bucket"].value_counts()
    order  = [b for b in ["ACCEPT", "REVIEW", "HARD_EXCLUDE", "MISSING_METADATA"]
               if b in counts.index]
    counts = counts.reindex(order)
    colors = [BUCKET_COLORS.get(b, "steelblue") for b in order]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel("Metadata policy bucket")
    ax.set_ylabel("Basin count")
    ax.set_title("Basin count by metadata policy bucket")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_site_type_counts(df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    plot_df = df.copy()
    plot_df["_code"] = plot_df["site_type_code"].fillna("MISSING")
    counts = plot_df["_code"].value_counts().head(top_n)

    # Colour bars by bucket
    bucket_lookup = (
        plot_df.drop_duplicates("_code")
        .set_index("_code")["metadata_policy_bucket"]
        .to_dict()
    )
    colors = [BUCKET_COLORS.get(bucket_lookup.get(c, ""), "steelblue") for c in counts.index]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = ax.barh(counts.index[::-1], counts.values[::-1],
                   color=colors[::-1], edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel("Basin count")
    ax.set_title(f"Basin count by site_type_code (top {top_n})")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_policy_by_class(df: pd.DataFrame, out_path: Path) -> None:
    if "candidate_class" not in df.columns:
        return
    buckets = ["ACCEPT", "REVIEW", "HARD_EXCLUDE", "MISSING_METADATA"]
    classes = [c for c in CANDIDATE_CLASS_ORDER if c in df["candidate_class"].values]
    ct = pd.crosstab(df["candidate_class"], df["metadata_policy_bucket"])
    ct = ct.reindex(index=classes, columns=buckets, fill_value=0)

    x     = np.arange(len(classes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for i, bucket in enumerate(buckets):
        vals = ct[bucket].values if bucket in ct.columns else np.zeros(len(classes))
        ax.bar(x + i * width, vals, width, label=bucket,
               color=BUCKET_COLORS.get(bucket, "steelblue"), edgecolor="white")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
    ax.set_ylabel("Basin count")
    ax.set_title("Metadata policy bucket by candidate class")
    ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_map(df: pd.DataFrame, out_path: Path) -> None:
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return
    sub = df.dropna(subset=["latitude", "longitude"])
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for bucket, color in BUCKET_COLORS.items():
        mask = sub["metadata_policy_bucket"] == bucket
        ax.scatter(sub.loc[mask, "longitude"], sub.loc[mask, "latitude"],
                   c=color, s=8, alpha=0.6, linewidths=0, label=bucket)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Basin metadata policy — WY2024 screening set")
    ax.legend(markerscale=3, fontsize=8)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary writers
# ---------------------------------------------------------------------------

def _bucket_row(stats: dict, bucket: str) -> str:
    n = stats["policy_counts"].get(bucket, 0)
    pct = 100.0 * n / max(stats["n_joined"], 1)
    return f"| {bucket} | {n} | {pct:.1f}% |"


def write_md_summary(joined: pd.DataFrame, stats: dict, out_path: Path) -> None:
    site_type_rows = (
        joined.groupby("site_type_code", dropna=False)
        .agg(count=("STAID", "count"),
             policy=("metadata_policy_bucket", lambda x: x.mode()[0]))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    lines = [
        "# Flash-NH USGS Site Metadata Audit",
        "",
        f"**Run timestamp**: {stats['run_timestamp']}",
        f"**Input**: WY2024 streamflow metrics ({stats['n_metrics_rows']} rows)",
        f"**Output folder**: `reports/flashnh_usgs_site_metadata_v001/`",
        "",
        "---",
        "",
        "## 1. Metadata Retrieval",
        "",
        "| Item | Count |",
        "|---|---|",
        f"| Input stations (WY2024 metrics) | {stats['n_metrics_rows']} |",
        f"| Stations with metadata retrieved | {stats['n_with_metadata']} |",
        f"| Stations with missing metadata | {stats['n_missing_metadata']} |",
        "",
        "---",
        "",
        "## 2. Policy Bucket Counts",
        "",
        "| Policy bucket | Count | Share |",
        "|---|---|---|",
        _bucket_row(stats, "ACCEPT"),
        _bucket_row(stats, "REVIEW"),
        _bucket_row(stats, "HARD_EXCLUDE"),
        _bucket_row(stats, "MISSING_METADATA"),
        "",
        "---",
        "",
        "## 3. Training Candidate Counts",
        "",
        "| Candidate set | Count | Definition |",
        "|---|---|---|",
        f"| main_training_candidate | {stats['n_main_training_candidate']} "
        "| hard_qc_pass AND metadata == ACCEPT |",
        f"| inclusive_training_candidate | {stats['n_inclusive_training_candidate']} "
        "| hard_qc_pass AND metadata in ACCEPT or REVIEW |",
        f"| original hard-QC exclusions | {stats['n_original_hard_qc_excluded']} "
        "| candidate_class == EXCLUDE_HARD_QC |",
        f"| hard-QC-passing basins removed by metadata HARD_EXCLUDE | "
        f"{stats['n_hard_qc_pass_removed_by_metadata']} | new metadata exclusions |",
        "",
        "---",
        "",
        "## 4. Site Type Counts",
        "",
        "| site_type_code | site_type_name | count | policy_bucket |",
        "|---|---|---|---|",
    ]

    for _, r in site_type_rows.iterrows():
        code = str(r["site_type_code"]) if pd.notna(r["site_type_code"]) else "MISSING"
        name = SITE_TYPE_NAMES.get(code.upper(), code)
        lines.append(f"| {code} | {name} | {int(r['count'])} | {r['policy']} |")

    lines += [
        "",
        "---",
        "",
        "## 5. Validation Checks",
        "",
        f"- Station 02247222 present in metadata: {stats['v_02247222_present']}",
        f"- Station 02247222 metadata_policy_bucket: {stats['v_02247222_bucket']}",
        f"- Station 02247222 metadata_exclusion_reason: {stats['v_02247222_reason']}",
        "",
        "---",
        "",
        "## 6. References",
        "",
        "| Document | Path |",
        "|---|---|",
        "| Basin screening decision memo | `docs/basin_screening_decision_memo.md` |",
        "| WY2024 metrics | `reports/flashnh_wy2024_streamflow_metrics_v002/`... |",
        "| USGS site types | https://api.waterdata.usgs.gov/ogcapi/v0/collections/site-types/items |",
        "| USGS NWIS site service | https://waterservices.usgs.gov/nwis/site/ |",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Summary MD: {out_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="USGS site-type metadata audit for Flash-NH WY2024 basins.")
    parser.add_argument("--max-sites", type=int, default=None,
                        help="Limit to first N stations (smoke test; skips cache write)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-fetch from USGS even if local cache exists")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    for d in [TABLES_DIR, SUMMARIES_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Load metrics ───────────────────────────────────────────────────────
    if not METRICS_CSV.exists():
        print(f"ERROR: metrics CSV not found: {METRICS_CSV}", file=sys.stderr)
        sys.exit(1)

    metrics = pd.read_csv(METRICS_CSV, dtype={"STAID": str})
    metrics["STAID"] = metrics["STAID"].str.strip()
    print(f"Metrics rows: {len(metrics)} (expected {EXPECTED_ROWS})")
    if len(metrics) != EXPECTED_ROWS:
        print(f"  WARNING: expected {EXPECTED_ROWS}, got {len(metrics)}")

    all_staids = metrics["STAID"].tolist()

    if args.max_sites:
        staids_to_fetch = all_staids[:args.max_sites]
        print(f"--max-sites {args.max_sites}: smoke-test mode, fetching {len(staids_to_fetch)} stations")
        smoke_test = True
    else:
        staids_to_fetch = all_staids
        smoke_test = False

    # ── Load or fetch metadata ─────────────────────────────────────────────
    use_cache = CACHE_CSV.exists() and not args.force_refresh and not smoke_test

    if use_cache:
        print(f"Loading cached metadata: {CACHE_CSV.relative_to(REPO_ROOT)}")
        meta_norm = pd.read_csv(CACHE_CSV, dtype={"STAID": str})
        meta_norm["STAID"] = meta_norm["STAID"].str.strip()
        print(f"  Cached rows: {len(meta_norm)}")
    else:
        if smoke_test:
            print("Smoke-test mode: fetching fresh (cache bypassed, not written)")
        elif args.force_refresh:
            print("--force-refresh: re-fetching from USGS NWIS")
        else:
            print("No local cache found — fetching from USGS NWIS...")
        print(f"  {len(staids_to_fetch)} stations in batches of {BATCH_SIZE}...")
        raw = fetch_nwis(staids_to_fetch)
        if len(raw) == 0:
            print("ERROR: USGS NWIS returned no data", file=sys.stderr)
            sys.exit(1)
        meta_norm = normalise_raw(raw)
        if not smoke_test:
            meta_norm.to_csv(CACHE_CSV, index=False)
            print(f"  Saved {len(meta_norm)} rows -> {CACHE_CSV.relative_to(REPO_ROOT)}")
        else:
            print(f"  Smoke test: {len(meta_norm)} rows retrieved (not cached)")

    # ── Join ───────────────────────────────────────────────────────────────
    joined = metrics.merge(meta_norm, on="STAID", how="left")
    print(f"Joined rows: {len(joined)}")

    # ── Policy + derived columns ───────────────────────────────────────────
    joined = add_policy(joined)
    joined = add_area_discrepancy(joined)

    # ── Key counts ────────────────────────────────────────────────────────
    n_with_meta    = int(joined["site_type_code"].notna().sum())
    n_missing_meta = len(joined) - n_with_meta
    n_orig_hard_qc = int((joined.get("candidate_class",
                                     pd.Series(dtype=str)) == "EXCLUDE_HARD_QC").sum())
    n_main_cand    = int(joined["main_training_candidate"].sum())
    n_incl_cand    = int(joined["inclusive_training_candidate"].sum())
    n_removed_by_meta = int(
        (joined["hard_qc_pass"].astype(bool) &
         (joined["metadata_policy_bucket"] == "HARD_EXCLUDE")).sum())

    policy_counts_df = (
        joined["metadata_policy_bucket"]
        .value_counts()
        .rename_axis("metadata_policy_bucket")
        .reset_index(name="count")
    )
    policy_counts_df["pct"] = (policy_counts_df["count"] / len(joined) * 100).round(2)

    # ── Validation checks ─────────────────────────────────────────────────
    probe     = "02247222"
    probe_row = joined[joined["STAID"] == probe]
    v_present = len(probe_row) > 0
    v_bucket  = str(probe_row.iloc[0]["metadata_policy_bucket"]) if v_present else "NOT_FOUND"
    v_reason  = str(probe_row.iloc[0]["metadata_exclusion_reason"]) if v_present else "NOT_FOUND"

    ok_bucket = v_bucket == "HARD_EXCLUDE"
    ok_reason = "tidal_stream" in v_reason

    print("\n--- Validation checks ---")
    print(f"  02247222 present:              {v_present}")
    print(f"  02247222 bucket:               {v_bucket}  {'OK' if ok_bucket else 'FAIL expected HARD_EXCLUDE'}")
    print(f"  02247222 exclusion reason:     {v_reason}  {'OK' if ok_reason else 'FAIL expected tidal_stream'}")
    print(f"  Total with metadata:           {n_with_meta}")
    print(f"  Total missing metadata:        {n_missing_meta}")
    print(f"  Original hard-QC exclusions:   {n_orig_hard_qc}")
    print(f"  Removed by metadata HARD_EXCL: {n_removed_by_meta}")
    print(f"  main_training_candidate:       {n_main_cand}")
    print(f"  inclusive_training_candidate:  {n_incl_cand}")

    # ── Build output tables ────────────────────────────────────────────────
    site_type_counts_df = (
        joined["site_type_code"].fillna("MISSING")
        .value_counts()
        .rename_axis("site_type_code")
        .reset_index(name="count")
    )
    site_type_counts_df["site_type_name"] = site_type_counts_df["site_type_code"].apply(
        lambda c: SITE_TYPE_NAMES.get(str(c).upper(), str(c)))
    site_type_counts_df["policy_bucket"] = site_type_counts_df["site_type_code"].apply(
        lambda c: classify(None if c == "MISSING" else c)[0])

    hard_excl   = joined[joined["metadata_policy_bucket"] == "HARD_EXCLUDE"].copy()
    review_cand = joined[joined["metadata_policy_bucket"] == "REVIEW"].copy()

    STATIC_COLS = [
        "STAID",
        "site_type_code", "site_type_group",
        "state_code_fips", "huc02_from_nwis", "huc8_code",
        "latitude", "longitude",
        "usgs_drainage_area_km2", "usgs_contributing_area_km2",
        "contributing_to_total_area_ratio",
        "altitude_ft", "altitude_datum",
    ]
    static_attrs = joined[[c for c in STATIC_COLS if c in joined.columns]].copy()

    # ── Write tables ───────────────────────────────────────────────────────
    print("\nWriting output tables...")
    joined.to_csv(
        TABLES_DIR / "wy2024_metrics_with_site_metadata.csv", index=False)
    policy_counts_df.to_csv(
        TABLES_DIR / "metadata_policy_counts.csv", index=False)
    site_type_counts_df.to_csv(
        TABLES_DIR / "site_type_counts.csv", index=False)
    hard_excl.to_csv(
        TABLES_DIR / "metadata_hard_exclusions.csv", index=False)
    review_cand.to_csv(
        TABLES_DIR / "metadata_review_candidates.csv", index=False)
    static_attrs.to_csv(
        TABLES_DIR / "static_metadata_attributes_candidates.csv", index=False)
    print(f"  6 tables written to {TABLES_DIR.relative_to(REPO_ROOT)}")

    # ── Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("Generating plots...")
        plot_policy_counts(joined, PLOTS_DIR / "metadata_policy_counts.png")
        plot_site_type_counts(joined, PLOTS_DIR / "site_type_counts.png")
        plot_policy_by_class(joined, PLOTS_DIR / "metadata_policy_by_candidate_class.png")
        plot_map(joined, PLOTS_DIR / "map_metadata_policy.png")
        print(f"  4 plots written to {PLOTS_DIR.relative_to(REPO_ROOT)}")

    # ── Summaries ──────────────────────────────────────────────────────────
    stats = {
        "run_timestamp":                    run_ts,
        "n_metrics_rows":                   len(metrics),
        "n_queried":                        len(staids_to_fetch),
        "n_joined":                         len(joined),
        "n_with_metadata":                  n_with_meta,
        "n_missing_metadata":               n_missing_meta,
        "n_original_hard_qc_excluded":      n_orig_hard_qc,
        "n_hard_qc_pass_removed_by_metadata": n_removed_by_meta,
        "n_main_training_candidate":        n_main_cand,
        "n_inclusive_training_candidate":   n_incl_cand,
        "policy_counts": policy_counts_df.set_index(
            "metadata_policy_bucket")["count"].to_dict(),
        "site_type_counts": site_type_counts_df.set_index(
            "site_type_code")["count"].head(30).to_dict(),
        "v_02247222_present": v_present,
        "v_02247222_bucket":  v_bucket,
        "v_02247222_reason":  v_reason,
        "smoke_test_mode":    smoke_test,
    }

    json_path = SUMMARIES_DIR / "usgs_site_metadata_audit_summary.json"
    json_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    write_md_summary(joined, stats,
                     SUMMARIES_DIR / "usgs_site_metadata_audit_summary.md")

    # ── Final report ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("USGS SITE METADATA AUDIT COMPLETE")
    print("=" * 60)
    print(f"  Metrics rows:                  {len(metrics)}")
    print(f"  Stations with metadata:        {n_with_meta}")
    print(f"  Missing metadata:              {n_missing_meta}")
    print()
    print("  Policy bucket counts:")
    for _, r in policy_counts_df.iterrows():
        print(f"    {r['metadata_policy_bucket']:25s} {int(r['count']):5d}  ({r['pct']:.1f}%)")
    print()
    print(f"  main_training_candidate:       {n_main_cand}")
    print(f"  inclusive_training_candidate:  {n_incl_cand}")
    print(f"  Hard-QC-pass removed by meta:  {n_removed_by_meta}")
    print()
    print(f"  Output: {OUT_DIR.relative_to(REPO_ROOT)}")
    if smoke_test:
        print("  NOTE: smoke-test run -- results are partial (first "
              f"{len(staids_to_fetch)} stations only)")


if __name__ == "__main__":
    main()
