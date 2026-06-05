#!/usr/bin/env python3
"""Generate compact diagnostic CSVs from the completed January 2023 pilot extraction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path("tmp/stage1_pilot_dryrun/03_basin_timeseries/stage1_pilot/january_2023")
MDIR    = Path("tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/january_2023_extraction")
OUT_DIR = MDIR / "diagnostics"
OUT_DIR.mkdir(exist_ok=True)

print("Loading Parquet files ...")
mrms    = pd.read_parquet(BASE / "mrms_hourly_basin_stats.parquet")
rtma    = pd.read_parquet(BASE / "rtma_hourly_basin_stats.parquet")
comb    = pd.read_parquet(BASE / "combined_hourly_basin_stats.parquet")
runtime = pd.read_csv(MDIR / "hourly_runtime_and_volume.csv")
print(f"  MRMS:     {mrms.shape}  RTMA: {rtma.shape}  Combined: {comb.shape}")

# ===========================================================================
# 1. january_value_summary_by_variable.csv
# ===========================================================================
rows = []
for (prod, var), g in comb.groupby(["product", "variable"], sort=True):
    wm  = g["weighted_mean"].dropna()
    vwf = g["valid_weight_fraction"].dropna()
    nc  = g["valid_cell_count"].dropna()
    rows.append({
        "product":                       prod,
        "variable":                      var,
        "units":                         g["units"].iloc[0],
        "variable_role":                 g["variable_role"].iloc[0],
        "recommended_for_initial_model": bool(g["recommended_for_initial_model"].iloc[0]),
        "n_rows":                        len(g),
        "n_unique_basins":               g["STAID"].nunique(),
        "n_unique_times":                g["valid_time_utc"].nunique(),
        "n_null_weighted_mean":          int(g["weighted_mean"].isna().sum()),
        "min":    float(wm.min())                   if len(wm) else np.nan,
        "p01":    float(np.percentile(wm, 1))       if len(wm) else np.nan,
        "p05":    float(np.percentile(wm, 5))       if len(wm) else np.nan,
        "median": float(np.median(wm))              if len(wm) else np.nan,
        "mean":   float(wm.mean())                  if len(wm) else np.nan,
        "p95":    float(np.percentile(wm, 95))      if len(wm) else np.nan,
        "p99":    float(np.percentile(wm, 99))      if len(wm) else np.nan,
        "max":    float(wm.max())                   if len(wm) else np.nan,
        "std":    float(wm.std())                   if len(wm) else np.nan,
        "min_valid_weight_fraction":    float(vwf.min())      if len(vwf) else np.nan,
        "median_valid_weight_fraction": float(np.median(vwf)) if len(vwf) else np.nan,
        "max_valid_weight_fraction":    float(vwf.max())      if len(vwf) else np.nan,
        "min_n_cells":    int(nc.min())             if len(nc) else np.nan,
        "median_n_cells": float(np.median(nc))      if len(nc) else np.nan,
        "max_n_cells":    int(nc.max())             if len(nc) else np.nan,
    })

summary_df = pd.DataFrame(rows)
out1 = OUT_DIR / "january_value_summary_by_variable.csv"
summary_df.to_csv(out1, index=False)
print(f"[1] {out1.name}: {len(summary_df)} rows")

# ===========================================================================
# 2. january_extreme_values_by_variable.csv
# ===========================================================================
KEEP = ["STAID", "valid_time_utc", "product", "variable", "weighted_mean",
        "units", "valid_cell_count", "valid_weight_fraction",
        "variable_role", "recommended_for_initial_model"]
parts = []
for (prod, var), g in comb.groupby(["product", "variable"], sort=True):
    g_valid = g.dropna(subset=["weighted_mean"])
    if len(g_valid) == 0:
        continue
    g_sorted = g_valid.sort_values("weighted_mean")
    n = min(10, len(g_sorted))
    for side, df_side in [("low", g_sorted.head(n)), ("high", g_sorted.tail(n))]:
        df_s = df_side[KEEP].copy()
        df_s["rank_type"] = side
        df_s["rank"]      = range(1, n + 1)
        parts.append(df_s)

ext_df = pd.concat(parts, ignore_index=True)
out2 = OUT_DIR / "january_extreme_values_by_variable.csv"
ext_df.to_csv(out2, index=False)
print(f"[2] {out2.name}: {len(ext_df)} rows")

# ===========================================================================
# 3. january_top_slow_hours.csv
# ===========================================================================
RT_KEEP = ["product", "valid_time_utc", "raw_file_size_bytes", "file_reused",
           "download_time_s", "decode_time_s", "extraction_time_s", "write_time_s",
           "total_processing_time_s", "n_output_rows", "status", "warning_message"]
slow = (runtime.sort_values("total_processing_time_s", ascending=False)
               .head(30)[RT_KEEP])
out3 = OUT_DIR / "january_top_slow_hours.csv"
slow.to_csv(out3, index=False)
print(f"[3] {out3.name}: {len(slow)} rows")

# ===========================================================================
# 4. january_rtma_large_raw_files.csv
# ===========================================================================
rtma_rt = runtime[runtime["product"] == "rtma_conus_aws_2p5km"].copy()
large = (rtma_rt[rtma_rt["raw_file_size_bytes"] > 75e6]
         [["valid_time_utc", "raw_file_path", "raw_file_size_bytes", "file_reused",
           "download_time_s", "total_processing_time_s"]]
         .sort_values("raw_file_size_bytes", ascending=False))
out4 = OUT_DIR / "january_rtma_large_raw_files.csv"
large.to_csv(out4, index=False)
print(f"[4] {out4.name}: {len(large)} RTMA files > 75 MB")

# ===========================================================================
# 5. january_output_preview_stratified.csv
# ===========================================================================
TARGET_STAIDS = ["10164500", "02344605", "03298135"]
TARGET_TIMES  = [
    "2023-01-01T00:00:00Z",
    "2023-01-05T00:00:00Z",
    "2023-01-10T00:00:00Z",
    "2023-01-15T00:00:00Z",
    "2023-01-20T00:00:00Z",
    "2023-01-25T00:00:00Z",
    "2023-01-31T23:00:00Z",
]
prev = comb[comb["STAID"].isin(TARGET_STAIDS) & comb["valid_time_utc"].isin(TARGET_TIMES)]
out5 = OUT_DIR / "january_output_preview_stratified.csv"
prev.to_csv(out5, index=False)
print(f"[5] {out5.name}: {len(prev)} rows  "
      f"STAIDs={sorted(prev['STAID'].unique())}  "
      f"n_times={prev['valid_time_utc'].nunique()}")

# ===========================================================================
# PRINT REPORT
# ===========================================================================
SEP = "=" * 72
print()
print(SEP)
print("DIAGNOSTIC REPORT — January 2023 Stage 1 Pilot Extraction")
print(SEP)

print("\n--- Shapes ---")
print(f"  MRMS:     {mrms.shape}")
print(f"  RTMA:     {rtma.shape}")
print(f"  Combined: {comb.shape}")

print("\n--- Columns (combined) ---")
print(" ", list(comb.columns))

print("\n--- Products ---")
print(" ", comb["product"].unique().tolist())

print("\n--- RTMA variables (expected 11) ---")
print(" ", sorted(rtma["variable"].unique()))
print(f"  Count: {rtma['variable'].nunique()}")

print("\n--- 10wdir / orog check ---")
print(f"  10wdir present: {'10wdir' in rtma['variable'].unique()}")
print(f"  orog   present: {'orog' in rtma['variable'].unique()}")

print("\n--- weighted_mean min/max by variable ---")
for (prod, var), g in comb.groupby(["product", "variable"], sort=True):
    wm  = g["weighted_mean"].dropna()
    u   = g["units"].iloc[0]
    n_n = int(g["weighted_mean"].isna().sum())
    print(f"  {prod[:22]:22s}  {var:8s}  min={wm.min():>12.4g}  max={wm.max():>12.4g}  nulls={n_n:4d}  [{u}]")

print("\n--- Top 10 slowest product-hours ---")
top10 = slow.head(10)[["product", "valid_time_utc", "total_processing_time_s",
                        "download_time_s", "raw_file_size_bytes", "status"]]
for _, r in top10.iterrows():
    mb = r["raw_file_size_bytes"] / 1e6 if pd.notna(r["raw_file_size_bytes"]) else float("nan")
    print(f"  {r['product'][:24]:24s}  {r['valid_time_utc']}  "
          f"proc={r['total_processing_time_s']:6.1f}s  dl={r['download_time_s']:6.1f}s  "
          f"size={mb:6.1f}MB  status={r['status']}")

print(f"\n--- RTMA raw files > 75 MB: {len(large)} ---")
if len(large):
    print("  Top entries (MB):")
    for _, r in large.head(5).iterrows():
        print(f"    {r['valid_time_utc']}  {r['raw_file_size_bytes']/1e6:.1f} MB  reused={r['file_reused']}")
else:
    print("  None — all RTMA files are selected-message size (<= 75 MB)")

print("\n--- Valid weight fraction range (all products) ---")
for (prod, var), g in comb.groupby(["product","variable"], sort=True):
    vwf = g["valid_weight_fraction"].dropna()
    if len(vwf):
        print(f"  {prod[:22]:22s}  {var:8s}  vwf [{vwf.min():.3f} – {vwf.max():.3f}]  "
              f"median={np.median(vwf):.3f}")

print()
print(f"All 5 diagnostics written to: {OUT_DIR}")
print(SEP)
