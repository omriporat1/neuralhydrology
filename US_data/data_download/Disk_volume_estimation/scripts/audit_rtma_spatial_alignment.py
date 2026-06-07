"""
audit_rtma_spatial_alignment.py — RTMA spatial sanity check for Flash-NH Stage 1 animations.

Verifies that no orientation/cropping bug exists for RTMA (analogous to the
MRMS lat-decreasing-with-row bug found in v1 animations).

RTMA grid convention (documented here):
  - Lambert Conformal Conic, curvilinear 2D lat/lon; 1597 rows × 2345 cols
  - lat[0,0]=19.229 N (SW corner), lat[-1,-1]=54.373 N (NE corner)
  - Latitude INCREASES with row (row 0 = southernmost) — OPPOSITE to MRMS
  - lon stored as 0-360; convert lon -= 360 where lon > 180 for geographic mapping
  - weight row_idx/col_idx index directly into data[row_idx, col_idx]; no formula needed
  - cfgrib variable names: 't2m' (parquet: '2t'), 'u10' (parquet: '10u'), 'v10' (parquet: '10v')
  - Units: 2t in K, 10u/10v in m/s

Script exits with code 1 if any basin-variable exceeds the tolerance threshold,
allowing it to be used as a gate before animation generation.

Audit frames (R02, R11, R06):
  - dry/baseline frame (first window hour)
  - peak-precipitation frame
  - near-peak-flow frame

Outputs (stable, no version suffix):
  tmp/.../10_animations/stage1_pilot/rtma_spatial_audit.csv
  tmp/.../10_animations/stage1_pilot/rtma_spatial_audit.json
"""

import json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cfgrib

warnings.filterwarnings("ignore")

ROOT          = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\Disk_volume_estimation")
RTMA_BASE     = ROOT / "tmp/stage1_pilot_dryrun/00_raw/rtma"
FORCING_PQ    = ROOT / "tmp/stage1_pilot_dryrun/03_basin_timeseries/stage1_pilot/january_2023/combined_hourly_basin_stats.parquet"
RTMA_WEIGHTS  = ROOT / "tmp/stage1_pilot_dryrun/02_basin_geometries/weights/rtma/pilot_rtma_weights.parquet"
CANDIDATES_CSV = ROOT / "tmp/stage1_pilot_dryrun/09_manifests/stage1_pilot/january_2023_event_qc/event_animation_candidates_refined.csv"

ANIM_DIR = ROOT / "tmp/stage1_pilot_dryrun/10_animations/stage1_pilot"
ANIM_DIR.mkdir(parents=True, exist_ok=True)

TOLERANCE_PCT = 1.0

RTMA_VARS = [
    ("t2m", "2t",  "K"),
    ("u10", "10u", "m/s"),
    ("v10", "10v", "m/s"),
]

AUDIT_FRAMES = {
    "R02": {
        "staid": "07263580",
        "description": "STRONG_WET AR small basin; peak precip Jan-29 08Z, peak flow Jan-29 09Z",
        "frames": [
            (0,  "2023-01-28T09:00:00Z", "dry_baseline"),
            (23, "2023-01-29T08:00:00Z", "peak_precip"),
            (24, "2023-01-29T09:00:00Z", "peak_flow"),
        ],
    },
    "R11": {
        "staid": "01100627",
        "description": "OFFSET_STRESS MA basin, gauge offset 4km; peak precip Jan-23 09Z",
        "frames": [
            (0,  "2023-01-21T06:00:00Z", "dry_baseline"),
            (51, "2023-01-23T09:00:00Z", "peak_precip"),
            (53, "2023-01-23T11:00:00Z", "peak_flow"),
        ],
    },
    "R06": {
        "staid": "05372995",
        "description": "MODERATE_COLD_REGION MN larger basin; peak precip Jan-03 18Z",
        "frames": [
            (0,  "2023-01-02T10:00:00Z", "dry_baseline"),
            (33, "2023-01-03T18:00:00Z", "peak_precip"),
        ],
    },
}


def rtma_path(ts: pd.Timestamp) -> Path:
    return RTMA_BASE / f"rtma2p5.{ts.strftime('%Y%m%d')}" / f"rtma2p5.t{ts.strftime('%H')}z.2dvaranl_ndfd.grb2_wexp"


def load_rtma_full(ts: pd.Timestamp):
    p = rtma_path(ts)
    meta = {"source_file": str(p), "exists": p.exists()}
    if not p.exists():
        return None, meta
    try:
        dsets = cfgrib.open_datasets(str(p), indexpath=None)
        result = {}
        for ds in dsets:
            dvars = set(ds.data_vars)
            for varname in ("t2m", "u10", "v10"):
                if varname in dvars:
                    result[varname] = ds[varname].values
                    if "lat_2d" not in result:
                        result["lat_2d"] = ds["latitude"].values
                        result["lon_2d"] = ds["longitude"].values
                    if "valid_time" not in result:
                        vt = ds.coords.get("valid_time")
                        if vt is not None:
                            try:
                                result["valid_time"] = pd.Timestamp(str(vt.values)).tz_localize("UTC")
                            except Exception:
                                result["valid_time"] = None
        meta["variables_found"] = [k for k in ("t2m", "u10", "v10") if k in result]
        if "lat_2d" in result:
            lat = result["lat_2d"]
            lon = result["lon_2d"]
            meta["lat_shape"]   = list(lat.shape)
            meta["lat_0_0"]     = round(float(lat[0, 0]), 4)
            meta["lat_m1_m1"]   = round(float(lat[-1, -1]), 4)
            meta["lat_incr_row"]= bool(lat[-1, 0] > lat[0, 0])
            meta["lon_range_0_360"] = [round(float(lon.min()), 4), round(float(lon.max()), 4)]
        if "valid_time" in result:
            meta["grib_valid_time_utc"] = result["valid_time"].strftime("%Y-%m-%dT%H:%M:%SZ")
        return result, meta
    except Exception as e:
        meta["error"] = str(e)
        return None, meta


def basin_mean_from_rtma(data_2d, w_rows, w_cols, w_norms):
    nrows, ncols = data_2d.shape
    mask = (w_rows >= 0) & (w_rows < nrows) & (w_cols >= 0) & (w_cols < ncols)
    if not mask.any():
        return np.nan
    return float(np.sum(data_2d[w_rows[mask], w_cols[mask]] * w_norms[mask]))


def main():
    print("=" * 72)
    print("RTMA Spatial Alignment Audit — Flash-NH Stage 1")
    print("=" * 72)

    cands_df = pd.read_csv(CANDIDATES_CSV)

    print("Loading forcing parquet and RTMA weights…")
    pq_df = pd.read_parquet(FORCING_PQ)
    pq_df["valid_time_utc"] = pd.to_datetime(pq_df["valid_time_utc"], utc=True)
    pq_df["STAID"] = pq_df["STAID"].astype(str).str.zfill(8)

    wdf = pd.read_parquet(RTMA_WEIGHTS)
    wdf["STAID"] = wdf["STAID"].astype(str).str.zfill(8)

    # Document grid orientation from a known file
    probe_ts = pd.Timestamp("2023-01-29T08:00:00Z")
    probe_data, probe_meta = load_rtma_full(probe_ts)
    print("\n-- RTMA Grid Orientation (probe: Jan-29 08Z) --")
    print(f"  Grid shape        : {probe_meta.get('lat_shape')}")
    print(f"  lat[0,0]          = {probe_meta.get('lat_0_0')} N  (SW corner)")
    print(f"  lat[-1,-1]        = {probe_meta.get('lat_m1_m1')} N  (NE corner)")
    print(f"  lat increases/row : {probe_meta.get('lat_incr_row')}  (row 0 = southernmost; OPPOSITE to MRMS)")
    print(f"  lon range (0-360) : {probe_meta.get('lon_range_0_360')}")
    print(f"  Indexing          : data[row_idx, col_idx] — direct, no formula")

    grid_convention = {
        "grid_type": "Lambert Conformal Conic, curvilinear 2D lat/lon",
        "grid_shape": probe_meta.get("lat_shape"),
        "lat_0_0_N": probe_meta.get("lat_0_0"),
        "lat_m1_m1_N": probe_meta.get("lat_m1_m1"),
        "lat_increases_with_row": probe_meta.get("lat_incr_row"),
        "row_0_is": "southernmost (~19.23 N)",
        "lon_range_0_360": probe_meta.get("lon_range_0_360"),
        "lon_for_mapping": "subtract 360 where lon > 180",
        "row_col_indexing": "data[row_idx, col_idx] — direct index, weights pre-computed on same grid",
        "mrms_contrast": "MRMS lat DECREASES with row (row 0 = 54.995 N); RTMA lat INCREASES with row",
    }

    rows_out = []
    any_fail = False

    for rid, cfg in AUDIT_FRAMES.items():
        staid = cfg["staid"]
        row_cand = cands_df[cands_df["candidate_id"] == rid]
        g_lat = float(row_cand.iloc[0]["lat"]) if len(row_cand) else None
        g_lon = float(row_cand.iloc[0]["lon"]) if len(row_cand) else None

        w_sub   = wdf[wdf["STAID"] == staid]
        w_rows  = w_sub["row_idx"].values.astype(int)
        w_cols  = w_sub["col_idx"].values.astype(int)
        w_norms = w_sub["normalized_weight"].values.astype(float)

        print(f"\n{'=' * 72}")
        print(f"  {rid}  STAID={staid}  {cfg['description']}")
        print(f"  Weight cells: {len(w_sub)}  "
              f"row {w_rows.min()}-{w_rows.max()}  col {w_cols.min()}-{w_cols.max()}")
        print(f"{'=' * 72}")

        for fi, ts_str, scenario in cfg["frames"]:
            ts = pd.Timestamp(ts_str, tz="UTC")
            print(f"\n  fr{fi:3d}  {ts_str}  [{scenario}]")

            rtma_data, rtma_meta = load_rtma_full(ts)
            grib_vt = rtma_meta.get("grib_valid_time_utc")
            vt_match = (grib_vt == ts.strftime("%Y-%m-%dT%H:%M:%SZ")) if grib_vt else None
            print(f"    valid_time match: {vt_match}  grib={grib_vt}")

            frame_row = {
                "candidate_id": rid, "staid": staid, "frame_index": fi,
                "scenario": scenario, "timestamp_utc": ts_str,
                "rtma_source_file": rtma_meta["source_file"],
                "rtma_grib_valid_time_utc": grib_vt,
                "filename_vs_grib_match": vt_match,
                "weight_cells_n": len(w_sub),
                "weight_row_range": f"{w_rows.min()}-{w_rows.max()}",
                "weight_col_range": f"{w_cols.min()}-{w_cols.max()}",
                "gauge_lat": g_lat, "gauge_lon": g_lon,
                "grid_convention": "lat_increases_with_row; row_idx direct index",
            }

            frame_pass = True
            for cfgrib_name, pq_var, units in RTMA_VARS:
                raster_mean = np.nan
                if rtma_data is not None and cfgrib_name in rtma_data:
                    raster_mean = basin_mean_from_rtma(rtma_data[cfgrib_name], w_rows, w_cols, w_norms)

                pq_row = pq_df[
                    (pq_df["STAID"] == staid) &
                    (pq_df["product"] == "rtma_conus_aws_2p5km") &
                    (pq_df["variable"] == pq_var) &
                    (pq_df["valid_time_utc"] == ts)]
                pq_mean = float(pq_row["weighted_mean"].iloc[0]) if len(pq_row) == 1 else np.nan

                abs_diff = rel_diff_pct = np.nan
                var_pass = False
                if not np.isnan(raster_mean) and not np.isnan(pq_mean):
                    abs_diff = abs(raster_mean - pq_mean)
                    rel_diff_pct = abs_diff / max(abs(pq_mean), 1e-6) * 100
                    var_pass = rel_diff_pct < TOLERANCE_PCT
                elif np.isnan(raster_mean) and np.isnan(pq_mean):
                    var_pass = True; rel_diff_pct = 0.0

                if not var_pass:
                    frame_pass = False; any_fail = True

                status = "PASS" if var_pass else "FAIL ***"
                print(f"    {pq_var:4s}  raster={raster_mean:10.5f} {units}  "
                      f"parquet={pq_mean:10.5f} {units}  "
                      f"diff={rel_diff_pct:.4f}%  [{status}]")
                if not var_pass:
                    print(f"    *** MISMATCH: raster={raster_mean:.6f}  parquet={pq_mean:.6f}")

                frame_row[f"{pq_var}_raster"] = round(raster_mean, 6) if not np.isnan(raster_mean) else None
                frame_row[f"{pq_var}_parquet"] = round(pq_mean, 6) if not np.isnan(pq_mean) else None
                frame_row[f"{pq_var}_rel_diff_pct"] = round(rel_diff_pct, 4) if not np.isnan(rel_diff_pct) else None
                frame_row[f"{pq_var}_pass"] = var_pass

            frame_row["frame_pass"] = frame_pass
            rows_out.append(frame_row)

    n_pass = sum(1 for r in rows_out if r["frame_pass"])
    n_total = len(rows_out)

    csv_path  = ANIM_DIR / "rtma_spatial_audit.csv"
    json_path = ANIM_DIR / "rtma_spatial_audit.json"
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)

    out = {
        "rtma_grid_convention": grid_convention,
        "tolerance_pct": TOLERANCE_PCT,
        "overall_pass": not any_fail,
        "frames_pass": n_pass, "frames_total": n_total,
        "frames": rows_out,
    }
    json_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 72}")
    print(f"  RTMA Spatial Audit: {n_pass}/{n_total} PASS")
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")

    if any_fail:
        print("\n  *** FAIL — do not proceed to animation generation ***")
        sys.exit(1)
    else:
        print("  ALL PASS — safe to proceed with animation generation.")


if __name__ == "__main__":
    main()
