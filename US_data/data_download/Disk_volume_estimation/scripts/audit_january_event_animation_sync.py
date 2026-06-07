"""
audit_january_event_animation_sync.py — MRMS raster vs parquet sync audit for Flash-NH Stage 1.

Verifies that basin-mean MRMS values recomputed directly from grib2.gz files
match values pre-extracted to combined_hourly_basin_stats.parquet, confirming
the extraction pipeline and animation rendering use consistent data.

MRMS convention (documented here):
  - Filename timestamp = grib valid_time = END of 1-hour accumulation
  - File 08:00Z covers QPE 07:00Z–08:00Z
  - Latitude DECREASES with row; row 0 = 54.995 N (northernmost)
  - Row formula: row = (54.995 - lat) / 0.01
  - Audit uses full MRMS grid (not a crop), so orientation does not affect results

Audit candidates: R02 and R11 at key event frames.

Outputs (stable, no version suffix):
  tmp/.../10_animations/stage1_pilot/sync_audit.csv
  tmp/.../10_animations/stage1_pilot/sync_audit.json
"""

import gzip, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import cfgrib
import netCDF4 as nc

warnings.filterwarnings("ignore")

ROOT        = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\Disk_volume_estimation")
CAMELSH_DIR = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\CAMELSH_resolution_test\data\raw\camelsh")

MRMS_BASE    = ROOT / "tmp/stage1_pilot_dryrun/00_raw/mrms/CONUS/MultiSensor_QPE_01H_Pass1_00.00"
FORCING_PQ   = ROOT / "tmp/stage1_pilot_dryrun/03_basin_timeseries/stage1_pilot/january_2023/combined_hourly_basin_stats.parquet"
MRMS_WEIGHTS = ROOT / "tmp/stage1_pilot_dryrun/02_basin_geometries/weights/mrms/pilot_mrms_weights.parquet"

ANIM_DIR = ROOT / "tmp/stage1_pilot_dryrun/10_animations/stage1_pilot"
ANIM_DIR.mkdir(parents=True, exist_ok=True)

MRMS_LAT_TOP = 54.995
MRMS_DLAT = MRMS_DLON = 0.01
MRMS_NROWS, MRMS_NCOLS = 3500, 7000

TOLERANCE_PCT = 2.0

AUDIT_FRAMES = {
    "R02": {
        "staid": "07263580",
        "frames": [
            (0,  "2023-01-28T09:00:00Z", "window_start_dry"),
            (15, "2023-01-29T00:00:00Z", "precip_onset"),
            (23, "2023-01-29T08:00:00Z", "peak_precip"),
            (24, "2023-01-29T09:00:00Z", "peak_flow"),
            (51, "2023-01-30T12:00:00Z", "recession"),
        ],
    },
    "R11": {
        "staid": "01100627",
        "frames": [
            (0,  "2023-01-21T06:00:00Z", "window_start_dry"),
            (33, "2023-01-22T15:00:00Z", "precip_onset"),
            (51, "2023-01-23T09:00:00Z", "peak_precip"),
            (53, "2023-01-23T11:00:00Z", "peak_flow"),
            (66, "2023-01-24T00:00:00Z", "recession"),
        ],
    },
}


def mrms_path(ts: pd.Timestamp) -> Path:
    ds = ts.strftime("%Y%m%d")
    hs = ts.strftime("%H0000")
    return MRMS_BASE / ds / f"MRMS_MultiSensor_QPE_01H_Pass1_00.00_{ds}-{hs}.grib2.gz"


def load_mrms_full(ts: pd.Timestamp, tmp_dir: Path):
    p = mrms_path(ts)
    if not p.exists():
        return None, {"error": "missing", "path": str(p)}
    tmp_f = tmp_dir / f"_sync_audit_{ts.strftime('%Y%m%d_%H')}.grib2"
    try:
        with gzip.open(p, "rb") as gz:
            tmp_f.write_bytes(gz.read())
        ds = cfgrib.open_dataset(str(tmp_f), indexpath=None)
        data = np.where(ds["unknown"].values < 0, 0.0, ds["unknown"].values)
        meta = {
            "source_file": str(p),
            "filename_timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        vt = getattr(ds, "valid_time", None)
        if vt is not None:
            try:
                vt_val = pd.Timestamp(str(vt.values)).tz_localize("UTC")
                meta["grib_valid_time_utc"] = vt_val.strftime("%Y-%m-%dT%H:%M:%SZ")
                meta["filename_vs_grib_match"] = (vt_val == ts)
            except Exception:
                meta["grib_valid_time_utc"] = str(vt.values)
                meta["filename_vs_grib_match"] = "parse_error"
        return data, meta
    except Exception as e:
        return None, {"error": str(e)}
    finally:
        if tmp_f.exists():
            tmp_f.unlink()


def basin_mean_full(data, w_rows, w_cols, w_norms):
    mask = (w_rows >= 0) & (w_rows < MRMS_NROWS) & (w_cols >= 0) & (w_cols < MRMS_NCOLS)
    if not mask.any():
        return 0.0
    return float(np.sum(data[w_rows[mask], w_cols[mask]] * w_norms[mask]))


def load_sf(staid, ts):
    f = CAMELSH_DIR / f"{staid}_hourly.nc"
    ds = nc.Dataset(f)
    tv = ds.variables["time"]
    tms = pd.to_datetime(
        [t.strftime("%Y-%m-%d %H:%M:%S")
         for t in nc.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))],
        utc=True)
    sf = pd.Series(
        np.ma.filled(ds.variables["streamflow"][:].squeeze().astype(float), np.nan), index=tms)
    ds.close()
    return float(sf.get(ts, np.nan))


def main():
    print("=" * 70)
    print("MRMS Sync Audit — Flash-NH Stage 1")
    print("=" * 70)

    pq_df = pd.read_parquet(FORCING_PQ)
    pq_df["valid_time_utc"] = pd.to_datetime(pq_df["valid_time_utc"], utc=True)
    pq_df["STAID"] = pq_df["STAID"].astype(str).str.zfill(8)

    wdf = pd.read_parquet(MRMS_WEIGHTS)
    wdf["STAID"] = wdf["STAID"].astype(str).str.zfill(8)

    tmp_dir = ANIM_DIR / "_sync_audit_tmp"
    tmp_dir.mkdir(exist_ok=True)

    weight_cache = {}
    for rid, cfg in AUDIT_FRAMES.items():
        staid = cfg["staid"]
        w = wdf[wdf["STAID"] == staid]
        weight_cache[staid] = (
            w["row_idx"].values.astype(int),
            w["col_idx"].values.astype(int),
            w["normalized_weight"].values.astype(float))

    rows = []
    any_fail = False

    for rid, cfg in AUDIT_FRAMES.items():
        staid = cfg["staid"]
        wr, wc, wn = weight_cache[staid]
        print(f"\n  {rid}  STAID={staid}")

        for fi, ts_str, scenario in cfg["frames"]:
            ts = pd.Timestamp(ts_str, tz="UTC")
            data, meta = load_mrms_full(ts, tmp_dir)

            raster_mean = basin_mean_full(data, wr, wc, wn) if data is not None else np.nan
            grib_vt     = meta.get("grib_valid_time_utc", "?")
            file_match  = meta.get("filename_vs_grib_match", False)

            pq_row = pq_df[
                (pq_df["STAID"] == staid) &
                (pq_df["product"] == "mrms_qpe_1h_pass1") &
                (pq_df["valid_time_utc"] == ts)]
            pq_mean = float(pq_row["weighted_mean"].iloc[0]) if len(pq_row) == 1 else np.nan

            pq_rt2m = pq_df[
                (pq_df["STAID"] == staid) &
                (pq_df["product"] == "rtma_conus_aws_2p5km") &
                (pq_df["variable"] == "2t") &
                (pq_df["valid_time_utc"] == ts)]
            t2m_c = float(pq_rt2m["weighted_mean"].iloc[0]) - 273.15 if len(pq_rt2m) == 1 else np.nan

            sf_val = load_sf(staid, ts)

            pq_raster_match = (
                abs(raster_mean - pq_mean) / max(abs(pq_mean), 0.01) * 100 < TOLERANCE_PCT
                if not (np.isnan(raster_mean) or np.isnan(pq_mean)) else
                (np.isnan(raster_mean) and np.isnan(pq_mean)))
            pass_flag = pq_raster_match and (file_match is True)

            if not pass_flag:
                any_fail = True

            diff_pct = (abs(raster_mean - pq_mean) / max(abs(pq_mean), 0.01) * 100
                        if not (np.isnan(raster_mean) or np.isnan(pq_mean)) else np.nan)

            flag = "PASS" if pass_flag else "FAIL ***"
            print(f"    fr{fi:3d}  {ts_str[:16]:16s}  {scenario:25s}  "
                  f"raster={raster_mean:8.4f}  pq={pq_mean:8.4f}  "
                  f"diff={diff_pct:.4f}%  [{flag}]")

            rows.append({
                "candidate_id":            rid,
                "staid":                   staid,
                "frame_index":             fi,
                "scenario":                scenario,
                "timestamp_utc":           ts_str,
                "mrms_raster_mean":        round(raster_mean, 4) if not np.isnan(raster_mean) else None,
                "mrms_parquet_mean":       round(pq_mean, 4) if not np.isnan(pq_mean) else None,
                "diff_pct":                round(diff_pct, 4) if not np.isnan(diff_pct) else None,
                "grib_valid_time_utc":     grib_vt,
                "filename_vs_grib_match":  str(file_match),
                "rtma_2t_C":               round(t2m_c, 3) if not np.isnan(t2m_c) else None,
                "sf_m3s":                  round(sf_val, 4) if not np.isnan(sf_val) else None,
                "pass_flag":               pass_flag,
            })

    n_pass  = sum(1 for r in rows if r["pass_flag"])
    n_total = len(rows)

    csv_path  = ANIM_DIR / "sync_audit.csv"
    json_path = ANIM_DIR / "sync_audit.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    out = {
        "mrms_convention": (
            "Filename timestamp = valid_time = END of 1h accumulation. "
            "File 08:00Z covers QPE 07:00Z-08:00Z."),
        "mrms_lat_convention": (
            "CORRECTED: lat DECREASES with row. Row 0 = 54.995 N. "
            "Audit uses full grid (no crop), so orientation does not affect results."),
        "tolerance_pct": TOLERANCE_PCT,
        "overall_pass": not any_fail,
        "frames_pass": n_pass,
        "frames_total": n_total,
        "frames": rows,
    }
    json_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    print(f"\n  MRMS Sync Audit: {n_pass}/{n_total} PASS")
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")

    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    if any_fail:
        print("\n  *** FAIL — raster/parquet mismatch detected ***")
        sys.exit(1)
    else:
        print("  ALL PASS — extraction and animation pipeline consistent.")


if __name__ == "__main__":
    main()
