"""
discover_rtma_urma_precip_january2023.py — RTMA/URMA-family precipitation discovery.

Diagnostic-only. Does NOT modify Stage 1 model inputs or forcing datasets.

Objective
---------
Find a precipitation field in the RTMA/URMA product family that can be used
as a qualitative cross-check against MRMS QPE for the January 2023 event windows,
thereby increasing confidence in RTMA/URMA spatial alignment and accumulation
handling already used in Stage 1.

Product inspection order
------------------------
1. Regular RTMA (exact product already used in Stage 1 extraction).
   Source: locally-cached files under tmp/.../00_raw/rtma/
   Why first: already validated, no new download needed. If it contains a
   precipitation field, that is the most direct diagnostic possible.

2. URMA QPE (Unrestricted Mesoscale Analysis precipitation analysis).
   Source: noaa-urma-pds S3 bucket.
   Why second: URMA is closely related to RTMA, likely on the same 2.5 km CONUS
   grid, and documentation suggests 1h QPE may be available here.
   If it is on the same grid, existing Stage 1 RTMA basin weights can be reused.

3. RTMA-RU (Rapid Update).
   Checked only as a fallback if 1 and 2 are both unavailable/unusable.
   Not implemented for extraction in this run.

Pilot candidates probed (from Milestone 2E refined candidates)
--------------------------------------------------------------
  R02 — AR STRONG_WET   peak Jan-29 08Z
  R06 — MN MODERATE     peak Jan-03 18Z
  R11 — MA OFFSET       peak Jan-23 09Z

Key RTMA grid reference (validated in Milestone 2E audits)
----------------------------------------------------------
  Lambert Conformal Conic, 1597 rows × 2345 cols
  lat increases with row; lat[0,0] ≈ 19.23 N; lat[-1,-1] ≈ 54.37 N
  lon stored 0-360; convert lon -= 360 where > 180 for geographic mapping
  data[row_idx, col_idx] — direct index into existing weight table

Stop conditions
---------------
  Script prints a clear STOP reason and exits with a non-zero code if:
  - No suitable precipitation field is found in any inspected product.
  - Any precipitation field found is grid-incompatible with existing RTMA weights.
  In both cases full JSON/Markdown reports are still written.

Outputs
-------
  tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics/discovery/
    rtma_urma_precip_inventory_sample.csv
    rtma_urma_precip_discovery_report.json
    rtma_urma_precip_discovery_report.md

Usage
-----
    python scripts/discover_rtma_urma_precip_january2023.py
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -- Paths ----------------------------------------------------------------------

ROOT = Path(r"C:\PhD\Python\neuralhydrology\US_data\data_download\Disk_volume_estimation")

RTMA_CACHE_BASE = ROOT / "tmp/stage1_pilot_dryrun/00_raw/rtma"
RTMA_WEIGHTS_PQ = (
    ROOT / "tmp/stage1_pilot_dryrun/02_basin_geometries/weights/rtma"
    / "pilot_rtma_weights.parquet"
)

DIAG_ROOT = ROOT / "tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics"
DISC_DIR  = DIAG_ROOT / "discovery"
TMP_DIR   = DISC_DIR / "_tmp_downloads"

# -- S3 buckets -----------------------------------------------------------------

# Regular RTMA (Stage 1 product already in use)
RTMA_BUCKET         = "noaa-rtma-pds"
RTMA_REGULAR_PREFIX = "rtma2p5."
RTMA_ANALYSIS_SUFFIX = "2dvaranl_ndfd.grb2_wexp"

# URMA (candidate for QPE)
URMA_BUCKET          = "noaa-urma-pds"
URMA_PREFIX          = "urma2p5."
URMA_ANALYSIS_SUFFIX = "2dvaranl_ndfd.grb2_wexp"

# RTMA-RU (fallback only; not implemented for extraction in this run)
RTMA_RU_PREFIX_CANDIDATES = ["rtma2p5_ru.", "rtma2p5.ru.", "rtma2p5ru."]

# -- Reference RTMA grid (validated Milestone 2E) ------------------------------

RTMA_REFERENCE_GRID = {
    "grid_shape":            [1597, 2345],
    "lat_0_0_N":             19.229,
    "lat_m1_m1_N":           54.373,
    "lat_increases_with_row": True,
    "lon_range_0_360":       [211.131, 299.066],
}

# -- Candidate event windows (peak ± 12 h) -------------------------------------

CANDIDATE_WINDOWS = {
    "R02": {
        "staid":           "07263580",
        "state":           "AR",
        "category":        "STRONG_WET",
        "peak_precip_utc": "2023-01-29T08:00:00Z",
        # representative hours to probe (hourly, ± ~12 h around peak)
        "probe_hours": [
            datetime(2023, 1, 28, 20),
            datetime(2023, 1, 29,  0),
            datetime(2023, 1, 29,  4),
            datetime(2023, 1, 29,  8),   # peak
            datetime(2023, 1, 29, 12),
            datetime(2023, 1, 29, 16),
            datetime(2023, 1, 29, 20),
        ],
    },
    "R06": {
        "staid":           "05372995",
        "state":           "MN",
        "category":        "MODERATE_COLD_REGION",
        "peak_precip_utc": "2023-01-03T18:00:00Z",
        "probe_hours": [
            datetime(2023, 1,  3,  6),
            datetime(2023, 1,  3, 10),
            datetime(2023, 1,  3, 14),
            datetime(2023, 1,  3, 18),   # peak
            datetime(2023, 1,  3, 22),
            datetime(2023, 1,  4,  2),
            datetime(2023, 1,  4,  6),
        ],
    },
    "R11": {
        "staid":           "01100627",
        "state":           "MA",
        "category":        "OFFSET_STRESS",
        "peak_precip_utc": "2023-01-23T09:00:00Z",
        "probe_hours": [
            datetime(2023, 1, 22, 21),
            datetime(2023, 1, 23,  0),
            datetime(2023, 1, 23,  4),
            datetime(2023, 1, 23,  9),   # peak
            datetime(2023, 1, 23, 13),
            datetime(2023, 1, 23, 17),
            datetime(2023, 1, 23, 20),
        ],
    },
}

# -- Precipitation field recognition -------------------------------------------

PRECIP_SHORT_NAMES: frozenset[str] = frozenset({
    "tp", "apcp", "tprec", "prate", "prec", "precip",
    "acpcp", "ncpcp", "asnow", "tp15m", "tp1h",
    "qpf",   # quantitative precipitation forecast (occasionally used)
})
PRECIP_GRIB_NAME_FRAGMENTS: frozenset[str] = frozenset({
    "total precipitation", "accumulated precipitation",
    "precipitation amount", "precipitation rate",
    "convective precip", "large scale precip",
    "quantitative precip",
})
PRECIP_PARAM_IDS: frozenset[int] = frozenset({61, 62, 63, 228, 260267})


# -- S3 helpers -----------------------------------------------------------------

def _s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    return boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))


def _list_prefixes(s3, bucket: str, prefix: str = "") -> list[str]:
    out: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            out.append(entry.get("Prefix", ""))
    return out


def _list_objects(s3, bucket: str, prefix: str) -> list[dict]:
    out: list[dict] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        out.extend(page.get("Contents", []))
    return out


def _download_key(s3, bucket: str, key: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        with local.open("wb") as fh:
            while chunk := resp["Body"].read(8 * 1024 * 1024):
                fh.write(chunk)
        return True
    except Exception as exc:
        print(f"      Download failed: {exc}")
        return False


def _parse_datetime_from_key(key: str) -> Optional[datetime]:
    """Parse datetime from a URMA/RTMA S3 key.  Handles HH00Z and HHmmZ patterns."""
    day_m    = re.search(r"\.(\d{8})/", key)
    hhmm_m   = re.search(r"\.t(\d{2})(\d{2})z\.", key)
    hh_m     = re.search(r"\.t(\d{2})z\.", key)
    day_str  = day_m.group(1) if day_m else None
    if not day_str:
        return None
    if hhmm_m:
        hh, mm = hhmm_m.group(1), hhmm_m.group(2)
    elif hh_m:
        hh, mm = hh_m.group(1), "00"
    else:
        return None
    try:
        return datetime.strptime(f"{day_str}{hh}{mm}", "%Y%m%d%H%M")
    except ValueError:
        return None


def _parse_dt_flexible(s) -> Optional[datetime]:
    """Parse a datetime value that may arrive in many string forms or already be a datetime.

    Accepted inputs:
      "2023-01-29T00:00:00"   "2023-01-29 00:00:00"
      "2023-01-29T00:00Z"     "2023-01-29 00:00Z"
      "2023-01-29T00:00:00Z"  pandas Timestamp  datetime object
    Returns a tz-naive UTC datetime or None on failure.
    """
    if s is None:
        return None
    if isinstance(s, datetime):
        return s.replace(tzinfo=None)
    # pandas Timestamp
    if hasattr(s, "to_pydatetime"):
        try:
            return s.to_pydatetime().replace(tzinfo=None)
        except Exception:
            pass
    s_norm = str(s).strip().rstrip("Z").strip().replace(" ", "T")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%dT%H",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s_norm, fmt)
        except ValueError:
            continue
    return None


def _check_dt_flexible_sanity() -> None:
    """Inline sanity check — raises AssertionError if parsing is broken."""
    cases = [
        ("2023-01-29T00:00:00",  datetime(2023, 1, 29,  0,  0,  0)),
        ("2023-01-29 00:00:00",  datetime(2023, 1, 29,  0,  0,  0)),
        ("2023-01-29T00:00Z",    datetime(2023, 1, 29,  0,  0,  0)),
        ("2023-01-29 00:00Z",    datetime(2023, 1, 29,  0,  0,  0)),
        ("2023-01-29T08:00:00Z", datetime(2023, 1, 29,  8,  0,  0)),
        (datetime(2023, 1, 3, 18), datetime(2023, 1,  3, 18,  0,  0)),
    ]
    for inp, expected in cases:
        got = _parse_dt_flexible(inp)
        assert got == expected, f"_parse_dt_flexible({inp!r}) -> {got!r}, expected {expected!r}"
    assert _parse_dt_flexible(None) is None


# -- GRIB inventory -------------------------------------------------------------

def _inventory_grib(path: Path) -> list[dict]:
    """Inventory all 2-D variables in a GRIB2 file via cfgrib; eccodes fallback."""
    records: list[dict] = []

    # ---- cfgrib ----
    try:
        import cfgrib
        datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
        for ds in datasets:
            for vname in list(ds.data_vars):
                try:
                    da    = ds[vname]
                    attrs = da.attrs
                    arr   = np.squeeze(np.asarray(da.values, dtype=np.float64))
                    flat  = arr[np.isfinite(arr)] if arr.ndim >= 2 else np.array([])

                    def _ts(cnames):
                        for cn in cnames:
                            c = da.coords.get(cn)
                            if c is not None:
                                raw = str(np.asarray(c.values).reshape(-1)[0]).split(".")[0]
                                return (raw + "Z") if ("T" in raw and not raw.endswith("Z")) else raw
                        return None

                    records.append({
                        "short_name":        str(attrs.get("GRIB_shortName", vname)).lower(),
                        "cfgrib_var_name":    vname,
                        "grib_name":          str(attrs.get("GRIB_name",     "?")),
                        "grib_param_id":      attrs.get("GRIB_paramId",     "?"),
                        "grib_units":         str(attrs.get("GRIB_units",    "?")),
                        "level":              attrs.get("GRIB_level",       "?"),
                        "level_type":         str(attrs.get("GRIB_typeOfLevel", "?")),
                        "step_range":         str(attrs.get("GRIB_stepRange",
                                                            attrs.get("GRIB_step", "?"))),
                        "step_type":          str(attrs.get("GRIB_stepType", "?")),
                        "valid_time_utc":     _ts(("valid_time", "time")),
                        "reference_time_utc": _ts(("time", "forecast_reference_time")),
                        "grid_shape":         list(arr.shape) if arr.ndim == 2 else ["?"],
                        "data_min":   round(float(flat.min()), 6) if len(flat) > 0 else None,
                        "data_max":   round(float(flat.max()), 6) if len(flat) > 0 else None,
                        "nonzero_frac": (round(float((flat > 0).mean()), 4)
                                         if len(flat) > 0 else 0.0),
                        "inventory_method": "cfgrib",
                        "source_file":      path.name,
                    })
                except Exception as exc:
                    records.append({
                        "short_name": "?error?", "cfgrib_var_name": vname,
                        "grib_name": str(exc), "inventory_method": "cfgrib_error",
                        "source_file": path.name,
                    })
        if records:
            return records
    except Exception as exc_cf:
        print(f"      cfgrib failed ({exc_cf}); trying eccodes …")

    # ---- eccodes fallback ----
    try:
        from eccodes import (codes_get, codes_get_string,
                             codes_grib_new_from_file, codes_release)

        def _g(gid, k, d="?"):
            try:
                return codes_get(gid, k)
            except Exception:
                return d

        def _gs(gid, k, d="?"):
            try:
                return codes_get_string(gid, k)
            except Exception:
                try:
                    return str(codes_get(gid, k))
                except Exception:
                    return d

        with path.open("rb") as fh:
            while True:
                gid = codes_grib_new_from_file(fh)
                if gid is None:
                    break
                try:
                    vd = _g(gid, "validityDate", 0)
                    vt = _g(gid, "validityTime", 0)
                    vt_str = None
                    if isinstance(vd, int) and vd > 0:
                        try:
                            vt_str = datetime.strptime(
                                f"{vd:08d}{vt:04d}", "%Y%m%d%H%M"
                            ).strftime("%Y-%m-%dT%H:%M:%SZ")
                        except Exception:
                            pass
                    nr = _g(gid, "Nj", "?")
                    nc = _g(gid, "Ni", "?")
                    records.append({
                        "short_name":   str(_gs(gid, "shortName", "?")).lower(),
                        "cfgrib_var_name": "?",
                        "grib_name":    _gs(gid, "name", "?"),
                        "grib_param_id": _g(gid, "paramId", "?"),
                        "grib_units":   _gs(gid, "units", "?"),
                        "level":        _g(gid, "level", "?"),
                        "level_type":   _gs(gid, "typeOfLevel", "?"),
                        "step_range":   _gs(gid, "stepRange", "?"),
                        "step_type":    _gs(gid, "stepType", "?"),
                        "valid_time_utc": vt_str,
                        "reference_time_utc": None,
                        "grid_shape":   [nr, nc],
                        "data_min": None, "data_max": None, "nonzero_frac": None,
                        "inventory_method": "eccodes",
                        "source_file":  path.name,
                    })
                except Exception:
                    pass
                finally:
                    try:
                        codes_release(gid)
                    except Exception:
                        pass
    except ImportError:
        print("      eccodes not available")
    except Exception as exc_ec:
        print(f"      eccodes failed: {exc_ec}")

    return records


def _is_precip(rec: dict) -> bool:
    sn  = str(rec.get("short_name", "")).lower()
    nm  = str(rec.get("grib_name",  "")).lower()
    pid = rec.get("grib_param_id", "?")
    return (
        sn in PRECIP_SHORT_NAMES
        or any(f in nm for f in PRECIP_GRIB_NAME_FRAGMENTS)
        or (isinstance(pid, int) and pid in PRECIP_PARAM_IDS)
    )


def _characterise_precip(rec: dict) -> dict:
    sr   = str(rec.get("step_range", "?"))
    st   = str(rec.get("step_type",  "?")).lower()
    un   = str(rec.get("grib_units", "?")).lower()
    # Infer accumulation window
    accum_min: Optional[int] = None
    try:
        if "-" in sr:
            a, b = sr.split("-", 1)
            accum_min = int(round((float(b) - float(a)) * 60))
        elif sr.endswith("h"):
            accum_min = int(float(sr[:-1]) * 60)
        elif sr.endswith("m"):
            accum_min = int(float(sr[:-1]))
    except Exception:
        pass
    return {
        "step_range":             sr,
        "step_type":              st,
        "grib_units":             un,
        "inferred_accum_minutes": accum_min,
        "is_accumulated":         st in {"accum", "accum_adj"},
        "is_instantaneous":       st in {"instant", "avg", "diff"},
        "units_equiv_mm":         any(u in un for u in ("kg/m2", "kg m-2", "mm", "m of water")),
        "units_rate":             any(u in un for u in ("/s", "/h", "s-1")),
        "data_max":               rec.get("data_max"),
        "nonzero_frac":           rec.get("nonzero_frac"),
    }


# -- Grid helpers ---------------------------------------------------------------

def _grid_from_grib(path: Path) -> Optional[dict]:
    """Return grid metadata dict from a cfgrib-readable GRIB2 file."""
    try:
        import cfgrib
        for ds in cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}):
            for vn in ds.data_vars:
                da  = ds[vn]
                lat = da.coords.get("latitude")
                lon = da.coords.get("longitude")
                if lat is not None and lat.values.ndim == 2:
                    lv, lnv = lat.values, lon.values
                    return {
                        "grid_shape":            list(lv.shape),
                        "lat_0_0_N":             round(float(lv[0, 0]),   4),
                        "lat_m1_m1_N":           round(float(lv[-1, -1]), 4),
                        "lat_increases_with_row": bool(lv[-1, 0] > lv[0, 0]),
                        "lon_range_0_360":       [round(float(lnv.min()), 4),
                                                  round(float(lnv.max()), 4)],
                        "source_file":           str(path),
                        "_lat_2d":               lv,
                        "_lon_2d":               lnv,
                    }
    except Exception as exc:
        print(f"      Grid extraction failed: {exc}")
    return None


def _check_compat(
    candidate_grid: dict,
    regular_grid:   Optional[dict],
    weights_df:     pd.DataFrame,
) -> dict:
    """Check whether existing RTMA weight table is compatible with candidate grid."""
    ru_shape = candidate_grid.get("grid_shape", [0, 0])
    ru_nr, ru_nc = ru_shape[0], ru_shape[1]
    reg_shape = regular_grid.get("grid_shape") if regular_grid else None

    w_rows = weights_df["row_idx"].values.astype(int)
    w_cols = weights_df["col_idx"].values.astype(int)
    inb    = (w_rows >= 0) & (w_rows < ru_nr) & (w_cols >= 0) & (w_cols < ru_nc)
    n_oob  = int((~inb).sum())

    # Lat/lon spot-check against regular RTMA
    lc: Optional[dict] = None
    sample_rows: list[dict] = []
    ru_lat  = candidate_grid.get("_lat_2d")
    ru_lon  = candidate_grid.get("_lon_2d")
    reg_lat = regular_grid.get("_lat_2d") if regular_grid else None
    reg_lon = regular_grid.get("_lon_2d") if regular_grid else None

    if ru_lat is not None and reg_lat is not None:
        step = max(1, len(w_rows) // 50)
        max_dlat = max_dlon = 0.0
        for i in range(0, len(w_rows), step):
            r, c = int(w_rows[i]), int(w_cols[i])
            if r < ru_nr and c < ru_nc and r < reg_lat.shape[0] and c < reg_lat.shape[1]:
                dlat = abs(float(ru_lat[r, c]) - float(reg_lat[r, c]))
                dlon = abs(float(ru_lon[r, c]) - float(reg_lon[r, c]))
                max_dlat = max(max_dlat, dlat)
                max_dlon = max(max_dlon, dlon)
                if len(sample_rows) < 10:
                    sample_rows.append({
                        "row": r, "col": c,
                        "lat_candidate": round(float(ru_lat[r, c]), 5),
                        "lon_candidate": round(float(ru_lon[r, c]), 5),
                        "lat_regular":   round(float(reg_lat[r, c]), 5),
                        "lon_regular":   round(float(reg_lon[r, c]), 5),
                        "dlat": round(dlat, 6), "dlon": round(dlon, 6),
                    })
        lc = {
            "n_cells_compared": len(sample_rows),
            "max_lat_diff_deg": round(max_dlat, 6),
            "max_lon_diff_deg": round(max_dlon, 6),
            "grids_coincident": (max_dlat < 0.001 and max_dlon < 0.001),
        }

    shapes_match = (ru_shape == reg_shape) if reg_shape else None
    compatible   = (
        (shapes_match is not False)        # None means we can't check — treat as ok
        and n_oob == 0
        and (lc is None or lc["grids_coincident"])
    )

    return {
        "candidate_grid_shape":    ru_shape,
        "regular_grid_shape":      reg_shape,
        "shapes_match":            shapes_match,
        "n_weight_cells_total":    len(w_rows),
        "n_weight_cells_oob":      n_oob,
        "frac_in_bounds":          round(float(inb.mean()), 6),
        "latlon_spot_check":       lc,
        "latlon_sample":           sample_rows,
        "weight_table_compatible": compatible,
    }


# -- Per-product inspection helpers --------------------------------------------

def _inspect_local_rtma(
    regular_grid_meta_holder: list,  # mutable; first element is set if found
) -> dict:
    """
    Inspect already-cached regular RTMA files for precipitation fields.
    Fills regular_grid_meta_holder[0] with grid metadata on success.
    Returns a result dict.
    """
    print("\n" + "-" * 70)
    print("PRODUCT 1 — Regular RTMA (locally cached Stage 1 files)")
    print("-" * 70)

    probe_times = [
        datetime(2023, 1, 29, 8),   # R02 peak
        datetime(2023, 1,  3, 18),  # R06 peak
        datetime(2023, 1, 23, 9),   # R11 peak
    ]
    result = {
        "product":        "rtma_conus_aws_2p5km",
        "bucket":         RTMA_BUCKET,
        "file_pattern":   f"{RTMA_REGULAR_PREFIX}YYYYMMDD/{RTMA_REGULAR_PREFIX}tHHz.{RTMA_ANALYSIS_SUFFIX}",
        "files_checked":  [],
        "all_variables":  [],
        "precip_fields":  [],
        "grid_meta":      None,
        "precip_found":   False,
    }

    all_vars: set[str] = set()
    for ts in probe_times:
        ds  = ts.strftime("%Y%m%d")
        hh  = ts.strftime("%H")
        p   = RTMA_CACHE_BASE / f"rtma2p5.{ds}" / f"rtma2p5.t{hh}z.{RTMA_ANALYSIS_SUFFIX}"
        exists = p.exists()
        print(f"\n  {p.name}  {'EXISTS' if exists else 'MISSING'}")
        result["files_checked"].append({"path": str(p), "exists": exists})
        if not exists:
            continue

        inv = _inventory_grib(p)
        snames = sorted({r.get("short_name", "?") for r in inv})
        all_vars.update(snames)
        print(f"    {len(inv)} GRIB messages.  Variables: {snames}")

        precip = [r for r in inv if _is_precip(r)]
        for pr in precip:
            char = _characterise_precip(pr)
            print(f"    *** PRECIP: {pr['short_name']!r}  units={pr['grib_units']}  "
                  f"step={pr['step_range']}/{pr['step_type']}  max={pr.get('data_max')}")
            result["precip_fields"].append({**pr, **char})

        # Grid meta — load once from first available file
        if result["grid_meta"] is None:
            gm = _grid_from_grib(p)
            if gm:
                result["grid_meta"] = {k: v for k, v in gm.items()
                                        if not k.startswith("_")}
                # Share with caller so URMA comparison has a reference
                regular_grid_meta_holder.append(gm)
                print(f"    Grid: {gm['grid_shape']}  lat[0,0]={gm['lat_0_0_N']} N  "
                      f"lat[-1,-1]={gm['lat_m1_m1_N']} N  "
                      f"lat↑row={gm['lat_increases_with_row']}")

    result["all_variables"] = sorted(all_vars)
    result["precip_found"]  = len(result["precip_fields"]) > 0

    if result["precip_found"]:
        print("\n  RESULT: Regular RTMA CONTAINS precipitation field(s).")
    else:
        print(f"\n  RESULT: Regular RTMA has NO precipitation field.")
        print(f"  Variables confirmed: {sorted(all_vars)}")

    return result


def _inspect_urma(
    s3,
    regular_grid: Optional[dict],
    weights_df:   pd.DataFrame,
) -> dict:
    """
    Inspect URMA (noaa-urma-pds) for QPE precipitation fields.
    Returns a result dict including grid compatibility information.
    """
    print("\n" + "-" * 70)
    print("PRODUCT 2 — URMA QPE (noaa-urma-pds)")
    print("-" * 70)

    result: dict[str, Any] = {
        "product":       "urma_conus_aws_2p5km",
        "bucket":        URMA_BUCKET,
        "bucket_accessible": False,
        "top_prefixes_found": [],
        "file_inventory":     {},   # date_str -> list of {key, size, dt}
        "sample_files":       [],
        "all_messages":       [],
        "precip_fields":      [],
        "grid_meta":          None,
        "grid_compat":        None,
        "precip_found":       False,
        "weights_compatible": False,
    }

    # ---- Bucket accessibility ----
    print(f"\n  Listing prefixes in s3://{URMA_BUCKET}/ …")
    try:
        top_pfx = _list_prefixes(s3, URMA_BUCKET)
        result["bucket_accessible"]   = True
        result["top_prefixes_found"]  = top_pfx
        n_pfx = len(top_pfx)
        print(f"  {n_pfx} top-level prefixes found.  Sample:")
        for p in sorted(top_pfx)[:30]:
            print(f"    {p}")
        if n_pfx > 30:
            print(f"    … and {n_pfx - 30} more")
    except Exception as exc:
        print(f"  Bucket not accessible: {exc}")
        result["bucket_error"] = str(exc)
        return result

    # ---- Per-date file listing ----
    probe_dates = sorted({dt.strftime("%Y%m%d")
                           for cfg in CANDIDATE_WINDOWS.values()
                           for dt in cfg["probe_hours"]})
    print(f"\n  Probing dates: {probe_dates}")

    found_any_files = False
    for date_str in probe_dates:
        pfx      = f"{URMA_PREFIX}{date_str}/"
        objs     = _list_objects(s3, URMA_BUCKET, pfx)
        day_info = []
        for obj in objs:
            key   = obj["Key"]
            size  = obj.get("Size", 0)
            dt_k  = _parse_datetime_from_key(key)
            fname = Path(key).name
            day_info.append({"key": key, "size": size, "datetime": dt_k,
                              "filename": fname})

        result["file_inventory"][date_str] = [
            {"key": d["key"], "size": d["size"],
             "datetime": str(d["datetime"]) if d["datetime"] else None,
             "filename": d["filename"]}
            for d in day_info
        ]

        # Classify filenames
        analysis_files = [f for f in day_info
                          if URMA_ANALYSIS_SUFFIX in f["filename"]]
        qpe_files      = [f for f in day_info
                          if "pcp" in f["filename"].lower()
                          or "qpe" in f["filename"].lower()]
        other_files    = [f for f in day_info
                          if f not in analysis_files and f not in qpe_files]

        if day_info:
            found_any_files = True
            print(f"\n  {date_str}: {len(day_info)} files total  "
                  f"(analysis={len(analysis_files)}, qpe={len(qpe_files)}, "
                  f"other={len(other_files)})")
            # Print QPE files first (most relevant)
            for f in qpe_files[:5]:
                dt_s = f["datetime"].strftime("%Y-%m-%dT%H:%MZ") if f["datetime"] else "?"
                print(f"    [QPE]      {f['filename']:<60s} {f['size']/1e6:6.1f} MB  {dt_s}")
            for f in analysis_files[:3]:
                dt_s = f["datetime"].strftime("%Y-%m-%dT%H:%MZ") if f["datetime"] else "?"
                print(f"    [analysis] {f['filename']:<60s} {f['size']/1e6:6.1f} MB  {dt_s}")
            if other_files:
                print(f"    [{len(other_files)} other files …]")
        else:
            print(f"\n  {date_str}: no files found under {pfx}")

    result["urma_found_any_files"] = found_any_files
    if not found_any_files:
        print("\n  RESULT: No URMA files found for any target date.")
        return result

    # ---- Download and inventory sample files ----
    # Prefer QPE-named files; fall back to analysis files.
    # Download at most 2 distinct sample files total.
    print("\n  Downloading sample URMA files for GRIB inventory …")
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    all_messages:   list[dict] = []
    precip_fields:  list[dict] = []
    sampled_keys:   set[str]   = set()
    n_samples = 0

    # Build a prioritised download list: (candidate_id, datetime, key, size)
    download_candidates: list[tuple] = []
    for rid, cfg in CANDIDATE_WINDOWS.items():
        peak_dt   = datetime.strptime(cfg["peak_precip_utc"], "%Y-%m-%dT%H:%M:%SZ")
        peak_date = peak_dt.strftime("%Y%m%d")
        day_files = [f for f in result["file_inventory"].get(peak_date, [])]
        # Sort QPE files by closeness to peak time
        for f in day_files:
            dt_f  = _parse_dt_flexible(f.get("datetime"))
            is_qpe = ("pcp" in f["filename"].lower() or "qpe" in f["filename"].lower())
            priority = 0 if is_qpe else 1
            diff = abs(dt_f - peak_dt).total_seconds() if dt_f else 999999
            download_candidates.append((priority, diff, rid, f["key"], f["size"]))

    download_candidates.sort()   # QPE first, then closest to peak

    urma_sample_grid: Optional[dict] = None
    for priority, diff, rid, key, size in download_candidates:
        if n_samples >= 2:
            break
        if key in sampled_keys:
            continue
        fname   = Path(key).name
        local_p = TMP_DIR / fname
        size_mb = size / 1e6
        print(f"\n    [{rid}] {fname}  ({size_mb:.1f} MB)  "
              f"{'[QPE]' if priority == 0 else '[analysis]'} …", end="", flush=True)
        t0_dl = time.time()
        ok    = _download_key(s3, URMA_BUCKET, key, local_p)
        dl_s  = time.time() - t0_dl
        if not ok:
            print(f" FAILED")
            continue
        print(f" {dl_s:.1f}s")
        sampled_keys.add(key)
        n_samples += 1

        msgs = _inventory_grib(local_p)
        for m in msgs:
            m["candidate_id"] = rid
            m["s3_key"]       = key
        all_messages.extend(msgs)

        print(f"    {len(msgs)} GRIB messages:")
        for m in msgs:
            tag = "  *** PRECIP ***" if _is_precip(m) else ""
            print(f"      {m.get('short_name','?'):15s} | "
                  f"{str(m.get('grib_name','?'))[:35]:35s} | "
                  f"{str(m.get('grib_units','?')):15s} | "
                  f"step={m.get('step_range','?')}/{m.get('step_type','?')}{tag}")

        pf = [m for m in msgs if _is_precip(m)]
        for m in pf:
            char = _characterise_precip(m)
            precip_fields.append({**m, **char})

        # Extract grid once
        if urma_sample_grid is None:
            urma_sample_grid = _grid_from_grib(local_p)
            if urma_sample_grid:
                result["grid_meta"] = {k: v for k, v in urma_sample_grid.items()
                                        if not k.startswith("_")}
                print(f"\n    Grid: {urma_sample_grid['grid_shape']}  "
                      f"lat[0,0]={urma_sample_grid['lat_0_0_N']} N  "
                      f"lat[-1,-1]={urma_sample_grid['lat_m1_m1_N']} N  "
                      f"lat↑row={urma_sample_grid['lat_increases_with_row']}")

        result["sample_files"].append({
            "candidate_id": rid, "s3_key": key,
            "size_bytes": size, "download_s": round(dl_s, 2),
            "n_messages": len(msgs),
        })

    result["all_messages"]  = all_messages
    result["precip_fields"] = [
        {k: v for k, v in pf.items() if not k.startswith("_")}
        for pf in precip_fields
    ]
    result["precip_found"]  = len(precip_fields) > 0

    if precip_fields:
        print(f"\n  Precipitation fields found ({len(precip_fields)}):")
        for pf in precip_fields:
            print(f"    short_name={pf['short_name']!r}  grib_name={pf['grib_name']!r}")
            print(f"      units={pf['grib_units']}  "
                  f"step={pf['step_range']}/{pf['step_type']}  "
                  f"accum_min={pf.get('inferred_accum_minutes')}  "
                  f"data_max={pf.get('data_max')}")
    else:
        all_snames = sorted({m.get("short_name", "?") for m in all_messages})
        print(f"\n  No precipitation fields found in sampled URMA files.")
        print(f"  Variables seen: {all_snames}")

    # ---- Grid compatibility ----
    if urma_sample_grid is not None:
        print("\n  Checking grid compatibility with existing RTMA weight table …")
        compat = _check_compat(urma_sample_grid, regular_grid, weights_df)
        result["grid_compat"] = {
            k: v for k, v in compat.items() if k != "latlon_sample"
        }
        result["grid_compat"]["latlon_sample"] = compat["latlon_sample"]
        print(f"    Shapes match           : {compat['shapes_match']}")
        print(f"    Weight cells OOB       : {compat['n_weight_cells_oob']}")
        print(f"    Lat/lon coincident     : "
              f"{compat['latlon_spot_check'].get('grids_coincident') if compat['latlon_spot_check'] else 'N/A'}")
        print(f"    WEIGHT TABLE COMPAT    : {compat['weight_table_compatible']}")
        result["weights_compatible"] = compat["weight_table_compatible"]

    return result


def _check_rtmaru_fallback(s3) -> dict:
    """
    Brief RTMA-RU availability check (fallback; not implemented for extraction).
    Returns a minimal dict documenting what was found.
    """
    print("\n" + "-" * 70)
    print("PRODUCT 3 — RTMA-RU (fallback check only; extraction NOT implemented)")
    print("-" * 70)

    found_any = False
    prefixes_found: list[str] = []
    file_counts: dict[str, int] = {}

    try:
        top = _list_prefixes(s3, RTMA_BUCKET)
    except Exception as exc:
        print(f"  Could not list {RTMA_BUCKET}: {exc}")
        return {"checked": False, "error": str(exc)}

    for pfx in top:
        for cand in RTMA_RU_PREFIX_CANDIDATES:
            if pfx.startswith(cand):
                prefixes_found.append(pfx)
                break

    # Also check per target date inside regular RTMA prefix for _ru_ filenames
    probe_dates = sorted({dt.strftime("%Y%m%d")
                           for cfg in CANDIDATE_WINDOWS.values()
                           for dt in cfg["probe_hours"]})
    ru_files_in_regular: dict[str, list[str]] = {}
    for date_str in probe_dates:
        regular_pfx = f"{RTMA_REGULAR_PREFIX}{date_str}/"
        objs = _list_objects(s3, RTMA_BUCKET, regular_pfx)
        ru_in_day = [o["Key"] for o in objs
                     if "_ru" in Path(o["Key"]).name.lower()
                     or "rapid" in Path(o["Key"]).name.lower()]
        if ru_in_day:
            ru_files_in_regular[date_str] = ru_in_day
            found_any = True

    if prefixes_found:
        found_any = True
        print(f"  RTMA-RU top-level prefixes found: {prefixes_found}")
        # List files in one date as a sample
        for date_str in probe_dates[:1]:
            for pfx in prefixes_found[:2]:
                objs = _list_objects(s3, RTMA_BUCKET, pfx + date_str + "/")
                if objs:
                    file_counts[pfx + date_str] = len(objs)
                    print(f"  {pfx}{date_str}/: {len(objs)} files")
                    for o in objs[:3]:
                        print(f"    {Path(o['Key']).name}  {o.get('Size', 0)/1e6:.1f} MB")
    else:
        print(f"  No RTMA-RU top-level prefixes found in {RTMA_BUCKET}")

    if ru_files_in_regular:
        print(f"  RTMA-RU files inside regular RTMA dirs: {dict(list(ru_files_in_regular.items())[:3])}")

    if not found_any:
        print("  RTMA-RU: no files found for target dates")

    return {
        "checked":                  True,
        "rtmaru_top_prefixes":      prefixes_found,
        "rtmaru_in_regular_prefix": {k: v[:5] for k, v in ru_files_in_regular.items()},
        "found_any":                found_any,
        "file_counts_sample":       file_counts,
        "note": (
            "RTMA-RU fallback only. Extraction not implemented in this run. "
            "If needed, use noaa-rtma-pds with rtma2p5_ru. prefix pattern."
        ),
    }


# -- Main -----------------------------------------------------------------------

def main() -> int:
    _check_dt_flexible_sanity()   # fail-fast if datetime parsing is broken

    # Ensure stdout can emit the Unicode characters used in progress messages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t_wall = time.time()
    DISC_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RTMA/URMA Family Precipitation Discovery — Flash-NH Stage 1 Diagnostic")
    print("=" * 70)
    print(f"Output: {DISC_DIR}")
    print(f"Diagnostic-only. Model inputs UNCHANGED.")
    print()

    report: dict[str, Any] = {
        "script":              "discover_rtma_urma_precip_january2023.py",
        "run_utc":             datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "candidates_probed":   list(CANDIDATE_WINDOWS.keys()),
        "diagnostic_only":     True,
        "modifies_model_inputs": False,
        "weight_table":        str(RTMA_WEIGHTS_PQ),
    }

    # -- Load weight table ------------------------------------------------------
    print("Loading RTMA weight table …")
    if not RTMA_WEIGHTS_PQ.exists():
        print(f"  ERROR: {RTMA_WEIGHTS_PQ}")
        return 1
    weights_df = pd.read_parquet(RTMA_WEIGHTS_PQ)
    weights_df["STAID"] = weights_df["STAID"].astype(str).str.zfill(8)
    print(f"  {len(weights_df)} rows, {weights_df['STAID'].nunique()} basins  "
          f"row [{weights_df['row_idx'].min()}, {weights_df['row_idx'].max()}]  "
          f"col [{weights_df['col_idx'].min()}, {weights_df['col_idx'].max()}]")
    report["weight_table_info"] = {
        "path":     str(RTMA_WEIGHTS_PQ),
        "n_rows":   len(weights_df),
        "n_basins": int(weights_df["STAID"].nunique()),
        "row_min":  int(weights_df["row_idx"].min()),
        "row_max":  int(weights_df["row_idx"].max()),
        "col_min":  int(weights_df["col_idx"].min()),
        "col_max":  int(weights_df["col_idx"].max()),
    }

    # -- Product 1: Regular RTMA ------------------------------------------------
    regular_grid_holder: list = []   # filled inside _inspect_local_rtma
    rtma_result = _inspect_local_rtma(regular_grid_holder)
    report["product_regular_rtma"] = {
        k: v for k, v in rtma_result.items() if k not in ("all_messages",)
    }

    regular_grid: Optional[dict] = regular_grid_holder[0] if regular_grid_holder else None

    all_inventory_rows: list[dict] = []

    # -- Product 2: URMA --------------------------------------------------------
    s3 = _s3_client()
    urma_result = _inspect_urma(s3, regular_grid, weights_df)
    report["product_urma"] = {
        k: v for k, v in urma_result.items() if k != "all_messages"
    }
    all_inventory_rows.extend(urma_result.get("all_messages", []))

    # -- Product 3: RTMA-RU (fallback check) -----------------------------------
    rtmaru_result = _check_rtmaru_fallback(s3)
    report["product_rtmaru_fallback"] = rtmaru_result

    # -- Overall verdict --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DISCOVERY SUMMARY")
    print("=" * 70)

    verdict:          str = "no_precipitation_found"
    best_product:     Optional[str] = None
    best_precip:      Optional[dict] = None
    weights_reusable: bool = False
    recommendation:   str = ""

    # Priority 1: regular RTMA with precip (most direct)
    if rtma_result["precip_found"] and rtma_result.get("grid_meta"):
        compat = _check_compat(
            {**rtma_result["grid_meta"],  # no _lat_2d/_lon_2d here — checked separately
             "_lat_2d": regular_grid.get("_lat_2d") if regular_grid else None,
             "_lon_2d": regular_grid.get("_lon_2d") if regular_grid else None,
             },
            regular_grid, weights_df
        )
        if compat["weight_table_compatible"]:
            verdict         = "pass_regular_rtma"
            best_product    = "rtma_conus_aws_2p5km"
            best_precip     = rtma_result["precip_fields"][0]
            weights_reusable = True
            recommendation  = (
                "Regular RTMA contains a precipitation field and the existing "
                "RTMA weight table is compatible. This is the most direct diagnostic path. "
                "Proceed to Part 2 extraction pilot using regular RTMA precip."
            )
        else:
            verdict       = "stop_rtma_precip_grid_incompatible"
            recommendation = (
                "Regular RTMA contains precipitation but grid is NOT compatible with "
                "existing RTMA weight table. A new weight table would be required."
            )

    # Priority 2: URMA with precip
    elif urma_result.get("precip_found"):
        if urma_result.get("weights_compatible"):
            verdict         = "pass_urma"
            best_product    = "urma_conus_aws_2p5km"
            best_precip     = urma_result["precip_fields"][0]
            weights_reusable = True
            recommendation  = (
                "URMA QPE contains a precipitation field and the existing RTMA weight "
                "table is compatible (same grid family). Proceed to Part 2 extraction "
                "pilot using URMA QPE for R02, R06, R11."
            )
        elif urma_result.get("grid_compat") is not None:
            verdict       = "stop_urma_precip_grid_incompatible"
            recommendation = (
                "URMA contains precipitation but grid is NOT compatible with existing "
                "RTMA weight table. New weights would be required — outside approved scope."
            )
        else:
            verdict       = "stop_urma_grid_not_checked"
            recommendation = "URMA has precip but grid could not be extracted for comparison."

    # Priority 2b: URMA accessible but no precip in sampled files
    elif urma_result.get("bucket_accessible"):
        n_qpe = sum(
            1 for date_files in urma_result.get("file_inventory", {}).values()
            for f in date_files
            if "pcp" in f.get("filename", "").lower() or "qpe" in f.get("filename", "").lower()
        )
        if n_qpe > 0:
            verdict       = "stop_urma_qpe_files_exist_but_no_precip_decoded"
            recommendation = (
                f"URMA bucket accessible, {n_qpe} QPE-named files found for target dates, "
                "but no precipitation GRIB field decoded from the sampled files. "
                "Try downloading a QPE file directly (not the analysis file) for GRIB inspection."
            )
        else:
            verdict       = "stop_urma_no_qpe_files"
            recommendation = (
                "URMA bucket accessible but no QPE-named files found for target dates. "
                "Possible reasons: QPE product not archived for Jan 2023, or filename "
                "pattern different from 'pcp'. Check URMA documentation."
            )

    # Priority 3: RTMA-RU found
    elif rtmaru_result.get("found_any"):
        verdict       = "stop_rtmaru_available_but_not_implemented"
        recommendation = (
            "Regular RTMA and URMA QPE have no suitable precipitation. "
            "RTMA-RU files were found in the S3 bucket. "
            "Implement RTMA-RU extraction only after explicit approval."
        )

    # Nothing found
    else:
        verdict       = "stop_no_precipitation_in_any_product"
        recommendation = (
            "No precipitation field found in regular RTMA, URMA QPE, or RTMA-RU "
            "for the target candidate dates. Consider alternative precipitation sources "
            "(Stage IV, CPC, IMERG) or accept that this diagnostic is not feasible "
            "with the current product family."
        )

    # Print summary
    print(f"\n  Regular RTMA has precip  : {rtma_result['precip_found']}")
    print(f"  URMA accessible          : {urma_result.get('bucket_accessible', False)}")
    print(f"  URMA has precip          : {urma_result.get('precip_found', False)}")
    print(f"  URMA weights compatible  : {urma_result.get('weights_compatible', False)}")
    print(f"  RTMA-RU found any files  : {rtmaru_result.get('found_any', False)}")
    print(f"\n  VERDICT  : {verdict}")
    print(f"  REUSABLE : existing RTMA weights reusable = {weights_reusable}")
    print(f"\n  RECOMMENDATION:")
    for line in recommendation.split(". "):
        if line:
            print(f"    {line.strip()}.")
    print()

    if verdict.startswith("stop_"):
        print(f"  STOP — do not proceed to extraction without addressing the above.")
    else:
        print(f"  PASS — safe to proceed to Part 2 extraction pilot.")
        if best_product and best_precip:
            print(f"  Best product : {best_product}")
            print(f"  Best field   : {best_precip.get('short_name')!r}  "
                  f"({best_precip.get('grib_name')})")
            print(f"  Accum min    : {best_precip.get('inferred_accum_minutes')}")
            print(f"  Next step    : implement Part 2 extraction pilot for R02/R06/R11")
    print("=" * 70)

    report["verdict"]           = verdict
    report["best_product"]      = best_product
    report["best_precip_field"] = best_precip
    report["weights_reusable"]  = weights_reusable
    report["recommendation"]    = recommendation
    report["overall_pass"]      = verdict.startswith("pass_")

    # -- Write outputs ----------------------------------------------------------
    report["runtime_s"] = round(time.time() - t_wall, 1)

    json_path = DISC_DIR / "rtma_urma_precip_discovery_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nJSON  -> {json_path}")

    if all_inventory_rows:
        csv_path = DISC_DIR / "rtma_urma_precip_inventory_sample.csv"
        pd.DataFrame(all_inventory_rows).to_csv(csv_path, index=False)
        print(f"CSV   -> {csv_path}")

    _write_md(report, DISC_DIR / "rtma_urma_precip_discovery_report.md")
    print(f"MD    -> {DISC_DIR / 'rtma_urma_precip_discovery_report.md'}")

    return 0 if report["overall_pass"] else 1


# -- Markdown report ------------------------------------------------------------

def _write_md(r: dict, path: Path) -> None:
    verdict = r.get("verdict", "?")
    overall = "**PASS**" if r.get("overall_pass") else f"**STOP — {verdict}**"
    lines = [
        "# RTMA/URMA Family Precipitation Discovery Report",
        "",
        f"**Run UTC**: {r.get('run_utc')}  ",
        f"**Candidates probed**: {', '.join(r.get('candidates_probed', []))}  ",
        f"**Overall result**: {overall}  ",
        f"**Diagnostic only** (model inputs unchanged): {r.get('diagnostic_only')}",
        "",
        "---",
        "",
        "## RTMA Weight Table (reused, not modified)",
        "",
    ]
    wt = r.get("weight_table_info", {})
    lines += [
        f"- Path: `{wt.get('path', '?')}`",
        f"- Rows: {wt.get('n_rows')}  |  Basins: {wt.get('n_basins')}",
        f"- row_idx: [{wt.get('row_min')}, {wt.get('row_max')}]",
        f"- col_idx: [{wt.get('col_min')}, {wt.get('col_max')}]",
        "",
        "---",
        "",
        "## Product 1 — Regular RTMA (Stage 1 product, locally cached)",
        "",
    ]
    rtma = r.get("product_regular_rtma", {})
    lines += [
        f"- File pattern: `{rtma.get('file_pattern', '?')}`",
        f"- Files checked: {len(rtma.get('files_checked', []))}",
        f"- Variables confirmed: {rtma.get('all_variables', [])}",
        f"- **Precipitation found: {rtma.get('precip_found', False)}**",
    ]
    if rtma.get("precip_fields"):
        lines.append("")
        lines.append("### Precipitation fields in regular RTMA")
        lines += ["| short_name | grib_name | units | step | accum_min | data_max |",
                  "|---|---|---|---|---|---|"]
        for pf in rtma["precip_fields"]:
            lines.append(
                f"| `{pf.get('short_name')}` | {pf.get('grib_name')} | "
                f"{pf.get('grib_units')} | {pf.get('step_range')}/{pf.get('step_type')} | "
                f"{pf.get('inferred_accum_minutes')} | {pf.get('data_max')} |"
            )
    lines += ["", "---", "", "## Product 2 — URMA QPE (noaa-urma-pds)", ""]
    urma = r.get("product_urma", {})
    lines += [
        f"- Bucket: `{urma.get('bucket', '?')}`",
        f"- Accessible: **{urma.get('bucket_accessible', False)}**",
        f"- Files found for target dates: {urma.get('urma_found_any_files', False)}",
        f"- **Precipitation found: {urma.get('precip_found', False)}**",
        f"- **Existing RTMA weights compatible: {urma.get('weights_compatible', False)}**",
    ]
    if urma.get("precip_fields"):
        lines.append("")
        lines.append("### Precipitation fields in URMA")
        lines += ["| short_name | grib_name | units | step | accum_min | data_max |",
                  "|---|---|---|---|---|---|"]
        for pf in urma["precip_fields"]:
            lines.append(
                f"| `{pf.get('short_name')}` | {pf.get('grib_name')} | "
                f"{pf.get('grib_units')} | {pf.get('step_range')}/{pf.get('step_type')} | "
                f"{pf.get('inferred_accum_minutes')} | {pf.get('data_max')} |"
            )
    gc = urma.get("grid_compat")
    if gc:
        lines += [
            "",
            "### URMA Grid Compatibility",
            "",
            f"| Check | Result |",
            f"|---|---|",
            f"| URMA grid shape | {gc.get('candidate_grid_shape')} |",
            f"| Regular RTMA shape | {gc.get('regular_grid_shape')} |",
            f"| Shapes match | **{gc.get('shapes_match')}** |",
            f"| Weight cells OOB | {gc.get('n_weight_cells_oob')} / {gc.get('n_weight_cells_total')} |",
        ]
        lc = gc.get("latlon_spot_check")
        if lc:
            lines.append(
                f"| Grids coincident (<0.001°) | **{lc.get('grids_coincident')}** "
                f"(max Δlat={lc.get('max_lat_diff_deg')} °, "
                f"max Δlon={lc.get('max_lon_diff_deg')} °) |"
            )
        lines.append(f"| **Weight table compatible** | **{gc.get('weight_table_compatible')}** |")
    lines += [
        "",
        "---",
        "",
        "## Product 3 — RTMA-RU (fallback, not implemented)",
        "",
    ]
    ru = r.get("product_rtmaru_fallback", {})
    lines += [
        f"- Checked: {ru.get('checked', False)}",
        f"- Top-level prefixes found: {ru.get('rtmaru_top_prefixes', [])}",
        f"- Found any files: {ru.get('found_any', False)}",
        f"- Note: {ru.get('note', 'N/A')}",
        "",
        "---",
        "",
        "## Verdict and Recommendation",
        "",
        f"**Verdict**: `{verdict}`  ",
        f"**Existing RTMA weights reusable**: {r.get('weights_reusable', False)}  ",
        f"**Overall pass**: {r.get('overall_pass', False)}",
        "",
        r.get("recommendation", ""),
        "",
        "---",
        f"*Runtime: {r.get('runtime_s', '?')} s  |  "
        f"Script: `scripts/discover_rtma_urma_precip_january2023.py`*",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())