#!/usr/bin/env python3
r"""
USGS IV hydrograph downloader (native IV cadence) for a set of site_ids.

- Input sites: C:\PhD\Python\neuralhydrology\US_data\basin_attribute_values_filtered.csv (column: site_id)
- Variable: Discharge, instantaneous, cubic feet per second (USGS parameter 00060), converted to cubic meters per second.
- Period: set START_ISO and END_ISO below (UTC ISO8601 recommended, e.g., "2000-01-01T00:00:00Z").
- Output: one CSV per site with datetime_utc, discharge_cms, qualifiers.
"""

from __future__ import annotations
import argparse
import datetime as dt
import io
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# -----------------------
# Settings (edit here)
# -----------------------
INPUT_SITES_CSV = r"C:\PhD\Python\neuralhydrology\US_data\basin_attribute_values_filtered.csv"
SITE_ID_COLUMN  = "site_id"
FIRST_N = 10  # process only the first N site_ids (0 = all)

# Period (UTC). Use "YYYY-MM-DD" or full ISO with time. CLI can override.
START_ISO = "2024-01-01T00:00:00Z"
END_ISO   = "2024-01-02T23:59:59Z"

# Output directory (CLI can override)
OUT_DIR = r"C:\PhD\Python\neuralhydrology\US_data\iv_downloads_sample"

# Request chunking (month-by-month reduces server load and avoids large payloads)
CHUNK_BY_MONTH = True
SLEEP_SEC = 0.1

# -----------------------
USGS_IV_ENDPOINT   = "https://waterservices.usgs.gov/nwis/iv/"
PARAM_CODE         = "00060"   # discharge, instantaneous, cubic feet per second (convert to m3/s)

USER_AGENT         = "camelsh-iv-download/0.1 (+https://usgs.gov)"
FT3S_TO_M3S        = 0.028316846592  # safety fallback if unit is unexpectedly ft^3/s

def make_session(retries: int = 5, backoff: float = 0.3, timeout: int = 30) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    try:
        retry = Retry(
            total=retries, read=retries, connect=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
    except TypeError:
        retry = Retry(
            total=retries, read=retries, connect=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            method_whitelist=frozenset(["GET"]),
            raise_on_status=False,
        )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    sess.request_timeout = timeout
    return sess

def parse_time_utc(s: str) -> pd.Timestamp:
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts

def iter_months(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while cur <= end_month:
        nxt = (cur + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        s = max(cur, start)
        e = min(nxt, end)
        yield s, e
        cur = (cur + pd.offsets.MonthBegin(2)).normalize() - pd.offsets.MonthBegin(1)

def fetch_iv_json(sess: requests.Session, site_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    params = {
        "sites": site_id,
        "parameterCd": PARAM_CODE,
        "startDT": start.isoformat().replace("+00:00", "Z"),
        "endDT":   end.isoformat().replace("+00:00", "Z"),
        "format": "json",
    }
    r = sess.get(USGS_IV_ENDPOINT, params=params, timeout=getattr(sess, "request_timeout", 30))
    r.raise_for_status()
    data = r.json()
    series = data.get("value", {}).get("timeSeries", [])
    rows = []
    for ts in series:
        unit_code = (((ts or {}).get("variable") or {}).get("unit") or {}).get("unitCode", "")
        u = str(unit_code).lower().strip()
        # Expect m3/s for 30209; convert only if API returns ft3/s unexpectedly
        if u in {"m3/s", "cms", "cubic meters per second"}:
            factor = 1.0
        elif u in {"ft3/s", "cfs", "ft^3/s", "cubic feet per second", ""}:
            factor = FT3S_TO_M3S
        else:
            factor = FT3S_TO_M3S
            tqdm.write(f"[warn] {site_id}: unexpected unitCode '{unit_code}', converting using ft3/s→m3/s factor")

        for vv in ts.get("values", []):
            for val in vv.get("value", []):
                dt_iso = val.get("dateTime")
                v_raw = pd.to_numeric(val.get("value"), errors="coerce")
                if pd.isna(v_raw) or dt_iso is None:
                    continue
                q = ",".join(val.get("qualifiers", [])) if isinstance(val.get("qualifiers"), list) else val.get("qualifiers")
                rows.append((dt_iso, float(v_raw) * factor, q))
    if not rows:
        return pd.DataFrame(columns=["datetime_utc", "discharge_cms", "qualifiers"])
    df = pd.DataFrame(rows, columns=["dateTime", "discharge_cms", "qualifiers"])
    df["datetime_utc"] = df["dateTime"].apply(parse_time_utc)
    df.drop(columns=["dateTime"], inplace=True)
    df = df.drop_duplicates(subset=["datetime_utc"], keep="last")
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    # Keep only requested window (defensive)
    df = df[(df["datetime_utc"] >= start) & (df["datetime_utc"] <= end)]
    return df[["datetime_utc", "discharge_cms", "qualifiers"]]

def snap_to_15min(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tol_minutes: float) -> pd.DataFrame:
    if df.empty:
        # return an empty 15-min grid (all NaN values)
        grid = pd.date_range(start=start, end=end, freq="15T", tz="UTC")
        return pd.DataFrame({"datetime_utc": grid, "discharge_cfs": pd.Series([pd.NA]*len(grid), dtype="float64"), "qualifiers": pd.NA})
    s_aligned = start.floor("15T")
    e_aligned = end.ceil("15T")
    grid = pd.date_range(start=s_aligned, end=e_aligned, freq="15T", tz="UTC")
    s = df.set_index("datetime_utc").sort_index()
    snapped = s.reindex(grid, method="nearest", tolerance=pd.Timedelta(minutes=tol_minutes))
    snapped = snapped.rename_axis("datetime_utc").reset_index()
    # return snapped[["datetime_utc", "discharge_cfs", "qualifiers"]]
    return snapped.rename(columns={"discharge_cfs": "discharge_cms"})[["datetime_utc","discharge_cms","qualifiers"]]

def load_site_ids(csv_path: str, id_col: str) -> List[str]:
    df = pd.read_csv(csv_path, dtype={id_col: str})
    if id_col not in df.columns:
        # try to auto-detect common names
        for cand in [id_col, "STAID", "site_no", "site_no_txt", "GAGE_ID", "USGS_ID"]:
            if cand in df.columns:
                id_col = cand
                break
    if id_col not in df.columns:
        raise SystemExit(f"Cannot find site id column in {csv_path}. Expected '{SITE_ID_COLUMN}'.")
    site_ids = df[id_col].astype(str).str.strip().str.zfill(8).unique().tolist()
    return site_ids

def save_site_csv(out_dir: str, site_id: str, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tag_start = start.strftime("%Y%m%d")
    tag_end   = end.strftime("%Y%m%d")
    out_path = Path(out_dir) / f"{site_id}_{PARAM_CODE}_iv_{tag_start}_{tag_end}_utc_cms.csv"
    out = df.copy()
    if "datetime_utc" not in out.columns:
        out["datetime_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
    else:
        out["datetime_utc"] = pd.to_datetime(out["datetime_utc"], utc=True, errors="coerce")
    if len(out) > 0:
        out["datetime_utc"] = out["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        out["datetime_utc"] = out["datetime_utc"].astype("string")
    out.insert(0, "site_id", site_id)
    out.to_csv(out_path, index=False)
    return str(out_path)

def run(
    sites_csv: str,
    out_dir: str,
    start_iso: str,
    end_iso: str,
    chunk_by_month: bool = True,
    first_n: int = FIRST_N,
):
    start = parse_time_utc(start_iso)
    end   = parse_time_utc(end_iso)
    if pd.isna(start) or pd.isna(end) or end < start:
        raise SystemExit("Invalid START/END period. Use ISO8601 (e.g., 2000-01-01 or 2000-01-01T00:00:00Z).")

    sites = load_site_ids(sites_csv, SITE_ID_COLUMN)
    if not sites:
        raise SystemExit("No site_ids found to download.")
    if first_n and first_n > 0:
        sites = sites[:first_n]
        print(f"[info] Limiting to the first {len(sites)} site(s).")

    sess = make_session()
    print(f"Downloading {PARAM_CODE} IV (cms) for {len(sites)} sites from {start} to {end} (UTC)")
    for site_id in tqdm(sites, desc="Sites"):
        try:
            parts: List[pd.DataFrame] = []
            if chunk_by_month:
                for s, e in iter_months(start, end):
                    df = fetch_iv_json(sess, site_id, s, e)
                    if not df.empty:
                        parts.append(df)
            else:
                df = fetch_iv_json(sess, site_id, start, end)
                if not df.empty:
                    parts.append(df)

            df_all = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["datetime_utc", "discharge_cms", "qualifiers"])
            if "datetime_utc" in df_all.columns:
                df_all["datetime_utc"] = pd.to_datetime(df_all["datetime_utc"], utc=True, errors="coerce")
                df_all = df_all.dropna(subset=["datetime_utc"])
            df_all = df_all.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"])

            out_file = save_site_csv(out_dir, site_id, df_all, start, end)
            tqdm.write(f"[saved] {site_id} -> {out_file}")
        except Exception as ex:
            tqdm.write(f"[ERROR] {site_id}: {ex}")

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--sites-csv", default=INPUT_SITES_CSV)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--start", default=START_ISO, help="ISO8601 (UTC). Example: 2000-01-01 or 2000-01-01T00:00:00Z")
    p.add_argument("--end",   default=END_ISO,   help="ISO8601 (UTC). Example: 2020-12-31 or 2020-12-31T23:59:59Z")
    p.add_argument("--no-month-chunks", dest="month_chunks", action="store_false")
    p.add_argument("--month-chunks", dest="month_chunks", action="store_true")
    p.set_defaults(month_chunks=CHUNK_BY_MONTH)
    p.add_argument("--first-n", type=int, default=FIRST_N,
                   help="Process only the first N site_ids (0 = all).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_cli()
    run(
        sites_csv=args.sites_csv,
        out_dir=args.out,
        start_iso=args.start,
        end_iso=args.end,
        chunk_by_month=args.month_chunks,
        first_n=args.first_n,
    )