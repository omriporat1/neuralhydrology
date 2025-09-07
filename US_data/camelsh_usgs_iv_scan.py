#!/usr/bin/env python3
"""
CAMELSH × USGS IV (15-min) availability scanner — VS Code friendly

Run by simply pressing "Run" in VS Code. All settings live in CONFIG below.
- Reads a CAMELSH-like attributes CSV with USGS gauge IDs and (optionally) drainage area.
- Queries USGS NWIS 'site' (seriesCatalogOutput=true) for discharge (00060) IV period-of-record.
- (Optional) Samples a short IV window to estimate native timestep (~15 min).
- Optional drainage-area filter.
- Writes a CSV summary. Supports resume & periodic checkpoints.

Deps: pandas, requests, tqdm
    pip install pandas requests tqdm
"""

from __future__ import annotations
import argparse
import datetime as dt
import io
import math
import os
import time
import random  # NEW
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# =========================
# CONFIG — edit these
# =========================
CONFIG = {
    # Paths
    "attrs_csv": r"C:\PhD\Python\neuralhydrology\US_data\attributes\attributes_gageii_BasinID.csv",
    "out_csv":   r"C:\PhD\Python\neuralhydrology\US_data\iv_scan_results.csv",
    # Column names (auto-detected if None)
    "id_field": "STAID",
    "area_field": None,
    # Optional drainage-area filter (km^2). Use None to skip.
    "min_area": None,
    "max_area": None,
    # API and runtime behavior
    "batch_size": 200,  # temporarily lower while validating
    "sleep_sec": 0.1,
    "timeout": 30,
    "retries": 5,
    "checkpoint_every": 5,
    "resume": True,
    # Optional: estimate native timestep by sampling recent IV
    "check_interval": True,
    "lookback_days": 7,
    # --- debug probe options ---
    "debug_probe": True,          # set True to print a small random sample
    "debug_probe_only": False,     # set True to exit after the probe
    "debug_n": 10,                # how many sites to sample
    "debug_seed": 42,             # RNG seed for reproducibility
}

# Advanced: keep CLI overrides available (optional)
ALLOW_CLI = True

# =========================
# Implementation
# =========================

USGS_SITE_ENDPOINT = "https://waterservices.usgs.gov/nwis/site/"
USGS_IV_ENDPOINT   = "https://waterservices.usgs.gov/nwis/iv/"

COMMON_ID_FIELDS   = ["STAID", "GAGE_ID", "USGS_ID", "site_no", "site_no_txt"]
COMMON_AREA_FIELDS = ["DRAIN_SQKM", "DRAINAGE_SQKM", "DRAIN_SQ_KM", "AREA_SQKM", "DA_SQKM", "drain_sqkm"]


def detect_column(cols: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def chunked(seq: List[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def make_session(total_retries: int = 5, backoff: float = 0.3, timeout: int = 30) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": "camelsh-iv-scan/0.1 (+https://usgs.gov)"})
    try:
        retry = Retry(
            total=total_retries, read=total_retries, connect=total_retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
    except TypeError:
        # Fallback for older urllib3
        retry = Retry(
            total=total_retries, read=total_retries, connect=total_retries,
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


def fetch_series_catalog(sess: requests.Session, site_ids: List[str]) -> dict | str:
    """Try JSON; on 400 fallback to RDB text."""
    base_params = {
        "sites": ",".join(site_ids),
        "seriesCatalogOutput": "true",
        "siteStatus": "all",
    }
    # First try JSON (minimal params)
    params_json = {**base_params, "format": "json"}
    r = sess.get(USGS_SITE_ENDPOINT, params=params_json, timeout=getattr(sess, "request_timeout", 30))
    if r.status_code < 400:
        return r.json()
    # Fallback to RDB
    params_rdb = {**base_params, "format": "rdb"}
    r2 = sess.get(USGS_SITE_ENDPOINT, params=params_rdb, timeout=getattr(sess, "request_timeout", 30))
    if r2.status_code >= 400:
        snippet = r2.text.replace("\n", " ")[:500]
        raise requests.HTTPError(f"{r2.status_code} {r2.reason} — {snippet}")
    return r2.text


def parse_series_catalog(catalog) -> dict:
    """Return mapping site_id -> list of (begin_iso, end_iso). Supports JSON or RDB text."""
    # JSON path (existing behavior)
    if isinstance(catalog, dict):
        out = {}
        series = catalog.get("value", {}).get("timeSeries", [])
        for ts in series:
            try:
                site = ts["sourceInfo"]["siteCode"][0]["value"]
                var  = ts["variable"]["variableCode"][0]["value"]
                if var != "00060":  # discharge
                    continue
                for vs in ts.get("values", []):
                    b = vs.get("beginDateTime")
                    e = vs.get("endDateTime")
                    if b or e:
                        out.setdefault(site, []).append((b, e))
            except Exception:
                continue
        return out

    # RDB path (fallback)
    txt = str(catalog)
    if not txt.strip():
        return {}
    df = pd.read_csv(io.StringIO(txt), sep="\t", comment="#", dtype=str)
    if df.empty:
        return {}
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"site_no", "parm_cd", "data_type_cd", "begin_date", "end_date"}
    if not needed.issubset(set(df.columns)):
        return {}

    # Accept both 'uv' (unit values / instantaneous) and 'iv' just in case
    df = df[(df["parm_cd"] == "00060") & (df["data_type_cd"].str.lower().isin(["uv", "iv"]))]

    out = {}
    for _, row in df.iterrows():
        site = str(row["site_no"]).zfill(8)
        b = str(row.get("begin_date") or "") or None
        e = str(row.get("end_date") or "") or None
        out.setdefault(site, []).append((b, e))
    return out


def iso_parse(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def summarize_ranges(ranges: List[tuple[Optional[str], Optional[str]]]) -> tuple[Optional[str], Optional[str]]:
    begins = [iso_parse(b) for b, _ in ranges if b]
    ends   = [iso_parse(e) for _, e in ranges if e]
    if not begins or not ends:
        return None, None
    return min(begins).isoformat(), max(ends).isoformat()


def fetch_iv_sample(
    sess: requests.Session,
    site_id: str,
    lookback_days: int = 7,
    parameter_cd: str = "00060",
    anchor_iso: Optional[str] = None,   # new: anchor around last known IV end if recent window is empty
):
    """Fetch IV timestamps. Try recent period; if empty and anchor_iso given, sample around anchor."""
    # 1) Prefer minimal query using period=P{n}D near "now"
    params1 = {
        "sites": site_id,
        "parameterCd": parameter_cd,
        "period": f"P{int(lookback_days)}D",
        "format": "json",
    }
    r = sess.get(USGS_IV_ENDPOINT, params=params1, timeout=getattr(sess, "request_timeout", 30))
    series = []
    if r.status_code < 400:
        try:
            data = r.json()
            series = data.get("value", {}).get("timeSeries", [])
        except Exception:
            series = []

    # 2) Fallback to explicit start/end around anchor (iv_end) if recent is empty
    if not series and anchor_iso:
        end = iso_parse(anchor_iso)
        if end is None:
            end = dt.datetime.utcnow()
        # ensure UTC, ISO with Z
        if end.tzinfo is None:
            end = end.replace(tzinfo=dt.timezone.utc)
        else:
            end = end.astimezone(dt.timezone.utc)
        start = end - dt.timedelta(days=lookback_days)
        params2 = {
            "sites": site_id,
            "parameterCd": parameter_cd,
            "startDT": start.isoformat().replace("+00:00", "Z"),
            "endDT":   end.isoformat().replace("+00:00", "Z"),
            "format": "json",
        }
        r2 = sess.get(USGS_IV_ENDPOINT, params=params2, timeout=getattr(sess, "request_timeout", 30))
        if r2.status_code < 400:
            try:
                data = r2.json()
                series = data.get("value", {}).get("timeSeries", [])
            except Exception:
                series = []
        else:
            series = []

    # 3) Extract timestamps
    pts = []
    for s in series or []:
        for v in s.get("values", []):
            for val in v.get("value", []):
                t = iso_parse(val.get("dateTime"))
                if t:
                    pts.append(t)
    pts.sort()
    if not pts or len(pts) < 3:
        return None
    return pts


def estimate_timestep_minutes(timestamps: List[dt.datetime]) -> Optional[float]:
    if not timestamps or len(timestamps) < 3:
        return None
    diffs = [(b - a).total_seconds()/60.0 for a, b in zip(timestamps[:-1], timestamps[1:])]
    diffs = [d for d in diffs if 0.5 <= d <= 180.0]
    if not diffs:
        return None
    diffs.sort()
    mid = len(diffs)//2
    median = diffs[mid] if len(diffs) % 2 == 1 else 0.5*(diffs[mid-1]+diffs[mid])
    return round(median, 2)


# NEW: small diagnostic probe
def debug_probe(sess: requests.Session, sites: List[str], df_area: pd.DataFrame, cfg: dict) -> None:
    n = min(int(cfg.get("debug_n", 10)), len(sites))
    rng = random.Random(cfg.get("debug_seed"))
    sample_sites = rng.sample(sites, n) if n > 0 else []
    print(f"[debug] probing {len(sample_sites)} random sites (lookback=P{cfg['lookback_days']}D)")

    # One request for series catalog
    cat = fetch_series_catalog(sess, sample_sites)
    parsed = parse_series_catalog(cat)

    # Build quick area lookup
    area_map = dict(zip(df_area["site_id"], df_area["drainage_km2"]))

    for site in sample_sites:
        ranges = parsed.get(site, [])
        has_iv = len(ranges) > 0
        start_iso, end_iso = summarize_ranges(ranges) if has_iv else (None, None)

        ts_recent = fetch_iv_sample(sess, site, lookback_days=cfg["lookback_days"])
        med_recent = estimate_timestep_minutes(ts_recent) if ts_recent else None

        ts_anchor = None
        med_anchor = None
        if (not ts_recent) and end_iso:
            ts_anchor = fetch_iv_sample(sess, site, lookback_days=cfg["lookback_days"], anchor_iso=end_iso)
            med_anchor = estimate_timestep_minutes(ts_anchor) if ts_anchor else None

        area_val = area_map.get(site)
        print(f"- site={site} area_km2={area_val} has_iv={has_iv} POR={start_iso} -> {end_iso}")
        print(f"  recent:  pts={0 if ts_recent is None else len(ts_recent)}  median_dt_min={med_recent}")
        print(f"  anchor:  pts={0 if ts_anchor is None else len(ts_anchor)}  median_dt_min={med_anchor}")
        # Optional: show first 3 timestamps to eyeball cadence
        def head3(ts):
            return [t.isoformat() for t in (ts[:3] if ts else [])]
        print(f"  head(recent)={head3(ts_recent)}  head(anchor)={head3(ts_anchor)}")


def run_scan(cfg: dict):
    # Validate input path early
    if not os.path.exists(cfg["attrs_csv"]):
        raise SystemExit(f"Attributes CSV not found. Set CONFIG['attrs_csv'] correctly: {cfg['attrs_csv']}")

    # Load attributes
    df = pd.read_csv(cfg["attrs_csv"])
    id_field   = cfg["id_field"]   or detect_column(df.columns, COMMON_ID_FIELDS)
    area_field = cfg["area_field"] or detect_column(df.columns, COMMON_AREA_FIELDS)
    if not id_field:
        raise SystemExit(f"Could not detect USGS ID column. Try setting CONFIG['id_field']. "
                         f"Looked for: {COMMON_ID_FIELDS}")

    sites_all = df[id_field].astype(str).str.zfill(8).tolist()
    if area_field:
        df_area = df[[id_field, area_field]].copy()
        df_area.columns = ["site_id", "drainage_km2"]
        df_area["site_id"] = df_area["site_id"].astype(str).str.zfill(8)
    else:
        df_area = pd.DataFrame({"site_id": df[id_field].astype(str).str.zfill(8), "drainage_km2": math.nan})

    # Optional area filter
    sites = sites_all
    if cfg["min_area"] is not None or cfg["max_area"] is not None:
        if df_area["drainage_km2"].notna().any():
            mask = pd.Series(True, index=df_area.index)
            if cfg["min_area"] is not None:
                mask &= df_area["drainage_km2"] >= float(cfg["min_area"])
            if cfg["max_area"] is not None:
                mask &= df_area["drainage_km2"] <= float(cfg["max_area"])
            keep = set(df_area.loc[mask, "site_id"].tolist())
            sites = [s for s in sites_all if s in keep]
            print(f"[area filter] kept {len(sites)} of {len(sites_all)} sites")
        else:
            print("[WARN] Area filter requested but area column not found; skipping filter.")

    # Resume (kept even in debug so we can still inspect written CSV if desired)
    processed = set()
    results = []
    if cfg["resume"] and os.path.exists(cfg["out_csv"]):
        prev = pd.read_csv(cfg["out_csv"])
        if "site_id" in prev.columns:
            processed = set(prev["site_id"].astype(str).str.zfill(8))
            results = prev.to_dict(orient="records")
            print(f"[resume] loaded {len(processed)} previously scanned sites from {cfg['out_csv']}")

    # HTTP session
    sess = make_session(total_retries=cfg["retries"], backoff=cfg["sleep_sec"], timeout=cfg["timeout"])

    # DEBUG PROBE
    if cfg.get("debug_probe") or cfg.get("debug_probe_only"):
        debug_probe(sess, sites, df_area, cfg)
        if cfg.get("debug_probe_only"):
            return

    # Scan
    batches = list(chunked([s for s in sites if s not in processed], cfg["batch_size"]))
    checkpoint_counter = 0
    for batch in tqdm(batches, desc="Query NWIS series catalog"):
        try:
            cat = fetch_series_catalog(sess, batch)
        except Exception as e:
            print(f"[ERROR] batch failed, retrying once: {e}")
            time.sleep(2 * cfg["sleep_sec"])
            try:
                cat = fetch_series_catalog(sess, batch)
            except Exception as e2:
                print(f"[ERROR] batch permanently failed, marking missing: {e2}")
                for site in batch:
                    results.append({"site_id": site, "has_iv": False, "iv_start": None, "iv_end": None})
                continue

        parsed = parse_series_catalog(cat)
        for site in batch:
            ranges = parsed.get(site, [])
            has_iv = len(ranges) > 0
            if has_iv:
                start_iso, end_iso = summarize_ranges(ranges)
            else:
                start_iso, end_iso = None, None
            results.append({"site_id": site, "has_iv": has_iv, "iv_start": start_iso, "iv_end": end_iso})

        time.sleep(cfg["sleep_sec"])
        checkpoint_counter += 1

        if cfg["checkpoint_every"] and (checkpoint_counter % cfg["checkpoint_every"] == 0):
            out_df = pd.DataFrame(results).merge(df_area, on="site_id", how="left")
            out_df.to_csv(cfg["out_csv"], index=False)
            print(f"[checkpoint] wrote {len(out_df)} rows to {cfg['out_csv']}")

    # Optional: estimate timestep
    out_df = pd.DataFrame(results).merge(df_area, on="site_id", how="left")
    if cfg["check_interval"]:
        est_rows = []
        iv_sites = out_df.loc[out_df["has_iv"], "site_id"].tolist()
        # make a quick lookup of iv_end to anchor fallback sampling
        iv_end_map = dict(zip(out_df["site_id"], out_df["iv_end"]))
        for site in tqdm(iv_sites, desc="Estimate timestep"):
            anchor = iv_end_map.get(site)
            try:
                ts = fetch_iv_sample(
                    sess, site,
                    lookback_days=cfg["lookback_days"],
                    anchor_iso=anchor
                )
                median_dt = estimate_timestep_minutes(ts) if ts else None
            except Exception:
                median_dt = None
            est_rows.append((site, median_dt))
            time.sleep(cfg["sleep_sec"])
        est_df = pd.DataFrame(est_rows, columns=["site_id", "median_dt_min"])

        # IMPORTANT: avoid suffix collisions when resuming
        if "median_dt_min" in out_df.columns:
            out_df.drop(columns=["median_dt_min"], inplace=True)

        out_df = out_df.merge(est_df, on="site_id", how="left")
    else:
        out_df["median_dt_min"] = None

    # Flag likely 15-min cadence
    out_df["likely_15min"] = out_df["median_dt_min"].apply(
        lambda x: (x is not None) and (10.0 <= float(x) <= 20.0) if pd.notna(x) else False
    )

    # Order and save
    cols = ["site_id", "has_iv", "iv_start", "iv_end", "median_dt_min", "likely_15min", "drainage_km2"]
    for c in cols:
        if c not in out_df.columns:
            out_df[c] = None
    out_df = out_df[cols]

    out_df.to_csv(cfg["out_csv"], index=False)
    print(f"[done] wrote {len(out_df)} rows to {cfg['out_csv']}")


def parse_cli() -> dict:
    """Optional CLI overrides; returns a dict of updates for CONFIG."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--attrs")
    p.add_argument("--out")
    p.add_argument("--id-field")
    p.add_argument("--area-field")
    p.add_argument("--min-area", type=float)
    p.add_argument("--max-area", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--sleep", dest="sleep_sec", type=float)
    p.add_argument("--timeout", type=int)
    p.add_argument("--retries", type=int)
    p.add_argument("--checkpoint-every", type=int)

    # IMPORTANT: booleans default to None so they don't override CONFIG unless passed
    p.add_argument("--resume", dest="resume", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.set_defaults(resume=None)

    p.add_argument("--check-interval", dest="check_interval", action="store_true")
    p.add_argument("--no-check-interval", dest="check_interval", action="store_false")
    p.set_defaults(check_interval=None)

    p.add_argument("--lookback-days", type=int)

    args, _ = p.parse_known_args()
    updates = {k: v for k, v in vars(args).items() if v is not None}
    # Map CLI names to config keys
    if "attrs" in updates:
        updates["attrs_csv"] = updates.pop("attrs")
    if "out" in updates:
        updates["out_csv"] = updates.pop("out")
    return updates


if __name__ == "__main__":
    # Merge optional CLI overrides so you can still run from a terminal if desired
    if ALLOW_CLI:
        CONFIG.update(parse_cli())
    # Run with the final config (press "Run" in VS Code to debug)
    print("Using configuration:\n", {k: v for k, v in CONFIG.items() if k not in {"id_field","area_field"}})
    run_scan(CONFIG)
