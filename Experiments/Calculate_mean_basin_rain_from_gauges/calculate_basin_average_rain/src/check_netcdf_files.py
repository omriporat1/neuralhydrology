from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

try:
    import xarray as xr
except ImportError:  # xarray is required for netCDF reading
    xr = None  # type: ignore

# Add: optional tqdm progress bar with fallback
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # tqdm not installed or not available
    def tqdm(iterable, **kwargs):
        return iterable


# -----------------------------
# Config and helpers
# -----------------------------

IL_NC_DIR_DEFAULT = r"C:\PhD\Data\Caravan\timeseries\netcdf\il"
IL_CSV_DIR_DEFAULT = r"C:\PhD\Data\Caravan\timeseries\csv\il"
REPORT_DIR_DEFAULT = r"C:\PhD\Data\Caravan\timeseries\reports\il"
FREQ = "10min"
FLOAT_ATOL = 1e-6

# Add: default CSV datetime format (ISO) and a configurable global
CSV_TIME_FORMAT_DEFAULT = "%Y-%m-%d %H:%M:%S"
CSV_TIME_FORMAT: Optional[str] = CSV_TIME_FORMAT_DEFAULT

# canonical column names used for comparison/validation
COL_SYNONYMS = {
    # gauges summary
    "min_gauge": "min_gauges",
    "min_gaug": "min_gauges",
    "min_gauges": "min_gauges",
    "max_gauge": "max_gauges",
    "max_gaug": "max_gauges",
    "max_gauges": "max_gauges",
    # rain stats
    "mean_rain": "mean_rain",
    "avg_rain": "mean_rain",
    "max_rain": "max_rain",
    # core fields
    "date": "date",
    "time": "date",
    "datetime": "date",
    "timestamp": "date",
    "Station_ID": "Station_ID",
    "Flow_m3_sec": "Flow_m3_sec",
    "Water_level_m": "Water_level_m",
    "Flow_type": "Flow_type",
    "Data_type": "Data_type",
    "Record_type": "Record_type",
    # gauges per-station (allow case/underscore variations)
    "Station_ID_gauge_1": "Station_ID_gauge_1",
    "Station_ID_gauge_2": "Station_ID_gauge_2",
    "Station_ID_gauge_3": "Station_ID_gauge_3",
    "Rain_gauge_1": "Rain_gauge_1",
    "Rain_gauge_2": "Rain_gauge_2",
    "Rain_gauge_3": "Rain_gauge_3",
}

NUMERIC_FLOAT_COLS = [
    "Flow_m3_sec",
    "Water_level_m",
    "Rain_gauge_1",
    "Rain_gauge_2",
    "Rain_gauge_3",
    "mean_rain",
    "max_rain",
]
INT_COLS = [
    "Station_ID",
    "Station_ID_gauge_1",
    "Station_ID_gauge_2",
    "Station_ID_gauge_3",
    "min_gauges",
    "max_gauges",
]
STR_COLS = ["Flow_type", "Data_type", "Record_type"]

# columns used to check "good" timesteps
GOODNESS_CHECK = {
    "Flow_m3_sec": ("nonnegative", float),
    "mean_rain": ("nonnegative", float),
    "max_rain": ("nonnegative", float),
    "min_gauges": ("integer_nonneg", int),
    "max_gauges": ("integer_nonneg", int),
}


def ensure_report_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_basin_id_from_name(name: str) -> Optional[str]:
    """
    Extract basin number from filenames like 'il_2105.nc' or 'il_2105.csv'.
    """
    m = re.match(r"^il[_-](\d+)\.(?:nc|csv)$", name, flags=re.IGNORECASE)
    return m.group(1) if m else None


def to_dayfirst_datetime(s: Iterable, allow_na=True) -> pd.Series:
    return pd.to_datetime(pd.Series(s), dayfirst=True, errors="coerce" if allow_na else "raise")


# New: robust mixed-format parser that also flags non-dayfirst rows
def parse_datetime_mixed(s: Iterable, allow_na: bool = True) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (dt, nonstandard_mask). nonstandard_mask=True where parsing succeeded
    only with dayfirst=False (i.e., different time structure than expected).
    """
    ser = pd.Series(s, copy=False)
    if pd.api.types.is_datetime64_any_dtype(ser):
        dt = pd.to_datetime(ser, errors=("coerce" if allow_na else "raise"))
        nonstd = pd.Series(False, index=dt.index)
        return dt, nonstd

    ser_str = ser.astype("string")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dt1 = pd.to_datetime(ser_str, dayfirst=True, errors=("coerce" if allow_na else "raise"))

    nonstd = pd.Series(False, index=ser_str.index)
    missing = dt1.isna()
    if missing.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            dt2 = pd.to_datetime(ser_str[missing], dayfirst=False, errors="coerce")
        # mark those successfully parsed only in the second pass
        parsed_second = dt2.notna()
        nonstd.loc[missing] = parsed_second
        dt1.loc[missing & parsed_second] = dt2
    return dt1, nonstd


def rename_by_synonyms(cols: Iterable[str]) -> Dict[str, str]:
    out = {}
    for c in cols:
        if c in COL_SYNONYMS:
            out[c] = COL_SYNONYMS[c]
            continue
        # try case-insensitive match
        lc = c.lower()
        match = None
        for k in COL_SYNONYMS.keys():
            if lc == k.lower():
                match = COL_SYNONYMS[k]
                break
        out[c] = match if match else c
    return out


def _coerce_numeric(series: pd.Series, kind: str) -> pd.Series:
    if kind == "float":
        s = pd.to_numeric(series, errors="coerce")
        return s.astype("Float64")
    if kind == "int":
        s = pd.to_numeric(series, errors="coerce")
        # Keep as nullable Int64 to preserve NaN
        return s.astype("Int64")
    return series


def standardize_dataframe(df: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    # rename
    df = df.rename(columns=rename_by_synonyms(df.columns))
    invalid_date_rows: List[pd.Timestamp] = []

    # parse/clean dates
    if "date" not in df.columns:
        for candidate in ["time", "datetime", "timestamp"]:
            if candidate in df.columns:
                df["date"] = df[candidate]
                break
    if "date" in df.columns:
        if source == "csv" and CSV_TIME_FORMAT:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                dates = pd.to_datetime(df["date"].astype("string"), format=CSV_TIME_FORMAT, errors="coerce")
            # when enforcing a format, no per-row "nonstandard" — failures remain NaT
            nonstd = pd.Series(False, index=df.index)
        else:
            # fallback/mixed parsing for non-CSV sources or when auto-detecting
            dates, nonstd = parse_datetime_mixed(df["date"])
    else:
        dates, nonstd = pd.Series(pd.NaT, index=df.index), pd.Series(False, index=df.index)

    invalid_mask = dates.isna()
    if invalid_mask.any():
        invalid_date_rows = [pd.NaT] * int(invalid_mask.sum())

    keep = ~invalid_mask
    df = df.loc[keep].copy()
    df["date"] = dates.loc[keep]
    df["_nonstandard_time"] = nonstd.loc[keep].astype("boolean")

    # coerce numeric columns
    for c in NUMERIC_FLOAT_COLS:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c], "float")
    for c in INT_COLS:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c], "int")
    for c in STR_COLS:
        if c in df.columns:
            # keep as strings; normalize empty as NaN to simplify checks
            df[c] = df[c].astype("string")

    # drop exact dup row/timestamp pairs but keep duplicates info for gap report later
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    df = df.set_index("date").sort_index()
    return df, invalid_date_rows


def read_csv_file(path: Path) -> Tuple[pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    """
    Returns standardized df, list of invalid-date rows (count only), and duplicate timestamp list.
    """
    if not path.exists():
        return pd.DataFrame(), [], []
    # sep=None lets pandas sniff comma/tab/semicolon
    df_raw = pd.read_csv(path, sep=None, engine="python", dtype="string")
    # track duplicates before standardization cleans them
    dt, _ = parse_datetime_mixed(df_raw.get("date", pd.Series(index=df_raw.index, dtype="string")))
    dup_mask = dt.duplicated(keep=False) & ~dt.isna()
    dups = dt[dup_mask].tolist()
    df, invalid = standardize_dataframe(df_raw, "csv")
    return df, invalid, dups


def read_netcdf_file(path: Path) -> Tuple[pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    if not path.exists():
        return pd.DataFrame(), [], []
    if xr is None:
        raise RuntimeError("xarray is required to read netCDF files. Install with `pip install xarray netcdf4`.")
    ds = xr.open_dataset(path)
    try:
        df_raw = ds.to_dataframe().reset_index()
    finally:
        ds.close()
    # Heuristic: prefer a coord/col named 'date' or common time names
    time_col = None
    for c in ["date", "time", "datetime", "timestamp"]:
        if c in df_raw.columns:
            time_col = c
            break
    if time_col and time_col != "date":
        df_raw["date"] = df_raw[time_col]
    # track duplicates
    dt, _ = parse_datetime_mixed(df_raw.get("date", pd.Series(index=df_raw.index, dtype="string")))
    dup_mask = dt.duplicated(keep=False) & ~dt.isna()
    dups = dt[dup_mask].tolist()
    df, invalid = standardize_dataframe(df_raw, "netcdf")
    return df, invalid, dups


def expected_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    # inclusive range at 10-min spacing
    return pd.date_range(start=start, end=end, freq=FREQ)


def align_time_range(
    df_nc: pd.DataFrame, df_csv: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex, pd.Series, pd.Series]:
    if df_nc.empty and df_csv.empty:
        return df_nc, df_csv, pd.DatetimeIndex([], dtype="datetime64[ns]"), pd.Series([], dtype=bool), pd.Series([], dtype=bool)
    min_ts = None
    max_ts = None
    if not df_nc.empty:
        min_ts = df_nc.index.min() if min_ts is None else min(min_ts, df_nc.index.min())
        max_ts = df_nc.index.max() if max_ts is None else max(max_ts, df_nc.index.max())
    if not df_csv.empty:
        min_ts = df_csv.index.min() if min_ts is None else min(min_ts, df_csv.index.min())
        max_ts = df_csv.index.max() if max_ts is None else max(max_ts, df_csv.index.max())
    # apply provided overrides
    if start is not None:
        min_ts = max(min_ts, start) if min_ts is not None else start
    if end is not None:
        max_ts = min(max_ts, end) if max_ts is not None else end
    if min_ts is None or max_ts is None or min_ts > max_ts:
        empty_idx = pd.DatetimeIndex([], dtype="datetime64[ns]")
        return df_nc.iloc[0:0], df_csv.iloc[0:0], empty_idx, pd.Series([], dtype=bool), pd.Series([], dtype=bool)

    idx = expected_index(min_ts, max_ts)

    # mark rows that originally existed before reindex
    if not df_nc.empty:
        df_nc = df_nc.copy()
        df_nc["_row_exists"] = True
    if not df_csv.empty:
        df_csv = df_csv.copy()
        df_csv["_row_exists"] = True

    df_nc = df_nc.reindex(idx)
    df_csv = df_csv.reindex(idx)

    # build existence masks (avoid downcasting warnings)
    if "_row_exists" in df_nc:
        exists_nc = df_nc["_row_exists"].astype("boolean").fillna(False).astype(bool)
    else:
        exists_nc = pd.Series(False, index=idx, dtype=bool)
    if "_row_exists" in df_csv:
        exists_csv = df_csv["_row_exists"].astype("boolean").fillna(False).astype(bool)
    else:
        exists_csv = pd.Series(False, index=idx, dtype=bool)

    df_nc = df_nc.drop(columns=["_row_exists"], errors="ignore")
    df_csv = df_csv.drop(columns=["_row_exists"], errors="ignore")

    return df_nc, df_csv, idx, exists_nc, exists_csv


def numeric_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    a_vals = a.astype(float)
    b_vals = b.astype(float)
    both_nan = a_vals.isna() & b_vals.isna()
    close = np.isclose(a_vals.fillna(0.0), b_vals.fillna(0.0), atol=FLOAT_ATOL)
    return pd.Series(close | both_nan, index=a.index)


def series_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return numeric_equal(a, b)
    # strings or mixed
    a_str = a.astype("string")
    b_str = b.astype("string")
    return (a_str.fillna("<NA>") == b_str.fillna("<NA>"))


def compare_dataframes(df_nc: pd.DataFrame, df_csv: pd.DataFrame) -> Tuple[int, int, int]:
    """
    Returns:
      missing_in_csv: timestamps present in NC but NaN everywhere in CSV (i.e., CSV missing)
      missing_in_nc: timestamps present in CSV but NaN everywhere in NC (i.e., NC missing)
      differing_rows: timestamps present in both where any comparable column differs
    """
    if df_nc.empty and df_csv.empty:
        return 0, 0, 0
    # columns to compare = intersection, exclude private helper columns (prefix "_")
    cols = sorted(c for c in (set(df_nc.columns) & set(df_csv.columns)) if not str(c).startswith("_"))
    # define "present" as having any non-NaN among comparable columns
    present_nc = df_nc[cols].notna().any(axis=1)
    present_csv = df_csv[cols].notna().any(axis=1)

    missing_in_csv = int((present_nc & ~present_csv).sum())
    missing_in_nc = int((present_csv & ~present_nc).sum())

    overlap = present_nc & present_csv
    if overlap.any():
        differs = pd.Series(False, index=df_nc.index)
        for c in cols:
            eq = series_equal(df_nc.loc[overlap, c], df_csv.loc[overlap, c])
            differs.loc[overlap] |= ~eq
        differing_rows = int(differs.sum())
    else:
        differing_rows = 0

    return missing_in_csv, missing_in_nc, differing_rows


def build_issue_series(df: pd.DataFrame, idx: pd.DatetimeIndex, exists: Optional[pd.Series] = None) -> Dict[str, pd.Series]:
    """
    Build boolean series per issue type aligned to idx.
    Separates:
      - missing line: timestamp absent in original data (exists=False)
      - empty row: row existed but all values are NaN
      - per-column NaN: only where a row exists
      - different time structure: timestamp parsed only with fallback
    """
    issues: Dict[str, pd.Series] = {}

    if exists is None:
        # Fallback: infer existence from any non-NaN (cannot distinguish empty row vs missing)
        exists = (df.notna().any(axis=1) if not df.empty else pd.Series(False, index=idx)).reindex(idx).fillna(False)
    else:
        exists = exists.reindex(idx).fillna(False)

    present = (df.notna().any(axis=1) if not df.empty else pd.Series(False, index=idx)).reindex(idx).fillna(False)

    # Missing line (no original row at timestamp)
    issues["missing line"] = ~exists

    # Row existed but contained no values in any field
    issues["empty row (no values)"] = exists & ~present

    # Different time structure (parsed by fallback), only where a row exists
    if "_nonstandard_time" in df.columns:
        nonstd = df["_nonstandard_time"].astype("boolean")
        nonstd_mask = nonstd.reindex(idx).fillna(False) & exists
        issues["different time structure"] = nonstd_mask

    def mark_issue(name: str, mask: pd.Series):
        issues[name] = mask.reindex(idx).fillna(False)

    # Negative numeric values (only relevant where a row exists)
    for col, (rule, _) in GOODNESS_CHECK.items():
        if col in df.columns:
            s = df[col]
            if rule == "nonnegative":
                neg = (s < 0) & s.notna() & exists
                mark_issue(f"negative value in {col}", neg)

    # NaN / no value in required columns — only for existing rows
    for col in ["Flow_m3_sec", "mean_rain", "max_rain", "min_gauges", "max_gauges"]:
        if col in df.columns:
            mark_issue(f"nan in {col}", df[col].isna() & exists)

    # Non-integer in gauge summary (or negative integer) — only for existing rows
    for col in ["min_gauges", "max_gauges"]:
        if col in df.columns:
            s = df[col]
            non_int = (s.isna() | (s.astype("Float64") % 1 != 0)) & exists
            negative = (s < 0) & s.notna() & exists
            mark_issue(f"non-integer in {col}", non_int)
            mark_issue(f"negative value in {col}", negative)

    return issues


def slugify_issue(name: str) -> str:
    # make a safe column suffix from the issue string
    s = name.strip().lower()
    s = s.replace("%", "pct")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def summarize_issues_for_summary(issues: Dict[str, pd.Series], idx: pd.DatetimeIndex, source_tag: str) -> Dict[str, float]:
    """
    Returns a dict with per-issue count and percent over idx length.
    Keys: f"{source_tag}_{slug}_count" and f"{source_tag}_{slug}_pct"
    """
    out: Dict[str, float] = {}
    total = len(idx)
    for issue_name, mask in issues.items():
        cnt = int(mask.sum()) if not mask.empty else 0
        slug = slugify_issue(issue_name)
        out[f"{source_tag}_{slug}_count"] = cnt
        out[f"{source_tag}_{slug}_pct"] = (cnt / total * 100.0) if total > 0 else 0.0
    return out


def group_runs(mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    """
    Given a boolean Series indexed by datetime, return list of (start, end, length) for True runs.
    Length is in number of timesteps (index points).
    """
    if mask.empty:
        return []
    # convert to int to find edges
    m = mask.astype(int).values
    idx = mask.index
    # find start/end positions of runs of 1s
    diffs = np.diff(np.r_[0, m, 0])
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    runs = []
    for s, e in zip(starts, ends):
        runs.append((idx[s], idx[e], int(e - s + 1)))
    return runs


def write_gaps_report(
    basin_id: str,
    out_dir: Path,
    idx: pd.DatetimeIndex,
    issues: Dict[str, pd.Series],
    invalid_date_count: int,
    duplicate_count: int,
    source_tag: str = "nc",
) -> Path:
    # define desired column order up-front
    cols = ["basin_id", "source", "issue", "first_ts", "last_ts", "length_timesteps"]

    rows = []
    for issue_name, series in issues.items():
        for first_ts, last_ts, length in group_runs(series):
            rows.append(
                {
                    "basin_id": basin_id,
                    "source": source_tag,
                    "issue": issue_name,
                    "first_ts": first_ts,
                    "last_ts": last_ts,
                    "length_timesteps": length,
                }
            )
    # invalid date rows (cannot assign timestamps)
    if invalid_date_count:
        rows.append(
            {
                "basin_id": basin_id,
                "source": source_tag,
                "issue": "invalid date (unparsable)",
                "first_ts": pd.NaT,
                "last_ts": pd.NaT,
                "length_timesteps": invalid_date_count,
            }
        )
    if duplicate_count:
        rows.append(
            {
                "basin_id": basin_id,
                "source": source_tag,
                "issue": "duplicate timestamps",
                "first_ts": pd.NaT,
                "last_ts": pd.NaT,
                "length_timesteps": duplicate_count,
            }
        )

    # Create DataFrame with correct columns even if rows is empty
    df_out = pd.DataFrame(rows, columns=cols)
    ensure_report_dir(out_dir)
    out_path = out_dir / f"gaps_il_{basin_id}_{source_tag}.csv"

    # Sort by time (chronological), not by issue
    if not df_out.empty:
        df_out["first_ts"] = pd.to_datetime(df_out["first_ts"], errors="coerce")
        df_out["last_ts"]  = pd.to_datetime(df_out["last_ts"], errors="coerce")
        df_out = df_out.sort_values(
            by=["first_ts", "last_ts", "issue"], ascending=[True, True, True], na_position="last"
        ).reset_index(drop=True)

    df_out.to_csv(out_path, index=False)
    return out_path


def _is_number(x) -> bool:
    try:
        # treat pandas NA as not a number for our tolerant compare
        if pd.isna(x):
            return False
        float(x)
        return True
    except Exception:
        return False


def _cells_equal(a, b) -> bool:
    # numeric tolerant compare with NaN-equality
    if pd.isna(a) and pd.isna(b):
        return True
    if _is_number(a) and _is_number(b):
        try:
            return bool(np.isclose(float(a), float(b), atol=FLOAT_ATOL))
        except Exception:
            pass
    # if only one is NA -> not equal
    if pd.isna(a) or pd.isna(b):
        return False
    # fallback exact compare as strings (force plain strings to avoid pd.NA tri-state)
    sa = "<NA>" if pd.isna(a) else str(a)
    sb = "<NA>" if pd.isna(b) else str(b)
    return sa == sb


def _summarize_mismatch(ts: pd.Timestamp, cols: List[str], row_nc: pd.Series, row_csv: pd.Series) -> Tuple[List[str], str]:
    diff_cols: List[str] = []
    snippets: List[str] = []
    for c in cols:
        a = row_nc.get(c, pd.NA)
        b = row_csv.get(c, pd.NA)
        if not _cells_equal(a, b):
            diff_cols.append(c)
            # short printable values
            va = "NaN" if pd.isna(a) else f"{a}"
            vb = "NaN" if pd.isna(b) else f"{b}"
            snippets.append(f"{c}: nc={va}, csv={vb}")
    # keep detail string concise
    detail = " | ".join(snippets)
    if len(detail) > 1000:
        detail = detail[:997] + "..."
    return diff_cols, detail


def write_diff_examples(
    basin_id: str,
    out_dir: Path,
    idx: pd.DatetimeIndex,
    df_nc_al: pd.DataFrame,
    df_csv_al: pd.DataFrame,
    max_examples: int = 100,
) -> Path:
    # columns considered for presence and comparison
    cols_nc_all = [c for c in df_nc_al.columns if not str(c).startswith("_")]
    cols_csv_all = [c for c in df_csv_al.columns if not str(c).startswith("_")]
    cols_compare = sorted(list(set(cols_nc_all) & set(cols_csv_all)))
    cols_union = sorted(list(set(cols_nc_all) | set(cols_csv_all)))

    # presence by any non-NA among that df's own data columns
    present_nc = df_nc_al[cols_nc_all].notna().any(axis=1) if cols_nc_all else pd.Series(False, index=idx)
    present_csv = df_csv_al[cols_csv_all].notna().any(axis=1) if cols_csv_all else pd.Series(False, index=idx)

    missing_in_csv_mask = present_nc & ~present_csv
    missing_in_nc_mask = present_csv & ~present_nc
    overlap_mask = present_nc & present_csv

    # find value mismatches on overlap
    differs_mask = pd.Series(False, index=idx)
    if overlap_mask.any() and cols_compare:
        for c in cols_compare:
            eq = series_equal(df_nc_al.loc[overlap_mask, c], df_csv_al.loc[overlap_mask, c])
            differs_mask.loc[overlap_mask] |= ~eq

    # helper: first timestamp of each contiguous run
    def run_starts(mask: pd.Series) -> List[pd.Timestamp]:
        return [first for (first, _last, _L) in group_runs(mask)]

    # helper: map run start -> (run_end, run_length)
    def run_map(mask: pd.Series) -> Dict[pd.Timestamp, Tuple[pd.Timestamp, int]]:
        return {first: (last, length) for (first, last, length) in group_runs(mask)}

    # one example per run (start) for each type
    starts_missing_csv = [(ts, "missing_in_csv") for ts in run_starts(missing_in_csv_mask)]
    starts_missing_nc  = [(ts, "missing_in_nc")  for ts in run_starts(missing_in_nc_mask)]
    starts_mismatch    = [(ts, "value_mismatch") for ts in run_starts(differs_mask)]

    # run metadata for sequence info
    runs_missing_csv = run_map(missing_in_csv_mask)
    runs_missing_nc  = run_map(missing_in_nc_mask)
    runs_mismatch    = run_map(differs_mask)

    # merge all run starts chronologically and cap to max_examples
    selected = starts_missing_csv + starts_missing_nc + starts_mismatch
    selected.sort(key=lambda x: (x[0], x[1]))
    selected = selected[:max_examples]

    rows = []
    for ts, kind in selected:
        row_nc = df_nc_al.loc[ts] if ts in df_nc_al.index else pd.Series(dtype="object")
        row_csv = df_csv_al.loc[ts] if ts in df_csv_al.index else pd.Series(dtype="object")

        mismatch_columns = ""
        mismatch_details = ""
        if kind == "value_mismatch":
            diff_cols, detail = _summarize_mismatch(ts, cols_compare, row_nc, row_csv)
            mismatch_columns = ";".join(diff_cols)
            mismatch_details = detail
            seq_last, seq_len = runs_mismatch.get(ts, (ts, 1))
        elif kind == "missing_in_csv":
            mismatch_details = "timestamp present in NC, missing in CSV"
            seq_last, seq_len = runs_missing_csv.get(ts, (ts, 1))
        elif kind == "missing_in_nc":
            mismatch_details = "timestamp present in CSV, missing in NC"
            seq_last, seq_len = runs_missing_nc.get(ts, (ts, 1))
        else:
            seq_last, seq_len = (ts, 1)

        record = {
            "basin_id": basin_id,
            "timestamp": ts,
            "difference_type": kind,
            "mismatch_columns": mismatch_columns,
            "mismatch_details": mismatch_details,
            "sequence_last_ts": seq_last,
            "sequence_length_timesteps": int(seq_len),
        }
        # include all union columns, with both sources
        for c in cols_union:
            record[f"{c}_nc"] = row_nc.get(c, pd.NA)
            record[f"{c}_csv"] = row_csv.get(c, pd.NA)
        rows.append(record)

    ensure_report_dir(out_dir)
    out_path = out_dir / f"diffs_il_{basin_id}.csv"
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.sort_values(by=["timestamp", "difference_type"]).reset_index(drop=True)
    df_out.to_csv(out_path, index=False)
    return out_path


def process_basin(
    basin_id: str,
    nc_path: Path,
    csv_path: Path,
    out_dir: Path,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> Dict[str, object]:
    # read both
    df_nc, invalid_nc, dup_nc = read_netcdf_file(nc_path)
    df_csv, invalid_csv, dup_csv = read_csv_file(csv_path)

    # align to common time range
    df_nc_al, df_csv_al, idx, exists_nc, exists_csv = align_time_range(df_nc, df_csv, start, end)

    # comparison summary
    missing_in_csv, missing_in_nc, differing_rows = compare_dataframes(df_nc_al, df_csv_al)
    identical = (missing_in_csv == 0) and (missing_in_nc == 0) and (differing_rows == 0)

    # count "different time structure" rows per source (only where a row exists)
    diff_time_nc = 0
    diff_time_csv = 0
    if "_nonstandard_time" in df_nc_al.columns:
        nonstd_nc = df_nc_al["_nonstandard_time"].astype("boolean").reindex(idx).fillna(False)
        diff_time_nc = int((nonstd_nc & exists_nc.astype("boolean")).sum())
    if "_nonstandard_time" in df_csv_al.columns:
        nonstd_csv = df_csv_al["_nonstandard_time"].astype("boolean").reindex(idx).fillna(False)
        diff_time_csv = int((nonstd_csv & exists_csv.astype("boolean")).sum())

    # per-basin gaps report (based on NC)
    issues_nc = build_issue_series(df_nc_al, idx, exists_nc)
    write_gaps_report(
        basin_id=basin_id,
        out_dir=out_dir,
        idx=idx,
        issues=issues_nc,
        invalid_date_count=len(invalid_nc),
        duplicate_count=len(dup_nc),
        source_tag="nc",
    )
    # Also produce CSV gaps to aid debugging
    issues_csv = {}
    if not df_csv_al.empty or invalid_csv or dup_csv:
        issues_csv = build_issue_series(df_csv_al, idx, exists_csv)
        write_gaps_report(
            basin_id=basin_id,
            out_dir=out_dir,
            idx=idx,
            issues=issues_csv,
            invalid_date_count=len(invalid_csv),
            duplicate_count=len(dup_csv),
            source_tag="csv",
        )

    # Also produce per-basin examples of differing lines (max 100)
    write_diff_examples(
        basin_id=basin_id,
        out_dir=out_dir,
        idx=idx,
        df_nc_al=df_nc_al,
        df_csv_al=df_csv_al,
        max_examples=100,
    )

    # Build the result row
    res: Dict[str, object] = {
        "basin_id": basin_id,
        "nc_file": str(nc_path),
        "csv_file": str(csv_path) if csv_path.exists() else "",
        "identical": identical,
        "missing_in_csv": missing_in_csv,
        "missing_in_nc": missing_in_nc,
        "differing_rows": differing_rows,
        "different_time_rows_nc": diff_time_nc,
        "different_time_rows_csv": diff_time_csv,
        "total_timesteps": len(idx),
    }

    # Add per-issue counts and percentages for NC and CSV
    res.update(summarize_issues_for_summary(issues_nc, idx, "nc"))
    if issues_csv:
        res.update(summarize_issues_for_summary(issues_csv, idx, "csv"))

    return res


def main():
    p = argparse.ArgumentParser(description="Compare IL netCDF and CSV time series and report gaps.")
    p.add_argument("--nc-dir", default=IL_NC_DIR_DEFAULT, help="Directory with il_*.nc files.")
    p.add_argument("--csv-dir", default=IL_CSV_DIR_DEFAULT, help="Directory with il_*.csv files.")
    p.add_argument("--out-dir", default=REPORT_DIR_DEFAULT, help="Directory to write reports.")
    p.add_argument(
        "--start",
        default=None,
        help="Optional start datetime (dayfirst). Example: 01/01/2000 00:00",
    )
    p.add_argument(
        "--end",
        default=None,
        help="Optional end datetime (dayfirst). Example: 31/12/2020 23:50",
    )
    # Add: flag to disable progress bars
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    p.add_argument(
        "--csv-time-format",
        default=CSV_TIME_FORMAT_DEFAULT,
        help='Datetime format for CSV "date" column (default: %Y-%m-%d %H:%M:%S). Use "auto" to auto-detect.',
    )
    args = p.parse_args()

    nc_dir = Path(args.nc_dir)
    csv_dir = Path(args.csv_dir)
    out_dir = ensure_report_dir(Path(args.out_dir))

    start = pd.to_datetime(args.start, dayfirst=True) if args.start else None
    end = pd.to_datetime(args.end, dayfirst=True) if args.end else None

    # Apply CSV time format preference
    global CSV_TIME_FORMAT
    CSV_TIME_FORMAT = None if (args.csv_time_format or "").lower() == "auto" else args.csv_time_format

    # discover basins from NC files
    basins = []
    for f in nc_dir.glob("il_*.nc"):
        bid = parse_basin_id_from_name(f.name)
        if bid:
            basins.append((bid, f))
    if not basins:
        print(f"No NC files found in {nc_dir}")
        return

    summary_rows = []
    # Add: wrap per-basin loop with tqdm unless disabled
    basin_iter = basins
    if not args.no_progress:
        basin_iter = tqdm(basins, desc="Processing basins", unit="basin", leave=False)

    for bid, nc_path in basin_iter:
        csv_path = csv_dir / f"il_{bid}.csv"
        res = process_basin(
            basin_id=bid,
            nc_path=nc_path,
            csv_path=csv_path,
            out_dir=out_dir,
            start=start,
            end=end,
        )
        summary_rows.append(res)

    df_summary = pd.DataFrame(summary_rows).sort_values("basin_id")

    # Fill NaNs only for numeric columns (counts/pcts) to make CSV cleaner
    numeric_cols = [c for c in df_summary.columns if c.endswith("_count") or c.endswith("_pct")]
    if numeric_cols:
        df_summary[numeric_cols] = df_summary[numeric_cols].fillna(0)

    summary_path = out_dir / "comparison_report.csv"
    df_summary.to_csv(summary_path, index=False)

    print(f"Wrote comparison summary: {summary_path}")
    print(f"Gaps reports written to: {out_dir}")


if __name__ == "__main__":
    main()
