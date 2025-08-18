import os
import re
import argparse
from typing import Dict, List, Optional

import pandas as pd
import xarray as xr


# === Default paths (can be overridden via CLI) ===
BASIN_FOLDER = r"C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output\basin_average_rain"
CARAVAN_FOLDER = r"C:\PhD\Data\Caravan\timeseries\csv\il"

OUT_CSV = r"C:\PhD\Data\Caravan\timeseries\csv\il_new"
OUT_NC = r"C:\PhD\Data\Caravan\timeseries\netcdf\il_new"

WINTER_OUT_CSV = r"C:\PhD\Data\Caravan\Caravan_winter\timeseries\csv\il_new"
WINTER_OUT_NC = r"C:\PhD\Data\Caravan\Caravan_winter\timeseries\netcdf\il_new"

TARGET_FREQ = "10min"


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def list_ids(folder: str) -> List[str]:
    ids: List[str] = []
    if not os.path.isdir(folder):
        return ids
    for fn in os.listdir(folder):
        if fn.lower().endswith(".csv") and fn.startswith("il_"):
            m = re.match(r"il_(\d+)\.csv$", fn, re.IGNORECASE)
            if m:
                ids.append(m.group(1))
    return sorted(set(ids))


def _robust_parse_datetime(series: pd.Series) -> pd.Series:
    # Try dayfirst=True first (common in IL data). If many NaT, retry without.
    s = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=False)
    nat_ratio = s.isna().mean() if len(s) else 0.0
    if nat_ratio > 0.5:
        s2 = pd.to_datetime(series, errors="coerce", dayfirst=False, utc=False)
        if s2.notna().sum() > s.notna().sum():
            s = s2
    # Ensure tz-naive
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def read_caravan_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize date column name
    if "date" not in df.columns:
        # Sometimes first column is date with potential BOM/whitespace
        for cand in df.columns:
            if cand.strip().lower() in {"date", "datetime", "time"}:
                df.rename(columns={cand: "date"}, inplace=True)
                break
    if "date" not in df.columns:
        raise ValueError(f"No date column found in Caravan file: {path}")
    df["date"] = _robust_parse_datetime(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df


def read_basin_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize potential date column names
    date_col = None
    for cand in df.columns:
        if str(cand).strip().lower() in {"date", "datetime", "time"}:
            date_col = cand
            break
    if date_col is None:
        # If the last column is likely date (as in sample screenshot)
        if len(df.columns) >= 1:
            date_col = df.columns[-1]
        else:
            raise ValueError(f"No columns in basin file: {path}")

    df.rename(columns={date_col: "date"}, inplace=True)

    # Normalize rain/gauges columns (handle typos/plurals)
    rename_map: Dict[str, str] = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"mean_rain", "avg_rain", "rain_mean"}:
            rename_map[c] = "mean_rain"
        elif cl in {"max_rain"}:
            rename_map[c] = "max_rain"
        elif cl in {"min_gauges", "min_gauge", "min_gaug", "mingauges", "mingauge"}:
            rename_map[c] = "min_gauges"
        elif cl in {"max_gauges", "max_gauge", "max_gaug", "maxgauges", "maxgauge"}:
            rename_map[c] = "max_gauges"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    wanted = ["date", "mean_rain", "max_rain", "min_gauges", "max_gauges"]
    # Keep only available wanted cols
    keep = [c for c in wanted if c in df.columns]
    if "date" not in keep:
        raise ValueError(f"Failed to identify 'date' in basin file: {path}")

    df = df[keep]
    df["date"] = _robust_parse_datetime(df["date"])
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"]).sort_values("date")
    return df


def merge_on_date(caravan_df: pd.DataFrame, basin_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if basin_df is None or basin_df.empty:
        return caravan_df.copy()
    # Left join on caravan time index to keep its exact rows
    left = caravan_df.copy()
    right = basin_df.copy()
    merged = pd.merge(left, right, on="date", how="left")
    return merged


def save_csv_and_netcdf(df: pd.DataFrame, csv_path: str, nc_path: str, set_freq: Optional[str] = None) -> None:
    # Save CSV
    df.to_csv(csv_path, index=False)

    # Save NetCDF
    dfi = df.copy()
    dfi.set_index("date", inplace=True)
    if set_freq:
        try:
            dfi.index.freq = set_freq
        except Exception:
            # If not perfectly regular, ignore frequency assignment
            pass
    ds = xr.Dataset.from_dataframe(dfi)
    ds.to_netcdf(nc_path)


# New helper to round rain columns to 2 decimals
def _round_rain_cols_inplace(df: pd.DataFrame) -> None:
    for col in ("mean_rain", "max_rain"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)


def filter_winter(df: pd.DataFrame) -> pd.DataFrame:
    # Keep Oct–Apr
    if "date" not in df.columns:
        raise ValueError("DataFrame missing 'date' column for winter filtering")
    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.month
    is_winter = ((tmp["month"] >= 10) & (tmp["month"] <= 12)) | ((tmp["month"] >= 1) & (tmp["month"] <= 4))
    tmp = tmp[is_winter].drop(columns=["month"], errors="ignore")
    return tmp


def reindex_fill_10min(df: pd.DataFrame, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    if df.empty:
        return df
    dfi = df.copy()
    dfi = dfi.set_index("date")
    if start is None:
        start = dfi.index.min()
    if end is None:
        end = dfi.index.max()
    full_index = pd.date_range(start=start, end=end, freq=TARGET_FREQ)
    dfi = dfi.reindex(full_index)
    dfi.index.name = "date"
    dfi = dfi.reset_index()
    return dfi


def process_id(basin_id: str,
               paths: Dict[str, str],
               copy_if_missing_basin: bool,
               write_outputs: bool,
               do_winter: bool) -> Dict[str, object]:
    caravan_file = os.path.join(paths["caravan"], f"il_{basin_id}.csv")
    basin_file = os.path.join(paths["basin"], f"il_{basin_id}.csv")

    has_caravan = os.path.exists(caravan_file)
    has_basin = os.path.exists(basin_file)

    result: Dict[str, object] = {
        "basin_id": basin_id,
        "caravan_file": caravan_file if has_caravan else "",
        "basin_file": basin_file if has_basin else "",
        "has_caravan": has_caravan,
        "has_basin": has_basin,
        "action": "skipped",
        "out_csv": "",
        "out_nc": "",
        "winter_out_csv": "",
        "winter_out_nc": "",
    }

    if not has_caravan and not has_basin:
        result["action"] = "missing_both"
        return result

    if not has_caravan and has_basin:
        result["action"] = "missing_caravan"
        return result

    # has_caravan True here
    caravan_df = read_caravan_csv(caravan_file)
    basin_df = read_basin_csv(basin_file) if has_basin else None
    merged = merge_on_date(caravan_df, basin_df)

    # Round rain columns to 2 decimals before saving
    _round_rain_cols_inplace(merged)

    if not has_basin and copy_if_missing_basin:
        result["action"] = "copied_caravan_only"
    elif has_basin:
        result["action"] = "merged"
    else:
        result["action"] = "caravan_only_no_copy" if not write_outputs else "caravan_only"

    out_csv = os.path.join(paths["out_csv"], f"il_{basin_id}.csv")
    out_nc = os.path.join(paths["out_nc"], f"il_{basin_id}.nc")
    winter_csv = os.path.join(paths["winter_csv"], f"il_{basin_id}.csv")
    winter_nc = os.path.join(paths["winter_nc"], f"il_{basin_id}.nc")

    if write_outputs:
        ensure_dirs(os.path.dirname(out_csv), os.path.dirname(out_nc))
        save_csv_and_netcdf(merged, out_csv, out_nc, set_freq=None)
        result["out_csv"] = out_csv
        result["out_nc"] = out_nc

        if do_winter:
            ensure_dirs(os.path.dirname(winter_csv), os.path.dirname(winter_nc))
            winter_df = filter_winter(merged)
            winter_df = reindex_fill_10min(winter_df)
            # Ensure rounding persists after reindexing (no-op for NaNs)
            _round_rain_cols_inplace(winter_df)
            save_csv_and_netcdf(winter_df, winter_csv, winter_nc, set_freq=TARGET_FREQ)
            result["winter_out_csv"] = winter_csv
            result["winter_out_nc"] = winter_nc

    return result


def main():
    parser = argparse.ArgumentParser(description="Merge basin-average rain columns into Caravan timeseries per basin, and produce winter-only versions.")
    parser.add_argument("--basin-folder", default=BASIN_FOLDER)
    parser.add_argument("--caravan-folder", default=CARAVAN_FOLDER)
    parser.add_argument("--out-csv", default=OUT_CSV)
    parser.add_argument("--out-nc", default=OUT_NC)
    parser.add_argument("--winter-out-csv", default=WINTER_OUT_CSV)
    parser.add_argument("--winter-out-nc", default=WINTER_OUT_NC)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N basin IDs (for quick tests)")
    parser.add_argument("--dry-run", action="store_true", help="Do not write any outputs; only print a report")
    parser.add_argument("--no-winter", action="store_true", help="Skip writing winter-only versions")
    parser.add_argument("--no-copy-if-missing-basin", action="store_true", help="If basin rain file is missing, still write outputs (caravan-only). Default is to copy caravan-only outputs anyway.")

    args = parser.parse_args()

    paths = {
        "basin": args.basin_folder,
        "caravan": args.caravan_folder,
        "out_csv": args.out_csv,
        "out_nc": args.out_nc,
        "winter_csv": args.winter_out_csv,
        "winter_nc": args.winter_out_nc,
    }

    # Discover IDs from both sources
    caravan_ids = list_ids(paths["caravan"])
    basin_ids = list_ids(paths["basin"])
    all_ids = sorted(set(caravan_ids) | set(basin_ids), key=lambda x: int(x))
    if args.limit:
        all_ids = all_ids[: args.limit]

    write_outputs = not args.dry_run
    do_winter = not args.no_winter
    copy_if_missing_basin = not args.no_copy_if_missing_basin

    results: List[Dict[str, object]] = []
    total = len(all_ids)
    ensure_dirs(paths["out_csv"])  # for report

    for i, bid in enumerate(all_ids, 1):
        res = process_id(
            basin_id=bid,
            paths=paths,
            copy_if_missing_basin=copy_if_missing_basin,
            write_outputs=write_outputs,
            do_winter=do_winter,
        )
        results.append(res)
        print(f"[{i}/{total}] {bid}: {res['action']}")

    # Save report
    report_df = pd.DataFrame(results)
    report_path = os.path.join(paths["out_csv"], "merge_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"Report saved: {report_path}")






if __name__ == "__main__":
    main()
