from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt

# --- config ---
DEFAULT_DATA_DIR = Path(r"C:\PhD\Data\Caravan\timeseries\netcdf\il")
DEFAULT_RUN_DIR = Path(r"C:\PhD\Python\neuralhydrology\Experiments\Calculate_mean_basin_rain_from_gauges\Train_single_model\results\job_0\run_014_av_rain_all_year")
OUT_CSV = "data_vs_outputs_diagnosis.csv"
MAKE_MISMATCH_PLOTS = True  # set True to also render per-basin mismatch plots

# --- helpers: load config and cached outputs ---
def load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yml not found in {run_dir}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def load_cached_outputs(cache_dir: Path) -> dict:
    results = {}
    if not cache_dir.exists():
        return results
    for nc in cache_dir.glob("*.nc"):
        try:
            basin = nc.stem.replace("__10min", "")
            ds = xr.load_dataset(nc)
            # Expect variables Flow_m3_sec_obs and Flow_m3_sec_sim; if not, try to infer
            if "Flow_m3_sec_obs" in ds and "Flow_m3_sec_sim" in ds:
                results[basin] = ds
            else:
                # Fallback: try to split a single variable into obs/sim is not possible; store as-is
                results[basin] = ds
        except Exception as e:
            print(f"[WARN] Failed loading cache file {nc}: {e}")
    return results

# --- helpers: load raw basin time series from Caravan ---
CANDIDATE_FILE_EXT = (".parquet", ".csv", ".feather", ".nc", ".nc4")
DATE_COLS = ("date", "time", "datetime", "Date", "Time", "Datetime")
BASIN_COLS = ("basin", "basin_id", "gauge_id", "station_id", "camels_id", "id", "Station_ID")
FLOW_COLS = ("Flow_m3_sec", "QObs", "Qobs", "qobs", "streamflow", "discharge")

def basin_id_to_number(basin_id: str) -> str:
    # "il_17168" -> "17168"
    m = re.search(r"(\d+)$", str(basin_id))
    return m.group(1) if m else str(basin_id)

def _parse_dates_in_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        if c in df.columns:
            # removed deprecated infer_datetime_format
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
            df = df.set_index(c).sort_index()
            return df
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    return df

def _select_basin_rows(df: pd.DataFrame, basin_id: str) -> pd.DataFrame:
    # Match by Station_ID or generic basin id columns, exact match preferred
    basin_num = basin_id_to_number(basin_id)
    for bc in BASIN_COLS:
        if bc in df.columns:
            col = df[bc].astype(str)
            exact = df[col == basin_num]
            if not exact.empty:
                return exact
            contains = df[col.str.contains(basin_num)]
            if not contains.empty:
                return contains
    return df

def _find_candidate_files(data_dir: Path, basin_id: str):
    # Look for files that include exact basin id or numeric suffix
    pats = [f"*{basin_id}*"]
    m = re.search(r"(\d+)$", basin_id)
    if m:
        pats.append(f"*{m.group(1)}*")
    files = []
    for pat in pats:
        for ext in CANDIDATE_FILE_EXT:
            files.extend(data_dir.rglob(pat + ext))
    # de-duplicate preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq

def load_raw_timeseries_from_netcdf(nc_dir: Path, basin_id: str) -> pd.DataFrame | None:
    """Load Flow_m3_sec for a basin from il_<basin>.nc with time/date index."""
    basin_num = basin_id_to_number(basin_id)
    nc_path = nc_dir / f"il_{basin_num}.nc"
    if not nc_path.exists():
        return None
    try:
        ds = xr.load_dataset(nc_path)
        # time coordinate could be 'date' or 'time'
        time_coord = "date" if "date" in ds.coords else ("time" if "time" in ds.coords else None)
        if time_coord is None:
            return None
        # variable expected: Flow_m3_sec; fallback to any var containing 'flow'
        var = "Flow_m3_sec" if "Flow_m3_sec" in ds.data_vars else None
        if var is None:
            for v in ds.data_vars:
                if "flow" in v.lower() or "discharge" in v.lower():
                    var = v
                    break
        if var is None:
            return None
        df = ds[[var]].to_dataframe().reset_index()
        df = _parse_dates_in_df(df)
        return df.rename(columns={var: "Flow_m3_sec"})
    except Exception as e:
        print(f"[WARN] Failed loading NetCDF for {basin_id}: {nc_path} ({e})")
        return None

def load_raw_timeseries_from_csv(root_dir: Path, basin_id: str) -> pd.DataFrame | None:
    """Load Flow_m3_sec for a basin from CSVs with header: date, Station_ID, Flow_m3_sec, ..."""
    basin_num = basin_id_to_number(basin_id)
    # Heuristic: search CSVs under root_dir for this basin number
    candidates = list(root_dir.rglob("*.csv"))
    for f in candidates:
        try:
            df = pd.read_csv(f)
            if "date" not in df.columns or "Station_ID" not in df.columns:
                continue
            # Parse dates in provided format; dayfirst to match examples
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
            df = df[df["Station_ID"].astype(str) == basin_num]
            if df.empty:
                continue
            df = df.set_index("date").sort_index()
            if "Flow_m3_sec" not in df.columns:
                continue
            return df[["Flow_m3_sec"]].copy()
        except Exception:
            continue
    return None

def load_raw_timeseries_for_basin(data_dir: Path, basin_id: str) -> pd.DataFrame | None:
    """Try NetCDF first (il_<basin>.nc), then CSV fallback structured as provided."""
    # NetCDF preferred
    df_nc = load_raw_timeseries_from_netcdf(data_dir, basin_id)
    if df_nc is not None and not df_nc.empty:
        print(f"[RAW] {basin_id}: loaded NetCDF from {data_dir}")
        return df_nc
    # CSV fallback: look one level up if data_dir is the nc folder
    csv_root = data_dir.parent if data_dir.name.lower() == "il" else data_dir
    df_csv = load_raw_timeseries_from_csv(csv_root, basin_id)
    if df_csv is not None and not df_csv.empty:
        print(f"[RAW] {basin_id}: loaded CSV from {csv_root}")
        return df_csv
    # As last resort, attempt previous generic search (kept for compatibility)
    candidates = _find_candidate_files(csv_root, basin_id)
    for f in candidates:
        try:
            if f.suffix.lower() == ".parquet":
                df = pd.read_parquet(f)
            elif f.suffix.lower() == ".feather":
                df = pd.read_feather(f)
            elif f.suffix.lower() in (".nc", ".nc4"):
                ds = xr.load_dataset(f)
                df = ds.to_dataframe().reset_index()
            else:
                continue
            df = _parse_dates_in_df(df)
            if df.empty:
                continue
            df = _select_basin_rows(df, basin_id)
            if df.empty:
                continue
            if "Flow_m3_sec" in df.columns:
                return df[["Flow_m3_sec"]]
            # fallback to any numeric column
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                return df[[num_cols[0]]].rename(columns={num_cols[0]: "Flow_m3_sec"})
        except Exception:
            continue
    return None

# --- core diagnosis ---
def _xr_to_series(ds: xr.Dataset, varname: str) -> pd.Series | None:
    if varname in ds:
        da = ds[varname]
        # date coord may be named 'date' (neuralhydrology)
        if "date" in da.coords:
            t = pd.to_datetime(da["date"].values)
            v = np.asarray(da).flatten()
            return pd.Series(v, index=t, name=varname)
        # try to infer
        for c in ("time", "datetime"):
            if c in da.coords:
                t = pd.to_datetime(da[c].values)
                v = np.asarray(da).flatten()
                return pd.Series(v, index=t, name=varname)
    return None

def summarize_series(s: pd.Series) -> dict:
    if s is None or s.empty:
        return dict(n=0, n_nan=0, start=None, end=None, mean=np.nan, median=np.nan, freq=np.nan)
    n = len(s)
    n_nan = int(np.isnan(s.values).sum())
    start = s.index.min()
    end = s.index.max()
    try:
        diffs = pd.Series(s.index[1:] - s.index[:-1]).dt.total_seconds()
        freq_val = diffs.mode().iat[0] if not diffs.empty else np.nan
    except Exception:
        freq_val = np.nan
    # ensure numeric float (or np.nan)
    try:
        freq_val = float(freq_val)
    except Exception:
        freq_val = np.nan
    return dict(
        n=int(n), n_nan=n_nan,
        start=str(start) if pd.notnull(start) else None,
        end=str(end) if pd.notnull(end) else None,
        mean=float(np.nanmean(s.values)) if n > 0 else np.nan,
        median=float(np.nanmedian(s.values)) if n > 0 else np.nan,
        freq=freq_val
    )

# --- helper: safe finite check ---
def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float, np.floating)) and np.isfinite(x)

# --- New: resolve run directory and cache directory robustly ---
def resolve_run_dir(run_dir: Path) -> Path:
    # If this dir already has a config.yml, use it
    if (run_dir / "config.yml").exists():
        return run_dir
    # Search recursively for config.yml and pick the most recently modified one
    candidates = list(run_dir.rglob("config.yml"))
    if not candidates:
        return run_dir
    choice = max(candidates, key=lambda p: p.stat().st_mtime).parent
    print(f"[INFO] Resolved actual run directory: {choice}")
    return choice

def find_cache_dir(run_dir: Path) -> Path | None:
    primary = run_dir / "cached_evaluation_10min"
    if primary.exists() and any(primary.glob("*.nc")):
        return primary
    # Search recursively and pick the cache dir that has the most recent file
    candidates = []
    for d in run_dir.rglob("cached_evaluation_10min"):
        nc_files = list(d.glob("*.nc"))
        if nc_files:
            latest_mtime = max(f.stat().st_mtime for f in nc_files)
            candidates.append((latest_mtime, d))
    if candidates:
        _, best = max(candidates, key=lambda t: t[0])
        print(f"[INFO] Using cache directory: {best}")
        return best
    return None

# --- New: pick a robust 2-week window to plot (prefer raw-only region) ---
def _pick_plot_window(raw_idx: pd.DatetimeIndex,
                      obs_idx: pd.DatetimeIndex | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    try:
        raw_idx = pd.DatetimeIndex(raw_idx) if raw_idx is not None else pd.DatetimeIndex([])
        obs_idx = pd.DatetimeIndex(obs_idx) if obs_idx is not None and len(obs_idx) else pd.DatetimeIndex([])
        if len(raw_idx) == 0:
            return None
        center = None
        if len(obs_idx):
            # Prefer a timestamp where raw has data but outputs miss
            missing = raw_idx.difference(obs_idx)
            if len(missing):
                center = missing.min()
            else:
                inter = raw_idx.intersection(obs_idx)
                if len(inter):
                    center = inter[int(len(inter) // 2)]
        if center is None:
            center = raw_idx[int(len(raw_idx) // 2)]
        win_start = center - pd.Timedelta(days=7)
        win_end = center + pd.Timedelta(days=7)
        return win_start, win_end
    except Exception:
        return None

# --- RESTORED: index comparison helper ---
def compare_indices(raw_idx: pd.DatetimeIndex, out_idx: pd.DatetimeIndex) -> tuple[int, int]:
    """Return counts of timestamps present in raw but missing in outputs, and vice versa."""
    if raw_idx is None or out_idx is None:
        return 0, 0
    try:
        raw_idx = pd.DatetimeIndex(raw_idx)
        out_idx = pd.DatetimeIndex(out_idx)
    except Exception:
        return 0, 0
    if len(raw_idx) == 0 or len(out_idx) == 0:
        return 0, 0
    missing_in_out = pd.Index(raw_idx).difference(pd.Index(out_idx))
    missing_in_raw = pd.Index(out_idx).difference(pd.Index(raw_idx))
    return int(len(missing_in_out)), int(len(missing_in_raw))

def main():
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DATA_DIR

    # New: resolve nested run subfolder containing config.yml
    run_dir = resolve_run_dir(run_dir)
    print(f"[INFO] Using run_dir={run_dir}")
    print(f"[INFO] Using data_dir={data_dir}")

    cfg = load_config(run_dir)

    # New: locate the correct cache directory (may be under a nested subfolder)
    cache_dir = find_cache_dir(run_dir)
    if cache_dir is None:
        print(f"[ERROR] No cached outputs found under {run_dir} (no cached_evaluation_10min/*.nc).")
        print("[HINT] Run your evaluation script with caching enabled first, then re-run this diagnosis.")
        return

    outputs = load_cached_outputs(cache_dir)
    if not outputs:
        print(f"[ERROR] Failed to load any cached outputs from {cache_dir}.")
        return

    rows = []
    diag_dir = run_dir / "diagnostics_compare"
    diag_dir.mkdir(exist_ok=True)

    plots_saved = 0  # New: counter
    for basin, ds in outputs.items():
        # Expected variables from evaluation
        s_obs = _xr_to_series(ds, "Flow_m3_sec_obs")
        s_sim = _xr_to_series(ds, "Flow_m3_sec_sim")
        # Fallback: if not present, try any variable names containing 'obs'/'sim'
        if s_obs is None:
            for v in ds.data_vars:
                if "obs" in v.lower():
                    s_obs = _xr_to_series(ds, v)
                    break
        if s_sim is None:
            for v in ds.data_vars:
                if "sim" in v.lower():
                    s_sim = _xr_to_series(ds, v)
                    break

        raw_df = load_raw_timeseries_for_basin(data_dir, basin)
        if raw_df is None:
            print(f"[WARN] Raw data not found for basin {basin} under {data_dir}")
            raw_idx = pd.DatetimeIndex([])
            flow_col = None
            raw_stats = dict(n=0, n_nan=0, start=None, end=None, mean=np.nan, median=np.nan, freq=np.nan)
        else:
            # pick a flow column
            flow_col = None
            for c in FLOW_COLS:
                if c in raw_df.columns:
                    flow_col = c
                    break
            if flow_col is None:
                # cannot find flow column, fallback to any numeric col
                num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
                flow_col = num_cols[0] if num_cols else None
            s_raw = raw_df[flow_col] if flow_col in raw_df.columns else None
            if s_raw is not None and not isinstance(s_raw.index, pd.DatetimeIndex):
                # parse index if needed
                s_raw = s_raw.copy()
                s_raw.index = pd.to_datetime(s_raw.index, errors="coerce")
                s_raw = s_raw.dropna()
            raw_idx = s_raw.index if s_raw is not None else pd.DatetimeIndex([])
            raw_stats = summarize_series(s_raw)

        obs_stats = summarize_series(s_obs) if s_obs is not None else summarize_series(pd.Series(dtype=float))
        sim_stats = summarize_series(s_sim) if s_sim is not None else summarize_series(pd.Series(dtype=float))

        # Compare coverage
        miss_out, miss_raw = compare_indices(raw_idx, s_obs.index if s_obs is not None else pd.DatetimeIndex([]))
        miss_out_sim, _ = compare_indices(raw_idx, s_sim.index if s_sim is not None else pd.DatetimeIndex([]))

        rows.append({
            "basin": basin,
            # raw
            "raw_n": raw_stats["n"], "raw_nan": raw_stats["n_nan"],
            "raw_start": raw_stats["start"], "raw_end": raw_stats["end"],
            "raw_mean": raw_stats["mean"], "raw_median": raw_stats["median"],
            "raw_dt_seconds_mode": raw_stats["freq"],
            # outputs obs
            "obs_n": obs_stats["n"], "obs_nan": obs_stats["n_nan"],
            "obs_start": obs_stats["start"], "obs_end": obs_stats["end"],
            "obs_mean": obs_stats["mean"], "obs_median": obs_stats["median"],
            "obs_dt_seconds_mode": obs_stats["freq"],
            # outputs sim
            "sim_n": sim_stats["n"], "sim_nan": sim_stats["n_nan"],
            "sim_start": sim_stats["start"], "sim_end": sim_stats["end"],
            "sim_mean": sim_stats["mean"], "sim_median": sim_stats["median"],
            "sim_dt_seconds_mode": sim_stats["freq"],
            # mismatches
            "raw_not_in_outputs_obs": miss_out,
            "raw_not_in_outputs_sim": miss_out_sim,
            "outputs_obs_not_in_raw": miss_raw,
            # quick flags
            "all_sim_nan": bool(sim_stats["n"] > 0 and sim_stats["n"] == sim_stats["n_nan"]),
            "outputs_empty": bool(obs_stats["n"] == 0 and sim_stats["n"] == 0),
            "freq_mismatch": not (_is_finite_number(raw_stats["freq"]) and _is_finite_number(obs_stats["freq"]) and abs(float(raw_stats["freq"]) - float(obs_stats["freq"])) < 1e-6)
        })

        # Optional: small plot highlighting a 2-week window
        # Relaxed: plot if any mismatch or any overlap (even if only obs or sim present)
        if MAKE_MISMATCH_PLOTS and raw_idx.size and ((miss_out > 0) or (miss_out_sim > 0) or (miss_raw > 0) or (s_obs is not None) or (s_sim is not None)):
            try:
                win = _pick_plot_window(raw_idx, s_obs.index if s_obs is not None else None)
                if win is None:
                    print(f"[WARN] {basin}: could not pick plot window; skipping.")
                    continue
                win_start, win_end = win

                fig = plt.figure(figsize=(12, 6))
                # Raw flow
                if raw_df is not None:
                    # Prefer Flow_m3_sec if present
                    plot_flow_col = "Flow_m3_sec" if "Flow_m3_sec" in raw_df.columns else (raw_df.select_dtypes(include=[np.number]).columns.tolist()[0] if len(raw_df.select_dtypes(include=[np.number]).columns) else None)
                    if plot_flow_col is not None:
                        s_raw_plot = raw_df[plot_flow_col].loc[win_start:win_end]
                        plt.plot(s_raw_plot.index, s_raw_plot.values, label="Raw Flow", alpha=0.6)
                # Eval Obs
                if s_obs is not None and len(s_obs):
                    s_obs_win = s_obs.loc[win_start:win_end]
                    if len(s_obs_win):
                        plt.plot(s_obs_win.index, s_obs_win.values, label="Eval Obs", linestyle="--")
                # Eval Sim
                if s_sim is not None and len(s_sim):
                    s_sim_win = s_sim.loc[win_start:win_end]
                    if len(s_sim_win):
                        plt.plot(s_sim_win.index, s_sim_win.values, label="Eval Sim", linestyle=":")

                plt.title(f"{basin} raw vs outputs ({win_start.date()}..{win_end.date()})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                out_png = diag_dir / f"{basin}_raw_vs_outputs.png"
                plt.savefig(out_png)
                plt.close(fig)
                plots_saved += 1
            except Exception as e:
                print(f"[WARN] Plotting mismatch for {basin} failed: {e}")

    df = pd.DataFrame(rows)
    out_csv = diag_dir / OUT_CSV
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote diagnosis summary to {out_csv}")
    print(f"[OK] Saved {plots_saved} plot(s) under {diag_dir}")

    # Quick console insights
    empty = df[df["outputs_empty"]]
    all_nan = df[df["all_sim_nan"]]
    freq_mis = df[df["freq_mismatch"]]
    print(f"[INSPECT] outputs_empty basins: {len(empty)}; all_sim_nan basins: {len(all_nan)}; freq_mismatch basins: {len(freq_mis)}")
    print("[HINT] If raw_not_in_outputs_* > 0 while raw_n >> 0, check period overrides (test vs validation) and data_dir alignment.")

if __name__ == "__main__":
    main()
