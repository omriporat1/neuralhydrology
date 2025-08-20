from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.utils.config import Config
import xarray
from datetime import datetime
import yaml

# --- Added helper metrics (Multiple-epochs compatible) ---
def nse(obs, sim):
	obs = np.asarray(obs)
	sim = np.asarray(sim)
	mean_obs = np.nanmean(obs)
	numerator = np.nansum((obs - sim) ** 2)
	denominator = np.nansum((obs - mean_obs) ** 2)
	return 1 - (numerator / denominator if denominator != 0 else np.nan)

def persistent_nse(obs, sim, lag_steps=18):
	obs = np.asarray(obs)
	sim = np.asarray(sim)
	persistence = np.roll(obs, lag_steps)
	if len(obs) > 0:
		persistence[:lag_steps] = obs[0]
	num = np.nansum((obs - sim) ** 2)
	den = np.nansum((obs - persistence) ** 2)
	return 1 - (num / den if den != 0 else np.nan)

def peak_flow_error(obs, sim):
	obs = np.asarray(obs)
	sim = np.asarray(sim)
	peak_obs = np.nanmax(obs) if obs.size else np.nan
	return (np.nanmax(sim) - peak_obs) / peak_obs if (peak_obs not in [0, np.nan]) else np.nan

def volume_error(obs, sim):
	obs = np.asarray(obs)
	sim = np.asarray(sim)
	vol_obs = np.nansum(obs)
	return (np.nansum(sim) - vol_obs) / vol_obs if vol_obs != 0 else np.nan

# --- New: lightweight diagnostics helpers ---
def _series_stats(da):
	try:
		values = np.asarray(da).flatten()
		n = values.size
		n_nan = int(np.isnan(values).sum())
		date_min = str(da['date'].values.min()) if ('date' in da.coords and n > 0) else None
		date_max = str(da['date'].values.max()) if ('date' in da.coords and n > 0) else None
		return {"n": int(n), "n_nan": n_nan, "date_min": date_min, "date_max": date_max}
	except Exception:
		return {"n": 0, "n_nan": 0, "date_min": None, "date_max": None}

# --- New: evaluation cache helpers ---
def _load_cached_results(cache_dir: Path):
	results = {}
	for nc in cache_dir.glob("*.nc"):
		try:
			basin = nc.stem.replace("__10min", "")
			ds = xarray.load_dataset(nc)
			results[basin] = {"10min": {"xr": ds}}
		except Exception as e:
			print(f"[WARN] Failed loading cache file {nc}: {e}")
	return results

def _save_results_to_cache(results: dict, cache_dir: Path):
	cache_dir.mkdir(exist_ok=True)
	for basin, by_scale in results.items():
		try:
			ds = by_scale["10min"]["xr"]
			if isinstance(ds, xarray.DataArray):
				ds = ds.to_dataset(name="Flow_m3_sec")
			out = cache_dir / f"{basin}__10min.nc"
			ds.to_netcdf(out)
		except Exception as e:
			print(f"[WARN] Could not cache results for basin {basin}: {e}")

def main():
    job_dir = Path(r"C:\PhD\Python\neuralhydrology\Experiments\Calculate_mean_basin_rain_from_gauges\Train_single_model\results\job_0")
    
    run_dir = job_dir / f"run_014_av_rain"
    print(f"\nProcessing run directory: {run_dir}")
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}")
        return
    
    run_subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not run_subdirs:
        print(f"No run subdirectories found in {run_dir}")
        return
    
    run_dir = run_subdirs[0]
    print(f"Using run directory: {run_dir}")
    
    max_events_path = Path("C:/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv")
    delay = 18  # 3 hours = 18 10-minute steps
    
    print(f"Checking if run directory exists: {run_dir.exists()}")
    print(f"Checking if max events file exists: {max_events_path.exists()}")
    
    if not run_dir.exists():
        print(f"Error: Run directory not found at {run_dir}")
        return
        
    if not max_events_path.exists():
        print(f"Error: Max events file not found at {max_events_path}")
        return

    LOCAL_BASIN_PATH = Path("C:/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static")
    
    # create a dictionary with the timerange of the events to be extracted - from day before max of each year to day after max of each year, for each basin, unless it's zero:
    max_events_df = pd.read_csv(max_events_path)
    max_events_df = max_events_df[max_events_df['max_discharge'] > 0]  # Filter out zero discharge events
    max_events_df['max_date'] = pd.to_datetime(max_events_df['max_date'], format='mixed', dayfirst=True)
    max_events_df['start_date'] = max_events_df['max_date'] - pd.Timedelta(days=1)
    max_events_df['end_date'] = max_events_df['max_date'] + pd.Timedelta(days=2)
    max_events_df = max_events_df.set_index('basin')

    # Process the single run directory
    config_path = run_dir / "config.yml"
    print(f"Checking if config file exists: {config_path.exists()}")
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    RUN_LOCALLY = True  # Set to False when running on HPC/original environment

    if RUN_LOCALLY:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Modify paths in the dictionary
        config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan/Caravan_winter"))
        config_dict['device'] = 'cpu'
        
        # IMPORTANT: Set the run_dir to the absolute local path
        config_dict['run_dir'] = str(run_dir.absolute())
        
        # Update basin files if they exist
        
        if 'test_basin_file' in config_dict and config_dict['test_basin_file']:
            basin_filename = Path(config_dict['test_basin_file']).name
            config_dict['test_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)
        
        if 'train_basin_file' in config_dict and config_dict['train_basin_file']:
            basin_filename = Path(config_dict['train_basin_file']).name
            config_dict['train_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)
        
        if 'validation_basin_file' in config_dict and config_dict['validation_basin_file']:
            basin_filename = Path(config_dict['validation_basin_file']).name
            config_dict['validation_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)

        # try to change the test start and end dates to those of the validation period:
        if 'test_start_date' in config_dict and 'test_end_date' in config_dict:
            config_dict['test_start_date'] = config_dict['validation_start_date']
            config_dict['test_end_date'] = config_dict['validation_end_date']


        # Create new config from modified dictionary
        config = Config(config_dict)
    else:
        # If running on HPC, use the original config file
        config = Config(config_path)
    
    
    # New: print configured eval periods for context
    print(f"Configured periods -> "
          f"train: {getattr(config, 'train_start_date', None)}..{getattr(config, 'train_end_date', None)}, "
          f"validation: {getattr(config, 'validation_start_date', None)}..{getattr(config, 'validation_end_date', None)}, "
          f"test: {getattr(config, 'test_start_date', None)}..{getattr(config, 'test_end_date', None)}")

    # --- New: cache-aware evaluation ---
    USE_CACHE = True
    cache_dir = run_dir / "cached_evaluation_10min"

    results = None
    if USE_CACHE and cache_dir.exists() and any(cache_dir.glob("*.nc")):
        print(f"[CACHE] Loading evaluation results from {cache_dir}")
        results = _load_cached_results(cache_dir)

    if results is None or len(results) == 0:
        print("[CACHE] No valid cache found. Running evaluation...")
        tester = get_tester(cfg=config, run_dir=run_dir, period="test", init_model=True)
        results = tester.evaluate(save_results=False, metrics=config.metrics)
        # Save cache for next runs
        try:
            _save_results_to_cache(results, cache_dir)
            print(f"[CACHE] Saved evaluation cache to {cache_dir}")
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")

    basins = results.keys()

    validation_hydrographs_dir = run_dir / "validation_hydrographs_and_analysis"
    validation_hydrographs_dir.mkdir(exist_ok=True)

    # New: collect basin-level diagnostics
    debug_rows = []

    for basin in basins:
        try:
            # Basin key matching logic
            if basin in max_events_df.index:
                basin_key = basin
            else:
                basin_id = basin.split('_')[-1] if '_' in basin else basin
                if int(basin_id) in max_events_df.index:
                    basin_key = int(basin_id)
                else:
                    # No matching events row: still create full-period outputs; metrics recorded later as erroneous
                    basin_key = None

            if basin_key is not None:
                basin_events = max_events_df.loc[basin_key]
                if isinstance(basin_events, pd.Series):
                    basin_events = pd.DataFrame([basin_events])
            else:
                basin_events = pd.DataFrame(columns=["start_date", "end_date", "max_date", "max_discharge"])

            basin_results = results[basin]["10min"]["xr"]
            qobs = basin_results["Flow_m3_sec_obs"]
            qsim = basin_results["Flow_m3_sec_sim"]
            if 'time_step' in qobs.dims:
                qobs = qobs.isel(time_step=-1)
            if 'time_step' in qsim.dims:
                qsim = qsim.isel(time_step=-1)

            # New: record diagnostics for this basin (before any plotting)
            obs_stats = _series_stats(qobs)
            sim_stats = _series_stats(qsim)
            debug_rows.append({
                "basin": basin,
                "obs_n": obs_stats["n"], "obs_nan": obs_stats["n_nan"],
                "sim_n": sim_stats["n"], "sim_nan": sim_stats["n_nan"],
                "date_min": obs_stats["date_min"], "date_max": obs_stats["date_max"],
                "all_sim_nan": bool(sim_stats["n"] > 0 and sim_stats["n"] == sim_stats["n_nan"])
            })
            print(f"[DEBUG] {basin}: obs_n={obs_stats['n']} sim_n={sim_stats['n']} "
                  f"obs_nan={obs_stats['n_nan']} sim_nan={sim_stats['n_nan']} "
                  f"date_range=[{obs_stats['date_min']}..{obs_stats['date_max']}]")

            # If there is no data, skip creating empty outputs
            if obs_stats["n"] == 0 or sim_stats["n"] == 0:
                print(f"[WARN] {basin}: Empty series (obs_n={obs_stats['n']}, sim_n={sim_stats['n']}). Skipping files.")
                continue
            if sim_stats["n"] == sim_stats["n_nan"]:
                print(f"[WARN] {basin}: All simulated values are NaN. Skipping files.")
                continue

            # Shifted obs for plotting
            try:
                fill_value = qobs.isel(date=0).item() if 'date' in qobs.dims and obs_stats["n"] > 0 else np.nan
            except Exception:
                fill_value = np.nan
            qobs_shift = qobs.shift(date=delay, fill_value=fill_value)

            # Per-event CSVs/plots (3-day window)
            for _, event in basin_events.iterrows():
                start_date = event['start_date']
                end_date = event['end_date']
                try:
                    qobs_event = qobs.sel(date=slice(start_date, end_date))
                    qsim_event = qsim.sel(date=slice(start_date, end_date))
                    qobs_shift_event = qobs_shift.sel(date=slice(start_date, end_date))
                except Exception as e:
                    print(f"[WARN] {basin}: slice {start_date}..{end_date} failed: {e}")
                    continue

                n_ev = int(np.asarray(qobs_event).size)
                n_ev_sim = int(np.asarray(qsim_event).size)
                if n_ev == 0 or n_ev_sim == 0 or np.all(np.isnan(np.asarray(qsim_event))):
                    print(f"[WARN] {basin}: No overlap or all-NaN for event {start_date}..{end_date}. Skipping files.")
                    continue

                event_df = pd.DataFrame({
                    'date': qobs_event['date'].values,
                    'observed': np.asarray(qobs_event).flatten(),
                    'simulated': np.asarray(qsim_event).flatten(),
                    'shifted': np.asarray(qobs_shift_event).flatten(),
                    'event_date': event['max_date'],
                    'event_discharge': event['max_discharge']
                })

                event_str = f"{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}"
                csv_file = validation_hydrographs_dir / f"{basin}_event_{event_str}.csv"
                event_df.to_csv(csv_file, index=False)
                print(f"Saved event hydrograph for basin {basin} ({event_str}) to {csv_file}")

                plt.figure(figsize=(12, 8))
                plt.plot(event_df['date'], event_df['observed'], label="Observed")
                plt.plot(event_df['date'], event_df['simulated'], linestyle='--', label="Simulated")
                plt.plot(event_df['date'], event_df['shifted'], linestyle=':', label="Observed shifted (3 hours)")
                plt.title(f"Event Hydrograph for Basin {basin}\n{event_str}")
                plt.xlabel("Date")
                plt.ylabel("Discharge (m³/s)")
                plt.grid(True)
                plt.legend()
                fig_file = validation_hydrographs_dir / f"{basin}_event_{event_str}.png"
                plt.savefig(fig_file)
                plt.close()

            # Full validation-period CSV/plot
            try:
                full_df = pd.DataFrame({
                    'date': qobs['date'].values,
                    'observed': np.asarray(qobs).flatten(),
                    'simulated': np.asarray(qsim).flatten(),
                    'shifted': np.asarray(qobs_shift).flatten()
                })
                full_csv_file = validation_hydrographs_dir / f"{basin}_full_period.csv"
                full_df.to_csv(full_csv_file, index=False)
                print(f"Saved full period data for basin {basin} to {full_csv_file}")

                plt.figure(figsize=(16, 10))
                plt.plot(full_df['date'], full_df['observed'], label="Observed")
                plt.plot(full_df['date'], full_df['simulated'], label="Simulated", linestyle='--')
                plt.plot(full_df['date'], full_df['shifted'], label="Observed shifted (3 hours)")
                # Highlight max-event windows if present
                for idx_ev, (_, event) in enumerate(basin_events.iterrows()):
                    event_start = event['start_date']
                    event_end = event['end_date']
                    event_max = event['max_date']
                    plt.axvspan(event_start, event_end, alpha=0.2, color='yellow')
                    max_discharge = event['max_discharge']
                    plt.scatter([event_max], [max_discharge], color='red', s=100, zorder=5,
								label=f"Max Discharge ({pd.to_datetime(event_max).strftime('%Y-%m-%d')})" if idx_ev == 0 else "")
                # Annotate NSE only (no RMSE)
                try:
                    obs_array = full_df['observed'].values
                    sim_array = full_df['simulated'].values
                    if not np.isnan(obs_array).all() and not np.isnan(sim_array).all():
                        nse_val = nse(obs_array, sim_array)
                        plt.text(0.02, 0.95, f"NSE: {nse_val:.3f}", transform=plt.gca().transAxes)
                except Exception as e:
                    print(f"Error calculating NSE for basin {basin}: {str(e)}")

                plt.title(f"Full Validation Period for Basin {basin}")
                plt.xlabel("Date")
                plt.ylabel("Discharge (m³/s)")
                plt.grid(True)
                plt.legend()
                plt.savefig(validation_hydrographs_dir / f"{basin}_full_period.png")
                plt.close()
            except Exception as e:
                print(f"Error creating full period visualization for basin {basin}: {str(e)}")
        except Exception as e:
            print(f"Error processing basin {basin}: {str(e)}")
            continue

    # New: save basin diagnostics
    if debug_rows:
        dbg_df = pd.DataFrame(debug_rows)
        dbg_csv = validation_hydrographs_dir / "basin_debug_summary.csv"
        dbg_df.to_csv(dbg_csv, index=False)
        print(f"Saved basin diagnostics to {dbg_csv}")

    # --- Unified metrics collection and saving (Multiple-epochs style) ---
    metrics_rows = []
    for basin in basins:
        try:
            basin_results = results[basin]["10min"]["xr"]
            qobs = basin_results["Flow_m3_sec_obs"]
            qsim = basin_results["Flow_m3_sec_sim"]
            if 'time_step' in qobs.dims:
                qobs = qobs.isel(time_step=-1)
            if 'time_step' in qsim.dims:
                qsim = qsim.isel(time_step=-1)

            # Full period metrics
            obs_array = np.asarray(qobs).flatten()
            sim_array = np.asarray(qsim).flatten()
            if obs_array.size == 0 or sim_array.size == 0 or np.all(np.isnan(sim_array)):
                metrics_rows.append({
                    "run": run_dir.name,
                    "basin": basin,
                    "event": "full_period",
                    "NSE": np.nan,
                    "pNSE": np.nan,
                    "PeakFlowError": np.nan,
                    "VolumeError": np.nan,
                    "erroneous": True
                })
            else:
                metrics_rows.append({
                    "run": run_dir.name,
                    "basin": basin,
                    "event": "full_period",
                    "NSE": nse(obs_array, sim_array),
                    "pNSE": persistent_nse(obs_array, sim_array, lag_steps=delay),
                    "PeakFlowError": peak_flow_error(obs_array, sim_array),
                    "VolumeError": volume_error(obs_array, sim_array),
                    "erroneous": False
                })

            # Per-event metrics (3-day windows)
            if basin in max_events_df.index:
                basin_key = basin
            else:
                basin_id = basin.split('_')[-1] if '_' in basin else basin
                try:
                    basin_key = int(basin_id)
                except Exception:
                    basin_key = None

            if basin_key is None or basin_key not in max_events_df.index:
                # No events mapping for this basin
                metrics_rows.append({
                    "run": run_dir.name,
                    "basin": basin,
                    "event": "no_events",
                    "NSE": np.nan,
                    "pNSE": np.nan,
                    "PeakFlowError": np.nan,
                    "VolumeError": np.nan,
                    "erroneous": True
                })
                continue

            basin_events = max_events_df.loc[basin_key]
            if isinstance(basin_events, pd.Series):
                basin_events = pd.DataFrame([basin_events])
            # Fix potential typo if triggered
            if isinstance(basin_events, pd.Series):
                basin_events = pd.DataFrame([basin_events])

            for _, event in basin_events.iterrows():
                start_date = event['start_date']
                end_date = event['end_date']
                try:
                    qobs_event = qobs.sel(date=slice(start_date, end_date))
                    qsim_event = qsim.sel(date=slice(start_date, end_date))
                    obs_event = np.asarray(qobs_event).flatten()
                    sim_event = np.asarray(qsim_event).flatten()
                    if obs_event.size == 0 or sim_event.size == 0 or np.all(np.isnan(sim_event)):
                        row = {
							"run": run_dir.name,
							"basin": basin,
							"event": f"{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}",
							"NSE": np.nan,
							"pNSE": np.nan,
							"PeakFlowError": np.nan,
							"VolumeError": np.nan,
							"erroneous": True
						}
                    else:
                        row = {
							"run": run_dir.name,
							"basin": basin,
							"event": f"{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}",
							"NSE": nse(obs_event, sim_event),
							"pNSE": persistent_nse(obs_event, sim_event, lag_steps=delay),
							"PeakFlowError": peak_flow_error(obs_event, sim_event),
							"VolumeError": volume_error(obs_event, sim_event),
							"erroneous": False
						}
                    metrics_rows.append(row)
                except Exception as e:
                    print(f"Error calculating event metrics for basin {basin}, event {start_date}-{end_date}: {e}")
                    metrics_rows.append({
						"run": run_dir.name,
						"basin": basin,
						"event": f"{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}",
						"NSE": np.nan,
						"pNSE": np.nan,
						"PeakFlowError": np.nan,
						"VolumeError": np.nan,
						"erroneous": True
					})
        except Exception as e:
            print(f"Error calculating metrics for basin {basin}: {e}")
            metrics_rows.append({
				"run": run_dir.name,
				"basin": basin,
				"event": "full_period",
				"NSE": np.nan,
				"pNSE": np.nan,
				"PeakFlowError": np.nan,
				"VolumeError": np.nan,
				"erroneous": True
			})
            continue

    # Save unified per-event metrics
    metrics_df = pd.DataFrame(metrics_rows)
    if "erroneous" not in metrics_df.columns:
        metrics_df["erroneous"] = False
    event_metrics_csv = validation_hydrographs_dir / "event_metrics.csv"
    metrics_df.to_csv(event_metrics_csv, index=False)
    print(f"Saved event metrics to {event_metrics_csv}")

    # Summary CSV with mean, median, std (exclude erroneous), plus erroneous_count
    if not metrics_df.empty:
        valid = metrics_df[~metrics_df["erroneous"]]
        if not valid.empty:
            to_agg = valid[["NSE", "pNSE", "PeakFlowError", "VolumeError"]]
            summary = to_agg.agg(["mean", "median", "std"])
            summary["erroneous_count"] = metrics_df["erroneous"].sum()
            summary_csv = validation_hydrographs_dir / "event_metrics_summary.csv"
            summary.to_csv(summary_csv)
            print(f"Saved event metrics summary to {summary_csv}")
        else:
            print("No valid rows for summary; all rows marked erroneous.")
    # ...existing code...

if __name__ == '__main__':
    try:
        print("Starting script execution...")
        main()
        print("Script execution completed.")
    except Exception as e:
        print(f"Error in main script execution: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"Error in main script execution: {str(e)}")
        import traceback
        traceback.print_exc()
