import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.utils.config import Config
import xarray
from datetime import datetime
import yaml



# %%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU


def main():
    # Accept run index and job id from command line
    run_id = int(sys.argv[1])
    job_id = int(41780893)

    job_dir = Path(f"/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results/job_{job_id}")
    run_dir = job_dir / f"run_{run_id:03d}"
    print(f"\nProcessing run directory: {run_dir}")
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}")
        return

    # Use specific run directory
    run_subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not run_subdirs:
        print(f"No run subdirectories found in {run_dir}")
        return

    # Use the first run subdirectory
    run_dir = run_subdirs[0]
    print(f"Using run directory: {run_dir}")

    max_events_path = Path("/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv")
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

    max_events_df = pd.read_csv(max_events_path)
    max_events_df = max_events_df[max_events_df['max_discharge'] > 0]
    max_events_df['max_date'] = pd.to_datetime(max_events_df['max_date'])
    max_events_df['start_date'] = max_events_df['max_date'] - pd.Timedelta(days=1)
    max_events_df['end_date'] = max_events_df['max_date'] + pd.Timedelta(days=1)
    max_events_df = max_events_df.set_index('basin')

    config_path = run_dir / "config.yml"
    print(f"Checking if config file exists: {config_path.exists()}")

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    RUN_LOCALLY = False  # Set to False when running on HPC/original environment

    if RUN_LOCALLY:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan"))
        config_dict['device'] = 'cpu'
        config_dict['run_dir'] = str(run_dir.absolute())

        if 'test_basin_file' in config_dict and config_dict['test_basin_file']:
            basin_filename = Path(config_dict['test_basin_file']).name
            config_dict['test_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)

        if 'train_basin_file' in config_dict and config_dict['train_basin_file']:
            basin_filename = Path(config_dict['train_basin_file']).name
            config_dict['train_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)

        if 'validation_basin_file' in config_dict and config_dict['validation_basin_file']:
            basin_filename = Path(config_dict['validation_basin_file']).name
            config_dict['validation_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)

        if 'test_start_date' in config_dict and 'test_end_date' in config_dict:
            config_dict['test_start_date'] = config_dict['validation_start_date']
            config_dict['test_end_date'] = config_dict['validation_end_date']

        config = Config(config_dict)
    else:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        if 'test_start_date' in config_dict and 'test_end_date' in config_dict:
            config_dict['test_start_date'] = config_dict['validation_start_date']
            config_dict['test_end_date'] = config_dict['validation_end_date']
        config = Config(config_dict)

    tester = get_tester(cfg=config, run_dir=run_dir, period="test", init_model=True)
    results = tester.evaluate(save_results=False, metrics=config.metrics)
    basins = results.keys()

    validation_hydrographs_dir = run_dir / "validation_hydrographs_cluster"
    validation_hydrographs_dir.mkdir(exist_ok=True)

    for basin in basins:
        try:
            if basin in max_events_df.index:
                basin_key = basin
            else:
                basin_id = basin.split('_')[-1] if '_' in basin else basin
                if int(basin_id) in max_events_df.index:
                    basin_key = int(basin_id)

            basin_events = max_events_df.loc[basin_key]
            if isinstance(basin_events, pd.Series):
                basin_events = pd.DataFrame([basin_events])

            basin_results = results[basin]["10min"]["xr"]
            qobs = basin_results["Flow_m3_sec_obs"]
            qsim = basin_results["Flow_m3_sec_sim"]
            if 'time_step' in qobs.dims:
                qobs = qobs.isel(time_step=-1)
            if 'time_step' in qsim.dims:
                qsim = qsim.isel(time_step=-1)

            fill_value = qobs.isel(date=0).item() if 'date' in qobs.dims else qobs[0].item()
            qobs_shift = qobs.shift(date=delay, fill_value=fill_value)

            for _, event in basin_events.iterrows():
                start_date = event['start_date']
                end_date = event['end_date']
                try:
                    qobs_event = qobs.sel(date=slice(start_date, end_date))
                    qsim_event = qsim.sel(date=slice(start_date, end_date))
                    qobs_shift_event = qobs_shift.sel(date=slice(start_date, end_date))
                except Exception as e:
                    print(f"Date range {start_date} to {end_date} not available for basin {basin}: {e}")
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
                for _, event in basin_events.iterrows():
                    event_start = event['start_date']
                    event_end = event['end_date']
                    event_max = event['max_date']
                    plt.axvspan(event_start, event_end, alpha=0.2, color='yellow')
                    max_discharge = event['max_discharge']
                    plt.scatter([event_max], [max_discharge], color='red', s=100, zorder=5, label=f"Max Discharge ({pd.to_datetime(event_max).strftime('%Y-%m-%d')})" if _ == 0 else "")
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

    # --- Metrics collection and saving ---
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

            obs_array = np.asarray(qobs).flatten()
            sim_array = np.asarray(qsim).flatten()
            mse_full = np.nanmean((obs_array - sim_array) ** 2)
            mean_obs = np.nanmean(obs_array)
            numerator = np.nansum((obs_array - sim_array) ** 2)
            denominator = np.nansum((obs_array - mean_obs) ** 2)
            nse_full = 1 - (numerator / denominator if denominator != 0 else np.nan)
            metrics_rows.append({
                "run": run_dir.name,
                "basin": basin,
                "event": "full_period",
                "MSE": mse_full,
                "NSE": nse_full
            })

            basin_key = basin if basin in max_events_df.index else int(basin.split('_')[-1])
            basin_events = max_events_df.loc[basin_key]
            if isinstance(basin_events, pd.Series):
                basin_events = pd.DataFrame([basin_events])

            event_mse_list = []
            event_nse_list = []
            for _, event in basin_events.iterrows():
                start_date = event['start_date']
                end_date = event['end_date']
                try:
                    qobs_event = qobs.sel(date=slice(start_date, end_date))
                    qsim_event = qsim.sel(date=slice(start_date, end_date))
                    obs_event = np.asarray(qobs_event).flatten()
                    sim_event = np.asarray(qsim_event).flatten()
                    mse_event = np.nanmean((obs_event - sim_event) ** 2)
                    mean_obs_event = np.nanmean(obs_event)
                    numerator_event = np.nansum((obs_event - sim_event) ** 2)
                    denominator_event = np.nansum((obs_event - mean_obs_event) ** 2)
                    nse_event = 1 - (numerator_event / denominator_event if denominator_event != 0 else np.nan)
                    metrics_rows.append({
                        "run": run_dir.name,
                        "basin": basin,
                        "event": f"{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}",
                        "MSE": mse_event,
                        "NSE": nse_event
                    })
                    event_mse_list.append(mse_event)
                    event_nse_list.append(nse_event)
                except Exception as e:
                    print(f"Error calculating event metrics for basin {basin}, event {start_date}-{end_date}: {e}")
                    continue

            if event_mse_list and event_nse_list:
                metrics_rows.append({
                    "run": run_dir.name,
                    "basin": basin,
                    "event": "events_average",
                    "MSE": np.nanmean(event_mse_list),
                    "NSE": np.nanmean(event_nse_list)
                })
        except Exception as e:
            print(f"Error calculating metrics for basin {basin}: {e}")
            continue

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_file = validation_hydrographs_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv_file, index=False)
    print(f"Saved metrics summary to {metrics_csv_file}")

if __name__ == '__main__':
    try:
        print("Starting script execution...")
        main()
        print("Script execution completed.")
    except Exception as e:
        print(f"Error in main script execution: {str(e)}")
        import traceback
        traceback.print_exc()
