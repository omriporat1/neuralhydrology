import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config
import yaml
import re
import traceback

def nse(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    mean_obs = np.nanmean(obs)
    numerator = np.nansum((obs - sim) ** 2)
    denominator = np.nansum((obs - mean_obs) ** 2)
    return 1 - (numerator / denominator if denominator != 0 else np.nan)

def peak_flow_error(obs, sim):
    peak_obs = np.nanmax(obs)
    return (np.nanmax(sim) - peak_obs) / peak_obs if peak_obs != 0 else np.nan

def volume_error(obs, sim):
    vol_obs = np.nansum(obs)
    return (np.nansum(sim) - vol_obs) / vol_obs if vol_obs != 0 else np.nan

def persistent_nse(obs, sim, lag_steps=18):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    persistence = np.roll(obs, lag_steps)
    persistence[:lag_steps] = obs[0]  # handle edge
    num = np.nansum((obs - sim) ** 2)
    denom = np.nansum((obs - persistence) ** 2)
    return 1 - (num / denom if denom != 0 else np.nan)

def main():
    run_id = int(sys.argv[1])
    job_id = int(41780893)

    job_dir = Path(f"/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results/job_{job_id}")
    run_dir = job_dir / f"run_{run_id:03d}"
    print(f"\nProcessing run directory: {run_dir}")
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}")
        return

    # Find the actual run subdirectory
    run_subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not run_subdirs:
        print(f"No run subdirectories found in {run_dir}")
        return
    run_dir = run_subdirs[0]
    print(f"Using run directory: {run_dir}")

    max_events_path = Path("/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv")
    delay = 18

    print(f"Checking if run directory exists: {run_dir.exists()}")
    print(f"Checking if max events file exists: {max_events_path.exists()}")

    if not run_dir.exists():
        print(f"Error: Run directory not found at {run_dir}")
        return

    if not max_events_path.exists():
        print(f"Error: Max events file not found at {max_events_path}")
        return

    new_basin_filename = "il_basins_high_qual_0_04_N35.txt"
    basin_filename = new_basin_filename

    max_events_df = pd.read_csv(max_events_path)
    max_events_df = max_events_df[max_events_df['max_discharge'] > 0]
    max_events_df['max_date'] = pd.to_datetime(max_events_df['max_date'], dayfirst=True)
    max_events_df['start_date'] = max_events_df['max_date'] - pd.Timedelta(days=1)
    max_events_df['end_date'] = max_events_df['max_date'] + pd.Timedelta(days=1)
    max_events_df = max_events_df.set_index('basin')

    config_path = run_dir / "config.yml"
    print(f"Checking if config file exists: {config_path.exists()}")

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    # Load config for cluster environment
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    remote_basin_path = Path(config_dict['test_basin_file'])
    basin_dir = remote_basin_path.parent
    config_dict['test_basin_file'] = str(basin_dir / basin_filename)
    config_dict['train_basin_file'] = str(basin_dir / basin_filename)
    config_dict['validation_basin_file'] = str(basin_dir / basin_filename)

    if 'test_start_date' in config_dict and 'test_end_date' in config_dict:
        config_dict['test_start_date'] = config_dict['validation_start_date']
        config_dict['test_end_date'] = config_dict['validation_end_date']
    config = Config(config_dict)

    # Find only the LATEST epoch model file for testing
    epoch_files = sorted(run_dir.glob("model_epoch*.pt"))
    if not epoch_files:
        print("No epoch files found!")
        return
    
    # Take only the last epoch for testing
    latest_epoch_file = epoch_files[-1]
    epoch = int(latest_epoch_file.stem.split("epoch")[-1])
    
    print(f"TESTING: Processing only latest epoch {epoch} ({latest_epoch_file.name})")

    # Create output folder for this epoch
    epoch_out_dir = run_dir / f"test_validation_epoch_{epoch:03d}"
    epoch_out_dir.mkdir(exist_ok=True)

    # Evaluate this epoch
    tester = get_tester(cfg=config, run_dir=run_dir, period="test", init_model=True)
    results = tester.evaluate(epoch=epoch, save_results=False, metrics=config.metrics)
    basins = results.keys()
    print(f"Number of basins in results: {len(basins)}")
    print(f"Basin names: {list(basins)[:5]}...")

    metrics_rows = []
    processed_basins = 0
    skipped_basins = 0
    
    # Process only first 3 basins for testing
    test_basins = list(basins)[:3]
    print(f"TESTING: Processing only first 3 basins: {test_basins}")
    
    for basin in test_basins:
        try:
            print(f"DEBUG: Processing basin: {basin}")
            # Find matching events for this basin
            if basin in max_events_df.index:
                basin_key = basin
            else:
                basin_id = basin.split('_')[-1] if '_' in basin else basin
                try:
                    basin_key = int(basin_id)
                except Exception:
                    print(f"Could not parse basin id for {basin}, skipping.")
                    skipped_basins += 1
                    continue
                if basin_key not in max_events_df.index:
                    print(f"Basin {basin} not found in max_events_df")
                    skipped_basins += 1
                    continue

            print(f"DEBUG: Basin key found: {basin_key}")
            basin_events = max_events_df.loc[basin_key]
            if isinstance(basin_events, pd.Series):
                basin_events = pd.DataFrame([basin_events])

            basin_results = results[basin]["10min"]["xr"]
            qobs = basin_results["Flow_m3_sec_obs"]
            qsim = basin_results["Flow_m3_sec_sim"]
            print(f"DEBUG: qobs shape: {qobs.shape}, qsim shape: {qsim.shape}")
            
            if 'time_step' in qobs.dims:
                qobs = qobs.isel(time_step=-1)
            if 'time_step' in qsim.dims:
                qsim = qsim.isel(time_step=-1)

            # Check for empty arrays or all NaN
            if len(qobs) == 0 or len(qsim) == 0:
                print(f"DEBUG: Basin {basin}: qobs or qsim is empty, skipping.")
                skipped_basins += 1
                continue
                
            obs_array = np.asarray(qobs).flatten()
            sim_array = np.asarray(qsim).flatten()
            print(f"DEBUG: obs_array shape: {obs_array.shape}, sim_array shape: {sim_array.shape}")
            print(f"DEBUG: obs_array nan count: {np.isnan(obs_array).sum()}, sim_array nan count: {np.isnan(sim_array).sum()}")
            
            if np.all(np.isnan(sim_array)):
                print(f"DEBUG: Basin {basin}: All simulated values are NaN, skipping.")
                skipped_basins += 1
                continue

            processed_basins += 1
            print(f"DEBUG: Basin {basin}: Processing data successfully")

            # Process events for this basin
            print(f"DEBUG: Processing {len(basin_events)} events for basin {basin}")

            fill_value = qobs.isel(date=0).item() if 'date' in qobs.dims else qobs[0].item()
            qobs_shift = qobs.shift(date=delay, fill_value=fill_value)

            # Process only first event for testing
            first_event = basin_events.iloc[0]
            start_date = first_event['start_date']
            end_date = first_event['end_date']
            print(f"DEBUG: Processing ONLY first event for basin {basin}: {start_date} to {end_date}")
            
            try:
                qobs_event = qobs.sel(date=slice(start_date, end_date))
                qsim_event = qsim.sel(date=slice(start_date, end_date))
                qobs_shift_event = qobs_shift.sel(date=slice(start_date, end_date))
                obs_event = np.asarray(qobs_event).flatten()
                sim_event = np.asarray(qsim_event).flatten()
                
                if len(obs_event) == 0 or len(sim_event) == 0 or np.all(np.isnan(sim_event)):
                    print(f"DEBUG: Event data invalid for basin {basin}, skipping event.")
                    continue
                    
                # Calculate metrics
                nse_event_val = nse(obs_event, sim_event)
                pnse_event = persistent_nse(obs_event, sim_event, lag_steps=delay)
                peak_err_event = peak_flow_error(obs_event, sim_event)
                vol_err_event = volume_error(obs_event, sim_event)
                
                metrics_rows.append({
                    "run": run_dir.name,
                    "basin": basin,
                    "event": f"{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}",
                    "NSE": nse_event_val,
                    "pNSE": pnse_event,
                    "PeakFlowError": peak_err_event,
                    "VolumeError": vol_err_event,
                    "erroneous": False
                })

                # Save event hydrograph and plot
                print(f"DEBUG: Saving event CSV and plot for basin {basin}, event {start_date}-{end_date}")
                event_df = pd.DataFrame({
                    'date': qobs_event['date'].values,
                    'observed': obs_event,
                    'simulated': sim_event,
                    'shifted': np.asarray(qobs_shift_event).flatten(),
                    'event_date': first_event['max_date'],
                    'event_discharge': first_event['max_discharge']
                })
                event_str = f"{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}"
                csv_file = epoch_out_dir / f"{basin}_event_{event_str}.csv"
                print(f"DEBUG: Saving event CSV to: {csv_file}")
                event_df.to_csv(csv_file, index=False)
                print(f"DEBUG: Event CSV saved successfully")
                
                plt.figure(figsize=(12, 8))
                plt.plot(event_df['date'], event_df['observed'], label="Observed")
                plt.plot(event_df['date'], event_df['simulated'], linestyle='--', label="Simulated")
                plt.plot(event_df['date'], event_df['shifted'], linestyle=':', label="Observed shifted (3 hours)")
                plt.title(f"Event Hydrograph for Basin {basin}\n{event_str}")
                plt.xlabel("Date")
                plt.ylabel("Discharge (m³/s)")
                plt.grid(True)
                plt.legend()
                fig_file = epoch_out_dir / f"{basin}_event_{event_str}.png"
                print(f"DEBUG: Saving event plot to: {fig_file}")
                plt.savefig(fig_file)
                plt.close()
                print(f"DEBUG: Event plot saved successfully")
                
            except Exception as e:
                print(f"DEBUG: Error processing event for basin {basin}: {e}")
                continue

            # Save full period hydrograph and plot
            print(f"DEBUG: Saving full period CSV and plot for basin {basin}")
            try:
                full_df = pd.DataFrame({
                    'date': qobs['date'].values,
                    'observed': obs_array,
                    'simulated': sim_array,
                    'shifted': np.asarray(qobs_shift).flatten()
                })
                full_csv_file = epoch_out_dir / f"{basin}_full_period.csv"
                print(f"DEBUG: Saving full period CSV to: {full_csv_file}")
                full_df.to_csv(full_csv_file, index=False)
                print(f"DEBUG: Full period CSV saved successfully")
                
                plt.figure(figsize=(16, 10))
                plt.plot(full_df['date'], full_df['observed'], label="Observed")
                plt.plot(full_df['date'], full_df['simulated'], label="Simulated", linestyle='--')
                plt.plot(full_df['date'], full_df['shifted'], label="Observed shifted (3 hours)")
                plt.title(f"Full Validation Period for Basin {basin}")
                plt.xlabel("Date")
                plt.ylabel("Discharge (m³/s)")
                plt.grid(True)
                plt.legend()
                plot_file = epoch_out_dir / f"{basin}_full_period.png"
                print(f"DEBUG: Saving full period plot to: {plot_file}")
                plt.savefig(plot_file)
                plt.close()
                print(f"DEBUG: Full period plot saved successfully")
            except Exception as e:
                print(f"DEBUG: Error creating full period visualization for basin {basin}: {str(e)}")
            
            print(f"DEBUG: Completed processing basin {basin} successfully")
            
        except Exception as e:
            print(f"DEBUG: Error processing basin {basin}: {str(e)}")
            traceback.print_exc()
            skipped_basins += 1
            continue

    print(f"TESTING SUMMARY:")
    print(f"- Processed basins: {processed_basins}")
    print(f"- Skipped basins: {skipped_basins}")
    print(f"- Metrics rows collected: {len(metrics_rows)}")
    
    # Check what files were actually created
    created_files = list(epoch_out_dir.glob("*"))
    print(f"- Files created in {epoch_out_dir}: {len(created_files)}")
    for f in created_files:
        print(f"  * {f.name}")

if __name__ == '__main__':
    try:
        print("Starting TESTING script execution...")
        main()
        print("TESTING script execution completed.")
    except Exception as e:
        print(f"Error in main script execution: {str(e)}")
        traceback.print_exc()
