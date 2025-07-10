import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import os
import xarray
from datetime import datetime
import yaml


def main():
    # Base directory for all runs
    experiments_dir = Path("C:/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results/job_41780893")
    max_events_path = Path("C:/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv")
    delay = 18  # 3 hours = 18 10-minute steps
    
    print(f"Checking experiments directory: {experiments_dir}")
    print(f"Checking max events file: {max_events_path}")
    
    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found at {experiments_dir}")
        return
        
    if not max_events_path.exists():
        print(f"Error: Max events file not found at {max_events_path}")
        return
    
    # Load max events data
    max_events_df = pd.read_csv(max_events_path)
    max_events_df = max_events_df[max_events_df['max_discharge'] > 0]  # Filter out zero discharge events
    max_events_df['max_date'] = pd.to_datetime(max_events_df['max_date'])
    max_events_df['start_date'] = max_events_df['max_date'] - pd.Timedelta(days=1)
    max_events_df['end_date'] = max_events_df['max_date'] + pd.Timedelta(days=1)
    max_events_df = max_events_df.set_index('basin')
    
    LOCAL_BASIN_PATH = Path("C:/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static")
    
    # Find all run_XXX directories
    run_dirs = [d for d in experiments_dir.glob("run_*") if d.is_dir()]
    print(f"Found {len(run_dirs)} run directories")
    
    # For testing, limit to just the first run directory
    # Comment this out to process all runs
    run_dirs = run_dirs[:1]
    print(f"Using first {len(run_dirs)} run directories for processing")
    
    # Process each run directory
    for run_base_dir in run_dirs:
        print(f"\nProcessing run directory: {run_base_dir}")
        
        # Find all subdirectories in the run directory (these contain the actual model runs)
        model_run_dirs = [d for d in run_base_dir.glob("*") if d.is_dir()]
        
        if not model_run_dirs:
            print(f"No model run subdirectories found in {run_base_dir}")
            continue
        
        # Use the first model run directory (which has a unique name like N38_A30_4CPU_SMI_0406_170648)
        run_dir = model_run_dirs[0]
        print(f"Using model run directory: {run_dir}")
        
        # Check if config.yml exists in this directory
        config_path = run_dir / "config.yml"
        if not config_path.exists():
            print(f"Config file not found at {config_path}, skipping this run")
            continue

        RUN_LOCALLY = True  # Set to False when running on HPC/original environment

        if RUN_LOCALLY:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Modify paths in the dictionary
            config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan"))
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
            
            # Create new config from modified dictionary
            config = Config(config_dict)
        else:
            # If running on HPC, use the original config file
            config = Config(config_path)
        
        try:
            print(f"Creating tester for {run_dir.name}...")
            tester = get_tester(cfg=config, run_dir=run_dir, period="validation", init_model=True)
            print(f"Evaluating model for {run_dir.name}...")
            results = tester.evaluate(save_results=False, metrics=config.metrics)
            print(f"Evaluation complete. Got results for {len(results.keys())} basins")
            basins = results.keys()

            # for each basin, create in the run directory a folder called "validation_hydrographs"
            # and save the hydrograph of the entire validation period in it,
            # in the format of csv: time column and observed and simulated values.
            # Also add a "shifted" column with the observed values shifted by 18 time steps (3 hours).
            validation_hydrographs_dir = run_dir / "validation_hydrographs"
            validation_hydrographs_dir.mkdir(exist_ok=True)
            
            print(f"Processing {len(basins)} basins for {run_dir.name}...")
            
            for basin in basins:
                try:
                    # Check if the basin exists directly
                    if basin in max_events_df.index:
                        basin_key = basin
                    else:
                        # Try to match with/without 'il_' prefix
                        if basin.startswith('il_') and basin[3:] in max_events_df.index:
                            basin_key = basin[3:]
                            print(f"Found basin {basin} as {basin_key} in max events data")
                        elif 'il_' + basin in max_events_df.index:
                            basin_key = 'il_' + basin
                            print(f"Found basin {basin} as {basin_key} in max events data")
                        else:
                            # Try to match by the numeric part (assuming format like 'prefix_number')
                            basin_id = basin.split('_')[-1] if '_' in basin else basin
                            matching_basins = [idx for idx in max_events_df.index 
                                    if idx.endswith(basin_id) or 
                                    (idx.split('_')[-1] if '_' in idx else idx) == basin_id]
                            
                            if matching_basins:
                                basin_key = matching_basins[0]
                                print(f"Matched basin {basin} to {basin_key} based on ID")
                            else:
                                print(f"Basin {basin} not found in max events data, skipping...")
                                continue
                        
                    # Get event dates for this basin
                    basin_events = max_events_df.loc[basin_key]
                    
                    # If there's only one event, convert to DataFrame with one row
                    if isinstance(basin_events, pd.Series):
                        basin_events = pd.DataFrame([basin_events])
                    
                    # Create dataframe to store all results
                    all_hydrographs = pd.DataFrame()
                    
                    # Process each maximum event for this basin
                    for _, event in basin_events.iterrows():
                        start_date = event['start_date']
                        end_date = event['end_date']
                        
                        # Extract observations and simulations for this event
                        basin_results = results[basin]["10min"]["xr"]
                        
                        # Get the discharge observations and simulations
                        # Try different column names based on normalization type
                        obs_columns = [col for col in basin_results if 'obs' in col and 'Flow' in col]
                        sim_columns = [col for col in basin_results if 'sim' in col and 'Flow' in col]
                        
                        if not obs_columns or not sim_columns:
                            print(f"Could not find observation or simulation columns for basin {basin}")
                            continue
                        
                        # Use the first available column
                        qobs = basin_results[obs_columns[0]]
                        qsim = basin_results[sim_columns[0]]
                        
                        # Filter to the event time range
                        try:
                            qobs_event = qobs.sel(date=slice(start_date, end_date))
                            qsim_event = qsim.sel(date=slice(start_date, end_date))
                        except KeyError:
                            print(f"Date range {start_date} to {end_date} not available for basin {basin}")
                            continue
                        
                        # Create shifted observations for lead time analysis
                        fill_value = qobs.isel(date=0, time_step=0).item()
                        qobs_shift = qobs.shift(date=delay, fill_value=fill_value)
                        qobs_shift_event = qobs_shift.sel(date=slice(start_date, end_date))
                        
                        # Create a DataFrame with date, observed, simulated, and shifted values
                        event_df = pd.DataFrame({
                            'date': qobs_event.date.values,
                            'observed': qobs_event.values,
                            'simulated': qsim_event.values,
                            'shifted': qobs_shift_event.values,
                            'event_date': event['max_date'],
                            'event_discharge': event['max_discharge']
                        })
                        
                        # Append to all hydrographs
                        all_hydrographs = pd.concat([all_hydrographs, event_df])
                    
                    # Save the combined hydrographs for this basin
                    if not all_hydrographs.empty:
                        csv_file = validation_hydrographs_dir / f"{basin}_validation_hydrographs.csv"
                        all_hydrographs.to_csv(csv_file, index=False)
                        print(f"Saved validation hydrographs for basin {basin} to {csv_file}")
                        
                        # Create visualization of the hydrographs
                        plt.figure(figsize=(12, 8))
                        for event_date in all_hydrographs['event_date'].unique():
                            event_data = all_hydrographs[all_hydrographs['event_date'] == event_date]
                            plt.plot(event_data['date'], event_data['observed'], label=f"Observed ({event_date.strftime('%Y-%m-%d')})")
                            plt.plot(event_data['date'], event_data['simulated'], linestyle='--', 
                                    label=f"Simulated ({event_date.strftime('%Y-%m-%d')})")
                            
                        plt.title(f"Validation Hydrographs for Basin {basin}")
                        plt.xlabel("Date")
                        plt.ylabel("Discharge (m³/s)")
                        plt.grid(True)
                        plt.legend()
                        
                        # Save the plot
                        plt.savefig(validation_hydrographs_dir / f"{basin}_validation_hydrographs.png")
                        plt.close()
                    
                    # Also save the entire evaluation period for this basin
                    try:
                        # Get the entire period data
                        basin_results = results[basin]["10min"]["xr"]
                        
                        # Get the discharge observations and simulations
                        obs_columns = [col for col in basin_results if 'obs' in col and 'Flow' in col]
                        sim_columns = [col for col in basin_results if 'sim' in col and 'Flow' in col]
                        
                        if obs_columns and sim_columns:
                            qobs_full = basin_results[obs_columns[0]]
                            qsim_full = basin_results[sim_columns[0]]
                            
                            # Create shifted observations for lead time analysis
                            fill_value = qobs_full.isel(date=0, time_step=0).item()
                            qobs_shift_full = qobs_full.shift(date=delay, fill_value=fill_value)
                            
                            # Create a DataFrame with all data
                            full_df = pd.DataFrame({
                                'date': qobs_full.date.values,
                                'observed': qobs_full.values,
                                'simulated': qsim_full.values,
                                'shifted': qobs_shift_full.values
                            })
                            
                            # Save the full period data to CSV
                            full_csv_file = validation_hydrographs_dir / f"{basin}_full_period.csv"
                            full_df.to_csv(full_csv_file, index=False)
                            print(f"Saved full period data for basin {basin} to {full_csv_file}")
                            
                            # Create visualization of the full period
                            plt.figure(figsize=(16, 10))
                            plt.plot(full_df['date'], full_df['observed'], label="Observed")
                            plt.plot(full_df['date'], full_df['simulated'], label="Simulated", linestyle='--')
                            plt.plot(full_df['date'], full_df['shifted'], label="Observed shifted (3 hours)")
                            
                            # Highlight the maximum event periods
                            for _, event in basin_events.iterrows():
                                event_start = event['start_date']
                                event_end = event['end_date']
                                event_max = event['max_date']
                                
                                # Add shaded rectangle for event period
                                plt.axvspan(event_start, event_end, alpha=0.2, color='yellow')
                                
                                # Mark the maximum discharge point
                                max_discharge = event['max_discharge']
                                plt.scatter([event_max], [max_discharge], color='red', s=100, zorder=5, 
                                           label=f"Max Discharge ({event_max.strftime('%Y-%m-%d')})" if _ == 0 else "")
                            
                            plt.title(f"Full Validation Period for Basin {basin}")
                            plt.xlabel("Date")
                            plt.ylabel("Discharge (m³/s)")
                            plt.grid(True)
                            plt.legend()
                            
                            # Add some statistics to the plot
                            # Calculate metrics for the entire period
                            metrics_full = {}
                            try:
                                # Calculate NSE and RMSE between observed and simulated
                                obs_array = full_df['observed'].values
                                sim_array = full_df['simulated'].values
                                shift_array = full_df['shifted'].values
                                
                                # NSE between observed and simulated
                                if not np.isnan(obs_array).all() and not np.isnan(sim_array).all():
                                    mean_obs = np.nanmean(obs_array)
                                    numerator = np.nansum((obs_array - sim_array) ** 2)
                                    denominator = np.nansum((obs_array - mean_obs) ** 2)
                                    nse = 1 - (numerator / denominator if denominator != 0 else np.nan)
                                    metrics_full['NSE'] = nse
                                    
                                    # RMSE between observed and simulated
                                    rmse = np.sqrt(np.nanmean((obs_array - sim_array) ** 2))
                                    metrics_full['RMSE'] = rmse
                                    
                                    # Add metrics to plot
                                    plt.text(0.02, 0.95, f"NSE: {metrics_full.get('NSE', 'N/A'):.3f}", transform=plt.gca().transAxes)
                                    plt.text(0.02, 0.91, f"RMSE: {metrics_full.get('RMSE', 'N/A'):.3f}", transform=plt.gca().transAxes)
                            except Exception as e:
                                print(f"Error calculating metrics for basin {basin}: {str(e)}")
                            
                            # Save the full period plot
                            plt.savefig(validation_hydrographs_dir / f"{basin}_full_period.png")
                            plt.close()
                    except Exception as e:
                        print(f"Error creating full period visualization for basin {basin}: {str(e)}")
                except Exception as e:
                    print(f"Error processing basin {basin}: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error evaluating model for run {run_dir.name}: {str(e)}")
            continue


if __name__ == '__main__':
    try:
        print("Starting script execution...")
        main()
        print("Script execution completed.")
    except Exception as e:
        print(f"Error in main script execution: {str(e)}")
        import traceback
        traceback.print_exc()
