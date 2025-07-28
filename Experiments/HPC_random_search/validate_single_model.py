import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from neuralhydrology.evaluation import get_tester
from neuralhydrology.utils.config import Config
import yaml
import traceback

def main():
    job_sub_id = 14  # Must match the training job
    job_id = 0       # Must match the training job
    
    # Path to the trained model
    job_dir = Path(f"results/job_{job_id}")
    run_dir = job_dir / f"run_{job_sub_id:03d}_a"
    
    print(f"Validating model in: {run_dir}")
    
    if not run_dir.exists():
        print(f"Error: Run directory not found at {run_dir}")
        return
    
    # Find the actual run subdirectory created by neuralhydrology
    run_subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not run_subdirs:
        print(f"No run subdirectories found in {run_dir}")
        return
    
    # Use the first (and should be only) run subdirectory
    actual_run_dir = run_subdirs[0]
    print(f"Using actual run directory: {actual_run_dir}")
    
    # Load max events data
    max_events_path = Path("/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv")
    delay = 18  # 3 hours = 18 10-minute steps
    
    if not max_events_path.exists():
        print(f"Error: Max events file not found at {max_events_path}")
        return
    
    # Load and process events data
    max_events_df = pd.read_csv(max_events_path)
    max_events_df = max_events_df[max_events_df['max_discharge'] > 0]
    max_events_df['max_date'] = pd.to_datetime(max_events_df['max_date'], dayfirst=True)
    max_events_df['start_date'] = max_events_df['max_date'] - pd.Timedelta(days=1)
    max_events_df['end_date'] = max_events_df['max_date'] + pd.Timedelta(days=1)
    max_events_df = max_events_df.set_index('basin')
    
    # Load config and modify for validation
    config_path = actual_run_dir / "config.yml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Modify config for validation period
    if 'test_start_date' in config_dict and 'test_end_date' in config_dict:
        config_dict['test_start_date'] = config_dict['validation_start_date']
        config_dict['test_end_date'] = config_dict['validation_end_date']
    
    config = Config(config_dict)
    
    # Get tester and evaluate
    print("Initializing tester...")
    tester = get_tester(cfg=config, run_dir=actual_run_dir, period="test", init_model=True)
    
    print("Running evaluation...")
    results = tester.evaluate(save_results=False, metrics=config.metrics)
    basins = results.keys()
    
    # Create output directory
    validation_hydrographs_dir = actual_run_dir / "validation_hydrographs_cluster"
    validation_hydrographs_dir.mkdir(exist_ok=True)
    
    print(f"Processing {len(basins)} basins...")
    
    # Process each basin
    metrics_rows = []
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
                    print(f"Basin {basin} not found in max events data")
                    continue
            
            basin_events = max_events_df.loc[basin_key]
            if isinstance(basin_events, pd.Series):
                basin_events = pd.DataFrame([basin_events])
            
            # Get basin results
            basin_results = results[basin]["10min"]["xr"]
            qobs = basin_results["Flow_m3_sec_obs"]
            qsim = basin_results["Flow_m3_sec_sim"]
            
            if 'time_step' in qobs.dims:
                qobs = qobs.isel(time_step=-1)
            if 'time_step' in qsim.dims:
                qsim = qsim.isel(time_step=-1)
            
            # Shifted obs for full period
            fill_value = qobs.isel(date=0).item() if 'date' in qobs.dims else qobs[0].item()
            qobs_shift = qobs.shift(date=delay, fill_value=fill_value)
            
            # Process each event
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
                
                # Create event dataframe
                event_df = pd.DataFrame({
                    'date': qobs_event['date'].values,
                    'observed': np.asarray(qobs_event).flatten(),
                    'simulated': np.asarray(qsim_event).flatten(),
                    'shifted': np.asarray(qobs_shift_event).flatten(),
                    'event_date': event['max_date'],
                    'event_discharge': event['max_discharge']
                })
                
                # Save event CSV
                event_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
                csv_file = validation_hydrographs_dir / f"{basin}_event_{event_str}.csv"
                event_df.to_csv(csv_file, index=False)
                
                # Calculate metrics for this event
                obs_vals = event_df['observed'].values
                sim_vals = event_df['simulated'].values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(obs_vals) | np.isnan(sim_vals))
                if valid_mask.sum() > 0:
                    obs_clean = obs_vals[valid_mask]
                    sim_clean = sim_vals[valid_mask]
                    
                    # Calculate NSE
                    numerator = np.sum((obs_clean - sim_clean) ** 2)
                    denominator = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
                    nse_event = 1 - (numerator / denominator) if denominator != 0 else np.nan
                    
                    # Calculate MSE
                    mse_event = np.mean((obs_clean - sim_clean) ** 2)
                else:
                    nse_event = np.nan
                    mse_event = np.nan
                
                metrics_rows.append({
                    "run": actual_run_dir.name,
                    "basin": basin,
                    "event": event_str,
                    "MSE": mse_event,
                    "NSE": nse_event
                })
            
            # Save full period data
            try:
                full_df = pd.DataFrame({
                    'date': qobs['date'].values,
                    'observed': np.asarray(qobs).flatten(),
                    'simulated': np.asarray(qsim).flatten(),
                    'shifted': np.asarray(qobs_shift).flatten()
                })
                full_csv_file = validation_hydrographs_dir / f"{basin}_full_period.csv"
                full_df.to_csv(full_csv_file, index=False)
                print(f"Saved full period data for basin {basin}")
                
                # Calculate full period metrics
                obs_vals = full_df['observed'].values
                sim_vals = full_df['simulated'].values
                valid_mask = ~(np.isnan(obs_vals) | np.isnan(sim_vals))
                
                if valid_mask.sum() > 0:
                    obs_clean = obs_vals[valid_mask]
                    sim_clean = sim_vals[valid_mask]
                    
                    numerator = np.sum((obs_clean - sim_clean) ** 2)
                    denominator = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
                    nse_full = 1 - (numerator / denominator) if denominator != 0 else np.nan
                    mse_full = np.mean((obs_clean - sim_clean) ** 2)
                else:
                    nse_full = np.nan
                    mse_full = np.nan
                
                metrics_rows.append({
                    "run": actual_run_dir.name,
                    "basin": basin,
                    "event": "full_period",
                    "MSE": mse_full,
                    "NSE": nse_full
                })
                
            except Exception as e:
                print(f"Error processing full period for basin {basin}: {e}")
                
        except Exception as e:
            print(f"Error processing basin {basin}: {e}")
            traceback.print_exc()
    
    # Save metrics summary
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_csv_file = validation_hydrographs_dir / "metrics_summary.csv"
        metrics_df.to_csv(metrics_csv_file, index=False)
        print(f"Saved metrics summary to {metrics_csv_file}")
        
        # Print summary statistics
        print("\nValidation Summary:")
        print(f"Total basins processed: {len(basins)}")
        print(f"Total events processed: {len(metrics_rows)}")
        
        event_metrics = metrics_df[metrics_df['event'] != 'full_period']
        if not event_metrics.empty:
            print(f"Average NSE (events): {event_metrics['NSE'].mean():.4f}")
            print(f"Average MSE (events): {event_metrics['MSE'].mean():.4f}")
        
        full_period_metrics = metrics_df[metrics_df['event'] == 'full_period']
        if not full_period_metrics.empty:
            print(f"Average NSE (full period): {full_period_metrics['NSE'].mean():.4f}")
            print(f"Average MSE (full period): {full_period_metrics['MSE'].mean():.4f}")
    
    print("Validation completed successfully!")

if __name__ == '__main__':
    try:
        print("Starting validation...")
        main()
        print("Validation completed.")
    except Exception as e:
        print(f"Error in validation: {str(e)}")
        traceback.print_exc()