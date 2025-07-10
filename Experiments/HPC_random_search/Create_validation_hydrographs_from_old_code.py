
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
    # Use specific run directory
    run_dir = Path("C:/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results/job_41780893/run_001")
    run_subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not run_subdirs:
        print(f"No run subdirectories found in {run_dir}")
        return
    
    # Use the first run subdirectory
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
    max_events_df['max_date'] = pd.to_datetime(max_events_df['max_date'])
    max_events_df['start_date'] = max_events_df['max_date'] - pd.Timedelta(days=1)
    max_events_df['end_date'] = max_events_df['max_date'] + pd.Timedelta(days=1)
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

        # try to change the test start and end dates to those of the validation period:
        if 'test_start_date' in config_dict and 'test_end_date' in config_dict:
            config_dict['test_start_date'] = config_dict['validation_start_date']
            config_dict['test_end_date'] = config_dict['validation_end_date']

            # for debugging purposes, change config_dict['test_end_date'] to 31/12/2023:
            config_dict['test_end_date'] = '31/10/2010'  # Change to a fixed date for debugging


        # Create new config from modified dictionary
        config = Config(config_dict)
    else:
        # If running on HPC, use the original config file
        config = Config(config_path)
    
    
    tester = get_tester(cfg=config, run_dir=run_dir, period="test", init_model=True)
    results = tester.evaluate(save_results=False, metrics=config.metrics)
    basins = results.keys()

    # for each basin, create in the run directory a folder called "validation_hydrographs"
    # and save the hydrograph of the entire validation period in it,
    # in the format of csv: time column and observed and simulated values.
    # Also add a "shifted" column with the observed values shifted by 18 time steps (3 hours).
    validation_hydrographs_dir = run_dir / "validation_hydrographs"
    validation_hydrographs_dir.mkdir(exist_ok=True)
    

    for basin in basins:
        try:
            # Basin key matching logic
            if basin in max_events_df.index:
                basin_key = basin
            else:
                basin_id = basin.split('_')[-1] if '_' in basin else basin
                if int(basin_id) in max_events_df.index:
                    basin_key = int(basin_id)                
                
                
                # if basin.startswith('il_') and basin[3:] in max_events_df.index:
                #     basin_key = basin[3:]
                #     print(f"Found basin {basin} as {basin_key} in max events data")
                # elif 'il_' + basin in max_events_df.index:
                #     basin_key = 'il_' + basin
                #     print(f"Found basin {basin} as {basin_key} in max events data")
                # else:
                #     basin_id = basin.split('_')[-1] if '_' in basin else basin
                #     matching_basins = [idx for idx in max_events_df.index 
                #         if idx.endswith(basin_id) or (idx.split('_')[-1] if '_' in idx else idx) == basin_id]
                #     if matching_basins:
                #         basin_key = matching_basins[0]
                #         print(f"Matched basin {basin} to {basin_key} based on ID")
                #     else:
                #         print(f"Basin {basin} not found in max events data, skipping...")
                #         continue

            basin_events = max_events_df.loc[basin_key]
            if isinstance(basin_events, pd.Series):
                basin_events = pd.DataFrame([basin_events])

            all_hydrographs = pd.DataFrame()
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
                # Ensure 1D arrays
                event_df = pd.DataFrame({
                    'date': qobs_event['date'].values,
                    'observed': np.asarray(qobs_event).flatten(),
                    'simulated': np.asarray(qsim_event).flatten(),
                    'shifted': np.asarray(qobs_shift_event).flatten(),
                    'event_date': event['max_date'],
                    'event_discharge': event['max_discharge']
                })
                all_hydrographs = pd.concat([all_hydrographs, event_df], ignore_index=True)

            # Save the combined hydrographs for this basin
            if not all_hydrographs.empty:
                csv_file = validation_hydrographs_dir / f"{basin}_validation_hydrographs.csv"
                all_hydrographs.to_csv(csv_file, index=False)
                print(f"Saved validation hydrographs for basin {basin} to {csv_file}")

                # Plotting, deduplicate legend
                plt.figure(figsize=(12, 8))
                used_labels = set()
                for event_date in all_hydrographs['event_date'].unique():
                    event_data = all_hydrographs[all_hydrographs['event_date'] == event_date]
                    label_obs = f"Observed ({pd.to_datetime(event_date).strftime('%Y-%m-%d')})"
                    label_sim = f"Simulated ({pd.to_datetime(event_date).strftime('%Y-%m-%d')})"
                    l1 = label_obs if label_obs not in used_labels else None
                    l2 = label_sim if label_sim not in used_labels else None
                    plt.plot(event_data['date'], event_data['observed'], label=l1)
                    plt.plot(event_data['date'], event_data['simulated'], linestyle='--', label=l2)
                    used_labels.update([label_obs, label_sim])
                plt.title(f"Validation Hydrographs for Basin {basin}")
                plt.xlabel("Date")
                plt.ylabel("Discharge (m³/s)")
                plt.grid(True)
                plt.legend()
                plt.savefig(validation_hydrographs_dir / f"{basin}_validation_hydrographs.png")
                plt.close()

            # Also save the entire evaluation period for this basin
            try:
                # Get the entire period data
                # qobs, qsim, qobs_shift already defined above
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
                # Highlight the maximum event periods
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
                # Add some statistics to the plot
                metrics_full = {}
                try:
                    obs_array = full_df['observed'].values
                    sim_array = full_df['simulated'].values
                    if not np.isnan(obs_array).all() and not np.isnan(sim_array).all():
                        mean_obs = np.nanmean(obs_array)
                        numerator = np.nansum((obs_array - sim_array) ** 2)
                        denominator = np.nansum((obs_array - mean_obs) ** 2)
                        nse = 1 - (numerator / denominator if denominator != 0 else np.nan)
                        metrics_full['NSE'] = nse
                        rmse = np.sqrt(np.nanmean((obs_array - sim_array) ** 2))
                        metrics_full['RMSE'] = rmse
                        plt.text(0.02, 0.95, f"NSE: {metrics_full.get('NSE', 'N/A'):.3f}", transform=plt.gca().transAxes)
                        plt.text(0.02, 0.91, f"RMSE: {metrics_full.get('RMSE', 'N/A'):.3f}", transform=plt.gca().transAxes)
                except Exception as e:
                    print(f"Error calculating metrics for basin {basin}: {str(e)}")
                plt.savefig(validation_hydrographs_dir / f"{basin}_full_period.png")
                plt.close()
            except Exception as e:
                print(f"Error creating full period visualization for basin {basin}: {str(e)}")
        except Exception as e:
            print(f"Error processing basin {basin}: {str(e)}")
            continue



    # run_dir = Path(
    #     "runs/HPC_training_zscore_norm_hidden_size256_batch_size512_learning_rate0001_2812_002447")  # you'll find this path in the output of the training above.
    # # run_config = Config(Path("Feature_normalization\Best_HPC_flow_only.yml"))
    # run_config = Config(run_dir / "config.yml")

    # data_dir = run_config.data_dir

    # max_events_path = (r"C:\PhD\Python\neuralhydrology-neuralhydrology-e4329c3"
    #                    r"\neuralhydrology\max_event_dates.csv")
    # # create a tester instance and start evaluation
    # tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
    # results = tester.evaluate(save_results=False, metrics=run_config.metrics)

    # # results.keys()
    # basins = results.keys()







    # # Normalization method:
    # # normalization = 'minmax_norm'
    # normalization = 'z_score_norm'
    # # normalization = 'unit_discharge'
    # # normalization = 'no_norm'

    # norm_dict = pickle.load(open(data_dir / r"timeseries\netcdf\il\normalization_dict.pkl", "rb"))

    # max_event_per_basin = pd.read_csv(max_events_path)
    # max_event_per_basin = max_event_per_basin.set_index("basin")

    # # create a "test figures" folder in the run directory:
    # test_figures_dir = run_dir / "test_figures_unnormalized"
    # test_figures_dir.mkdir(exist_ok=True)

    # delay = 18  # 3 hours = 18 10-minute steps
    # for basin in basins:
    #     start_date = max_event_per_basin.loc[basin, "event_start"]
    #     end_date = max_event_per_basin.loc[basin, "event_end"]

    #     # use relevant normalization values of the basin to rescale the data:
    #     basin_norm_dict = norm_dict[basin]
    #     if normalization == 'minmax_norm':
    #         # extract observations and simulations for a specific station in a specific date range:
    #         qobs = results[basin]["10min"]["xr"]["Flow_m3_sec_minmax_norm_obs"]
    #         qsim = results[basin]["10min"]["xr"]["Flow_m3_sec_minmax_norm_sim"]
    #         min_values = basin_norm_dict["features"]["Flow_m3_sec"]["min"]
    #         max_values = basin_norm_dict["features"]["Flow_m3_sec"]["max"]
    #         qobs = qobs * (max_values - min_values) + min_values
    #         qsim = qsim * (max_values - min_values) + min_values
    #         norm_type = 'training minmax normalization'

    #     elif normalization == 'z_score_norm':
    #         # extract observations and simulations for a specific station in a specific date range:
    #         qobs = results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_obs"]
    #         qsim = results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_sim"]
    #         mean_values = basin_norm_dict["features"]["Flow_m3_sec"]["mean"]
    #         std_values = basin_norm_dict["features"]["Flow_m3_sec"]["std"]
    #         qobs = qobs * std_values + mean_values
    #         qsim = qsim * std_values + mean_values
    #         norm_type = 'training z-score normalization'

    #     elif normalization == 'unit_discharge':
    #         # extract observations and simulations for a specific station in a specific date range:
    #         qobs = results[basin]["10min"]["xr"]["unit_discharge_m3_sec_km_obs"]
    #         qsim = results[basin]["10min"]["xr"]["unit_discharge_m3_sec_km_sim"]
    #         basin_area = basin_norm_dict["basin_area"]
    #         qobs = qobs * basin_area
    #         qsim = qsim * basin_area
    #         norm_type = 'unit discharge normalization'

    #     else:
    #         # extract observations and simulations for a specific station in a specific date range:
    #         qobs = results[basin]["10min"]["xr"]["Flow_m3_sec_obs"]
    #         qsim = results[basin]["10min"]["xr"]["Flow_m3_sec_sim"]
    #         norm_type = 'no normalization'

    #     # Shift the observed data by the delay
    #     fill_value = qobs.isel(date=0, time_step=0).item()
    #     qobs_shift = qobs.shift(date=18, fill_value=fill_value)

    #     # Filter qobs based on the date range
    #     qobs_dates = qobs.sel(date=slice(start_date, end_date))
    #     qsim_dates = qsim.sel(date=slice(start_date, end_date))
    #     qobs_shift_dates = qobs_shift.sel(date=slice(start_date, end_date))
    #     values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1), "10min")
    #     ref_values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qobs_shift.isel(time_step=-1), "10min")

    #     fig, ax = plt.subplots(figsize=(16, 10))
    #     ax.plot(qobs_dates["date"], qobs_dates, label="Observed")
    #     ax.plot(qsim_dates["date"], qsim_dates, label="Simulated")
    #     ax.plot(qobs_shift_dates["date"], qobs_shift_dates, label="Observed shifted")

    #     original_xlim = ax.get_xlim()
    #     original_ylim = ax.get_ylim()

    #     peaks_data = values['Peak-time-discharge']
    #     peaks_data_shifted = ref_values['Peak-time-discharge']
    #     N_observed_peaks = len(peaks_data['Observed'])

    #     ax.scatter(xarray.concat(peaks_data['Observed'], dim='peaks').coords['date'].values,
    #                xarray.concat(peaks_data['Observed'], dim='peaks').values, color='blue', label='Observed Peak',
    #                marker='x')
    #     ax.scatter(xarray.concat(peaks_data['Simulated'], dim='peaks').coords['date'].values,
    #                xarray.concat(peaks_data['Simulated'], dim='peaks').values, color='orange', label='Simulated Peak',
    #                marker='x')
    #     ax.scatter(xarray.concat(peaks_data_shifted['Simulated'], dim='peaks').coords['date'].values,
    #                xarray.concat(peaks_data_shifted['Simulated'], dim='peaks').values, color='green',
    #                label='lagged Peak', marker='x')

    #     ax.set_xlim(original_xlim)
    #     ax.set_ylim(original_ylim)

    #     ax.grid()
    #     ax.set_ylabel("Discharge (m^3/s)")
    #     ax.set_title(f"Test period - basin {basin} - learning rate {run_config.learning_rate}, hidden size {run_config.hidden_size}, sequence length {run_config.seq_length}, {norm_type}")
    #     # add axis to the plot
    #     ax.legend()
    #     plt.show()

    #     # Save the figure
    #     fig.savefig(test_figures_dir / f"{basin}_dates.png")

    #     fig, ax = plt.subplots(figsize=(16, 10))
    #     ax.plot(qobs["date"], qobs, label="Observed", color='blue')
    #     ax.plot(qsim["date"], qsim, label="Simulated", color='orange')
    #     ax.plot(qobs_shift["date"], qobs_shift, label="Observed shifted", color='green')
    #     ax.grid()
    #     ax.set_ylabel("Discharge (m^3/s)")
    #     ax.set_title(f"Test period - basin {basin} - learning rate {run_config.learning_rate}, hidden size {run_config.hidden_size}, sequence length {run_config.seq_length}, {norm_type})")
    #     # add axis to the plot
    #     ax.text(0.01, 0.96, "Observation - lagged metrics", transform=ax.transAxes, fontweight='bold')
    #     ax.text(0.01, 0.92, f"NSE: {ref_values['NSE']:.2f}", transform=ax.transAxes)
    #     ax.text(0.01, 0.88, f"RMSE: {ref_values['RMSE']:.2f}", transform=ax.transAxes)
    #     ax.text(0.01, 0.84, f"FHV: {ref_values['FHV']:.1f}", transform=ax.transAxes)
    #     ax.text(0.01, 0.80, f"Peak-Timing: {ref_values['Peak-Timing']*10:.0f} minutes, N_obs={N_observed_peaks}", transform=ax.transAxes)
    #     ax.text(0.01, 0.76, f"Missed-Peaks: {ref_values['Missed-Peaks']:.1f}, N_obs={N_observed_peaks}", transform=ax.transAxes)
    #     ax.text(0.01, 0.72, f"Peak-MAPE: {ref_values['Peak-MAPE']:.1f}, N_obs={N_observed_peaks}", transform=ax.transAxes)

    #     ax.text(0.23, 0.96, "Observation - simulation metrics", transform=ax.transAxes, fontweight='bold')
    #     ax.text(0.23, 0.92, f"NSE: {values['NSE']:.2f}", transform=ax.transAxes)
    #     ax.text(0.23, 0.88, f"RMSE: {values['RMSE']:.2f}", transform=ax.transAxes)
    #     ax.text(0.23, 0.84, f"FHV: {values['FHV']:.1f}", transform=ax.transAxes)
    #     ax.text(0.23, 0.80, f"Peak-Timing: {values['Peak-Timing']*10:.0f} minutes, N_obs={N_observed_peaks}", transform=ax.transAxes)
    #     ax.text(0.23, 0.76, f"Missed-Peaks: {values['Missed-Peaks']:.1f}, N_obs={N_observed_peaks}", transform=ax.transAxes)
    #     ax.text(0.23, 0.72, f"Peak-MAPE: {values['Peak-MAPE']:.1f}, N_obs={N_observed_peaks}", transform=ax.transAxes)

    #     ax.scatter(xarray.concat(peaks_data['Observed'], dim='peaks').coords['date'].values,
    #                xarray.concat(peaks_data['Observed'], dim='peaks').values, color='blue', label='Observed Peak', marker='x')
    #     ax.scatter(xarray.concat(peaks_data['Simulated'], dim='peaks').coords['date'].values,
    #                xarray.concat(peaks_data['Simulated'], dim='peaks').values, color='orange', label='Simulated Peak', marker='x')
    #     ax.scatter(xarray.concat(peaks_data_shifted['Simulated'], dim='peaks').coords['date'].values,
    #                xarray.concat(peaks_data_shifted['Simulated'], dim='peaks').values, color='green', label='lagged Peak', marker='x')
    #     ax.legend()
    #     ax.set_xlim(datetime(2018, 1, 1), datetime(2018, 3, 1))

    #     plt.show()

    #     fig.savefig(test_figures_dir / f"{basin}.png")


if __name__ == '__main__':
    try:
        print("Starting script execution...")
        main()
        print("Script execution completed.")
    except Exception as e:
        print(f"Error in main script execution: {str(e)}")
        import traceback
        traceback.print_exc()
