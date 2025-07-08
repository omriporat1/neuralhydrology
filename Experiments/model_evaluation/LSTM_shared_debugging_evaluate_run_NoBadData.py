import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import os
import xarray
from datetime import datetime



# %%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU


def main():
    run_dir = Path(
        "runs/HPC_training_zscore_norm_hidden_size256_batch_size512_learning_rate0001_2812_002447")  # you'll find this path in the output of the training above.
    # run_config = Config(Path("Feature_normalization\Best_HPC_flow_only.yml"))
    run_config = Config(run_dir / "config.yml")

    data_dir = run_config.data_dir

    max_events_path = (r"C:\PhD\Python\neuralhydrology-neuralhydrology-e4329c3"
                       r"\neuralhydrology\max_event_dates.csv")
    # create a tester instance and start evaluation
    tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
    results = tester.evaluate(save_results=False, metrics=run_config.metrics)

    # results.keys()
    basins = results.keys()


    # Normalization method:
    # normalization = 'minmax_norm'
    normalization = 'z_score_norm'
    # normalization = 'unit_discharge'
    # normalization = 'no_norm'

    norm_dict = pickle.load(open(data_dir / r"timeseries\netcdf\il\normalization_dict.pkl", "rb"))

    max_event_per_basin = pd.read_csv(max_events_path)
    max_event_per_basin = max_event_per_basin.set_index("basin")

    # create a "test figures" folder in the run directory:
    test_figures_dir = run_dir / "test_figures_unnormalized"
    test_figures_dir.mkdir(exist_ok=True)

    delay = 18  # 3 hours = 18 10-minute steps
    for basin in basins:
        start_date = max_event_per_basin.loc[basin, "event_start"]
        end_date = max_event_per_basin.loc[basin, "event_end"]

        # use relevant normalization values of the basin to rescale the data:
        basin_norm_dict = norm_dict[basin]
        if normalization == 'minmax_norm':
            # extract observations and simulations for a specific station in a specific date range:
            qobs = results[basin]["10min"]["xr"]["Flow_m3_sec_minmax_norm_obs"]
            qsim = results[basin]["10min"]["xr"]["Flow_m3_sec_minmax_norm_sim"]
            min_values = basin_norm_dict["features"]["Flow_m3_sec"]["min"]
            max_values = basin_norm_dict["features"]["Flow_m3_sec"]["max"]
            qobs = qobs * (max_values - min_values) + min_values
            qsim = qsim * (max_values - min_values) + min_values
            norm_type = 'training minmax normalization'

        elif normalization == 'z_score_norm':
            # extract observations and simulations for a specific station in a specific date range:
            qobs = results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_obs"]
            qsim = results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_sim"]
            mean_values = basin_norm_dict["features"]["Flow_m3_sec"]["mean"]
            std_values = basin_norm_dict["features"]["Flow_m3_sec"]["std"]
            qobs = qobs * std_values + mean_values
            qsim = qsim * std_values + mean_values
            norm_type = 'training z-score normalization'

        elif normalization == 'unit_discharge':
            # extract observations and simulations for a specific station in a specific date range:
            qobs = results[basin]["10min"]["xr"]["unit_discharge_m3_sec_km_obs"]
            qsim = results[basin]["10min"]["xr"]["unit_discharge_m3_sec_km_sim"]
            basin_area = basin_norm_dict["basin_area"]
            qobs = qobs * basin_area
            qsim = qsim * basin_area
            norm_type = 'unit discharge normalization'

        else:
            # extract observations and simulations for a specific station in a specific date range:
            qobs = results[basin]["10min"]["xr"]["Flow_m3_sec_obs"]
            qsim = results[basin]["10min"]["xr"]["Flow_m3_sec_sim"]
            norm_type = 'no normalization'

        # Shift the observed data by the delay
        fill_value = qobs.isel(date=0, time_step=0).item()
        qobs_shift = qobs.shift(date=18, fill_value=fill_value)

        # Filter qobs based on the date range
        qobs_dates = qobs.sel(date=slice(start_date, end_date))
        qsim_dates = qsim.sel(date=slice(start_date, end_date))
        qobs_shift_dates = qobs_shift.sel(date=slice(start_date, end_date))
        values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1), "10min")
        ref_values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qobs_shift.isel(time_step=-1), "10min")

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(qobs_dates["date"], qobs_dates, label="Observed")
        ax.plot(qsim_dates["date"], qsim_dates, label="Simulated")
        ax.plot(qobs_shift_dates["date"], qobs_shift_dates, label="Observed shifted")

        original_xlim = ax.get_xlim()
        original_ylim = ax.get_ylim()

        peaks_data = values['Peak-time-discharge']
        peaks_data_shifted = ref_values['Peak-time-discharge']
        N_observed_peaks = len(peaks_data['Observed'])

        ax.scatter(xarray.concat(peaks_data['Observed'], dim='peaks').coords['date'].values,
                   xarray.concat(peaks_data['Observed'], dim='peaks').values, color='blue', label='Observed Peak',
                   marker='x')
        ax.scatter(xarray.concat(peaks_data['Simulated'], dim='peaks').coords['date'].values,
                   xarray.concat(peaks_data['Simulated'], dim='peaks').values, color='orange', label='Simulated Peak',
                   marker='x')
        ax.scatter(xarray.concat(peaks_data_shifted['Simulated'], dim='peaks').coords['date'].values,
                   xarray.concat(peaks_data_shifted['Simulated'], dim='peaks').values, color='green',
                   label='lagged Peak', marker='x')

        ax.set_xlim(original_xlim)
        ax.set_ylim(original_ylim)

        ax.grid()
        ax.set_ylabel("Discharge (m^3/s)")
        ax.set_title(f"Test period - basin {basin} - learning rate {run_config.learning_rate}, hidden size {run_config.hidden_size}, sequence length {run_config.seq_length}, {norm_type}")
        # add axis to the plot
        ax.legend()
        plt.show()

        # Save the figure
        fig.savefig(test_figures_dir / f"{basin}_dates.png")

        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(qobs["date"], qobs, label="Observed", color='blue')
        ax.plot(qsim["date"], qsim, label="Simulated", color='orange')
        ax.plot(qobs_shift["date"], qobs_shift, label="Observed shifted", color='green')
        ax.grid()
        ax.set_ylabel("Discharge (m^3/s)")
        ax.set_title(f"Test period - basin {basin} - learning rate {run_config.learning_rate}, hidden size {run_config.hidden_size}, sequence length {run_config.seq_length}, {norm_type})")
        # add axis to the plot
        ax.text(0.01, 0.96, "Observation - lagged metrics", transform=ax.transAxes, fontweight='bold')
        ax.text(0.01, 0.92, f"NSE: {ref_values['NSE']:.2f}", transform=ax.transAxes)
        ax.text(0.01, 0.88, f"RMSE: {ref_values['RMSE']:.2f}", transform=ax.transAxes)
        ax.text(0.01, 0.84, f"FHV: {ref_values['FHV']:.1f}", transform=ax.transAxes)
        ax.text(0.01, 0.80, f"Peak-Timing: {ref_values['Peak-Timing']*10:.0f} minutes, N_obs={N_observed_peaks}", transform=ax.transAxes)
        ax.text(0.01, 0.76, f"Missed-Peaks: {ref_values['Missed-Peaks']:.1f}, N_obs={N_observed_peaks}", transform=ax.transAxes)
        ax.text(0.01, 0.72, f"Peak-MAPE: {ref_values['Peak-MAPE']:.1f}, N_obs={N_observed_peaks}", transform=ax.transAxes)

        ax.text(0.23, 0.96, "Observation - simulation metrics", transform=ax.transAxes, fontweight='bold')
        ax.text(0.23, 0.92, f"NSE: {values['NSE']:.2f}", transform=ax.transAxes)
        ax.text(0.23, 0.88, f"RMSE: {values['RMSE']:.2f}", transform=ax.transAxes)
        ax.text(0.23, 0.84, f"FHV: {values['FHV']:.1f}", transform=ax.transAxes)
        ax.text(0.23, 0.80, f"Peak-Timing: {values['Peak-Timing']*10:.0f} minutes, N_obs={N_observed_peaks}", transform=ax.transAxes)
        ax.text(0.23, 0.76, f"Missed-Peaks: {values['Missed-Peaks']:.1f}, N_obs={N_observed_peaks}", transform=ax.transAxes)
        ax.text(0.23, 0.72, f"Peak-MAPE: {values['Peak-MAPE']:.1f}, N_obs={N_observed_peaks}", transform=ax.transAxes)

        ax.scatter(xarray.concat(peaks_data['Observed'], dim='peaks').coords['date'].values,
                   xarray.concat(peaks_data['Observed'], dim='peaks').values, color='blue', label='Observed Peak', marker='x')
        ax.scatter(xarray.concat(peaks_data['Simulated'], dim='peaks').coords['date'].values,
                   xarray.concat(peaks_data['Simulated'], dim='peaks').values, color='orange', label='Simulated Peak', marker='x')
        ax.scatter(xarray.concat(peaks_data_shifted['Simulated'], dim='peaks').coords['date'].values,
                   xarray.concat(peaks_data_shifted['Simulated'], dim='peaks').values, color='green', label='lagged Peak', marker='x')
        ax.legend()
        ax.set_xlim(datetime(2018, 1, 1), datetime(2018, 3, 1))

        plt.show()

        fig.savefig(test_figures_dir / f"{basin}.png")


if __name__ == '__main__':
    main()
