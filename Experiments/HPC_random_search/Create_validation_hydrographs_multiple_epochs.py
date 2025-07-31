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

def extract_train_metrics_from_log(log_path):
    """Extract train and validation metrics per epoch from a log file matching the actual log format."""
    epochs = []
    train_losses = []
    val_losses = []
    val_nses = []
    with open(log_path, "r") as f:
        for line in f:
            # Match: Epoch 1 average loss: avg_loss: 0.05254, avg_total_loss: 0.05254
            train_match = re.search(r"Epoch\s+(\d+) average loss: avg_loss: ([0-9.eE+-]+)", line)
            if train_match:
                epoch = int(train_match.group(1))
                train_loss = float(train_match.group(2))
                # Store epoch and train loss
                epochs.append(epoch)
                train_losses.append(train_loss)
            # Match: Epoch 1 average validation loss: 0.11096 -- Median validation metrics: avg_loss: 0.11096, NSE: 0.77376
            val_match = re.search(r"Epoch\s+(\d+) average validation loss: ([0-9.eE+-]+) -- Median validation metrics: avg_loss: ([0-9.eE+-]+), NSE: ([0-9.eE+-]+)", line)
            if val_match:
                epoch = int(val_match.group(1))
                val_loss = float(val_match.group(2))
                val_nse = float(val_match.group(4))
                # Ensure lists are aligned by epoch
                while len(val_losses) < len(epochs) - 1:
                    val_losses.append(float('nan'))
                    val_nses.append(float('nan'))
                val_losses.append(val_loss)
                val_nses.append(val_nse)
    # Pad lists to same length
    while len(train_losses) < len(epochs):
        train_losses.append(float('nan'))
    while len(val_losses) < len(epochs):
        val_losses.append(float('nan'))
    while len(val_nses) < len(epochs):
        val_nses.append(float('nan'))
    return epochs, train_losses, val_losses, val_nses

def main():
    run_id = int(sys.argv[1])
    job_id = int(41780893)

    job_dir = Path(f"/sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results/job_{job_id}")
    run_dir = job_dir / f"run_{run_id:03d}"
    print(f"\nProcessing run directory: {run_dir}")
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}")
        return

    # Find the actual run subdirectory (e.g., N38_A30_4CPU_SMI_0506_225956)
    run_subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
    if not run_subdirs:
        print(f"No run subdirectories found in {run_dir}")
        return
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
    
    new_basin_filename = "il_basins_high_qual_0_04_N35.txt"
    basin_filename = new_basin_filename

    max_events_df = pd.read_csv(max_events_path)
    max_events_df = max_events_df[max_events_df['max_discharge'] > 0]
    max_events_df['max_date'] = pd.to_datetime(max_events_df['max_date'], dayfirst=True)
    # For 72 hours: day before, day of, day after
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
            # basin_filename = Path(config_dict['test_basin_file']).name
            config_dict['test_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)

        if 'train_basin_file' in config_dict and config_dict['train_basin_file']:
            # basin_filename = Path(config_dict['train_basin_file']).name
            config_dict['train_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)

        if 'validation_basin_file' in config_dict and config_dict['validation_basin_file']:
            # basin_filename = Path(config_dict['validation_basin_file']).name
            config_dict['validation_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)

        if 'test_start_date' in config_dict and 'test_end_date' in config_dict:
            config_dict['test_start_date'] = config_dict['validation_start_date']
            config_dict['test_end_date'] = config_dict['validation_end_date']

        config = Config(config_dict)
    else:
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


    # Find all epoch model files
    epoch_files = sorted(run_dir.glob("model_epoch*.pt"))
    epoch_numbers = [int(f.stem.split("epoch")[-1]) for f in epoch_files]

    # Prepare to collect metrics
    epoch_metrics = []

    # Look for output.log directly in run_dir
    log_file = run_dir / "output.log"
    if log_file.exists():
        print(f"Found log file: {log_file}")
        train_epochs, train_losses, val_losses, val_nses = extract_train_metrics_from_log(log_file)
        train_metrics_dict = {e: (l, v, n) for e, l, v, n in zip(train_epochs, train_losses, val_losses, val_nses)}
    else:
        print("output.log not found in", run_dir)
        print("Files in run_dir:", list(run_dir.iterdir()))
        train_metrics_dict = {}

    # Prepare validation hydrographs output directory
    validation_hydrographs_dir = run_dir / "validation_hydrographs_cluster"
    validation_hydrographs_dir.mkdir(exist_ok=True)

    # For each epoch, evaluate and collect metrics
    for epoch, epoch_file in zip(epoch_numbers, epoch_files):
        print(f"Evaluating epoch {epoch} ({epoch_file.name})")
        tester = get_tester(cfg=config, run_dir=run_dir, period="test", init_model=True)
        results = tester.evaluate(epoch=epoch, save_results=False, metrics=config.metrics)
        basins = results.keys()

        # Aggregate full-period metrics across all basins
        mse_list = []
        nse_list = []
        for basin in basins:
            basin_results = results[basin]["10min"]["xr"]
            qobs = basin_results["Flow_m3_sec_obs"]
            qsim = basin_results["Flow_m3_sec_sim"]
            if 'time_step' in qobs.dims:
                qobs = qobs.isel(time_step=-1)
            if 'time_step' in qsim.dims:
                qsim = qsim.isel(time_step=-1)
            obs_array = np.asarray(qobs).flatten()
            sim_array = np.asarray(qsim).flatten()
            mse = np.nanmean((obs_array - sim_array) ** 2)
            mean_obs = np.nanmean(obs_array)
            numerator = np.nansum((obs_array - sim_array) ** 2)
            denominator = np.nansum((obs_array - mean_obs) ** 2)
            nse = 1 - (numerator / denominator if denominator != 0 else np.nan)
            mse_list.append(mse)
            nse_list.append(nse)
        val_mse = np.nanmean(mse_list)
        val_nse = np.nanmean(nse_list)

        # Get train/val loss/NSE if available
        train_loss, val_loss, val_nse_log = train_metrics_dict.get(epoch, (np.nan, np.nan, np.nan))

        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss_log": val_loss,
            "val_NSE_log": val_nse_log,
            "val_MSE": val_mse,
            "val_NSE": val_nse
        })

    # Save metrics CSV
    metrics_df = pd.DataFrame(epoch_metrics)
    metrics_df = metrics_df.sort_values("epoch")
    metrics_csv = run_dir / "epoch_metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved epoch metrics summary to {metrics_csv}")

    # Plot metrics as function of epoch
    plt.figure(figsize=(10, 7))
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train Loss")
    plt.plot(metrics_df["epoch"], metrics_df["train_MSE"], label="Train MSE")
    plt.plot(metrics_df["epoch"], metrics_df["val_MSE"], label="Validation MSE")
    plt.plot(metrics_df["epoch"], metrics_df["val_NSE"], label="Validation NSE")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"Training and Validation Metrics vs. Epoch ({run_dir.name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "epoch_metrics_summary.png")
    plt.close()

    # Find all epoch model files
    epoch_files = sorted(run_dir.glob("model_epoch*.pt"))
    epoch_numbers = [int(f.stem.split("epoch")[-1]) for f in epoch_files]

    for epoch, epoch_file in zip(epoch_numbers, epoch_files):
        print(f"Processing epoch {epoch} ({epoch_file.name})")

        # Create output folder for this epoch
        epoch_out_dir = run_dir / f"validation_hydrographs_epoch_{epoch:03d}"
        epoch_out_dir.mkdir(exist_ok=True)

        # Evaluate this epoch
        tester = get_tester(cfg=config, run_dir=run_dir, period="test", init_model=True)
        results = tester.evaluate(epoch=epoch, save_results=False, metrics=config.metrics)
        basins = results.keys()

        metrics_rows = []
        for basin in basins:
            try:
                # Find matching events for this basin
                if basin in max_events_df.index:
                    basin_key = basin
                else:
                    basin_id = basin.split('_')[-1] if '_' in basin else basin
                    try:
                        basin_key = int(basin_id)
                    except Exception:
                        print(f"Could not parse basin id for {basin}, skipping.")
                        continue
                    if basin_key not in max_events_df.index:
                        print(f"Basin {basin} not found in max_events_df")
                        continue

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

                # Check for empty arrays or all NaN
                if len(qobs) == 0 or len(qsim) == 0:
                    print(f"Basin {basin}: qobs or qsim is empty, skipping.")
                    continue
                obs_array = np.asarray(qobs).flatten()
                sim_array = np.asarray(qsim).flatten()
                if np.all(np.isnan(sim_array)):
                    print(f"Basin {basin}: All simulated values are NaN, skipping.")
                    continue

                fill_value = qobs.isel(date=0).item() if 'date' in qobs.dims else qobs[0].item()
                qobs_shift = qobs.shift(date=delay, fill_value=fill_value)

                # --- Full period metrics ---
                try:
                    nse_full_val = nse(obs_array, sim_array)
                    pnse_full = persistent_nse(obs_array, sim_array, lag_steps=delay)
                    peak_err_full = peak_flow_error(obs_array, sim_array)
                    vol_err_full = volume_error(obs_array, sim_array)
                    metrics_rows.append({
                        "run": run_dir.name,
                        "basin": basin,
                        "event": "full_period",
                        "NSE": nse_full_val,
                        "pNSE": pnse_full,
                        "PeakFlowError": peak_err_full,
                        "VolumeError": vol_err_full,
                        "erroneous": False
                    })
                except Exception as e:
                    print(f"Error calculating full period metrics for basin {basin}: {e}")
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

                # --- Event metrics (72h window) ---
                for _, event in basin_events.iterrows():
                    start_date = event['start_date']
                    end_date = event['end_date']
                    try:
                        qobs_event = qobs.sel(date=slice(start_date, end_date))
                        qsim_event = qsim.sel(date=slice(start_date, end_date))
                        qobs_shift_event = qobs_shift.sel(date=slice(start_date, end_date))
                        obs_event = np.asarray(qobs_event).flatten()
                        sim_event = np.asarray(qsim_event).flatten()
                        if len(obs_event) == 0 or len(sim_event) == 0 or np.all(np.isnan(sim_event)):
                            print(f"Basin {basin} event {start_date}-{end_date}: empty or all-NaN, skipping.")
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
                            continue
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
                        event_df = pd.DataFrame({
                            'date': qobs_event['date'].values,
                            'observed': obs_event,
                            'simulated': sim_event,
                            'shifted': np.asarray(qobs_shift_event).flatten(),
                            'event_date': event['max_date'],
                            'event_discharge': event['max_discharge']
                        })
                        event_str = f"{pd.to_datetime(start_date).strftime('%Y%m%d')}_{pd.to_datetime(end_date).strftime('%Y%m%d')}"
                        csv_file = epoch_out_dir / f"{basin}_event_{event_str}.csv"
                        event_df.to_csv(csv_file, index=False)
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
                        plt.savefig(fig_file)
                        plt.close()
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
                        continue

                # Save full period hydrograph and plot
                try:
                    full_df = pd.DataFrame({
                        'date': qobs['date'].values,
                        'observed': obs_array,
                        'simulated': sim_array,
                        'shifted': np.asarray(qobs_shift).flatten()
                    })
                    full_csv_file = epoch_out_dir / f"{basin}_full_period.csv"
                    full_df.to_csv(full_csv_file, index=False)
                    plt.figure(figsize=(16, 10))
                    plt.plot(full_df['date'], full_df['observed'], label="Observed")
                    plt.plot(full_df['date'], full_df['simulated'], label="Simulated", linestyle='--')
                    plt.plot(full_df['date'], full_df['shifted'], label="Observed shifted (3 hours)")
                    for idx, (_, event) in enumerate(basin_events.iterrows()):
                        event_start = event['start_date']
                        event_end = event['end_date']
                        event_max = event['max_date']
                        plt.axvspan(event_start, event_end, alpha=0.2, color='yellow')
                        max_discharge = event['max_discharge']
                        plt.scatter([event_max], [max_discharge], color='red', s=100, zorder=5, label=f"Max Discharge ({pd.to_datetime(event_max).strftime('%Y-%m-%d')})" if idx == 0 else "")
                    plt.title(f"Full Validation Period for Basin {basin}")
                    plt.xlabel("Date")
                    plt.ylabel("Discharge (m³/s)")
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(epoch_out_dir / f"{basin}_full_period.png")
                    plt.close()
                except Exception as e:
                    print(f"Error creating full period visualization for basin {basin}: {str(e)}")
            except Exception as e:
                print(f"Error processing basin {basin}: {str(e)}")
                continue

        # --- Save metrics and summary for this epoch ---
        metrics_df = pd.DataFrame(metrics_rows)
        if "erroneous" not in metrics_df.columns:
            metrics_df["erroneous"] = False
        metrics_csv_file = epoch_out_dir / "event_metrics.csv"
        metrics_df.to_csv(metrics_csv_file, index=False)
        print(f"Saved event metrics to {metrics_csv_file}")

        # Summary statistics and plotting
        metrics_to_analyze = ["NSE", "pNSE", "PeakFlowError", "VolumeError"]
        if not metrics_df.empty and "erroneous" in metrics_df.columns:
            summary = metrics_df[~metrics_df["erroneous"]].agg({
                "NSE": ["mean", "std"],
                "pNSE": ["mean", "std"],
                "PeakFlowError": ["mean", "std"],
                "VolumeError": ["mean", "std"]
            })
            summary["erroneous_count"] = metrics_df["erroneous"].sum()
            summary_csv_file = epoch_out_dir / "event_metrics_summary.csv"
            summary.to_csv(summary_csv_file)
            print(f"Saved event metrics summary to {summary_csv_file}")

            # Bar plot of mean metrics
            plt.figure(figsize=(8, 5))
            mean_values = [summary.loc["mean", m] for m in metrics_to_analyze if m in summary.columns]
            plt.bar(metrics_to_analyze[:len(mean_values)], mean_values)
            plt.ylabel("Mean Metric Value")
            plt.title(f"Run {run_dir.name} Epoch {epoch} Event Metrics (mean)")
            plt.tight_layout()
            plt.savefig(epoch_out_dir / "event_metrics_summary.png")
            plt.close()

            # Boxplot for each metric
            plt.figure(figsize=(12, 8))
            for i, metric in enumerate(metrics_to_analyze):
                plt.subplot(2, 2, i+1)
                sns.boxplot(y=metrics_df.loc[~metrics_df["erroneous"], metric])
                plt.title(f"{metric} Distribution")
            plt.tight_layout()
            plt.savefig(epoch_out_dir / "event_metrics_boxplots.png")
            plt.close()
        else:
            print("No valid metrics to summarize or plot for this epoch.")

if __name__ == '__main__':
    try:
        print("Starting script execution...")
        main()
        print("Script execution completed.")
    except Exception as e:
        print(f"Error in main script execution: {str(e)}")
        import traceback
        traceback.print_exc()
