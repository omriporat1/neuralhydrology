import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


# Helper functions for metrics
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
    # Load event maxima
    max_events_df = pd.read_csv(
        "c:/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv",
        parse_dates=["max_date"]
    )
    
    results_dir = Path("c:/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results")
    run_dirs = [d for d in results_dir.glob("job_*/run_*/*") if d.is_dir()]

    all_metrics = []

    for run_dir in run_dirs:
        hydro_dir = run_dir / "validation_hydrographs_cluster"
        
        output_dir = hydro_dir / "metrics_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not hydro_dir.exists():
            continue
        hydro_files = list(hydro_dir.glob("il_*_full_period.csv"))
        run_metrics = []
        for hydro_file in hydro_files:
            basin_id = hydro_file.stem.split("_")[1]
            basin_events = max_events_df[max_events_df["basin"] == int(basin_id)]
            if basin_events.empty:
                continue
            df = pd.read_csv(hydro_file, parse_dates=["date"])
            for _, event in basin_events.iterrows():
                event_date = event["max_date"]
                start = pd.Timestamp(event_date) - pd.Timedelta(days=1)
                start = start.replace(hour=0, minute=0, second=0)
                end = pd.Timestamp(event_date) + pd.Timedelta(days=1)
                end = end.replace(hour=23, minute=59, second=59)
                event_df = df[(df["date"] >= start) & (df["date"] <= end)].copy()
                if event_df.empty:
                    run_metrics.append({
                        "run": run_dir.name,
                        "basin": basin_id,
                        "event_date": event_date,
                        "NSE": np.nan,
                        "pNSE": np.nan,
                        "PeakFlowError": np.nan,
                        "VolumeError": np.nan,
                        "erroneous": True
                    })
                    continue
                event_str = f"{pd.to_datetime(start).strftime('%Y%m%d')}_{pd.to_datetime(end).strftime('%Y%m%d')}"
                event_csv = output_dir / f"{basin_id}_event_{event_str}.csv"
                event_df.to_csv(event_csv, index=False)
                plt.figure(figsize=(10, 6))
                plt.plot(event_df["date"], event_df["observed"], label="Observed")
                plt.plot(event_df["date"], event_df["predicted"], label="Predicted")
                plt.plot(event_df["date"], event_df["shifted"], label="Shifted")
                plt.title(f"Basin {basin_id} Event {event_str}")
                plt.xlabel("Date")
                plt.ylabel("Discharge")
                plt.legend()
                plt.tight_layout()
                fig_file = output_dir / f"{basin_id}_event_{event_str}.png"
                plt.savefig(fig_file)
                plt.close()
                obs = event_df["observed"].values
                sim = event_df["predicted"].values
                shifted = event_df["shifted"].values
                nse_val = nse(obs, sim)
                pnse_val = persistent_nse(obs, sim, lag_steps=18)
                peak_err = peak_flow_error(obs, sim)
                vol_err = volume_error(obs, sim)
                run_metrics.append({
                    "run": run_dir.name,
                    "basin": basin_id,
                    "event_date": event_date,
                    "NSE": nse_val,
                    "pNSE": pnse_val,
                    "PeakFlowError": peak_err,
                    "VolumeError": vol_err,
                    "erroneous": False
                })
        metrics_df = pd.DataFrame(run_metrics)
        metrics_df.to_csv(hydro_dir / "event_metrics.csv", index=False)
        summary = metrics_df[~metrics_df["erroneous"]].agg({
            "NSE": ["mean", "std"],
            "pNSE": ["mean", "std"],
            "PeakFlowError": ["mean", "std"],
            "VolumeError": ["mean", "std"]
        })
        summary["erroneous_count"] = metrics_df["erroneous"].sum()
        summary.to_csv(hydro_dir / "event_metrics_summary.csv")
        plt.figure(figsize=(8, 5))
        plt.bar(["NSE", "pNSE", "PeakFlowError", "VolumeError"], summary.loc["mean"])
        plt.ylabel("Mean Metric Value")
        plt.title(f"Run {run_dir.name} Event Metrics (mean)")
        plt.tight_layout()
        plt.savefig(hydro_dir / "event_metrics_summary.png")
        plt.close()
        all_metrics.append(summary)

    # Optionally, aggregate across all runs
    all_metrics_df = pd.concat([pd.DataFrame(m).T for m in all_metrics])
    all_metrics_df.to_csv(results_dir / "all_event_metrics_summary.csv")

if __name__ == "__main__":
    main()