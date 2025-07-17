import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        output_dir = hydro_dir / "metrics_evaluation_shifted"
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
                event_date = pd.to_datetime(event["max_date"], dayfirst=True)
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
                        "NSE_shifted": np.nan,
                        "pNSE_shifted": np.nan,
                        "PeakFlowError_shifted": np.nan,
                        "VolumeError_shifted": np.nan,
                        "erroneous": True
                    })
                    continue
                event_str = f"{pd.to_datetime(start).strftime('%Y%m%d')}_{pd.to_datetime(end).strftime('%Y%m%d')}"
                event_csv = output_dir / f"{basin_id}_event_{event_str}_shifted.csv"
                event_df.to_csv(event_csv, index=False)
                plt.figure(figsize=(10, 6))
                plt.plot(event_df["date"], event_df["observed"], label="Observed")
                plt.plot(event_df["date"], event_df["shifted"], linestyle=':', label="Observed Shifted (3 hours)")
                plt.title(f"Basin {basin_id} Event {event_str} (Shifted)")
                plt.xlabel("Date")
                plt.ylabel("Discharge [m³/s]")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                fig_file = output_dir / f"{basin_id}_event_{event_str}_shifted.png"
                plt.savefig(fig_file)
                plt.close()
                obs = event_df["observed"].values
                shifted = event_df["shifted"].values
                nse_val = nse(obs, shifted)
                pnse_val = persistent_nse(obs, shifted, lag_steps=18)
                peak_err = peak_flow_error(obs, shifted)
                vol_err = volume_error(obs, shifted)
                run_metrics.append({
                    "run": run_dir.name,
                    "basin": basin_id,
                    "event_date": event_date,
                    "NSE_shifted": nse_val,
                    "pNSE_shifted": pnse_val,
                    "PeakFlowError_shifted": peak_err,
                    "VolumeError_shifted": vol_err,
                    "erroneous": False
                })
        metrics_df = pd.DataFrame(run_metrics)
        metrics_df.to_csv(hydro_dir / "event_metrics_shifted.csv", index=False)
        summary = metrics_df[~metrics_df["erroneous"]].agg({
            "NSE_shifted": ["mean", "std"],
            "pNSE_shifted": ["mean", "std"],
            "PeakFlowError_shifted": ["mean", "std"],
            "VolumeError_shifted": ["mean", "std"]
        })
        summary["erroneous_count"] = metrics_df["erroneous"].sum()
        summary.to_csv(hydro_dir / "event_metrics_summary_shifted.csv")
        plt.figure(figsize=(8, 5))
        metrics_to_plot = ["NSE_shifted", "pNSE_shifted", "PeakFlowError_shifted", "VolumeError_shifted"]
        mean_values = [summary.loc["mean", m] for m in metrics_to_plot if m in summary.columns]
        plt.bar(metrics_to_plot[:len(mean_values)], mean_values)
        plt.ylabel("Mean Metric Value")
        plt.title(f"Run {run_dir.name} Event Metrics (mean, shifted)")
        plt.tight_layout()
        plt.savefig(hydro_dir / "event_metrics_summary_shifted.png")
        plt.close()
        all_metrics.append(summary)

    # Optionally, aggregate across all runs
    all_metrics_df = pd.concat([pd.DataFrame(m).T for m in all_metrics])
    all_metrics_df.to_csv(results_dir / "all_event_metrics_summary_shifted.csv")
    plt.figure(figsize=(8, 5))
    all_mean_values = [all_metrics_df.loc["mean", m] for m in metrics_to_plot if m in all_metrics_df.columns]
    plt.bar(metrics_to_plot[:len(all_mean_values)], all_mean_values)
    plt.ylabel("Mean Metric Value")
    plt.title("All Runs Event Metrics (mean, shifted)")
    plt.tight_layout()
    plt.savefig(results_dir / "all_event_metrics_summary_shifted.png")
    plt.close()

    # create a summary plot for all runs showing the mean and sd values for each metric for each run:
    plt.figure(figsize=(10, 6))
    for metric in metrics_to_plot:
        if metric in all_metrics_df.columns:
            plt.errorbar(
                all_metrics_df.index, 
                all_metrics_df[metric]['mean'], 
                yerr=all_metrics_df[metric]['std'], 
                label=metric, 
                marker='o'
            )
    plt.xlabel("Run")
    plt.ylabel("Metric Value")
    plt.title("Event Metrics Summary Across All Runs (Shifted)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "all_runs_event_metrics_summary_shifted.png")
    plt.close()

    # Further analysis: aggregate and visualize shifted metrics across models
    # (same as your further analysis code, but using _shifted columns and filenames)
    metrics_to_analyze = ["NSE_shifted", "pNSE_shifted", "PeakFlowError_shifted", "VolumeError_shifted"]
    summary_rows = []

    for run_dir in run_dirs:
        hydro_dir = run_dir / "validation_hydrographs_cluster"
        metrics_file = hydro_dir / "event_metrics_shifted.csv"
        if not metrics_file.exists():
            continue
        df = pd.read_csv(metrics_file)
        df = df[~df["erroneous"]]  # Exclude erroneous events
        
        row = {"run": run_dir.name}
        for metric in metrics_to_analyze:
            if metric in df.columns:
                vals = df[metric].dropna()
                if len(vals) > 0:
                    row[f"{metric}_mean"] = vals.mean()
                    row[f"{metric}_median"] = vals.median()
                    row[f"{metric}_std"] = vals.std()
                    row[f"{metric}_p25"] = np.percentile(vals, 25)
                    row[f"{metric}_p75"] = np.percentile(vals, 75)
                    row[f"{metric}_count"] = len(vals)
                else:
                    for stat in ["mean", "median", "std", "p25", "p75", "count"]:
                        row[f"{metric}_{stat}"] = np.nan
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = results_dir / "improved_event_metrics_summary_shifted.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")

    # 1. Separate plots for each shifted metric with proper scaling
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_analyze):
        ax = axes[i]
        data_to_plot = summary_df[f"{metric}_mean"].dropna()
        ax.boxplot([data_to_plot], labels=[f'{metric} Mean'])
        ax.set_title(f'{metric} Distribution Across Models (Shifted)')
        ax.set_ylabel(f'{metric} Value')
        ax.grid(True, alpha=0.3)
        if len(data_to_plot) > 0:
            ax.text(0.02, 0.98, f'Models: {len(data_to_plot)}\nBest: {data_to_plot.max():.3f}\nWorst: {data_to_plot.min():.3f}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(results_dir / "metrics_distribution_by_type_shifted.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Top 10 and Bottom 10 models for each shifted metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_analyze):
        ax = axes[i]
        sorted_df = summary_df.sort_values(f"{metric}_mean", ascending=(metric in ["PeakFlowError_shifted", "VolumeError_shifted"]))
        top_10 = sorted_df.head(10)
        bottom_10 = sorted_df.tail(10)
        x_pos = np.arange(len(top_10))
        ax.bar(x_pos, top_10[f"{metric}_mean"], color='green', alpha=0.7, label='Top 10')
        ax.errorbar(x_pos, top_10[f"{metric}_mean"], yerr=top_10[f"{metric}_std"], 
                   fmt='none', color='black', capsize=3)
        x_pos_bottom = np.arange(len(bottom_10)) + len(top_10) + 1
        ax.bar(x_pos_bottom, bottom_10[f"{metric}_mean"], color='red', alpha=0.7, label='Bottom 10')
        ax.errorbar(x_pos_bottom, bottom_10[f"{metric}_mean"], yerr=bottom_10[f"{metric}_std"], 
                   fmt='none', color='black', capsize=3)
        all_labels = list(top_10['run'].str[-6:]) + [''] + list(bottom_10['run'].str[-6:])
        ax.set_xticks(list(x_pos) + [len(top_10)] + list(x_pos_bottom))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_title(f'{metric}: Best vs Worst Models (Shifted)')
        ax.set_ylabel(f'{metric} Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "top_bottom_models_comparison_shifted.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Ranking matrix for shifted metrics
    ranking_df = pd.DataFrame(index=summary_df['run'])
    for metric in metrics_to_analyze:
        ascending = metric in ["PeakFlowError_shifted", "VolumeError_shifted"]
        ranking_df[f'{metric}_rank'] = summary_df[f"{metric}_mean"].rank(ascending=ascending)
    ranking_df['avg_rank'] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.sort_values('avg_rank')
    plt.figure(figsize=(12, 8))
    sns.heatmap(ranking_df.iloc[:20, :-1], annot=True, fmt='.0f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank (1=Best)'})
    plt.title('Model Rankings Across Shifted Metrics (Top 20 Models)')
    plt.xlabel('Metrics')
    plt.ylabel('Models')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(results_dir / "model_rankings_heatmap_shifted.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Scatter plots to see correlations between shifted metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metric_pairs = [
        ('NSE_shifted', 'pNSE_shifted'), 
        ('NSE_shifted', 'PeakFlowError_shifted'), 
        ('NSE_shifted', 'VolumeError_shifted'),
        ('pNSE_shifted', 'PeakFlowError_shifted'), 
        ('pNSE_shifted', 'VolumeError_shifted'), 
        ('PeakFlowError_shifted', 'VolumeError_shifted')
    ]
    for i, (metric1, metric2) in enumerate(metric_pairs):
        ax = axes[i//3, i%3]
        x = summary_df[f"{metric1}_mean"]
        y = summary_df[f"{metric2}_mean"]
        ax.scatter(x, y, alpha=0.7, s=50)
        ax.set_xlabel(f'{metric1} Mean')
        ax.set_ylabel(f'{metric2} Mean')
        ax.set_title(f'{metric1} vs {metric2} (Shifted)')
        ax.grid(True, alpha=0.3)
        corr = np.corrcoef(x.dropna(), y.dropna())[0, 1] if len(x.dropna()) > 1 else np.nan
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(results_dir / "metrics_correlation_plots_shifted.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Summary table of best models for shifted metrics
    best_models = pd.DataFrame()
    for metric in metrics_to_analyze:
        ascending = metric in ["PeakFlowError_shifted", "VolumeError_shifted"]
        best_model = summary_df.loc[summary_df[f"{metric}_mean"].idxmax() if not ascending 
                                   else summary_df[f"{metric}_mean"].idxmin()]
        best_models[metric] = [best_model['run'], best_model[f"{metric}_mean"]]
    best_models.index = ['Model', 'Value']
    best_models.to_csv(results_dir / "best_models_per_metric_shifted.csv")
    overall_best = ranking_df.index[0]
    print(f"\nOverall best model (lowest average rank, shifted): {overall_best}")
    print(f"Average rank: {ranking_df.loc[overall_best, 'avg_rank']:.2f}")

    return summary_df, ranking_df

if __name__ == "__main__":
    summary_df, ranking_df = main()