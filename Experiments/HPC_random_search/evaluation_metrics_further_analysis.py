import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def main():
    results_dir = Path("c:/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results")
    run_dirs = [d for d in results_dir.glob("job_*/run_*/*") if d.is_dir()]
    metrics_to_analyze = ["NSE", "pNSE", "PeakFlowError", "VolumeError"]

    summary_rows = []

    for run_dir in run_dirs:
        hydro_dir = run_dir / "validation_hydrographs_cluster"
        metrics_file = hydro_dir / "event_metrics.csv"
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
    summary_csv = results_dir / "improved_event_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")

    # 1. Separate plots for each metric with proper scaling
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_analyze):
        ax = axes[i]
        
        # Box plot showing distribution of means across models
        data_to_plot = summary_df[f"{metric}_mean"].dropna()
        ax.boxplot([data_to_plot], labels=[f'{metric} Mean'])
        ax.set_title(f'{metric} Distribution Across Models')
        ax.set_ylabel(f'{metric} Value')
        ax.grid(True, alpha=0.3)
        
        # Add some statistics as text
        if len(data_to_plot) > 0:
            ax.text(0.02, 0.98, f'Models: {len(data_to_plot)}\nBest: {data_to_plot.max():.3f}\nWorst: {data_to_plot.min():.3f}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(results_dir / "metrics_distribution_by_type.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Top 10 and Bottom 10 models for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_analyze):
        ax = axes[i]
        
        # Sort by metric mean and get top/bottom 10
        sorted_df = summary_df.sort_values(f"{metric}_mean", ascending=(metric in ["PeakFlowError", "VolumeError"]))
        top_10 = sorted_df.head(10)
        bottom_10 = sorted_df.tail(10)
        
        # Plot top 10 in green, bottom 10 in red
        x_pos = np.arange(len(top_10))
        ax.bar(x_pos, top_10[f"{metric}_mean"], color='green', alpha=0.7, label='Top 10')
        ax.errorbar(x_pos, top_10[f"{metric}_mean"], yerr=top_10[f"{metric}_std"], 
                   fmt='none', color='black', capsize=3)
        
        x_pos_bottom = np.arange(len(bottom_10)) + len(top_10) + 1
        ax.bar(x_pos_bottom, bottom_10[f"{metric}_mean"], color='red', alpha=0.7, label='Bottom 10')
        ax.errorbar(x_pos_bottom, bottom_10[f"{metric}_mean"], yerr=bottom_10[f"{metric}_std"], 
                   fmt='none', color='black', capsize=3)
        
        # Formatting
        all_labels = list(top_10['run'].str[-6:]) + [''] + list(bottom_10['run'].str[-6:])
        ax.set_xticks(list(x_pos) + [len(top_10)] + list(x_pos_bottom))
        ax.set_xticklabels(all_labels, rotation=45, ha='right')
        ax.set_title(f'{metric}: Best vs Worst Models')
        ax.set_ylabel(f'{metric} Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / "top_bottom_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Ranking matrix - shows which models are consistently good
    ranking_df = pd.DataFrame(index=summary_df['run'])
    for metric in metrics_to_analyze:
        # Rank models (1 = best)
        ascending = metric in ["PeakFlowError", "VolumeError"]  # Lower is better for these
        ranking_df[f'{metric}_rank'] = summary_df[f"{metric}_mean"].rank(ascending=ascending)
    
    # Calculate average rank
    ranking_df['avg_rank'] = ranking_df.mean(axis=1)
    ranking_df = ranking_df.sort_values('avg_rank')
    
    # Plot ranking heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(ranking_df.iloc[:20, :-1], annot=True, fmt='.0f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Rank (1=Best)'})
    plt.title('Model Rankings Across Metrics (Top 20 Models)')
    plt.xlabel('Metrics')
    plt.ylabel('Models')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(results_dir / "model_rankings_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Scatter plots to see correlations between metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    metric_pairs = [('NSE', 'pNSE'), ('NSE', 'PeakFlowError'), ('NSE', 'VolumeError'),
                   ('pNSE', 'PeakFlowError'), ('pNSE', 'VolumeError'), ('PeakFlowError', 'VolumeError')]
    
    for i, (metric1, metric2) in enumerate(metric_pairs):
        ax = axes[i//3, i%3]
        
        x = summary_df[f"{metric1}_mean"]
        y = summary_df[f"{metric2}_mean"]
        
        ax.scatter(x, y, alpha=0.7, s=50)
        ax.set_xlabel(f'{metric1} Mean')
        ax.set_ylabel(f'{metric2} Mean')
        ax.set_title(f'{metric1} vs {metric2}')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(x.dropna(), y.dropna())[0, 1] if len(x.dropna()) > 1 else np.nan
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(results_dir / "metrics_correlation_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Summary table of best models
    best_models = pd.DataFrame()
    for metric in metrics_to_analyze:
        ascending = metric in ["PeakFlowError", "VolumeError"]
        best_model = summary_df.loc[summary_df[f"{metric}_mean"].idxmax() if not ascending 
                                   else summary_df[f"{metric}_mean"].idxmin()]
        best_models[metric] = [best_model['run'], best_model[f"{metric}_mean"]]
    
    best_models.index = ['Model', 'Value']
    best_models.to_csv(results_dir / "best_models_per_metric.csv")
    
    # Overall best model (by average rank)
    overall_best = ranking_df.index[0]
    print(f"\nOverall best model (lowest average rank): {overall_best}")
    print(f"Average rank: {ranking_df.loc[overall_best, 'avg_rank']:.2f}")
    
    return summary_df, ranking_df

if __name__ == "__main__":
    summary_df, ranking_df = main()