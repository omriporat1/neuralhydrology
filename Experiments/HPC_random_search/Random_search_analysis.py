import pandas as pd
import numpy as np
from pathlib import Path


def analyze_random_search():
    # Read SLURM job stats
    slurm_stats = pd.read_csv(r'C:\PhD\Python\neuralhydrology\Experiments\HPC_random_search\results\job_41780893\slurm_job_stat.csv')

    # Read hyperparameter configurations
    configs = pd.read_csv(r'C:\PhD\Python\neuralhydrology\Experiments\HPC_random_search\random_search_configurations.csv')

    # Process job completion times
    slurm_stats['Duration'] = pd.to_timedelta(slurm_stats['Elapsed'])
    slurm_stats['Hours'] = slurm_stats['Duration'].dt.total_seconds() / 3600

    # Basic statistics
    completed_jobs = slurm_stats[slurm_stats['State'] == 'COMPLETED']
    timeout_jobs = slurm_stats[slurm_stats['State'] == 'TIMEOUT']

    print(f"Total jobs: {len(slurm_stats)}")
    print(f"Completed jobs: {len(completed_jobs)}")
    print(f"Timeout jobs: {len(timeout_jobs)}")
    print(f"Average runtime: {completed_jobs['Hours'].mean():.2f} hours")
    print(f"Min runtime: {completed_jobs['Hours'].min():.2f} hours")
    print(f"Max runtime: {completed_jobs['Hours'].max():.2f} hours")

    # Merge with configurations if needed
    if len(configs) == len(slurm_stats):
        results = pd.concat([configs, slurm_stats['State']], axis=1)
        success_rate = results.groupby(configs.columns.tolist())['State'].agg(
            lambda x: (x == 'COMPLETED').mean()
        ).sort_values(ascending=False)

        print(r"Top 5 most successful configurations:")
        print(success_rate.head())

    return slurm_stats, configs


if __name__ == "__main__":
    stats, configs = analyze_random_search()
