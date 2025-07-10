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

def process_single_run():
    try:
        print("Starting to process a single run...")
        # Use absolute paths
        run_dir = Path("C:/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results/job_41780893/run_000/runs")
        
        # Find the run directory inside
        if run_dir.exists():
            print(f"Run directory exists: {run_dir}")
            run_subdirs = list(run_dir.glob("*/"))
            if run_subdirs:
                print(f"Found {len(run_subdirs)} run subdirectories")
                run_subdir = run_subdirs[0]
                print(f"Using first run subdir: {run_subdir}")
                
                # Check for config.yml
                config_path = run_subdir / "config.yml"
                if config_path.exists():
                    print(f"Config file exists: {config_path}")
                    
                    # Load the configuration
                    with open(config_path, 'r') as f:
                        config_dict = yaml.safe_load(f)
                    
                    # Modify paths in the dictionary
                    config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan"))
                    config_dict['device'] = 'cpu'
                    
                    # IMPORTANT: Set the run_dir to the absolute local path
                    config_dict['run_dir'] = str(run_subdir.absolute())
                    
                    # Create new config from modified dictionary
                    config = Config(config_dict)
                    
                    # Try to get tester
                    print("Creating tester...")
                    tester = get_tester(cfg=config, run_dir=run_subdir, period="validation", init_model=True)
                    
                    # Try to evaluate
                    print("Evaluating model...")
                    results = tester.evaluate(save_results=False, metrics=config.metrics)
                    
                    print(f"Evaluation complete. Got results for {len(results.keys())} basins")
                    
                    return "Success!"
                else:
                    print(f"Config file not found at {config_path}")
            else:
                print(f"No run subdirectories found in {run_dir}")
        else:
            print(f"Run directory does not exist: {run_dir}")
        
        return "Failed to process run"
    
    except Exception as e:
        print(f"Error in process_single_run: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

if __name__ == "__main__":
    result = process_single_run()
    print(f"Final result: {result}")
