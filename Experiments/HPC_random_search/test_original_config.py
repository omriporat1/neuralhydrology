from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.evaluate import start_evaluation
import pickle
import pandas as pd
import yaml

def test_original_config():
    """Test validation evaluation with the original config (DD/MM/YYYY format)."""
    
    experiment_folder = Path("Experiments/HPC_random_search/results/job_41780893/run_000/N38_A30_4CPU_SMI_0406_170648")
    config_path = experiment_folder / "config.yml"
    
    print("=== TESTING ORIGINAL CONFIG WITH LOCAL PATHS ===")
    
    # Load original config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    print(f"Original validation dates (DD/MM/YYYY format):")
    print(f"  Start: {config_dict['validation_start_date']}")
    print(f"  End: {config_dict['validation_end_date']}")
    
    # Only modify paths, keep original dates
    config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan"))
    config_dict['device'] = 'cpu'
    config_dict['run_dir'] = str(experiment_folder.absolute())
    
    # Update basin files to local paths
    LOCAL_BASIN_PATH = Path("C:/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static")
    
    for basin_key in ['test_basin_file', 'train_basin_file', 'validation_basin_file']:
        if basin_key in config_dict and config_dict[basin_key]:
            basin_filename = Path(config_dict[basin_key]).name
            config_dict[basin_key] = str(LOCAL_BASIN_PATH / basin_filename)
            print(f"Updated {basin_key}: {config_dict[basin_key]}")
    
    # Create config (should work with DD/MM/YYYY format)
    config = Config(config_dict)
    
    print(f"Config created successfully!")
    print(f"Parsed validation dates:")
    print(f"  Start: {config.validation_start_date}")
    print(f"  End: {config.validation_end_date}")
    
    # Check if validation basin file exists
    val_basin_file = Path(config.validation_basin_file)
    if not val_basin_file.exists():
        print(f"ERROR: Validation basin file not found: {val_basin_file}")
        return False
    
    # Load validation basins
    with open(val_basin_file, 'r') as f:
        basins = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(basins)} validation basins")
    
    # Check a sample basin data file
    sample_basin = basins[0]
    data_file = Path(config.data_dir) / "timeseries" / "netcdf" / sample_basin.split('_')[0] / f"{sample_basin}.nc"
    print(f"Sample data file: {data_file}")
    
    if not data_file.exists():
        print(f"ERROR: Sample data file not found: {data_file}")
        return False
    
    print("All files exist - proceeding with evaluation...")
    
    # Set up PyTorch for CPU
    import torch
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Run validation evaluation
    print("Starting validation evaluation...")
    try:
        start_evaluation(cfg=config, run_dir=experiment_folder, epoch=None, period="validation")
        print("Evaluation completed!")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check results
    validation_results_files = list(experiment_folder.glob("validation/model_epoch*/validation_results.p"))
    
    if validation_results_files:
        latest_results_file = sorted(validation_results_files)[-1]
        print(f"Loading results from: {latest_results_file}")
        
        with open(latest_results_file, 'rb') as f:
            results = pickle.load(f)
        
        if results and len(results) > 0:
            print(f"🎉 SUCCESS! Validation evaluation produced {len(results)} basins")
            
            # Show sample structure
            sample_basin_id = list(results.keys())[0]
            sample_data = results[sample_basin_id]
            print(f"Sample basin: {sample_basin_id}")
            print(f"Sample data keys: {list(sample_data.keys())}")
            
            if '10min' in sample_data and 'xr' in sample_data['10min']:
                xr_data = sample_data['10min']['xr']
                df = xr_data.to_dataframe().reset_index()
                print(f"DataFrame columns: {list(df.columns)}")
                print(f"Data shape: {df.shape}")
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                
                # Save sample CSV
                output_dir = Path("Experiments/HPC_random_search/validation_test")
                output_dir.mkdir(exist_ok=True)
                
                csv_file = output_dir / f"{sample_basin_id}_validation_hydrograph.csv"
                df_output = pd.DataFrame({
                    'datetime': pd.to_datetime(df['date']),
                    'observed': df['Flow_m3_sec_obs'].values,
                    'predicted': df['Flow_m3_sec_sim'].values
                })
                df_output.to_csv(csv_file, index=False)
                print(f"✅ Saved sample CSV: {csv_file}")
                print(f"CSV has {len(df_output)} time steps")
                
            return True
        else:
            print("❌ Results file is still empty")
            
            # Debug: Check the CSV metrics file instead
            metrics_files = list(experiment_folder.glob("validation/model_epoch*/validation_metrics.csv"))
            if metrics_files:
                latest_metrics = sorted(metrics_files)[-1]
                print(f"Checking metrics file: {latest_metrics}")
                
                try:
                    metrics_df = pd.read_csv(latest_metrics)
                    print(f"Metrics shape: {metrics_df.shape}")
                    if len(metrics_df) > 0:
                        print("Metrics exist but pickle results are empty - there might be an evaluation issue")
                    else:
                        print("Both metrics and results are empty")
                except Exception as e:
                    print(f"Error reading metrics: {e}")
            
            return False
    else:
        print("❌ No validation results files found")
        return False

if __name__ == "__main__":
    success = test_original_config()
    if success:
        print("\n🎉 VALIDATION WORKING WITH ORIGINAL CONFIG!")
        print("The issue was just the local paths, not the date format")
    else:
        print("\n❌ Still not working - need further investigation")