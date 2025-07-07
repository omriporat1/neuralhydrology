from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.evaluate import start_evaluation
import pickle
import pandas as pd
import yaml

def test_corrected_validation():
    """Test validation evaluation with corrected date format."""
    
    corrected_config_path = Path("Experiments/HPC_random_search/corrected_config.yml")
    experiment_folder = Path("Experiments/HPC_random_search/results/job_41780893/run_000/N38_A30_4CPU_SMI_0406_170648")
    
    if not corrected_config_path.exists():
        print("Please run compare_test_vs_validation.py first to create corrected config")
        return
    
    print("=== TESTING CORRECTED VALIDATION CONFIG ===")
    
    # Load corrected config as dictionary (Config class expects dict, not file path)
    with open(corrected_config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = Config(config_dict)
    
    print(f"Using corrected validation dates:")
    print(f"  Start: {config.validation_start_date}")
    print(f"  End: {config.validation_end_date}")
    
    # Set up PyTorch for CPU
    import torch
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Run validation evaluation
    print("Starting validation evaluation with corrected config...")
    start_evaluation(cfg=config, run_dir=experiment_folder, epoch=None, period="validation")
    
    # Check results
    validation_results_files = list(experiment_folder.glob("validation/model_epoch*/validation_results.p"))
    
    if validation_results_files:
        latest_results_file = sorted(validation_results_files)[-1]
        print(f"Loading results from: {latest_results_file}")
        
        with open(latest_results_file, 'rb') as f:
            results = pickle.load(f)
        
        if results and len(results) > 0:
            print(f"SUCCESS! Validation evaluation produced {len(results)} basins")
            
            # Show sample structure (should match test structure)
            sample_basin = list(results.keys())[0]
            sample_data = results[sample_basin]
            print(f"Sample basin: {sample_basin}")
            print(f"Sample data keys: {list(sample_data.keys())}")
            
            if '10min' in sample_data and 'xr' in sample_data['10min']:
                xr_data = sample_data['10min']['xr']
                df = xr_data.to_dataframe().reset_index()
                print(f"DataFrame columns: {list(df.columns)}")
                print(f"Data shape: {df.shape}")
                print(f"Date range: {df['date'].min()} to {df['date'].max()}")
                
                # Save sample CSV to verify structure
                output_dir = Path("Experiments/HPC_random_search/validation_test")
                output_dir.mkdir(exist_ok=True)
                
                csv_file = output_dir / f"{sample_basin}_validation_hydrograph.csv"
                df_output = pd.DataFrame({
                    'datetime': pd.to_datetime(df['date']),
                    'observed': df['Flow_m3_sec_obs'].values,
                    'predicted': df['Flow_m3_sec_sim'].values
                })
                df_output.to_csv(csv_file, index=False)
                print(f"Saved sample CSV: {csv_file}")
                print(f"CSV has {len(df_output)} time steps")
                
            return True
        else:
            print("Results file is still empty")
            return False
    else:
        print("No validation results files found")
        return False

if __name__ == "__main__":
    success = test_corrected_validation()
    if success:
        print("\n🎉 VALIDATION EVALUATION WORKING!")
        print("Now you can apply this fix to all experiments")