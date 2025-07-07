from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.datasetzoo.caravan import Caravan
from neuralhydrology.datautils.utils import load_scaler
import yaml

def simple_debug():
    """Simple debugging approach using Caravan dataset directly."""
    
    experiment_folder = Path("Experiments/HPC_random_search/results/job_41780893/run_000/N38_A30_4CPU_SMI_0406_170648")
    config_path = experiment_folder / "config.yml"
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply local modifications
    config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan"))
    config_dict['device'] = 'cpu'
    config_dict['run_dir'] = str(experiment_folder.absolute())
    
    # Update basin files
    LOCAL_BASIN_PATH = Path("C:/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static")
    config_dict['validation_basin_file'] = str(LOCAL_BASIN_PATH / "il_basins_high_qual_0_04_N38.txt")
    
    config = Config(config_dict)
    
    print("=== SIMPLE DATASET DEBUG ===")
    print(f"Dataset: {config.dataset}")
    print(f"Sequence length: {config.seq_length}")
    print(f"Validation period: {config.validation_start_date} to {config.validation_end_date}")
    
    try:
        # Load the scaler from the training run
        print("Loading scaler from training data...")
        scaler = load_scaler(experiment_folder)
        print(f"Scaler loaded successfully with keys: {list(scaler.keys())}")
        
        # Create Caravan dataset for validation
        print("Creating Caravan validation dataset...")
        dataset = Caravan(cfg=config, is_train=False, period="validation", scaler=scaler)
        
        print(f"Dataset created successfully!")
        print(f"Dataset length: {len(dataset)}")
        print(f"Dataset basins: {getattr(dataset, 'basins', 'No basins')}")
        
        if len(dataset) == 0:
            print("❌ Dataset is empty!")
            
            # Now let's debug why it's empty
            print(f"Dataset period: {getattr(dataset, 'period', 'No period')}")
            
            # Check sample basin data loading
            with open(config.validation_basin_file, 'r') as f:
                basins = [line.strip() for line in f if line.strip()]
            
            sample_basin = basins[0]
            print(f"\nTesting manual data loading for basin: {sample_basin}")
            
            from neuralhydrology.datasetzoo.caravan import load_caravan_timeseries
            
            try:
                df = load_caravan_timeseries(Path(config.data_dir), sample_basin)
                print(f"Loaded timeseries shape: {df.shape}")
                print(f"Date range: {df.index.min()} to {df.index.max()}")
                print(f"Columns: {list(df.columns)}")
                
                # Check validation period overlap
                val_start = config.validation_start_date
                val_end = config.validation_end_date
                
                val_mask = (df.index >= val_start) & (df.index <= val_end)
                val_data = df[val_mask]
                
                print(f"Validation period data: {len(val_data)} rows")
                
                if len(val_data) > 0:
                    target_var = config.target_variables[0]
                    if target_var in val_data.columns:
                        valid_target = val_data[target_var].notna().sum()
                        print(f"Valid target values in validation period: {valid_target}")
                        print(f"Required for sequence length {config.seq_length}: {config.seq_length + 1}")
                        
                        # Check if there are enough continuous valid values
                        continuous_valid = check_continuous_sequences(val_data[target_var], config.seq_length + 1)
                        print(f"Longest continuous valid sequence: {continuous_valid}")
                        
                        if continuous_valid >= config.seq_length + 1:
                            print("✅ This basin should pass validation requirements")
                        else:
                            print(f"❌ Not enough continuous valid data ({continuous_valid} < {config.seq_length + 1})")
                    else:
                        print(f"❌ Target variable '{target_var}' not found")
                        print(f"Available: {list(val_data.columns)}")
                else:
                    print("❌ No data in validation period")
                
            except Exception as e:
                print(f"❌ Error loading timeseries: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("✅ Dataset has data!")
            
            # Test getting a sample
            try:
                sample = dataset[0]
                print(f"Sample structure: {sample.keys()}")
            except Exception as e:
                print(f"Error getting sample: {e}")
        
        # Compare with test
        print("\n=== TESTING TEST DATASET ===")
        test_dataset = Caravan(cfg=config, is_train=False, period="test", scaler=scaler)
        print(f"Test dataset length: {len(test_dataset)}")
        
        if len(test_dataset) > 0:
            print("✅ Test dataset works - validation should work too")
        else:
            print("❌ Test dataset is also empty")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def check_continuous_sequences(series, min_length):
    """Check the longest continuous sequence of valid (non-NaN) values."""
    valid_mask = series.notna()
    max_continuous = 0
    current_continuous = 0
    
    for is_valid in valid_mask:
        if is_valid:
            current_continuous += 1
            max_continuous = max(max_continuous, current_continuous)
        else:
            current_continuous = 0
    
    return max_continuous

if __name__ == "__main__":
    simple_debug()