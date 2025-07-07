from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import get_tester
from neuralhydrology.datautils import get_basin_dataset
import pandas as pd
import xarray as xr
import yaml

def debug_validation_filtering():
    """Debug why validation produces 0 basins while test works."""
    
    experiment_folder = Path("Experiments/HPC_random_search/results/job_41780893/run_000/N38_A30_4CPU_SMI_0406_170648")
    config_path = experiment_folder / "config.yml"
    
    print("=== DEBUGGING VALIDATION FILTERING ===")
    
    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply local modifications
    config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan"))
    config_dict['device'] = 'cpu'
    config_dict['run_dir'] = str(experiment_folder.absolute())
    
    # Update basin files
    LOCAL_BASIN_PATH = Path("C:/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static")
    
    for basin_key in ['test_basin_file', 'train_basin_file', 'validation_basin_file']:
        if basin_key in config_dict and config_dict[basin_key]:
            basin_filename = Path(config_dict[basin_key]).name
            config_dict[basin_key] = str(LOCAL_BASIN_PATH / basin_filename)
    
    config = Config(config_dict)
    
    print(f"Sequence length: {config.seq_length}")
    print(f"Validation period: {config.validation_start_date} to {config.validation_end_date}")
    print(f"Test period: {config.test_start_date} to {config.test_end_date}")
    
    # Load basin list
    with open(config.validation_basin_file, 'r') as f:
        basins = [line.strip() for line in f if line.strip()]
    
    print(f"\nChecking {len(basins)} basins individually...")
    
    valid_basins = []
    invalid_basins = []
    
    for i, basin in enumerate(basins):
        print(f"\n--- Basin {i+1}/{len(basins)}: {basin} ---")
        
        try:
            # Load basin data
            data_file = Path(config.data_dir) / "timeseries" / "netcdf" / basin.split('_')[0] / f"{basin}.nc"
            
            if not data_file.exists():
                print(f"❌ Data file not found: {data_file}")
                invalid_basins.append((basin, "Data file missing"))
                continue
            
            ds = xr.open_dataset(data_file)
            dates = pd.to_datetime(ds['date'].values)
            
            print(f"Data available: {dates.min()} to {dates.max()}")
            
            # Check validation period coverage
            val_start = pd.to_datetime(config.validation_start_date)
            val_end = pd.to_datetime(config.validation_end_date)
            
            val_mask = (dates >= val_start) & (dates <= val_end)
            val_points = val_mask.sum()
            
            print(f"Validation period data points: {val_points}")
            
            if val_points == 0:
                print(f"❌ No data in validation period")
                invalid_basins.append((basin, "No data in validation period"))
                ds.close()
                continue
            
            # Check target variable availability
            target_var = config.target_variables[0]  # 'Flow_m3_sec'
            
            if target_var not in ds.variables:
                print(f"❌ Target variable '{target_var}' not found")
                invalid_basins.append((basin, f"Missing target variable {target_var}"))
                ds.close()
                continue
            
            # Check valid target values in validation period
            target_data = ds[target_var].values[val_mask]
            valid_target_points = (~pd.isna(target_data)).sum()
            
            print(f"Valid target values in validation period: {valid_target_points}/{val_points}")
            
            # Check sequence length requirement
            min_required = config.seq_length + 1  # Need seq_length + 1 for sequences
            
            print(f"Minimum required continuous points: {min_required}")
            
            if valid_target_points < min_required:
                print(f"❌ Not enough valid target values ({valid_target_points} < {min_required})")
                invalid_basins.append((basin, f"Insufficient valid data ({valid_target_points} < {min_required})"))
                ds.close()
                continue
            
            # Check for continuous sequences
            valid_mask = ~pd.isna(target_data)
            
            # Find longest continuous sequence
            max_continuous = 0
            current_continuous = 0
            
            for is_valid in valid_mask:
                if is_valid:
                    current_continuous += 1
                    max_continuous = max(max_continuous, current_continuous)
                else:
                    current_continuous = 0
            
            print(f"Longest continuous sequence: {max_continuous}")
            
            if max_continuous < min_required:
                print(f"❌ Longest continuous sequence too short ({max_continuous} < {min_required})")
                invalid_basins.append((basin, f"Continuous sequence too short ({max_continuous} < {min_required})"))
                ds.close()
                continue
            
            # Check dynamic inputs
            missing_inputs = []
            for input_var in config.dynamic_inputs:
                if input_var not in ds.variables:
                    missing_inputs.append(input_var)
            
            if missing_inputs:
                print(f"❌ Missing dynamic inputs: {missing_inputs}")
                invalid_basins.append((basin, f"Missing inputs: {missing_inputs}"))
                ds.close()
                continue
            
            print(f"✅ Basin passes all checks")
            valid_basins.append(basin)
            
            ds.close()
            
        except Exception as e:
            print(f"❌ Error processing basin: {e}")
            invalid_basins.append((basin, f"Error: {e}"))
    
    print(f"\n=== SUMMARY ===")
    print(f"Valid basins: {len(valid_basins)}")
    print(f"Invalid basins: {len(invalid_basins)}")
    
    if len(valid_basins) > 0:
        print(f"Valid basins: {valid_basins}")
    
    if len(invalid_basins) > 0:
        print(f"\nInvalid basins and reasons:")
        for basin, reason in invalid_basins:
            print(f"  {basin}: {reason}")
    
    # Now compare with test period for the same basins
    print(f"\n=== COMPARING WITH TEST PERIOD ===")
    
    test_start = pd.to_datetime(config.test_start_date)
    test_end = pd.to_datetime(config.test_end_date)
    
    sample_basin = basins[0]
    data_file = Path(config.data_dir) / "timeseries" / "netcdf" / sample_basin.split('_')[0] / f"{sample_basin}.nc"
    
    ds = xr.open_dataset(data_file)
    dates = pd.to_datetime(ds['date'].values)
    target_var = config.target_variables[0]
    
    # Test period analysis
    test_mask = (dates >= test_start) & (dates <= test_end)
    test_points = test_mask.sum()
    test_target_data = ds[target_var].values[test_mask]
    valid_test_points = (~pd.isna(test_target_data)).sum()
    
    print(f"Test period for {sample_basin}:")
    print(f"  Total points: {test_points}")
    print(f"  Valid target points: {valid_test_points}")
    print(f"  Passes sequence requirement: {valid_test_points >= config.seq_length + 1}")
    
    ds.close()

if __name__ == "__main__":
    debug_validation_filtering()