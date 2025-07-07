from pathlib import Path
import yaml
import pandas as pd
import xarray as xr

def compare_test_vs_validation_config():
    """Compare test vs validation settings to find the difference."""
    
    config_path = Path("Experiments/HPC_random_search/results/job_41780893/run_000/N38_A30_4CPU_SMI_0406_170648/config.yml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== COMPARING TEST VS VALIDATION SETTINGS ===")
    print(f"Test start date: {config.get('test_start_date')}")
    print(f"Test end date: {config.get('test_end_date')}")
    print(f"Validation start date: {config.get('validation_start_date')}")
    print(f"Validation end date: {config.get('validation_end_date')}")
    
    print(f"\nTest basin file: {config.get('test_basin_file')}")
    print(f"Validation basin file: {config.get('validation_basin_file')}")
    
    # Check if validation period is properly defined
    val_start = config.get('validation_start_date')
    val_end = config.get('validation_end_date')
    test_start = config.get('test_start_date')
    test_end = config.get('test_end_date')
    
    print(f"\n=== DATE FORMAT ANALYSIS ===")
    print(f"Validation start format: {val_start} (type: {type(val_start)})")
    print(f"Test start format: {test_start} (type: {type(test_start)})")
    
    # Try to parse the dates
    try:
        val_start_parsed = pd.to_datetime(val_start)
        val_end_parsed = pd.to_datetime(val_end)
        print(f"Validation period parsed: {val_start_parsed} to {val_end_parsed}")
    except Exception as e:
        print(f"ERROR parsing validation dates: {e}")
    
    try:
        test_start_parsed = pd.to_datetime(test_start)
        test_end_parsed = pd.to_datetime(test_end)
        print(f"Test period parsed: {test_start_parsed} to {test_end_parsed}")
    except Exception as e:
        print(f"ERROR parsing test dates: {e}")
    
    # Check data availability for both periods
    basin = "il_12130"
    data_file = f"C:/PhD/Data/Caravan/timeseries/netcdf/{basin.split('_')[0]}/{basin}.nc"
    
    print(f"\n=== DATA AVAILABILITY CHECK ===")
    ds = xr.open_dataset(data_file)
    dates = pd.to_datetime(ds['date'].values)
    
    print(f"Data covers: {dates.min()} to {dates.max()}")
    
    # Check test period overlap
    if test_start and test_end:
        test_mask = (dates >= pd.to_datetime(test_start)) & (dates <= pd.to_datetime(test_end))
        print(f"Test period overlap: {test_mask.sum()} time steps")
    
    # Check validation period overlap
    if val_start and val_end:
        val_mask = (dates >= pd.to_datetime(val_start)) & (dates <= pd.to_datetime(val_end))
        print(f"Validation period overlap: {val_mask.sum()} time steps")
    
    ds.close()
    
    # Show the problematic date formats
    print(f"\n=== POTENTIAL ISSUES ===")
    if val_start and "/" in str(val_start):
        print(f"WARNING: Validation dates use DD/MM/YYYY format: {val_start}")
        print("This might cause parsing issues in neuralhydrology")
    
    if test_start and "-" in str(test_start):
        print(f"Test dates use YYYY-MM-DD format: {test_start}")
        print("This is the standard format")

def create_corrected_validation_config():
    """Create a version that runs validation with corrected date format."""
    
    config_path = Path("Experiments/HPC_random_search/results/job_41780893/run_000/N38_A30_4CPU_SMI_0406_170648/config.yml")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    print("=== CREATING CORRECTED CONFIG ===")
    
    # Fix the date format issue
    original_val_start = config_dict.get('validation_start_date')
    original_val_end = config_dict.get('validation_end_date')
    
    # Convert DD/MM/YYYY to YYYY-MM-DD
    if original_val_start and "/" in str(original_val_start):
        # Parse DD/MM/YYYY and convert to YYYY-MM-DD
        val_start_fixed = pd.to_datetime(original_val_start, format='%d/%m/%Y').strftime('%Y-%m-%d')
        val_end_fixed = pd.to_datetime(original_val_end, format='%d/%m/%Y').strftime('%Y-%m-%d')
        
        print(f"Fixed validation start: {original_val_start} -> {val_start_fixed}")
        print(f"Fixed validation end: {original_val_end} -> {val_end_fixed}")
        
        # Apply local modifications
        config_dict['validation_start_date'] = val_start_fixed
        config_dict['validation_end_date'] = val_end_fixed
        config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan"))
        config_dict['device'] = 'cpu'
        config_dict['run_dir'] = str(config_path.parent.absolute())
        
        # Update basin files
        LOCAL_BASIN_PATH = Path("C:/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static")
        
        for basin_key in ['test_basin_file', 'train_basin_file', 'validation_basin_file']:
            if basin_key in config_dict and config_dict[basin_key]:
                basin_filename = Path(config_dict[basin_key]).name
                config_dict[basin_key] = str(LOCAL_BASIN_PATH / basin_filename)
        
        # Save corrected config
        corrected_config_path = Path("Experiments/HPC_random_search/corrected_config.yml")
        with open(corrected_config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        print(f"Saved corrected config to: {corrected_config_path}")
        return corrected_config_path
    
    return None

if __name__ == "__main__":
    compare_test_vs_validation_config()
    corrected_config = create_corrected_validation_config()
    
    if corrected_config:
        print(f"\nNext step: Run validation evaluation with corrected config:")
        print(f"Use the config file: {corrected_config}")