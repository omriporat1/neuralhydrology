import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import get_tester
from neuralhydrology.datautils.utils import load_basin_file
import logging
import xarray as xr

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_experiment_folders(base_dir):
    """Find all experiment folders containing config.yml files and model files."""
    experiment_folders = []
    base_path = Path(base_dir)
    
    for item in base_path.rglob("config.yml"):
        config_dir = item.parent
        
        # Check if this directory contains model files (indicating it's the actual experiment directory)
        model_files = list(config_dir.glob("model_epoch*.pt"))
        
        if model_files:
            experiment_folders.append(config_dir)
            logger.debug(f"Found experiment directory with {len(model_files)} model files: {config_dir}")
        else:
            logger.debug(f"Skipping config.yml without model files: {config_dir}")
    
    return experiment_folders

def find_results_pickle_files(experiment_folder):
    """Find existing results pickle files in the experiment folder."""
    # Look for test results
    test_files = list(experiment_folder.glob("test/model_epoch*/test_results.p"))
    
    # Look for validation results
    val_files = list(experiment_folder.glob("validation/model_epoch*/validation_results.p"))
    
    # all_files = test_files + val_files
    all_files = test_files

    if all_files:
        # Get the most recent one (highest epoch number)
        all_files.sort(key=lambda x: int(x.parent.name.split('epoch')[1]))
        return all_files[-1]  # Return the latest epoch
    
    return None

def has_valid_pickle_results(experiment_folder):
    """Check if experiment has valid (non-empty) pickle results."""
    results_file = find_results_pickle_files(experiment_folder)
    
    if not results_file:
        logger.debug(f"No pickle file found for {experiment_folder.name}")
        return False
    
    logger.debug(f"Checking pickle file: {results_file}")
    
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        logger.debug(f"Loaded pickle for {experiment_folder.name}: {type(results)}, {len(results) if hasattr(results, '__len__') else 'no length'} entries")
        
        # Check if results actually contain data
        if results and len(results) > 0:
            logger.debug(f"Results has {len(results)} entries, checking content...")
            
            # Additional check: make sure at least one basin has actual data
            valid_basins = 0
            for basin_id, basin_data in results.items():
                if basin_data:  # If any basin has data
                    logger.debug(f"Basin {basin_id} has data: {type(basin_data)}")
                    if isinstance(basin_data, dict):
                        logger.debug(f"  Basin data keys: {list(basin_data.keys())}")
                    valid_basins += 1
                    break  # We only need to find one valid basin
            
            if valid_basins > 0:
                logger.debug(f"Found {valid_basins} valid basins in {experiment_folder.name}")
                return True
            else:
                logger.debug(f"All basins are empty in {experiment_folder.name}")
                return False
        else:
            logger.debug(f"Results dict is empty for {experiment_folder.name}")
            return False
            
    except Exception as e:
        logger.debug(f"Error checking pickle file {results_file}: {e}")
        return False

def has_existing_csv_results(experiment_folder):
    """Check if experiment already has CSV results generated."""
    results_dir = experiment_folder / "test_results"
    if results_dir.exists():
        csv_files = list(results_dir.glob("*.csv"))
        return len(csv_files) > 0
    return False

def get_last_epoch_number(run_dir):
    """Get the last epoch number from model files."""
    model_files = list(run_dir.glob("model_epoch*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {run_dir}")
    
    # Extract epoch numbers and return the maximum
    epochs = []
    for model_file in model_files:
        epoch_str = model_file.stem.split("epoch")[-1]
        epochs.append(int(epoch_str))
    
    return max(epochs)

def debug_data_availability(config):
    """Debug what data is actually available for the configured period and basins."""
    logger.info("=== DETAILED DATA DEBUGGING ===")
    
    # Get validation basins
    try:
        val_basins = load_basin_file(config.validation_basin_file)
        logger.info(f"Validation basins from config: {len(val_basins)} basins")
        logger.info(f"First 5 basins: {val_basins[:5]}")
    except Exception as e:
        logger.error(f"Could not load validation basin file: {e}")
        return
    
    # Check data availability for a sample basin
    sample_basin = val_basins[0]
    data_file = Path(config.data_dir) / "timeseries" / "netcdf" / "il" / f"{sample_basin}.nc"
    
    if data_file.exists():
        logger.info(f"Checking data for sample basin: {sample_basin}")
        
        try:
            # Load the NetCDF file
            ds = xr.open_dataset(data_file)
            logger.info(f"NetCDF variables: {list(ds.variables.keys())}")
            
            # Check date range
            if 'date' in ds.variables:
                dates = pd.to_datetime(ds['date'].values)
                logger.info(f"Data date range: {dates.min()} to {dates.max()}")
                
                # Check validation period overlap
                val_start = pd.to_datetime(f"{config.validation_start_date}")
                val_end = pd.to_datetime(f"{config.validation_end_date}")
                
                logger.info(f"Validation period: {val_start} to {val_end}")
                
                # Check if there's overlap
                overlap_mask = (dates >= val_start) & (dates <= val_end)
                overlap_count = overlap_mask.sum()
                
                logger.info(f"Overlapping time steps: {overlap_count}")
                
                if overlap_count > 0:
                    logger.info(f"Overlap range: {dates[overlap_mask].min()} to {dates[overlap_mask].max()}")
                else:
                    logger.warning("NO OVERLAP between validation period and available data!")
                
                # Check for discharge data
                discharge_vars = [var for var in ds.variables if 'flow' in var.lower() or 'discharge' in var.lower() or 'q' in var.lower()]
                logger.info(f"Potential discharge variables: {discharge_vars}")
                
                # Check for missing data in validation period
                if overlap_count > 0 and discharge_vars:
                    for var in discharge_vars:
                        var_data = ds[var].values[overlap_mask]
                        valid_count = (~pd.isna(var_data)).sum()
                        logger.info(f"Variable {var}: {valid_count}/{overlap_count} valid values in validation period")
            
            ds.close()
            
        except Exception as e:
            logger.error(f"Error reading NetCDF file: {e}")
    else:
        logger.error(f"Data file not found: {data_file}")
    
    logger.info("=== END DATA DEBUGGING ===")
    
    

def save_basin_hydrographs(results, output_dir):
    """Save individual basin hydrographs as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    if not results or len(results) == 0:
        logger.warning("No results to save")
        return saved_count
    
    logger.info(f"Attempting to save hydrographs for {len(results)} basins to {output_path}")
    
    for basin_id, basin_data in results.items():
        try:
            logger.debug(f"Processing basin {basin_id}")
            
            # Initialize variables
            observed = None
            predicted = None
            dates = None
            
            # Handle different possible data structures
            if isinstance(basin_data, dict):
                logger.debug(f"Basin data keys: {list(basin_data.keys())}")
                
                # Check for frequency-based structure (e.g., '10min', '1D', '1H', etc.)
                freq_keys = [key for key in basin_data.keys() if any(char.isdigit() for char in key)]
                if freq_keys:
                    freq_key = freq_keys[0]  # Use the first frequency
                    logger.debug(f"Using frequency key: {freq_key}")
                    freq_data = basin_data[freq_key]
                    
                    if isinstance(freq_data, dict) and 'xr' in freq_data:
                        logger.debug("Found xarray data in frequency structure")
                        xr_data = freq_data['xr']
                        logger.debug(f"Xarray data variables: {list(xr_data.variables.keys())}")
                        
                        # Convert to DataFrame
                        df = xr_data.to_dataframe().reset_index()
                        logger.debug(f"DataFrame columns: {list(df.columns)}")
                        
                        # Extract time series data
                        if 'date' in df.columns:
                            dates = pd.to_datetime(df['date'])
                        
                        # Look for observed and predicted columns in neuralhydrology format
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'qobs' in col_lower or ('obs' in col_lower and 'flow' in col_lower):
                                observed = df[col].values
                                logger.debug(f"Found observed data in column: {col}")
                            elif 'qsim' in col_lower or ('sim' in col_lower and 'flow' in col_lower):
                                predicted = df[col].values
                                logger.debug(f"Found predicted data in column: {col}")
                
                # Check for direct xarray structure (backup)
                elif 'xr' in basin_data:
                    logger.debug("Found direct xarray structure")
                    xr_data = basin_data['xr']
                    df = xr_data.to_dataframe().reset_index()
                    
                    # Extract time series data
                    if 'date' in df.columns:
                        dates = pd.to_datetime(df['date'])
                    
                    # Look for observed and predicted columns
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'qobs' in col_lower or ('obs' in col_lower and 'flow' in col_lower):
                            observed = df[col].values
                            logger.debug(f"Found observed data in column: {col}")
                        elif 'qsim' in col_lower or ('sim' in col_lower and 'flow' in col_lower):
                            predicted = df[col].values
                            logger.debug(f"Found predicted data in column: {col}")
                
                # Check for legacy frequency structure
                elif any(key in ['1D', '1H', '3H', '6H', '12H', '24H', '10min'] for key in basin_data.keys()):
                    freq_key = next(key for key in basin_data.keys() if key in ['1D', '1H', '3H', '6H', '12H', '24H', '10min'])
                    freq_data = basin_data[freq_key]
                    
                    if 'xr' in freq_data:
                        xr_data = freq_data['xr']
                        df = xr_data.to_dataframe().reset_index()
                        
                        if 'date' in df.columns:
                            dates = pd.to_datetime(df['date'])
                        
                        for col in df.columns:
                            col_lower = col.lower()
                            if 'qobs' in col_lower or ('obs' in col_lower and 'flow' in col_lower):
                                observed = df[col].values
                                logger.debug(f"Found observed data in column: {col}")
                            elif 'qsim' in col_lower or ('sim' in col_lower and 'flow' in col_lower):
                                predicted = df[col].values
                                logger.debug(f"Found predicted data in column: {col}")
            
            # Create DataFrame if we have the required data
            if observed is not None and predicted is not None:
                # Ensure arrays are 1D
                observed = np.array(observed).flatten()
                predicted = np.array(predicted).flatten()
                
                logger.debug(f"Data shapes - Observed: {observed.shape}, Predicted: {predicted.shape}")
                
                # Create DataFrame
                data_dict = {
                    'observed': observed,
                    'predicted': predicted
                }
                
                # Add dates if available
                if dates is not None:
                    dates = pd.to_datetime(np.array(dates).flatten())
                    data_dict['datetime'] = dates
                    logger.debug(f"Date range: {dates.min()} to {dates.max()}")
                else:
                    # Create a simple index if no dates
                    data_dict['time_step'] = range(len(observed))
                    logger.debug("No dates found - using time step index")
                
                df = pd.DataFrame(data_dict)
                
                # Save to CSV
                csv_file = output_path / f"{basin_id}_hydrograph.csv"
                df.to_csv(csv_file, index=False)
                
                logger.debug(f"Saved hydrograph for {basin_id}: {len(df)} time steps")
                saved_count += 1
                
            else:
                logger.warning(f"Could not extract observed/predicted data for basin {basin_id}")
                # More detailed debugging for first few failures
                if saved_count < 3 and isinstance(basin_data, dict):
                    logger.warning(f"Available keys for {basin_id}: {list(basin_data.keys())}")
                    
                    # Try to explore the frequency structure more deeply
                    for key, value in basin_data.items():
                        if isinstance(value, dict):
                            logger.warning(f"  {key} contains: {list(value.keys())}")
                            if 'xr' in value:
                                xr_data = value['xr']
                                logger.warning(f"    xr variables: {list(xr_data.variables.keys())}")
                
        except Exception as e:
            logger.error(f"Error saving hydrograph for basin {basin_id}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    logger.info(f"Successfully saved {saved_count} hydrographs")
    return saved_count

def extract_hydrographs_from_pickle(experiment_folder):
    """Extract hydrographs from existing pickle files without re-running evaluation."""
    logger.info(f"Processing existing results for: {experiment_folder.name}")
    
    # Find existing results pickle file
    results_file = find_results_pickle_files(experiment_folder)
    
    if not results_file:
        logger.warning(f"No results pickle file found in {experiment_folder}")
        return False
    
    logger.info(f"Found results file: {results_file}")
    
    try:
        # Load the results
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        
        logger.info(f"Loaded results: {type(results)}, {len(results)} entries")
        
        if not results or len(results) == 0:
            logger.warning("Results file is empty")
            return False
        
        # Create output directory
        output_dir = experiment_folder / "test_results"
        
        # Save individual basin CSV files
        saved_count = save_basin_hydrographs(results, output_dir)
        
        if saved_count > 0:
            logger.info(f"Successfully saved {saved_count} basin hydrographs to {output_dir}")
            return True
        else:
            logger.warning("No hydrographs were saved")
            return False
            
    except Exception as e:
        logger.error(f"Error processing pickle file {results_file}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def run_full_evaluation(experiment_folder):
    """Run full evaluation for experiments that don't have results yet."""
    try:
        config_path = experiment_folder / "config.yml"
        if not config_path.exists():
            logger.warning(f"No config.yml found in {experiment_folder}")
            return False
        
        # Load configuration
        config = Config(config_path)

        # User-defined local modifications (set RUN_LOCALLY to True for local execution)
        RUN_LOCALLY = True  # Set to False when running on HPC/original environment

        if RUN_LOCALLY:
            logger.info("Applying local configuration modifications...")
            
            # Load the YAML file directly and modify it
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Modify paths in the dictionary
            config_dict['data_dir'] = str(Path("C:/PhD/Data/Caravan"))
            config_dict['device'] = 'cpu'
            
            # IMPORTANT: Set the run_dir to the absolute local path
            config_dict['run_dir'] = str(experiment_folder.absolute())
            
            # Update basin files if they exist
            LOCAL_BASIN_PATH = Path("C:/PhD/Python/neuralhydrology/Experiments/expand_stations_and_periods/new_configuration_wide_data_with_static")
            
            if 'test_basin_file' in config_dict and config_dict['test_basin_file']:
                basin_filename = Path(config_dict['test_basin_file']).name
                config_dict['test_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)
            
            if 'train_basin_file' in config_dict and config_dict['train_basin_file']:
                basin_filename = Path(config_dict['train_basin_file']).name
                config_dict['train_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)
            
            if 'validation_basin_file' in config_dict and config_dict['validation_basin_file']:
                basin_filename = Path(config_dict['validation_basin_file']).name
                config_dict['validation_basin_file'] = str(LOCAL_BASIN_PATH / basin_filename)
            
            # Create new config from modified dictionary
            config = Config(config_dict)
            logger.info("Local configuration modifications applied successfully")
        else:
            logger.info("Running on HPC/cluster - using original configuration paths")
            # On cluster, just ensure the run_dir is set correctly
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Only modify run_dir to the current experiment folder absolute path
            config_dict['run_dir'] = str(experiment_folder.absolute())
            
            # Create config from potentially modified dictionary
            config = Config(config_dict)
            logger.info("HPC configuration applied successfully")

        logger.info(f"Running full evaluation for: {config.experiment_name}")
        
        # Get the last epoch
        try:
            last_epoch = get_last_epoch_number(experiment_folder)
            logger.info(f"Using model from epoch {last_epoch}")
        except FileNotFoundError as e:
            logger.error(f"Error finding model files: {e}")
            return False
        
        # Create tester instance
        try:
            import torch
            
            # Set device appropriately
            if RUN_LOCALLY:
                # Force CPU for local execution
                if hasattr(torch, 'cuda') and torch.cuda.is_available():
                    logger.info("CUDA available but forcing CPU for local evaluation")
                torch.set_default_tensor_type('torch.FloatTensor')  # Ensure CPU tensors
            else:
                # On cluster, use the device specified in config (likely GPU)
                logger.info(f"Running on cluster with device: {config.device}")
                if config.device.startswith('cuda') and torch.cuda.is_available():
                    torch.set_default_tensor_type('torch.cuda.FloatTensor')
                else:
                    torch.set_default_tensor_type('torch.FloatTensor')
            
            # Try test period first since it worked before
            try:
                tester = get_tester(cfg=config, run_dir=experiment_folder, period="test", init_model=True)
                logger.info("Tester created successfully using test period")
                period_used = "test"
            except Exception as test_error:
                logger.warning(f"Could not create tester for test period: {test_error}")
                try:
                    tester = get_tester(cfg=config, run_dir=experiment_folder, period="validation", init_model=True)
                    logger.info("Tester created successfully using validation period")
                    period_used = "validation"
                except Exception as val_error:
                    logger.error(f"Could not create tester for validation period either: {val_error}")
                    raise val_error
            
        except Exception as e:
            logger.error(f"Error creating tester: {e}")
            return False
        
        # Run the evaluation
        try:
            logger.info(f"Starting evaluation on {period_used} period...")
            results = tester.evaluate()
            logger.info(f"Evaluation completed. Results type: {type(results)}")
            
            if results and len(results) > 0:
                logger.info(f"Results available with {len(results)} entries")
                
                # Create output directory for this experiment
                output_dir = experiment_folder / "test_results"
                
                # Save individual basin CSV files
                saved_count = save_basin_hydrographs(results, output_dir)
                
                if saved_count > 0:
                    logger.info(f"Successfully saved {saved_count} basin hydrographs to {output_dir}")
                    return True
                else:
                    logger.warning("No hydrographs were saved")
                    return False
                    
            else:
                logger.warning("No results returned from evaluation")
                return False
                
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error processing {experiment_folder}: {e}")
        return False

def main():
    """Main function to process all experiments."""
    # Define the base directory containing experiments
    base_experiment_dir = Path("Experiments/HPC_random_search/results")

    if not base_experiment_dir.exists():
        logger.error(f"Could not find experiments directory: {base_experiment_dir}")
        return
    
    logger.info(f"Searching for experiments in: {base_experiment_dir.absolute()}")
    
    # Find all experiment folders
    experiment_folders = find_experiment_folders(base_experiment_dir)
    logger.info(f"Found {len(experiment_folders)} experiment folders")
    
    if not experiment_folders:
        logger.warning("No experiment folders with config.yml found")
        return
    
    # Show sample paths for verification
    logger.info("Sample experiment paths:")
    for folder in experiment_folders[:3]:
        logger.info(f"  - {folder}")
        # Also show what pickle files exist
        test_files = list(folder.glob("test/model_epoch*/test_results.p"))
        val_files = list(folder.glob("validation/model_epoch*/validation_results.p"))
        logger.info(f"    Test files: {len(test_files)}")
        logger.info(f"    Validation files: {len(val_files)}")
        if test_files:
            logger.info(f"    Latest test: {test_files[-1]}")
        if val_files:
            logger.info(f"    Latest validation: {val_files[-1]}")
    
    # Categorize experiments - now with proper validation
    experiments_with_results = []
    experiments_without_results = []
    experiments_with_csv = []
    
    logger.info("Categorizing experiments...")
    for i, folder in enumerate(experiment_folders, 1):
        logger.info(f"Checking experiment {i}/{len(experiment_folders)}: {folder.name}")
        
        if has_existing_csv_results(folder):
            experiments_with_csv.append(folder)
            logger.info(f"  -> Already has CSV results")
        elif has_valid_pickle_results(folder):
            experiments_with_results.append(folder)
            logger.info(f"  -> Has valid pickle results")
        else:
            experiments_without_results.append(folder)
            logger.info(f"  -> Needs full evaluation")
    
    logger.info(f"Categorization complete:")
    logger.info(f"  - Already have CSV results: {len(experiments_with_csv)}")
    logger.info(f"  - Have valid pickle results (need CSV extraction): {len(experiments_with_results)}")
    logger.info(f"  - Need full evaluation: {len(experiments_without_results)}")
    
    # For debugging, let's also manually check one of the "failed" experiments
    if experiments_without_results:
        sample_folder = experiments_without_results[0]
        logger.info(f"DEBUG: Manually checking {sample_folder.name}")
        
        # Check what files actually exist
        test_files = list(sample_folder.glob("test/model_epoch*/test_results.p"))
        val_files = list(sample_folder.glob("validation/model_epoch*/validation_results.p"))
        
        logger.info(f"DEBUG: Found {len(test_files)} test files, {len(val_files)} validation files")
        
        if test_files:
            latest_file = test_files[-1]
            logger.info(f"DEBUG: Checking latest file: {latest_file}")
            
            try:
                with open(latest_file, 'rb') as f:
                    results = pickle.load(f)
                logger.info(f"DEBUG: Loaded {type(results)} with {len(results) if hasattr(results, '__len__') else 'unknown'} entries")
                
                if results:
                    sample_basin = list(results.keys())[0] if results else None
                    if sample_basin:
                        logger.info(f"DEBUG: Sample basin {sample_basin}: {type(results[sample_basin])}")
                        if isinstance(results[sample_basin], dict):
                            logger.info(f"DEBUG: Sample basin keys: {list(results[sample_basin].keys())}")
            except Exception as e:
                logger.error(f"DEBUG: Error loading {latest_file}: {e}")
    
    # Continue with processing...
    # 1. Skip experiments that already have CSV results
    if experiments_with_csv:
        logger.info(f"Skipping {len(experiments_with_csv)} experiments that already have CSV results")
        successful_count += len(experiments_with_csv)
    
    # 2. Process experiments with existing valid pickle results (fast)
    if experiments_with_results:
        logger.info(f"Processing {len(experiments_with_results)} experiments with valid pickle results...")
        for i, folder in enumerate(experiments_with_results, 1):
            logger.info(f"Extracting from pickle {i}/{len(experiments_with_results)}: {folder.name}")
            
            try:
                success = extract_hydrographs_from_pickle(folder)
                if success:
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to extract from pickle {folder}: {e}")
                failed_count += 1
    
    # 3. Process experiments without results (slow - full evaluation)
    if experiments_without_results:
        logger.info(f"Running full evaluation for {len(experiments_without_results)} experiments...")
        for i, folder in enumerate(experiments_without_results, 1):
            logger.info(f"Full evaluation {i}/{len(experiments_without_results)}: {folder.name}")
            
            try:
                success = run_full_evaluation(folder)
                if success:
                    successful_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Failed to evaluate {folder}: {e}")
                failed_count += 1
    
    logger.info(f"Processing complete. Successful: {successful_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main()