import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.evaluation import get_tester
import logging
import xarray as xr
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_experiment_folders(base_dir):
    """Find all experiment folders containing config.yml files and model files."""
    experiment_folders = []
    base_path = Path(base_dir)
    
    for item in base_path.rglob("config.yml"):
        config_dir = item.parent
        # Check if this directory contains model files
        model_files = list(config_dir.glob("model_epoch*.pt"))
        if model_files:
            experiment_folders.append(config_dir)
    
    return experiment_folders

def has_validation_csv_results(experiment_folder):
    """Check if experiment already has validation CSV results."""
    results_dir = experiment_folder / "validation_hydrographs"
    if results_dir.exists():
        csv_files = list(results_dir.glob("*.csv"))
        return len(csv_files) > 0
    return False

def has_validation_pickle_results(experiment_folder):
    """Check if experiment has validation pickle results with actual data."""
    val_files = list(experiment_folder.glob("validation/model_epoch*/validation_results.p"))
    
    if not val_files:
        return False
    
    # Check if the most recent pickle file actually contains data
    try:
        latest_file = sorted(val_files)[-1]
        with open(latest_file, 'rb') as f:
            results = pickle.load(f)
        
        # Check if results actually contain data
        if results and len(results) > 0:
            # Additional check: make sure at least one basin has actual data
            for basin_id, basin_data in results.items():
                if basin_data:  # If any basin has data, consider it valid
                    return True
            return False  # All basins are empty
        else:
            return False  # Results dict is empty
            
    except Exception as e:
        logger.debug(f"Error checking validation pickle file: {e}")
        return False

def extract_hydrographs_from_results(results, output_dir):
    """Extract and save hydrographs from evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    
    if not results or len(results) == 0:
        logger.warning("No results to save")
        return saved_count
    
    logger.info(f"Extracting hydrographs for {len(results)} basins")
    
    for basin_id, basin_data in results.items():
        try:
            observed = None
            predicted = None
            dates = None
            
            if isinstance(basin_data, dict):
                logger.debug(f"Processing basin {basin_id}, data keys: {list(basin_data.keys())}")
                
                # Look for frequency-based structure (most common in neuralhydrology)
                freq_keys = [key for key in basin_data.keys() if any(char.isdigit() for char in key)]
                
                if freq_keys:
                    freq_key = freq_keys[0]  # Use the first frequency (e.g., '10min')
                    freq_data = basin_data[freq_key]
                    logger.debug(f"Using frequency key: {freq_key}")
                    
                    if isinstance(freq_data, dict) and 'xr' in freq_data:
                        # Extract from xarray structure
                        xr_data = freq_data['xr']
                        df = xr_data.to_dataframe().reset_index()
                        
                        logger.debug(f"DataFrame columns: {list(df.columns)}")
                        
                        # Get dates
                        if 'date' in df.columns:
                            dates = pd.to_datetime(df['date'])
                        
                        # Find observed and predicted columns - be more flexible
                        for col in df.columns:
                            col_lower = col.lower()
                            # Look for observed data
                            if any(obs_pattern in col_lower for obs_pattern in ['obs', 'qobs', 'target']):
                                observed = df[col].values
                                logger.debug(f"Found observed in column: {col}")
                            # Look for predicted data
                            elif any(sim_pattern in col_lower for sim_pattern in ['sim', 'qsim', 'pred', 'prediction']):
                                predicted = df[col].values
                                logger.debug(f"Found predicted in column: {col}")
            
            # Create and save CSV if we have the required data
            if observed is not None and predicted is not None and dates is not None:
                # Create DataFrame with correct column order: datetime, observed, predicted
                df_output = pd.DataFrame({
                    'datetime': dates,
                    'observed': observed.flatten(),
                    'predicted': predicted.flatten()
                })
                
                # Save to CSV
                csv_file = output_path / f"{basin_id}_validation_hydrograph.csv"
                df_output.to_csv(csv_file, index=False)
                
                logger.debug(f"Saved {basin_id}: {len(df_output)} time steps")
                saved_count += 1
            else:
                logger.warning(f"Could not extract data for basin {basin_id}")
                # Debug info for first few failures
                if saved_count < 3:
                    logger.warning(f"  Observed: {observed is not None}")
                    logger.warning(f"  Predicted: {predicted is not None}")
                    logger.warning(f"  Dates: {dates is not None}")
                
        except Exception as e:
            logger.error(f"Error processing basin {basin_id}: {e}")
    
    logger.info(f"Successfully saved {saved_count} hydrographs")
    return saved_count

def debug_validation_config(config):
    """Debug validation configuration to understand basin filtering."""
    logger.info("=== DEBUGGING VALIDATION CONFIG ===")
    
    # Check validation period
    logger.info(f"Validation start date: {config.validation_start_date}")
    logger.info(f"Validation end date: {config.validation_end_date}")
    
    # Check basin file
    logger.info(f"Validation basin file: {config.validation_basin_file}")
    
    # Check if validation basin file exists
    if not Path(config.validation_basin_file).exists():
        logger.error(f"Validation basin file does not exist: {config.validation_basin_file}")
        return False
    
    # Load validation basins
    from neuralhydrology.datautils.utils import load_basin_file
    try:
        val_basins = load_basin_file(config.validation_basin_file)
        logger.info(f"Loaded {len(val_basins)} validation basins")
        logger.info(f"First 5 basins: {val_basins[:5]}")
    except Exception as e:
        logger.error(f"Error loading validation basins: {e}")
        return False
    
    # Check sequence length requirement
    seq_length = getattr(config, 'seq_length', 365)
    logger.info(f"Sequence length requirement: {seq_length}")
    
    # Check other filtering parameters
    logger.info(f"Target variables: {getattr(config, 'target_variables', 'Not set')}")
    logger.info(f"Min samples: {getattr(config, 'min_training_samples', 'Not set')}")
    
    # Sample a basin and check data availability
    sample_basin = val_basins[0]
    logger.info(f"Checking sample basin: {sample_basin}")
    
    # Check data file
    data_file = Path(config.data_dir) / "timeseries" / "netcdf" / sample_basin.split('_')[0] / f"{sample_basin}.nc"
    logger.info(f"Data file path: {data_file}")
    
    if not data_file.exists():
        logger.error(f"Data file does not exist: {data_file}")
        return False
    
    # Load and check data
    try:
        import xarray as xr
        ds = xr.open_dataset(data_file)
        
        # Check date range
        if 'date' in ds.variables:
            dates = pd.to_datetime(ds['date'].values)
            logger.info(f"Data covers: {dates.min()} to {dates.max()}")
            
            # Check overlap with validation period
            val_start = pd.to_datetime(config.validation_start_date)
            val_end = pd.to_datetime(config.validation_end_date)
            
            overlap_mask = (dates >= val_start) & (dates <= val_end)
            overlap_count = overlap_mask.sum()
            logger.info(f"Validation period overlap: {overlap_count} time steps")
            
            if overlap_count == 0:
                logger.error("NO OVERLAP between validation period and data!")
                return False
                
            # Check target variable availability
            target_vars = config.target_variables if hasattr(config, 'target_variables') else ['discharge_vol', 'QObs(mm/d)']
            
            for target_var in target_vars:
                if target_var in ds.variables:
                    target_data = ds[target_var].values[overlap_mask]
                    valid_count = (~pd.isna(target_data)).sum()
                    logger.info(f"Target variable '{target_var}': {valid_count}/{len(target_data)} valid values")
                    
                    if valid_count < seq_length:
                        logger.warning(f"Not enough valid values for sequence length!")
                    else:
                        logger.info(f"Sufficient valid values for sequence length")
                        
                else:
                    logger.warning(f"Target variable '{target_var}' not found in data")
        
        ds.close()
        return True
        
    except Exception as e:
        logger.error(f"Error checking data: {e}")
        return False

def run_validation_evaluation_using_start_evaluation(experiment_folder):
    """Run validation evaluation using neuralhydrology's standard evaluation workflow."""
    try:
        config_path = experiment_folder / "config.yml"
        if not config_path.exists():
            logger.warning(f"No config.yml found in {experiment_folder}")
            return False
        
        logger.info(f"Processing experiment: {experiment_folder.name}")
        
        # Load original config to check validation period
        import yaml
        with open(config_path, 'r') as f:
            original_config = yaml.safe_load(f)
            
        logger.info("=== ORIGINAL CONFIG VALIDATION SETTINGS ===")
        logger.info(f"Original validation_start_date: {original_config.get('validation_start_date', 'Not set')}")
        logger.info(f"Original validation_end_date: {original_config.get('validation_end_date', 'Not set')}")
        logger.info(f"Original validation_basin_file: {original_config.get('validation_basin_file', 'Not set')}")
        
        # Check if validation period is actually defined
        if not original_config.get('validation_start_date') or not original_config.get('validation_end_date'):
            logger.error("Validation period not defined in config!")
            return False
            
        # Load and modify configuration for local execution
        config_dict = original_config.copy()
        
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
        
        # Create config
        config = Config(config_dict)
        
        # Debug validation configuration
        if not debug_validation_config(config):
            logger.error("Validation configuration is invalid")
            return False
        
        # Set up PyTorch for CPU
        import torch
        torch.set_default_tensor_type('torch.FloatTensor')
        
        # Use neuralhydrology's standard evaluation workflow
        logger.info("Starting validation evaluation using start_evaluation...")
        start_evaluation(cfg=config, run_dir=experiment_folder, epoch=None, period="validation")
        
        # After evaluation, load the results and extract hydrographs
        validation_results_files = list(experiment_folder.glob("validation/model_epoch*/validation_results.p"))
        
        if validation_results_files:
            # Get the latest validation results file
            latest_results_file = sorted(validation_results_files)[-1]
            logger.info(f"Loading results from: {latest_results_file}")
            
            with open(latest_results_file, 'rb') as f:
                results = pickle.load(f)
            
            if results and len(results) > 0:
                logger.info(f"Loaded results with {len(results)} basins")
                
                # Create output directory
                output_dir = experiment_folder / "validation_hydrographs"
                
                # Extract and save hydrographs
                saved_count = extract_hydrographs_from_results(results, output_dir)
                
                if saved_count > 0:
                    logger.info(f"Successfully saved {saved_count} validation hydrographs")
                    return True
                else:
                    logger.warning("No hydrographs were saved")
                    return False
            else:
                logger.warning("Results file is empty")
                return False
        else:
            logger.warning("No validation results files found after evaluation")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {experiment_folder.name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_validation_evaluation_using_tester(experiment_folder):
    """Alternative approach using tester directly (like your example file)."""
    try:
        config_path = experiment_folder / "config.yml"
        if not config_path.exists():
            logger.warning(f"No config.yml found in {experiment_folder}")
            return False
        
        logger.info(f"Processing experiment: {experiment_folder.name}")
        
        # Load and modify configuration for local execution
        import yaml
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
        
        # Create config
        config = Config(config_dict)
        
        # Set up PyTorch for CPU
        import torch
        torch.set_default_tensor_type('torch.FloatTensor')
        
        # Create tester instance (following your example pattern)
        logger.info("Creating tester for validation period...")
        tester = get_tester(cfg=config, run_dir=experiment_folder, period="validation", init_model=True)
        
        # Run evaluation (following your example pattern)
        logger.info("Running evaluation...")
        results = tester.evaluate(save_results=True, metrics=config.metrics)
        
        if results and len(results) > 0:
            logger.info(f"Evaluation completed with {len(results)} basins")
            
            # Create output directory
            output_dir = experiment_folder / "validation_hydrographs"
            
            # Extract and save hydrographs
            saved_count = extract_hydrographs_from_results(results, output_dir)
            
            if saved_count > 0:
                logger.info(f"Successfully saved {saved_count} validation hydrographs")
                return True
            else:
                logger.warning("No hydrographs were saved")
                return False
        else:
            logger.warning("No results returned from evaluation")
            return False
            
    except Exception as e:
        logger.error(f"Error processing {experiment_folder.name}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main function to process all experiments for validation evaluation."""
    
    # Find experiments
    base_experiment_dir = Path("Experiments/HPC_random_search/results")
    
    if not base_experiment_dir.exists():
        logger.error(f"Could not find experiments directory: {base_experiment_dir}")
        return
    
    logger.info(f"Searching for experiments in: {base_experiment_dir.absolute()}")
    experiment_folders = find_experiment_folders(base_experiment_dir)
    logger.info(f"Found {len(experiment_folders)} experiment folders")
    
    if not experiment_folders:
        logger.warning("No experiment folders found")
        return
    
    # Categorize experiments with proper validation
    experiments_with_validation_csv = []
    experiments_with_validation_pickle = []
    experiments_to_process = []
    
    logger.info("Categorizing experiments (checking actual content)...")
    for i, folder in enumerate(experiment_folders, 1):
        logger.info(f"Checking experiment {i}/{len(experiment_folders)}: {folder.name}")
        
        if has_validation_csv_results(folder):
            experiments_with_validation_csv.append(folder)
            logger.info(f"  -> Already has CSV results")
        elif has_validation_pickle_results(folder):
            experiments_with_validation_pickle.append(folder)
            logger.info(f"  -> Has valid pickle results")
        else:
            experiments_to_process.append(folder)
            logger.info(f"  -> Needs validation evaluation")
    
    logger.info(f"Categorization complete:")
    logger.info(f"  - Already have validation CSV results: {len(experiments_with_validation_csv)}")
    logger.info(f"  - Have valid validation pickle results: {len(experiments_with_validation_pickle)}")
    logger.info(f"  - Need validation evaluation: {len(experiments_to_process)}")
    
    # First, extract CSVs from existing valid pickle files
    if experiments_with_validation_pickle:
        logger.info(f"Extracting CSVs from {len(experiments_with_validation_pickle)} valid pickle files...")
        for i, folder in enumerate(experiments_with_validation_pickle, 1):
            logger.info(f"Extracting {i}/{len(experiments_with_validation_pickle)}: {folder.name}")
            try:
                validation_results_files = list(folder.glob("validation/model_epoch*/validation_results.p"))
                if validation_results_files:
                    latest_results_file = sorted(validation_results_files)[-1]
                    with open(latest_results_file, 'rb') as f:
                        results = pickle.load(f)
                    
                    if results and len(results) > 0:
                        output_dir = folder / "validation_hydrographs"
                        saved_count = extract_hydrographs_from_results(results, output_dir)
                        if saved_count > 0:
                            logger.info(f"  -> Extracted {saved_count} hydrographs")
                        else:
                            logger.warning(f"  -> No hydrographs extracted")
                    else:
                        logger.warning(f"  -> Pickle file is empty")
            except Exception as e:
                logger.error(f"Error extracting from {folder.name}: {e}")
    
    # Then process experiments that need evaluation
    if experiments_to_process:
        logger.info(f"Processing {len(experiments_to_process)} experiments that need evaluation...")
        
        # Process only the first experiment for debugging
        folder = experiments_to_process[0]
        logger.info(f"Processing experiment for debugging: {folder.name}")
        
        try:
            # Try the standard evaluation approach first
            success = run_validation_evaluation_using_start_evaluation(folder)
            if not success:
                logger.info("Standard evaluation failed, trying tester approach...")
                success = run_validation_evaluation_using_tester(folder)
            
            if success:
                logger.info("Successfully processed experiment")
            else:
                logger.error("Failed to process experiment with both approaches")
                
        except Exception as e:
            logger.error(f"Failed to process {folder.name}: {e}")
    
    else:
        logger.info("All experiments already have validation results!")

if __name__ == "__main__":
    main()
