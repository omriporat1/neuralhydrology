import os
import pickle
import pandas as pd
from pathlib import Path
from neuralhydrology.utils.config import Config
from neuralhydrology.evaluation import get_tester
from neuralhydrology.datautils.utils import load_basin_file
import logging

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

def extract_hydrographs_from_results(results, basin_list):
    """Extract hydrograph data from evaluation results."""
    hydrographs = {}
    
    for basin in basin_list:
        if basin in results:
            basin_data = results[basin]
            
            # Handle different frequency structures
            if isinstance(basin_data, dict):
                # Check if there are frequency keys (like '1D', '1h')
                freq_keys = [k for k in basin_data.keys() if k not in ['qobs', 'date']]
                
                if freq_keys:
                    # Use the first frequency found
                    freq_key = freq_keys[0]
                    if 'xr' in basin_data[freq_key]:
                        xr_data = basin_data[freq_key]['xr']
                        df = xr_data.to_dataframe().reset_index()
                        
                        # Ensure we have datetime and flow columns
                        if 'date' in df.columns:
                            # Create clean hydrograph with datetime and flow values
                            hydrograph_df = pd.DataFrame()
                            hydrograph_df['datetime'] = pd.to_datetime(df['date'])
                            
                            # Look for flow columns (observed and simulated)
                            flow_cols = [col for col in df.columns if col not in ['date', 'basin']]
                            for col in flow_cols:
                                hydrograph_df[col] = df[col]
                            
                            hydrographs[basin] = hydrograph_df
                else:
                    # Direct xarray data
                    if 'xr' in basin_data:
                        xr_data = basin_data['xr']
                        df = xr_data.to_dataframe().reset_index()
                        
                        # Ensure we have datetime and flow columns
                        if 'date' in df.columns:
                            hydrograph_df = pd.DataFrame()
                            hydrograph_df['datetime'] = pd.to_datetime(df['date'])
                            
                            # Look for flow columns
                            flow_cols = [col for col in df.columns if col not in ['date', 'basin']]
                            for col in flow_cols:
                                hydrograph_df[col] = df[col]
                            
                            hydrographs[basin] = hydrograph_df
            
            # If no structured data found, log warning
            if basin not in hydrographs:
                logger.warning(f"Could not extract hydrograph data for basin {basin}")
    
    return hydrographs

def create_test_hydrographs(experiment_folder):
    """Create test hydrographs for a single experiment."""
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

        logger.info(f"Processing experiment: {config.experiment_name}")
        
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
            
            # Ensure CPU device is set in config
            logger.info(f"Config device before tester creation: {config.device}")
            
            # Debug: Check if model files exist
            model_files = list(experiment_folder.glob("model_epoch*.pt"))
            logger.info(f"Found {len(model_files)} model files in {experiment_folder}")
            
            # Debug: Check scaler files
            train_data_dir = experiment_folder / "train_data"
            scaler_yml = train_data_dir / "train_data_scaler.yml"
            scaler_pickle = train_data_dir / "train_data_scaler.p"
            logger.info(f"Scaler YML exists: {scaler_yml.exists()}")
            logger.info(f"Scaler pickle exists: {scaler_pickle.exists()}")
            
            # Set device to CPU explicitly
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                logger.info("CUDA available but forcing CPU for evaluation")
            torch.set_default_tensor_type('torch.FloatTensor')  # Ensure CPU tensors
            
            tester = get_tester(cfg=config, run_dir=experiment_folder, period="validation", init_model=True)
            logger.info(f"Tester created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tester: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
        
        # Debug validation basin file and data availability
        logger.info("=== DEBUGGING VALIDATION SETUP ===")
        if hasattr(config, 'validation_basin_file') and config.validation_basin_file:
            val_basin_path = Path(config.validation_basin_file)
            logger.info(f"Validation basin file: {val_basin_path}")
            logger.info(f"Validation basin file exists: {val_basin_path.exists()}")
            
            if val_basin_path.exists():
                try:
                    val_basins = load_basin_file(val_basin_path)
                    logger.info(f"Total validation basins from file: {len(val_basins)}")
                    logger.info(f"First 5 validation basins: {val_basins[:5]}")
                    
                    # Debug: Check validation period
                    if hasattr(config, 'val_start_date') and hasattr(config, 'val_end_date'):
                        logger.info(f"Validation period: {config.val_start_date} to {config.val_end_date}")
                    else:
                        logger.warning("No validation period found in config")
                    
                    # Debug: Check data directory and available basins
                    data_dir = Path(config.data_dir)
                    logger.info(f"Data directory: {data_dir}")
                    logger.info(f"Data directory exists: {data_dir.exists()}")
                    
                    if data_dir.exists():
                        # Check what basins are actually available in the data directory
                        all_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
                        logger.info(f"Total directories in data folder: {len(all_dirs)}")
                        logger.info(f"First 5 directories: {all_dirs[:5]}")
                        
                        # Check which validation basins are available
                        available_val_basins = [b for b in val_basins if b in all_dirs]
                        logger.info(f"Available validation basins in data: {len(available_val_basins)}")
                        
                        missing_basins = [b for b in val_basins if b not in all_dirs]
                        if missing_basins:
                            logger.warning(f"Missing basins count: {len(missing_basins)}")
                            logger.warning(f"First 5 missing basins: {missing_basins[:5]}")
                        
                        # Check if any validation basin has data files for the validation period
                        if available_val_basins:
                            sample_basin = available_val_basins[0]
                            basin_dir = data_dir / sample_basin
                            data_files = list(basin_dir.glob("*.csv"))
                            logger.info(f"Sample basin {sample_basin} has {len(data_files)} data files")
                            
                            # Check date range in the first data file
                            if data_files:
                                try:
                                    sample_df = pd.read_csv(data_files[0], parse_dates=['date'], nrows=10)
                                    if 'date' in sample_df.columns:
                                        logger.info(f"Sample data date range starts: {sample_df['date'].min()}")
                                        
                                        # Read the whole file to get end date
                                        full_df = pd.read_csv(data_files[0], parse_dates=['date'])
                                        logger.info(f"Sample data date range ends: {full_df['date'].max()}")
                                except Exception as e:
                                    logger.warning(f"Could not read sample data file: {e}")
                    
                except Exception as e:
                    logger.error(f"Error in validation debugging: {e}")
        else:
            logger.warning("No validation_basin_file found in config")
        
        logger.info("=== END DEBUGGING ===")
            
    except Exception as e:
        logger.error(f"Unexpected error processing {experiment_folder}: {e}")
        return False

def main():
    """Main function to process all experiments."""
    # Define the base directory containing experiments
    # Adjust this path based on your workspace structure
    base_experiment_dir = Path("Experiments/HPC_random_search/results")  # Change this to your experiments directory

    if not base_experiment_dir.exists():
        # Try alternative paths
        alternative_paths = [
            Path("runs"),
            Path("./runs"),
            Path("../runs"),
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                base_experiment_dir = alt_path
                break
        else:
            logger.error(f"Could not find experiments directory. Please specify the correct path.")
            return
    
    logger.info(f"Searching for experiments in: {base_experiment_dir.absolute()}")
    
    # Find all experiment folders
    experiment_folders = find_experiment_folders(base_experiment_dir)
    logger.info(f"Found {len(experiment_folders)} experiment folders")
    
    if not experiment_folders:
        logger.warning("No experiment folders with config.yml found")
        return
    
    # Process each experiment
    successful_count = 0
    failed_count = 0
    
    for i, folder in enumerate(experiment_folders, 1):
        logger.info(f"Processing experiment {i}/{len(experiment_folders)}: {folder.name}")
        
        try:
            success = create_test_hydrographs(folder)
            if success:
                successful_count += 1
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"Failed to process {folder}: {e}")
            failed_count += 1
    
    logger.info(f"Processing complete. Successful: {successful_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main()