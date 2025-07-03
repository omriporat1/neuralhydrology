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
    """Find all experiment folders containing config.yml files."""
    experiment_folders = []
    base_path = Path(base_dir)
    
    for item in base_path.rglob("config.yml"):
        experiment_folders.append(item.parent)
    
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
            # Original config loading
            config = Config(config_path)
            config.device = "cpu"  # This works because device is settable

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
            tester = get_tester(cfg=config, run_dir=experiment_folder, period="validation", init_model=True)
        except Exception as e:
            logger.error(f"Error creating tester: {e}")
            return False
        
        # Run evaluation
        try:
            results = tester.evaluate(epoch=last_epoch, save_results=False, save_all_output=False, metrics=[])
            logger.info(f"Evaluation completed for {len(results)} basins")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return False
        
        # Load basin list
        try:
            if hasattr(config, 'test_basin_file') and config.test_basin_file:
                basin_list = load_basin_file(Path(config.test_basin_file))
            else:
                basin_list = list(results.keys())
        except Exception as e:
            logger.warning(f"Could not load basin file, using results keys: {e}")
            basin_list = list(results.keys())
        
        # Extract hydrographs
        hydrographs = extract_hydrographs_from_results(results, basin_list)
        
        # Create hydrographs directory
        hydrographs_dir = experiment_folder / "hydrographs"
        hydrographs_dir.mkdir(exist_ok=True)
        
        # Save hydrographs to CSV files
        saved_count = 0
        for basin, df in hydrographs.items():
            try:
                output_file = hydrographs_dir / f"{basin}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Saved hydrograph for basin {basin}")
                saved_count += 1
            except Exception as e:
                logger.error(f"Error saving hydrograph for basin {basin}: {e}")
        
        logger.info(f"Successfully saved {saved_count} hydrographs to {hydrographs_dir}")
        return True
        
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