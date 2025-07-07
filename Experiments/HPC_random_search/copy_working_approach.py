from pathlib import Path
import pickle

def find_working_test_results():
    """Find experiments that have working test results and copy the approach."""
    
    base_dir = Path("Experiments/HPC_random_search/results")
    
    for config_path in base_dir.rglob("config.yml"):
        experiment_folder = config_path.parent
        
        # Look for test results (which seem to work)
        test_files = list(experiment_folder.glob("test/model_epoch*/test_results.p"))
        
        if test_files:
            latest_test_file = sorted(test_files)[-1]
            
            try:
                with open(latest_test_file, 'rb') as f:
                    results = pickle.load(f)
                
                if results and len(results) > 0:
                    print(f"WORKING EXPERIMENT: {experiment_folder.name}")
                    print(f"Test results file: {latest_test_file}")
                    print(f"Number of basins: {len(results)}")
                    
                    # Show the structure
                    sample_basin = list(results.keys())[0]
                    sample_data = results[sample_basin]
                    print(f"Sample basin: {sample_basin}")
                    print(f"Sample data type: {type(sample_data)}")
                    
                    if isinstance(sample_data, dict):
                        print(f"Sample data keys: {list(sample_data.keys())}")
                        
                        # Look for the structure
                        for key, value in sample_data.items():
                            print(f"  {key}: {type(value)}")
                            if isinstance(value, dict) and 'xr' in value:
                                xr_data = value['xr']
                                df = xr_data.to_dataframe().reset_index()
                                print(f"    DataFrame columns: {list(df.columns)}")
                    
                    return experiment_folder, latest_test_file
                    
            except Exception as e:
                continue
    
    return None, None

if __name__ == "__main__":
    working_folder, working_file = find_working_test_results()
    if working_folder:
        print(f"\nFound working approach in: {working_folder}")
        print("Now you can modify this to use validation period instead of test period")