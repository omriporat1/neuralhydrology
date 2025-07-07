import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_results_file(results_file):
    """Inspect the structure of the results file"""
    print(f"Loading results from: {results_file}")
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Results type: {type(results)}")
    
    if isinstance(results, dict):
        print(f"Number of keys: {len(results)}")
        if len(results) == 0:
            print("Results dictionary is empty!")
            return results
            
        print(f"Keys: {list(results.keys())[:5]}...")  # Show first 5 keys
        
        # Inspect first entry
        first_key = list(results.keys())[0]
        first_value = results[first_key]
        print(f"\nFirst entry key: {first_key}")
        print(f"First entry type: {type(first_value)}")
        
        if isinstance(first_value, dict):
            print(f"First entry keys: {list(first_value.keys())}")
            for key, value in first_value.items():
                print(f"  {key}: {type(value)} - shape: {getattr(value, 'shape', 'N/A')}")
        else:
            print(f"First entry value: {first_value}")
    else:
        print(f"Results content: {results}")
    
    return results

def extract_all_hydrographs(results_file, output_dir):
    """Extract hydrographs for all basins from validation_results.p"""
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded results: {type(results)}, length: {len(results) if hasattr(results, '__len__') else 'N/A'}")
    
    if not results or len(results) == 0:
        print("No results to extract - file is empty!")
        return {}
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_hydrographs = {}
    
    for basin_id, basin_data in results.items():
        try:
            print(f"Processing basin: {basin_id}")
            print(f"Basin data type: {type(basin_data)}")
            
            # Handle different possible structures
            if isinstance(basin_data, dict):
                print(f"Basin data keys: {list(basin_data.keys())}")
                
                # Look for discharge data
                obs = None
                sim = None
                dates = None
                
                # Check different possible key names
                for obs_key in ['qobs', 'obs', 'observed', 'y_obs']:
                    if obs_key in basin_data:
                        obs = basin_data[obs_key]
                        print(f"Found observed data with key: {obs_key}, shape: {getattr(obs, 'shape', 'N/A')}")
                        break
                
                for sim_key in ['qsim', 'sim', 'simulated', 'y_sim', 'y_hat']:
                    if sim_key in basin_data:
                        sim = basin_data[sim_key]
                        print(f"Found simulated data with key: {sim_key}, shape: {getattr(sim, 'shape', 'N/A')}")
                        break
                
                for date_key in ['date', 'dates', 'time']:
                    if date_key in basin_data:
                        dates = basin_data[date_key]
                        print(f"Found dates with key: {date_key}, shape: {getattr(dates, 'shape', 'N/A')}")
                        break
                
                if obs is not None and sim is not None:
                    # Ensure arrays are 1D
                    obs_flat = obs.flatten() if hasattr(obs, 'flatten') else obs
                    sim_flat = sim.flatten() if hasattr(sim, 'flatten') else sim
                    
                    # Create DataFrame
                    df_data = {
                        'observed': obs_flat,
                        'simulated': sim_flat
                    }
                    
                    if dates is not None:
                        dates_flat = dates.flatten() if hasattr(dates, 'flatten') else dates
                        df_data['date'] = dates_flat
                    
                    df = pd.DataFrame(df_data)
                    
                    # Save individual basin CSV
                    df.to_csv(output_path / f'{basin_id}_hydrograph.csv', index=False)
                    
                    # Store in dictionary
                    all_hydrographs[basin_id] = df
                    
                    print(f"Extracted hydrograph for {basin_id}: {len(df)} time steps")
                else:
                    print(f"No observed/simulated data found for {basin_id}")
                
        except Exception as e:
            print(f"Error processing basin {basin_id}: {e}")
    
    return all_hydrographs

def main():
    """Main function to extract hydrographs from validation results."""
    # Fix the path - use forward slashes or raw string
    results_path = Path("C:/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results/job_41780893/run_000/N38_A30_4CPU_SMI_0406_170648/test/model_epoch009/test_results.p")
    
    # Check if file exists
    if not results_path.exists():
        print(f"File not found: {results_path}")
        # Try validation_results.p instead
        validation_path = results_path.parent / "validation_results.p"
        if validation_path.exists():
            print(f"Found validation_results.p instead: {validation_path}")
            results_path = validation_path
        else:
            print("No results file found!")
            return
    
    # First inspect the file
    print("=== INSPECTING RESULTS FILE ===")
    results = inspect_results_file(results_path)
    
    print("\n=== EXTRACTING HYDROGRAPHS ===")
    output_dir = "extracted_hydrographs"
    hydrographs = extract_all_hydrographs(results_path, output_dir)
    
    print(f"Successfully extracted hydrographs for {len(hydrographs)} basins")

if __name__ == "__main__":
    main()