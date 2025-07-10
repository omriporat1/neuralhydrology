from pathlib import Path

def check_files():
    experiments_dir = Path("C:/PhD/Python/neuralhydrology/Experiments/HPC_random_search/results/job_41780893")
    max_events_path = Path("C:/PhD/Python/neuralhydrology/Experiments/extract_extreme_events/from_daily_max/annual_max_discharge_dates.csv")
    
    print(f"Checking if experiments directory exists: {experiments_dir.exists()}")
    print(f"Checking if max events file exists: {max_events_path.exists()}")
    
    if experiments_dir.exists():
        print("Contents of experiments directory:")
        for item in experiments_dir.iterdir():
            print(f"  {item}")
    
    if max_events_path.exists():
        print(f"Max events file size: {max_events_path.stat().st_size} bytes")
        
    # Test read of max events file
    try:
        import pandas as pd
        df = pd.read_csv(max_events_path)
        print(f"Successfully read max events file. Shape: {df.shape}")
        print("First few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error reading max events file: {str(e)}")

if __name__ == "__main__":
    try:
        check_files()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
