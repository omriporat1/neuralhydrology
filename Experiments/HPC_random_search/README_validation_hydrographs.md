# Validation Hydrographs Extraction

This script processes NeuralHydrology model outputs and creates validation hydrographs for extreme events and full time periods.

## Purpose

The script performs the following tasks:

1. Extracts, for each basin and each year, the event windows around the annual maximum discharge events
2. For each basin, saves hydrographs (CSV and plots) for:
   - The event windows around maximum discharge events
   - The entire validation period
3. Includes observed, simulated, and shifted observed values (3-hour shift) in the outputs
4. Calculates and displays performance metrics (NSE, RMSE) on the plots

## How to Run

### Option 1: Run with Batch File
1. Double-click the `run_script.bat` file in Windows Explorer
2. The script will execute and show progress in the command window

### Option 2: Run in Python Environment
1. Open a command prompt or PowerShell
2. Navigate to the script directory:
   ```
   cd "c:\PhD\Python\neuralhydrology\Experiments\HPC_random_search"
   ```
3. Run the script:
   ```
   python Create_validation_hydrographs_fixed.py
   ```

## Output Files

For each basin in each model run, the script creates:

1. `{basin}_validation_hydrographs.csv` - CSV file with date, observed, simulated, and shifted values for all event windows
2. `{basin}_validation_hydrographs.png` - Plot of all event hydrographs
3. `{basin}_full_period.csv` - CSV file with the entire validation period data
4. `{basin}_full_period.png` - Plot of the entire validation period with highlighted event windows

## Configuration

- The script is currently set to process only the first run directory for testing
- To process all runs, uncomment the line: `# run_dirs = run_dirs[:1]`
- The script is configured to run locally with CPU (set `RUN_LOCALLY = False` to use original HPC configuration)

## Error Handling

The script includes comprehensive error handling to catch and report issues at each step:
- Missing directories or files
- Basins not found in the maximum events data
- Missing date ranges in model outputs
- Issues with model evaluation or visualization
