import os
import pandas as pd
import xarray as xr

# Define paths
input_folder = r"C:/PhD/Data/Caravan/timeseries/csv/il"
output_folder_netcdf = r"C:/PhD/Data/Caravan/timeseries/netcdf/il"

# Ensure output directories exist
os.makedirs(output_folder_netcdf, exist_ok=True)


# Process each CSV file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.startswith('il') and file_name.endswith('.csv'):
        input_csv_path = os.path.join(input_folder, file_name)

        # Extract numeric ID (e.g., 1234 from 1234_combined.csv)
        numeric_id = file_name.split('_')[1].split('.')[0]  # Assumes the format is il_XXXX.csv
        # basin_id = numeric_id

        # Read time series data
        df = pd.read_csv(input_csv_path, na_values=['', ' '], parse_dates=['datetime'], dayfirst=True)
        # change the column 'datetime' to 'date':
        df.rename(columns={'datetime': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Convert DataFrame to xarray Dataset
        ds = xr.Dataset.from_dataframe(df)

        # Save as NetCDF
        netcdf_filename = f"il_{numeric_id}.nc"
        ds.to_netcdf(os.path.join(output_folder_netcdf, netcdf_filename))



        print(f"Saved: {file_name} → {netcdf_filename}")

print("All files processed.")
