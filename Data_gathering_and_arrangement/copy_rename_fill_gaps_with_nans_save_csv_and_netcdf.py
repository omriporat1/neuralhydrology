import os
import pandas as pd
import xarray as xr

# === Define paths ===
input_folder_path = r'C:\PhD\Data\IHS_IMS_wide\IHS_IMS_wide_unified_2025-05-15'
output_folder_path = r'C:\PhD\Data\Caravan\timeseries\csv\il'
output_netcdf_folder_path = r'C:\PhD\Data\Caravan\timeseries\netcdf\il'

# Ensure output directories exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(output_netcdf_folder_path, exist_ok=True)

# Set the time frequency for reindexing (10-minute intervals)
target_freq = '10min'

# === Process each file ===
for filename in os.listdir(input_folder_path):
    if filename.endswith("_combined.csv"):
        # Extract numeric ID
        numeric_id = filename.split('_')[0]
        new_filename = f"il_{numeric_id}.csv"

        # Define paths
        input_file_path = os.path.join(input_folder_path, filename)
        output_csv_path = os.path.join(output_folder_path, new_filename)
        output_netcdf_path = os.path.join(output_netcdf_folder_path, f"il_{numeric_id}.nc")

        # --- Load and process CSV ---
        df = pd.read_csv(
            input_file_path,
            parse_dates=['datetime'],
            dayfirst=False,  # safer given your format
            na_values=['', ' ']
        )
        df.rename(columns={'datetime': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Reindex to fill time gaps
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq)
        df = df.reindex(full_index)

        # === Save cleaned CSV ===
        df.index.name = 'date'
        df.reset_index(inplace=True)
        df.to_csv(output_csv_path, index=False)

        # === Save NetCDF ===
        df.set_index('date', inplace=True)  # Required again for xarray
        df.index.freq = target_freq  # Explicitly assign frequency
        ds = xr.Dataset.from_dataframe(df)
        ds.to_netcdf(output_netcdf_path)

        print(f"Processed: {filename} → CSV: {new_filename}, NetCDF: il_{numeric_id}.nc")

print("All files processed: copied, gaps filled, CSV and NetCDF saved.")
