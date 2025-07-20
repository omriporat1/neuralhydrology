import os
import pandas as pd
import xarray as xr

# === Define paths ===
input_folder_path = r'C:\PhD\Data\Caravan\timeseries\csv\il'
output_folder_path = r'C:\PhD\Data\Caravan\Caravan_winter\timeseries\csv\il'
output_netcdf_folder_path = r'C:\PhD\Data\Caravan\Caravan_winter\timeseries\netcdf\il'

# Ensure output directories exist
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(output_netcdf_folder_path, exist_ok=True)

# Set the time frequency for reindexing (10-minute intervals)
target_freq = '10min'

def filter_winter(df):
    # Assumes 'date' is datetime and index or column
    if 'date' not in df.columns:
        df = df.reset_index()
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # For months Oct-Dec, use current year; for Jan-Apr, use previous year as 'winter_year'
    df['winter_year'] = df['year']
    df.loc[df['month'] < 10, 'winter_year'] = df['year'] - 1

    # Keep only Oct 1 - Apr 30 for each winter
    is_winter = (
        ((df['month'] >= 10) & (df['month'] <= 12)) | ((df['month'] >= 1) & (df['month'] <= 4))
    )
    # For Oct-Dec: from Oct 1 00:00, for Jan-Apr: until Apr 30 23:59
    df = df[is_winter]
    # Remove helper columns
    df = df.drop(columns=['year', 'month', 'winter_year'])
    return df

# === Process each file ===
for filename in os.listdir(input_folder_path):
    if filename.endswith(".csv"):
        input_file_path = os.path.join(input_folder_path, filename)
        output_csv_path = os.path.join(output_folder_path, filename)
        output_netcdf_path = os.path.join(output_netcdf_folder_path, filename.replace('.csv', '.nc'))

        # --- Load and process CSV ---
        df = pd.read_csv(
            input_file_path,
            parse_dates=['date'],
            dayfirst=False,
            na_values=['', ' ']
        )
        df = filter_winter(df)
        if df.empty:
            print(f"Skipped {filename}: no winter data found.")
            continue

        # Reindex to fill time gaps
        df.set_index('date', inplace=True)
        full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=target_freq)
        df = df.reindex(full_index)
        df.index.name = 'date'
        df.reset_index(inplace=True)

        # === Save cleaned CSV ===
        df.to_csv(output_csv_path, index=False)

        # === Save NetCDF ===
        df.set_index('date', inplace=True)
        df.index.freq = target_freq
        ds = xr.Dataset.from_dataframe(df)
        ds.to_netcdf(output_netcdf_path)

        print(f"Processed: {filename} → CSV: {os.path.basename(output_csv_path)}, NetCDF: {os.path.basename(output_netcdf_path)}")

print("All files processed: winter records copied, gaps filled, CSV and NetCDF saved.")