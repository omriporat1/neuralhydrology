# This file implements the entire process of calculating basin-average rainfall from point rain gauge data using IDW interpolation.

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import os

# Function to fill missing values
def fill_missing(data, timestep_minutes):
    """
    Fill missing values in the data for single timestep gaps.

    Parameters:
    data : pd.DataFrame
        DataFrame containing time series data for rain gauges.
    timestep_minutes : int
        Temporal resolution in minutes.

    Returns:
    pd.DataFrame
        DataFrame with missing values filled for single timestep gaps.
    """
    filled_data = []
    timestep = pd.Timedelta(minutes=timestep_minutes)

    for station_id, station_data in data.groupby('Station_ID'):
        station_data = station_data.sort_values(by='datetime')
        for col in station_data.columns:
            if col not in ['Station_ID', 'datetime']:
                gaps = (station_data['datetime'].diff() == timestep) & station_data[col].isna()
                station_data.loc[gaps, col] = station_data[col].interpolate(method='linear', limit=1, limit_direction='both')
        filled_data.append(station_data)

    return pd.concat(filled_data, ignore_index=True)

# Function for quality control
def quality_check(data, excessive_missing_threshold=0.5, max_value=50, log_folder='log'):
    """
    Perform quality control on the data and add flags to indicate missing or out-of-range values.

    Parameters:
    data : pd.DataFrame
        DataFrame containing time series data for rain gauges.
    excessive_missing_threshold : float, optional
        Threshold for excessive missing data (default is 0.5).
    max_value : float, optional
        Maximum valid value for data (default is 50).
    log_folder : str, optional
        Folder to save the log files (default is 'log').

    Returns:
    pd.DataFrame
        DataFrame with quality-controlled data and flags.
    """
    os.makedirs(log_folder, exist_ok=True)
    unified_log = []

    # Initialize QC_Flag column in the original data
    data['QC_Flag'] = 0

    for station_id, station_data in data.groupby('Station_ID'):
        total_values = len(station_data)
        missing_values = station_data.isna().sum().sum()
        missing_percentage = missing_values / total_values

        # Add a flag column to indicate missing or out-of-range values
        station_data['QC_Flag'] = 0

        # Log missing values and flagged values per station
        for col in station_data.columns:
            if col not in ['Station_ID', 'datetime', 'QC_Flag']:
                out_of_range = (station_data[col] < 0) | (station_data[col] > max_value)
                station_data.loc[out_of_range, 'QC_Flag'] = 1  # Flag out-of-range values
                station_data.loc[station_data[col].isna(), 'QC_Flag'] = 2  # Flag missing values

                unified_log.append({
                    'Station_ID': station_id,
                    'Column': col,
                    'Flagged Values': out_of_range.sum(),
                    'Missing Values': missing_values,
                    'Missing Percentage': missing_percentage
                })

        # Flag stations with excessive missing data
        if missing_percentage > excessive_missing_threshold:
            print(f"Warning: Station {station_id} has excessive missing data ({missing_percentage:.2%}).")

        # Update the original data with changes made to station_data
        data.loc[station_data.index, 'QC_Flag'] = station_data['QC_Flag']

    # Save unified log to CSV
    unified_log_df = pd.DataFrame(unified_log)
    unified_log_df.to_csv(os.path.join(log_folder, 'quality_control_log.csv'), index=False)

    return data





# Main script
def main():
    # Load data
    gauges = pd.read_csv(r'C:\PhD\Data\IMS\Data_by_station\available_stations.csv')
    basins = gpd.read_file(r'C:\PhD\Data\Caravan\shapefiles\il\il_basin_shapes.shp')

    # Process data
    # Read all CSV files in the specified directory
    # data_dir = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted'
    data_dir = r'C:\PhD\Data\IMS\Data_by_station\5_stations_filtered_2022_2023'

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

    # Combine all files into a single DataFrame
    gauges_data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

   
    # Process the combined data
    gauges_data['datetime'] = pd.to_datetime(gauges_data['datetime'], dayfirst=True)
    # print the first few rows to verify datetime conversion
    print(gauges_data.head())
    
    # Sort and fill missing values
    gauges_data = gauges_data.sort_values(by=['Station_ID', 'datetime'])
    gauges_data = fill_missing(gauges_data, timestep_minutes=10)
    gauges_data = quality_check(gauges_data, log_folder=r'C:\PhD\Data\IMS\Data_by_station\5_stations_filtered_2022_2023\log')

    
if __name__ == '__main__':
    main()