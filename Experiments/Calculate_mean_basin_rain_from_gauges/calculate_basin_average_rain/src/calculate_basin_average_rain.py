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
    Perform quality control on the data.

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
        DataFrame with quality-controlled data.
    """
    os.makedirs(log_folder, exist_ok=True)
    unified_log = []

    for station_id, station_data in data.groupby('Station_ID'):
        total_values = len(station_data)
        missing_values = station_data.isna().sum().sum()
        missing_percentage = missing_values / total_values

        # Log missing values and flagged values per station
        for col in station_data.columns:
            if col not in ['Station_ID', 'datetime']:
                out_of_range = (station_data[col] < 0) | (station_data[col] > max_value)
                unified_log.append({
                    'Station_ID': station_id,
                    'Column': col,
                    'Flagged Values': out_of_range.sum(),
                    'Missing Values': missing_values,
                    'Missing Percentage': missing_percentage
                })
                station_data.loc[out_of_range, col] = np.nan

        # Flag stations with excessive missing data
        if missing_percentage > excessive_missing_threshold:
            print(f"Warning: Station {station_id} has excessive missing data ({missing_percentage:.2%}).")

    # Save unified log to CSV
    unified_log_df = pd.DataFrame(unified_log)
    unified_log_df.to_csv(os.path.join(log_folder, 'quality_control_log.csv'), index=False)

    return data

# Function for IDW interpolation
def interpolate_idw(points, values, grid_x, grid_y, power=2):
    """
    Perform IDW interpolation.

    Parameters:
    points : array-like
        Coordinates of the known data points.
    values : array-like
        Values at the known data points.
    grid_x : array-like
        X-coordinates of the grid.
    grid_y : array-like
        Y-coordinates of the grid.
    power : float
        Power parameter for IDW.

    Returns:
    np.ndarray
        Interpolated grid values.
    """
    tree = cKDTree(points)
    grid_values = np.zeros(grid_x.shape)
    for i in range(grid_x.shape[0]):
        for j in range(grid_x.shape[1]):
            distances, indices = tree.query((grid_x[i, j], grid_y[i, j]), k=len(points))
            distances = np.where(distances == 0, 1e-10, distances)
            weights = 1 / distances**power
            weights /= weights.sum()
            grid_values[i, j] = np.dot(weights, values[indices])
    return grid_values

# Function to compute basin average rainfall
def compute_basin_avg(grid, basins):
    """
    Compute basin average rainfall.

    Parameters:
    grid : np.ndarray
        Interpolated rainfall grid.
    basins : gpd.GeoDataFrame
        GeoDataFrame containing basin polygons.

    Returns:
    pd.DataFrame
        DataFrame with basin average rainfall.
    """
    stats = zonal_stats(basins, grid, stats=['mean'], geojson_out=True)
    basin_avg = pd.DataFrame({
        'Basin_ID': [feature['properties']['id'] for feature in stats],
        'Avg_Rainfall': [feature['properties']['mean'] for feature in stats]
    })
    return basin_avg

# Function to summarize statistics
def summarize_statistics(data):
    """
    Summarize rainfall statistics.

    Parameters:
    data : pd.DataFrame
        DataFrame containing basin rainfall data.

    Returns:
    pd.DataFrame
        DataFrame with summary statistics.
    """
    summary = data.describe()
    summary.to_csv('summary_statistics.csv')
    return summary

# Function to plot rainfall map
def plot_rain_map(grid, basins, gauges, timestep):
    """
    Plot rainfall map for a given timestep.

    Parameters:
    grid : np.ndarray
        Interpolated rainfall grid.
    basins : gpd.GeoDataFrame
        GeoDataFrame containing basin polygons.
    gauges : pd.DataFrame
        DataFrame containing gauge locations and rainfall values.
    timestep : str
        Timestep for the plot title.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, extent=(grid.min(), grid.max()), origin='lower', cmap='Blues')
    basins.boundary.plot(ax=plt.gca(), color='black')
    plt.scatter(gauges['ITM_X'], gauges['ITM_Y'], c=gauges['Rain'], cmap='Reds', edgecolor='black')
    plt.title(f'Rainfall Map - {timestep}')
    plt.colorbar(label='Rainfall (mm)')
    plt.show()

# Function to create grid
def create_grid(x_min, x_max, y_min, y_max, resolution):
    """
    Create a grid of points for interpolation.

    Parameters:
    x_min : float
        Minimum x-coordinate.
    x_max : float
        Maximum x-coordinate.
    y_min : float
        Minimum y-coordinate.
    y_max : float
        Maximum y-coordinate.
    resolution : float
        Distance between grid points.

    Returns:
    grid_x : array
        X-coordinates of the grid points.
    grid_y : array
        Y-coordinates of the grid points.
    """
    import numpy as np

    grid_x = np.arange(x_min, x_max, resolution)
    grid_y = np.arange(y_min, y_max, resolution)

    return np.meshgrid(grid_x, grid_y)

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

    # Create grid
    grid_x, grid_y = create_grid(x_min=0, x_max=100, y_min=0, y_max=100, resolution=1)

    # Interpolate rainfall
    points = gauges[['ITM_X', 'ITM_Y']].values
    values = gauges_data.iloc[:, 1:].mean(axis=1).values
    grid = interpolate_idw(points, values, grid_x, grid_y)

    # Compute basin averages
    basin_avg = compute_basin_avg(grid, basins)
    basin_avg.to_csv('basin_avg_rainfall.csv', index=False)

    # Summarize statistics
    summarize_statistics(basin_avg)

    # Plot example map
    plot_rain_map(grid, basins, gauges, timestep='2025-08-03 12:00')

if __name__ == '__main__':
    main()