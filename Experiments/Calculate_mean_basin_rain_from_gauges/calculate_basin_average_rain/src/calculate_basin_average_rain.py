# This file implements the entire process of calculating basin-average rainfall from point rain gauge data using IDW interpolation.

import pandas as pd
import numpy as np
import geopandas as gpd
from scipy.spatial import cKDTree
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests

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

# Function to extract grid edges
def extract_grid_edges(basins, buffer=5000):
    """
    Extract the grid edges from the basins shapefile and return the overall min and max values of x and y as integers, with an optional buffer.

    Parameters:
    basins : gpd.GeoDataFrame
        GeoDataFrame containing the basin shapes.
    buffer : int, optional
        Buffer value to add/subtract to the min and max x and y values (default is 5000).

    Returns:
    dict
        Dictionary containing the overall min and max values of x and y with keys ['minx', 'miny', 'maxx', 'maxy'] as integers, adjusted by the buffer.
    """
    # Ensure the basins GeoDataFrame has a valid geometry column
    if 'geometry' not in basins.columns:
        raise ValueError("The provided GeoDataFrame does not contain a 'geometry' column.")

    # Extract the bounds (minx, miny, maxx, maxy) for each basin
    bounds = basins.geometry.bounds

    # Calculate the overall min and max values of x and y and convert to integers
    overall_bounds = {
        'minx': int(bounds['minx'].min()) - buffer,
        'miny': int(bounds['miny'].min()) - buffer,
        'maxx': int(bounds['maxx'].max()) + buffer,
        'maxy': int(bounds['maxy'].max()) + buffer
    }

    return overall_bounds



def plot_map(basins, gauges, grid_edges):
    """
    Plot a map with the basins, gauges, grid edges, and international borders in ITM (EPSG:2039).

    Parameters:
    basins : gpd.GeoDataFrame
        GeoDataFrame containing the basin shapes.
    gauges : pd.DataFrame
        DataFrame containing the rain gauge data.
    grid_edges : dict
        Dictionary containing the grid edges with keys ['minx', 'miny', 'maxx', 'maxy'].
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot basins
    basins.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5)

    # Plot gauges
    ax.scatter(gauges['ITM_X'], gauges['ITM_Y'], color='red', label='Rain Gauges')

    # Draw grid edges
    rect = plt.Rectangle((grid_edges['minx'], grid_edges['miny']),
                         grid_edges['maxx'] - grid_edges['minx'],
                         grid_edges['maxy'] - grid_edges['miny'],
                         linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # --- New code to download and cache Natural Earth data ---
    # Define the path for the cached shapefile
    cache_dir = os.path.join(os.path.expanduser('~'), '.geopandas_cache')
    os.makedirs(cache_dir, exist_ok=True)
    shapefile_path = os.path.join(cache_dir, 'ne_10m_admin_0_countries.shp')

    # Download and unzip if the shapefile doesn't exist
    if not os.path.exists(shapefile_path):
        url = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
        zip_path = os.path.join(cache_dir, 'ne_10m_admin_0_countries.zip')
        
        try:
            import requests
            import zipfile
            import io

            print("Downloading high-resolution Natural Earth countries dataset...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            # Unzip the content
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(cache_dir)
            print(f"Dataset cached at {cache_dir}")

        except Exception as e:
            print(f"Failed to download or process the Natural Earth dataset: {e}")
            # As a fallback, we will not plot the borders
            shapefile_path = None

    # Plot the borders if the shapefile is available
    if shapefile_path and os.path.exists(shapefile_path):
        world = gpd.read_file(shapefile_path)
        world = world.to_crs(epsg=2039)
        world.boundary.plot(ax=ax, edgecolor='gray', linewidth=0.5)
    # --- End of new code ---

    map_buffer = 200000  # Buffer for the map edges
    # Set plot limits to focus on the area of interest
    ax.set_xlim(grid_edges['minx'] - map_buffer, grid_edges['maxx'] + map_buffer)
    ax.set_ylim(grid_edges['miny'] - map_buffer, grid_edges['maxy'] + map_buffer)

    ax.set_title('Basin Map with Rain Gauges, Grid Edges, and International Borders (EPSG:2039)')
    ax.set_xlabel('X Coordinate (meters)')
    ax.set_ylabel('Y Coordinate (meters)')
    ax.legend()

    plt.show()


# Main script
def main():
    # Load data
    gauges = pd.read_csv(r'C:\PhD\Data\IMS\Data_by_station\available_stations.csv')
    basins = gpd.read_file(r'C:\PhD\Data\Caravan\shapefiles\il\il_basin_shapes.shp')
    grid_resolution = 1000  # in meters

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

    # extract grid edges from the basins shapefile
    # if basins are not in EPSG:2039 - transform them from their current CRS to EPSG:2039
    if basins.crs is None or basins.crs.to_epsg() != 2039:
        basins = basins.to_crs(epsg=2039)    
    grid_edges = extract_grid_edges(basins)

    # Print the extracted grid edges
    print("Extracted grid edges:", grid_edges)

    # plot a map with the basins, gauges, and grid edges - in a function
    plot_map(basins, gauges, grid_edges)


    

if __name__ == '__main__':
    main()