import logging
import concurrent.futures
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
import xarray as xr


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
from matplotlib.animation import FuncAnimation

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


# --- Moved out for multiprocessing pickling ---
def interpolate_single_timestep(i, t, gauges_data, grid_points, grid_shape, power, max_radius):
    frame = gauges_data[gauges_data['datetime'] == t]
    logging.info(f"IDW timestep {i+1}: {t} (type: {type(t)})")
    logging.info(f"Number of gauges: {len(frame)}")
    if not frame.empty:
        logging.debug(f"Gauge rainfall values: {frame['rain'].values}")
        logging.debug(f"Gauge coordinates: {list(zip(frame['ITM_X'].values, frame['ITM_Y'].values))}")
    else:
        logging.warning("No gauges found for this timestep.")
    if frame.empty:
        nan_grid = np.full(grid_shape, np.nan, dtype=np.float32)
        zero_grid = np.zeros(grid_shape, dtype=np.int16)
        return nan_grid, zero_grid
    coords = frame[['ITM_X', 'ITM_Y']].values
    values = frame['rain'].values if 'rain' in frame.columns else frame.iloc[:, 3].values
    # Only use valid (not NaN, not out-of-range) stations
    valid_mask = (~np.isnan(values)) & (values >= 0) & (values <= 50)
    coords = coords[valid_mask]
    values = values[valid_mask]
    if len(values) == 0:
        nan_grid = np.full(grid_shape, np.nan, dtype=np.float32)
        zero_grid = np.zeros(grid_shape, dtype=np.int16)
        logging.warning(f"No valid stations for timestep {t}!")
        return nan_grid, zero_grid
    dists = np.sqrt(((grid_points[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    with np.errstate(divide='ignore'):
        weights = 1 / np.power(dists, power)
        weights[dists > max_radius] = 0
        weights[dists == 0] = 1e12
    # Count number of stations used for each pixel (within radius and valid)
    gauge_count = np.sum((dists <= max_radius), axis=1).reshape(grid_shape)
    grid_vals = np.nansum(weights * values, axis=1) / np.nansum(weights, axis=1)
    result = grid_vals.reshape(grid_shape)
    if np.all(np.isnan(result)):
        logging.warning(f"Interpolated grid is all NaN for timestep {t}!")
    else:
        logging.info(f"Grid min: {np.nanmin(result)}, max: {np.nanmax(result)}")
    return result, gauge_count


# Wrapper for multiprocessing to unpack arguments (must be top-level for pickling)
def interpolate_single_timestep_wrapper(args):
    return interpolate_single_timestep(*args)

def idw_interpolation_grid(gauges_data, grid_edges, power=2, max_radius=50000, output_dir="output", date_range=None, grid_resolution=1000):
    os.makedirs(output_dir, exist_ok=True)
    x = np.arange(grid_edges['minx'], grid_edges['maxx'], grid_resolution)
    y = np.arange(grid_edges['miny'], grid_edges['maxy'], grid_resolution)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    if date_range:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        if hasattr(gauges_data['datetime'].dt, 'tz') and gauges_data['datetime'].dt.tz is not None:
            if start.tzinfo is None:
                start = start.tz_localize(gauges_data['datetime'].dt.tz)
            else:
                start = start.tz_convert(gauges_data['datetime'].dt.tz)
            if end.tzinfo is None:
                end = end.tz_localize(gauges_data['datetime'].dt.tz)
            else:
                end = end.tz_convert(gauges_data['datetime'].dt.tz)
        mask = (gauges_data['datetime'] >= start) & (gauges_data['datetime'] <= end)
        gauges_data = gauges_data[mask]
    times = np.sort(gauges_data['datetime'].unique())
    times = np.array(times, dtype='datetime64[ns]')
    grid_shape = xx.shape
    grid_array = np.full((len(times), grid_shape[0], grid_shape[1]), np.nan, dtype=np.float32)
    count_array = np.zeros((len(times), grid_shape[0], grid_shape[1]), dtype=np.int16)
    logging.info(f"Starting IDW interpolation for {len(times)} timesteps using {os.cpu_count()} CPUs...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(
            interpolate_single_timestep_wrapper,
            [(i, t, gauges_data, grid_points, grid_shape, power, max_radius) for i, t in enumerate(times)]
        ))
    for i, (rain_arr, count_arr) in enumerate(results):
        grid_array[i] = rain_arr
        count_array[i] = count_arr
    logging.info("IDW interpolation completed.")
    ds = xr.Dataset({
        "rain": (("time", "y", "x"), grid_array),
        "gauge_count": (("time", "y", "x"), count_array)
    }, coords={
        "time": times,
        "y": y,
        "x": x
    })
    nc_path = os.path.join(output_dir, "rain_grid.nc")
    ds.to_netcdf(nc_path, encoding={
        "rain": {"dtype": "float32", "zlib": True, "complevel": 4},
        "gauge_count": {"dtype": "int16", "zlib": True, "complevel": 1}
    })
    logging.info(f"Saved interpolated grid and gauge counts to {nc_path}")
    return ds






def animate_grid(ds, date_range=None):
    """
    Animate the rain grid for a selected date range.

    Parameters:
    ds : xarray.Dataset
        Dataset containing the rain grid
    date_range : tuple, optional
        (start_date, end_date) as strings or pd.Timestamp
    """
    rain = ds["rain"]
    times = ds["time"].values
    if date_range:
        start, end = np.datetime64(date_range[0]), np.datetime64(date_range[1])
        idx = (times >= start) & (times <= end)
        rain = rain[idx]
        times = times[idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(rain[0], origin="lower", cmap="Blues", interpolation="nearest")
    ax.set_title(str(times[0]))
    plt.colorbar(im, ax=ax, label="Rain (mm)")

    def update(frame):
        im.set_data(rain[frame])
        ax.set_title(str(times[frame]))
        return [im]

    anim = FuncAnimation(fig, update, frames=len(times), interval=300, blit=True)
    # Save animation as MP4 in output directory
    output_dir = os.getcwd()
    if hasattr(ds, 'encoding') and 'rain' in ds.encoding and 'source' in ds.encoding['rain']:
        output_dir = os.path.dirname(ds.encoding['rain']['source'])
    anim_path = os.path.join(output_dir, 'rain_grid_animation.mp4')
    try:
        anim.save(anim_path, writer='ffmpeg', dpi=150)
        print(f"Saved animation to {anim_path}")
    except Exception as e:
        print(f"Could not save animation as MP4: {e}")
    plt.show()
    return anim

def plot_top_rainfall_timesteps(idw_ds, gauges_data, gauges, top_n=5, output_dir=None):
    """
    Create a separate figure for each of the top_n timesteps with highest rainfall.
    Overlay gauges colored by their value at each timestep, and adjust colorscale per map.
    """
    # Find top_n timesteps with highest grid rainfall
    top_times = idw_ds['time'].values[np.argsort(idw_ds['rain'].max(dim=('y', 'x')).values)[-top_n:]]
    print(f"Top {top_n} timesteps (raw):", top_times)

    for idx, t in enumerate(top_times):
        print(f"\n--- Plotting timestep {idx+1}/{top_n} ---")
        print(f"Raw time value: {t} (type: {type(t)})")
        rain_grid = idw_ds['rain'].sel(time=t)
        print(f"Rain grid shape: {rain_grid.shape}, min: {np.nanmin(rain_grid.values)}, max: {np.nanmax(rain_grid.values)}")
        # Get gauge values for this timestep
        if isinstance(t, np.datetime64):
            t_pd = pd.to_datetime(str(t))
        else:
            t_pd = pd.to_datetime(t)
        print(f"Converted pandas datetime: {t_pd} (type: {type(t_pd)})")
        # Try to match gauges by rounding to minute
        gauges_data_dt_rounded = gauges_data.copy()
        gauges_data_dt_rounded['dt_rounded'] = gauges_data_dt_rounded['datetime'].dt.round('min')
        t_pd_rounded = pd.to_datetime(t_pd).round('min')
        gauges_at_t = gauges_data_dt_rounded[gauges_data_dt_rounded['dt_rounded'] == t_pd_rounded]
        print(f"Number of gauges at this timestep: {len(gauges_at_t)}")
        if not gauges_at_t.empty:
            print("Gauge rainfall values:", gauges_at_t['rain'].values)
            print("Gauge coordinates:", list(zip(gauges_at_t['ITM_X'].values, gauges_at_t['ITM_Y'].values)))
        else:
            print("No gauges found for this timestep.")
        # Set up figure in ITM coordinates (meters)
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plot rain grid
        vmin = np.nanmin(rain_grid.values)
        vmax = np.nanmax(rain_grid.values)
        im = ax.imshow(rain_grid, origin="lower", cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax,
                      extent=[idw_ds['x'].values[0], idw_ds['x'].values[-1], idw_ds['y'].values[0], idw_ds['y'].values[-1]])
        ax.set_title(f"Rainfall at {t_pd}")
        plt.colorbar(im, ax=ax, label="Rain (mm)")
        # Overlay gauges colored by their value
        if not gauges_at_t.empty:
            sc = ax.scatter(gauges_at_t['ITM_X'], gauges_at_t['ITM_Y'], c=gauges_at_t['rain'], cmap="Reds", edgecolor='black', s=80, vmin=vmin, vmax=vmax, label='Gauges')
            plt.colorbar(sc, ax=ax, label="Gauge Rain (mm)")
        ax.set_xlabel('ITM X (meters)')
        ax.set_ylabel('ITM Y (meters)')
        ax.legend()
        # Save figure to output_dir
        fname = f"rain_map_{idx+1}_{t_pd.strftime('%Y%m%d_%H%M')}.png"
        fig_path = os.path.join(output_dir, fname)
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        plt.close(fig)


# Main script
def main():
    # Load data
    gauges = pd.read_csv('/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/available_stations.csv')
    basins = gpd.read_file('/sci/labs/efratmorin/omripo/PhD/Data/Caravan/shapefiles/il/il_basin_shapes.shp')
    grid_resolution = 1000  # in meters

    # Process data
    # Read all CSV files in the specified directory
    # data_dir = '/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/5_stations_filtered_2022_2023'
    data_dir = '/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted'

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

    # Combine all files into a single DataFrame
    gauges_data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    # Merge with available stations to get ITM_X and ITM_Y
    gauges_data = pd.merge(gauges_data, gauges[['Station_ID', 'ITM_X', 'ITM_Y']], on='Station_ID', how='left')
    
    # Process the combined data
    gauges_data['datetime'] = pd.to_datetime(gauges_data['datetime'], dayfirst=True)
    # Remove timezone info to make all datetimes naive for robust matching with grid times
    if gauges_data['datetime'].dt.tz is not None:
        gauges_data['datetime'] = gauges_data['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
    # Ensure there is a 'rain' column (case-insensitive search)
    if 'rain' not in gauges_data.columns:
        # Try to find a likely candidate
        rain_candidates = [col for col in gauges_data.columns if col.lower() in ['rain', 'rainfall', 'precip', 'precipitation']]
        if rain_candidates:
            gauges_data.rename(columns={rain_candidates[0]: 'rain'}, inplace=True)
            print(f"Renamed column '{rain_candidates[0]}' to 'rain'.")
        else:
            raise ValueError("No 'rain' column found in gauge data. Please check your input CSVs for a valid rainfall column.")
    # print the first few rows to verify datetime conversion
    # print(gauges_data.head())
    
    # Sort and fill missing values
    gauges_data = gauges_data.sort_values(by=['Station_ID', 'datetime'])
    gauges_data = fill_missing(gauges_data, timestep_minutes=10)
    gauges_data = quality_check(gauges_data, log_folder='/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/log')

    # extract grid edges from the basins shapefile
    # if basins are not in EPSG:2039 - transform them from their current CRS to EPSG:2039
    if basins.crs is None or basins.crs.to_epsg() != 2039:
        basins = basins.to_crs(epsg=2039)    
    grid_edges = extract_grid_edges(basins)

    # Print the extracted grid edges
    # print("Extracted grid edges:", grid_edges)

    # plot a map with the basins, gauges, and grid edges - in a function
    # plot_map(basins, gauges, grid_edges)

    # Interpolate to grid and save as NetCDF
    # output_dir = '/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/5_stations_filtered_2022_2023/output'
    output_dir = '/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/output'
    # Export first 10 unique gauge datetimes

    # We'll get these after idw_ds is created
    idw_ds = idw_interpolation_grid(gauges_data, grid_edges, power=2, max_radius=50000, output_dir=output_dir, grid_resolution=grid_resolution, date_range=("2022-10-01", "2023-10-01"))


    # Animate the grid for a specific date range
    # animate_grid(idw_ds, date_range=("2022-01-01", "2022-01-03"))
    
    # Plot top rainfall timesteps as separate figures
    plot_top_rainfall_timesteps(idw_ds, gauges_data, gauges, top_n=5, output_dir=output_dir)
    

if __name__ == '__main__':
    main()