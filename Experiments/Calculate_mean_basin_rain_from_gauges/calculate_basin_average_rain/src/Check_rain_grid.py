import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import geopandas as gpd


def load_rain_grid(nc_path):
    """Load the NetCDF rain grid as xarray.Dataset."""
    return xr.open_dataset(nc_path)

def load_gauge_data(csv_path):
    """Load gauge data CSV and ensure datetime is parsed."""
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    return df

def load_event_days(txt_path):
    """Load event days from a txt file, one date per line, format YYYY-MM-DD."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Parse each line to datetime.date
    dates = [pd.to_datetime(line, format='%Y-%m-%d', errors='coerce').date() for line in lines]
    # Remove any failed parses
    dates = [d for d in dates if pd.notnull(d)]
    return dates

def rain_grid_stats(rain_ds, output_dir):
    """Compute and save statistics about the rain grid."""
    stats = {}
    rain = rain_ds['rain']
    times = rain_ds['time'].values
    stats['start_time'] = str(times[0])
    stats['end_time'] = str(times[-1])
    stats['num_timesteps'] = len(times)
    # Find missing timesteps (if time is regular, check for gaps)
    time_diffs = np.diff(times)
    expected = np.median(time_diffs)
    missing = np.sum(time_diffs != expected)
    stats['missing_timesteps'] = int(missing)
    stats['min'] = float(np.nanmin(rain.values))
    stats['max'] = float(np.nanmax(rain.values))
    stats['mean'] = float(np.nanmean(rain.values))
    stats['nan_percent_per_timestep'] = [float(np.mean(np.isnan(rain[i].values)))*100 for i in range(rain.shape[0])]
    stats['overall_nan_percent'] = float(np.mean(np.isnan(rain.values)))*100
    # Save stats to file
    stats_path = os.path.join(output_dir, 'rain_grid_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved rain grid stats to {stats_path}")
    return stats

def animate_extreme_dates(rain_ds, gauges_df, extreme_dates, output_dir):
    """Create and save an animation for each extreme date, overlaying gauge values."""
    # Load basins shapefile
    basins_path = r'C:\PhD\Data\Caravan\shapefiles\il\il_basin_shapes.shp'
    basins = gpd.read_file(basins_path)
    if basins.crs is None or basins.crs.to_epsg() != 2039:
        basins = basins.to_crs(epsg=2039)
    rain = rain_ds['rain']
    x = rain_ds['x'].values
    y = rain_ds['y'].values
    times = pd.to_datetime(rain_ds['time'].values)
    # Set constant colorscales
    vmin = 0
    vmax = 1.5  # Not too extreme, but visible
    gauge_vmin = 0
    gauge_vmax = 1.5
    # Create a GIF animation for each event date, showing all 24 hours (every 10 min)
    fps = 10  # Default frames per second
    for date in extreme_dates:
        day_mask = times.date == date.date()
        day_indices = np.where(day_mask)[0]
        if len(day_indices) == 0:
            print(f"No rain grid timesteps found for {date.strftime('%Y-%m-%d')}")
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax_grid = divider.append_axes("right", size="5%", pad=0.05)
        cax_gauge = divider.append_axes("bottom", size="5%", pad=0.4)
        def update_hourly(frame_idx):
            ax.clear()
            cax_grid.cla()
            cax_gauge.cla()
            idx = day_indices[frame_idx]
            grid = rain[idx].values
            im = ax.imshow(grid, origin="lower", cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax,
                          extent=[x[0], x[-1], y[0], y[-1]])
            basins.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
            tstamp = times[idx]
            ax.set_title(f"Rainfall at {tstamp.strftime('%Y-%m-%d %H:%M')}")
            gauges_at_t = gauges_df[gauges_df['datetime'] == tstamp]
            sc = None
            if not gauges_at_t.empty:
                sc = ax.scatter(gauges_at_t['ITM_X'], gauges_at_t['ITM_Y'], c=gauges_at_t['Rain'], cmap="Reds", edgecolor='black', s=80, vmin=gauge_vmin, vmax=gauge_vmax, label='Gauges')
            ax.set_xlabel('ITM X (meters)')
            ax.set_ylabel('ITM Y (meters)')
            ax.legend()
            # Add colorbars
            fig.colorbar(im, cax=cax_grid, label="Rain (mm)")
            if sc is not None:
                fig.colorbar(sc, cax=cax_gauge, orientation='horizontal', label="Gauge Rain (mm)")
            return [im]
        anim = FuncAnimation(fig, update_hourly, frames=len(day_indices), interval=1000/fps, blit=False)
        anim_path = os.path.join(output_dir, f'rain_extreme_{date.strftime("%Y%m%d")}_animation.gif')
        try:
            anim.save(anim_path, writer='pillow', dpi=150, fps=fps)
            print(f"Saved animation to {anim_path}")
        except Exception as e:
            print(f"Could not save animation as gif: {e}")
        plt.close(fig)
    # Animation across all extreme dates
    fig, ax = plt.subplots(figsize=(8, 8))
    def update(i):
        ax.clear()
        idx = np.argmin(np.abs(times - extreme_dates[i]))
        grid = rain[idx].values
        im = ax.imshow(grid, origin="lower", cmap="Blues", interpolation="nearest", vmin=vmin, vmax=vmax,
                      extent=[x[0], x[-1], y[0], y[-1]])
        basins.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
        ax.set_title(f"Rainfall at {extreme_dates[i].strftime('%Y-%m-%d')}")
        gauges_at_t = gauges_df[gauges_df['datetime'] == extreme_dates[i]]
        if not gauges_at_t.empty:
            sc = ax.scatter(gauges_at_t['ITM_X'], gauges_at_t['ITM_Y'], c=gauges_at_t['Rain'], cmap="Reds", edgecolor='black', s=80, vmin=gauge_vmin, vmax=gauge_vmax, label='Gauges')
        ax.set_xlabel('ITM X (meters)')
        ax.set_ylabel('ITM Y (meters)')
        ax.legend()
        return [im]
    anim = FuncAnimation(fig, update, frames=len(extreme_dates), interval=1000, blit=False)
    anim_path = os.path.join(output_dir, 'rain_extreme_dates_animation.gif')
    try:
        anim.save(anim_path, writer='pillow', dpi=150)
        print(f"Saved animation to {anim_path}")
    except Exception as e:
        print(f"Could not save animation as gif: {e}")
    plt.close(fig)

def main():
    # Paths (edit as needed)
    nc_path = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output\rain_grid.nc'
    gauge_csv = r'C:\PhD\Data\IMS\Data_by_station\available_stations.csv'
    event_days_txt = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output\max_dates_2022_2023.txt'
    output_dir = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output'
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    rain_ds = load_rain_grid(nc_path)
    # Get relevant event days from txt
    event_days = load_event_days(event_days_txt)
    # Convert event_days to pandas Timestamps and filter to rain grid time range
    rain_times = pd.to_datetime(rain_ds['time'].values)
    min_time = rain_times.min()
    max_time = rain_times.max()
    # Convert event_days to Timestamps
    event_days_ts = pd.to_datetime(event_days)
    # Only keep event days within rain grid time range
    event_days_in_range = [d for d in event_days_ts if (d >= min_time) and (d <= max_time)]
    if not event_days_in_range:
        print("No event days within rain grid time range. Exiting.")
        return
    # Merge only relevant rows from all gauge CSVs
    gauge_data_dir = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted'
    gauge_csv_files = [os.path.join(gauge_data_dir, f) for f in os.listdir(gauge_data_dir) if f.endswith('.csv') and 'available_stations' not in f]
    cache_path = os.path.join(output_dir, f'relevant_gauges_{os.path.splitext(os.path.basename(event_days_txt))[0]}.csv')
    if os.path.exists(cache_path):
        print(f"Loading cached relevant gauges from {cache_path}")
        gauges_df = pd.read_csv(cache_path, parse_dates=['datetime'])
    else:
        relevant_gauges = []
        for f in gauge_csv_files:
            df = pd.read_csv(f, parse_dates=['datetime'], dayfirst=True)
            # Remove timezone info for robust matching
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
            # Match by date only (ignore hour)
            df['date'] = df['datetime'].dt.date
            event_days_dates = set([d.date() for d in event_days_in_range])
            df_relevant = df[df['date'].isin(event_days_dates)]
            if not df_relevant.empty:
                relevant_gauges.append(df_relevant)
        gauges_df = pd.concat(relevant_gauges, ignore_index=True)
        # Merge with locations before caching
        stations_df = pd.read_csv(gauge_csv)
        gauges_df = pd.merge(gauges_df, stations_df[['Station_ID', 'ITM_X', 'ITM_Y']], on='Station_ID', how='left')
        gauges_df.to_csv(cache_path, index=False)
        print(f"Saved relevant gauges to {cache_path}")
    # Stats
    # rain_grid_stats(rain_ds, output_dir)
    # Animation/figures
    animate_extreme_dates(rain_ds, gauges_df, event_days_in_range, output_dir)

if __name__ == '__main__':
    main()
