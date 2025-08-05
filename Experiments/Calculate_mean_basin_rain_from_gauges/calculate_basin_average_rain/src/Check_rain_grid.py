import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def load_rain_grid(nc_path):
    """Load the NetCDF rain grid as xarray.Dataset."""
    return xr.open_dataset(nc_path)

def load_gauge_data(csv_path):
    """Load gauge data CSV and ensure datetime is parsed."""
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    return df

def load_extreme_dates(csv_path):
    """Load unique dates from annual_max_discharge_dates.csv."""
    df = pd.read_csv(csv_path)
    # Parse max_date column to datetime
    df['max_date'] = pd.to_datetime(df['max_date'], dayfirst=True, errors='coerce')
    unique_dates = pd.Series(df['max_date'].unique()).dropna().sort_values()
    return unique_dates

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
    rain = rain_ds['rain']
    x = rain_ds['x'].values
    y = rain_ds['y'].values
    times = pd.to_datetime(rain_ds['time'].values)
    for date in extreme_dates:
        # Find closest timestep in grid
        idx = np.argmin(np.abs(times - date))
        grid = rain[idx].values
        # Get gauge values for this date
        gauges_at_t = gauges_df[gauges_df['datetime'] == date]
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(grid, origin="lower", cmap="Blues", interpolation="nearest",
                      extent=[x[0], x[-1], y[0], y[-1]])
        ax.set_title(f"Rainfall at {date.strftime('%Y-%m-%d')}")
        plt.colorbar(im, ax=ax, label="Rain (mm)")
        if not gauges_at_t.empty:
            sc = ax.scatter(gauges_at_t['ITM_X'], gauges_at_t['ITM_Y'], c=gauges_at_t['rain'], cmap="Reds", edgecolor='black', s=80, label='Gauges')
            plt.colorbar(sc, ax=ax, label="Gauge Rain (mm)")
        ax.set_xlabel('ITM X (meters)')
        ax.set_ylabel('ITM Y (meters)')
        ax.legend()
        fname = f"rain_extreme_{date.strftime('%Y%m%d')}.png"
        fig_path = os.path.join(output_dir, fname)
        plt.savefig(fig_path, bbox_inches='tight')
        print(f"Saved figure to {fig_path}")
        plt.close(fig)
    # Optionally, create an animation across all extreme dates
    fig, ax = plt.subplots(figsize=(8, 8))
    def update(i):
        ax.clear()
        grid = rain[np.argmin(np.abs(times - extreme_dates[i]))].values
        im = ax.imshow(grid, origin="lower", cmap="Blues", interpolation="nearest",
                      extent=[x[0], x[-1], y[0], y[-1]])
        ax.set_title(f"Rainfall at {extreme_dates[i].strftime('%Y-%m-%d')}")
        gauges_at_t = gauges_df[gauges_df['datetime'] == extreme_dates[i]]
        if not gauges_at_t.empty:
            sc = ax.scatter(gauges_at_t['ITM_X'], gauges_at_t['ITM_Y'], c=gauges_at_t['rain'], cmap="Reds", edgecolor='black', s=80, label='Gauges')
        ax.set_xlabel('ITM X (meters)')
        ax.set_ylabel('ITM Y (meters)')
        ax.legend()
        return [im]
    anim = FuncAnimation(fig, update, frames=len(extreme_dates), interval=1000, blit=False)
    anim_path = os.path.join(output_dir, 'rain_extreme_dates_animation.mp4')
    try:
        anim.save(anim_path, writer='ffmpeg', dpi=150)
        print(f"Saved animation to {anim_path}")
    except Exception as e:
        print(f"Could not save animation as MP4: {e}")
    plt.close(fig)

def main():
    # Paths (edit as needed)
    nc_path = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output\rain_grid.nc'
    gauge_csv = r'C:\PhD\Data\IMS\Data_by_station\available_stations.csv'
    gauge_data_csv = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\all_gauges.csv'  # or merge all station CSVs
    extreme_dates_csv = r'C:\PhD\Python\neuralhydrology\Experiments\extract_extreme_events\from_daily_max\annual_max_discharge_dates.csv'
    output_dir = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output'
    os.makedirs(output_dir, exist_ok=True)
    # Load data
    rain_ds = load_rain_grid(nc_path)
    gauges_df = load_gauge_data(gauge_data_csv)
    # Merge with locations
    stations_df = pd.read_csv(gauge_csv)
    gauges_df = pd.merge(gauges_df, stations_df[['Station_ID', 'ITM_X', 'ITM_Y']], on='Station_ID', how='left')
    extreme_dates = load_extreme_dates(extreme_dates_csv)
    # Stats
    rain_grid_stats(rain_ds, output_dir)
    # Animation/figures
    animate_extreme_dates(rain_ds, gauges_df, extreme_dates, output_dir)

if __name__ == '__main__':
    main()
