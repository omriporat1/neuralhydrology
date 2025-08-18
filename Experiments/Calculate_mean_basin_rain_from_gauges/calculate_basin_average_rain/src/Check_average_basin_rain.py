import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# Progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


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

# New: read per-basin CSVs into a time x basin_id DataFrame of mean_rain
def read_basin_timeseries(csv_dir):
    csvs = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
            if f.lower().endswith('.csv') and f.lower() not in ('run.log',)]
    if not csvs:
        raise FileNotFoundError(f"No basin CSVs found in {csv_dir}")
    series = []
    for fp in sorted(csvs):
        stem = os.path.splitext(os.path.basename(fp))[0]
        try:
            df = pd.read_csv(fp, usecols=['date', 'mean_rain'])
        except Exception:
            df = pd.read_csv(fp)
            if 'date' not in df.columns or 'mean_rain' not in df.columns:
                continue
        # generator uses dd/mm/YYYY HH:MM
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['date']).drop_duplicates(subset=['date']).set_index('date').sort_index()
        s = df['mean_rain'].astype(float)
        s.name = stem
        series.append(s)
    if not series:
        raise RuntimeError("No usable basin CSVs with columns ['date','mean_rain'] were read.")
    ts = pd.concat(series, axis=1).sort_index()  # index=time, columns=basin_id
    return ts

def animate_extreme_dates(ts_basin, gauges_df, extreme_dates, output_dir):
    """Animate basins colored by mean rain for each event day, overlaying gauges."""
    # Load basins shapefile
    basins_path = r'C:\PhD\Data\Caravan\shapefiles\il\il_basin_shapes.shp'
    basins = gpd.read_file(basins_path)
    if basins.crs is None or basins.crs.to_epsg() != 2039:
        basins = basins.to_crs(epsg=2039)

    # ID column that matches CSV stems (e.g., 'il_12130')
    id_col = 'gauge_id'
    if id_col not in basins.columns:
        raise KeyError(f"'{id_col}' not found in basins shapefile. Adjust id_col to match your attribute.")

    basins[id_col] = basins[id_col].astype(str)
    # Keep only basins we have CSVs for
    common = basins[basins[id_col].isin(ts_basin.columns)].copy()
    if common.empty:
        raise RuntimeError("No basins in shapefile match CSV IDs.")

    # Drop invalid/empty geometries and fix minor topology
    common = common[common.geometry.notna()]
    if not common.empty:
        common["geometry"] = common.geometry.buffer(0)
        common = common[~common.geometry.is_empty]

    if common.empty:
        # Fallback: use all basins to compute bounds so we don't get NaN extents
        bounds = basins.total_bounds
    else:
        # Order to match columns in ts_basin when possible
        ordered_cols = [c for c in ts_basin.columns if c in set(common[id_col])]
        if ordered_cols:
            common = common.set_index(id_col).loc[ordered_cols].reset_index()
        bounds = common.total_bounds

    # Robust bounds fallback
    if not np.isfinite(bounds).all():
        bounds = basins.total_bounds
    if not np.isfinite(bounds).all():
        raise ValueError("Map bounds are NaN/Inf; check shapefile geometries and CRS.")

    # Color scale from data
    vmin = 0.0
    finite_vals = np.asarray(ts_basin.values, dtype=float)
    finite = np.isfinite(finite_vals)
    vmax = float(np.nanpercentile(finite_vals[finite], 99)) if finite.any() else 1.0
    # Use non-deprecated colormap getter
    cmap = plt.colormaps.get('Blues')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Gauge color scale (unchanged)
    gauge_vmin = 0.0
    gauge_vmax = vmax  # align to basin scale by default

    # Create a GIF animation for each event date, showing all timesteps of that day
    fps = 10
    ts_index = pd.to_datetime(ts_basin.index)

    for date in extreme_dates:
        # frames for that date
        mask = (ts_index.date == date.date())
        frame_times = ts_index[mask]
        if len(frame_times) == 0:
            print(f"No basin timesteps found for {date.strftime('%Y-%m-%d')}")
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        divider = make_axes_locatable(ax)
        cax_basin = divider.append_axes("right", size="5%", pad=0.05)
        cax_gauge = divider.append_axes("bottom", size="5%", pad=0.4)

        # Static map extent
        ax.set_aspect('equal')
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])

        def update_hourly(i):
            ax.clear()
            cax_basin.cla()
            cax_gauge.cla()

            # Re-apply extent each frame after clear
            ax.set_aspect('equal')
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])

            tstamp = frame_times[i]
            # Join values for this timestep to basins
            vals = ts_basin.loc[tstamp]
            plot_gdf = common.copy()
            plot_gdf['mean_rain'] = plot_gdf[id_col].map(vals.to_dict())

            # Plot basins colored by mean_rain (gray for NaNs)
            plot_gdf.plot(
                ax=ax, column='mean_rain', cmap=cmap, vmin=vmin, vmax=vmax,
                edgecolor='black', linewidth=0.6,
                missing_kwds={"color": "#d9d9d9", "edgecolor": "black", "hatch": "///", "label": "No data"}
            )
            ax.set_title(f"Basin mean rain at {tstamp.strftime('%Y-%m-%d %H:%M')}")
            ax.set_xlabel('ITM X (meters)')
            ax.set_ylabel('ITM Y (meters)')

            # Gauges at this timestamp
            gauges_at_t = gauges_df[gauges_df['datetime'] == tstamp]
            sc = None
            if not gauges_at_t.empty:
                sc = ax.scatter(
                    gauges_at_t['ITM_X'], gauges_at_t['ITM_Y'],
                    c=gauges_at_t['Rain'], cmap="Reds",
                    edgecolor='black', s=80,
                    vmin=gauge_vmin, vmax=gauge_vmax, label='Gauges'
                )
                fig.colorbar(sc, cax=cax_gauge, orientation='horizontal', label="Gauge Rain (mm)")

            # Basin colorbar
            sm.set_array([])
            fig.colorbar(sm, cax=cax_basin, label="Mean basin rain (mm)")

            if sc is not None:
                ax.legend(loc='upper right')

            return []

        anim = FuncAnimation(fig, update_hourly, frames=len(frame_times), interval=1000/fps, blit=False)
        anim_path = os.path.join(output_dir, f'basins_mean_{date.strftime("%Y%m%d")}.gif')

        # Progress bar while saving frames
        if tqdm is not None:
            pbar = tqdm(total=len(frame_times), desc=f"Rendering {date.strftime('%Y-%m-%d')}", unit="frame")
            def _progress(i, n):
                pbar.update(1)
        else:
            pbar = None
            def _progress(i, n):  # no-op
                return

        try:
            anim.save(anim_path, writer='pillow', dpi=150, fps=fps, progress_callback=_progress)
            print(f"Saved animation to {anim_path}")
        except Exception as e:
            print(f"Could not save animation as gif: {e}")
        finally:
            if pbar is not None:
                pbar.close()
            plt.close(fig)

def main():
    # Paths (edit as needed)
    # Folder with per-basin CSVs you attached
    basin_csv_dir = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output\basin_average_rain'
    gauge_csv = r'C:\PhD\Data\IMS\Data_by_station\available_stations.csv'
    event_days_txt = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output\max_dates_2022_2023.txt'
    output_dir = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output'
    os.makedirs(output_dir, exist_ok=True)

    # Load basin-average timeseries
    ts_basin = read_basin_timeseries(basin_csv_dir)

    # Get event days and keep those within timeseries range
    event_days = load_event_days(event_days_txt)
    ts_times = pd.to_datetime(ts_basin.index)
    min_time = ts_times.min()
    max_time = ts_times.max()
    event_days_ts = pd.to_datetime(event_days)
    event_days_in_range = [d for d in event_days_ts if (d >= min_time) and (d <= max_time)]
    if not event_days_in_range:
        print("No event days within basin timeseries range. Exiting.")
        return

    # Merge only relevant rows from all gauge CSVs (same as your original script)
    gauge_data_dir = r'C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted'
    gauge_csv_files = [os.path.join(gauge_data_dir, f) for f in os.listdir(gauge_data_dir)
                       if f.endswith('.csv') and 'available_stations' not in f]
    cache_path = os.path.join(output_dir, f'relevant_gauges_{os.path.splitext(os.path.basename(event_days_txt))[0]}.csv')
    if os.path.exists(cache_path):
        print(f"Loading cached relevant gauges from {cache_path}")
        gauges_df = pd.read_csv(cache_path, parse_dates=['datetime'])
    else:
        relevant_gauges = []
        for f in gauge_csv_files:
            df = pd.read_csv(f, parse_dates=['datetime'], dayfirst=True)
            if df['datetime'].dt.tz is not None:
                df['datetime'] = df['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
            df['date'] = df['datetime'].dt.date
            event_days_dates = set([d.date() for d in event_days_in_range])
            df_relevant = df[df['date'].isin(event_days_dates)]
            if not df_relevant.empty:
                relevant_gauges.append(df_relevant)
        if relevant_gauges:
            gauges_df = pd.concat(relevant_gauges, ignore_index=True)
            stations_df = pd.read_csv(gauge_csv)
            gauges_df = pd.merge(gauges_df, stations_df[['Station_ID', 'ITM_X', 'ITM_Y']],
                                 on='Station_ID', how='left')
            gauges_df.to_csv(cache_path, index=False)
            print(f"Saved relevant gauges to {cache_path}")
        else:
            gauges_df = pd.DataFrame(columns=['datetime', 'Station_ID', 'ITM_X', 'ITM_Y', 'Rain'])

    # Animate
    animate_extreme_dates(ts_basin, gauges_df, event_days_in_range, output_dir)

if __name__ == '__main__':
    main()
