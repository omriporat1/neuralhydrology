import os
import sys
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
from rasterstats import zonal_stats
from datetime import datetime
import concurrent.futures
import rasterio

def setup_logging(log_file=None):
    if log_file:
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def format_date(dt):
    # dt: np.datetime64 or pd.Timestamp
    if isinstance(dt, np.datetime64):
        dt = pd.to_datetime(str(dt))
    return dt.strftime('%d/%m/%Y %H:%M')

def extract_basin_stats_for_timestep(args):
    # For parallelization: (basin_idx, basin_geom, rain_grid, gauge_grid, x_coords, y_coords, time, basin_id, log_dir)
    (basin_idx, basin_geom, rain_grid, gauge_grid, x_coords, y_coords, time, basin_id, log_dir) = args
    try:
        # Prepare a temporary raster for this timestep
        affine = rasterio.transform.from_bounds(
            x_coords[0], y_coords[0], x_coords[-1], y_coords[-1], rain_grid.shape[1], rain_grid.shape[0])
        zs_rain = zonal_stats(
            [basin_geom], rain_grid, affine=affine, stats=["mean", "max"], nodata=np.nan, all_touched=True)
        zs_gauges = zonal_stats(
            [basin_geom], gauge_grid, affine=affine, stats=["min", "max"], nodata=np.nan, all_touched=True)
        result = {
            'date': format_date(time),
            'mean_rain': zs_rain[0]['mean'],
            'max_rain': zs_rain[0]['max'],
            'min_gauges': zs_gauges[0]['min'],
            'max_gauges': zs_gauges[0]['max']
        }
        return (basin_id, result)
    except Exception as e:
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, f'basin_{basin_id}_error.log'), 'a') as f:
                f.write(f"Error at {format_date(time)}: {e}\n")
        return (basin_id, None)

def _find_yearly_netcdfs(netcdf_path_or_dir):
    # Returns a sorted list of nc file paths. If a directory is provided,
    # it looks for <dir>/<year>/rain_grid.nc in each subdirectory.
    if os.path.isdir(netcdf_path_or_dir):
        nc_files = []
        for name in sorted(os.listdir(netcdf_path_or_dir)):
            subdir = os.path.join(netcdf_path_or_dir, name)
            if os.path.isdir(subdir):
                candidate = os.path.join(subdir, 'rain_grid.nc')
                if os.path.isfile(candidate):
                    nc_files.append(candidate)
        if not nc_files:
            raise FileNotFoundError(f"No rain_grid.nc files found under {netcdf_path_or_dir}")
        return nc_files
    if os.path.isfile(netcdf_path_or_dir):
        return [netcdf_path_or_dir]
    raise FileNotFoundError(f"{netcdf_path_or_dir} is neither a file nor a directory")

def process_basin_over_files(basin_idx, basin_row, nc_files, log_dir, out_dir, start_time=None, end_time=None, parallel=False):
    basin_id = basin_row['id'] if 'id' in basin_row else basin_idx
    basin_geom = basin_row['geometry']

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"basin_{basin_id}.csv")
    first_write = not os.path.exists(csv_path)

    # Convert start/end to np.datetime64 if provided
    start = np.datetime64(start_time) if start_time else None
    end = np.datetime64(end_time) if end_time else None

    for nc_path in nc_files:
        try:
            # Open one year at a time; iterate per timestep to keep RAM low
            with xr.open_dataset(nc_path) as ds:
                # Apply time filter per file if needed
                if start or end:
                    ds = ds.sel(time=slice(start or ds['time'].values[0], end or ds['time'].values[-1]))
                if ds['time'].size == 0:
                    continue

                x_coords = ds['x'].values
                y_coords = ds['y'].values

                times = ds['time'].values
                rain_da = ds['rain']              # keep as DataArray
                gauges_da = ds['gauge_count']     # keep as DataArray

                args_list = [
                    (
                        basin_idx,
                        basin_geom,
                        rain_da.isel(time=i).values,          # 2D numpy per timestep
                        gauges_da.isel(time=i).values,        # 2D numpy per timestep
                        x_coords,
                        y_coords,
                        times[i],
                        basin_id,
                        log_dir
                    )
                    for i in range(len(times))
                ]

                results = []
                if parallel:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        for _, res in executor.map(extract_basin_stats_for_timestep, args_list):
                            if res:
                                results.append(res)
                else:
                    for a in args_list:
                        _, res = extract_basin_stats_for_timestep(a)
                        if res:
                            results.append(res)

                if results:
                    df = pd.DataFrame(results)
                    df.to_csv(
                        csv_path,
                        index=False,
                        encoding='utf-8',
                        mode='w' if first_write else 'a',
                        header=first_write
                    )
                    first_write = False
                    logging.info(f"Saved basin {basin_id} results from {os.path.basename(nc_path)} to {csv_path}")
        except Exception as e:
            logging.exception(f"Basin {basin_id}: failed processing {nc_path}: {e}")

def process_basin_timeseries(basin_idx, basin_row, ds, x_coords, y_coords, log_dir, out_dir, parallel=False):
    basin_id = basin_row['id'] if 'id' in basin_row else basin_idx
    basin_geom = basin_row['geometry']
    times = ds['time'].values
    rain = ds['rain'].values
    gauge_count = ds['gauge_count'].values
    results = []
    args_list = [
        (basin_idx, basin_geom, rain[i], gauge_count[i], x_coords, y_coords, times[i], basin_id, log_dir)
        for i in range(len(times))
    ]
    if parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for basin_id, res in executor.map(extract_basin_stats_for_timestep, args_list):
                if res:
                    results.append(res)
    else:
        for args in args_list:
            basin_id, res = extract_basin_stats_for_timestep(args)
            if res:
                results.append(res)
    # Save to CSV
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f"basin_{basin_id}.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logging.info(f"Saved basin {basin_id} results to {csv_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract mean/max rain and gauge stats for each basin from NetCDF rain surfaces.")
    parser.add_argument('--basins', type=str, required=True, help='Path to basins shapefile')
    parser.add_argument('--netcdf', type=str, required=True, help='Path to a NetCDF file OR a directory with yearly subfolders containing rain_grid.nc')
    parser.add_argument('--out_dir', type=str, default='basin_stats_output', help='Output directory for CSVs')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for log files')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing (per basin)')
    parser.add_argument('--log_file', type=str, default=None, help='Log file path')
    parser.add_argument('--start_time', type=str, default=None, help='Start time (inclusive) in YYYY-MM-DD[ HH:MM] format')
    parser.add_argument('--end_time', type=str, default=None, help='End time (inclusive) in YYYY-MM-DD[ HH:MM] format')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (defaults to SLURM_CPUS_PER_TASK or os.cpu_count())')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    setup_logging(args.log_file)

    logging.info(f"Loading basins from {args.basins}")
    basins = gpd.read_file(args.basins)
    if basins.crs is None or basins.crs.to_epsg() != 2039:
        basins = basins.to_crs(epsg=2039)

    # Discover yearly NetCDF files
    nc_files = _find_yearly_netcdfs(args.netcdf)
    logging.info(f"Found {len(nc_files)} NetCDF file(s) to process.")

    logging.info(f"Processing {len(basins)} basins...")
    basin_rows = list(basins.iterrows())
    if args.parallel:
        # Respect SLURM cpus-per-task if present
        import multiprocessing as mp
        n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count() or 1))
        logging.info(f"Starting per-basin multiprocessing with {n_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futs = [
                executor.submit(
                    process_basin_over_files,
                    idx,
                    row,
                    nc_files,
                    args.log_dir,
                    args.out_dir,
                    args.start_time,
                    args.end_time,
                    False
                )
                for idx, row in basin_rows
            ]
            total = len(futs)
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                try:
                    fut.result()
                except Exception as e:
                    logging.exception(f"A basin task failed: {e}")
                finally:
                    logging.info(f"Progress: {i}/{total} basins finished")
    else:
        total = len(basin_rows)
        for i, (idx, row) in enumerate(basin_rows, 1):
            process_basin_over_files(
                idx, row, nc_files, args.log_dir, args.out_dir, args.start_time, args.end_time, False
            )
            logging.info(f"Progress: {i}/{total} basins finished")

    logging.info("All basins processed.")

if __name__ == '__main__':
    main()
