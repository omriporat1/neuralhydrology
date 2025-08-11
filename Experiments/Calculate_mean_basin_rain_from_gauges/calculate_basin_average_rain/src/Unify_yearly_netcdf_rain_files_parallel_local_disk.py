import os
import xarray as xr
import dask
import dask.array as da
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from dask.distributed import Client

def unify_yearly_netcdf_files(output_dir, merged_filename="rain_grid_full_parallel_local.nc"):
    """
    Unify all yearly NetCDF files (rain_grid.nc in each output/year_YYYY folder) into a single NetCDF file.
    """
    # Find all year folders in the output directory
    print("[DEBUG] Listing year folders...")
    year_folders = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("year_")]
    print(f"[DEBUG] Found {len(year_folders)} year folders.")
    year_folders.sort()
    print("[DEBUG] Collecting NetCDF file paths...")
    nc_files = [os.path.join(folder, "rain_grid.nc") for folder in year_folders]
    print(f"[DEBUG] Checking which NetCDF files exist...")
    existing_files = []
    missing_files = []
    for idx, f in enumerate(nc_files):
        print(f"[DEBUG] Checking file {idx+1}/{len(nc_files)}: {f}")
        if os.path.isfile(f):
            print(f"[DEBUG]   Exists: {f}")
            existing_files.append(f)
        else:
            print(f"[DEBUG]   MISSING: {f}")
            missing_files.append(f)
    print(f"[DEBUG] {len(existing_files)} files found, {len(missing_files)} missing.")
    if missing_files:
        print(f"Warning: {len(missing_files)} yearly NetCDF files are missing and will be skipped:")
        for f in missing_files:
            print(f"  Missing: {f}")
    if not existing_files:
        print("No existing yearly files to merge!")
        return
    print(f"Merging {len(existing_files)} yearly NetCDFs into one...")
    # Write to local scratch disk for fast I/O, then copy to output_dir
    scratch_dir = os.environ.get('SCRATCH', f"/tmp/{os.environ.get('USER', 'user')}")
    print(f"Using scratch directory: {scratch_dir}")
    os.makedirs(scratch_dir, exist_ok=True)
    local_merged_path = os.path.join(scratch_dir, merged_filename)
    print(f"[DEBUG] Opening {len(existing_files)} NetCDF files with xarray.open_mfdataset...")
    ds_merged = xr.open_mfdataset(existing_files, combine='by_coords', chunks={'time': 100})
    print(f"[DEBUG] Writing merged NetCDF to local scratch: {local_merged_path}")
    with ProgressBar():
        ds_merged.to_netcdf(local_merged_path, engine='netcdf4', compute=True)
    print(f"Merged file saved to local scratch: {local_merged_path}")
    # Copy to final output directory
    final_merged_path = os.path.join(output_dir, merged_filename)
    print(f"[DEBUG] Copying merged file to final output directory: {final_merged_path}")
    import shutil
    shutil.copy2(local_merged_path, final_merged_path)
    print(f"Copied merged file to {final_merged_path}")

if __name__ == "__main__":
    n_workers = int(os.environ.get("NHY_DASK_WORKERS", 4))
    client = Client(n_workers=n_workers, threads_per_worker=1, dashboard_address=":8787")
    print(f"Dask client started with {n_workers} workers (1 thread per worker)")

    # Set the output directory where the yearly folders are located
    output_dir = '/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/output'
    unify_yearly_netcdf_files(output_dir)
