import os
import xarray as xr
import dask
import dask.array as da
from dask.distributed import Client
from tqdm import tqdm
from dask.diagnostics import ProgressBar

client = Client()  # This will use all available cores on the node

def unify_yearly_netcdf_files(output_dir, merged_filename="rain_grid_full_parallel.nc"):
    """
    Unify all yearly NetCDF files (rain_grid.nc in each output/year_YYYY folder) into a single NetCDF file.
    """
    # Find all year folders in the output directory
    year_folders = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                   if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("year_")]
    # Sort folders by year
    year_folders.sort()
    # Collect paths to rain_grid.nc in each year folder
    nc_files = [os.path.join(folder, "rain_grid.nc") for folder in year_folders]
    existing_files = []
    missing_files = []
    for f in tqdm(nc_files, desc="Checking yearly files"):
        if os.path.isfile(f):
            existing_files.append(f)
        else:
            missing_files.append(f)
    if missing_files:
        print(f"Warning: {len(missing_files)} yearly NetCDF files are missing and will be skipped:")
        for f in missing_files:
            print(f"  Missing: {f}")
    if not existing_files:
        print("No existing yearly files to merge!")
        return
    print(f"Merging {len(existing_files)} yearly NetCDFs into one...")
    merged_path = os.path.join(output_dir, merged_filename)
    ds_merged = xr.open_mfdataset(existing_files, combine='by_coords', chunks={})
    with ProgressBar():
        ds_merged.to_netcdf(merged_path)
    print(f"Merged file saved to {merged_path}")

if __name__ == "__main__":
    # Set the output directory where the yearly folders are located
    output_dir = '/sci/labs/efratmorin/omripo/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/output'
    unify_yearly_netcdf_files(output_dir)
