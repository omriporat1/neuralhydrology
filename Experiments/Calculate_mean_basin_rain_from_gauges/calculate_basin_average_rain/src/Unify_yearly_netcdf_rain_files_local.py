import os
import time
import xarray as xr

def merge_yearly_netcdf_files(output_dir, merged_filename="rain_grid_full_merged_local.nc"):
    print("=== Starting local NetCDF merge ===", flush=True)
    start_time = time.time()
    # Find all year folders
    year_folders = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
                    if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("year_")]
    year_folders.sort()
    print(f"Found {len(year_folders)} year folders.", flush=True)
    # Collect NetCDF file paths
    nc_files = [os.path.join(folder, "rain_grid.nc") for folder in year_folders]
    existing_files = []
    for idx, f in enumerate(nc_files):
        print(f"Checking file {idx+1}/{len(nc_files)}: {f}", flush=True)
        if os.path.isfile(f):
            size_mb = os.path.getsize(f) / (1024*1024)
            print(f"  Exists ({size_mb:.1f} MB)", flush=True)
            existing_files.append(f)
        else:
            print("  MISSING", flush=True)
    if not existing_files:
        print("No NetCDF files found to merge!", flush=True)
        return
    print(f"Merging {len(existing_files)} files...", flush=True)
    merge_start = time.time()
    ds_merged = xr.open_mfdataset(existing_files, combine='by_coords', chunks={'time': 100})
    print("Writing merged NetCDF file...", flush=True)
    merged_path = os.path.join(output_dir, merged_filename)
    ds_merged.to_netcdf(merged_path, engine='netcdf4', compute=True)
    print(f"Merged file saved to {merged_path}", flush=True)
    print(f"Merge step took {time.time() - merge_start:.1f} seconds.", flush=True)
    print(f"Total script time: {time.time() - start_time:.1f} seconds.", flush=True)

if __name__ == "__main__":
    # Set your local output directory here:
    output_dir = r"C:\PhD\Data\IMS\Data_by_station\Data_by_station_formatted\output"
    merge_yearly_netcdf_files(output_dir)