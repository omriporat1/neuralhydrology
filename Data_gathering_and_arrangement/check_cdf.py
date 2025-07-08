'''from netCDF4 import Dataset
import numpy as np

# Open the NetCDF file in write mode
file_path = "../../../../data/Caravan/timeseries/netcdf/il/il_8146.nc"
dataset = Dataset(file_path, mode="r+")  # 'r+' allows reading and writing

# Print the structure of the file
print("Variables in the file:")
print(dataset.variables.keys())

# Print details of a specific variable
var_name = ["date", "Flow_m3_sec", "Water_level_m", "Rain_gauge_1", "Rain_gauge_2", "Rain_gauge_3"]
for var in var_name:
    if var in dataset.variables:
        print(f"\nDetails of variable '{var}':")
        print(dataset.variables[var])
    else:
        print(f"Variable '{var}' not found in the file.")

'''

import xarray as xr
import pandas as pd

# Step 1: Open the .nc file
file_path = f'../../../../data/Caravan/timeseries/netcdf/il/il_8146.nc'  # Replace with your file path
dataset = xr.open_dataset(file_path)

# Step 2: View the dataset's structure
print(dataset)

# Step 3: Access variables or convert to DataFrame
# Example: Convert a specific variable to a DataFrame
var_name = ["date", "Flow_m3_sec", "Water_level_m", "Rain_gauge_1", "Rain_gauge_2", "Rain_gauge_3"]
for var in var_name:
    if var in dataset:
        variable_data = dataset[var]
        df = variable_data.to_dataframe().reset_index()  # Converts to a DataFrame
        print(df.head())
    else:
        print("Variable not found. Available variables:", list(dataset.variables))
