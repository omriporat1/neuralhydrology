# notebooks/inspect_camelsh.py

import xarray as xr
import os
from pathlib import Path

# EDIT: put your actual filename here (one CAMELSH .nc file you downloaded)
FN = Path("data/raw/camelsh/0136230002_hourly.nc")

if not FN.exists():
    raise SystemExit(f"File not found: {FN}\nDownload a CAMELSH .nc and place it there, or adjust FN.")

print("Opening:", FN)
ds = xr.open_dataset(FN)

print("\n=== Dataset summary ===")
print(ds)

print("\n=== Data variables ===")
print(list(ds.data_vars))

if "Streamflow" in ds:
    print("\n=== Streamflow attrs ===")
    for k, v in ds["Streamflow"].attrs.items():
        print(f"{k}: {v}")
else:
    first_var = list(ds.data_vars)[0]
    print(f"\nNOTE: 'Streamflow' not found. Showing attrs for first var: {first_var}")
    for k, v in ds[first_var].attrs.items():
        print(f"{k}: {v}")

print("\n=== time coord (first 20) ===")
print(ds.coords["time"][:20].values)

print("\n=== time coord attrs ===")
for k, v in ds.coords["time"].attrs.items():
    print(f"{k}: {v}")
