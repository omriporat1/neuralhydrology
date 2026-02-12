# src/data_camelsh.py
from pathlib import Path
import xarray as xr
import pandas as pd

def load_camelsh_hourly(gauge_id: str, year: int, root: str = "data/raw/camelsh") -> pd.Series:
    fn = Path(root) / f"{gauge_id}_hourly.nc"
    if not fn.exists():
        raise FileNotFoundError(f"CAMELSH file not found: {fn}")

    ds = xr.open_dataset(fn)
    if "streamflow" not in ds:
        raise KeyError(f"'streamflow' not found in {fn}. Variables: {list(ds.data_vars)}")

    q = ds["streamflow"].to_series()
    # CAMELSH timestamps are intended as UTC; make that explicit
    if q.index.tz is None:
        q.index = q.index.tz_localize("UTC")

    q = q.loc[f"{year}-01-01 00:00:00+00:00":f"{year}-12-31 23:00:00+00:00"]
    q.name = "q_camelsh_m3s"
    return q
