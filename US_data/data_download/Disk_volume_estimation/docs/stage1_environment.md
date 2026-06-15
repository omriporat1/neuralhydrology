# Flash-NH Stage 1 — Environment Design

Last updated: 2026-06-15

---

## Overview

This document describes the dedicated conda environment for Flash-NH Stage 1
preprocessing work on h2o (`h2o.es.huji.ac.il`).

The environment is intentionally scoped to **CPU preprocessing and data work only**.
It does not include GPU libraries, PyTorch, or NeuralHydrology training dependencies.
Those belong in a separate Moriah training environment (`envs/environment-stage1-moriah.yml`,
to be created when Moriah access is confirmed).

---

## Why a dedicated environment

The shared `iacpy3_2025` environment on h2o is used for smoke tests and early acquisition.
It is not version-pinned to this project, is not version-controlled, and may change or
disappear without notice. All production Flash-NH preprocessing work should use the
project-controlled environment described here.

---

## Why GPU / PyTorch are excluded

h2o has no usable GPU (`nvidia-smi` not found; PI-confirmed). NeuralHydrology model
training is designated for the Moriah GPU cluster, not h2o. Including GPU or PyTorch
dependencies in the h2o environment would be dead weight at best and a source of
installation fragility at worst. The h2o env is a lean preprocessing env; the Moriah
env will carry PyTorch + NeuralHydrology + CUDA.

---

## Spec file

Repository path: `envs/environment-stage1-h2o.yml`

Primary channel: `conda-forge`

### Key dependency groups

| Group | Packages |
|---|---|
| Python | `python=3.11` |
| Core numeric | `numpy`, `pandas`, `scipy` |
| xarray / NetCDF | `xarray`, `netcdf4`, `h5netcdf`, `h5py` |
| Parquet / columnar | `pyarrow` |
| Dask parallel | `dask`, `distributed`, `zarr` |
| HTTP / config / progress | `requests`, `pyyaml`, `tqdm` |
| Plotting / QC | `matplotlib` |
| Geospatial | `shapely`, `geopandas`, `pyproj`, `rasterio`, `pyogrio`, `rioxarray`, `fiona` |
| GRIB / met | `cfgrib`, `eccodes` |
| Jupyter | `ipykernel`, `ipython` |
| Testing | `pytest` |
| pip extra | `neuralhydrology` (import-only on h2o; no training) |

---

## Install location

```
/data42/omrip/Flash-NH/envs/flashnh-stage1
```

---

## Install command

**Preferred — mamba (faster solver):**

```bash
mamba env create \
    --file /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/envs/environment-stage1-h2o.yml \
    --prefix /data42/omrip/Flash-NH/envs/flashnh-stage1
```

**Fallback — conda (if mamba is unavailable):**

```bash
conda env create \
    --file /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/envs/environment-stage1-h2o.yml \
    --prefix /data42/omrip/Flash-NH/envs/flashnh-stage1
```

If neither `mamba` nor `conda` is available in PATH, check for `micromamba`:

```bash
micromamba env create \
    --file /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/envs/environment-stage1-h2o.yml \
    --prefix /data42/omrip/Flash-NH/envs/flashnh-stage1
```

---

## Activation

```bash
conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1
```

Or, if using micromamba:

```bash
micromamba activate /data42/omrip/Flash-NH/envs/flashnh-stage1
```

---

## Smoke test

Run these after installing to confirm the environment is functional.
All commands should complete without error.

### Core imports

```python
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import pyarrow
import requests
import yaml
print("Core imports OK")
```

### Geospatial imports

```python
import geopandas as gpd
import rasterio
import pyproj
import shapely
print("Geospatial imports OK")
```

### GRIB / meteorological imports (may fail — see Known Risks)

```python
try:
    import cfgrib
    import eccodes
    print("cfgrib / eccodes OK")
except ImportError as e:
    print(f"GRIB imports failed: {e}")
    print("See Known Risks — GRIB fallback applies")
```

### NetCDF write/read under tmp/

```python
import numpy as np
import xarray as xr
import os

tmp_path = "/data42/omrip/Flash-NH/tmp/env_smoke_test.nc"
ds = xr.Dataset({"test": ("time", np.arange(5.0))},
                coords={"time": np.arange(5)})
ds.to_netcdf(tmp_path)
ds2 = xr.open_dataset(tmp_path)
assert len(ds2["test"]) == 5
os.remove(tmp_path)
print("NetCDF write/read OK")
```

### Parquet write/read under tmp/

```python
import pandas as pd
import os

tmp_path = "/data42/omrip/Flash-NH/tmp/env_smoke_test.parquet"
df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
df.to_parquet(tmp_path)
df2 = pd.read_parquet(tmp_path)
assert len(df2) == 3
os.remove(tmp_path)
print("Parquet write/read OK")
```

---

## Known risks

### GRIB / eccodes / cfgrib (highest fragility)

`cfgrib` and `eccodes` are the most likely packages to cause installation problems.
`eccodes` is a C library with system-level dependencies; conda-forge packages it, but
library linking can fail on older glibc versions or unusual system configurations.

**Fallback strategy if GRIB fails:**

1. Remove `cfgrib` and `eccodes` from the spec and rebuild the environment without them.
2. Document that GRIB handling is disabled.
3. Use `wgrib2` (system binary, if available on h2o) or transfer GRIB files to a machine
   where GRIB tools are confirmed functional.
4. Do not block streamflow acquisition, NetCDF assembly, or geospatial preprocessing
   on GRIB dependency resolution — those work stacks are fully independent.

### pyogrio vs fiona

Both `pyogrio` and `fiona` are included; `pyogrio` is the faster modern reader and
`geopandas ≥ 0.13` uses it by default. If there is a conflict between them, remove
`fiona` — `pyogrio` covers the required use cases.

### neuralhydrology (pip install on h2o)

`neuralhydrology` is installed via pip for import-only use (reading configs, testing
package structure). No training will run on h2o. If the pip install fails due to a
missing torch wheel, install neuralhydrology without its optional extras:

```bash
pip install neuralhydrology --no-deps
```

and install the required non-GPU deps manually if needed.

---

## Moriah training environment

The Moriah training environment (`envs/environment-stage1-moriah.yml`) is a separate
design task, to be created after:

1. Moriah cluster access is confirmed.
2. SLURM partition names and GPU queue policy are documented.
3. CUDA driver version on Moriah nodes is confirmed.

The Moriah env will carry: `pytorch`, `neuralhydrology`, CUDA libraries, and any
training-specific tooling. It will not carry the geospatial preprocessing stack
from this env.

---

## Update history

| Date | Change |
|---|---|
| 2026-06-15 | Initial spec created; Python 3.11, conda-forge primary channel |