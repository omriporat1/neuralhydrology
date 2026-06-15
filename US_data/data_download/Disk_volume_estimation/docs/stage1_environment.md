# Flash-NH Stage 1 — Environment Design

Last updated: 2026-06-15

**Install status: INSTALLED AND SMOKE-TESTED (2026-06-15)**
Smoke log: `/data42/omrip/Flash-NH/tmp/env_smoke_20260615T120918Z/env_smoke.log`
All 7 smoke checks PASS. See [Smoke test results](#smoke-test-results) below.

---

## Overview

This document describes the dedicated conda environment for Flash-NH Stage 1
preprocessing work on h2o (`h2o.es.huji.ac.il`).

The environment is intentionally scoped to **CPU preprocessing and data work only**.
It does not include intentional GPU libraries or NeuralHydrology training dependencies.
Those belong in a separate Moriah training environment (`envs/environment-stage1-moriah.yml`,
to be created when Moriah access is confirmed).

> **Note:** The current installed env (7.0 G) is larger than intended because `neuralhydrology`
> pulled in a CUDA-enabled `torch==2.12.0+cu130` and associated NVIDIA packages via pip.
> `torch.cuda.is_available()` returns `False` on h2o (no GPU), so this is harmless for
> preprocessing work. A future spec revision should install `neuralhydrology --no-deps`
> or pin a CPU-only torch to keep the h2o env lean. See [Known risks](#known-risks) for details.

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

### What actually worked on h2o (2026-06-15)

Two failures occurred before the successful install:

1. **`mamba` failed** — `/opt/conda/bin/mamba` could not load `libmamba.so.4` (broken
   system mamba installation on h2o).
2. **`conda` default solver failed** — the default libmamba solver hit a shared
   package-cache permission error under `/opt/conda/pkgs/cache/`.

**Successful workaround:**

```bash
export CONDA_PKGS_DIRS=/home/omrip/.conda/pkgs
conda env create \
    --solver classic \
    --file /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/envs/environment-stage1-h2o.yml \
    --prefix /data42/omrip/Flash-NH/envs/flashnh-stage1
```

Two key flags: `--solver classic` bypasses the broken libmamba solver; setting
`CONDA_PKGS_DIRS` redirects the package cache to a location where the user has write
access.

### Original install guidance (for reference / future reinstalls)

**Preferred — mamba (faster solver):**

```bash
mamba env create \
    --file /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/envs/environment-stage1-h2o.yml \
    --prefix /data42/omrip/Flash-NH/envs/flashnh-stage1
```

**Fallback — conda classic (use this on h2o):**

```bash
export CONDA_PKGS_DIRS=/home/omrip/.conda/pkgs
conda env create \
    --solver classic \
    --file /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/envs/environment-stage1-h2o.yml \
    --prefix /data42/omrip/Flash-NH/envs/flashnh-stage1
```

If micromamba is available and mamba/conda both fail:

```bash
micromamba env create \
    --file /data42/omrip/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/envs/environment-stage1-h2o.yml \
    --prefix /data42/omrip/Flash-NH/envs/flashnh-stage1
```

---

## Activation

On h2o, conda is not automatically initialized in non-login shells. The `conda activate`
command will fail with "CommandNotFoundError" unless conda is sourced first:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1
```

This two-step sequence is required in fresh SSH sessions and `screen` windows on h2o.
Add `source /opt/conda/etc/profile.d/conda.sh` to your `.bashrc` or to the top of any
`screen`-launched script to avoid the error.

If using micromamba:

```bash
micromamba activate /data42/omrip/Flash-NH/envs/flashnh-stage1
```

---

## Smoke test results

**Status: ALL PASS (2026-06-15)**
Log: `/data42/omrip/Flash-NH/tmp/env_smoke_20260615T120918Z/env_smoke.log`
Python: `3.11.15` | Env size: `7.0 G` | `git status --short`: clean after smoke

| Check | Result |
|---|---|
| `core_imports` (numpy, pandas, xarray, netCDF4, pyarrow, requests, yaml) | PASS |
| `geospatial_imports` (geopandas, rasterio, pyproj, shapely) | PASS |
| `dask_imports` | PASS |
| `grib_imports_cfgrib_eccodes` | PASS |
| `netcdf_write_read_tmp` | PASS |
| `parquet_write_read_tmp` | PASS |
| `neuralhydrology_import_only` | PASS |
| `torch_import_check` (`torch==2.12.0+cu130`; `cuda_available=False`) | PASS |

> **Torch/CUDA note:** `neuralhydrology` pip-installed a CUDA-enabled torch (2.12.0+cu130)
> and NVIDIA packages, inflating the env to 7.0 G. `torch.cuda.is_available()` returns
> `False` on h2o — no GPU hardware is present, so CUDA is inert. This is functionally
> harmless for preprocessing but differs from the lean CPU-only intent. See
> [Known risks](#known-risks) for the planned mitigation.

---

## Smoke test

Run these after a fresh install to confirm the environment is functional.
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

### neuralhydrology + CUDA torch (known oversizing issue)

`neuralhydrology` is installed via pip for import-only use on h2o; no training will run.
On the 2026-06-15 install, pip resolved `neuralhydrology`'s dependencies and pulled in
`torch==2.12.0+cu130` plus associated NVIDIA CUDA packages (~3–4 G of the 7.0 G total).
`torch.cuda.is_available()` returns `False` because h2o has no GPU, so CUDA is inert.

This is acceptable for now, but a future revision of `environment-stage1-h2o.yml` should
avoid the oversizing. Planned mitigation options (choose one at revision time):

1. `pip install neuralhydrology --no-deps` and declare only the non-torch extras manually.
2. Pin `torch` to a CPU-only wheel before installing neuralhydrology:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install neuralhydrology
   ```
3. Remove `neuralhydrology` from the h2o spec entirely and install it only on Moriah,
   where GPU torch is needed anyway.

Until this is addressed, the 7.0 G env size and CUDA-enabled torch on h2o are documented
as a known caveat, not a blocker.

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
| 2026-06-15 | Install completed on h2o; all 7 smoke checks PASS; mamba/solver/cache workaround documented; CUDA torch caveat added |