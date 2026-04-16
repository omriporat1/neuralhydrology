from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import tempfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


@dataclass(frozen=True)
class DerivedSpec:
	n_basins: int
	start: datetime
	end: datetime
	temporal_resolution: str  # "hourly" or "daily"
	n_vars: int
	dtype: str
	output_format: str


_PARQUET_RATIO_CACHE: Dict[Tuple[int, str], float] = {}
LOGGER = logging.getLogger(__name__)


def _timesteps_inclusive(start: datetime, end: datetime, temporal_resolution: str) -> int:
	if temporal_resolution == "hourly":
		delta_hours = int((end - start).total_seconds() // 3600)
		return delta_hours + 1
	if temporal_resolution == "daily":
		delta_days = (end.date() - start.date()).days
		return delta_days + 1
	raise ValueError(f"Unsupported temporal resolution: {temporal_resolution}")


def _dtype_bytes(dtype: str) -> int:
	return np.dtype(dtype).itemsize


def calibrate_parquet_ratio(n_vars: int, dtype: str) -> float:
	cache_key = (n_vars, dtype)
	if cache_key in _PARQUET_RATIO_CACHE:
		return _PARQUET_RATIO_CACHE[cache_key]

	hours = 7 * 24
	n_rows = hours
	data = {}
	for i in range(n_vars):
		col = np.random.default_rng(42 + i).normal(size=n_rows).astype(dtype)
		data[f"var_{i+1}"] = col

	df = pd.DataFrame(data)
	table = pa.Table.from_pandas(df, preserve_index=False)

	with tempfile.TemporaryDirectory() as tmp_dir:
		tmp_path = Path(tmp_dir) / "_parquet_ratio_tmp.parquet"
		pq.write_table(table, tmp_path, compression="snappy")
		parquet_bytes = tmp_path.stat().st_size

	n_columns = len(df.columns)
	bytes_per_value = _dtype_bytes(dtype)
	raw_bytes = n_rows * n_columns * bytes_per_value
	ratio = parquet_bytes / raw_bytes if raw_bytes else 1.0
	if ratio > 1.0:
		LOGGER.warning(
			"Parquet ratio > 1 detected (parquet_bytes=%s, raw_bytes=%s, ratio=%.4f). Clamping to 1.0.",
			parquet_bytes,
			raw_bytes,
			ratio,
		)
		ratio = 1.0
	_PARQUET_RATIO_CACHE[cache_key] = ratio
	return ratio


def compute_derived_bytes(spec: DerivedSpec) -> int:
	timesteps = _timesteps_inclusive(spec.start, spec.end, spec.temporal_resolution)
	raw_bytes = spec.n_basins * timesteps * spec.n_vars * _dtype_bytes(spec.dtype)

	if spec.output_format == "parquet":
		ratio = calibrate_parquet_ratio(spec.n_vars, spec.dtype)
		return int(raw_bytes * ratio)

	if spec.output_format in {"zarr", "netcdf"}:
		raise NotImplementedError(
			"zarr/netcdf output not implemented. Install zarr + xarray/netCDF4 "
			"and implement format-specific overhead handling."
		)

	raise ValueError(f"Unsupported output format: {spec.output_format}")
