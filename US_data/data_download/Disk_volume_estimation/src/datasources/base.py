from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import warnings

from src.derived_size import DerivedSpec


@dataclass(frozen=True)
class Region:
	name: str
	bbox: tuple[float, float, float, float]


CONUS_BBOX = Region("conus_bbox", (-126.0, 24.0, -66.0, 50.0))


@dataclass(frozen=True)
class ConusCropValidation:
	cropped_array: np.ndarray
	crop_bounds: Optional[tuple[float, float, float, float]]
	valid: bool
	reason: Optional[str] = None


@dataclass(frozen=True)
class RemoteObject:
	url: str
	key: str
	datetime: datetime
	variables: list[str]
	estimated_bytes: Optional[int] = None
	lead_time_h: Optional[int] = None





def validate_conus_crop(
	arr: np.ndarray,
	coords,
	bbox: tuple[float, float, float, float] = CONUS_BBOX.bbox,
	source_name: str = "unknown_source",
	crop_kind: str = "CONUS crop",
	logger: Optional[logging.Logger] = None,
	mark_invalid: Optional[Callable[[str], None]] = None,
) -> ConusCropValidation:
	lon_min, lat_min, lon_max, lat_max = bbox
	lat = coords.get("latitude") if "latitude" in coords else coords.get("lat")
	lon = coords.get("longitude") if "longitude" in coords else coords.get("lon")
	if lat is None or lon is None:
		reason = "missing latitude/longitude coordinates"
		empty = arr[:0, :0] if arr.ndim >= 2 else arr[:0]
		message = f"{source_name} {crop_kind}: {reason}; marking datasource INVALID"
		warnings.warn(message, RuntimeWarning, stacklevel=2)
		if logger is not None:
			logger.warning(message)
		if mark_invalid is not None:
			mark_invalid(reason)
		return ConusCropValidation(empty, None, False, reason)

	lat_arr = np.asarray(lat.values)
	lon_arr = np.asarray(lon.values)
	if lat_arr.ndim == 1 and lon_arr.ndim == 1:
		lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
	else:
		lat_grid = lat_arr
		lon_grid = lon_arr

	lon_grid = np.where(lon_grid > 180.0, lon_grid - 360.0, lon_grid)
	mask = (
		(lat_grid >= lat_min)
		& (lat_grid <= lat_max)
		& (lon_grid >= lon_min)
		& (lon_grid <= lon_max)
	)
	if mask.shape != arr.shape:
		reason = f"mask shape {mask.shape} does not match array shape {arr.shape}"
		empty = arr[:0, :0] if arr.ndim >= 2 else arr[:0]
		message = f"{source_name} {crop_kind}: {reason}; marking datasource INVALID"
		warnings.warn(message, RuntimeWarning, stacklevel=2)
		if logger is not None:
			logger.warning(message)
		if mark_invalid is not None:
			mark_invalid(reason)
		return ConusCropValidation(empty, None, False, reason)

	row_idx = np.where(mask.any(axis=1))[0]
	col_idx = np.where(mask.any(axis=0))[0]
	if row_idx.size == 0 or col_idx.size == 0:
		reason = "CONUS crop has no overlapping cells"
		empty = arr[:0, :0] if arr.ndim >= 2 else arr[:0]
		message = f"{source_name} {crop_kind}: {reason}; marking datasource INVALID"
		warnings.warn(message, RuntimeWarning, stacklevel=2)
		if logger is not None:
			logger.warning(message)
		if mark_invalid is not None:
			mark_invalid(reason)
		return ConusCropValidation(empty, None, False, reason)

	cropped = arr[row_idx.min():row_idx.max() + 1, col_idx.min():col_idx.max() + 1]
	selected_lat = lat_grid[mask]
	selected_lon = lon_grid[mask]
	crop_bounds = (
		float(np.min(selected_lon)),
		float(np.min(selected_lat)),
		float(np.max(selected_lon)),
		float(np.max(selected_lat)),
	)
	message = (
		f"{source_name} {crop_kind}: shape={tuple(int(v) for v in cropped.shape)} "
		f"bounds=lon[{crop_bounds[0]:.3f}, {crop_bounds[2]:.3f}] lat[{crop_bounds[1]:.3f}, {crop_bounds[3]:.3f}]"
	)
	if logger is not None:
		logger.info(message)
	valid = cropped.size > 0 and crop_bounds[0] <= lon_max and crop_bounds[2] >= lon_min and crop_bounds[1] <= lat_max and crop_bounds[3] >= lat_min
	if not valid:
		reason = "cropped array does not overlap CONUS bounds"
		warning_message = f"{message}; marking datasource INVALID"
		warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
		if logger is not None:
			logger.warning(warning_message)
		if mark_invalid is not None:
			mark_invalid(reason)
		return ConusCropValidation(cropped[:0, :0], crop_bounds, False, reason)

	return ConusCropValidation(cropped, crop_bounds, True, None)

class DataSource(ABC):
	name: str
	temporal_resolution: str  # "hourly", "daily", or "forecast"

	@abstractmethod
	def list_sample_objects(
		self,
		start: datetime,
		end: datetime,
		region: Region,
		variables: list[str],
		lead_times: Optional[Iterable[int]] = None,
	) -> list[RemoteObject]:
		raise NotImplementedError

	@abstractmethod
	def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
		raise NotImplementedError

	@abstractmethod
	def measure_bytes(self, files: Iterable[Path]) -> int:
		raise NotImplementedError

	@abstractmethod
	def estimate_raw_total(
		self,
		sample_bytes: int,
		sample_start: datetime,
		sample_end: datetime,
		full_start: datetime,
		full_end: datetime,
	) -> int:
		raise NotImplementedError

	@abstractmethod
	def estimate_derived_total(self, spec: DerivedSpec) -> int:
		raise NotImplementedError

	@abstractmethod
	def estimate_peak_local(
		self,
		raw_total_bytes: int,
		scratch_multiplier: float,
		concurrency: int,
	) -> int:
		raise NotImplementedError

	@abstractmethod
	def assumptions(self) -> dict:
		raise NotImplementedError
