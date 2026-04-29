from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import gzip
import logging
from pathlib import Path
import tempfile
from typing import Optional
import time
import warnings

import numpy as np

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region
from src.datasources.era5_landt import Era5LandTDataSource
from src.datasources.gdas import GdasAwsAntecedentDataSource
from src.datasources.gfs import GfsAwsConusDataSource
from src.datasources.imerg import ImergLateDailyDataSource
from src.datasources.ifs import IfsMarsDataSource
from src.datasources.mrms import MrmsAwsQpe1hPass1, MrmsDataSource
from src.datasources.rtma import RtmaAwsConusDataSource
from src.derived_size import calibrate_parquet_ratio


LOGGER = logging.getLogger(__name__)


def _suppress_cfgrib_merge_futurewarning() -> None:
	warnings.filterwarnings(
		"ignore",
		message=(
			r"In a future version of xarray the default value for compat will change "
			r"from compat='no_conflicts' to compat='override'.*"
		),
		category=FutureWarning,
		module=r"cfgrib\.xarray_store",
	)

@dataclass(frozen=True)
class Config:
	sample_start: datetime
	sample_end: datetime
	full_start: datetime
	full_end: datetime
	region: Region
	out_dir: Path
	report_dir: Path
	variables: list[str]
	include_longwave_stage2: bool
	derived_dtype: str
	derived_format: str
	scratch_multiplier: float
	n_basins: int
	concurrency: int
	dry_run: bool
	mrms_backend: str
	mrms_debug_listing: bool
	range_mode: str
	make_preview: bool
	gfs_max_lead: int
	ifs_max_lead: int


@dataclass
class EstimateResult:
	source: str
	sample_range: str
	sample_files: int
	raw_sample_bytes: Optional[int]
	raw_sample_full_file_bytes: Optional[int]
	raw_sample_selected_bytes: Optional[int]
	raw_sample_selected_conus_bytes: Optional[int]
	raw_hot_bytes: Optional[int]
	raw_hot_full_file_bytes: Optional[int]
	raw_hot_selected_bytes: Optional[int]
	raw_hot_selected_conus_bytes: Optional[int]
	derived_hot_bytes: Optional[int]
	raw_cold_bytes: Optional[int]
	peak_local_bytes: Optional[int]
	source_available_start: Optional[str]
	source_available_end: Optional[str]
	mrms_available_start: Optional[str]
	mrms_available_end: Optional[str]
	effective_start: Optional[str]
	effective_end: Optional[str]
	effective_prediction_start: Optional[str]
	effective_prediction_end: Optional[str]
	era5_land_required_data_start: Optional[str]
	era5_land_required_data_end: Optional[str]
	gdas_required_data_start: Optional[str]
	gdas_required_data_end: Optional[str]
	imerg_required_data_start: Optional[str]
	imerg_required_data_end: Optional[str]
	range_mode: str
	covers_effective_range: bool
	notes: str
	assumptions: dict


@dataclass
class BenchmarkResult:
	concurrency: int
	total_wall_time_s: float
	files_downloaded: int
	total_bytes: int
	files_per_s: float
	mb_per_s: float
	retry_count: int
	warning_count: int
	avg_file_time_s: float
	p90_file_time_s: float


def get_enabled_sources(config: Config) -> list[DataSource]:
	if config.mrms_backend == "planetary":
		return [
			MrmsDataSource(),
			RtmaAwsConusDataSource(download_concurrency=config.concurrency),
			GfsAwsConusDataSource(max_lead_h=config.gfs_max_lead, download_concurrency=config.concurrency),
			IfsMarsDataSource(max_lead_h=config.ifs_max_lead, download_concurrency=config.concurrency),
			Era5LandTDataSource(download_concurrency=config.concurrency),
			GdasAwsAntecedentDataSource(download_concurrency=config.concurrency),
			ImergLateDailyDataSource(download_concurrency=config.concurrency),
		]
	return [
		MrmsAwsQpe1hPass1(
			debug_listing=config.mrms_debug_listing,
			download_concurrency=config.concurrency,
		)
		,
		RtmaAwsConusDataSource(download_concurrency=config.concurrency),
		GfsAwsConusDataSource(max_lead_h=config.gfs_max_lead, download_concurrency=config.concurrency),
		IfsMarsDataSource(max_lead_h=config.ifs_max_lead, download_concurrency=config.concurrency),
		Era5LandTDataSource(download_concurrency=config.concurrency),
		GdasAwsAntecedentDataSource(download_concurrency=config.concurrency),
		ImergLateDailyDataSource(download_concurrency=config.concurrency),
	]


def _read_available_extent(source: DataSource) -> tuple[Optional[datetime], Optional[datetime]]:
	start = getattr(source, "_available_start", None)
	end = getattr(source, "_available_end", None)
	return start, end


def _intersect_ranges(
	start_a: datetime,
	end_a: datetime,
	start_b: Optional[datetime],
	end_b: Optional[datetime],
) -> tuple[Optional[datetime], Optional[datetime]]:
	if start_b is None or end_b is None:
		return start_a, end_a
	start = max(start_a, start_b)
	end = min(end_a, end_b)
	if start > end:
		return None, None
	return start, end


def _to_iso(value: Optional[datetime]) -> Optional[str]:
	return value.isoformat() if value is not None else None


def _compute_era5_landt_windows(
	mrms_effective_start: Optional[datetime],
	mrms_effective_end: Optional[datetime],
	source_available_start: Optional[datetime],
	source_available_end: Optional[datetime],
	latency_days: int,
	lookback_days: int,
) -> dict[str, Optional[datetime]]:
	if (
		mrms_effective_start is None
		or mrms_effective_end is None
		or source_available_start is None
		or source_available_end is None
	):
		return {
			"effective_prediction_start": None,
			"effective_prediction_end": None,
			"required_data_start": None,
			"required_data_end": None,
		}

	lag = timedelta(days=latency_days)
	lookback = timedelta(days=lookback_days)

	pred_start = max(mrms_effective_start, source_available_start + lookback)
	pred_end = min(mrms_effective_end, source_available_end + lag)
	if pred_start > pred_end:
		return {
			"effective_prediction_start": None,
			"effective_prediction_end": None,
			"required_data_start": None,
			"required_data_end": None,
		}

	required_start = pred_start - lookback
	required_end = pred_end - lag
	if required_start > required_end:
		return {
			"effective_prediction_start": None,
			"effective_prediction_end": None,
			"required_data_start": None,
			"required_data_end": None,
		}

	return {
		"effective_prediction_start": pred_start,
		"effective_prediction_end": pred_end,
		"required_data_start": required_start,
		"required_data_end": required_end,
	}


def _compute_antecedent_windows(
	mrms_effective_start: Optional[datetime],
	mrms_effective_end: Optional[datetime],
	source_available_start: Optional[datetime],
	source_available_end: Optional[datetime],
	latency_days: int,
	lookback_days: int,
) -> dict[str, Optional[datetime]]:
	return _compute_era5_landt_windows(
		mrms_effective_start=mrms_effective_start,
		mrms_effective_end=mrms_effective_end,
		source_available_start=source_available_start,
		source_available_end=source_available_end,
		latency_days=latency_days,
		lookback_days=lookback_days,
	)


def _extract_precip_grid(file_path: Path) -> tuple[np.ndarray, str, str]:
	work_path = file_path
	tmp_file_path: Optional[Path] = None
	if file_path.suffix == ".gz":
		tmp_dir = tempfile.TemporaryDirectory()
		tmp_file_path = Path(tmp_dir.name) / file_path.stem
		with gzip.open(file_path, "rb") as src, tmp_file_path.open("wb") as dst:
			dst.write(src.read())
		work_path = tmp_file_path

	try:
		try:
			import xarray as xr
		except Exception as exc:  # noqa: BLE001
			raise RuntimeError(
				"MRMS preview backend unavailable. Install with: "
				"pip install xarray cfgrib eccodes"
			) from exc

		try:
			_suppress_cfgrib_merge_futurewarning()
			ds = xr.open_dataset(work_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
			try:
				var_name = next(iter(ds.data_vars))
				arr = np.asarray(ds[var_name].values)
				arr = np.squeeze(arr)
				time_val = ds[var_name].coords.get("time")
				if time_val is not None:
					time_item = np.asarray(time_val.values).reshape(-1)[0]
					timestamp_str = str(time_item)
				else:
					timestamp_str = "unknown_time"
				product_name = str(ds[var_name].attrs.get("GRIB_name", var_name))
				return arr, timestamp_str, product_name
			finally:
				ds.close()
		except Exception as exc:  # noqa: BLE001
			raise RuntimeError(f"cfgrib backend failed: {exc}") from exc
	finally:
		if tmp_file_path is not None:
			try:
				tmp_file_path.unlink(missing_ok=True)
			except Exception:  # noqa: BLE001
				pass


def _generate_mrms_preview(sample_files: list[Path], report_dir: Path, source_name: str) -> None:
	if not sample_files:
		return

	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("MRMS preview skipped (matplotlib unavailable): %s", exc)
		return

	preview_dir = report_dir / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)
	preview_source = sample_files[0]

	try:
		arr, timestamp_str, product_name = _extract_precip_grid(preview_source)
		if arr.ndim != 2:
			LOGGER.warning("MRMS preview skipped (unexpected grid shape %s).", arr.shape)
			return

		fig, ax = plt.subplots(figsize=(8, 5))
		image = ax.imshow(arr, origin="lower")
		fig.colorbar(image, ax=ax, label="precip")
		ax.set_title(f"{source_name} | {product_name} | {timestamp_str}")
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		safe_ts = timestamp_str.replace(":", "-").replace(" ", "_")
		out_path = preview_dir / f"{source_name}_{safe_ts}.png"
		fig.tight_layout()
		fig.savefig(out_path, dpi=150)
		plt.close(fig)
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("MRMS preview generation failed and was skipped: %s", exc)


def _read_rtma_variable_arrays(file_path: Path) -> tuple[dict[str, np.ndarray], str]:
	import cfgrib

	_suppress_cfgrib_merge_futurewarning()
	datasets = cfgrib.open_datasets(str(file_path), backend_kwargs={"indexpath": ""})
	var_map: dict[str, np.ndarray] = {}
	timestamp_str = "unknown_time"
	try:
		for ds in datasets:
			for var_name in ds.data_vars:
				arr = np.asarray(ds[var_name].values)
				arr = np.squeeze(arr)
				if arr.ndim != 2:
					continue
				short_name = str(ds[var_name].attrs.get("GRIB_shortName", "")).lower()
				if short_name in {"2t", "t2m"} and "TMP" not in var_map:
					var_map["TMP"] = arr
				if short_name in {"10u", "u10"} and "UGRD" not in var_map:
					var_map["UGRD"] = arr
				if short_name in {"10v", "v10"} and "VGRD" not in var_map:
					var_map["VGRD"] = arr
				if short_name in {"2sh", "sh2", "spfh"} and "SPFH" not in var_map:
					var_map["SPFH"] = arr
				if short_name in {"2d", "d2m", "dpt"} and "DPT" not in var_map:
					var_map["DPT"] = arr
				if short_name in {"sp", "pres"} and "PRES" not in var_map:
					var_map["PRES"] = arr
				if short_name in {"tcc"} and "TCDC" not in var_map:
					var_map["TCDC"] = arr
				if timestamp_str == "unknown_time":
					time_coord = ds[var_name].coords.get("time")
					if time_coord is not None:
						time_item = np.asarray(time_coord.values).reshape(-1)[0]
						timestamp_str = str(time_item)
	finally:
		for ds in datasets:
			ds.close()
	return var_map, timestamp_str


def _qc_stats(arr: np.ndarray) -> dict[str, float]:
	arrf = arr.astype("float64", copy=False)
	nan_mask = np.isnan(arrf)
	nan_pct = float(nan_mask.sum() / arrf.size * 100.0) if arrf.size else 100.0
	return {
		"min": float(np.nanmin(arrf)),
		"max": float(np.nanmax(arrf)),
		"mean": float(np.nanmean(arrf)),
		"nan_pct": nan_pct,
	}


def _generate_rtma_preview(sample_files: list[Path], report_dir: Path, source_name: str) -> dict:
	if not sample_files:
		return {}

	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("RTMA preview skipped (matplotlib unavailable): %s", exc)
		return {}

	preview_dir = report_dir / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)
	preview_source = sample_files[0]

	try:
		var_map, timestamp_str = _read_rtma_variable_arrays(preview_source)
		humidity_key = "SPFH" if "SPFH" in var_map else ("DPT" if "DPT" in var_map else None)
		target_vars = ["TMP", "UGRD"]
		if humidity_key is not None:
			target_vars.append(humidity_key)

		qc_summary: dict[str, dict[str, float]] = {}
		for var_name in target_vars:
			if var_name not in var_map:
				continue
			arr = var_map[var_name]
			stats = _qc_stats(arr)
			qc_summary[var_name] = stats
			print(
				"RTMA QC "
				f"{var_name}: min={stats['min']:.6g}, max={stats['max']:.6g}, "
				f"mean={stats['mean']:.6g}, nan_pct={stats['nan_pct']:.3f}"
			)

			fig, ax = plt.subplots(figsize=(8, 5))
			image = ax.imshow(arr, origin="lower")
			fig.colorbar(image, ax=ax, label=var_name)
			ax.set_title(f"{source_name} | {var_name} | {timestamp_str}")
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			safe_ts = timestamp_str.replace(":", "-").replace(" ", "_")
			out_path = preview_dir / f"{source_name}_{var_name}_{safe_ts}.png"
			fig.tight_layout()
			fig.savefig(out_path, dpi=150)
			plt.close(fig)

		return qc_summary
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("RTMA preview generation failed and was skipped: %s", exc)
		return {}


def _crop_gfs_to_conus(arr: np.ndarray, coords) -> np.ndarray:
	lat = coords.get("latitude") if "latitude" in coords else coords.get("lat")
	lon = coords.get("longitude") if "longitude" in coords else coords.get("lon")
	if lat is None or lon is None:
		return arr

	lat_arr = np.asarray(lat.values)
	lon_arr = np.asarray(lon.values)
	if lat_arr.ndim == 1 and lon_arr.ndim == 1:
		lon_grid, lat_grid = np.meshgrid(lon_arr, lat_arr)
	else:
		lat_grid = lat_arr
		lon_grid = lon_arr
	lon_grid = np.where(lon_grid > 180.0, lon_grid - 360.0, lon_grid)
	lon_min, lat_min, lon_max, lat_max = CONUS_BBOX.bbox
	mask = (
		(lat_grid >= lat_min)
		& (lat_grid <= lat_max)
		& (lon_grid >= lon_min)
		& (lon_grid <= lon_max)
	)
	if mask.shape != arr.shape:
		return arr
	row_idx = np.where(mask.any(axis=1))[0]
	col_idx = np.where(mask.any(axis=0))[0]
	if row_idx.size == 0 or col_idx.size == 0:
		return arr
	return arr[row_idx.min():row_idx.max() + 1, col_idx.min():col_idx.max() + 1]


def _read_gfs_variable_arrays(file_path: Path) -> tuple[dict[str, np.ndarray], str]:
	import cfgrib

	_suppress_cfgrib_merge_futurewarning()
	datasets = cfgrib.open_datasets(str(file_path), backend_kwargs={"indexpath": ""})
	var_map: dict[str, np.ndarray] = {}
	timestamp_str = "unknown_time"
	try:
		for ds in datasets:
			for var_name in ds.data_vars:
				da = ds[var_name]
				arr = np.asarray(da.values)
				arr = np.squeeze(arr)
				if arr.ndim != 2:
					continue
				arr = _crop_gfs_to_conus(arr, da.coords)
				if arr.size == 0:
					continue
				short_name = str(ds[var_name].attrs.get("GRIB_shortName", "")).lower()
				if short_name in {"2t", "tmp", "t2m"} and "TMP" not in var_map:
					var_map["TMP"] = arr
				if short_name in {"10u", "ugrd", "u10"} and "UGRD" not in var_map:
					var_map["UGRD"] = arr
				if short_name in {"prate", "apcp"} and "PRATE" not in var_map:
					var_map["PRATE"] = arr
				if timestamp_str == "unknown_time":
					time_coord = ds[var_name].coords.get("time")
					if time_coord is not None:
						time_item = np.asarray(time_coord.values).reshape(-1)[0]
						timestamp_str = str(time_item)
	finally:
		for ds in datasets:
			ds.close()
	return var_map, timestamp_str


def _generate_gfs_preview(sample_files: list[Path], report_dir: Path, source_name: str) -> dict:
	if not sample_files:
		return {}

	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("GFS preview skipped (matplotlib unavailable): %s", exc)
		return {}

	preview_dir = report_dir / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)
	preview_source = sample_files[0]

	try:
		var_map, timestamp_str = _read_gfs_variable_arrays(preview_source)
		target_vars = ["TMP", "UGRD", "PRATE"]

		qc_summary: dict[str, dict[str, float]] = {}
		for var_name in target_vars:
			if var_name not in var_map:
				continue
			arr = var_map[var_name]
			stats = _qc_stats(arr)
			qc_summary[var_name] = stats
			print(
				"GFS QC "
				f"{var_name}: min={stats['min']:.6g}, max={stats['max']:.6g}, "
				f"mean={stats['mean']:.6g}, nan_pct={stats['nan_pct']:.3f}"
			)

			fig, ax = plt.subplots(figsize=(8, 5))
			image = ax.imshow(arr, origin="lower")
			fig.colorbar(image, ax=ax, label=var_name)
			ax.set_title(f"{source_name} | {var_name} | {timestamp_str}")
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			safe_ts = timestamp_str.replace(":", "-").replace(" ", "_")
			out_path = preview_dir / f"{source_name}_{var_name}_{safe_ts}.png"
			fig.tight_layout()
			fig.savefig(out_path, dpi=150)
			plt.close(fig)

		return qc_summary
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("GFS preview generation failed and was skipped: %s", exc)
		return {}


def _read_ifs_variable_arrays(file_path: Path) -> tuple[dict[str, np.ndarray], str]:
	import cfgrib

	_suppress_cfgrib_merge_futurewarning()
	datasets = cfgrib.open_datasets(str(file_path), backend_kwargs={"indexpath": ""})
	var_map: dict[str, np.ndarray] = {}
	timestamp_str = "unknown_time"
	try:
		for ds in datasets:
			for var_name in ds.data_vars:
				arr = np.asarray(ds[var_name].values)
				arr = np.squeeze(arr)
				if arr.ndim == 3:
					# IFS cycle files contain multiple lead times; use first lead for quick-look preview.
					arr = arr[0, :, :]
				if arr.ndim != 2:
					continue
				short_name = str(ds[var_name].attrs.get("GRIB_shortName", "")).lower()
				if short_name in {"2t", "t2m", "tmp"} and "TMP" not in var_map:
					var_map["TMP"] = arr
				if short_name in {"10u", "u10", "ugrd"} and "UGRD" not in var_map:
					var_map["UGRD"] = arr
				if short_name in {"tp", "prate", "apcp"} and "TP" not in var_map:
					var_map["TP"] = arr
				if timestamp_str == "unknown_time":
					time_coord = ds[var_name].coords.get("time")
					if time_coord is not None:
						time_item = np.asarray(time_coord.values).reshape(-1)[0]
						timestamp_str = str(time_item)
	finally:
		for ds in datasets:
			ds.close()
	return var_map, timestamp_str


def _generate_ifs_preview(sample_files: list[Path], report_dir: Path, source_name: str) -> dict:
	if not sample_files:
		return {}

	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("IFS preview skipped (matplotlib unavailable): %s", exc)
		return {}

	preview_dir = report_dir / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)
	preview_source = sample_files[0]

	try:
		var_map, timestamp_str = _read_ifs_variable_arrays(preview_source)
		target_vars = ["TMP", "UGRD", "TP"]

		qc_summary: dict[str, dict[str, float]] = {}
		for var_name in target_vars:
			if var_name not in var_map:
				continue
			arr = var_map[var_name]
			stats = _qc_stats(arr)
			qc_summary[var_name] = stats
			print(
				"IFS QC "
				f"{var_name}: min={stats['min']:.6g}, max={stats['max']:.6g}, "
				f"mean={stats['mean']:.6g}, nan_pct={stats['nan_pct']:.3f}"
			)

			fig, ax = plt.subplots(figsize=(8, 5))
			image = ax.imshow(arr, origin="lower")
			fig.colorbar(image, ax=ax, label=var_name)
			ax.set_title(f"{source_name} | {var_name} | {timestamp_str}")
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			safe_ts = timestamp_str.replace(":", "-").replace(" ", "_")
			out_path = preview_dir / f"{source_name}_{var_name}_{safe_ts}.png"
			fig.tight_layout()
			fig.savefig(out_path, dpi=150)
			plt.close(fig)

		return qc_summary
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("IFS preview generation failed and was skipped: %s", exc)
		return {}


def _read_era5_landt_variable_arrays(file_path: Path) -> tuple[dict[str, np.ndarray], str]:
	import cfgrib

	_suppress_cfgrib_merge_futurewarning()
	datasets = cfgrib.open_datasets(str(file_path), backend_kwargs={"indexpath": ""})
	var_map: dict[str, np.ndarray] = {}
	timestamp_str = "unknown_time"
	try:
		for ds in datasets:
			for var_name in ds.data_vars:
				arr = np.asarray(ds[var_name].values)
				arr = np.squeeze(arr)
				while arr.ndim > 2:
					if arr.shape[0] == 0:
						break
					chosen = arr[0]
					for idx in range(arr.shape[0]):
						candidate = arr[idx]
						if np.isfinite(candidate).any():
							chosen = candidate
							break
					arr = chosen
				if arr.ndim != 2:
					continue
				short_name = str(ds[var_name].attrs.get("GRIB_shortName", "")).lower()
				if short_name in {"2t", "t2m", "tmp"} and "TMP" not in var_map:
					var_map["TMP"] = arr
				if short_name in {"swvl1", "swvl"} and "SWVL1" not in var_map:
					var_map["SWVL1"] = arr
				if short_name in {"swvl2"} and "SWVL2" not in var_map:
					var_map["SWVL2"] = arr
				if short_name in {"swvl3"} and "SWVL3" not in var_map:
					var_map["SWVL3"] = arr
				if short_name in {"swvl4"} and "SWVL4" not in var_map:
					var_map["SWVL4"] = arr
				if short_name in {"ssrd", "dswrf"} and "SSRD" not in var_map:
					var_map["SSRD"] = arr
				if short_name in {"sd", "sde"} and "SD" not in var_map:
					var_map["SD"] = arr
				if timestamp_str == "unknown_time":
					time_coord = ds[var_name].coords.get("time")
					if time_coord is not None:
						time_item = np.asarray(time_coord.values).reshape(-1)[0]
						timestamp_str = str(time_item)
	finally:
		for ds in datasets:
			ds.close()
	return var_map, timestamp_str


def _generate_era5_landt_preview(sample_files: list[Path], report_dir: Path, source_name: str) -> dict:
	if not sample_files:
		return {}

	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("ERA5-Land-T preview skipped (matplotlib unavailable): %s", exc)
		return {}

	preview_dir = report_dir / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)
	preview_source = sample_files[0]

	try:
		var_map, timestamp_str = _read_era5_landt_variable_arrays(preview_source)
		target_vars = ["TMP", "SSRD", "SWVL1", "SWVL2", "SWVL3", "SWVL4", "SD"]

		qc_summary: dict[str, dict[str, float]] = {}
		for var_name in target_vars:
			if var_name not in var_map:
				continue
			arr = var_map[var_name]
			stats = _qc_stats(arr)
			qc_summary[var_name] = stats
			print(
				"ERA5-Land-T QC "
				f"{var_name}: min={stats['min']:.6g}, max={stats['max']:.6g}, "
				f"mean={stats['mean']:.6g}, nan_pct={stats['nan_pct']:.3f}"
			)

			fig, ax = plt.subplots(figsize=(8, 5))
			image = ax.imshow(arr, origin="lower")
			fig.colorbar(image, ax=ax, label=var_name)
			ax.set_title(f"{source_name} | {var_name} | {timestamp_str}")
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			safe_ts = timestamp_str.replace(":", "-").replace(" ", "_")
			out_path = preview_dir / f"{source_name}_{var_name}_{safe_ts}.png"
			fig.tight_layout()
			fig.savefig(out_path, dpi=150)
			plt.close(fig)

		return qc_summary
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("ERA5-Land-T preview generation failed and was skipped: %s", exc)
		return {}


def _read_gdas_variable_arrays(file_path: Path) -> tuple[dict[str, np.ndarray], str]:
	import cfgrib

	_suppress_cfgrib_merge_futurewarning()
	datasets = cfgrib.open_datasets(str(file_path), backend_kwargs={"indexpath": ""})
	var_map: dict[str, np.ndarray] = {}
	timestamp_str = "unknown_time"
	try:
		for ds in datasets:
			for var_name in ds.data_vars:
				da = ds[var_name]
				arr = np.asarray(da.values)
				arr = np.squeeze(arr)
				while arr.ndim > 2:
					if arr.shape[0] == 0:
						break
					arr = arr[0]
				if arr.ndim != 2:
					continue
				arr = _crop_gfs_to_conus(arr, da.coords)
				if arr.size == 0:
					continue
				short_name = str(ds[var_name].attrs.get("GRIB_shortName", "")).lower()
				if short_name in {"2t", "tmp", "t2m"} and "TMP" not in var_map:
					var_map["TMP"] = arr
				if short_name in {"rh"} and "RH" not in var_map:
					var_map["RH"] = arr
				if short_name in {"prmsl", "msl"} and "PRMSL" not in var_map:
					var_map["PRMSL"] = arr
				if short_name in {"dswrf", "ssrd"} and "DSWRF" not in var_map:
					var_map["DSWRF"] = arr
				if short_name in {"10u", "ugrd", "u10"} and "UGRD" not in var_map:
					var_map["UGRD"] = arr
				if timestamp_str == "unknown_time":
					time_coord = ds[var_name].coords.get("time")
					if time_coord is not None:
						time_item = np.asarray(time_coord.values).reshape(-1)[0]
						timestamp_str = str(time_item)
	finally:
		for ds in datasets:
			ds.close()
	return var_map, timestamp_str


def _generate_gdas_preview(sample_files: list[Path], report_dir: Path, source_name: str) -> dict:
	if not sample_files:
		return {}

	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("GDAS preview skipped (matplotlib unavailable): %s", exc)
		return {}

	preview_dir = report_dir / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)
	preview_source = sample_files[0]

	try:
		var_map, timestamp_str = _read_gdas_variable_arrays(preview_source)
		moisture_or_pressure = "RH" if "RH" in var_map else ("PRMSL" if "PRMSL" in var_map else None)
		target_vars = ["TMP"]
		if moisture_or_pressure is not None:
			target_vars.append(moisture_or_pressure)
		if "DSWRF" in var_map:
			target_vars.append("DSWRF")
		elif "UGRD" in var_map:
			target_vars.append("UGRD")

		qc_summary: dict[str, dict[str, float]] = {}
		for var_name in target_vars:
			if var_name not in var_map:
				continue
			arr = var_map[var_name]
			stats = _qc_stats(arr)
			qc_summary[var_name] = stats
			print(
				"GDAS QC "
				f"{var_name}: min={stats['min']:.6g}, max={stats['max']:.6g}, "
				f"mean={stats['mean']:.6g}, nan_pct={stats['nan_pct']:.3f}"
			)

			fig, ax = plt.subplots(figsize=(8, 5))
			image = ax.imshow(arr, origin="lower")
			fig.colorbar(image, ax=ax, label=var_name)
			ax.set_title(f"{source_name} | {var_name} | {timestamp_str}")
			ax.set_xlabel("x")
			ax.set_ylabel("y")
			safe_ts = timestamp_str.replace(":", "-").replace(" ", "_")
			out_path = preview_dir / f"{source_name}_{var_name}_{safe_ts}.png"
			fig.tight_layout()
			fig.savefig(out_path, dpi=150)
			plt.close(fig)

		return qc_summary
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("GDAS preview generation failed and was skipped: %s", exc)
		return {}


def _read_imerg_variable_arrays(file_path: Path) -> tuple[dict[str, np.ndarray], str, Optional[dict[str, float]]]:
	try:
		import xarray as xr
	except Exception as exc:  # noqa: BLE001
		raise RuntimeError("IMERG preview requires xarray and netcdf support.") from exc

	ds = xr.open_dataset(file_path)
	var_map: dict[str, np.ndarray] = {}
	timestamp_str = "unknown_time"
	crop_info: Optional[dict[str, float]] = None
	try:
		if "precipitation" in ds.data_vars:
			da = ds["precipitation"]
		else:
			first_var = next(iter(ds.data_vars), None)
			if first_var is None:
				return var_map, timestamp_str, crop_info
			da = ds[first_var]

		arr = np.asarray(da.values)
		arr = np.squeeze(arr)
		while arr.ndim > 2:
			arr = arr[0]
		lat_name = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
		lon_name = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
		if lat_name is not None and lon_name is not None:
			lon_min, lat_min, lon_max, lat_max = CONUS_BBOX.bbox
			da, crop_info = _subset_imerg_to_bbox(da, lat_name, lon_name, lon_min, lat_min, lon_max, lat_max)
			arr = np.asarray(da.values)
			arr = np.squeeze(arr)
			while arr.ndim > 2:
				arr = arr[0]
			if crop_info is not None:
				crop_info = dict(crop_info)
				crop_info["shape"] = tuple(int(v) for v in arr.shape)

		if arr.ndim == 2 and arr.size > 0:
			var_map["PRECIP"] = arr

		time_coord = da.coords.get("time")
		if time_coord is not None:
			time_item = np.asarray(time_coord.values).reshape(-1)[0]
			timestamp_str = str(time_item)
	finally:
		ds.close()

	return var_map, timestamp_str, crop_info


def _subset_imerg_to_bbox(da, lat_name: str, lon_name: str, lon_min: float, lat_min: float, lon_max: float, lat_max: float):
	lat_vals = np.asarray(da[lat_name].values)
	lon_vals = np.asarray(da[lon_name].values)

	if lat_vals.ndim != 1 or lon_vals.ndim != 1:
		return da, None

	lon_vals_norm = np.where(lon_vals > 180.0, lon_vals - 360.0, lon_vals)
	lon_order = np.argsort(lon_vals_norm)
	if not np.array_equal(lon_order, np.arange(lon_vals_norm.size)):
		da = da.isel({lon_name: lon_order})
		lon_vals_norm = lon_vals_norm[lon_order]

	da = da.assign_coords({lon_name: lon_vals_norm})

	lat_ascending = bool(lat_vals[0] <= lat_vals[-1])
	lon_ascending = bool(lon_vals_norm[0] <= lon_vals_norm[-1])
	lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
	lon_slice = slice(lon_min, lon_max) if lon_ascending else slice(lon_max, lon_min)
	subset = da.sel({lat_name: lat_slice, lon_name: lon_slice})
	sub_lat = np.asarray(subset[lat_name].values)
	sub_lon = np.asarray(subset[lon_name].values)
	if sub_lat.size == 0 or sub_lon.size == 0:
		return subset, None
	crop_info = {
		"lon_min": float(np.min(sub_lon)),
		"lon_max": float(np.max(sub_lon)),
		"lat_min": float(np.min(sub_lat)),
		"lat_max": float(np.max(sub_lat)),
	}
	return subset, crop_info


def _generate_imerg_preview(sample_files: list[Path], report_dir: Path, source_name: str) -> dict:
	if not sample_files:
		return {}

	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("IMERG preview skipped (matplotlib unavailable): %s", exc)
		return {}

	preview_dir = report_dir / "preview"
	preview_dir.mkdir(parents=True, exist_ok=True)
	preview_source = sample_files[0]

	try:
		var_map, timestamp_str, crop_info = _read_imerg_variable_arrays(preview_source)
		if "PRECIP" not in var_map:
			return {}

		arr = var_map["PRECIP"]
		if crop_info is not None:
			print(
				"IMERG preview crop bounds: "
				f"lon=[{crop_info['lon_min']:.3f}, {crop_info['lon_max']:.3f}], "
				f"lat=[{crop_info['lat_min']:.3f}, {crop_info['lat_max']:.3f}], "
				f"shape={crop_info.get('shape', tuple(arr.shape))}"
			)
		stats = _qc_stats(arr)
		qc_summary = {"PRECIP": stats}
		print(
			"IMERG QC "
			f"PRECIP: min={stats['min']:.6g}, max={stats['max']:.6g}, "
			f"mean={stats['mean']:.6g}, nan_pct={stats['nan_pct']:.3f}"
		)

		fig, ax = plt.subplots(figsize=(8, 5))
		if crop_info is not None:
			image = ax.imshow(
				arr,
				origin="lower",
				extent=[crop_info["lon_min"], crop_info["lon_max"], crop_info["lat_min"], crop_info["lat_max"]],
				aspect="auto",
			)
			ax.set_xlabel("longitude")
			ax.set_ylabel("latitude")
		else:
			image = ax.imshow(arr, origin="lower")
			ax.set_xlabel("x")
			ax.set_ylabel("y")
		fig.colorbar(image, ax=ax, label="PRECIP")
		ax.set_title(f"{source_name} | PRECIP | {timestamp_str}")
		safe_ts = timestamp_str.replace(":", "-").replace(" ", "_")
		out_path = preview_dir / f"{source_name}_PRECIP_{safe_ts}.png"
		fig.tight_layout()
		fig.savefig(out_path, dpi=150)
		plt.close(fig)
		return qc_summary
	except Exception as exc:  # noqa: BLE001
		LOGGER.warning("IMERG preview generation failed and was skipped: %s", exc)
		return {}


def run_estimation(config: Config) -> tuple[list[EstimateResult], dict]:
	ratio_info: dict = {"parquet_ratio_by_source": {}, "preview_qc_by_source": {}}
	results: list[EstimateResult] = []
	sources = get_enabled_sources(config)
	object_cache: dict[str, list] = {}
	mrms_available_start: Optional[datetime] = None
	mrms_available_end: Optional[datetime] = None
	mrms_effective_start: Optional[datetime] = None
	mrms_effective_end: Optional[datetime] = None

	for source in sources:
		source_variables = config.variables
		if source.name == "mrms_qpe_1h_pass1":
			source_variables = ["precip"]
		if source.name == "rtma_conus_aws_2p5km":
			source_variables = ["TMP", "SPFH_or_DPT", "UGRD", "VGRD", "PRES"]
		if source.name == "gfs_conus_aws_0p25":
			source_variables = ["PRATE", "TMP", "RH", "UGRD", "VGRD", "PRMSL", "DSWRF"]
		if source.name == "ifs_mars_conus":
			source_variables = ["TP", "2T", "2D", "10U", "10V", "SP", "SSRD"]
		if source.name == "era5_land_t_conus":
			source_variables = ["TP", "2T", "2D", "10U", "10V", "SP", "SSRD", "SWVL1", "SWVL2", "SWVL3", "SWVL4", "SD"]
		if source.name == "gdas_conus_aws_0p25":
			source_variables = ["TMP", "RH", "UGRD", "VGRD", "PRMSL", "DSWRF", "PRATE"]
		if source.name == "imerg_late_daily_conus":
			source_variables = ["PRECIP"]
		objects = source.list_sample_objects(
			start=config.sample_start,
			end=config.sample_end,
			region=config.region,
			variables=source_variables,
		)
		object_cache[source.name] = objects
		if source.name == "mrms_qpe_1h_pass1":
			mrms_available_start, mrms_available_end = _read_available_extent(source)
			break

	if config.range_mode == "source_full":
		mrms_effective_start, mrms_effective_end = _intersect_ranges(
			config.full_start,
			config.full_end,
			mrms_available_start,
			mrms_available_end,
		)
	else:
		mrms_effective_start, mrms_effective_end = _intersect_ranges(
			config.full_start,
			config.full_end,
			mrms_available_start,
			mrms_available_end,
		)

	for source in sources:
		source_variables = config.variables
		if source.name == "mrms_qpe_1h_pass1":
			source_variables = ["precip"]
		if source.name == "rtma_conus_aws_2p5km":
			source_variables = ["TMP", "SPFH_or_DPT", "UGRD", "VGRD", "PRES"]
		if source.name == "gfs_conus_aws_0p25":
			source_variables = ["PRATE", "TMP", "RH", "UGRD", "VGRD", "PRMSL", "DSWRF"]
		if source.name == "ifs_mars_conus":
			source_variables = ["TP", "2T", "2D", "10U", "10V", "SP", "SSRD"]
		if source.name == "era5_land_t_conus":
			source_variables = ["TP", "2T", "2D", "10U", "10V", "SP", "SSRD", "SWVL1", "SWVL2", "SWVL3", "SWVL4", "SD"]
		if source.name == "gdas_conus_aws_0p25":
			source_variables = ["TMP", "RH", "UGRD", "VGRD", "PRMSL", "DSWRF", "PRATE"]
		if source.name == "imerg_late_daily_conus":
			source_variables = ["PRECIP"]
		if config.derived_format == "parquet":
			ratio_info["parquet_ratio_by_source"][source.name] = calibrate_parquet_ratio(
				n_vars=len(source_variables),
				dtype=config.derived_dtype,
			)

		objects = object_cache.get(source.name)
		if objects is None:
			objects = source.list_sample_objects(
				start=config.sample_start,
				end=config.sample_end,
				region=config.region,
				variables=source_variables,
			)

		source_available_start, source_available_end = _read_available_extent(source)
		effective_prediction_start: Optional[datetime] = None
		effective_prediction_end: Optional[datetime] = None
		era5_required_data_start: Optional[datetime] = None
		era5_required_data_end: Optional[datetime] = None
		gdas_required_data_start: Optional[datetime] = None
		gdas_required_data_end: Optional[datetime] = None
		imerg_required_data_start: Optional[datetime] = None
		imerg_required_data_end: Optional[datetime] = None
		if config.range_mode == "source_full":
			effective_start, effective_end = _intersect_ranges(
				config.full_start,
				config.full_end,
				source_available_start,
				source_available_end,
			)
		else:
			mrms_clip_start, mrms_clip_end = _intersect_ranges(
				config.full_start,
				config.full_end,
				mrms_available_start,
				mrms_available_end,
			)
			if mrms_clip_start is None or mrms_clip_end is None:
				effective_start, effective_end = None, None
			else:
				effective_start, effective_end = _intersect_ranges(
					mrms_clip_start,
					mrms_clip_end,
					source_available_start,
					source_available_end,
				)

		if source.name == "era5_land_t_conus":
			windows = _compute_antecedent_windows(
				mrms_effective_start=mrms_effective_start,
				mrms_effective_end=mrms_effective_end,
				source_available_start=source_available_start,
				source_available_end=source_available_end,
				latency_days=getattr(source, "LATENCY_DAYS", 5),
				lookback_days=getattr(source, "LOOKBACK_DAYS", 365),
			)
			effective_prediction_start = windows["effective_prediction_start"]
			effective_prediction_end = windows["effective_prediction_end"]
			era5_required_data_start = windows["required_data_start"]
			era5_required_data_end = windows["required_data_end"]
			effective_start = era5_required_data_start
			effective_end = era5_required_data_end
			if hasattr(source, "set_required_window"):
				source.set_required_window(era5_required_data_start, era5_required_data_end)

		if source.name == "gdas_conus_aws_0p25":
			windows = _compute_antecedent_windows(
				mrms_effective_start=mrms_effective_start,
				mrms_effective_end=mrms_effective_end,
				source_available_start=source_available_start,
				source_available_end=source_available_end,
				latency_days=getattr(source, "LATENCY_DAYS", 1),
				lookback_days=getattr(source, "LOOKBACK_DAYS", 365),
			)
			effective_prediction_start = windows["effective_prediction_start"]
			effective_prediction_end = windows["effective_prediction_end"]
			gdas_required_data_start = windows["required_data_start"]
			gdas_required_data_end = windows["required_data_end"]
			effective_start = gdas_required_data_start
			effective_end = gdas_required_data_end
			if hasattr(source, "set_required_window"):
				source.set_required_window(gdas_required_data_start, gdas_required_data_end)

		if source.name == "imerg_late_daily_conus":
			windows = _compute_antecedent_windows(
				mrms_effective_start=mrms_effective_start,
				mrms_effective_end=mrms_effective_end,
				source_available_start=source_available_start,
				source_available_end=source_available_end,
				latency_days=getattr(source, "LATENCY_DAYS", 1),
				lookback_days=getattr(source, "LOOKBACK_DAYS", 365),
			)
			effective_prediction_start = windows["effective_prediction_start"]
			effective_prediction_end = windows["effective_prediction_end"]
			imerg_required_data_start = windows["required_data_start"]
			imerg_required_data_end = windows["required_data_end"]
			effective_start = imerg_required_data_start
			effective_end = imerg_required_data_end
			if hasattr(source, "set_required_window"):
				source.set_required_window(imerg_required_data_start, imerg_required_data_end)

		covers_effective_range = (
			effective_start is not None
			and effective_end is not None
			and (
				source_available_start is None
				or source_available_end is None
				or (
					source_available_start <= effective_start
					and source_available_end >= effective_end
				)
			)
		)

		if not objects:
			results.append(
				EstimateResult(
					source=source.name,
					sample_range=f"{config.sample_start.date()} to {config.sample_end.date()}",
					sample_files=0,
					raw_sample_bytes=None,
					raw_sample_full_file_bytes=None,
					raw_sample_selected_bytes=None,
					raw_sample_selected_conus_bytes=None,
					raw_hot_bytes=None,
					raw_hot_full_file_bytes=None,
					raw_hot_selected_bytes=None,
					raw_hot_selected_conus_bytes=None,
					derived_hot_bytes=None,
					raw_cold_bytes=None,
					peak_local_bytes=None,
					source_available_start=_to_iso(source_available_start),
					source_available_end=_to_iso(source_available_end),
					mrms_available_start=_to_iso(mrms_available_start),
					mrms_available_end=_to_iso(mrms_available_end),
					effective_start=_to_iso(effective_start),
					effective_end=_to_iso(effective_end),
					effective_prediction_start=_to_iso(effective_prediction_start),
					effective_prediction_end=_to_iso(effective_prediction_end),
					era5_land_required_data_start=_to_iso(era5_required_data_start),
					era5_land_required_data_end=_to_iso(era5_required_data_end),
					gdas_required_data_start=_to_iso(gdas_required_data_start),
					gdas_required_data_end=_to_iso(gdas_required_data_end),
					imerg_required_data_start=_to_iso(imerg_required_data_start),
					imerg_required_data_end=_to_iso(imerg_required_data_end),
					range_mode=config.range_mode,
					covers_effective_range=covers_effective_range,
					notes=(
						"No sample objects returned. "
						f"Totals based on {config.range_mode} range."
					),
					assumptions=source.assumptions(),
				)
			)
			continue

		if effective_start is None or effective_end is None:
			results.append(
				EstimateResult(
					source=source.name,
					sample_range=f"{config.sample_start.date()} to {config.sample_end.date()}",
					sample_files=len(objects),
					raw_sample_bytes=None,
					raw_sample_full_file_bytes=None,
					raw_sample_selected_bytes=None,
					raw_sample_selected_conus_bytes=None,
					raw_hot_bytes=None,
					raw_hot_full_file_bytes=None,
					raw_hot_selected_bytes=None,
					raw_hot_selected_conus_bytes=None,
					derived_hot_bytes=None,
					raw_cold_bytes=None,
					peak_local_bytes=None,
					source_available_start=_to_iso(source_available_start),
					source_available_end=_to_iso(source_available_end),
					mrms_available_start=_to_iso(mrms_available_start),
					mrms_available_end=_to_iso(mrms_available_end),
					effective_start=None,
					effective_end=None,
					effective_prediction_start=_to_iso(effective_prediction_start),
					effective_prediction_end=_to_iso(effective_prediction_end),
					era5_land_required_data_start=_to_iso(era5_required_data_start),
					era5_land_required_data_end=_to_iso(era5_required_data_end),
					gdas_required_data_start=_to_iso(gdas_required_data_start),
					gdas_required_data_end=_to_iso(gdas_required_data_end),
					imerg_required_data_start=_to_iso(imerg_required_data_start),
					imerg_required_data_end=_to_iso(imerg_required_data_end),
					range_mode=config.range_mode,
					covers_effective_range=False,
					notes=(
						"No overlap between requested and effective range. "
						f"Totals based on {config.range_mode} range."
					),
					assumptions=source.assumptions(),
				)
			)
			continue

		if config.dry_run:
			full_sample_bytes = 0
			if all(obj.estimated_bytes is not None for obj in objects):
				full_sample_bytes = sum(obj.estimated_bytes or 0 for obj in objects)
			else:
				full_sample_bytes = None
			selected_sample_bytes = None
			selected_conus_sample_bytes = None
		else:
			files = source.download_sample(config.out_dir / source.name, objects)
			if config.make_preview and source.name == "mrms_qpe_1h_pass1":
				_generate_mrms_preview(files, config.report_dir, source.name)
			if config.make_preview and source.name == "rtma_conus_aws_2p5km":
				ratio_info["preview_qc_by_source"][source.name] = _generate_rtma_preview(
					files,
					config.report_dir,
					source.name,
				)
			if config.make_preview and source.name == "gfs_conus_aws_0p25":
				ratio_info["preview_qc_by_source"][source.name] = _generate_gfs_preview(
					files,
					config.report_dir,
					source.name,
				)
			if config.make_preview and source.name == "ifs_mars_conus":
				ratio_info["preview_qc_by_source"][source.name] = _generate_ifs_preview(
					files,
					config.report_dir,
					source.name,
				)
			if config.make_preview and source.name == "era5_land_t_conus":
				ratio_info["preview_qc_by_source"][source.name] = _generate_era5_landt_preview(
					files,
					config.report_dir,
					source.name,
				)
			if config.make_preview and source.name == "gdas_conus_aws_0p25":
				ratio_info["preview_qc_by_source"][source.name] = _generate_gdas_preview(
					files,
					config.report_dir,
					source.name,
				)
			if config.make_preview and source.name == "imerg_late_daily_conus":
				ratio_info["preview_qc_by_source"][source.name] = _generate_imerg_preview(
					files,
					config.report_dir,
					source.name,
				)
			full_sample_bytes = source.measure_bytes(files)
			if hasattr(source, "measure_full_file_sample_bytes"):
				full_candidate = source.measure_full_file_sample_bytes(objects)
				if full_candidate is not None:
					full_sample_bytes = full_candidate
			selected_sample_bytes = None
			if hasattr(source, "measure_selected_variable_bytes"):
				selected_sample_bytes, _ = source.measure_selected_variable_bytes(files, objects)
			selected_conus_sample_bytes = None
			if hasattr(source, "measure_selected_conus_bytes"):
				selected_conus_sample_bytes, _ = source.measure_selected_conus_bytes(files, config.region)

		sample_bytes = (
			selected_conus_sample_bytes
			if selected_conus_sample_bytes is not None
			else (selected_sample_bytes if selected_sample_bytes is not None else full_sample_bytes)
		)

		raw_hot_full = (
			source.estimate_raw_total(
				sample_bytes=full_sample_bytes,
				sample_start=config.sample_start,
				sample_end=config.sample_end,
				full_start=effective_start,
				full_end=effective_end,
			)
			if full_sample_bytes is not None
			else None
		)

		raw_hot_selected = (
			source.estimate_raw_total(
				sample_bytes=selected_sample_bytes,
				sample_start=config.sample_start,
				sample_end=config.sample_end,
				full_start=effective_start,
				full_end=effective_end,
			)
			if selected_sample_bytes is not None
			else None
		)

		raw_hot_selected_conus = (
			source.estimate_raw_total(
				sample_bytes=selected_conus_sample_bytes,
				sample_start=config.sample_start,
				sample_end=config.sample_end,
				full_start=effective_start,
				full_end=effective_end,
			)
			if selected_conus_sample_bytes is not None
			else None
		)

		raw_hot = (
			raw_hot_selected_conus
			if raw_hot_selected_conus is not None
			else (raw_hot_selected if raw_hot_selected is not None else raw_hot_full)
		)

		derived_spec = DerivedSpec(
			n_basins=config.n_basins,
			start=effective_start,
			end=effective_end,
			temporal_resolution=source.temporal_resolution,
			n_vars=len(source_variables),
			dtype=config.derived_dtype,
			output_format=config.derived_format,
		)
		derived_hot = source.estimate_derived_total(derived_spec)
		raw_cold = raw_hot
		peak_local = (
			source.estimate_peak_local(raw_hot, config.scratch_multiplier, config.concurrency)
			if raw_hot is not None
			else None
		)

		results.append(
			EstimateResult(
				source=source.name,
				sample_range=f"{config.sample_start.date()} to {config.sample_end.date()}",
				sample_files=len(objects),
				raw_sample_bytes=sample_bytes,
				raw_sample_full_file_bytes=full_sample_bytes,
				raw_sample_selected_bytes=selected_sample_bytes,
				raw_sample_selected_conus_bytes=selected_conus_sample_bytes,
				raw_hot_bytes=raw_hot,
				raw_hot_full_file_bytes=raw_hot_full,
				raw_hot_selected_bytes=raw_hot_selected,
				raw_hot_selected_conus_bytes=raw_hot_selected_conus,
				derived_hot_bytes=derived_hot,
				raw_cold_bytes=raw_cold,
				peak_local_bytes=peak_local,
				source_available_start=_to_iso(source_available_start),
				source_available_end=_to_iso(source_available_end),
				mrms_available_start=_to_iso(mrms_available_start),
				mrms_available_end=_to_iso(mrms_available_end),
				effective_start=_to_iso(effective_start),
				effective_end=_to_iso(effective_end),
				effective_prediction_start=_to_iso(effective_prediction_start),
				effective_prediction_end=_to_iso(effective_prediction_end),
				era5_land_required_data_start=_to_iso(era5_required_data_start),
				era5_land_required_data_end=_to_iso(era5_required_data_end),
				gdas_required_data_start=_to_iso(gdas_required_data_start),
				gdas_required_data_end=_to_iso(gdas_required_data_end),
				imerg_required_data_start=_to_iso(imerg_required_data_start),
				imerg_required_data_end=_to_iso(imerg_required_data_end),
				range_mode=config.range_mode,
				covers_effective_range=covers_effective_range,
				notes=(
					("OK" if sample_bytes is not None else "Missing sample bytes (dry-run without file:size).")
					+ (
						" Using selected-CONUS raw estimate."
						if raw_hot_selected_conus is not None
						else (
							" Using selected-variable raw estimate."
							if raw_hot_selected is not None
							else ""
						)
					)
					+ (
						" CONUS crop is local."
						if source.name in {"gfs_conus_aws_0p25", "gdas_conus_aws_0p25", "imerg_late_daily_conus"} and raw_hot_selected_conus is not None
						else ""
					)
					+ f" Totals based on {config.range_mode} range."
				),
				assumptions=source.assumptions(),
			)
		)

	return results, ratio_info


def benchmark_mrms_aws_concurrency(config: Config, concurrency_levels: list[int]) -> tuple[list[BenchmarkResult], int]:
	if config.mrms_backend != "aws":
		raise ValueError("Benchmark mode is supported for MRMS AWS backend only.")

	results: list[BenchmarkResult] = []	
	for concurrency in concurrency_levels:
		source = MrmsAwsQpe1hPass1(
			debug_listing=config.mrms_debug_listing,
			download_concurrency=concurrency,
		)
		objects = source.list_sample_objects(
			start=config.sample_start,
			end=config.sample_end,
			region=config.region,
			variables=["precip"],
		)

		run_start = time.perf_counter()
		files = source.download_sample(config.out_dir / f"mrms_benchmark_c{concurrency}", objects)
		total_bytes = source.measure_bytes(files)
		total_wall_time_s = time.perf_counter() - run_start

		file_times = getattr(source, "_last_file_times", [])
		avg_file_time_s = float(np.mean(file_times)) if file_times else 0.0
		p90_file_time_s = float(np.percentile(file_times, 90)) if file_times else 0.0
		files_downloaded = len(files)
		files_per_s = files_downloaded / total_wall_time_s if total_wall_time_s > 0 else 0.0
		mb_per_s = (total_bytes / (1024 * 1024)) / total_wall_time_s if total_wall_time_s > 0 else 0.0

		results.append(
			BenchmarkResult(
				concurrency=concurrency,
				total_wall_time_s=total_wall_time_s,
				files_downloaded=files_downloaded,
				total_bytes=total_bytes,
				files_per_s=files_per_s,
				mb_per_s=mb_per_s,
				retry_count=getattr(source, "_retry_count", 0),
				warning_count=getattr(source, "_warning_count", 0),
				avg_file_time_s=avg_file_time_s,
				p90_file_time_s=p90_file_time_s,
			)
		)

	if not results:
		raise RuntimeError("No benchmark results were produced.")

	best = max(results, key=lambda item: item.mb_per_s)
	return results, best.concurrency


def config_to_dict(config: Config) -> dict:
	data = asdict(config)
	data["region"] = config.region.name
	data["out_dir"] = str(config.out_dir)
	data["report_dir"] = str(config.report_dir)
	data["sample_start"] = config.sample_start.isoformat()
	data["sample_end"] = config.sample_end.isoformat()
	data["full_start"] = config.full_start.isoformat()
	data["full_end"] = config.full_end.isoformat()
	data["mrms_backend"] = config.mrms_backend
	data["mrms_debug_listing"] = config.mrms_debug_listing
	data["range_mode"] = config.range_mode
	data["make_preview"] = config.make_preview
	return data


def default_region() -> Region:
	return CONUS_BBOX
