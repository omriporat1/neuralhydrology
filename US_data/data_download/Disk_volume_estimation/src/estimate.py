from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import gzip
import logging
from pathlib import Path
import tempfile
from typing import Optional
import time

import numpy as np

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region
from src.datasources.mrms import MrmsAwsQpe1hPass1, MrmsDataSource
from src.derived_size import calibrate_parquet_ratio


LOGGER = logging.getLogger(__name__)


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


@dataclass
class EstimateResult:
	source: str
	sample_range: str
	sample_files: int
	raw_sample_bytes: Optional[int]
	raw_hot_bytes: Optional[int]
	derived_hot_bytes: Optional[int]
	raw_cold_bytes: Optional[int]
	peak_local_bytes: Optional[int]
	source_available_start: Optional[str]
	source_available_end: Optional[str]
	mrms_available_start: Optional[str]
	mrms_available_end: Optional[str]
	effective_start: Optional[str]
	effective_end: Optional[str]
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
		return [MrmsDataSource()]
	return [
		MrmsAwsQpe1hPass1(
			debug_listing=config.mrms_debug_listing,
			download_concurrency=config.concurrency,
		)
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


def run_estimation(config: Config) -> tuple[list[EstimateResult], dict]:
	ratio_info: dict = {"parquet_ratio_by_source": {}}
	results: list[EstimateResult] = []
	sources = get_enabled_sources(config)
	object_cache: dict[str, list] = {}
	mrms_available_start: Optional[datetime] = None
	mrms_available_end: Optional[datetime] = None

	for source in sources:
		source_variables = ["precip"] if source.name == "mrms_qpe_1h_pass1" else config.variables
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

	for source in sources:
		source_variables = ["precip"] if source.name == "mrms_qpe_1h_pass1" else config.variables
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
					raw_hot_bytes=None,
					derived_hot_bytes=None,
					raw_cold_bytes=None,
					peak_local_bytes=None,
					source_available_start=_to_iso(source_available_start),
					source_available_end=_to_iso(source_available_end),
					mrms_available_start=_to_iso(mrms_available_start),
					mrms_available_end=_to_iso(mrms_available_end),
					effective_start=_to_iso(effective_start),
					effective_end=_to_iso(effective_end),
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
					raw_hot_bytes=None,
					derived_hot_bytes=None,
					raw_cold_bytes=None,
					peak_local_bytes=None,
					source_available_start=_to_iso(source_available_start),
					source_available_end=_to_iso(source_available_end),
					mrms_available_start=_to_iso(mrms_available_start),
					mrms_available_end=_to_iso(mrms_available_end),
					effective_start=None,
					effective_end=None,
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
			sample_bytes = 0
			if all(obj.estimated_bytes is not None for obj in objects):
				sample_bytes = sum(obj.estimated_bytes or 0 for obj in objects)
			else:
				sample_bytes = None
		else:
			files = source.download_sample(config.out_dir / source.name, objects)
			if config.make_preview and source.name == "mrms_qpe_1h_pass1":
				_generate_mrms_preview(files, config.report_dir, source.name)
			sample_bytes = source.measure_bytes(files)

		raw_hot = (
			source.estimate_raw_total(
				sample_bytes=sample_bytes,
				sample_start=config.sample_start,
				sample_end=config.sample_end,
				full_start=effective_start,
				full_end=effective_end,
			)
			if sample_bytes is not None
			else None
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
				raw_hot_bytes=raw_hot,
				derived_hot_bytes=derived_hot,
				raw_cold_bytes=raw_cold,
				peak_local_bytes=peak_local,
				source_available_start=_to_iso(source_available_start),
				source_available_end=_to_iso(source_available_end),
				mrms_available_start=_to_iso(mrms_available_start),
				mrms_available_end=_to_iso(mrms_available_end),
				effective_start=_to_iso(effective_start),
				effective_end=_to_iso(effective_end),
				range_mode=config.range_mode,
				covers_effective_range=covers_effective_range,
				notes=(
					("OK" if sample_bytes is not None else "Missing sample bytes (dry-run without file:size).")
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
