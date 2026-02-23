from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region
from src.datasources.mrms import MrmsAwsQpe1hPass1, MrmsDataSource
from src.derived_size import calibrate_parquet_ratio


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
	notes: str
	assumptions: dict


def get_enabled_sources(config: Config) -> list[DataSource]:
	if config.mrms_backend == "planetary":
		return [MrmsDataSource()]
	return [MrmsAwsQpe1hPass1(debug_listing=config.mrms_debug_listing)]


def run_estimation(config: Config) -> tuple[list[EstimateResult], dict]:
	ratio_info: dict = {"parquet_ratio_by_source": {}}
	results: list[EstimateResult] = []
	for source in get_enabled_sources(config):
		source_variables = ["precip"] if source.name == "mrms_qpe_1h_pass1" else config.variables
		if config.derived_format == "parquet":
			ratio_info["parquet_ratio_by_source"][source.name] = calibrate_parquet_ratio(
				n_vars=len(source_variables),
				dtype=config.derived_dtype,
			)

		objects = source.list_sample_objects(
			start=config.sample_start,
			end=config.sample_end,
			region=config.region,
			variables=source_variables,
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
					notes="No sample objects returned.",
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
			sample_bytes = source.measure_bytes(files)

		raw_hot = (
			source.estimate_raw_total(
				sample_bytes=sample_bytes,
				sample_start=config.sample_start,
				sample_end=config.sample_end,
				full_start=config.full_start,
				full_end=config.full_end,
			)
			if sample_bytes is not None
			else None
		)

		derived_spec = DerivedSpec(
			n_basins=config.n_basins,
			start=config.full_start,
			end=config.full_end,
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
				notes="OK" if sample_bytes is not None else "Missing sample bytes (dry-run without file:size).",
				assumptions=source.assumptions(),
			)
		)

	return results, ratio_info


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
	return data


def default_region() -> Region:
	return CONUS_BBOX
