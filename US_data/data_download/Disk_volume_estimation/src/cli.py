from __future__ import annotations

import argparse
from datetime import datetime, time
from pathlib import Path

from src.estimate import Config, benchmark_mrms_aws_concurrency, config_to_dict, default_region, run_estimation
from src.report import print_benchmark_table, print_table, write_benchmark_csv, write_csv, write_json


def _parse_datetime(value: str, end_of_day: bool = False) -> datetime:
	if len(value) == 10:
		date_val = datetime.fromisoformat(value).date()
		return datetime.combine(date_val, time.max if end_of_day else time.min)
	return datetime.fromisoformat(value)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Disk volume estimation tool")
	parser.add_argument("--sample-start", required=True)
	parser.add_argument("--sample-end", required=True)
	parser.add_argument("--full-start", required=True)
	parser.add_argument("--full-end", required=True)
	parser.add_argument("--region", default="conus_bbox")
	parser.add_argument("--out", default="./data/sample")
	parser.add_argument("--report", default="./reports")
	parser.add_argument("--vars", default="core6+nice")
	parser.add_argument("--include-longwave-stage2", default="true")
	parser.add_argument("--derived-dtype", default="float32")
	parser.add_argument("--derived-format", default="parquet")
	parser.add_argument("--scratch-multiplier", type=float, default=1.5)
	parser.add_argument("--n-basins", type=int, default=9000)
	parser.add_argument(
		"--concurrency",
		type=int,
		default=16,
		help="Download worker count (default: 16). For MRMS AWS bulk downloads, current recommended setting is 32.",
	)
	parser.add_argument("--dry-run", action="store_true")
	parser.add_argument("--mrms-backend", default="aws", choices=["aws", "planetary"])
	parser.add_argument("--mrms-debug-listing", action="store_true")
	parser.add_argument("--range-mode", default="mrms_aligned", choices=["mrms_aligned", "source_full"])
	parser.add_argument("--make-preview", default="false", choices=["true", "false"])
	parser.add_argument(
		"--gfs-max-lead",
		type=int,
		default=24,
		help="Maximum GFS forecast lead hour per cycle (default: 24).",
	)
	parser.add_argument("--benchmark-concurrency", default="")
	return parser


def _parse_benchmark_concurrency(value: str) -> list[int]:
	if not value.strip():
		return []
	levels = [int(v.strip()) for v in value.split(",") if v.strip()]
	levels = [v for v in levels if v > 0]
	return sorted(set(levels))


def _parse_vars(value: str) -> list[str]:
	if value == "core6":
		return [
			"precip",
			"temp_2m",
			"humidity",
			"wind_10m",
			"surface_pressure",
			"shortwave_down",
		]
	if value == "core6+nice":
		return [
			"precip",
			"temp_2m",
			"humidity",
			"wind_10m",
			"surface_pressure",
			"shortwave_down",
			"soil_moisture_top",
			"swe_or_snow_depth",
		]
	return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	sample_start = _parse_datetime(args.sample_start, end_of_day=False)
	sample_end = _parse_datetime(args.sample_end, end_of_day=True)
	full_start = _parse_datetime(args.full_start, end_of_day=False)
	full_end = _parse_datetime(args.full_end, end_of_day=True)

	region = default_region()
	if args.region != region.name:
		raise ValueError("Only conus_bbox is supported in this sprint.")

	config = Config(
		sample_start=sample_start,
		sample_end=sample_end,
		full_start=full_start,
		full_end=full_end,
		region=region,
		out_dir=Path(args.out),
		report_dir=Path(args.report),
		variables=_parse_vars(args.vars),
		include_longwave_stage2=str(args.include_longwave_stage2).lower() == "true",
		derived_dtype=args.derived_dtype,
		derived_format=args.derived_format,
		scratch_multiplier=args.scratch_multiplier,
		n_basins=args.n_basins,
		concurrency=args.concurrency,
		dry_run=args.dry_run,
		mrms_backend=args.mrms_backend,
		mrms_debug_listing=args.mrms_debug_listing,
		range_mode=args.range_mode,
		make_preview=str(args.make_preview).lower() == "true",
		gfs_max_lead=args.gfs_max_lead,
	)

	benchmark_levels = _parse_benchmark_concurrency(args.benchmark_concurrency)
	if benchmark_levels:
		results, recommended = benchmark_mrms_aws_concurrency(config, benchmark_levels)
		print_benchmark_table(results)
		csv_path = write_benchmark_csv(results, config.report_dir)
		print(f"Benchmark CSV saved: {csv_path}")
		print(f"Recommended default concurrency based on best MB/s: {recommended}")
		return

	results, ratio_info = run_estimation(config)
	print_table(results)
	write_csv(results, config.report_dir)
	metadata = {
		"config": config_to_dict(config),
		"ratio_info": ratio_info,
	}
	write_json(results, config.report_dir, metadata)


if __name__ == "__main__":
	main()
