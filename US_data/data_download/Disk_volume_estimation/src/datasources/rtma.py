from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Iterable, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
from typing import Any

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region, RemoteObject, log_request
from src.derived_size import compute_derived_bytes


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RtmaAwsConfig:
	bucket: str = "noaa-rtma-pds"
	prefix_root: str = "rtma2p5."
	analysis_suffix: str = "2dvaranl_ndfd.grb2_wexp"


class RtmaAwsConusDataSource(DataSource):
	name = "rtma_conus_aws_2p5km"
	temporal_resolution = "hourly"

	def __init__(self, config: Optional[RtmaAwsConfig] = None, download_concurrency: int = 4) -> None:
		self.config = config or RtmaAwsConfig()
		self.download_concurrency = max(1, download_concurrency)
		self._available_start: Optional[datetime] = None
		self._available_end: Optional[datetime] = None
		self._s3_client_cached = None
		self._retry_count: int = 0
		self._warning_count: int = 0
		self._last_file_times: list[float] = []
		self._last_total_download_time: float = 0.0
		self._last_listing_time: float = 0.0
		self._last_full_file_sample_bytes: Optional[int] = None
		self._last_selected_sample_bytes: Optional[int] = None
		self._last_selected_accounting_mode: str = "unknown"

	def list_sample_objects(
		self,
		start: datetime,
		end: datetime,
		region: Region,
		variables: list[str],
		lead_times: Optional[Iterable[int]] = None,
	) -> list[RemoteObject]:
		if region.name != CONUS_BBOX.name:
			raise ValueError("RTMA AWS implementation currently supports only CONUS bbox.")

		listing_started = time.perf_counter()
		s3 = self._s3_client()
		self._discover_available_extent(s3)

		objects: list[RemoteObject] = []
		for day_prefix in self._iter_day_prefixes(start, end):
			for obj in self._list_objects_with_prefix(s3, day_prefix):
				key = obj["Key"]
				dt = self._parse_datetime_from_key(key)
				if dt is None or dt < start or dt > end:
					continue
				objects.append(
					RemoteObject(
						url=f"s3://{self.config.bucket}/{key}",
						key=key,
						datetime=dt,
						variables=variables,
						estimated_bytes=int(obj.get("Size") or 0),
					)
				)

		objects = sorted(objects, key=lambda obj: obj.datetime)
		listing_seconds = time.perf_counter() - listing_started
		self._last_listing_time = listing_seconds
		print(
			"RTMA AWS timing: "
			f"listing_time={listing_seconds:.2f}s, sample_objects={len(objects)}"
		)
		return objects

	def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
		s3 = self._s3_client()
		out_dir.mkdir(parents=True, exist_ok=True)
		if not objects:
			return []

		total_started = time.perf_counter()
		self._retry_count = 0
		self._warning_count = 0
		paths: list[Path] = []
		file_times: list[float] = []
		with ThreadPoolExecutor(max_workers=self.download_concurrency) as executor:
			futures = [
				executor.submit(self._download_one_object_timed, s3, out_dir, obj)
				for obj in objects
			]
			for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading RTMA AWS sample", unit="file"):
				out_path, file_seconds = future.result()
				paths.append(out_path)
				file_times.append(file_seconds)
				print(f"RTMA AWS timing: file={out_path.name} download_time={file_seconds:.3f}s")

		total_seconds = time.perf_counter() - total_started
		self._last_file_times = list(file_times)
		self._last_total_download_time = total_seconds
		avg_seconds = sum(file_times) / len(file_times) if file_times else 0.0
		print(
			"RTMA AWS timing summary: "
			f"files={len(paths)}, total_download_time={total_seconds:.2f}s, "
			f"avg_file_download_time={avg_seconds:.3f}s, concurrency={self.download_concurrency}"
		)
		return paths

	def measure_bytes(self, files: Iterable[Path]) -> int:
		return sum(path.stat().st_size for path in files)

	def measure_selected_variable_bytes(self, files: list[Path], objects: list[RemoteObject]) -> tuple[Optional[int], str]:
		"""Return selected-variable byte total and accounting mode.

		Modes:
		- server_side_range: based on .idx inventory and S3 byte-range accounting.
		- local_message_scan: based on local GRIB message lengths.
		- unavailable: could not determine selected bytes.
		"""
		targets = self._selected_targets(include_optional_tcdc=True)
		humidity_present_any = False
		server_total = 0
		server_counted = 0
		s3 = self._s3_client()

		for obj in objects:
			try:
				selected_bytes, humidity_present = self._measure_selected_from_index_and_ranges(s3, obj)
				server_total += selected_bytes
				server_counted += 1
				humidity_present_any = humidity_present_any or humidity_present
			except Exception:
				# Fall back to local message scanning for this/remaining files.
				server_counted = 0
				server_total = 0
				break

		if server_counted == len(objects) and objects:
			self._last_selected_sample_bytes = int(server_total)
			self._last_selected_accounting_mode = "server_side_range"
			return int(server_total), "server_side_range"

		local_total, humidity_present_any = self._measure_selected_from_local_files(files, targets)
		if local_total is None:
			self._last_selected_sample_bytes = None
			self._last_selected_accounting_mode = "unavailable"
			return None, "unavailable"

		self._last_selected_sample_bytes = int(local_total)
		self._last_selected_accounting_mode = "local_message_scan"
		return int(local_total), "local_message_scan"

	def estimate_raw_total(
		self,
		sample_bytes: int,
		sample_start: datetime,
		sample_end: datetime,
		full_start: datetime,
		full_end: datetime,
	) -> int:
		sample_hours = int((sample_end - sample_start).total_seconds() // 3600) + 1
		available_start = self._available_start or full_start
		available_end = self._available_end or full_end
		clipped_start = max(full_start, available_start)
		clipped_end = min(full_end, available_end)
		full_hours = int((clipped_end - clipped_start).total_seconds() // 3600) + 1
		if sample_hours <= 0:
			raise ValueError("Sample period must be at least one hour.")
		if full_hours <= 0:
			raise ValueError("Requested full range does not overlap available RTMA data.")
		return int(sample_bytes * (full_hours / sample_hours))

	def estimate_derived_total(self, spec: DerivedSpec) -> int:
		return compute_derived_bytes(spec)

	def estimate_peak_local(
		self,
		raw_total_bytes: int,
		scratch_multiplier: float,
		concurrency: int,
	) -> int:
		return int(raw_total_bytes * scratch_multiplier)

	def assumptions(self) -> dict:
		selected_mode = self._last_selected_accounting_mode
		return {
			"backend": "AWS noaa-rtma-pds",
			"region": "CONUS rtma2p5 archive",
			"cadence": "hourly",
			"variables_included": ["TMP", "SPFH_or_DPT", "UGRD", "VGRD", "PRES"],
			"optional_variable_if_available": ["TCDC"],
			"subsetting": "message-level byte accounting enabled",
			"selected_raw_bytes_mode": selected_mode,
			"selected_raw_bytes_mode_notes": {
				"server_side_range": "Computed from .idx offsets and S3 Range byte windows.",
				"local_message_scan": "Computed from local GRIB message totalLength values.",
				"unavailable": "Could not derive selected message bytes.",
			},
			"notes": "Using rtma2p5.tHHz.2dvaranl_ndfd.grb2_wexp analysis files.",
		}

	def _s3_client(self):
		if self._s3_client_cached is None:
			self._s3_client_cached = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
		return self._s3_client_cached

	@staticmethod
	def _log_retry_attempt(retry_state) -> None:
		instance = retry_state.args[0] if retry_state.args else None
		if isinstance(instance, RtmaAwsConusDataSource):
			instance._retry_count += 1
			instance._warning_count += 1
		attempt = retry_state.attempt_number
		next_sleep = retry_state.next_action.sleep if retry_state.next_action else None
		exc = retry_state.outcome.exception() if retry_state.outcome else None
		print(
			"RTMA AWS retry: "
			f"attempt={attempt}, backoff_s={next_sleep}, error={exc}"
		)

	@retry(
		stop=stop_after_attempt(3),
		wait=wait_exponential(multiplier=1, min=1, max=10),
		before_sleep=_log_retry_attempt,
	)
	def _download_s3_object(self, s3, bucket: str, key: str, out_path: Path) -> None:
		log_request(LOGGER, self.name, "s3.get_object", url=f"s3://{bucket}/{key}", params={"Bucket": bucket, "Key": key})
		response = s3.get_object(Bucket=bucket, Key=key)
		body = response["Body"]
		with out_path.open("wb") as f:
			while True:
				chunk = body.read(8 * 1024 * 1024)
				if not chunk:
					break
				f.write(chunk)

	def _download_one_object_timed(self, s3, out_dir: Path, obj: RemoteObject) -> tuple[Path, float]:
		bucket, key = self._parse_s3_url(obj.url)
		out_path = out_dir / obj.key
		out_path.parent.mkdir(parents=True, exist_ok=True)
		started = time.perf_counter()
		self._download_s3_object(s3, bucket, key, out_path)
		elapsed = time.perf_counter() - started
		return out_path, elapsed

	def _measure_selected_from_index_and_ranges(self, s3, obj: RemoteObject) -> tuple[int, bool]:
		bucket, key = self._parse_s3_url(obj.url)
		idx_key = f"{key}.idx"
		log_request(LOGGER, self.name, "s3.get_object", url=f"s3://{bucket}/{idx_key}", params={"Bucket": bucket, "Key": idx_key})
		idx_response = s3.get_object(Bucket=bucket, Key=idx_key)
		idx_text = idx_response["Body"].read().decode("utf-8", errors="replace")
		entries = self._parse_idx_entries(idx_text)
		if not entries:
			raise RuntimeError(f"Empty or unparseable idx for {key}")

		selector = self._build_selector(entries)
		ranges = self._selected_ranges(entries, selector)
		if not ranges:
			raise RuntimeError(f"No selected RTMA ranges from idx for {key}")

		selected_bytes = 0
		for start, end in ranges:
			range_header = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
			log_request(LOGGER, self.name, "s3.get_object", url=f"s3://{bucket}/{key}", params={"Bucket": bucket, "Key": key, "Range": range_header})
			resp = s3.get_object(Bucket=bucket, Key=key, Range=range_header)
			content_len = resp.get("ContentLength")
			if content_len is None:
				content_len = 0
				while True:
					chunk = resp["Body"].read(8 * 1024 * 1024)
					if not chunk:
						break
					content_len += len(chunk)
			selected_bytes += int(content_len)

		return selected_bytes, selector["has_spfh"]

	def _measure_selected_from_local_files(self, files: list[Path], targets: dict[str, set[str]]) -> tuple[Optional[int], bool]:
		try:
			from eccodes import codes_get, codes_grib_new_from_file, codes_release
		except Exception:
			return None, False

		total_selected = 0
		spfh_present_any = False

		for file_path in files:
			short_names: list[str] = []
			msg_lengths: list[int] = []
			with file_path.open("rb") as fh:
				while True:
					gid = codes_grib_new_from_file(fh)
					if gid is None:
						break
					try:
						short_name = str(codes_get(gid, "shortName") or "").lower()
						total_len = int(codes_get(gid, "totalLength"))
					except Exception:
						codes_release(gid)
						continue
					short_names.append(short_name)
					msg_lengths.append(total_len)
					codes_release(gid)

			has_spfh = any(sn in targets["SPFH"] for sn in short_names)
			spfh_present_any = spfh_present_any or has_spfh
			for sn, total_len in zip(short_names, msg_lengths):
				if self._is_selected_short_name(sn, targets, has_spfh):
					total_selected += total_len

		return total_selected, spfh_present_any

	@staticmethod
	def _selected_targets(include_optional_tcdc: bool) -> dict[str, set[str]]:
		targets = {
			"TMP": {"2t", "t2m", "tmp"},
			"SPFH": {"2sh", "sh2", "spfh", "q"},
			"DPT": {"2d", "d2m", "dpt"},
			"UGRD": {"10u", "u10", "ugrd"},
			"VGRD": {"10v", "v10", "vgrd"},
			"PRES": {"sp", "pres", "pressfc"},
			"TCDC": {"tcc", "tcdc"},
		}
		if not include_optional_tcdc:
			targets["TCDC"] = set()
		return targets

	def _build_selector(self, entries: list[dict[str, Any]]) -> dict[str, bool]:
		targets = self._selected_targets(include_optional_tcdc=True)
		has_spfh = any(entry.get("short_name", "") in targets["SPFH"] for entry in entries)
		return {"has_spfh": has_spfh}

	def _selected_ranges(self, entries: list[dict[str, Any]], selector: dict[str, bool]) -> list[tuple[int, Optional[int]]]:
		targets = self._selected_targets(include_optional_tcdc=True)
		has_spfh = selector["has_spfh"]
		ranges: list[tuple[int, Optional[int]]] = []
		for entry in entries:
			if not self._is_selected_short_name(entry["short_name"], targets, has_spfh):
				continue
			ranges.append((entry["offset"], entry["end_offset"]))
		return self._merge_ranges(ranges)

	@staticmethod
	def _is_selected_short_name(short_name: str, targets: dict[str, set[str]], has_spfh: bool) -> bool:
		sn = short_name.lower()
		if sn in targets["TMP"] or sn in targets["UGRD"] or sn in targets["VGRD"] or sn in targets["PRES"]:
			return True
		if sn in targets["TCDC"]:
			return True
		if has_spfh:
			return sn in targets["SPFH"]
		return sn in targets["DPT"]

	@staticmethod
	def _parse_idx_entries(idx_text: str) -> list[dict[str, Any]]:
		entries: list[dict[str, Any]] = []
		for raw_line in idx_text.splitlines():
			line = raw_line.strip()
			if not line:
				continue
			parts = line.split(":")
			if len(parts) < 4:
				continue
			try:
				offset = int(parts[1])
			except ValueError:
				continue
			short_name = parts[3].strip().lower()
			entries.append({"offset": offset, "short_name": short_name})

		entries.sort(key=lambda e: e["offset"])
		for idx, entry in enumerate(entries):
			next_offset = entries[idx + 1]["offset"] if idx + 1 < len(entries) else None
			entry["end_offset"] = (next_offset - 1) if next_offset is not None else None
		return entries

	@staticmethod
	def _merge_ranges(ranges: list[tuple[int, Optional[int]]]) -> list[tuple[int, Optional[int]]]:
		if not ranges:
			return []
		sorted_ranges = sorted(ranges, key=lambda r: r[0])
		merged: list[tuple[int, Optional[int]]] = []
		for start, end in sorted_ranges:
			if not merged:
				merged.append((start, end))
				continue
			prev_start, prev_end = merged[-1]
			if prev_end is None:
				continue
			if start <= prev_end + 1:
				if end is None or prev_end is None:
					merged[-1] = (prev_start, None)
				else:
					merged[-1] = (prev_start, max(prev_end, end))
			else:
				merged.append((start, end))
		return merged

	def _discover_available_extent(self, s3) -> None:
		log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{self.config.bucket}/{self.config.prefix_root}", params={"Bucket": self.config.bucket, "Prefix": self.config.prefix_root, "Delimiter": "/"})
		prefixes = list(self._iter_common_prefixes(s3, self.config.prefix_root))
		days: list[str] = []
		for prefix in prefixes:
			match = re.search(r"rtma2p5\.(\d{8})/$", prefix)
			if match:
				days.append(match.group(1))
		if not days:
			print("RTMA AWS availability: no day prefixes discovered.")
			return
		day_min = min(days)
		day_max = max(days)
		self._available_start = datetime.strptime(day_min, "%Y%m%d")
		self._available_end = datetime.strptime(day_max, "%Y%m%d") + timedelta(hours=23, minutes=59, seconds=59)
		print(
			"RTMA AWS availability: "
			f"earliest={self._available_start.isoformat()}Z, "
			f"latest={self._available_end.isoformat()}Z"
		)

	def _iter_day_prefixes(self, start: datetime, end: datetime) -> Iterable[str]:
		current = datetime(start.year, start.month, start.day)
		end_day = datetime(end.year, end.month, end.day)
		while current <= end_day:
			yield f"rtma2p5.{current.year:04d}{current.month:02d}{current.day:02d}/"
			current += timedelta(days=1)

	def _list_objects_with_prefix(self, s3, prefix: str) -> list[dict]:
		objects: list[dict] = []
		continuation: Optional[str] = None
		while True:
			kwargs = {
				"Bucket": self.config.bucket,
				"Prefix": prefix,
			}
			if continuation:
				kwargs["ContinuationToken"] = continuation
			log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{self.config.bucket}/{prefix}", params=kwargs)
			response = s3.list_objects_v2(**kwargs)
			for obj in response.get("Contents", []):
				key = obj.get("Key", "")
				if not key.endswith(self.config.analysis_suffix):
					continue
				objects.append(obj)
			if not response.get("IsTruncated"):
				break
			continuation = response.get("NextContinuationToken")
		return objects

	def _iter_common_prefixes(self, s3, prefix: str) -> Iterable[str]:
		paginator = s3.get_paginator("list_objects_v2")
		for page in paginator.paginate(
			Bucket=self.config.bucket,
			Prefix=prefix,
			Delimiter="/",
		):
			for entry in page.get("CommonPrefixes", []):
				yield entry.get("Prefix")

	@staticmethod
	def _parse_datetime_from_key(key: str) -> Optional[datetime]:
		match = re.search(r"\.t(\d{2})z\.2dvaranl_ndfd\.grb2_wexp$", key)
		day_match = re.search(r"rtma2p5\.(\d{8})/", key)
		if not day_match or not match:
			return None
		day = day_match.group(1)
		hour = match.group(1)
		stamp = f"{day}{hour}"
		try:
			return datetime.strptime(stamp, "%Y%m%d%H")
		except ValueError:
			return None

	@staticmethod
	def _parse_s3_url(url: str) -> tuple[str, str]:
		if not url.startswith("s3://"):
			raise ValueError("Expected s3:// URL for AWS RTMA object.")
		bucket_and_key = url[5:]
		bucket, key = bucket_and_key.split("/", 1)
		return bucket, key
