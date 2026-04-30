from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import time

import boto3
import httpx
import planetary_computer
from pystac_client import Client
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region, RemoteObject, log_request
from src.derived_size import compute_derived_bytes


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MrmsConfig:
	stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1"
	collection: str = "noaa-mrms-qpe-1h-pass1"


class MrmsDataSource(DataSource):
	name = "mrms_qpe_1h_pass1"
	temporal_resolution = "hourly"

	def __init__(self, config: Optional[MrmsConfig] = None) -> None:
		self.config = config or MrmsConfig()

	def list_sample_objects(
		self,
		start: datetime,
		end: datetime,
		region: Region,
		variables: list[str],
		lead_times: Optional[Iterable[int]] = None,
	) -> list[RemoteObject]:
		if region.name != CONUS_BBOX.name:
			raise ValueError("MRMS implementation currently supports only CONUS bbox.")

		client = Client.open(self.config.stac_url)
		start_iso = start.replace(microsecond=0).isoformat() + "Z"
		end_iso = end.replace(microsecond=0).isoformat() + "Z"
		datetime_range = f"{start_iso}/{end_iso}"
		log_request(
			LOGGER,
			self.name,
			"stac.search",
			url=self.config.stac_url,
			params={"collections": [self.config.collection], "bbox": list(region.bbox), "datetime": datetime_range},
		)
		search = client.search(
			collections=[self.config.collection],
			bbox=list(region.bbox),
			datetime=datetime_range,
		)

		objects: list[RemoteObject] = []
		for item in search.items():
			asset_key = self._pick_asset_key(item.assets)
			asset = item.assets[asset_key]
			href = planetary_computer.sign(asset.href)
			file_size = asset.extra_fields.get("file:size")
			objects.append(
				RemoteObject(
					url=href,
					key=f"{item.id}_{asset_key}.grib2",
					datetime=item.datetime,
					variables=variables,
					estimated_bytes=int(file_size) if file_size is not None else None,
				)
			)

		return sorted(objects, key=lambda obj: obj.datetime)

	@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
	def _download_one(self, url: str, out_path: Path) -> None:
		out_path.parent.mkdir(parents=True, exist_ok=True)
		log_request(LOGGER, self.name, "http.get", url=url, params={"timeout_s": 60})
		with httpx.stream("GET", url, timeout=60) as response:
			response.raise_for_status()
			with out_path.open("wb") as f:
				for chunk in response.iter_bytes():
					f.write(chunk)

	def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
		out_dir.mkdir(parents=True, exist_ok=True)
		paths: list[Path] = []
		for obj in tqdm(objects, desc="Downloading MRMS sample", unit="file"):
			out_path = out_dir / obj.key
			self._download_one(obj.url, out_path)
			paths.append(out_path)
		return paths

	def measure_bytes(self, files: Iterable[Path]) -> int:
		return sum(path.stat().st_size for path in files)

	def estimate_raw_total(
		self,
		sample_bytes: int,
		sample_start: datetime,
		sample_end: datetime,
		full_start: datetime,
		full_end: datetime,
	) -> int:
		sample_hours = int((sample_end - sample_start).total_seconds() // 3600) + 1
		full_hours = int((full_end - full_start).total_seconds() // 3600) + 1
		if sample_hours <= 0:
			raise ValueError("Sample period must be at least one hour.")
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
		return {
			"region": "CONUS bbox only",
			"cadence": "hourly",
			"variables": ["precip"],
			"server_side_subset": False,
			"notes": "STAC GRIB2 assets downloaded in full; no byte-range subsetting.",
		}

	@staticmethod
	def _pick_asset_key(assets: dict) -> str:
		for key, asset in assets.items():
			media_type = getattr(asset, "media_type", "") or ""
			if "grib" in media_type.lower():
				return key
		return next(iter(assets.keys()))


@dataclass(frozen=True)
class MrmsAwsConfig:
	bucket: str = "noaa-mrms-pds"
	conus_prefix: str = "CONUS/"
	product_match: str = "MultiSensor_QPE_01H_Pass1"


class MrmsAwsQpe1hPass1(DataSource):
	name = "mrms_qpe_1h_pass1"
	temporal_resolution = "hourly"

	def __init__(
		self,
		config: Optional[MrmsAwsConfig] = None,
		debug_listing: bool = False,
		download_concurrency: int = 4,
	) -> None:
		self.config = config or MrmsAwsConfig()
		self.debug_listing = debug_listing
		self.download_concurrency = max(1, download_concurrency)
		self._product_prefix: Optional[str] = None
		self._available_start: Optional[datetime] = None
		self._available_end: Optional[datetime] = None
		self._date_prefix_format: Optional[str] = None
		self._s3_client_cached = None
		self._retry_count: int = 0
		self._warning_count: int = 0
		self._last_file_times: list[float] = []
		self._last_total_download_time: float = 0.0
		self._last_listing_time: float = 0.0

	def list_sample_objects(
		self,
		start: datetime,
		end: datetime,
		region: Region,
		variables: list[str],
		lead_times: Optional[Iterable[int]] = None,
	) -> list[RemoteObject]:
		if region.name != CONUS_BBOX.name:
			raise ValueError("MRMS AWS implementation currently supports only CONUS bbox.")

		listing_started = time.perf_counter()
		s3 = self._s3_client()
		try:
			product_prefix = self._discover_product_prefix(s3)
		except RuntimeError:
			if self.debug_listing:
				self._debug_listings(s3, None)
			raise
		self._discover_available_extent(s3, product_prefix)
		if self.debug_listing:
			self._debug_listings(s3, product_prefix)

		objects: list[RemoteObject] = []
		for prefix in self._iter_candidate_prefixes(s3, product_prefix, start, end):
			for obj in self._list_objects_with_prefix(s3, prefix):
				key = obj["Key"]
				dt = self._parse_timestamp_from_key(key)
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
			"MRMS AWS timing: "
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
			for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading MRMS AWS sample", unit="file"):
				out_path, file_seconds = future.result()
				paths.append(out_path)
				file_times.append(file_seconds)
				print(f"MRMS AWS timing: file={out_path.name} download_time={file_seconds:.3f}s")

		total_seconds = time.perf_counter() - total_started
		self._last_file_times = list(file_times)
		self._last_total_download_time = total_seconds
		avg_seconds = sum(file_times) / len(file_times) if file_times else 0.0
		print(
			"MRMS AWS timing summary: "
			f"files={len(paths)}, total_download_time={total_seconds:.2f}s, "
			f"avg_file_download_time={avg_seconds:.3f}s, concurrency={self.download_concurrency}"
		)
		return paths

	def measure_bytes(self, files: Iterable[Path]) -> int:
		return sum(path.stat().st_size for path in files)

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
			raise ValueError("Requested full range does not overlap available MRMS data.")
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
		return {
			"region": "CONUS bbox only",
			"cadence": "hourly",
			"variables": ["precip"],
			"server_side_subset": True,
			"notes": "AWS MRMS CONUS product; listing filtered by timestamp in keys.",
		}

	def _s3_client(self):
		if self._s3_client_cached is None:
			self._s3_client_cached = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
		return self._s3_client_cached

	@staticmethod
	def _log_retry_attempt(retry_state) -> None:
		instance = retry_state.args[0] if retry_state.args else None
		if isinstance(instance, MrmsAwsQpe1hPass1):
			instance._retry_count += 1
			instance._warning_count += 1
		attempt = retry_state.attempt_number
		next_sleep = retry_state.next_action.sleep if retry_state.next_action else None
		exc = retry_state.outcome.exception() if retry_state.outcome else None
		print(
			"MRMS AWS retry: "
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

	def _discover_product_prefix(self, s3) -> str:
		if self._product_prefix is not None:
			return self._product_prefix
		log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{self.config.bucket}/{self.config.conus_prefix}", params={"Bucket": self.config.bucket, "Prefix": self.config.conus_prefix, "Delimiter": "/"})
		response = s3.list_objects_v2(
			Bucket=self.config.bucket,
			Prefix=self.config.conus_prefix,
			Delimiter="/",
		)
		prefixes = [p["Prefix"] for p in response.get("CommonPrefixes", [])]
		for prefix in prefixes:
			if self.config.product_match in prefix:
				self._product_prefix = prefix
				break
		if self._product_prefix is None:
			for prefix in prefixes:
				child = s3.list_objects_v2(
					Bucket=self.config.bucket,
					Prefix=prefix,
					Delimiter="/",
				)
				child_prefixes = [p["Prefix"] for p in child.get("CommonPrefixes", [])]
				for child_prefix in child_prefixes:
					if self.config.product_match in child_prefix:
						self._product_prefix = child_prefix
						break
				if self._product_prefix is not None:
					break
		if self._product_prefix is None:
			raise RuntimeError("Could not discover MRMS product prefix under CONUS/.")
		return self._product_prefix

	def _discover_available_extent(self, s3, product_prefix: str) -> None:
		log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{self.config.bucket}/{product_prefix}", params={"Bucket": self.config.bucket, "Prefix": product_prefix, "Delimiter": "/"})
		day_min, day_max, year_min, year_max = self._scan_date_prefixes(s3, product_prefix)
		if day_min and day_max:
			self._date_prefix_format = "day"
			self._available_start = datetime.strptime(day_min, "%Y%m%d")
			self._available_end = datetime.strptime(day_max, "%Y%m%d") + timedelta(hours=23, minutes=59, seconds=59)
			print(
				"MRMS AWS availability: "
				f"earliest={self._available_start.isoformat()}Z, "
				f"latest={self._available_end.isoformat()}Z"
			)
			return
		if year_min is not None and year_max is not None:
			self._date_prefix_format = "year"
			self._available_start = datetime(year_min, 1, 1, 0, 0, 0)
			self._available_end = datetime(year_max, 12, 31, 23, 59, 59)
			print(
				"MRMS AWS availability: "
				f"earliest={self._available_start.isoformat()}Z, "
				f"latest={self._available_end.isoformat()}Z"
			)
			return
		inferred = self._infer_extent_from_keys(s3, product_prefix)
		if inferred is None:
			print("MRMS AWS availability: no year prefixes discovered.")
			return
		self._available_start, self._available_end = inferred
		print(
			"MRMS AWS availability: "
			f"earliest={self._available_start.isoformat()}Z, "
			f"latest={self._available_end.isoformat()}Z"
		)

	def _scan_date_prefixes(
		self, s3, product_prefix: str
	) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
		day_min: Optional[str] = None
		day_max: Optional[str] = None
		year_min: Optional[int] = None
		year_max: Optional[int] = None
		for prefix in self._iter_common_prefixes(s3, product_prefix):
			day_match = re.search(r"/(\d{8})/$", prefix)
			if day_match:
				day = day_match.group(1)
				day_min = day if day_min is None else min(day_min, day)
				day_max = day if day_max is None else max(day_max, day)
				continue
			year_match = re.search(r"/(\d{4})/$", prefix)
			if year_match:
				year = int(year_match.group(1))
				year_min = year if year_min is None else min(year_min, year)
				year_max = year if year_max is None else max(year_max, year)
		return day_min, day_max, year_min, year_max

	def _iter_candidate_prefixes(
		self,
		s3,
		product_prefix: str,
		start: datetime,
		end: datetime,
	) -> Iterable[str]:
		if self._date_prefix_format == "day":
			current = datetime(start.year, start.month, start.day)
			end_day = datetime(end.year, end.month, end.day)
			while current <= end_day:
				yield f"{product_prefix}{current.year:04d}{current.month:02d}{current.day:02d}/"
				current += timedelta(days=1)
			return
		year_prefixes = list(self._iter_common_prefixes(s3, product_prefix))
		year_prefixes = [p for p in year_prefixes if re.search(r"/\d{4}/$", p)]
		if not year_prefixes:
			yield product_prefix
			return
		has_month_prefix = False
		if year_prefixes:
			probe = year_prefixes[0]
			response = s3.list_objects_v2(
				Bucket=self.config.bucket,
				Prefix=probe,
				Delimiter="/",
			)
			month_prefixes = response.get("CommonPrefixes", [])
			has_month_prefix = len(month_prefixes) > 0

		if not has_month_prefix:
			years = sorted({dt.year for dt in self._iter_hours(start, end)})
			for year in years:
				yield f"{product_prefix}{year}/"
			return

		current = datetime(start.year, start.month, start.day)
		end_day = datetime(end.year, end.month, end.day)
		while current <= end_day:
			yield f"{product_prefix}{current.year}/{current.month:02d}/{current.day:02d}/"
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
			objects.extend(response.get("Contents", []))
			if not response.get("IsTruncated"):
				break
			continuation = response.get("NextContinuationToken")
		return objects

	def _list_objects_with_prefix_limited(self, s3, prefix: str, max_keys: int) -> list[dict]:
		log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{self.config.bucket}/{prefix}", params={"Bucket": self.config.bucket, "Prefix": prefix, "MaxKeys": max_keys})
		response = s3.list_objects_v2(
			Bucket=self.config.bucket,
			Prefix=prefix,
			MaxKeys=max_keys,
		)
		return response.get("Contents", [])

	def _infer_extent_from_keys(self, s3, product_prefix: str) -> Optional[tuple[datetime, datetime]]:
		objects = self._list_objects_with_prefix_limited(s3, product_prefix, max_keys=200)
		timestamps: list[datetime] = []
		for obj in objects:
			dt = self._parse_timestamp_from_key(obj.get("Key", ""))
			if dt:
				timestamps.append(dt)
		if not timestamps:
			return None
		return min(timestamps), max(timestamps)

	def _debug_listings(self, s3, product_prefix: Optional[str]) -> None:
		bucket = self.config.bucket
		print("MRMS AWS debug listing: top-level prefixes")
		log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{bucket}/", params={"Bucket": bucket, "Delimiter": "/"})
		top = s3.list_objects_v2(Bucket=bucket, Delimiter="/")
		print([p["Prefix"] for p in top.get("CommonPrefixes", [])])
		print("MRMS AWS debug listing: CONUS/ prefixes")
		log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{bucket}/{self.config.conus_prefix}", params={"Bucket": bucket, "Prefix": self.config.conus_prefix, "Delimiter": "/"})
		conus = s3.list_objects_v2(Bucket=bucket, Prefix=self.config.conus_prefix, Delimiter="/")
		conus_prefixes = [p["Prefix"] for p in conus.get("CommonPrefixes", [])]
		print(conus_prefixes)
		best = product_prefix
		candidate = None
		for prefix in conus_prefixes:
			if self.config.product_match in prefix:
				candidate = prefix
				break
		if candidate is None and conus_prefixes:
			log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{bucket}/{conus_prefixes[0]}", params={"Bucket": bucket, "Prefix": conus_prefixes[0], "Delimiter": "/"})
			child = s3.list_objects_v2(Bucket=bucket, Prefix=conus_prefixes[0], Delimiter="/")
			child_prefixes = [p["Prefix"] for p in child.get("CommonPrefixes", [])]
			print("MRMS AWS debug listing: next-level prefixes under", conus_prefixes[0])
			print(child_prefixes)
			for child_prefix in child_prefixes:
				if self.config.product_match in child_prefix:
					candidate = child_prefix
					break
		best = best or candidate
		if best:
			log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{bucket}/{best}", params={"Bucket": bucket, "Prefix": best, "Delimiter": "/"})
			child = s3.list_objects_v2(Bucket=bucket, Prefix=best, Delimiter="/")
			child_prefixes = [p["Prefix"] for p in child.get("CommonPrefixes", [])]
			if child_prefixes:
				print("MRMS AWS debug listing: next-level prefixes under", best)
				print(child_prefixes)
			print("MRMS AWS debug listing: sample keys")
			for obj in self._list_objects_with_prefix_limited(s3, best, max_keys=50):
				print(obj.get("Key"), obj.get("Size"))

	def _iter_common_prefixes(self, s3, prefix: str) -> Iterable[str]:
		paginator = s3.get_paginator("list_objects_v2")
		for page in paginator.paginate(
			Bucket=self.config.bucket,
			Prefix=prefix,
			Delimiter="/",
		):
			for entry in page.get("CommonPrefixes", []):
				yield entry.get("Prefix")

	def _parse_timestamp_from_key(self, key: str) -> Optional[datetime]:
		match = re.search(r"(\d{8})-(\d{6})", key)
		if not match:
			return None
		stamp = match.group(1) + match.group(2)
		try:
			return datetime.strptime(stamp, "%Y%m%d%H%M%S")
		except ValueError:
			return None

	@staticmethod
	def _parse_s3_url(url: str) -> tuple[str, str]:
		if not url.startswith("s3://"):
			raise ValueError("Expected s3:// URL for AWS MRMS object.")
		bucket_and_key = url[5:]
		bucket, key = bucket_and_key.split("/", 1)
		return bucket, key

	@staticmethod
	def _iter_hours(start: datetime, end: datetime) -> Iterable[datetime]:
		current = start
		while current <= end:
			yield current
			current += timedelta(hours=1)
