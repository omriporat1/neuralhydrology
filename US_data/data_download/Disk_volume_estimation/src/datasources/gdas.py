from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
import time
from typing import Any, Iterable, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region, RemoteObject, log_request, validate_conus_crop
from src.derived_size import compute_derived_bytes


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GdasAwsConfig:
    bucket: str = "noaa-gfs-bdp-pds"
    day_prefix_root: str = "gdas."
    product: str = "pgrb2.0p25"


class GdasAwsAntecedentDataSource(DataSource):
    """GDAS antecedent meteorology via NOAA AWS Open Data.

    Uses selected-variable byte-range extraction from .idx inventories for raw
    accounting. Spatial subset accounting is estimated locally for CONUS.
    """

    name = "gdas_conus_aws_0p25"
    temporal_resolution = "daily"
    LATENCY_DAYS = 1
    LOOKBACK_DAYS = 365
    CYCLE_HOURS = (0, 6, 12, 18)

    def __init__(self, config: Optional[GdasAwsConfig] = None, download_concurrency: int = 4) -> None:
        self.config = config or GdasAwsConfig()
        self.download_concurrency = max(1, int(download_concurrency))
        self._available_start: Optional[datetime] = None
        self._available_end: Optional[datetime] = None
        self._s3_client_cached = None
        self._retry_count: int = 0
        self._warning_count: int = 0

        self._last_selected_accounting_mode: str = "server_side_range"
        self._last_selected_conus_accounting_mode: str = "local_spatial_crop"
        self._last_required_data_start: Optional[datetime] = None
        self._last_required_data_end: Optional[datetime] = None
        self._validation_status: str = "VALID"
        self._validation_reason: Optional[str] = None

    def _mark_invalid(self, reason: str) -> None:
        self._validation_status = "INVALID"
        self._validation_reason = reason

    def list_sample_objects(
        self,
        start: datetime,
        end: datetime,
        region: Region,
        variables: list[str],
        lead_times: Optional[Iterable[int]] = None,
    ) -> list[RemoteObject]:
        del lead_times
        if region.name != CONUS_BBOX.name:
            raise ValueError("GDAS implementation currently supports only CONUS bbox.")

        s3 = self._s3_client()
        listing_started = time.perf_counter()
        self._discover_available_extent(s3)

        objects: list[RemoteObject] = []
        day = datetime(start.year, start.month, start.day)
        end_day = datetime(end.year, end.month, end.day)
        while day <= end_day:
            selected_cycle = self._find_best_cycle_for_day(s3, day)
            if selected_cycle is not None:
                key = self._object_key(day, selected_cycle)
                try:
                    meta = s3.head_object(Bucket=self.config.bucket, Key=key)
                    size = int(meta.get("ContentLength") or 0)
                except Exception:
                    size = None
                objects.append(
                    RemoteObject(
                        url=f"s3://{self.config.bucket}/{key}",
                        key=key,
                        datetime=day.replace(hour=selected_cycle),
                        variables=variables,
                        estimated_bytes=size,
                        lead_time_h=0,
                    )
                )
            day += timedelta(days=1)

        elapsed = time.perf_counter() - listing_started
        print(
            "GDAS AWS timing: "
            f"listing_time={elapsed:.2f}s, sample_objects={len(objects)}"
        )
        return objects

    def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
        s3 = self._s3_client()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not objects:
            return []

        total_started = time.perf_counter()
        paths: list[Path] = []
        with ThreadPoolExecutor(max_workers=self.download_concurrency) as executor:
            futures = [executor.submit(self._download_one_selected_subset, s3, out_dir, obj) for obj in objects]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading GDAS selected sample", unit="file"):
                paths.append(future.result())

        total_seconds = time.perf_counter() - total_started
        avg_seconds = total_seconds / len(paths) if paths else 0.0
        print(
            "GDAS AWS timing summary: "
            f"files={len(paths)}, total_download_time={total_seconds:.2f}s, "
            f"avg_file_download_time={avg_seconds:.3f}s, concurrency={self.download_concurrency}"
        )
        return paths

    def measure_bytes(self, files: Iterable[Path]) -> int:
        return int(sum(path.stat().st_size for path in files))

    def measure_full_file_sample_bytes(self, objects: list[RemoteObject]) -> Optional[int]:
        if not objects:
            return 0
        if any(obj.estimated_bytes is None for obj in objects):
            return None
        return int(sum(obj.estimated_bytes or 0 for obj in objects))

    def measure_selected_variable_bytes(self, files: list[Path], objects: list[RemoteObject]) -> tuple[Optional[int], str]:
        del objects
        self._last_selected_accounting_mode = "server_side_range"
        return self.measure_bytes(files), self._last_selected_accounting_mode

    def measure_selected_conus_bytes(self, files: list[Path], region: Region = CONUS_BBOX) -> tuple[Optional[int], str]:
        try:
            import cfgrib
        except Exception:
            self._last_selected_conus_accounting_mode = "unavailable"
            return None, "unavailable"

        total_bytes = 0
        for file_path in files:
            datasets = cfgrib.open_datasets(str(file_path), backend_kwargs={"indexpath": ""})
            try:
                for ds in datasets:
                    for var_name in ds.data_vars:
                        da = ds[var_name]
                        arr = np.asarray(da.values)
                        arr = np.squeeze(arr)
                        if arr.ndim != 2:
                            continue
                        crop = validate_conus_crop(
                            arr,
                            da.coords,
                            region.bbox,
                            source_name=self.name,
                            crop_kind="GDAS CONUS crop",
                            logger=LOGGER,
                            mark_invalid=self._mark_invalid,
                        )
                        if crop.cropped_array.size == 0:
                            continue
                        total_bytes += int(crop.cropped_array.size * crop.cropped_array.dtype.itemsize)
            finally:
                for ds in datasets:
                    ds.close()

        self._last_selected_conus_accounting_mode = "local_spatial_crop"
        return int(total_bytes), self._last_selected_conus_accounting_mode

    def estimate_raw_total(
        self,
        sample_bytes: int,
        sample_start: datetime,
        sample_end: datetime,
        full_start: datetime,
        full_end: datetime,
    ) -> int:
        sample_days = (sample_end.date() - sample_start.date()).days + 1
        full_days = (full_end.date() - full_start.date()).days + 1
        if sample_days <= 0:
            raise ValueError("GDAS sample period must include at least one day.")
        if full_days <= 0:
            raise ValueError("GDAS required archive range is empty.")
        return int(sample_bytes * (full_days / sample_days))

    def estimate_derived_total(self, spec: DerivedSpec) -> int:
        return compute_derived_bytes(spec)

    def estimate_peak_local(
        self,
        raw_total_bytes: int,
        scratch_multiplier: float,
        concurrency: int,
    ) -> int:
        del concurrency
        return int(raw_total_bytes * scratch_multiplier)

    def assumptions(self) -> dict:
        return {
            "backend": "AWS noaa-gfs-bdp-pds (GDAS path)",
            "region": "CONUS accounting; selected-variable extraction server-side, CONUS crop local",
            "raw_cadence": "6-hourly GDAS analyses (daily sampling uses 00z f000)",
            "derived_cadence": "daily basin-average",
            "variables_included": [
                "TMP_2m",
                "RH_2m",
                "UGRD_10m",
                "VGRD_10m",
                "PRMSL_msl",
                "DSWRF_surface",
                "PRATE_surface",
            ],
            "latency_days": self.LATENCY_DAYS,
            "lookback_days": self.LOOKBACK_DAYS,
            "server_side_spatial_subset": False,
            "selected_raw_bytes_mode": self._last_selected_accounting_mode,
            "selected_conus_raw_bytes_mode": self._last_selected_conus_accounting_mode,
            "validation_status": self._validation_status,
            "validation_reason": self._validation_reason,
            "required_historical_archive_window": {
                "start": self._last_required_data_start.isoformat() if self._last_required_data_start else None,
                "end": self._last_required_data_end.isoformat() if self._last_required_data_end else None,
            },
            "notes": "Selected-variable bytes are computed using .idx + S3 Range; CONUS crop is local for accounting.",
        }

    def set_required_window(self, start: Optional[datetime], end: Optional[datetime]) -> None:
        self._last_required_data_start = start
        self._last_required_data_end = end

    def _s3_client(self):
        if self._s3_client_cached is None:
            self._s3_client_cached = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
        return self._s3_client_cached

    def _discover_available_extent(self, s3) -> None:
        log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{self.config.bucket}/{self.config.day_prefix_root}", params={"Bucket": self.config.bucket, "Prefix": self.config.day_prefix_root, "Delimiter": "/"})
        prefixes = list(self._iter_common_prefixes(s3, self.config.day_prefix_root))
        days: list[str] = []
        for prefix in prefixes:
            match = re.search(r"gdas\.(\d{8})/$", prefix)
            if match:
                days.append(match.group(1))
        if not days:
            print("GDAS AWS availability: no day prefixes discovered.")
            return
        day_min = min(days)
        day_max = max(days)
        self._available_start = datetime.strptime(day_min, "%Y%m%d")
        self._available_end = datetime.strptime(day_max, "%Y%m%d") + timedelta(hours=23, minutes=59, seconds=59)
        print(
            "GDAS AWS availability: "
            f"earliest={self._available_start.isoformat()}Z, "
            f"latest={self._available_end.isoformat()}Z"
        )

    def _iter_common_prefixes(self, s3, prefix: str) -> Iterable[str]:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix, Delimiter="/"):
            log_request(LOGGER, self.name, "s3.list_objects_v2", url=f"s3://{self.config.bucket}/{prefix}", params={"Bucket": self.config.bucket, "Prefix": prefix, "Delimiter": "/"})
            for entry in page.get("CommonPrefixes", []):
                yield entry.get("Prefix")

    def _find_best_cycle_for_day(self, s3, day: datetime) -> Optional[int]:
        for cycle_h in self.CYCLE_HOURS:
            key = self._object_key(day, cycle_h)
            try:
                log_request(LOGGER, self.name, "s3.head_object", url=f"s3://{self.config.bucket}/{key}", params={"Bucket": self.config.bucket, "Key": key})
                s3.head_object(Bucket=self.config.bucket, Key=key)
                return cycle_h
            except Exception:
                continue
        return None

    def _object_key(self, day: datetime, cycle_h: int) -> str:
        ymd = f"{day.year:04d}{day.month:02d}{day.day:02d}"
        return f"gdas.{ymd}/{cycle_h:02d}/atmos/gdas.t{cycle_h:02d}z.{self.config.product}.f000"

    @staticmethod
    def _parse_s3_url(url: str) -> tuple[str, str]:
        if not url.startswith("s3://"):
            raise ValueError("Expected s3:// URL for GDAS AWS object.")
        bucket_and_key = url[5:]
        bucket, key = bucket_and_key.split("/", 1)
        return bucket, key

    @staticmethod
    def _log_retry_attempt(retry_state) -> None:
        instance = retry_state.args[0] if retry_state.args else None
        if isinstance(instance, GdasAwsAntecedentDataSource):
            instance._retry_count += 1
            instance._warning_count += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=_log_retry_attempt,
    )
    def _download_s3_range(self, s3, bucket: str, key: str, byte_range: str) -> bytes:
        log_request(LOGGER, self.name, "s3.get_object", url=f"s3://{bucket}/{key}", params={"Bucket": bucket, "Key": key, "Range": byte_range})
        resp = s3.get_object(Bucket=bucket, Key=key, Range=byte_range)
        return resp["Body"].read()

    def _download_one_selected_subset(self, s3, out_dir: Path, obj: RemoteObject) -> Path:
        bucket, key = self._parse_s3_url(obj.url)
        idx_key = f"{key}.idx"
        log_request(LOGGER, self.name, "s3.get_object", url=f"s3://{bucket}/{idx_key}", params={"Bucket": bucket, "Key": idx_key})
        idx_resp = s3.get_object(Bucket=bucket, Key=idx_key)
        idx_text = idx_resp["Body"].read().decode("utf-8", errors="replace")

        entries = self._parse_idx_entries(idx_text)
        ranges = self._selected_ranges(entries)

        rel_dir = Path(obj.key).parent
        out_path = out_dir / rel_dir / "selected_f000.grib2"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("wb") as f:
            for start, end in ranges:
                hdr = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
                chunk = self._download_s3_range(s3, bucket, key, hdr)
                f.write(chunk)

        return out_path

    def _selected_ranges(self, entries: list[dict[str, Any]]) -> list[tuple[int, Optional[int]]]:
        selected: list[tuple[int, Optional[int]]] = []
        for entry in entries:
            var = entry["var"]
            level = entry["level"]
            if self._is_selected_message(var, level):
                selected.append((entry["offset"], entry["end_offset"]))

        merged = self._merge_ranges(selected)
        if not merged:
            raise RuntimeError("No selected GDAS messages found from idx.")
        return merged

    @staticmethod
    def _is_selected_message(var: str, level: str) -> bool:
        v = var.upper()
        l = level.lower()
        if v == "TMP" and l == "2 m above ground":
            return True
        if v == "RH" and l == "2 m above ground":
            return True
        if v == "UGRD" and l == "10 m above ground":
            return True
        if v == "VGRD" and l == "10 m above ground":
            return True
        if v == "PRMSL" and l == "mean sea level":
            return True
        if v == "DSWRF" and l == "surface":
            return True
        if v == "PRATE" and l == "surface":
            return True
        return False

    @staticmethod
    def _parse_idx_entries(idx_text: str) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for raw_line in idx_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < 5:
                continue
            try:
                offset = int(parts[1])
            except ValueError:
                continue
            var = parts[3].strip()
            level = parts[4].strip()
            entries.append({"offset": offset, "var": var, "level": level})

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
                if end is None:
                    merged[-1] = (prev_start, None)
                else:
                    merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _crop_array_to_bbox(
        arr: np.ndarray,
        coords,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        lon_min, lat_min, lon_max, lat_max = bbox
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
        mask = (
            (lat_grid >= lat_min)
            & (lat_grid <= lat_max)
            & (lon_grid >= lon_min)
            & (lon_grid <= lon_max)
        )
        if mask.shape != arr.shape:
            return arr
        return arr[mask]