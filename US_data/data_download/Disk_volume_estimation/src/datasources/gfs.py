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

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region, RemoteObject, validate_conus_crop
from src.derived_size import compute_derived_bytes


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GfsAwsConfig:
    bucket: str = "noaa-gfs-bdp-pds"
    product: str = "pgrb2.0p25"
    day_prefix_root: str = "gfs."


class GfsAwsConusDataSource(DataSource):
    """GFS forecast forcing via NOAA AWS Open Data.

    We use forecast files on S3 and selected-variable byte-range extraction using each
    file's .idx inventory. This provides message-level (variable/level) accounting.
    Spatial subsetting (CONUS window) is not applied at S3-object level in this path.
    """

    name = "gfs_conus_aws_0p25"
    temporal_resolution = "hourly"
    CYCLE_HOURS = (0, 6, 12, 18)

    def __init__(self, max_lead_h: int = 24, config: Optional[GfsAwsConfig] = None, download_concurrency: int = 4) -> None:
        self.max_lead_h = max(0, int(max_lead_h))
        self.config = config or GfsAwsConfig()
        self.download_concurrency = max(1, int(download_concurrency))
        self._available_start: Optional[datetime] = None
        self._available_end: Optional[datetime] = None
        self._s3_client_cached = None
        self._retry_count: int = 0
        self._warning_count: int = 0

        self._last_full_file_sample_bytes: Optional[int] = None
        self._last_selected_sample_bytes: Optional[int] = None
        self._last_selected_conus_sample_bytes: Optional[int] = None
        self._last_selected_accounting_mode: str = "server_side_range"
        self._last_conus_accounting_mode: str = "local_spatial_crop"
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
        if region.name != CONUS_BBOX.name:
            raise ValueError("GFS implementation currently supports only CONUS bbox.")

        s3 = self._s3_client()
        listing_started = time.perf_counter()
        self._discover_available_extent(s3)

        leads = list(lead_times) if lead_times is not None else list(range(0, self.max_lead_h + 1))
        cycle_times = list(self._iter_cycle_datetimes(start, end))

        objects: list[RemoteObject] = []
        for cycle_dt in cycle_times:
            size_by_lead = self._list_cycle_sizes(s3, cycle_dt)
            for lead in leads:
                key = self._object_key(cycle_dt, lead)
                size = size_by_lead.get(lead)
                if size is None:
                    continue
                objects.append(
                    RemoteObject(
                        url=f"s3://{self.config.bucket}/{key}",
                        key=key,
                        datetime=cycle_dt,
                        variables=variables,
                        estimated_bytes=size,
                        lead_time_h=lead,
                    )
                )

        objects = sorted(objects, key=lambda obj: (obj.datetime, obj.lead_time_h or 0))
        self._last_full_file_sample_bytes = int(sum(obj.estimated_bytes or 0 for obj in objects))

        elapsed = time.perf_counter() - listing_started
        print(
            "GFS AWS timing: "
            f"listing_time={elapsed:.2f}s, cycles={len(cycle_times)}, sample_objects={len(objects)}, "
            f"max_lead_h={self.max_lead_h}"
        )
        return objects

    def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
        s3 = self._s3_client()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not objects:
            self._last_selected_sample_bytes = 0
            return []

        total_started = time.perf_counter()
        self._retry_count = 0
        self._warning_count = 0

        paths: list[Path] = []
        with ThreadPoolExecutor(max_workers=self.download_concurrency) as executor:
            futures = [
                executor.submit(self._download_one_selected_subset, s3, out_dir, obj)
                for obj in objects
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading GFS selected sample", unit="file"):
                out_path = future.result()
                paths.append(out_path)

        total_seconds = time.perf_counter() - total_started
        self._last_selected_sample_bytes = int(sum(path.stat().st_size for path in paths))
        avg_per_file = total_seconds / len(paths) if paths else 0.0
        print(
            "GFS AWS timing summary: "
            f"files={len(paths)}, total_download_time={total_seconds:.2f}s, "
            f"avg_file_time={avg_per_file:.3f}s, concurrency={self.download_concurrency}"
        )
        return paths

    def measure_bytes(self, files: Iterable[Path]) -> int:
        return int(sum(path.stat().st_size for path in files))

    def measure_selected_variable_bytes(self, files: list[Path], objects: list[RemoteObject]) -> tuple[Optional[int], str]:
        del objects
        selected = self.measure_bytes(files)
        self._last_selected_sample_bytes = selected
        self._last_selected_accounting_mode = "server_side_range"
        return selected, "server_side_range"

    def measure_selected_conus_bytes(self, files: list[Path], region: Region = CONUS_BBOX) -> tuple[Optional[int], str]:
        try:
            import cfgrib
        except Exception:
            self._last_selected_conus_sample_bytes = None
            self._last_conus_accounting_mode = "unavailable"
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
                            crop_kind="GFS CONUS crop",
                            logger=LOGGER,
                            mark_invalid=self._mark_invalid,
                        )
                        if crop.cropped_array.size == 0:
                            continue
                        total_bytes += int(crop.cropped_array.size * crop.cropped_array.dtype.itemsize)
            finally:
                for ds in datasets:
                    ds.close()

        self._last_selected_conus_sample_bytes = int(total_bytes)
        self._last_conus_accounting_mode = "local_spatial_crop"
        return int(total_bytes), "local_spatial_crop"

    def estimate_raw_total(
        self,
        sample_bytes: int,
        sample_start: datetime,
        sample_end: datetime,
        full_start: datetime,
        full_end: datetime,
    ) -> int:
        sample_cycles = self._count_cycles(sample_start, sample_end)
        if sample_cycles <= 0:
            raise ValueError("GFS sample must include at least one 6-hour cycle.")

        effective_days = (full_end.date() - full_start.date()).days + 1
        if effective_days <= 0:
            raise ValueError("Requested GFS effective range must include at least one day.")

        bytes_per_cycle = float(sample_bytes) / float(sample_cycles)
        return int(bytes_per_cycle * 4.0 * float(effective_days))

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
            "backend": "AWS noaa-gfs-bdp-pds",
            "region": "CONUS forcing workflow",
            "cadence": "forecast cycles 00/06/12/18 with hourly leads",
            "max_lead_h": self.max_lead_h,
            "variables_included": [
                "PRATE_surface",
                "TMP_2m",
                "RH_2m",
                "UGRD_10m",
                "VGRD_10m",
                "PRMSL_msl",
                "DSWRF_surface",
            ],
            "subsetting": "message-level selected-variable extraction via .idx + S3 Range",
            "selected_raw_bytes_mode": self._last_selected_accounting_mode,
            "selected_conus_raw_bytes_mode": self._last_conus_accounting_mode,
            "validation_status": self._validation_status,
            "validation_reason": self._validation_reason,
            "spatial_crop": {
                "server_side": False,
                "local": True,
                "method": "Local bbox crop on selected-variable grids before byte accounting.",
            },
            "notes": "Selected-variable bytes are server-side range extracted; CONUS crop accounting is local.",
        }

    def measure_full_file_sample_bytes(self, objects: list[RemoteObject]) -> Optional[int]:
        if not objects:
            return 0
        if any(obj.estimated_bytes is None for obj in objects):
            return None
        return int(sum(obj.estimated_bytes or 0 for obj in objects))

    def _s3_client(self):
        if self._s3_client_cached is None:
            self._s3_client_cached = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
        return self._s3_client_cached

    @staticmethod
    def _log_retry_attempt(retry_state) -> None:
        instance = retry_state.args[0] if retry_state.args else None
        if isinstance(instance, GfsAwsConusDataSource):
            instance._retry_count += 1
            instance._warning_count += 1
        attempt = retry_state.attempt_number
        next_sleep = retry_state.next_action.sleep if retry_state.next_action else None
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        print(f"GFS AWS retry: attempt={attempt}, backoff_s={next_sleep}, error={exc}")

    def _discover_available_extent(self, s3) -> None:
        prefixes = list(self._iter_common_prefixes(s3, self.config.day_prefix_root))
        days: list[str] = []
        for prefix in prefixes:
            match = re.search(r"gfs\.(\d{8})/$", prefix)
            if match:
                days.append(match.group(1))
        if not days:
            print("GFS AWS availability: no day prefixes discovered.")
            return
        day_min = min(days)
        day_max = max(days)
        self._available_start = datetime.strptime(day_min, "%Y%m%d")
        self._available_end = datetime.strptime(day_max, "%Y%m%d") + timedelta(hours=23, minutes=59, seconds=59)
        print(
            "GFS AWS availability: "
            f"earliest={self._available_start.isoformat()}Z, "
            f"latest={self._available_end.isoformat()}Z"
        )

    def _iter_common_prefixes(self, s3, prefix: str) -> Iterable[str]:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.config.bucket, Prefix=prefix, Delimiter="/"):
            for entry in page.get("CommonPrefixes", []):
                yield entry.get("Prefix")

    def _iter_cycle_datetimes(self, start: datetime, end: datetime) -> Iterable[datetime]:
        day = datetime(start.year, start.month, start.day)
        end_day = datetime(end.year, end.month, end.day)
        while day <= end_day:
            for cycle_h in self.CYCLE_HOURS:
                cycle_dt = day.replace(hour=cycle_h)
                if cycle_dt < start or cycle_dt > end:
                    continue
                yield cycle_dt
            day += timedelta(days=1)

    def _count_cycles(self, start: datetime, end: datetime) -> int:
        return sum(1 for _ in self._iter_cycle_datetimes(start, end))

    def _list_cycle_sizes(self, s3, cycle_dt: datetime) -> dict[int, int]:
        sizes: dict[int, int] = {}
        prefix = (
            f"gfs.{cycle_dt.year:04d}{cycle_dt.month:02d}{cycle_dt.day:02d}/"
            f"{cycle_dt.hour:02d}/atmos/gfs.t{cycle_dt.hour:02d}z.{self.config.product}.f"
        )
        continuation: Optional[str] = None
        while True:
            kwargs = {"Bucket": self.config.bucket, "Prefix": prefix}
            if continuation:
                kwargs["ContinuationToken"] = continuation
            resp = s3.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                key = obj.get("Key", "")
                lead = self._parse_lead_from_key(key, cycle_dt.hour)
                if lead is None:
                    continue
                if lead > self.max_lead_h:
                    continue
                sizes[lead] = int(obj.get("Size") or 0)
            if not resp.get("IsTruncated"):
                break
            continuation = resp.get("NextContinuationToken")
        return sizes

    def _object_key(self, cycle_dt: datetime, lead_h: int) -> str:
        ymd = f"{cycle_dt.year:04d}{cycle_dt.month:02d}{cycle_dt.day:02d}"
        return (
            f"gfs.{ymd}/{cycle_dt.hour:02d}/atmos/"
            f"gfs.t{cycle_dt.hour:02d}z.{self.config.product}.f{lead_h:03d}"
        )

    @staticmethod
    def _parse_lead_from_key(key: str, cycle_hour: int) -> Optional[int]:
        match = re.search(rf"gfs\.t{cycle_hour:02d}z\.pgrb2\.0p25\.f(\d{{3}})$", key)
        if not match:
            return None
        return int(match.group(1))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=_log_retry_attempt,
    )
    def _download_s3_range(self, s3, bucket: str, key: str, byte_range: str) -> bytes:
        resp = s3.get_object(Bucket=bucket, Key=key, Range=byte_range)
        return resp["Body"].read()

    def _download_one_selected_subset(self, s3, out_dir: Path, obj: RemoteObject) -> Path:
        bucket, key = self._parse_s3_url(obj.url)
        idx_key = f"{key}.idx"
        idx_resp = s3.get_object(Bucket=bucket, Key=idx_key)
        idx_text = idx_resp["Body"].read().decode("utf-8", errors="replace")

        entries = self._parse_idx_entries(idx_text)
        ranges = self._selected_ranges(entries)

        rel_dir = Path(obj.key).parent
        lead_part = f"f{(obj.lead_time_h or 0):03d}"
        out_path = out_dir / rel_dir / f"selected_{lead_part}.grib2"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("wb") as f:
            for start, end in ranges:
                if end is None:
                    hdr = f"bytes={start}-"
                else:
                    hdr = f"bytes={start}-{end}"
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
            raise RuntimeError("No selected GFS messages found from idx.")
        return merged

    @staticmethod
    def _is_selected_message(var: str, level: str) -> bool:
        v = var.upper()
        l = level.lower()
        if v == "PRATE" and l == "surface":
            return True
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
    def _parse_s3_url(url: str) -> tuple[str, str]:
        if not url.startswith("s3://"):
            raise ValueError("Expected s3:// URL for GFS AWS object.")
        bucket_and_key = url[5:]
        bucket, key = bucket_and_key.split("/", 1)
        return bucket, key

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
