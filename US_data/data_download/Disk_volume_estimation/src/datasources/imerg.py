from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import re
import time
from typing import Iterable, Optional

import requests
import numpy as np

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region, RemoteObject, log_request, validate_conus_crop
from src.derived_size import compute_derived_bytes


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImergLateDailyConfig:
    base_url: str = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.07"
    dataset: str = "GPM_3IMERGDL.07"
    file_pattern: str = "3B-DAY-L.MS.MRG.3IMERG.{yyyymmdd}-S000000-E235959.V07"


class ImergLateDailyDataSource(DataSource):
    """IMERG Late Run daily precipitation antecedent datasource.

    Raw product is daily accumulated precipitation NetCDF4 (one timestep/day).
    """

    name = "imerg_late_daily_conus"
    temporal_resolution = "daily"
    LATENCY_DAYS = 1
    LOOKBACK_DAYS = 365

    def __init__(self, config: Optional[ImergLateDailyConfig] = None, download_concurrency: int = 1) -> None:
        self.config = config or ImergLateDailyConfig()
        self.download_concurrency = max(1, int(download_concurrency))
        self._available_start: Optional[datetime] = datetime(2000, 6, 1, 0, 0, 0)
        self._available_end: Optional[datetime] = None
        self._last_selected_accounting_mode: str = "full_file_single_variable"
        self._last_selected_conus_accounting_mode: str = "local_spatial_crop"
        self._last_selected_conus_crop_info: Optional[dict] = None
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
            raise ValueError("IMERG Late Daily implementation currently supports only CONUS bbox.")

        self._refresh_availability()
        listing_started = time.perf_counter()
        session = requests.Session()

        objects: list[RemoteObject] = []
        day = datetime(start.year, start.month, start.day)
        end_day = datetime(end.year, end.month, end.day)
        while day <= end_day:
            file_name = self._resolve_filename_for_day(session, day)
            if file_name is not None:
                url = f"{self.config.base_url}/{day.year:04d}/{day.month:02d}/{file_name}"
                size = self._head_size(session, url)
                objects.append(
                    RemoteObject(
                        url=url,
                        key=file_name,
                        datetime=day,
                        variables=variables,
                        estimated_bytes=size,
                    )
                )
            day += timedelta(days=1)

        elapsed = time.perf_counter() - listing_started
        print(
            "IMERG Late Daily timing: "
            f"listing_time={elapsed:.2f}s, sample_objects={len(objects)}"
        )
        return objects

    def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not objects:
            return []

        session = requests.Session()
        started = time.perf_counter()
        paths: list[Path] = []
        for obj in objects:
            out_path = out_dir / obj.key
            out_path.parent.mkdir(parents=True, exist_ok=True)
            self._download_file(session, obj.url, out_path)
            paths.append(out_path)

        elapsed = time.perf_counter() - started
        avg_seconds = elapsed / len(paths) if paths else 0.0
        print(
            "IMERG Late Daily timing summary: "
            f"files={len(paths)}, total_download_time={elapsed:.2f}s, "
            f"avg_file_download_time={avg_seconds:.3f}s"
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
        self._last_selected_accounting_mode = "full_file_single_variable"
        return self.measure_bytes(files), self._last_selected_accounting_mode

    def measure_selected_conus_bytes(self, files: list[Path], region: Region = CONUS_BBOX) -> tuple[Optional[int], str]:
        try:
            import xarray as xr
        except Exception:
            self._last_selected_conus_accounting_mode = "unavailable"
            return None, "unavailable"

        total_bytes = 0
        self._last_selected_conus_crop_info = None
        lon_min, lat_min, lon_max, lat_max = region.bbox
        for file_path in files:
            ds = xr.open_dataset(file_path)
            try:
                data_var = "precipitation" if "precipitation" in ds.data_vars else next(iter(ds.data_vars), None)
                if data_var is None:
                    continue
                da = ds[data_var]
                lat_name = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else None)
                lon_name = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else None)
                if lat_name is None or lon_name is None:
                    continue
                subset, crop_info = _subset_to_conus(da, lat_name, lon_name, lon_min, lat_min, lon_max, lat_max)
                validation = validate_conus_crop(
                    np.asarray(subset.values),
                    subset.coords,
                    region.bbox,
                    source_name=self.name,
                    crop_kind="IMERG CONUS crop",
                    logger=LOGGER,
                    mark_invalid=self._mark_invalid,
                )
                arr = validation.cropped_array
                if validation.valid and self._last_selected_conus_crop_info is None and validation.crop_bounds is not None:
                    validated_crop_info = {
                        "lon_min": validation.crop_bounds[0],
                        "lon_max": validation.crop_bounds[2],
                        "lat_min": validation.crop_bounds[1],
                        "lat_max": validation.crop_bounds[3],
                        "shape": tuple(int(v) for v in arr.shape),
                    }
                    self._last_selected_conus_crop_info = validated_crop_info
                    print(
                        "IMERG selected_conus crop bounds: "
                        f"lon=[{validated_crop_info['lon_min']:.3f}, {validated_crop_info['lon_max']:.3f}], "
                        f"lat=[{validated_crop_info['lat_min']:.3f}, {validated_crop_info['lat_max']:.3f}], "
                        f"shape={validated_crop_info['shape']}"
                    )
                if arr.size == 0:
                    continue
                total_bytes += int(arr.size * arr.dtype.itemsize)
            finally:
                ds.close()

        self._last_selected_conus_accounting_mode = "local_spatial_crop"
        return int(total_bytes), self._last_selected_conus_accounting_mode

    def get_last_selected_conus_crop_info(self) -> Optional[dict]:
        return self._last_selected_conus_crop_info

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
            raise ValueError("IMERG sample period must include at least one day.")
        if full_days <= 0:
            raise ValueError("IMERG required archive range is empty.")
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
            "backend": "NASA GES DISC HTTPS (Earthdata)",
            "dataset": "IMERG Late Run Daily (GPM_3IMERGDL.07)",
            "region": "Global file download; CONUS crop for selected_conus accounting",
            "raw_cadence": "daily accumulated precipitation",
            "derived_cadence": "daily basin-average",
            "variables_included": ["precipitation"],
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
            "notes": "Uses native IMERG Late daily product (no half-hour aggregation).",
        }

    def set_required_window(self, start: Optional[datetime], end: Optional[datetime]) -> None:
        self._last_required_data_start = start
        self._last_required_data_end = end

    @staticmethod
    def _setup_error_message() -> str:
        return (
            "IMERG setup requires Earthdata authentication. Configure one of:\n"
            "1) ~/.netrc with machine urs.earthdata.nasa.gov login <user> password <pass>\n"
            "2) Earthdata bearer token/cookie flow for GES DISC downloads"
        )

    def _refresh_availability(self) -> None:
        latest_date = datetime.utcnow().date() - timedelta(days=self.LATENCY_DAYS)
        self._available_end = datetime(latest_date.year, latest_date.month, latest_date.day, 23, 59, 59)

    def _resolve_filename_for_day(self, session: requests.Session, day: datetime) -> Optional[str]:
        ymd = f"{day.year:04d}{day.month:02d}{day.day:02d}"
        month_url = f"{self.config.base_url}/{day.year:04d}/{day.month:02d}/"
        try:
            log_request(LOGGER, self.name, "http.get", url=month_url, params={"timeout_s": 30, "allow_redirects": True})
            resp = session.get(month_url, timeout=30)
            resp.raise_for_status()
        except Exception:
            return None

        pattern = re.compile(
            rf"3B-DAY-L\.MS\.MRG\.3IMERG\.{ymd}-S000000-E235959\.V07[^\"\s]*\.nc4"
        )
        matches = pattern.findall(resp.text)
        if not matches:
            return None
        return sorted(set(matches))[-1]

    def _head_size(self, session: requests.Session, url: str) -> Optional[int]:
        try:
            log_request(LOGGER, self.name, "http.head", url=url, params={"timeout_s": 30, "allow_redirects": True})
            resp = session.head(url, timeout=30, allow_redirects=True)
            if resp.status_code >= 400:
                return None
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
        except Exception:
            return None

    def _download_file(self, session: requests.Session, url: str, out_path: Path) -> None:
        try:
            log_request(LOGGER, self.name, "http.get", url=url, params={"stream": True, "timeout_s": 120, "allow_redirects": True})
            with session.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
                if resp.status_code == 401 or "urs.earthdata.nasa.gov" in str(resp.url):
                    raise RuntimeError(self._setup_error_message())
                resp.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("IMERG download failed. " + self._setup_error_message()) from exc


def _subset_to_conus(da, lat_name: str, lon_name: str, lon_min: float, lat_min: float, lon_max: float, lat_max: float):
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