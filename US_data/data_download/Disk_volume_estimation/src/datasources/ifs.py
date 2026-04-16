from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Iterable, Optional

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region, RemoteObject
from src.derived_size import compute_derived_bytes


@dataclass(frozen=True)
class IfsMarsConfig:
    mars_class: str = "od"
    stream: str = "oper"
    type: str = "fc"
    levtype: str = "sfc"
    grid: str = "0.25/0.25"
    area: str = "50/-126/24/-66"  # North/West/South/East (CONUS)
    param: str = "228.128/167.128/168.128/165.128/166.128/134.128/169.128"
    target_ext: str = "grib"


class IfsMarsDataSource(DataSource):
    """ECMWF IFS forcing datasource via MARS (authenticated user access)."""

    name = "ifs_mars_conus"
    temporal_resolution = "hourly"
    CYCLE_HOURS = (0, 6, 12, 18)

    def __init__(self, max_lead_h: int = 24, config: Optional[IfsMarsConfig] = None, download_concurrency: int = 2) -> None:
        self.max_lead_h = max(0, int(max_lead_h))
        self.config = config or IfsMarsConfig()
        self.download_concurrency = max(1, int(download_concurrency))
        self._available_start: Optional[datetime] = None
        self._available_end: Optional[datetime] = None
        self._retry_count: int = 0
        self._warning_count: int = 0

        self._last_selected_sample_bytes: Optional[int] = None
        self._last_selected_conus_sample_bytes: Optional[int] = None
        self._last_request_example: Optional[dict] = None

    def list_sample_objects(
        self,
        start: datetime,
        end: datetime,
        region: Region,
        variables: list[str],
        lead_times: Optional[Iterable[int]] = None,
    ) -> list[RemoteObject]:
        if region.name != CONUS_BBOX.name:
            raise ValueError("IFS MARS implementation currently supports only CONUS bbox.")

        del lead_times  # Leads are encoded in per-cycle request step range.
        cycles = list(self._iter_cycle_datetimes(start, end))
        objects: list[RemoteObject] = []
        for cycle_dt in cycles:
            key = f"ifs_{cycle_dt.year:04d}{cycle_dt.month:02d}{cycle_dt.day:02d}_{cycle_dt.hour:02d}"
            objects.append(
                RemoteObject(
                    url=f"mars://{key}",
                    key=key,
                    datetime=cycle_dt,
                    variables=variables,
                    estimated_bytes=None,
                    lead_time_h=self.max_lead_h,
                )
            )

        print(
            "IFS MARS timing: "
            f"simulated_listing_cycles={len(cycles)}, sample_objects={len(objects)}, max_lead_h={self.max_lead_h}"
        )
        return objects

    def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not objects:
            self._last_selected_sample_bytes = 0
            self._last_selected_conus_sample_bytes = 0
            return []

        total_started = time.perf_counter()
        self._retry_count = 0
        self._warning_count = 0

        paths: list[Path] = []
        with ThreadPoolExecutor(max_workers=self.download_concurrency) as executor:
            futures = [executor.submit(self._download_one_cycle, out_dir, obj) for obj in objects]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading IFS MARS sample", unit="cycle"):
                paths.append(future.result())

        total_seconds = time.perf_counter() - total_started
        selected_bytes = self.measure_bytes(paths)
        self._last_selected_sample_bytes = selected_bytes
        self._last_selected_conus_sample_bytes = selected_bytes

        avg_cycle_time = total_seconds / len(paths) if paths else 0.0
        print(
            "IFS MARS timing summary: "
            f"cycles={len(paths)}, total_download_time={total_seconds:.2f}s, "
            f"avg_cycle_time={avg_cycle_time:.3f}s, concurrency={self.download_concurrency}"
        )
        return paths

    def measure_bytes(self, files: Iterable[Path]) -> int:
        return int(sum(path.stat().st_size for path in files))

    def measure_selected_variable_bytes(self, files: list[Path], objects: list[RemoteObject]) -> tuple[Optional[int], str]:
        del objects
        value = self.measure_bytes(files)
        self._last_selected_sample_bytes = value
        return value, "server_side_variable_and_spatial_subset"

    def measure_selected_conus_bytes(self, files: list[Path], region: Region = CONUS_BBOX) -> tuple[Optional[int], str]:
        del region
        value = self.measure_bytes(files)
        self._last_selected_conus_sample_bytes = value
        return value, "server_side_spatial_subset"

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
            raise ValueError("IFS sample must include at least one 6-hour cycle.")

        effective_days = (full_end.date() - full_start.date()).days + 1
        if effective_days <= 0:
            raise ValueError("Requested IFS effective range must include at least one day.")

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
            "backend": "ECMWF MARS via ecmwfapi",
            "region": "CONUS bbox server-side area crop",
            "cadence": "forecast cycles 00/06/12/18 with hourly leads",
            "max_lead_h": self.max_lead_h,
            "variables_included": [
                "TP",
                "2T",
                "2D",
                "10U",
                "10V",
                "SP",
                "SSRD",
            ],
            "subsetting": "server-side variable + spatial subset in MARS request",
            "selected_raw_bytes_mode": "server_side_variable_and_spatial_subset",
            "selected_conus_raw_bytes_mode": "server_side_spatial_subset",
            "mars_area": self.config.area,
            "mars_grid": self.config.grid,
            "mars_param": self.config.param,
            "request_example": self._last_request_example,
            "notes": "Sample object listing is simulated because MARS has no listing API in this workflow.",
        }

    def _iter_cycle_datetimes(self, start: datetime, end: datetime):
        day = datetime(start.year, start.month, start.day)
        end_day = datetime(end.year, end.month, end.day)
        from datetime import timedelta

        while day <= end_day:
            for hour in self.CYCLE_HOURS:
                cycle = day.replace(hour=hour)
                if cycle < start or cycle > end:
                    continue
                yield cycle
            day += timedelta(days=1)

    def _count_cycles(self, start: datetime, end: datetime) -> int:
        return sum(1 for _ in self._iter_cycle_datetimes(start, end))

    @staticmethod
    def _setup_error_message() -> str:
        return (
            "IFS MARS setup is incomplete. Install and configure ECMWF API client:\n"
            "1) pip install ecmwf-api-client\n"
            "2) Create ~/.ecmwfapirc with your ECMWF URL, key, and email\n"
            "3) Ensure your account has Access MARS permissions"
        )

    @staticmethod
    def _build_step_string(max_lead_h: int) -> str:
        return f"0/to/{int(max_lead_h)}/by/1"

    def _build_request(self, cycle_dt: datetime, target_path: Path) -> dict:
        req = {
            "class": self.config.mars_class,
            "stream": self.config.stream,
            "type": self.config.type,
            "levtype": self.config.levtype,
            "date": cycle_dt.strftime("%Y-%m-%d"),
            "time": f"{cycle_dt.hour:02d}:00:00",
            "step": self._build_step_string(self.max_lead_h),
            "param": self.config.param,
            "area": self.config.area,
            "grid": self.config.grid,
            "target": str(target_path),
        }
        self._last_request_example = dict(req)
        return req

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def _download_one_cycle(self, out_dir: Path, obj: RemoteObject) -> Path:
        try:
            from ecmwfapi import ECMWFService  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(self._setup_error_message()) from exc

        file_name = f"{obj.key}.{self.config.target_ext}"
        out_path = out_dir / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        request = self._build_request(obj.datetime, out_path)
        mars_request = {k: v for k, v in request.items() if k != "target"}
        print(f"IFS MARS request: {mars_request}")
        print(f"IFS MARS target: {out_path}")
        print(f"IFS MARS variables: {mars_request['param']}")
        print(f"IFS MARS area: {mars_request['area']}")

        try:
            server = ECMWFService("mars")
            server.execute(mars_request, str(out_path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "IFS MARS request failed. Verify credentials and MARS permissions.\n"
                + self._setup_error_message()
            ) from exc

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError("IFS MARS returned no output file. Check request permissions and parameters.")

        return out_path
