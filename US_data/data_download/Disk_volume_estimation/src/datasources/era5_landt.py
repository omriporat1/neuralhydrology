from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import importlib
from pathlib import Path
import time
from typing import Iterable, Optional
import zipfile

from tqdm import tqdm

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region, RemoteObject
from src.derived_size import compute_derived_bytes


@dataclass(frozen=True)
class Era5LandTConfig:
    dataset: str = "reanalysis-era5-land"
    area: tuple[float, float, float, float] = (50.0, -126.0, 24.0, -66.0)  # north, west, south, east
    format: str = "grib"


class Era5LandTDataSource(DataSource):
    """ERA5-Land-T antecedent datasource via CDS API.

    This datasource estimates raw download volume from hourly gridded files,
    while derived sizing is based on daily basin-average outputs.
    """

    name = "era5_land_t_conus"
    temporal_resolution = "daily"
    LATENCY_DAYS = 5
    LOOKBACK_DAYS = 365

    _CDS_VAR_MAP = {
        "TP": "total_precipitation",
        "2T": "2m_temperature",
        "2D": "2m_dewpoint_temperature",
        "10U": "10m_u_component_of_wind",
        "10V": "10m_v_component_of_wind",
        "SP": "surface_pressure",
        "SSRD": "surface_solar_radiation_downwards",
        "SWVL1": "volumetric_soil_water_layer_1",
        "SD": "snow_depth",
    }

    def __init__(self, config: Optional[Era5LandTConfig] = None, download_concurrency: int = 1) -> None:
        self.config = config or Era5LandTConfig()
        self.download_concurrency = max(1, int(download_concurrency))
        self._available_start: Optional[datetime] = datetime(1950, 1, 1, 0, 0, 0)
        self._available_end: Optional[datetime] = None
        self._last_selected_accounting_mode: str = "server_side_variable_subset"
        self._last_selected_conus_accounting_mode: str = "server_side_spatial_subset"
        self._last_required_data_start: Optional[datetime] = None
        self._last_required_data_end: Optional[datetime] = None

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
            raise ValueError("ERA5-Land-T implementation currently supports only CONUS bbox.")

        self._refresh_availability()
        listing_started = time.perf_counter()
        objects: list[RemoteObject] = []

        day = datetime(start.year, start.month, start.day)
        end_day = datetime(end.year, end.month, end.day)
        while day <= end_day:
            key = f"era5landt_{day.strftime('%Y%m%d')}"
            objects.append(
                RemoteObject(
                    url=f"cds://{self.config.dataset}/{key}",
                    key=key,
                    datetime=day,
                    variables=variables,
                    estimated_bytes=None,
                )
            )
            day += timedelta(days=1)

        elapsed = time.perf_counter() - listing_started
        print(
            "ERA5-Land-T timing: "
            f"listing_time={elapsed:.2f}s, sample_objects={len(objects)}"
        )
        return objects

    def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not objects:
            return []

        try:
            cdsapi = importlib.import_module("cdsapi")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(self._setup_error_message()) from exc

        client = cdsapi.Client(quiet=True)
        paths: list[Path] = []
        started = time.perf_counter()
        for obj in tqdm(objects, desc="Downloading ERA5-Land-T sample", unit="day"):
            out_path = out_dir / f"{obj.key}.grib"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            request = self._build_request(obj)
            client.retrieve(self.config.dataset, request, str(out_path))
            self._maybe_extract_wrapped_grib(out_path)
            paths.append(out_path)

        elapsed = time.perf_counter() - started
        avg_seconds = elapsed / len(paths) if paths else 0.0
        print(
            "ERA5-Land-T timing summary: "
            f"files={len(paths)}, total_download_time={elapsed:.2f}s, "
            f"avg_file_download_time={avg_seconds:.3f}s"
        )
        return paths

    def measure_bytes(self, files: Iterable[Path]) -> int:
        return int(sum(path.stat().st_size for path in files))

    def measure_selected_variable_bytes(self, files: list[Path], objects: list[RemoteObject]) -> tuple[Optional[int], str]:
        del objects
        self._last_selected_accounting_mode = "server_side_variable_subset"
        return self.measure_bytes(files), self._last_selected_accounting_mode

    def measure_selected_conus_bytes(self, files: list[Path], region: Region = CONUS_BBOX) -> tuple[Optional[int], str]:
        del region
        self._last_selected_conus_accounting_mode = "server_side_spatial_subset"
        return self.measure_bytes(files), self._last_selected_conus_accounting_mode

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
            raise ValueError("ERA5-Land-T sample period must include at least one day.")
        if full_days <= 0:
            raise ValueError("ERA5-Land-T required archive range is empty.")
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
        required_start = self._last_required_data_start.isoformat() if self._last_required_data_start else None
        required_end = self._last_required_data_end.isoformat() if self._last_required_data_end else None
        return {
            "backend": "CDS API (ERA5-Land as ERA5-Land-T proxy)",
            "dataset": self.config.dataset,
            "region": "CONUS server-side area subset",
            "raw_cadence": "hourly gridded",
            "derived_cadence": "daily basin-average",
            "variables_included": [
                "total_precipitation",
                "2m_temperature",
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
                "surface_solar_radiation_downwards",
                "volumetric_soil_water_layer_1",
                "snow_depth",
            ],
            "latency_days": self.LATENCY_DAYS,
            "lookback_days": self.LOOKBACK_DAYS,
            "server_side_spatial_subset": True,
            "selected_raw_bytes_mode": self._last_selected_accounting_mode,
            "selected_conus_raw_bytes_mode": self._last_selected_conus_accounting_mode,
            "required_historical_archive_window": {
                "start": required_start,
                "end": required_end,
            },
            "notes": (
                "Raw estimate reflects selected variables with server-side CONUS area subset. "
                "Derived estimate reflects daily antecedent time series over the required archive window."
            ),
        }

    def set_required_window(self, start: Optional[datetime], end: Optional[datetime]) -> None:
        self._last_required_data_start = start
        self._last_required_data_end = end

    @classmethod
    def cds_variables(cls, source_variables: list[str]) -> list[str]:
        variables: list[str] = []
        for token in source_variables:
            mapped = cls._CDS_VAR_MAP.get(token)
            if mapped is not None and mapped not in variables:
                variables.append(mapped)
        return variables

    @staticmethod
    def _setup_error_message() -> str:
        return (
            "ERA5-Land-T setup is incomplete. Install and configure CDS API:\n"
            "1) pip install cdsapi\n"
            "2) Create ~/.cdsapirc with your CDS URL and API key\n"
            "3) Ensure your account has access to ERA5-Land data"
        )

    def _refresh_availability(self) -> None:
        # ERA5-Land-T is modeled with a fixed operational latency.
        latest_date = datetime.utcnow().date() - timedelta(days=self.LATENCY_DAYS)
        self._available_end = datetime(latest_date.year, latest_date.month, latest_date.day, 23, 59, 59)

    def _build_request(self, obj: RemoteObject) -> dict:
        cds_vars = self.cds_variables(obj.variables)
        if not cds_vars:
            raise ValueError("No ERA5-Land-T variables mapped for CDS request.")

        return {
            "product_type": "reanalysis",
            "variable": cds_vars,
            "year": f"{obj.datetime.year:04d}",
            "month": f"{obj.datetime.month:02d}",
            "day": f"{obj.datetime.day:02d}",
            "time": [f"{hour:02d}:00" for hour in range(24)],
            "area": list(self.config.area),
            "format": self.config.format,
        }

    @staticmethod
    def _maybe_extract_wrapped_grib(file_path: Path) -> None:
        if not file_path.exists() or file_path.stat().st_size < 4:
            return
        with file_path.open("rb") as f:
            signature = f.read(4)
        if signature != b"PK\x03\x04":
            return

        with zipfile.ZipFile(file_path, "r") as zf:
            members = [name for name in zf.namelist() if name.lower().endswith((".grib", ".grib2"))]
            if not members:
                return
            data = zf.read(members[0])
        file_path.write_bytes(data)
