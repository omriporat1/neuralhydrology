from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from src.derived_size import DerivedSpec


@dataclass(frozen=True)
class Region:
	name: str
	bbox: tuple[float, float, float, float]


CONUS_BBOX = Region("conus_bbox", (-126.0, 24.0, -66.0, 50.0))


@dataclass(frozen=True)
class RemoteObject:
	url: str
	key: str
	datetime: datetime
	variables: list[str]
	estimated_bytes: Optional[int] = None
	lead_time_h: Optional[int] = None




class DataSource(ABC):
	name: str
	temporal_resolution: str  # "hourly", "daily", or "forecast"

	@abstractmethod
	def list_sample_objects(
		self,
		start: datetime,
		end: datetime,
		region: Region,
		variables: list[str],
		lead_times: Optional[Iterable[int]] = None,
	) -> list[RemoteObject]:
		raise NotImplementedError

	@abstractmethod
	def download_sample(self, out_dir: Path, objects: list[RemoteObject]) -> list[Path]:
		raise NotImplementedError

	@abstractmethod
	def measure_bytes(self, files: Iterable[Path]) -> int:
		raise NotImplementedError

	@abstractmethod
	def estimate_raw_total(
		self,
		sample_bytes: int,
		sample_start: datetime,
		sample_end: datetime,
		full_start: datetime,
		full_end: datetime,
	) -> int:
		raise NotImplementedError

	@abstractmethod
	def estimate_derived_total(self, spec: DerivedSpec) -> int:
		raise NotImplementedError

	@abstractmethod
	def estimate_peak_local(
		self,
		raw_total_bytes: int,
		scratch_multiplier: float,
		concurrency: int,
	) -> int:
		raise NotImplementedError

	@abstractmethod
	def assumptions(self) -> dict:
		raise NotImplementedError
