"""Pipeline configuration dataclasses and YAML loader for Flash-NH.

Config hierarchy (first wins):
  1. --data-root CLI flag passed to a script
  2. FLASHNH_DATA_ROOT environment variable
  3. data_root.local value in the YAML config file
  4. Default: <repo_parent>/Flash-NH_data

Usage:
    from src.pipeline.config import load_config, PipelineConfig
    cfg = load_config(Path("configs/pilot_stage1.yaml"))
    data_root = cfg.effective_data_root()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_ROOT = str(REPO_ROOT.parent / "Flash-NH_data")


@dataclass
class BasinSelectionConfig:
    n_train_core_keep: int = 40
    n_holdout_review: int = 5
    n_exclude_qc_only: int = 5
    fallback_all_train: bool = True
    random_seed: int = 42
    basin_status_file: Optional[str] = None


@dataclass
class PilotConfig:
    n_basins: int = 50
    time_start: str = "2023-01-01T00:00:00"
    time_end: str = "2023-01-31T23:00:00"
    basin_selection: BasinSelectionConfig = field(default_factory=BasinSelectionConfig)


@dataclass
class CamelsHConfig:
    basin_polygons: Optional[str] = None
    static_attributes: Optional[str] = None
    hourly_streamflow: Optional[str] = None


@dataclass
class PipelineConfig:
    project_name: str = "flash-nh"
    version: str = "stage1_pilot_v001"
    data_root_local: Optional[str] = None
    data_root_hpc: Optional[str] = None
    pilot: PilotConfig = field(default_factory=PilotConfig)
    camelsh: CamelsHConfig = field(default_factory=CamelsHConfig)

    # Ordered list of named output directories under data_root.
    # Changing this mapping here is the only place needed to rename dirs.
    _OUTPUT_DIR_NAMES: dict[str, str] = field(default_factory=lambda: {
        "raw":                "00_raw",
        "standardized_grids": "01_standardized_grids",
        "basin_geometries":   "02_basin_geometries",
        "basin_timeseries":   "03_basin_timeseries",
        "ml_datasets":        "04_ml_datasets",
        "splits":             "05_splits",
        "qc_reports":         "06_qc_reports",
        "logs":               "08_logs",
        "manifests":          "09_manifests",
        "tmp":                "tmp",
    })

    def effective_data_root(self, override: Optional[str] = None) -> Path:
        """Return effective data root respecting override > env > config > default."""
        if override:
            return Path(override)
        env = os.environ.get("FLASHNH_DATA_ROOT")
        if env:
            return Path(env)
        if self.data_root_local:
            return Path(self.data_root_local)
        return Path(_DEFAULT_DATA_ROOT)

    def output_dir(self, key: str, data_root: Optional[Path] = None) -> Path:
        """Resolve a named output directory under data_root."""
        root = data_root or self.effective_data_root()
        names = self._OUTPUT_DIR_NAMES
        if key not in names:
            raise KeyError(f"Unknown output key {key!r}. Valid: {sorted(names)}")
        return root / names[key]

    def resolve_basin_status_file(self) -> Optional[Path]:
        """Return path to final_basin_training_status.csv (auto-discovered from repo if not set)."""
        explicit = self.pilot.basin_selection.basin_status_file
        if explicit:
            return Path(explicit)
        candidate = (
            REPO_ROOT / "reports" / "flashnh_final_basin_selection_v001"
            / "tables" / "final_basin_training_status.csv"
        )
        return candidate if candidate.exists() else None


def load_config(path: Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file."""
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return _parse_raw(raw)


def _parse_raw(raw: dict[str, Any]) -> PipelineConfig:
    cfg = PipelineConfig()

    proj = raw.get("project") or {}
    cfg.project_name = str(proj.get("name", cfg.project_name))
    cfg.version = str(proj.get("version", cfg.version))

    dr = raw.get("data_root") or {}
    if isinstance(dr, dict):
        local = dr.get("local")
        hpc = dr.get("hpc")
        if local:
            cfg.data_root_local = str(local)
        if hpc:
            cfg.data_root_hpc = str(hpc)
    elif isinstance(dr, str) and dr:
        cfg.data_root_local = dr

    pilot_raw = raw.get("pilot") or {}
    if "n_basins" in pilot_raw:
        cfg.pilot.n_basins = int(pilot_raw["n_basins"])
    tr = pilot_raw.get("time_range") or {}
    if "start" in tr:
        cfg.pilot.time_start = str(tr["start"])
    if "end" in tr:
        cfg.pilot.time_end = str(tr["end"])

    bs = pilot_raw.get("basin_selection") or {}
    if "n_train_core_keep" in bs:
        cfg.pilot.basin_selection.n_train_core_keep = int(bs["n_train_core_keep"])
    if "n_holdout_review" in bs:
        cfg.pilot.basin_selection.n_holdout_review = int(bs["n_holdout_review"])
    if "n_exclude_qc_only" in bs:
        cfg.pilot.basin_selection.n_exclude_qc_only = int(bs["n_exclude_qc_only"])
    if "fallback_all_train" in bs:
        cfg.pilot.basin_selection.fallback_all_train = bool(bs["fallback_all_train"])
    if "random_seed" in bs:
        cfg.pilot.basin_selection.random_seed = int(bs["random_seed"])
    if bs.get("basin_status_file"):
        cfg.pilot.basin_selection.basin_status_file = str(bs["basin_status_file"])

    camelsh_raw = raw.get("camelsh") or {}
    if camelsh_raw.get("basin_polygons"):
        cfg.camelsh.basin_polygons = str(camelsh_raw["basin_polygons"])
    if camelsh_raw.get("static_attributes"):
        cfg.camelsh.static_attributes = str(camelsh_raw["static_attributes"])
    if camelsh_raw.get("hourly_streamflow"):
        cfg.camelsh.hourly_streamflow = str(camelsh_raw["hourly_streamflow"])

    return cfg


def config_to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    """Serialize config to a plain dict suitable for JSON/YAML provenance snapshots."""
    return {
        "project": {"name": cfg.project_name, "version": cfg.version},
        "data_root": {"local": cfg.data_root_local, "hpc": cfg.data_root_hpc},
        "pilot": {
            "n_basins": cfg.pilot.n_basins,
            "time_range": {"start": cfg.pilot.time_start, "end": cfg.pilot.time_end},
            "basin_selection": {
                "n_train_core_keep": cfg.pilot.basin_selection.n_train_core_keep,
                "n_holdout_review": cfg.pilot.basin_selection.n_holdout_review,
                "n_exclude_qc_only": cfg.pilot.basin_selection.n_exclude_qc_only,
                "fallback_all_train": cfg.pilot.basin_selection.fallback_all_train,
                "random_seed": cfg.pilot.basin_selection.random_seed,
                "basin_status_file": cfg.pilot.basin_selection.basin_status_file,
            },
        },
        "camelsh": {
            "basin_polygons": cfg.camelsh.basin_polygons,
            "static_attributes": cfg.camelsh.static_attributes,
            "hourly_streamflow": cfg.camelsh.hourly_streamflow,
        },
    }
