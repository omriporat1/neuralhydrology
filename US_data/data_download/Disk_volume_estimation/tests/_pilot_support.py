"""Shared fixture helpers for tests/test_pilot_*.py (Stage 1 lead-6
optimization pilot, task item 10). Not itself a test module (no test_
prefix -- pytest will not collect it).

Reuses the real, committed Stage 1 scientific baseline policy and canonical
split files, exactly like tests/test_nh_config_generation.py and
tests/test_nh_full_population_config_generation.py, plus the
``build_pilot_bundle``/``validate_full_population_basin_membership`` full-
union package-manifest contract discovered while verifying
pilot_orchestration.py: the fake package's manifest/attributes must cover
the FULL development ∪ spatial_holdout_nonca union (2,557 basins), never
just the small subset a given test actually reads time-series for.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.baseline.nh_seed_evaluation import weight_stem
from src.baseline.policy import load_stage1_baseline_policy
from src.baseline.splits import load_eligible_basins

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_POLICY_PATH = REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"
SPLITS_DIR = REPO_ROOT / "config" / "stage1_baseline_splits_v001"
PILOT_POLICY_PATH = REPO_ROOT / "config" / "stage1_lead06_pilot_v001.yaml"

BASELINE_POLICY = load_stage1_baseline_policy(BASELINE_POLICY_PATH)
REAL_DYNAMIC_INPUTS = list(BASELINE_POLICY["dynamic_inputs"])
STATIC_COUNT = BASELINE_POLICY["static_attributes"]["expected_model_input_columns"]

REAL_DEVELOPMENT = sorted(load_eligible_basins(SPLITS_DIR / "development_train.txt"))
REAL_SPATIAL_HOLDOUT = sorted(load_eligible_basins(SPLITS_DIR / "spatial_holdout_nonca.txt"))
REAL_FULL_UNION = REAL_DEVELOPMENT + REAL_SPATIAL_HOLDOUT


def _subst_mappings() -> dict[str, Path]:
    """Return active ``subst`` drive mappings, keyed by uppercase letter."""
    result = subprocess.run(["subst"], check=True, capture_output=True, text=True)
    mappings: dict[str, Path] = {}
    for line in result.stdout.splitlines():
        if ":\\: => " in line:
            drive, target = line.split(":\\: => ", 1)
            mappings[drive.upper()] = Path(target.strip())
    return mappings


def _windows_project_alias() -> tuple[Path, bool]:
    """Create a short test-only alias, or safely reuse an existing one."""
    mappings = _subst_mappings()
    project_root = REPO_ROOT
    for letter, target in mappings.items():
        if target.exists() and project_root.exists() and target.samefile(project_root):
            return Path(f"{letter}:\\"), False
    for letter in ("T", "U", "V", "W", "X", "Y", "Z"):
        if letter not in mappings and not Path(f"{letter}:\\").exists():
            subprocess.run(["subst", f"{letter}:", str(project_root)], check=True)
            return Path(f"{letter}:\\"), True
    raise RuntimeError("no safe free drive letter available for project-local pilot-test scratch")


def _make_short_root_dir() -> tuple[Path, "Callable[[], None]"]:
    """A short-rooted temporary directory.

    This pilot's real NH continuation-directory nesting
    (``continue_training_from_epoch###/``, potentially several levels deep
    across successive chunks -- see ``pilot_orchestration.py``'s
    continuation-epoch-semantics note) combined with pytest's own long
    default ``tmp_path`` prefix
    (``AppData/Local/Temp/pytest-of-<user>/pytest-NNN/<test-name>/...``) can
    exceed Windows' default 260-character MAX_PATH -- a Windows-only local
    test-environment limitation (Linux, where real Moriah/h2o runs happen,
    has no such limit; see ``docs/stage1_lead06_pilot_v001.md``). Callers own
    cleanup of the returned directory."""
    if sys.platform != "win32":
        return Path(tempfile.mkdtemp()), (lambda: None)
    # ``tmp/`` is the project's intentionally ignored disposable scratch
    # location. Its short path preserves room for real continuation nesting.
    # Do not call ``resolve()`` here: a test-only short drive alias must
    # remain visible to pathlib while the files still live under the project.
    alias_root, created_alias = _windows_project_alias()
    root = alias_root / "tmp"
    root.mkdir(exist_ok=True)
    def cleanup_alias() -> None:
        if created_alias:
            subprocess.run(["subst", alias_root.drive, "/D"], check=True)
    return Path(tempfile.mkdtemp(prefix="t", dir=str(root))), cleanup_alias


@pytest.fixture
def short_tmp_path():
    """Like the built-in ``tmp_path`` fixture, but rooted at a short path on
    Windows (see :func:`_make_short_root_dir`) -- use for any pilot test
    whose NH run directory may nest multiple nested continuation
    directories deep."""
    path, cleanup_alias = _make_short_root_dir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        cleanup_alias()


def pick_development_basins(n: int = 5) -> list:
    assert len(REAL_DEVELOPMENT) >= n
    return REAL_DEVELOPMENT[:n]


def static_columns(n: int = STATIC_COUNT) -> list:
    return [f"col_{i:04d}" for i in range(n)]


def build_full_union_package(root: Path, ts_basin_ids=()) -> Path:
    """Write manifests/package_manifest.json + attributes/attributes.csv
    covering the full 2,557-basin development-union (required by
    ``validate_full_population_basin_membership``, which every pilot bundle
    builder calls) -- cheap even at full scale since no NetCDF is written
    for most of them. Real NetCDF time-series files are written only for
    ``ts_basin_ids`` (the small subset an individual test actually reads),
    matching config-generation's own on-disk requirements: config
    generation never touches time-series files, only screening/full-
    validation checkpoint evaluation does, and only for the basins it is
    told to evaluate.
    """
    columns = static_columns()
    manifests_dir = root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    columns_sha256 = hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()
    manifest = {
        "schema_name": "stage1_compact_package_manifest",
        "schema_version": 1,
        "package_role": "compact_scientific_package",
        "basin_count": len(REAL_FULL_UNION),
        "basin_ids": list(REAL_FULL_UNION),
        "dynamic_variables": REAL_DYNAMIC_INPUTS,
        "static_model_input_columns": columns,
        "static_model_input_columns_sha256": columns_sha256,
    }
    (manifests_dir / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    attrs_dir = root / "attributes"
    attrs_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"gauge_id": b, **{c: 0.0 for c in columns}} for b in REAL_FULL_UNION]
    df = pd.DataFrame(rows, columns=["gauge_id"] + columns)
    df.to_csv(attrs_dir / "attributes.csv", index=False)

    if ts_basin_ids:
        write_basin_netcdfs(root, ts_basin_ids)
    return root


def write_basin_netcdfs(package_root: Path, basin_ids, *, n: int = 400, area_km2: float = 50.0, lead_hours: int = 6, seed: int = 0) -> None:
    ts_dir = package_root / "time_series"
    ts_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for b in basin_ids:
        qobs_m3s = rng.uniform(1.0, 200.0, size=n)
        usable_n = n - lead_hours
        target = np.full(n, np.nan)
        target[:usable_n] = 3.6 * qobs_m3s[lead_hours:lead_hours + usable_n] / area_km2
        xr.Dataset(
            {"qobs_m3s": ("date", qobs_m3s), "qobs_mm_per_h_lead06": ("date", target)},
            coords={"date": np.arange(n)},
        ).to_netcdf(ts_dir / f"{b}.nc")


def write_perfect_validation_results(nh_run_dir: Path, epoch: int, basins, package_root: Path) -> None:
    """Write a validation_results.p whose sim==obs exactly (NSE=1.0) for
    every basin -- deterministic, no torch/NH needed."""
    period_dir = nh_run_dir / "validation" / weight_stem(epoch)
    period_dir.mkdir(parents=True, exist_ok=True)
    basin_results = {}
    for b in basins:
        ds = xr.open_dataset(package_root / "time_series" / f"{b}.nc")
        target = ds["qobs_mm_per_h_lead06"].values
        obs = target.copy()
        sim = obs.copy()
        xr_ds = xr.Dataset(
            {"qobs_mm_per_h_lead06_obs": ("date", obs), "qobs_mm_per_h_lead06_sim": ("date", sim)},
            coords={"date": np.arange(len(obs))},
        )
        basin_results[b] = {"1h": {"xr": xr_ds}}
    with open(period_dir / "validation_results.p", "wb") as fh:
        pickle.dump(basin_results, fh)


def write_screening_basin_ids_file(path: Path, basin_ids) -> Path:
    path.write_text("\n".join(basin_ids) + "\n", encoding="utf-8")
    return path
