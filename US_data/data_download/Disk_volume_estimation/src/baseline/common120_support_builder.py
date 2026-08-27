"""Authoritative, candidate-independent Common-120 support construction."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import xarray as xr

from .fixed_support_contract_v2 import build_fixed_support_contract, validate_fixed_support_contract
from .nh_seed_evaluation import basin_netcdf_path
from .gap_mask_io import MRMS_PRODUCT, RTMA_PRODUCT, load_gap_timestamps_json
from .nh_config_generation import read_package_manifest, validate_full_population_basin_membership
from .pilot_lead06_config import load_screening_basin_ids
from .policy import load_stage1_baseline_policy
from .policy_v2_six_axis import load_stage1_baseline_policy_v2_six_axis
from .sweep_v1_production_adapter import PreparationPaths, _verify_artifact_identities
from . import sweep_v1_campaign as sweep
from .sweep_v2_six_axis_campaign import OBJECTIVE_ID_V2, SEQ_LENGTH_MAX
from .validity_mask import bad_hour_mask_from_timestamps, compute_history_valid, _validate_hourly_index

__all__ = ["Common120SupportError", "Common120BuildResult", "build_common120_support"]


class Common120SupportError(ValueError):
    pass


@dataclass(frozen=True)
class Common120BuildResult:
    contract: dict
    accounting: dict


def _dates_and_target(path: Path, target: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if not path.is_file():
        raise Common120SupportError(f"missing basin NetCDF: {path}")
    with xr.open_dataset(path) as data:
        if "date" not in data.coords or target not in data:
            raise Common120SupportError(f"{path}: required date/target is missing")
        values = np.asarray(data[target].values)
        dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(data["date"].values)))
    if values.ndim != 1 or values.shape != (len(dates),):
        raise Common120SupportError(f"{path}: target must be one-dimensional and date-aligned")
    try:
        _validate_hourly_index(dates)
    except Exception as exc:
        raise Common120SupportError(f"{path}: invalid package-native timeline: {exc}") from exc
    return dates, values


def build_common120_support(*, package_root, splits_dir, screening_basin_ids_path,
                            baseline_policy_path, policy_overlay_path) -> Common120BuildResult:
    """Build, but never write, the frozen 400-basin issue-time support contract."""
    baseline = load_stage1_baseline_policy(baseline_policy_path)
    effective = load_stage1_baseline_policy_v2_six_axis(baseline_policy_path, policy_overlay_path)
    if effective["gap_policy"]["include_rtma_in_history_mask"] is not True:
        raise Common120SupportError("frozen production policy must include RTMA in the package gap mask")
    validation = baseline["temporal_split"]["validation"]
    # For hourly grids, +23h is the final point of an inclusive end date,
    # equivalent to NeuralHydrology's date + 1 day - 1 second convention.
    start, end = pd.Timestamp(validation["start"]), pd.Timestamp(validation["end"]) + pd.Timedelta(hours=23)
    target, lead = "qobs_mm_per_h_lead06", 6
    identities = _verify_artifact_identities(PreparationPaths(
        Path(baseline_policy_path), Path(package_root), Path(splits_dir), Path(screening_basin_ids_path)))
    manifest = read_package_manifest(package_root)
    scope = manifest.get("gap_product_scope")
    if scope != [MRMS_PRODUCT, RTMA_PRODUCT]:
        raise Common120SupportError(f"package gap_product_scope must be complete MRMS+RTMA, got {scope!r}")
    membership = validate_full_population_basin_membership(manifest, splits_dir)
    basins = load_screening_basin_ids(screening_basin_ids_path, development_basins=membership.development_basins,
                                     expected_count=400, expected_sha256=sweep.SCREENING_ARTIFACT_SHA256)
    if basins != sorted(set(basins)):
        raise Common120SupportError("screening basins must be exactly 400 sorted unique IDs")
    gap_path = Path(package_root) / "masks" / "gap_timestamps.json"
    declared_gap = (manifest.get("gap_timestamp_artifact") or {}).get("sha256")
    actual_gap = hashlib.sha256(gap_path.read_bytes()).hexdigest() if gap_path.is_file() else None
    if not isinstance(declared_gap, str) or actual_gap != declared_gap:
        raise Common120SupportError("package gap-timestamp artifact checksum contradicts package manifest")
    first_dates, _ = _dates_and_target(basin_netcdf_path(package_root, basins[0]), target)
    gaps = load_gap_timestamps_json(gap_path)
    try:
        bad = bad_hour_mask_from_timestamps(first_dates, gaps, on_out_of_range="ignore")
        history = compute_history_valid(first_dates, bad, SEQ_LENGTH_MAX)
    except Exception as exc:
        raise Common120SupportError(f"invalid packaged gap mask: {exc}") from exc
    global_valid = history & (first_dates >= start) & (first_dates <= end) & (first_dates + pd.Timedelta(hours=lead) <= end)
    per_dates, per_admitted, counts = {}, {}, {}
    for basin in basins:
        dates, qobs = _dates_and_target(basin_netcdf_path(package_root, basin), target)
        if not dates.equals(first_dates):
            raise Common120SupportError(f"{basin}: timeline differs from package-native authoritative timeline")
        admitted = global_valid & np.isfinite(qobs)
        if not admitted.any():
            raise Common120SupportError(f"{basin}: zero Common-120 support")
        per_dates[basin], per_admitted[basin] = dates.to_numpy(), admitted
        counts[basin] = int(admitted.sum())
    contract = build_fixed_support_contract(
        contract_id=OBJECTIVE_ID_V2, lead_hours=lead, target_variable=target, period="validation",
        date_start=str(start.date()), date_end=str(end.date()), source_gap_policy_identity="stage1_policy_mrms_rtma_history_v001",
        screening_basin_ids_sha256=sweep.SCREENING_ARTIFACT_SHA256, per_basin_date=per_dates,
        per_basin_admitted=per_admitted, seq_length_floor=SEQ_LENGTH_MAX, **identities,
    )
    validate_fixed_support_contract(contract)
    values = np.array(list(counts.values()), dtype=int)
    accounting = {"n_basins": len(basins), "global_validation_issue_times": int(global_valid.sum()),
                  "total_retained": int(values.sum()), "per_basin_retained": counts,
                  "min_retained": int(values.min()), "max_retained": int(values.max()),
                  "median_retained": float(np.median(values)), "gap_count": len(gaps),
                  "package_identities": identities}
    return Common120BuildResult(contract=contract, accounting=accounting)
