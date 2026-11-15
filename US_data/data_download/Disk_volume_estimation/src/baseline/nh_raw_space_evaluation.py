"""Raw-space (m^3/s) evaluation layer for a completed NeuralHydrology 1.13
run (Stage 1 initial full-population seed training, see
``docs/decision_log.md``'s 2026-07-25 entry).

NH's own evaluation output (``<period>_results.p``, read by
``nh_evaluation_check.py``) reports observations/predictions in the
area-normalized training scale (``qobs_mm_per_h_lead06``, mm/h-equivalent
runoff depth), never raw discharge. This module is the ONLY place in the
repository that reconverts those values back to raw m^3/s
(``docs/stage1_scientific_baseline_design.md``'s binding primary metric
space) and computes hydrological skill metrics on them.

Basin area is not stored anywhere in ``stage1_scientific_package_v002`` (no
``DRAIN_SQKM`` field in any NetCDF or manifest -- confirmed by inspection of
``package_netcdf.py``). Rather than requiring a new h2o->Moriah transfer of
the external static-attribute matrix or a separate area CSV, this module
self-derives each basin's area directly from data already present in its own
package NetCDF: the diagnostic raw ``qobs_m3s`` series and the built
lead-shifted target (e.g. ``qobs_mm_per_h_lead06``) satisfy, by construction
(see ``docs/stage1_scientific_baseline_design.md`` Sec 5a's own round-trip
audit requirement)::

    qobs_mm_per_h_lead{L}[t] == discharge_m3s_to_runoff_mm_per_h(qobs_m3s[t + L], area_km2)

so ``area_km2`` can be recovered algebraically from many (t) samples per
basin, reduced robustly (median + a relative-MAD consistency check) rather
than trusting a single sample.

Reuses :mod:`src.baseline.units` for the discharge<->runoff-depth conversion
contract (float64, NaN-preserving, infinite-discharge-rejecting,
positive-finite-area-enforcing) and
:class:`src.baseline.package_audit.AuditReport` / ``sha256_file`` for the
generic evidence-reporting idiom used by every other Stage 1 audit module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .units import runoff_mm_per_h_to_discharge_m3s

__all__ = [
    "RawSpaceEvaluationError",
    "BasinAreaResult",
    "derive_basin_area_km2",
    "derive_basin_area_km2_from_netcdf",
    "OBSERVATION_CUTOFF_CONFIG_KEY",
    "parse_observation_cutoff",
    "read_run_observation_cutoff",
    "resolve_run_observation_cutoff",
    "select_through_observation_cutoff",
    "SPLIT_CONTRACT_CONFIG_KEY",
    "ROTATION_CYCLE_CONFIG_KEY",
    "RAW_OBSERVATION_VARIABLE",
    "observation_offset_hours",
    "select_readable_observations",
    "RotationSealing",
    "parse_rotation_sealing",
    "read_run_rotation_sealing",
    "SEALED_DATASET_KEY",
    "SPLIT_SEALING_RECEIPT_TEMPLATE",
    "FROZEN_SPLIT_CONTRACT_CHECKSUMS",
    "split_sealing_receipt_path",
    "run_has_split_sealing_receipts",
    "verify_run_sealing_receipts",
    "observation_times_outside_windows",
    "PeriodConversionResult",
    "convert_period_to_raw_space",
    "raw_space_metrics",
    "aggregate_raw_space_metrics",
    "evaluate_basin_raw_space",
    "pooled_raw_space_metrics",
]

DEFAULT_MIN_AREA_SAMPLES = 100
DEFAULT_MAX_RELATIVE_MAD = 1e-4


class RawSpaceEvaluationError(Exception):
    """Raised for a setup/contract problem (insufficient samples, an
    inconsistent algebraic round-trip, shape mismatch, etc.), never for an
    ordinary "this basin/epoch has poor skill" outcome."""


# ---------------------------------------------------------------------------
# Optional observation cutoff (experiment-scoped sealed-period boundary)
# ---------------------------------------------------------------------------

#: NH config key naming the last package timestamp whose observation a run may
#: read (ISO timestamp, e.g. ``"2024-12-31T23:00:00"``). Absent -> unchanged
#: whole-file behaviour (every historical run/evaluation).
OBSERVATION_CUTOFF_CONFIG_KEY = "flashnh_observation_cutoff"
_LEAD_VARIABLE_RE = re.compile(r"_lead(\d+)$")


def parse_observation_cutoff(value) -> Optional["pd.Timestamp"]:
    """``None``/``"None"`` -> ``None``; otherwise a timezone-naive hourly
    ``pandas.Timestamp`` (raises :class:`RawSpaceEvaluationError`)."""
    import pandas as pd

    if value is None or value == "None":
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RawSpaceEvaluationError(f"invalid observation cutoff {value!r}: {exc}") from exc
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    if ts != ts.floor("h"):
        raise RawSpaceEvaluationError(f"observation cutoff {value!r} is not on the hourly grid")
    return ts


def read_run_observation_cutoff(run_dir) -> Optional["pd.Timestamp"]:
    """The cutoff declared in an NH run's own ``config.yml`` (``None`` if the
    key is absent or the run directory has no ``config.yml``)."""
    from ruamel.yaml import YAML

    config_path = Path(run_dir) / "config.yml"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = YAML(typ="safe").load(fh) or {}
    return parse_observation_cutoff(cfg.get(OBSERVATION_CUTOFF_CONFIG_KEY))


def resolve_run_observation_cutoff(run_dir, observation_cutoff=None) -> Optional["pd.Timestamp"]:
    """Explicit ``observation_cutoff`` and the run's declared cutoff must agree
    when both are present; either alone applies; neither -> ``None``.

    A run whose ``config.yml`` declares a water-year rotation split contract
    (``flashnh_split_contract`` / ``flashnh_rotation_cycle``) is refused: its
    observations are sealed by readable windows, not by a single cutoff, and a
    caller that only knows the cutoff would read sealed observations."""
    if run_dir is not None:
        declared_cfg = _read_run_config(run_dir)
        if run_has_split_sealing_receipts(run_dir) or not (
                _unset(declared_cfg.get(SPLIT_CONTRACT_CONFIG_KEY))
                and _unset(declared_cfg.get(ROTATION_CYCLE_CONFIG_KEY))):
            raise RawSpaceEvaluationError(
                f"run {run_dir} declares a water-year rotation split contract (or carries its sealing "
                "receipts); this caller only applies an "
                "observation cutoff and would read sealed observations (use the rotation-sealing path)")
    explicit = parse_observation_cutoff(observation_cutoff)
    declared = read_run_observation_cutoff(run_dir) if run_dir is not None else None
    if explicit is not None and declared is not None and explicit != declared:
        raise RawSpaceEvaluationError(
            f"observation cutoff {explicit} contradicts the run's declared {OBSERVATION_CUTOFF_CONFIG_KEY} {declared}")
    return explicit if explicit is not None else declared


def select_through_observation_cutoff(ds, *, cutoff, coordinate: str):
    """Lazily restrict an (unloaded) package ``xarray.Dataset`` so that no
    observation later than ``cutoff`` is read: every variable is sliced to
    ``coordinate <= cutoff``, and every lead-shifted ``*_lead<NN>`` variable
    (whose value at ``t`` is the observation at ``t + NN h``) is read only
    through ``cutoff - NN h`` and NaN-filled -- never read -- after that."""
    import pandas as pd

    if coordinate not in ds.coords:
        raise RawSpaceEvaluationError(f"package dataset has no {coordinate!r} coordinate")
    sub = ds.sel({coordinate: slice(None, cutoff)})
    for name in list(sub.data_vars):
        match = _LEAD_VARIABLE_RE.search(str(name))
        if match is None:
            continue
        head = ds[name].sel({coordinate: slice(None, cutoff - pd.Timedelta(hours=int(match.group(1))))})
        sub[name] = head.reindex({coordinate: sub[coordinate]})
    return sub


# ---------------------------------------------------------------------------
# Water-year rotation sealing (split contract + rotation cycle)
# ---------------------------------------------------------------------------

#: NH config key naming the frozen water-year rotation split contract JSON
#: (``src.baseline.wy_rotation_split``). A relative path is resolved against
#: the project root, so one config works locally and on Moriah.
SPLIT_CONTRACT_CONFIG_KEY = "flashnh_split_contract"
#: NH config key naming the rotation cycle (int) of that contract.
ROTATION_CYCLE_CONFIG_KEY = "flashnh_rotation_cycle"
#: The raw discharge observation in a package NetCDF (observation at row time).
RAW_OBSERVATION_VARIABLE = "qobs_m3s"
#: NH dataset key of the sealed loader (``src.baseline.nh_register``). A run
#: declaring a split contract under any other ``dataset`` was loaded unsealed.
SEALED_DATASET_KEY = "flashnh"
#: Per-period receipt FlashNHDataset writes in NH's ``train_dir``
#: (``<run_dir>/train_data``) after a sealed load.
SPLIT_SEALING_RECEIPT_TEMPLATE = "flashnh_split_sealing_{period}.json"
#: Content checksums of the frozen split contracts this code accepts, pinned
#: here so an edit to the contract-generating code cannot silently produce a
#: different contract that still verifies against itself.
FROZEN_SPLIT_CONTRACT_CHECKSUMS = {
    "flashnh_wy_rotation_split_v001": "125b071b261fcc8a0bbb50254dc3babf9ad0eb2cb3b15e804910ff66d7422983",
}
#: Receipt fields that identify the sealing (the contract path is excluded:
#: it legitimately differs between machines).
_RECEIPT_IDENTITY_KEYS = ("contract_id", "contract_checksum_sha256", "rotation_cycle", "lead_hours",
                          "observation_upper_bound_utc", "readable_observation_windows")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def observation_offset_hours(name) -> Optional[int]:
    """Hours from row time to observation time for an observation variable:
    0 for ``qobs_m3s``, ``NN`` for any ``*_lead<NN>`` variable, ``None`` for a
    variable that holds no observation (forcing, gap flags, anything else)."""
    name = str(name)
    if name == RAW_OBSERVATION_VARIABLE:
        return 0
    match = _LEAD_VARIABLE_RE.search(name)
    return None if match is None else int(match.group(1))


def _check_hourly_coordinate(ds, coordinate: str):
    import pandas as pd

    if coordinate not in ds.coords:
        raise RawSpaceEvaluationError(f"package dataset has no {coordinate!r} coordinate")
    index = pd.DatetimeIndex(ds.indexes[coordinate])
    if index.tz is not None:
        raise RawSpaceEvaluationError(f"package coordinate {coordinate!r} must be naive UTC")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise RawSpaceEvaluationError(f"package coordinate {coordinate!r} must be strictly increasing")
    return index


def select_readable_observations(ds, *, windows, coordinate: str, forcing_variables, upper_bound):
    """Lazily restrict an (unloaded) package ``xarray.Dataset`` to what one
    rotation period may read, BEFORE any observation value is read.

    - ``forcing_variables`` are kept whole (forcing look-back may cross any
      water-year boundary). Each must exist and must not be an observation
      variable (see :func:`observation_offset_hours`).
    - Every observation variable (``qobs_m3s`` and every ``*_lead<NN>``, not
      only the configured target) keeps a value only at rows whose
      observation time (row time + offset) lies inside one of ``windows``
      (inclusive :class:`~src.baseline.wy_rotation_split.TimeWindow`-like
      objects with ``first``/``last``) and not after ``upper_bound``. Only those
      row ranges are read (label slices on the lazy array); every other row is
      NaN and never read.
    - Every other variable is dropped, so an unknown variable is sealed by
      default.

    Returns a new Dataset on the file's full coordinate."""
    import pandas as pd
    import xarray as xr

    index = _check_hourly_coordinate(ds, coordinate)
    upper_bound = pd.Timestamp(upper_bound)
    forcing_variables = list(dict.fromkeys(str(v) for v in forcing_variables))
    bad = [v for v in forcing_variables if observation_offset_hours(v) is not None]
    if bad:
        raise RawSpaceEvaluationError(f"observation variable(s) {bad} cannot be read as unrestricted forcing")
    missing = [v for v in forcing_variables if v not in ds.data_vars]
    if missing:
        raise RawSpaceEvaluationError(f"forcing variable(s) {missing} not in the package dataset")

    full = xr.DataArray(index.values, dims=(coordinate,), name=coordinate)
    out = xr.Dataset(coords={coordinate: full})
    for name in forcing_variables:
        out[name] = ds[name]
    for name in ds.data_vars:
        offset = observation_offset_hours(name)
        if offset is None:
            continue
        shift = pd.Timedelta(hours=offset)
        pieces = []
        for window in windows:
            first, last = pd.Timestamp(window.first), min(pd.Timestamp(window.last), upper_bound)
            if last < first:
                continue
            piece = ds[name].sel({coordinate: slice(first - shift, last - shift)})
            if piece.sizes[coordinate]:
                pieces.append(piece.load())
        if pieces:
            values = xr.concat(pieces, dim=coordinate) if len(pieces) > 1 else pieces[0]
            if not pd.DatetimeIndex(values.indexes[coordinate]).is_unique:
                raise RawSpaceEvaluationError(f"readable windows overlap for {name!r}")
            values = values.astype("float64", copy=False).reindex({coordinate: full})
        else:
            values = xr.DataArray(np.full(len(index), np.nan), dims=(coordinate,), coords={coordinate: full})
        out[name] = values.astype(ds[name].dtype if np.issubdtype(ds[name].dtype, np.floating) else "float64")
        out[name].attrs = dict(ds[name].attrs)
    return out


def observation_times_outside_windows(times, *, windows, upper_bound) -> "np.ndarray":
    """Boolean mask: which observation ``times`` lie outside every one of
    ``windows`` (inclusive) or after ``upper_bound``."""
    import pandas as pd

    values = np.asarray(times, dtype="datetime64[ns]")
    inside = np.zeros(values.shape, dtype=bool)
    for window in windows:
        inside |= ((values >= np.datetime64(pd.Timestamp(window.first), "ns"))
                   & (values <= np.datetime64(pd.Timestamp(window.last), "ns")))
    inside &= values <= np.datetime64(pd.Timestamp(upper_bound), "ns")
    return ~inside


@dataclass(frozen=True)
class RotationSealing:
    """One rotation run's sealing identity, resolved from its config and the
    verified frozen contract. ``windows_by_period`` maps each NH period
    (train / validation / test) to its readable observation windows."""

    contract_path: str
    contract_id: str
    contract_checksum_sha256: str
    cycle: int
    lead_hours: int
    upper_bound: "pd.Timestamp"
    windows_by_period: Mapping
    nh_dates: Mapping

    def readable_windows(self, period: str) -> tuple:
        if period not in self.windows_by_period:
            raise RawSpaceEvaluationError(f"unknown NH period {period!r}")
        return self.windows_by_period[period]

    @property
    def training_windows(self) -> tuple:
        return self.windows_by_period["train"]

    def receipt(self) -> dict:
        return {"split_contract": self.contract_path, "contract_id": self.contract_id,
                "contract_checksum_sha256": self.contract_checksum_sha256, "rotation_cycle": self.cycle,
                "lead_hours": self.lead_hours, "observation_upper_bound_utc": self.upper_bound.isoformat(),
                "readable_observation_windows": {
                    p: [[w.first.isoformat(), w.last.isoformat()] for w in ws]
                    for p, ws in self.windows_by_period.items()}}


def _unset(value) -> bool:
    return value is None or value == "None"


def parse_rotation_sealing(config: Mapping, *, project_root=None) -> Optional[RotationSealing]:
    """Resolve ``flashnh_split_contract`` + ``flashnh_rotation_cycle`` from an
    NH config mapping. Neither -> ``None``. Fail closed
    (:class:`RawSpaceEvaluationError`) when only one is set, when
    ``flashnh_observation_cutoff`` is also set, when the contract does not
    verify as the frozen contract, or when the cycle is not an int in its
    table."""
    from .wy_rotation_split import (
        NH_PERIODS, OBSERVATION_UPPER_BOUND, WyRotationSplitError, cycle_nh_dates, load_split_contract,
        readable_observation_windows,
    )
    from .wy_rotation_split import _cycles_from_contract

    path_value = config.get(SPLIT_CONTRACT_CONFIG_KEY)
    cycle_value = config.get(ROTATION_CYCLE_CONFIG_KEY)
    if _unset(path_value) and _unset(cycle_value):
        return None
    if _unset(path_value) or _unset(cycle_value):
        raise RawSpaceEvaluationError(
            f"{SPLIT_CONTRACT_CONFIG_KEY} and {ROTATION_CYCLE_CONFIG_KEY} must be set together "
            f"(got {path_value!r} / {cycle_value!r})")
    if not _unset(config.get(OBSERVATION_CUTOFF_CONFIG_KEY)):
        raise RawSpaceEvaluationError(
            f"{OBSERVATION_CUTOFF_CONFIG_KEY} cannot be combined with {SPLIT_CONTRACT_CONFIG_KEY}")
    if type(cycle_value) is not int:
        raise RawSpaceEvaluationError(f"{ROTATION_CYCLE_CONFIG_KEY} must be an int, got {cycle_value!r}")
    path = Path(str(path_value))
    if not path.is_absolute():
        path = Path(project_root if project_root is not None else _PROJECT_ROOT) / path
    try:
        contract = load_split_contract(path)
    except WyRotationSplitError as exc:
        raise RawSpaceEvaluationError(f"split contract {path} refused: {exc}") from exc
    pinned = FROZEN_SPLIT_CONTRACT_CHECKSUMS.get(contract["contract_id"])
    if pinned is None or contract["checksum_sha256"] != pinned:
        raise RawSpaceEvaluationError(
            f"split contract {path} ({contract['contract_id']!r}, checksum {contract['checksum_sha256']}) is not "
            "a pinned frozen contract")
    cycles = {c.cycle: c for c in _cycles_from_contract(contract)}
    if cycle_value not in cycles:
        raise RawSpaceEvaluationError(
            f"{ROTATION_CYCLE_CONFIG_KEY} {cycle_value} is not in the contract table {sorted(cycles)}")
    cycle = cycles[cycle_value]
    bound = OBSERVATION_UPPER_BOUND
    if contract["observation_upper_bound_utc"] != bound.strftime("%Y-%m-%dT%H:%M:%SZ"):
        raise RawSpaceEvaluationError("contract observation upper bound differs from the frozen value")
    return RotationSealing(
        contract_path=str(path), contract_id=contract["contract_id"],
        contract_checksum_sha256=contract["checksum_sha256"], cycle=cycle.cycle,
        lead_hours=int(contract["conventions"]["lead_hours"]), upper_bound=bound,
        windows_by_period={p: readable_observation_windows(cycle, p) for p in NH_PERIODS},
        nh_dates=cycle_nh_dates(cycle))


def _read_run_config(run_dir) -> dict:
    from ruamel.yaml import YAML

    config_path = Path(run_dir) / "config.yml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        return YAML(typ="safe").load(fh) or {}


def split_sealing_receipt_path(run_dir, period: str) -> Path:
    """Where FlashNHDataset leaves its sealing receipt for ``period`` in an NH
    run directory (NH's ``train_dir`` is ``<run_dir>/train_data``)."""
    return Path(run_dir) / "train_data" / SPLIT_SEALING_RECEIPT_TEMPLATE.format(period=period)


def run_has_split_sealing_receipts(run_dir) -> bool:
    """Whether a sealed loader left any split-sealing receipt in ``run_dir``."""
    train_dir = Path(run_dir) / "train_data"
    return train_dir.is_dir() and any(train_dir.glob(SPLIT_SEALING_RECEIPT_TEMPLATE.format(period="*")))


def read_run_rotation_sealing(run_dir, *, project_root=None) -> Optional[RotationSealing]:
    """The rotation sealing declared in an NH run's own ``config.yml``;
    ``None`` for a run with neither contract key and no sealing receipt.

    Fail closed when the run has sealing receipts but no ``config.yml``, or a
    ``config.yml`` that declares no contract (the record of how the run was
    loaded is missing or contradicts the receipts), and when the config
    declares a contract under a ``dataset`` other than ``flashnh`` (the
    contract keys are then ignored by NH and the data were loaded unsealed)."""
    has_receipts = run_has_split_sealing_receipts(run_dir)
    if not (Path(run_dir) / "config.yml").is_file():
        if has_receipts:
            raise RawSpaceEvaluationError(
                f"run {run_dir} has split-sealing receipts but no config.yml; cannot verify how it was loaded")
        return None
    config = _read_run_config(run_dir)
    sealing = parse_rotation_sealing(config, project_root=project_root)
    if sealing is None:
        if has_receipts:
            raise RawSpaceEvaluationError(
                f"run {run_dir} has split-sealing receipts but its config.yml declares no split contract")
        return None
    if config.get("dataset") != SEALED_DATASET_KEY:
        raise RawSpaceEvaluationError(
            f"run {run_dir} declares a split contract under dataset {config.get('dataset')!r}; only the "
            f"{SEALED_DATASET_KEY!r} dataset applies the sealing, so this run was loaded unsealed")
    return sealing


def verify_run_sealing_receipts(run_dir, sealing: RotationSealing, periods) -> None:
    """Fail closed unless the sealed loader left a receipt for every one of
    ``periods`` in ``run_dir`` whose sealing identity (contract id, checksum,
    cycle, lead, upper bound, readable windows) equals ``sealing``'s."""
    import json

    want = sealing.receipt()
    for period in dict.fromkeys(periods):
        path = split_sealing_receipt_path(run_dir, period)
        if not path.is_file():
            raise RawSpaceEvaluationError(
                f"run {run_dir}: no split-sealing receipt for the {period} period ({path}); the {period} data "
                "were not loaded by the sealed loader")
        try:
            got = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RawSpaceEvaluationError(f"unreadable split-sealing receipt {path}: {exc}") from exc
        if not isinstance(got, dict) or got.get("period") != period:
            raise RawSpaceEvaluationError(f"split-sealing receipt {path} is not for the {period} period")
        differs = [k for k in _RECEIPT_IDENTITY_KEYS if got.get(k) != want[k]]
        if differs:
            raise RawSpaceEvaluationError(
                f"split-sealing receipt {path} does not match the run's declared sealing: {differs}")


# ---------------------------------------------------------------------------
# Basin-area self-derivation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BasinAreaResult:
    """Result of self-deriving one basin's area from its own package data."""

    basin_id: str
    area_km2: float
    n_samples: int
    relative_mad: float
    consistent: bool


def derive_basin_area_km2(
    qobs_m3s: np.ndarray,
    target_mm_per_h: np.ndarray,
    lead_hours: int,
    *,
    basin_id: str = "",
    min_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
) -> BasinAreaResult:
    """Self-derive one basin's area [km^2] from its own hourly time series.

    ``qobs_m3s`` and ``target_mm_per_h`` (e.g. ``qobs_mm_per_h_lead06``) must
    be the SAME basin's full, aligned hourly series (same length, same
    timeline, no reindexing performed here -- callers own alignment).
    ``target_mm_per_h[i]`` is expected (by the Stage 1 package build
    contract) to equal the mm/h-conversion of ``qobs_m3s[i + lead_hours]``;
    inverting that identity for each valid ``i`` gives one area estimate:

        area_km2_estimate[i] = 3.6 * qobs_m3s[i + lead_hours] / target_mm_per_h[i]

    Only strictly positive, finite pairs are used. Raises
    :class:`RawSpaceEvaluationError` if fewer than ``min_samples`` valid
    pairs are available (an area estimate from too few samples is not
    trustworthy). Returns a :class:`BasinAreaResult` whose ``consistent``
    flag is ``False`` (never silently dropped -- callers must check it) when
    the relative median-absolute-deviation of the per-sample estimates
    exceeds ``max_relative_mad``, which would indicate the algebraic
    identity does not hold cleanly for this basin (e.g. a build-time area
    inconsistency) rather than ordinary floating-point noise.
    """
    qobs_m3s = np.asarray(qobs_m3s, dtype=np.float64)
    target_mm_per_h = np.asarray(target_mm_per_h, dtype=np.float64)
    if qobs_m3s.ndim != 1 or target_mm_per_h.ndim != 1:
        raise RawSpaceEvaluationError("qobs_m3s and target_mm_per_h must both be 1-D series")
    if len(qobs_m3s) != len(target_mm_per_h):
        raise RawSpaceEvaluationError(
            f"qobs_m3s (len {len(qobs_m3s)}) and target_mm_per_h (len {len(target_mm_per_h)}) "
            "must be the same length (same aligned timeline)"
        )
    if lead_hours <= 0:
        raise RawSpaceEvaluationError(f"lead_hours must be positive, got {lead_hours}")

    n = len(target_mm_per_h)
    usable_n = n - lead_hours
    if usable_n <= 0:
        raise RawSpaceEvaluationError(
            f"series length {n} is too short for lead_hours={lead_hours}"
        )

    target_head = target_mm_per_h[:usable_n]
    qobs_shifted = qobs_m3s[lead_hours:lead_hours + usable_n]

    valid = np.isfinite(target_head) & np.isfinite(qobs_shifted) & (target_head > 0.0) & (qobs_shifted > 0.0)
    n_valid = int(valid.sum())
    if n_valid < min_samples:
        raise RawSpaceEvaluationError(
            f"basin {basin_id!r}: only {n_valid} valid area-estimate sample(s) "
            f"(need >= {min_samples}); cannot derive a trustworthy area"
        )

    estimates = 3.6 * qobs_shifted[valid] / target_head[valid]
    median = float(np.median(estimates))
    mad = float(np.median(np.abs(estimates - median)))
    relative_mad = mad / median if median > 0.0 else float("inf")
    consistent = relative_mad <= max_relative_mad

    return BasinAreaResult(
        basin_id=basin_id,
        area_km2=median,
        n_samples=n_valid,
        relative_mad=relative_mad,
        consistent=consistent,
    )


def derive_basin_area_km2_from_netcdf(
    nc_path,
    *,
    basin_id: str,
    target_variable: str,
    lead_hours: int,
    min_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
    observation_cutoff=None,
    readable_windows=None,
    observation_upper_bound=None,
) -> BasinAreaResult:
    """I/O wrapper: reads ``qobs_m3s`` and ``target_variable`` from one
    package basin NetCDF (``time_series/<basin_id>.nc``) and calls
    :func:`derive_basin_area_km2`. With ``observation_cutoff`` the read is
    restricted first (:func:`select_through_observation_cutoff`), so the area
    is derived only from observations at or before the cutoff.

    With ``readable_windows`` (a rotation run's TRAINING readable windows) and
    ``observation_upper_bound`` -- both required together, and exclusive with
    ``observation_cutoff`` -- the read goes through
    :func:`select_readable_observations` first, so the area is derived only
    from observations whose observation time lies in those windows and not
    after the bound. Each area sample pairs ``target[t]`` with
    ``qobs_m3s[t + lead]``, which share one observation time, so a sample is
    either wholly readable or wholly sealed."""
    import xarray as xr

    nc_path = Path(nc_path)
    cutoff = parse_observation_cutoff(observation_cutoff)
    if (readable_windows is None) != (observation_upper_bound is None):
        raise RawSpaceEvaluationError("readable_windows and observation_upper_bound must be given together")
    if readable_windows is not None and cutoff is not None:
        raise RawSpaceEvaluationError("observation_cutoff cannot be combined with readable_windows")
    if readable_windows is not None and not tuple(readable_windows):
        raise RawSpaceEvaluationError("readable_windows is empty")
    if not nc_path.is_file():
        raise RawSpaceEvaluationError(f"basin NetCDF not found: {nc_path}")
    with xr.open_dataset(nc_path) as ds:
        if cutoff is not None:
            coordinate = "date" if "date" in ds.coords else "time"
            ds = select_through_observation_cutoff(ds, cutoff=cutoff, coordinate=coordinate)
        elif readable_windows is not None:
            coordinate = "date" if "date" in ds.coords else "time"
            ds = select_readable_observations(ds, windows=tuple(readable_windows), coordinate=coordinate,
                                              forcing_variables=(), upper_bound=observation_upper_bound)
        if "qobs_m3s" not in ds.data_vars:
            raise RawSpaceEvaluationError(f"{nc_path}: missing diagnostic variable 'qobs_m3s'")
        if target_variable not in ds.data_vars:
            raise RawSpaceEvaluationError(f"{nc_path}: missing target variable {target_variable!r}")
        qobs_m3s = ds["qobs_m3s"].values
        target_mm_per_h = ds[target_variable].values

    return derive_basin_area_km2(
        qobs_m3s,
        target_mm_per_h,
        lead_hours,
        basin_id=basin_id,
        min_samples=min_samples,
        max_relative_mad=max_relative_mad,
    )


# ---------------------------------------------------------------------------
# mm/h -> m^3/s conversion with consistent NaN masking
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PeriodConversionResult:
    """One basin/period's raw-space series plus the NaN-masking accounting
    needed to audit "consistent NaN masking" (item required by Part B)."""

    obs_m3s: np.ndarray
    sim_m3s: np.ndarray
    admitted_mask: np.ndarray
    n_admitted: int
    n_total: int
    n_sim_nonfinite_at_admitted: int


def convert_period_to_raw_space(
    obs_mm_per_h: np.ndarray,
    sim_mm_per_h: np.ndarray,
    area_km2: float,
) -> PeriodConversionResult:
    """Convert one basin/period's NH-native mm/h obs+sim series to raw m^3/s.

    "Admitted" samples are exactly those with a finite observation -- the
    same convention ``nh_evaluation_check.py`` already uses (a qobs-NaN hour
    is excluded from loss/metrics, but the model still emits an output for
    every window). Model outputs that are non-finite at an admitted position
    are counted (``n_sim_nonfinite_at_admitted``) rather than silently
    dropped, so callers can surface it as an evaluation-integrity error
    instead of a quietly-shrunk sample count.
    """
    obs_mm_per_h = np.asarray(obs_mm_per_h, dtype=np.float64)
    sim_mm_per_h = np.asarray(sim_mm_per_h, dtype=np.float64)
    if obs_mm_per_h.shape != sim_mm_per_h.shape:
        raise RawSpaceEvaluationError(
            f"obs shape {obs_mm_per_h.shape} != sim shape {sim_mm_per_h.shape}"
        )

    admitted_mask = np.isfinite(obs_mm_per_h)
    n_admitted = int(admitted_mask.sum())
    sim_at_admitted = sim_mm_per_h[admitted_mask]
    n_sim_nonfinite_at_admitted = int((~np.isfinite(sim_at_admitted)).sum())

    obs_m3s_full = np.full_like(obs_mm_per_h, np.nan, dtype=np.float64)
    sim_m3s_full = np.full_like(sim_mm_per_h, np.nan, dtype=np.float64)
    # Convert only at admitted positions; runoff_mm_per_h_to_discharge_m3s
    # rejects +/-inf but tolerates NaN, so non-finite sim values at admitted
    # positions safely become NaN in raw space rather than raising -- the
    # n_sim_nonfinite_at_admitted count above is the authoritative signal for
    # that condition, matching nh_evaluation_check.py's error-not-silent-drop
    # convention.
    obs_admitted = obs_mm_per_h[admitted_mask]
    sim_admitted = np.where(np.isfinite(sim_at_admitted), sim_at_admitted, np.nan)
    obs_m3s_full[admitted_mask] = runoff_mm_per_h_to_discharge_m3s(obs_admitted, area_km2)
    sim_m3s_full[admitted_mask] = runoff_mm_per_h_to_discharge_m3s(sim_admitted, area_km2)

    return PeriodConversionResult(
        obs_m3s=obs_m3s_full,
        sim_m3s=sim_m3s_full,
        admitted_mask=admitted_mask,
        n_admitted=n_admitted,
        n_total=int(obs_mm_per_h.size),
        n_sim_nonfinite_at_admitted=n_sim_nonfinite_at_admitted,
    )


# ---------------------------------------------------------------------------
# Raw-space metrics
# ---------------------------------------------------------------------------

def raw_space_metrics(obs_m3s: np.ndarray, sim_m3s: np.ndarray) -> dict:
    """Raw-space (m^3/s) skill metrics on a single basin/period's admitted
    samples. Both inputs must already be NaN-free and equal-length (callers
    pass ``PeriodConversionResult.obs_m3s[mask]`` / ``sim_m3s[mask]``, or
    equivalent). RMSE/MAE/bias are defined for a single finite pair; PBIAS
    additionally requires a nonzero observed sum. NSE/KGE/Pearson-r are NaN
    when fewer than 2 finite samples or the observation series has zero
    variance (undefined, not zero) -- both conditions are already enforced
    below by the ``denom > 0.0`` / ``obs_std > 0.0`` gates, which a
    single-sample series can never satisfy (its own-mean variance is
    identically zero), so no separate n>=2 branch is needed for those."""
    obs = np.asarray(obs_m3s, dtype=np.float64)
    sim = np.asarray(sim_m3s, dtype=np.float64)
    if obs.shape != sim.shape:
        raise RawSpaceEvaluationError(f"obs shape {obs.shape} != sim shape {sim.shape}")
    finite = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[finite]
    sim = sim[finite]
    n = int(obs.size)

    result = {
        "n_samples": n,
        "nse": float("nan"),
        "kge": float("nan"),
        "kge_r": float("nan"),
        "kge_alpha": float("nan"),
        "kge_beta": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "pearson_r": float("nan"),
        "bias": float("nan"),
        "pbias": float("nan"),
    }
    if n < 1:
        return result

    obs_mean = float(np.mean(obs))
    error = sim - obs
    result["rmse"] = float(np.sqrt(np.mean(error ** 2)))
    result["mae"] = float(np.mean(np.abs(error)))
    result["bias"] = float(np.mean(error))

    obs_sum = float(np.sum(obs))
    result["pbias"] = float(100.0 * np.sum(error) / obs_sum) if obs_sum != 0.0 else float("nan")

    denom = float(np.sum((obs - obs_mean) ** 2))
    if denom > 0.0:
        result["nse"] = float(1.0 - np.sum(error ** 2) / denom)

    obs_std = float(np.std(obs))
    sim_std = float(np.std(sim))
    sim_mean = float(np.mean(sim))
    if obs_std > 0.0 and sim_std > 0.0:
        r = float(np.corrcoef(obs, sim)[0, 1])
        alpha = sim_std / obs_std
        beta = sim_mean / obs_mean if obs_mean != 0.0 else float("nan")
        result["pearson_r"] = r
        result["kge_r"] = r
        result["kge_alpha"] = alpha
        result["kge_beta"] = beta
        if np.isfinite(beta):
            result["kge"] = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    return result


def evaluate_basin_raw_space(
    *,
    basin_id: str,
    obs_mm_per_h: np.ndarray,
    sim_mm_per_h: np.ndarray,
    area_km2: float,
    return_admitted_arrays: bool = False,
) -> dict:
    """Convert one basin/period to raw space and compute its metrics in one
    step. Returns a flat dict merging :class:`PeriodConversionResult`
    accounting fields with :func:`raw_space_metrics`'s output.

    ``return_admitted_arrays``, if set, additionally includes the basin's
    own admitted (finite-observation) raw m^3/s obs/sim arrays under
    ``"_admitted_obs_m3s"``/``"_admitted_sim_m3s"`` -- leading underscore to
    signal these are for an immediate in-process pooling reduction (see
    :func:`pooled_raw_space_metrics`), not for JSON serialization; callers
    that persist this dict as evidence must pop them first. Defaults to
    ``False`` so existing callers/tests are unaffected.
    """
    conversion = convert_period_to_raw_space(obs_mm_per_h, sim_mm_per_h, area_km2)
    metrics = raw_space_metrics(
        conversion.obs_m3s[conversion.admitted_mask],
        conversion.sim_m3s[conversion.admitted_mask],
    )
    result = {
        "basin_id": basin_id,
        "area_km2": area_km2,
        "n_total": conversion.n_total,
        "n_admitted": conversion.n_admitted,
        "n_sim_nonfinite_at_admitted": conversion.n_sim_nonfinite_at_admitted,
        **metrics,
    }
    if return_admitted_arrays:
        result["_admitted_obs_m3s"] = conversion.obs_m3s[conversion.admitted_mask]
        result["_admitted_sim_m3s"] = conversion.sim_m3s[conversion.admitted_mask]
    return result


def pooled_raw_space_metrics(obs_m3s_list: Sequence[np.ndarray], sim_m3s_list: Sequence[np.ndarray]) -> dict:
    """Diagnostic-only pooled metric (never the primary median-per-basin
    metric -- see ``docs/stage1_scientific_baseline_design.md``'s binding
    metric policy): concatenate every basin's already-admitted raw-space
    obs/sim samples into one array and compute a single set of metrics on
    the pooled population via the same :func:`raw_space_metrics` formulas
    used per-basin (no separate pooled formula)."""
    obs_all = (
        np.concatenate([np.asarray(a, dtype=np.float64) for a in obs_m3s_list])
        if obs_m3s_list else np.array([], dtype=np.float64)
    )
    sim_all = (
        np.concatenate([np.asarray(a, dtype=np.float64) for a in sim_m3s_list])
        if sim_m3s_list else np.array([], dtype=np.float64)
    )
    metrics = raw_space_metrics(obs_all, sim_all)
    metrics["n_basins_pooled"] = len(obs_m3s_list)
    return metrics


# ---------------------------------------------------------------------------
# Cross-basin aggregation
# ---------------------------------------------------------------------------

_AGGREGATE_METRIC_NAMES = ("nse", "kge", "rmse", "mae", "pearson_r", "bias", "pbias")


def aggregate_raw_space_metrics(per_basin_results: Sequence[Mapping]) -> dict:
    """Aggregate a list of :func:`evaluate_basin_raw_space` dicts into
    per-metric median/mean/quartiles, finite-value counts, and total sample
    counts across the whole basin population."""
    n_basins = len(per_basin_results)
    aggregate: dict = {
        "n_basins": n_basins,
        "n_admitted_total": int(sum(r["n_admitted"] for r in per_basin_results)),
        "n_sim_nonfinite_at_admitted_total": int(
            sum(r["n_sim_nonfinite_at_admitted"] for r in per_basin_results)
        ),
        "metrics": {},
    }
    for metric_name in _AGGREGATE_METRIC_NAMES:
        values = np.array([r[metric_name] for r in per_basin_results], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        n_finite = int(finite_values.size)
        if n_finite == 0:
            aggregate["metrics"][metric_name] = {
                "n_finite_basins": 0,
                "median": float("nan"),
                "mean": float("nan"),
                "q25": float("nan"),
                "q75": float("nan"),
            }
            continue
        aggregate["metrics"][metric_name] = {
            "n_finite_basins": n_finite,
            "median": float(np.median(finite_values)),
            "mean": float(np.mean(finite_values)),
            "q25": float(np.percentile(finite_values, 25)),
            "q75": float(np.percentile(finite_values, 75)),
        }
    return aggregate
