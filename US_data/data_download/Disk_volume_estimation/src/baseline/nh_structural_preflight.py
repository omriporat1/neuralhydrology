"""Stage 1 NH structural preflight (local implementation increment, no training).

Two independent, composable layers, mirroring the independence-boundary
style already used by :mod:`src.baseline.package_audit`:

1. **File-only structural checks** (:func:`check_generated_config_structure`,
   :func:`inspect_dynamic_nan_inventory`) -- never import
   ``neuralhydrology``, never construct a dataset. These can run against
   either a synthetic fixture (this task) or, later, the real certified
   package on h2o (not executed there by this task).
2. **Real-NH dataset-construction checks**
   (:func:`check_flashnh_dataset_construction`) -- import
   ``neuralhydrology``, register ``flashnh``, build a real ``Config`` and
   real ``FlashNHDataset`` instances for train/validation/test. Only ever
   exercised here against synthetic fixtures (never a real package, never
   training -- construction only).

:func:`run_structural_preflight` composes both layers into one
:class:`~src.baseline.package_audit.AuditReport`. ``AuditReport``/
``CheckRecord`` are reused directly from ``package_audit`` (a generic,
non-scientific reporting container) rather than reimplemented; this module
does not import any other scientific re-derivation logic from
``package_audit``, ``package_builder``, or ``package_assembly``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .gap_mask_io import GapMaskIOError, load_gap_timestamps_json
from .nh_register import FLASHNH_DATASET_KEY, register_flashnh_dataset
from .package_audit import AuditReport
from .staid import normalize_staid

__all__ = [
    "NHStructuralPreflightError",
    "DynamicNanInventory",
    "RTMA_GAP_FLAG_VARIABLE",
    "RTMA_CONTINUOUS_VARIABLES",
    "check_generated_config_structure",
    "check_flashnh_dataset_construction",
    "inspect_dynamic_nan_inventory",
    "run_structural_preflight",
]

_FORBIDDEN_KEY_SUBSTRINGS = (
    "partition", "gres", "gpu", "hostname", "username", "password", "token", "secret", "credential",
)
_TRACKED_SUBDIRS = ("config", "src", "scripts", "tests", "docs")

# Stage 1's two real gap-flag variables do not share one shape: precipitation
# has its own dedicated flag (mrms_qpe_1h_mm_gap : mrms_qpe_1h_mm, 1:1,
# already expressible by the generic "<name>+suffix" inference below), but
# all five RTMA continuous variables share a single flag (rtma_gap). No
# "<name>+suffix" pattern can express that many-to-one relationship, so it is
# named explicitly here rather than guessed. This is a fixed Stage 1 fact
# (the certified package's actual variable set), not a general mapping
# framework -- do not extend this beyond the real Stage 1 dynamic inputs.
RTMA_GAP_FLAG_VARIABLE = "rtma_gap"
RTMA_CONTINUOUS_VARIABLES = ("rtma_2t_K", "rtma_2d_K", "rtma_2sh_kgkg", "rtma_10u_ms", "rtma_10v_ms")


class NHStructuralPreflightError(ValueError):
    """Raised for a malformed generated bundle, missing preflight input, or
    an unreadable package artifact."""


@dataclass(frozen=True)
class DynamicNanInventory:
    """Read-only NaN/gap-consistency inventory over a package's per-basin
    NetCDF files (task item 6). Never modifies the package."""

    per_variable_nan_counts: dict
    per_basin_nan_counts: dict
    basin_masks_identical: bool
    nan_outside_documented_gaps: dict
    gap_flag_mismatches: dict


def _iter_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _iter_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_keys(item)


def _check_no_forbidden_keys(report: AuditReport, obj, label: str) -> None:
    bad = sorted({k for k in _iter_keys(obj) if any(f in k.lower() for f in _FORBIDDEN_KEY_SUBSTRINGS)})
    if bad:
        report.error(f"no_forbidden_keys[{label}]", f"forbidden key(s) found: {bad}")
    else:
        report.ok(f"no_forbidden_keys[{label}]")


def _check_output_location_safety(report: AuditReport, generated_dir, repo_root) -> None:
    generated_dir = Path(generated_dir).resolve()
    if repo_root is None:
        report.warn("output_outside_tracked_paths", "repo_root not provided; skipped")
        return
    repo_root = Path(repo_root).resolve()
    for sub in _TRACKED_SUBDIRS:
        tracked = repo_root / sub
        try:
            generated_dir.relative_to(tracked)
        except ValueError:
            continue
        report.error(
            "output_outside_tracked_paths",
            f"generated output dir {generated_dir} is inside tracked path {tracked}",
        )
        return
    report.ok("output_outside_tracked_paths", f"{generated_dir} is not inside any tracked source directory")


def _find_basin_netcdf(package_root, basin) -> "Path | None":
    ts_dir = Path(package_root) / "time_series"
    for suffix in (".nc", ".nc4"):
        candidate = ts_dir / f"{basin}{suffix}"
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Layer 1: file-only structural checks (no NeuralHydrology import)
# ---------------------------------------------------------------------------

def check_generated_config_structure(
    report: AuditReport,
    *,
    generated_dir,
    package_root,
    expected_basin_count: int,
    expected_seq_length: int,
    expected_target_variable: str,
    expected_dynamic_inputs: list,
    expected_static_column_count: int,
    expected_predict_last_n,
    expected_dates: dict,
    raw_target_variable: str = "qobs_m3s",
    repo_root=None,
) -> dict:
    """File-only checks against a generated config bundle + the package it
    was rendered against. Returns the parsed ``{"config": ..., "manifest":
    ...}`` for callers that want to inspect more. Records checks on
    ``report``; never raises for a failed *check* (only for a structurally
    unreadable input, via ``NHStructuralPreflightError``)."""
    generated_dir = Path(generated_dir)
    package_root = Path(package_root)

    if package_root.is_dir():
        report.ok("package_root_exists")
    else:
        report.error("package_root_exists", f"package root not found: {package_root}")

    required = {
        "config.yaml": generated_dir / "config.yaml",
        "generation_manifest.json": generated_dir / "generation_manifest.json",
        "train_basins.txt": generated_dir / "train_basins.txt",
        "validation_basins.txt": generated_dir / "validation_basins.txt",
        "test_basins.txt": generated_dir / "test_basins.txt",
    }
    for label, p in required.items():
        if p.is_file():
            report.ok(f"generated_output_exists[{label}]")
        else:
            report.error(f"generated_output_exists[{label}]", f"missing: {p}")
    if report.error_count:
        raise NHStructuralPreflightError(
            f"cannot continue structural preflight: {report.failed_messages()}"
        )

    raw_cfg = yaml.safe_load(required["config.yaml"].read_text(encoding="utf-8"))
    manifest = json.loads(required["generation_manifest.json"].read_text(encoding="utf-8"))

    _check_no_forbidden_keys(report, raw_cfg, "config.yaml")
    _check_no_forbidden_keys(report, manifest, "generation_manifest.json")
    _check_output_location_safety(report, generated_dir, repo_root)

    if raw_cfg.get("dataset") == FLASHNH_DATASET_KEY:
        report.ok("dataset_key")
    else:
        report.error("dataset_key", f"expected dataset == {FLASHNH_DATASET_KEY!r}, got {raw_cfg.get('dataset')!r}")

    if "nan_handling_method" in raw_cfg:
        report.error("nan_handling_method_absent", "nan_handling_method must be unset for the hard-exclusion baseline")
    else:
        report.ok("nan_handling_method_absent")

    if raw_cfg.get("seq_length") == expected_seq_length:
        report.ok("seq_length_exact")
    else:
        report.error("seq_length_exact", f"expected seq_length == {expected_seq_length}, got {raw_cfg.get('seq_length')!r}")

    if raw_cfg.get("predict_last_n") == expected_predict_last_n:
        report.ok("predict_last_n_documented", f"predict_last_n == {expected_predict_last_n!r}")
    else:
        report.error(
            "predict_last_n_documented",
            f"expected predict_last_n == {expected_predict_last_n!r}, got {raw_cfg.get('predict_last_n')!r}",
        )

    target_variables = raw_cfg.get("target_variables")
    if target_variables == [expected_target_variable]:
        report.ok("target_variable_exact")
    else:
        report.error(
            "target_variable_exact",
            f"expected target_variables == [{expected_target_variable!r}], got {target_variables!r}",
        )
    if raw_target_variable in (target_variables or []):
        report.error("raw_target_not_configured", f"{raw_target_variable!r} must never appear in target_variables")
    else:
        report.ok("raw_target_not_configured")

    if list(raw_cfg.get("dynamic_inputs") or []) == list(expected_dynamic_inputs):
        report.ok("dynamic_inputs_exact_order")
    else:
        report.error(
            "dynamic_inputs_exact_order",
            f"expected dynamic_inputs == {list(expected_dynamic_inputs)}, got {raw_cfg.get('dynamic_inputs')!r}",
        )

    static_attrs = raw_cfg.get("static_attributes") or []
    if len(static_attrs) == expected_static_column_count:
        report.ok("static_attribute_count_exact", f"{len(static_attrs)} columns")
    else:
        report.error(
            "static_attribute_count_exact",
            f"expected {expected_static_column_count} static columns, got {len(static_attrs)}",
        )

    date_mismatches = {
        key: (raw_cfg.get(key), expected)
        for key, expected in expected_dates.items()
        if raw_cfg.get(key) != expected
    }
    if date_mismatches:
        report.error("dates_exact", f"date mismatch(es): {date_mismatches}")
    else:
        report.ok("dates_exact")

    basin_lists = {}
    for label, path in (
        ("train", required["train_basins.txt"]),
        ("validation", required["validation_basins.txt"]),
        ("test", required["test_basins.txt"]),
    ):
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        basin_lists[label] = lines
        if len(lines) == expected_basin_count and len(set(lines)) == expected_basin_count:
            report.ok(f"basin_membership_count[{label}]", f"{len(lines)} basins")
        else:
            report.error(
                f"basin_membership_count[{label}]",
                f"expected {expected_basin_count} unique basins, got {len(lines)} ({len(set(lines))} unique)",
            )

    train_set, val_set, test_set = set(basin_lists["train"]), set(basin_lists["validation"]), set(basin_lists["test"])
    if train_set == val_set == test_set:
        report.ok("basin_membership_identical_across_periods")
    else:
        report.error(
            "basin_membership_identical_across_periods",
            "train/validation/test basin lists are not identical (temporal-only separation requires identical basins)",
        )

    missing_nc = [b for b in sorted(train_set) if _find_basin_netcdf(package_root, b) is None]
    if missing_nc:
        report.error("basin_netcdf_present", f"basin(s) with no time-series NetCDF: {missing_nc}")
    else:
        report.ok("basin_netcdf_present", f"{len(train_set)} basin(s) all have a time-series NetCDF")

    gap_path = package_root / "masks" / "gap_timestamps.json"
    try:
        gap_timestamps = load_gap_timestamps_json(gap_path)
        report.ok("gap_timestamp_artifact_parses", f"{len(gap_timestamps)} gap timestamp(s)")
    except GapMaskIOError as exc:
        report.error("gap_timestamp_artifact_parses", str(exc))

    package_manifest_path = package_root / "manifests" / "package_manifest.json"
    if package_manifest_path.is_file():
        package_manifest = json.loads(package_manifest_path.read_text(encoding="utf-8"))
        recorded_identity = manifest.get("package_manifest_identity", {})
        live_identity = {
            "schema_name": package_manifest.get("schema_name"),
            "schema_version": package_manifest.get("schema_version"),
            "package_role": package_manifest.get("package_role"),
            "basin_count": package_manifest.get("basin_count"),
            "static_model_input_columns_sha256": package_manifest.get("static_model_input_columns_sha256"),
        }
        if recorded_identity == live_identity:
            report.ok("package_manifest_identity_consistent")
        else:
            report.error(
                "package_manifest_identity_consistent",
                f"generation manifest's recorded identity {recorded_identity} != live package manifest {live_identity}",
            )

        dynamic_vars = set(package_manifest.get("dynamic_variables", []))
        missing_dynamic = [v for v in expected_dynamic_inputs if v not in dynamic_vars]
        if missing_dynamic:
            report.error("package_has_required_dynamic_variables", f"missing from package manifest: {missing_dynamic}")
        else:
            report.ok("package_has_required_dynamic_variables")

        # Basin-list exact-equality contract: package manifest basin_ids,
        # generation manifest basin_ids, and the train/validation/test basin
        # files must all agree exactly. The generator normalizes+sorts
        # package_manifest["basin_ids"] to derive generation_manifest["basin_ids"]
        # and every basin file (see nh_config_generation.py::validate_basin_membership
        # / write_generated_config), so re-deriving the same normalized+sorted
        # order from the package manifest here and comparing it against all
        # four other sources catches any divergence introduced after
        # generation (e.g. a hand-edited or stale basin file) that a mere
        # train==validation==test check would miss.
        try:
            package_basin_ids = sorted(normalize_staid(b) for b in package_manifest.get("basin_ids", []))
        except (TypeError, ValueError) as exc:
            report.error("basin_ids_exact_equality_all_sources", f"could not normalize package manifest basin_ids: {exc}")
            package_basin_ids = None

        if package_basin_ids is not None:
            generation_basin_ids = list(manifest.get("basin_ids", []))
            sources = {
                "package_manifest": package_basin_ids,
                "generation_manifest": generation_basin_ids,
                "train_basins.txt": basin_lists["train"],
                "validation_basins.txt": basin_lists["validation"],
                "test_basins.txt": basin_lists["test"],
            }
            mismatched = sorted(label for label, ids in sources.items() if ids != package_basin_ids)
            if mismatched:
                report.error(
                    "basin_ids_exact_equality_all_sources",
                    f"basin ID list(s) not identical (order-preserving) to the normalized+sorted package "
                    f"manifest basin_ids: {mismatched}",
                )
            else:
                report.ok(
                    "basin_ids_exact_equality_all_sources",
                    f"{len(package_basin_ids)} basin(s) identical, same order, across package manifest, "
                    "generation manifest, and train/validation/test lists",
                )
    else:
        report.error("package_manifest_identity_consistent", f"package manifest not found: {package_manifest_path}")

    return {"config": raw_cfg, "manifest": manifest}


# ---------------------------------------------------------------------------
# Layer 2: real-NH dataset-construction checks
# ---------------------------------------------------------------------------

def _scalers_equal(a, b) -> bool:
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        va, vb = a[k], b[k]
        equals = getattr(va, "equals", None)
        if callable(equals):
            if not va.equals(vb):
                return False
        elif va != vb:
            return False
    return True


def check_flashnh_dataset_construction(report: AuditReport, config_path, *, register: bool = True) -> dict:
    """Instantiate real ``FlashNHDataset`` train/validation/test instances
    against ``config_path`` and record scaler/finiteness/reuse checks.

    Intended for synthetic-fixture use in this task. The same function is
    directly reusable, unmodified, against a real package on a system with
    NeuralHydrology + real data available (e.g. Moriah) -- not executed
    there by this task.
    """
    if register:
        register_flashnh_dataset()
    from neuralhydrology.datasetzoo import get_dataset
    from neuralhydrology.utils.config import Config
    import torch

    cfg = Config(Path(config_path))
    if cfg.dataset == FLASHNH_DATASET_KEY:
        report.ok("dataset_registered_key")
    else:
        report.error("dataset_registered_key", f"cfg.dataset == {cfg.dataset!r}, expected {FLASHNH_DATASET_KEY!r}")

    # NH mechanics, not a scientific decision: a train-period BaseDataset
    # persists its id-to-int mapping and scaler under cfg.train_dir, which is
    # normally created by BaseTrainer.initialize_training() before dataset
    # construction. Construction-only checks here never run a trainer, so
    # this directory is set up directly (mirrors tests/_nh_synthetic.py's
    # prepare_run_dirs helper).
    if cfg.train_dir is None:
        run_dir = Path(cfg.run_dir) if cfg.run_dir is not None else Path(config_path).resolve().parent / "runs"
        train_dir = run_dir / "train_data"
        train_dir.mkdir(parents=True, exist_ok=True)
        cfg.train_dir = train_dir

    try:
        # NH 1.13 mechanics, not a scientific decision: BaseDataset.__init__'s
        # ``scaler`` parameter defaults to a *mutable* ``{}`` that Python
        # creates once and shares across every call site that omits it. NH's
        # own normal call pattern (get_dataset(..., is_train=True) with no
        # scaler=) relies on that default starting empty for the process's
        # first train-period construction; any later train-period
        # construction in the same process inherits the same, by-then
        # already-populated dict, so its own ``not scaler`` check goes False
        # and NH silently reuses a previous, unrelated dataset's scaler
        # instead of computing its own -- corrupting normalization and, via
        # xarray's intersecting arithmetic, silently dropping any dynamic
        # input/target column absent from that stale scaler. Passing an
        # explicit fresh dict here avoids relying on -- or polluting -- that
        # process-global default, independent of how many other train-period
        # datasets this process has already constructed.
        train_ds = get_dataset(cfg=cfg, is_train=True, period="train", scaler={})
    except Exception as exc:  # noqa: BLE001 -- surfaced as a report failure, not a crash
        report.error("train_dataset_construction", str(exc))
        return {}
    report.ok("train_dataset_construction")

    center_da = train_ds.scaler["xarray_feature_center"]
    scale_da = train_ds.scaler["xarray_feature_scale"]
    nonfinite_vars = []
    for name in cfg.dynamic_inputs:
        try:
            c = float(center_da[name].item())
            s = float(scale_da[name].item())
        except Exception as exc:  # noqa: BLE001
            report.error("scaler_present", f"could not read scaler entry for {name!r}: {exc}")
            continue
        if not (np.isfinite(c) and np.isfinite(s)):
            nonfinite_vars.append(name)
    if nonfinite_vars:
        report.error("scaler_finite", f"non-finite scaler center/scale for: {nonfinite_vars}")
    else:
        report.ok("scaler_finite")

    try:
        val_ds = get_dataset(cfg=cfg, is_train=False, period="validation", scaler=train_ds.scaler)
        test_ds = get_dataset(cfg=cfg, is_train=False, period="test", scaler=train_ds.scaler)
    except Exception as exc:  # noqa: BLE001
        report.error("eval_dataset_construction", str(exc))
        return {"train": train_ds}
    report.ok("eval_dataset_construction")

    if _scalers_equal(train_ds.scaler, val_ds.scaler) and _scalers_equal(train_ds.scaler, test_ds.scaler):
        report.ok("scaler_reused_unchanged")
    else:
        report.error("scaler_reused_unchanged", "validation/test scaler differs from the training scaler")

    datasets = {"train": train_ds, "validation": val_ds, "test": test_ds}
    for label, ds in datasets.items():
        n_nonfinite = 0
        for i in range(len(ds)):
            sample = ds[i]
            for v in sample["x_d"].values():
                if not torch.isfinite(v).all():
                    n_nonfinite += 1
        if n_nonfinite:
            report.error(f"finite_admitted_batches[{label}]", f"{n_nonfinite} non-finite x_d tensor(s) among {len(ds)} sample(s)")
        else:
            report.ok(f"finite_admitted_batches[{label}]", f"{len(ds)} admitted sample(s), all finite")

        stats = getattr(ds, "flashnh_filter_stats", None)
        if stats is None:
            report.error(f"filter_stats_available[{label}]", "flashnh_filter_stats not populated")
        else:
            report.ok(f"filter_stats_available[{label}]", str(stats))

    sample0 = train_ds[0]
    y = sample0["y"]
    n_targets = y.shape[-1] if hasattr(y, "shape") else len(y)
    if n_targets == 1:
        report.ok("single_target_tensor")
    else:
        report.error("single_target_tensor", f"expected exactly one target in the y tensor, got size {n_targets}")

    return datasets


# ---------------------------------------------------------------------------
# Task item 6: read-only real-package NaN inventory (no NeuralHydrology import)
# ---------------------------------------------------------------------------

def inspect_dynamic_nan_inventory(
    package_root,
    *,
    dynamic_inputs,
    basin_ids,
    gap_flag_suffix: str = "_gap",
) -> DynamicNanInventory:
    """Read-only per-variable/per-basin dynamic-NaN inventory (task item 6).

    Reports per-variable NaN counts, per-basin NaN counts, whether all
    basins share an identical NaN mask, which (variable, basin) pairs have a
    NaN outside the documented ``masks/gap_timestamps.json`` set, and where
    a ``<var>_gap`` flag disagrees with the corresponding base variable's
    actual NaN positions. Never writes to ``package_root``.

    Intended to be run later against the real certified package on h2o (not
    executed there by this task); tested here only against synthetic
    fixtures. Deliberately does not repeat the "does an admitted FlashNHDataset
    sample contain a non-finite value" question -- that is
    :func:`check_flashnh_dataset_construction`'s concern, which requires
    constructing a real dataset; this function is read-only and NH-free.
    """
    package_root = Path(package_root)
    gap_path = package_root / "masks" / "gap_timestamps.json"
    try:
        gap_timestamps = set(load_gap_timestamps_json(gap_path))
    except GapMaskIOError as exc:
        raise NHStructuralPreflightError(str(exc)) from exc

    dynamic_inputs = list(dynamic_inputs)
    per_variable_nan_counts = {name: 0 for name in dynamic_inputs}
    per_basin_nan_masks: dict = {}
    nan_outside_documented_gaps: dict = {name: [] for name in dynamic_inputs}
    gap_flag_mismatches: dict = {}

    flag_pairs = {
        name: f"{name}{gap_flag_suffix}"
        for name in dynamic_inputs
        if not name.endswith(gap_flag_suffix) and f"{name}{gap_flag_suffix}" in dynamic_inputs
    }
    # RTMA_GAP_FLAG_VARIABLE covers all five RTMA continuous variables, not a
    # single "<name>+suffix" match -- see the module-level comment above.
    if RTMA_GAP_FLAG_VARIABLE in dynamic_inputs:
        for base_var in RTMA_CONTINUOUS_VARIABLES:
            if base_var in dynamic_inputs:
                flag_pairs[base_var] = RTMA_GAP_FLAG_VARIABLE

    for basin in basin_ids:
        nc_path = _find_basin_netcdf(package_root, basin)
        if nc_path is None:
            raise NHStructuralPreflightError(f"no time-series NetCDF found for basin {basin!r} under {package_root}")
        with xr.open_dataset(nc_path) as ds:
            df = ds.to_dataframe()
        basin_masks = {}
        for name in dynamic_inputs:
            if name not in df.columns:
                raise NHStructuralPreflightError(f"basin {basin!r}: dynamic input {name!r} missing from {nc_path}")
            is_nan = df[name].isna()
            per_variable_nan_counts[name] += int(is_nan.sum())
            basin_masks[name] = is_nan.to_numpy()
            outside = [ts for ts in df.index[is_nan.to_numpy()] if pd.Timestamp(ts) not in gap_timestamps]
            if outside:
                nan_outside_documented_gaps[name].append({"basin": basin, "count": len(outside)})
        per_basin_nan_masks[basin] = basin_masks

        for base_var, flag_var in flag_pairs.items():
            flag_is_set = df[flag_var].fillna(0).astype(bool).to_numpy()
            mismatch = int(np.sum(basin_masks[base_var] != flag_is_set))
            if mismatch:
                gap_flag_mismatches.setdefault(base_var, {})[basin] = mismatch

    basin_list = list(per_basin_nan_masks.keys())
    basin_masks_identical = True
    if len(basin_list) > 1:
        reference = per_basin_nan_masks[basin_list[0]]
        for other in basin_list[1:]:
            if any(not np.array_equal(reference[name], per_basin_nan_masks[other][name]) for name in dynamic_inputs):
                basin_masks_identical = False
                break

    return DynamicNanInventory(
        per_variable_nan_counts=per_variable_nan_counts,
        per_basin_nan_counts={
            basin: {name: int(mask.sum()) for name, mask in masks.items()}
            for basin, masks in per_basin_nan_masks.items()
        },
        basin_masks_identical=basin_masks_identical,
        nan_outside_documented_gaps={k: v for k, v in nan_outside_documented_gaps.items() if v},
        gap_flag_mismatches=gap_flag_mismatches,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_structural_preflight(
    *,
    generated_dir,
    package_root,
    expected_basin_count: int,
    expected_seq_length: int,
    expected_target_variable: str,
    expected_dynamic_inputs: list,
    expected_static_column_count: int,
    expected_predict_last_n,
    expected_dates: dict,
    repo_root=None,
    run_dataset_construction: bool = True,
) -> AuditReport:
    """Compose Layer 1 (always) and Layer 2 (optional, real-NH) checks into
    one :class:`AuditReport`. Never runs training."""
    report = AuditReport()
    check_generated_config_structure(
        report,
        generated_dir=generated_dir,
        package_root=package_root,
        expected_basin_count=expected_basin_count,
        expected_seq_length=expected_seq_length,
        expected_target_variable=expected_target_variable,
        expected_dynamic_inputs=expected_dynamic_inputs,
        expected_static_column_count=expected_static_column_count,
        expected_predict_last_n=expected_predict_last_n,
        expected_dates=expected_dates,
        repo_root=repo_root,
    )
    if run_dataset_construction and report.error_count == 0:
        config_path = Path(generated_dir) / "config.yaml"
        check_flashnh_dataset_construction(report, config_path)
    return report
