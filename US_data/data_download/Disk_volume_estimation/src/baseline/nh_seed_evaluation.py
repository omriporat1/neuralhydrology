"""Orchestration for evaluating a Stage 1 full-population seed run
(``docs/decision_log.md``'s 2026-07-25 "initial full-population seed training
profile" entry): reading NH's own per-checkpoint evaluation pickles and
turning them into raw-space (m^3/s) metrics via
:mod:`src.baseline.nh_raw_space_evaluation`, and preparing the small
"external-scaler evaluation run directory" a spatial-holdout evaluation needs.

Two distinct concerns live here, kept in one module because both are thin
orchestration around already-established conventions (never modeling logic,
never a new scaler fit):

1. **Reading NH's evaluation pickles.** Mirrors
   ``nh_evaluation_check.py``'s ``<period>_results.p`` reading convention
   exactly (``run_dir / period / f"model_epoch{epoch:03d}" /
   f"{period}_results.p"``, ``xr_ds[f"{target_variable}_obs"/"_sim"]``,
   "admitted" = finite observation) so the two modules never silently drift
   apart on what a period's results look like.

2. **Preparing a spatial-holdout evaluation run directory.** NeuralHydrology's
   ``Tester`` always reloads its scaler from ``run_dir/train_data/
   train_data_scaler.yml`` and its basin/period settings from
   ``run_dir/config.yml`` -- both read from the SAME run_dir. Evaluating the
   spatial-holdout population with the already-trained development checkpoint
   therefore needs a run_dir that combines the holdout bundle's own
   config.yaml (correct test-period basin list, dates) with the development
   run's checkpoint and its train_data_scaler.yml, copied byte-for-byte
   (never refit -- see ``nh_config_generation.write_generated_config``'s
   ``TEST_ONLY_DO_NOT_TRAIN.txt`` text and
   ``nh_structural_preflight.check_flashnh_external_scaler_test_construction``,
   which already established and tested that reusing an external scaler this
   way reproduces it unchanged at the ``FlashNHDataset`` level).
   :func:`prepare_external_scaler_eval_run_dir` builds exactly that directory
   as a plain file-copy operation (no torch/NH import required, so it is
   fully unit-testable locally), then the caller runs the ordinary
   ``scripts/run_stage1_nh.py eval <out_run_dir> --period test --epoch N``
   against it -- no new evaluation code path, no custom inference loop.
"""
from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from .nh_config_generation import HOLDOUT_MARKER_FILENAME
from .nh_raw_space_evaluation import (
    DEFAULT_MAX_RELATIVE_MAD,
    DEFAULT_MIN_AREA_SAMPLES,
    RawSpaceEvaluationError,
    aggregate_raw_space_metrics,
    derive_basin_area_km2_from_netcdf,
    evaluate_basin_raw_space,
    pooled_raw_space_metrics,
)
from .package_audit import sha256_file

__all__ = [
    "NHSeedEvaluationError",
    "weight_stem",
    "load_period_results",
    "basin_netcdf_path",
    "raw_space_metrics_for_run_period",
    "require_holdout_bundle",
    "prepare_external_scaler_eval_run_dir",
]


class NHSeedEvaluationError(Exception):
    """Raised for a setup/contract problem (missing run artifact, wrong bundle
    type, pickle format mismatch), never for an ordinary poor-skill outcome."""


def weight_stem(epoch: int) -> str:
    return f"model_epoch{epoch:03d}"


def load_period_results(run_dir, period: str, epoch: int) -> dict:
    """Reads ``run_dir/period/model_epoch{epoch:03d}/{period}_results.p``,
    the same pickle path convention ``nh_evaluation_check.py`` uses."""
    run_dir = Path(run_dir)
    result_pickle = run_dir / period / weight_stem(epoch) / f"{period}_results.p"
    if not result_pickle.exists():
        raise NHSeedEvaluationError(f"missing {period} results pickle: {result_pickle}")
    with open(result_pickle, "rb") as fh:
        return pickle.load(fh)


def basin_netcdf_path(package_root, basin_id: str) -> Path:
    """The certified package's per-basin time-series file path convention
    (confirmed via ``package_builder.py``): ``time_series/<basin_id>.nc``."""
    return Path(package_root) / "time_series" / f"{basin_id}.nc"


def raw_space_metrics_for_run_period(
    *,
    run_dir,
    period: str,
    epoch: int,
    package_root,
    target_variable: str,
    lead_hours: int,
    basin_ids: Optional[Sequence[str]] = None,
    min_area_samples: int = DEFAULT_MIN_AREA_SAMPLES,
    max_relative_mad: float = DEFAULT_MAX_RELATIVE_MAD,
    compute_pooled: bool = False,
) -> dict:
    """Reads one completed (period, epoch)'s NH evaluation pickle, self-derives
    each basin's area from the package's own data, converts obs/sim back to
    raw m^3/s, and returns per-basin + aggregate raw-space metrics.

    Basins with an inconsistent or under-sampled area derivation are excluded
    from the metrics but listed under ``"area_derivation_excluded"`` with
    their reason -- never silently dropped without a trace.

    ``compute_pooled``, if set, additionally attaches
    ``result["pooled"]`` -- a diagnostic-only pooled-sample metric set (see
    :func:`src.baseline.nh_raw_space_evaluation.pooled_raw_space_metrics`),
    never the primary median-per-basin metric. Defaults to ``False`` so
    existing callers are unaffected.
    """
    results = load_period_results(run_dir, period, epoch)
    actual_basin_ids = sorted(results.keys())
    if basin_ids is not None:
        wanted = set(basin_ids)
        missing = sorted(wanted - set(actual_basin_ids))
        if missing:
            raise NHSeedEvaluationError(
                f"{period}/epoch{epoch}: requested basin(s) not present in results: {missing}"
            )
        actual_basin_ids = sorted(wanted)

    obs_key = f"{target_variable}_obs"
    sim_key = f"{target_variable}_sim"

    per_basin_metrics = []
    area_derivation_excluded = []
    pooled_obs_arrays = []
    pooled_sim_arrays = []
    for basin_id in actual_basin_ids:
        freq_results = results[basin_id]
        for freq, freq_result in freq_results.items():
            xr_ds = freq_result.get("xr")
            if xr_ds is None or obs_key not in xr_ds.data_vars or sim_key not in xr_ds.data_vars:
                area_derivation_excluded.append(
                    {"basin_id": basin_id, "freq": freq, "reason": "missing xr result or target data vars"}
                )
                continue

            nc_path = basin_netcdf_path(package_root, basin_id)
            try:
                area_result = derive_basin_area_km2_from_netcdf(
                    nc_path,
                    basin_id=basin_id,
                    target_variable=target_variable,
                    lead_hours=lead_hours,
                    min_samples=min_area_samples,
                    max_relative_mad=max_relative_mad,
                )
            except RawSpaceEvaluationError as exc:
                area_derivation_excluded.append({"basin_id": basin_id, "freq": freq, "reason": str(exc)})
                continue
            if not area_result.consistent:
                area_derivation_excluded.append(
                    {
                        "basin_id": basin_id,
                        "freq": freq,
                        "reason": (
                            f"area derivation inconsistent: relative_mad={area_result.relative_mad:.6g} "
                            f"> max_relative_mad={max_relative_mad:.6g}"
                        ),
                    }
                )
                continue

            obs_mm_per_h = xr_ds[obs_key].values.reshape(-1)
            sim_mm_per_h = xr_ds[sim_key].values.reshape(-1)
            basin_metrics = evaluate_basin_raw_space(
                basin_id=basin_id,
                obs_mm_per_h=obs_mm_per_h,
                sim_mm_per_h=sim_mm_per_h,
                area_km2=area_result.area_km2,
                return_admitted_arrays=compute_pooled,
            )
            if compute_pooled:
                pooled_obs_arrays.append(basin_metrics.pop("_admitted_obs_m3s"))
                pooled_sim_arrays.append(basin_metrics.pop("_admitted_sim_m3s"))
            basin_metrics["freq"] = freq
            basin_metrics["area_n_samples"] = area_result.n_samples
            basin_metrics["area_relative_mad"] = area_result.relative_mad
            per_basin_metrics.append(basin_metrics)

    aggregate = aggregate_raw_space_metrics(per_basin_metrics) if per_basin_metrics else {
        "n_basins": 0,
        "n_admitted_total": 0,
        "n_sim_nonfinite_at_admitted_total": 0,
        "metrics": {},
    }

    result = {
        "run_dir": str(run_dir),
        "period": period,
        "epoch": epoch,
        "target_variable": target_variable,
        "lead_hours": lead_hours,
        "n_basins_requested": len(actual_basin_ids),
        "n_basins_evaluated": len(per_basin_metrics),
        "n_basins_area_excluded": len(area_derivation_excluded),
        "area_derivation_excluded": area_derivation_excluded,
        "per_basin": per_basin_metrics,
        "aggregate": aggregate,
    }
    if compute_pooled:
        result["pooled"] = pooled_raw_space_metrics(pooled_obs_arrays, pooled_sim_arrays)
    return result


def require_holdout_bundle(generated_dir) -> None:
    """Inverse of :func:`src.baseline.nh_config_generation.raise_if_holdout_bundle`
    -- raises if ``generated_dir`` is NOT a spatial-holdout (test-only) bundle.
    Used to guard the external-scaler eval run_dir preparation, which must
    only ever be pointed at the holdout bundle (never the development one --
    the development bundle's own test period is evaluated directly via its
    own already-trained run_dir, with no external-scaler staging needed)."""
    marker_path = Path(generated_dir) / HOLDOUT_MARKER_FILENAME
    if not marker_path.is_file():
        raise NHSeedEvaluationError(
            f"{generated_dir} is not a spatial-holdout bundle (missing {marker_path}); "
            "external-scaler eval run_dir preparation must only be pointed at the holdout bundle"
        )


def prepare_external_scaler_eval_run_dir(
    *,
    development_run_dir,
    epoch: int,
    holdout_generated_dir,
    out_run_dir,
    force: bool = False,
) -> dict:
    """Builds a minimal NH-Tester-compatible run directory for evaluating the
    spatial-holdout population with the already-trained development
    checkpoint, reusing the development run's scaler byte-for-byte (never
    refit). Returns a manifest dict (also written to
    ``out_run_dir/EXTERNAL_SCALER_EVAL_MANIFEST.json``) recording exact
    provenance and sha256 of every copied artifact.

    Pure file I/O -- imports no NH/torch code, so this is fully testable
    locally with a fabricated development_run_dir/holdout_generated_dir.
    """
    development_run_dir = Path(development_run_dir)
    holdout_generated_dir = Path(holdout_generated_dir)
    out_run_dir = Path(out_run_dir)

    require_holdout_bundle(holdout_generated_dir)

    holdout_config_src = holdout_generated_dir / "config.yaml"
    checkpoint_src = development_run_dir / f"{weight_stem(epoch)}.pt"
    scaler_src = development_run_dir / "train_data" / "train_data_scaler.yml"
    for label, p in (
        ("holdout config.yaml", holdout_config_src),
        ("development checkpoint", checkpoint_src),
        ("development scaler", scaler_src),
    ):
        if not p.is_file():
            raise NHSeedEvaluationError(f"missing required {label}: {p}")

    if out_run_dir.exists():
        if not force:
            raise NHSeedEvaluationError(f"out_run_dir already exists (pass force=True to overwrite): {out_run_dir}")
        shutil.rmtree(out_run_dir)
    out_run_dir.mkdir(parents=True)
    (out_run_dir / "train_data").mkdir()

    config_dst = out_run_dir / "config.yml"
    checkpoint_dst = out_run_dir / f"{weight_stem(epoch)}.pt"
    scaler_dst = out_run_dir / "train_data" / "train_data_scaler.yml"

    shutil.copy2(holdout_config_src, config_dst)
    shutil.copy2(checkpoint_src, checkpoint_dst)
    shutil.copy2(scaler_src, scaler_dst)

    scaler_src_sha256 = sha256_file(scaler_src)
    scaler_dst_sha256 = sha256_file(scaler_dst)
    if scaler_src_sha256 != scaler_dst_sha256:
        raise NHSeedEvaluationError(
            f"scaler copy corrupted: source sha256 {scaler_src_sha256} != dest sha256 {scaler_dst_sha256}"
        )
    checkpoint_src_sha256 = sha256_file(checkpoint_src)
    checkpoint_dst_sha256 = sha256_file(checkpoint_dst)
    if checkpoint_src_sha256 != checkpoint_dst_sha256:
        raise NHSeedEvaluationError(
            f"checkpoint copy corrupted: source sha256 {checkpoint_src_sha256} != dest sha256 {checkpoint_dst_sha256}"
        )

    manifest = {
        "schema_name": "stage1_external_scaler_eval_run_dir_manifest",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_run_dir": str(development_run_dir),
        "epoch": epoch,
        "holdout_generated_dir": str(holdout_generated_dir),
        "out_run_dir": str(out_run_dir),
        "scaler_reused_unchanged": True,
        "scaler_sha256": scaler_dst_sha256,
        "checkpoint_sha256": checkpoint_dst_sha256,
        "config_yaml_sha256": sha256_file(config_dst),
    }
    manifest_path = out_run_dir / "EXTERNAL_SCALER_EVAL_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest
