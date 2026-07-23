"""Evidence checks for a completed NeuralHydrology 1.13 evaluation invocation
(``scripts/run_stage1_nh.py eval ...``), run against an already-completed
training run directory. Construction/training/evaluation itself lives
entirely in ``scripts/run_stage1_nh.py`` and installed NeuralHydrology; this
module only reads back what was written (the pickled per-basin/per-frequency
``xarray.Dataset`` results and the run's own ``config.yml``) and reports
whether it matches the expectations for a specific evaluation invocation
(target variable, epoch, basin membership, calendar year, finiteness, and
that no training artifact was overwritten).

Reuses :class:`src.baseline.package_audit.AuditReport` / ``sha256_file`` --
generic, non-scientific reporting helpers already used by the Stage 1
package auditor -- rather than declaring a second reporting dataclass.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .package_audit import AuditReport, sha256_file

__all__ = [
    "NHEvaluationCheckError",
    "check_period_evaluation",
    "check_training_artifacts_unchanged",
    "run_evaluation_check",
]


class NHEvaluationCheckError(Exception):
    """Raised for setup problems (missing run_dir/config.yml), not per-check failures."""


def _weight_stem(epoch: int) -> str:
    return f"model_epoch{epoch:03d}"


def _load_period_results(run_dir: Path, period: str, epoch: int) -> dict:
    result_pickle = run_dir / period / _weight_stem(epoch) / f"{period}_results.p"
    if not result_pickle.exists():
        raise NHEvaluationCheckError(f"missing {period} results pickle: {result_pickle}")
    with open(result_pickle, "rb") as fh:
        return pickle.load(fh)


def check_period_evaluation(
    report: AuditReport,
    *,
    run_dir: Path,
    period: str,
    epoch: int,
    target_variable: str,
    expected_basin_ids: Sequence[str],
    expected_year: int,
    expected_metric_names: Sequence[str],
) -> dict:
    """Check one evaluated period's ``<period>_results.p`` (and sibling
    ``<period>_metrics.csv``). Returns
    ``{"basin_count", "sample_count", "metric_names", "output_dir"}``.
    """
    prefix = f"evaluation[{period}]"
    period_dir = run_dir / period / _weight_stem(epoch)

    try:
        results = _load_period_results(run_dir, period, epoch)
    except NHEvaluationCheckError as exc:
        report.error(f"{prefix}.results_exist", str(exc))
        return {"basin_count": 0, "sample_count": 0, "metric_names": [], "output_dir": str(period_dir)}
    report.ok(f"{prefix}.results_exist", str(period_dir / f"{period}_results.p"))

    metrics_csv = period_dir / f"{period}_metrics.csv"
    if metrics_csv.exists():
        report.ok(f"{prefix}.metrics_csv_exists", str(metrics_csv))
    else:
        report.error(f"{prefix}.metrics_csv_exists", f"missing {metrics_csv}")

    actual_basin_ids = sorted(results.keys())
    expected_sorted = sorted(expected_basin_ids)
    if actual_basin_ids == expected_sorted:
        report.ok(f"{prefix}.basin_membership", f"{len(actual_basin_ids)} basins match expected set exactly")
    else:
        missing = sorted(set(expected_sorted) - set(actual_basin_ids))
        extra = sorted(set(actual_basin_ids) - set(expected_sorted))
        report.error(f"{prefix}.basin_membership", f"missing={missing} extra={extra}")

    obs_key = f"{target_variable}_obs"
    sim_key = f"{target_variable}_sim"
    sample_count = 0
    found_metric_names: set = set()

    for basin in actual_basin_ids:
        for freq, freq_results in results[basin].items():
            xr_ds = freq_results.get("xr")
            if xr_ds is None:
                report.error(f"{prefix}.xr_present[{basin}][{freq}]", "missing 'xr' key in results entry")
                continue
            if obs_key not in xr_ds.data_vars or sim_key not in xr_ds.data_vars:
                report.error(
                    f"{prefix}.target_variable_present[{basin}][{freq}]",
                    f"expected data vars {obs_key}/{sim_key}, got {sorted(xr_ds.data_vars)}",
                )
                continue

            obs = xr_ds[obs_key].values
            sim = xr_ds[sim_key].values
            if obs.shape != sim.shape:
                report.error(f"{prefix}.shape_match[{basin}][{freq}]", f"obs {obs.shape} vs sim {sim.shape}")
            else:
                report.ok(f"{prefix}.shape_match[{basin}][{freq}]", str(obs.shape))

            # "Admitted" samples are exactly those with a finite observation
            # (qobs-NaN hours are excluded from loss/metrics but the model
            # still emits an output for every window in the sequence).
            finite_mask = np.isfinite(obs)
            admitted = int(finite_mask.sum())
            sample_count += admitted
            if admitted and not np.all(np.isfinite(sim[finite_mask])):
                n_bad = int((~np.isfinite(sim[finite_mask])).sum())
                report.error(
                    f"{prefix}.predictions_finite[{basin}][{freq}]",
                    f"{n_bad} non-finite prediction(s) at {admitted} admitted (obs-finite) position(s)",
                )
            else:
                report.ok(f"{prefix}.predictions_finite[{basin}][{freq}]", f"{admitted} admitted samples")

            dates = pd.to_datetime(np.asarray(xr_ds["date"].values))
            bad_years = sorted({int(y) for y in dates.year if y != expected_year})
            if bad_years:
                report.error(
                    f"{prefix}.dates_in_expected_year[{basin}][{freq}]",
                    f"expected all dates in {expected_year}, found year(s) {bad_years}",
                )
            else:
                report.ok(f"{prefix}.dates_in_expected_year[{basin}][{freq}]", f"{len(dates)} date(s) all in {expected_year}")

            found_metric_names.update(key for key in freq_results.keys() if key != "xr")

    for metric in expected_metric_names:
        if metric in found_metric_names:
            report.ok(f"{prefix}.metric_present[{metric}]")
        else:
            report.error(
                f"{prefix}.metric_present[{metric}]",
                f"expected metric '{metric}' not found among {sorted(found_metric_names)}",
            )

    return {
        "basin_count": len(actual_basin_ids),
        "sample_count": sample_count,
        "metric_names": sorted(found_metric_names),
        "output_dir": str(period_dir),
    }


def check_training_artifacts_unchanged(
    report: AuditReport, *, run_dir: Path, pre_eval_sha256: Mapping[str, str]
) -> None:
    """Compares current sha256 of each ``run_dir``-relative path against the
    hash captured by the caller *before* any evaluation invocation ran."""
    for rel_path, expected_hash in pre_eval_sha256.items():
        actual_path = run_dir / rel_path
        check_id = f"training_artifact_unchanged[{rel_path}]"
        if not actual_path.exists():
            report.error(check_id, f"missing after evaluation: {actual_path}")
            continue
        actual_hash = sha256_file(actual_path)
        if actual_hash == expected_hash:
            report.ok(check_id)
        else:
            report.error(check_id, f"sha256 changed: expected {expected_hash}, got {actual_hash}")


def run_evaluation_check(
    *,
    run_dir: Path,
    epoch: int,
    expected_target_variable: str,
    expected_basin_count: int,
    expected_validation_year: int,
    expected_test_year: int,
    expected_metric_names: Sequence[str],
    pre_eval_sha256: Mapping[str, str],
) -> "tuple[AuditReport, dict]":
    """Runs every Section-5-style evidence check for one completed epoch's
    validation + test evaluation. Never runs training or evaluation itself.
    """
    from neuralhydrology.datautils.utils import load_basin_file
    from neuralhydrology.utils.config import Config

    run_dir = Path(run_dir)
    report = AuditReport()
    output: dict = {}

    cfg_path = run_dir / "config.yml"
    if not cfg_path.exists():
        report.error("config_exists", f"missing {cfg_path}")
        return report, output
    report.ok("config_exists", str(cfg_path))
    cfg = Config(cfg_path)

    if cfg.target_variables == [expected_target_variable]:
        report.ok("target_variable_matches_config", expected_target_variable)
    else:
        report.error(
            "target_variable_matches_config",
            f"expected [{expected_target_variable}], got {cfg.target_variables}",
        )

    checkpoint_path = run_dir / f"{_weight_stem(epoch)}.pt"
    if checkpoint_path.exists():
        report.ok("epoch_checkpoint_exists", str(checkpoint_path))
    else:
        report.error("epoch_checkpoint_exists", f"missing {checkpoint_path}")

    scaler_path = run_dir / "train_data" / "train_data_scaler.yml"
    if scaler_path.exists():
        report.ok("training_scaler_exists", str(scaler_path))
    else:
        report.error("training_scaler_exists", f"missing {scaler_path}")

    period_specs = [
        ("validation", expected_validation_year, cfg.validation_basin_file),
        ("test", expected_test_year, cfg.test_basin_file),
    ]
    for period, expected_year, basin_file in period_specs:
        expected_basin_ids = load_basin_file(basin_file)
        if len(expected_basin_ids) == expected_basin_count:
            report.ok(f"evaluation[{period}].expected_basin_count", str(expected_basin_count))
        else:
            report.error(
                f"evaluation[{period}].expected_basin_count",
                f"basin file {basin_file} has {len(expected_basin_ids)} basin(s), expected {expected_basin_count}",
            )

        output[period] = check_period_evaluation(
            report,
            run_dir=run_dir,
            period=period,
            epoch=epoch,
            target_variable=expected_target_variable,
            expected_basin_ids=expected_basin_ids,
            expected_year=expected_year,
            expected_metric_names=expected_metric_names,
        )

    check_training_artifacts_unchanged(report, run_dir=run_dir, pre_eval_sha256=pre_eval_sha256)

    return report, output
