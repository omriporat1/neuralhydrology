"""RD1-C4-E -- production results/evidence writer for the formal RD1-C4-B
hydrological review
(:func:`src.baseline.stage1_v2_12plus12_hydrological_consumer.assemble_rd1_hydrological_review`).

This module contains no scientific logic of its own. It:

1. authenticates the frozen 24-trial roster via
   :func:`src.baseline.rd1_c4_trial_authentication.authenticate_trial_roster`
   (the same receipt-derived roster D1 consumes -- ``run_dir``, ``best_epoch``,
   ``official_objective``, and identity are never caller-asserted);
2. calls the already-qualified
   :func:`~src.baseline.stage1_v2_12plus12_hydrological_consumer.assemble_rd1_hydrological_review`
   batch entry point, which is the sole source of every NSE/KGE/Q98 number
   this module writes out; and
3. serializes the already-qualified per-basin/per-trial numbers to flat,
   reviewable CSV/JSON evidence tables plus a manifest with SHA-256 hashes of
   every produced file.

It never reopens ``validation_results.p``, never recomputes a metric, and
never derives an identity independently of the qualified consumer/roster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fixed_support_contract_v2 import load_fixed_support_contract
from .rd1_c4_trial_authentication import TrialAuthenticationError, authenticate_trial_roster
from .stage1_v2_12plus12_hydrological_consumer import (
    HydrologicalConsumerError,
    HydrologicalReviewResult,
    Q98ConfigurationBasinDiagnostics,
    assemble_rd1_hydrological_review,
)

__all__ = [
    "ResultsRunnerError",
    "produce_rd1_c4_results",
    "write_rd1_c4_results",
    "build_parser",
    "main",
]

_FROZEN_QUANTILE_LEVELS = (1, 5, 25, 50, 75, 95, 99)


class ResultsRunnerError(ValueError):
    """Raised for a roster/authentication failure or a hydrological-consumer
    contract violation surfaced while producing results. Never raised for an
    ordinary poor-skill scientific outcome, and never raised to paper over a
    real contract violation from the qualified consumer."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _trial_identity_columns(result) -> dict:
    source = result.best_epoch_source
    return {
        "trial_id": source.trial_id,
        "search_arm": source.search_arm,
        "configuration_id": source.configuration_id,
        "proposal_id": source.proposal_id,
        "proposal_order": source.proposal_order,
        "best_epoch": source.best_epoch,
        "official_objective": source.official_objective,
        "rescored_objective": result.rescored_objective,
    }


def per_basin_metric_rows(review: HydrologicalReviewResult) -> list:
    """Flatten every trial's already-qualified ``fixed_support_result["per_basin"]``
    rows (raw-space NSE/KGE/components from
    :mod:`src.baseline.nh_raw_space_evaluation`) into one flat table, tagged
    with trial/arm identity. Every value is passed through as-is -- no metric
    is recomputed here."""
    rows = []
    for trial_id in sorted(review.results_by_trial_id):
        result = review.results_by_trial_id[trial_id]
        meta = _trial_identity_columns(result)
        for basin_row in result.fixed_support_result["per_basin"]:
            row = dict(meta)
            for key, value in basin_row.items():
                if key.startswith("_") or key == "freq":
                    continue
                row[key] = value
            rows.append(row)
    return rows


def q98_diagnostic_rows(review: HydrologicalReviewResult) -> list:
    """Flatten every trial's already-qualified
    :class:`~src.baseline.stage1_v2_12plus12_hydrological_consumer.Q98ConfigurationBasinDiagnostics`
    into one flat table, tagged with trial/arm identity."""
    rows = []
    for trial_id in sorted(review.results_by_trial_id):
        result = review.results_by_trial_id[trial_id]
        source = result.best_epoch_source
        for basin_id in sorted(result.q98_diagnostics_by_basin):
            diag = result.q98_diagnostics_by_basin[basin_id]
            row = {
                "trial_id": trial_id,
                "search_arm": source.search_arm,
                "configuration_id": source.configuration_id,
            }
            row.update(asdict(diag))
            rows.append(row)
    return rows


def basin_distribution_summary_rows(review: HydrologicalReviewResult) -> list:
    """One row per trial: the already-qualified RD1-C3 frozen seven-quantile
    NSE core (:class:`~src.baseline.stage1_v2_12plus12_basin_analysis.FrozenQuantileCore`)
    plus finite/nonfinite basin coverage, tagged with trial/arm identity."""
    rows = []
    for trial_id in sorted(review.results_by_trial_id):
        result = review.results_by_trial_id[trial_id]
        source = result.best_epoch_source
        bd = result.basin_distribution
        row = {
            "trial_id": trial_id,
            "search_arm": source.search_arm,
            "configuration_id": bd.configuration_id,
            "n_total_basins": bd.n_total_basins,
            "n_finite_basins": bd.n_finite_basins,
            "n_nonfinite_basins": bd.n_nonfinite_basins,
        }
        row.update({f"nse_{key}": value for key, value in bd.frozen_core.to_dict().items()})
        rows.append(row)
    return rows


def canonical_q98_facts_rows(review: HydrologicalReviewResult) -> list:
    """One row per basin: the shared candidate-independent Q98 threshold /
    high-flow-sample-count / observed-peak facts every trial's Q98
    diagnostics were computed against."""
    rows = []
    for basin_id in sorted(review.canonical_q98_facts_by_basin):
        facts = review.canonical_q98_facts_by_basin[basin_id]
        rows.append(
            {
                "basin_id": facts.basin_id,
                "n_admitted": facts.n_admitted,
                "q98_threshold": facts.q98_threshold,
                "n_high_flow": facts.n_high_flow,
                "observed_peak_value": facts.observed_peak_value,
            }
        )
    return rows


def coverage_summary(review: HydrologicalReviewResult) -> dict:
    return {
        name: {
            "n_total": coverage.n_total,
            "n_finite": coverage.n_finite,
            "n_unavailable": coverage.n_unavailable,
        }
        for name, coverage in review.coverage.items()
    }


def write_rd1_c4_results(review: HydrologicalReviewResult, out_dir: "str | Path") -> dict:
    """Write the complete RD1-C4 results evidence bundle to ``out_dir`` and
    return a manifest mapping each produced file's repo-relative-style name
    to its size and SHA-256. Refuses to overwrite an existing non-empty
    ``out_dir`` (fresh output only, mirroring the D1 reducer's
    fresh-``D1_OUT_DIR`` discipline)."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ResultsRunnerError(f"out_dir {out_dir} already exists and is non-empty -- refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_basin = per_basin_metric_rows(review)
    per_basin_fields = list(_trial_identity_columns(next(iter(review.results_by_trial_id.values()))).keys())
    for row in per_basin:
        for key in row:
            if key not in per_basin_fields:
                per_basin_fields.append(key)
    _write_csv(out_dir / "per_basin_metrics.csv", per_basin, per_basin_fields)

    q98_rows = q98_diagnostic_rows(review)
    q98_fields = ["trial_id", "search_arm", "configuration_id"] + [
        f.name for f in fields(Q98ConfigurationBasinDiagnostics)
    ]
    _write_csv(out_dir / "q98_diagnostics.csv", q98_rows, q98_fields)

    dist_rows = basin_distribution_summary_rows(review)
    dist_fields = list(dist_rows[0].keys()) if dist_rows else []
    _write_csv(out_dir / "basin_distribution_summary.csv", dist_rows, dist_fields)

    facts_rows = canonical_q98_facts_rows(review)
    facts_fields = list(facts_rows[0].keys()) if facts_rows else []
    _write_csv(out_dir / "canonical_q98_facts.csv", facts_rows, facts_fields)

    identity = {
        "n_configurations": review.n_configurations,
        "n_bayesian": review.n_bayesian,
        "n_random_control": review.n_random_control,
        "n_basins": review.n_basins,
        "contract_id": review.contract_id,
        "trial_ids": sorted(review.results_by_trial_id),
        "coverage": coverage_summary(review),
    }
    with open(out_dir / "review_identity.json", "w", encoding="utf-8") as handle:
        json.dump(identity, handle, indent=2, sort_keys=True)

    manifest = {}
    for produced in sorted(out_dir.iterdir()):
        if produced.is_file():
            manifest[produced.name] = {
                "size_bytes": produced.stat().st_size,
                "sha256": _sha256_path(produced),
            }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def produce_rd1_c4_results(
    *,
    trial_list_path: "str | Path",
    contract_path: "str | Path",
    package_root: "str | Path",
    out_dir: "str | Path",
) -> dict:
    """End-to-end production entry point: authenticate the frozen 24-trial
    roster against the frozen fixed-support contract, run the formal
    RD1-C4-B batch consumer over it, and write the results evidence bundle.
    Raises :class:`ResultsRunnerError` on any roster/authentication or
    hydrological-consumer contract violation -- never silently narrows the
    population or substitutes a partial result."""
    contract = load_fixed_support_contract(contract_path)
    try:
        roster = authenticate_trial_roster(trial_list_path=Path(trial_list_path), contract=contract)
    except TrialAuthenticationError as exc:
        raise ResultsRunnerError(f"{trial_list_path}: trial roster is not receipt-qualified: {exc}") from exc

    best_epoch_sources = [target.best_epoch_source for target in roster]
    try:
        review = assemble_rd1_hydrological_review(
            best_epoch_sources=best_epoch_sources, package_root=package_root, contract=contract
        )
    except HydrologicalConsumerError as exc:
        raise ResultsRunnerError(f"formal RD1-C4-B assembly failed: {exc}") from exc

    return write_rd1_c4_results(review, out_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rd1_c4_results",
        description="RD1-C4 production results/evidence writer (no scientific logic; serializes the "
        "qualified assemble_rd1_hydrological_review() output).",
    )
    parser.add_argument("--trial-list", required=True, help="the same frozen 24-trial roster JSON D1 used")
    parser.add_argument("--contract", required=True, help="frozen fixed-support contract JSON")
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = produce_rd1_c4_results(
            trial_list_path=args.trial_list,
            contract_path=args.contract,
            package_root=args.package_root,
            out_dir=args.out_dir,
        )
    except ResultsRunnerError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    print(f"wrote {len(manifest)} files to {args.out_dir}", flush=True)
    for name in sorted(manifest):
        print(f"  {name}: {manifest[name]['size_bytes']} bytes sha256={manifest[name]['sha256']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
