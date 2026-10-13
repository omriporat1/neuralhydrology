"""RD1 P17/P18/P19 recovered-original-trial evaluation.

Session A correction task (2026-10-08): the original training artifacts for
P17/P18/P19 (bayesian_continuation_36) were found intact at the Flash-NH
REPO-ROOT .scratch_local paths (not the nested-project paths Round 1/2 of
this investigation mistakenly checked). This script produces the complete
qualified 400-basin fixed-support hydrological evaluation for each of these
three recovered ORIGINAL trials, at their documented best_epoch, reusing
only already-qualified scientific primitives:

  - src.baseline.fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics
    (raw-space NSE/KGE/rmse/mae/pearson_r/bias/pbias per basin, raw-space
    conversion handled internally)
  - src.baseline.fixed_support_contract_v2.extract_v2_objective_from_fixed_support_result
    (the only function permitted to produce the v2 primary objective)
  - src.baseline.sweep_v1_campaign.derive_trajectory_diagnostics
    (the documented, authoritative first-achieved-best-epoch selection rule)
  - src.baseline.fixed_support_contract_v2.derive_canonical_package_observed_series
    + stage1_v2_12plus12_hydrological_consumer.derive_canonical_basin_q98_facts
    + stage1_v2_12plus12_hydrological_consumer.compute_q98_configuration_basin_diagnostics
    (Q98-normalized RMSE, relative volume bias, observed-peak-time magnitude
    error, high-flow NSE)

This script deliberately does NOT go through the receipt-bound production
entry points (build_v2_best_epoch_source / evaluate_v2_configuration_hydrological_result /
assemble_rd1_hydrological_review / rd1_c4_results_runner), because no
execution_provenance.json receipt exists on disk for these three trials.
Fabricating one well enough to pass build_v2_best_epoch_source's internal
re-derivation (configuration_id_v2/proposal_id_v2/trial_id_v2 recomputation)
would produce an artifact that looks like genuine machine-written execution
provenance without being one. Calling the underlying primitives directly
(exactly the composition those entry points use internally -- see
rd1_c4_results_runner.py / stage1_v2_12plus12_hydrological_consumer.py for
the reference composition) avoids that risk while reusing 100% of the
qualified scientific math. Trial identity for the three trials evaluated
here is established independently (job-log run_dir extraction + config.yml
six-axis/seed cross-check against historical_trial_axes.json), not via a
receipt.

Usage:
  python rd1_p17p19_recovered_evaluation_runner.py <repo_workdir> <out_dir>

All per-trial facts (run_dir, trial_id, recorded_objective) are hardcoded
below because they are frozen historical identity facts already verified
against primary evidence (job stdout logs, recovered config.yml files,
W&B wandb-summary.json), not free parameters.
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

repo_workdir = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(repo_workdir))

from src.baseline.fixed_support_contract_v2 import (  # noqa: E402
    FixedSupportContractError,
    derive_canonical_package_observed_series,
    evaluate_fixed_support_raw_space_metrics,
    extract_v2_objective_from_fixed_support_result,
    load_fixed_support_contract,
)
from src.baseline.package_identity_qualification import qualify_package_identity  # noqa: E402
from src.baseline.stage1_v2_12plus12_hydrological_consumer import (  # noqa: E402
    HydrologicalConsumerError,
    _audit_admitted_observation_provenance,
    compute_q98_configuration_basin_diagnostics,
    derive_canonical_basin_q98_facts,
)
from src.baseline.sweep_v1_campaign import derive_trajectory_diagnostics  # noqa: E402

PACKAGE_ROOT = "/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_scientific_package_v002"
FIXED_SUPPORT_CONTRACT_PATH = (
    "/sci/labs/efratmorin/omripo/Flash-NH/data/fixed_support_contracts/"
    "stage1_v2_common120_fixed_support_contract_v001.json"
)
EXPECTED_FIXED_SUPPORT_CONTRACT_SHA256 = "cb4ebe86afa501ef3d5929ead5b455f8df06e7d38b58ebf4148f8545fe6851ef"
EXPECTED_COMMIT = "dabd2ca851bd2b3a03035886cfa50015f2c864b4"

TRIALS = {
    "P17": {
        "wandb_run_id": "mwxip3q4",
        "trial_id": (
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal017__"
            "sweep_v2_cfg_224202d59859c8e969d1__mf12x50000__seedA967139__attempt001"
        ),
        "run_dir": (
            "/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/.scratch_local/"
            "p17_launch_2026-09-24_20260924T134251Z/training_output/proposal_000017/"
            "execution_generation_001/"
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal017__"
            "sweep_v2_cfg_224202d59859c8e969d1__mf12x50000__seedA967139__attempt001/runs/"
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal017__"
            "sweep_v2_cfg_224202d59859c8e969d1__mf12x50000__seedA967139__attempt001_2409_165146"
        ),
        "recorded_objective_wandb_common120_raw_space_nse_v001": 0.42106726019345797,
    },
    "P18": {
        "wandb_run_id": "x2dv6l4b",
        "trial_id": (
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal018__"
            "sweep_v2_cfg_ecfc86475926fac9cf25__mf12x50000__seedA967139__attempt001"
        ),
        "run_dir": (
            "/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/.scratch_local/"
            "rd1_continuation_chain_20260925T123812Z/training_output/proposal_000018/"
            "execution_generation_001/"
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal018__"
            "sweep_v2_cfg_ecfc86475926fac9cf25__mf12x50000__seedA967139__attempt001/runs/"
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal018__"
            "sweep_v2_cfg_ecfc86475926fac9cf25__mf12x50000__seedA967139__attempt001_2609_123959"
        ),
        "recorded_objective_wandb_common120_raw_space_nse_v001": 0.4541781146131809,
    },
    "P19": {
        "wandb_run_id": "onbe4e1c",
        "trial_id": (
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal019__"
            "sweep_v2_cfg_bfcf674895deb7f57f88__mf12x50000__seedA967139__attempt001"
        ),
        "run_dir": (
            "/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/.scratch_local/"
            "rd1_continuation_chain_20260925T123812Z/training_output/proposal_000019/"
            "execution_generation_001/"
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal019__"
            "sweep_v2_cfg_bfcf674895deb7f57f88__mf12x50000__seedA967139__attempt001/runs/"
            "stage1_phase_b_sweep_v2_six_axis_common120_v001__bayesian__proposal019__"
            "sweep_v2_cfg_bfcf674895deb7f57f88__mf12x50000__seedA967139__attempt001_2609_235512"
        ),
        "recorded_objective_wandb_common120_raw_space_nse_v001": 0.4364637928204671,
    },
}


def jsonify(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    return value


def main() -> int:
    try:
        actual_commit = subprocess.run(
            ["git", "-C", str(repo_workdir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        actual_commit = f"<git rev-parse failed: {exc}>"

    print(f"repo_workdir={repo_workdir}", flush=True)
    print(f"actual_commit={actual_commit} expected_commit={EXPECTED_COMMIT}", flush=True)

    contract = load_fixed_support_contract(FIXED_SUPPORT_CONTRACT_PATH)
    contract_sha256 = contract["checksum_sha256"]
    print(
        f"loaded fixed-support contract from {FIXED_SUPPORT_CONTRACT_PATH} "
        f"checksum_sha256={contract_sha256} expected={EXPECTED_FIXED_SUPPORT_CONTRACT_SHA256} "
        f"match={contract_sha256 == EXPECTED_FIXED_SUPPORT_CONTRACT_SHA256}",
        flush=True,
    )

    package_identity = qualify_package_identity(package_root=PACKAGE_ROOT, contract=contract)
    print("package identity qualified", flush=True)

    overall_manifest = {
        "task": "rd1_p17p19_recovered_evaluation",
        "generated_by": "rd1_p17p19_recovered_evaluation_runner.py",
        "repo_workdir": str(repo_workdir),
        "actual_commit": actual_commit,
        "expected_commit": EXPECTED_COMMIT,
        "commit_match": actual_commit == EXPECTED_COMMIT,
        "package_root": PACKAGE_ROOT,
        "fixed_support_contract_path": FIXED_SUPPORT_CONTRACT_PATH,
        "fixed_support_contract_sha256": contract_sha256,
        "expected_fixed_support_contract_sha256": EXPECTED_FIXED_SUPPORT_CONTRACT_SHA256,
        "fixed_support_contract_sha256_match": contract_sha256 == EXPECTED_FIXED_SUPPORT_CONTRACT_SHA256,
        "trials": {},
    }

    for label, trial in TRIALS.items():
        print(f"=== {label} ({trial['trial_id']}) ===", flush=True)
        run_dir = trial["run_dir"]
        trial_id = trial["trial_id"]
        trial_report = {
            "wandb_run_id": trial["wandb_run_id"],
            "trial_id": trial_id,
            "run_dir": run_dir,
            "recorded_objective_wandb_common120_raw_space_nse_v001": trial[
                "recorded_objective_wandb_common120_raw_space_nse_v001"
            ],
            "epoch_trajectory": {},
            "errors": [],
        }

        trajectory = {}
        for epoch in range(1, 13):
            try:
                result = evaluate_fixed_support_raw_space_metrics(
                    run_dir=run_dir,
                    epoch=epoch,
                    package_root=PACKAGE_ROOT,
                    contract=contract,
                    package_identity=package_identity,
                    require_full_screening_population=True,
                    return_admitted_series=False,
                )
                objective = extract_v2_objective_from_fixed_support_result(result)
                trajectory[epoch] = objective
                print(f"  epoch {epoch:02d}: objective={objective:.6f}", flush=True)
            except FixedSupportContractError as exc:
                trial_report["errors"].append(f"epoch {epoch}: {exc}")
                print(f"  epoch {epoch:02d}: FAILED: {exc}", flush=True)

        trial_report["epoch_trajectory"] = trajectory

        if len(trajectory) != 12:
            trial_report["status"] = "INCOMPLETE_TRAJECTORY"
            overall_manifest["trials"][label] = trial_report
            print(f"  {label}: incomplete trajectory ({len(trajectory)}/12 epochs) -- skipping best-epoch evaluation", flush=True)
            continue

        diagnostics = derive_trajectory_diagnostics(trajectory)
        best_epoch = diagnostics["best_epoch"]
        recomputed_objective = diagnostics["best_score"]
        trial_report["trajectory_diagnostics"] = diagnostics
        trial_report["best_epoch"] = best_epoch
        trial_report["recomputed_objective"] = recomputed_objective
        recorded = trial_report["recorded_objective_wandb_common120_raw_space_nse_v001"]
        trial_report["objective_agreement"] = {
            "recorded": recorded,
            "recomputed": recomputed_objective,
            "abs_diff": abs(recorded - recomputed_objective),
            "exact_match": recorded == recomputed_objective,
        }
        print(
            f"  best_epoch={best_epoch} recomputed_objective={recomputed_objective:.9f} "
            f"recorded_objective={recorded:.9f} abs_diff={abs(recorded - recomputed_objective):.3e}",
            flush=True,
        )

        result_best = evaluate_fixed_support_raw_space_metrics(
            run_dir=run_dir,
            epoch=best_epoch,
            package_root=PACKAGE_ROOT,
            contract=contract,
            package_identity=package_identity,
            require_full_screening_population=True,
            return_admitted_series=True,
        )
        trial_report["n_basins_requested"] = result_best["n_basins_requested"]
        trial_report["n_basins_evaluated"] = result_best["n_basins_evaluated"]
        trial_report["n_basins_excluded"] = result_best["n_basins_excluded"]
        trial_report["basins_excluded"] = result_best["basins_excluded"]
        trial_report["aggregate"] = result_best["aggregate"]

        admitted_series_by_basin = result_best["admitted_series_by_basin"]
        per_basin_rows = {row["basin_id"]: row for row in result_best["per_basin"]}

        q98_rows = {}
        q98_errors = []
        provenance_audit_rows = {}
        for basin_id, admitted in admitted_series_by_basin.items():
            try:
                canonical = derive_canonical_package_observed_series(
                    package_root=PACKAGE_ROOT, basin_id=basin_id, contract=contract,
                    package_identity=package_identity,
                )
                facts = derive_canonical_basin_q98_facts(
                    basin_id=basin_id, date=canonical.date, obs_m3s=canonical.obs_m3s,
                )
                # Report-only provenance/consistency evidence between this
                # trial's OWN admitted observed discharge (from its
                # validation_results.p) and the package-canonical series --
                # per docs/decision_log.md 2026-09-17 (RD1-C4-E), this never
                # determines any Q98 fact or metric and is not fatal on a
                # within-envelope float difference.
                audit = _audit_admitted_observation_provenance(
                    trial_id=trial_id, basin_id=basin_id,
                    admitted_date=admitted.date, admitted_obs_m3s=admitted.obs_m3s,
                    canonical=canonical,
                )
                provenance_audit_rows[basin_id] = {
                    "n_compared": audit.n_compared,
                    "n_exceeding": audit.n_exceeding,
                    "envelope_exceeded": audit.envelope_exceeded,
                    "max_abs_diff_m3s": audit.max_abs_diff_m3s,
                }
                # Q98 facts/metrics are derived solely from the
                # package-canonical observed series (obs_m3s=facts.canonical_obs_m3s
                # here, matching the formal assemble_rd1_hydrological_review
                # call pattern exactly) -- never from this trial's own
                # admitted series, which is provenance evidence only (above).
                diag = compute_q98_configuration_basin_diagnostics(
                    trial_id=trial_id, facts=facts,
                    date=admitted.date, obs_m3s=facts.canonical_obs_m3s, sim_m3s=admitted.sim_m3s,
                )
                q98_rows[basin_id] = {
                    "n_high_flow": diag.n_high_flow,
                    "q98_normalized_rmse": diag.q98_normalized_rmse,
                    "relative_volume_bias": diag.relative_volume_bias,
                    "observed_peak_time_magnitude_error": diag.observed_peak_time_magnitude_error,
                    "high_flow_nse": diag.high_flow_nse,
                    "high_flow_nse_available": diag.high_flow_nse_available,
                }
            except (HydrologicalConsumerError, FixedSupportContractError) as exc:
                q98_errors.append({"basin_id": basin_id, "error": str(exc)})

        trial_report["q98_errors"] = q98_errors
        trial_report["n_q98_basins_evaluated"] = len(q98_rows)
        trial_report["n_q98_basins_failed"] = len(q98_errors)
        trial_report["provenance_audit_by_basin"] = provenance_audit_rows
        trial_report["provenance_audit_summary"] = {
            "n_basins_with_exceedance": sum(1 for r in provenance_audit_rows.values() if r["envelope_exceeded"]),
            "max_abs_diff_m3s_overall": (
                max((r["max_abs_diff_m3s"] for r in provenance_audit_rows.values()), default=float("nan"))
            ),
        }

        combined_per_basin = []
        for basin_id, row in sorted(per_basin_rows.items()):
            combined = {k: v for k, v in row.items() if not str(k).startswith("_") and k != "freq"}
            if basin_id in q98_rows:
                combined.update(q98_rows[basin_id])
            combined_per_basin.append(combined)
        trial_report["per_basin"] = combined_per_basin

        def _finite_median(key: str):
            values = [row[key] for row in combined_per_basin if key in row and np.isfinite(row[key])]
            return float(np.median(values)) if values else float("nan")

        trial_report["q98_aggregate_medians"] = {
            "q98_normalized_rmse_median": _finite_median("q98_normalized_rmse"),
            "relative_volume_bias_median": _finite_median("relative_volume_bias"),
            "observed_peak_time_magnitude_error_median": _finite_median("observed_peak_time_magnitude_error"),
            "high_flow_nse_median": _finite_median("high_flow_nse"),
        }

        trial_report["status"] = "EVALUATED"
        overall_manifest["trials"][label] = trial_report

        trial_out_path = out_dir / f"{label}_recovered_evaluation.json"
        with open(trial_out_path, "w", encoding="utf-8") as handle:
            json.dump(jsonify(trial_report), handle, indent=2, sort_keys=True)
        print(f"  wrote {trial_out_path}", flush=True)

    manifest_path = out_dir / "rd1_p17p19_recovered_evaluation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(jsonify(overall_manifest), handle, indent=2, sort_keys=True)
    print(f"wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
