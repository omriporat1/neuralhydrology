#!/usr/bin/env python
"""Run exactly ONE of the 15 ``seqlen_common336_seed3_v001`` trials:
NeuralHydrology training (via the existing MONOLITHIC executor
``pilot_orchestration.execute_prepared_pilot_run_monolithic``), post-hoc
natural-support pilot screening (unchanged, diagnostic), and the Common-336
raw-space screening per epoch (``evaluate_common336_screening``), then select
the best epoch by Common-336 median per-basin raw-space NSE.

Never contacts W&B or any sweep/controller; it reads no sealed scope (the
screening path is ``validation`` only). Intended caller:
``scripts/run_seqlen_common336_moriah.sbatch`` (one array task per trial).

Failure behaviour: any identity/provenance mismatch, an incomplete checkpoint
set, an incomplete 400-basin Common-336 screening, or a loader that did not
apply the support raises and writes a FAILED receipt; it never falls back.
Restart: a COMPLETE receipt makes a rerun a no-op; an existing NH run dir is
consumed by the monolithic executor's own restart-safe logic (it never
continues an interrupted training -- that is reported as blocked).
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline import pilot_orchestration  # noqa: E402
from src.baseline import seqlen_common336_campaign as camp  # noqa: E402
from src.baseline import sweep_v1_campaign as sweep  # noqa: E402
from src.baseline.nh_config_generation import (  # noqa: E402
    read_package_manifest, validate_full_population_basin_membership,
)
from src.baseline.pilot_lead06_config import load_pilot_policy, load_screening_basin_ids  # noqa: E402
from src.baseline.seqlen_common336_support import (  # noqa: E402
    evaluate_common336_screening, load_common336_screening_contract, load_training_support,
)
from src.baseline.sweep_v1_runtime_contract import RuntimeContractError, verify_commit_and_interpreter  # noqa: E402

_CANONICAL_MORIAH_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
_DEFAULT_PACKAGE_ROOT = "/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_scientific_package_v002"
_DEFAULT_SCREENING_BASIN_IDS = ("/sci/labs/efratmorin/omripo/Flash-NH/data/screening_subsets/"
                                "stage1_provisional_operational_screening_subset_v001/screening_subset_basin_ids.txt")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--campaign-manifest", type=Path, required=True)
    p.add_argument("--index", type=int, default=None, help="0..14; default $SLURM_ARRAY_TASK_ID")
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--expected-runtime-python", default=_CANONICAL_MORIAH_PYTHON)
    p.add_argument("--package-root", type=Path, default=Path(_DEFAULT_PACKAGE_ROOT))
    p.add_argument("--screening-basin-ids", type=Path, default=Path(_DEFAULT_SCREENING_BASIN_IDS))
    p.add_argument("--splits-dir", type=Path, default=ROOT / "config" / "stage1_baseline_splits_v001")
    p.add_argument("--base-pilot-policy-path", type=Path, default=ROOT / "config" / "stage1_lead06_pilot_v001.yaml")
    return p


def _write_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = _parser().parse_args()
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    try:
        provenance = verify_commit_and_interpreter(repo_root=ROOT, expected_commit=args.expected_commit,
                                                   expected_runtime_python=args.expected_runtime_python)
    except RuntimeContractError as exc:
        raise SystemExit(f"runtime provenance check failed, refusing to run trial: {exc}") from exc

    manifest = json.loads(args.campaign_manifest.read_text(encoding="utf-8"))
    row = camp.verify_campaign_row(manifest, index)
    config_dir = Path(row["config_dir"])
    receipt_path = config_dir / camp.TRIAL_RECEIPT_NAME
    if receipt_path.is_file() and json.loads(receipt_path.read_text(encoding="utf-8")).get("status") == "COMPLETE":
        print(f"{row['run_id']}: COMPLETE receipt already present, nothing to do")
        return 0

    receipt = {"status": "STARTED", "run_id": row["run_id"], "index": index, "started_utc": time.time(),
               "campaign_manifest": str(args.campaign_manifest),
               "campaign_manifest_sha256": camp.sha256_file(args.campaign_manifest), **provenance,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
               "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "host": socket.gethostname()}
    _write_json(receipt_path, receipt)
    try:
        support = load_training_support(manifest["training_support"]["path"])
        contract = load_common336_screening_contract(manifest["screening_contract"]["path"])
        membership = validate_full_population_basin_membership(read_package_manifest(args.package_root), args.splits_dir)
        screening = load_screening_basin_ids(args.screening_basin_ids, development_basins=membership.development_basins,
                                             expected_count=400, expected_sha256=sweep.SCREENING_ARTIFACT_SHA256)
        validation_file = Path(manifest["basin_files"]["validation_basin_file"]["path"])
        validation_basins = {line.strip() for line in validation_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        if not (set(screening) == set(contract["basin_ids"]) == validation_basins):
            raise camp.SeqlenCampaignError("screening ids, Common-336 contract basins and validation basin file differ")

        policy = replace(load_pilot_policy(args.base_pilot_policy_path), lead_hours=6,
                         screening_validation_every_n_epochs=1, initial_training_epochs=1,
                         pilot_max_epoch_budget=sweep.TARGET_EPOCH, performance_early_stopping_enabled=False,
                         screening_basin_ids_path=str(args.screening_basin_ids), screening_expected_count=400,
                         screening_expected_sha256=sweep.SCREENING_ARTIFACT_SHA256)
        basins = tuple(sorted(screening))

        def common336_epoch_evaluator(run_dir: Path, epoch: int) -> dict:
            return {"common336": evaluate_common336_screening(
                run_dir=run_dir, epoch=epoch, package_root=args.package_root, contract=contract,
                basin_ids=basins, require_full_screening_population=True)}

        result = pilot_orchestration.execute_prepared_pilot_run_monolithic(
            execution_policy=policy, config_dir=config_dir, experiment_name=row["run_id"],
            package_root=args.package_root, target_variable=support["target_variable"], lead_hours=6,
            screening_basin_ids=screening, target_epoch=sweep.TARGET_EPOCH,
            supplemental_epoch_evaluator=common336_epoch_evaluator)
        receipt["nh_run_dir"] = str(result.nh_run_dir)
        applied = camp.verify_applied_training_support(result.nh_run_dir, seq_length=row["seq_length"],
                                                       seed=row["seed"], support=support)
        summary = camp.summarize_trial(result, row=row)
        _write_json(config_dir / camp.PER_EPOCH_RESULTS_NAME, {
            epoch: {k: entry["common336"][k] for k in ("per_basin", "aggregate", "basins_excluded",
                                                       "evaluation_source", "contract_checksum_sha256")}
            for epoch, entry in result.supplemental_epoch_results.items()})
        receipt.update(status="COMPLETE", finished_utc=time.time(), training_support_applied=applied,
                       summary={k: v for k, v in summary.items() if k != "per_epoch_aggregate"},
                       natural_support_screening_events=len(result.screening_events))
        _write_json(receipt_path, receipt)
        print(json.dumps({"run_id": row["run_id"], "best_epoch": summary["best_epoch"],
                          "best_common336_median_nse": summary["best_score"]}))
        return 0
    except BaseException as exc:  # recorded, then re-raised: never a silent pass
        receipt.update(status="FAILED", finished_utc=time.time(), error=repr(exc), traceback=traceback.format_exc())
        _write_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
