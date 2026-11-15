#!/usr/bin/env python
"""Run ONE water-year rotation run (a campaign cycle or a canary), score-blind:
NeuralHydrology training of the sealed cycle config, then NH evaluation of the
checkpoint year (NH ``validation`` period) for EVERY epoch over all 2,307
development basins with no metrics, then a receipts/keys/shapes verification.

It computes, prints and stores no skill score: no in-loop validation
(``validate_every: null``), ``metrics=[]`` at evaluation, no metrics key, no
W&B. The raw ``validation_results.p`` per epoch is left for the later,
separately approved checkpoint rule. The assessment year (NH ``test``) is never
evaluated here. Intended caller: a Moriah sbatch (the canary's is
``.scratch_local/wy_rotation_runs_v001/run_wy_rotation_canary_moriah.sbatch``).

Evidence written to the config dir: ``wy_rotation_run_receipt.json``
(STARTED -> COMPLETE/FAILED; provenance, Slurm ids, timing, peak memory) and
``wy_rotation_score_blind_evidence.json`` (verify_score_blind_run).

Restart: a COMPLETE receipt makes a rerun a no-op. An existing NH run dir
with all checkpoints is not retrained (only the missing per-epoch pickles are
written); one with an incomplete checkpoint set is refused (continuing an
interrupted training is a separate decision).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline import pilot_orchestration as po  # noqa: E402
from src.baseline import wy_rotation_runs as rot  # noqa: E402
from src.baseline.wy_rotation_split import load_split_contract, verify_basin_population  # noqa: E402

_CANONICAL_MORIAH_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
RECEIPT_NAME = "wy_rotation_run_receipt.json"
EVIDENCE_NAME = "wy_rotation_score_blind_evidence.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", type=Path, required=True, help="campaign_manifest.json or canary_manifest.json")
    p.add_argument("--index", type=int, default=None, help="campaign row 0..4; default $SLURM_ARRAY_TASK_ID; "
                                                           "ignored for a canary manifest")
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--expected-runtime-python", default=_CANONICAL_MORIAH_PYTHON)
    return p


def _write_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _peak_memory() -> dict:
    out = {}
    try:
        import resource

        out["max_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            out["cuda_max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            out["cuda_max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    return out


def _tester_factory(nh_run_dir: Path):
    from neuralhydrology.utils.logging_utils import setup_logging

    log_path = nh_run_dir / "output.log"
    if not po.root_logger_has_file_handler(log_path):
        setup_logging(str(log_path))
    return rot._default_tester_factory(nh_run_dir)


def _raise_on_sigterm(signum, frame):
    # Slurm sends SIGTERM at the time limit / on scancel: turn it into an
    # exception so the receipt records FAILED instead of staying STARTED.
    raise SystemExit(f"terminated by signal {signum}")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    from src.baseline.sweep_v1_runtime_contract import RuntimeContractError, verify_commit_and_interpreter

    try:
        provenance = verify_commit_and_interpreter(repo_root=ROOT, expected_commit=args.expected_commit,
                                                   expected_runtime_python=args.expected_runtime_python)
    except RuntimeContractError as exc:
        raise SystemExit(f"runtime provenance check failed, refusing to run: {exc}") from exc

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("is_canary"):
        checked = rot.verify_canary_manifest(manifest)
        campaign = checked["campaign"]
    else:
        index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
        checked = rot.verify_campaign_row(manifest, index)
        campaign = manifest
    row, config = checked["row"], checked["config"]
    config_dir = Path(row["config_dir"])
    epochs = list(range(1, int(config["epochs"]) + 1))
    receipt_path = config_dir / RECEIPT_NAME
    if receipt_path.is_file() and json.loads(receipt_path.read_text(encoding="utf-8")).get("status") == "COMPLETE":
        print(f"{row['run_id']}: COMPLETE receipt already present, nothing to do")
        return 0

    receipt = {"status": "STARTED", "run_id": row["run_id"], "cycle": row["cycle"],
               "is_canary": bool(row.get("is_canary")), "epochs": len(epochs),
               "max_updates_per_epoch": config["max_updates_per_epoch"], "started_utc": time.time(),
               "manifest": str(args.manifest), "manifest_sha256": rot.sha256_file(args.manifest), **provenance,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
               "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "host": socket.gethostname()}
    _write_json(receipt_path, receipt)
    try:
        contract = load_split_contract(campaign["split_contract"]["path"])
        population = verify_basin_population(contract, ROOT)

        nh_run_dir = po._try_discover_nh_run_dir(config_dir, row["run_id"])
        if nh_run_dir is None:
            t0 = time.monotonic()
            po.default_train_chunk(po.TrainChunkRequest(
                is_first_chunk=True, config_path=checked["config_path"], nh_run_dir=None, current_epoch=None,
                logical_target_epoch=len(epochs), additional_epochs=len(epochs)))
            receipt["train_wall_s"] = round(time.monotonic() - t0, 1)
            receipt["train_peak_memory"] = _peak_memory()
            nh_run_dir = po.discover_nh_run_dir(config_dir, row["run_id"])
            receipt["trained_in_this_job"] = True
        else:
            receipt["trained_in_this_job"] = False
        receipt["nh_run_dir"] = str(nh_run_dir)
        checkpoints = po.discover_physical_checkpoints(nh_run_dir)
        if sorted(checkpoints) != epochs or any(c.owning_run_dir != nh_run_dir for c in checkpoints.values()):
            raise rot.WyRotationRunsError(
                f"checkpoint set {sorted(checkpoints)} under {nh_run_dir} is not the flat 1..{len(epochs)} set of a "
                "completed training; continuing an interrupted run needs a separate decision")
        receipt["epoch_end_timestamps"] = rot.log_epoch_timestamps(nh_run_dir / "output.log")
        _write_json(receipt_path, receipt)

        t0 = time.monotonic()
        receipt["evaluation"] = rot.evaluate_all_epochs(nh_run_dir, epochs, tester_factory=_tester_factory)
        receipt["evaluation_wall_s"] = round(time.monotonic() - t0, 1)
        receipt["evaluation_peak_memory"] = _peak_memory()
        _write_json(receipt_path, receipt)

        evidence = rot.verify_score_blind_run(nh_run_dir, cycle=row["cycle"], epochs=epochs, population=population)
        _write_json(config_dir / EVIDENCE_NAME, evidence)
        if not evidence["passed"]:
            raise rot.WyRotationRunsError(f"score-blind verification failed: {evidence['problems']}")
        receipt.update(status="COMPLETE", finished_utc=time.time())
        _write_json(receipt_path, receipt)
        print(json.dumps({"run_id": row["run_id"], "status": "COMPLETE", "epochs_evaluated": len(epochs),
                          "n_basins_per_epoch": sorted({e["n_basins"] for e in evidence["per_epoch"].values()})}))
        return 0
    except BaseException as exc:  # recorded, then re-raised: never a silent pass
        receipt.update(status="FAILED", finished_utc=time.time(), error=repr(exc), traceback=traceback.format_exc())
        _write_json(receipt_path, receipt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
