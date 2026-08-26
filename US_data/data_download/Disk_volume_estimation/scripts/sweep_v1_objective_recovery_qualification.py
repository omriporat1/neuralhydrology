"""QUALIFICATION ONLY -- NON-SCIENTIFIC.

Operationally qualifies ``src.baseline.sweep_v1_objective_recovery`` end to
end against one small, disposable, explicitly non-scientific REAL W&B
sweep/run. This is Section F of the ATTEMPT005 closure/provenance-continuity/
objective-recovery-qualification task: the recovery module's pure/local
logic was already unit-tested with a fake ``wandb`` module (see
``tests/test_sweep_v1_objective_recovery.py``); this script is what actually
exercises the network-facing ``recover_and_publish_objective`` path that
those unit tests deliberately never touch.

Hard rules enforced by this script itself, not just by convention:
  * NEVER calls ``wandb.agent()`` -- creates a disposable sweep, then starts
    exactly ONE run directly under it via ``wandb.init(settings=wandb.
    Settings(sweep_id=...))``, mirroring ``run_sweep_v1_exact_retry_bridge.
    py``'s production pattern, not the agent-loop pattern.
  * NEVER references production sweep ``4x3btz2s`` or run ``ardib08c`` --
    asserted immediately after the disposable sweep/run are created, and the
    synthetic record builder refuses either identity outright.
  * NEVER trains, never imports neuralhydrology/torch, never touches
    scientific data or the real Sweep-v1 campaign module.
  * Uses a ``campaign_id``/``trial_id`` namespace
    (``objective_recovery_qualification_v001``) that can never collide with
    a real Sweep-v1 identity.

Must run inside a CPU Slurm allocation on Moriah (small, short), never the
login node -- see ``sweep_v1_objective_recovery_qualification_moriah.sbatch``.
Requires real W&B network credentials (the Moriah ``flashnh-moriah``
environment already has these from prior qualification work); the local
Windows dev environment does not, which is why this does not run locally.

Exercises, in order:
  1. Disposable sweep + exactly one disposable run (real, online, clean
     finish verified via ``wandb.Api()``).
  2. A synthetic immutable VALID record referencing that real run/sweep
     identity, with a finite frozen objective and a full identity/source-
     hash shape.
  3. One successful publication via the real ``recover_and_publish_objective``
     (real ``wandb.Api()``, never ``wandb.init()``/``wandb.agent()`` inside
     the recovery call itself) -- verified by reading the run's summary back.
  4. An identical repeated reconciliation proving idempotency -- with
     ``wandb`` poisoned in ``sys.modules`` beforehand, so a second network
     call would raise ``ImportError`` rather than silently succeed.
  5. Rejection of all 8 named negative cases: INVALID record, incomplete
     record, non-finite objective, mismatched W&B run, mismatched sweep
     (the one negative case that genuinely needs a live ``wandb.Api()``
     call), changed objective, missing source hashes, contradictory
     identity.
  6. Durable local JSON evidence of every result above.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.baseline.sweep_v1_objective_recovery import (  # noqa: E402
    ObjectiveRecoveryError, recover_and_publish_objective,
)

_CANONICAL_RUNTIME_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
_DEFAULT_PROJECT = "flashnh-stage1"
_DEFAULT_OUT_DIR = _REPO_ROOT / "reports" / "sweep_v1_objective_recovery_qualification_v001" / "run"

# Deliberately outside the real Sweep-v1 campaign identity namespace
# (src.baseline.sweep_v1_campaign.CAMPAIGN_ID) -- this can never be mistaken
# for a real proposal/trial.
_QUAL_CAMPAIGN_ID = "objective_recovery_qualification_v001"
_QUAL_TAGS = ("qualification", "non_scientific", _QUAL_CAMPAIGN_ID)
_TOY_METRIC_NAME = "qualification/toy_objective"

FORBIDDEN_PRODUCTION_SWEEP_ID = "4x3btz2s"
FORBIDDEN_PRODUCTION_RUN_ID = "ardib08c"


def _git_head(repo_root: Path) -> "str | None":
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _git_dirty_tracked(repo_root: Path) -> "list[str] | None":
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=str(repo_root), capture_output=True, text=True, check=True, timeout=10,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return None


def _guard_or_die(*, expected_commit: str, expected_runtime_python: str) -> None:
    actual_head = _git_head(_REPO_ROOT)
    if actual_head != expected_commit:
        raise SystemExit(f"REFUSING: git HEAD {actual_head!r} != expected commit {expected_commit!r}")
    dirty = _git_dirty_tracked(_REPO_ROOT)
    if dirty:
        raise SystemExit(f"REFUSING: tracked tree is dirty: {dirty!r}")
    if sys.executable != expected_runtime_python:
        raise SystemExit(
            f"REFUSING: python executable {sys.executable!r} != expected canonical runtime {expected_runtime_python!r}"
        )


def _assert_not_production_identity(*, wandb_run_id: str, wandb_sweep_id: str) -> None:
    if wandb_run_id == FORBIDDEN_PRODUCTION_RUN_ID or wandb_sweep_id == FORBIDDEN_PRODUCTION_SWEEP_ID:
        raise SystemExit(
            "REFUSING: qualification must never reference the production attempt005 "
            f"run/sweep identity (run={wandb_run_id!r}, sweep={wandb_sweep_id!r})"
        )


def _build_synthetic_valid_record(
    *, wandb_run_id: str, wandb_sweep_id: str, trial_suffix: str, overrides: "dict[str, Any] | None" = None,
) -> dict:
    _assert_not_production_identity(wandb_run_id=wandb_run_id, wandb_sweep_id=wandb_sweep_id)
    trial_id = f"{_QUAL_CAMPAIGN_ID}__trial_{trial_suffix}"
    record = {
        "campaign_id": _QUAL_CAMPAIGN_ID,
        "proposal_id": f"{_QUAL_CAMPAIGN_ID}__proposal001",
        "configuration_id": "objective_recovery_qual_cfg_0001",
        "trial_id": trial_id,
        "retry_of_trial_id": None,
        "execution_generation": 1,
        "search_arm": "qualification",
        "execution_status": "VALID",
        "objective_score": 0.123456,
        "generated_nh_config_sha256": hashlib.sha256(trial_id.encode("utf-8")).hexdigest(),
        "wandb_run_id": wandb_run_id,
        "wandb_sweep_id": wandb_sweep_id,
    }
    if overrides:
        record.update(overrides)
    return record


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expected-commit", type=str, required=True)
    parser.add_argument("--expected-runtime-python", type=str, default=_CANONICAL_RUNTIME_PYTHON)
    parser.add_argument("--project", type=str, default=_DEFAULT_PROJECT)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    args = parser.parse_args()

    _guard_or_die(expected_commit=args.expected_commit, expected_runtime_python=args.expected_runtime_python)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import wandb  # lazy import, repo-wide convention; real package, real network

    checks: "dict[str, Any]" = {}
    negative: "dict[str, Any]" = {}

    # --- 1. disposable sweep + exactly one disposable run (never wandb.agent) ---
    sweep_config = {
        "method": "bayes",
        "name": f"{_QUAL_CAMPAIGN_ID}_toy",
        "metric": {"name": _TOY_METRIC_NAME, "goal": "maximize"},
        "parameters": {"toy_axis": {"values": [1, 2, 3]}},
    }
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    if sweep_id == FORBIDDEN_PRODUCTION_SWEEP_ID:
        raise SystemExit("REFUSING: newly created disposable sweep collided with the production sweep id")

    run = wandb.init(
        settings=wandb.Settings(sweep_id=sweep_id),
        project=args.project, entity=args.entity,
        job_type="objective_recovery_qualification",
        tags=list(_QUAL_TAGS),
        config={"qualification_campaign_id": _QUAL_CAMPAIGN_ID, "scientific_trial": False, "toy_axis": 1},
    )
    run_id = run.id
    if run_id == FORBIDDEN_PRODUCTION_RUN_ID:
        raise SystemExit("REFUSING: newly created disposable run collided with the production run id")
    run.log({_TOY_METRIC_NAME: 0.5})
    run.finish()

    api = wandb.Api()
    run_path = f"{args.entity}/{args.project}/{run_id}" if args.entity else f"{args.project}/{run_id}"
    fetched = api.run(run_path)
    checks["disposable_run_id"] = run_id
    checks["disposable_sweep_id"] = sweep_id
    checks["run_state_after_finish"] = fetched.state
    checks["clean_wandb_finish"] = fetched.state == "finished"

    # --- 2 & 3. synthetic VALID record, positive publication ---
    record = _build_synthetic_valid_record(wandb_run_id=run_id, wandb_sweep_id=sweep_id, trial_suffix="0001")
    record_path = _write_json(out_dir / "synthetic_valid_record.json", record)
    marker_path = out_dir / "marker_positive.json"
    expected_identity = {
        "trial_id": record["trial_id"], "configuration_id": record["configuration_id"],
        "wandb_run_id": run_id, "wandb_sweep_id": sweep_id,
    }

    result1 = recover_and_publish_objective(
        execution_provenance_path=record_path, expected_identity=expected_identity,
        marker_path=marker_path, project=args.project, entity=args.entity,
    )
    checks["positive_publish_result"] = result1

    # A fresh wandb.Api() instance -- NOT the `api` object above -- is used
    # deliberately: wandb.Api() caches Run objects per-instance by path, so
    # reusing `api` here would silently return the SAME cached (pre-publish)
    # Run/Summary object rather than a genuine server round-trip, making
    # this readback check pass or fail on stale local state instead of the
    # real thing recover_and_publish_objective just wrote (confirmed via a
    # live wandb.Api() inspection on Moriah: this is exactly what happened
    # in the first real run of this script, job 45950133).
    refetched = wandb.Api().run(run_path)
    checks["summary_flashnh_objective_score_readback"] = refetched.summary.get("flashnh/objective_score")
    checks["summary_matches_record"] = refetched.summary.get("flashnh/objective_score") == record["objective_score"]
    checks["summary_flashnh_valid_readback"] = refetched.summary.get("flashnh/valid")

    # --- 4. idempotent repeat -- poison wandb to PROVE no second network call ---
    real_wandb_module = sys.modules["wandb"]
    sys.modules["wandb"] = None
    try:
        result2 = recover_and_publish_objective(
            execution_provenance_path=record_path, expected_identity=expected_identity,
            marker_path=marker_path, project=args.project, entity=args.entity,
        )
        checks["idempotent_repeat_result"] = result2
        checks["idempotent_repeat_never_imported_wandb"] = True
    finally:
        sys.modules["wandb"] = real_wandb_module

    # --- 5. eight negative cases ---
    def _expect_reject(name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
            negative[name] = {"rejected": False, "error_type": None, "error": None}
        except Exception as exc:  # noqa: BLE001 -- any raise is the expected "rejected" signal here
            negative[name] = {"rejected": True, "error_type": type(exc).__name__, "error": str(exc)}

    def _poisoned(fn: Callable[[], Any]) -> Callable[[], Any]:
        def _wrapped():
            sys.modules["wandb"] = None
            try:
                return fn()
            finally:
                sys.modules["wandb"] = real_wandb_module
        return _wrapped

    invalid_record = _build_synthetic_valid_record(
        wandb_run_id=run_id, wandb_sweep_id=sweep_id, trial_suffix="0002",
        overrides={"execution_status": "INVALID", "objective_score": None},
    )
    invalid_path = _write_json(out_dir / "negative_invalid_record.json", invalid_record)
    _expect_reject("invalid_record", _poisoned(lambda: recover_and_publish_objective(
        execution_provenance_path=invalid_path, expected_identity={},
        marker_path=out_dir / "marker_negative_invalid.json", project=args.project, entity=args.entity,
    )))

    incomplete_record = _build_synthetic_valid_record(
        wandb_run_id=run_id, wandb_sweep_id=sweep_id, trial_suffix="0003", overrides={"trial_id": None},
    )
    incomplete_path = _write_json(out_dir / "negative_incomplete_record.json", incomplete_record)
    _expect_reject("incomplete_record", _poisoned(lambda: recover_and_publish_objective(
        execution_provenance_path=incomplete_path, expected_identity={},
        marker_path=out_dir / "marker_negative_incomplete.json", project=args.project, entity=args.entity,
    )))

    non_finite_record = _build_synthetic_valid_record(
        wandb_run_id=run_id, wandb_sweep_id=sweep_id, trial_suffix="0004", overrides={"objective_score": float("nan")},
    )
    non_finite_path = _write_json(out_dir / "negative_non_finite_record.json", non_finite_record)
    _expect_reject("non_finite_objective", _poisoned(lambda: recover_and_publish_objective(
        execution_provenance_path=non_finite_path, expected_identity={},
        marker_path=out_dir / "marker_negative_non_finite.json", project=args.project, entity=args.entity,
    )))

    missing_hash_record = dict(_build_synthetic_valid_record(
        wandb_run_id=run_id, wandb_sweep_id=sweep_id, trial_suffix="0005",
    ))
    del missing_hash_record["generated_nh_config_sha256"]
    missing_hash_path = _write_json(out_dir / "negative_missing_source_hash_record.json", missing_hash_record)
    _expect_reject("missing_source_hashes", _poisoned(lambda: recover_and_publish_objective(
        execution_provenance_path=missing_hash_path, expected_identity={},
        marker_path=out_dir / "marker_negative_missing_hash.json", project=args.project, entity=args.entity,
    )))

    _expect_reject("mismatched_wandb_run_id", _poisoned(lambda: recover_and_publish_objective(
        execution_provenance_path=record_path, expected_identity={"wandb_run_id": "not-the-real-disposable-run-id"},
        marker_path=out_dir / "marker_negative_mismatched_run.json", project=args.project, entity=args.entity,
    )))

    _expect_reject("contradictory_identity", _poisoned(lambda: recover_and_publish_objective(
        execution_provenance_path=record_path, expected_identity={"configuration_id": "some-other-configuration-id"},
        marker_path=out_dir / "marker_negative_contradictory.json", project=args.project, entity=args.entity,
    )))

    stale_payload = {
        "flashnh/valid": True, "flashnh/objective_score": 0.999999, "flashnh/trial_id": record["trial_id"],
        "flashnh/retry_of_trial_id": None, "flashnh/execution_generation": 1, "flashnh/objective_recovered": True,
    }
    changed_marker_path = out_dir / "marker_negative_changed_objective.json"
    _write_json(changed_marker_path, {"wandb_run_id": run_id, "published_payload": stale_payload})
    _expect_reject("changed_objective", _poisoned(lambda: recover_and_publish_objective(
        execution_provenance_path=record_path, expected_identity=expected_identity,
        marker_path=changed_marker_path, project=args.project, entity=args.entity,
    )))

    # mismatched sweep is the ONE negative case that genuinely needs a live
    # wandb.Api() call: the run really exists and really belongs to
    # `sweep_id`, so a record claiming a different sweep must be caught by
    # the live sweepId comparison inside recover_and_publish_objective, not
    # by any pre-wandb local check.
    wrong_sweep_record = _build_synthetic_valid_record(
        wandb_run_id=run_id, wandb_sweep_id="not-the-real-disposable-sweep-id", trial_suffix="0006",
    )
    wrong_sweep_path = _write_json(out_dir / "negative_mismatched_sweep_record.json", wrong_sweep_record)
    _expect_reject("mismatched_sweep", lambda: recover_and_publish_objective(
        execution_provenance_path=wrong_sweep_path, expected_identity={"wandb_run_id": run_id},
        marker_path=out_dir / "marker_negative_mismatched_sweep.json", project=args.project, entity=args.entity,
    ))

    checks["negative_cases"] = negative
    checks["all_eight_negative_cases_rejected"] = all(v["rejected"] for v in negative.values())
    checks["negative_case_count"] = len(negative)

    overall_pass = (
        checks.get("clean_wandb_finish") is True
        and result1.get("status") == "published"
        and checks.get("summary_matches_record") is True
        and result2.get("status") == "already_published"
        and checks.get("idempotent_repeat_never_imported_wandb") is True
        and checks.get("all_eight_negative_cases_rejected") is True
        and checks.get("negative_case_count") == 8
    )

    evidence = {
        "qualification_kind": "sweep_v1_objective_recovery_qualification",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_head": args.expected_commit,
        "project": args.project,
        "entity": args.entity,
        "never_used_production_sweep_id": FORBIDDEN_PRODUCTION_SWEEP_ID,
        "never_used_production_run_id": FORBIDDEN_PRODUCTION_RUN_ID,
        "checks": checks,
        "overall_pass": overall_pass,
    }
    evidence_path = _write_json(out_dir / "qualification_evidence.json", evidence)

    print(json.dumps(evidence, indent=2, sort_keys=True, default=str))
    print(f"\nEvidence written to {evidence_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
