"""QUALIFICATION ONLY -- NON-SCIENTIFIC.

Empirically determines -- never assumes -- whether a manually-initialized
W&B run (``wandb.init(settings=wandb.Settings(sweep_id=...))``, never
``wandb.agent()``) is incorporated into a real hosted sweep's run population
and Bayesian-controller-visible history, and whether doing so consumes an
additional controller-issued proposal. This is the disposable, real-package
qualification the exact-retry bridge design
(``scripts/run_sweep_v1_exact_retry_bridge.py``) rests on -- source-code
inspection of the installed wandb 0.28.1 SDK (``sdk/internal/sender.py``:
``upsert_run(..., sweep_name=run.sweep_id or None, ...)`` is the SAME
run-creation call an agent-launched run makes; the controller's
``register_agent``/``agent_heartbeat``/``_command_run`` machinery that
requests a NEW proposal lives entirely in ``wandb_agent.py`` and is never
invoked by bare ``wandb.init()``) strongly predicted this mechanism works,
but this script proves it against a real, disposable, non-scientific hosted
sweep rather than resting on that prediction alone.

Must run inside a CPU Slurm allocation, never the login node. Never imports
neuralhydrology/torch, never touches scientific data, never writes to any
file/counter the real frozen Sweep-v1 campaign
(``src/baseline/sweep_v1_campaign.py``) reads, and never touches the real
production sweep or any real Sweep-v1 run identity. Every run/sweep this
script creates is clearly labeled non-scientific
(``qualification_kind``, ``online_sweep_qualification: True``,
``scientific_trial: False`` -- reusing
``scripts.wandb_online_sweep_qualification_toy.build_run_identity``) and
uses a campaign/sweep/metric name deliberately distinct from both the real
Sweep-v1 identity and the pre-existing
``phase_b_wandb_online_sweep_qualification_v001`` toy-agent qualification
wave.

Scenario exercised, all against ONE fresh disposable sweep:
  1. ``wandb.agent(sweep_id, function=..., count=1)`` -- one real
     controller-issued proposal ("original"), objective logged and finite.
  2. A MANUAL run join with the EXACT SAME five axes as (1)
     (``wandb.init(settings=wandb.Settings(sweep_id=sweep_id), config=...)``,
     no ``wandb.agent()`` anywhere around this call) -- simulates an exact
     VALID retry: logs a finite objective.
  3. A second MANUAL run join, same mechanism, but that deliberately never
     logs the objective metric -- simulates an exact INVALID retry: no
     objective is ever published for it.
  4. ``wandb.Api()`` inspection: the sweep's run population includes all of
     (1)-(3); the manual runs' ``run.sweep_id`` matches; the metric is
     present (finite) for (1)/(2) and absent for (3).
  5. A second REAL ``wandb.agent(sweep_id, function=..., count=1)`` proposal
     -- proves the controller can still legitimately serve a next proposal
     over a history containing manually-joined runs, without error, and
     that the sweep's run population grows to 4 without disturbing (1)-(3).
  6. Re-fetch (1)-(3) via the API after step 5 and confirm their config/
     summary are byte-identical to what was recorded right after they
     finished -- the manual joins/second proposal never mutated them.

This script does NOT attempt to statistically prove the controller's
Bayesian posterior was numerically influenced by the manually-joined runs
(that would require many repeated sweeps and is out of scope) -- it proves
the STRUCTURAL claims the exact-retry design depends on: no additional
proposal is consumed by a manual join, the manual run is exposed as a
first-class sweep observation via the same API real Sweep-v1 tooling would
use, a missing objective is preserved as missing (never fabricated), and
none of this corrupts the controller's ability to keep serving proposals.

Usage:
    python scripts/wandb_exact_retry_join_qualification.py \\
        --expected-commit <sha> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.wandb_online_sweep_qualification_toy import (  # noqa: E402
    DEFAULT_PROJECT,
    build_run_identity,
    check_flashnh_legality,
    compute_toy_objective,
)
from src.baseline.sweep_v1_campaign import SEARCH_DOMAIN  # noqa: E402

_CANONICAL_RUNTIME_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
_DEFAULT_OUT_DIR = _REPO_ROOT / "reports" / "wandb_exact_retry_join_qualification_v001"

QUALIFICATION_CAMPAIGN_ID = "phase_b_wandb_exact_retry_join_qualification_v001"
TOY_METRIC_NAME = "qualification/exact_retry_toy_objective"
SWEEP_NAME = "phase_b_wandb_exact_retry_join_qualification_v001_bayes"
QUALIFICATION_TAGS = ("qualification", "non_scientific", "exact_retry_join", QUALIFICATION_CAMPAIGN_ID)

_AXES = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")


def _build_sweep_config() -> dict:
    lr = SEARCH_DOMAIN["learning_rate"]; hidden = SEARCH_DOMAIN["hidden_size"]
    emb = SEARCH_DOMAIN["embedding_dropout"]; out = SEARCH_DOMAIN["output_dropout"]; batch = SEARCH_DOMAIN["batch_size"]
    return {
        "method": "bayes", "name": SWEEP_NAME, "metric": {"name": TOY_METRIC_NAME, "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "log_uniform_values", "min": lr["lower"], "max": lr["upper"]},
            "hidden_size": {"values": list(hidden["values"])},
            "embedding_dropout": {"distribution": "uniform", "min": emb["lower"], "max": emb["upper"]},
            "output_dropout": {"distribution": "uniform", "min": out["lower"], "max": out["upper"]},
            "batch_size": {"values": list(batch["values"])},
        },
    }


def _git_head(repo_root: Path) -> "str | None":
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, check=True, timeout=10)
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
        raise SystemExit(f"REFUSING: python executable {sys.executable!r} != expected canonical runtime {expected_runtime_python!r}")


def _verify_online(run, label: str, captured: dict) -> None:
    reported_mode = getattr(getattr(run, "settings", None), "mode", None)
    try:
        hosted_url = run.get_url()
    except Exception:  # noqa: BLE001 -- absence itself is the signal
        hosted_url = None
    if reported_mode != "online" or not hosted_url or not run.id:
        captured[f"{label}_online_confirmed"] = False
        raise SystemExit(
            f"QUALIFICATION FAIL ({label}): run did not verifiably start ONLINE "
            f"(mode={reported_mode!r}, url={hosted_url!r}, id={run.id!r}). No silent offline fallback is accepted."
        )
    captured[f"{label}_online_confirmed"] = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expected-commit", type=str, required=True)
    parser.add_argument("--expected-runtime-python", type=str, default=_CANONICAL_RUNTIME_PYTHON)
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    args = parser.parse_args()

    _guard_or_die(expected_commit=args.expected_commit, expected_runtime_python=args.expected_runtime_python)

    import wandb

    sweep_config = _build_sweep_config()
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)

    captured: dict = {}

    # --- Step 1: real controller-issued proposal ("original") -----------
    def _agent_fn_original() -> None:
        run = wandb.init(mode="online", project=args.project, entity=args.entity, group=SWEEP_NAME,
                          job_type="qualification_controller_original", tags=list(QUALIFICATION_TAGS),
                          config=build_run_identity("controller_original"))
        try:
            _verify_online(run, "original", captured)
            proposed = {k: run.config[k] for k in _AXES}
            objective = compute_toy_objective(proposed)
            run.log({TOY_METRIC_NAME: objective})
            captured["original"] = {
                "run_id": run.id, "entity": run.entity, "project": run.project,
                "proposed_config": proposed, "objective": objective,
                "legality": check_flashnh_legality(proposed),
            }
        finally:
            run.finish()

    wandb.agent(sweep_id, function=_agent_fn_original, project=args.project, entity=args.entity, count=1)

    entity = captured["original"]["entity"]
    project = captured["original"]["project"]
    exact_axes = captured["original"]["proposed_config"]

    # --- Step 2: manual join, EXACT SAME axes, VALID retry (logs objective) --
    manual_valid_run = wandb.init(
        mode="online", settings=wandb.Settings(sweep_id=sweep_id), project=project, entity=entity,
        config=dict(exact_axes), group=SWEEP_NAME, job_type="qualification_manual_exact_retry_valid",
        tags=list(QUALIFICATION_TAGS) + ["manual_join", "no_agent", "simulated_valid_retry"],
    )
    _verify_online(manual_valid_run, "manual_valid", captured)
    manual_valid_sweep_id_matches = manual_valid_run.sweep_id == sweep_id
    manual_valid_objective = compute_toy_objective(exact_axes)
    manual_valid_run.log({TOY_METRIC_NAME: manual_valid_objective})
    manual_valid_run.summary["qualification/exact_retry_of_run_id"] = captured["original"]["run_id"]
    manual_valid_run_id = manual_valid_run.id
    manual_valid_run.finish()

    # --- Step 3: manual join, EXACT SAME axes, INVALID retry (no objective logged) --
    manual_invalid_run = wandb.init(
        mode="online", settings=wandb.Settings(sweep_id=sweep_id), project=project, entity=entity,
        config=dict(exact_axes), group=SWEEP_NAME, job_type="qualification_manual_exact_retry_invalid",
        tags=list(QUALIFICATION_TAGS) + ["manual_join", "no_agent", "simulated_invalid_retry"],
    )
    _verify_online(manual_invalid_run, "manual_invalid", captured)
    manual_invalid_sweep_id_matches = manual_invalid_run.sweep_id == sweep_id
    manual_invalid_run.summary["qualification/exact_retry_of_run_id"] = captured["original"]["run_id"]
    manual_invalid_run.summary["qualification/simulated_invalid_no_objective"] = True
    manual_invalid_run_id = manual_invalid_run.id
    manual_invalid_run.finish()

    # --- Step 4: API inspection after manual joins, before any 2nd proposal --
    api = wandb.Api()
    sweep_obj = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs_after_manual = {r.id: r for r in sweep_obj.runs}
    manual_valid_present = manual_valid_run_id in runs_after_manual
    manual_invalid_present = manual_invalid_run_id in runs_after_manual
    original_present = captured["original"]["run_id"] in runs_after_manual
    manual_valid_metric_visible = runs_after_manual[manual_valid_run_id].summary.get(TOY_METRIC_NAME) is not None if manual_valid_present else False
    manual_invalid_metric_absent = runs_after_manual[manual_invalid_run_id].summary.get(TOY_METRIC_NAME) is None if manual_invalid_present else False
    run_count_after_manual = len(runs_after_manual)

    # --- Step 5: a second REAL controller proposal over the now-3-run history --
    def _agent_fn_second() -> None:
        run = wandb.init(mode="online", project=args.project, entity=args.entity, group=SWEEP_NAME,
                          job_type="qualification_controller_second", tags=list(QUALIFICATION_TAGS),
                          config=build_run_identity("controller_second"))
        try:
            _verify_online(run, "second", captured)
            proposed = {k: run.config[k] for k in _AXES}
            objective = compute_toy_objective(proposed)
            run.log({TOY_METRIC_NAME: objective})
            captured["second"] = {"run_id": run.id, "proposed_config": proposed, "objective": objective}
        finally:
            run.finish()

    wandb.agent(sweep_id, function=_agent_fn_second, project=args.project, entity=args.entity, count=1)

    # --- Step 6: re-fetch and confirm nothing prior was mutated ----------
    sweep_obj_final = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs_final = {r.id: r for r in sweep_obj_final.runs}
    run_count_final = len(runs_final)
    manual_valid_unchanged = (
        manual_valid_run_id in runs_final
        and dict(runs_final[manual_valid_run_id].config) == dict(exact_axes)
        and runs_final[manual_valid_run_id].summary.get(TOY_METRIC_NAME) == manual_valid_objective
    )
    manual_invalid_unchanged = (
        manual_invalid_run_id in runs_final
        and dict(runs_final[manual_invalid_run_id].config) == dict(exact_axes)
        and runs_final[manual_invalid_run_id].summary.get(TOY_METRIC_NAME) is None
    )
    original_unchanged = (
        captured["original"]["run_id"] in runs_final
        and runs_final[captured["original"]["run_id"]].summary.get(TOY_METRIC_NAME) == captured["original"]["objective"]
    )

    checks = {
        "step1_original_online": captured.get("original_online_confirmed") is True,
        "step2_manual_valid_online": captured.get("manual_valid_online_confirmed") is True,
        "step2_manual_valid_sweep_id_matches": manual_valid_sweep_id_matches,
        "step3_manual_invalid_online": captured.get("manual_invalid_online_confirmed") is True,
        "step3_manual_invalid_sweep_id_matches": manual_invalid_sweep_id_matches,
        "step4_run_count_after_manual_is_3": run_count_after_manual == 3,
        "step4_original_present": original_present,
        "step4_manual_valid_present": manual_valid_present,
        "step4_manual_invalid_present": manual_invalid_present,
        "step4_manual_valid_metric_visible": manual_valid_metric_visible,
        "step4_manual_invalid_metric_absent": manual_invalid_metric_absent,
        "step5_second_proposal_online": captured.get("second_online_confirmed") is True,
        "step6_run_count_final_is_4": run_count_final == 4,
        "step6_manual_valid_unchanged": manual_valid_unchanged,
        "step6_manual_invalid_unchanged": manual_invalid_unchanged,
        "step6_original_unchanged": original_unchanged,
    }
    all_checks_passed = all(checks.values())

    record = {
        "qualification_kind": "wandb_exact_retry_join_qualification",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_head": args.expected_commit,
        "project": project, "entity": entity, "sweep_id": sweep_id, "sweep_config": sweep_config,
        "original_run_id": captured["original"]["run_id"],
        "manual_valid_run_id": manual_valid_run_id,
        "manual_invalid_run_id": manual_invalid_run_id,
        "second_proposal_run_id": captured.get("second", {}).get("run_id"),
        "exact_axes_used_for_manual_joins": exact_axes,
        "run_count_after_manual_joins": run_count_after_manual,
        "run_count_final": run_count_final,
        "checks": checks,
        "all_checks_passed": all_checks_passed,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exact_retry_join_qualification_record.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    print(f"\nQualification record written to {out_path}")
    print(f"all_checks_passed={all_checks_passed}")

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
