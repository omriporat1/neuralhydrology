"""QUALIFICATION ONLY -- NON-SCIENTIFIC.

Creates (or reuses) one tiny REAL hosted W&B Bayesian sweep and runs exactly
ONE agent proposal against it, online. This is the compute-node half of the
Phase-B online W&B sweep qualification gate (docs/
stage1_phase_b_sweep_v1_launch_contract.md: "Before scientific sweep
launch, complete online W&B qualification on a CPU Slurm allocation...").

Must run inside a CPU Slurm allocation, never the login node. Never imports
neuralhydrology/torch, never touches scientific data, never generates an NH
training config, and never writes to any file/counter the real frozen
Sweep-v1 campaign (``src/baseline/sweep_v1_campaign.py``) reads.

Architecture qualified here matches the production design: one bounded
Slurm allocation -> one W&B agent -> exactly one proposal/run
(``wandb.agent(..., count=1)``); this script never loops over multiple
proposals.

Online-mode is verified, not assumed: after ``wandb.init(mode="online")``
this script checks the resulting run's reported mode, hosted URL, and run
ID. If W&B silently degrades to offline, this script FAILS qualification
rather than reporting partial success -- per this task's explicit
instruction not to accept a silent offline fallback as a pass.

Usage (first job, creates the sweep):
    python scripts/wandb_online_sweep_qualification_run.py \\
        --expected-commit <sha> --proposal-label first --out-dir <dir>

Usage (second job, reuses the sweep created by the first):
    python scripts/wandb_online_sweep_qualification_run.py \\
        --expected-commit <sha> --proposal-label second \\
        --sweep-id <entity/project/sweep_id> --out-dir <dir>
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
    QUALIFICATION_TAGS,
    SWEEP_NAME,
    build_run_identity,
    build_sweep_config,
    check_flashnh_legality,
    compute_toy_objective,
)

_CANONICAL_RUNTIME_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
_DEFAULT_OUT_DIR = _REPO_ROOT / "reports" / "wandb_online_sweep_qualification_v001" / "run"


def _git_head(repo_root: Path) -> "str | None":
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, check=True, timeout=10
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
        raise SystemExit(f"REFUSING: python executable {sys.executable!r} != expected canonical runtime {expected_runtime_python!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expected-commit", type=str, required=True)
    parser.add_argument("--expected-runtime-python", type=str, default=_CANONICAL_RUNTIME_PYTHON)
    parser.add_argument("--proposal-label", type=str, required=True, help='e.g. "first" or "second"')
    parser.add_argument("--sweep-id", type=str, default=None, help="Reuse an existing sweep instead of creating a new one.")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    args = parser.parse_args()

    _guard_or_die(expected_commit=args.expected_commit, expected_runtime_python=args.expected_runtime_python)

    import wandb

    created_new_sweep = args.sweep_id is None
    if created_new_sweep:
        sweep_config = build_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
    else:
        sweep_id = args.sweep_id
        sweep_config = None

    captured: dict = {}

    def _agent_fn() -> None:
        run = wandb.init(
            mode="online",
            project=args.project,
            entity=args.entity,
            group=SWEEP_NAME,
            job_type="qualification_toy_agent",
            tags=list(QUALIFICATION_TAGS),
            config=build_run_identity(args.proposal_label),
        )
        try:
            reported_mode = getattr(getattr(run, "settings", None), "mode", None)
            hosted_url = None
            try:
                hosted_url = run.get_url()
            except Exception:  # noqa: BLE001 -- absence itself is the signal
                hosted_url = None

            if reported_mode != "online" or not hosted_url or not run.id:
                captured["online_confirmed"] = False
                captured["failure_reason"] = (
                    f"non-online or unverifiable run: mode={reported_mode!r} "
                    f"url={hosted_url!r} run_id={run.id!r}"
                )
                raise SystemExit(
                    "QUALIFICATION FAIL: run did not verifiably start ONLINE "
                    f"(mode={reported_mode!r}, url={hosted_url!r}, id={run.id!r}). "
                    "No silent offline fallback is accepted."
                )

            proposed = {k: run.config[k] for k in ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")}
            objective = compute_toy_objective(proposed)
            legality = check_flashnh_legality(proposed)

            run.log({"qualification/toy_objective": objective})

            captured.update(
                {
                    "online_confirmed": True,
                    "reported_mode": reported_mode,
                    "run_id": run.id,
                    "run_url": hosted_url,
                    "proposed_config": proposed,
                    "toy_objective": objective,
                    "flashnh_legality": legality,
                }
            )
        finally:
            run.finish()

    wandb.agent(sweep_id, function=_agent_fn, project=args.project, entity=args.entity, count=1)

    record = {
        "qualification_kind": "wandb_online_sweep_qualification_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_head": args.expected_commit,
        "proposal_label": args.proposal_label,
        "project": args.project,
        "entity": args.entity,
        "sweep_id": sweep_id,
        "created_new_sweep": created_new_sweep,
        "sweep_config": sweep_config,
        **captured,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_record_{args.proposal_label}.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    print(f"\nRun record written to {out_path}")

    return 0 if captured.get("online_confirmed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
