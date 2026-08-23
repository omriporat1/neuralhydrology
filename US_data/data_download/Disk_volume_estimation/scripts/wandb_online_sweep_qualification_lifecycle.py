"""QUALIFICATION ONLY -- NON-SCIENTIFIC.

Sweep-lifecycle control (pause / resume / stop / status) for the toy sweep
created by ``wandb_online_sweep_qualification_run.py``. Qualifies the W&B
lifecycle primitive that production Boundary Review 1/2 (~12 / ~24 valid
Bayesian results; docs/stage1_phase_b_sweep_v1_launch_contract.md, "Boundary
reviews and immutable waves") will need to pause new-proposal issuance while
in-flight scientific jobs finish. This script does NOT implement the
production checkpoint controller -- it only proves the underlying W&B
primitive works from this project's environment, via the ``wandb`` CLI
(``wandb sweep --pause/--resume/--stop <entity/project/sweep_id>``) for the
state transition and ``wandb.Api()`` for a read-only state query before/
after.

Must run inside a CPU Slurm allocation, never the login node (a ``status``
call alone makes an online API request). Never deliberately launches a
hanging agent merely to observe a paused state.

Usage:
    python scripts/wandb_online_sweep_qualification_lifecycle.py \\
        --sweep-id <entity/project/sweep_id> --action pause --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT_DIR = _REPO_ROOT / "reports" / "wandb_online_sweep_qualification_v001" / "lifecycle"

_VALID_ACTIONS = ("status", "pause", "resume", "stop")


def _wandb_cli_path() -> str:
    sibling = Path(sys.executable).parent / "wandb"
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("wandb")
    if found is None:
        raise SystemExit("wandb CLI not found next to the current interpreter or on PATH")
    return found


def _query_state(sweep_id: str) -> "str | None":
    import wandb

    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    return getattr(sweep, "state", None)


def _run_cli_action(cli_path: str, flag: str, sweep_id: str) -> dict:
    result = subprocess.run([cli_path, "sweep", flag, sweep_id], capture_output=True, text=True, timeout=60)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep-id", type=str, required=True, help="entity/project/sweep_id")
    parser.add_argument("--action", type=str, required=True, choices=_VALID_ACTIONS)
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    args = parser.parse_args()

    cli_path = _wandb_cli_path()

    state_before = _query_state(args.sweep_id)

    cli_result = None
    if args.action != "status":
        flag = f"--{args.action}"
        cli_result = _run_cli_action(cli_path, flag, args.sweep_id)

    state_after = _query_state(args.sweep_id)

    record = {
        "qualification_kind": "wandb_online_sweep_qualification_lifecycle",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "sweep_id": args.sweep_id,
        "action": args.action,
        "state_before": state_before,
        "state_after": state_after,
        "cli_result": cli_result,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"lifecycle_record_{args.action}.json"
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str), encoding="utf-8")

    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    print(f"\nLifecycle record written to {out_path}")

    if args.action != "status" and cli_result is not None and cli_result["returncode"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
