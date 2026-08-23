"""QUALIFICATION ONLY -- NON-SCIENTIFIC.

CPU-only environment/credential preflight for the Phase-B online W&B sweep
qualification (docs/stage1_phase_b_sweep_v1_launch_contract.md, "Before
scientific sweep launch, complete online W&B qualification..."). Must run
inside a CPU Slurm allocation, never the login node -- importing the real
``wandb`` package counts as the compute workload this project's remote-
operations policy reserves for Slurm (docs/remote_operations.md section 2.2).

Checks, all read-only / no mutation:
  * exact Git commit guard (refuses on mismatch or a dirty tracked tree);
  * exact canonical runtime guard (refuses unless invoked with the intended
    production interpreter, unless explicitly overridden for local testing);
  * real ``wandb`` import + version;
  * ``wandb`` CLI availability in the same environment;
  * a CREDENTIAL-PRESENT BOOLEAN ONLY -- this script never prints, stores,
    or otherwise serializes ``WANDB_API_KEY`` or any netrc password. It
    checks presence via the environment variable and, as a fallback, via
    the existence of a netrc entry for the W&B API host (existence check
    only, the discovered login/password tuple is immediately discarded and
    never referenced again).

Never imports neuralhydrology or torch, never generates an NH config, never
touches a sealed temporal-test/spatial-holdout set, never installs or
upgrades any package.

Usage:
    python scripts/wandb_online_sweep_qualification_preflight.py \\
        --expected-commit <sha> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import netrc
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_CANONICAL_RUNTIME_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
_WANDB_NETRC_HOST = "api.wandb.ai"
_DEFAULT_OUT_DIR = _REPO_ROOT / "reports" / "wandb_online_sweep_qualification_v001" / "preflight"


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


def _credential_present() -> bool:
    """Boolean only. Never returns, logs, or stores the credential value."""
    env_value = os.environ.get("WANDB_API_KEY", "")
    if env_value.strip():
        return True
    try:
        authenticators = netrc.netrc().authenticators(_WANDB_NETRC_HOST)
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return authenticators is not None


def _wandb_cli_path() -> "str | None":
    sibling = Path(sys.executable).parent / "wandb"
    if sibling.is_file():
        return str(sibling)
    found = shutil.which("wandb")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expected-commit", type=str, required=True)
    parser.add_argument(
        "--expected-runtime-python", type=str, default=_CANONICAL_RUNTIME_PYTHON,
        help="Override only for local/off-Moriah testing; production qualification must use the default.",
    )
    parser.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    args = parser.parse_args()

    actual_head = _git_head(_REPO_ROOT)
    dirty_tracked = _git_dirty_tracked(_REPO_ROOT)

    runtime_python_matches = sys.executable == args.expected_runtime_python

    wandb_import_error: "str | None" = None
    wandb_version: "str | None" = None
    try:
        import wandb
        wandb_version = wandb.__version__
    except Exception as exc:  # noqa: BLE001 -- report, never raise past this point
        wandb_import_error = f"{type(exc).__name__}: {exc}"

    wandb_cli_path = _wandb_cli_path()
    credential_available = _credential_present()

    checks = {
        "git_commit_matches_expected": actual_head is not None and actual_head == args.expected_commit,
        "git_tracked_tree_clean": not dirty_tracked,
        "runtime_python_is_canonical": runtime_python_matches,
        "wandb_import_ok": wandb_import_error is None,
        "wandb_cli_available": wandb_cli_path is not None,
        "credential_available": credential_available,
    }
    all_checks_passed = all(checks.values())

    record = {
        "qualification_kind": "wandb_online_sweep_qualification_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git_head_actual": actual_head,
        "git_head_expected": args.expected_commit,
        "git_dirty_tracked_files": dirty_tracked,
        "python_executable_actual": sys.executable,
        "python_executable_expected": args.expected_runtime_python,
        "python_version": sys.version,
        "wandb_version": wandb_version,
        "wandb_import_error": wandb_import_error,
        "wandb_cli_path": wandb_cli_path,
        # BOOLEAN ONLY -- see _credential_present() docstring. No key/token
        # value is ever assigned to any variable retained past that call.
        "credential_available": credential_available,
        "checks": checks,
        "all_checks_passed": all_checks_passed,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight_record.json").write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(record, indent=2, sort_keys=True))
    print(f"\nPreflight record written to {out_dir / 'preflight_record.json'}")
    print(f"all_checks_passed={all_checks_passed}")

    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
