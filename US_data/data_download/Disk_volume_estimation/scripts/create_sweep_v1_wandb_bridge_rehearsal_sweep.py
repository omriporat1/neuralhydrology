"""QUALIFICATION ONLY -- NON-SCIENTIFIC.

Registers one disposable, non-scientific Sweep-v1 REHEARSAL sweep against
the real W&B service, and writes the matching
``src.baseline.sweep_v1_wandb_bridge_manifest`` launch manifest that
``scripts/run_sweep_v1_wandb_bridge.py main_from_manifest`` will consume when
a real ``wandb agent`` invokes it (see
``scripts/run_sweep_v1_wandb_bridge_rehearsal_moriah.sbatch``).

This is the one-time setup step for Section G of the fresh-proposal
W&B-agent bridge qualification: it never calls ``wandb.agent()`` or
``wandb.init()`` itself, never trains, never imports neuralhydrology/torch,
and never references the production sweep (``4x3btz2s``) or production run
(``ardib08c``) -- the manifest schema (``sweep_v1_wandb_bridge_manifest``)
independently refuses a rehearsal manifest that targets the production
sweep, and this script additionally asserts it before writing anything.

Ordering (chicken-and-egg, resolved the same way
``sweep_v1_exact_retry_rehearsal_v001``'s own rehearsal manifest was built):
the disposable sweep's own ``command`` field must embed the manifest's
absolute path (so a real ``wandb agent`` round trip routes to
``main_from_manifest``), so the manifest PATH is fixed first, the sweep is
registered referencing that not-yet-existing path, and only once a real
``wandb_sweep_id`` comes back is the manifest itself written (with that real
id inside it).

Must run inside a CPU Slurm allocation (or an interactive CPU-only session)
on Moriah with the canonical ``flashnh-moriah`` interpreter -- never the
login node for anything beyond this cheap registration call, and never a
GPU. Requires real W&B network credentials already established on Moriah
(the same credential store every other Sweep-v1 qualification script here
uses); the local Windows dev environment does not have them, which is why
this does not run locally.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.baseline.sweep_v1_execution import build_wandb_bridge_rehearsal_sweep_config  # noqa: E402
from src.baseline.sweep_v1_runtime_contract import run_full_runtime_contract  # noqa: E402
from src.baseline.sweep_v1_wandb_bridge_manifest import (  # noqa: E402
    MODE_REHEARSAL, PRODUCTION_WANDB_SWEEP_ID, REHEARSAL_RESERVED_EXECUTION_GENERATION,
    REHEARSAL_RESERVED_PROPOSAL_ORDER, write_wandb_bridge_manifest,
)

_CANONICAL_RUNTIME_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"
_DEFAULT_WANDB_PROJECT = "flashnh-stage1"
_DEFAULT_WANDB_ENTITY = "omri-porat1-huji"

# Real, already-qualified production identities (mirrored, not rederived,
# from .scratch_local/sweep_v1_launch_manifests/attempt005_v001.json) -- the
# fresh bridge needs the same real package/screening/policy inputs the
# production bridge uses; only the sweep id, output root, proposal
# order/generation, and stop_before_training diverge for rehearsal.
_REAL_PACKAGE_ROOT = "/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_scientific_package_v002"
_REAL_SCREENING_BASIN_IDS_PATH = (
    "/sci/labs/efratmorin/omripo/Flash-NH/data/screening_subsets/"
    "stage1_provisional_operational_screening_subset_v001/screening_subset_basin_ids.txt"
)
_REAL_BASELINE_POLICY_PATH = str(_REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml")
_REAL_BASE_PILOT_POLICY_PATH = str(_REPO_ROOT / "config" / "stage1_lead06_pilot_v001.yaml")

_DEFAULT_MANIFEST_PATH = (
    _REPO_ROOT / ".scratch_local" / "sweep_v1_wandb_bridge_launch_manifests" / "rehearsal_v001.json"
)
_DEFAULT_OUTPUT_ROOT = "/sci/labs/efratmorin/omripo/Flash-NH/evidence/sweep_v1_wandb_bridge_rehearsal_v001"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--manifest-path", type=Path, default=_DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wandb-project", default=_DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=_DEFAULT_WANDB_ENTITY)
    args = parser.parse_args()

    run_full_runtime_contract(
        repo_root=_REPO_ROOT, expected_commit=args.expected_commit,
        expected_runtime_python=_CANONICAL_RUNTIME_PYTHON,
    )

    manifest_path = args.manifest_path
    if manifest_path.exists():
        raise SystemExit(f"REFUSING: manifest path already exists: {manifest_path}")

    sweep_config = build_wandb_bridge_rehearsal_sweep_config(
        program="scripts/run_sweep_v1_wandb_bridge.py", manifest_path=str(manifest_path.resolve()),
    )

    import wandb
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)

    if sweep_id == PRODUCTION_WANDB_SWEEP_ID:
        raise SystemExit(
            f"REFUSING: W&B returned the production sweep id ({PRODUCTION_WANDB_SWEEP_ID!r}) for what must be a "
            "disposable rehearsal sweep; aborting before writing any manifest"
        )

    manifest = write_wandb_bridge_manifest(
        manifest_path,
        manifest_label="sweep_v1_wandb_bridge_rehearsal_v001",
        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode=MODE_REHEARSAL,
        expected_commit=args.expected_commit,
        expected_runtime_python=_CANONICAL_RUNTIME_PYTHON,
        package_root=_REAL_PACKAGE_ROOT,
        screening_basin_ids_path=_REAL_SCREENING_BASIN_IDS_PATH,
        output_root=args.output_root,
        baseline_policy_path=_REAL_BASELINE_POLICY_PATH,
        base_pilot_policy_path=_REAL_BASE_PILOT_POLICY_PATH,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_sweep_id=sweep_id,
        proposal_order=REHEARSAL_RESERVED_PROPOSAL_ORDER,
        execution_generation=REHEARSAL_RESERVED_EXECUTION_GENERATION,
        stop_before_training=True,
    )

    print(json.dumps({
        "wandb_sweep_id": sweep_id,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "output_root": args.output_root,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
