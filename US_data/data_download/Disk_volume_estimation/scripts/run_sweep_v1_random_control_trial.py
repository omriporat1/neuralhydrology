"""Non-W&B-agent-driven entry point for one Sweep-v1 random-control trial.

Random-control rows are a fixed, pre-committed manifest
(``config/stage1_phase_b_sweep_v1_original_domain_v001_random_control_manifest.json``)
-- unlike the Bayesian arm, no live W&B search proposes them, so this script
never imports or calls ``wandb`` at all. It reuses exactly the same
prepare -> write -> execute -> interpret path as the Bayesian bridge
(``scripts/run_sweep_v1_wandb_bridge.py``):

  1. ``prepare_random_control_row(...)`` -- verifies the manifest's frozen
     SHA-256 and that the requested row is an exact committed manifest row
     (never regenerated/recomputed).
  2. ``write_prepared_proposal(...)``.
  3. ``run_prepared_trial_in_production(...)`` -- the same real, fully-tested
     Sweep-v1 execution/interpretation layer the Bayesian arm uses;
     arm-agnostic per ``src/baseline/sweep_v1_execution.py``'s
     ``run_prepared_trial_in_production`` docstring (neither
     ``build_execution_context`` nor ``execute_prepared_trial`` branches on
     ``search_arm``).

One Slurm allocation runs exactly one random-control row -- the same
one-allocation-per-trial rule as ``run_sweep_v1_wandb_agent_moriah.sbatch``
applies to the Bayesian arm, enforced here by requiring exactly one row
index per invocation rather than looping over the manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline.sweep_v1_execution import run_prepared_trial_in_production
from src.baseline.sweep_v1_production_adapter import (
    PreparationPaths, prepare_random_control_row, write_prepared_proposal,
)

_DEFAULT_MANIFEST = ROOT / "config/stage1_phase_b_sweep_v1_original_domain_v001_random_control_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--row-index", type=int, required=True, help="0-based index into the committed manifest's rows.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--screening-basin-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-policy-path", type=Path, default=ROOT / "config/stage1_scientific_baseline_v001.yaml")
    parser.add_argument("--base-pilot-policy-path", type=Path, default=ROOT / "config/stage1_lead06_pilot_v001.yaml")
    args = parser.parse_args()

    payload = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if not (0 <= args.row_index < len(rows)):
        raise SystemExit(f"--row-index {args.row_index} out of range for {len(rows)} committed rows")
    row = rows[args.row_index]

    canonical_splits = ROOT / "config" / "stage1_baseline_splits_v001"
    paths = PreparationPaths(args.baseline_policy_path, args.package_root, canonical_splits, args.screening_basin_ids)

    prepared = prepare_random_control_row(row=row, manifest_path=args.manifest_path, paths=paths)
    output_dir = args.output_root / prepared.trial_id
    record = write_prepared_proposal(prepared, output_dir)

    outcome = run_prepared_trial_in_production(
        prepared_record=record, output_dir=output_dir, paths=paths,
        base_pilot_policy_path=args.base_pilot_policy_path,
    )
    trial = outcome["review_records"]["trial_summary"]
    print(json.dumps({"trial_id": record["trial_id"], "valid": outcome["valid"],
                      "objective_score": trial["objective_score"]}, indent=2))
    return 0 if outcome["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
