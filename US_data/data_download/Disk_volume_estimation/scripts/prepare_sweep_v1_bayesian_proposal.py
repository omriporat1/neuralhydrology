"""Prepare exactly one production Sweep-v1 Bayesian proposal; never train."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.baseline.sweep_v1_production_adapter import PreparationPaths, prepare_bayesian_proposal, write_prepared_proposal

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal-json", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--screening-basin-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", required=True)
    parser.add_argument("--baseline-policy-path", type=Path, default=ROOT / "config/stage1_scientific_baseline_v001.yaml")
    args = parser.parse_args()
    proposal = json.loads(args.proposal_json.read_text(encoding="utf-8"))
    canonical_splits = ROOT / "config" / "stage1_baseline_splits_v001"
    prepared = prepare_bayesian_proposal(proposal=proposal, paths=PreparationPaths(args.baseline_policy_path, args.package_root, canonical_splits, args.screening_basin_ids))
    record = write_prepared_proposal(prepared, args.output_dir)
    record_path = args.output_dir / "preparation_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(record_path)

if __name__ == "__main__":
    main()
