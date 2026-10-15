#!/usr/bin/env python
"""Prepare the seqlen_common336_seed3_v001 inputs (CPU only; no training).

Subcommands:
  support  -- build the Common-336 training support + screening contract and
              per-basin accounting from the verified package/split/screening
              identities (``build_common336_support``). ``--subset-n N``
              calibrates on the first N training + first N screening basins
              (accounting only; no artifact is written).
  configs  -- derive the 15 NH configs + campaign_manifest.json from the frozen
              P19 config (``prepare_campaign``).
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline import seqlen_common336_campaign as camp  # noqa: E402
from src.baseline import seqlen_common336_support as c336  # noqa: E402

_DATA = "/sci/labs/efratmorin/omripo/Flash-NH/data"


def _support(args) -> int:
    train = [l.strip() for l in args.train_basin_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    screening = [l.strip() for l in args.screening_basin_ids.read_text(encoding="utf-8").splitlines() if l.strip()]
    subset = None
    if args.subset_n:
        subset = sorted(set(sorted(train)[:args.subset_n]) | set(sorted(screening)[:args.subset_n]))
    started = time.time()

    def progress(i, n, basin):
        if i == 1 or i % 100 == 0 or i == n:
            print(f"[{time.time() - started:8.1f}s] {i}/{n} {basin}", flush=True)

    result = c336.build_common336_support(
        package_root=args.package_root, splits_dir=args.splits_dir, screening_basin_ids_path=args.screening_basin_ids,
        baseline_policy_path=args.baseline_policy_path, train_basin_file=args.train_basin_file,
        dynamic_inputs=args.dynamic_inputs, basin_subset=subset, progress=progress)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accounting = result["accounting"]
    accounting["elapsed_s"] = time.time() - started
    accounting["max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    tag = f"subset{args.subset_n}" if subset else "full"
    (args.output_dir / f"accounting_{tag}.json").write_text(json.dumps(accounting, indent=1, sort_keys=True))
    print(json.dumps({"failures": len(accounting["failures"]), "rows": len(accounting["rows"]),
                      "elapsed_s": round(accounting["elapsed_s"], 1), "max_rss_mb": round(accounting["max_rss_mb"])}))
    if result["training_support"] is not None:
        c336.write_training_support(result["training_support"], args.output_dir / "common336_training_support_v001.json")
        c336.write_common336_screening_contract(result["screening_contract"],
                                                args.output_dir / "common336_screening_contract_v001.json")
    return 1 if accounting["failures"] else 0


def _configs(args) -> int:
    result = camp.prepare_campaign(base_config_path=args.base_config, base_manifest_path=args.base_manifest,
                                   training_support_path=args.training_support,
                                   screening_contract_path=args.screening_contract, output_root=args.output_root)
    print(json.dumps([{k: r[k] for k in ("index", "run_id", "config_sha256")} for r in result["runs"]], indent=1))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("support")
    s.add_argument("--package-root", type=Path, default=Path(f"{_DATA}/stage1_scientific_package_v002"))
    s.add_argument("--splits-dir", type=Path, default=ROOT / "config" / "stage1_baseline_splits_v001")
    s.add_argument("--screening-basin-ids", type=Path, default=Path(
        f"{_DATA}/screening_subsets/stage1_provisional_operational_screening_subset_v001/screening_subset_basin_ids.txt"))
    s.add_argument("--baseline-policy-path", type=Path, default=ROOT / "config" / "stage1_scientific_baseline_v001.yaml")
    s.add_argument("--train-basin-file", type=Path, required=True)
    s.add_argument("--dynamic-inputs", nargs="+", default=["mrms_qpe_1h_mm", "rtma_2t_K"])
    s.add_argument("--subset-n", type=int, default=0)
    s.add_argument("--output-dir", type=Path, required=True)
    c = sub.add_parser("configs")
    c.add_argument("--base-config", type=Path, required=True)
    c.add_argument("--base-manifest", type=Path, required=True)
    c.add_argument("--training-support", type=Path, required=True)
    c.add_argument("--screening-contract", type=Path, required=True)
    c.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    return _support(args) if args.cmd == "support" else _configs(args)


if __name__ == "__main__":
    raise SystemExit(main())
