"""Build or dry-run the package-bound Common-120 support contract."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.baseline.common120_support_builder import Common120SupportError, build_common120_support
from src.baseline.fixed_support_contract_v2 import FixedSupportContractError, load_fixed_support_contract, write_fixed_support_contract

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    for name in ("package-root", "splits-dir", "screening-basin-ids-path", "baseline-policy-path", "policy-overlay-path"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--output")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not a.dry_run and not a.output: p.error("--output is required unless --dry-run")
    try:
        result = build_common120_support(
            package_root=a.package_root, splits_dir=a.splits_dir,
            screening_basin_ids_path=a.screening_basin_ids_path,
            baseline_policy_path=a.baseline_policy_path, policy_overlay_path=a.policy_overlay_path,
        )
        if not a.dry_run:
            write_fixed_support_contract(result.contract, a.output)
            load_fixed_support_contract(a.output)
    except (Common120SupportError, FixedSupportContractError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"dry_run": a.dry_run, "accounting": result.accounting, "checksum_sha256": result.contract["checksum_sha256"]}, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
