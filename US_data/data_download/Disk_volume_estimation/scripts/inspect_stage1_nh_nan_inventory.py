#!/usr/bin/env python
"""Read-only dynamic-input NaN/gap-flag inventory over a Stage 1 package
(local implementation increment, no training).

Argument parsing and I/O wiring only. All check logic lives in
:func:`src.baseline.nh_structural_preflight.inspect_dynamic_nan_inventory`.
Never writes to --package-root; intended to be run against the real
certified Stage 1 Compact Scientific Package on h2o to verify that
continuous-forcing NaNs occur only at the documented MRMS/RTMA gap
timestamps and that mrms_qpe_1h_mm_gap / rtma_gap agree with the actual
per-variable missingness they are meant to flag.

Usage:
    python scripts/inspect_stage1_nh_nan_inventory.py \\
        --package-root /path/to/stage1_compact_scientific_package_v001 \\
        --basin-ids-file tmp/stage1_nh_config_lead06_seq24_v001/train_basins.txt \\
        --dynamic-inputs mrms_qpe_1h_mm rtma_2t_K rtma_2d_K rtma_2sh_kgkg \\
            rtma_10u_ms rtma_10v_ms mrms_qpe_1h_mm_gap rtma_gap \\
        --out inventory.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.nh_structural_preflight import NHStructuralPreflightError, inspect_dynamic_nan_inventory


def _fail(message: str) -> int:
    print(f"FATAL: {message}", file=sys.stderr)
    return 2


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package-root", required=True, help="Package root to read (never modified)")
    p.add_argument("--basin-ids-file", required=True,
                   help="Newline-delimited basin-ID file, e.g. train_basins.txt from generate_stage1_nh_config.py")
    p.add_argument("--dynamic-inputs", nargs="+", required=True,
                   help="Dynamic input + gap-flag variable names to inspect, in any order")
    p.add_argument("--gap-flag-suffix", default="_gap", help="Suffix used for 1:1 <name>+suffix gap-flag inference")
    p.add_argument("--out", default=None, help="Optional path to also write the JSON result to")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    basin_ids = [
        ln.strip() for ln in Path(args.basin_ids_file).read_text(encoding="utf-8").splitlines() if ln.strip()
    ]

    try:
        inventory = inspect_dynamic_nan_inventory(
            args.package_root,
            dynamic_inputs=args.dynamic_inputs,
            basin_ids=basin_ids,
            gap_flag_suffix=args.gap_flag_suffix,
        )
    except NHStructuralPreflightError as exc:
        return _fail(str(exc))

    result = {
        "package_root": str(Path(args.package_root)),
        "basin_count": len(basin_ids),
        "dynamic_inputs": list(args.dynamic_inputs),
        "per_variable_nan_counts": inventory.per_variable_nan_counts,
        "per_basin_nan_counts": inventory.per_basin_nan_counts,
        "basin_masks_identical": inventory.basin_masks_identical,
        "nan_outside_documented_gaps": inventory.nan_outside_documented_gaps,
        "gap_flag_mismatches": inventory.gap_flag_mismatches,
        "clean": not inventory.nan_outside_documented_gaps and not inventory.gap_flag_mismatches,
    }
    text = json.dumps(result, indent=2, default=str)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if result["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
