#!/usr/bin/env python
"""Render Stage 1 observed-vs-predicted hydrographs (Part L.3).

Thin CLI over :mod:`src.baseline.hydrograph_rendering`: turns an existing
NeuralHydrology validation-results pickle into a deterministic compact
~6-8-basin comparison panel and/or a full rendering of the existing Part C
24-basin hydrograph atlas. Reads only -- never trains or evaluates, and
never connects to a remote cluster.

Only ``--period validation`` is permitted. ``--epoch`` must be the run's
*selected* checkpoint epoch (e.g. the one recorded in the run's closure
decision), not necessarily its last or stopping epoch -- this script has no
notion of "stop epoch" and never infers one; the caller must supply the
selected checkpoint explicitly.

Usage (paths and epoch are placeholders):
    python scripts/render_stage1_hydrographs.py \\
        --run-dir /path/to/nh_run_dir \\
        --period validation \\
        --epoch <selected_checkpoint_epoch> \\
        --package-root /path/to/stage1_scientific_package_v002 \\
        --target-variable qobs_mm_per_h_lead06 \\
        --lead-hours 6 \\
        --atlas-csv reports/stage1_validation_optimization_foundation_v001/part_c_hydrograph_atlas/selection_v001/hydrograph_atlas_basin_selection.csv \\
        --out-dir tmp/stage1_hydrograph_rendering_v001 \\
        --mode both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.hydrograph_rendering import (
    DEFAULT_COMPACT_TARGET_COUNT,
    HydrographRenderingError,
    render_stage1_hydrographs,
)


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None, help="NH run directory (used with --period/--epoch to locate the result pickle)")
    p.add_argument("--result-pickle", default=None, help="Explicit path to a <period>_results.p (alternative to --run-dir)")
    p.add_argument("--period", required=True, help="Evaluation period; only 'validation' is permitted")
    p.add_argument("--epoch", required=True, type=int, help="Checkpoint epoch")
    p.add_argument("--package-root", required=True, help="Certified Stage 1 package root (for basin-area self-derivation)")
    p.add_argument("--target-variable", required=True, help="Target variable name, e.g. qobs_mm_per_h_lead06")
    p.add_argument("--lead-hours", required=True, type=int, help="Lead time in hours, e.g. 6")
    p.add_argument("--atlas-csv", required=True, help="Path to the hydrograph-atlas basin-selection CSV")
    p.add_argument("--out-dir", required=True, help="Output directory for generated (untracked) artifacts")
    p.add_argument("--mode", choices=["compact", "full", "both"], default="both")
    p.add_argument("--compact-target-count", type=int, default=DEFAULT_COMPACT_TARGET_COUNT)
    p.add_argument("--freq", default=None, help="NH result-pickle frequency key, if the run has more than one")
    p.add_argument("--dry-run", action="store_true", help="Compute the selection/series but write no files")
    p.add_argument("--force", action="store_true", help="Allow writing into a non-empty --out-dir")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        summary = render_stage1_hydrographs(
            run_dir=args.run_dir,
            result_pickle=args.result_pickle,
            period=args.period,
            epoch=args.epoch,
            package_root=args.package_root,
            target_variable=args.target_variable,
            lead_hours=args.lead_hours,
            atlas_csv=args.atlas_csv,
            out_dir=args.out_dir,
            mode=args.mode,
            compact_target_count=args.compact_target_count,
            freq=args.freq,
            write_outputs=not args.dry_run,
            force=args.force,
            repo_root=REPO_ROOT,
        )
    except HydrographRenderingError as exc:
        _fail(str(exc))
        return 1

    if not args.dry_run:
        run_command_path = Path(args.out_dir) / "run_command.txt"
        argv_for_record = argv if argv is not None else sys.argv[1:]
        run_command_path.write_text(
            "python scripts/render_stage1_hydrographs.py " + " ".join(argv_for_record) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({k: v for k, v in summary.items() if k != "manifest"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
