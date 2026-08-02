#!/usr/bin/env python
"""Build an evaluation-only NH run directory for re-evaluating an already
completed Stage 1 lead-06 pilot checkpoint against the fixed 24-basin
hydrograph atlas (Part L), instead of the run's original ~400-basin
screening-subset validation scope.

Composes three already-established, unmodified building blocks -- never
retrains, never refits a scaler, never touches temporal-test / spatial-holdout
/ California data:

1. ``src.baseline.hydrograph_rendering.load_atlas_selection_csv`` -- reads the
   committed Part C 24-basin atlas selection.
2. ``src.baseline.pilot_lead06_config.build_pilot_bundle_with_validation_scope``
   -- the same generalized bundle builder the original ~400-basin screening
   config was built with, given the atlas's 24 basin IDs as
   ``validation_basin_ids`` and the SAME ``run_profile_name`` as the
   original run, so architecture/hyperparameters/dates are reproduced
   unchanged except for basin-file membership.
3. ``src.baseline.nh_seed_evaluation.prepare_development_population_eval_run_dir``
   -- copies the original run's checkpoint + scaler byte-for-byte into a new,
   separate run directory and writes an EVALUATION_ONLY_DO_NOT_TRAIN.txt
   marker; never writes into the original training run directory.

After this script succeeds, evaluate with the ordinary:
    python scripts/run_stage1_nh.py eval <out-run-dir> --period validation --epoch <epoch>

Usage:
    python scripts/build_stage1_atlas24_eval_run_dir.py \\
        --development-run-dir /path/to/.../stage1_lead06_pilot_emb128x64_seedA_v001_2807_113156 \\
        --epoch 6 \\
        --package-root /path/to/stage1_scientific_package_v002 \\
        --atlas-csv reports/.../hydrograph_atlas_basin_selection.csv \\
        --generated-bundle-dir /path/to/tmp/atlas24_eval_generated \\
        --out-run-dir /path/to/emb128x64_seedA_epoch006_atlas24_validation_eval_v001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baseline.hydrograph_rendering import HydrographRenderingError, load_atlas_selection_csv
from src.baseline.nh_config_generation import (
    PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME,
    NHConfigGenerationError,
    write_generated_config,
)
from src.baseline.nh_seed_evaluation import NHSeedEvaluationError, prepare_development_population_eval_run_dir
from src.baseline.package_audit import sha256_file
from src.baseline.pilot_lead06_config import PilotConfigError, build_pilot_bundle_with_validation_scope

ATLAS24_POPULATION_ROLE = "atlas24_hydrograph_validation_visualization_only"


def _fail(message: str) -> None:
    print(f"FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--development-run-dir", required=True,
                   help="Original completed training run dir (holds model_epoch{E:03d}.pt + train_data/)")
    p.add_argument("--epoch", type=int, required=True, help="Frozen/selected checkpoint epoch, e.g. 6")
    p.add_argument("--package-root", required=True, help="Certified stage1_scientific_package_v002 root")
    p.add_argument("--policy", default=str(REPO_ROOT / "config" / "stage1_scientific_baseline_v001.yaml"),
                   help="Path to the Stage 1 scientific baseline policy YAML")
    p.add_argument("--splits-dir", default=str(REPO_ROOT / "config" / "stage1_baseline_splits_v001"),
                   help="Directory containing development_train.txt / spatial_holdout_nonca.txt / california_all.txt")
    p.add_argument("--atlas-csv", required=True, help="Path to the hydrograph-atlas basin-selection CSV (24 basins)")
    p.add_argument("--run-profile", default=PILOT_LEAD06_EMB128X64_SEEDA_PROFILE_NAME,
                   help="Named training-hyperparameter profile; MUST match the original run's profile "
                        "so architecture/hyperparameters are reproduced unchanged (default: the "
                        "emb128x64_seedA pilot profile)")
    p.add_argument("--generated-bundle-dir", required=True,
                   help="Output dir for the generated config.yaml/basin-lists/generation_manifest.json")
    p.add_argument("--out-run-dir", required=True,
                   help="Output dir for the final NH-Tester-compatible evaluation-only run directory")
    p.add_argument("--force", action="store_true", help="Allow overwriting non-empty output directories")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        atlas_df = load_atlas_selection_csv(args.atlas_csv)
    except HydrographRenderingError as exc:
        _fail(f"atlas CSV: {exc}")
        return 1
    atlas_basin_ids = sorted(atlas_df["gauge_id"])
    if len(atlas_basin_ids) != 24:
        _fail(f"atlas CSV {args.atlas_csv} yielded {len(atlas_basin_ids)} basin IDs, expected exactly 24")
        return 1

    try:
        bundle = build_pilot_bundle_with_validation_scope(
            baseline_policy_path=args.policy,
            package_root=args.package_root,
            splits_dir=args.splits_dir,
            lead_hours=6,
            seq_length=24,
            run_profile_name=args.run_profile,
            validation_basin_ids=atlas_basin_ids,
            population_role=ATLAS24_POPULATION_ROLE,
            package_type="stage1_lead06_pilot_emb128x64_seedA_atlas24_eval_v001",
        )
    except PilotConfigError as exc:
        _fail(f"bundle construction (basin membership / policy contract): {exc}")
        return 1

    try:
        generated_paths = write_generated_config(
            bundle, args.generated_bundle_dir,
            experiment_name="stage1_lead06_pilot_emb128x64_seedA_atlas24_eval_v001",
            force=args.force,
        )
    except NHConfigGenerationError as exc:
        _fail(f"write_generated_config: {exc}")
        return 1

    try:
        run_dir_manifest = prepare_development_population_eval_run_dir(
            development_run_dir=args.development_run_dir,
            epoch=args.epoch,
            eval_generated_dir=args.generated_bundle_dir,
            out_run_dir=args.out_run_dir,
            force=args.force,
        )
    except NHSeedEvaluationError as exc:
        _fail(f"prepare_development_population_eval_run_dir: {exc}")
        return 1

    summary = {
        "atlas_basin_count": len(atlas_basin_ids),
        "atlas_basin_ids": atlas_basin_ids,
        "atlas_csv_path": str(Path(args.atlas_csv)),
        "atlas_csv_sha256": sha256_file(args.atlas_csv),
        "run_profile_name": args.run_profile,
        "generated_bundle_dir": str(Path(args.generated_bundle_dir)),
        "generated_config_yaml": str(generated_paths["config.yaml"]),
        "out_run_dir": str(Path(args.out_run_dir)),
        "out_run_dir_manifest": run_dir_manifest,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
