#!/usr/bin/env python3
"""Create the Flash-NH data root directory structure.

Creates the numbered output directories under the configured data root,
writes an identifying README.txt in each top-level directory, and writes
a bootstrap manifest to 09_manifests/bootstrap/. Idempotent: safe to run
multiple times; existing directories are left untouched.

Usage:
    python scripts/bootstrap_data_root.py --config configs/pilot_stage1.yaml
    python scripts/bootstrap_data_root.py --data-root /my/external/Flash-NH_data
    FLASHNH_DATA_ROOT=/my/external/Flash-NH_data python scripts/bootstrap_data_root.py

Dry-run (print only, no writes):
    python scripts/bootstrap_data_root.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import load_config, config_to_dict, PipelineConfig
from src.pipeline.provenance import write_run_manifest


# Human-readable descriptions written into each top-level directory's README.txt
DIR_DESCRIPTIONS: dict[str, str] = {
    "00_raw": (
        "Raw acquired files in vendor-native formats (GRIB2, NC4, HDF, gzip).\n"
        "Organized by source/date. Never commit large raw files to git.\n"
        "Raw files are often re-downloadable; use manifests to track provenance."
    ),
    "01_standardized_grids": (
        "Optional standardized gridded products (Zarr, NetCDF).\n"
        "Standardization means consistent variable names, units, coordinates,\n"
        "and metadata — NOT resampling to a common grid. Each source keeps its\n"
        "native resolution. Not required for the first pilot."
    ),
    "02_basin_geometries": (
        "Canonical CAMELSH basin polygons (GeoPackage) and precomputed\n"
        "basin-grid weight tables (Parquet) for each gridded source product.\n"
        "Weights are computed once and reused for all timesteps."
    ),
    "03_basin_timeseries": (
        "Basin-average time series (Parquet). Derived from gridded products\n"
        "via precomputed weights. Organized by stage and version.\n"
        "Both long-form (basin/time/source/variable/value) and\n"
        "wide-form (basin/time/one-column-per-variable) formats."
    ),
    "04_ml_datasets": (
        "Frozen versioned model-ready datasets for NeuralHydrology training.\n"
        "Each version folder contains dataset_config.yaml, manifest.json,\n"
        "dynamic_inputs.parquet, static_attributes.parquet,\n"
        "target_streamflow.parquet, split files, and normalization stats."
    ),
    "05_splits": (
        "Train/validation/test split definitions.\n"
        "train_basins.txt, val_basins.txt, test_basins.txt, time_splits.yaml.\n"
        "Small files; commit to git for reproducibility."
    ),
    "06_qc_reports": (
        "Automated QC reports, static plots, and missingness summaries.\n"
        "Organized by stage. Not committed to git (regeneratable).\n"
        "Level 1: numeric checks. Level 2: static plots. Level 3: animations."
    ),
    "08_logs": (
        "Pipeline run logs and acquisition transcripts.\n"
        "Not committed to git. Keep recent logs; archive older ones."
    ),
    "09_manifests": (
        "Provenance manifests, run summaries, and request specs.\n"
        "Small JSON/YAML files. Commit curated manifests to git."
    ),
    "tmp": (
        "Scratch and temporary intermediate files.\n"
        "Never committed to git. Safe to delete at any time."
    ),
}

# Product-specific subdirectories to create under each top-level dir
PRODUCT_SUBDIRS: dict[str, list[str]] = {
    "00_raw": ["mrms", "rtma"],
    "02_basin_geometries": ["weights/mrms", "weights/rtma"],
    "03_basin_timeseries": ["stage1_pilot"],
    "04_ml_datasets": ["stage1_pilot_v001"],
    "06_qc_reports": ["stage1_pilot"],
    "09_manifests": ["stage1_pilot", "bootstrap"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "pilot_stage1.yaml"),
        help="Path to YAML config (default: configs/pilot_stage1.yaml)",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Override data root path (else uses config / FLASHNH_DATA_ROOT env)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without writing anything",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        cfg = load_config(config_path)
        print(f"Config: {config_path}")
    else:
        cfg = PipelineConfig()
        print(f"Config not found at {config_path}; using defaults.")

    data_root = cfg.effective_data_root(override=args.data_root)
    dry_run = args.dry_run

    label = "[DRY-RUN] " if dry_run else ""
    print(f"{label}Data root: {data_root}")
    print()

    dirs_created: list[str] = []
    dirs_existing: list[str] = []

    def _handle_dir(target: Path, readme_text: str | None = None) -> None:
        if dry_run:
            tag = "EXISTS" if target.exists() else "CREATE"
            print(f"  [{tag}] {target}")
            return
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            if readme_text:
                (target / "README.txt").write_text(
                    f"Flash-NH data: {target.name}\n\n{readme_text}\n",
                    encoding="utf-8",
                )
            dirs_created.append(str(target))
        else:
            dirs_existing.append(str(target))

    for name, description in DIR_DESCRIPTIONS.items():
        _handle_dir(data_root / name, readme_text=description)
        for subdir in PRODUCT_SUBDIRS.get(name, []):
            _handle_dir(data_root / name / subdir, readme_text=None)

    if dry_run:
        top_count = len(DIR_DESCRIPTIONS)
        sub_count = sum(len(v) for v in PRODUCT_SUBDIRS.values())
        print(f"\n[DRY-RUN] Would create up to {top_count + sub_count} directories under: {data_root}")
        return

    # Validate and write manifest
    all_present = all((data_root / d).exists() for d in DIR_DESCRIPTIONS)
    validation = {
        "data_root_accessible": data_root.exists(),
        "all_top_level_dirs_present": all_present,
        "n_dirs_created": len(dirs_created) > 0 or all_present,
    }

    manifest_dir = data_root / "09_manifests" / "bootstrap"
    run_cmd = f"python scripts/bootstrap_data_root.py --config {args.config}"
    if args.data_root:
        run_cmd += f" --data-root {args.data_root}"

    write_run_manifest(
        manifest_dir,
        run_command=run_cmd,
        config_dict=config_to_dict(cfg),
        output_paths={"data_root": str(data_root)},
        validation_results=validation,
    )

    print(f"Bootstrap complete.")
    print(f"  Created:  {len(dirs_created)} directories")
    print(f"  Existing: {len(dirs_existing)} directories")
    print(f"  Manifest: {manifest_dir / 'manifest.json'}")

    if not all_present:
        missing = [d for d in DIR_DESCRIPTIONS if not (data_root / d).exists()]
        print(f"\nWARNING: missing top-level dirs: {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
