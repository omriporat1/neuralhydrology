#!/usr/bin/env python3
"""Select the 50-basin pilot subset for the Flash-NH Stage 1 pilot.

SELECTION LOGIC
---------------
Preferred composition (configured in YAML basin_selection section):
  - 40 basins from TRAIN_CORE or TRAIN_SOFT_KEEP (initial training set)
  -  5 basins from HOLDOUT_REVIEW (withheld for secondary review; used here for QC)
  -  5 basins from EXCLUDE_TRAINING (hard exclusions; used here for forced-failure QC only)

If fallback_all_train=true and the HOLDOUT/EXCLUDE strata do not have enough basins,
all 50 basins are selected from the training set. This limitation is documented in
the output manifest.

REPRODUCIBILITY
---------------
Selection uses Python's random.Random with a fixed seed (random_seed in config).
The seed, selection logic, and source file are recorded in the output manifest.

OUTPUT
------
Writes pilot_basin_manifest.csv to {data_root}/09_manifests/stage1_pilot/.
Also writes provenance files (manifest.json, summary.md, etc.) to the same dir.

Usage:
    python scripts/select_pilot_basins.py --config configs/pilot_stage1.yaml
    python scripts/select_pilot_basins.py --data-root /my/data/root --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.config import load_config, config_to_dict, PipelineConfig
from src.pipeline.provenance import write_run_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "pilot_stage1.yaml"),
        help="Path to YAML config",
    )
    p.add_argument("--data-root", default=None, help="Override data root path")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selection without writing files",
    )
    return p.parse_args()


def _load_status_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def _select(
    rows: list[dict],
    n_train: int,
    n_holdout: int,
    n_exclude: int,
    seed: int,
    fallback_all_train: bool,
) -> tuple[list[dict], dict]:
    """Return (selected_rows, selection_metadata)."""
    import random
    rng = random.Random(seed)

    train = [r for r in rows if r.get("final_training_status") in ("TRAIN_CORE", "TRAIN_SOFT_KEEP")]
    holdout = [r for r in rows if r.get("final_training_status") == "HOLDOUT_REVIEW"]
    exclude = [r for r in rows if r.get("final_training_status") == "EXCLUDE_TRAINING"]

    has_holdout = len(holdout) >= n_holdout
    has_exclude = len(exclude) >= n_exclude

    if fallback_all_train and (not has_holdout or not has_exclude):
        total = n_train + n_holdout + n_exclude
        sel = [dict(r, pilot_role="TRAIN") for r in rng.sample(train, min(total, len(train)))]
        meta = {
            "used_fallback_all_train": True,
            "fallback_reason": (
                f"HOLDOUT_REVIEW available={len(holdout)} (need {n_holdout}), "
                f"EXCLUDE_TRAINING available={len(exclude)} (need {n_exclude})"
            ),
            "composition": {"TRAIN": len(sel)},
        }
        return sel, meta

    sel_train = [dict(r, pilot_role="TRAIN") for r in rng.sample(train, min(n_train, len(train)))]
    sel_holdout = [dict(r, pilot_role="HOLDOUT_QC") for r in rng.sample(holdout, min(n_holdout, len(holdout)))]
    sel_exclude = [dict(r, pilot_role="EXCLUDE_QC") for r in rng.sample(exclude, min(n_exclude, len(exclude)))]

    meta = {
        "used_fallback_all_train": False,
        "composition": {
            "TRAIN": len(sel_train),
            "HOLDOUT_QC": len(sel_holdout),
            "EXCLUDE_QC": len(sel_exclude),
        },
    }
    return sel_train + sel_holdout + sel_exclude, meta


def _write_manifest_csv(out_path: Path, basins: list[dict]) -> None:
    if not basins:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("STAID,final_training_status,pilot_role\n", encoding="utf-8")
        return
    # Canonical front columns; remaining columns appended
    front = ["STAID", "final_training_status", "pilot_role"]
    rest = [k for k in basins[0] if k not in set(front)]
    fieldnames = front + rest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(basins)


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
    bs = cfg.pilot.basin_selection

    # Locate basin status file
    status_file = cfg.resolve_basin_status_file()
    if status_file is None or not status_file.exists():
        expected = (
            REPO_ROOT / "reports" / "flashnh_final_basin_selection_v001"
            / "tables" / "final_basin_training_status.csv"
        )
        print(f"ERROR: basin status file not found.")
        print(f"  Expected: {expected}")
        print(f"  Set pilot.basin_selection.basin_status_file in config if it lives elsewhere.")
        sys.exit(1)

    print(f"Basin status file: {status_file}")
    rows = _load_status_csv(status_file)
    print(f"  Loaded {len(rows):,} basins from status file")

    selected, sel_meta = _select(
        rows,
        n_train=bs.n_train_core_keep,
        n_holdout=bs.n_holdout_review,
        n_exclude=bs.n_exclude_qc_only,
        seed=bs.random_seed,
        fallback_all_train=bs.fallback_all_train,
    )

    composition = sel_meta.get("composition", {})
    print(f"\nPilot selection (seed={bs.random_seed}):")
    for role, count in sorted(composition.items()):
        print(f"  {role:20s}: {count}")
    print(f"  {'TOTAL':20s}: {len(selected)}")

    if sel_meta.get("used_fallback_all_train"):
        print(f"\n  NOTE: fallback_all_train triggered — all basins from training set.")
        print(f"  Reason: {sel_meta.get('fallback_reason', '')}")

    manifest_dir = data_root / "09_manifests" / "stage1_pilot"
    manifest_csv = manifest_dir / "pilot_basin_manifest.csv"

    if dry_run:
        print(f"\n[DRY-RUN] Would write pilot manifest to: {manifest_csv}")
        print(f"[DRY-RUN] Sample STAIDs: {[r['STAID'] for r in selected[:5]]}")
        return

    _write_manifest_csv(manifest_csv, selected)
    print(f"\nWritten: {manifest_csv}")

    run_cmd = f"python scripts/select_pilot_basins.py --config {args.config}"
    if args.data_root:
        run_cmd += f" --data-root {args.data_root}"

    validation = {
        "status_file_found": True,
        "n_basins_selected_nonzero": len(selected) > 0,
        "manifest_csv_written": manifest_csv.exists(),
        "composition_as_expected": not sel_meta.get("used_fallback_all_train", False),
    }
    extra = {
        "n_basins_selected": len(selected),
        "selection_seed": bs.random_seed,
        "composition": composition,
        "selection_metadata": sel_meta,
    }

    write_run_manifest(
        manifest_dir,
        run_command=run_cmd,
        config_dict=config_to_dict(cfg),
        input_paths={"basin_status_file": str(status_file)},
        output_paths={"pilot_basin_manifest": str(manifest_csv)},
        validation_results=validation,
        extra=extra,
    )
    print(f"Manifest: {manifest_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
