"""Prepare one Phase-B batch-size OPERATIONAL QUALIFICATION ONLY smoke.

This is deliberately not a Sweep-v1 candidate generator and does not rank
batch sizes.  It writes two configurations for exactly one approved batch
size: the intended Sweep-v1 configuration (50,000 updates/epoch) and a
separate one-epoch, eight-update operational smoke configuration.  The latter
exists solely to exercise the real NH CudaLSTM/data path on an allocated GPU.

This command performs no training.  The paired Moriah launcher is the only
supported execution path and calls this command before invoking NH.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

_REPO_WORKDIR = Path(__file__).resolve().parent.parent
if str(_REPO_WORKDIR) not in sys.path:
    sys.path.insert(0, str(_REPO_WORKDIR))

from src.baseline.nh_config_generation import (
    read_package_manifest,
    validate_full_population_basin_membership,
    write_generated_config,
)
from src.baseline.pilot_lead06_config import build_pilot_bundle_with_validation_scope
APPROVED_BATCH_SIZES = (128, 256, 512)
HIDDEN_SIZE_STRESS_POINT = 256
SWEEP_V1_UPDATES_PER_EPOCH = 50_000
OPERATIONAL_SMOKE_UPDATES = 8
OPERATIONAL_SMOKE_EPOCHS = 1
_PROFILE = "pilot_lead06_emb128x32_seedA_v001"
_LEAD_HOURS = 6
_SEQ_LENGTH = 72
_LEARNING_RATE = 3e-4
_EMBEDDING_DROPOUT = 0.10
_OUTPUT_DROPOUT = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(args, *, max_updates_per_epoch: int):
    """Build the normal development-only PT config with frozen H256 settings."""
    development_ids = validate_full_population_basin_membership(
        read_package_manifest(args.package_root), args.splits_dir
    ).development_basins
    return build_pilot_bundle_with_validation_scope(
        baseline_policy_path=args.baseline_policy_path,
        package_root=args.package_root,
        splits_dir=args.splits_dir,
        lead_hours=_LEAD_HOURS,
        seq_length=_SEQ_LENGTH,
        run_profile_name=_PROFILE,
        validation_basin_ids=development_ids,
        population_role="development_operational_qualification_only",
        package_type="phase_b_batch_size_operational_qualification_only",
        max_updates_per_epoch=max_updates_per_epoch,
        learning_rate=_LEARNING_RATE,
        hidden_size=HIDDEN_SIZE_STRESS_POINT,
        embedding_dropout=_EMBEDDING_DROPOUT,
        output_dropout=_OUTPUT_DROPOUT,
        batch_size=args.batch_size,
        dynamic_inputs=("mrms_qpe_1h_mm", "rtma_2t_K"),
    )


def _write_smoke_bundle(args, out_dir: Path):
    bundle = _make_bundle(args, max_updates_per_epoch=OPERATIONAL_SMOKE_UPDATES)
    mapping = dict(bundle.config_mapping)
    # This is intentionally distinct from the intended config above.  It
    # never validates or reports scientific metrics; eight actual training
    # updates are enough to exercise repeated forward/backward/optimizer work.
    mapping.update({"epochs": OPERATIONAL_SMOKE_EPOCHS, "validate_every": 100, "save_weights_every": 1})
    smoke_bundle = dataclasses.replace(bundle, config_mapping=mapping)
    return write_generated_config(
        smoke_bundle,
        out_dir,
        experiment_name=f"phase_b_batch_size_operational_qualification_only_bs{args.batch_size}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True, choices=APPROVED_BATCH_SIZES)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, default=_REPO_WORKDIR / "config" / "stage1_baseline_splits_v001")
    parser.add_argument("--baseline-policy-path", type=Path, default=_REPO_WORKDIR / "config" / "stage1_scientific_baseline_v001.yaml")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error(f"refusing to overwrite non-empty qualification directory: {args.out_dir}")

    intended_dir = args.out_dir / "intended_sweep_v1_config"
    intended_bundle = _make_bundle(args, max_updates_per_epoch=SWEEP_V1_UPDATES_PER_EPOCH)
    intended_paths = write_generated_config(
        intended_bundle,
        intended_dir,
        experiment_name=f"phase_b_sweep_v1_intended_bs{args.batch_size}",
    )
    smoke_paths = _write_smoke_bundle(args, args.out_dir / "operational_smoke_config")
    record = {
        "schema_name": "phase_b_batch_size_operational_qualification_preparation",
        "schema_version": 1,
        "classification": "OPERATIONAL QUALIFICATION ONLY — NOT SCIENTIFIC PERFORMANCE EVIDENCE",
        "batch_size": args.batch_size,
        "hidden_size": HIDDEN_SIZE_STRESS_POINT,
        "approved_batch_sizes": list(APPROVED_BATCH_SIZES),
        "intended_sweep_v1_updates_per_epoch": SWEEP_V1_UPDATES_PER_EPOCH,
        "operational_smoke_updates": OPERATIONAL_SMOKE_UPDATES,
        "operational_smoke_epochs": OPERATIONAL_SMOKE_EPOCHS,
        "no_validation_or_metric_ranking": True,
        "intended_config": {name: str(path) for name, path in intended_paths.items()},
        "operational_smoke_config": {name: str(path) for name, path in smoke_paths.items()},
        "config_sha256": {
            "intended": _sha256(intended_paths["config.yaml"]),
            "operational_smoke": _sha256(smoke_paths["config.yaml"]),
        },
        "development_membership_source": "validated full-population development split",
    }
    record_path = args.out_dir / "qualification_preparation.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
