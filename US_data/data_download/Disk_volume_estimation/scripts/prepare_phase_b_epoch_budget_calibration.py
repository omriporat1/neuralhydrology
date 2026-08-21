"""Prepare-only contract for Phase-B Track-A epoch-budget calibration.

This module defines exactly the frozen C1--C5 cohort.  It writes generated
NH configurations and a compact audit only; it has no launcher, training,
W&B, HPO, or sealed-scope code.  Later operational work must call the
existing ``run_pilot`` machinery with the policy returned here, preserving a
single logical trajectory per candidate through epoch 14.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

_REPO_WORKDIR = Path(__file__).resolve().parent.parent
if str(_REPO_WORKDIR) not in sys.path:
    sys.path.insert(0, str(_REPO_WORKDIR))

from src.baseline.pilot_early_stopping import build_effective_policy
from src.baseline.pilot_lead06_config import PilotRunSpec, build_pilot_bundle, load_pilot_policy
from src.baseline.nh_config_generation import write_generated_config

CAMPAIGN_NAME = "phase_b_epoch_budget_calibration_seedA_v001"
TARGET_EPOCH = 14
MAX_UPDATES_PER_EPOCH = 50_000
SEED_A = 967139
PT_DYNAMIC_INPUTS = ("mrms_qpe_1h_mm", "rtma_2t_K")
_PROFILE = "pilot_lead06_emb128x32_seedA_v001"

EPOCH_BUDGET_CALIBRATION_RUN_SPECS = {
    "C1_anchor": PilotRunSpec("C1_anchor", "learned_fc_embedding", [128, 32], "seed_a", SEED_A, _PROFILE,
                               max_updates_per_epoch=MAX_UPDATES_PER_EPOCH, learning_rate=3e-4, hidden_size=128,
                               output_dropout=0.25, batch_size=256, seq_length=72, dynamic_inputs=PT_DYNAMIC_INPUTS),
    "C2_low_lr": PilotRunSpec("C2_low_lr", "learned_fc_embedding", [128, 32], "seed_a", SEED_A, _PROFILE,
                               max_updates_per_epoch=MAX_UPDATES_PER_EPOCH, learning_rate=1e-4, hidden_size=128,
                               output_dropout=0.25, batch_size=256, seq_length=72, dynamic_inputs=PT_DYNAMIC_INPUTS),
    "C3_high_lr": PilotRunSpec("C3_high_lr", "learned_fc_embedding", [128, 32], "seed_a", SEED_A, _PROFILE,
                                max_updates_per_epoch=MAX_UPDATES_PER_EPOCH, learning_rate=1e-3, hidden_size=128,
                                output_dropout=0.25, batch_size=256, seq_length=72, dynamic_inputs=PT_DYNAMIC_INPUTS),
    "C4_late_h64": PilotRunSpec("C4_late_h64", "learned_fc_embedding", [128, 32], "seed_a", SEED_A, _PROFILE,
                                 max_updates_per_epoch=MAX_UPDATES_PER_EPOCH, learning_rate=3e-4, hidden_size=64,
                                 output_dropout=0.25, batch_size=256, seq_length=72, dynamic_inputs=PT_DYNAMIC_INPUTS),
    "C5_convergence_stress": PilotRunSpec("C5_convergence_stress", "learned_fc_embedding", [128, 32], "seed_a", SEED_A, _PROFILE,
                                           max_updates_per_epoch=MAX_UPDATES_PER_EPOCH, learning_rate=3e-4, hidden_size=256,
                                           output_dropout=0.25, batch_size=128, seq_length=72, dynamic_inputs=PT_DYNAMIC_INPUTS),
}


def _resolve_policy_relative_paths(policy):
    def absolute(raw: str) -> str:
        path = Path(raw)
        return str(path if path.is_absolute() else _REPO_WORKDIR / path)
    return dataclasses.replace(policy, screening_basin_ids_path=absolute(policy.screening_basin_ids_path),
                               base_early_stopping_policy_path=absolute(policy.base_early_stopping_policy_path),
                               wandb_policy_path=absolute(policy.wandb_policy_path))


def build_epoch_budget_calibration_policy(base_policy):
    """Return an in-memory, explicit opt-in policy; historical YAML is untouched."""
    collisions = set(EPOCH_BUDGET_CALIBRATION_RUN_SPECS) & set(base_policy.runs)
    if collisions:
        raise RuntimeError(f"calibration run-id collision with historical pilot: {sorted(collisions)}")
    runs = dict(base_policy.runs)
    runs.update(EPOCH_BUDGET_CALIBRATION_RUN_SPECS)
    raw = dict(base_policy.raw)
    raw["policy_name"] = CAMPAIGN_NAME
    return dataclasses.replace(
        base_policy, raw=raw, runs=runs, seq_length=72,
        pilot_max_epoch_budget=TARGET_EPOCH, screening_validation_every_n_epochs=1,
        diagnostic_only_epoch=0, stopping_eligible_from_epoch=1,
        initial_training_epochs=TARGET_EPOCH, performance_early_stopping_enabled=False,
    )


def prepare_campaign(*, pilot_policy_path, baseline_policy_path, package_root, splits_dir, output_dir) -> Path:
    """Generate all five configs and write ``config_audit.json``; never train."""
    base = _resolve_policy_relative_paths(load_pilot_policy(pilot_policy_path))
    policy = build_epoch_budget_calibration_policy(base)
    effective = build_effective_policy(policy)
    output = Path(output_dir)
    rows = []
    for run_id in EPOCH_BUDGET_CALIBRATION_RUN_SPECS:
        bundle = build_pilot_bundle(pilot_policy=policy, run_id=run_id, baseline_policy_path=baseline_policy_path,
                                    package_root=package_root, splits_dir=splits_dir)
        config_paths = write_generated_config(bundle, output / run_id)
        config = bundle.config_mapping
        rows.append({
            "candidate_id": run_id, "learning_rate": config["learning_rate"], "hidden_size": config["hidden_size"],
            "batch_size": config["batch_size"], "dynamic_inputs": config["dynamic_inputs"], "seq_length": config["seq_length"],
            "statics_embedding": config["statics_embedding"], "output_dropout": config["output_dropout"], "seed": config["seed"],
            "training_segment_epochs": config["epochs"], "checkpoint_save_every_epochs": config["save_weights_every"],
            "target_epoch": TARGET_EPOCH,
            "max_updates_per_epoch": config["max_updates_per_epoch"], "screening_epochs": list(range(1, TARGET_EPOCH + 1)),
            "performance_early_stopping_enabled": False, "sealed_scope": False,
                "config_path": str(config_paths["config.yaml"]), "generation_manifest": str(config_paths["generation_manifest.json"]),
        })
    audit = {"campaign": CAMPAIGN_NAME, "audit_scope": "LOCAL_STRUCTURAL_AUDIT_ONLY",
             "canonical_package_validation_required_before_training": True, "candidate_count": len(rows), "target_epoch": TARGET_EPOCH,
             "max_updates_per_epoch": MAX_UPDATES_PER_EPOCH, "screening_epochs": list(range(1, TARGET_EPOCH + 1)),
             "training_cadence": "one_uninterrupted_segment_through_target_epoch",
             "checkpoint_retention": "save_weights_every_epoch",
             "raw_space_evaluation_cadence": "every_retained_epoch_post_training_or_existing_result",
             "performance_early_stopping_enabled": False, "effective_policy": effective,
             "no_wandb_or_hpo": True, "no_sealed_scope": True, "candidates": rows}
    output.mkdir(parents=True, exist_ok=True)
    audit_path = output / "config_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pilot-policy-path", type=Path, default=_REPO_WORKDIR / "config" / "stage1_lead06_pilot_v001.yaml")
    parser.add_argument("--baseline-policy-path", type=Path, default=_REPO_WORKDIR / "config" / "stage1_scientific_baseline_v001.yaml")
    parser.add_argument("--splits-dir", type=Path, default=_REPO_WORKDIR / "config" / "stage1_baseline_splits_v001")
    args = parser.parse_args()
    print(prepare_campaign(pilot_policy_path=args.pilot_policy_path, baseline_policy_path=args.baseline_policy_path,
                           package_root=args.package_root, splits_dir=args.splits_dir, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
