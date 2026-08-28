"""One-agent v2 fresh-proposal bridge. Never creates controller proposals."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import pilot_orchestration
from src.baseline.fixed_support_contract_v2 import load_fixed_support_contract
from src.baseline.nh_config_generation import read_package_manifest, validate_full_population_basin_membership
from src.baseline.pilot_lead06_config import load_screening_basin_ids
from src.baseline import sweep_v1_campaign as sweep
from src.baseline.sweep_v1_execution import enrich_layer_b_provenance
from src.baseline.sweep_v2_six_axis_campaign import (
    FORBIDDEN_V1_SWEEP_ID, SweepV2CampaignError, canonical_hyperparameters_v2, normalize_seq_length_axis,
)
from src.baseline.sweep_v2_six_axis_config import V2_METRIC_NAME, V2_REHEARSAL_PLACEHOLDER_METRIC_NAME
from src.baseline.sweep_v2_six_axis_execution import (
    build_execution_context_v2, build_v2_epoch_evaluator, build_v2_objective_publication_payload,
    execute_prepared_trial_v2, select_executor_mode_v2, write_proposal_intake_provenance_v2,
)
from src.baseline.sweep_v2_six_axis_production_adapter import (
    PreparationPathsV2, prepare_bayesian_proposal_v2, write_prepared_proposal_v2,
)

_AXES = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size", "seq_length")


class SweepV2BridgeRefusal(ValueError):
    """A controlled bridge boundary refused an unsafe controller assignment."""


def _incident(root: Path, run_id: str | None, stage: str, reason: str) -> None:
    path = root / f"bootstrap_assignment_rejected__wandb_run_{run_id or 'unknown'}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "execution_provenance.json").write_text(
        json.dumps({"provenance_stage": stage, "rejection_reason": reason, "wandb_run_id": run_id}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _controller_axes(run: Any, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return immutable-by-convention raw controller axes and canonical axes.

    The first mapping is the exact six-key controller representation.  The
    second is a separate mapping whose seq_length has passed the authoritative
    normalizer before it can reach identity, intake, paths, or execution.
    """
    keys = set(dict(run.config)) - {"_wandb"}
    if keys != set(_AXES):
        reason = f"exact six-axis run.config required; missing={sorted(set(_AXES) - keys)} unexpected={sorted(keys - set(_AXES))}"
        _incident(output_root, getattr(run, "id", None), "controller_config_shape_rejected", reason)
        raise SweepV2BridgeRefusal(reason)
    raw = {key: run.config[key] for key in _AXES}
    canonical = dict(raw)
    try:
        canonical["seq_length"] = normalize_seq_length_axis(raw["seq_length"])
    except SweepV2CampaignError as exc:
        reason = "controller_axis_value_rejected"
        _incident(output_root, getattr(run, "id", None), reason, "controller seq_length violates the v2 domain")
        raise SweepV2BridgeRefusal(reason) from exc
    return raw, canonical_hyperparameters_v2(canonical)


def _validate_objective_metric_contract(run: Any, *, manifest: Mapping[str, Any], output_root: Path) -> None:
    """Check the joined live sweep's optimizer metric before proposal intake."""
    import wandb

    sweep_path = f"{getattr(run, 'entity', None)}/{getattr(run, 'project', None)}/{run.sweep_id}"
    try:
        config = wandb.Api().sweep(sweep_path).config
    # W&B does not guarantee one stable public exception base across client
    # versions.  Keep this broad boundary confined to the sole external call;
    # no later local metric/configuration validation is caught here.
    except Exception as exc:
        reason = "objective_metric_contract_unverifiable"
        _incident(output_root, getattr(run, "id", None), reason, "unable to verify the joined sweep metric contract")
        raise SweepV2BridgeRefusal(reason) from exc
    expected_metric_name = (
        V2_METRIC_NAME
        if manifest["mode"] == "production"
        else V2_REHEARSAL_PLACEHOLDER_METRIC_NAME
    )
    metric = config.get("metric") or {}
    if (config.get("method") != "bayes" or metric.get("name") != expected_metric_name
            or metric.get("goal") != "maximize"):
        reason = (
            f"joined sweep metric contract mismatch: mode={manifest['mode']!r} "
            f"method={config.get('method')!r} metric={metric!r} "
            f"expected={{'name': {expected_metric_name!r}, 'goal': 'maximize'}}"
        )
        _incident(output_root, getattr(run, "id", None), "objective_metric_contract_rejected", reason)
        raise SweepV2BridgeRefusal(reason)


def _validate_manifest_repository_root(manifest: Mapping[str, Any]) -> Path:
    declared = Path(str(manifest["repository_root"])).resolve()
    actual = ROOT.resolve()
    if declared != actual:
        raise SweepV2BridgeRefusal(f"manifest repository_root {declared!s} does not match bridge repository root {actual!s}")
    return declared


def _validate_screening_binding(manifest: Mapping[str, Any]) -> None:
    """Establish frozen-policy = manifest = loaded-artifact checksum agreement."""
    declared = manifest["screening_basin_ids_sha256"]
    if declared != sweep.SCREENING_ARTIFACT_SHA256:
        raise SweepV2BridgeRefusal("manifest screening checksum contradicts frozen screening identity")
    membership = validate_full_population_basin_membership(
        read_package_manifest(manifest["package_root"]), ROOT / "config" / "stage1_baseline_splits_v001",
    )
    load_screening_basin_ids(
        manifest["screening_basin_ids_path"], development_basins=membership.development_basins,
        expected_count=400, expected_sha256=declared,
    )


def _execute(manifest: Mapping[str, Any]) -> int:
    if manifest["wandb_sweep_id"] == FORBIDDEN_V1_SWEEP_ID:
        raise SweepV2BridgeRefusal("v1 sweep forbidden")
    _validate_manifest_repository_root(manifest)
    contract = load_fixed_support_contract(manifest["fixed_support_contract_path"])
    if (contract["checksum_sha256"] != manifest["fixed_support_contract_sha256"]
            or contract["contract_id"] != manifest["fixed_support_contract_version"]
            or len(contract["basin_ids"]) != 400):
        raise SweepV2BridgeRefusal("fixed-support manifest binding mismatch or non-production support")
    _validate_screening_binding(manifest)

    import wandb
    run = wandb.init()
    try:
        if run.sweep_id != manifest["wandb_sweep_id"]:
            _incident(Path(manifest["output_root"]), run.id, "sweep_identity_rejected", "unexpected sweep")
            raise SweepV2BridgeRefusal("unexpected sweep")
        _validate_objective_metric_contract(run, manifest=manifest, output_root=Path(manifest["output_root"]))
        raw_axes, canonical_axes = _controller_axes(run, Path(manifest["output_root"]))
        intake = write_proposal_intake_provenance_v2(
            output_root=manifest["output_root"], axes=canonical_axes, search_arm="bayesian",
            proposal_order=manifest["proposal_order"], wandb_sweep_id=run.sweep_id, wandb_run_id=run.id,
            support_contract_version=contract["contract_id"], support_contract_sha256=contract["checksum_sha256"],
            execution_generation=manifest["execution_generation"],
        )
        output_dir = Path(manifest["output_root"]) / intake["trial_id"]
        enrich_layer_b_provenance(output_dir=output_dir, stage="proposal_intake", fields={
            "launch_manifest_sha256": manifest["manifest_sha256"],
            "raw_controller_axes": raw_axes,
            "normalized_controller_axes": canonical_axes,
        })
        paths = PreparationPathsV2(
            Path(manifest["baseline_policy_path"]), Path(manifest["policy_overlay_path"]),
            Path(manifest["package_root"]), ROOT / "config" / "stage1_baseline_splits_v001",
            Path(manifest["screening_basin_ids_path"]), Path(manifest["fixed_support_contract_path"]),
        )
        proposal = {**canonical_axes, "proposal_order": manifest["proposal_order"],
                    "execution_generation": manifest["execution_generation"], "wandb_sweep_id": run.sweep_id,
                    "wandb_run_id": run.id}
        prepared = prepare_bayesian_proposal_v2(proposal=proposal, paths=paths)
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields=prepared.evidence)
        record = write_prepared_proposal_v2(prepared, output_dir, allow_layer_b_provenance=True)
        mode = select_executor_mode_v2(record)
        enrich_layer_b_provenance(output_dir=output_dir, stage="executor_mode_selected", fields={"executor_mode": mode})
        if manifest["stop_before_training"]:
            run.summary.update({"flashnh/rehearsal_stopped_before_training": True, "flashnh/trial_id": intake["trial_id"]})
            return 0
        context = build_execution_context_v2(
            prepared_record=record, paths=paths, base_pilot_policy_path=manifest["base_pilot_policy_path"],
        )
        evaluator = build_v2_epoch_evaluator(
            support_contract=contract, package_root=paths.package_root, screening_basin_ids=context.screening_basin_ids,
        )

        def execute():
            return pilot_orchestration.execute_prepared_pilot_run_monolithic(
                execution_policy=context.execution_policy, config_dir=context.config_dir,
                experiment_name=context.experiment_name, package_root=context.package_root,
                target_variable=context.target_variable, lead_hours=context.lead_hours,
                screening_basin_ids=context.screening_basin_ids, target_epoch=record["target_epoch"],
                supplemental_epoch_evaluator=evaluator,
            )

        outcome = execute_prepared_trial_v2(
            prepared_record=record, output_dir=output_dir, expected_screening_population=400,
            execute_prepared_run_fn=execute, executor_mode=mode,
        )
        if outcome["valid"]:
            payload = build_v2_objective_publication_payload(outcome["provenance"])
            run.log({V2_METRIC_NAME: payload[V2_METRIC_NAME]})
            run.summary.update(payload)
            return 0
        return 1
    finally:
        run.finish()


def main_from_manifest(path: str | Path) -> int:
    from src.baseline.sweep_v1_runtime_contract import run_full_runtime_contract
    from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import load_v2_wandb_bridge_manifest

    manifest = load_v2_wandb_bridge_manifest(path)
    repo_root = _validate_manifest_repository_root(manifest)
    run_full_runtime_contract(
        repo_root=repo_root, expected_commit=manifest["expected_commit"],
        expected_runtime_python=manifest["expected_runtime_python"],
    )
    return _execute(manifest)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("one manifest path required")
    try:
        raise SystemExit(main_from_manifest(sys.argv[1]))
    except SweepV2BridgeRefusal as exc:
        raise SystemExit(f"REFUSING: {exc}") from exc
