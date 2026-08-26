"""One-agent v2 fresh-proposal bridge. Never creates controller proposals."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.baseline.fixed_support_contract_v2 import load_fixed_support_contract
from src.baseline.sweep_v1_execution import enrich_layer_b_provenance
from src.baseline import pilot_orchestration
from src.baseline.sweep_v2_six_axis_campaign import CAMPAIGN_ID_V2, FORBIDDEN_V1_SWEEP_ID, canonical_hyperparameters_v2, normalize_seq_length_axis
from src.baseline.sweep_v2_six_axis_config import V2_METRIC_NAME, build_production_sweep_config_v2
from src.baseline.sweep_v2_six_axis_execution import build_execution_context_v2, build_v2_epoch_evaluator, build_v2_objective_publication_payload, execute_prepared_trial_v2, select_executor_mode_v2, write_proposal_intake_provenance_v2
from src.baseline.sweep_v2_six_axis_production_adapter import PreparationPathsV2, prepare_bayesian_proposal_v2, write_prepared_proposal_v2
_AXES=("learning_rate","hidden_size","embedding_dropout","output_dropout","batch_size","seq_length")

def _incident(root: Path, run_id: str | None, stage: str, reason: str) -> None:
    p=root/f"bootstrap_assignment_rejected__wandb_run_{run_id or 'unknown'}"; p.mkdir(parents=True,exist_ok=True)
    (p/"execution_provenance.json").write_text(json.dumps({"provenance_stage":stage,"rejection_reason":reason,"wandb_run_id":run_id},sort_keys=True)+"\n")

def _controller_axes(run: Any, output_root: Path) -> dict[str, Any]:
    keys=set(dict(run.config))-{"_wandb"}
    if keys != set(_AXES):
        reason=f"exact six-axis run.config required; missing={sorted(set(_AXES)-keys)} unexpected={sorted(keys-set(_AXES))}"; _incident(output_root,getattr(run,'id',None),'controller_config_shape_rejected',reason); raise ValueError(reason)
    raw={key:run.config[key] for key in _AXES}; raw["seq_length"]=normalize_seq_length_axis(raw["seq_length"])
    return raw

def _execute(manifest: dict[str,Any]) -> int:
    if manifest["wandb_sweep_id"] == FORBIDDEN_V1_SWEEP_ID: raise ValueError("v1 sweep forbidden")
    contract=load_fixed_support_contract(manifest["fixed_support_contract_path"])
    if contract["checksum_sha256"] != manifest["fixed_support_contract_sha256"] or contract["contract_id"] != manifest["fixed_support_contract_version"] or len(contract["basin_ids"]) != 400: raise ValueError("fixed-support manifest binding mismatch or non-production support")
    import wandb
    run=wandb.init(); valid=False
    try:
        if run.sweep_id != manifest["wandb_sweep_id"]: _incident(Path(manifest["output_root"]),run.id,'sweep_identity_rejected','unexpected sweep'); raise ValueError('unexpected sweep')
        axes=_controller_axes(run,Path(manifest["output_root"])); canonical_hyperparameters_v2(axes)
        intake=write_proposal_intake_provenance_v2(output_root=manifest["output_root"],axes=axes,search_arm="bayesian",proposal_order=manifest["proposal_order"],wandb_sweep_id=run.sweep_id,wandb_run_id=run.id,support_contract_version=contract["contract_id"],support_contract_sha256=contract["checksum_sha256"],execution_generation=manifest["execution_generation"])
        out=Path(manifest["output_root"])/intake["trial_id"]
        enrich_layer_b_provenance(output_dir=out,stage='proposal_intake',fields={"launch_manifest_sha256":manifest["manifest_sha256"],"raw_controller_axes":dict(run.config),"normalized_controller_axes":axes})
        paths=PreparationPathsV2(Path(manifest["baseline_policy_path"]),Path(manifest["policy_overlay_path"]),Path(manifest["package_root"]),ROOT/'config/stage1_baseline_splits_v001',Path(manifest["screening_basin_ids_path"]),Path(manifest["fixed_support_contract_path"]))
        proposal={**axes,"proposal_order":manifest["proposal_order"],"execution_generation":manifest["execution_generation"],"wandb_sweep_id":run.sweep_id,"wandb_run_id":run.id}
        prepared=prepare_bayesian_proposal_v2(proposal=proposal,paths=paths); enrich_layer_b_provenance(output_dir=out,stage='prepared',fields=prepared.evidence)
        record=write_prepared_proposal_v2(prepared,out,allow_layer_b_provenance=True); mode=select_executor_mode_v2(record); enrich_layer_b_provenance(output_dir=out,stage='executor_mode_selected',fields={"executor_mode":mode})
        if manifest["stop_before_training"]:
            run.summary.update({"flashnh/rehearsal_stopped_before_training":True,"flashnh/trial_id":intake["trial_id"]}); return 0
        context=build_execution_context_v2(prepared_record=record,paths=paths,base_pilot_policy_path=manifest["base_pilot_policy_path"])
        evaluator=build_v2_epoch_evaluator(support_contract=contract,package_root=paths.package_root,screening_basin_ids=context.screening_basin_ids)
        def execute(): return pilot_orchestration.execute_prepared_pilot_run_monolithic(execution_policy=context.execution_policy,config_dir=context.config_dir,experiment_name=context.experiment_name,package_root=context.package_root,target_variable=context.target_variable,lead_hours=context.lead_hours,screening_basin_ids=context.screening_basin_ids,target_epoch=record['target_epoch'],supplemental_epoch_evaluator=evaluator)
        outcome=execute_prepared_trial_v2(prepared_record=record,output_dir=out,expected_screening_population=400,execute_prepared_run_fn=execute,executor_mode=mode); valid=outcome['valid']
        if valid:
            payload=build_v2_objective_publication_payload(outcome['provenance']); run.log({V2_METRIC_NAME:payload[V2_METRIC_NAME]}); run.summary.update(payload)
        return 0 if valid else 1
    finally: run.finish()

def main_from_manifest(path: str|Path)->int:
    from src.baseline.sweep_v1_runtime_contract import run_full_runtime_contract
    from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import load_v2_wandb_bridge_manifest
    m=load_v2_wandb_bridge_manifest(path); run_full_runtime_contract(repo_root=ROOT,expected_commit=m['expected_commit'],expected_runtime_python=m['expected_runtime_python']); return _execute(m)
if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('one manifest path required')
    raise SystemExit(main_from_manifest(sys.argv[1]))
