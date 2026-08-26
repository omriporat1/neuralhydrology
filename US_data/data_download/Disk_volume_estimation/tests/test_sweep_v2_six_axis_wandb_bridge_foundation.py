"""Local, fake-only contract tests for the v2 bridge foundation."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest
from src.baseline.sweep_v2_six_axis_campaign import *
from src.baseline.sweep_v2_six_axis_config import V2_METRIC_NAME, build_production_sweep_config_v2
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import *

ROOT=Path(__file__).parents[1]
BRIDGE=ROOT/'scripts/run_sweep_v2_six_axis_wandb_bridge.py'

def _fields(**changes):
    value=dict(manifest_label='fixture',created_at_utc='2026-08-26T00:00:00Z',mode='rehearsal',expected_commit='a'*40,repository_root=str(ROOT),expected_runtime_python='/canonical/python',wandb_project='fixture',wandb_sweep_id='disposable-v2',output_root=str(ROOT/'tmp/out'),package_root=str(ROOT/'tmp/pkg'),screening_basin_ids_path=str(ROOT/'tmp/screening.txt'),screening_basin_ids_sha256='b'*64,fixed_support_contract_path=str(ROOT/'tmp/support.json'),fixed_support_contract_version=OBJECTIVE_ID_V2,fixed_support_contract_sha256='c'*64,baseline_policy_path=str(ROOT/'config/stage1_scientific_baseline_v001.yaml'),policy_overlay_path=str(ROOT/'config/stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml'),base_pilot_policy_path=str(ROOT/'config/stage1_lead06_pilot_v001.yaml'),proposal_order=900001,execution_generation=900001,stop_before_training=True,max_agents=1,campaign_id=CAMPAIGN_ID_V2,domain_version=DOMAIN_VERSION_V2,canonicalization_version=CONFIGURATION_CANONICALIZATION_VERSION_V2,objective_id=OBJECTIVE_ID_V2)
    value.update(changes); return value

def _bridge_module():
    spec=importlib.util.spec_from_file_location('v2_bridge_test',BRIDGE); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def test_manifest_checksum_unknown_secret_modes_and_single_agent(tmp_path):
    data=build_v2_wandb_bridge_manifest(**_fields()); assert data['manifest_sha256']==compute_manifest_checksum(data)
    path=tmp_path/'m.json'; write_v2_wandb_bridge_manifest(path,**_fields()); assert load_v2_wandb_bridge_manifest(path)==data
    with pytest.raises(SweepV2BridgeManifestError): write_v2_wandb_bridge_manifest(path,**_fields())
    for changed in ({'unexpected':1},{'api_key':'secret'},{'max_agents':2},{'mode':'production'},
                    {'wandb_sweep_id':FORBIDDEN_V1_SWEEP_ID},{'screening_basin_ids_sha256':'not-a-sha'}):
        with pytest.raises(SweepV2BridgeManifestError): build_v2_wandb_bridge_manifest(**_fields(**changed))

def test_controller_shape_normalizes_before_identity_or_intake(tmp_path):
    bridge=_bridge_module()
    class Run: id='fake'; config={'learning_rate':3e-4,'hidden_size':128,'embedding_dropout':.1,'output_dropout':.2,'batch_size':256,'seq_length':72.0}
    axes=bridge._controller_axes(Run(),tmp_path); assert axes['seq_length']==72
    assert canonical_hyperparameters_v2(axes)['seq_length']==72
    Run.config['extra']=1
    with pytest.raises(ValueError,match='exact six-axis'): bridge._controller_axes(Run(),tmp_path)

@pytest.mark.parametrize('bad',[72.5,47,float('nan'),float('inf'),'72',True])
def test_controller_rejects_bad_seq_length(tmp_path,bad):
    bridge=_bridge_module()
    class Run: id='fake'; config={'learning_rate':3e-4,'hidden_size':128,'embedding_dropout':.1,'output_dropout':.2,'batch_size':256,'seq_length':bad}
    with pytest.raises(Exception): bridge._controller_axes(Run(),tmp_path)

def test_sweep_config_and_launchers_have_exact_v2_metric_and_one_agent():
    cfg=build_production_sweep_config_v2(program='bridge.py')
    assert cfg['metric']['name']==V2_METRIC_NAME and cfg['parameters']['seq_length']=={'distribution':'q_uniform','min':48,'max':120,'q':12}
    assert 'flashnh/best_score' not in str(cfg)
    scripts = [
        ROOT/'scripts/run_sweep_v2_six_axis_wandb_bridge_rehearsal_moriah.sbatch',
        ROOT/'scripts/run_sweep_v2_six_axis_wandb_agent_moriah.sbatch',
    ]
    for script in scripts:
        text = script.read_text()
        code = '\n'.join(line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#'))
        assert code.count('wandb agent') == 1
        assert 'wandb agent --count 1 "${WANDB_SWEEP_ID}"' in code
        assert 'FORBIDDEN_PRODUCTION_SWEEP_ID="4x3btz2s"' in code
        assert 'if [ "${WANDB_SWEEP_ID}" = "${FORBIDDEN_PRODUCTION_SWEEP_ID}" ]' in code
        assert 'export WANDB_PROJECT WANDB_ENTITY' in code
        assert 'export PATH="$(dirname "${CANONICAL_PYTHON}"):${PATH}"' in code
    rehearsal = scripts[0].read_text()
    production = scripts[1].read_text()
    assert '#SBATCH --partition=glacier' in rehearsal and '--gres=' not in rehearsal
    assert '#SBATCH --partition=catfish' in production and '#SBATCH --gres=gpu:l4:1' in production
