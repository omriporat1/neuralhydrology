"""Local, fake-only contract tests for the v2 bridge foundation."""
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
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
    raw, axes=bridge._controller_axes(Run(),tmp_path)
    assert raw['seq_length'] == 72.0
    assert axes['seq_length']==72
    assert raw is not axes
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
        assert 'date -u' in code and 'hostname' in code and 'pwd' in code
        assert 'EXPECTED_COMMIT:' in code and 'V2_BRIDGE_MANIFEST:' in code
    rehearsal = scripts[0].read_text()
    production = scripts[1].read_text()
    assert '#SBATCH --partition=glacier' in rehearsal and '--gres=' not in rehearsal
    assert '#SBATCH --partition=catfish' in production and '#SBATCH --gres=gpu:l4:1' in production


class _FakeRun:
    def __init__(self, axes, *, sweep_id='disposable-v2', metric=V2_METRIC_NAME):
        self.id = 'fake-run'
        self.sweep_id = sweep_id
        self.entity, self.project = 'entity', 'project'
        self.config = dict(axes)
        self.summary, self.logged, self.finished = {}, [], 0
        self._metric = metric
    def log(self, payload): self.logged.append(payload)
    def finish(self): self.finished += 1


def _fake_wandb(run):
    return SimpleNamespace(
        init=lambda: run,
        Api=lambda: SimpleNamespace(sweep=lambda _: SimpleNamespace(config={
            'method': 'bayes', 'metric': {'name': run._metric, 'goal': 'maximize'},
        })),
    )


def _bridge_axes(seq=72.0):
    return {'learning_rate':3e-4, 'hidden_size':128, 'embedding_dropout':.1,
            'output_dropout':.2, 'batch_size':256, 'seq_length':seq}


def _wire_fake_bridge(monkeypatch, bridge, tmp_path, *, terminal=None):
    events = []
    contract = {'checksum_sha256': 'c'*64, 'contract_id': OBJECTIVE_ID_V2, 'basin_ids': list(map(str, range(400)))}
    monkeypatch.setattr(bridge, 'load_fixed_support_contract', lambda _: contract)
    monkeypatch.setattr(bridge, '_validate_screening_binding', lambda _: None)
    def intake(**kwargs):
        events.append(('intake', kwargs))
        output_dir = Path(kwargs['output_root']) / 'trial'
        output_dir.mkdir(parents=True, exist_ok=True)
        return {'trial_id': 'trial', 'provenance_stage': 'proposal_intake', 'hyperparameters': kwargs['axes']}
    monkeypatch.setattr(bridge, 'write_proposal_intake_provenance_v2', intake)
    monkeypatch.setattr(bridge, 'enrich_layer_b_provenance', lambda **kwargs: events.append(('enrich', kwargs)))
    monkeypatch.setattr(bridge, 'prepare_bayesian_proposal_v2', lambda **_: SimpleNamespace(evidence={}))
    record = {'target_epoch': 12}
    monkeypatch.setattr(bridge, 'write_prepared_proposal_v2', lambda *_a, **_k: record)
    monkeypatch.setattr(bridge, 'select_executor_mode_v2', lambda _: 'monolithic')
    if terminal is not None:
        context = SimpleNamespace(execution_policy='policy', config_dir=tmp_path, experiment_name='x', package_root=tmp_path,
                                  target_variable='qobs_mm_per_h_lead06', lead_hours=6, screening_basin_ids=['a']*400)
        evaluator = object()
        monkeypatch.setattr(bridge, 'build_execution_context_v2', lambda **_: context)
        monkeypatch.setattr(bridge, 'build_v2_epoch_evaluator', lambda **_: evaluator)
        def monolithic(**kwargs):
            events.append(('monolithic', kwargs))
            return object()
        monkeypatch.setattr(bridge.pilot_orchestration, 'execute_prepared_pilot_run_monolithic', monolithic)
        def execute_trial(**kwargs):
            assert kwargs['executor_mode'] == 'monolithic'
            kwargs['execute_prepared_run_fn']()
            return terminal
        monkeypatch.setattr(bridge, 'execute_prepared_trial_v2', execute_trial)
    return contract, events


def test_rehearsal_main_from_manifest_persists_raw_and_canonical_before_stop(tmp_path, monkeypatch):
    bridge = _bridge_module()
    output = tmp_path/'out'
    manifest_path = tmp_path/'manifest.json'
    write_v2_wandb_bridge_manifest(manifest_path, **_fields(output_root=str(output), repository_root=str(ROOT)))
    run = _FakeRun(_bridge_axes(72.0))
    monkeypatch.setitem(sys.modules, 'wandb', _fake_wandb(run))
    _, events = _wire_fake_bridge(monkeypatch, bridge, tmp_path)
    monkeypatch.setattr('src.baseline.sweep_v1_runtime_contract.run_full_runtime_contract', lambda **_: None)
    assert bridge.main_from_manifest(manifest_path) == 0
    intake = next(value for kind, value in events if kind == 'intake')
    enrichment = next(value for kind, value in events if kind == 'enrich')
    assert intake['axes']['seq_length'] == 72
    assert enrichment['fields']['raw_controller_axes']['seq_length'] == 72.0
    assert enrichment['fields']['normalized_controller_axes']['seq_length'] == 72
    assert run.logged == [] and run.finished == 1
    assert run.summary['flashnh/rehearsal_stopped_before_training'] is True


def test_production_valid_publishes_only_fixed_support_metric_once(tmp_path, monkeypatch):
    bridge = _bridge_module(); run = _FakeRun(_bridge_axes())
    monkeypatch.setitem(sys.modules, 'wandb', _fake_wandb(run))
    terminal = {'valid': True, 'provenance': {'execution_status':'VALID', 'objective_eligible':True,
                'objective_score':0.42, 'best_epoch':3, 'fixed_support_epoch_trajectory': {3: 0.42},
                'fixed_support_metric_name': V2_METRIC_NAME, 'trial_id': 'trial'}}
    _, events = _wire_fake_bridge(monkeypatch, bridge, tmp_path, terminal=terminal)
    manifest = build_v2_wandb_bridge_manifest(**_fields(mode='production', stop_before_training=False,
        output_root=str(tmp_path/'out'), repository_root=str(ROOT)))
    assert bridge._execute(manifest) == 0
    assert run.logged == [{V2_METRIC_NAME: 0.42}]
    assert 'flashnh/best_score' not in str(run.logged)
    assert next(value for kind, value in events if kind == 'monolithic')['supplemental_epoch_evaluator'] is not None
    assert run.finished == 1


def test_metric_mismatch_refuses_before_intake_and_finishes(tmp_path, monkeypatch):
    bridge = _bridge_module(); run = _FakeRun(_bridge_axes(), metric='flashnh/best_score')
    monkeypatch.setitem(sys.modules, 'wandb', _fake_wandb(run))
    _wire_fake_bridge(monkeypatch, bridge, tmp_path)
    monkeypatch.setattr(bridge, 'write_proposal_intake_provenance_v2', lambda **_: pytest.fail('intake must not run'))
    manifest = build_v2_wandb_bridge_manifest(**_fields(output_root=str(tmp_path/'out'), repository_root=str(ROOT)))
    with pytest.raises(bridge.SweepV2BridgeRefusal, match='metric contract'):
        bridge._execute(manifest)
    assert run.logged == [] and run.finished == 1
    assert (tmp_path/'out'/f'bootstrap_assignment_rejected__wandb_run_{run.id}'/'execution_provenance.json').is_file()


def test_support_mismatch_refuses_before_wandb_init(tmp_path, monkeypatch):
    bridge = _bridge_module(); called = []
    monkeypatch.setattr(bridge, 'load_fixed_support_contract', lambda _: {'checksum_sha256':'x'*64, 'contract_id':OBJECTIVE_ID_V2, 'basin_ids':list(map(str, range(400)))})
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=lambda: called.append(True)))
    manifest = build_v2_wandb_bridge_manifest(**_fields(output_root=str(tmp_path/'out'), repository_root=str(ROOT)))
    with pytest.raises(bridge.SweepV2BridgeRefusal, match='fixed-support'):
        bridge._execute(manifest)
    assert called == []


def test_sweep_mismatch_and_invalid_terminal_publish_nothing(tmp_path, monkeypatch):
    bridge = _bridge_module(); run = _FakeRun(_bridge_axes(), sweep_id='wrong-sweep')
    monkeypatch.setitem(sys.modules, 'wandb', _fake_wandb(run))
    _wire_fake_bridge(monkeypatch, bridge, tmp_path)
    monkeypatch.setattr(bridge, 'write_proposal_intake_provenance_v2', lambda **_: pytest.fail('intake must not run'))
    manifest = build_v2_wandb_bridge_manifest(**_fields(output_root=str(tmp_path/'out'), repository_root=str(ROOT)))
    with pytest.raises(bridge.SweepV2BridgeRefusal, match='unexpected sweep'):
        bridge._execute(manifest)
    assert run.logged == [] and run.finished == 1

    run = _FakeRun(_bridge_axes())
    monkeypatch.setitem(sys.modules, 'wandb', _fake_wandb(run))
    terminal = {'valid': False, 'provenance': {'execution_status': 'INVALID', 'objective_eligible': False}}
    _wire_fake_bridge(monkeypatch, bridge, tmp_path, terminal=terminal)
    manifest = build_v2_wandb_bridge_manifest(**_fields(mode='production', stop_before_training=False,
        output_root=str(tmp_path/'out2'), repository_root=str(ROOT)))
    assert bridge._execute(manifest) == 1
    assert run.logged == [] and run.finished == 1


def test_screening_and_repository_binding_refuse_contradictions(tmp_path, monkeypatch):
    bridge = _bridge_module()
    manifest = build_v2_wandb_bridge_manifest(**_fields(repository_root=str(ROOT), output_root=str(tmp_path/'out')))
    changed = dict(manifest, repository_root=str(tmp_path/'elsewhere'))
    with pytest.raises(bridge.SweepV2BridgeRefusal, match='repository_root'):
        bridge._validate_manifest_repository_root(changed)
    changed = dict(manifest, screening_basin_ids_sha256='a'*64)
    with pytest.raises(bridge.SweepV2BridgeRefusal, match='screening checksum'):
        bridge._validate_screening_binding(changed)
