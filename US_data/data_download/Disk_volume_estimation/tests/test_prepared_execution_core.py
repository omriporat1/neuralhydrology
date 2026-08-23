from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from src.baseline import pilot_orchestration as orchestration


def test_prepared_core_never_prepares_or_rewrites_source_config_and_supports_every_epoch(tmp_path, monkeypatch):
    """The extracted core is scheduling/composition only: it consumes a written config."""
    config_dir = tmp_path / "prepared"; config_dir.mkdir()
    config = config_dir / "config.yaml"; config.write_bytes(b"frozen: true\n")
    before = config.read_bytes(); before_sha = hashlib.sha256(before).hexdigest()
    calls = []
    policy = SimpleNamespace(lead_hours=6)
    monkeypatch.setattr(orchestration, "build_effective_policy", lambda _: {"max_epoch_budget": 12})
    monkeypatch.setattr(orchestration, "chunk_epoch_targets", lambda *_: list(range(1, 13)))
    monkeypatch.setattr(orchestration, "_try_discover_nh_run_dir", lambda *_: None)
    monkeypatch.setattr(orchestration, "prepare_pilot_run", lambda *_, **__: (_ for _ in ()).throw(AssertionError("must not prepare")))
    def fake_chunk(**kwargs):
        epoch = kwargs["chunk_target_epoch"]; calls.append(epoch)
        return {"blocked": False, "stopped": False, "stop_reason": None,
                "screening_results": [{"epoch": epoch}], "checkpoint_dir_for_target": tmp_path / "nh",
                "nh_run_dir": tmp_path / "nh", "state": {"stopped": False}}
    monkeypatch.setattr(orchestration, "run_pilot_chunk", fake_chunk)
    result = orchestration.execute_prepared_pilot_run(
        execution_policy=policy, config_dir=config_dir, experiment_name="already_written",
        package_root=tmp_path, target_variable="qobs", lead_hours=6, screening_basin_ids=["x"], run_id="prepared",
    )
    assert calls == list(range(1, 13))
    assert [row["epoch"] for row in result["screening_events"]] == list(range(1, 13))
    assert config.read_bytes() == before
    assert hashlib.sha256(config.read_bytes()).hexdigest() == before_sha
