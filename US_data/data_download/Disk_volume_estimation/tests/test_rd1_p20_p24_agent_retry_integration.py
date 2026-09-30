"""Integration tests for the actual shell-wrapper -> Python-driver retry
path: ``scripts/rd1_p20_p24_job.py run-agent-with-retry`` invoked as a real
subprocess, with a stub bash "agent launcher" and a stub ``wandb`` Python
module -- never touches Moriah, a real W&B account, or Slurm.

Coordination pattern: each test writes its own tiny stub bash launcher
script (with its per-attempt outcome sequence and a counter file path baked
in literally, matching how the real launcher hardcodes its own paths) and
its own stub ``wandb.py`` module (importable via ``PYTHONPATH``, reading a
per-test JSON state file named by ``RD1_TEST_WANDB_STATE_PATH`` for its
sweep-runs-audit call sequence). Both live entirely under ``tmp_path``.

Scenarios (matching the four properties Phase A.1 requires this repair to
prove about the one live retry path):
  1. No retry when the before/after W&B audit shows a new run appeared,
     even though the failed attempt's own output matches the exact
     socket-failure marker.
  2. Exactly one retry for the exact socket-failure marker with a
     confirmed-empty before/after audit diff, then success and a resolved
     run id.
  3. No retry when the read-only audit itself is unavailable (every call
     raises): the retry decision must fail closed rather than assume "no
     new run". This test incurs the real ~6s exponential backoff delay in
     ``fetch_with_backoff`` (2s + 4s, two sleeps before it gives up) since
     that function is exercised unmodified, not mocked.
  4. No second retry once the attempt budget is exhausted, even when the
     second failure also matches the marker with an empty diff.

Regression coverage (review fix, 2026-09-27): a real production P20 attempt
failed with ``Invalid path: 'wta85z3b' (missing project)`` because the
audit's ``wandb.Api().sweep(...)`` call was passing the bare sweep id
instead of the ``entity/project/sweep_id`` path the real W&B API requires.
The stub ``_Api.sweep`` below now records every path it is called with so
these additional scenarios can assert on it directly:
  5. The audit always calls ``sweep()`` with the full canonical
     ``entity/project/sweep_id`` path built from the launch manifest, never
     the bare sweep id.
  6. A manifest with no usable ``wandb_project``/``wandb_entity`` fails
     closed before the agent launcher (and W&B) are ever contacted.
  7. A manifest whose own ``wandb_sweep_id`` disagrees with the CLI's
     ``--wandb-sweep-id`` fails closed before the agent launcher (and W&B)
     are ever contacted.

Regression coverage (dual-provenance diagnostic fix, 2026-09-30): a real
diagnostic attempt (Moriah job 46259195) failed because the narrow
attempt-env allowlist built by ``_cmd_run_agent_with_retry`` never included
``REPO_WORKDIR``, so the agent-launcher subprocess always fell back to
``run_sweep_v2_six_axis_wandb_agent_moriah.sbatch``'s own hardcoded
production default instead of the caller's intended value -- invisible in
production (where they already coincide) but fatal whenever they must
differ (e.g. a payload checkout used for provenance isolation), since the
bridge script then refuses on a ``repository_root`` mismatch.
  8. The exact ``--repo-workdir`` value passed on the CLI reaches the agent
     launcher subprocess's environment as ``REPO_WORKDIR``, verified by a
     stub launcher that echoes it back rather than by inspecting production
     refusal behavior.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.baseline.sweep_v2_six_axis_campaign import (
    CAMPAIGN_ID_V2,
    CONFIGURATION_CANONICALIZATION_VERSION_V2,
    DOMAIN_VERSION_V2,
    OBJECTIVE_ID_V2,
    configuration_id_v2,
    proposal_id_v2,
    trial_id_v2,
)
from src.baseline.sweep_v2_six_axis_wandb_bridge_manifest import write_v2_wandb_bridge_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_PY = REPO_ROOT / "scripts" / "rd1_p20_p24_job.py"

sys.path.insert(0, str(REPO_ROOT))

from scripts.rd1_p20_p24_job import V2_METRIC_NAME  # noqa: E402

_STUB_WANDB_MODULE = '''
import json
import os

_STATE_PATH = os.environ["RD1_TEST_WANDB_STATE_PATH"]


def _load():
    with open(_STATE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save(state):
    with open(_STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh)


class _Run:
    def __init__(self, run_id):
        self.id = run_id
        self.created_at = run_id


class _Sweep:
    def __init__(self, run_ids):
        self.runs = [_Run(r) for r in run_ids]


class _TerminalRun:
    def __init__(self, state, summary):
        self.state = state
        self.summary = summary


class _Api:
    def sweep(self, sweep_path):
        state = _load()
        state.setdefault("sweep_paths_seen", []).append(sweep_path)
        idx = state["call_index"]
        snapshots = state["snapshots"]
        entry = snapshots[min(idx, len(snapshots) - 1)]
        state["call_index"] = idx + 1
        _save(state)
        if entry == "RAISE":
            raise RuntimeError("stub: W&B sweep audit unreachable")
        return _Sweep(entry)

    def run(self, run_path):
        state = _load()
        state.setdefault("run_paths_seen", []).append(run_path)
        _save(state)
        lookup = state.get("run_lookup")
        if not lookup:
            raise RuntimeError(f"stub: no run_lookup configured for {run_path!r}")
        return _TerminalRun(lookup["state"], lookup["summary"])


def Api():
    return _Api()
'''

_STUB_LAUNCHER_TEMPLATE = """#!/bin/bash
COUNTER_FILE="{counter_file}"
N=$(cat "${{COUNTER_FILE}}" 2>/dev/null || echo 0)
N=$((N + 1))
echo "$N" > "${{COUNTER_FILE}}"
case "$N" in
{cases}
esac
"""


def _write_stub_wandb(tmp_path: Path) -> Path:
    site_dir = tmp_path / "stub_site"
    site_dir.mkdir()
    (site_dir / "wandb.py").write_text(_STUB_WANDB_MODULE, encoding="utf-8")
    return site_dir


def _write_wandb_state(tmp_path: Path, snapshots: list, *, run_lookup: "dict | None" = None) -> Path:
    state_path = tmp_path / "wandb_state.json"
    state_path.write_text(
        json.dumps({"call_index": 0, "snapshots": snapshots, "run_lookup": run_lookup}), encoding="utf-8"
    )
    return state_path


_PROVENANCE_HYPERPARAMETERS = {
    "learning_rate": 5e-4,
    "hidden_size": 128,
    "embedding_dropout": 0.1,
    "output_dropout": 0.1,
    "batch_size": 256,
    "seq_length": 72,
}
_PROVENANCE_SUPPORT_CONTRACT_SHA256 = "c" * 64


def _write_success_provenance(
    output_root: Path, *, order: int, sweep_id: str, run_id: str, objective_score: float = 0.5
) -> None:
    """A fully valid local ``execution_provenance.json`` record for a
    successful attempt, built from the real v2 identity-derivation
    primitives -- mirrors ``test_rd1_p20_p24_job.py``'s
    ``_build_provenance_record``/``_write_provenance_file`` -- so that
    ``resolve_run_id_from_local_provenance``'s full acceptance gate
    (identity re-derivation + the stubbed terminal W&B lookup below) has
    real local evidence to accept rather than a hand-typed fixture."""
    configuration_id = configuration_id_v2(
        _PROVENANCE_HYPERPARAMETERS,
        support_contract_version=OBJECTIVE_ID_V2,
        support_contract_sha256=_PROVENANCE_SUPPORT_CONTRACT_SHA256,
    )
    proposal_id = proposal_id_v2("bayesian", order)
    trial_id = trial_id_v2(configuration_id, proposal_id, execution_generation=1)
    record = {
        "hyperparameters": dict(_PROVENANCE_HYPERPARAMETERS),
        "search_arm": "bayesian",
        "proposal_order": order,
        "execution_generation": 1,
        "configuration_id": configuration_id,
        "proposal_id": proposal_id,
        "trial_id": trial_id,
        "campaign_id": CAMPAIGN_ID_V2,
        "domain_version": DOMAIN_VERSION_V2,
        "wandb_sweep_id": sweep_id,
        "wandb_run_id": run_id,
        "support_contract_version": OBJECTIVE_ID_V2,
        "support_contract_sha256": _PROVENANCE_SUPPORT_CONTRACT_SHA256,
        "execution_status": "VALID",
        "objective_eligible": True,
        "fixed_support_metric_name": V2_METRIC_NAME,
        "objective_score": objective_score,
    }
    # A short, fixed directory name -- not the real (long) trial_id -- keeps
    # this well under Windows' MAX_PATH once nested beneath pytest's own
    # deep tmp_path; find_local_execution_provenance(_by_order) matches on
    # the record's own JSON fields, never the enclosing directory name.
    trial_dir = output_root / "t"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "execution_provenance.json").write_text(json.dumps(record), encoding="utf-8")


def _write_stub_launcher(tmp_path: Path, counter_file: Path, outcomes: list[tuple[int, str]]) -> Path:
    cases = []
    for i, (rc, msg) in enumerate(outcomes, start=1):
        cases.append(f'  {i}) echo "{msg}" >&2; exit {rc} ;;')
    last_rc, last_msg = outcomes[-1]
    cases.append(f'  *) echo "{last_msg}" >&2; exit {last_rc} ;;')
    script = _STUB_LAUNCHER_TEMPLATE.format(counter_file=counter_file.as_posix(), cases="\n".join(cases))
    launcher_path = tmp_path / "stub_launcher.sh"
    launcher_path.write_text(script, encoding="utf-8")
    launcher_path.chmod(0o755)
    return launcher_path


_DEFAULT_MANIFEST_SWEEP_ID = "stub-sweep-id"
_DEFAULT_MANIFEST_PROJECT = "rd1-test-project"
_DEFAULT_MANIFEST_ENTITY = "rd1-test-entity"


def _write_manifest(
    path: Path,
    *,
    wandb_sweep_id: str = _DEFAULT_MANIFEST_SWEEP_ID,
    wandb_project: "str | None" = _DEFAULT_MANIFEST_PROJECT,
    wandb_entity: "str | None" = _DEFAULT_MANIFEST_ENTITY,
) -> None:
    """The one launch-identity manifest the CLI's audit calls must resolve
    entity/project from -- built with the same real, loader-validated
    manifest writer used everywhere else in this codebase (never a
    hand-rolled JSON fixture), so these tests exercise the exact schema the
    production driver actually reads. ``wandb_project``/``wandb_entity`` of
    ``None`` is how a test simulates a manifest that carries no usable
    identity for that field (empty string for project, since it is a
    required-but-unchecked-for-emptiness field; omitted entirely for
    entity, an optional field)."""
    fields = dict(
        manifest_label="rd1-p20p24-retry-integration-test",
        created_at_utc="2026-09-27T00:00:00Z",
        mode="rehearsal",
        expected_commit="a" * 40,
        repository_root=str(REPO_ROOT),
        expected_runtime_python="/canonical/python",
        wandb_project="" if wandb_project is None else wandb_project,
        wandb_sweep_id=wandb_sweep_id,
        output_root=str(path.parent / "out"),
        package_root=str(REPO_ROOT / "tmp/pkg"),
        screening_basin_ids_path=str(REPO_ROOT / "tmp/screening.txt"),
        screening_basin_ids_sha256="b" * 64,
        fixed_support_contract_path=str(REPO_ROOT / "tmp/support.json"),
        fixed_support_contract_version=OBJECTIVE_ID_V2,
        fixed_support_contract_sha256="c" * 64,
        baseline_policy_path=str(REPO_ROOT / "config/stage1_scientific_baseline_v001.yaml"),
        policy_overlay_path=str(REPO_ROOT / "config/stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml"),
        base_pilot_policy_path=str(REPO_ROOT / "config/stage1_lead06_pilot_v001.yaml"),
        proposal_order=1,
        execution_generation=1,
        stop_before_training=True,
        max_agents=1,
        campaign_id=CAMPAIGN_ID_V2,
        domain_version=DOMAIN_VERSION_V2,
        canonicalization_version=CONFIGURATION_CANONICALIZATION_VERSION_V2,
        objective_id=OBJECTIVE_ID_V2,
    )
    if wandb_entity is not None:
        fields["wandb_entity"] = wandb_entity
    write_v2_wandb_bridge_manifest(path, **fields)


def _run_cli(
    tmp_path: Path,
    site_dir: Path,
    state_path: Path,
    launcher_path: Path,
    extra_args: list[str],
    manifest_kwargs: "dict | None" = None,
    repo_workdir: "str | None" = None,
    order: int = 1,
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["RD1_TEST_WANDB_STATE_PATH"] = str(state_path)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, **(manifest_kwargs or {}))
    cmd = [
        sys.executable,
        str(JOB_PY),
        "run-agent-with-retry",
        "--agent-launcher-path",
        str(launcher_path),
        "--manifest-path",
        str(manifest_path),
        "--wandb-sweep-id",
        _DEFAULT_MANIFEST_SWEEP_ID,
        "--path",
        env.get("PATH", ""),
        "--home",
        str(tmp_path),
        "--repo-workdir",
        repo_workdir if repo_workdir is not None else str(tmp_path),
        "--order",
        str(order),
        *extra_args,
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60)


SOCKET_MARKER = "Failed to connect to service on socket"


def test_no_retry_when_new_run_id_appears_in_audit(tmp_path):
    site_dir = _write_stub_wandb(tmp_path)
    state_path = _write_wandb_state(tmp_path, snapshots=[["r0"], ["r0", "r1"]])
    counter_file = tmp_path / "attempt_counter.txt"
    launcher_path = _write_stub_launcher(tmp_path, counter_file, outcomes=[(1, f"wandb: ERROR {SOCKET_MARKER}")])

    proc = _run_cli(tmp_path, site_dir, state_path, launcher_path, extra_args=["--max-attempts", "2"])

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "AGENT_FAILED" in (proc.stdout + proc.stderr)
    assert counter_file.read_text().strip() == "1"


def test_one_retry_on_empty_diff_then_success(tmp_path):
    site_dir = _write_stub_wandb(tmp_path)
    _write_success_provenance(
        tmp_path / "out", order=1, sweep_id=_DEFAULT_MANIFEST_SWEEP_ID, run_id="r1"
    )
    state_path = _write_wandb_state(
        tmp_path,
        snapshots=[["r0"], ["r0"], ["r0", "r1"]],
        run_lookup={"state": "finished", "summary": {"flashnh/valid": True, V2_METRIC_NAME: 0.5}},
    )
    counter_file = tmp_path / "attempt_counter.txt"
    launcher_path = _write_stub_launcher(
        tmp_path,
        counter_file,
        outcomes=[(1, f"wandb: ERROR {SOCKET_MARKER}"), (0, "ok")],
    )

    proc = _run_cli(tmp_path, site_dir, state_path, launcher_path, extra_args=["--max-attempts", "2"])

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AGENT_OK run_id=r1" in proc.stdout
    assert counter_file.read_text().strip() == "2"


def test_no_retry_when_audit_fails(tmp_path):
    site_dir = _write_stub_wandb(tmp_path)
    state_path = _write_wandb_state(tmp_path, snapshots=[["r0"], "RAISE"])
    counter_file = tmp_path / "attempt_counter.txt"
    launcher_path = _write_stub_launcher(tmp_path, counter_file, outcomes=[(1, f"wandb: ERROR {SOCKET_MARKER}")])

    proc = _run_cli(tmp_path, site_dir, state_path, launcher_path, extra_args=["--max-attempts", "2"])

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "retry_audit_unavailable" in (proc.stdout + proc.stderr)
    assert counter_file.read_text().strip() == "1"


def test_no_second_retry_after_budget_exhausted(tmp_path):
    site_dir = _write_stub_wandb(tmp_path)
    state_path = _write_wandb_state(tmp_path, snapshots=[["r0"], ["r0"]])
    counter_file = tmp_path / "attempt_counter.txt"
    launcher_path = _write_stub_launcher(
        tmp_path,
        counter_file,
        outcomes=[(1, f"wandb: ERROR {SOCKET_MARKER}"), (1, f"wandb: ERROR {SOCKET_MARKER}")],
    )

    proc = _run_cli(tmp_path, site_dir, state_path, launcher_path, extra_args=["--max-attempts", "2"])

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "AGENT_FAILED" in (proc.stdout + proc.stderr)
    assert counter_file.read_text().strip() == "2"


def test_audit_uses_full_canonical_sweep_path_never_bare_id(tmp_path):
    """Regression test for the real production failure: the stub records
    every path it was called with, and every one of them must be the full
    ``entity/project/sweep_id`` path -- never the bare sweep id alone."""
    site_dir = _write_stub_wandb(tmp_path)
    _write_success_provenance(
        tmp_path / "out", order=1, sweep_id=_DEFAULT_MANIFEST_SWEEP_ID, run_id="r1"
    )
    state_path = _write_wandb_state(
        tmp_path,
        snapshots=[["r0"], ["r0", "r1"]],
        run_lookup={"state": "finished", "summary": {"flashnh/valid": True, V2_METRIC_NAME: 0.5}},
    )
    counter_file = tmp_path / "attempt_counter.txt"
    launcher_path = _write_stub_launcher(tmp_path, counter_file, outcomes=[(0, "ok")])

    proc = _run_cli(tmp_path, site_dir, state_path, launcher_path, extra_args=["--max-attempts", "2"])

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AGENT_OK run_id=r1" in proc.stdout
    final_state = json.loads(state_path.read_text())
    expected_path = f"{_DEFAULT_MANIFEST_ENTITY}/{_DEFAULT_MANIFEST_PROJECT}/{_DEFAULT_MANIFEST_SWEEP_ID}"
    assert final_state["sweep_paths_seen"], "the audit must call sweep() at least once"
    for seen in final_state["sweep_paths_seen"]:
        assert seen == expected_path, f"expected full canonical path, got bare/partial path {seen!r}"
        assert seen != _DEFAULT_MANIFEST_SWEEP_ID


def test_missing_manifest_project_or_entity_fails_closed_before_agent_launch(tmp_path):
    site_dir = _write_stub_wandb(tmp_path)
    state_path = _write_wandb_state(tmp_path, snapshots=[["r0"]])
    counter_file = tmp_path / "attempt_counter.txt"
    launcher_path = _write_stub_launcher(tmp_path, counter_file, outcomes=[(0, "ok")])

    proc = _run_cli(
        tmp_path,
        site_dir,
        state_path,
        launcher_path,
        extra_args=["--max-attempts", "2"],
        manifest_kwargs={"wandb_project": None, "wandb_entity": None},
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "retry_audit_unavailable" in (proc.stdout + proc.stderr)
    assert not counter_file.exists(), "the agent launcher must never run when identity resolution fails closed"
    final_state = json.loads(state_path.read_text())
    assert final_state.get("sweep_paths_seen", []) == [], "wandb must never be contacted when identity is unresolved"


def test_repo_workdir_env_var_forwarded_to_launcher(tmp_path):
    """Regression test for the real diagnostic failure (Moriah job
    46259195): the launcher must observe REPO_WORKDIR exactly as passed on
    the CLI, not a value inherited from this test process's own
    environment (the subprocess env is a from-scratch allowlist, never
    inherited -- see ``_subprocess_attempt_fn``)."""
    site_dir = _write_stub_wandb(tmp_path)
    _write_success_provenance(
        tmp_path / "out", order=1, sweep_id=_DEFAULT_MANIFEST_SWEEP_ID, run_id="r1"
    )
    state_path = _write_wandb_state(
        tmp_path,
        snapshots=[["r0"]],
        run_lookup={"state": "finished", "summary": {"flashnh/valid": True, V2_METRIC_NAME: 0.5}},
    )
    probe_file = tmp_path / "repo_workdir_seen.txt"
    launcher_path = tmp_path / "env_probe_launcher.sh"
    launcher_path.write_text(
        '#!/bin/bash\necho -n "${REPO_WORKDIR:-UNSET}" > "' + probe_file.as_posix() + '"\nexit 0\n',
        encoding="utf-8",
    )
    launcher_path.chmod(0o755)
    expected_repo_workdir = str(tmp_path / "distinct_payload_checkout")

    proc = _run_cli(
        tmp_path,
        site_dir,
        state_path,
        launcher_path,
        extra_args=["--max-attempts", "2"],
        repo_workdir=expected_repo_workdir,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert probe_file.read_text() == expected_repo_workdir


def test_manifest_sweep_id_mismatch_fails_closed_before_agent_launch(tmp_path):
    site_dir = _write_stub_wandb(tmp_path)
    state_path = _write_wandb_state(tmp_path, snapshots=[["r0"]])
    counter_file = tmp_path / "attempt_counter.txt"
    launcher_path = _write_stub_launcher(tmp_path, counter_file, outcomes=[(0, "ok")])

    proc = _run_cli(
        tmp_path,
        site_dir,
        state_path,
        launcher_path,
        extra_args=["--max-attempts", "2"],
        manifest_kwargs={"wandb_sweep_id": "a-different-sweep-id"},
    )

    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "retry_audit_unavailable" in (proc.stdout + proc.stderr)
    assert not counter_file.exists(), "the agent launcher must never run when identity resolution fails closed"
    final_state = json.loads(state_path.read_text())
    assert final_state.get("sweep_paths_seen", []) == [], "wandb must never be contacted when identity is unresolved"
