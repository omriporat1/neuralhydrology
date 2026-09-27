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
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_PY = REPO_ROOT / "scripts" / "rd1_p20_p24_job.py"

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


class _Api:
    def sweep(self, sweep_id):
        state = _load()
        idx = state["call_index"]
        snapshots = state["snapshots"]
        entry = snapshots[min(idx, len(snapshots) - 1)]
        state["call_index"] = idx + 1
        _save(state)
        if entry == "RAISE":
            raise RuntimeError("stub: W&B sweep audit unreachable")
        return _Sweep(entry)


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


def _write_wandb_state(tmp_path: Path, snapshots: list) -> Path:
    state_path = tmp_path / "wandb_state.json"
    state_path.write_text(json.dumps({"call_index": 0, "snapshots": snapshots}), encoding="utf-8")
    return state_path


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


def _run_cli(tmp_path: Path, site_dir: Path, state_path: Path, launcher_path: Path, extra_args: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["RD1_TEST_WANDB_STATE_PATH"] = str(state_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    cmd = [
        sys.executable,
        str(JOB_PY),
        "run-agent-with-retry",
        "--agent-launcher-path",
        str(launcher_path),
        "--manifest-path",
        str(manifest_path),
        "--wandb-sweep-id",
        "stub-sweep-id",
        "--path",
        env.get("PATH", ""),
        "--home",
        str(tmp_path),
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
    state_path = _write_wandb_state(tmp_path, snapshots=[["r0"], ["r0"], ["r0", "r1"]])
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
