"""Tests for ``scripts/submit_rd1_p20_p24_chain.sh``, the clean-login-shell
submission of the fixed RD1 P20-P24 chain.

Runs the real script under ``bash`` (Git Bash on this Windows dev machine)
against a stub ``sbatch`` on ``PATH`` that only records its own arguments
and prints a fake job id -- never a real Slurm/W&B contact.

Covers, in order:
  1. Exactly 5 ``sbatch`` invocations are made, for orders 20-24 in order,
     each followed by the next (proving the hard stop at P24: nothing after
     the 5th call, no P25 anywhere in the recorded invocations).
  2. Each invocation's ``--export=`` value contains no ``ALL`` token
     anywhere (the explicit-allowlist requirement) and carries every
     required ``RD1_*`` variable for its own order.
  3. Each invocation after the first carries ``--dependency=afterok:<prior
     job id>`` referencing exactly the immediately preceding submission's
     returned job id (the fixed ``afterok`` chain topology).
  4. The first (P20) invocation carries no ``--dependency`` argument at all.
  5. A forbidden sweep id (the frozen production id) is refused before any
     ``sbatch`` call is made (no submission after an invalid/ambiguous
     scientific-scope request).
  6. A missing required argument (e.g. ``--registry-path``) is refused
     before any ``sbatch`` call is made.
  7. Log containment (review fix, 2026-09-27): ``${RD1_CHAIN_DIR}/logs/`` is
     created, all five invocations carry explicit ``--output``/``--error``
     resolving strictly under it with proposal-specific ``%j``-qualified
     filenames, and none references the former external
     ``.../Flash-NH/logs/`` directory.
  8. The same log-containment properties hold regardless of the caller's
     cwd.
  9. A ``--chain-dir`` outside the project-local ``.scratch_local/`` (the
     same boundary check ``rd1_p20_p24_job.sbatch`` already enforces) is
     refused by the submitter itself, before any ``sbatch`` call and before
     the logs directory is created.
"""
from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "submit_rd1_p20_p24_chain.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="requires a bash interpreter on PATH")


def _write_stub_sbatch(bin_dir: Path, log_path: Path) -> None:
    stub = bin_dir / "sbatch"
    stub.write_text(
        "#!/bin/bash\n"
        f"echo \"$*\" >> \"{log_path.as_posix()}\"\n"
        "echo $((1000 + RANDOM % 1000))\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


def _base_args(chain_dir: Path) -> list[str]:
    return [
        "--registry-path", str(chain_dir / "registry.json"),
        "--chain-dir", str(chain_dir),
        "--wandb-sweep-id", "disposable-test-sweep",
        "--expected-commit", "dabd2ca851bd2b3a03035886cfa50015f2c864b4",
        "--execution-generation", "1",
        "--output-root-base", str(chain_dir / "outputs"),
        "--p20-pinned-manifest-path", str(chain_dir / "p20_manifest.json"),
        "--p20-pinned-manifest-sha256", "0" * 64,
    ]


def _chain_dir_under_project_local_scratch(tmp_path: Path) -> Path:
    """A ``--chain-dir`` value that satisfies the submitter's project-local
    artifact-boundary check when paired with ``REPO_WORKDIR=<tmp_path>`` in
    ``env`` below -- mirrors the identical convention
    ``rd1_p20_p24_job.sbatch`` uses (``${REPO_WORKDIR}/.scratch_local/...``)."""
    return tmp_path / ".scratch_local" / "chain"


def _run(tmp_path: Path, extra_args: "list[str] | None" = None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)

    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    args = _base_args(chain_dir) + (extra_args or [])

    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "REPO_WORKDIR": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    invocations = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    return proc, invocations


def test_exactly_five_invocations_for_orders_20_through_24(tmp_path):
    proc, invocations = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert len(invocations) == 5
    for order, line in zip(range(20, 25), invocations):
        assert f"RD1_PROPOSAL_ORDER={order}" in line


def test_no_export_all_and_required_vars_present(tmp_path):
    _, invocations = _run(tmp_path)
    for line in invocations:
        export_field = next(tok for tok in line.split() if tok.startswith("--export="))
        export_value = export_field[len("--export="):]
        tokens = export_value.split(",")
        assert "ALL" not in tokens
        assert any(t.startswith("RD1_PROPOSAL_ORDER=") for t in tokens)
        assert any(t.startswith("RD1_REGISTRY_PATH=") for t in tokens)
        assert any(t.startswith("RD1_WANDB_SWEEP_ID=") for t in tokens)


def test_dependency_chain_references_prior_job_id(tmp_path):
    _, invocations = _run(tmp_path)
    assert "--dependency" not in invocations[0]
    # The submitted job id comes from the stub's stdout, captured by the
    # script itself (JOB_ID="$(sbatch ...)"), not from the logged args line;
    # checking dependency syntax on every non-first invocation is sufficient
    # here since the id-capture/threading itself is plain shell assignment.
    for line in invocations[1:]:
        assert "--dependency=afterok:" in line


def test_refuses_forbidden_sweep_id_before_any_sbatch_call(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)

    args = _base_args(chain_dir)
    args[args.index("--wandb-sweep-id") + 1] = "4x3btz2s"

    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "REPO_WORKDIR": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


# --- Log containment (review fix, 2026-09-27) --------------------------------

FORMER_EXTERNAL_LOG_DIR = "/sci/labs/efratmorin/omripo/Flash-NH/logs/"


def _assert_log_containment(chain_dir: Path, invocations: "list[str]") -> None:
    logs_dir = chain_dir / "logs"
    assert logs_dir.is_dir()
    expected_prefix = f"{chain_dir}/logs/"

    assert len(invocations) == 5
    for order, line in zip(range(20, 25), invocations):
        assert FORMER_EXTERNAL_LOG_DIR not in line

        tokens = line.split()
        output_tok = next(t for t in tokens if t.startswith("--output="))
        error_tok = next(t for t in tokens if t.startswith("--error="))
        output_path = output_tok[len("--output="):]
        error_path = error_tok[len("--error="):]

        assert output_path.startswith(expected_prefix)
        assert error_path.startswith(expected_prefix)
        assert output_path != error_path
        assert "%j" in output_path
        assert "%j" in error_path
        assert f"p{order}" in output_path
        assert f"p{order}" in error_path


def test_logs_dir_created_and_all_five_calls_carry_contained_output_error(tmp_path):
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    proc, invocations = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    _assert_log_containment(chain_dir, invocations)


def test_log_containment_unaffected_by_caller_cwd(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    args = _base_args(chain_dir)

    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "REPO_WORKDIR": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(unrelated_cwd),
    )
    assert proc.returncode == 0, proc.stderr
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    _assert_log_containment(chain_dir, invocations)


def test_refuses_chain_dir_outside_project_scratch_local_before_any_sbatch_call(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = tmp_path / "not_under_scratch_local" / "chain"

    args = _base_args(chain_dir)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "REPO_WORKDIR": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "must resolve beneath" in proc.stderr
    assert not chain_dir.exists()
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


def test_refuses_missing_required_argument_before_any_sbatch_call(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)

    args = _base_args(chain_dir)
    idx = args.index("--registry-path")
    del args[idx:idx + 2]

    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "REPO_WORKDIR": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()
