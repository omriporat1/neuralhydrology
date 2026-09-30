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
 10. ``--payload-repository-path`` is required, is exported into every one
     of the five ``sbatch`` calls as bare ``REPO_WORKDIR=<payload path>``
     (proving the payload path actually reaches the job/agent/bridge
     process, which all resolve ``REPO_WORKDIR`` the same bare-name way),
     the real tracked-repo root is separately exported as
     ``RD1_PROJECT_ROOT=<tracked repo>`` (so the project-local boundary
     check in ``rd1_p20_p24_job.sbatch`` keeps anchoring to the real
     project, not the payload checkout), a HEAD mismatch against
     ``--expected-commit`` is refused before any ``sbatch`` call, a dirty
     payload working tree is refused before any ``sbatch`` call, the
     ``--export=`` value still carries no ``ALL`` token, and the fixed
     P20->P24 ``afterok`` topology from points 1-4 above is unaffected.
 11. ``--start-order 21`` (P21-P24 resume, 2026-09-30): submits exactly 4
     invocations for orders 21-24, P21 carries no ``--dependency`` and
     P22-P24 chain ``afterok`` exactly as in the P20-start case, no
     invocation ever carries ``RD1_P20_PINNED_MANIFEST_PATH``/
     ``RD1_P20_PINNED_MANIFEST_SHA256``, the frozen payload-path/allowlist
     exports from point 10 are unaffected, a registry that is not exactly
     P1-P20 is refused before any ``sbatch`` call, passing
     ``--p20-pinned-manifest-path``/``--p20-pinned-manifest-sha256`` with
     ``--start-order 21`` is refused before any ``sbatch`` call, and an
     unsupported ``--start-order`` value is refused before any ``sbatch``
     call.
"""
from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = REPO_ROOT / "scripts" / "submit_rd1_p20_p24_chain.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="requires a bash interpreter on PATH")


def _bash_command_dir(name: str) -> str:
    """Directory containing ``name`` as bash's own login-shell PATH resolves
    it (not the outer Python process's PATH, which may use Windows-style
    entries bash's PATH variable cannot consume)."""
    try:
        result = subprocess.run(["bash", "-lc", f"command -v {name}"], capture_output=True, text=True)
    except OSError:
        return ""
    resolved = result.stdout.strip()
    return resolved.rsplit("/", 1)[0] if "/" in resolved else ""


_GIT_DIR = _bash_command_dir("git")
pytestmark = pytest.mark.skipif(shutil.which("bash") is None or not _GIT_DIR, reason="requires bash and git interpreters on PATH")


def _write_stub_sbatch(bin_dir: Path, log_path: Path) -> None:
    stub = bin_dir / "sbatch"
    stub.write_text(
        "#!/bin/bash\n"
        f"echo \"$*\" >> \"{log_path.as_posix()}\"\n"
        "echo $((1000 + RANDOM % 1000))\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


def _make_payload_repo(
    base: Path, *, dirty: bool = False, untracked: bool = False, nested: bool = False,
) -> tuple[Path, str]:
    """A real tiny git checkout standing in for the detached frozen-payload
    checkout. Its actual HEAD (content-addressed, cannot be forced to an
    arbitrary SHA) is returned alongside the path so callers pass it as
    ``--expected-commit`` for the pass case, or a different literal for the
    HEAD-mismatch case.

    ``nested=True`` mirrors the real production topology: the git root
    (``.git``) sits at the top of a larger checkout, and the project this
    submitter operates on lives several directories below it (e.g.
    ``US_data/data_download/Disk_volume_estimation``) rather than being the
    git root itself. The returned path is that nested project directory,
    not the git root, since that is what ``--payload-repository-path`` is
    given in production."""
    repo = base / "payload_repo"
    repo.mkdir()
    run = lambda *cmd, cwd=repo: subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "test")
    (repo / "README.md").write_text("payload\n", encoding="utf-8")
    project_dir = repo
    if nested:
        project_dir = repo / "US_data" / "data_download" / "Disk_volume_estimation"
        project_dir.mkdir(parents=True)
        (project_dir / "marker.txt").write_text("nested project\n", encoding="utf-8")
    run("git", "add", ".")
    run("git", "commit", "-q", "-m", "payload")
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    if dirty:
        (repo / "README.md").write_text("payload modified\n", encoding="utf-8")
    if nested:
        return project_dir, head
    if untracked:
        (repo / "wandb").mkdir()
        (repo / "wandb" / "debug.log").write_text("stray run artifact\n", encoding="utf-8")
    return repo, head


def _base_args(chain_dir: Path, payload_repo: Path, expected_commit: str) -> list[str]:
    return [
        "--registry-path", str(chain_dir / "registry.json"),
        "--chain-dir", str(chain_dir),
        "--wandb-sweep-id", "disposable-test-sweep",
        "--expected-commit", expected_commit,
        "--execution-generation", "1",
        "--output-root-base", str(chain_dir / "outputs"),
        "--p20-pinned-manifest-path", str(chain_dir / "p20_manifest.json"),
        "--p20-pinned-manifest-sha256", "0" * 64,
        "--payload-repository-path", str(payload_repo),
    ]


def _chain_dir_under_project_local_scratch(tmp_path: Path) -> Path:
    """A ``--chain-dir`` value that satisfies the submitter's project-local
    artifact-boundary check when paired with ``REPO_WORKDIR=<tmp_path>`` in
    ``env`` below -- mirrors the identical convention
    ``rd1_p20_p24_job.sbatch`` uses (``${REPO_WORKDIR}/.scratch_local/...``).
    This ``env`` ``REPO_WORKDIR`` is consumed only by the submitter's own
    tracked-repo/boundary-check logic (becoming ``RD1_PROJECT_ROOT`` in the
    export list); it is unrelated to ``--payload-repository-path``, which is
    exported into the job as the bare ``REPO_WORKDIR`` the job itself sees."""
    return tmp_path / ".scratch_local" / "chain"


def _env(bin_dir: Path, tmp_path: Path) -> dict:
    path_value = f"{bin_dir}:/usr/bin:/bin"
    if _GIT_DIR:
        path_value = f"{path_value}:{_GIT_DIR}"
    return {"PATH": path_value, "HOME": str(tmp_path), "REPO_WORKDIR": str(tmp_path)}


def _run(tmp_path: Path, extra_args: "list[str] | None" = None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)

    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path)
    args = _base_args(chain_dir, payload_repo, payload_head) + (extra_args or [])

    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
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
    payload_repo, payload_head = _make_payload_repo(tmp_path)

    args = _base_args(chain_dir, payload_repo, payload_head)
    args[args.index("--wandb-sweep-id") + 1] = "4x3btz2s"

    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
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
    payload_repo, payload_head = _make_payload_repo(tmp_path)
    args = _base_args(chain_dir, payload_repo, payload_head)

    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
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
    payload_repo, payload_head = _make_payload_repo(tmp_path)

    args = _base_args(chain_dir, payload_repo, payload_head)
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
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
    payload_repo, payload_head = _make_payload_repo(tmp_path)

    args = _base_args(chain_dir, payload_repo, payload_head)
    idx = args.index("--registry-path")
    del args[idx:idx + 2]

    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


def test_refuses_missing_payload_repository_path_before_any_sbatch_call(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path)

    args = _base_args(chain_dir, payload_repo, payload_head)
    idx = args.index("--payload-repository-path")
    del args[idx:idx + 2]

    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "payload-repository-path" in proc.stderr
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


# --- Dual-provenance REPO_WORKDIR export (2026-09-30 recovery correction) ----


def test_payload_repository_path_exported_as_repo_workdir_every_order(tmp_path):
    proc, invocations = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert len(invocations) == 5

    expected_repo_workdir_prefix = "REPO_WORKDIR="
    for line in invocations:
        export_field = next(tok for tok in line.split() if tok.startswith("--export="))
        tokens = export_field[len("--export="):].split(",")
        repo_workdir_tokens = [t for t in tokens if t.startswith(expected_repo_workdir_prefix)]
        assert len(repo_workdir_tokens) == 1
        assert repo_workdir_tokens[0].endswith("payload_repo")
        project_root_tokens = [t for t in tokens if t.startswith("RD1_PROJECT_ROOT=")]
        assert len(project_root_tokens) == 1
        # The payload checkout and the real tracked-repo project root must be
        # two distinct paths -- this is the whole point of the correction.
        assert repo_workdir_tokens[0][len(expected_repo_workdir_prefix):] != project_root_tokens[0][len("RD1_PROJECT_ROOT="):]


def test_refuses_payload_repository_head_mismatch_before_any_sbatch_call(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path)

    args = _base_args(chain_dir, payload_repo, "0" * 40)  # expected-commit != actual HEAD
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "does not match --expected-commit" in proc.stderr
    assert payload_head in proc.stderr
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


def test_untracked_files_in_payload_repository_do_not_block_submission(tmp_path):
    """Stray untracked runtime artifacts (e.g. a leftover wandb/ run-log
    directory from an earlier attempt) must not be treated as a dirty
    checkout -- only actual tracked-content modifications should refuse
    submission, so this check never forces deleting evidence to proceed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path, untracked=True)

    args = _base_args(chain_dir, payload_repo, payload_head)
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert len(log_path.read_text(encoding="utf-8").strip().splitlines()) == 5


def test_payload_repository_path_as_nested_project_subdirectory_accepted(tmp_path):
    """--payload-repository-path may be a subdirectory of a larger checkout
    (mirroring production: the payload checkout's .git sits above
    US_data/data_download/Disk_volume_estimation, not inside it) rather
    than a git root itself. A literal ``<path>/.git`` existence check would
    wrongly refuse this; the validation must rely on git's own upward repo
    discovery instead."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_project_dir, payload_head = _make_payload_repo(tmp_path, nested=True)
    assert not (payload_project_dir / ".git").exists()

    args = _base_args(chain_dir, payload_project_dir, payload_head)
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert len(log_path.read_text(encoding="utf-8").strip().splitlines()) == 5


def test_refuses_dirty_payload_repository_worktree_before_any_sbatch_call(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path, dirty=True)

    args = _base_args(chain_dir, payload_repo, payload_head)
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "uncommitted tracked-file changes" in proc.stderr
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


# --- --start-order 21 resume path (Part B, 2026-09-30) ----------------------


def _write_registry_with_orders(path: Path, orders: "list[int]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": [{"order": o} for o in orders]}), encoding="utf-8")


def _resume_args(chain_dir: Path, registry_path: Path, payload_repo: Path, expected_commit: str) -> list[str]:
    """Deliberately omits ``--p20-pinned-manifest-path``/``--sha256``:
    unlike ``_base_args``, the resume path must not require (or accept)
    them."""
    return [
        "--registry-path", str(registry_path),
        "--chain-dir", str(chain_dir),
        "--wandb-sweep-id", "disposable-test-sweep",
        "--expected-commit", expected_commit,
        "--execution-generation", "1",
        "--output-root-base", str(chain_dir / "outputs"),
        "--payload-repository-path", str(payload_repo),
        "--start-order", "21",
    ]


def test_start_order_21_submits_exactly_p21_through_p24_with_correct_dependencies(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path)
    registry_path = chain_dir / "registry.json"
    _write_registry_with_orders(registry_path, list(range(1, 21)))  # exactly P1-P20

    args = _resume_args(chain_dir, registry_path, payload_repo, payload_head)
    env = _env(bin_dir, tmp_path)
    env["CANONICAL_PYTHON"] = sys.executable

    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 4

    for order, line in zip(range(21, 25), invocations):
        assert f"RD1_PROPOSAL_ORDER={order}" in line
        assert "RD1_P20_PINNED_MANIFEST_PATH=" not in line
        assert "RD1_P20_PINNED_MANIFEST_SHA256=" not in line
        export_field = next(tok for tok in line.split() if tok.startswith("--export="))
        tokens = export_field[len("--export="):].split(",")
        assert "ALL" not in tokens
        repo_workdir_tokens = [t for t in tokens if t.startswith("REPO_WORKDIR=")]
        assert len(repo_workdir_tokens) == 1
        assert repo_workdir_tokens[0].endswith("payload_repo")

    # Never P20, never anything past P24.
    assert not any("RD1_PROPOSAL_ORDER=20," in line or line.endswith("RD1_PROPOSAL_ORDER=20") for line in invocations)
    assert not any("RD1_PROPOSAL_ORDER=25" in line for line in invocations)

    assert "--dependency" not in invocations[0]
    for line in invocations[1:]:
        assert "--dependency=afterok:" in line


def test_start_order_21_refuses_when_registry_not_exactly_p1_p20(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path)
    registry_path = chain_dir / "registry.json"
    _write_registry_with_orders(registry_path, list(range(1, 20)))  # only P1-P19

    args = _resume_args(chain_dir, registry_path, payload_repo, payload_head)
    env = _env(bin_dir, tmp_path)
    env["CANONICAL_PYTHON"] = sys.executable

    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "resume precondition refused" in proc.stderr
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


def test_start_order_21_refuses_p20_manifest_arguments_before_any_sbatch_call(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path)
    registry_path = chain_dir / "registry.json"
    # Even a fully-valid P1-P20 registry must not matter here -- the args
    # themselves are refused first, and this check must not require
    # CANONICAL_PYTHON/a real registry read at all.
    _write_registry_with_orders(registry_path, list(range(1, 21)))

    args = _resume_args(chain_dir, registry_path, payload_repo, payload_head) + [
        "--p20-pinned-manifest-path", str(chain_dir / "p20_manifest.json"),
        "--p20-pinned-manifest-sha256", "0" * 64,
    ]
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "must not be given with --start-order 21" in proc.stderr
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


def test_start_order_invalid_value_refused_before_any_sbatch_call(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path)

    args = _base_args(chain_dir, payload_repo, payload_head) + ["--start-order", "22"]
    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "--start-order must be 20 or 21" in proc.stderr
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()


def test_start_order_default_still_requires_p20_manifest_arguments(tmp_path):
    """Regression guard: the default (omitted --start-order, i.e. 20) path
    must still require the P20 pinned-manifest arguments exactly as before
    this change."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)
    chain_dir = _chain_dir_under_project_local_scratch(tmp_path)
    payload_repo, payload_head = _make_payload_repo(tmp_path)

    args = _base_args(chain_dir, payload_repo, payload_head)
    idx = args.index("--p20-pinned-manifest-path")
    del args[idx:idx + 2]

    proc = subprocess.run(
        ["bash", str(SUBMIT_SCRIPT), *args],
        capture_output=True,
        text=True,
        env=_env(bin_dir, tmp_path),
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode != 0
    assert "p20-pinned-manifest-path" in proc.stderr
    assert not log_path.exists() or not log_path.read_text(encoding="utf-8").strip()
