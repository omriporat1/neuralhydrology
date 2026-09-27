"""Tests for ``scripts/rd1_p20_p24_job.sbatch``: the production Catfish GPU
header, the fail-closed ``.scratch_local`` project-boundary assertion on
``RD1_CHAIN_DIR``, deployed-harness path resolution, project-local temporary-
file containment, and invocation-time deployed-harness integrity
verification.

Runs the real script under ``bash`` (Git Bash on this Windows dev machine),
with ``REPO_WORKDIR``/``CANONICAL_PYTHON``/``FLASHNH_BASE`` overridden to
point at disposable ``tmp_path`` fixtures and a stub interpreter that only
logs its own argv -- never a real Slurm/W&B/Moriah contact. Every case below
uses ``RD1_DRY_RUN=--dry-run`` so the script exits before any W&B contact,
by design (see the sbatch file's own ``RD1_DRY_RUN`` handling), except the
project-local-temp-file and harness-verification tests below, which need to
observe behavior past that point / need a real interpreter respectively and
say so explicitly.

Covers, in order:
  1. The ``#SBATCH`` header carries the exact production Catfish GPU spec
     (partition, GPU class, CPUs, memory, walltime) -- a regression test for
     the class of failure where Slurm auto-tags a GPU-less job ``cpuonly``
     and it dies ``BadConstraints`` (see the Moriah catfish operational
     lesson this repairs).
  2. A ``RD1_CHAIN_DIR`` outside the project-local ``.scratch_local/`` is
     refused before any directory is created and before the deployed-
     harness check is even reached.
  3. A ``RD1_CHAIN_DIR`` beneath the broader Flash-NH base directory but not
     beneath *this* project's own ``.scratch_local/`` is refused (guards
     against the exact wrong convention the chain-submission script's own
     usage comment previously documented).
  4. A ``RD1_CHAIN_DIR`` beneath the project's ``.scratch_local/`` with no
     deployed harness present is refused, naming the expected deployed
     driver path.
  5. Once the harness is deployed at ``${RD1_CHAIN_DIR}/harness/scripts/
     rd1_p20_p24_job.py``, the dry run succeeds and the Python driver is
     invoked with exactly that absolute, ``RD1_CHAIN_DIR``-derived path --
     never the bare ``scripts/rd1_p20_p24_job.py`` string that would only
     resolve correctly by accident of the caller's current directory.
  6. (Review fix, 2026-09-27, task A) No default/global ``mktemp`` use: a
     static regression guard on the script text, plus a behavioral, full
     non-dry-run run proving the agent-attempt result file is created
     beneath ``${RD1_CHAIN_DIR}/tmp/`` while the driver subprocess runs and
     is gone (trap-cleaned) once the script exits.
  7. (Review fix, 2026-09-27, task B) Invocation-time deployed-harness
     integrity verification: a valid receipt lets the run proceed into
     preflight; a tampered deployed driver/module/submitter/sbatch-script or
     a malformed/missing/incomplete/mismatched receipt is refused, with
     ``HARNESS_VERIFY_FAILED`` on stderr, strictly before preflight (and
     therefore before any agent launch) is ever reached. These tests use a
     real Python interpreter (``sys.executable``), not the generic
     argv-logging stub, since the verifier is real stdlib Python.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_SBATCH = REPO_ROOT / "scripts" / "rd1_p20_p24_job.sbatch"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="requires a bash interpreter on PATH")

# Mirrors scripts/deploy_rd1_p20_p24_harness.py's DEPLOY_MANIFEST exactly, and
# is deliberately duplicated (not imported) here: the sbatch script's own new
# verifier duplicates this same literal list rather than trusting the
# deploy-time tool, and these tests check that actual, independent contract.
REQUIRED_HARNESS_PATHS: tuple[str, ...] = (
    "scripts/submit_rd1_p20_p24_chain.sh",
    "scripts/rd1_p20_p24_job.sbatch",
    "scripts/rd1_p20_p24_job.py",
    "src/__init__.py",
    "src/baseline/__init__.py",
    "src/baseline/rd1_p20_p24_env.py",
    "src/baseline/rd1_p20_p24_registry.py",
    "src/baseline/rd1_p20_p24_retry.py",
)
FROZEN_SCIENTIFIC_COMMIT = "dabd2ca851bd2b3a03035886cfa50015f2c864b4"


def _sbatch_text() -> str:
    return JOB_SBATCH.read_text(encoding="utf-8")


def _norm(path_text: str) -> str:
    """Forward-slash-normalized form, so a Windows-rendered ``pathlib.Path``
    string can be compared against a bash-constructed (mixed-separator, on
    this Windows dev machine) path for the same location."""
    return path_text.replace("\\", "/")


def test_sbatch_header_matches_production_catfish_gpu_spec():
    text = _sbatch_text()
    assert "#SBATCH --partition=catfish" in text
    assert "#SBATCH --gres=gpu:l4:1" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --mem=128G" in text
    assert "#SBATCH --time=08:00:00" in text


def _write_stub_python(bin_path: Path, log_path: Path) -> None:
    bin_path.write_text(
        "#!/bin/bash\n"
        f"printf '%s\\n' \"$@\" >> \"{log_path.as_posix()}\"\n"
        "echo '---' >> \"" + log_path.as_posix() + "\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)


def _base_env(tmp_path: Path, *, repo_workdir: Path, chain_dir: Path, python_log: Path) -> dict:
    stub_python = tmp_path / "stub_python.sh"
    _write_stub_python(stub_python, python_log)
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "REPO_WORKDIR": str(repo_workdir),
        "CANONICAL_PYTHON": str(stub_python),
        "FLASHNH_BASE": str(tmp_path / "unused_flashnh_base"),
        "RD1_PROPOSAL_ORDER": "21",
        "RD1_CHAIN_DIR": str(chain_dir),
        "RD1_REGISTRY_PATH": str(tmp_path / "registry.json"),
        "RD1_LOCK_PATH": str(tmp_path / "registry.lock"),
        "RD1_WANDB_SWEEP_ID": "disposable-test-sweep",
        "RD1_EXPECTED_COMMIT": "dabd2ca851bd2b3a03035886cfa50015f2c864b4",
        "RD1_EXECUTION_GENERATION": "1",
        "RD1_OUTPUT_ROOT_BASE": str(tmp_path / "outputs"),
        "RD1_DRY_RUN": "--dry-run",
    }


def _run(env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(JOB_SBATCH)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_refuses_chain_dir_outside_project_scratch_local(tmp_path):
    repo_workdir = tmp_path / "repo_root"
    repo_workdir.mkdir()
    chain_dir = tmp_path / "not_under_scratch_local" / "chain"  # sibling of repo_workdir, no .scratch_local
    env = _base_env(tmp_path, repo_workdir=repo_workdir, chain_dir=chain_dir, python_log=tmp_path / "py.log")

    proc = _run(env)

    assert proc.returncode != 0
    assert "must resolve beneath" in proc.stderr
    assert not chain_dir.exists()


def test_refuses_chain_dir_under_wrong_flashnh_base_scratch_local(tmp_path):
    repo_workdir = tmp_path / "repo_root"
    repo_workdir.mkdir()
    # Mirrors the exact wrong convention the submission script's usage
    # comment previously documented: .scratch_local as a sibling of the
    # project root, not beneath it.
    chain_dir = tmp_path / ".scratch_local" / "rd1_p20_p24_chain"
    env = _base_env(tmp_path, repo_workdir=repo_workdir, chain_dir=chain_dir, python_log=tmp_path / "py.log")

    proc = _run(env)

    assert proc.returncode != 0
    assert "must resolve beneath" in proc.stderr


def test_refuses_when_harness_not_deployed(tmp_path):
    repo_workdir = tmp_path / "repo_root"
    repo_workdir.mkdir()
    chain_dir = repo_workdir / ".scratch_local" / "rd1_p20_p24_chain"
    env = _base_env(tmp_path, repo_workdir=repo_workdir, chain_dir=chain_dir, python_log=tmp_path / "py.log")

    proc = _run(env)

    assert proc.returncode != 0
    expected_driver = chain_dir / "harness" / "scripts" / "rd1_p20_p24_job.py"
    assert "MISSING deployed harness job driver" in proc.stderr
    assert _norm(str(expected_driver)) in _norm(proc.stderr)


def test_succeeds_and_invokes_deployed_harness_absolute_path_not_cwd_relative(tmp_path):
    repo_workdir = tmp_path / "repo_root"
    repo_workdir.mkdir()
    chain_dir = repo_workdir / ".scratch_local" / "rd1_p20_p24_chain"

    # A decoy at the OLD cwd-relative location: if path resolution ever
    # regresses to a bare "scripts/rd1_p20_p24_job.py" lookup relative to
    # REPO_WORKDIR, this is what it would silently pick up instead.
    decoy_scripts_dir = repo_workdir / "scripts"
    decoy_scripts_dir.mkdir(parents=True)
    (decoy_scripts_dir / "rd1_p20_p24_job.py").write_text("# decoy, must never be invoked\n", encoding="utf-8")

    deployed_scripts_dir = chain_dir / "harness" / "scripts"
    deployed_scripts_dir.mkdir(parents=True)
    deployed_driver = deployed_scripts_dir / "rd1_p20_p24_job.py"
    deployed_driver.write_text("# deployed harness stub, never actually executed by the stub interpreter\n", encoding="utf-8")

    python_log = tmp_path / "py.log"
    env = _base_env(tmp_path, repo_workdir=repo_workdir, chain_dir=chain_dir, python_log=python_log)

    proc = _run(env)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DRY_RUN" in proc.stdout

    logged_lines = [_norm(line) for line in python_log.read_text(encoding="utf-8").splitlines()]
    assert _norm(str(deployed_driver)) in logged_lines
    # The bare cwd-relative token must never appear as its own logged argv
    # entry (it legitimately appears only as a trailing substring of the
    # deployed absolute path, e.g. ".../harness/scripts/rd1_p20_p24_job.py").
    assert "scripts/rd1_p20_p24_job.py" not in logged_lines


# --- Task A: project-local temporary-file containment (review fix, 2026-09-27) ---


def test_no_bare_mktemp_and_result_file_template_is_chain_dir_local():
    text = _sbatch_text()
    assert 'RESULT_FILE="$(mktemp)"' not in text
    assert "RD1_TMP_DIR=\"${RD1_CHAIN_DIR}/tmp\"" in text
    assert 'RESULT_FILE="$(mktemp "${RD1_TMP_DIR}/' in text
    assert "trap 'rm -f \"${RESULT_FILE}\"' EXIT" in text


def _write_universal_stub_python(bin_path: Path, log_path: Path, tmp_listing_path: Path) -> None:
    """A stub CANONICAL_PYTHON that intercepts every invocation (preflight,
    manifest build/validate, sanitize-env-probe, run-agent-with-retry,
    append-registry-row): logs argv and exits 0 for all of them, except for
    ``run-agent-with-retry``, where it additionally snapshots
    ``${RD1_CHAIN_DIR}/tmp`` (inherited from the parent shell's exported
    environment) into ``tmp_listing_path`` *while it runs* -- i.e. while the
    agent-attempt result file the sbatch script just ``mktemp``'d is open for
    this very subprocess' redirected stdout -- before printing the expected
    ``AGENT_OK`` marker the wrapper greps for."""
    bin_path.write_text(
        "#!/bin/bash\n"
        "for _a in \"$@\"; do\n"
        '  if [ "${_a}" = "run-agent-with-retry" ]; then\n'
        f'    ls -1 "${{RD1_CHAIN_DIR}}/tmp" > "{tmp_listing_path.as_posix()}" 2>/dev/null || true\n'
        '    echo "AGENT_OK run_id=stubrun123"\n'
        "    exit 0\n"
        "  fi\n"
        "done\n"
        f'printf \'%s\\n\' "$@" >> "{log_path.as_posix()}"\n'
        f'echo \'---\' >> "{log_path.as_posix()}"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)


def test_result_file_lives_under_chain_dir_tmp_and_is_removed_on_exit(tmp_path):
    repo_workdir = tmp_path / "repo_root"
    repo_workdir.mkdir()
    chain_dir = repo_workdir / ".scratch_local" / "chain"
    deployed_scripts_dir = chain_dir / "harness" / "scripts"
    deployed_scripts_dir.mkdir(parents=True)
    (deployed_scripts_dir / "rd1_p20_p24_job.py").write_text("# stub, intercepted by CANONICAL_PYTHON\n", encoding="utf-8")

    agent_launcher_dir = repo_workdir / "scripts"
    agent_launcher_dir.mkdir(exist_ok=True)
    (agent_launcher_dir / "run_sweep_v2_six_axis_wandb_agent_moriah.sbatch").write_text("# decoy\n", encoding="utf-8")

    python_log = tmp_path / "py.log"
    tmp_listing = tmp_path / "tmp_listing_during_run.txt"
    # Deliberately a different filename from the generic stub `_base_env`
    # writes at `tmp_path / "stub_python.sh"` -- `_base_env` unconditionally
    # (re)writes that exact path with its own boring argv-logging stub, which
    # would silently clobber a same-named custom stub written before it.
    stub_python = tmp_path / "stub_python_agent_intercept.sh"
    _write_universal_stub_python(stub_python, python_log, tmp_listing)

    env = _base_env(tmp_path, repo_workdir=repo_workdir, chain_dir=chain_dir, python_log=python_log)
    env["CANONICAL_PYTHON"] = str(stub_python)
    del env["RD1_DRY_RUN"]  # must run past the dry-run early exit to reach mktemp/agent-launch

    proc = _run(env)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "AGENT_OK run_id=stubrun123" in proc.stdout or "resolved run_id=stubrun123" in proc.stdout

    tmp_dir = chain_dir / "tmp"
    assert tmp_dir.is_dir()
    # During the run-agent-with-retry subprocess, the RESULT_FILE (already
    # mktemp'd under this exact directory) was present.
    during_run_listing = tmp_listing.read_text(encoding="utf-8").strip()
    assert during_run_listing, "expected at least one file under ${RD1_CHAIN_DIR}/tmp during the agent attempt"
    # After the script exits, the EXIT trap has removed it: nothing left.
    assert list(tmp_dir.iterdir()) == []


# --- Task B: invocation-time deployed-harness integrity verification -------
# (review fix, 2026-09-27). Uses a REAL Python interpreter as CANONICAL_PYTHON
# (the verifier is real stdlib Python, never the generic argv-logging stub),
# and stops the assertions at "did verification pass/fail correctly and
# strictly before preflight", not at overall script success -- these disposable
# fake-repo fixtures do not need to make every later production step
# (build-manifest/validate-launch against a real script) succeed too.


def _git_bash_realpath(path: Path) -> str:
    """The exact string the sbatch script's own boundary check computes for
    ``RD1_CHAIN_DIR`` (``RD1_CHAIN_DIR_REAL="$(realpath -m "${RD1_CHAIN_DIR}")"``,
    see ``scripts/rd1_p20_p24_job.sbatch``), so a receipt's ``chain_dir`` field
    matches exactly what the invocation-time verifier is handed as its second
    argv. Deliberately NOT ``cd "<dir>" && pwd``: some Git Bash installs mount
    the Windows temp directory at ``/tmp``, a mapping ``cd``/``pwd`` follow but
    a literal ``realpath -m`` string transform does not -- using ``cd``/``pwd``
    here previously produced a spurious ``/tmp/...`` value that disagreed with
    the script's own ``realpath -m``-derived path for the exact same directory."""
    posix_path = str(path).replace("\\", "/")
    proc = subprocess.run(["bash", "-c", f'realpath -m "{posix_path}"'], capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _deploy_valid_harness(chain_dir: Path) -> dict:
    """Deploys a minimal-but-real harness bundle (the driver is a no-op
    stub, real enough for a real Python interpreter to execute without
    error; the other 7 required files are inert placeholder content) plus a
    matching, fully valid ``deploy_receipt.json`` under
    ``${chain_dir}/harness/``. Returns the receipt dict actually written."""
    harness_dir = chain_dir / "harness"
    file_records = []
    for rel_path in REQUIRED_HARNESS_PATHS:
        dest = harness_dir / Path(rel_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if rel_path == "scripts/rd1_p20_p24_job.py":
            content = b"# deployed driver stub: always succeeds, ignores argv\n"
        else:
            content = f"# deployed stub content for {rel_path}\n".encode("utf-8")
        dest.write_bytes(content)
        file_records.append({"path": rel_path, "sha256": _sha256_hex(content), "deployed_path": str(dest), "executable": False})

    receipt = {
        "frozen_scientific_commit": FROZEN_SCIENTIFIC_COMMIT,
        "operational_base_commit": "0" * 40,
        "harness_commit": "1" * 40,
        "chain_dir": _git_bash_realpath(chain_dir),
        "deployed_at_utc": "2026-09-27T00:00:00Z",
        "provenance_audit": {},
        "files": file_records,
    }
    (harness_dir / "deploy_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def _base_env_for_verification(tmp_path: Path, *, repo_workdir: Path, chain_dir: Path) -> dict:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "REPO_WORKDIR": str(repo_workdir),
        "CANONICAL_PYTHON": sys.executable,
        "FLASHNH_BASE": str(tmp_path / "unused_flashnh_base"),
        "RD1_PROPOSAL_ORDER": "20",
        "RD1_CHAIN_DIR": str(chain_dir),
        "RD1_REGISTRY_PATH": str(tmp_path / "registry.json"),
        "RD1_LOCK_PATH": str(tmp_path / "registry.lock"),
        "RD1_WANDB_SWEEP_ID": "disposable-test-sweep",
        "RD1_EXPECTED_COMMIT": FROZEN_SCIENTIFIC_COMMIT,
        "RD1_EXECUTION_GENERATION": "1",
        "RD1_OUTPUT_ROOT_BASE": str(tmp_path / "outputs"),
        "RD1_DRY_RUN": "--dry-run",
        "RD1_P20_PINNED_MANIFEST_PATH": str(tmp_path / "p20_manifest.json"),
        "RD1_P20_PINNED_MANIFEST_SHA256": _sha256_hex(b"placeholder manifest, never opened for real"),
    }


def test_valid_receipt_permits_preflight(tmp_path):
    repo_workdir = tmp_path / "repo_root"
    repo_workdir.mkdir()
    chain_dir = repo_workdir / ".scratch_local" / "chain"
    chain_dir.mkdir(parents=True)
    _deploy_valid_harness(chain_dir)
    (tmp_path / "p20_manifest.json").write_text("placeholder manifest, never opened for real", encoding="utf-8")

    env = _base_env_for_verification(tmp_path, repo_workdir=repo_workdir, chain_dir=chain_dir)
    proc = _run(env)

    assert "HARNESS_VERIFY_OK" in proc.stdout, proc.stdout + proc.stderr
    verify_idx = proc.stdout.index("HARNESS_VERIFY_OK")
    preflight_idx = proc.stdout.index("--- preflight ---")
    assert verify_idx < preflight_idx


def _run_verification_case(tmp_path: Path, mutate) -> subprocess.CompletedProcess:
    repo_workdir = tmp_path / "repo_root"
    repo_workdir.mkdir()
    chain_dir = repo_workdir / ".scratch_local" / "chain"
    chain_dir.mkdir(parents=True)
    _deploy_valid_harness(chain_dir)
    (tmp_path / "p20_manifest.json").write_text("placeholder manifest, never opened for real", encoding="utf-8")

    mutate(chain_dir)

    env = _base_env_for_verification(tmp_path, repo_workdir=repo_workdir, chain_dir=chain_dir)
    return _run(env)


def _assert_refused_before_preflight(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode != 0
    assert "HARNESS_VERIFY_FAILED" in proc.stderr, proc.stdout + proc.stderr
    assert "--- preflight ---" not in proc.stdout
    assert "AGENT_OK" not in proc.stdout


def test_tampered_deployed_driver_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        (chain_dir / "harness" / "scripts" / "rd1_p20_p24_job.py").write_bytes(b"# tampered after deployment\n")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_tampered_imported_module_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        (chain_dir / "harness" / "src" / "baseline" / "rd1_p20_p24_env.py").write_bytes(b"# tampered after deployment\n")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_tampered_deployed_submitter_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        (chain_dir / "harness" / "scripts" / "submit_rd1_p20_p24_chain.sh").write_bytes(b"# tampered after deployment\n")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_tampered_deployed_sbatch_script_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        (chain_dir / "harness" / "scripts" / "rd1_p20_p24_job.sbatch").write_bytes(b"# tampered after deployment\n")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_missing_receipt_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        (chain_dir / "harness" / "deploy_receipt.json").unlink()

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_malformed_receipt_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        (chain_dir / "harness" / "deploy_receipt.json").write_text("{not valid json", encoding="utf-8")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_incomplete_receipt_missing_required_file_entry_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        receipt_path = chain_dir / "harness" / "deploy_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["files"] = [f for f in receipt["files"] if f["path"] != "src/baseline/rd1_p20_p24_retry.py"]
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_mismatched_frozen_scientific_commit_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        receipt_path = chain_dir / "harness" / "deploy_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["frozen_scientific_commit"] = "0" * 40
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_mismatched_chain_dir_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        receipt_path = chain_dir / "harness" / "deploy_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["chain_dir"] = receipt["chain_dir"] + "_not_the_real_one"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))


def test_checksum_mismatch_refused_before_preflight(tmp_path):
    def mutate(chain_dir: Path) -> None:
        receipt_path = chain_dir / "harness" / "deploy_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for entry in receipt["files"]:
            if entry["path"] == "src/baseline/rd1_p20_p24_registry.py":
                entry["sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    _assert_refused_before_preflight(_run_verification_case(tmp_path, mutate))
