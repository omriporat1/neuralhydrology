"""RD1-C4-D1 section I: the Slurm entry points are implemented, not launchable
by accident.

These tests read the wrapper files as text. They deliberately do not execute
them: the point is that submitting the wrong file, or the right file without
the right explicit inputs, cannot start work -- and proving that by running
anything would defeat it. Nothing here submits a job, contacts a scheduler,
or touches a real Moriah product.

Checks that ask "does this file *do* X" run against the executable text only:
comments and quoted strings are stripped first. The wrappers deliberately
*talk about* the array, the reducer and W&B -- telling the operator which
file to use and what is not automatic is part of the accident-proofing, and
a test that could not tell advice from action would forbid the advice.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "rd1_c4_d1"
BENCHMARK = SCRIPTS / "rd1_c4_d1_benchmark.sbatch"
ARRAY = SCRIPTS / "rd1_c4_d1_array.sbatch"
REDUCE = SCRIPTS / "rd1_c4_d1_reduce.sbatch"
COMMON = SCRIPTS / "rd1_c4_d1_common.sh"

ENTRY_POINTS = [BENCHMARK, ARRAY, REDUCE]


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_quoted(line: str) -> str:
    """Drop the contents of quoted spans, keeping the command words."""
    out = []
    quote = ""
    for char in line:
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in ('"', "'"):
            quote = char
            continue
        out.append(char)
    return "".join(out)


def _uncommented_text(path: Path) -> str:
    """The file with comment lines removed, quoted text kept.

    Guard expressions such as ``"${SLURM_ARRAY_TASK_ID:-}"`` are quoted in
    shell, so a presence check has to look at this text, not at the
    quote-stripped text used for the "does not do X" checks.
    """
    lines = [
        line.strip()
        for line in _text(path).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return "\n".join(lines)


def _executable_text(path: Path) -> str:
    """The file with comment lines and quoted text removed."""
    lines = []
    for line in _text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(_strip_quoted(stripped))
    return "\n".join(lines)


@pytest.mark.parametrize("path", [*ENTRY_POINTS, COMMON], ids=lambda path: path.name)
def test_every_entry_point_exists(path):
    assert path.is_file(), f"missing D1 entry point {path.name}"


@pytest.mark.parametrize("path", ENTRY_POINTS, ids=lambda path: path.name)
def test_no_entry_point_carries_an_array_directive(path):
    """An ``#SBATCH --array=`` directive in the file would make the full
    24-trial array the default behaviour of a bare ``sbatch``. The operator
    must supply the range and the concurrency cap explicitly on the command
    line, every time."""
    for line in _text(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("#SBATCH") and "--array" in stripped:
            pytest.fail(f"{path.name} carries an array directive: {stripped}")


def test_the_benchmark_refuses_to_run_inside_an_array_allocation():
    """'The implementation should make it difficult to launch the full array
    accidentally when only a benchmark is intended.' Submitting the benchmark
    file *with* ``--array`` is exactly that accident, so the benchmark tests
    the array variables and refuses."""
    text = _text(BENCHMARK)
    assert "SLURM_ARRAY_TASK_ID:-" in _uncommented_text(BENCHMARK)
    assert "SLURM_ARRAY_JOB_ID:-" in _uncommented_text(BENCHMARK)
    assert "REFUSING: the benchmark was submitted with --array" in text


def test_the_benchmark_needs_one_explicitly_named_trial():
    assert "D1_BENCHMARK_TRIAL_ID" in _uncommented_text(BENCHMARK)
    assert "REFUSING: export D1_BENCHMARK_TRIAL_ID" in _text(BENCHMARK)


def test_the_array_refuses_without_the_explicit_confirmation_token():
    assert "D1_ARRAY_CONFIRM:-" in _uncommented_text(ARRAY)
    assert '"run-all-24-trials"' in _text(ARRAY)


def test_the_array_refuses_when_submitted_without_an_array_range():
    """A bare ``sbatch`` of the array file produces a single task with no
    ``SLURM_ARRAY_TASK_ID``. That must refuse rather than quietly run trial
    zero, or one trial would be produced under a job that looks like the
    whole campaign."""
    assert "SLURM_ARRAY_TASK_ID:-" in _uncommented_text(ARRAY)
    assert "REFUSING: no SLURM_ARRAY_TASK_ID" in _text(ARRAY)


def test_the_array_refuses_an_index_outside_the_frozen_24_trial_range():
    assert "-ge 24" in _uncommented_text(ARRAY)
    assert "outside the frozen 24-trial range 0-23" in _text(ARRAY)


def test_the_array_does_not_invoke_the_reducer():
    """Section I: no automatic reduction before all 24 receipts qualify. The
    array wrapper may *say* the reduction is manual -- and does -- but must
    not call it."""
    executable = _executable_text(ARRAY)
    assert "rd1_c4_d1_reduce" not in executable
    assert "reduce" not in executable
    assert "Reduction is NOT automatic" in _text(ARRAY)


@pytest.mark.parametrize("path", ENTRY_POINTS, ids=lambda path: path.name)
def test_no_entry_point_submits_anything(path):
    """None of these files may submit a job -- not itself, not a successor."""
    assert "sbatch" not in _executable_text(path)


@pytest.mark.parametrize("path", ENTRY_POINTS, ids=lambda path: path.name)
def test_no_entry_point_contacts_wandb_a_controller_or_the_network(path):
    executable = _executable_text(path).lower()
    for forbidden in ("wandb", "curl", "wget", "ssh ", "controller"):
        assert forbidden not in executable, f"{path.name} runs {forbidden!r}"


@pytest.mark.parametrize("path", ENTRY_POINTS, ids=lambda path: path.name)
def test_every_entry_point_is_marked_not_launched(path):
    assert "NOT LAUNCHED" in _text(path)


@pytest.mark.parametrize("path", ENTRY_POINTS, ids=lambda path: path.name)
def test_no_entry_point_reserves_more_than_the_provisional_one_cpu(path):
    """'Do not reserve eight CPUs unless within-task parallelism is
    implemented and benchmarked.' It is not implemented, so the reservation
    stays at one CPU."""
    directives = [
        line.strip()
        for line in _text(path).splitlines()
        if line.strip().startswith("#SBATCH") and "cpus-per-task" in line
    ]
    assert directives, f"{path.name} does not state a CPU reservation"
    for directive in directives:
        assert directive.endswith("=1"), f"{path.name}: {directive}"


# --- benchmark store-root isolation: real executable guard behaviour -------
#
# The checks above are text-based by design (see module docstring): they
# prove a *file* never contains a dangerous action. The store-root isolation
# guard below is different -- it is branching logic over environment values,
# and "the words REFUSING and D1_BENCHMARK_STORE_ROOT appear somewhere in the
# file" cannot prove which paths it actually accepts or rejects. So these
# tests really execute the guard.
#
# This is still harmless: rd1_c4_d1_benchmark.sbatch guards its own main body
# with `if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then return/exit; fi`, so
# *sourcing* it (as every test below does) only defines
# `_d1_benchmark_validate_store_root` and friends -- it never reaches the
# array/trial-id checks, `rd1_c4_d1_require_inputs`, `mkdir`, `rd1_c4_d1_setup_env`,
# or the python invocation. No sbatch, no python, no network, no module/conda
# activation, ever happens here. Two further tests near the end of this
# section do execute the file as a script (not source it) specifically to
# prove the pre-existing array and one-trial refusals still fire first --
# but even those exit (via the early "REFUSING" checks) long before
# `rd1_c4_d1_require_inputs`, `mkdir`, or `rd1_c4_d1_setup_env` are reached.

pytestmark_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required to exercise the guard")


def _bash_available() -> bool:
    return shutil.which("bash") is not None


def _guard_env(tmp_path: Path, **d1_vars: "str | None") -> tuple[dict, Path]:
    """A minimal, isolated environment: a fresh FLASHNH_BASE under tmp_path,
    with any D1_* variables the parent test process happens to carry removed
    first so "unset" really means unset."""
    flashnh_base = tmp_path / "flashnh_base"
    flashnh_base.mkdir(exist_ok=True)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("D1_") or key.startswith("SLURM_"):
            env.pop(key)
    env["FLASHNH_BASE"] = str(flashnh_base)
    for key, value in d1_vars.items():
        if value is not None:
            env[key] = value
    return env, flashnh_base


def _call_guard(tmp_path: Path, **d1_vars: "str | None") -> tuple[subprocess.CompletedProcess, Path]:
    """Source the benchmark script (main body never runs -- see above) and
    call the store-root guard function directly, returning its exit code,
    stdout (the resolved path, on success) and stderr (the refusal, on
    failure)."""
    env, flashnh_base = _guard_env(tmp_path, **d1_vars)
    script = f'source "{BENCHMARK}" && _d1_benchmark_validate_store_root'
    result = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, flashnh_base


@pytestmark_bash
def test_benchmark_store_root_guard_refuses_when_unset(tmp_path):
    result, _ = _call_guard(tmp_path)
    assert result.returncode != 0
    assert "REFUSING" in result.stderr
    assert "D1_BENCHMARK_STORE_ROOT" in result.stderr


@pytestmark_bash
def test_benchmark_store_root_guard_refuses_when_empty(tmp_path):
    result, _ = _call_guard(tmp_path, D1_BENCHMARK_STORE_ROOT="")
    assert result.returncode != 0
    assert "REFUSING" in result.stderr
    assert "D1_BENCHMARK_STORE_ROOT" in result.stderr


@pytestmark_bash
def test_benchmark_store_root_guard_refuses_the_shared_default_store(tmp_path):
    flashnh_base = tmp_path / "flashnh_base"
    shared_default = str(flashnh_base / "tmp" / "rd1_c4_d1" / "store")
    result, _ = _call_guard(tmp_path, D1_BENCHMARK_STORE_ROOT=shared_default)
    assert result.returncode != 0
    assert "REFUSING" in result.stderr


@pytestmark_bash
def test_benchmark_store_root_guard_refuses_equality_with_supplied_shared_store_root(tmp_path):
    env, flashnh_base = _guard_env(tmp_path)
    shared = str(flashnh_base / "tmp" / "rd1_c4_d1" / "benchmarks" / "shared_reused")
    env["D1_STORE_ROOT"] = shared
    env["D1_BENCHMARK_STORE_ROOT"] = shared
    script = f'source "{BENCHMARK}" && _d1_benchmark_validate_store_root'
    result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "REFUSING" in result.stderr
    assert "D1_STORE_ROOT" in result.stderr


@pytestmark_bash
def test_benchmark_store_root_guard_refuses_a_path_outside_the_approved_area(tmp_path):
    flashnh_base = tmp_path / "flashnh_base"
    outside = str(flashnh_base / "tmp" / "rd1_c4_d1" / "not_benchmarks" / "attempt1")
    result, _ = _call_guard(tmp_path, D1_BENCHMARK_STORE_ROOT=outside)
    assert result.returncode != 0
    assert "REFUSING" in result.stderr
    assert "approved benchmark area" in result.stderr


@pytestmark_bash
def test_benchmark_store_root_guard_refuses_a_store_root_that_already_exists(tmp_path):
    flashnh_base = tmp_path / "flashnh_base"
    existing = flashnh_base / "tmp" / "rd1_c4_d1" / "benchmarks" / "already_there"
    existing.mkdir(parents=True)
    result, _ = _call_guard(tmp_path, D1_BENCHMARK_STORE_ROOT=str(existing))
    assert result.returncode != 0
    assert "REFUSING" in result.stderr
    assert "already exists" in result.stderr


@pytestmark_bash
def test_benchmark_store_root_guard_accepts_a_fresh_path_inside_the_approved_area(tmp_path):
    flashnh_base = tmp_path / "flashnh_base"
    fresh = flashnh_base / "tmp" / "rd1_c4_d1" / "benchmarks" / "attempt_2026_09_16"
    result, _ = _call_guard(tmp_path, D1_BENCHMARK_STORE_ROOT=str(fresh))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() != ""
    assert not fresh.exists(), "the guard must never create the store root itself"


@pytestmark_bash
def test_benchmark_store_root_guard_refuses_a_dangling_symlink():
    """RD1-C4-D1 independent-review finding: the raw operator-supplied
    pathname must be rejected if it exists in ANY form -- including as a
    symlink whose target does not exist -- before it is ever passed to
    ``realpath``, which would otherwise canonicalize the dangling link away
    and let it slip past the existence checks undetected.

    This uses a real symlink, so it needs an OS-level ability the sandboxed
    Windows account this suite may be running under does not necessarily
    have (``SeCreateSymbolicLinkPrivilege``). Only that exact failure mode
    (Windows error 1314, ERROR_PRIVILEGE_NOT_HELD) is treated as a skip; any
    other error is a genuine test failure and is re-raised.
    """
    project_root = Path(__file__).resolve().parents[1]
    scratch_root = project_root / "tmp" / f"rd1_c4_d1_symlink_guard_test_{uuid.uuid4().hex[:12]}"
    scratch_root.mkdir(parents=True)
    try:
        flashnh_base = scratch_root / "flashnh_base"
        benchmark_area = flashnh_base / "tmp" / "rd1_c4_d1" / "benchmarks"
        benchmark_area.mkdir(parents=True)
        link_path = benchmark_area / "attempt"
        target_path = benchmark_area / "nonexistent_target"

        try:
            link_path.symlink_to(target_path)
        except OSError as exc:
            if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
                pytest.skip(
                    "Windows process lacks SeCreateSymbolicLinkPrivilege; "
                    "behavioral symlink guard requires Linux/Moriah verification"
                )
            raise

        assert link_path.is_symlink()
        assert not target_path.exists()

        env, _ = _guard_env(scratch_root, D1_BENCHMARK_STORE_ROOT=str(link_path))
        script = f'source "{BENCHMARK}" && _d1_benchmark_validate_store_root'
        result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=30)

        assert result.returncode != 0
        assert "REFUSING" in result.stderr

        assert not target_path.exists(), "validation must never create the dangling symlink's target"
        created = {entry.name for entry in benchmark_area.iterdir()}
        assert created == {"attempt"}, f"validation created unexpected output under the benchmark area: {created}"
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)


@pytestmark_bash
def test_benchmark_still_refuses_inside_an_array_allocation_when_actually_executed(tmp_path):
    """Section I's existing no-array protection, proven by real execution
    rather than text matching. The guard fires before the store-root check,
    `rd1_c4_d1_require_inputs`, or `rd1_c4_d1_setup_env`, so nothing beyond a
    fresh FLASHNH_BASE directory is ever touched."""
    flashnh_base = tmp_path / "flashnh_base"
    fresh = flashnh_base / "tmp" / "rd1_c4_d1" / "benchmarks" / "attempt_array_check"
    env, _ = _guard_env(
        tmp_path,
        D1_BENCHMARK_STORE_ROOT=str(fresh),
        D1_BENCHMARK_TRIAL_ID="some_trial",
        SLURM_ARRAY_TASK_ID="3",
    )
    result = subprocess.run(["bash", str(BENCHMARK)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "REFUSING: the benchmark was submitted with --array" in result.stderr
    assert not fresh.exists()


@pytestmark_bash
def test_benchmark_still_refuses_without_an_explicit_trial_when_actually_executed(tmp_path):
    """Section I's existing one-trial protection, proven by real execution.
    The trial-id guard fires before the store-root check, so a missing
    D1_BENCHMARK_TRIAL_ID is refused even with an otherwise-valid store
    root."""
    flashnh_base = tmp_path / "flashnh_base"
    fresh = flashnh_base / "tmp" / "rd1_c4_d1" / "benchmarks" / "attempt_trial_check"
    env, _ = _guard_env(tmp_path, D1_BENCHMARK_STORE_ROOT=str(fresh))
    result = subprocess.run(["bash", str(BENCHMARK)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "REFUSING: export D1_BENCHMARK_TRIAL_ID" in result.stderr
    assert not fresh.exists()


# --- job 46175271: spool-copy self-location regression ---------------------
#
# On Moriah, `sbatch path/to/script.sbatch` executes a per-job spool copy
# (/var/spool/slurmd/job<id>/slurm_script), not the checked-out file. Job
# 46175271 failed in 2 seconds because each entry point located its sibling
# rd1_c4_d1_common.sh via `dirname "${BASH_SOURCE[0]}"`, which pointed at the
# empty spool directory rather than the isolated repo clone. The tests below
# reproduce that exact shape -- a copy under a generic "slurm_script" name in
# an unrelated temp directory that does NOT contain rd1_c4_d1_common.sh -- and
# prove the corrected entry points resolve the common file from
# REPO_CLONE_DIR instead, reaching each script's own pre-existing input guard
# rather than a "No such file or directory" / "command not found" failure.

REPO_ROOT_FOR_TESTS = Path(__file__).resolve().parents[1].parent.parent.parent
assert (
    REPO_ROOT_FOR_TESTS / "US_data" / "data_download" / "Disk_volume_estimation" / "scripts" / "rd1_c4_d1" / "rd1_c4_d1_common.sh"
).is_file(), "REPO_ROOT_FOR_TESTS does not resolve to a checkout containing rd1_c4_d1_common.sh"


def _spool_copy(tmp_path: Path, entry_point: Path) -> Path:
    """Copy ``entry_point`` into an unrelated directory under a generic
    Slurm-spool-style name, deliberately without its sibling common file --
    the exact shape of /var/spool/slurmd/job<id>/slurm_script."""
    spool_dir = tmp_path / "spool" / "jobXXXXX"
    spool_dir.mkdir(parents=True)
    spool_script = spool_dir / "slurm_script"
    shutil.copy2(entry_point, spool_script)
    assert not (spool_dir / "rd1_c4_d1_common.sh").exists()
    return spool_script


_SPOOL_REGRESSION_CASES = [
    (BENCHMARK, {"D1_BENCHMARK_TRIAL_ID": "probe_trial", "D1_BENCHMARK_STORE_ROOT": None}),
    (ARRAY, {"D1_ARRAY_CONFIRM": "run-all-24-trials", "SLURM_ARRAY_TASK_ID": "0"}),
    (REDUCE, {}),
]


@pytestmark_bash
@pytest.mark.parametrize("entry_point,extra_env", _SPOOL_REGRESSION_CASES, ids=lambda v: getattr(v, "name", None) or "env")
def test_entry_point_locates_common_script_from_repo_clone_when_run_from_a_spool_copy(tmp_path, entry_point, extra_env):
    spool_script = _spool_copy(tmp_path, entry_point)

    flashnh_base = tmp_path / "flashnh_base"
    env, _ = _guard_env(tmp_path)
    env["SLURM_JOB_ID"] = "46175271"
    env["REPO_CLONE_DIR"] = str(REPO_ROOT_FOR_TESTS)
    if entry_point is BENCHMARK:
        extra_env = dict(extra_env)
        extra_env["D1_BENCHMARK_STORE_ROOT"] = str(
            flashnh_base / "tmp" / "rd1_c4_d1" / "benchmarks" / "spool_regression_attempt"
        )
    env.update(extra_env)

    result = subprocess.run(["bash", str(spool_script)], env=env, capture_output=True, text=True, timeout=30)

    # The historical failure mode from job 46175271, must not recur.
    assert "rd1_c4_d1_common.sh: No such file or directory" not in result.stderr
    assert "command not found" not in result.stderr
    assert "REFUSING: cannot locate rd1_c4_d1_common.sh" not in result.stderr

    # It must instead reach the entry point's own pre-existing guard, proving
    # rd1_c4_d1_common.sh (and rd1_c4_d1_require_inputs, which it defines)
    # was actually sourced from the repo clone.
    assert result.returncode == 2
    assert "REFUSING" in result.stderr

    # No benchmark store, log, scheduler action, Python process, or network
    # contact: every one of these guards fires before mkdir/exec/python.
    assert not (flashnh_base / "tmp" / "rd1_c4_d1").exists()


@pytestmark_bash
@pytest.mark.parametrize("entry_point", ENTRY_POINTS, ids=lambda path: path.name)
def test_entry_point_fails_closed_when_slurm_mode_repo_clone_lacks_the_common_script(tmp_path, entry_point):
    """A Slurm-mode REPO_CLONE_DIR that does not contain the common script
    must refuse with the new explicit message -- it must never silently fall
    back to a common.sh sitting next to the spool copy, which would recreate
    the exact ambiguity job 46175271 exposed."""
    spool_script = _spool_copy(tmp_path, entry_point)

    # A decoy common.sh next to the spool copy: if the corrected bootstrap
    # ever fell back to BASH_SOURCE-sibling resolution under Slurm, this is
    # what it would source instead of failing closed.
    decoy = spool_script.parent / "rd1_c4_d1_common.sh"
    decoy.write_text('echo "DECOY_SOURCED_FROM_SPOOL_SIBLING"\nrd1_c4_d1_require_inputs() { return 0; }\n', encoding="utf-8")

    empty_clone = tmp_path / "empty_repo_clone"
    empty_clone.mkdir()

    env, _ = _guard_env(tmp_path)
    env["SLURM_JOB_ID"] = "46175271"
    env["REPO_CLONE_DIR"] = str(empty_clone)

    result = subprocess.run(["bash", str(spool_script)], env=env, capture_output=True, text=True, timeout=30)

    assert "DECOY_SOURCED_FROM_SPOOL_SIBLING" not in result.stdout
    assert result.returncode != 0
    assert "REFUSING: cannot locate rd1_c4_d1_common.sh" in result.stderr
    assert "resolution=slurm" in result.stderr
    assert str(empty_clone) in result.stderr
    assert "REPO_CLONE_DIR" in result.stderr
