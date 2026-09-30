"""Tests for ``scripts/deploy_rd1_p20_p24_harness.py``: the tracked
deployment script for the RD1 P20-P24 operational harness.

Uses disposable local git repositories under ``tmp_path`` (real ``git``
commits, never the actual project repository) so the frozen-payload
provenance audit and the file-manifest/checksum/receipt behavior can be
exercised against real ``git show``/``git diff`` output, without touching
Moriah, W&B, or this project's own git history -- except for the dedicated
"actual repository history" test near the end, which deliberately runs
against the real Flash-NH repository's already-existing, immutable commits
(never a commit that does not yet exist).

Covers, in order:
  1. ``FROZEN_SCIENTIFIC_COMMIT`` / ``APPROVED_OPERATIONAL_BASE_COMMITS``
     regression guards.
  2. A ``RD1_CHAIN_DIR`` outside the project-local ``.scratch_local/`` is
     refused before any git or file I/O.
  3. A harness commit that changes only allowed operational-harness paths
     relative to the base ("frozen scientific") commit deploys
     successfully: every manifest file (including the two shell/sbatch
     entrypoints) lands on disk with the correct content, a verified
     SHA-256, and the correct executable bit, and ``deploy_receipt.json``
     matches the returned receipt and carries all three distinguished
     commits.
  4. A harness commit that additionally touches a path outside the allowed
     set is refused by the provenance audit -- no file and no receipt are
     written -- and the audit does not rely on any manifest-level
     ``expected_commit`` string, only on real ``git diff`` output.
  5. The project-prefix-aware audit correctly (a) derives an approved
     operational-base commit's allowed paths live from git in a nested
     (monorepo-like) repo layout, and (b) fails closed when an "approved"
     commit or the harness commit touches a path outside the project
     subtree, rather than silently excluding or misattributing it.
  6. A defense-in-depth "decomposition" check: history that looks
     individually approved in each of the two audit stages, but that
     cancels out in the full frozen-to-harness diff (e.g. a doc file
     changed then reverted), fails closed rather than being accepted.
  7. A relocation/deployment-layout test: the deployed
     ``submit_rd1_p20_p24_chain.sh`` invokes the DEPLOYED
     ``rd1_p20_p24_job.sbatch`` (never the repository-tracked path,
     regardless of caller cwd).
  8. A dedicated test against the actual Flash-NH repository's real,
     already-existing history (frozen scientific commit ``dabd2ca...`` and
     the already-approved operational-base commit ``04a487e0...``), proving
     the audit interface works against real history before any harness
     commit exists yet.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import deploy_rd1_p20_p24_harness as deploy_mod  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="requires git on PATH")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


def _write(repo: Path, rel_path: str, content: str) -> None:
    path = repo / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_base_commit(repo: Path) -> str:
    _write(repo, "src/__init__.py", "# pre-existing package marker\n")
    _write(repo, "src/baseline/__init__.py", "# pre-existing baseline package marker\n")
    _write(repo, "scripts/unrelated_science.py", "# unrelated pre-existing scientific script\n")
    # Pre-existing, frozen-scientific-era dependency modules the harness
    # driver imports transitively via
    # src/baseline/sweep_v2_six_axis_wandb_bridge_manifest.py (job-46226308
    # forensic repair) -- written into the BASE commit, never the harness
    # commit, exactly mirroring the real repository: these four modules are
    # unchanged between FROZEN_SCIENTIFIC_COMMIT and the real harness
    # commits, never new operational-harness surface.
    _write(repo, "src/baseline/sweep_v1_campaign.py", "# pre-existing sweep v1 campaign module\n")
    _write(
        repo,
        "src/baseline/sweep_v2_six_axis_campaign.py",
        "# pre-existing sweep v2 six-axis campaign module\n",
    )
    _write(repo, "src/baseline/sweep_v1_launch_manifest.py", "# pre-existing sweep v1 launch manifest module\n")
    _write(
        repo,
        "src/baseline/sweep_v2_six_axis_wandb_bridge_manifest.py",
        "# pre-existing sweep v2 six-axis W&B bridge manifest module\n",
    )
    return _commit(repo, "base scientific commit")


def _add_harness_files(repo: Path) -> None:
    _write(repo, "scripts/rd1_p20_p24_job.py", "# harness driver\n")
    _write(repo, "scripts/rd1_p20_p24_job.sbatch", "#!/bin/bash\n# harness sbatch\n")
    _write(repo, "scripts/submit_rd1_p20_p24_chain.sh", "#!/bin/bash\n# harness submit\n")
    _write(repo, "scripts/deploy_rd1_p20_p24_harness.py", "# this deploy script\n")
    _write(repo, "src/baseline/rd1_p20_p24_env.py", "# env module\n")
    _write(repo, "src/baseline/rd1_p20_p24_registry.py", "# registry module\n")
    _write(repo, "src/baseline/rd1_p20_p24_retry.py", "# retry module\n")
    _write(repo, "tests/test_rd1_p20_p24_env.py", "# env test\n")


def test_frozen_scientific_commit_constant_unchanged():
    assert deploy_mod.FROZEN_SCIENTIFIC_COMMIT == "dabd2ca851bd2b3a03035886cfa50015f2c864b4"


def test_approved_operational_base_commits_constant_unchanged():
    assert deploy_mod.APPROVED_OPERATIONAL_BASE_COMMITS == (
        "04a487e0c2daddf0ae5c700402b6b976fb2b076b",
    )


def test_deploy_manifest_covers_both_shell_sbatch_entrypoints():
    assert "scripts/submit_rd1_p20_p24_chain.sh" in deploy_mod.DEPLOY_MANIFEST
    assert "scripts/rd1_p20_p24_job.sbatch" in deploy_mod.DEPLOY_MANIFEST
    assert deploy_mod.EXECUTABLE_MANIFEST_PATHS == {
        "scripts/submit_rd1_p20_p24_chain.sh",
        "scripts/rd1_p20_p24_job.sbatch",
        "scripts/rd1_p20_p24_job.py",
    }


def test_refuses_chain_dir_outside_project_scratch_local(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    base_commit = _make_base_commit(repo)

    chain_dir = tmp_path / "not_under_scratch_local" / "chain"
    with pytest.raises(deploy_mod.DeploymentRefused, match="must resolve beneath"):
        deploy_mod.deploy(
            repo_root=repo,
            chain_dir=chain_dir,
            harness_commit=base_commit,
            operational_base_commit=base_commit,
            frozen_scientific_commit=base_commit,
            approved_operational_base_commits=(),
        )
    assert not chain_dir.exists()


def test_successful_deploy_writes_verified_files_and_receipt(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    base_commit = _make_base_commit(repo)
    _add_harness_files(repo)
    harness_commit = _commit(repo, "add RD1 P20-P24 operational harness")

    chain_dir = repo / ".scratch_local" / "rd1_p20_p24_chain"
    receipt = deploy_mod.deploy(
        repo_root=repo,
        chain_dir=chain_dir,
        harness_commit=harness_commit,
        operational_base_commit=base_commit,
        frozen_scientific_commit=base_commit,
        approved_operational_base_commits=(),
    )

    assert receipt["frozen_scientific_commit"] == base_commit
    assert receipt["operational_base_commit"] == base_commit
    assert receipt["harness_commit"] == harness_commit
    assert receipt["provenance_audit"]["passed"] is True
    assert receipt["provenance_audit"]["disallowed_paths"] == []
    assert receipt["provenance_audit"]["decomposition_matches"] is True
    assert len(receipt["files"]) == len(deploy_mod.DEPLOY_MANIFEST)

    harness_dir = chain_dir / "harness"
    for rec in receipt["files"]:
        deployed_path = Path(rec["deployed_path"])
        assert deployed_path.exists()
        assert deployed_path == harness_dir / Path(rec["path"])
        assert hashlib.sha256(deployed_path.read_bytes()).hexdigest() == rec["sha256"]
        if rec["path"] in deploy_mod.EXECUTABLE_MANIFEST_PATHS:
            assert rec["executable"] is True
            if os.name != "nt":
                assert deployed_path.stat().st_mode & stat.S_IXUSR
        else:
            assert rec["executable"] is False

    driver_text = (harness_dir / "scripts" / "rd1_p20_p24_job.py").read_text(encoding="utf-8")
    assert driver_text == "# harness driver\n"
    submit_text = (harness_dir / "scripts" / "submit_rd1_p20_p24_chain.sh").read_text(encoding="utf-8")
    assert submit_text == "#!/bin/bash\n# harness submit\n"
    sbatch_text = (harness_dir / "scripts" / "rd1_p20_p24_job.sbatch").read_text(encoding="utf-8")
    assert sbatch_text == "#!/bin/bash\n# harness sbatch\n"

    receipt_path = chain_dir / "deploy_receipt.json"
    assert receipt_path.exists()
    on_disk = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert on_disk == receipt

    # Credential-free: no token/password/secret-shaped keys anywhere.
    flat = json.dumps(on_disk).lower()
    for forbidden in ("password", "token", "secret", "api_key", "apikey"):
        assert forbidden not in flat


def test_disallowed_changed_path_refuses_and_writes_nothing(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    base_commit = _make_base_commit(repo)
    _add_harness_files(repo)
    # Also modify a pre-existing, unrelated scientific file -- exactly the
    # class of change the provenance audit must catch.
    _write(repo, "scripts/unrelated_science.py", "# unrelated pre-existing scientific script (MODIFIED)\n")
    harness_commit = _commit(repo, "add harness AND accidentally touch unrelated science file")

    chain_dir = repo / ".scratch_local" / "rd1_p20_p24_chain"
    with pytest.raises(deploy_mod.DeploymentRefused, match="provenance audit failed"):
        deploy_mod.deploy(
            repo_root=repo,
            chain_dir=chain_dir,
            harness_commit=harness_commit,
            operational_base_commit=base_commit,
            frozen_scientific_commit=base_commit,
            approved_operational_base_commits=(),
        )

    assert not (chain_dir / "harness").exists()
    assert not (chain_dir / "deploy_receipt.json").exists()


def test_provenance_audit_is_independent_of_manifest_expected_commit_string(tmp_path):
    """The audit must be computed from real git history, not trusted from
    any manifest-level ``expected_commit`` value an operator could type in.
    Passing a harness commit that is IDENTICAL to the base commit (i.e. adds
    nothing) still passes -- there is nothing disallowed to have changed --
    proving the check is a live git diff, not a string comparison against a
    manifest field that was never inspected here."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    base_commit = _make_base_commit(repo)

    audit = deploy_mod.provenance_audit(
        repo,
        harness_commit=base_commit,
        operational_base_commit=base_commit,
        frozen_scientific_commit=base_commit,
        approved_operational_base_commits=(),
    )
    assert audit["changed_paths_frozen_to_operational_base"] == []
    assert audit["changed_paths_operational_base_to_harness"] == []
    assert audit["changed_paths_frozen_to_harness"] == []
    assert audit["decomposition_matches"] is True
    assert audit["passed"] is True


# --- Monorepo path-prefix awareness (task 2) --------------------------------
#
# The real Flash-NH repository's git top-level is three directories above
# this project's root, so a plain ``git diff --name-only`` returns paths
# relative to that top-level (e.g.
# "US_data/data_download/Disk_volume_estimation/AGENTS.md"), NOT relative to
# the project root the way DEPLOY_MANIFEST/ALLOWED_HARNESS_CHANGED_PATHS are
# written. These tests reproduce that nesting with a disposable repo so the
# prefix-stripping logic (_project_prefix / _diff_project_relative) is
# exercised the same way it will be against the real repository.


def _init_nested_repo(tmp_path: Path) -> tuple[Path, Path]:
    git_root = tmp_path / "monorepo"
    _init_repo(git_root)
    project_root = git_root / "some" / "nested" / "project"
    project_root.mkdir(parents=True)
    return git_root, project_root


def test_approved_operational_base_commit_paths_derived_correctly_in_nested_repo(tmp_path):
    git_root, project_root = _init_nested_repo(tmp_path)

    frozen_rel = "some/nested/project/src/__init__.py"
    _write(git_root, frozen_rel, "# pre-existing package marker\n")
    _write(git_root, "some/nested/project/scripts/unrelated_science.py", "# unrelated\n")
    frozen_commit = _commit(git_root, "frozen scientific commit (nested)")

    _write(git_root, "some/nested/project/CLAUDE.md", "hardened boundary text\n")
    doc_commit = _commit(git_root, "approved doc-hardening commit (nested)")

    _add_harness_files(project_root)
    harness_commit = _commit(git_root, "add RD1 P20-P24 operational harness (nested)")

    audit = deploy_mod.provenance_audit(
        project_root,
        harness_commit=harness_commit,
        operational_base_commit=doc_commit,
        frozen_scientific_commit=frozen_commit,
        approved_operational_base_commits=(doc_commit,),
    )
    assert audit["changed_paths_frozen_to_operational_base"] == ["CLAUDE.md"]
    assert audit["passed"] is True
    assert audit["decomposition_matches"] is True


def test_change_outside_project_subtree_fails_closed_in_nested_repo(tmp_path):
    git_root, project_root = _init_nested_repo(tmp_path)

    _write(git_root, "some/nested/project/src/__init__.py", "# pre-existing package marker\n")
    _write(git_root, "other_project/unrelated_file.txt", "v1\n")
    frozen_commit = _commit(git_root, "frozen scientific commit (nested, with sibling project)")

    # An "approved" commit that actually touches a file OUTSIDE this
    # project's subtree -- the audit must refuse to bless this rather than
    # silently ignore or misattribute the change.
    _write(git_root, "other_project/unrelated_file.txt", "v2 (outside project)\n")
    outside_commit = _commit(git_root, "touches only a sibling project's file")

    with pytest.raises(deploy_mod.DeploymentRefused, match="outside this project's subtree"):
        deploy_mod.provenance_audit(
            project_root,
            harness_commit=outside_commit,
            operational_base_commit=outside_commit,
            frozen_scientific_commit=frozen_commit,
            approved_operational_base_commits=(outside_commit,),
        )


# --- Decomposition consistency (defense in depth) ---------------------------


def test_decomposition_mismatch_fails_closed_when_change_cancels_out(tmp_path):
    """A doc file changed between frozen and operational_base (individually
    approved via the doc commit), then reverted between operational_base and
    harness (individually looking like a no-op net change, but NOT itself an
    approved harness path) must still fail closed: both because the reverting
    change is not in ALLOWED_HARNESS_CHANGED_PATHS, and because the full
    frozen->harness diff for that file disagrees with the stage-by-stage
    decomposition."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write(repo, "CLAUDE.md", "original content\n")
    frozen_commit = _make_base_commit(repo)

    _write(repo, "CLAUDE.md", "approved doc change\n")
    doc_commit = _commit(repo, "approved doc-hardening commit")

    _add_harness_files(repo)
    _write(repo, "CLAUDE.md", "original content\n")  # reverted on top of the doc commit
    harness_commit = _commit(repo, "add harness AND revert CLAUDE.md back to original")

    audit = deploy_mod.provenance_audit(
        repo,
        harness_commit=harness_commit,
        operational_base_commit=doc_commit,
        frozen_scientific_commit=frozen_commit,
        approved_operational_base_commits=(doc_commit,),
    )
    assert "CLAUDE.md" in audit["changed_paths_operational_base_to_harness"]
    assert "CLAUDE.md" in audit["disallowed_paths"]
    assert audit["decomposition_matches"] is False
    assert audit["passed"] is False


# --- Relocation / deployment-layout (task 1) --------------------------------


def _bash_abspath(dir_str: str, filename: str) -> str:
    """Resolves ``dir_str/filename`` to bash's own absolute path notion via
    ``cd "<dir>" && pwd`` -- the exact same mechanism
    ``submit_rd1_p20_p24_chain.sh`` uses for its own self-location
    (``_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"``). Using
    any other path-normalization (e.g. Python's ``Path.resolve()``, or
    bash's ``realpath`` on a raw Windows-style string) can disagree with
    this on Git Bash, since some Git Bash installs mount the Windows temp
    directory at ``/tmp`` -- a mapping ``cd``/``pwd`` follow but a literal
    ``realpath "C:\\Users\\...\\Temp\\..."`` string does not."""
    proc = subprocess.run(["bash", "-c", f'cd "{dir_str}" && pwd'], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return f"{proc.stdout.strip()}/{filename}"


def _bash_git_dir() -> str:
    """Resolves the directory bash's own ``command -v git`` would find, as a
    POSIX-style path bash can consume directly in ``$PATH`` -- unlike
    ``shutil.which("git")`` (a Windows-style ``C:\\...`` path from this
    Python process's own PATH), which collides with ``$PATH``'s ``:``
    separator when spliced into a bash-only env (the drive-letter colon
    splits it into two bogus entries). Run with the inherited environment
    (never the minimal per-test allowlist), matching ``_bash_abspath``."""
    proc = subprocess.run(
        ["bash", "-c", 'dirname "$(command -v git)"'], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _write_stub_sbatch(bin_dir: Path, log_path: Path) -> None:
    stub = bin_dir / "sbatch"
    stub.write_text(
        "#!/bin/bash\n"
        f"echo \"$*\" >> \"{log_path.as_posix()}\"\n"
        "echo $((1000 + RANDOM % 1000))\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)


@pytest.mark.skipif(shutil.which("bash") is None, reason="requires a bash interpreter on PATH")
def test_deployed_submitter_invokes_deployed_job_sbatch_not_repo_path(tmp_path):
    """Proves the complete path resolution required by task 1: once the
    harness (including the two shell/sbatch entrypoints) is deployed, the
    DEPLOYED ``submit_rd1_p20_p24_chain.sh`` must submit the DEPLOYED
    ``rd1_p20_p24_job.sbatch`` -- never falling back to caller cwd or the
    repository-tracked ``scripts/`` path. This must fail if the submitter or
    job script would resolve to the repository script path instead of the
    deployed harness path."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    base_commit = _make_base_commit(repo)
    _add_harness_files(repo)
    harness_commit = _commit(repo, "add RD1 P20-P24 operational harness")

    chain_dir = repo / ".scratch_local" / "rd1_p20_p24_chain"
    receipt = deploy_mod.deploy(
        repo_root=repo,
        chain_dir=chain_dir,
        harness_commit=harness_commit,
        operational_base_commit=base_commit,
        frozen_scientific_commit=base_commit,
        approved_operational_base_commits=(),
    )
    harness_dir = chain_dir / "harness"
    deployed_submitter = harness_dir / "scripts" / "submit_rd1_p20_p24_chain.sh"
    deployed_job_sbatch = harness_dir / "scripts" / "rd1_p20_p24_job.sbatch"
    assert deployed_submitter.exists()
    assert deployed_job_sbatch.exists()

    # Overwrite the deployed submitter with a minimal real copy of the
    # self-location logic under test (rather than the disposable-repo stub
    # content written by _add_harness_files), so this test exercises the
    # actual resolution mechanism, not a placeholder string.
    real_submitter_src = REPO_ROOT / "scripts" / "submit_rd1_p20_p24_chain.sh"
    deployed_submitter.write_bytes(real_submitter_src.read_bytes())
    deployed_submitter.chmod(deployed_submitter.stat().st_mode | stat.S_IEXEC)
    real_job_sbatch_src = REPO_ROOT / "scripts" / "rd1_p20_p24_job.sbatch"
    deployed_job_sbatch.write_bytes(real_job_sbatch_src.read_bytes())

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "sbatch_invocations.log"
    _write_stub_sbatch(bin_dir, log_path)

    args = [
        "--registry-path", str(chain_dir / "registry.json"),
        "--chain-dir", str(chain_dir),
        "--wandb-sweep-id", "disposable-test-sweep",
        # The submitter's --payload-repository-path check requires this to be
        # the payload checkout's real HEAD (harness_commit below) -- the
        # previous literal FROZEN_SCIENTIFIC_COMMIT hash was never a real
        # commit of that checkout and predates --payload-repository-path
        # existing at all.
        "--expected-commit", harness_commit,
        "--execution-generation", "1",
        "--output-root-base", str(chain_dir / "outputs"),
        "--p20-pinned-manifest-path", str(chain_dir / "p20_manifest.json"),
        "--p20-pinned-manifest-sha256", "0" * 64,
        # The disposable repo itself is a clean checkout pinned exactly at
        # harness_commit, so it doubles as the required payload checkout.
        "--payload-repository-path", str(repo),
    ]
    # The submitter's --payload-repository-path check now shells out to git
    # itself (a requirement that postdates this env allowlist), so git's own
    # directory must be reachable too -- resolved via bash's own notion of
    # where git lives (POSIX-style), since the module-level skipif above
    # already guarantees git is on PATH here.
    git_dir = _bash_git_dir()
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin:{git_dir}", "HOME": str(tmp_path), "REPO_WORKDIR": str(repo)}
    # Deliberately run from an unrelated cwd (NOT the harness dir, NOT the
    # repo) to prove resolution does not depend on caller cwd either.
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()
    proc = subprocess.run(
        ["bash", str(deployed_submitter), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(unrelated_cwd),
    )
    assert proc.returncode == 0, proc.stderr
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 5

    repo_job_sbatch = REPO_ROOT / "scripts" / "rd1_p20_p24_job.sbatch"
    deployed_real = _bash_abspath(str(deployed_job_sbatch.parent), deployed_job_sbatch.name)
    repo_real = _bash_abspath(str(repo_job_sbatch.parent), repo_job_sbatch.name)
    assert deployed_real != repo_real
    for line in invocations:
        args_tokens = line.split()
        job_script_arg = args_tokens[-1]
        assert job_script_arg == deployed_real
        assert job_script_arg != repo_real

    del receipt  # only used to trigger + sanity-check the deployment itself


# --- Actual Flash-NH repository history (task 2) ----------------------------


def test_actual_flashnh_repository_history_audit_interface():
    """The provenance audit must be designed to run against the actual
    Flash-NH repository history, not only a disposable test repo. This test
    exercises it against the real REPO_ROOT and the real, already-existing
    commits ``dabd2ca...`` (frozen scientific) and ``04a487e0...``
    (already-approved operational-base doc hardening, currently HEAD) --
    before any harness commit exists. ``harness_commit`` is set to the
    current HEAD too, so stage B (operational_base -> harness) is trivially
    empty and this test needs no harness commit to exist yet."""
    repo_root = deploy_mod.REPO_ROOT
    head = _git(repo_root, "rev-parse", "HEAD")

    audit = deploy_mod.provenance_audit(
        repo_root,
        harness_commit=head,
        operational_base_commit=head,
        frozen_scientific_commit=deploy_mod.FROZEN_SCIENTIFIC_COMMIT,
    )

    assert audit["frozen_scientific_commit"] == deploy_mod.FROZEN_SCIENTIFIC_COMMIT
    assert audit["operational_base_commit"] == head
    assert audit["harness_commit"] == head
    assert audit["changed_paths_operational_base_to_harness"] == []

    if head == "04a487e0c2daddf0ae5c700402b6b976fb2b076b":
        assert set(audit["changed_paths_frozen_to_operational_base"]) == {
            "AGENTS.md",
            "CLAUDE.md",
            "docs/repo_policy.md",
        }
        assert audit["decomposition_matches"] is True
        assert audit["passed"] is True
    else:
        # HEAD has moved past the already-approved doc-hardening commit this
        # constant names -- e.g. because this very task's harness commit (or
        # something else) has since landed. The audit must still run without
        # raising; whether it passes now legitimately depends on what
        # actually changed, which this test does not assume.
        pass


# --- _git_show_bytes monorepo-prefix regression (corrective patch) ---------
#
# scripts/deploy_rd1_p20_p24_harness.py's _git_show_bytes() used to pass a
# bare project-relative path straight to ``git show <commit>:<path>``. A
# bare (non "./"-prefixed) <path> in that form resolves against the git
# TOP LEVEL, not against ``cwd`` -- so in the real Flash-NH monorepo (a
# non-empty _project_prefix) it silently looked in the wrong place and
# failed to find e.g. "scripts/submit_rd1_p20_p24_chain.sh". These tests
# cover both the non-empty-prefix real-repository case and the disposable
# empty-prefix case, without relying only on mocks.


def test_git_show_bytes_empty_prefix_disposable_repo(tmp_path):
    """When the project root IS the git top level (empty _project_prefix,
    the layout every disposable repo above uses), _git_show_bytes must still
    return the exact tracked content for a project-relative path."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _write(repo, "scripts/rd1_p20_p24_job.py", "# harness driver v1\n")
    commit = _commit(repo, "add harness driver")

    assert deploy_mod._project_prefix(repo) == ""
    blob = deploy_mod._git_show_bytes(repo, commit, "scripts/rd1_p20_p24_job.py")
    assert blob == b"# harness driver v1\n"


def test_git_show_bytes_matches_real_repo_tracked_content_at_d5de66e():
    """Regression for the monorepo path bug: against the ACTUAL Flash-NH
    repository (non-empty _project_prefix, e.g.
    "US_data/data_download/Disk_volume_estimation/"), _git_show_bytes for a
    project-relative DEPLOY_MANIFEST path at the real, already-existing
    harness commit ``d5de66e`` must return exactly that file's tracked
    content. Uses ``src/baseline/rd1_p20_p24_env.py`` (untouched by this
    corrective patch) so the expected bytes can be read straight off the
    working tree, which is checked out at that same commit."""
    repo_root = deploy_mod.REPO_ROOT
    harness_commit = "d5de66eb97defad2f454a8b5dc6152b2ad807473"
    prefix = deploy_mod._project_prefix(repo_root)
    assert prefix, (
        "expected a non-empty monorepo prefix here -- otherwise this test "
        "would not actually exercise the bug this patch fixes"
    )

    rel_path = "src/baseline/rd1_p20_p24_env.py"
    blob = deploy_mod._git_show_bytes(repo_root, harness_commit, rel_path)
    on_disk = (repo_root / rel_path).read_bytes()
    assert blob == on_disk
    assert len(blob) > 0


# --- Real local monorepo end-to-end deployment (corrective patch) ----------


def test_real_monorepo_end_to_end_deployment_at_d5de66e():
    """A local, non-mocked end-to-end deployment against the actual
    Flash-NH monorepo using the real, already-existing harness commit
    ``d5de66e``. Writes only under a fresh, uniquely-named directory inside
    this project's own gitignored ``.scratch_local/`` (never outside the
    project root -- see CLAUDE.md's project-local artifact boundary), and
    removes it afterwards. No W&B, Slurm, registry, or manifest action."""
    repo_root = deploy_mod.REPO_ROOT
    harness_commit = "d5de66eb97defad2f454a8b5dc6152b2ad807473"

    chain_dir = repo_root / ".scratch_local" / f"test_deploy_real_monorepo_{harness_commit[:12]}"
    if chain_dir.exists():
        shutil.rmtree(chain_dir)
    try:
        receipt = deploy_mod.deploy(
            repo_root=repo_root,
            chain_dir=chain_dir,
            harness_commit=harness_commit,
        )

        assert receipt["frozen_scientific_commit"] == deploy_mod.FROZEN_SCIENTIFIC_COMMIT
        assert receipt["harness_commit"] == harness_commit
        assert receipt["provenance_audit"]["passed"] is True
        assert receipt["provenance_audit"]["decomposition_matches"] is True
        assert len(receipt["files"]) == len(deploy_mod.DEPLOY_MANIFEST) == 12

        chain_dir_real = chain_dir.resolve()
        for rec in receipt["files"]:
            deployed_path = Path(rec["deployed_path"]).resolve()
            assert deployed_path.exists()
            assert deployed_path.is_file()
            assert hashlib.sha256(deployed_path.read_bytes()).hexdigest() == rec["sha256"]
            # Raises ValueError (failing the test) if deployed_path is not
            # beneath chain_dir_real.
            deployed_path.relative_to(chain_dir_real)

        receipt_path = chain_dir / "deploy_receipt.json"
        assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    finally:
        if chain_dir.exists():
            shutil.rmtree(chain_dir)


# --- Isolated deployed-bundle import preflight (job-46226308 forensic repair) ---
#
# Direct regression coverage for the defect that failed Slurm job 46226308 in
# ~3 seconds: ``scripts/rd1_p20_p24_job.py`` imports
# ``src.baseline.sweep_v2_six_axis_wandb_bridge_manifest``, but that module
# (and its own transitive ``src.baseline.sweep_v1_launch_manifest`` /
# ``sweep_v2_six_axis_campaign`` / ``sweep_v1_campaign`` dependency closure)
# was missing from DEPLOY_MANIFEST -- so the deployed bundle's checksum
# verification (which only proves deployed bytes match tracked bytes for
# whatever IS in the manifest) passed, but the job died on
# ``ModuleNotFoundError`` the moment it actually tried to run. These tests
# use the REAL, current Flash-NH repository HEAD and the REAL driver file --
# never a disposable stub -- so they actually exercise the real local
# monorepo deployment path this bug lived in.


def test_real_monorepo_end_to_end_deployment_at_current_head_import_preflight_passes():
    """A local, non-mocked end-to-end deployment against the actual, current
    Flash-NH monorepo HEAD, whose real ``scripts/rd1_p20_p24_job.py`` really
    does import ``src.baseline.sweep_v2_six_axis_wandb_bridge_manifest`` (not
    a stub). Proves the isolated deployed-bundle import preflight actually
    passes end to end against the real driver and its real transitive
    dependency closure now that DEPLOY_MANIFEST includes all four of them."""
    repo_root = deploy_mod.REPO_ROOT
    head = _git(repo_root, "rev-parse", "HEAD")

    chain_dir = repo_root / ".scratch_local" / f"test_deploy_import_preflight_{head[:12]}"
    if chain_dir.exists():
        shutil.rmtree(chain_dir)
    try:
        receipt = deploy_mod.deploy(
            repo_root=repo_root,
            chain_dir=chain_dir,
            harness_commit=head,
            # Current HEAD normally sits many ordinary harness-iteration
            # commits ahead of any single approved base -- so, exactly like
            # test_real_monorepo_end_to_end_deployment_at_d5de66e's sibling
            # test in tests/test_rd1_p20_p24_job_sbatch.py already does, the
            # provenance audit's base must be pinned explicitly to the one
            # already-approved operational-base commit rather than left to
            # default to ``head^`` (which is just the previous ordinary
            # iteration commit, not an approved base, and would make the
            # audit refuse for reasons unrelated to this regression).
            operational_base_commit="04a487e0c2daddf0ae5c700402b6b976fb2b076b",
        )

        assert receipt["import_preflight"]["returncode"] == 0
        output_path = Path(receipt["import_preflight"]["output_path"])
        assert output_path.is_file()
        assert "returncode=0" in output_path.read_text(encoding="utf-8")

        # Prove the deployed driver on disk really is the real one (contains
        # the actual import this whole regression is about), not a no-op
        # stand-in that would make a passing preflight meaningless.
        deployed_driver = Path(receipt["files"][2]["deployed_path"])
        assert deployed_driver.name == "rd1_p20_p24_job.py"
        assert "sweep_v2_six_axis_wandb_bridge_manifest" in deployed_driver.read_text(encoding="utf-8")

        assert (chain_dir / "deploy_receipt.json").exists()
    finally:
        if chain_dir.exists():
            shutil.rmtree(chain_dir)


def test_import_preflight_fails_closed_when_manifest_omits_a_required_transitive_dependency():
    """Directly reproduces the job-46226308 defect class: deploy the real,
    current HEAD harness commit but with a DEPLOY_MANIFEST that omits
    ``src/baseline/sweep_v2_six_axis_wandb_bridge_manifest.py`` -- exactly
    what the pre-repair DEPLOY_MANIFEST actually omitted. The isolated import
    preflight must fail closed (``ModuleNotFoundError``, non-zero return
    code) and ``deploy_receipt.json`` must NOT be written, so nothing
    downstream -- the sbatch script's own invocation-time receipt check, or a
    human operator -- can mistake this deployment for valid."""
    repo_root = deploy_mod.REPO_ROOT
    head = _git(repo_root, "rev-parse", "HEAD")

    omitted = "src/baseline/sweep_v2_six_axis_wandb_bridge_manifest.py"
    broken_manifest = tuple(p for p in deploy_mod.DEPLOY_MANIFEST if p != omitted)
    assert len(broken_manifest) == len(deploy_mod.DEPLOY_MANIFEST) - 1

    chain_dir = repo_root / ".scratch_local" / f"test_deploy_broken_manifest_{head[:12]}"
    if chain_dir.exists():
        shutil.rmtree(chain_dir)
    try:
        with pytest.raises(
            deploy_mod.DeploymentRefused,
            match="isolated deployed-bundle import preflight failed",
        ):
            deploy_mod.deploy(
                repo_root=repo_root,
                chain_dir=chain_dir,
                harness_commit=head,
                operational_base_commit="04a487e0c2daddf0ae5c700402b6b976fb2b076b",
                manifest=broken_manifest,
            )

        assert not (chain_dir / "deploy_receipt.json").exists()
        assert not (chain_dir / "harness" / omitted).exists()

        output_path = chain_dir / "import_preflight" / "import_preflight_output.txt"
        assert output_path.is_file()
        assert "ModuleNotFoundError" in output_path.read_text(encoding="utf-8")
    finally:
        if chain_dir.exists():
            shutil.rmtree(chain_dir)
