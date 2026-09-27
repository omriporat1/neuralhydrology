#!/usr/bin/env python
"""Deploys the tracked RD1 P20-P24 operational harness into a chain-run's
project-local ``.scratch_local`` directory, and records a credential-free
receipt.

Why this exists: the scientific manifest/expected-commit identity for the
RD1 P20-P24 chain stays frozen at a single commit
(``FROZEN_SCIENTIFIC_COMMIT`` below). The repaired operational harness
(``scripts/rd1_p20_p24_job.py`` and its ``src/baseline/rd1_p20_p24_*``
modules) is tracked as a *separate* commit on top of that frozen point. This
script is the one place that bridges the two: it deploys the harness commit's
files into ``${chain_dir}/harness/`` (never reading them from the tracked
``REPO_WORKDIR`` checkout at job-run time -- see ``scripts/
rd1_p20_p24_job.sbatch``), and it independently proves -- via ``git diff``,
not via the manifest's own format-only ``expected_commit`` field -- that the
harness commit changed nothing but the new operational-harness surface.

Usage (never run automatically by any job -- see scripts/
submit_rd1_p20_p24_chain.sh's own comment: only a human, from a clean login
shell, deploys the harness before submitting the chain)::

    python scripts/deploy_rd1_p20_p24_harness.py \\
        --chain-dir /path/to/project/.scratch_local/rd1_p20_p24_<timestamp> \\
        --harness-commit <sha of the tracked operational-harness commit>

Three distinct commits matter to the provenance audit below, and are never
conflated:

- ``frozen_scientific_commit`` -- the immutable scientific execution
  identity (``FROZEN_SCIENTIFIC_COMMIT``). Never a CLI flag.
- ``operational_base_commit`` -- the commit the harness commit is actually
  built on top of. This may be later than the frozen scientific commit
  because of legitimate, already-approved, non-scientific commits (e.g. a
  project-policy documentation hardening) landing in between. Defaults to
  ``<harness_commit>^`` (the harness commit's own first parent) when not
  given explicitly.
- ``harness_commit`` -- the commit that adds the new operational-harness
  source/deployment/test files.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# The scientific execution identity this chain's manifests are pinned to.
# Frozen: not a CLI flag, so nothing at deployment time can silently swap it
# for a different scientific commit.
FROZEN_SCIENTIFIC_COMMIT = "dabd2ca851bd2b3a03035886cfa50015f2c864b4"

# Commits already reviewed and approved as legitimate, non-scientific
# changes that may sit between FROZEN_SCIENTIFIC_COMMIT and a harness
# commit's operational_base_commit. The paths each one is allowed to have
# touched are derived LIVE from git (see _approved_operational_base_paths),
# never hardcoded here -- so this list can only ever name a commit, not
# guess at its content.
APPROVED_OPERATIONAL_BASE_COMMITS: tuple[str, ...] = (
    # "Harden Flash-NH project-local artifact boundary" -- AGENTS.md,
    # CLAUDE.md, docs/repo_policy.md only (already pushed, current HEAD).
    "04a487e0c2daddf0ae5c700402b6b976fb2b076b",
)

# The exact files that make up the deployed, runtime operational harness
# (the Python driver plus the ``src/baseline`` package path it imports via
# its own ``sys.path.insert(0, parents[1])`` -- see rd1_p20_p24_job.py --
# plus the shell/sbatch entrypoints a production launch actually invokes).
# Paths are POSIX-style, relative to the project root, and this is the
# single source of truth for what gets copied into ``${chain_dir}/harness/``.
DEPLOY_MANIFEST: tuple[str, ...] = (
    "scripts/submit_rd1_p20_p24_chain.sh",
    "scripts/rd1_p20_p24_job.sbatch",
    "scripts/rd1_p20_p24_job.py",
    "src/__init__.py",
    "src/baseline/__init__.py",
    "src/baseline/rd1_p20_p24_env.py",
    "src/baseline/rd1_p20_p24_registry.py",
    "src/baseline/rd1_p20_p24_retry.py",
)

# Manifest paths that must be deployed executable (the two shell/sbatch
# entrypoints a human or Slurm actually invokes directly, plus the Python
# driver which scripts/rd1_p20_p24_job.sbatch also execs directly).
EXECUTABLE_MANIFEST_PATHS: frozenset[str] = frozenset(
    {
        "scripts/submit_rd1_p20_p24_chain.sh",
        "scripts/rd1_p20_p24_job.sbatch",
        "scripts/rd1_p20_p24_job.py",
    }
)

# The only paths a diff between operational_base_commit and a candidate
# harness commit may touch. This is the independent provenance check the
# manifest's format-only ``_validate_expected_commit`` cannot provide: it
# looks at the actual committed tree, not a string an operator typed in.
# ``src/__init__.py`` and ``src/baseline/__init__.py`` are deliberately
# excluded here: they are pre-existing, shared package markers the harness
# commit must not need to touch, so a diff touching them is still refused.
ALLOWED_HARNESS_CHANGED_PATHS: frozenset[str] = frozenset(
    {
        "scripts/rd1_p20_p24_job.py",
        "scripts/rd1_p20_p24_job.sbatch",
        "scripts/submit_rd1_p20_p24_chain.sh",
        "scripts/deploy_rd1_p20_p24_harness.py",
        "src/baseline/rd1_p20_p24_env.py",
        "src/baseline/rd1_p20_p24_registry.py",
        "src/baseline/rd1_p20_p24_retry.py",
        "tests/test_rd1_p20_p24_env.py",
        "tests/test_rd1_p20_p24_registry.py",
        "tests/test_rd1_p20_p24_retry.py",
        "tests/test_rd1_p20_p24_job.py",
        "tests/test_rd1_p20_p24_job_sbatch.py",
        "tests/test_submit_rd1_p20_p24_chain.py",
        "tests/test_rd1_p20_p24_agent_retry_integration.py",
        "tests/test_deploy_rd1_p20_p24_harness.py",
    }
)

# Explicitly approved operational documentation paths. Empty until the user
# approves a specific doc; edited here, never widened by a runtime flag.
APPROVED_OPERATIONAL_DOC_PATHS: frozenset[str] = frozenset()


class DeploymentRefused(Exception):
    """Raised for any fail-closed refusal before deployment proceeds."""


def _run_git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DeploymentRefused(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def _project_prefix(repo_root: Path) -> str:
    """The project root's path relative to the actual git top-level, e.g.
    ``"US_data/data_download/Disk_volume_estimation/"`` in the real Flash-NH
    monorepo, or ``""`` when repo_root IS the git top-level (as every
    disposable single-level test repo below is). Needed because a plain
    ``git diff --name-only`` returns paths relative to the git top-level, not
    to ``repo_root`` -- without stripping this prefix, every path returned
    for the real repository would silently fail to match DEPLOY_MANIFEST /
    ALLOWED_HARNESS_CHANGED_PATHS, which are written relative to the project
    root."""
    return _run_git(repo_root, "rev-parse", "--show-prefix").strip()


def _diff_project_relative(repo_root: Path, commit_a: str, commit_b: str) -> tuple[list[str], list[str]]:
    """Returns ``(project_relative_paths, outside_project_paths)`` changed
    between ``commit_a`` and ``commit_b``. Uses the FULL, unscoped diff (no
    ``-- <pathspec>`` and no ``--relative``, which would silently exclude
    changes outside the project subtree) so a change anywhere in the
    monorepo is surfaced -- one outside this project's subtree cannot be
    expressed as a project-relative path and is always treated as
    disallowed, never silently dropped."""
    prefix = _project_prefix(repo_root)
    diff_output = _run_git(repo_root, "diff", "--name-only", commit_a, commit_b)
    raw_paths = [line for line in diff_output.splitlines() if line]
    project_relative: list[str] = []
    outside_project: list[str] = []
    for p in raw_paths:
        if not prefix:
            project_relative.append(p)
        elif p.startswith(prefix):
            project_relative.append(p[len(prefix):])
        else:
            outside_project.append(p)
    return sorted(project_relative), sorted(outside_project)


def _approved_operational_base_paths(
    repo_root: Path, approved_commits: Sequence[str]
) -> frozenset[str]:
    """Derives, LIVE from git, the path set an already-approved operational-
    base commit is allowed to have touched: the union of what each commit in
    ``approved_commits`` actually changed relative to its own first parent.
    Never hardcoded, so this cannot drift from what those commits really
    contain. A commit that touches a path outside this project's subtree is
    refused outright -- this audit has no way to bless that."""
    allowed: set[str] = set()
    for commit in approved_commits:
        parent = _run_git(repo_root, "rev-parse", f"{commit}^").strip()
        project_relative, outside_project = _diff_project_relative(repo_root, parent, commit)
        if outside_project:
            raise DeploymentRefused(
                f"approved operational-base commit {commit} changes path(s) "
                f"outside this project's subtree, cannot be blessed: {outside_project}"
            )
        allowed.update(project_relative)
    return frozenset(allowed)


def _git_show_bytes(repo_root: Path, commit: str, rel_path: str) -> bytes:
    proc = subprocess.run(
        ["git", "show", f"{commit}:{rel_path}"],
        cwd=str(repo_root),
        capture_output=True,
    )
    if proc.returncode != 0:
        raise DeploymentRefused(
            f"git show {commit}:{rel_path} failed: {proc.stderr.decode('utf-8', 'replace').strip()}"
        )
    return proc.stdout


def assert_chain_dir_within_project_scratch_local(repo_root: Path, chain_dir: Path) -> Path:
    """Fail-closed project-local boundary check (CLAUDE.md, non-negotiable):
    mirrors the identical check in ``scripts/rd1_p20_p24_job.sbatch``. Runs
    before any directory is created and before any git or file I/O below.
    Returns the resolved, normalized chain-dir path on success."""
    repo_root_real = repo_root.resolve()
    chain_dir_real = chain_dir.resolve()
    scratch_local_root = repo_root_real / ".scratch_local"
    try:
        chain_dir_real.relative_to(scratch_local_root)
    except ValueError as exc:
        raise DeploymentRefused(
            f"RD1_CHAIN_DIR must resolve beneath {scratch_local_root}/ (got {chain_dir_real})"
        ) from exc
    return chain_dir_real


def provenance_audit(
    repo_root: Path,
    *,
    harness_commit: str,
    operational_base_commit: str | None = None,
    frozen_scientific_commit: str = FROZEN_SCIENTIFIC_COMMIT,
    approved_operational_base_commits: Sequence[str] = APPROVED_OPERATIONAL_BASE_COMMITS,
    allowed_harness_paths: frozenset[str] = ALLOWED_HARNESS_CHANGED_PATHS,
    approved_doc_paths: frozenset[str] = APPROVED_OPERATIONAL_DOC_PATHS,
) -> dict:
    """Independently proves, from actual git history run against the real
    repository (never from the manifest's own format-only ``expected_commit``
    field, and never assuming a disposable single-level test repo), that
    everything that changed between the immutable ``frozen_scientific_commit``
    and the candidate ``harness_commit`` is either an already-approved,
    non-scientific operational-base change, or part of the new operational-
    harness surface approved in this task.

    Two independently-checked stages, both fully recorded:

      stage A: frozen_scientific_commit -> operational_base_commit.
               Must be a subset of what ``approved_operational_base_commits``
               actually changed (derived live from git, never hardcoded).
      stage B: operational_base_commit -> harness_commit.
               Must be a subset of the new-harness allowlist.

    A third, independent cross-check recomputes the FULL
    frozen_scientific_commit -> harness_commit diff directly and requires it
    to equal the union of stage A's and stage B's changed paths. Any
    mismatch -- e.g. non-linear or rebased history where a path changes in
    one stage and is reverted in the other, so it cancels out of the full
    diff while still appearing "approved" in each stage individually -- is
    treated as a failure too, not silently accepted.
    """
    if operational_base_commit is None:
        operational_base_commit = _run_git(repo_root, "rev-parse", f"{harness_commit}^").strip()

    stage_a_paths, stage_a_outside = _diff_project_relative(
        repo_root, frozen_scientific_commit, operational_base_commit
    )
    allowed_base_paths = _approved_operational_base_paths(repo_root, approved_operational_base_commits)
    stage_a_disallowed = sorted(
        set(p for p in stage_a_paths if p not in allowed_base_paths) | set(stage_a_outside)
    )

    stage_b_paths, stage_b_outside = _diff_project_relative(
        repo_root, operational_base_commit, harness_commit
    )
    allowed_harness = allowed_harness_paths | approved_doc_paths
    stage_b_disallowed = sorted(
        set(p for p in stage_b_paths if p not in allowed_harness) | set(stage_b_outside)
    )

    full_paths, full_outside = _diff_project_relative(
        repo_root, frozen_scientific_commit, harness_commit
    )
    decomposition_expected = sorted(set(stage_a_paths) | set(stage_b_paths))
    decomposition_matches = (
        full_paths == decomposition_expected
        and not full_outside
        and not stage_a_outside
        and not stage_b_outside
    )

    disallowed = sorted(set(stage_a_disallowed) | set(stage_b_disallowed) | set(full_outside))
    passed = not disallowed and decomposition_matches

    return {
        "frozen_scientific_commit": frozen_scientific_commit,
        "operational_base_commit": operational_base_commit,
        "harness_commit": harness_commit,
        "changed_paths_frozen_to_operational_base": stage_a_paths,
        "changed_paths_operational_base_to_harness": stage_b_paths,
        "changed_paths_frozen_to_harness": full_paths,
        "decomposition_matches": decomposition_matches,
        "disallowed_paths": disallowed,
        "passed": passed,
    }


def deploy_harness_files(
    repo_root: Path,
    chain_dir_real: Path,
    harness_commit: str,
    manifest: Sequence[str] = DEPLOY_MANIFEST,
    executable_paths: frozenset[str] = EXECUTABLE_MANIFEST_PATHS,
) -> list[dict]:
    """Copies every ``manifest`` path's content at ``harness_commit`` into
    ``${chain_dir_real}/harness/<path>``, verifying each deployed file's
    on-disk SHA-256 against the SHA-256 of the git blob it was written from,
    and marking the entrypoints in ``executable_paths`` executable (git blob
    content alone does not carry the executable bit)."""
    harness_dir = chain_dir_real / "harness"
    records: list[dict] = []
    for rel_path in manifest:
        blob = _git_show_bytes(repo_root, harness_commit, rel_path)
        expected_sha256 = hashlib.sha256(blob).hexdigest()

        dest_path = harness_dir / PurePosixPath(rel_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(blob)

        deployed_sha256 = hashlib.sha256(dest_path.read_bytes()).hexdigest()
        if deployed_sha256 != expected_sha256:
            raise DeploymentRefused(
                f"checksum mismatch after deploying {rel_path}: "
                f"expected {expected_sha256}, got {deployed_sha256}"
            )
        if rel_path in executable_paths:
            dest_path.chmod(dest_path.stat().st_mode | 0o111)
        records.append(
            {
                "path": rel_path,
                "sha256": expected_sha256,
                "deployed_path": str(dest_path),
                "executable": rel_path in executable_paths,
            }
        )
    return records


def build_receipt(
    *,
    chain_dir_real: Path,
    audit: dict,
    file_records: list[dict],
    deployed_at_utc: str,
) -> dict:
    return {
        "frozen_scientific_commit": audit["frozen_scientific_commit"],
        "operational_base_commit": audit["operational_base_commit"],
        "harness_commit": audit["harness_commit"],
        "chain_dir": str(chain_dir_real),
        "deployed_at_utc": deployed_at_utc,
        "provenance_audit": audit,
        "files": file_records,
    }


def deploy(
    *,
    repo_root: Path,
    chain_dir: Path,
    harness_commit: str,
    operational_base_commit: str | None = None,
    frozen_scientific_commit: str = FROZEN_SCIENTIFIC_COMMIT,
    approved_operational_base_commits: Sequence[str] = APPROVED_OPERATIONAL_BASE_COMMITS,
    manifest: Sequence[str] = DEPLOY_MANIFEST,
    executable_paths: frozenset[str] = EXECUTABLE_MANIFEST_PATHS,
    allowed_harness_paths: frozenset[str] = ALLOWED_HARNESS_CHANGED_PATHS,
    approved_doc_paths: frozenset[str] = APPROVED_OPERATIONAL_DOC_PATHS,
) -> dict:
    """End-to-end: boundary check -> provenance audit (gates deployment) ->
    file deployment+verification -> receipt. Raises ``DeploymentRefused``
    (writing no files and no receipt) on any refusal."""
    chain_dir_real = assert_chain_dir_within_project_scratch_local(repo_root, chain_dir)

    audit = provenance_audit(
        repo_root,
        harness_commit=harness_commit,
        operational_base_commit=operational_base_commit,
        frozen_scientific_commit=frozen_scientific_commit,
        approved_operational_base_commits=approved_operational_base_commits,
        allowed_harness_paths=allowed_harness_paths,
        approved_doc_paths=approved_doc_paths,
    )
    if not audit["passed"]:
        raise DeploymentRefused(
            "frozen-payload provenance audit failed: harness commit "
            f"{harness_commit} (operational_base_commit="
            f"{audit['operational_base_commit']}) is not composed exclusively "
            f"of approved operational-base and operational-harness changes "
            f"relative to frozen scientific commit {frozen_scientific_commit}: "
            f"disallowed={audit['disallowed_paths']} "
            f"decomposition_matches={audit['decomposition_matches']}"
        )

    file_records = deploy_harness_files(repo_root, chain_dir_real, harness_commit, manifest, executable_paths)

    deployed_at_utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt = build_receipt(
        chain_dir_real=chain_dir_real,
        audit=audit,
        file_records=file_records,
        deployed_at_utc=deployed_at_utc,
    )
    receipt_path = chain_dir_real / "deploy_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chain-dir", required=True, help="RD1_CHAIN_DIR to deploy the harness into")
    parser.add_argument("--harness-commit", required=True, help="tracked operational-harness commit sha")
    parser.add_argument(
        "--operational-base-commit",
        default=None,
        help="commit the harness commit is built on top of (defaults to <harness-commit>^)",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="repo root (defaults to this script's own checkout)")
    args = parser.parse_args(argv)

    try:
        receipt = deploy(
            repo_root=Path(args.repo_root),
            chain_dir=Path(args.chain_dir),
            harness_commit=args.harness_commit,
            operational_base_commit=args.operational_base_commit,
        )
    except DeploymentRefused as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
