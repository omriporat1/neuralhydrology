"""Tests for the shared production/rehearsal runtime-safety contract,
``src.baseline.sweep_v1_runtime_contract``.

Covers, in order:
  1. ``verify_commit_and_interpreter`` passes on a matching commit/clean
     tree/matching interpreter, using an isolated fixture git repo (never the
     real development repo's own working tree, which is routinely dirty
     mid-task -- see fixture docstring).
  2. ``verify_commit_and_interpreter`` raises on a commit mismatch.
  3. ``verify_commit_and_interpreter`` raises on a dirty tracked tree.
  4. ``verify_commit_and_interpreter`` raises on an interpreter mismatch.
  5. ``verify_home_and_netrc`` returns safe diagnostics for a present,
     safely-permissioned ``.netrc`` under a fixture HOME -- never file
     contents.
  6. ``verify_home_and_netrc`` raises when ``.netrc`` is required but absent.
  7. ``verify_home_and_netrc`` does not raise (and reports ``netrc_exists=False``)
     when ``.netrc`` is absent and ``require_netrc=False``.
  8. ``safe_wandb_env_var_names`` reports only ``WANDB_*`` NAMES, never values.
  9. ``RuntimeDiagnostics.to_safe_dict()`` never contains a credential value,
     even when a ``WANDB_API_KEY``-shaped env var is set (name only).
  10. ``run_full_runtime_contract`` composes both checks end-to-end and fails
      fast on the commit/interpreter check before ever touching HOME/netrc.

Uses only fixture HOME/repo directories under ``tmp_path`` -- never reads or
touches the real developer/operator ``~/.netrc``, and never asserts on this
development repo's own (routinely dirty mid-task) working-tree state.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.baseline.sweep_v1_runtime_contract import (
    RuntimeContractError, git_head, run_full_runtime_contract, safe_wandb_env_var_names,
    verify_commit_and_interpreter, verify_home_and_netrc,
)

ROOT = Path(__file__).resolve().parent.parent


def _init_fixture_git_repo(tmp_path: Path) -> "tuple[Path, str]":
    """An isolated, throwaway git repo with one committed file -- lets
    commit/dirty-tree checks be tested deterministically, independent of this
    development repo's own (routinely dirty mid-task) working-tree state."""
    repo = tmp_path / "fixture_repo"
    repo.mkdir()
    run = lambda *args: subprocess.run(args, cwd=repo, check=True, capture_output=True, text=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "test")
    (repo / "file.txt").write_text("v1\n", encoding="utf-8")
    run("git", "add", "file.txt")
    run("git", "commit", "-q", "-m", "init")
    head = git_head(repo)
    assert head is not None
    return repo, head


# --- verify_commit_and_interpreter ---------------------------------------------

def test_verify_commit_and_interpreter_passes_on_matching_commit_and_interpreter(tmp_path):
    repo, head = _init_fixture_git_repo(tmp_path)
    result = verify_commit_and_interpreter(
        repo_root=repo, expected_commit=head, expected_runtime_python=sys.executable,
        actual_executable=sys.executable,
    )
    assert result == {"git_commit": head, "runtime_python": sys.executable}


def test_verify_commit_and_interpreter_raises_on_commit_mismatch(tmp_path):
    repo, _head = _init_fixture_git_repo(tmp_path)
    with pytest.raises(RuntimeContractError, match="git HEAD"):
        verify_commit_and_interpreter(
            repo_root=repo, expected_commit="0" * 40, expected_runtime_python=sys.executable,
            actual_executable=sys.executable,
        )


def test_verify_commit_and_interpreter_raises_on_dirty_tracked_tree(tmp_path):
    repo, head = _init_fixture_git_repo(tmp_path)
    (repo / "file.txt").write_text("v2 -- uncommitted edit\n", encoding="utf-8")
    with pytest.raises(RuntimeContractError, match="dirty"):
        verify_commit_and_interpreter(
            repo_root=repo, expected_commit=head, expected_runtime_python=sys.executable,
            actual_executable=sys.executable,
        )


def test_verify_commit_and_interpreter_raises_on_interpreter_mismatch(tmp_path):
    repo, head = _init_fixture_git_repo(tmp_path)
    with pytest.raises(RuntimeContractError, match="python executable"):
        verify_commit_and_interpreter(
            repo_root=repo, expected_commit=head,
            expected_runtime_python="/some/other/canonical/python",
            actual_executable=sys.executable,
        )


# --- verify_home_and_netrc ------------------------------------------------------

def test_verify_home_and_netrc_reports_safe_diagnostics_for_a_present_file(tmp_path):
    home = tmp_path / "fixture_home"
    home.mkdir()
    netrc = home / ".netrc"
    netrc.write_text("machine api.wandb.ai\n  login user\n  password not-a-real-secret\n", encoding="utf-8")
    netrc.chmod(0o600)

    diagnostics = verify_home_and_netrc(home=str(home), require_netrc=True)

    assert diagnostics["home"] == str(home)
    assert diagnostics["netrc_exists"] is True
    if os.name == "posix":
        # chmod bits are not meaningfully enforced on Windows/NTFS -- this
        # exact-mode assertion is only meaningful on the real POSIX targets
        # (local dev on Linux/mac, and the actual Moriah remote).
        assert diagnostics["netrc_mode_octal"] == "0o600"
        assert diagnostics["netrc_owner_matches_effective_user"] is True
        assert diagnostics["netrc_group_or_world_accessible"] is False
    # Never file contents.
    assert "not-a-real-secret" not in str(diagnostics)
    assert "password" not in str(diagnostics)


def test_verify_home_and_netrc_raises_when_required_and_absent(tmp_path):
    home = tmp_path / "fixture_home_no_netrc"
    home.mkdir()
    with pytest.raises(RuntimeContractError, match="credential store not found"):
        verify_home_and_netrc(home=str(home), require_netrc=True)


def test_verify_home_and_netrc_does_not_raise_when_not_required_and_absent(tmp_path):
    home = tmp_path / "fixture_home_no_netrc_optional"
    home.mkdir()
    diagnostics = verify_home_and_netrc(home=str(home), require_netrc=False)
    assert diagnostics["netrc_exists"] is False
    assert diagnostics["netrc_mode_octal"] is None


def test_verify_home_and_netrc_flags_a_group_or_world_readable_file(tmp_path):
    home = tmp_path / "fixture_home_loose_perms"
    home.mkdir()
    netrc = home / ".netrc"
    netrc.write_text("machine api.wandb.ai\n  login user\n  password not-a-real-secret\n", encoding="utf-8")
    netrc.chmod(0o644)

    diagnostics = verify_home_and_netrc(home=str(home), require_netrc=True)
    if os.name == "posix":
        assert diagnostics["netrc_group_or_world_accessible"] is True


def test_verify_home_and_netrc_raises_when_home_is_not_set(monkeypatch):
    monkeypatch.delenv("HOME", raising=False)
    with pytest.raises(RuntimeContractError, match="HOME is not set"):
        verify_home_and_netrc(home=None, require_netrc=True)


# --- safe_wandb_env_var_names ----------------------------------------------------

def test_safe_wandb_env_var_names_reports_names_only_never_values(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "totally-secret-value-should-never-appear")
    monkeypatch.setenv("WANDB_PROJECT", "flashnh-stage1")
    monkeypatch.delenv("SOME_OTHER_VAR", raising=False)

    names = safe_wandb_env_var_names()

    assert "WANDB_API_KEY" in names
    assert "WANDB_PROJECT" in names
    assert names == sorted(names)
    assert "totally-secret-value-should-never-appear" not in names


# --- run_full_runtime_contract ---------------------------------------------------

def test_run_full_runtime_contract_fails_fast_on_commit_mismatch_before_netrc_check(monkeypatch, tmp_path):
    # Point HOME at a directory with no .netrc at all -- if the commit check
    # did not fail FIRST, this would raise a DIFFERENT (netrc) error instead.
    monkeypatch.setenv("HOME", str(tmp_path / "unused_home"))
    with pytest.raises(RuntimeContractError, match="git HEAD"):
        run_full_runtime_contract(
            repo_root=ROOT, expected_commit="0" * 40, expected_runtime_python=sys.executable,
        )


def test_run_full_runtime_contract_returns_full_safe_diagnostics(monkeypatch, tmp_path):
    repo, head = _init_fixture_git_repo(tmp_path)
    home = tmp_path / "fixture_home"
    home.mkdir()
    (home / ".netrc").write_text("machine api.wandb.ai\n", encoding="utf-8")
    (home / ".netrc").chmod(0o600)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("WANDB_MODE", "offline")

    diagnostics = run_full_runtime_contract(
        repo_root=repo, expected_commit=head, expected_runtime_python=sys.executable,
    )

    safe = diagnostics.to_safe_dict()
    assert safe["git_commit"] == head
    assert safe["runtime_python"] == sys.executable
    assert safe["netrc_exists"] is True
    assert "WANDB_MODE" in safe["wandb_env_var_names"]
    assert "machine api.wandb.ai" not in str(safe)
