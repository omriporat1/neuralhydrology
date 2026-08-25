"""Shared runtime-safety contract for Sweep-v1 production and rehearsal launches.

Generalizes the commit/interpreter-pin pattern already qualified in
``scripts/wandb_exact_retry_join_qualification.py`` (``_guard_or_die``) and
its ``.sbatch`` launcher's inline shell-level equivalent, and adds HOME/
``.netrc`` presence-and-permission diagnostics. Production and disposable
rehearsal launches call this exact module so the same invariants are
enforced, and fail durably, before any W&B call.

Every function here is safe to print, log, or persist as durable evidence:
nothing ever reads or returns ``.netrc`` file *contents*, and W&B environment
state is only ever reported as variable *names*, never values.
"""
from __future__ import annotations

import os
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class RuntimeContractError(ValueError):
    """A required production/rehearsal runtime invariant did not hold."""


def git_head(repo_root: "str | Path") -> "str | None":
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root), capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return None


def git_dirty_tracked(repo_root: "str | Path") -> "list[str] | None":
    """Porcelain status of TRACKED files only (untracked scratch/evidence is expected)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=no"],
            cwd=str(repo_root), capture_output=True, text=True, check=True, timeout=10,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return None


def verify_commit_and_interpreter(
    *, repo_root: "str | Path", expected_commit: str, expected_runtime_python: str,
    actual_executable: "str | None" = None,
) -> dict[str, Any]:
    """Hard-fail unless HEAD matches exactly, the tracked tree is clean, and
    the running interpreter is exactly the pinned canonical path. Avoids
    relying on shell activation or PATH lookup: ``expected_runtime_python``
    and ``sys.executable`` are both absolute paths, compared literally.
    """
    actual_executable = actual_executable if actual_executable is not None else sys.executable

    actual_head = git_head(repo_root)
    if actual_head != expected_commit:
        raise RuntimeContractError(f"git HEAD {actual_head!r} != expected commit {expected_commit!r}")

    dirty = git_dirty_tracked(repo_root)
    if dirty:
        raise RuntimeContractError(f"tracked working tree is dirty, refusing: {dirty!r}")

    if actual_executable != expected_runtime_python:
        raise RuntimeContractError(
            f"python executable {actual_executable!r} != expected canonical runtime {expected_runtime_python!r}"
        )

    return {"git_commit": actual_head, "runtime_python": actual_executable}


def verify_home_and_netrc(*, home: "str | None" = None, require_netrc: bool = True) -> dict[str, Any]:
    """Safe-only HOME/.netrc diagnostics: existence, ownership, and
    permission bits -- never file contents. Returns a JSON-safe dict fit to
    print, log, or persist as durable evidence.
    """
    home = home if home is not None else os.environ.get("HOME")
    if not home:
        raise RuntimeContractError("HOME is not set")

    netrc_path = Path(home) / ".netrc"
    diagnostics: dict[str, Any] = {
        "home": home,
        "netrc_path": str(netrc_path),
        "netrc_exists": netrc_path.is_file(),
    }
    if not diagnostics["netrc_exists"]:
        if require_netrc:
            raise RuntimeContractError(f"required credential store not found: {netrc_path}")
        diagnostics.update(
            netrc_mode_octal=None,
            netrc_owner_matches_effective_user=None,
            netrc_group_or_world_accessible=None,
        )
        return diagnostics

    file_stat = netrc_path.stat()
    diagnostics["netrc_mode_octal"] = oct(stat.S_IMODE(file_stat.st_mode))
    if hasattr(os, "geteuid"):
        diagnostics["netrc_owner_matches_effective_user"] = file_stat.st_uid == os.geteuid()
        diagnostics["netrc_group_or_world_accessible"] = bool(file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO))
    else:
        diagnostics["netrc_owner_matches_effective_user"] = None
        diagnostics["netrc_group_or_world_accessible"] = None
    return diagnostics


def safe_wandb_env_var_names() -> list[str]:
    """Sorted ``WANDB_*`` environment variable NAMES currently set -- never values."""
    return sorted(name for name in os.environ if name.startswith("WANDB_"))


@dataclass(frozen=True)
class RuntimeDiagnostics:
    git_commit: str
    runtime_python: str
    home: str
    netrc_path: str
    netrc_exists: bool
    netrc_mode_octal: "str | None"
    netrc_owner_matches_effective_user: "bool | None"
    netrc_group_or_world_accessible: "bool | None"
    wandb_env_var_names: "list[str]"

    def to_safe_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_full_runtime_contract(
    *, repo_root: "str | Path", expected_commit: str, expected_runtime_python: str, require_netrc: bool = True,
) -> RuntimeDiagnostics:
    """Full shared runtime contract: commit/dirty-tree/interpreter pin, then
    HOME/.netrc presence+permission diagnostics, then WANDB_* env var NAMES
    only. Raises ``RuntimeContractError`` durably on the first failing
    invariant, before any W&B call. Production and rehearsal call this
    identically.
    """
    commit_info = verify_commit_and_interpreter(
        repo_root=repo_root, expected_commit=expected_commit, expected_runtime_python=expected_runtime_python,
    )
    netrc_info = verify_home_and_netrc(require_netrc=require_netrc)
    return RuntimeDiagnostics(
        git_commit=commit_info["git_commit"],
        runtime_python=commit_info["runtime_python"],
        home=netrc_info["home"],
        netrc_path=netrc_info["netrc_path"],
        netrc_exists=netrc_info["netrc_exists"],
        netrc_mode_octal=netrc_info.get("netrc_mode_octal"),
        netrc_owner_matches_effective_user=netrc_info.get("netrc_owner_matches_effective_user"),
        netrc_group_or_world_accessible=netrc_info.get("netrc_group_or_world_accessible"),
        wandb_env_var_names=safe_wandb_env_var_names(),
    )
