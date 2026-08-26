"""Repo-wide pytest fixtures."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def short_tmp_path():
    """A short-rooted scratch directory for tests that need to create deeply
    nested paths on Windows.

    Some Sweep-v2 six-axis identity strings (e.g. ``trial_id_v2``, which
    embeds both ``proposal_id`` and ``configuration_id`` for controller-
    proposal collision-safety -- see that function's docstring) are long
    enough that using one as a directory name overflows Windows' 260-char
    ``MAX_PATH`` once nested under pytest's own deeply-nested default
    ``tmp_path`` -- and this repo's own absolute root is already deep enough
    on its own to leave too little budget for a long trial_id plus an
    atomic-write temp suffix, so a plain gitignored ``tmp/`` subdirectory is
    not short enough either. Use the Windows extended-length (``\\\\?\\``)
    path prefix instead, which makes NTFS bypass ``MAX_PATH`` entirely
    (verified directly against ``os.mkdir``/``tempfile.mkstemp``); this is a
    pure OS-level escape hatch, not a change to any identity/path logic in
    the functions under test. Real execution happens on Moriah (Linux),
    where ``MAX_PATH`` does not exist at all.
    """
    root = Path(__file__).parents[1] / "tmp" / "pytest_v2_execution_scratch"
    root.mkdir(parents=True, exist_ok=True)
    real_path = root / uuid.uuid4().hex[:12]
    real_path.mkdir()
    yield_path = Path("\\\\?\\" + str(real_path.resolve())) if os.name == "nt" else real_path
    try:
        yield yield_path
    finally:
        shutil.rmtree(real_path, ignore_errors=True)
