"""Tests for environment sanitization in ``src.baseline.rd1_p20_p24_env``.

Covers, in order:
  1. ``sanitize_environment`` strips every ``WANDB_``-prefixed variable and
     the exact bare name ``WANDB``, leaving unrelated variables untouched.
  2. ``sanitize_environment`` strips the exact stale-socket variable
     (``WANDB_SERVICE``) that caused the P20 forensic failure.
  3. ``sanitize_environment`` preserves a name explicitly listed in
     ``keep`` even though it matches the ``WANDB_`` prefix.
  4. ``sanitize_environment`` is a no-op on an environment with no
     ``WANDB*`` variables at all.

Correction (Phase A.1 review): the four tests that previously covered the
dead ``resolve_wandb_api_key()`` / ``CredentialError`` code were removed
along with that code (see ``rd1_p20_p24_env.py`` docstring for the
maintained-launcher credential dependency this module now documents
instead of reimplementing).

Never touches Moriah, W&B, or Slurm; never contacts a network.
"""
from __future__ import annotations

from src.baseline.rd1_p20_p24_env import sanitize_environment


def test_sanitize_environment_strips_all_wandb_prefixed_vars():
    env = {
        "WANDB_API_KEY": "secret",
        "WANDB_PROJECT": "flashnh-stage1",
        "WANDB": "1",
        "PATH": "/usr/bin",
        "HOME": "/home/omripo",
    }
    cleaned = sanitize_environment(env)
    assert cleaned == {"PATH": "/usr/bin", "HOME": "/home/omripo"}


def test_sanitize_environment_strips_wandb_service_socket_variable():
    env = {"WANDB_SERVICE": "unix:///tmp/stale.sock", "PATH": "/usr/bin"}
    cleaned = sanitize_environment(env)
    assert "WANDB_SERVICE" not in cleaned
    assert cleaned == {"PATH": "/usr/bin"}


def test_sanitize_environment_preserves_explicit_keep_list():
    env = {"WANDB_PROJECT": "flashnh-stage1", "WANDB_ENTITY": "omri-porat1-huji"}
    cleaned = sanitize_environment(env, keep=frozenset({"WANDB_PROJECT"}))
    assert cleaned == {"WANDB_PROJECT": "flashnh-stage1"}


def test_sanitize_environment_noop_when_no_wandb_vars():
    env = {"PATH": "/usr/bin", "HOME": "/home/omripo"}
    assert sanitize_environment(env) == dict(env)
