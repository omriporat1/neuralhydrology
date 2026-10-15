"""Idempotent NeuralHydrology registration for FlashNHDataset.

Deliberately not registered at import time: importing this module has no
side effects until ``register_flashnh_dataset()`` is called explicitly, so
ordinary test collection / unrelated imports of ``src.baseline`` cannot
accidentally mutate NeuralHydrology's global dataset registry.

Registration also adds the read-only NH ``Config`` properties
``flashnh_common_support_file`` and ``flashnh_observation_cutoff`` (``None``
when unset) so that NH's unknown-key check accepts those FlashNHDataset keys;
configs without them are unaffected.
"""
from __future__ import annotations

from src.baseline.nh_dataset import FlashNHDataset

__all__ = ["register_flashnh_dataset", "FLASHNH_DATASET_KEY"]

FLASHNH_DATASET_KEY = "flashnh"
COMMON_SUPPORT_CONFIG_KEY = "flashnh_common_support_file"
OBSERVATION_CUTOFF_CONFIG_KEY = "flashnh_observation_cutoff"

_registered = False


def register_flashnh_dataset() -> None:
    """Register FlashNHDataset under the "flashnh" dataset key. Safe to call
    more than once (a repeat call is a no-op)."""
    global _registered
    if _registered:
        return
    from neuralhydrology.datasetzoo import register_dataset
    from neuralhydrology.utils.config import Config

    register_dataset(FLASHNH_DATASET_KEY, FlashNHDataset)
    for key in (COMMON_SUPPORT_CONFIG_KEY, OBSERVATION_CUTOFF_CONFIG_KEY):
        if not isinstance(getattr(Config, key, None), property):
            setattr(Config, key, property(lambda self, _key=key: self._cfg.get(_key, None)))
    _registered = True
