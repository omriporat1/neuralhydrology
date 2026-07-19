"""Idempotent NeuralHydrology registration for FlashNHDataset.

Deliberately not registered at import time: importing this module has no
side effects until ``register_flashnh_dataset()`` is called explicitly, so
ordinary test collection / unrelated imports of ``src.baseline`` cannot
accidentally mutate NeuralHydrology's global dataset registry.
"""
from __future__ import annotations

from src.baseline.nh_dataset import FlashNHDataset

__all__ = ["register_flashnh_dataset", "FLASHNH_DATASET_KEY"]

FLASHNH_DATASET_KEY = "flashnh"

_registered = False


def register_flashnh_dataset() -> None:
    """Register FlashNHDataset under the "flashnh" dataset key. Safe to call
    more than once (a repeat call is a no-op)."""
    global _registered
    if _registered:
        return
    from neuralhydrology.datasetzoo import register_dataset

    register_dataset(FLASHNH_DATASET_KEY, FlashNHDataset)
    _registered = True
