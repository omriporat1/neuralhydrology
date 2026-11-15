"""Idempotent NeuralHydrology registration for FlashNHDataset.

Deliberately not registered at import time: importing this module has no
side effects until ``register_flashnh_dataset()`` is called explicitly, so
ordinary test collection / unrelated imports of ``src.baseline`` cannot
accidentally mutate NeuralHydrology's global dataset registry.

Registration also adds the read-only NH ``Config`` properties
``flashnh_common_support_file``, ``flashnh_observation_cutoff``,
``flashnh_split_contract`` and ``flashnh_rotation_cycle`` (``None`` when unset)
so that NH's unknown-key check accepts those FlashNHDataset keys; configs
without them are unaffected. It also chains a check onto that key check that
refuses the split-contract keys under any ``dataset`` other than ``flashnh``
(:func:`check_split_contract_dataset`), so such a run stops at launch, and
re-runs that check on the merged config after ``Config.update_config``
(continue-training / finetune overlays).
"""
from __future__ import annotations

from src.baseline.nh_dataset import FlashNHDataset

__all__ = ["register_flashnh_dataset", "FLASHNH_DATASET_KEY"]

FLASHNH_DATASET_KEY = "flashnh"
COMMON_SUPPORT_CONFIG_KEY = "flashnh_common_support_file"
OBSERVATION_CUTOFF_CONFIG_KEY = "flashnh_observation_cutoff"
SPLIT_CONTRACT_CONFIG_KEY = "flashnh_split_contract"
ROTATION_CYCLE_CONFIG_KEY = "flashnh_rotation_cycle"

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
    for key in (COMMON_SUPPORT_CONFIG_KEY, OBSERVATION_CUTOFF_CONFIG_KEY, SPLIT_CONTRACT_CONFIG_KEY,
                ROTATION_CYCLE_CONFIG_KEY):
        if not isinstance(getattr(Config, key, None), property):
            setattr(Config, key, property(lambda self, _key=key: self._cfg.get(_key, None)))
    _install_split_contract_dataset_check(Config)
    _install_split_contract_merged_check(Config)
    _registered = True


def check_split_contract_dataset(cfg: dict, *, partial: bool = False) -> None:
    """Refuse a config that sets a split-contract key under a ``dataset``
    other than ``flashnh``: only FlashNHDataset applies the sealing, so such
    a run would train unsealed (user decision 2026-11-15, question c: stop it
    when the config is loaded, i.e. at launch, rather than only at scoring).

    ``partial=True`` (NH's per-``Config`` key check, which also sees the
    partial overlays passed to ``update_config``) skips a dict without a
    ``dataset`` key; the merged config is checked after the update instead."""
    if partial and "dataset" not in cfg:
        return
    keys = [k for k in (SPLIT_CONTRACT_CONFIG_KEY, ROTATION_CYCLE_CONFIG_KEY) if cfg.get(k) is not None]
    if keys and cfg.get("dataset") != FLASHNH_DATASET_KEY:
        raise ValueError(f"{keys} set under dataset {cfg.get('dataset')!r}: a water-year rotation split contract "
                         f"is applied only by the {FLASHNH_DATASET_KEY!r} dataset, so this run would load unsealed")


def _install_split_contract_dataset_check(config_cls) -> None:
    """Chain :func:`check_split_contract_dataset` onto NH's
    ``Config._check_cfg_keys`` (run by every ``Config(...)`` outside
    dev_mode), once. The raw-space scorer refuses such runs independently."""
    original = config_cls.__dict__["_check_cfg_keys"].__func__
    if getattr(original, "_flashnh_split_contract_check", False):
        return

    def checked(cfg: dict):
        original(cfg)
        check_split_contract_dataset(cfg, partial=True)

    checked._flashnh_split_contract_check = True
    config_cls._check_cfg_keys = staticmethod(checked)


def _install_split_contract_merged_check(config_cls) -> None:
    """Wrap NH's ``Config.update_config`` (used by continue-training and
    finetune to overlay a partial config on the base run's config), once, so
    the merged config is checked: NH checks only the partial overlay. Skipped
    under NH ``dev_mode``, like NH's own key check."""
    original = config_cls.__dict__["update_config"]
    if getattr(original, "_flashnh_split_contract_check", False):
        return

    def update_config(self, yml_path_or_dict, dev_mode: bool = False):
        original(self, yml_path_or_dict, dev_mode=dev_mode)
        if not (dev_mode or self._cfg.get("dev_mode", False)):
            check_split_contract_dataset(self._cfg)

    update_config.__doc__ = original.__doc__
    update_config._flashnh_split_contract_check = True
    config_cls.update_config = update_config
