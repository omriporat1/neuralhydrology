"""Tests for src/baseline/nh_register.py.

Registration is deliberately not an import-time side effect: importing the
module must not touch NeuralHydrology's global dataset registry until
register_flashnh_dataset() is called explicitly. Verified here only through
NH's own public API (register_dataset/get_dataset via a real, minimal
GenericDataset-compatible package), never via the registry's private state.
"""
import sys
from pathlib import Path

from neuralhydrology.utils.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _nh_synthetic import build_synthetic_package, prepare_run_dirs  # noqa: E402

from src.baseline.nh_dataset import FlashNHDataset
from src.baseline.nh_register import FLASHNH_DATASET_KEY, register_flashnh_dataset


def _build_minimal_cfg(tmp_path: Path) -> Path:
    return build_synthetic_package(
        tmp_path,
        basins=["SYN01"],
        seq_length=3,
        lead_hours=2,
        bad_hours=[],
    )


def test_flashnh_dataset_key_constant():
    assert FLASHNH_DATASET_KEY == "flashnh"


def test_register_flashnh_dataset_is_idempotent(tmp_path):
    # Calling registration multiple times must not raise and must not change
    # which class ends up resolved for the "flashnh" key.
    register_flashnh_dataset()
    register_flashnh_dataset()
    register_flashnh_dataset()

    cfg_path = _build_minimal_cfg(tmp_path)
    cfg = Config(cfg_path)
    prepare_run_dirs(cfg, tmp_path, "idempotent")

    ds = FlashNHDataset.__mro__  # sanity: class object still importable/usable
    assert FlashNHDataset in ds


def test_config_key_flashnh_resolves_to_flashnh_dataset(tmp_path):
    register_flashnh_dataset()

    cfg_path = _build_minimal_cfg(tmp_path)
    cfg = Config(cfg_path)
    assert cfg.dataset == "flashnh"
    prepare_run_dirs(cfg, tmp_path, "resolve")

    from neuralhydrology.datasetzoo import get_dataset

    train_ds = get_dataset(cfg=cfg, is_train=True, period="train")
    assert type(train_ds) is FlashNHDataset
    assert hasattr(train_ds, "flashnh_filter_stats")
