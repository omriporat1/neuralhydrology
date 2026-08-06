"""Focused tests for scripts/extract_pilot_tb_losses.py -- the read-only
TensorBoard transformed-space loss extractor.

The real 'tensorboard' package is not importable in this local Windows
Python environment (confirmed while building this script -- see its module
docstring/job 45762023/45762029/45762034), so these tests never import
tensorboard.backend.event_processing.event_accumulator. Instead they
monkeypatch the script's own `_load_scalar_tags` seam with an in-memory fake
EventAccumulator built from plain Python data, and exercise the pure
tag-resolution / epoch-mapping / output-schema logic that seam feeds into --
exactly the logic that was empirically corrected twice against the real
incumbent event file. Real extraction was already exercised end-to-end via a
short Slurm CPU job against the real event file (job 45762034); these tests
cover the decision logic itself, deterministically and without Slurm.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import namedtuple
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "extract_pilot_tb_losses.py"

FakeEvent = namedtuple("FakeEvent", ["wall_time", "step", "value"])


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "extract_pilot_tb_losses_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def extractor(monkeypatch, tmp_path):
    return _load_module()


class FakeEventAccumulator:
    """Minimal stand-in for tensorboard's EventAccumulator: a fixed
    tag -> [FakeEvent, ...] mapping, nothing else."""

    def __init__(self, scalars: dict):
        self._scalars = scalars

    def Scalars(self, tag):
        return self._scalars[tag]


def _install_fake_tags(monkeypatch, extractor, tags, scalars, event_path: Path):
    event_path.write_bytes(b"")  # extract_losses() only checks is_file()
    fake_ea = FakeEventAccumulator(scalars)

    def _fake_load_scalar_tags(path):
        assert Path(path) == event_path
        return fake_ea, tags

    monkeypatch.setattr(extractor, "_load_scalar_tags", _fake_load_scalar_tags)
    return event_path


REAL_INCUMBENT_TAGS = [
    "train/avg_loss", "train/avg_total_loss", "train/loss", "train/total_loss",
    "valid/avg_loss", "valid/avg_total_loss",
]


def _basic_scalars():
    return {
        "train/avg_total_loss": [
            FakeEvent(wall_time=100.0, step=1, value=7.5),
            FakeEvent(wall_time=200.0, step=2, value=7.2),
            FakeEvent(wall_time=300.0, step=3, value=7.1),
        ],
        "valid/avg_total_loss": [
            FakeEvent(wall_time=310.0, step=3, value=8.1),
        ],
        "train/avg_loss": [FakeEvent(wall_time=100.0, step=1, value=6.0)],
        "valid/avg_loss": [FakeEvent(wall_time=310.0, step=3, value=6.9)],
        "train/loss": [FakeEvent(wall_time=100.0, step=1478891, value=6.0)],
        "train/total_loss": [FakeEvent(wall_time=100.0, step=1478891, value=7.5)],
    }


# --- exact-tag resolution ----------------------------------------------------

def test_exact_avg_total_loss_tags_resolved_when_multiple_loss_shaped_candidates_present(
    extractor, monkeypatch, tmp_path
):
    """Mirrors the real incumbent event file's shape (job 45762023/45762029):
    both {train,valid}/avg_loss and {train,valid}/avg_total_loss exist. The
    extractor must resolve to avg_total_loss on both sides, never guess."""
    event_path = _install_fake_tags(
        monkeypatch, extractor, REAL_INCUMBENT_TAGS, _basic_scalars(), tmp_path / "events.fake"
    )
    result = extractor.extract_losses(event_path, run_id="emb128x64_seedA_cap_low_cal")
    assert result["train_transformed_space_loss_tag"] == "train/avg_total_loss"
    assert result["valid_transformed_space_loss_tag"] == "valid/avg_total_loss"


def test_missing_train_avg_total_loss_tag_fails_clearly(extractor, monkeypatch, tmp_path):
    tags = ["train/avg_loss", "valid/avg_total_loss"]
    scalars = {
        "train/avg_loss": [FakeEvent(1.0, 1, 6.0)],
        "valid/avg_total_loss": [FakeEvent(1.0, 1, 8.0)],
    }
    event_path = _install_fake_tags(monkeypatch, extractor, tags, scalars, tmp_path / "events.fake")
    with pytest.raises(extractor.TBLossExtractionError) as exc_info:
        extractor.extract_losses(event_path)
    assert "train/avg_total_loss" in str(exc_info.value)


def test_missing_valid_avg_total_loss_tag_fails_clearly_no_validation_loss(extractor, monkeypatch, tmp_path):
    tags = ["train/avg_total_loss", "valid/avg_loss"]
    scalars = {
        "train/avg_total_loss": [FakeEvent(1.0, 1, 7.0)],
        "valid/avg_loss": [FakeEvent(1.0, 1, 6.9)],
    }
    event_path = _install_fake_tags(monkeypatch, extractor, tags, scalars, tmp_path / "events.fake")
    with pytest.raises(extractor.TBLossExtractionError) as exc_info:
        extractor.extract_losses(event_path)
    assert "valid/avg_total_loss" in str(exc_info.value)


def test_no_loss_shaped_tags_at_all_fails_with_no_fuzzy_candidates_listed(extractor, monkeypatch, tmp_path):
    tags = ["valid/median_nse"]
    scalars = {"valid/median_nse": [FakeEvent(1.0, 1, 0.5)]}
    event_path = _install_fake_tags(monkeypatch, extractor, tags, scalars, tmp_path / "events.fake")
    with pytest.raises(extractor.TBLossExtractionError) as exc_info:
        extractor.extract_losses(event_path)
    msg = str(exc_info.value)
    assert "train side=[]" in msg
    assert "valid side=[]" in msg


# --- terminology separation: raw-space NSE never folded into loss ----------

def test_raw_space_nse_tags_recorded_separately_never_labeled_loss(extractor, monkeypatch, tmp_path):
    tags = REAL_INCUMBENT_TAGS + ["valid/median_nse", "valid/mean_nse"]
    scalars = _basic_scalars()
    scalars["valid/median_nse"] = [FakeEvent(310.0, 3, 0.62)]
    scalars["valid/mean_nse"] = [FakeEvent(310.0, 3, 0.55)]
    event_path = _install_fake_tags(monkeypatch, extractor, tags, scalars, tmp_path / "events.fake")

    result = extractor.extract_losses(event_path)

    assert result["raw_space_nse_tags_present"] == ["valid/mean_nse", "valid/median_nse"]
    epoch3 = next(e for e in result["inventory"] if e["epoch"] == 3)
    assert "raw_space_nse_scalars" in epoch3
    assert epoch3["raw_space_nse_scalars"] == {"valid/mean_nse": 0.55, "valid/median_nse": 0.62}
    # never folded into either loss column
    assert epoch3["valid_transformed_space_loss"] == 8.1
    assert epoch3["train_transformed_space_loss"] == 7.1


def test_no_raw_nse_tags_present_when_none_logged(extractor, monkeypatch, tmp_path):
    event_path = _install_fake_tags(
        monkeypatch, extractor, REAL_INCUMBENT_TAGS, _basic_scalars(), tmp_path / "events.fake"
    )
    result = extractor.extract_losses(event_path)
    assert result["raw_space_nse_tags_present"] == []
    for entry in result["inventory"]:
        assert "raw_space_nse_scalars" not in entry


# --- epoch mapping: unambiguous integer step == epoch -----------------------

def test_epoch_mapping_merges_train_and_valid_rows_by_step(extractor, monkeypatch, tmp_path):
    event_path = _install_fake_tags(
        monkeypatch, extractor, REAL_INCUMBENT_TAGS, _basic_scalars(), tmp_path / "events.fake"
    )
    result = extractor.extract_losses(event_path)
    epochs = [e["epoch"] for e in result["inventory"]]
    assert epochs == [1, 2, 3]
    assert result["n_epochs_with_train_loss"] == 3
    assert result["n_epochs_with_valid_loss"] == 1
    epoch1 = result["inventory"][0]
    assert "valid_transformed_space_loss" not in epoch1
    epoch3 = result["inventory"][2]
    assert epoch3["valid_transformed_space_loss"] == 8.1


def test_non_integer_step_raises_unresolvable_epoch_mapping(extractor, monkeypatch, tmp_path):
    tags = ["train/avg_total_loss", "valid/avg_total_loss"]
    scalars = {
        "train/avg_total_loss": [FakeEvent(1.0, 1.5, 7.0)],
        "valid/avg_total_loss": [FakeEvent(1.0, 1, 8.0)],
    }
    event_path = _install_fake_tags(monkeypatch, extractor, tags, scalars, tmp_path / "events.fake")
    with pytest.raises(extractor.TBLossExtractionError, match="UNRESOLVABLE_EPOCH_MAPPING"):
        extractor.extract_losses(event_path)


def test_conflicting_duplicate_step_values_raises_unresolvable_epoch_mapping(extractor, monkeypatch, tmp_path):
    tags = ["train/avg_total_loss", "valid/avg_total_loss"]
    scalars = {
        "train/avg_total_loss": [FakeEvent(1.0, 1, 7.0), FakeEvent(2.0, 1, 7.9)],
        "valid/avg_total_loss": [FakeEvent(1.0, 1, 8.0)],
    }
    event_path = _install_fake_tags(monkeypatch, extractor, tags, scalars, tmp_path / "events.fake")
    with pytest.raises(extractor.TBLossExtractionError, match="UNRESOLVABLE_EPOCH_MAPPING"):
        extractor.extract_losses(event_path)


# --- deterministic extraction -----------------------------------------------

def test_extraction_is_deterministic_across_repeated_calls(extractor, monkeypatch, tmp_path):
    event_path = _install_fake_tags(
        monkeypatch, extractor, REAL_INCUMBENT_TAGS, _basic_scalars(), tmp_path / "events.fake"
    )
    result_a = extractor.extract_losses(event_path, run_id="x")
    result_b = extractor.extract_losses(event_path, run_id="x")
    assert result_a == result_b
    assert result_a["discovered_scalar_tags"] == sorted(result_a["discovered_scalar_tags"])


def test_missing_event_file_fails_before_any_tensorboard_import(extractor, tmp_path):
    with pytest.raises(extractor.TBLossExtractionError, match="does not exist"):
        extractor.extract_losses(tmp_path / "does_not_exist.tfevents")


# --- CSV/JSON output schema + deterministic ordering ------------------------

def test_write_outputs_produces_json_and_csv_with_expected_columns(extractor, monkeypatch, tmp_path):
    event_path = _install_fake_tags(
        monkeypatch, extractor, REAL_INCUMBENT_TAGS, _basic_scalars(), tmp_path / "events.fake"
    )
    result = extractor.extract_losses(event_path, run_id="emb128x64_seedA_cap_low_cal")
    out_dir = tmp_path / "out"
    json_path, csv_path = extractor._write_outputs(result, out_dir)

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["train_transformed_space_loss_tag"] == "train/avg_total_loss"
    assert written["valid_transformed_space_loss_tag"] == "valid/avg_total_loss"
    assert [e["epoch"] for e in written["inventory"]] == [1, 2, 3]

    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    assert header == [
        "epoch", "run_id",
        "train_transformed_space_loss", "train_wall_time",
        "valid_transformed_space_loss", "valid_wall_time",
        "raw_space_nse_scalars", "source_event_file",
    ]
    data_rows = rows[1:]
    assert [row[0] for row in data_rows] == ["1", "2", "3"]


def test_write_outputs_is_deterministic_byte_for_byte(extractor, monkeypatch, tmp_path):
    event_path = _install_fake_tags(
        monkeypatch, extractor, REAL_INCUMBENT_TAGS, _basic_scalars(), tmp_path / "events.fake"
    )
    result = extractor.extract_losses(event_path, run_id="r")
    json_path_a, csv_path_a = extractor._write_outputs(result, tmp_path / "out_a")
    json_path_b, csv_path_b = extractor._write_outputs(result, tmp_path / "out_b")
    assert json_path_a.read_text(encoding="utf-8") == json_path_b.read_text(encoding="utf-8")
    assert csv_path_a.read_text(encoding="utf-8") == csv_path_b.read_text(encoding="utf-8")


# --- terminology separation: dict keys never conflate the three concepts ---

def test_result_schema_keeps_train_loss_valid_loss_and_raw_nse_as_distinct_fields(
    extractor, monkeypatch, tmp_path
):
    tags = REAL_INCUMBENT_TAGS + ["valid/median_nse"]
    scalars = _basic_scalars()
    scalars["valid/median_nse"] = [FakeEvent(310.0, 3, 0.62)]
    event_path = _install_fake_tags(monkeypatch, extractor, tags, scalars, tmp_path / "events.fake")

    result = extractor.extract_losses(event_path)

    assert "train_transformed_space_loss_tag" in result
    assert "valid_transformed_space_loss_tag" in result
    assert "raw_space_nse_tags_present" in result
    assert result["train_transformed_space_loss_tag"] != result["valid_transformed_space_loss_tag"]
    assert "valid/median_nse" not in (
        result["train_transformed_space_loss_tag"], result["valid_transformed_space_loss_tag"]
    )
    assert "aggregation_comparability_note" in result
    assert "not computed via an identical aggregation" in result["aggregation_comparability_note"].lower() or \
        "NOT computed via an identical aggregation" in result["aggregation_comparability_note"]


# --- tag pattern sanity (used only for the fuzzy diagnostic-candidate list) -

def test_train_loss_pattern_matches_expected_candidates(extractor):
    assert extractor.TRAIN_LOSS_TAG_PATTERN.match("train/avg_loss")
    assert extractor.TRAIN_LOSS_TAG_PATTERN.match("train/avg_total_loss")
    assert not extractor.TRAIN_LOSS_TAG_PATTERN.match("valid/avg_loss")
    assert not extractor.TRAIN_LOSS_TAG_PATTERN.match("train/loss")


def test_valid_loss_pattern_matches_expected_candidates(extractor):
    assert extractor.VALID_LOSS_TAG_PATTERN.match("valid/avg_loss")
    assert extractor.VALID_LOSS_TAG_PATTERN.match("valid/avg_total_loss")
    assert not extractor.VALID_LOSS_TAG_PATTERN.match("train/avg_loss")


def test_raw_nse_pattern_matches_only_median_or_mean_nse(extractor):
    assert extractor.RAW_NSE_TAG_PATTERN.match("valid/median_nse")
    assert extractor.RAW_NSE_TAG_PATTERN.match("valid/mean_nse")
    assert not extractor.RAW_NSE_TAG_PATTERN.match("valid/avg_total_loss")
    assert not extractor.RAW_NSE_TAG_PATTERN.match("train/median_nse")


# --- CLI: clear failure surfaces as a clean non-zero exit -------------------

def test_cli_main_exits_nonzero_with_clear_message_on_extraction_failure(
    extractor, monkeypatch, tmp_path, capsys
):
    def _raise(*a, **k):
        raise extractor.TBLossExtractionError("NO_VALIDATION_LOSS_TAG: boom")

    monkeypatch.setattr(extractor, "extract_losses", _raise)
    monkeypatch.setattr(
        sys, "argv",
        [
            "extract_pilot_tb_losses.py",
            "--event-file", str(tmp_path / "events.fake"),
            "--out-dir", str(tmp_path / "out"),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        extractor.main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "TB_LOSS_EXTRACTION_FAILED" in captured.err
    assert "NO_VALIDATION_LOSS_TAG" in captured.err


def test_cli_main_writes_outputs_and_prints_ok_status_on_success(extractor, monkeypatch, tmp_path, capsys):
    event_path = _install_fake_tags(
        monkeypatch, extractor, REAL_INCUMBENT_TAGS, _basic_scalars(), tmp_path / "events.fake"
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv",
        [
            "extract_pilot_tb_losses.py",
            "--event-file", str(event_path),
            "--run-id", "emb128x64_seedA_cap_low_cal",
            "--out-dir", str(out_dir),
        ],
    )
    extractor.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "OK"
    assert printed["train_transformed_space_loss_tag"] == "train/avg_total_loss"
    assert printed["valid_transformed_space_loss_tag"] == "valid/avg_total_loss"
    assert (out_dir / "pilot_tb_loss_extraction.json").is_file()
    assert (out_dir / "pilot_tb_loss_extraction.csv").is_file()
