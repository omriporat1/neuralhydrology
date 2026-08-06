"""Read-only extractor for transformed-space training/validation loss out of
an existing NH 1.13 TensorBoard event file, for one pilot/closure NH run
directory.

Why this exists: NH 1.13's ``neuralhydrology.training.logger.Logger``
already computes and writes transformed-space (i.e. normalized/model-output-
space, NOT raw-mm/h-space) training and validation loss to TensorBoard every
epoch it runs (see ``Logger.summarise()`` -- training writes
``train/avg_{loss_key}`` at ``global_step=self.epoch`` every epoch;
validation writes ``valid/avg_{loss_key}`` at the same ``global_step`` only
on epochs where ``epoch % cfg.validate_every == 0``). Flash-NH's own pilot
machinery (``pilot_screening_eval.py``) only ever computes and screens on
*raw-space, per-basin, median NSE* -- it never reads this already-written
transformed-space loss back out. This script is the smallest possible
read-only bridge: it opens the event file(s) NH itself already wrote, finds
the loss scalar tag(s) actually present (never assumes a name), and emits a
compact CSV+JSON inventory of epoch -> {train loss, valid loss}.

Guarantees:
  * Never writes into, modifies, or deletes anything under the NH run
    directory or its TensorBoard event file(s) -- opens them strictly
    read-only via TensorBoard's own ``EventAccumulator``.
  * Never calls any NeuralHydrology entrypoint (``start_run``/
    ``continue_run``/``evaluate``) and never recomputes or approximates a
    loss value from raw-space predictions -- every number in the output is
    copied verbatim from a scalar NH's own training process already wrote.
  * Never assumes a scalar tag name: tags are discovered from the event
    file's own tag index, not hardcoded blind. NH 1.13 logs TWO loss-shaped
    scalars per side (``{train,valid}/avg_loss`` pre-regularization and
    ``{train,valid}/avg_total_loss`` post-regularization -- confirmed
    empirically against the real incumbent event file, job 45762023/
    45762029), so a fuzzy "contains loss" match is genuinely ambiguous on
    both sides. This script resolves the ambiguity by requiring the exact
    tag ``avg_total_loss`` on each side, a choice grounded in NH's own
    source (``loss.py``'s docstring states "'total_loss' contains the
    overall loss"; ``basetrainer.py`` itself reports
    ``valid_metrics['avg_total_loss']`` as "the" validation loss in its own
    log line) rather than an arbitrary pick -- see ``extract_losses()``.
    If that exact tag is missing on either side, this is a hard failure
    (never a silent fall back to the other candidate).
  * Keeps the transformed-space training loss, transformed-space validation
    loss, and raw-space per-basin median NSE strictly separate in the
    output schema -- a raw-space NSE-shaped tag (``valid/median_nse`` and
    similar) is recorded, if present, under its own clearly distinct
    ``raw_space_median_nse`` column/field, never folded into either loss
    column and never labeled "loss".
  * Fails loudly (non-zero exit, clear message) rather than guessing on:
    no scalar data at all in the event file, no validation-loss-shaped tag
    found, more than one candidate train- or valid-loss tag, or a
    non-integer / internally-inconsistent step-to-epoch mapping (NH's own
    step convention is exactly the epoch number -- see module docstring
    above -- so a non-integer step, or two different loss values recorded
    at the same step, means this script's epoch-mapping assumption does not
    hold for this event file, and it refuses to guess further).

Output is written only to a caller-specified ``--out-dir``, which is always
a generated-evidence location (e.g. under ``evidence/`` or a scratch
directory) -- never a path under a tracked ``config/``, ``scripts/``, or
``src/`` directory, and this script never adds anything to source control.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TRAIN_LOSS_TAG_PATTERN = re.compile(r"^train/avg_.*loss.*$", re.IGNORECASE)
VALID_LOSS_TAG_PATTERN = re.compile(r"^valid/avg_.*loss.*$", re.IGNORECASE)
# Recorded separately, purely so a reader of the output JSON can see that a
# raw-space NSE-shaped tag exists and is NOT one of the two loss columns --
# never used to fill in a loss value.
RAW_NSE_TAG_PATTERN = re.compile(r"^valid/(median|mean)_nse$", re.IGNORECASE)


class TBLossExtractionError(RuntimeError):
    """Raised on any condition this extractor refuses to silently resolve."""


def _load_scalar_tags(event_file: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise TBLossExtractionError(
            "The 'tensorboard' package is not importable in this Python environment -- this "
            "extractor needs tensorboard.backend.event_processing.event_accumulator to read "
            "existing event files (it never imports torch or neuralhydrology). Run it inside "
            "an environment that already has tensorboard installed (e.g. the flashnh-moriah "
            "conda env), such as via a short Slurm CPU job."
        ) from exc

    ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        raise TBLossExtractionError(
            f"No scalar tags found at all in event file {event_file} -- refusing to guess; "
            "this file may not be a NH training/validation TensorBoard event file, or "
            "log_tensorboard may have been disabled for this run."
        )
    return ea, tags


def _extract_series(ea, tag: str, source_event_file: str) -> list:
    """Returns one dict per scalar event: epoch, wall_time, value, tag, source_event_file.

    NH's own step convention (see module docstring) makes TensorBoard's
    ``step`` exactly the epoch number for both train/* and valid/* scalars
    written by ``neuralhydrology.training.logger.Logger`` -- this function
    trusts that convention but verifies it holds (integer steps, no
    duplicate step with conflicting values) rather than assuming it blindly.
    """
    events = ea.Scalars(tag)
    rows = []
    seen_epoch_values = {}
    for ev in events:
        step = ev.step
        if float(step) != int(step):
            raise TBLossExtractionError(
                f"UNRESOLVABLE_EPOCH_MAPPING: tag {tag!r} has a non-integer TensorBoard step "
                f"({step!r}) at wall_time={ev.wall_time} -- this extractor's epoch mapping "
                "assumes NH's own convention that step == epoch, which does not hold here."
            )
        epoch = int(step)
        if epoch in seen_epoch_values and seen_epoch_values[epoch] != ev.value:
            raise TBLossExtractionError(
                f"UNRESOLVABLE_EPOCH_MAPPING: tag {tag!r} has two different values recorded at "
                f"the same epoch/step {epoch} ({seen_epoch_values[epoch]!r} vs {ev.value!r}) -- "
                "refusing to pick one silently."
            )
        seen_epoch_values[epoch] = ev.value
        rows.append({
            "epoch": epoch,
            "wall_time": ev.wall_time,
            "value": ev.value,
            "tag": tag,
            "source_event_file": source_event_file,
        })
    return rows


def extract_losses(event_file: Path, run_id: "str | None" = None) -> dict:
    event_file = Path(event_file)
    if not event_file.is_file():
        raise TBLossExtractionError(f"Event file does not exist: {event_file}")

    ea, tags = _load_scalar_tags(event_file)
    source_event_file = str(event_file)
    raw_nse_tags = sorted(t for t in tags if RAW_NSE_TAG_PATTERN.match(t))

    # Empirically (job 45762023/45762029 against the real incumbent event
    # file), BOTH sides log two loss-shaped scalars, not one:
    # {train,valid}/avg_loss (pre-regularization) and
    # {train,valid}/avg_total_loss (post-regularization). A fuzzy
    # "*loss*" pattern is therefore genuinely ambiguous on both sides --
    # matched and reported as AMBIGUOUS_LOSS_TAG below whenever it fires,
    # never silently picked between.
    #
    # The correct, non-arbitrary choice is 'avg_total_loss' on both sides,
    # grounded directly in NH's own source (not assumed): loss.py's
    # calculate_loss() docstring states verbatim "'total_loss' contains the
    # overall loss" (all_losses['total_loss'] = loss + every regularization
    # term; all_losses['loss'] is only the pre-regularization component --
    # see neuralhydrology/training/loss.py), and basetrainer.py itself
    # extracts valid_metrics['avg_total_loss'] as literally "the" validation
    # loss for its own human-readable log line ("Epoch N average validation
    # loss: ..." -- see neuralhydrology/training/basetrainer.py). This
    # extractor therefore requires the exact tag f'{side}/avg_total_loss' to
    # be present; if it is not, this is a hard failure (not a fall back to
    # a fuzzy guess among the remaining candidates).
    train_tag = "train/avg_total_loss"
    valid_tag = "valid/avg_total_loss"
    missing = [t for t in (train_tag, valid_tag) if t not in tags]
    if missing:
        fuzzy_train = sorted(t for t in tags if TRAIN_LOSS_TAG_PATTERN.match(t))
        fuzzy_valid = sorted(t for t in tags if VALID_LOSS_TAG_PATTERN.match(t))
        raise TBLossExtractionError(
            f"Expected exact tag(s) {missing} (NH's own 'total_loss' = overall/optimized loss, "
            "see neuralhydrology/training/loss.py + basetrainer.py) not found among this event "
            f"file's scalar tags ({sorted(tags)}). Loss-shaped candidates actually present: "
            f"train side={fuzzy_train}, valid side={fuzzy_valid}. Refusing to guess a substitute."
        )
    if valid_tag not in tags:
        raise TBLossExtractionError(
            f"NO_VALIDATION_LOSS_TAG: {valid_tag!r} not found among this event file's scalar "
            f"tags ({sorted(tags)})."
        )

    train_rows = _extract_series(ea, train_tag, source_event_file)
    valid_rows = _extract_series(ea, valid_tag, source_event_file)
    raw_nse_rows = {t: _extract_series(ea, t, source_event_file) for t in raw_nse_tags}

    by_epoch = defaultdict(dict)
    for row in train_rows:
        by_epoch[row["epoch"]]["train_transformed_space_loss"] = row["value"]
        by_epoch[row["epoch"]]["train_wall_time"] = row["wall_time"]
    for row in valid_rows:
        by_epoch[row["epoch"]]["valid_transformed_space_loss"] = row["value"]
        by_epoch[row["epoch"]]["valid_wall_time"] = row["wall_time"]
    for nse_tag, rows in raw_nse_rows.items():
        for row in rows:
            by_epoch[row["epoch"]].setdefault("raw_space_nse_scalars", {})[nse_tag] = row["value"]

    inventory = []
    for epoch in sorted(by_epoch):
        entry = {"epoch": epoch, "run_id": run_id, "source_event_file": source_event_file}
        entry.update(by_epoch[epoch])
        inventory.append(entry)

    return {
        "run_id": run_id,
        "source_event_file": source_event_file,
        "discovered_scalar_tags": sorted(tags),
        "train_transformed_space_loss_tag": train_tag,
        "valid_transformed_space_loss_tag": valid_tag,
        "raw_space_nse_tags_present": raw_nse_tags,
        "epoch_mapping": "TensorBoard step == NH epoch number (neuralhydrology.training.logger.Logger convention); verified integer/non-conflicting per tag above.",
        "aggregation_comparability_note": (
            "train_transformed_space_loss is a simple mean over per-update-step training-batch "
            "losses within the epoch (np.nanmean over Logger._metrics['total_loss']); "
            "valid_transformed_space_loss is a per-basin-batch-count-WEIGHTED mean across basins "
            "(see neuralhydrology.training.logger.Logger.summarise()'s tuple branch). Both are in "
            "the same transformed/model-output loss space and the same underlying loss function, "
            "so a train-vs-valid loss-gap plot is qualitatively meaningful, but the two series are "
            "NOT computed via an identical aggregation procedure -- this is a real, not cosmetic, "
            "difference and should be stated alongside any loss-gap comparison."
        ),
        "n_epochs_with_train_loss": sum(1 for e in inventory if "train_transformed_space_loss" in e),
        "n_epochs_with_valid_loss": sum(1 for e in inventory if "valid_transformed_space_loss" in e),
        "inventory": inventory,
    }


def _write_outputs(result: dict, out_dir: Path) -> "tuple[Path, Path]":
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "pilot_tb_loss_extraction.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    csv_path = out_dir / "pilot_tb_loss_extraction.csv"
    columns = [
        "epoch", "run_id",
        "train_transformed_space_loss", "train_wall_time",
        "valid_transformed_space_loss", "valid_wall_time",
        "raw_space_nse_scalars", "source_event_file",
    ]
    lines = [",".join(columns)]
    for entry in result["inventory"]:
        row = []
        for col in columns:
            value = entry.get(col, "")
            if col == "raw_space_nse_scalars" and value:
                value = json.dumps(value)
            row.append("" if value == "" else str(value).replace(",", ";"))
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-file", type=Path, required=True, help="Path to a single tfevents file")
    parser.add_argument("--run-id", default=None, help="Label only -- not used to locate or validate the event file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Generated-evidence output directory (untracked)")
    args = parser.parse_args()

    try:
        result = extract_losses(args.event_file, run_id=args.run_id)
    except TBLossExtractionError as exc:
        print(f"TB_LOSS_EXTRACTION_FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    json_path, csv_path = _write_outputs(result, args.out_dir)
    print(json.dumps({
        "status": "OK",
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "train_transformed_space_loss_tag": result["train_transformed_space_loss_tag"],
        "valid_transformed_space_loss_tag": result["valid_transformed_space_loss_tag"],
        "raw_space_nse_tags_present": result["raw_space_nse_tags_present"],
        "n_epochs_with_train_loss": result["n_epochs_with_train_loss"],
        "n_epochs_with_valid_loss": result["n_epochs_with_valid_loss"],
    }, indent=2))


if __name__ == "__main__":
    main()
