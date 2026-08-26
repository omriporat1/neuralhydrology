"""Stage 1 lead-6 optimization pilot: training/evaluation orchestration
(task item 6).

Composes, unmodified, every subsystem built for this pilot --
:mod:`src.baseline.pilot_lead06_config` (bundle/config generation),
:mod:`src.baseline.pilot_screening_eval` (provisional 400-basin screening),
:mod:`src.baseline.pilot_early_stopping` (restart-safe stopping state
machine), :mod:`src.baseline.pilot_tracking` (optional W&B logging), and
:mod:`src.baseline.pilot_evidence_bundle` (compact evidence write) -- plus
the exact NH entrypoints ``scripts/run_stage1_nh.py`` already wraps
(``neuralhydrology.nh_run.start_run`` / ``continue_run``). No modeling,
metric, or stopping logic is duplicated here; this module is purely the
"smallest practical launcher" the task's item 6 asks for.

Bounded-chunk design (matches the comment already committed in
``nh_config_generation._PILOT_LEAD06_BASE_PROFILE``): every pilot run's
frozen initial config trains only through epoch 6 (this pilot's
``stopping_eligible_from_epoch``). This module NEVER edits that frozen
profile to "restart" at a larger epoch count -- instead it extends training
past epoch 6 via NH's own ``continue_run`` plus a small ``epochs``-overlay
file, one bounded chunk at a time (each chunk advancing by
``screening_validation_every_n_epochs``), until either early stopping fires
or this pilot's 36-epoch sub-cap is reached.

Directory-nesting note: :func:`src.baseline.nh_config_generation.write_generated_config`
points NH's own ``run_dir`` config key at ``config_out_dir/runs``, but NH's
``start_run`` then creates one further nested, timestamped experiment
directory under that path at actual training time. :func:`discover_nh_run_dir`
locates that actual nested directory -- every other pilot function in this
task (``evaluate_screening_checkpoint``, ``record_screening_event``,
``write_pilot_evidence_bundle``, NH's own ``continue_run``/``start_evaluation``)
must be pointed at it, never at ``config_out_dir/runs`` itself.

Restart safety: this module keeps NO training-decision state of its own
beyond a small, purely-advisory ``pilot_orchestration_state.json`` (which
screening epochs have already been logged to the tracking backend, to avoid
duplicate W&B log entries on resume -- logging is append-only and not
otherwise idempotent). The actual source-of-truth restart state is always
re-derived from disk: which epoch was last actually trained comes from NH's
own checkpoint files; whether training should stop comes from
:mod:`src.baseline.pilot_early_stopping`'s own persisted, idempotent-replay
state. Calling :func:`run_pilot` again on a partially- or fully-completed
run is always safe: already-completed chunks are not retrained, already
-recorded screening epochs are not re-logged, and the full accumulated
screening history is always re-derived for the evidence bundle.

Nothing in this module is called against the real certified package, a real
training run, or Moriah anywhere in this task -- see
``docs/stage1_lead06_pilot_v001.md``'s "known limitations" section. It
exists so that a later, explicit Moriah launch (task item 7's sbatch script)
has a single, already-tested entrypoint to call, rather than ad hoc
per-run scripting under time pressure.

Explicit-evaluation correction (found during the first real Moriah
workflow-qualification run, ``emb128x64_seedA``, job 45695059): NH's own
``validate_every``-driven in-training validation does NOT reliably persist
the ``validation/model_epochNNN/validation_results.p`` pickle that
:func:`src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`
(via :func:`src.baseline.nh_seed_evaluation.raw_space_metrics_for_run_period`)
requires -- see ``docs/stage1_lead06_pilot_v001.md``'s qualification-run
section for the confirmed root cause and evidence. This module therefore
never assumes that pickle already exists at a screening checkpoint.
:func:`ensure_validation_results` checks
:func:`src.baseline.nh_seed_evaluation.period_results_path` (the single
canonical path helper, never independently reconstructed here) before every
screening-cadence epoch's :func:`~src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`
call: an already-saved result (from a prior explicit evaluation, or from
NH's own in-training validation on the epochs where it happens to have
worked) is reused unchanged; a missing one is produced by one explicit call
to an injectable ``evaluate_checkpoint_fn`` (mirroring ``train_chunk_fn``'s
seam -- :func:`default_evaluate_checkpoint` is the real NH call, exactly
``scripts/run_stage1_nh.py``'s own ``eval`` subcommand), and the pickle is
then required to exist or the run fails loudly rather than silently
screening against stale/absent data. This is purely an inference
prerequisite, never a second metric implementation or an authoritative
evaluation -- see ``pilot_screening_eval.py``'s corrected module docstring.

Continuation-epoch-semantics correction (found during the second real Moriah
qualification-run failure, ``emb128x64_seedA``, job 45705457, following the
first repair above): NH's own ``epochs`` config key is an ADDITIVE
epoch count on top of whatever epoch a continuation resumes from
(``neuralhydrology.training.basetrainer.BaseTrainer.train_and_validate``'s
``range(self._epoch + 1, self._epoch + self.cfg.epochs + 1)``), never an
absolute target -- so a continuation overlay that (incorrectly) wrote the
chunk's absolute target epoch into ``epochs`` trained far past the intended
chunk boundary (continuing from epoch 6 with ``epochs: 9`` produced
checkpoints 7-15, not 7-9). :class:`TrainChunkRequest` therefore carries an
explicit ``additional_epochs`` (always exactly ``logical_target_epoch -
current_epoch``, what ``default_train_chunk`` writes into the overlay's
``epochs`` key) and an explicit ``current_epoch`` (written into the overlay's
``continue_from_epoch`` key, a real, recognized NH ``Config`` property that
selects an EXACT continuation start epoch, overriding NH's default "highest
checkpoint in this directory" selection -- see
``neuralhydrology/training/basetrainer.py``'s ``_get_start_epoch_number`` /
``_restore_training_state``) -- never an ambiguous absolute ``target_epoch``
overlay value again.

NH also nests a fresh ``continue_training_from_epoch{start:03d}/``
subdirectory under whatever ``run_dir`` a ``continue_run`` call is pointed at
(``_create_folder_structure`` in the same module; nesting recurses
arbitrarily -- continuing again from an already-continued run_dir creates a
further-nested subdirectory), so a checkpoint's PHYSICAL owning directory is
not always ``nh_run_dir``/the base run directory. :func:`discover_physical_checkpoints`
walks base + every nested continuation directory to build a canonical
epoch -> physical-checkpoint inventory, failing loudly (never heuristically)
if more than one physical file ever claims the same logical epoch anywhere in
the tree. Because the real qualification run's own continuation directory
(``continue_training_from_epoch006``) still physically contains accidental
overshoot checkpoints 10-15 alongside the legitimate epoch-9 checkpoint (a
byproduct of the additive-``epochs`` bug above, now fixed prospectively but
not retroactively undone -- these files are never deleted/renamed/rewritten,
see module docstring's "never invent an unsafe workaround" note),
:func:`resolve_trusted_chunk_checkpoint` never treats "a checkpoint file with
this epoch number exists somewhere" as sufficient justification to reuse or
screen it: a chunk's target epoch is only trusted when its physical owning
directory is EXACTLY the continuation directory that a clean, chunk-sized
``continue_run`` call starting at this chunk's own ``previous_target_epoch``
would itself create (or the base run directory itself, only when that
directory's own ``start_run`` -- never ``continue_run`` -- already produced
the target epoch's checkpoint directly, e.g. the first chunk's target
already fully satisfied by a prior process's completed ``start_run`` call)
-- e.g. epoch 9 is trusted for chunk (6 -> 9) because its owning
directory is exactly ``continue_training_from_epoch006``, but epoch 12
sitting in that SAME directory is NOT trusted for chunk (9 -> 12), because
that directory was never continued from epoch 9. :func:`run_pilot_chunk`
never trains into an epoch range that a physical, untrusted checkpoint
already occupies (:func:`untrusted_overshoot_epochs`) -- doing so would
create a second, differently-nested physical file claiming the same logical
epoch. It also never re-attempts a ``train_chunk_fn`` call whose expected
continuation directory already exists but is untrusted/incomplete (verified
against ``neuralhydrology.training.basetrainer.BaseTrainer._create_folder_structure``,
which raises ``RuntimeError`` rather than resuming into or recreating an
already-existing run directory) -- this covers an interrupted continuation
attempt that was killed before producing even its first new checkpoint,
which :func:`untrusted_overshoot_epochs` alone would not catch. When either
condition occurs, the chunk is reported as ``blocked`` with a clear reason (never a raised exception that would look like an ordinary crash,
and never a silent resume from the highest physical checkpoint) --
:func:`run_pilot` stops advancing further chunks in that case, still writes
the evidence bundle reflecting the last successfully-processed epoch, and
:func:`compute_pilot_status_fields` reports ``safe_to_continue_automatically
= False`` with the specific overshoot epochs listed, requiring a human to
resolve the pre-existing overshoot artifacts before this pilot run can
safely proceed past its current logical frontier. See
``docs/stage1_lead06_pilot_v001.md``'s second-qualification-run-failure
section for the confirmed evidence this fix models.

Unconditional-nesting correction (found by inspecting
``neuralhydrology.nh_run.continue_run`` directly, not merely inferred from
the qualification run's evidence): ``continue_run`` sets
``base_config.is_continue_training = True`` unconditionally, on EVERY call,
regardless of whether ``continue_from_epoch`` is also set -- so
``_create_folder_structure``'s nesting behavior above is not specific to
"later" chunks. A genuinely PARTIAL first chunk (e.g. the base run's own
frozen ``epochs: 6`` profile was interrupted after only epochs 1-4) is, once
resumed via ``continue_run``, physically indistinguishable from any other
bounded continuation -- it nests a
``continue_training_from_epoch004/`` directory under the base run directory
just the same. :func:`run_pilot_chunk`'s ``previous_target_epoch == 0``
branch therefore only ever treats the base run directory itself as the
checkpoint's owning directory when the base profile's own target is ALREADY
fully satisfied on disk (``start_run`` produced it directly, no
``continue_run`` involved at all); any partial-first-chunk resumption is
routed through the exact same :func:`_advance_chunk_via_continuation` trust
logic as every later chunk, just with ``start_dir`` pinned to the base run
directory and ``resume_from_epoch`` set to the highest checkpoint already
found there (never the literal ``0``).

Explicit, run-specific overshoot adoption (added after a human review of the
real ``emb128x64_seedA`` overshoot artifacts from job 45705457, confirmed by
recovery/verification jobs 45718473/45718742/45721557): the untrusted-
overshoot guard above is, by design, permanent and unconditional for every
run -- it never reinterprets a checkpoint as trustworthy on its own. Adopting
a specific pre-existing overshoot checkpoint is therefore only ever possible
through an explicit, human-authored, per-run
:func:`load_accepted_continuation_manifest` file (``pilot_accepted_continuation.json``,
stored in the base run directory next to ``pilot_early_stopping_state.json``
-- never committed to git, never a general CLI override flag). Each entry
pins one epoch's model+optimizer checkpoint by exact relative path and
SHA-256; :func:`_advance_chunk_via_continuation` consults it only as a
fallback, and only for the exact ``chunk_target_epoch`` a given chunk call is
already trying to resolve -- so an accepted epoch 15 entry is never consulted
while epoch 12 is the chunk still being resolved, preserving strict epoch-12-
before-epoch-15 sequencing without any dedicated sequencing code (the
existing chunk-by-chunk loop in :func:`run_pilot` already provides it). A
manifest for the wrong ``run_id``, or any entry path resolving outside the
run directory, is rejected loudly at load time; a hash mismatch is rejected
loudly only when that specific epoch is actually consulted. This never
triggers training -- it only ever adopts a checkpoint that already exists on
disk -- and never changes the default strict behavior for any run without
such a manifest.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

import yaml

from .nh_config_generation import write_generated_config
from .nh_seed_evaluation import period_results_path
from .pilot_early_stopping import (
    build_effective_policy,
    load_or_init_pilot_state,
    pilot_best_checkpoint_epoch,
    record_screening_event,
)
from .pilot_evidence_bundle import write_pilot_evidence_bundle
from .pilot_lead06_config import (
    PilotPolicy,
    PilotRunSpec,
    build_pilot_bundle,
    resolve_pilot_run_spec,
)
from .pilot_screening_eval import (
    classify_screening_epoch_role,
    evaluate_screening_checkpoint,
    load_validated_screening_basin_ids,
)
from .pilot_tracking import (
    build_pilot_run_identity,
    finish_pilot_run,
    init_pilot_tracking_run,
    log_pilot_checkpoint_reference,
    log_pilot_screening_event,
)
from .splits import sha256_of

__all__ = [
    "PilotOrchestrationError",
    "TrainChunkRequest",
    "default_train_chunk",
    "EvaluationRequest",
    "root_logger_has_file_handler",
    "default_evaluate_checkpoint",
    "ensure_validation_results",
    "discover_nh_run_dir",
    "PhysicalCheckpoint",
    "discover_physical_checkpoints",
    "read_actual_optimizer_updates",
    "actual_optimizer_updates_by_epoch",
    "resolve_trusted_chunk_checkpoint",
    "untrusted_overshoot_epochs",
    "ACCEPTED_CONTINUATION_FILENAME",
    "AcceptedContinuationEntry",
    "load_accepted_continuation_manifest",
    "compute_pilot_status_fields",
    "CAP_IDENTITY_STATE_FILENAME",
    "enforce_pilot_cap_identity",
    "LR_IDENTITY_STATE_FILENAME",
    "enforce_pilot_learning_rate_identity",
    "HIDDEN_SIZE_IDENTITY_STATE_FILENAME",
    "enforce_pilot_hidden_size_identity",
    "EMBEDDING_DROPOUT_IDENTITY_STATE_FILENAME",
    "enforce_pilot_embedding_dropout_identity",
    "OUTPUT_DROPOUT_IDENTITY_STATE_FILENAME",
    "enforce_pilot_output_dropout_identity",
    "BATCH_SIZE_IDENTITY_STATE_FILENAME",
    "enforce_pilot_batch_size_identity",
    "SEQ_LENGTH_IDENTITY_STATE_FILENAME",
    "enforce_pilot_seq_length_identity",
    "DYNAMIC_INPUTS_IDENTITY_STATE_FILENAME",
    "enforce_pilot_dynamic_inputs_identity",
    "chunk_epoch_targets",
    "screening_epochs_in_chunk",
    "prepare_pilot_run",
    "PREPARATION_RESULT_FILENAME",
    "prepare_pilot_run_only",
    "run_pilot_chunk",
    "logged_screening_epochs",
    "PreparedPilotExecutionResult",
    "execute_prepared_pilot_run",
    "execute_prepared_pilot_run_monolithic",
    "run_pilot",
]

_CHECKPOINT_GLOB = "model_epoch*.pt"
_ORCHESTRATION_STATE_FILENAME = "pilot_orchestration_state.json"
_CONTINUATION_DIR_RE = re.compile(r"^continue_training_from_epoch(\d{3})$")
_CHECKPOINT_FILE_RE = re.compile(r"^model_epoch(\d{3})\.pt$")

# Name of the small JSON record persisted under the NH run directory once it
# exists, recording which (pilot_policy_name, run_id, max_updates_per_epoch)
# identity this run directory was first used for -- an always-active
# (W&B-independent) safeguard; see enforce_pilot_cap_identity.
CAP_IDENTITY_STATE_FILENAME = "pilot_cap_identity.json"

# Name of the small JSON record persisted under the NH run directory once it
# exists, recording which (pilot_policy_name, run_id, resolved_learning_rate)
# identity this run directory was first used for -- an always-active
# (W&B-independent) safeguard mirroring CAP_IDENTITY_STATE_FILENAME/
# enforce_pilot_cap_identity; see enforce_pilot_learning_rate_identity.
LR_IDENTITY_STATE_FILENAME = "pilot_lr_identity.json"

# Name of the small JSON record persisted under the NH run directory once it
# exists, recording which (pilot_policy_name, run_id, resolved_hidden_size)
# identity this run directory was first used for -- an always-active
# (W&B-independent) safeguard mirroring CAP_IDENTITY_STATE_FILENAME/
# LR_IDENTITY_STATE_FILENAME; see enforce_pilot_hidden_size_identity. Adopted
# for the Hidden-size-A range-characterization campaign.
HIDDEN_SIZE_IDENTITY_STATE_FILENAME = "pilot_hidden_size_identity.json"

# Name of the small JSON record persisted under the NH run directory once it
# exists, recording which (pilot_policy_name, run_id,
# resolved_embedding_dropout) identity this run directory was first used for
# -- an always-active (W&B-independent) safeguard mirroring
# CAP_IDENTITY_STATE_FILENAME/LR_IDENTITY_STATE_FILENAME/
# HIDDEN_SIZE_IDENTITY_STATE_FILENAME; see
# enforce_pilot_embedding_dropout_identity. Adopted for the
# Embedding-Dropout-A range-characterization campaign.
EMBEDDING_DROPOUT_IDENTITY_STATE_FILENAME = "pilot_embedding_dropout_identity.json"
OUTPUT_DROPOUT_IDENTITY_STATE_FILENAME = "pilot_output_dropout_identity.json"
BATCH_SIZE_IDENTITY_STATE_FILENAME = "pilot_batch_size_identity.json"

# Name of the small JSON record persisted under the NH run directory once it
# exists, recording which (pilot_policy_name, run_id, resolved_seq_length)
# identity this run directory was first used for -- an always-active
# (W&B-independent) safeguard mirroring CAP_IDENTITY_STATE_FILENAME/
# LR_IDENTITY_STATE_FILENAME/HIDDEN_SIZE_IDENTITY_STATE_FILENAME/
# EMBEDDING_DROPOUT_IDENTITY_STATE_FILENAME; see
# enforce_pilot_seq_length_identity. Adopted for the Sequence-Length-A
# range-characterization campaign.
SEQ_LENGTH_IDENTITY_STATE_FILENAME = "pilot_seq_length_identity.json"

# Name of the small JSON record persisted under the NH run directory once it
# exists, recording which (pilot_policy_name, run_id,
# resolved_dynamic_inputs) identity this run directory was first used for --
# an always-active (W&B-independent) safeguard mirroring
# CAP_IDENTITY_STATE_FILENAME/LR_IDENTITY_STATE_FILENAME/
# HIDDEN_SIZE_IDENTITY_STATE_FILENAME/
# EMBEDDING_DROPOUT_IDENTITY_STATE_FILENAME/SEQ_LENGTH_IDENTITY_STATE_FILENAME;
# see enforce_pilot_dynamic_inputs_identity. Adopted for the
# Dynamic-Input-Family-A range-characterization campaign.
DYNAMIC_INPUTS_IDENTITY_STATE_FILENAME = "pilot_dynamic_inputs_identity.json"

ACCEPTED_CONTINUATION_FILENAME = "pilot_accepted_continuation.json"


class PilotOrchestrationError(Exception):
    """Raised for an invalid orchestration request (unknown run_id, missing
    NH run directory, empty chunk schedule) -- never for an ordinary
    training/screening outcome."""


@dataclass(frozen=True)
class TrainChunkRequest:
    """One bounded NH training call this module asks its (injectable)
    ``train_chunk_fn`` to perform. ``is_first_chunk=True`` maps to
    ``neuralhydrology.nh_run.start_run(config_file=config_path)``;
    otherwise to ``neuralhydrology.nh_run.continue_run(run_dir=nh_run_dir,
    config_file=<small epochs-overlay file>)`` -- exactly
    ``scripts/run_stage1_nh.py``'s own train/continue behavior, never
    duplicated NH training logic.

    ``nh_run_dir`` for a continuation is the PHYSICAL checkpoint-owning
    directory training should resume from (never assumed to be the base run
    directory -- see module docstring's continuation-epoch-semantics note).
    ``current_epoch`` is the exact epoch to continue from -- written into
    the overlay's ``continue_from_epoch`` key when not ``None``, overriding
    NH's default "highest checkpoint in this directory" selection; ``None``
    only for the fully-degenerate corner where the base run directory has no
    checkpoint at all yet (see ``run_pilot_chunk``'s ``previous_target_epoch
    == 0`` branch and ``_advance_chunk_via_continuation``'s docstring) --
    every other continuation, including a partial first chunk, passes an
    explicit ``current_epoch``. ``additional_epochs`` is the exact (always additive,
    never absolute) count written into the overlay's ``epochs`` key.
    ``logical_target_epoch`` is retained only for logging/test clarity and
    is never itself written into any NH config."""

    is_first_chunk: bool
    config_path: Path
    nh_run_dir: "Path | None"
    current_epoch: "int | None"
    logical_target_epoch: int
    additional_epochs: int


def _continuation_overlay(request: TrainChunkRequest) -> dict:
    """The exact ``epochs``/``continue_from_epoch`` overlay dict a
    continuation chunk writes for NH -- pure and NH/torch-free so it can be
    unit-tested directly, independent of :func:`default_train_chunk`'s
    otherwise-untestable-locally NH call (this is the precise piece of logic
    responsible for both the additive-vs-absolute-epoch bug and the
    continuation-nesting bug documented in the module docstring -- it must
    never again change without a direct test catching it)."""
    overlay = {"epochs": request.additional_epochs}
    if request.current_epoch is not None:
        overlay["continue_from_epoch"] = request.current_epoch
    return overlay


def default_train_chunk(request: TrainChunkRequest) -> None:
    """Real NH training call -- lazy-imports neuralhydrology/torch so this
    module (and everything that composes it, including tests) stays
    importable without either installed. NEVER invoked by this task itself;
    a future real Moriah launch is the only intended caller. See module
    docstring."""
    from .nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    if request.is_first_chunk:
        from neuralhydrology.nh_run import start_run

        start_run(config_file=request.config_path)
    else:
        from neuralhydrology.nh_run import continue_run

        overlay = _continuation_overlay(request)
        overlay_path = Path(request.nh_run_dir) / "pilot_epoch_overlay.yaml"
        overlay_path.write_text(yaml.safe_dump(overlay), encoding="utf-8")
        continue_run(run_dir=request.nh_run_dir, config_file=overlay_path)


@dataclass(frozen=True)
class EvaluationRequest:
    """One explicit NH evaluation call this module asks its (injectable)
    ``evaluate_checkpoint_fn`` to perform, when
    :func:`ensure_validation_results` finds the expected saved-result pickle
    absent for a screening-cadence epoch. ``period`` is always
    ``"validation"`` here -- this pilot's screening path never touches the
    sealed temporal-test period (see ``pilot_screening_eval.py``)."""

    nh_run_dir: Path
    epoch: int
    period: str = "validation"


def root_logger_has_file_handler(log_path: "Path | str") -> bool:
    """True if the root logger already has a ``logging.FileHandler`` writing
    to ``log_path``.

    ``neuralhydrology.utils.logging_utils.setup_logging`` unconditionally
    opens a brand-new ``FileHandler`` (eagerly, holding open a real file
    descriptor) and ``StreamHandler`` on every call, then attaches both via
    ``logging.basicConfig(handlers=...)`` -- which is a documented no-op once
    the root logger already has any handlers. So a first call attaches NH's
    handlers as expected, but every subsequent call in the same process (this
    pilot's :func:`default_evaluate_checkpoint` may be called once per
    screening epoch, many times per process) still opens and leaks an extra,
    never-attached file descriptor pointed at the same ``output.log``, even
    though no duplicate log lines are produced. Checking for an
    already-attached handler on the same path first (rather than suppressing
    NH's logging outright) lets repeated evaluation calls skip only the
    redundant handler creation."""
    target = Path(log_path).resolve()
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == target:
            return True
    return False


def default_evaluate_checkpoint(request: EvaluationRequest) -> None:
    """Real NH evaluation call -- lazy-imports neuralhydrology/torch exactly
    like :func:`default_train_chunk`, so this module stays importable
    without either installed. Mirrors ``scripts/run_stage1_nh.py``'s own
    ``eval`` subcommand precisely (load ``run_dir/config.yml``, append to
    ``run_dir/output.log`` via the same ``setup_logging`` call, then
    ``start_evaluation``) rather than duplicating that behavior, except that
    ``setup_logging`` is skipped when :func:`root_logger_has_file_handler`
    shows this process already has it wired up (see that function's
    docstring) -- this pilot may call this function once per screening
    epoch within a single process, unlike the single-shot CLI it mirrors.
    NEVER invoked by this task's own tests; only a real Moriah resume calls
    it, and only when :func:`ensure_validation_results` finds the expected
    result pickle missing."""
    from .nh_register import register_flashnh_dataset

    register_flashnh_dataset()
    from neuralhydrology.evaluation.evaluate import start_evaluation
    from neuralhydrology.utils.config import Config
    from neuralhydrology.utils.logging_utils import setup_logging

    nh_run_dir = Path(request.nh_run_dir)
    config = Config(nh_run_dir / "config.yml")
    log_path = nh_run_dir / "output.log"
    if not root_logger_has_file_handler(log_path):
        setup_logging(str(log_path))
    start_evaluation(cfg=config, run_dir=nh_run_dir, epoch=request.epoch, period=request.period)


def ensure_validation_results(
    *,
    nh_run_dir,
    epoch: int,
    evaluate_checkpoint_fn: "Callable[[EvaluationRequest], None]" = default_evaluate_checkpoint,
) -> Path:
    """Restart-safe prerequisite check for one screening-cadence epoch's
    saved NH ``"validation"``-period result pickle (see module docstring's
    "Explicit-evaluation correction" note).

    Uses :func:`src.baseline.nh_seed_evaluation.period_results_path` as the
    single source of truth for the expected path -- never reconstructed
    independently here. If the pickle already exists (whether from NH's own
    in-training validation or a prior explicit evaluation call on an earlier
    invocation of this function), it is reused as-is and
    ``evaluate_checkpoint_fn`` is never called -- this is what makes resume
    avoid re-evaluating already-processed epochs. If absent,
    ``evaluate_checkpoint_fn`` is invoked exactly once; any exception it
    raises propagates unchanged (fails loudly, leaves no partial screening
    or early-stopping state behind, and is safely retryable on the next
    resume). After that call, the pickle is required to exist -- if it still
    doesn't, raises :class:`PilotOrchestrationError` rather than letting a
    silently-failed evaluation fall through to the metric reader.
    """
    result_path = period_results_path(nh_run_dir, "validation", epoch)
    if result_path.is_file():
        return result_path

    evaluate_checkpoint_fn(EvaluationRequest(nh_run_dir=Path(nh_run_dir), epoch=epoch))

    if not result_path.is_file():
        raise PilotOrchestrationError(
            f"explicit NH evaluation for epoch {epoch} did not produce the expected "
            f"validation result pickle: {result_path}"
        )
    return result_path


def discover_nh_run_dir(config_out_dir, experiment_name: str) -> Path:
    """Locate the actual, NH-created timestamped experiment directory under
    ``config_out_dir/runs`` (see module docstring's directory-nesting note).
    Raises if no matching directory exists yet. Raises loudly, listing every
    candidate, if more than one matches -- this is an ambiguous state (e.g. a
    stale directory from an earlier, abandoned attempt) that must be resolved
    by a human, never silently resolved by picking the newest one, since that
    could resume the wrong run."""
    runs_root = Path(config_out_dir) / "runs"
    if not runs_root.is_dir():
        raise PilotOrchestrationError(
            f"NH runs root does not exist yet: {runs_root} -- has the first training chunk run?"
        )
    candidates = sorted(
        (p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(experiment_name))
    )
    if not candidates:
        raise PilotOrchestrationError(
            f"no NH experiment directory found under {runs_root} whose name starts with {experiment_name!r}"
        )
    if len(candidates) > 1:
        raise PilotOrchestrationError(
            f"ambiguous NH experiment directory: {len(candidates)} candidates under {runs_root} "
            f"whose name starts with {experiment_name!r}, refusing to guess which is the real run: "
            f"{[str(c) for c in candidates]}"
        )
    return candidates[0]


def _try_discover_nh_run_dir(config_out_dir, experiment_name: str) -> "Path | None":
    try:
        return discover_nh_run_dir(config_out_dir, experiment_name)
    except PilotOrchestrationError:
        return None


def _last_completed_epoch(nh_run_dir) -> int:
    checkpoints = list(Path(nh_run_dir).glob(_CHECKPOINT_GLOB))
    if not checkpoints:
        return 0
    return max(int(p.stem.replace("model_epoch", "")) for p in checkpoints)


@dataclass(frozen=True)
class PhysicalCheckpoint:
    """One physical NH checkpoint file discovered anywhere under a base run
    directory (base dir itself, or a nested ``continue_training_from_epoch###``
    continuation directory at any depth). ``owning_run_dir`` is the exact
    directory the file lives in -- never inferred from the epoch number."""

    epoch: int
    path: Path
    owning_run_dir: Path


def discover_physical_checkpoints(base_run_dir) -> "dict[int, PhysicalCheckpoint]":
    """Recursively inventories every NH checkpoint file physically present
    under ``base_run_dir``: directly (the base/original run), and inside any
    nested ``continue_training_from_epoch###`` continuation directory NH's
    own ``continue_run`` creates (nesting recurses arbitrarily -- continuing
    again from an already-continued run_dir creates a further-nested
    directory; see module docstring).

    Only files matching NH's own exact ``model_epoch###.pt`` naming
    convention are collected, and only directories matching NH's own exact
    ``continue_training_from_epoch###`` naming convention are recursed into
    -- any other file or directory name is ignored, never guessed at.

    Raises :class:`PilotOrchestrationError` immediately if more than one
    physical file anywhere in the tree claims the same logical epoch number
    -- this is never resolved heuristically (by mtime, path depth, or
    directory name), since doing so could silently pick an untrustworthy
    checkpoint (see module docstring's overshoot note)."""
    base_run_dir = Path(base_run_dir)
    found: "dict[int, list[PhysicalCheckpoint]]" = {}

    def _scan(directory: Path) -> None:
        if not directory.is_dir():
            return
        for entry in sorted(directory.iterdir()):
            if entry.is_file():
                m = _CHECKPOINT_FILE_RE.match(entry.name)
                if m:
                    epoch = int(m.group(1))
                    found.setdefault(epoch, []).append(
                        PhysicalCheckpoint(epoch=epoch, path=entry, owning_run_dir=directory)
                    )
            elif entry.is_dir() and _CONTINUATION_DIR_RE.match(entry.name):
                _scan(entry)

    _scan(base_run_dir)

    duplicates = {epoch: cands for epoch, cands in found.items() if len(cands) > 1}
    if duplicates:
        details = "; ".join(
            f"epoch {epoch}: {[str(c.path) for c in cands]}" for epoch, cands in sorted(duplicates.items())
        )
        raise PilotOrchestrationError(
            f"ambiguous physical checkpoint inventory under {base_run_dir}: more than one file claims "
            f"the same logical epoch, refusing to guess which is authoritative -- {details}"
        )
    return {epoch: cands[0] for epoch, cands in found.items()}


def read_actual_optimizer_updates(optimizer_state_path) -> int:
    """Read the exact cumulative number of ``optimizer.step()`` calls
    performed through this checkpoint, straight from PyTorch's own
    unconditionally-saved optimizer state file (``optimizer_state_epochNNN.pt``,
    written by NH's ``BaseTrainer._save_weights_and_optimizer`` alongside
    every ``model_epochNNN.pt`` -- see ``neuralhydrology/training/basetrainer.py``).

    This is genuine structured evidence, not an inference: NH's only two
    supported optimizers (``torch.optim.Adam``/``AdamW``, see
    ``neuralhydrology/training/__init__.py``'s ``get_optimizer``) both
    maintain a per-parameter ``state[p]['step']`` counter that PyTorch itself
    increments exactly once per ``optimizer.step()`` call for that parameter
    -- i.e. exactly once per non-NaN training batch (NH's own
    ``BaseTrainer._train_epoch`` skips ``optimizer.step()``, but never
    ``experiment_logger.log_step()``, on a NaN-loss batch), making this a
    MORE precise source than NH's in-memory ``Logger.update`` counter, which
    increments on every batch including NaN-skipped ones. The value is
    cumulative since this run directory's first ``start_run`` call --
    optimizer state, and therefore this counter, survives every
    ``continue_run`` via ``BaseTrainer._restore_training_state``'s
    ``optimizer.load_state_dict``.

    Requires no modification to vendored/installed NH core code -- this only
    reads an artifact NH already writes unconditionally today. Lazily
    imports torch (never imported at this module's import time), so this
    module stays importable without torch installed; only a caller that
    actually invokes this function against a real optimizer-state file needs
    torch present.

    Raises :class:`PilotOrchestrationError` if the file is missing, contains
    no per-parameter state, or -- an anomaly worth surfacing loudly, never
    silently averaged or guessed over -- different parameters disagree on
    their step count.
    """
    import torch

    optimizer_state_path = Path(optimizer_state_path)
    if not optimizer_state_path.is_file():
        raise PilotOrchestrationError(
            f"optimizer state file not found: {optimizer_state_path} -- cannot read actual "
            "optimizer-update evidence for this checkpoint"
        )
    state_dict = torch.load(optimizer_state_path, map_location="cpu")
    param_states = state_dict.get("state", {})
    if not param_states:
        raise PilotOrchestrationError(
            f"optimizer state file {optimizer_state_path} contains no per-parameter state -- "
            "cannot read actual optimizer-update evidence"
        )

    steps = set()
    for param_state in param_states.values():
        step = param_state["step"]
        steps.add(int(step.item()) if hasattr(step, "item") else int(step))

    if len(steps) != 1:
        raise PilotOrchestrationError(
            f"optimizer state file {optimizer_state_path} has disagreeing per-parameter step "
            f"counts {sorted(steps)} -- refusing to guess which is authoritative"
        )
    return steps.pop()


def actual_optimizer_updates_by_epoch(nh_run_dir) -> "dict[int, int]":
    """Cumulative actual-optimizer-update evidence (see
    :func:`read_actual_optimizer_updates`) for every physically-checkpointed
    epoch under ``nh_run_dir`` (base run directory plus any nested
    continuation directory -- see :func:`discover_physical_checkpoints`).

    Requires torch. NOT called automatically by
    :func:`~src.baseline.pilot_evidence_bundle.write_pilot_evidence_bundle`
    (which stays torch-free so every existing local/test caller -- all of
    which use byte-content fake checkpoint files, never real torch state --
    is unaffected). A real Moriah launch calls this explicitly and passes
    its result into that function's optional
    ``actual_optimizer_updates_by_epoch`` parameter."""
    inventory = discover_physical_checkpoints(nh_run_dir)
    return {
        epoch: read_actual_optimizer_updates(ckpt.path.parent / f"optimizer_state_epoch{epoch:03d}.pt")
        for epoch, ckpt in inventory.items()
    }


def _expected_continuation_dir(previous_checkpoint_dir: Path, previous_target_epoch: int) -> Path:
    """The exact physical directory a clean, chunk-sized ``continue_run``
    call -- continuing from ``previous_target_epoch``'s checkpoint, which
    physically lives in ``previous_checkpoint_dir`` -- would itself create
    (or, for ``previous_target_epoch == 0``, the base run directory itself,
    since the first bounded chunk's checkpoints live there directly)."""
    if previous_target_epoch == 0:
        return Path(previous_checkpoint_dir)
    return Path(previous_checkpoint_dir) / f"continue_training_from_epoch{previous_target_epoch:03d}"


def resolve_trusted_chunk_checkpoint(
    inventory: "dict[int, PhysicalCheckpoint]",
    previous_checkpoint_dir: Path,
    previous_target_epoch: int,
    epoch: int,
) -> "PhysicalCheckpoint | None":
    """Returns ``epoch``'s physical checkpoint IFF it is trusted -- i.e. its
    owning directory is EXACTLY the directory a clean continuation started
    at ``previous_target_epoch`` (from ``previous_checkpoint_dir``) would
    itself create. Returns ``None`` otherwise, whether ``epoch`` is truly
    absent from the inventory or merely present as untrusted overshoot in
    some OTHER (already-existing, differently-started) directory -- see
    module docstring's worked example distinguishing epoch 9 (trusted) from
    epoch 12 (untrusted overshoot in the same physical directory)."""
    ckpt = inventory.get(epoch)
    if ckpt is None:
        return None
    expected_dir = _expected_continuation_dir(previous_checkpoint_dir, previous_target_epoch)
    if ckpt.owning_run_dir != expected_dir:
        return None
    return ckpt


def untrusted_overshoot_epochs(
    inventory: "dict[int, PhysicalCheckpoint]", previous_target_epoch: int, chunk_target_epoch: int
) -> "list[int]":
    """Physical checkpoint epochs strictly within
    ``(previous_target_epoch, chunk_target_epoch]`` that already exist on
    disk. Only meaningful to call once :func:`resolve_trusted_chunk_checkpoint`
    has already found ``chunk_target_epoch`` untrusted (or absent): if the
    target itself isn't the trusted product of a continuation cleanly
    started at ``previous_target_epoch``, then by construction NO checkpoint
    in this range can be either (a legitimate continuation from
    ``previous_target_epoch`` would have produced ALL of them together, in
    the one new directory that check just found missing/mismatched) -- so
    any physically-present epoch here is untrusted overshoot that a fresh
    training attempt would collide with."""
    return sorted(e for e in inventory if previous_target_epoch < e <= chunk_target_epoch)


@dataclass(frozen=True)
class AcceptedContinuationEntry:
    """One human-reviewed, run-specific accepted-checkpoint entry from an
    ``ACCEPTED_CONTINUATION_FILENAME`` manifest (see module docstring's
    "Explicit, run-specific overshoot adoption" note). Paths are already
    resolved to absolute, containment-checked locations; hashes are the
    manifest's claimed SHA-256 values, not yet verified against the real
    files (verification happens lazily, only when this entry's epoch is
    actually consulted -- see :func:`_resolve_accepted_checkpoint`)."""

    model_path: Path
    model_sha256: str
    optimizer_path: Path
    optimizer_sha256: str


def load_accepted_continuation_manifest(
    nh_run_dir, run_id: str
) -> "dict[int, AcceptedContinuationEntry]":
    """Load this run's explicit continuation-adoption manifest, if present.

    Returns ``{}`` when no ``ACCEPTED_CONTINUATION_FILENAME`` file exists in
    ``nh_run_dir`` -- adoption is opt-in per run, and the default
    :func:`resolve_trusted_chunk_checkpoint` / :func:`untrusted_overshoot_epochs`
    guard is completely unaffected for every run without one.

    This is deliberately not a general override mechanism: the manifest's own
    ``run_id`` must exactly match ``run_id`` (a manifest left in, or copied
    to, the wrong run directory raises rather than silently being honored),
    every entry's ``model_path``/``optimizer_path`` must resolve strictly
    inside ``nh_run_dir`` (never an absolute path elsewhere on disk, never a
    ``..`` escape), and both paths must share the manifest's declared
    ``accepted_directory`` as their parent (catches an internally
    inconsistent manifest rather than trusting one entry's directory over
    another's), and must be named exactly ``model_epoch{E:03d}.pt`` /
    ``optimizer_state_epoch{E:03d}.pt`` for its own key epoch ``E`` (an entry
    keyed epoch 12 pointing at correctly-hashed epoch 15 files is rejected --
    a hash alone does not bind an entry to the epoch it is meant to
    authenticate). SHA-256 hashes are recorded here but not yet compared
    against the real files -- see :func:`_resolve_accepted_checkpoint`,
    called only for the one epoch a chunk is actually trying to resolve.
    """
    manifest_path = Path(nh_run_dir) / ACCEPTED_CONTINUATION_FILENAME
    if not manifest_path.is_file():
        return {}

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("schema_version") != 1:
        raise PilotOrchestrationError(
            f"accepted-continuation manifest {manifest_path} has unsupported "
            f"schema_version={manifest.get('schema_version')!r}"
        )
    if manifest.get("run_id") != run_id:
        raise PilotOrchestrationError(
            f"accepted-continuation manifest {manifest_path} is for "
            f"run_id={manifest.get('run_id')!r}, but this run is {run_id!r} -- "
            "refusing to use a manifest reviewed for a different run"
        )

    run_dir_resolved = Path(nh_run_dir).resolve()
    accepted_directory = manifest.get("accepted_directory")
    if not accepted_directory:
        raise PilotOrchestrationError(
            f"accepted-continuation manifest {manifest_path} is missing 'accepted_directory'"
        )
    expected_parent = (run_dir_resolved / accepted_directory).resolve()
    if not expected_parent.is_relative_to(run_dir_resolved):
        raise PilotOrchestrationError(
            f"accepted-continuation manifest {manifest_path} 'accepted_directory' "
            f"resolves outside the run directory: {expected_parent}"
        )

    entries: "dict[int, AcceptedContinuationEntry]" = {}
    for epoch_str, raw_entry in manifest.get("accepted_checkpoints", {}).items():
        try:
            epoch = int(epoch_str)
            model_path = (run_dir_resolved / raw_entry["model_path"]).resolve()
            optimizer_path = (run_dir_resolved / raw_entry["optimizer_path"]).resolve()
            model_sha256 = raw_entry["model_sha256"]
            optimizer_sha256 = raw_entry["optimizer_sha256"]
        except (KeyError, ValueError) as exc:
            raise PilotOrchestrationError(
                f"accepted-continuation manifest {manifest_path} has a malformed entry "
                f"for epoch {epoch_str!r}: {exc}"
            ) from exc

        for candidate in (model_path, optimizer_path):
            if not candidate.is_relative_to(run_dir_resolved):
                raise PilotOrchestrationError(
                    f"accepted-continuation manifest {manifest_path} entry for epoch "
                    f"{epoch} points outside the run directory: {candidate}"
                )
            if candidate.parent != expected_parent:
                raise PilotOrchestrationError(
                    f"accepted-continuation manifest {manifest_path} entry for epoch "
                    f"{epoch} is not inside its own declared accepted_directory "
                    f"({expected_parent}): {candidate}"
                )

        expected_model_name = f"model_epoch{epoch:03d}.pt"
        expected_optimizer_name = f"optimizer_state_epoch{epoch:03d}.pt"
        if model_path.name != expected_model_name or optimizer_path.name != expected_optimizer_name:
            raise PilotOrchestrationError(
                f"accepted-continuation manifest {manifest_path} entry keyed for epoch "
                f"{epoch} must point to {expected_model_name!r}/{expected_optimizer_name!r}, "
                f"not {model_path.name!r}/{optimizer_path.name!r} -- an entry's key epoch "
                "must match the epoch of the files it authenticates, otherwise a correctly "
                "hashed file for a DIFFERENT epoch could be silently substituted"
            )

        entries[epoch] = AcceptedContinuationEntry(
            model_path=model_path,
            model_sha256=model_sha256,
            optimizer_path=optimizer_path,
            optimizer_sha256=optimizer_sha256,
        )
    return entries


def _resolve_accepted_checkpoint(entry: AcceptedContinuationEntry, epoch: int) -> Path:
    """Verify one manifest entry's model+optimizer SHA-256 hashes against the
    real files on disk before trusting it -- both artifacts are required, and
    a missing file or hash mismatch raises rather than silently falling back
    to the default blocked status, since that would mask exactly the
    discrepancy this manifest was reviewed to rule out. Returns the shared
    owning directory (the manifest's ``accepted_directory``, already
    verified to be both paths' parent) on success."""
    for path, expected_hash, label in (
        (entry.model_path, entry.model_sha256, "model"),
        (entry.optimizer_path, entry.optimizer_sha256, "optimizer"),
    ):
        if not path.is_file():
            raise PilotOrchestrationError(
                f"accepted-continuation manifest entry for epoch {epoch} references a "
                f"missing {label} file: {path}"
            )
        actual_hash = sha256_of(path)
        if actual_hash != expected_hash:
            raise PilotOrchestrationError(
                f"accepted-continuation manifest entry for epoch {epoch} {label} hash "
                f"mismatch: expected {expected_hash}, got {actual_hash} for {path}"
            )
    return entry.model_path.parent


def compute_pilot_status_fields(nh_run_dir, pilot_policy: "PilotPolicy | None" = None) -> dict:
    """Human/launcher-facing status snapshot (task item 7) distinguishing
    the highest PHYSICAL checkpoint epoch on disk from the highest
    logically-screened (stopping-eligible-recorded) epoch, from the next
    epoch this pilot's schedule intends to screen next, and from any
    untrusted overshoot epochs sitting beyond the logical frontier.

    Computed directly from disk state (physical checkpoint inventory +
    this pilot's own restart-safe early-stopping state file), independent
    of any particular :func:`run_pilot` call's in-memory result -- safe to
    call at any time, including after a wall-time SIGTERM killed the pilot
    process before it could print its own JSON result (see
    ``scripts/run_stage1_lead06_pilot_moriah.sbatch``'s fallback status
    path). ``pilot_policy`` is optional: when given,
    ``next_intended_screening_epoch`` is also computed from this pilot's own
    chunk schedule; when omitted (the sbatch fallback path does not have
    convenient access to the full policy object), it is reported as
    ``None``.

    Never asserts ``safe_to_continue_automatically`` unless the run is
    neither already stopped nor blocked by untrusted overshoot -- "a
    checkpoint exists past the logical frontier" must never be read as
    "screening progressed that far" (see module docstring)."""
    nh_run_dir = Path(nh_run_dir)
    inventory = discover_physical_checkpoints(nh_run_dir)
    highest_physical_checkpoint_epoch = max(inventory) if inventory else None

    es_path = nh_run_dir / "pilot_early_stopping_state.json"
    highest_screened_epoch = None
    stopped = False
    if es_path.is_file():
        with open(es_path, "r", encoding="utf-8") as fh:
            es_state = json.load(fh)
        history = es_state.get("history", [])
        if history:
            highest_screened_epoch = max(entry["epoch"] for entry in history)
        stopped = bool(es_state.get("stopped"))

    if highest_screened_epoch is not None:
        overshoot_epochs = sorted(e for e in inventory if e > highest_screened_epoch)
    else:
        overshoot_epochs = []

    next_intended_screening_epoch = None
    if pilot_policy is not None and highest_screened_epoch is not None and not stopped:
        effective_policy = build_effective_policy(pilot_policy)
        targets = chunk_epoch_targets(pilot_policy, effective_policy["max_epoch_budget"])
        remaining = [t for t in targets if t > highest_screened_epoch]
        next_intended_screening_epoch = remaining[0] if remaining else None

    safe_to_continue_automatically = (
        not stopped
        and not overshoot_epochs
        and highest_screened_epoch is not None
        and (pilot_policy is None or next_intended_screening_epoch is not None)
    )

    return {
        "highest_physical_checkpoint_epoch": highest_physical_checkpoint_epoch,
        "highest_screened_epoch": highest_screened_epoch,
        "next_intended_screening_epoch": next_intended_screening_epoch,
        "overshoot_epochs": overshoot_epochs,
        "stopped": stopped,
        "safe_to_continue_automatically": safe_to_continue_automatically,
    }


def _load_orchestration_state(nh_run_dir) -> dict:
    path = Path(nh_run_dir) / _ORCHESTRATION_STATE_FILENAME
    if not path.is_file():
        return {"logged_screening_epochs": []}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_orchestration_state(nh_run_dir, state: dict) -> None:
    path = Path(nh_run_dir) / _ORCHESTRATION_STATE_FILENAME
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def logged_screening_epochs(nh_run_dir) -> "list[int]":
    """Public accessor for the durable, restart-safe set of screening epochs
    already logged for ``nh_run_dir`` (``pilot_orchestration_state.json``'s
    ``logged_screening_epochs`` -- see :func:`run_pilot_chunk`'s docstring).
    This is every screening-cadence epoch ever processed across every past
    invocation for this run directory, both ``diagnostic_only`` and
    ``stopping_eligible`` roles -- the authoritative source for
    full-history reconstruction. The public entry point external consumers
    should use instead of reaching into ``_load_orchestration_state``
    directly."""
    state = _load_orchestration_state(nh_run_dir)
    return sorted(state["logged_screening_epochs"])


def _load_scalar_identity_record(nh_run_dir, state_filename: str) -> "dict | None":
    path = Path(nh_run_dir) / state_filename
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_scalar_identity_record(nh_run_dir, state_filename: str, record: dict) -> None:
    path = Path(nh_run_dir) / state_filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _enforce_pilot_scalar_identity(
    *,
    run_identity: dict,
    nh_run_dir: "str | Path | None",
    state_filename: str,
    identity_key: str,
    axis_label: str,
    contradiction_detail: str,
) -> None:
    """Shared engine underneath every ``enforce_pilot_*_identity`` wrapper
    (cap, learning-rate, hidden-size, embedding-dropout, seq-length): each
    wrapper is the documented/tested public entry point and owns its own
    state filename, ``run_identity`` key, axis label, and contradiction
    detail text; this function only implements the common
    load-if-absent/compare-if-present/raise-loudly-on-mismatch mechanics
    shared by all of them. Not part of the public API -- see any wrapper's
    docstring (e.g. :func:`enforce_pilot_cap_identity`) for the full
    always-active, W&B-independent safeguard contract this implements."""
    if nh_run_dir is None:
        return
    nh_run_dir = Path(nh_run_dir)
    if not nh_run_dir.is_dir():
        return

    current = {
        "pilot_policy_name": run_identity["pilot_policy_name"],
        "run_id": run_identity["run_id"],
        identity_key: run_identity[identity_key],
    }
    record = _load_scalar_identity_record(nh_run_dir, state_filename)
    if record is None:
        _save_scalar_identity_record(nh_run_dir, state_filename, current)
        return

    if record != current:
        raise PilotOrchestrationError(
            f"NH run directory {nh_run_dir} already has a persisted {axis_label} "
            f"identity {record!r}, which contradicts this call's identity {current!r} "
            f"-- {contradiction_detail}"
        )


def enforce_pilot_cap_identity(*, run_identity: dict, nh_run_dir: "str | Path | None") -> None:
    """Always-active, W&B-independent safeguard: a candidate's
    ``max_updates_per_epoch`` -- capped (positive int) or uncapped (``None``)
    -- is this run directory's frozen identity for its entire on-disk
    lifetime, exactly like ``pilot_policy_name``/``run_id``. Unlike
    ``pilot_tracking.resolve_pilot_wandb_run_id``'s contradiction check
    (active only when W&B tracking is enabled, which is NOT this pilot's
    default -- see ``pilot_tracking.py``'s module docstring), this check
    always runs: a cap-identity contradiction is a training-safety concern,
    not a tracking concern, so it must be caught even when tracking is
    disabled.

    If ``nh_run_dir`` does not exist yet (this candidate's very first call,
    before any NH run directory has been created), this function persists
    nothing and returns -- there is nothing yet to contradict. Once a run
    directory exists, the first call for it persists the current
    ``run_identity``'s ``(pilot_policy_name, run_id, max_updates_per_epoch)``
    triple; every later call for the same directory must match it exactly,
    or :class:`PilotOrchestrationError` is raised -- before any further
    tracking or training call. See this function's call site in
    :func:`run_pilot` (right after ``existing_nh_run_dir`` is discovered,
    before ``init_pilot_tracking_run`` and before any chunk is trained) and
    the module docstring's restart-safety discipline."""
    _enforce_pilot_scalar_identity(
        run_identity=run_identity,
        nh_run_dir=nh_run_dir,
        state_filename=CAP_IDENTITY_STATE_FILENAME,
        identity_key="max_updates_per_epoch",
        axis_label="cap",
        contradiction_detail=(
            "max_updates_per_epoch (capped vs uncapped, or two different int caps) "
            "must never change across a continuation of the same run directory"
        ),
    )


def enforce_pilot_learning_rate_identity(*, run_identity: dict, nh_run_dir: "str | Path | None") -> None:
    """Always-active, W&B-independent safeguard for the LR-A range-
    characterization campaign: a candidate's resolved learning rate is this
    run directory's frozen identity for its entire on-disk lifetime, exactly
    like ``pilot_policy_name``/``run_id`` and mirroring
    :func:`enforce_pilot_cap_identity`'s design (a learning-rate-identity
    contradiction is a training-safety concern, not a tracking concern, so it
    must be caught even when W&B tracking is disabled -- this pilot's
    default).

    Compares ``run_identity["resolved_learning_rate"]`` (see
    ``pilot_tracking.build_pilot_run_identity``), never the
    ``learning_rate_override`` field alone -- an override and an
    unset-override-that-resolves-to-the-same-profile-value are the same
    training identity, but two different resolved values (whether from
    different overrides, or an override vs. a differing profile default) are
    always a contradiction.

    If ``nh_run_dir`` does not exist yet (this candidate's very first call,
    before any NH run directory has been created), this function persists
    nothing and returns -- there is nothing yet to contradict. Once a run
    directory exists, the first call for it persists the current
    ``run_identity``'s ``(pilot_policy_name, run_id, resolved_learning_rate)``
    triple; every later call for the same directory must match it exactly,
    or :class:`PilotOrchestrationError` is raised -- before any further
    tracking or training call. See this function's call site in
    :func:`run_pilot`, immediately alongside ``enforce_pilot_cap_identity``."""
    _enforce_pilot_scalar_identity(
        run_identity=run_identity,
        nh_run_dir=nh_run_dir,
        state_filename=LR_IDENTITY_STATE_FILENAME,
        identity_key="resolved_learning_rate",
        axis_label="learning-rate",
        contradiction_detail=(
            "resolved_learning_rate must never change across a continuation of "
            "the same run directory"
        ),
    )


def enforce_pilot_hidden_size_identity(*, run_identity: dict, nh_run_dir: "str | Path | None") -> None:
    """Always-active, W&B-independent safeguard for the Hidden-size-A range-
    characterization campaign: a candidate's resolved hidden size is this run
    directory's frozen identity for its entire on-disk lifetime, exactly like
    ``pilot_policy_name``/``run_id`` and mirroring
    :func:`enforce_pilot_cap_identity`/:func:`enforce_pilot_learning_rate_identity`'s
    design (a hidden-size-identity contradiction is a training-safety
    concern, not a tracking concern, so it must be caught even when W&B
    tracking is disabled -- this pilot's default).

    Compares ``run_identity["resolved_hidden_size"]`` (see
    ``pilot_tracking.build_pilot_run_identity``), never the
    ``hidden_size_override`` field alone -- an override and an
    unset-override-that-resolves-to-the-same-profile-value are the same
    training identity, but two different resolved values (whether from
    different overrides, or an override vs. a differing profile default) are
    always a contradiction.

    If ``nh_run_dir`` does not exist yet (this candidate's very first call,
    before any NH run directory has been created), this function persists
    nothing and returns -- there is nothing yet to contradict. Once a run
    directory exists, the first call for it persists the current
    ``run_identity``'s ``(pilot_policy_name, run_id, resolved_hidden_size)``
    triple; every later call for the same directory must match it exactly,
    or :class:`PilotOrchestrationError` is raised -- before any further
    tracking or training call. See this function's call site in
    :func:`run_pilot`, immediately alongside ``enforce_pilot_cap_identity``
    and ``enforce_pilot_learning_rate_identity``."""
    _enforce_pilot_scalar_identity(
        run_identity=run_identity,
        nh_run_dir=nh_run_dir,
        state_filename=HIDDEN_SIZE_IDENTITY_STATE_FILENAME,
        identity_key="resolved_hidden_size",
        axis_label="hidden-size",
        contradiction_detail=(
            "resolved_hidden_size must never change across a continuation of "
            "the same run directory"
        ),
    )


def enforce_pilot_embedding_dropout_identity(*, run_identity: dict, nh_run_dir: "str | Path | None") -> None:
    """Always-active, W&B-independent safeguard for the Embedding-Dropout-A
    range-characterization campaign: a candidate's resolved
    ``statics_embedding`` dropout is this run directory's frozen identity for
    its entire on-disk lifetime, exactly like ``pilot_policy_name``/``run_id``
    and mirroring
    :func:`enforce_pilot_cap_identity`/:func:`enforce_pilot_learning_rate_identity`/
    :func:`enforce_pilot_hidden_size_identity`'s design (an embedding-dropout-
    identity contradiction is a training-safety concern, not a tracking
    concern, so it must be caught even when W&B tracking is disabled -- this
    pilot's default).

    Compares ``run_identity["resolved_embedding_dropout"]`` (see
    ``pilot_tracking.build_pilot_run_identity``), never the
    ``embedding_dropout_override`` field alone -- an override and an
    unset-override-that-resolves-to-the-same-profile-value are the same
    training identity, but two different resolved values (whether from
    different overrides, or an override vs. a differing profile default) are
    always a contradiction. ``0.0`` (the drop00 candidate) is a valid,
    distinct-from-``None`` resolved value here -- the comparison is a plain
    dict-equality check, never a truthiness check, so it is never confused
    with an unset/omitted value.

    If ``nh_run_dir`` does not exist yet (this candidate's very first call,
    before any NH run directory has been created), this function persists
    nothing and returns -- there is nothing yet to contradict. Once a run
    directory exists, the first call for it persists the current
    ``run_identity``'s ``(pilot_policy_name, run_id,
    resolved_embedding_dropout)`` triple; every later call for the same
    directory must match it exactly, or :class:`PilotOrchestrationError` is
    raised -- before any further tracking or training call. See this
    function's call site in :func:`run_pilot`, immediately alongside
    ``enforce_pilot_cap_identity``, ``enforce_pilot_learning_rate_identity``,
    and ``enforce_pilot_hidden_size_identity``."""
    _enforce_pilot_scalar_identity(
        run_identity=run_identity,
        nh_run_dir=nh_run_dir,
        state_filename=EMBEDDING_DROPOUT_IDENTITY_STATE_FILENAME,
        identity_key="resolved_embedding_dropout",
        axis_label="embedding-dropout",
        contradiction_detail=(
            "resolved_embedding_dropout must never change across a continuation of "
            "the same run directory"
        ),
    )


def enforce_pilot_output_dropout_identity(*, run_identity: dict, nh_run_dir: "str | Path | None") -> None:
    """Keep resolved CUDALSTM output dropout immutable across a continuation."""
    _enforce_pilot_scalar_identity(
        run_identity=run_identity, nh_run_dir=nh_run_dir,
        state_filename=OUTPUT_DROPOUT_IDENTITY_STATE_FILENAME,
        identity_key="resolved_output_dropout", axis_label="output-dropout",
        contradiction_detail="resolved_output_dropout must never change across a continuation of the same run directory",
    )


def enforce_pilot_batch_size_identity(*, run_identity: dict, nh_run_dir: "str | Path | None") -> None:
    """Keep resolved NH DataLoader batch size immutable across a continuation."""
    _enforce_pilot_scalar_identity(
        run_identity=run_identity, nh_run_dir=nh_run_dir,
        state_filename=BATCH_SIZE_IDENTITY_STATE_FILENAME,
        identity_key="resolved_batch_size", axis_label="batch-size",
        contradiction_detail="resolved_batch_size must never change across a continuation of the same run directory",
    )


def enforce_pilot_seq_length_identity(*, run_identity: dict, nh_run_dir: "str | Path | None") -> None:
    """Always-active, W&B-independent safeguard for the Sequence-Length-A
    range-characterization campaign: a candidate's resolved ``seq_length`` is
    this run directory's frozen identity for its entire on-disk lifetime,
    exactly like ``pilot_policy_name``/``run_id`` and mirroring
    :func:`enforce_pilot_cap_identity`/:func:`enforce_pilot_learning_rate_identity`/
    :func:`enforce_pilot_hidden_size_identity`/
    :func:`enforce_pilot_embedding_dropout_identity`'s design (a
    seq-length-identity contradiction is a training-safety concern, not a
    tracking concern, so it must be caught even when W&B tracking is
    disabled -- this pilot's default).

    Compares ``run_identity["resolved_seq_length"]`` (see
    ``pilot_tracking.build_pilot_run_identity``), never the
    ``seq_length_override`` field alone -- an override and an
    unset-override-that-resolves-to-the-same-policy-default are the same
    training identity, but two different resolved values are always a
    contradiction.

    If ``nh_run_dir`` does not exist yet (this candidate's very first call,
    before any NH run directory has been created), this function persists
    nothing and returns -- there is nothing yet to contradict. Once a run
    directory exists, the first call for it persists the current
    ``run_identity``'s ``(pilot_policy_name, run_id, resolved_seq_length)``
    triple; every later call for the same directory must match it exactly,
    or :class:`PilotOrchestrationError` is raised -- before any further
    tracking or training call. See this function's call site in
    :func:`run_pilot`, immediately alongside the other four scalar-identity
    guards."""
    _enforce_pilot_scalar_identity(
        run_identity=run_identity,
        nh_run_dir=nh_run_dir,
        state_filename=SEQ_LENGTH_IDENTITY_STATE_FILENAME,
        identity_key="resolved_seq_length",
        axis_label="seq-length",
        contradiction_detail=(
            "resolved_seq_length must never change across a continuation of "
            "the same run directory"
        ),
    )


def enforce_pilot_dynamic_inputs_identity(*, run_identity: dict, nh_run_dir: "str | Path | None") -> None:
    """Always-active, W&B-independent safeguard for the Dynamic-Input-Family-A
    range-characterization campaign: a candidate's resolved
    ``dynamic_inputs`` list is this run directory's frozen identity for its
    entire on-disk lifetime, exactly like ``pilot_policy_name``/``run_id``
    and mirroring :func:`enforce_pilot_cap_identity`/
    :func:`enforce_pilot_learning_rate_identity`/
    :func:`enforce_pilot_hidden_size_identity`/
    :func:`enforce_pilot_embedding_dropout_identity`/
    :func:`enforce_pilot_seq_length_identity`'s design (a dynamic-inputs
    identity contradiction is a training-safety concern, not a tracking
    concern, so it must be caught even when W&B tracking is disabled --
    this pilot's default).

    Compares ``run_identity["resolved_dynamic_inputs"]`` (see
    ``pilot_tracking.build_pilot_run_identity``), never the
    ``dynamic_inputs_override`` field alone -- an override and an
    unset-override-that-resolves-to-the-same-policy-default are the same
    training identity, but two different resolved lists (including two
    lists containing the same variables in a different order) are always a
    contradiction: order is part of the identity here, since
    ``resolved_dynamic_inputs`` is compared by plain equality after a JSON
    round-trip.

    If ``nh_run_dir`` does not exist yet (this candidate's very first call,
    before any NH run directory has been created), this function persists
    nothing and returns -- there is nothing yet to contradict. Once a run
    directory exists, the first call for it persists the current
    ``run_identity``'s ``(pilot_policy_name, run_id,
    resolved_dynamic_inputs)`` triple; every later call for the same
    directory must match it exactly, or :class:`PilotOrchestrationError` is
    raised -- before any further tracking or training call. See this
    function's call site in :func:`run_pilot`, immediately alongside the
    other five scalar-identity guards."""
    _enforce_pilot_scalar_identity(
        run_identity=run_identity,
        nh_run_dir=nh_run_dir,
        state_filename=DYNAMIC_INPUTS_IDENTITY_STATE_FILENAME,
        identity_key="resolved_dynamic_inputs",
        axis_label="dynamic-inputs",
        contradiction_detail=(
            "resolved_dynamic_inputs must never change across a continuation "
            "of the same run directory"
        ),
    )


def chunk_epoch_targets(pilot_policy: PilotPolicy, effective_max_epoch_budget: int) -> "list[int]":
    """The sequence of epoch counts each bounded training chunk trains up TO
    (inclusive). The first chunk ends at the explicitly configured
    ``pilot_policy.initial_training_epochs`` (historically 6). Each later
    chunk advances by
    ``screening_validation_every_n_epochs`` (3), since NH's own
    ``validate_every=3`` already produces a screening-cadence
    checkpoint+validation at every chunk boundary without this module
    re-triggering anything. Capped at ``effective_max_epoch_budget`` (this
    pilot's 36-epoch sub-cap, see
    :func:`src.baseline.pilot_early_stopping.build_effective_policy`)."""
    first_target = pilot_policy.initial_training_epochs
    step = pilot_policy.screening_validation_every_n_epochs
    if first_target > effective_max_epoch_budget:
        raise PilotOrchestrationError(
            f"initial_training_epochs={first_target} exceeds "
            f"effective_max_epoch_budget={effective_max_epoch_budget}"
        )
    targets = list(range(first_target, effective_max_epoch_budget + 1, step))
    if targets[-1] != effective_max_epoch_budget:
        targets.append(effective_max_epoch_budget)
    return targets


def screening_epochs_in_chunk(previous_target: int, new_target: int, pilot_policy: PilotPolicy) -> "list[int]":
    """Epochs newly reached within one chunk that fall on this pilot's fixed
    screening cadence (diagnostic epoch 3 included, since it divides evenly
    into the 3-epoch cadence -- :func:`src.baseline.pilot_screening_eval.classify_screening_epoch_role`
    is what actually distinguishes diagnostic vs stopping-eligible; this
    function only enumerates cadence epochs, never classifies them)."""
    step = pilot_policy.screening_validation_every_n_epochs
    start = previous_target + step if previous_target > 0 else step
    return [e for e in range(start, new_target + 1, step) if e % step == 0]


def prepare_pilot_run(
    *,
    pilot_policy: PilotPolicy,
    run_id: str,
    baseline_policy_path,
    package_root,
    splits_dir,
    config_out_dir,
    static_column_manifest_path=None,
    force: bool = False,
):
    """Build this run's ``GeneratedConfigBundle`` and write (or, on resume,
    reuse) its generated config under ``config_out_dir``. Idempotent: if
    ``config_out_dir`` already contains a previously-written
    ``config.yaml``/``generation_manifest.json`` and ``force`` is False,
    reuses it rather than regenerating -- regenerating would silently change
    the frozen training config NH itself may already be resuming from.
    Returns ``(run_spec, bundle, config_out_dir, experiment_name)``. Does
    not call NH -- see :func:`run_pilot_chunk` for the actual training
    step."""
    run_spec = resolve_pilot_run_spec(pilot_policy, run_id)
    bundle = build_pilot_bundle(
        pilot_policy=pilot_policy,
        run_id=run_id,
        baseline_policy_path=baseline_policy_path,
        package_root=package_root,
        splits_dir=splits_dir,
        static_column_manifest_path=static_column_manifest_path,
    )

    config_out_dir = Path(config_out_dir)
    experiment_name = f"stage1_lead06_pilot_{run_id}_v001"
    config_path = config_out_dir / "config.yaml"
    manifest_path = config_out_dir / "generation_manifest.json"
    if force or not (config_path.is_file() and manifest_path.is_file()):
        write_generated_config(bundle, config_out_dir, experiment_name=experiment_name, force=force)

    return run_spec, bundle, config_out_dir, experiment_name


PREPARATION_RESULT_FILENAME = "pilot_preparation_result.json"


def _assert_no_prior_training_state(
    *, config_out_dir: Path, experiment_name: str, preparation_out_dir, run_id: str
) -> None:
    """Fail loudly if this candidate already has a real NH run directory (or
    an evidence bundle from a real training invocation) -- ``--prepare-only``
    (see :func:`prepare_pilot_run_only`) only ever prepares a brand-new,
    untrained candidate, and must never describe an already-trained (or
    ambiguously-trained: more than one matching run directory) candidate as a
    clean preparation."""
    runs_root = Path(config_out_dir) / "runs"
    if runs_root.is_dir():
        candidates = sorted(
            p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(experiment_name)
        )
        if candidates:
            raise PilotOrchestrationError(
                f"--prepare-only refuses to proceed for run_id={run_id!r}: "
                f"{len(candidates)} existing NH run director{'y' if len(candidates) == 1 else 'ies'} "
                f"already present under {runs_root} matching {experiment_name!r} "
                f"({[str(c) for c in candidates]}) -- this indicates training has already started "
                "(or an ambiguous/partial prior attempt exists); --prepare-only only supports "
                "preparing a brand-new, untrained candidate. Use the ordinary (non---prepare-only) "
                "invocation to continue training this run_id, or resolve the existing directory "
                "manually before re-preparing."
            )

    preparation_out_dir = Path(preparation_out_dir)
    if preparation_out_dir.is_dir():
        unexpected = sorted(
            p.name for p in preparation_out_dir.iterdir() if p.name != PREPARATION_RESULT_FILENAME
        )
        if unexpected:
            raise PilotOrchestrationError(
                f"--prepare-only refuses to proceed: {preparation_out_dir} already contains "
                f"unexpected file(s)/directory(ies) not written by a prior --prepare-only call "
                f"({unexpected}) -- this looks like a real training evidence bundle or other "
                "non-preparation state; refusing to silently overwrite it. Point "
                "--evidence-out-dir at a fresh location, or resolve the existing directory "
                "manually."
            )


def prepare_pilot_run_only(
    *,
    pilot_policy: PilotPolicy,
    run_id: str,
    baseline_policy_path,
    package_root,
    splits_dir,
    config_out_dir,
    preparation_out_dir,
    static_column_manifest_path=None,
    tracking_generation: str = "g1",
    commands_used: "list[str] | None" = None,
    force: bool = False,
) -> dict:
    """Expose exactly :func:`run_pilot`'s config-generation phase --
    :func:`prepare_pilot_run` plus this pilot's provenance/identity
    computation (:func:`~src.baseline.pilot_tracking.build_pilot_run_identity`)
    -- and stop, one early-exit mode around existing preparation, not a
    second, generalized lifecycle framework.

    Calls no NH entrypoint (``start_run``/``continue_run``) and initializes
    no W&B backend (never calls
    :func:`~src.baseline.pilot_tracking.init_pilot_tracking_run`): the only
    NH/W&B-shaped values this function computes
    (``wandb_run_id``/``wandb_policy_sha256`` inside ``run_identity``) are
    pure, deterministic computations over already-in-memory config/policy
    data, never a call into NH or the W&B SDK. Creates no checkpoint,
    optimizer-state, validation-result pickle, screening-event,
    early-stopping-state, or orchestration-state file, and accesses no
    temporal-test-period, spatial-holdout, or California data (this pilot's
    screening machinery, which does touch the development-population
    validation period, is never invoked here -- only
    :func:`~src.baseline.pilot_screening_eval.load_validated_screening_basin_ids`
    is, to validate the screening-subset file itself resolves).

    Restart-safe like :func:`run_pilot`, but strictly narrower:
    :func:`prepare_pilot_run`'s own idempotent config-file reuse is
    unaffected by this function, and :func:`_assert_no_prior_training_state`
    additionally refuses to run at all if this candidate already has any real
    NH run directory or evidence-bundle output.

    Returns a JSON-serializable dict describing the prepared candidate (also
    written verbatim to ``preparation_out_dir/PREPARATION_RESULT_FILENAME``).
    """
    run_spec, bundle, config_dir, experiment_name = prepare_pilot_run(
        pilot_policy=pilot_policy,
        run_id=run_id,
        baseline_policy_path=baseline_policy_path,
        package_root=package_root,
        splits_dir=splits_dir,
        config_out_dir=config_out_dir,
        static_column_manifest_path=static_column_manifest_path,
        force=force,
    )

    _assert_no_prior_training_state(
        config_out_dir=config_dir,
        experiment_name=experiment_name,
        preparation_out_dir=preparation_out_dir,
        run_id=run_id,
    )

    screening_basin_ids = load_validated_screening_basin_ids(
        pilot_policy=pilot_policy, package_root=package_root, splits_dir=splits_dir
    )
    effective_policy = build_effective_policy(pilot_policy)

    run_identity = build_pilot_run_identity(
        pilot_policy=pilot_policy,
        run_spec=run_spec,
        bundle=bundle,
        effective_early_stopping_policy=effective_policy,
        tracking_generation=tracking_generation,
    )

    config_path = config_dir / "config.yaml"
    manifest_path = config_dir / "generation_manifest.json"

    preparation_out_dir = Path(preparation_out_dir)
    preparation_out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_name": "stage1_lead06_pilot_preparation_result",
        "schema_version": 1,
        "status": "PREPARED_ONLY",
        "run_id": run_id,
        "experiment_name": experiment_name,
        "config_out_dir": str(config_dir),
        "generated_config_path": str(config_path),
        "generation_manifest_path": str(manifest_path),
        "n_screening_basins": len(screening_basin_ids),
        "wandb_policy_sha256": run_identity["wandb_policy_sha256"],
        "tracking_generation": run_identity["tracking_generation"],
        "run_identity": run_identity,
        "commands_used": list(commands_used) if commands_used else [],
        "training_started": False,
        "evaluation_started": False,
        "wandb_backend_initialized": False,
        "statement": (
            "This --prepare-only invocation generated only this candidate's NH config and "
            "generation manifest, via the same prepare_pilot_run() step run_pilot() itself "
            "calls first. It never called neuralhydrology.nh_run.start_run or continue_run, "
            "never initialized a W&B backend or offline run directory, and never created any "
            "checkpoint, optimizer-state, validation-result pickle, screening-event, "
            "early-stopping-state, or orchestration-state file. It never accessed the "
            "temporal-test period, any spatial-holdout basin, or any California basin."
        ),
    }

    result_path = preparation_out_dir / PREPARATION_RESULT_FILENAME
    result_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["preparation_result_path"] = str(result_path)
    return result


def _advance_chunk_via_continuation(
    *,
    nh_base_run_dir: Path,
    config_dir: Path,
    start_dir: Path,
    resume_from_epoch: int,
    chunk_target_epoch: int,
    train_chunk_fn,
    accepted_checkpoints: "dict[int, AcceptedContinuationEntry] | None" = None,
) -> "tuple[Path | None, str | None]":
    """Resolve ``chunk_target_epoch``'s trusted physical checkpoint,
    producing it via one bounded ``train_chunk_fn`` continuation call from
    ``resume_from_epoch``'s checkpoint (physically owned by ``start_dir``)
    if it does not already exist. Returns
    ``(checkpoint_dir_for_target, blocked_reason)`` -- exactly one is
    ``None``.

    Shared by both branches of :func:`run_pilot_chunk` that continue an
    existing checkpoint (mid-first-chunk resumption with
    ``resume_from_epoch > 0``, and any later bounded chunk): NH's own
    ``continue_run`` sets ``is_continue_training = True`` unconditionally
    (``neuralhydrology.nh_run.continue_run``), so
    ``BaseTrainer._create_folder_structure`` always nests a fresh
    ``continue_training_from_epoch{resume_from_epoch:03d}/`` directory
    under ``start_dir`` -- there is no "flat" continuation case, regardless
    of which logical chunk this is (see module docstring).

    ``accepted_checkpoints``, when given, is only ever consulted for the one
    epoch this call is already trying to resolve (``chunk_target_epoch``) --
    never for any other epoch present in the same manifest -- which is what
    keeps a later chunk's accepted entry (e.g. epoch 15) from ever being
    consulted while an earlier chunk (e.g. epoch 12) is still the one being
    resolved.
    """
    inventory = discover_physical_checkpoints(nh_base_run_dir)
    expected_dir = _expected_continuation_dir(start_dir, resume_from_epoch)
    trusted = resolve_trusted_chunk_checkpoint(inventory, start_dir, resume_from_epoch, chunk_target_epoch)
    if trusted is not None:
        return expected_dir, None

    conflicts = untrusted_overshoot_epochs(inventory, resume_from_epoch, chunk_target_epoch)
    if conflicts:
        accepted_entry = (accepted_checkpoints or {}).get(chunk_target_epoch)
        if accepted_entry is not None:
            accepted_dir = _resolve_accepted_checkpoint(accepted_entry, chunk_target_epoch)
            return accepted_dir, None
        return None, (
            f"cannot safely continue training from epoch {resume_from_epoch} to "
            f"{chunk_target_epoch}: untrusted physical checkpoint(s) already occupy epoch(s) "
            f"{conflicts}, but not in the directory a clean continuation from epoch "
            f"{resume_from_epoch} would produce ({expected_dir}) -- refusing to train "
            "into a range that would create a duplicate physical claim on an existing "
            "epoch number; manual review of the pre-existing checkpoint(s) is required "
            "before this pilot can safely continue automatically"
        )
    if resume_from_epoch != 0 and expected_dir.is_dir():
        # NeuralHydrology's own continue_run refuses to create an
        # already-existing run directory
        # (basetrainer.BaseTrainer._create_folder_structure raises
        # RuntimeError rather than resuming into or overwriting it), so a
        # fresh train_chunk_fn call here would crash inside NH itself
        # rather than failing through this module's own clear status --
        # even though no checkpoint in this chunk's target range happens
        # to be present yet (e.g. the directory was created but killed
        # before its first epoch finished). Halting here with an explicit
        # blocked status, never letting that NH crash surface as an
        # undifferentiated exception. (Guarded on resume_from_epoch != 0:
        # when resume_from_epoch is 0, _expected_continuation_dir returns
        # start_dir itself -- which trivially already exists since it is
        # the directory this call is already operating within -- so that
        # case is not a meaningful "pre-existing partial continuation"
        # signal at all.)
        return None, (
            f"cannot safely continue training from epoch {resume_from_epoch} to "
            f"{chunk_target_epoch}: the directory a fresh continuation would create "
            f"already exists ({expected_dir}) but does not yet contain a trusted "
            f"checkpoint at epoch {chunk_target_epoch} -- NeuralHydrology's own "
            "continue_run refuses to create an already-existing run directory, so this "
            "looks like a previously interrupted continuation attempt that cannot be "
            "safely retried automatically; manual review is required (verify whether the "
            "partial directory's contents are trustworthy, then either complete training "
            "manually or remove the partial directory before resuming this pilot)"
        )

    train_chunk_fn(
        TrainChunkRequest(
            is_first_chunk=False,
            config_path=Path(config_dir) / "config.yaml",
            nh_run_dir=start_dir,
            current_epoch=resume_from_epoch if resume_from_epoch != 0 else None,
            logical_target_epoch=chunk_target_epoch,
            additional_epochs=chunk_target_epoch - resume_from_epoch,
        )
    )
    inventory = discover_physical_checkpoints(nh_base_run_dir)
    trusted = resolve_trusted_chunk_checkpoint(inventory, start_dir, resume_from_epoch, chunk_target_epoch)
    if trusted is None:
        raise PilotOrchestrationError(
            f"training chunk to epoch {chunk_target_epoch} did not produce the expected "
            f"checkpoint at {expected_dir}"
        )
    return expected_dir, None


def run_pilot_chunk(
    *,
    pilot_policy: PilotPolicy,
    config_dir,
    experiment_name: str,
    package_root,
    target_variable: str,
    lead_hours: int,
    screening_basin_ids,
    effective_policy: dict,
    chunk_target_epoch: int,
    previous_target_epoch: int,
    is_first_chunk: bool,
    previous_checkpoint_dir: "Path | None" = None,
    tracking_run=None,
    train_chunk_fn: "Callable[[TrainChunkRequest], None]" = default_train_chunk,
    evaluate_checkpoint_fn: "Callable[[EvaluationRequest], None]" = default_evaluate_checkpoint,
    run_id: str = "",
) -> dict:
    """Run exactly one bounded training chunk (``previous_target_epoch`` ->
    ``chunk_target_epoch``) and process every screening-cadence epoch newly
    reached within it. Returns ``{"nh_run_dir", "blocked", "blocked_reason",
    "stopped", "stop_reason", "state", "screening_results",
    "checkpoint_dir_for_target"}``.

    ``previous_checkpoint_dir`` is the physical directory that owns
    ``previous_target_epoch``'s checkpoint (``None``/ignored when
    ``previous_target_epoch == 0``, meaning "the base run directory" --
    see module docstring). The caller (:func:`run_pilot`) is responsible for
    threading each chunk's returned ``checkpoint_dir_for_target`` into the
    NEXT chunk's ``previous_checkpoint_dir`` -- this module never
    re-derives it via any multi-level trust chain, only ever compares
    directly against what the caller asserts is the current trusted frontier.

    If the target epoch's checkpoint is not yet a TRUSTED physical product
    of a continuation cleanly started at ``previous_target_epoch`` (see
    :func:`resolve_trusted_chunk_checkpoint`), and physical checkpoints
    already occupy the epoch range this chunk would need to produce (see
    :func:`untrusted_overshoot_epochs`), this chunk is refused rather than
    attempted: no training call is made, no screening is performed, and the
    returned dict has ``blocked=True`` with a human-readable
    ``blocked_reason`` -- never a raised exception (which would look like an
    ordinary crash) and never a silent resume from whatever the highest
    physical checkpoint happens to be.

    Idempotent on resume: an already-trained epoch is not retrained (checked
    against the physical checkpoint inventory); an already-saved validation
    result pickle is not re-evaluated (checked via
    :func:`ensure_validation_results`, see module docstring); a screening
    epoch already present in this run's advisory
    ``pilot_orchestration_state.json`` (``logged_screening_epochs``) is
    skipped entirely -- neither re-evaluated nor re-fed through
    :func:`src.baseline.pilot_early_stopping.record_screening_event`, since
    that function's own idempotent-replay semantics only cover replaying
    the exact last-recorded history entry, not an earlier one that a later
    chunk's screening has since superseded (real Moriah job 45718742:
    replaying epoch 6 once epoch 9 was already the last recorded entry
    raised ``PilotEarlyStoppingError`` -- "out of order"). So
    ``screening_results`` reflects only the newly-processed epochs of this
    particular call, not this chunk's full cadence history when resuming
    past already-logged epochs. Canonical pilot state (early-stopping + orchestration state) is always
    kept in the BASE run directory, never per-continuation-directory -- one
    logical pilot history regardless of how many physical continuation
    directories exist (see module docstring's "one logical pilot history"
    note).
    """
    config_dir = Path(config_dir)

    if is_first_chunk:
        train_chunk_fn(
            TrainChunkRequest(
                is_first_chunk=True,
                config_path=config_dir / "config.yaml",
                nh_run_dir=None,
                current_epoch=None,
                logical_target_epoch=chunk_target_epoch,
                additional_epochs=chunk_target_epoch,
            )
        )
        nh_base_run_dir = discover_nh_run_dir(config_dir, experiment_name)
        checkpoint_dir_for_target = nh_base_run_dir
        blocked_reason = None
    else:
        nh_base_run_dir = discover_nh_run_dir(config_dir, experiment_name)
        blocked_reason = None
        accepted_checkpoints = load_accepted_continuation_manifest(nh_base_run_dir, run_id)

        if previous_target_epoch == 0:
            # Base-run resumption (this orchestration process never itself
            # ran the first chunk to completion -- e.g. the real second
            # Moriah failure's shape, where checkpoints 1-6 already exist
            # from a prior process's start_run call). Only two shapes are
            # supported here: no checkpoints at all yet (handled by
            # _advance_chunk_via_continuation's resume_from_epoch=0 corner),
            # or the frozen base profile's own target already fully
            # satisfied, with its checkpoints living FLAT in nh_base_run_dir
            # (start_run, unlike continue_run, never sets
            # is_continue_training and so never nests) -- nothing to
            # continue. A genuinely PARTIAL first chunk (e.g. epochs 1-4
            # exist, target is 6) is intentionally unsupported and rejected
            # below: this module resolves exactly ONE physical checkpoint
            # directory per chunk for every screening epoch in that chunk
            # (see this function's docstring), but a partial-first-chunk
            # continuation would place epoch 3 flat in nh_base_run_dir and
            # epoch 6 in a newly nested continue_training_from_epoch{highest:03d}/
            # directory -- two different physical directories for the same
            # chunk's screening epochs, which this module does not attempt
            # to reconcile automatically.
            highest = _last_completed_epoch(nh_base_run_dir)
            if 0 < highest < chunk_target_epoch:
                blocked_reason = (
                    f"cannot safely process the first chunk (epoch 0 to {chunk_target_epoch}): "
                    f"the base run directory already has checkpoint(s) through epoch {highest}, "
                    f"a partial first chunk -- continuing it would require a training call "
                    f"from epoch {highest}, nesting the remaining checkpoints into a new "
                    f"continue_training_from_epoch{highest:03d}/ directory while the earlier "
                    f"checkpoints stay flat in the base run directory, so screening epochs "
                    f"{screening_epochs_in_chunk(0, chunk_target_epoch, pilot_policy)} would span "
                    "two different physical run directories; partial continuation within the "
                    "initial chunk is intentionally unsupported -- manual review is required "
                    "(either complete the first chunk to its full target outside this pilot "
                    "module, or start it over from no checkpoints at all)"
                )
                checkpoint_dir_for_target = None
            elif highest >= chunk_target_epoch:
                checkpoint_dir_for_target = nh_base_run_dir
            else:
                checkpoint_dir_for_target, blocked_reason = _advance_chunk_via_continuation(
                    nh_base_run_dir=nh_base_run_dir,
                    config_dir=config_dir,
                    start_dir=nh_base_run_dir,
                    resume_from_epoch=highest,
                    chunk_target_epoch=chunk_target_epoch,
                    train_chunk_fn=train_chunk_fn,
                    accepted_checkpoints=accepted_checkpoints,
                )
        else:
            start_dir = Path(previous_checkpoint_dir) if previous_checkpoint_dir is not None else nh_base_run_dir
            checkpoint_dir_for_target, blocked_reason = _advance_chunk_via_continuation(
                nh_base_run_dir=nh_base_run_dir,
                config_dir=config_dir,
                start_dir=start_dir,
                resume_from_epoch=previous_target_epoch,
                chunk_target_epoch=chunk_target_epoch,
                train_chunk_fn=train_chunk_fn,
                accepted_checkpoints=accepted_checkpoints,
            )

    if blocked_reason is not None:
        return {
            "nh_run_dir": nh_base_run_dir,
            "blocked": True,
            "blocked_reason": blocked_reason,
            "stopped": False,
            "stop_reason": None,
            "state": load_or_init_pilot_state(nh_base_run_dir, effective_policy),
            "screening_results": [],
            "checkpoint_dir_for_target": None,
        }

    orchestration_state = _load_orchestration_state(nh_base_run_dir)
    logged_epochs = set(orchestration_state["logged_screening_epochs"])

    es_state = load_or_init_pilot_state(nh_base_run_dir, effective_policy)
    screening_results = []
    for epoch in screening_epochs_in_chunk(previous_target_epoch, chunk_target_epoch, pilot_policy):
        role = classify_screening_epoch_role(epoch, pilot_policy)

        if epoch in logged_epochs:
            # Already fully processed on a prior invocation of this pilot
            # (this run's persisted pilot_orchestration_state.json already
            # lists it). Re-evaluating and re-recording it here would, for a
            # stopping-eligible epoch, replay it into
            # early_stopping.record_official_validation_event once a LATER
            # epoch is already the last recorded history entry -- real
            # Moriah job 45718742: history already ended at epoch 9, so
            # re-recording epoch 6 raised PilotEarlyStoppingError "epoch 6
            # is not after the last recorded epoch 9 -- out of order".
            # Trust the persisted logged_screening_epochs contract instead
            # of re-deriving anything. Light consistency check only (not
            # broad reconciliation): a stopping-eligible epoch marked
            # logged must actually be present in the already-reloaded
            # early-stopping history, or state is genuinely inconsistent
            # and must not be silently skipped.
            if role == "stopping_eligible" and not any(
                entry["epoch"] == epoch for entry in es_state.get("history", [])
            ):
                raise PilotOrchestrationError(
                    f"epoch {epoch} is marked logged in this run's orchestration state but is "
                    "absent from its early-stopping history -- refusing to silently skip "
                    "genuinely inconsistent persisted state"
                )
            continue

        ensure_validation_results(
            nh_run_dir=checkpoint_dir_for_target, epoch=epoch, evaluate_checkpoint_fn=evaluate_checkpoint_fn
        )
        result = evaluate_screening_checkpoint(
            run_dir=checkpoint_dir_for_target,
            epoch=epoch,
            package_root=package_root,
            target_variable=target_variable,
            lead_hours=lead_hours,
            screening_basin_ids=screening_basin_ids,
            pilot_policy=pilot_policy,
        )
        screening_results.append(result)
        es_state = record_screening_event(
            run_dir=nh_base_run_dir,
            epoch=epoch,
            epoch_role=role,
            primary_metric_median=result["primary_metric_median"],
            effective_policy=effective_policy,
        )

        # Persist authoritative orchestration state for this epoch BEFORE
        # any optional W&B telemetry call, not after the whole loop: real
        # Moriah job 45731908 raised an uncaught TrackingError out of the
        # checkpoint-reference call below, so the old post-loop-only save
        # never ran even though this epoch's validation + early-stopping
        # state was already durable on disk. Tracking is now also fully
        # non-fatal (see log_pilot_screening_event/log_pilot_checkpoint_reference),
        # so this ordering is a durability improvement, not a correctness
        # dependency of the fix.
        logged_epochs.add(epoch)
        orchestration_state["logged_screening_epochs"] = sorted(logged_epochs)
        _save_orchestration_state(nh_base_run_dir, orchestration_state)

        if tracking_run is not None:
            log_pilot_screening_event(tracking_run, epoch=epoch, screening_result=result, early_stopping_state=es_state)
            ckpt_path = Path(checkpoint_dir_for_target) / f"model_epoch{epoch:03d}.pt"
            if ckpt_path.is_file():
                log_pilot_checkpoint_reference(tracking_run, epoch=epoch, path=ckpt_path, checksum=sha256_of(ckpt_path))

        if es_state.get("stopped"):
            break

    return {
        "nh_run_dir": nh_base_run_dir,
        "blocked": False,
        "blocked_reason": None,
        "stopped": bool(es_state.get("stopped")),
        "stop_reason": es_state.get("stop_reason"),
        "state": es_state,
        "screening_results": screening_results,
        "checkpoint_dir_for_target": checkpoint_dir_for_target,
    }


@dataclass(frozen=True)
class PreparedPilotExecutionResult:
    """Generic, campaign-agnostic receipt for one
    :func:`execute_prepared_pilot_run` call -- the sole interface a
    higher-level scientific workflow needs, so it never has to reopen the
    internal per-chunk dict shape, crawl the NH run directory itself, or
    reconstruct screening history by hand.

    ``screening_events`` is always this run's COMPLETE, epoch-ordered
    screening history -- every screening-cadence epoch ever logged for
    ``nh_run_dir`` across every past and current invocation (see
    :func:`logged_screening_epochs`), not just the epochs newly processed by
    this particular call. Each entry is exactly
    :func:`src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`'s
    return shape. A successfully reconstructed entry for epoch E is itself
    the evidence that NH validation/evaluation for E completed -- the
    ``ensure_validation_results`` -> ``evaluate_screening_checkpoint`` chain
    only ever produces a result by reading an already-saved validation
    pickle. Consumers should treat ``{e["epoch"] for e in screening_events}``
    as the authoritative NH-evaluation-coverage set and must not build a
    second, parallel evaluation-coverage model.

    ``checkpoint_inventory`` is exactly :func:`discover_physical_checkpoints`'s
    return shape (``dict[int, PhysicalCheckpoint]``) -- physical checkpoint
    presence only, no campaign-specific completeness interpretation.

    This type intentionally carries no campaign-specific concept (no
    VALID/INVALID, best_epoch/best_score, Bayesian objective, or W&B field).
    A caller needing actual optimizer-update evidence must call
    :func:`actual_optimizer_updates_by_epoch` on ``nh_run_dir`` explicitly --
    it is deliberately not eagerly computed here (torch-dependent, and not
    every consumer needs it).
    """

    final_status: str
    blocked_reason: "str | None"
    effective_policy: dict
    nh_run_dir: Path
    blocked: bool
    stopped: bool
    stop_reason: "str | None"
    checkpoint_inventory: "dict[int, PhysicalCheckpoint]"
    early_stopping_state: dict
    screening_events: list
    # Optional campaign-supplied, post-pickle facts keyed by epoch.  The
    # generic executor never interprets these values; this narrow callback
    # seam lets a scientific consumer evaluate already-materialized NH
    # results without changing v1 screening or triggering inference.
    supplemental_epoch_results: dict = field(default_factory=dict)


def _reconstruct_screening_history(
    *, nh_run_dir, checkpoint_inventory: "dict[int, PhysicalCheckpoint]",
    package_root, target_variable: str, lead_hours: int, screening_basin_ids, pilot_policy,
) -> list:
    """Rebuild this run's COMPLETE, epoch-ordered screening-event history
    from durable state, fixing the pre-existing bug where a resumed
    :func:`execute_prepared_pilot_run` call only returned the CURRENT
    invocation's screening results (prior chunks'/invocations' events were
    silently missing). Never persists a second screening-event
    representation, never re-runs NH inference, and never recomputes
    hydrologic metrics itself: each event is produced by re-reading the
    already-written validation pickle through the same mature
    :func:`evaluate_screening_checkpoint` helper :func:`run_pilot_chunk`
    itself uses, driven by the durable :func:`logged_screening_epochs` set
    (the authoritative record of which epochs were ever actually screened)
    and the authoritative physical checkpoint inventory (for each epoch's
    exact owning run directory -- never inferred from the base run
    directory, since an earlier chunk's screening epoch may physically live
    in a different, now-superseded continuation directory than the
    base/current one)."""
    events = []
    for epoch in logged_screening_epochs(nh_run_dir):
        checkpoint = checkpoint_inventory.get(epoch)
        if checkpoint is None:
            raise PilotOrchestrationError(
                f"epoch {epoch} is recorded as logged in this run's orchestration state "
                f"but has no physical checkpoint anywhere under {nh_run_dir} -- refusing to "
                "fabricate a screening-history entry for a checkpoint that no longer exists"
            )
        result = evaluate_screening_checkpoint(
            run_dir=checkpoint.owning_run_dir,
            epoch=epoch,
            package_root=package_root,
            target_variable=target_variable,
            lead_hours=lead_hours,
            screening_basin_ids=screening_basin_ids,
            pilot_policy=pilot_policy,
        )
        events.append(result)
    return events


def execute_prepared_pilot_run(
    *, execution_policy, config_dir, experiment_name: str, package_root,
    target_variable: str, lead_hours: int, screening_basin_ids, run_id: str,
    tracking_run=None, train_chunk_fn: "Callable[[TrainChunkRequest], None]" = default_train_chunk,
    evaluate_checkpoint_fn: "Callable[[EvaluationRequest], None]" = default_evaluate_checkpoint,
    max_target_epoch: "int | None" = None,
) -> PreparedPilotExecutionResult:
    """Execute an already-written NH config through the mature chunk primitive.

    This intentionally does not resolve a run spec or write ``config.yaml``.
    It reads only the existing execution-policy surface used by the former
    inline loop: chunk scheduling, lead hours, and the effective stop policy.

    Returns a generic, campaign-agnostic :class:`PreparedPilotExecutionResult`
    -- see that class's docstring for field semantics, in particular
    ``screening_events``' full-history reconstruction and its relationship
    to NH evaluation coverage.
    """
    effective_policy = build_effective_policy(execution_policy)
    targets = chunk_epoch_targets(execution_policy, effective_policy["max_epoch_budget"])
    if not targets:
        raise PilotOrchestrationError("chunk_epoch_targets returned no targets -- nothing to train")
    if max_target_epoch is not None:
        targets = [target for target in targets if target <= max_target_epoch]
        if not targets:
            raise PilotOrchestrationError(f"max_target_epoch={max_target_epoch} excludes every chunk target")
    have_started = _try_discover_nh_run_dir(config_dir, experiment_name) is not None
    previous_target = 0; previous_checkpoint_dir = None
    last_chunk_result = None; final_status = "not_started"; blocked_reason = None
    for idx, target in enumerate(targets):
        chunk_result = run_pilot_chunk(
            pilot_policy=execution_policy, config_dir=config_dir, experiment_name=experiment_name,
            package_root=package_root, target_variable=target_variable, lead_hours=lead_hours,
            screening_basin_ids=screening_basin_ids, effective_policy=effective_policy,
            chunk_target_epoch=target, previous_target_epoch=previous_target,
            is_first_chunk=(not have_started) and idx == 0, previous_checkpoint_dir=previous_checkpoint_dir,
            tracking_run=tracking_run, train_chunk_fn=train_chunk_fn,
            evaluate_checkpoint_fn=evaluate_checkpoint_fn, run_id=run_id,
        )
        have_started = True
        if chunk_result["blocked"]:
            if last_chunk_result is None:
                last_chunk_result = chunk_result
            blocked_reason = chunk_result["blocked_reason"]
            final_status = "blocked_continuation_overshoot_conflict"
            break
        last_chunk_result = chunk_result
        previous_target, previous_checkpoint_dir = target, chunk_result["checkpoint_dir_for_target"]
        if chunk_result["stopped"]:
            final_status = f"stopped_{chunk_result['stop_reason']}"
            break
    else:
        final_status = ("paused_at_max_target_epoch" if max_target_epoch is not None
                        and targets[-1] < effective_policy["max_epoch_budget"] else "budget_exhausted_not_stopped")
    if last_chunk_result is None:
        raise PilotOrchestrationError("prepared execution produced no chunk result")

    nh_run_dir = last_chunk_result["nh_run_dir"]
    checkpoint_inventory = discover_physical_checkpoints(nh_run_dir)
    screening_events = _reconstruct_screening_history(
        nh_run_dir=nh_run_dir, checkpoint_inventory=checkpoint_inventory,
        package_root=package_root, target_variable=target_variable, lead_hours=lead_hours,
        screening_basin_ids=screening_basin_ids, pilot_policy=execution_policy,
    )

    return PreparedPilotExecutionResult(
        final_status=final_status,
        blocked_reason=blocked_reason,
        effective_policy=effective_policy,
        nh_run_dir=nh_run_dir,
        blocked=blocked_reason is not None,
        stopped=bool(last_chunk_result["stopped"]),
        stop_reason=last_chunk_result["stop_reason"],
        checkpoint_inventory=checkpoint_inventory,
        early_stopping_state=last_chunk_result["state"],
        screening_events=screening_events,
    )


def execute_prepared_pilot_run_monolithic(
    *, execution_policy: PilotPolicy, config_dir, experiment_name: str, package_root,
    target_variable: str, lead_hours: int, screening_basin_ids, target_epoch: int,
    train_chunk_fn: "Callable[[TrainChunkRequest], None]" = default_train_chunk,
    evaluate_checkpoint_fn: "Callable[[EvaluationRequest], None]" = default_evaluate_checkpoint,
    supplemental_epoch_evaluator: "Callable[[Path, int], Mapping] | None" = None,
) -> PreparedPilotExecutionResult:
    """Execute one MONOLITHIC prepared NH training invocation -- for a
    generated config whose own ``epochs`` key is already baked to
    ``target_epoch`` (e.g. Sweep-v1's ``sweep_v1_production_adapter.py``
    single-shot-fidelity design) -- then perform post-hoc raw-space
    screening for every epoch ``1..target_epoch``.

    This is a SIBLING to :func:`execute_prepared_pilot_run`, not a
    modification of it: that function's bounded-chunk design (
    :func:`chunk_epoch_targets` / :func:`run_pilot_chunk` /
    :func:`_advance_chunk_via_continuation` /
    :func:`resolve_trusted_chunk_checkpoint` /
    :func:`untrusted_overshoot_epochs`) assumes the generated config trains
    only an initial small chunk and relies on repeated ``continue_run`` calls
    to advance further -- exactly the assumption a monolithic
    already-``target_epoch``-baked config violates (this is the confirmed
    root cause of the real ``blocked_continuation_overshoot_conflict``
    trial-1 failure: a single ``start_run`` call trains straight through
    every epoch, so the second chunk's continuation-trust check finds epoch 2
    already sitting flat in the base run directory and refuses to proceed).

    This function NEVER calls ``continue_run``: it invokes ``train_chunk_fn``
    with ``is_first_chunk=True`` at most once, only when no NH run directory
    yet exists for this experiment. If a run directory already exists but is
    missing any required checkpoint ``1..target_epoch`` -- or has a
    checkpoint for one of those epochs physically nested under a
    continuation directory, something a genuine single ``start_run`` call
    never produces -- this is reported as ``blocked`` rather than attempting
    any continuation to complete it; resuming an interrupted monolithic run
    is out of scope here and requires human review, exactly mirroring
    :func:`execute_prepared_pilot_run`'s own refuse-rather-than-guess
    posture for untrusted checkpoint state.

    Reuses, unchanged: :func:`_try_discover_nh_run_dir`/
    :func:`discover_nh_run_dir`, :func:`discover_physical_checkpoints`,
    :func:`ensure_validation_results`,
    :func:`~src.baseline.pilot_screening_eval.evaluate_screening_checkpoint`
    (via :func:`_reconstruct_screening_history`), and
    :func:`_load_orchestration_state`/:func:`_save_orchestration_state` for
    restart-safe per-epoch screening idempotency. Every epoch
    ``1..target_epoch`` must classify as a screening epoch under
    ``execution_policy`` (i.e. ``screening_validation_every_n_epochs=1``) --
    ``execution_policy``'s cadence field is used ONLY for
    :func:`~src.baseline.pilot_screening_eval.classify_screening_epoch_role`
    here, never for chunk scheduling (:func:`chunk_epoch_targets` is never
    called by this function).

    Requires ``performance_early_stopping_enabled=False`` -- this monolithic
    mode has no notion of stopping training early (the config already bakes
    in the full ``target_epoch`` budget before this function ever runs), so
    a policy that claims otherwise indicates a caller/context mismatch and is
    rejected immediately rather than silently ignored.

    ``supplemental_epoch_evaluator``, when supplied, is an optional pure
    post-pickle evaluator. It receives only the already-materialized run
    directory and epoch, must not train, infer, publish, or otherwise mutate
    external state, and is intentionally at-least-once: an interrupted call
    returns no receipt and a later call recomputes the complete trajectory
    from immutable epoch artifacts. V1 callers leave it unset.

    Returns the same generic, campaign-agnostic
    :class:`PreparedPilotExecutionResult` :func:`execute_prepared_pilot_run`
    returns, so a caller's VALID/INVALID interpretation layer (e.g.
    ``sweep_v1_execution.py``'s ``_derive_validity``) needs zero changes to
    consume either execution mode's receipt.
    """
    effective_policy = build_effective_policy(execution_policy)
    if effective_policy["performance_early_stopping_enabled"] is not False:
        raise PilotOrchestrationError(
            "execute_prepared_pilot_run_monolithic requires performance_early_stopping_enabled=False "
            f"in the effective policy, got {effective_policy['performance_early_stopping_enabled']!r}"
        )

    required = set(range(1, target_epoch + 1))
    nh_run_dir = _try_discover_nh_run_dir(config_dir, experiment_name)
    if nh_run_dir is None:
        train_chunk_fn(
            TrainChunkRequest(
                is_first_chunk=True,
                config_path=Path(config_dir) / "config.yaml",
                nh_run_dir=None,
                current_epoch=None,
                logical_target_epoch=target_epoch,
                additional_epochs=target_epoch,
            )
        )
        nh_run_dir = discover_nh_run_dir(config_dir, experiment_name)

    checkpoint_inventory = discover_physical_checkpoints(nh_run_dir)
    missing = sorted(required - set(checkpoint_inventory))
    nested = sorted(
        epoch for epoch, ckpt in checkpoint_inventory.items()
        if epoch in required and ckpt.owning_run_dir != Path(nh_run_dir)
    )
    if missing or nested:
        reasons = []
        if missing:
            reasons.append(f"missing required checkpoint(s) {missing}")
        if nested:
            reasons.append(
                f"checkpoint(s) for epoch(s) {nested} are physically nested under a continuation "
                "directory, which a genuine single monolithic start_run call never produces"
            )
        return PreparedPilotExecutionResult(
            final_status="blocked_incomplete_monolithic_training",
            blocked_reason=(
                f"monolithic training did not produce a complete, flat 1..{target_epoch} checkpoint "
                f"set directly under {nh_run_dir}: " + "; ".join(reasons) + " -- this function never "
                "attempts a continuation to complete an interrupted monolithic run; manual review is "
                "required"
            ),
            effective_policy=effective_policy, nh_run_dir=nh_run_dir, blocked=True,
            stopped=False, stop_reason=None, checkpoint_inventory=checkpoint_inventory,
            early_stopping_state={}, screening_events=[],
        )

    orchestration_state = _load_orchestration_state(nh_run_dir)
    already_logged = set(orchestration_state.get("logged_screening_epochs", []))
    for epoch in sorted(required - already_logged):
        ensure_validation_results(nh_run_dir=nh_run_dir, epoch=epoch, evaluate_checkpoint_fn=evaluate_checkpoint_fn)
        evaluate_screening_checkpoint(
            run_dir=nh_run_dir, epoch=epoch, package_root=package_root,
            target_variable=target_variable, lead_hours=lead_hours,
            screening_basin_ids=screening_basin_ids, pilot_policy=execution_policy,
        )
        already_logged.add(epoch)
        orchestration_state["logged_screening_epochs"] = sorted(already_logged)
        _save_orchestration_state(nh_run_dir, orchestration_state)

    screening_events = _reconstruct_screening_history(
        nh_run_dir=nh_run_dir, checkpoint_inventory=checkpoint_inventory,
        package_root=package_root, target_variable=target_variable, lead_hours=lead_hours,
        screening_basin_ids=screening_basin_ids, pilot_policy=execution_policy,
    )
    screened = {int(event["epoch"]) for event in screening_events}
    missing_screening = sorted(required - screened)
    if missing_screening:
        return PreparedPilotExecutionResult(
            final_status="blocked_incomplete_post_hoc_screening",
            blocked_reason=f"post-hoc screening did not produce results for epoch(s) {missing_screening}",
            effective_policy=effective_policy, nh_run_dir=nh_run_dir, blocked=True,
            stopped=False, stop_reason=None, checkpoint_inventory=checkpoint_inventory,
            early_stopping_state={}, screening_events=screening_events,
        )

    supplemental_epoch_results = {}
    if supplemental_epoch_evaluator is not None:
        for epoch in sorted(required):
            # The evaluator receives the existing run directory and epoch only;
            # it must consume the validation pickle just established above.
            supplemental_epoch_results[epoch] = dict(supplemental_epoch_evaluator(nh_run_dir, epoch))

    return PreparedPilotExecutionResult(
        final_status="monolithic_training_and_screening_complete",
        blocked_reason=None, effective_policy=effective_policy, nh_run_dir=nh_run_dir,
        blocked=False, stopped=False, stop_reason=None,
        checkpoint_inventory=checkpoint_inventory, early_stopping_state={},
        screening_events=screening_events,
        supplemental_epoch_results=supplemental_epoch_results,
    )


def run_pilot(
    *,
    pilot_policy: PilotPolicy,
    run_id: str,
    baseline_policy_path,
    package_root,
    splits_dir,
    config_out_dir,
    evidence_out_dir,
    screening_basin_ids: "list | None" = None,
    static_column_manifest_path=None,
    slurm_identity: "dict | None" = None,
    commands_used: "list[str] | None" = None,
    force: bool = False,
    tracking_generation: str = "g1",
    train_chunk_fn: "Callable[[TrainChunkRequest], None]" = default_train_chunk,
    evaluate_checkpoint_fn: "Callable[[EvaluationRequest], None]" = default_evaluate_checkpoint,
    max_target_epoch: "int | None" = None,
    require_tracking: bool = False,
) -> dict:
    """Top-level pilot orchestration for one ``run_id``: prepare the config,
    train in bounded chunks via NH's own ``start_run``/``continue_run``
    (through ``train_chunk_fn``), ensure each screening-cadence epoch's
    saved validation result exists (through ``evaluate_checkpoint_fn`` --
    see :func:`ensure_validation_results`), screen at every cadence epoch,
    apply restart-safe early stopping, log to W&B if enabled, and write the
    compact evidence bundle. Safe to call repeatedly on the same
    ``config_out_dir``/``evidence_out_dir`` -- the evidence bundle is always
    (re)written regardless of ``force``, since it is this function's own
    output and is expected to reflect the latest cumulative state on every
    resume. ``force`` instead controls only whether an already-generated NH
    config is regenerated (see ``prepare_pilot_run``); leave it ``False`` for
    ordinary resumes so a restart never silently rewrites the frozen
    training config NH may already be resuming from -- see module
    docstring's restart-safety note.

    NOT called against real data anywhere in this task -- see
    ``docs/stage1_lead06_pilot_v001.md``'s "known limitations" section. A
    real Moriah launch (task item 7's sbatch script, driving a thin CLI
    wrapper) is the only intended caller with ``train_chunk_fn`` left at its
    default (real NH training); tests and any local dry run must always pass
    a fake ``train_chunk_fn``.

    ``tracking_generation`` (default ``"g1"``) is passed straight through to
    :func:`src.baseline.pilot_tracking.build_pilot_run_identity` -- see that
    function and ``pilot_tracking.py``'s module docstring for when a caller
    should deliberately pass a different value. Leaving it at the default is
    correct for every ordinary bounded-chunk continuation of an in-progress
    candidate.

    ``max_target_epoch``, if given, bounds this call to chunk targets at or
    below it (see :func:`chunk_epoch_targets`) -- e.g. ``max_target_epoch=6``
    processes only the first chunk target and returns without attempting the
    next one, even though this call is neither ``blocked`` nor ``stopped``.
    Leave it ``None`` (the default) for ordinary operation, where one call
    walks every chunk target up to the policy's full epoch budget. This
    exists only so one specific call can stop for human review at a
    caller-chosen epoch (see the job 45731908 recovery note in
    docs/stage1_lead06_pilot_v001.md); it never changes what is
    trained/evaluated/recorded for any target at or below it, only whether
    later targets are attempted within this same call -- a later call with a
    higher (or absent) ``max_target_epoch`` resumes exactly where this one
    left off, through the same idempotent-resume path as any other restart.

    ``require_tracking`` (default ``False``, preserving all existing callers'
    behavior) is passed straight through to
    :func:`src.baseline.pilot_tracking.init_pilot_tracking_run` -- see that
    function's docstring for the Hidden-size-A campaign's strict launch
    contract (tracking init failure or a resolved null/untracked run is
    fatal rather than silently downgraded) that this parameter opts into
    when set ``True``.
    """
    run_spec, bundle, config_dir, experiment_name = prepare_pilot_run(
        pilot_policy=pilot_policy,
        run_id=run_id,
        baseline_policy_path=baseline_policy_path,
        package_root=package_root,
        splits_dir=splits_dir,
        config_out_dir=config_out_dir,
        static_column_manifest_path=static_column_manifest_path,
        force=force,
    )

    if screening_basin_ids is None:
        screening_basin_ids = load_validated_screening_basin_ids(
            pilot_policy=pilot_policy, package_root=package_root, splits_dir=splits_dir
        )

    effective_policy = build_effective_policy(pilot_policy)

    run_identity = build_pilot_run_identity(
        pilot_policy=pilot_policy,
        run_spec=run_spec,
        bundle=bundle,
        effective_early_stopping_policy=effective_policy,
        tracking_generation=tracking_generation,
        slurm_job_id=(slurm_identity or {}).get("job_id"),
        slurm_node=(slurm_identity or {}).get("node"),
        slurm_partition=(slurm_identity or {}).get("partition"),
        slurm_gres=(slurm_identity or {}).get("gres"),
    )

    # Discovered BEFORE starting the tracking run (not after, as a fresh NH
    # run directory does not yet exist on this candidate's very first call):
    # a resumed run passes its already-existing NH run directory into
    # init_pilot_tracking_run so the W&B run-identity persistence/
    # contradiction-check (see pilot_tracking.resolve_pilot_wandb_run_id)
    # can see it on this and every later continuation.
    existing_nh_run_dir = _try_discover_nh_run_dir(config_dir, experiment_name)
    have_started = existing_nh_run_dir is not None

    # Always active, regardless of W&B tracking policy -- must run before any
    # tracking or training call so a cap-identity contradiction is caught
    # before it can affect either. See enforce_pilot_cap_identity's docstring.
    enforce_pilot_cap_identity(run_identity=run_identity, nh_run_dir=existing_nh_run_dir)
    # Likewise always active: see enforce_pilot_learning_rate_identity's
    # docstring (LR-A range-characterization campaign).
    enforce_pilot_learning_rate_identity(run_identity=run_identity, nh_run_dir=existing_nh_run_dir)
    # Likewise always active: see enforce_pilot_hidden_size_identity's
    # docstring (Hidden-size-A range-characterization campaign).
    enforce_pilot_hidden_size_identity(run_identity=run_identity, nh_run_dir=existing_nh_run_dir)
    # Likewise always active: see enforce_pilot_embedding_dropout_identity's
    # docstring (Embedding-Dropout-A range-characterization campaign).
    enforce_pilot_embedding_dropout_identity(run_identity=run_identity, nh_run_dir=existing_nh_run_dir)
    enforce_pilot_output_dropout_identity(run_identity=run_identity, nh_run_dir=existing_nh_run_dir)
    enforce_pilot_batch_size_identity(run_identity=run_identity, nh_run_dir=existing_nh_run_dir)
    # Likewise always active: see enforce_pilot_seq_length_identity's
    # docstring (Sequence-Length-A range-characterization campaign).
    enforce_pilot_seq_length_identity(run_identity=run_identity, nh_run_dir=existing_nh_run_dir)
    # Likewise always active: see enforce_pilot_dynamic_inputs_identity's
    # docstring (Dynamic-Input-Family-A range-characterization campaign).
    enforce_pilot_dynamic_inputs_identity(run_identity=run_identity, nh_run_dir=existing_nh_run_dir)

    tracking_run = init_pilot_tracking_run(
        pilot_policy, run_identity, nh_run_dir=existing_nh_run_dir, require_tracking=require_tracking
    )

    execution = execute_prepared_pilot_run(
        execution_policy=pilot_policy, config_dir=config_dir, experiment_name=experiment_name,
        package_root=package_root, target_variable=bundle.target_variable, lead_hours=pilot_policy.lead_hours,
        screening_basin_ids=screening_basin_ids, run_id=run_id, tracking_run=tracking_run,
        train_chunk_fn=train_chunk_fn, evaluate_checkpoint_fn=evaluate_checkpoint_fn,
        max_target_epoch=max_target_epoch,
    )
    final_status = execution.final_status
    blocked_reason = execution.blocked_reason

    best_epoch = pilot_best_checkpoint_epoch(execution.early_stopping_state)
    finish_pilot_run(tracking_run, final_status=final_status, best_epoch=best_epoch)

    status_fields = compute_pilot_status_fields(execution.nh_run_dir, pilot_policy=pilot_policy)

    evidence_path = write_pilot_evidence_bundle(
        out_dir=evidence_out_dir,
        config_dir=config_dir,
        nh_run_dir=execution.nh_run_dir,
        pilot_policy=pilot_policy,
        run_spec=run_spec,
        tracking_run=tracking_run,
        early_stopping_state=execution.early_stopping_state,
        screening_events=execution.screening_events,
        run_status=final_status,
        commands_used=list(commands_used) if commands_used else [],
        slurm_identity=slurm_identity,
        continuation_status=status_fields,
        # Always overwrite: the evidence bundle is run_pilot's own output,
        # meant to reflect the latest cumulative state on every call
        # (including resumes where evidence_out_dir already exists from a
        # prior call). This is independent of the caller's `force`, which
        # instead guards config regeneration in prepare_pilot_run above --
        # conflating the two would force-regenerate the frozen NH training
        # config on a routine resume just to allow the evidence dir to be
        # rewritten.
        force=True,
    )

    return {
        "run_id": run_id,
        "final_status": final_status,
        "best_checkpoint_epoch": best_epoch,
        "nh_run_dir": execution.nh_run_dir,
        "evidence_bundle_path": evidence_path,
        "highest_physical_checkpoint_epoch": status_fields["highest_physical_checkpoint_epoch"],
        "highest_screened_epoch": status_fields["highest_screened_epoch"],
        "next_intended_screening_epoch": status_fields["next_intended_screening_epoch"],
        "overshoot_epochs": status_fields["overshoot_epochs"],
        "safe_to_continue_automatically": status_fields["safe_to_continue_automatically"],
        "blocked_reason": blocked_reason,
    }
