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
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

from .nh_config_generation import write_generated_config
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
    "discover_nh_run_dir",
    "chunk_epoch_targets",
    "screening_epochs_in_chunk",
    "prepare_pilot_run",
    "run_pilot_chunk",
    "run_pilot",
]

_CHECKPOINT_GLOB = "model_epoch*.pt"
_ORCHESTRATION_STATE_FILENAME = "pilot_orchestration_state.json"


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
    duplicated NH training logic."""

    is_first_chunk: bool
    config_path: Path
    nh_run_dir: "Path | None"
    target_epoch: int


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

        overlay_path = Path(request.nh_run_dir) / "pilot_epoch_overlay.yaml"
        overlay_path.write_text(yaml.safe_dump({"epochs": request.target_epoch}), encoding="utf-8")
        continue_run(run_dir=request.nh_run_dir, config_file=overlay_path)


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


def chunk_epoch_targets(pilot_policy: PilotPolicy, effective_max_epoch_budget: int) -> "list[int]":
    """The sequence of epoch counts each bounded training chunk trains up TO
    (inclusive). The first chunk always ends at
    ``pilot_policy.stopping_eligible_from_epoch`` (6) -- this is the frozen
    base profile's own ``epochs: 6``
    (``nh_config_generation._PILOT_LEAD06_BASE_PROFILE``), never edited
    here. Each later chunk advances by
    ``screening_validation_every_n_epochs`` (3), since NH's own
    ``validate_every=3`` already produces a screening-cadence
    checkpoint+validation at every chunk boundary without this module
    re-triggering anything. Capped at ``effective_max_epoch_budget`` (this
    pilot's 36-epoch sub-cap, see
    :func:`src.baseline.pilot_early_stopping.build_effective_policy`)."""
    first_target = pilot_policy.stopping_eligible_from_epoch
    step = pilot_policy.screening_validation_every_n_epochs
    if first_target > effective_max_epoch_budget:
        raise PilotOrchestrationError(
            f"stopping_eligible_from_epoch={first_target} exceeds "
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
    tracking_run=None,
    train_chunk_fn: "Callable[[TrainChunkRequest], None]" = default_train_chunk,
) -> dict:
    """Run exactly one bounded training chunk (``previous_target_epoch`` ->
    ``chunk_target_epoch``) and process every screening-cadence epoch newly
    reached within it. Returns
    ``{"nh_run_dir", "stopped", "stop_reason", "state", "screening_results"}``.

    Idempotent on resume: an already-trained epoch is not retrained (checked
    against NH's own checkpoint files), and an already-logged screening
    epoch (per this run's advisory ``pilot_orchestration_state.json``) is
    not re-logged to the tracking backend -- but IS still re-evaluated and
    re-fed through :func:`src.baseline.pilot_early_stopping.record_screening_event`,
    whose own idempotent-replay semantics make that safe, so the returned
    ``screening_results`` always reflects this chunk's full cadence history.
    """
    config_dir = Path(config_dir)

    if is_first_chunk:
        train_chunk_fn(
            TrainChunkRequest(
                is_first_chunk=True,
                config_path=config_dir / "config.yaml",
                nh_run_dir=None,
                target_epoch=chunk_target_epoch,
            )
        )
        nh_run_dir = discover_nh_run_dir(config_dir, experiment_name)
    else:
        nh_run_dir = discover_nh_run_dir(config_dir, experiment_name)
        if _last_completed_epoch(nh_run_dir) < chunk_target_epoch:
            train_chunk_fn(
                TrainChunkRequest(
                    is_first_chunk=False,
                    config_path=config_dir / "config.yaml",
                    nh_run_dir=nh_run_dir,
                    target_epoch=chunk_target_epoch,
                )
            )

    orchestration_state = _load_orchestration_state(nh_run_dir)
    logged_epochs = set(orchestration_state["logged_screening_epochs"])

    es_state = load_or_init_pilot_state(nh_run_dir, effective_policy)
    screening_results = []
    for epoch in screening_epochs_in_chunk(previous_target_epoch, chunk_target_epoch, pilot_policy):
        role = classify_screening_epoch_role(epoch, pilot_policy)
        result = evaluate_screening_checkpoint(
            run_dir=nh_run_dir,
            epoch=epoch,
            package_root=package_root,
            target_variable=target_variable,
            lead_hours=lead_hours,
            screening_basin_ids=screening_basin_ids,
            pilot_policy=pilot_policy,
        )
        screening_results.append(result)
        es_state = record_screening_event(
            run_dir=nh_run_dir,
            epoch=epoch,
            epoch_role=role,
            primary_metric_median=result["primary_metric_median"],
            effective_policy=effective_policy,
        )

        if tracking_run is not None and epoch not in logged_epochs:
            log_pilot_screening_event(tracking_run, epoch=epoch, screening_result=result, early_stopping_state=es_state)
            ckpt_path = nh_run_dir / f"model_epoch{epoch:03d}.pt"
            if ckpt_path.is_file():
                log_pilot_checkpoint_reference(tracking_run, epoch=epoch, path=ckpt_path, checksum=sha256_of(ckpt_path))
            logged_epochs.add(epoch)

        if es_state.get("stopped"):
            break

    orchestration_state["logged_screening_epochs"] = sorted(logged_epochs)
    _save_orchestration_state(nh_run_dir, orchestration_state)

    return {
        "nh_run_dir": nh_run_dir,
        "stopped": bool(es_state.get("stopped")),
        "stop_reason": es_state.get("stop_reason"),
        "state": es_state,
        "screening_results": screening_results,
    }


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
    train_chunk_fn: "Callable[[TrainChunkRequest], None]" = default_train_chunk,
) -> dict:
    """Top-level pilot orchestration for one ``run_id``: prepare the config,
    train in bounded chunks via NH's own ``start_run``/``continue_run``
    (through ``train_chunk_fn``), screen at every cadence epoch, apply
    restart-safe early stopping, log to W&B if enabled, and write the
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
        slurm_job_id=(slurm_identity or {}).get("job_id"),
        slurm_node=(slurm_identity or {}).get("node"),
        slurm_partition=(slurm_identity or {}).get("partition"),
        slurm_gres=(slurm_identity or {}).get("gres"),
    )
    tracking_run = init_pilot_tracking_run(pilot_policy, run_identity)

    targets = chunk_epoch_targets(pilot_policy, effective_policy["max_epoch_budget"])
    if not targets:
        raise PilotOrchestrationError("chunk_epoch_targets returned no targets -- nothing to train")

    existing_nh_run_dir = _try_discover_nh_run_dir(config_dir, experiment_name)
    have_started = existing_nh_run_dir is not None

    previous_target = 0
    all_screening_results: "list[dict]" = []
    last_chunk_result = None
    final_status = "not_started"
    for idx, target in enumerate(targets):
        is_first_chunk = (not have_started) and idx == 0
        last_chunk_result = run_pilot_chunk(
            pilot_policy=pilot_policy,
            config_dir=config_dir,
            experiment_name=experiment_name,
            package_root=package_root,
            target_variable=bundle.target_variable,
            lead_hours=pilot_policy.lead_hours,
            screening_basin_ids=screening_basin_ids,
            effective_policy=effective_policy,
            chunk_target_epoch=target,
            previous_target_epoch=previous_target,
            is_first_chunk=is_first_chunk,
            tracking_run=tracking_run,
            train_chunk_fn=train_chunk_fn,
        )
        have_started = True
        all_screening_results.extend(last_chunk_result["screening_results"])
        previous_target = target
        if last_chunk_result["stopped"]:
            final_status = f"stopped_{last_chunk_result['stop_reason']}"
            break
    else:
        final_status = "budget_exhausted_not_stopped"

    best_epoch = pilot_best_checkpoint_epoch(last_chunk_result["state"])
    finish_pilot_run(tracking_run, final_status=final_status, best_epoch=best_epoch)

    evidence_path = write_pilot_evidence_bundle(
        out_dir=evidence_out_dir,
        config_dir=config_dir,
        nh_run_dir=last_chunk_result["nh_run_dir"],
        pilot_policy=pilot_policy,
        run_spec=run_spec,
        tracking_run=tracking_run,
        early_stopping_state=last_chunk_result["state"],
        screening_events=all_screening_results,
        run_status=final_status,
        commands_used=list(commands_used) if commands_used else [],
        slurm_identity=slurm_identity,
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
        "nh_run_dir": last_chunk_result["nh_run_dir"],
        "evidence_bundle_path": evidence_path,
    }
