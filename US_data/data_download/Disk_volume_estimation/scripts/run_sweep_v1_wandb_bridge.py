"""Thin production W&B bridge for one Sweep-v1 Bayesian trial.

This is the ``program`` a real ``wandb agent`` invokes once per Bayesian
proposal (see ``build_production_sweep_config`` /
``scripts/build_sweep_v1_production_sweep_config.py``, whose sweep config's
``program`` field defaults to this script's path). Each invocation:

  1. ``wandb.init()`` to join the sweep-assigned run and read the proposed
     hyperparameters via ``run.config`` -- exactly the five frozen search
     axes ``build_production_sweep_config`` declares under ``parameters``,
     pulled by name (never a blind ``dict(run.config)``) so an unrelated key
     wandb might inject can never silently reach the preparation layer.
  2. ``write_proposal_intake_provenance`` -- durably records this exact
     W&B-assigned proposal (five axes, canonical IDs, W&B sweep/run identity)
     to local disk BEFORE any failure-prone step below, so it is always
     recoverable and retryable without asking W&B for a new configuration
     even if preparation or config generation raises. See
     ``src/baseline/sweep_v1_execution.py``'s docstring for the full Layer-B
     provenance design.
  3. ``canonicalize_wandb_proposal`` + ``prepare_bayesian_proposal`` +
     ``write_prepared_proposal`` -- the same real, already-qualified
     production-adapter path exercised by
     ``tests/test_sweep_v1_production_adapter.py``, with
     ``enrich_layer_b_provenance`` progressively enriching the same durable
     record (stage ``prepared`` after preparation succeeds, stage
     ``prepared_with_config`` after the config is written) rather than
     creating a second provenance authority.
  4. ``run_prepared_trial_in_production`` -- the real, fully-tested Sweep-v1
     execution/interpretation layer (``src/baseline/sweep_v1_execution.py``),
     which wires the mature NH orchestration
     (``pilot_orchestration.execute_prepared_pilot_run``) and derives
     VALID/INVALID + ``best_score`` from the authoritative prepared-execution
     receipt. This script never re-derives or second-guesses that result.
  5. Logs ``flashnh/best_score`` (matching ``build_production_sweep_config``'s
     ``metric.name``) as a time-series metric so the Bayesian optimizer can
     use it, and records the remaining outcome fields as run-summary values.
     W&B is a telemetry shell only here: it never determines VALID/INVALID or
     the objective value, and ``sweep_v1_execution.py`` has already written
     the authoritative ``review_records.json``/``execution_provenance.json``
     to ``output_dir`` before this script logs anything.

``proposal_order`` is a required, caller-supplied positive integer (e.g. a
Slurm job-array index or an external monotonic counter maintained by
whatever launches ``wandb agent``), never inferred from W&B or auto-numbered
here -- ``wandb`` sweep controllers do not expose a stable sequential
proposal count, and fabricating one would be an invented completeness rule
the prepared-execution contract does not authorize.

Never imports ``wandb`` at module scope (repo-wide lazy-import convention;
see ``scripts/wandb_online_sweep_qualification_run.py``). Never trains or
executes anything itself -- the sole real-training call in this whole path
is inside ``run_prepared_trial_in_production`` ->
``pilot_orchestration.execute_prepared_pilot_run``. Writing/reading this
script performs no live W&B call and starts no training; it only runs when a
real ``wandb agent`` process invokes it against a real sweep, which is
outside this task's authorized scope.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline.sweep_v1_execution import (
    enrich_layer_b_provenance, run_prepared_trial_in_production, write_proposal_intake_provenance,
)
from src.baseline.sweep_v1_production_adapter import (
    PreparationPaths, canonicalize_wandb_proposal, prepare_bayesian_proposal, write_prepared_proposal,
)

_AXES = ("learning_rate", "hidden_size", "embedding_dropout", "output_dropout", "batch_size")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--screening-basin-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--proposal-order", type=int, required=True,
                        help="Positive, caller-tracked Bayesian proposal sequence number.")
    parser.add_argument("--baseline-policy-path", type=Path, default=ROOT / "config/stage1_scientific_baseline_v001.yaml")
    parser.add_argument("--base-pilot-policy-path", type=Path, default=ROOT / "config/stage1_lead06_pilot_v001.yaml")
    args = parser.parse_args()

    canonical_splits = ROOT / "config" / "stage1_baseline_splits_v001"
    paths = PreparationPaths(args.baseline_policy_path, args.package_root, canonical_splits, args.screening_basin_ids)

    import wandb
    run = wandb.init()
    valid = False
    try:
        proposed_axes = {key: run.config[key] for key in _AXES}

        # DURABLE PROPOSAL-INTAKE PROVENANCE -- written immediately after the
        # five-axis proposal passes canonical legality validation and BEFORE
        # any failure-prone artifact/package verification, prepared-proposal
        # construction, config write, or mature execution, so this exact
        # W&B-assigned proposal is always locally recoverable without asking
        # W&B for a new one, even if everything below raises.
        intake = write_proposal_intake_provenance(
            output_root=args.output_root, axes=proposed_axes, search_arm="bayesian",
            proposal_order=args.proposal_order, wandb_sweep_id=run.sweep_id, wandb_run_id=run.id,
        )
        output_dir = args.output_root / intake["trial_id"]

        proposal = canonicalize_wandb_proposal(
            proposed_axes,
            metadata={"proposal_order": args.proposal_order, "wandb_sweep_id": run.sweep_id, "wandb_run_id": run.id},
        )
        prepared = prepare_bayesian_proposal(proposal=proposal, paths=paths)
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields=prepared.evidence)

        # allow_layer_b_provenance=True: output_dir already legitimately
        # holds the durable proposal_intake/prepared-stage
        # execution_provenance.json written above -- write_generated_config's
        # empty-directory guard would otherwise always reject it. This flag
        # tolerates ONLY that one pre-existing file (never overwritten,
        # deleted, or moved) and additionally verifies, before writing
        # anything, that its recorded trial_id matches this exact trial --
        # any other pre-existing entry, or a stale/foreign provenance file,
        # is a hard error raised before any generated artifact is written.
        record = write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared_with_config", fields=record)

        outcome = run_prepared_trial_in_production(
            prepared_record=record, output_dir=output_dir, paths=paths,
            base_pilot_policy_path=args.base_pilot_policy_path,
        )
        valid = outcome["valid"]
        trial = outcome["review_records"]["trial_summary"]

        if trial["objective_score"] is not None:
            run.log({"flashnh/best_score": trial["objective_score"]})
        run.summary["flashnh/valid"] = valid
        run.summary["flashnh/workflow_status"] = trial["workflow_status"]
        run.summary["flashnh/failure_category"] = trial["failure_category"]
        run.summary["flashnh/trial_id"] = trial["trial_id"]
        run.summary["flashnh/output_dir"] = str(output_dir)
    finally:
        run.finish()

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
