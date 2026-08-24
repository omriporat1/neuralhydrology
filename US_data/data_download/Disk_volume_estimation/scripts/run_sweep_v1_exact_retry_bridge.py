"""Operator-facing EXACT-RETRY bridge for one frozen Sweep-v1 attempt.

Unlike ``run_sweep_v1_wandb_bridge.py`` (the sole production caller of the
Bayesian-controller-driven path, invoked once per proposal by ``wandb
agent``), this script is manually operated and NEVER calls ``wandb.agent()``
or requests a new proposal from W&B. It exists to retry an already-INVALID
attempt with the EXACT SAME five hyperparameters, never a resampled or
hand-edited configuration:

  1. Load a durable frozen Layer-B proposal-intake record (the original
     attempt's own ``execution_provenance.json``, written by
     ``write_proposal_intake_provenance``) via
     ``src.baseline.sweep_v1_retry.load_frozen_proposal_record`` --
     independently re-derives ``configuration_id``/``proposal_id``/
     ``trial_id`` from the record's own axes rather than trusting the
     persisted ids at face value.
  2. Cross-check it against an operator-authored, explicitly reviewed
     pinned-expected-identity JSON file via
     ``sweep_v1_retry.assert_matches_pinned_identity`` -- refuses to proceed
     on ANY contradiction (wrong file, tampered record, wrong campaign).
  3. Derive the retry's identity via ``sweep_v1_retry.derive_exact_retry_identity``:
     identical hyperparameters/configuration_id/proposal_id, a strictly
     later ``execution_generation``, a freshly derived ``trial_id``, and
     ``retry_of_trial_id`` pointing at the original record's ``trial_id``.
  4. Refuse if the freshly derived output directory already exists --
     BEFORE any W&B call -- so a re-invocation can never overwrite or
     duplicate an attempt, and the original attempt's directory is never
     touched by this script (it is never opened for writing).
  5. ``wandb.init(settings=wandb.Settings(sweep_id=<production sweep>),
     config=<exact frozen hyperparameters>)`` -- deliberately NOT
     ``wandb.agent()``. Associating a run with a sweep via
     ``Settings.sweep_id`` and letting ``wandb.init()`` create it directly
     uses the identical backend run-creation field
     (``internal_api.upsert_run(..., sweep_name=...)``) that an
     agent-launched run uses, but that association happens entirely inside
     ``wandb.init()``'s own run-creation call -- the separate
     ``register_agent``/``agent_heartbeat``/``_command_run`` machinery that
     actually requests a NEW Bayesian-controller-assigned configuration
     lives only in ``wandb_agent.py`` and is never invoked here. This
     mechanism is empirically confirmed against wandb 0.28.1 source (see
     the disposable, non-scientific qualification in
     ``scripts/wandb_exact_retry_join_qualification.py``) rather than
     assumed.
  6. The remaining steps reuse the exact same, already-qualified production
     path as ``run_sweep_v1_wandb_bridge.py``: ``write_proposal_intake_provenance``
     -> ``canonicalize_wandb_proposal`` + ``prepare_bayesian_proposal`` +
     ``write_prepared_proposal`` (with ``allow_layer_b_provenance=True``) ->
     ``run_prepared_trial_in_production`` (passing ``retry_of_trial_id``
     through) -> logging ``flashnh/best_score`` only if the retry is VALID.

This script never reads, writes, or deletes anything under the ORIGINAL
attempt's output directory -- only under the freshly derived retry output
directory, which by construction (a strictly later ``execution_generation``)
is a different, previously-nonexistent path.

Never imports ``wandb`` at module scope (repo-wide lazy-import convention).
Writing/reading this script performs no live W&B call and starts no
training; it only executes real work when explicitly invoked with a real
frozen record, a real pinned-identity file, and real package/screening
artifacts -- outside this task's authorized scope, and not done here.
"""
from __future__ import annotations

import argparse
import json
import os
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
from src.baseline.sweep_v1_retry import (
    assert_matches_pinned_identity, derive_exact_retry_identity, load_frozen_proposal_record,
)

ENV_SELFTEST = "FLASHNH_SWEEP_V1_RETRY_BRIDGE_SELFTEST"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-proposal-record", type=Path, required=True,
                        help="Path to the original attempt's own execution_provenance.json.")
    parser.add_argument("--expected-identity", type=Path, required=True,
                        help="Path to an operator-authored JSON file pinning the expected identity "
                             "(any subset of proposal_order/proposal_id/configuration_id/trial_id/"
                             "search_arm/wandb_sweep_id/model_seed/five hyperparameters).")
    parser.add_argument("--execution-generation", type=int, required=True,
                        help="Strictly greater than the frozen record's own execution_generation.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--screening-basin-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-policy-path", type=Path, default=ROOT / "config/stage1_scientific_baseline_v001.yaml")
    parser.add_argument("--base-pilot-policy-path", type=Path, default=ROOT / "config/stage1_lead06_pilot_v001.yaml")
    parser.add_argument("--project", type=str, default="flashnh-stage1")
    parser.add_argument("--entity", type=str, default=None)
    args = parser.parse_args()

    record = load_frozen_proposal_record(args.frozen_proposal_record)
    pinned = json.loads(Path(args.expected_identity).read_text(encoding="utf-8"))
    assert_matches_pinned_identity(record, pinned)
    retry = derive_exact_retry_identity(record, execution_generation=args.execution_generation)

    output_root = args.output_root
    output_dir = output_root / retry["trial_id"]
    if output_dir.exists():
        raise SystemExit(
            f"REFUSING: retry output directory already exists, refusing to reuse or overwrite it: {output_dir}"
        )

    if os.environ.get(ENV_SELFTEST) == "resolve_only":
        # Deterministic, network-free hook: proves identity loading,
        # pinned-identity validation, and retry-identity derivation all
        # succeed and land on a fresh output directory, without importing
        # wandb, calling wandb.init(), or touching the network/training.
        print(json.dumps({"retry_identity": retry, "output_dir": str(output_dir)}, indent=2, sort_keys=True, default=str))
        return 0

    canonical_splits = ROOT / "config" / "stage1_baseline_splits_v001"
    paths = PreparationPaths(args.baseline_policy_path, args.package_root, canonical_splits, args.screening_basin_ids)

    import wandb
    run = wandb.init(
        settings=wandb.Settings(sweep_id=retry["wandb_sweep_id"]),
        project=args.project,
        entity=args.entity,
        config=retry["hyperparameters"],
        group=retry["proposal_id"],
        job_type="exact_retry",
        tags=["exact_retry", f"retry_of_{retry['retry_of_trial_id']}"],
    )
    valid = False
    try:
        if run.sweep_id != retry["wandb_sweep_id"]:
            raise SystemExit(
                f"REFUSING: run.sweep_id ({run.sweep_id!r}) did not associate with the requested "
                f"production sweep ({retry['wandb_sweep_id']!r}); no silent fallback is accepted."
            )

        intake = write_proposal_intake_provenance(
            output_root=output_root, axes=retry["hyperparameters"], search_arm=retry["search_arm"],
            proposal_order=retry["proposal_order"], wandb_sweep_id=run.sweep_id, wandb_run_id=run.id,
            execution_generation=retry["execution_generation"], retry_of_trial_id=retry["retry_of_trial_id"],
        )
        if intake["trial_id"] != retry["trial_id"]:
            raise SystemExit(
                f"REFUSING: intake trial_id ({intake['trial_id']!r}) disagrees with the derived retry "
                f"trial_id ({retry['trial_id']!r})."
            )

        proposal = canonicalize_wandb_proposal(
            retry["hyperparameters"],
            metadata={
                "proposal_order": retry["proposal_order"], "wandb_sweep_id": run.sweep_id, "wandb_run_id": run.id,
                "execution_generation": retry["execution_generation"],
            },
        )
        prepared = prepare_bayesian_proposal(proposal=proposal, paths=paths)
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared", fields=prepared.evidence)

        record_ = write_prepared_proposal(prepared, output_dir, allow_layer_b_provenance=True)
        enrich_layer_b_provenance(output_dir=output_dir, stage="prepared_with_config", fields=record_)

        outcome = run_prepared_trial_in_production(
            prepared_record=record_, output_dir=output_dir, paths=paths,
            base_pilot_policy_path=args.base_pilot_policy_path,
            retry_of_trial_id=retry["retry_of_trial_id"], slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        )
        valid = outcome["valid"]
        trial = outcome["review_records"]["trial_summary"]

        if trial["objective_score"] is not None:
            run.log({"flashnh/best_score": trial["objective_score"]})
        run.summary["flashnh/valid"] = valid
        run.summary["flashnh/workflow_status"] = trial["workflow_status"]
        run.summary["flashnh/failure_category"] = trial["failure_category"]
        run.summary["flashnh/trial_id"] = trial["trial_id"]
        run.summary["flashnh/retry_of_trial_id"] = retry["retry_of_trial_id"]
        run.summary["flashnh/execution_generation"] = retry["execution_generation"]
        run.summary["flashnh/exact_retry"] = True
        run.summary["flashnh/output_dir"] = str(output_dir)
    finally:
        run.finish()

    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
