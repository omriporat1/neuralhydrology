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
  5. Write the durable Layer-B proposal-intake record (``write_proposal_intake_provenance``,
     with ``wandb_run_id=None`` -- never fabricated) BEFORE any W&B call, so
     the retry's full identity (including any prior failed attempts recorded
     via ``--prior-attempts``, e.g. attempt002/job 45939764) survives even if
     W&B association itself fails. This ordering, and the bounded tag scheme
     in step 6, are the direct fix for the attempt002/job 45939764 incident,
     in which a 125-character tag was rejected deep inside ``wandb.init()``'s
     own ``Settings`` validation with no durable evidence yet written.
  6. Build a small, fixed-shape, deterministic tag set
     (``sweep_v1_retry.build_bounded_wandb_tags``) and validate every tag is
     at most ``MAX_WANDB_TAG_LENGTH`` characters
     (``sweep_v1_retry.validate_wandb_tags``) BEFORE calling ``wandb.init()``.
     Tags are non-authoritative conveniences; the complete retry/trial/
     proposal/configuration identity always lives in the durable intake
     record and in the W&B run's own ``config`` (never only in a tag).
  7. ``wandb.init(settings=wandb.Settings(sweep_id=<production sweep>),
     config=<exact frozen hyperparameters + retry_identity>)`` -- deliberately
     NOT ``wandb.agent()``. Associating a run with a sweep via
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
     assumed. Any tag-validation failure or ``wandb.init()`` exception is
     recorded onto the SAME durable intake record (stages
     ``wandb_tags_rejected`` / ``wandb_init_failed``) before re-raising --
     evidence is never lost, and no training ever starts without a
     successful W&B association. A successful association is itself
     recorded (stage ``wandb_associated``) before any preparation/training
     step runs.
  8. The remaining steps reuse the exact same, already-qualified production
     path as ``run_sweep_v1_wandb_bridge.py``: ``canonicalize_wandb_proposal``
     + ``prepare_bayesian_proposal`` + ``write_prepared_proposal`` (with
     ``allow_layer_b_provenance=True``) -> a pure executor-mode selection
     (``sweep_v1_execution.select_executor_mode``) -> ``run_prepared_trial_in_production``
     (passing ``retry_of_trial_id`` through) -> logging ``flashnh/best_score``
     only if the retry is VALID.

This script never reads, writes, or deletes anything under the ORIGINAL
attempt's output directory -- only under the freshly derived retry output
directory, which by construction (a strictly later ``execution_generation``)
is a different, previously-nonexistent path.

Never imports ``wandb`` at module scope (repo-wide lazy-import convention).
Writing/reading this script performs no live W&B call and starts no
training; it only executes real work when explicitly invoked with a real
frozen record, a real pinned-identity file, and real package/screening
artifacts -- outside this task's authorized scope, and not done here.

Two entry points share one execution core (``_execute_retry``), so
production and a disposable rehearsal can never silently diverge:

* :func:`main` -- the original, unchanged CLI-flag interface (kept for
  back-compat; not used by new manifest-driven launches).
* :func:`main_from_manifest` -- consumes one
  ``src.baseline.sweep_v1_launch_manifest`` JSON file instead of a long flag
  list. When ``manifest["mode"] == "rehearsal"``, the shared core runs the
  real runtime contract (commit/interpreter/HOME/netrc,
  ``src.baseline.sweep_v1_runtime_contract``), targets the manifest's own
  (necessarily non-production) ``wandb_sweep_id`` instead of the frozen
  record's historical one, records the pure executor-mode selection, and
  stops BEFORE ``run_prepared_trial_in_production`` is ever called -- it
  never starts NH training. When ``mode == "production"`` the same core runs
  end-to-end exactly like :func:`main` does today.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline.sweep_v1_execution import (
    enrich_layer_b_provenance, run_prepared_trial_in_production, select_executor_mode,
    write_proposal_intake_provenance,
)
from src.baseline.sweep_v1_production_adapter import (
    PreparationPaths, canonicalize_wandb_proposal, prepare_bayesian_proposal, write_prepared_proposal,
)
from src.baseline.sweep_v1_retry import (
    SweepV1RetryError, assert_matches_pinned_identity, build_bounded_wandb_tags, derive_exact_retry_identity,
    load_frozen_proposal_record, validate_wandb_tags,
)

ENV_SELFTEST = "FLASHNH_SWEEP_V1_RETRY_BRIDGE_SELFTEST"


def _execute_retry(
    *, record: dict[str, Any], pinned: dict[str, Any], execution_generation: int,
    prior_attempts: list, package_root: Path, screening_basin_ids: Path, output_root: Path,
    baseline_policy_path: Path, base_pilot_policy_path: Path, project: str, entity: "str | None",
    target_sweep_id_override: "str | None", stop_before_training: bool,
    extra_intake_fields: "dict[str, Any] | None" = None,
) -> int:
    """Shared execution core for both :func:`main` (legacy CLI) and
    :func:`main_from_manifest`. See module docstring for the exact contract.
    """
    assert_matches_pinned_identity(record, pinned)

    retry = derive_exact_retry_identity(
        record, execution_generation=execution_generation, prior_attempts=prior_attempts,
    )

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
    paths = PreparationPaths(baseline_policy_path, package_root, canonical_splits, screening_basin_ids)

    target_sweep_id = target_sweep_id_override if target_sweep_id_override is not None else retry["wandb_sweep_id"]

    # Durable retry-intake BEFORE any W&B call: no fabricated wandb_run_id,
    # and this record (including the operational link to any prior failed
    # attempt such as attempt002/job 45939764, via retry_history) survives
    # even if tag validation or wandb.init() itself fails below.
    intake = write_proposal_intake_provenance(
        output_root=output_root, axes=retry["hyperparameters"], search_arm=retry["search_arm"],
        proposal_order=retry["proposal_order"], wandb_sweep_id=target_sweep_id, wandb_run_id=None,
        execution_generation=retry["execution_generation"], retry_of_trial_id=retry["retry_of_trial_id"],
        retry_history=prior_attempts,
    )
    if extra_intake_fields:
        # The launch manifest itself becomes part of durable intake
        # provenance -- merged onto the SAME record, at the SAME
        # proposal_intake stage, still strictly before any W&B call.
        intake = enrich_layer_b_provenance(
            output_dir=output_dir, stage=intake["provenance_stage"], fields=extra_intake_fields,
        )
    if intake["trial_id"] != retry["trial_id"]:
        raise SystemExit(
            f"REFUSING: intake trial_id ({intake['trial_id']!r}) disagrees with the derived retry "
            f"trial_id ({retry['trial_id']!r})."
        )

    tags = build_bounded_wandb_tags(
        proposal_order=retry["proposal_order"], execution_generation=retry["execution_generation"],
        configuration_id=retry["configuration_id"],
    )
    try:
        validate_wandb_tags(tags)
    except SweepV1RetryError as exc:
        enrich_layer_b_provenance(
            output_dir=output_dir, stage="wandb_tags_rejected", fields={"error": str(exc), "tags": tags},
        )
        raise SystemExit(f"REFUSING: bounded W&B tag set failed validation: {exc}") from exc

    wandb_config = {
        **retry["hyperparameters"],
        "retry_identity": {
            "trial_id": retry["trial_id"],
            "retry_of_trial_id": retry["retry_of_trial_id"],
            "proposal_id": retry["proposal_id"],
            "configuration_id": retry["configuration_id"],
            "execution_generation": retry["execution_generation"],
        },
    }

    import wandb
    try:
        run = wandb.init(
            settings=wandb.Settings(sweep_id=target_sweep_id),
            project=project,
            entity=entity,
            config=wandb_config,
            group=retry["proposal_id"],
            job_type="exact_retry",
            tags=tags,
        )
    except Exception as exc:
        enrich_layer_b_provenance(
            output_dir=output_dir, stage="wandb_init_failed",
            fields={"error": str(exc), "error_type": type(exc).__name__, "tags": tags},
        )
        raise SystemExit(
            f"REFUSING: wandb.init() failed for exact-retry attempt {retry['trial_id']!r}: {exc}"
        ) from exc

    valid = False
    try:
        if run.sweep_id != target_sweep_id:
            enrich_layer_b_provenance(
                output_dir=output_dir, stage="wandb_association_mismatch",
                fields={
                    "wandb_run_id": run.id, "actual_sweep_id": run.sweep_id,
                    "expected_sweep_id": target_sweep_id,
                },
            )
            raise SystemExit(
                f"REFUSING: run.sweep_id ({run.sweep_id!r}) did not associate with the requested "
                f"sweep ({target_sweep_id!r}); no silent fallback is accepted."
            )

        enrich_layer_b_provenance(
            output_dir=output_dir, stage="wandb_associated",
            fields={"trial_id": retry["trial_id"], "wandb_run_id": run.id, "wandb_sweep_id": run.sweep_id},
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

        mode = select_executor_mode(record_)
        enrich_layer_b_provenance(
            output_dir=output_dir, stage="executor_mode_selected", fields={"executor_mode": mode},
        )

        if stop_before_training:
            run.summary["flashnh/rehearsal_stopped_before_training"] = True
            run.summary["flashnh/executor_mode_selected"] = mode
            run.summary["flashnh/trial_id"] = retry["trial_id"]
            run.summary["flashnh/retry_of_trial_id"] = retry["retry_of_trial_id"]
            run.summary["flashnh/execution_generation"] = retry["execution_generation"]
            return 0

        outcome = run_prepared_trial_in_production(
            prepared_record=record_, output_dir=output_dir, paths=paths,
            base_pilot_policy_path=base_pilot_policy_path,
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


def main_from_manifest(manifest_path: "str | Path") -> int:
    """Manifest-driven entry point: one positional JSON file replaces the
    long CLI/``--export`` flag list. Runs the shared commit/interpreter/
    HOME/netrc runtime contract before any durable-intake or W&B step (fails
    durably, before W&B, on any invariant violation), then delegates to the
    same :func:`_execute_retry` core :func:`main` uses.
    """
    from src.baseline.sweep_v1_launch_manifest import load_launch_manifest
    from src.baseline.sweep_v1_runtime_contract import run_full_runtime_contract

    manifest = load_launch_manifest(manifest_path)

    run_full_runtime_contract(
        repo_root=ROOT,
        expected_commit=manifest["expected_commit"],
        expected_runtime_python=manifest["expected_runtime_python"],
    )

    record = load_frozen_proposal_record(manifest["frozen_proposal_record_path"])
    pinned = dict(manifest["expected_identity"])

    prior_attempts: list = []
    prior_attempts_path = manifest.get("prior_attempts_path")
    if prior_attempts_path:
        loaded_prior_attempts = json.loads(Path(prior_attempts_path).read_text(encoding="utf-8"))
        if not isinstance(loaded_prior_attempts, list):
            raise SystemExit(f"prior_attempts_path must contain a JSON array: {prior_attempts_path}")
        prior_attempts = loaded_prior_attempts

    manifest_path_resolved = str(Path(manifest_path).resolve())
    return _execute_retry(
        record=record, pinned=pinned, execution_generation=int(manifest["execution_generation"]),
        prior_attempts=prior_attempts, package_root=Path(manifest["package_root"]),
        screening_basin_ids=Path(manifest["screening_basin_ids_path"]), output_root=Path(manifest["output_root"]),
        baseline_policy_path=Path(manifest["baseline_policy_path"]),
        base_pilot_policy_path=Path(manifest["base_pilot_policy_path"]),
        project=manifest["wandb_project"], entity=manifest.get("wandb_entity"),
        target_sweep_id_override=manifest["wandb_sweep_id"],
        stop_before_training=bool(manifest["stop_before_training"]),
        extra_intake_fields={
            "launch_manifest_path": manifest_path_resolved,
            "launch_manifest_sha256": manifest["manifest_sha256"],
            "launch_manifest_label": manifest["manifest_label"],
        },
    )


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
    parser.add_argument("--prior-attempts", type=Path, default=None,
                        help="Optional path to an operator-authored JSON array of prior recorded attempts "
                             "for this trial family (e.g. a failed attempt002/Slurm job that crashed before "
                             "wandb.init() and left no other durable trace). Each element is a small dict "
                             "such as {\"execution_generation\": 2, \"slurm_job_id\": \"45939764\", "
                             "\"status\": \"failed_before_wandb_association\"}. Used to (a) refuse reusing "
                             "any already-attempted execution_generation and (b) carry the operational link "
                             "forward into the retry-intake record's retry_history, without overloading "
                             "retry_of_trial_id.")
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

    prior_attempts: list = []
    if args.prior_attempts is not None:
        loaded_prior_attempts = json.loads(Path(args.prior_attempts).read_text(encoding="utf-8"))
        if not isinstance(loaded_prior_attempts, list):
            raise SystemExit(f"--prior-attempts must contain a JSON array: {args.prior_attempts}")
        prior_attempts = loaded_prior_attempts

    return _execute_retry(
        record=record, pinned=pinned, execution_generation=args.execution_generation,
        prior_attempts=prior_attempts, package_root=args.package_root,
        screening_basin_ids=args.screening_basin_ids, output_root=args.output_root,
        baseline_policy_path=args.baseline_policy_path, base_pilot_policy_path=args.base_pilot_policy_path,
        project=args.project, entity=args.entity,
        target_sweep_id_override=None, stop_before_training=False, extra_intake_fields=None,
    )


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        # Single positional argument: a launch-manifest path (Design Decision
        # 2 -- no long `--export=ALL,VAR=value,...` interface for new
        # manifest-driven launches). Any flag-style invocation (including
        # every existing CLI test in tests/test_sweep_v1_exact_retry_bridge.py)
        # falls through to the original argparse-driven main() unchanged.
        raise SystemExit(main_from_manifest(sys.argv[1]))
    raise SystemExit(main())
