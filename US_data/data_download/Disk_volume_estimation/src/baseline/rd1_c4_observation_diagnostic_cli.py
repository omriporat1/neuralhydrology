"""Thin CLI wrapper for the RD1-C4-D1 observation diagnostic.

Three subcommands, matching the three Slurm entry points in
``scripts/rd1_c4_d1/``:

``benchmark``
    Run exactly one trial, selected by explicit ``--trial-id``, to measure
    per-trial cost before anything is sized. Refuses to run under an array
    allocation at all, so a benchmark can never quietly become 24 tasks.

``array-task``
    Run exactly one trial, selected either by ``--array-index`` (resolved
    against the frozen trial list, in its own frozen order) or by explicit
    ``--trial-id``. Exactly one of the two must be given.

``reduce``
    Reduce all 24 completed shards. Refuses unless every one of them is
    receipt-qualified -- there is no partial-reduction flag.

This module contains no scientific logic: it parses arguments, loads the
frozen inputs, and calls the library. It never submits a job, never contacts
W&B or a controller, and never touches a sealed scope.

The trial list is a JSON file supplied by the operator with a frozen,
explicit order. It selects trials; it does not describe them (RD1-C4-D1
BLOCKER A1)::

    {"schema_name": "flashnh_rd1_c4_d1_trial_roster", "schema_version": 1,
     "campaign_id": "...", "support_contract_version": "...",
     "support_contract_sha256": "...",
     "trials": [{"trial_id": "...", "search_arm": "bayesian",
                 "source_receipt_path": "...",
                 "source_receipt_sha256": "..."}, ...]}

``run_dir``, ``best_epoch``, ``official_objective``, the configuration and
proposal identities, the evaluation scope and the sealed-scope state are all
REJECTED if present: every one of them is derived from the trial's own
hash-verified execution receipt by
:func:`~.rd1_c4_trial_authentication.authenticate_trial_roster`, so this CLI
cannot be pointed at an arbitrary run directory by editing a JSON file.

The order in that file IS the array-index mapping, and the file's own
SHA-256 is recorded in every receipt, so a reordered or edited list produces
an identity conflict rather than a silently different assignment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

from .fixed_support_contract_v2 import load_fixed_support_contract
from .rd1_c4_observation_diagnostic import (
    EXPECTED_BASIN_COUNT,
    ObservationDiagnosticError,
    run_trial_observation_diagnostic,
)
from .rd1_c4_trial_authentication import (
    AuthenticatedTrialRoster,
    AuthenticatedTrialTarget,
    EXPECTED_ARM_COUNTS,
    EXPECTED_TRIAL_COUNT,
    TrialAuthenticationError,
    authenticate_trial_roster,
)
from .rd1_c4_observation_diagnostic_reduce import (
    ReductionError,
    reduce_observation_diagnostic,
)

__all__ = ["main", "build_parser", "load_trial_list"]

_ARRAY_ENV_KEYS = ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_JOB_ID")


def load_trial_list(path, contract) -> tuple:
    """Authenticate the frozen trial list, returning ``(roster, list_sha256)``.

    Every trial is loaded through
    :func:`~.rd1_c4_trial_authentication.authenticate_trial_roster`, which
    opens and hashes each named execution receipt and derives the trial's
    run directory, official best epoch and objective, arm, identity,
    evaluation scope and sealed-scope state from it (RD1-C4-D1 BLOCKER A1).
    The list itself may only say *which* receipt, and what that receipt's
    SHA-256 must be.

    The order in the file is authoritative and is never sorted here: array
    index ``i`` means ``trials[i]``, and nothing else. Whole-roster shape
    (24 trials, 12 bayesian + 12 random_control, no duplicate trial, receipt,
    run directory or validation product) is enforced for every subcommand,
    so a single array task cannot run against a roster that is missing,
    repeating or substituting a trial.
    """
    path = Path(path)
    try:
        roster = authenticate_trial_roster(trial_list_path=path, contract=contract)
    except TrialAuthenticationError as exc:
        raise ObservationDiagnosticError(f"{path}: trial roster is not receipt-qualified: {exc}") from exc
    return roster, roster.trial_list_sha256


def _resolve_target(
    roster: AuthenticatedTrialRoster, *, trial_id: "str | None", array_index: "int | None"
) -> AuthenticatedTrialTarget:
    if (trial_id is None) == (array_index is None):
        raise ObservationDiagnosticError(
            "select exactly one trial: pass --trial-id OR --array-index, never both and never neither"
        )
    targets = list(roster)
    if trial_id is not None:
        matches = [target for target in targets if target.trial_id == trial_id]
        if len(matches) != 1:
            raise ObservationDiagnosticError(f"trial_id {trial_id!r} does not appear exactly once in the list")
        return matches[0]
    if not 0 <= array_index < len(targets):
        raise ObservationDiagnosticError(
            f"array index {array_index} is outside the frozen trial list of {len(targets)} trials"
        )
    return targets[array_index]


def _attempt_token() -> str:
    """A token unique to this attempt: the Slurm identity where there is one
    (so an attempt directory is traceable to the job that made it), a UUID
    otherwise."""
    job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if job and task:
        return f"slurm{job}_{task}"
    if job:
        return f"slurm{job}"
    return f"local{uuid.uuid4().hex[:12]}"


def _run_one(args, *, forbid_array: bool) -> int:
    if forbid_array:
        present = [key for key in _ARRAY_ENV_KEYS if os.environ.get(key)]
        if present:
            # Section I: make it difficult to launch the full array when only
            # a benchmark is intended. A benchmark that finds itself inside
            # an array allocation refuses outright rather than running one
            # arbitrary task of it.
            print(
                f"REFUSING: the benchmark entry point was invoked under a Slurm array allocation ({present}). "
                "The benchmark runs exactly one trial and must not be submitted with --array. "
                "Use the array entry point deliberately if 24 tasks are intended.",
                file=sys.stderr,
            )
            return 2

    # The contract is loaded first: a trial roster is authenticated AGAINST
    # a specific fixed-support contract identity, so there is no point at
    # which a trial exists without one.
    contract = load_fixed_support_contract(args.contract)
    roster, trial_list_sha256 = load_trial_list(args.trial_list, contract)
    target = _resolve_target(roster, trial_id=args.trial_id, array_index=args.array_index)

    print(f"trial_id={target.trial_id}", flush=True)
    print(f"search_arm={target.search_arm}", flush=True)
    print(f"trial_list_sha256={trial_list_sha256}", flush=True)
    print(f"attempt_token={_attempt_token()}", flush=True)

    summary = run_trial_observation_diagnostic(
        trial=target,
        contract=contract,
        package_root=args.package_root,
        store_root=args.store_root,
        repo_root=args.repo_root,
        attempt_token=_attempt_token(),
        expected_basin_count=args.expected_basin_count,
    )
    if summary.get("reused_existing_shard"):
        print("reused an existing identity-identical completed shard; nothing recomputed", flush=True)
    else:
        print("cell_status_counts=" + json.dumps(summary["cell_status_counts"], sort_keys=True), flush=True)
        print(f"detail_rows={summary['detail_index']['n_rows']}", flush=True)
        print(f"elapsed_s={summary['elapsed_s']}", flush=True)
    return 0


def _run_reduce(args) -> int:
    contract = load_fixed_support_contract(args.contract)
    roster, _ = load_trial_list(args.trial_list, contract)
    expected_trials = {target.trial_id: target.search_arm for target in roster}
    summary = reduce_observation_diagnostic(
        store_root=args.store_root,
        out_dir=args.out_dir,
        expected_trials=expected_trials,
        expected_basin_ids=list(contract["basin_ids"]),
        expected_trial_count=args.expected_trial_count,
        expected_arm_counts=EXPECTED_ARM_COUNTS,
        expected_basin_count=args.expected_basin_count,
    )
    print("cell_status_counts=" + json.dumps(summary["cell_status_counts"], sort_keys=True), flush=True)
    print(f"n_cells={summary['n_cells']}", flush=True)
    print(f"outputs written to {args.out_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rd1_c4_d1",
        description="RD1-C4-D1 package-versus-validation-pickle observation diagnostic (measurement only).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _common(sub):
        sub.add_argument("--trial-list", required=True, help="frozen trial list JSON (order is authoritative)")
        sub.add_argument("--contract", required=True, help="frozen fixed-support contract JSON")
        sub.add_argument("--store-root", required=True, help="shard store root directory")
        sub.add_argument("--expected-basin-count", type=int, default=EXPECTED_BASIN_COUNT)

    benchmark = subparsers.add_parser("benchmark", help="run exactly one trial to measure cost")
    _common(benchmark)
    benchmark.add_argument("--package-root", required=True)
    benchmark.add_argument("--repo-root", required=True)
    benchmark.add_argument("--trial-id", required=True, help="explicit trial id -- benchmarks are never indexed")
    benchmark.set_defaults(array_index=None, handler=lambda args: _run_one(args, forbid_array=True))

    array_task = subparsers.add_parser("array-task", help="run exactly one trial of the array")
    _common(array_task)
    array_task.add_argument("--package-root", required=True)
    array_task.add_argument("--repo-root", required=True)
    array_task.add_argument("--array-index", type=int, default=None)
    array_task.add_argument("--trial-id", default=None)
    array_task.set_defaults(handler=lambda args: _run_one(args, forbid_array=False))

    reduce_parser = subparsers.add_parser("reduce", help="reduce all 24 completed shards (no partial mode)")
    _common(reduce_parser)
    reduce_parser.add_argument("--out-dir", required=True)
    reduce_parser.add_argument("--expected-trial-count", type=int, default=EXPECTED_TRIAL_COUNT)
    reduce_parser.set_defaults(handler=_run_reduce)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ObservationDiagnosticError, ReductionError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
