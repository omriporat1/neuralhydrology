"""Non-W&B-agent-driven entry point for one Sweep-v2 six-axis random-control trial.

Random-control rows are a fixed, pre-committed manifest
(``config/stage1_phase_b_sweep_v2_six_axis_random_control_v001_random_control_manifest.json``)
frozen AFTER Bayesian observation 1 and drawn only from the exact committed
v2 six-axis priors -- unlike the Bayesian arm, no live W&B search proposes
them, so this script never imports or calls ``wandb`` at all and never
contacts the production Bayesian controller. It reuses exactly the same
prepare -> write -> execute -> interpret path as the Bayesian bridge
(``scripts/run_sweep_v2_six_axis_wandb_bridge.py``):

  1. Runtime provenance is pinned FIRST via the shared
     ``src.baseline.sweep_v1_runtime_contract.verify_commit_and_interpreter``
     (exact HEAD == ``--expected-commit``, clean tracked tree, exact
     canonical interpreter) -- the same module the Sweep-v1 production
     launcher uses; no second provenance framework is introduced here.
  2. ``prepare_random_control_proposal_v2(...)`` -- verifies the manifest's
     frozen SHA-256 and that the requested row is an exact committed manifest
     row (never regenerated/recomputed), then runs the identical
     ``_prepare_proposal_v2`` path the Bayesian arm uses with
     ``expected_arm="random_control"``.
  3. ``write_prepared_proposal_v2(...)``.
  4. ``run_prepared_trial_in_production_v2(...)`` -- the same real,
     focused-tested Sweep-v2 execution/interpretation layer the Bayesian arm
     uses; arm-agnostic per ``src/baseline/sweep_v2_six_axis_execution.py``'s
     ``run_prepared_trial_in_production_v2`` docstring (neither
     ``build_execution_context_v2`` nor ``execute_prepared_trial_v2`` branches
     on ``search_arm``). Explicit retry lineage (``--retry-of-trial-id`` /
     ``--prior-attempt-generation``) is resolved through the established v2
     retry contract (``derive_exact_retry_identity_v2``) and threaded into
     the durable execution provenance and review records.

One Slurm allocation runs exactly one random-control row -- the same
one-allocation-per-trial rule as ``run_sweep_v2_six_axis_wandb_agent_moriah.sbatch``
applies to the Bayesian arm, enforced here by requiring exactly one row
index per invocation rather than looping over the manifest. The intended
execution shape is three waves of four concurrent independent jobs
(row indices 0-3, 4-7, 8-11), each writing to ``output_root / <trial_id>``
so the four concurrent output roots never collide.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import sweep_v2_six_axis_random_control as rc
from src.baseline.sweep_v1_retry import SweepV1RetryError
from src.baseline.sweep_v1_runtime_contract import RuntimeContractError, verify_commit_and_interpreter
from src.baseline.sweep_v2_six_axis_campaign import _AXES_V2
from src.baseline.sweep_v2_six_axis_execution import run_prepared_trial_in_production_v2
from src.baseline.sweep_v2_six_axis_production_adapter import (
    PreparationPathsV2, prepare_random_control_proposal_v2, write_prepared_proposal_v2,
)
from src.baseline.sweep_v2_six_axis_retry import SweepV2RetryError, derive_exact_retry_identity_v2

_DEFAULT_MANIFEST = ROOT / (
    "config/stage1_phase_b_sweep_v2_six_axis_random_control_v001_random_control_manifest.json"
)
# Authoritative source for these real operational paths:
# ``src/baseline/sweep_v2_six_axis_wandb_bridge_registration.py`` (_REAL_* constants)
# and ``config/stage1_v2_common120_fixed_support_artifact_identity_v001.json``
# (artifact.deployment_provenance_moriah_absolute_path). Kept as literals here
# to avoid importing the W&B-registration module into this W&B-free runner.
_DEFAULT_PACKAGE_ROOT = "/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_scientific_package_v002"
_DEFAULT_SCREENING_BASIN_IDS = (
    "/sci/labs/efratmorin/omripo/Flash-NH/data/screening_subsets/"
    "stage1_provisional_operational_screening_subset_v001/screening_subset_basin_ids.txt"
)
_DEFAULT_FIXED_SUPPORT_CONTRACT = (
    "/sci/labs/efratmorin/omripo/Flash-NH/data/fixed_support_contracts/"
    "stage1_v2_common120_fixed_support_contract_v001.json"
)
# Canonical Moriah Flash-NH interpreter -- identical value/convention as the
# Sweep-v1 production launcher (see scripts/run_sweep_v1_exact_retry_moriah.sbatch).
_CANONICAL_MORIAH_PYTHON = "/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/bin/python"

_FULL_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
# Set to "resolve_only" to exercise argument validation + retry-lineage
# resolution without touching the runtime contract, the package, or the
# filesystem (used by the focused launcher/runner contract tests).
_SELFTEST_ENV = "FLASHNH_SWEEP_V2_RANDOM_CONTROL_SELFTEST"


def _validate_expected_commit(value: str) -> str:
    """Require a full, legal 40-hex-character git commit identity."""
    if not isinstance(value, str) or _FULL_COMMIT_RE.fullmatch(value) is None:
        raise SystemExit(
            f"--expected-commit must be a full 40-hex-character git commit id, got {value!r}"
        )
    return value


def _synthesize_attempt001_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the frozen attempt001 record shape for one committed row,
    so the established v2 retry contract can derive the exact advanced
    identity from it. The row's own committed
    ``configuration_id``/``proposal_id``/``trial_id_attempt001`` are used
    verbatim -- nothing is re-derived from the hyperparameters here."""
    return {
        "hyperparameters": {axis: row[axis] for axis in _AXES_V2},
        "search_arm": rc.RANDOM_CONTROL_ARM,
        "proposal_order": row["proposal_order"],
        "execution_generation": 1,
        "configuration_id": row["configuration_id"],
        "proposal_id": row["proposal_id"],
        "trial_id": row["trial_id_attempt001"],
        "wandb_sweep_id": None,
        "support_contract_version": rc.SUPPORT_CONTRACT_VERSION_V2,
        "support_contract_sha256": rc.SUPPORT_CONTRACT_SHA256_V2,
    }


def resolve_random_control_retry_lineage(
    row: Mapping[str, Any], *, execution_generation: int, retry_of_trial_id: "str | None",
    prior_attempt_generations: Sequence[int] = (),
) -> "tuple[str | None, str]":
    """Resolve ``(retry_of_trial_id, expected_trial_id)`` for one frozen row.

    ``execution_generation == 1`` represents attempt001 and has no retry
    predecessor -- a supplied ``--retry-of-trial-id`` or any prior attempt
    generation is rejected. A later generation is an infrastructure retry of
    the SAME frozen row: it must name the row's own frozen attempt001
    ``trial_id`` as its predecessor, and the exact advanced identity is then
    derived through the established v2 retry contract
    (:func:`derive_exact_retry_identity_v2`, which retains
    ``configuration_id``/``proposal_id`` and advances only the attempt
    suffix), never re-invented here. Impossible or inconsistent combinations
    fail before execution.
    """
    if not isinstance(execution_generation, int) or execution_generation < 1:
        raise SystemExit(
            f"--execution-generation must be a positive integer, got {execution_generation!r}"
        )
    prior = tuple(int(g) for g in (prior_attempt_generations or ()))
    attempt001_trial_id = row["trial_id_attempt001"]

    if execution_generation == 1:
        if retry_of_trial_id is not None:
            raise SystemExit(
                "execution_generation 1 is attempt001 and has no retry predecessor; "
                "omit --retry-of-trial-id"
            )
        if prior:
            raise SystemExit("execution_generation 1 cannot declare prior attempt generations")
        return None, attempt001_trial_id

    if retry_of_trial_id != attempt001_trial_id:
        raise SystemExit(
            "an infrastructure retry (execution_generation > 1) must set --retry-of-trial-id to the "
            f"frozen attempt001 trial_id for proposal_order {row['proposal_order']} "
            f"({attempt001_trial_id!r})"
        )
    try:
        identity = derive_exact_retry_identity_v2(
            _synthesize_attempt001_record(row),
            execution_generation=execution_generation,
            prior_attempts=[{"execution_generation": g} for g in prior],
        )
    except (SweepV2RetryError, SweepV1RetryError) as exc:
        raise SystemExit(f"invalid random-control retry lineage: {exc}") from exc
    return identity["retry_of_trial_id"], identity["trial_id"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", type=Path, default=_DEFAULT_MANIFEST)
    parser.add_argument("--row-index", type=int, required=True, help="0-based index into the committed manifest's rows.")
    parser.add_argument("--package-root", type=Path, default=Path(_DEFAULT_PACKAGE_ROOT))
    parser.add_argument("--screening-basin-ids", type=Path, default=Path(_DEFAULT_SCREENING_BASIN_IDS))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fixed-support-contract-path", type=Path, default=Path(_DEFAULT_FIXED_SUPPORT_CONTRACT))
    parser.add_argument("--baseline-policy-path", type=Path,
                        default=ROOT / "config/stage1_scientific_baseline_v001.yaml")
    parser.add_argument("--policy-overlay-path", type=Path,
                        default=ROOT / "config/stage1_scientific_baseline_v2_six_axis_overlay_v001.yaml")
    parser.add_argument("--base-pilot-policy-path", type=Path,
                        default=ROOT / "config/stage1_lead06_pilot_v001.yaml")
    parser.add_argument("--execution-generation", type=int, default=1,
                        help="Infrastructure-retry generation; retains configuration/proposal identity, "
                             "only the trial attempt suffix advances.")
    parser.add_argument("--expected-commit", required=True,
                        help="Full 40-hex-character git commit this launch is pinned to; the canonical "
                             "checkout's HEAD must equal it and its tracked tree must be clean.")
    parser.add_argument("--expected-runtime-python", type=Path, default=Path(_CANONICAL_MORIAH_PYTHON),
                        help="Absolute path of the canonical interpreter this trial must run under.")
    parser.add_argument("--retry-of-trial-id", default=None,
                        help="For an infrastructure retry (execution-generation > 1): the frozen "
                             "attempt001 trial_id of this same row.")
    parser.add_argument("--prior-attempt-generation", type=int, action="append", default=None,
                        help="Repeatable. A previously reserved/attempted execution generation for this "
                             "row's trial family; the requested generation must not already be listed.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _validate_expected_commit(args.expected_commit)

    payload = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if not (0 <= args.row_index < len(rows)):
        raise SystemExit(f"--row-index {args.row_index} out of range for {len(rows)} committed rows")
    row = rows[args.row_index]

    resolved_retry_of, expected_trial_id = resolve_random_control_retry_lineage(
        row, execution_generation=args.execution_generation,
        retry_of_trial_id=args.retry_of_trial_id,
        prior_attempt_generations=args.prior_attempt_generation or (),
    )

    if os.environ.get(_SELFTEST_ENV) == "resolve_only":
        print(json.dumps({
            "selftest": "resolve_only", "row_index": args.row_index,
            "proposal_order": row["proposal_order"],
            "execution_generation": args.execution_generation,
            "attempt001_trial_id": row["trial_id_attempt001"],
            "retry_of_trial_id": resolved_retry_of,
            "expected_trial_id": expected_trial_id,
        }, indent=2))
        return 0

    try:
        verify_commit_and_interpreter(
            repo_root=ROOT, expected_commit=args.expected_commit,
            expected_runtime_python=str(args.expected_runtime_python),
        )
    except RuntimeContractError as exc:
        raise SystemExit(f"runtime provenance check failed, refusing to run trial: {exc}") from exc

    canonical_splits = ROOT / "config" / "stage1_baseline_splits_v001"
    paths = PreparationPathsV2(
        baseline_policy_path=args.baseline_policy_path, policy_overlay_path=args.policy_overlay_path,
        package_root=args.package_root, splits_dir=canonical_splits,
        screening_basin_ids_path=args.screening_basin_ids,
        fixed_support_contract_path=args.fixed_support_contract_path,
    )

    prepared = prepare_random_control_proposal_v2(
        row=row, manifest_path=args.manifest_path, paths=paths,
        execution_generation=args.execution_generation,
    )
    attempt_suffix = f"__attempt{args.execution_generation:03d}"
    if not prepared.trial_id.endswith(attempt_suffix) or not expected_trial_id.endswith(attempt_suffix):
        raise SystemExit(
            f"prepared trial_id {prepared.trial_id!r} / expected {expected_trial_id!r} do not carry the "
            f"attempt suffix for execution generation {args.execution_generation}"
        )
    output_dir = args.output_root / prepared.trial_id
    record = write_prepared_proposal_v2(prepared, output_dir)

    outcome = run_prepared_trial_in_production_v2(
        prepared_record=record, output_dir=output_dir, paths=paths,
        base_pilot_policy_path=args.base_pilot_policy_path,
        retry_of_trial_id=resolved_retry_of,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    trial = outcome["review_records"]["trial_summary"]
    print(json.dumps({"trial_id": record["trial_id"], "valid": outcome["valid"],
                      "retry_of_trial_id": resolved_retry_of,
                      "objective_score": trial["objective_score"]}, indent=2))
    return 0 if outcome["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
