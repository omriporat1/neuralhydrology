"""Deterministic synthetic fixture v002 -- human-review refinement pass.

Regenerates the Sweep-v1 review-layer synthetic fixture with the same
narrative patterns as v001 (LR lower-boundary pressure, H=256 categorical
enrichment, interior embedding_dropout optimum, natural output_dropout
preference, late-best / C3-like-unstable / stable-typical trajectories,
failure+retry, heterogeneous cost) PLUS the specific fix requested after
human visual review of v001: the continuous LR-lower-boundary case must read
as genuinely early/uncertain at Boundary Review 1 (~13 valid Bayesian),
visibly strengthening at Boundary Review 2 (~25), and convincingly STRONG by
Final Closure (36) -- not settled (or still ambiguous) from the very first
checkpoint.

This is achieved with a staged Bayesian LR-exploitation schedule: proposals
sample uniformly across the domain through roughly the first half of the
campaign (so Boundary Review 1's slice carries no artificial concentration
or drift), then ramp into increasing lower-LR concentration so Boundary
Review 2 shows a real, but partial, drift/occupancy signal, and Final
Closure's full 36-trial population is dominated by the exploited tail.

Reuses `src.baseline.sweep_v1_campaign` for all campaign/domain/configuration
identity (`configuration_id`, `trial_id`, `proposal_id`,
`generate_random_control_rows`, `derive_trajectory_diagnostics`) and
`src.baseline.sweep_v1_review_analysis` for the boundary-pressure table used
to thread checkpoint-to-checkpoint evolution evidence -- this script defines
no scientific domain, objective, or fidelity of its own, and does not touch
the real (frozen) random-control manifest.

Run: python scripts/build_sweep_v1_visualization_fixture_v002.py [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline import sweep_v1_campaign as sweep
from src.baseline import sweep_v1_review_analysis as analysis
from src.baseline import sweep_v1_review_rendering as rendering

DEFAULT_OUTPUT_DIR = Path(".scratch_local/phase_b_sweep_v1_visualization_fixture_v002")
FIXTURE_RNG_SEED_LABEL = "sweep_v1_review_fixture_v002"
N_BAYESIAN_VALID = 36
WAVE_ID = f"{sweep.DOMAIN_VERSION}_wave1_v002"

# Staged exploitation schedule (§2): weight is 0 (pure uniform exploration)
# through progress <= RAMP_START, then ramps linearly to 1.0 (fully
# exploiting) by progress >= RAMP_END.  With these constants the
# checkpoint_12 slice (proposal_order 1-13, progress <= 0.361) is entirely
# pre-ramp -- no artificial LR concentration or drift can appear there --
# checkpoint_24 (order 1-25, progress <= 0.694) sits mid-ramp (a real but
# partial signal), and the final slice (order 1-36) includes a fully
# exploiting tail (order >= 32).
RAMP_START = 0.50
RAMP_END = 0.85


def _seeded_rng(*parts: Any) -> random.Random:
    return random.Random("::".join(str(p) for p in (FIXTURE_RNG_SEED_LABEL, *parts)))


def _true_objective(hp: dict[str, Any], rng: random.Random) -> float:
    """Ground-truth synthetic scoring surface -- identical shape to v001:
    LR lower-boundary pressure, H=256 categorical enrichment (with overlap),
    an interior embedding_dropout optimum, and a mild natural-boundary
    output_dropout preference, all with realistic noise/overlap."""
    lr, ed, od = hp["learning_rate"], hp["embedding_dropout"], hp["output_dropout"]
    hs, bs = hp["hidden_size"], hp["batch_size"]
    base = 0.340
    lr_term = 0.022 * (math.log10(1e-3) - math.log10(lr)) / (math.log10(1e-3) - math.log10(1e-4))
    hs_term = {64: -0.006, 128: 0.0, 256: 0.013}[hs]
    ed_term = -0.30 * (ed - 0.15) ** 2
    od_term = -0.018 * od
    bs_term = {128: 0.0, 256: 0.004, 512: -0.003}[bs]
    noise = rng.gauss(0, 0.013)
    return base + lr_term + hs_term + ed_term + od_term + bs_term + noise


def _exploitation_weight(proposal_order: int, n_total: int) -> float:
    progress = proposal_order / n_total
    return min(1.0, max(0.0, (progress - RAMP_START) / (RAMP_END - RAMP_START)))


def _sample_bayesian_hyperparameters(rng: random.Random, proposal_order: int, n_total: int) -> dict[str, Any]:
    """Staged exploration -> exploitation sampler (§2): early proposals are
    pure uniform draws across the legal domain; only once ``proposal_order``
    passes roughly the campaign midpoint does the sampler start concentrating
    toward the LR lower boundary and hidden_size=256, and even then a
    residual uniform-draw probability remains so the fixture stays
    realistically noisy rather than snapping to a deterministic edge."""
    exploitation_weight = _exploitation_weight(proposal_order, n_total)
    domain = sweep.SEARCH_DOMAIN
    log_lower, log_upper = math.log10(domain["learning_rate"]["lower"]), math.log10(domain["learning_rate"]["upper"])

    if rng.random() < exploitation_weight:
        log_lr = log_lower + rng.betavariate(1.0, 10.0) * (log_upper - log_lower)
    else:
        log_lr = rng.uniform(log_lower, log_upper)
    learning_rate = 10 ** log_lr

    if rng.random() < exploitation_weight:
        hidden_size = rng.choices([64, 128, 256], weights=[0.12, 0.23, 0.65])[0]
    else:
        hidden_size = rng.choice(domain["hidden_size"]["values"])

    return {
        "learning_rate": learning_rate,
        "embedding_dropout": rng.uniform(0.0, 0.4),
        "output_dropout": rng.uniform(0.0, 0.4),
        "hidden_size": hidden_size,
        "batch_size": rng.choice(domain["batch_size"]["values"]),
    }


def _runtime_seconds(rng: random.Random, hidden_size: int, batch_size: int) -> float:
    base = {64: 6900.0, 128: 7800.0, 256: 8700.0}[hidden_size]
    batch_adj = {128: 300.0, 256: 0.0, 512: -250.0}[batch_size]
    return max(3600.0, base + batch_adj + rng.gauss(0, 350))


# ------------------------------------------------------------------
# trajectory archetypes (rejection-sampled against the real, reused
# sweep.derive_trajectory_diagnostics to guarantee the intended pattern) --
# identical shapes to v001, reseeded under FIXTURE_RNG_SEED_LABEL v002.
# ------------------------------------------------------------------

def _stable_trajectory(rng: random.Random, true_best: float, peak_epoch: int) -> dict[int, float]:
    values = {}
    for e in range(1, 13):
        if e < peak_epoch:
            frac = 0.5 + 0.5 * (e / peak_epoch)
            level = true_best * (0.90 + 0.10 * frac)
        elif e == peak_epoch:
            level = true_best
        else:
            level = true_best - rng.uniform(0.0, 0.012)
        values[e] = level + rng.gauss(0, 0.006)
    return values


def _late_best_trajectory(rng: random.Random, true_best: float, peak_epoch: int) -> dict[int, float]:
    plateau = true_best - rng.uniform(0.014, 0.024)
    values = {}
    for e in range(1, 11):
        frac = e / 10
        values[e] = plateau * (0.75 + 0.25 * frac) + rng.gauss(0, 0.005)
    for e in (11, 12):
        values[e] = (true_best if e == peak_epoch else plateau) + rng.gauss(0, 0.004)
    return values


def _unstable_trajectory(rng: random.Random, true_best: float, peak_epoch: int) -> dict[int, float]:
    values = {}
    for e in range(1, peak_epoch):
        frac = 0.6 + 0.4 * (e / peak_epoch)
        values[e] = true_best * frac + rng.gauss(0, 0.006)
    values[peak_epoch] = true_best + rng.gauss(0, 0.002)
    collapse_floor = rng.uniform(0.01, 0.05)
    for e in range(peak_epoch + 1, 13):
        step = e - peak_epoch
        values[e] = collapse_floor + step * rng.uniform(0.01, 0.03) + rng.gauss(0, 0.004)
    return values


def _build_with_constraint(build_fn, check_fn, seed_label: str, max_tries: int = 120, **kwargs):
    for attempt in range(max_tries):
        rng = _seeded_rng(seed_label, attempt)
        trajectory = {e: round(v, 6) for e, v in build_fn(rng, **kwargs).items()}
        diagnostics = sweep.derive_trajectory_diagnostics(trajectory)
        if check_fn(diagnostics):
            return trajectory, diagnostics
    raise RuntimeError(f"could not satisfy fixture constraint for {seed_label!r} after {max_tries} attempts")


def _archetype_for_index(index: int, n_total: int) -> tuple[str, dict[str, Any]]:
    """Deterministically assign archetypes across the 36 Bayesian slots:
    2 unstable (C3-like), 3 late-best, remainder stable/typical."""
    if index in (13, 29):
        return "unstable", {"peak_epoch": 6 if index == 13 else 8}
    if index in (7, 20, 33):
        return "late_best", {"peak_epoch": 11 if index != 33 else 12}
    return "stable", {"peak_epoch": 7 + (index % 4)}


def _build_trial(*, rng: random.Random, hyperparameters: dict[str, Any], search_arm: str, proposal_order: int,
                  archetype: str, archetype_kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[int, float]]:
    true_best = _true_objective(hyperparameters, rng)
    seed_label = f"{search_arm}_traj_{proposal_order}"
    if archetype == "unstable":
        trajectory, diagnostics = _build_with_constraint(
            _unstable_trajectory,
            lambda d: d["best_minus_final"] > 0.10 and d["best_epoch"] <= 9,
            seed_label, true_best=true_best, peak_epoch=archetype_kwargs["peak_epoch"])
    elif archetype == "late_best":
        trajectory, diagnostics = _build_with_constraint(
            _late_best_trajectory,
            lambda d: d["late_best"] and d["late_gain_10_to_12"] > 0.006,
            seed_label, true_best=true_best, peak_epoch=archetype_kwargs["peak_epoch"])
    else:
        trajectory, diagnostics = _build_with_constraint(
            _stable_trajectory,
            lambda d: d["best_epoch"] <= 10 and d["best_minus_final"] < 0.02,
            seed_label, true_best=true_best, peak_epoch=archetype_kwargs["peak_epoch"])
    return diagnostics, trajectory


def _configuration_role(diagnostics: dict[str, Any], archetype: str) -> str:
    if archetype == "unstable":
        return "unstable"
    if archetype == "late_best":
        return "late_best"
    return "strong_stable" if diagnostics["best_score"] >= 0.365 else "typical"


def build_fixture() -> dict[str, pd.DataFrame]:
    """Build the full synthetic campaign tables (all attempts, all epochs,
    all proposals, all operations records) as pandas DataFrames."""
    trial_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    operations_rows: list[dict[str, Any]] = []
    job_counter = 9_500_000

    def _next_job_id() -> str:
        nonlocal job_counter
        job_counter += 1
        return f"SYN-JOB-{job_counter}"

    retry_slots_bayesian = {5, 22}  # proposals that fail once then retry

    for proposal_order in range(1, N_BAYESIAN_VALID + 1):
        hp_rng = _seeded_rng("bayesian_hp", proposal_order)
        hyperparameters = _sample_bayesian_hyperparameters(hp_rng, proposal_order, N_BAYESIAN_VALID)
        config_id = sweep.configuration_id(hyperparameters)
        pid = sweep.proposal_id("bayesian", proposal_order)
        archetype, archetype_kwargs = _archetype_for_index(proposal_order, N_BAYESIAN_VALID)
        score_rng = _seeded_rng("bayesian_score", proposal_order)
        diagnostics, trajectory = _build_trial(rng=score_rng, hyperparameters=hyperparameters,
                                               search_arm="bayesian", proposal_order=proposal_order,
                                               archetype=archetype, archetype_kwargs=archetype_kwargs)
        role = _configuration_role(diagnostics, archetype)

        generation = 1
        if proposal_order in retry_slots_bayesian:
            fail_rng = _seeded_rng("bayesian_fail", proposal_order)
            failed_trial_id = sweep.trial_id(config_id, execution_generation=1)
            crash_epoch = fail_rng.choice([2, 3])
            trial_rows.append({
                "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
                "search_arm": "bayesian", "proposal_id": pid, "configuration_id": config_id,
                "trial_id": failed_trial_id, "workflow_status": "failed", "objective_score": None,
                "best_epoch": None, "best_score": None, "final_epoch_score": None, "best_minus_final": None,
                "best_score_10": None, "best_score_12": None, "late_gain_10_to_12": None, "late_best": None,
                **hyperparameters, "runtime_seconds": round(fail_rng.uniform(500, 1600), 1),
                "gpu_hours": round(fail_rng.uniform(500, 1600) / 3600, 4),
                "execution_generation": 1, "retry_of_trial_id": None, "failure_category": "node_failure",
                "proposal_order": proposal_order, "valid_result_order": None, "wave_id": WAVE_ID,
                "boundary_review_checkpoint": None, "synthetic_archetype": None,
            })
            for e in range(1, crash_epoch + 1):
                trajectory_rows.append({
                    "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
                    "configuration_id": config_id, "trial_id": failed_trial_id, "search_arm": "bayesian",
                    "epoch": e, "median_raw_space_nse": round(trajectory[e] * 0.5, 6),
                    "evaluation_status": "incomplete_failed_attempt",
                })
            operations_rows.append({
                "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
                "configuration_id": config_id, "trial_id": failed_trial_id, "search_arm": "bayesian",
                "execution_generation": 1, "slurm_job_id": _next_job_id(), "slurm_state": "NODE_FAIL",
                "runtime_seconds": round(fail_rng.uniform(500, 1600), 1),
                "gpu_hours": round(fail_rng.uniform(500, 1600) / 3600, 4),
                "retry_of_trial_id": None, "failure_category": "node_failure",
            })
            generation = 2

        trial_id = sweep.trial_id(config_id, execution_generation=generation)
        runtime_rng = _seeded_rng("bayesian_runtime", proposal_order)
        runtime_seconds = _runtime_seconds(runtime_rng, hyperparameters["hidden_size"], hyperparameters["batch_size"])
        trial_rows.append({
            "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
            "search_arm": "bayesian", "proposal_id": pid, "configuration_id": config_id, "trial_id": trial_id,
            "workflow_status": "pass", "objective_score": diagnostics["best_score"], **diagnostics,
            **hyperparameters, "runtime_seconds": round(runtime_seconds, 1),
            "gpu_hours": round(runtime_seconds / 3600, 4), "execution_generation": generation,
            "retry_of_trial_id": sweep.trial_id(config_id, execution_generation=1) if generation > 1 else None,
            "failure_category": None, "proposal_order": proposal_order, "valid_result_order": proposal_order,
            "wave_id": WAVE_ID, "boundary_review_checkpoint": None, "synthetic_archetype": role,
        })
        for epoch, value in trajectory.items():
            trajectory_rows.append({
                "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
                "configuration_id": config_id, "trial_id": trial_id, "search_arm": "bayesian",
                "epoch": epoch, "median_raw_space_nse": value, "evaluation_status": "pass",
            })
        operations_rows.append({
            "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
            "configuration_id": config_id, "trial_id": trial_id, "search_arm": "bayesian",
            "execution_generation": generation, "slurm_job_id": _next_job_id(), "slurm_state": "COMPLETED",
            "runtime_seconds": round(runtime_seconds, 1), "gpu_hours": round(runtime_seconds / 3600, 4),
            "retry_of_trial_id": sweep.trial_id(config_id, execution_generation=1) if generation > 1 else None,
            "failure_category": None,
        })
        proposal_rows.append({
            "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION, "search_arm": "bayesian",
            "proposal_id": pid, "proposal_order": proposal_order, "configuration_id": config_id,
            **hyperparameters, "valid_result_order": proposal_order,
            "boundary_review_checkpoint": None, "wave_id": WAVE_ID,
        })

    # Real, frozen random-control manifest -- reused verbatim, untouched by
    # any v002 fixture change above.
    random_rows = sweep.generate_random_control_rows()
    retry_random_index = 6
    for row in random_rows:
        hyperparameters = {
            "learning_rate": float(row["learning_rate"]),
            "hidden_size": row["hidden_size"],
            "embedding_dropout": float(row["embedding_dropout"]),
            "output_dropout": float(row["output_dropout"]),
            "batch_size": row["batch_size"],
        }
        config_id = row["configuration_id"]
        pid = row["proposal_id"]
        manifest_index = row["manifest_index"]
        score_rng = _seeded_rng("random_score", manifest_index)
        diagnostics, trajectory = _build_trial(rng=score_rng, hyperparameters=hyperparameters,
                                               search_arm="random_control", proposal_order=manifest_index,
                                               archetype="stable",
                                               archetype_kwargs={"peak_epoch": 7 + (manifest_index % 4)})

        generation = 1
        if manifest_index == retry_random_index:
            fail_rng = _seeded_rng("random_fail", manifest_index)
            failed_trial_id = sweep.trial_id(config_id, execution_generation=1)
            crash_epoch = 2
            trial_rows.append({
                "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
                "search_arm": "random_control", "proposal_id": pid, "configuration_id": config_id,
                "trial_id": failed_trial_id, "workflow_status": "failed", "objective_score": None,
                "best_epoch": None, "best_score": None, "final_epoch_score": None, "best_minus_final": None,
                "best_score_10": None, "best_score_12": None, "late_gain_10_to_12": None, "late_best": None,
                **hyperparameters, "runtime_seconds": round(fail_rng.uniform(400, 1200), 1),
                "gpu_hours": round(fail_rng.uniform(400, 1200) / 3600, 4),
                "execution_generation": 1, "retry_of_trial_id": None, "failure_category": "slurm_timeout",
                "proposal_order": manifest_index, "valid_result_order": None, "wave_id": WAVE_ID,
                "boundary_review_checkpoint": None, "synthetic_archetype": None,
            })
            for e in range(1, crash_epoch + 1):
                trajectory_rows.append({
                    "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
                    "configuration_id": config_id, "trial_id": failed_trial_id, "search_arm": "random_control",
                    "epoch": e, "median_raw_space_nse": round(trajectory[e] * 0.5, 6),
                    "evaluation_status": "incomplete_failed_attempt",
                })
            operations_rows.append({
                "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
                "configuration_id": config_id, "trial_id": failed_trial_id, "search_arm": "random_control",
                "execution_generation": 1, "slurm_job_id": _next_job_id(), "slurm_state": "TIMEOUT",
                "runtime_seconds": round(fail_rng.uniform(400, 1200), 1),
                "gpu_hours": round(fail_rng.uniform(400, 1200) / 3600, 4),
                "retry_of_trial_id": None, "failure_category": "slurm_timeout",
            })
            generation = 2

        trial_id = sweep.trial_id(config_id, execution_generation=generation)
        runtime_rng = _seeded_rng("random_runtime", manifest_index)
        runtime_seconds = _runtime_seconds(runtime_rng, hyperparameters["hidden_size"], hyperparameters["batch_size"])
        trial_rows.append({
            "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
            "search_arm": "random_control", "proposal_id": pid, "configuration_id": config_id, "trial_id": trial_id,
            "workflow_status": "pass", "objective_score": diagnostics["best_score"], **diagnostics,
            **hyperparameters, "runtime_seconds": round(runtime_seconds, 1),
            "gpu_hours": round(runtime_seconds / 3600, 4), "execution_generation": generation,
            "retry_of_trial_id": sweep.trial_id(config_id, execution_generation=1) if generation > 1 else None,
            "failure_category": None, "proposal_order": manifest_index, "valid_result_order": manifest_index,
            "wave_id": WAVE_ID, "boundary_review_checkpoint": None, "synthetic_archetype": "typical",
        })
        for epoch, value in trajectory.items():
            trajectory_rows.append({
                "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
                "configuration_id": config_id, "trial_id": trial_id, "search_arm": "random_control",
                "epoch": epoch, "median_raw_space_nse": value, "evaluation_status": "pass",
            })
        operations_rows.append({
            "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION,
            "configuration_id": config_id, "trial_id": trial_id, "search_arm": "random_control",
            "execution_generation": generation, "slurm_job_id": _next_job_id(), "slurm_state": "COMPLETED",
            "runtime_seconds": round(runtime_seconds, 1), "gpu_hours": round(runtime_seconds / 3600, 4),
            "retry_of_trial_id": sweep.trial_id(config_id, execution_generation=1) if generation > 1 else None,
            "failure_category": None,
        })
        proposal_rows.append({
            "campaign_id": sweep.CAMPAIGN_ID, "domain_version": sweep.DOMAIN_VERSION, "search_arm": "random_control",
            "proposal_id": pid, "proposal_order": manifest_index, "configuration_id": config_id,
            **hyperparameters, "valid_result_order": manifest_index,
            "boundary_review_checkpoint": None, "wave_id": WAVE_ID,
        })

    return {
        "trial_df": pd.DataFrame(trial_rows),
        "trajectory_df": pd.DataFrame(trajectory_rows),
        "proposal_df": pd.DataFrame(proposal_rows),
        "operations_df": pd.DataFrame(operations_rows),
    }


def render_all_checkpoints(output_dir: Path, tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Render checkpoint_12 / checkpoint_24 / final, threading each
    checkpoint's boundary-pressure table into the NEXT checkpoint's render
    call as ``previous_boundary_df`` (§10) so the STRENGTHENING? panel and
    the standalone evolution figure have real previous-checkpoint evidence
    to compare against, not just a "first checkpoint" placeholder at every
    stage."""
    checkpoints = [
        ("checkpoint_12", 13, output_dir / "checkpoint_12"),
        ("checkpoint_24", 25, output_dir / "checkpoint_24"),
        ("final", N_BAYESIAN_VALID, output_dir / "final"),
    ]
    results: dict[str, dict[str, Any]] = {}
    previous_boundary_df = None
    for label, valid_count, checkpoint_dir in checkpoints:
        results[label] = rendering.render_checkpoint_packet(
            checkpoint_dir, trial_df=tables["trial_df"], trajectory_df=tables["trajectory_df"],
            proposal_df=tables["proposal_df"], operations_df=tables["operations_df"],
            checkpoint_label=label, checkpoint_valid_bayesian_count=valid_count,
            random_control_count=12, synthetic=True, previous_boundary_df=previous_boundary_df)
        trial_slice = analysis.checkpoint_slice(tables["trial_df"], valid_count, 12)
        previous_boundary_df = analysis.derive_boundary_pressure_table(trial_slice)
    return results


def _write_fixture_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    fixture_dir = output_dir / "_fixture_tables"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(fixture_dir / f"{name}.csv", index=False)
    (fixture_dir / "README.md").write_text(
        f"{rendering.SYNTHETIC_BANNER}\n\n"
        "Raw unsliced fixture tables (v002, human-review refinement pass) used to render all checkpoints.\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    tables = build_fixture()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_fixture_tables(args.output_dir, tables)
    results = render_all_checkpoints(args.output_dir, tables)
    for label, generated in results.items():
        print(f"[{label}] {len(generated)} artifacts -> {args.output_dir / label}")


if __name__ == "__main__":
    main()
