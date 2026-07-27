# Stage 1 validation and optimization foundation

Phase opened and closed 2026-07-26, immediately following the full-population
seed run's closure (`docs/decision_log.md`, "2026-07-26 — Stage 1
full-population seed-run closure decisions", Decisions 1-13). This phase
builds the **design, tooling, and documentation foundation** that a
subsequent, separately-scoped training/optimization phase will use — it
does **not** itself run a hyperparameter sweep, train an embedded-static or
EA-LSTM model, or evaluate the temporal test or spatial holdout sets. Every
Part below states its own non-goals; none of them are relaxed here.

This document is the single index for the phase. Full technical detail for
each Part lives in its own evidence subdirectory under
`reports/stage1_validation_optimization_foundation_v001/`.

## Binding decisions carried into this phase (unchanged, not re-litigated)

- **Primary metric:** median per-basin raw-space NSE on development
  validation (`docs/decision_log.md` Decision 7).
- **Early-stopping policy:** save every epoch; no stop before epoch 6;
  official validation every 2-3 epochs; min meaningful improvement 0.005
  median NSE; patience 3 validation events; max 30-40 epoch budget; always
  retain the best checkpoint (Decision 10). Policy engine implemented and
  tested (Part E); real training-orchestration integration remains pending.
- **Sealed test policy:** temporal-test and spatial-holdout data are never
  read for stopping, selection, sweep, or tracking decisions during Stage 1
  optimization (`docs/stage1_scientific_baseline_design.md` §2.4 / §9d).
- **Staged architecture strategy:** (1) seed config as trained, (2) CudaLSTM
  with a *verified* learned static representation, (3) EA-LSTM family, (4)
  architecture-specific light tuning, (5) deeper tuning for promoted
  families only (`docs/stage1_scientific_baseline_design.md` §9c). Hyper-
  parameters are not forced identical across families.

## Parts and status

| Part | Section | Scope | Status | Evidence |
|---|---|---|---|---|
| A | §5 | Seed percentile diagnostic closure: the center of the NSE distribution (p25-p99) was effectively flat across all 11 epochs (~0.02-0.04 NSE band); lower-tail percentiles (p1/p5) were unstable and non-monotonic. Supports the adopted early-stopping policy but does not by itself justify making it more aggressive, from one raw-static run. | Complete | `part_a_percentile_diagnostics/` |
| B | §6 | Static-pathway audit: confirmed seed used raw concatenation (`nn.Identity()`), not a learned embedding | Complete | `part_b_static_pathway_audit/` |
| C | §7 | **Deterministic provisional hydrograph-atlas selection v001** — deterministic event/case selection method plus a 24-basin realization, balanced by skill stratum, area class, and east/west geography. Selection/event-design tooling is complete; the final observed-vs-predicted atlas itself is not yet generated. Exact basin identity may be revised when the atlas is built, without reopening the selection framework. | Complete (deterministic provisional selection v001) | `part_c_hydrograph_atlas/` |
| D | §8 | **Provisional operational screening subset v001** — 400-basin development-validation subset, deterministic, reproducible, and stratified (geography, physical/hydroclimatic attributes, flow variability, seed skill); tracks the full 2,307-basin population well across all 11 seed-run epochs (Spearman 0.90, Kendall 0.82, max abs. diff 0.0053). Accepted for operational use (frequent feedback, early pruning), not yet scientifically authoritative or permanently frozen; the full 2,307-basin population remains authoritative for final run/checkpoint/architecture/hyperparameter selection. Prospective check: compare subset-based vs. full-population conclusions across the first ~3-5 materially different future optimization runs before reconsidering or freezing. | Complete (provisional operational subset v001) | `part_d_screening_subset/` |
| E | §9 | Early-stopping policy as an executable, restart-safe state machine. Engine implemented and tested; real training-orchestration integration remains pending. | Complete | `src/baseline/early_stopping.py`, `config/stage1_early_stopping_policy_v001.yaml`, `tests/test_early_stopping.py` (29 tests), `part_e_early_stopping/` |
| F | §10 | Optional, disabled-by-default W&B tracking wrapper. Implemented and tested; integration into the real training/validation harness remains pending. | Complete | `src/baseline/wandb_tracking.py`, `config/stage1_wandb_tracking_policy_v001.yaml`, `tests/test_wandb_tracking.py` (30 tests), `part_f_wandb_tracking/` |
| G | §11 | Evidence-grounded next-run operational defaults (Slurm resources, validation cadence) | Complete | `part_g_operational_defaults/` |
| H | §12 | Architecture-strategy status note; no winner declared | Complete | `part_h_architecture_strategy/` |
| I | §13 | First embedded-static CudaLSTM candidate (`embedded_static_cudalstm_pilot`): design/config + structural-smoke-only construction choice, not a scientific or tuned candidate; embedding width/depth/activation/dropout remain open optimization hyperparameters for a later phase | Complete | `src/baseline/nh_config_generation.py` extension, `part_i_embedded_static_pilot/`, 8 new tests in `tests/test_nh_config_generation.py` |
| J | §14 | Disposition of two retained diagnostic utility scripts: keep, recommend commit, no code change | Complete | `part_j_utility_script_disposition/` |
| K | §15 | Full test verification for Parts A/C/D/E/F/I | Complete | `part_k_test_verification/` |

## What this phase explicitly does NOT do

Restated verbatim from the phase's governing prohibitions, unchanged:

- Does not redesign the project.
- Does not launch a broad hyperparameter sweep.
- Does not launch a full embedded-static training run.
- Does not train EA-LSTM.
- Does not evaluate temporal test.
- Does not evaluate spatial holdout.
- Does not evaluate California.
- Does not modify the certified Stage 1 Compact Scientific Package.
- Does not regenerate the canonical basin splits.
- Does not commit any evidence or code automatically — see §19's proposed
  commit structure for a human-actionable recommendation only.

## Test verification summary (Part K, full detail in `part_k_test_verification/`)

- Focused tests for every Part with new code (A/C/D/E/F/I): 189 tests, all
  passing, run together as one combined invocation.
- Full repository regression suite (`flashnh-nh113-dev` env, 1094 tests
  collected): 1092 passed / 2 failed on the first pass; both failures
  (`test_package_audit.py`, `test_package_builder.py`) confirmed transient
  Windows file-lock flakiness unrelated to any module touched this phase —
  both pass cleanly in isolation. Net: 1094/1094.

## Commit-readiness pass (2026-07-27) — epoch-7 vs epoch-9 anchor-epoch sensitivity

Before recommending this phase's artifacts for commit, a conservative check
was run on one open ambiguity: Parts C and D's skill-quartile stratification
used epoch 9's per-basin NSE as a pragmatic plateau checkpoint. Using
already-persisted per-basin metrics for epochs 7 and 9 (no inference rerun),
skill-quartile edges, screening-subset selection, and hydrograph-atlas
selection were recomputed at epoch 7 with the identical policy/seed and
compared against the existing epoch-9 candidates.

**Result: exact basin membership is sensitive to the anchor checkpoint.**
Only 75.8% of the 2,307 development basins retain the same skill stratum
between epoch 7 and epoch 9 (vs. an initial ~90% review heuristic); the
screening-subset basin overlap is 82/400 (Jaccard 0.114); the atlas overlap
is 3/24 (Jaccard 0.067); and the epoch-7 candidate subset tracks the
full-population validation curve markedly worse than the epoch-9 candidate
(Spearman 0.48 vs 0.90, Kendall 0.35 vs 0.82, max abs. median-NSE diff 0.0175
vs 0.0053). Full detail, machine-readable comparison data, and checksums:
`reports/stage1_validation_optimization_foundation_v001/commit_readiness_epoch7_epoch9_sensitivity/`
(untracked).

**Final status resolution (2026-07-27, same date).** This sensitivity is
expected given the selection design's many small composite strata and
seeded within-cell draws, and does not invalidate either artifact's
*operational* purpose (frequent cheap feedback for the subset; structured
visual-inspection breadth for the atlas) — it does mean neither artifact is
yet a permanent, scientifically authoritative sample. Per the user-adopted
interpretation, the existing epoch-9 candidates are retained exactly as
built (no regeneration, no replacement, no new skill definition such as
across-epoch-median) and reclassified as **provisional**: the screening
subset as `provisional operational screening subset v001` (see Part D row
above), the atlas selection as `deterministic provisional hydrograph-atlas
selection v001` (see Part C row above). Full decision text:
`docs/decision_log.md` (2026-07-27, "final status resolution" entry).

## Cross-references

- `docs/decision_log.md` — full decision history, including the seed-run
  closure decisions this phase builds on.
- `docs/stage1_scientific_baseline_design.md` §9c/§9d/§10 — the binding
  policy this phase operationalizes (addenda added 2026-07-26 pointing back
  here).
- `docs/FLASHNH_CURRENT_STATE.md` — project-level current-state summary,
  updated to reference this phase's completion.
