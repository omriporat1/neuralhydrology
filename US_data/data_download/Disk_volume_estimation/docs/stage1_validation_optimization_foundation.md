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
| L | §16 | Post-`emb128x64_seedA` roadmap addendum: Stage A (structural contrasts) vs. Stage B (proper HPO) framing, hydrograph-diagnostic timing, W&B adoption sequencing, and the `max_updates_per_epoch` multi-fidelity direction. Documentation only — no run launched, no hydrograph generated, no W&B tracking enabled, no update cap adopted. | Complete (roadmap documented; several sub-items remain open/deferred by design — see §16) | `docs/decision_log.md` (2026-08-02 entry) |

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

## Part L — post-`emb128x64_seedA` roadmap addendum (§16, 2026-08-02)

Documentation-only addendum, written after `emb128x64_seedA` (the first of
the six `stage1_lead06_pilot_v001` runs) completed screening and stopped at
epoch 15 (`docs/decision_log.md`, 2026-07-30 closure entry; job `45722908`).
**This addendum launches nothing, generates no evidence, enables no
tracking, and adopts no numerical update cap.** Full decision text:
`docs/decision_log.md`, 2026-08-02 entry.

**L.1 — Stage A vs. Stage B (adopted framing, binding).** Stage A is the
existing closed six-run `stage1_lead06_pilot_v001` structural pilot
(Part I / `docs/stage1_lead06_pilot_v001.md`) — it answers only raw-vs-
learned-embedding static pathway, approximate embedding shape, and limited
two-seed robustness, with `seq_length`, `hidden_size`, `output_dropout`,
`batch_size`, learning rate, embedding dropout/activation, and scheduler
frozen identically across all six runs. **Stage A must not be described as
a hyperparameter sweep.** Stage B is proper HPO — deferred, not designed
here — and begins only once Stage A yields enough structural evidence to
select or narrow the architecture family. Likely Stage B dimensions:
learning rate, hidden size, output dropout, embedding width/depth,
embedding dropout, batch size, sequence length, and possibly
activation/scheduler after the architecture family is chosen. Stage B's
search space, optimizer, trial budget, fidelity policy, and promotion rules
remain unfrozen.

**L.2 — Remaining Stage A candidates (adopted direction, not a launch
authorization).** The preferred next candidate is `raw_seedA` (clean
same-seed contrast against completed `emb128x64_seedA`). Results are
reviewed between candidates; the repository is not committed to launching
`emb128x64_seedB`, `emb64_seedA`, `emb128_seedA`, or `raw_seedB`
automatically, sequentially, or in parallel — any may be deprioritized if
its scientific value becomes redundant given earlier results. Parallel
execution of `raw_seedA` and `emb128x64_seedB` remains an available
operational option, not the current preferred workflow. See
`docs/stage1_lead06_pilot_v001.md`'s current-status/next-step section for
the per-candidate detail.

**L.3 — Hydrograph timing (adopted direction; implementation not started).**
Hydrographs move earlier in the workflow as an early scientific diagnostic:
a deterministic compact ~6-8-basin comparison panel (reusable across
candidates, reproducibly derived from the existing Part C atlas-selection
metadata rather than chosen ad hoc) plus the existing provisional 24-basin
atlas (Part C), both showing observed-vs-predicted discharge in raw space
with full validation-period context, selected event windows, basin ID, and
supporting metrics. They are an early diagnostic available for inspection
and grounds for pausing on suspicious behavior or strong conflict with
aggregate metrics — **not** a mandatory formal approval gate after every
routine candidate or every short low-fidelity HPO trial. No hydrograph has
been generated by this addendum; the compact-panel plotting/assembly code
does not yet exist (confirmed absent during the preceding inspection pass).
Generated figures and large evaluation artifacts remain untracked, per
`docs/repo_policy.md`.

**L.4 — W&B adoption sequencing (adopted direction; not yet qualified or
enabled).** Order: (1) ordinary tracking qualification, (2) an offline-mode
real-path test preferred before relying on online tracking, (3) controlled
live tracking for one structural candidate, (4) sweeps only after Stage B's
search space/objective/fidelity/promotion rules are frozen. Repository code
remains authoritative for basin membership, sealed-set protection, metric
computation, early stopping, checkpoint provenance, and package identity;
W&B is telemetry/comparison infrastructure, never the scientific source of
truth (Part F). No W&B tracking has been qualified or enabled by this
addendum — the wrapper remains `enabled: false` / `mode: disabled`
(`config/stage1_wandb_tracking_policy_v001.yaml`, unchanged). A
project-specific W&B learning guide remains a required, not-yet-written
next documentation artifact — not authored in this addendum.

**L.5 — Multi-fidelity direction (adopted direction; no cap adopted).** The
preferred first multi-fidelity mechanism for Stage B is NeuralHydrology's
built-in `max_updates_per_epoch`, in preference to a reduced training-basin
subset — it preserves all 2,307 development basins as the sampling
universe, avoids defining a second basin population, and reuses an existing
NH mechanism. **Provisional, non-binding fidelity fractions**: low ≈10-15%
of one confirmed full uncapped epoch's optimizer-update count; medium
≈25-50%; full = uncapped (unchanged current practice). These are starting
points for later calibration, not approved integer update caps. The exact
full-epoch DataLoader/optimizer-update count has not been established
against the real Moriah NeuralHydrology 1.13 environment and real training
configuration; a same-repository, local NeuralHydrology **1.12** source
inspection (a different version than the 1.13 that actually trains on
Moriah) produced only a rough, explicitly-labeled order-of-magnitude
inference during the preceding inspection pass — that inference is not
restated as a number here, is not authoritative, and does not establish
real Moriah 1.13 behavior. Before any sweep, several materially different
configurations (naturally, the Stage A candidates once complete) must be
compared at full/medium/low fidelity to test whether ranking is
approximately preserved — absolute NSE agreement across fidelities is not
required.

**L.6 — Capped-update stopping and promotion (explicitly open, not
implemented).** Unresolved, requiring technical calibration before any
capped-fidelity screening runs: epoch-based vs. cumulative-update-based
screening/patience; the unit of patience for capped trials; fair budget
comparison across fidelities; promotion thresholds; continuation-from-
checkpoint vs. restart-from-seed for promoted finalists; learning-rate
scheduling implications; reliable cumulative-optimizer-update provenance
across a resumed/continued run (NeuralHydrology's own resumed-logger update
count reconstructs `len(loader) * epoch`, assuming every prior epoch ran
the full uncapped loader length — a known mismatch if a prior epoch was
itself capped; **NH's internal resumed logger count is not adopted as
authoritative for this purpose until its capped-epoch behavior is verified
against the real environment**). Current provisional recommendation for the
first capped-update campaign, **not an immutable scientific decision**: use
capped fidelities for screening/ranking only; restart promoted finalists
from their original seed at full fidelity under the already-qualified
uncapped protocol, rather than continuing from a lower-fidelity checkpoint;
reconsider checkpoint continuation at higher fidelity only after evidence
shows it is methodologically fair and operationally reliable.

## Cross-references

- `docs/decision_log.md` — full decision history, including the seed-run
  closure decisions this phase builds on, and the 2026-08-02 roadmap entry
  (Part L).
- `docs/stage1_scientific_baseline_design.md` §9c/§9d/§10 — the binding
  policy this phase operationalizes (addenda added 2026-07-26 pointing back
  here). Not edited by Part L; still the right place for a future small
  cross-reference to the Stage A/Stage B roadmap once one is judged
  necessary.
- `docs/FLASHNH_CURRENT_STATE.md` — project-level current-state summary,
  updated to reference this phase's completion and the Part L roadmap.
- `docs/stage1_lead06_pilot_v001.md` — Stage A run design and current
  status/next-step section (added alongside Part L).
