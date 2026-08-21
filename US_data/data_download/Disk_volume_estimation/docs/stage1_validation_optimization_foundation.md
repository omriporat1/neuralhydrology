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
| L | §16 | Post-`emb128x64_seedA` roadmap addendum (L.1-L.4); `max_updates_per_epoch` capped-update mechanism, implementation and calibration (L.5-L.9); 25k Seed-A embedding-shape neighborhood screening closed (L.10); 50k Seed-A embedding-shape comparison closed, `[128,32]` adopted as working default (L.11); LR-A design frozen (L.12) and implemented (L.13); LR-A closed, `3e-4` adopted as provisional Phase-A working anchor over carried-forward range `1e-4`-`1e-3`, cadence and W&B findings recorded (L.14); Hidden-size range characterization (Phase-A) design frozen (L.15) and closed, H=128 provisional anchor/H=64 live alternative/Phase-B support `{64,128,256}`, plus frozen validation-compatible 8-basin hydrograph panel v001 and standing Phase-A hydrograph rule (L.16); Embedding-dropout range characterization (Phase-A) design frozen, one-dimensional five-candidate `embedding_dropout` range characterization (`0.00,0.05,0.10,0.20,0.40`) at the LR-A/Hidden-size-A anchors, all five fresh, dropout-specific fidelity caveat (L.17); implementation and preparation-only validation complete, all 8 planned items built and tested, still not launched (L.18); Embedding-Dropout-A **closed** — five-candidate real Moriah campaign executed, weak sensitivity over `0.00`-`0.40`, no candidate robustly dominates, `drop10` retained as provisional working anchor (not proven optimal), exact/deterministic reproducibility against the historical H=128/dropout=0.10 comparator, standing 8-basin hydrograph sanity check clean (no repeated candidate-specific pathology), revised Phase-A/Phase-B roadmap adopted superseding L.1/L.10's sequence-length exclusion (L.19). Real Moriah execution occurred for L.3c, L.7-L.9, L.10's screening batch, L.11's 50k comparison, L.14's four LR-A runs, L.15/L.16's four hidden-size runs plus hydrograph rendering, and L.19's five embedding-dropout training runs plus retrospective diagnostic evaluation, optimizer/update verification, and hydrograph rendering; L.17/L.18 and this row's own documentation edits are otherwise documentation-only. | Complete through L.19 (several sub-items remain open/deferred by design — see §16; no final Stage 1 hyperparameters selected; final embedding-dropout selection deferred to Phase B; next stage: reusable Phase-A/HPO campaign infrastructure consolidation) | `docs/decision_log.md` (2026-08-02, 2026-08-04, 2026-08-05, 2026-08-06, 2026-08-08, 2026-08-09, 2026-08-10, 2026-08-11, 2026-08-13 entries) |

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
embedding dropout, batch size, and possibly activation/scheduler after the
architecture family is chosen. Stage B's search space, optimizer, trial
budget, fidelity policy, and promotion rules remain unfrozen.
**Corrected (2026-08-05, see Part L.10):** sequence length is explicitly
excluded from this list — it is fixed at 24 for the current model family
and is not an ordinary Stage B/HPO dimension. Alternative sequence lengths
define separate temporal-context model families (different antecedent-
memory assumptions, input construction, compute/memory cost, and cross-
basin response-time interpretation), not a value to be tuned inside this
funnel; any future sequence-length study compares whole model families
against a mature 24-hour model and is out of scope for Stage B as framed
here. **Further revised (2026-08-13, see Part L.19):** the
Embedding-Dropout-A closure adopts a revised roadmap that schedules a
dedicated Sequence-Length-A characterization (`seq_length={12,24,48,72}`),
reframing sequence length as a bounded, structural/calibratable model
parameter rather than a dimension permanently excluded from calibration.
This passage and L.10's correction of it are preserved as historical
framing, not rewritten — see L.19 for the current framing.

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
routine candidate or every short low-fidelity HPO trial. Generated figures
and large evaluation artifacts remain untracked, per `docs/repo_policy.md`.

**L.3a — rendering implementation (local, synthetic-verified; no real
hydrograph generated).** The local rendering machinery now exists:
`src/baseline/hydrograph_rendering.py` (library) plus
`scripts/render_stage1_hydrographs.py` (thin CLI), with focused tests in
`tests/test_hydrograph_rendering.py`. It reuses, rather than reimplements,
every scientific building block: `nh_seed_evaluation.period_results_path`/
`load_period_results`/`basin_netcdf_path` for result-pickle and basin-NetCDF
I/O; `nh_raw_space_evaluation` for basin-area self-derivation, mm/h -> m^3/s
conversion, and every skill metric; `hydrograph_atlas_events.select_atlas_events`
for event-window selection (observed discharge only — that function has no
parameter that could carry predicted/simulated discharge, so predictions
cannot influence event selection by construction).

- **Input contract**: an NH run directory + period + epoch (resolved via
  `period_results_path`, never reconstructed independently) or an explicit
  result-pickle path; a package root (for basin-area self-derivation); a
  target variable; a lead time; the existing hydrograph-atlas selection CSV
  (`hydrograph_atlas_basin_selection.csv`); an output directory; a mode
  (`compact` / `full` / `both`). Only `period="validation"` is permitted —
  the safest boundary until a separately-approved mode for other periods
  exists; any other period is rejected immediately.
- **Deterministic compact selection**: `select_compact_basins` derives ~6-8
  basins from the 24-basin atlas CSV by greedily maximizing coverage across
  `skill_stratum` x `area_class` x `geo_side` (sorted by `gauge_id` first, so
  the result does not depend on input row order; no random seed needed). The
  requested fourth dimension, flashy-vs-smoother hydrologic behavior, is
  **not** supported by current atlas metadata — the only superficially
  similar field, `hydro_class`, is an aridity (wet/dry) tercile of
  `ari_ix_uav` (`src/baseline/splits.py`), not a hydrograph-shape
  classification. No substitute was invented; this limitation is recorded
  verbatim in every rendering manifest's `compact_selection.dimension_omitted`
  field.
- **Output structure** (all untracked, written under a caller-supplied
  `--out-dir`): `compact_panel.png`, `atlas/<basin_id>.png` (one file per
  atlas basin), `per_basin_metrics.csv` (full raw-space metric table),
  `compact_basin_membership.json`, `event_window_table.csv`,
  `rendering_manifest.json` (input paths + SHA-256, epoch/period/target
  variable, compact-selection rule/version, output file list + checksums,
  generation timestamp, explicit `event_selection_basis:
  "observed_discharge_only"` and `raw_space_conversion_source` fields), and
  `summary.json`. The CLI additionally writes `run_command.txt`.
- **Safety**: missing basins, missing target variables, malformed
  result-pickle structure, and non-`validation` periods all raise
  `HydrographRenderingError` immediately rather than being silently skipped;
  no observation/prediction values are interpolated.
- **Selected checkpoint vs. stop epoch (do not conflate)**: `emb128x64_seedA`
  stopped screening at epoch 15 (`patience_exhausted`), but its *selected*
  checkpoint — the one the reference hydrograph atlas must render — is
  **epoch 6** (median raw-space screening NSE `0.20454161610527344`; see
  `docs/decision_log.md`, 2026-07-30 closure entry). The renderer itself is
  epoch-agnostic: `--epoch`/`epoch=` is always caller-supplied with no
  default and no internal notion of "stop epoch" vs. "selected epoch" —
  callers are responsible for passing the selected checkpoint's epoch, not
  the run's last or stopping epoch.
- **Verification performed**: 28 focused tests (synthetic fixtures only, no
  NeuralHydrology/torch import) plus a manual CLI dry run against a small
  synthetic 6-basin fixture (compact + full-atlas modes, and `--dry-run`),
  all local; outputs inspected then deleted. No Moriah/h2o access. **The
  real `emb128x64_seedA` hydrograph atlas has not been generated** — doing
  so requires the real epoch-6 `validation_results.p` (the selected
  checkpoint, not the epoch-15 stop point) and basin-area evidence for the
  24 atlas basins, per `docs/repo_policy.md`'s remote-evidence policy, which
  this task explicitly did not perform. **The full certified package root
  must not be transferred locally by default** — see the real-execution
  workflow note below. **Superseded (2026-08-02): the real atlas has since
  been generated and visually reviewed — see L.3c below.**

Example command (placeholders, not machine-specific paths):

```bash
python scripts/render_stage1_hydrographs.py \
    --run-dir /path/to/nh_run_dir \
    --period validation --epoch 6 \
    --package-root /path/to/stage1_scientific_package_v002 \
    --target-variable qobs_mm_per_h_lead06 --lead-hours 6 \
    --atlas-csv reports/.../hydrograph_atlas_basin_selection.csv \
    --out-dir tmp/stage1_hydrograph_rendering_v001 --mode both
```

**L.3b — real execution/transfer workflow (not yet chosen or performed).**
The renderer's only real-data dependencies are: the epoch-6
`validation_results.p`; each atlas basin's package NetCDF (`time_series/
<basin_id>.nc`, read via `basin_netcdf_path`, one file per rendered basin —
24 for the full atlas, ~6-8 for the compact panel only); and, optionally,
`manifests/package_manifest.json` for a package-identity checksum. It never
reads the other ~2,533 non-atlas basins or any static-attribute matrix.
Rendering on Moriah under Slurm (transferring back only the small generated
figures/CSVs/manifest) is the preferred first real-execution path, since it
avoids transferring any package data at all; a reduced local bundle
containing only the epoch-6 pickle plus the ≤24 needed basin NetCDFs is the
fallback if local rendering is required. Neither has been performed by this
addendum, and **the full 2,557-basin package must not be transferred to a
local machine for this purpose.** This choice remains a later operational
decision, not implemented here.

**L.3c — real execution completed (2026-08-02): atlas24 evaluation-only
derivative + rendering PASS, visual review adopted.** The workflow described
in L.3b's Moriah-Slurm path was carried out for real, against the completed
`emb128x64_seedA` run's selected epoch-6 checkpoint (not the epoch-15 stop
point — see the "Checkpoint identity" note in the evidence write-up below).

- **Build/evaluate/integrity** (`scripts/build_stage1_atlas24_eval_run_dir.py`,
  `scripts/run_stage1_atlas24_eval_moriah.sbatch`, job `45729427`, `PASS`):
  built a disposable, evaluation-only NH run directory pointed at exactly the
  24 fixed atlas basins for the validation period, reusing the original
  checkpoint and scaler byte-for-byte (never refit) and reusing the original
  frozen config except for the approved basin-file/run-dir/experiment-name
  fields (config-diff-checked, zero unexpected differences). Confirms:
  original checkpoint, scaler, config, and the original ~400-basin screening
  validation results pickle are sha256-identical before and after; the
  derivative's own results pickle contains exactly the 24 atlas basin IDs;
  an `EVALUATION_ONLY_DO_NOT_TRAIN.txt` marker is written into the derivative
  directory.
- **Rendering** (`scripts/render_stage1_hydrographs.py`, via
  `scripts/render_stage1_hydrographs_moriah.sbatch`, job `45729449`,
  `PASS`): reused the existing rendering tooling (L.3a) unchanged, CPU-only
  (`glacier`, no GPU), against the derivative's results pickle — 24
  individual atlas panels, one deterministic 8-basin compact panel, per-basin
  metrics, 96 event windows (observed-discharge-only selection, unchanged),
  and a checksummed rendering manifest/summary.
- **Commit-readiness review (this pass).** Reviewing the code written to
  support this operation found two concrete gaps in the shared
  evaluation-only-derivative helpers (`src/baseline/nh_seed_evaluation.py`,
  used by both `prepare_external_scaler_eval_run_dir` and
  `prepare_development_population_eval_run_dir`), now fixed and covered by
  new focused tests:
  1. Neither helper checked that its `out_run_dir` differs from (or is
     nested with) the protected `development_run_dir` before its
     `force`-gated directory removal. Fixed by an unconditional
     `_raise_if_out_run_dir_collides_with_development_run_dir` check that
     cannot be bypassed by `force=True`, checked before any deletion.
  2. `scripts/run_stage1_nh.py`'s ordinary `train`/`continue` commands had no
     check against pointing the trainer at an evaluation-only derivative
     directory (whose checkpoint must never be refit). Fixed by a new
     `raise_if_evaluation_only_bundle` guard (mirroring the existing
     `raise_if_holdout_bundle` spatial-holdout guard), wired into both
     commands before NeuralHydrology's own `start_run`/`continue_run` are
     invoked.
  All other reviewed properties (exactly-24-basin enforcement,
  development-only/spatial-holdout basin-membership validation, explicit
  required `--epoch`, Slurm-only Python execution, no credentials in either
  launcher, generated outputs written outside any tracked directory,
  explicit rerun/force semantics, aggregate PASS/FAIL status that cannot
  mask a sub-check failure, and full command-line reproducibility from the
  evidence bundle) were already satisfied by the existing design and did not
  require new tests.
- **Adopted visual interpretation.** Recorded in full in
  `docs/decision_log.md`'s 2026-08-02 entry for this operation (same
  wording); summarized here: genuine hydrologic signal with many predicted
  events in approximately the correct temporal neighborhood, no obvious
  universal six-hour displacement or global raw-space conversion failure
  (a visual diagnostic observation, not a formal proof) — but performance
  remains weak and hydrologically inconsistent, with commonly attenuated
  large peaks, some false/exaggerated predicted peaks, often poor
  recession/baseflow behavior, and strongly basin-varying bias. Supports
  continuing structural optimization; does not establish model adequacy,
  architecture superiority, full development-validation performance, or
  final Stage 1 readiness. The atlas's aggregate median NSE (≈0.14) is not a
  representative substitute for the provisional ~400-basin screening metric.
- **Full technical write-up and checksums** (untracked, per
  `docs/repo_policy.md`): `reports/stage1_validation_optimization_foundation_v001/part_l_atlas24_eval_emb128x64_seedA_v001/part_l_atlas24_eval_emb128x64_seedA_v001.md`.
- **Not changed by this pass:** no model-selection decision; no full
  development/population evaluation; no sealed-set access; no `raw_seedA`
  launch; no W&B activity.

**L.3d — hydrograph-demonstration standard, revised (2026-08-05, adopted
design; not yet implemented).** Written as part of closing the embedding-
shape neighborhood screening (Part L.10) and recorded here because it
extends this section's rendering design, not because any new rendering ran.
For 50k-promoted candidates, the standard compact demonstration package
should include:

1. A fixed eight-basin compact hydrograph panel (reusing `select_compact_basins`,
   L.3a).
2. Basin area in every panel title — km², drawn from the authoritative
   basin-area field (`derive_basin_area_km2_from_netcdf`, already used
   internally by `src/baseline/hydrograph_rendering.py` but not yet surfaced
   in panel titles).
3. Basin-average hourly precipitation (MRMS QPE, mm h⁻¹) as blue bars
   descending from the top of a secondary right-hand y-axis (zero at the
   top, precipitation increasing downward) — not present in the current
   renderer.
4. Explicit time alignment: precipitation plotted at its physical valid
   time; observations plotted at physical discharge time; lead-6
   predictions plotted at the target valid time they predict; no artificial
   six-hour shift of rainfall relative to discharge. This convention must be
   stated explicitly in the rendering manifest/metadata and in the
   accompanying interpretation text.
5. Matched comparison scales across candidates being compared: identical
   time windows, identical discharge limits for the same basin/window,
   identical precipitation limits where practical, identical plot
   conventions.
6. A compact-panel metrics table (reusing the existing per-basin raw-space
   metric computation).
7. A short hydrograph interpretation discussing peak magnitude, peak
   timing, false peaks, recession, baseflow, basin-specific bias, and
   rainfall-runoff timing where visible.
8. The full 24-basin atlas is **not** required for every 50k candidate —
   reserved for ambiguous cases, integrated candidates, or authoritative
   finalists.

**Demonstration-output cadence (adopted).** 25k coarse screening: strategic
metrics and learning curves only, no routine hydrograph package. 50k serious
triage: the compact eight-basin panel (items 1-7 above) plus a strategic
review packet. Integrated or uncapped finalists: the compact panel plus the
full 24-basin atlas plus a standardized 6-8 figure package plus a
comprehensive scientific summary.

**Status.** This is a design update to the existing L.3/L.3a rendering
standard, not a new visualization framework — items 2-5 above are gaps in
the current `render_basin_panel`/`render_compact_panel` implementation
(confirmed by source inspection: basin area is derived but not titled, and
no precipitation axis exists today) to be closed when a 50k-promoted
candidate first needs this package, not by this documentation-only entry.
No rendering code changed by this entry; no new panel generated.

**L.3d implementation note (2026-08-06).** Compact-renderer support for
items 1-8 above is now implemented in `src/baseline/hydrograph_rendering.py`
(`BasinSeries`, `load_basin_series`, `load_mrms_series`,
`render_stage1_compact_comparison_package`), superseding the "not yet
implemented" status above for the rendering code itself. A read-only
timestamp-semantics audit confirmed the following and added cross-series and
event-window regression tests to `tests/test_hydrograph_rendering.py`:

- the NeuralHydrology result-pickle `date` coordinate is issue time (the
  last input timestep), not physical discharge time;
- observed and predicted lead-6 discharge (`_obs`/`_sim`) are plotted at
  target-valid time, `issue_time + lead_hours` (6 h here) — the time the
  values physically represent — and both series share one date axis, so
  they are always shifted identically;
- MRMS is plotted at its own unshifted physical valid time from the
  package NetCDF, never re-indexed or shifted by `lead_hours`;
- raw-space scientific metrics (NSE/KGE/RMSE/MAE/PBIAS) were unaffected by
  the prior visualization timestamp issue — `convert_period_to_raw_space`/
  `raw_space_metrics` take plain positional arrays with no date argument,
  so the plotting timestamp axis never entered metric computation.

**Caveat — pre-existing atlas artifact.** The real atlas evidence under
`reports/stage1_validation_optimization_foundation_v001/part_l_atlas24_eval_emb128x64_seedA_v001/`
was generated on 2026-08-02 (`plotting_implementation_git_commit`
`3ff9eaff90e277953026991fb793bafea603563d`) using the pre-correction
renderer: its x-axes and `event_window_table.csv`
`peak_time`/`window_start`/`window_end` columns are issue time, not
target-valid time, and should not be interpreted as target-valid until that
artifact is regenerated. Regeneration is deferred until this atlas is next
scientifically needed; it was not performed as part of this note.

**L.4 — W&B adoption sequencing (Stage (1) and (2) complete; (3)/(4) not yet
started).** Order: (1) ordinary tracking qualification, (2) an offline-mode
real-path test preferred before relying on online tracking, (3) controlled
live tracking for one structural candidate, (4) sweeps only after Stage B's
search space/objective/fidelity/promotion rules are frozen. Repository code
remains authoritative for basin membership, sealed-set protection, metric
computation, early stopping, checkpoint provenance, and package identity;
W&B is telemetry/comparison infrastructure, never the scientific source of
truth (Part F).

Stage (1) — **wrapper contract, fake backend.** Implemented and fixed in
`src/baseline/wandb_tracking.py`/`src/baseline/pilot_tracking.py`: (a)
failure isolation after `wandb.init` (any backend call that raises is
caught, warned once per operation, recorded as
`degraded`/`degraded_operations` on the `TrackingRun`, never propagated
into training/screening/early-stopping/checkpoint-selection code); (b) a
stable W&B run identity across bounded Slurm continuations,
`derive_pilot_wandb_run_id`/`resolve_pilot_wandb_run_id` — a deterministic
id from `(pilot_policy_name, run_id, tracking_generation)` passed as
`wandb.init(id=..., resume="allow")`, cross-checked against a small
persisted record in the NH run directory. `tracking_generation` (default
`"g1"`) was added during the review that produced this addendum to close a
real collision gap: NH run directories are timestamped/prefix-matched, not
one fixed path per `run_id`, so a deliberate operator restart-from-scratch
under the same `run_id` (after abandoning its prior NH run directory) was
indistinguishable at call time from a genuine first attempt — both present
`existing_nh_run_dir = None`. An explicit, manually-bumped generation
string is the smallest durable fix; it stays at its default for every
ordinary continuation. The tracking metadata contract was also extended
(`max_updates_per_epoch: None`, `baseline_policy_sha256`, `splits_dir`,
`tracking_generation`, `wandb_run_id` in `build_pilot_run_identity`;
`mode`, `wandb_run_id`, `degraded`, `degraded_operations` in the evidence
bundle's `"wandb"` block). This stage was exercised entirely through
pytest against an in-process fake `wandb` module (`sys.modules`
monkeypatching) — never the real package, never network access, nothing
run outside `tmp_path` — across `tests/test_wandb_tracking.py`,
`tests/test_pilot_tracking.py`, `tests/test_pilot_orchestration.py`, and
`tests/test_wandb_offline_qualification.py` (15 numbered scenarios); the
combined suite passes (140 tests across the four files as of this
addendum). **This proves the wrapper's contract only — it does not by
itself prove real W&B offline I/O, serialization, or resume semantics**;
that distinction is the entire reason Stage (2) below is a separate,
later exercise, not a restatement of Stage (1).

Stage (2) — **real package, offline mode: qualified.**
`scripts/wandb_real_offline_qualification_smoke.py` drives this repo's
actual tracking code (never a reimplementation) against the real,
locally-installed **wandb 0.28.1** package, `mode="offline"`, no API key,
no network call, as two genuinely separate OS processes reusing one stable
run id (standing in for two bounded Slurm jobs continuing one candidate).
Confirmed: installed version; `wandb.init(mode="offline", id=<stable_id>,
resume="allow")`; config/hyperparameter serialization (embedded in
wandb-core's binary `run-<id>.wandb` transaction log — this wandb version
does **not** additionally emit a separate `files/config.yaml` in offline
mode, unlike older wandb releases' documented layout); scientific/resource
metric logging; a compact checkpoint-reference artifact record (path +
checksum + size only, never the checkpoint's bytes); a clean finish;
degradation handling against a *real* backend exception
(`wandb.errors.UsageError` raised by logging to an already-finished real
run — caught by `_guard_backend_call`, recorded as `degraded`, never
propagated); and no network attempt anywhere. **One assumption this smoke
run corrected**: the wrapper/Stage-(1) description above (and the prior
draft of this addendum) assumed same-id + `resume="allow"` makes a second
invocation append to the first's local run directory. It does not, in
offline mode — wandb prints `` WARNING `resume` will be ignored since W&B
syncing is set to `offline`. Starting a new run with run id <id>. `` and
each invocation gets its own fresh, timestamped local `offline-run-
<timestamp>-<id>/` directory. Reconciling same-id invocations into one
logical run is a **server-side, sync-time** operation (`wandb sync`,
matched by run id + project), never a local merge; no such sync has been
performed against a real server by this project. `resume="allow"` behaves
as originally described only in `online` mode (unqualified, see below).
The implementation itself required no change for this — it already only
passes `id=`/`resume=` through to `wandb.init` and never assumed a merged
local directory — only the documentation's description of the resulting
behavior was corrected (here and in `docs/stage1_wandb_user_guide.md`
§12). A secondary, non-blocking observation: an artificially long run id
in an earlier trial run of this smoke script silently truncated
wandb-core's binary transaction log path past Windows' `MAX_PATH`, with no
error or warning surfaced anywhere; real production ids
(`flashnh-{policy_name}-{run_id}-{generation}`, e.g. `flashnh-
stage1_lead06_pilot_v001-raw_seedA-g1`, ~46 chars) are well clear of that
threshold, so this is recorded as a caveat, not a defect requiring a code
change. The qualification record (commands, version, directory inventory,
findings) is at `reports/wandb_real_offline_qualification_v001/
qualification_record.json` (untracked, not part of this patch). Scope
limits: single machine, Windows, one short local run per invocation, no
GPU/NeuralHydrology training, no real multi-node Slurm continuation (two
subprocesses on one machine stood in for it).

**Online tracking (stage 3) remains not yet qualified** — `mode: online`
is implemented and policy-selectable but has never been exercised against
a live network connection; treat it as unqualified until it has been.
**Sweeps (stage 4) remain deferred**, unchanged, until Stage B's search
space/objective/fidelity/promotion rules are frozen. The wrapper's shipped
default remains `enabled: false` / `mode: disabled`
(`config/stage1_wandb_tracking_policy_v001.yaml`, unchanged) — none of the
above turned tracking on for any real candidate. `raw_seedA` remains the
next scientific candidate to launch, tracking-optional as before, and W&B
is not yet approved for operational use on it (stage 3 is unqualified and
a live raw_seedA run would be W&B's first real production exercise). The
previously-required project-specific W&B learning guide has now been
authored: `docs/stage1_wandb_user_guide.md`.

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

**L.7 — Capped-update mechanism implemented (2026-08-03); calibration and
adoption remain open.** The optional `max_updates_per_epoch` mechanism
described in L.5 is now implemented end-to-end in Flash-NH (config
generation, pilot/candidate policy, run identity, continuation/checkpoint
safeguards, evidence recording) and locally tested. This entry records only
that the mechanism exists and is safe to use later; it changes nothing
stated in L.5/L.6 about calibration — **no numerical cap has been adopted,
no capped run has been launched on Moriah, no speedup has been measured, and
the true uncapped optimizer-updates-per-epoch count has still not been
established against the real environment.**

Verified NeuralHydrology 1.13 semantics (read directly from the vendored
source, not inferred): `Config.max_updates_per_epoch` (`neuralhydrology/
utils/config.py`) is `Optional[int]`, read once per trainer construction
(`BaseTrainer.__init__`, `neuralhydrology/training/basetrainer.py`) and
re-applied fresh at the start of every `_train_epoch` call — it truncates
that epoch's DataLoader iteration to a deterministic index-based prefix
(`enumerate(pbar)`, `break` once the counter reaches the cap), not a
randomized subset; the counter resets every epoch; the scheduler still
steps once per epoch and checkpointing (`model_epochNNN.pt` and
`optimizer_state_epochNNN.pt`, the latter carrying PyTorch's own persisted
Adam/AdamW `state[p]['step']` counter) still occurs unconditionally once per
epoch regardless of the cap — this is also why no NH core-code modification
was needed to obtain real per-epoch actual-update evidence.

The Flash-NH contract: `max_updates_per_epoch: int | None`, `None` (the
default, and the only value any existing candidate — including `raw_seedA`
and `emb128x64_seedA` — has ever used) means uncapped/unchanged behavior;
any other value must be a positive integer (`0`, negative integers, bools,
floats, and strings are all rejected before config generation). The cap is
frozen for a candidate's entire trajectory: continuation/resume against an
already-started NH run directory is rejected before any training call if
the freshly-resolved cap disagrees with the cap already recorded for that
run directory (covers uncapped→capped, capped→uncapped, and capped-N→
capped-M, in both directions). Capped and uncapped runs are always distinct
identities; a capped checkpoint can never be adopted as the continuation
source for an uncapped trajectory or vice versa; a promoted (adopted-cap)
finalist config is expected to start a new full-fidelity trajectory from
its original seed, never continue from a capped checkpoint (still the
provisional recommendation from L.6, unchanged here). Preparation-only mode
records the declared cap in the run identity without starting training.
Evidence bundles record the configured cap and, where measured, the actual
per-epoch optimizer-update count (`actual_optimizer_updates_by_epoch`, read
from the real `optimizer_state_epochNNN.pt` step counter) as two distinct,
never-conflated fields.

Test coverage added across `tests/test_pilot_lead06_config.py`,
`tests/test_pilot_orchestration.py`, `tests/test_pilot_tracking.py`, and
`tests/test_pilot_evidence_bundle.py` (config validation, identity
conflict/continuation safeguards including the specific null↔int and
int↔int mismatch cases, `MAX_TARGET_EPOCH`/early-stopping/W&B-disabled
independence, offline-W&B cap recording, actual-optimizer-update-evidence
extraction and rejection of malformed/disagreeing optimizer state). A full
run of each affected test file showed no regressions; `raw_seedA` and
`emb128x64_seedA` behavior is unaffected (their `max_updates_per_epoch`
remains `null`, exactly as before this change).

**L.8 — Smallest Moriah calibration plan (prepared 2026-08-03, not
executed).** This plan exists so a future session can calibrate real caps
without redesigning the approach; nothing in it has been run, and it
authorizes no Slurm submission by itself.

*Step 1 — measure, don't guess (1 job).* Launch one short calibration-only
run, fixed architecture (`raw_seedA`'s static pathway) and fixed seed
(reuse `967139` — read-only reuse of the seed value, never of `raw_seedA`'s
checkpoints or run identity), `max_updates_per_epoch: null` (uncapped),
for exactly 1 epoch, under a distinct run id (e.g.
`calib_uncapped_probe_v001`, never `raw_seedA`/`emb128x64_seedA`). Purpose:
read the real `optimizer_state_epoch001.pt` step count via the
already-implemented `read_actual_optimizer_updates` to get the true
uncapped optimizer-updates-per-epoch count for the full 2,307-basin
development population on real Moriah hardware — the one number every
later step depends on and that no existing local evidence establishes
authoritatively. If a still-running or already-completed uncapped epoch
from the separate `raw_seedA` continuation already has its
`optimizer_state_epochNNN.pt` on disk by the time this step is reached,
read that instead (read-only) and skip launching a redundant probe job —
per the plan's own point (2), avoid a redundant baseline if existing
evidence already suffices.

*Step 2 — two caps only, informed by step 1 (2 jobs).* Using the measured
count from step 1, derive one **medium** integer cap and one **low**
integer cap (both explicit positive integers in the generated config, never
a fraction — the fractions in L.5 are non-binding starting intuition for
picking these two integers, not the values themselves). Launch two short
calibration-only runs (`calib_medium_v001`, `calib_low_v001`), same fixed
architecture/seed as step 1, each for enough epochs (expected 2-4) to
observe: cap enforcement (actual updates == configured cap, every epoch),
wall-clock time per epoch vs. step 1's uncapped baseline, training-loss
trajectory, checkpoint/optimizer-state creation every epoch, one
continuation (Slurm requeue or manual resume) to confirm the frozen-cap
identity safeguard accepts a matching resume and would reject a changed
one, and — only if the above all look sound — one screening point to see
whether the screening metric is even directionally informative at that
fidelity. No promotion decision is made from this alone.

*Explicitly out of scope for this plan*: any change to `raw_seedA`'s or
`emb128x64_seedA`'s identity, checkpoints, or trajectory; more than two
caps; more than one architecture/seed; a promotion threshold; a broader
sweep or campaign of any kind. The plan stops after step 2's evidence is
in hand and is reviewed before anything further is authorized.

**Estimate.** 3 Slurm jobs total (1 uncapped probe + 2 capped, or 2 total
if an existing `raw_seedA` epoch's optimizer state can be read instead of
launching the probe). Approximate GPU time: well under the ~2h40m
`raw_seedA` took for 6 full epochs, since every calibration job here is
1-4 epochs on the same architecture/data population — order of a few GPU
hours total, not a campaign. Expected calendar duration: well under a day
of wall-clock elapsed time (each job is short and Moriah queueing is the
main variable, not compute). Evidence required before adopting *any*
numerical cap for real screening use: step 1's measured uncapped
updates/epoch count, step 2's two caps' enforcement/wall-time/loss/
continuation results, and an explicit review confirming the screening
metric at the chosen fidelity is directionally trustworthy relative to
this pilot's existing full-fidelity results — none of which exists yet.

**Superseded:** the calibration this section describes has now been executed; see L.9 below.

**L.9 — Calibration executed (2026-08-04): mechanism qualified, L.5/L.6/L.8's open questions substantially answered, provisional fidelity workflow adopted.** Three real Moriah calibration exercises ran this session, under unchanged commit `ac98f6b3ad9b1687a26a7509f98a02df3c06381b`, closing L.8's step-1/step-2 plan and extending it with two further bounded, separately-authorized comparisons. Full decision text: `docs/decision_log.md`'s 2026-08-04 entry. Local evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/{cap_learning_diagnostics_v001,emb50k_architecture_diagnostic_v001,cap_parallel_batch_v001}/`.

*Step 1/2 result (closes L.8's plan).* The true uncapped optimizer-updates-per-epoch count for the full 2,307-basin development population, real Moriah NeuralHydrology 1.13, Seed A raw pathway (`raw_seedA`'s own trajectory): **237,298**, read from `optimizer_state_epochNNN.pt`'s persisted step counter (`cap_learning_diagnostics_v001`, job 45738246). Two calibration caps were derived and run for 6 epochs each: `raw_seedA_cap_medium_cal` (100,000) and `raw_seedA_cap_low_cal` (50,000), both enforcing their cap exactly every epoch (zero drift). At the three cumulative-update points where the two caps coincide (100k/200k/300k), `low_cal`'s median NSE was equal to or higher than `medium_cal`'s by 0.002-0.010 — a small, non-monotonic difference, not evidence that either cap is superior on its own. A continuation (Slurm requeue/resume) confirmed the frozen-cap identity safeguard.

*Every-epoch diagnostic finding (retrospective, diagnostic-only).* Retrospectively evaluating epochs 1/2/4/5 (never fed into early-stopping/checkpoint-selection state) showed per-epoch median NSE is not monotonic in any of the three trajectories (uncapped, 100k, 50k) — e.g. `low_cal` ranges 0.2176-0.2473 across epochs 1-6 with a local dip at epoch 3 before recovering — confirming that an epoch-3/epoch-6-only official cadence can hide intermediate peaks, dips, and recoveries that a coarser view would miss. `frac(NSE>0)` (0.75-0.83) and `frac(NSE<-1)` (0.0425-0.06) stayed tightly bounded throughout, in all three trajectories — no collapse or runaway gain at any fidelity tested.

*Matched raw-vs-embedded 50k comparison (`emb50k_architecture_diagnostic_v001`).* `raw_seedA_cap_low_cal` vs `emb128x64_seedA_cap_low_cal` (Seed A, identical settings except static pathway, both capped 50,000/epoch). Aggregate per-run medians favor the embedded pathway at 5 of 6 epochs (epoch 4 a near-tie, raw nominally 0.0011 higher); true per-basin paired differences favor embedded at every one of the 6 epochs (median paired diff -0.0025 to -0.0217 favoring embedded; win share narrows from roughly 56/39 to 44.25/42.25 by epoch 6). Basin-level IQR (~0.35-0.37 in both runs) is roughly an order of magnitude wider than the aggregate architecture gap. **This does not resolve the raw-vs-embedded question** — see the adopted framing change below.

*Four-candidate parallel batch (`cap_parallel_batch_v001`, jobs 45745666-45745669).* `raw_seedA_cap25k_cal`, `emb128x64_seedA_cap25k_cal` (both 25,000/epoch, Seed A) and `raw_seedB_cap50k_cal`, `emb128x64_seedB_cap50k_cal` (both 50,000/epoch, Seed B 1729) — all four trained cleanly through epoch 6 (`max_target_epoch`), exact cap enforcement verified every epoch for all four (job 45754600, `all_epochs_match_cap_exactly: true` for every candidate), fully isolated run identities, no sealed-population access, offline W&B throughout, full retrospective epoch-1-6 trajectories produced (jobs 45754619-45754622) and consolidated into per-run/per-epoch metrics (job 45754682) and true per-basin paired comparisons across three named comparisons A/B/C (job 45754688, `comparison_evidence_v2.json`). Representative results: comparison A (Seed-A cap sensitivity, 25k vs the protected 50k reference) shows small, sign-inconsistent epoch-6 median paired diffs (+0.0108 raw favoring 50k, -0.0092 embedded favoring 25k, both well inside a ~0.10-wide paired IQR); comparison B (Seed-B architecture at 50k) shows embedded favored at all 6 epochs by a modest, growing paired margin (epoch 6: median diff +0.0206, 62.25% embedded-favoring); comparison C (cross-seed at 50k) shows small, sign-inconsistent Seed-A-vs-Seed-B paired differences in both pathways. No candidate showed divergence, collapse, or an aggregate/paired sign flip large enough to overturn the "capped runs are for coarse screening only" conclusion below.

**Adopted findings (supersede L.5/L.6/L.8's "unresolved"/"not implemented" status on these specific points; full wording in `docs/decision_log.md`'s 2026-08-04 entry).**
1. Mechanism is operationally qualified across uncapped/100k/50k/25k, raw/embedded pathways, Seed A/B, sequential and parallel execution.
2. Capped-run aggregate metrics stay coherent (no collapse/divergence) but paired basin-level spread routinely exceeds the aggregate/architecture/seed effect sizes under study — **capped runs support coarse rejection and second-stage triage only, not fine ranking; capped performance is not evidence that fewer updates are scientifically superior; capped checkpoints must never be promoted to full-fidelity trajectories** (L.6's restart-from-seed recommendation is retained, now with direct supporting evidence rather than being only a provisional default).
3. Wall-clock time does not scale linearly with the update cap (halving the cap reduced measured elapsed time by only ~12-18% in the two matched pairs above) — fixed validation/startup/checkpointing overhead dominates; **do not assume ideal linear wall-time scaling.**
4. **Provisional three-tier fidelity workflow adopted, not a final scientific method:** 25k = first-pass coarse rejection; 50k = second-stage triage for plausible candidates; uncapped = finalists only, each fidelity a distinct run identity, no cross-fidelity checkpoint continuation, promoted candidates restart from their original seed.
5. **Retrospective per-epoch evaluation stays diagnostic-only and selective going forward:** official screening cadence and early-stopping semantics are unchanged; every-epoch retrospective evaluation is used for close/promising/puzzling/new-family candidates during this calibration phase, not applied automatically to every future 25k candidate in a routine broad campaign.
6. **Static embedding remains unresolved**, reframed as a bounded hyperparameter family (raw; `[64]`; `[128]`; `[128,64]`) rather than a settled raw-vs-embedded question. Next approved, not-yet-started batch: one-layer `[64]` and one-layer `[128]` embeddings, Seed A, 25k cap, against the existing Seed-A raw/`[128,64]` references, with embedding activation/dropout, output dropout, `hidden_size`, and learning rate held fixed. **These runs have not started.**

**Still open (not answered by this calibration; L.6's remaining unresolved items).** Fair cross-fidelity budget comparison beyond the coarse observations above; a quantitative promotion threshold; learning-rate scheduling implications for capped trials; NH's own resumed-logger cumulative-update count remains unverified as authoritative for capped-epoch resumption (this session always read the real per-epoch `optimizer_state_epochNNN.pt` counter directly instead, so this gap is sidestepped for evidence purposes but not closed as a general question).

**Strategic review packet standard (new, documentation-level, for future structural-comparison tasks only).** See `docs/decision_log.md`'s 2026-08-04 entry for the full 7-component definition. Not applied retroactively to the evidence summarized above.

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no numerical cap adopted for production use, no generated evidence committed, no sealed temporal-test/spatial-holdout data accessed.

**L.10 — Embedding-shape neighborhood screening closed (2026-08-05): `[128,64]`/`[128,32]` structural survivors, next 50k comparison designed, sequence length reframed, revised hyperparameter order, learning-curve standard revised.** Real Moriah screening batch this session, under unchanged commit `5aba586dc4856ecb05945b41d3ff29a34f096cb7`, closing L.9 point 6's "next approved, not-yet-started batch." Full decision text: `docs/decision_log.md`'s 2026-08-05 entry. Local evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/embedding_shape_neighborhood_seedA_25k_v001/`.

*Batch.* `emb64x32_seedA_cap25k_cal` (`[64,32]`), `emb128x32_seedA_cap25k_cal` (`[128,32]`), `emb256x64_seedA_cap25k_cal` (`[256,64]`) — Seed A, 25k cap, all other settings matching L.9's `emb128x64_seedA_cap25k_cal` reference (which was not touched, re-run, or continued). All three completed cleanly through epoch 6, exact cap enforcement verified every epoch for all three (job 45756766, cumulative 25,000/50,000/75,000/100,000/125,000/150,000, zero drift), full retrospective epoch-1-6 trajectories (jobs 45756721-45756723), and a true per-basin paired comparison against the reference for all 3 candidates x 6 epochs (job 45756761).

*Result (provisional, coarse-screening resolution only).* No candidate shows broad, consistent superiority over `[128,64]`. `[128,32]` has the mildest positive edge (positive median paired diff at 5/6 epochs, win rate never exceeding ≈0.53) — a plausible challenger, not a demonstrated winner. `[64,32]` stays broadly close to `[128,64]` (positive median paired diff at all 6 epochs but no stable/broad advantage). `[256,64]` is the weakest tested shape (negative median paired diff at 4/6 epochs; lowest official median NSE at both official epochs 3 and 6) — provisionally rejected at this tier. The 25k cap remains useful for divergence detection/coarse rejection (it separates `[256,64]`) but is not precise enough for fine ranking among `[64,32]`/`[128,32]`/`[128,64]`, all of whose paired win rates sit close to chance — extending L.9 point 2's "coarse rejection, not fine ranking" finding to this finer shape granularity.

*Structural survivors.* `[128,64]` (incumbent), `[128,32]` (challenger). Not a final architecture decision.

*Next approved structural phase (design only, not launched).* Existing Seed-A `[128,64]` trajectory (`emb128x64_seedA_cap_low_cal`) continued to 50k vs. a new Seed-A `[128,32]` trajectory at 50k. `max_updates_per_epoch=50000`; target up to epoch 12; official screening at epochs 3/6/9/12; existing early-stopping policy authoritative (stopping-eligible from epoch 6, minimum improvement 0.005, patience 3 eligible screening events); every epoch saved; retrospective checkpoint evaluation usable diagnostically; no cross-fidelity checkpoint reuse; the new `[128,32]` candidate starts from the original Seed-A initialization; the existing `[128,64]` candidate may continue only within its own unchanged candidate identity and fidelity. Purpose: close the embedding-structure question at a more informative fidelity and avoid further width/depth exploration unless new evidence later justifies it. **Not started.**

*Sequence length (adopted, binding — corrects L.1 above).* Sequence length is fixed at 24 for the current model family and is not an ordinary Stage B/HPO dimension. Alternative sequence lengths are separate temporal-context model families (different antecedent-memory assumptions, input construction, compute/memory cost, cross-basin response-time interpretation); a later study may compare such families against a mature 24-hour model, but that is a separate, later phase, not part of the current funnel. L.1's Stage B dimension list has been corrected to remove sequence length. **Further revised (2026-08-13, see Part L.19):** the Embedding-Dropout-A closure schedules a dedicated Sequence-Length-A characterization (`seq_length={12,24,48,72}`) as roadmap item 2, reframing sequence length as a bounded, structural/calibratable model parameter rather than a dimension permanently excluded from calibration. This passage is preserved as historical, not rewritten — see L.19 for the current framing.

*Revised hyperparameter order, within the fixed `seq_length=24` model family.* (1) Close embedding structure at 50k (`[128,32]` vs `[128,64]`); (2) learning rate (bounded contrast around 0.001, values TBD); (3) LSTM hidden size (bounded capacity contrast, candidates TBD); (4) embedding dropout; (5) output dropout; (6) small integration/interaction checks; (7) Seed-B confirmation for top integrated candidates only; (8) uncapped authoritative finalists; (9) a separate later temporal-context model-family study for sequence length. Rationale: hybrid of expected scientific/optimization impact, dependency/interaction structure, experimental clarity, operational cost.

*Learning-curve standard, revised (adopted for future serious-triage/finalist packets).* Training diagnostics: mean training loss vs. epoch and vs. cumulative optimizer updates. Validation/scientific diagnostics: median raw-space per-basin NSE vs. epoch, p25-p75 distributional band, `frac(NSE>0)`, explicit official-vs-retrospective markers. Transformed-space validation loss, if cheaply available, is a training diagnostic only, never the official model-selection metric. **Preserved:** NH losses may be transformed-space diagnostics; official Flash-NH benchmark metrics are always computed after full inverse conversion to raw m³/s; raw-space screening metrics remain authoritative for candidate selection; raw-space median NSE is never labeled "validation loss."

*Hydrograph-demonstration standard, revised.* See new L.3d above (basin area in titles, MRMS precipitation bars on an inverted secondary axis, explicit valid-time alignment, matched scales, compact metrics/interpretation, demonstration cadence by fidelity tier).

*W&B.* All four screening runs stayed offline throughout (`tracking_generation=g1`, no sync). No new W&B capability qualified by this batch; the previously-qualified single-segment sync (`docs/stage1_wandb_user_guide.md` §17) covers two unrelated runs and is unaffected.

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout data accessed, no run described as started.

**L.11 — 50k Seed-A embedding-shape comparison closed (2026-08-06): `[128,32]` adopted as working default, further embedding-shape exploration paused, bounded learning-rate calibration approved next.** Real Moriah closure batch this session, under unchanged commit `a4c5456331d97af61c71167a39bf5a6a0644d1ab`, closing L.10's "next approved structural phase." Full decision text: `docs/decision_log.md`'s 2026-08-06 entry. Local evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/cap50k_closure_comparison_audit_2026-08-06/`; Moriah archive `cap50k_closure_comparison_audit_2026-08-06.tar.gz` (SHA256 `9ff1960bf7537da78ea62e5046805c28c0436bd1804395086e12c13c1a347207`, independently re-verified locally against `MANIFEST.csv`, 38/38 files).

*Batch.* Existing Seed-A `[128,64]` trajectory (`emb128x64_seedA_cap_low_cal`, job 45762223, continued from epoch 6) vs. new Seed-A `[128,32]` trajectory (`emb128x32_seedA_cap_low_cal`, job 45762224, fresh Seed-A initialization), both `max_updates_per_epoch=50000`, both reaching the fixed epoch-12 closure bound cleanly (exit `0:0`, `PAUSED_AT_MAX_TARGET_EPOCH`), otherwise identical configuration (Seed A 967139, `qobs_mm_per_h_lead06`, lead 6h, `seq_length=24`, `hidden_size=128`, tanh embedding activation, embedding dropout 0.1, output dropout 0.25, Adam `lr=0.001`, NSE training loss, fixed 2,307-basin training population, fixed 400-basin screening set), differing only in `statics_embedding.hiddens`.

*Official screening-epoch median NSE (400-basin population).* Epoch 3: incumbent 0.2418, challenger 0.2480. Epoch 6: incumbent 0.2547, challenger 0.2541. Epoch 9: incumbent 0.2367, challenger 0.2569. Epoch 12: incumbent 0.2427, challenger 0.2464.

*True per-basin paired result (challenger minus incumbent, 400/400 matched basins, tie tolerance ±0.01).* Epoch 3: median +0.0136, Q25 -0.0293, Q75 +0.0650, challenger better 53.5%. Epoch 6: median +0.0145, Q25 -0.0294, Q75 +0.0640, challenger better 53.25%. Epoch 9: median +0.0160, Q25 -0.0330, Q75 +0.0636, challenger better 54.25%. Epoch 12: median +0.0072, Q25 -0.0447, Q75 +0.0709, challenger better 48.5%. Computed by a new minimal untracked helper (`tmp/paired_basin_csv_join.py`, pure CSV join+describe, no pickle access, no metric recomputation) after the three pre-existing `tmp/operations/` paired-comparison helpers were inspected and found unsuitable (all three recompute NSE from `validation_results.p` pickles rather than joining the already-certified per-basin CSVs). Run under Slurm, job 45763464, exit `0:0`.

*Interpretation (adopted, cautious).* `[128,32]` is at least comparable to `[128,64]`, with a small, directionally consistent paired advantage at epochs 3/6/9 that weakens by epoch 12 (median ΔNSE narrows to +0.0072, win-rate margin narrows from ~53-54%/33-35% to 48.5%/41.5%). Not decisively superior. Effect is modest relative to cross-basin heterogeneity (paired IQR ≈0.10 at every epoch). Single seed only. Training-loss diagnostics point the same direction but remain non-authoritative. No inference is drawn about the importance of static attributes generally — this compares two embedding widths only.

*Decision (adopted).* `[128,32]` becomes the working default embedding shape (at least as competitive, more economical). Further embedding-shape (width/depth) exploration is paused, not permanently closed. No model-family switch proposed.

*Early-stopping / closure interpretation.* Both trajectories' best official screening epoch is 6 (incumbent 0.25474, challenger 0.25414); neither early-stopped before epoch 12 (`stopped=false`). Epoch-12 termination was the fixed closure bound only, not early stopping. `next_intended_screening_epoch=15` in both bundles was never executed and is not a planned continuation.

*Next approved phase (design only, not launched).* Bounded learning-rate calibration around the current 0.001 baseline, `[128,32]` fixed, candidate values TBD, same 400-basin raw-space validation contract, staged promotion. **Not launched.**

*Operational efficiency (deferred engineering item, not a blocker).* Each nested NH continuation boundary added roughly 20-40 minutes of dataset/lookup-table/dataloader rebuild against a roughly 4-minute steady-state epoch (~25-45% of total wall time across the two boundaries here), quantified from checkpoint mtimes (both bundles' `epoch_timing_table` empty). Recorded as a future optimization target; not fixed here.

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout data accessed, no learning-rate experiment implemented or launched.

**L.12 — LR-A (bounded learning-rate range characterization) design frozen (2026-08-08): five-candidate range, all-epoch evaluation gap identified, `1e-3` reuse from L.10's `emb128x32_seedA_cap25k_cal`, multidimensional HPO roadmap recorded.** Documentation-only design-freeze task, under unchanged commit `9b3b56f7dd68e876c9d02c8a6e5993698b0a9437` (confirmed against local `HEAD` and `origin/master` before this update; clean tracked tree). Full decision text: `docs/decision_log.md`'s 2026-08-08 entry; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s 2026-08-08 section.

*Design (frozen, not launched).* Five learning-rate candidates — `1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2` — all other settings frozen at the `[128,32]`/Seed-A/25k-cap configuration established by L.10-L.11, fixed six-epoch budget for every candidate regardless of trajectory shape, checkpoint every epoch. Purpose: characterize the useful LR region around 0.001 and inform later checkpoint-cadence/objective design for Phase B — not final LR optimization, not a five-candidate tournament.

*All-epoch evaluation audit finding.* `pilot_screening_eval.evaluate_screening_checkpoint()` structurally rejects off-cadence epochs (1, 2, 4, 5); the underlying `pilot_orchestration.ensure_validation_results()` and `nh_seed_evaluation.raw_space_metrics_for_run_period()` are epoch-agnostic and can be called directly for any epoch without touching early-stopping state (`record_screening_event()` is unreachable outside the official chunk-boundary loop). No committed helper does this today — a small, additive diagnostic-evaluation function is the one identified implementation gap; see the decision_log entry for the full plan.

*Reuse.* `emb128x32_seedA_cap25k_cal` (L.10) is adopted as the LR-A `1e-3` reference without retraining — verified field-for-field scientific equivalence, zero relevant code/config drift since `5aba586` (`git log` confirms no commits touched the eight relevant files between `5aba586` and current `HEAD`).

*Naming.* `emb128x32_seedA_lr1em4_cap25k_cal`, `emb128x32_seedA_lr3em4_cap25k_cal`, `emb128x32_seedA_lr3em3_cap25k_cal`, `emb128x32_seedA_lr1em2_cap25k_cal`; campaign `lr_range_seedA_25k_v001`; closure-splice pattern following `run_stage1_cap50k_closure.py`.

*Minimum implementation plan (not implemented by this entry).* Optional `learning_rate` override field on `PilotRunSpec` plus a post-profile-merge override in `nh_config_generation.py` (preferred over duplicating named profiles); a new diagnostic-evaluation helper closing the gap above; a new closure-splice launcher pair (`CLOSURE_MAX_TARGET_EPOCH=6`); promotion/generalization of the ad hoc paired-comparison CSV-join helper to N-candidates-vs-1-reference.

*Future roadmap (recorded, not implemented).* Phase A (this design, 1D range characterization, non-binding, axes LR/hidden size/embedding dropout/output dropout) → Phase B (joint multidimensional HPO, capped-fidelity screening → interaction inspection → higher-fidelity promotion → optional adaptive search → Seed-B confirmation → uncapped finalists), with W&B reserved as a future search/index layer, never a scientific authority.

**L.13 — LR-A implementation and preparation-only validation complete (2026-08-08): all L.12 minimum-implementation-plan items built and tested, real (unmocked) preparation-only config validation for the four new candidates, no launch.** Implementation task following L.12's design freeze. Full decision text: `docs/decision_log.md`'s "LR-A implementation and preparation-only validation complete" 2026-08-08 entry; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s "LR-A implementation complete" 2026-08-08 section.

*Built (all six L.12 minimum-implementation-plan items).* (1) `PilotRunSpec.learning_rate: float | None = None` (additive, default-preserving). (2) `nh_config_generation.build_nh_config_mapping()` applies an optional `learning_rate` override after the named run-profile merge (always wins); `validate_learning_rate_override()` rejects non-numeric/bool/zero/negative/NaN/inf; `write_generated_config()`'s manifest records `learning_rate_override`/`resolved_learning_rate`. (3) `pilot_diagnostic_eval.py`: `evaluate_diagnostic_checkpoint()`/`evaluate_all_diagnostic_checkpoints()` evaluate any epoch 1-6 via the epoch-agnostic primitives identified in L.12's audit, tagging off-cadence epochs `evaluation_role="retrospective_diagnostic"` (`authoritative=False`, `stopping_eligible=False`) and on-cadence epochs (3, 6) `evaluation_role="official"` via delegation to `pilot_screening_eval.evaluate_screening_checkpoint()`; test-confirmed to never import `record_screening_event`. (4) `scripts/run_stage1_lr_range_seedA_closure.py` + `..._moriah.sbatch`, `LR_A_MAX_TARGET_EPOCH=6` fixed, four new run_ids only, `REFERENCE_RUN_ID` reachable exclusively via `--status-only`. (5) `checkpoint_comparison.py`: `build_n_vs_one_comparison()` (reshapes L.12's evaluation payloads into an N-vs-1-reference table, no new metric math), `derive_trajectory_summary()` (late-window epochs 4-6 direction, best checkpoint — deliberately no `score`/`rank`/`winner`/`composite_score`/`is_best` key, preserving "no single decision statistic"), `cadence_sensitivity_view()`. (6) Tests: 32 (`test_checkpoint_comparison.py`) + 11 (`test_pilot_diagnostic_eval.py`) + 46 (`test_run_stage1_lr_range_seedA_closure_cli.py`) + 44 (`test_lr_range_seedA_closure_sbatch_launcher.py`) + 15 (`test_lr_range_seedA_closure_preparation.py`, see below) — all passing, alongside a clean full-suite regression.

*Preparation-only validation (real code, no mocking).* `prepare_pilot_run_only()` called for real against a synthetic package covering the actual full 2,557-basin union, for each of the four new candidates: confirmed architecture/target/lead/seq_length/embedding-shape/dropout/seed/cap invariants match L.12's frozen contract; confirmed each candidate's `learning_rate` and its explicit manifest provenance; confirmed pairwise config diffs are limited to `learning_rate` plus identity/path metadata (`experiment_name`, basin-list file paths, `run_dir`), with `data_dir` and basin-list file *contents* identical across all four and every identity field pairwise-unique. One locally-synthesized fifth `PilotRunSpec` (distinct run_id, never `REFERENCE_RUN_ID`, `learning_rate=1e-3`) validated the same code path in isolation — this is explicitly not a reproduction of the real historical `1e-3` reference, which stays external/read-only (L.12's reuse-audit caveat about that reference's own generation manifest remains open and non-blocking).

*Not done.* No LR candidate launched, no Slurm job submitted, no real NH training/evaluation call, no W&B Sweep, no scientific-design change, no full-population validation, no hydrograph package, nothing committed automatically.

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no LR-A candidate launched, no W&B Sweep/HPO framework implemented.

**L.14 — LR-A (bounded learning-rate range characterization) closed (2026-08-09): range evidence recorded, `3e-4` adopted as provisional Phase-A working anchor, cadence and W&B findings documented.** Documentation-only closure task recording the completed Moriah execution of L.12's design under L.13's implementation, both already merged (design freeze `f300cb9`, implementation `bc8f253bed9231fc4a98233ffb2b92b16af8f743`). No training, evaluation, Slurm job, W&B sync, package generation, or new analysis was run by this task; no source code was modified. Full decision text: `docs/decision_log.md`'s 2026-08-09 entry; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s final 2026-08-09 closing section.

*Runs and evaluation.* All four new candidates (`emb128x32_seedA_lr1em4_cap25k_cal`, `emb128x32_seedA_lr3em4_cap25k_cal`, `emb128x32_seedA_lr3em3_cap25k_cal`, `emb128x32_seedA_lr1em2_cap25k_cal`) completed six-epoch/25k-cap training on Moriah (150,000 cumulative optimizer updates each, verified); the `1e-3` candidate is the L.12-adopted reused reference (`emb128x32_seedA_cap25k_cal`), never retrained. All five candidates evaluated at all six epochs via L.13's diagnostic-evaluation helper — 30/30 cells complete, epochs 3/6 `official`, epochs 1/2/4/5 `retrospective_diagnostic`.

*Result (range characterization, not final selection).* Epoch-6 median raw-space NSE ordering: `3e-4 (0.268) > 1e-4 (0.259) > 1e-3 (0.253) > 3e-3 (0.178) > 1e-2 (0.021)`. `3e-4` has a positive median paired NSE difference vs. the `1e-3` reference at all six epochs and is better on ~55-68% of the exactly-matched 400 screening basins depending on epoch (all 24 paired-comparison rows use exact 400/400 basin matching); `3e-3` and `1e-2` are worse than the reference at every epoch (`1e-3` better on ~76-92% of basins). Training-loss trajectories corroborate the optimization interpretation: `1e-4`/`3e-4`/`1e-3` decrease normally; `3e-3` and especially `1e-2` are non-monotonic/elevated, consistent with too-large step sizes. **Adopted:** useful LR region approximately `1e-4`-`1e-3`; `3e-4` adopted as the **provisional Phase-A working anchor**, not a final selected learning rate; `1e-4` broadly competitive but plateaus early (`best_observed_epoch=2`); `3e-3` clearly too high at this fidelity; `1e-2` decisively poor/unstable (`best_observed_epoch=1` for both `3e-3` and `1e-2`, i.e. best performance at the first checkpoint). No evidence supports extending the interval below `1e-4` or above `1e-2`. **Not proof `3e-4` is globally optimal** — Phase B will revisit learning rate jointly with other hyperparameters.

*Cadence finding.* A 3/6-only cadence would have missed the true best-observed checkpoint for all 5/5 candidates (`best_observed_epoch`: `lr1em4`=2, `lr3em4`=5, `ref1em3`=4, `lr3em3`=1, `lr1em2`=1 — none land on 3 or 6). A 2/4/6 cadence recovers it for only 2/5 (`lr1em4`, `ref1em3`). Does not imply every future run must evaluate every epoch; does imply a 3/6-only cadence is too sparse for short 25k screening trajectories when checkpoint localization or trajectory shape matters. Adopted recommendation: denser evaluation or a sustained-performance objective for future broad HPO.

*W&B operational finding.* All four new runs used the committed default tracking policy (`enabled: false`, `mode: disabled`) plus launcher `WANDB_MODE=offline`, with no `WANDB_POLICY_PATH` offline-enabled override supplied — backend `null`, no real W&B run IDs created. Recorded as an operational tracking omission with no effect on scientific validity (evidence was computed directly from checkpoints/raw-space metrics, independent of tracking). Not fixed by this entry; planned as the next small increment (see below).

*Audit incidents (resolved, no contamination of final evidence).* (a) First diagnostic-evaluation sbatch attempt omitted `cd "$REPO"`, so the relative screening-subset path failed to resolve from Slurm's default working directory; all five jobs failed within ~9s before any inference began; corrected and resubmitted, all five completed successfully. (b) Evidence-builder script read the per-epoch median via key `"median"` instead of the actual `"p50"` key, producing `None` medians and degenerate trajectory/cadence summaries; corrected, full build+plot pipeline re-run, results cross-checked against job stdout logs.

*Evidence packet (durable local, checksum-verified).* `.scratch_local/moriah_evidence/lr_a_five_lr_evidence_v001/` and `lr_a_five_lr_evidence_v001.tar.gz` (SHA256 `624c5df4e1823e00b00a303a1c577790c3a72005cc217fcee5dc3e65f186f61c`); manifest verification 23/23 files OK; untracked/gitignored under the existing `.scratch_local/` convention, not staged or committed.

*Next planned stage (recorded, not started).* Small W&B offline-tracking launch-contract fix/qualification (explicit reviewed offline-enabled policy override, or an explicit documented waiver) before the next launch; next Phase-A one-dimensional range characterization, likely hidden size; joint Phase B multidimensional HPO later per L.12's roadmap. No final Stage 1 learning-rate selection is made by this entry.

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no new analysis, no production code/tests/config/policy-YAML change, no generated evidence staged or committed, no W&B tracking-contract fix implemented, no final Stage 1 learning-rate selection.

**L.15 — Hidden-size range characterization (Phase-A) design frozen (2026-08-09): four-candidate `hidden_size` sweep at the LR-A anchor, fresh H=128 (not reused), mandatory tracked-W&B contract, full-trajectory evaluation plan.** Documentation-only design-freeze task, under unchanged commit `785e631f0111fd352035b5b234aec4a774f4aa97` (confirmed against local `HEAD` and `origin/master` before this update; clean tracked tree). Full decision text: `docs/decision_log.md`'s 2026-08-09 entry (topmost); candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-09 section.

*Design (frozen, not launched).* Four hidden-size candidates — `64`, `128`, `256`, `512` — all other settings frozen at L.10-L.14's `[128,32]`/Seed-A/25k-cap contract plus LR-A's provisional `3e-4` anchor held fixed across all four (not re-tuned per hidden size), fixed six-epoch budget, checkpoint every epoch, one uninterrupted epoch-1→6 segment, no continuation beyond epoch 6. Run_ids: `emb128x32_seedA_h64_lr3em4_cap25k_cal`, `emb128x32_seedA_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_h256_lr3em4_cap25k_cal`, `emb128x32_seedA_h512_lr3em4_cap25k_cal`; campaign `hidden_size_range_seedA_25k_v001`.

*Fresh H=128 decision (corrects an earlier proposal, adopted).* The historical LR-A `3e-4` candidate (`emb128x32_seedA_lr3em4_cap25k_cal`, also `hidden_size=128`) is **not** reused as this campaign's H=128 point. A fresh candidate is trained instead, for uniform campaign identity/provenance/tracking across all four points and because the historical run predates this campaign's mandatory tracked-W&B contract (it ran with tracking disabled, backend `null` — LR-A closure entry item (9)). The historical run is retained strictly as a read-only, non-pooled, non-cherry-picked reproducibility comparator; its own descriptive-only reproducibility question against the fresh H=128 run is explicitly deferred until after the fresh run completes.

*W&B contract (new, strict, adopted).* The campaign launcher must default to the reviewed offline-enabled policy (`config/stage1_wandb_tracking_policy_offline_v001.yaml`) and must hard-fail a real training launch if tracking initialization fails or resolves to backend `null`/no run id, unless an explicit human waiver flag is passed — closing LR-A item (9)'s operational gap going forward without retroactively changing any prior run.

*Evaluation design.* Official on-cadence screening epochs 3/6 unchanged; retrospective evaluation of epochs 1/2/4/5 for all four candidates via L.13's already-qualified `pilot_diagnostic_eval.py`, reused unmodified. Full epoch 1-6 trajectories required in the final evidence packet for all four hidden sizes; interpretation must not rely on epochs 3/6 alone, per L.14's cadence finding.

*Scientific caveats (adopted, for the future evidence packet's interpretation section).* LR×hidden-size interaction is untested by this campaign and deliberately deferred to Phase B — LR is held fixed, not tuned per hidden size. The fixed `[128,32]` static embedding is not scaled with hidden size, so its capacity relative to the recurrent pathway changes across the sweep (proportionally larger at H=64, smaller at H=512) — a deliberate simplification for a clean one-dimensional sweep, to be stated as a caveat, not treated as an oversight.

*Minimum implementation plan (not implemented by this entry).* Optional `hidden_size: int | None = None` field on `PilotRunSpec`, threaded through `pilot_lead06_config.py` exactly as `learning_rate` already is; a post-profile-merge `hidden_size` override plus `validate_hidden_size_override()` in `nh_config_generation.py` (no new named run profile — all four candidates reuse LR-A's `pilot_lead06_emb128x32_seedA_v001` profile with both `learning_rate` and `hidden_size` overrides applied); matching manifest/run-identity fields (`hidden_size_override`/`resolved_hidden_size`) in `nh_config_generation.py`'s manifest and `pilot_tracking.build_pilot_run_identity()`; a new `enforce_pilot_hidden_size_identity()` continuation-safety guard in `pilot_orchestration.py`, mirroring the existing cap/LR identity guards exactly; a new opt-in `require_tracking` parameter on `init_pilot_tracking_run()`/`run_pilot()` that hard-fails instead of silently downgrading to backend `null` when set; a new closure-splice launcher (`scripts/run_stage1_hidden_size_range_seedA_closure.py` + `..._moriah.sbatch`) following the LR-A closure launcher precedent, with the historical H=128 run reachable only via `--status-only`; a minimum of eight test categories (validation, config generation, identity/provenance, continuation safety, campaign allowlist, single-segment contract, preparation-only structural comparison, W&B contract).

*Future roadmap (unchanged, reaffirmed).* Phase A: one-dimensional range characterizations (LR-A closed; Hidden-size-A this design; embedding/output dropout remain candidate future axes, not committed). Phase B: joint multidimensional HPO per L.12's funnel, unchanged by this entry.

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no hidden-size candidate launched, no reproducibility comparison against the historical H=128 run performed, no final Stage 1 hidden-size selection.

**L.16 — Hidden-size range characterization (Phase-A) closed (2026-08-10): H=128 provisional anchor, H=64 live alternative, Phase-B support `{64,128,256}`; validation-compatible fixed 8-basin hydrograph panel v001 frozen and accepted; standing Phase-A hydrograph rule adopted; evidence-manifest packaging bug fixed.** Documentation-only closure task recording the completed, already-executed L.15 campaign (four Moriah training runs) and the separately-executed, human-reviewed hydrograph sanity check. No training, evaluation, Slurm job, config/HPO change, or basin reselection was performed by this task; the one code change is a packaging fix to an untracked evidence-assembly script (see below), plus a metadata-only `status` field update in the already-committed selection driver script. Full decision text: `docs/decision_log.md`'s 2026-08-10 entry (topmost).

*Result (from `hsz_trajectory_table.json`/`paired_comparisons_vs_fresh_h128.json`/`h128_reproducibility_comparison.json`, canonical 400-basin screening-validation subset, raw-space NSE, epochs 1-6).* Hidden size is **not sharply sensitive** over the tested `{64,128,256,512}` range at this Seed-A/LR=3e-4/25k-update-cap/6-epoch Phase-A fidelity: all four candidates stay in a broadly similar epochs-4-6 median-NSE band (~0.255-0.278), ordering is non-monotonic in hidden size, and the epoch-6-only ordering (H=64 best) disagrees with the epochs-4-6 window ordering (H=128 best) at the top spot. **H=128 is adopted as the provisional working anchor** for subsequent one-dimensional Phase-A characterization — highest late-window (epochs 4-6) median-of-medians (0.2777) and narrowest late-window range (0.0219) — but is explicitly **not** a final winner or an optimized value. **H=64 is a genuine near-tie and remains a live alternative**, to be carried into later joint Phase-B HPO: single best observed median NSE anywhere in the campaign (0.2922 at epoch 6), no capacity-insufficiency signal, still-rising late trajectory. **H=256 remains a plausible upper useful capacity point** (flat, unremarkable plateau, persistently but only mildly behind H=128). **H=512 showed no demonstrated validation benefit at this fidelity** (fastest-falling training loss did not convert to validation-metric advantage; weakest late-window median-of-medians; behind H=128 in paired comparison at 4/6 epochs) and is **not part of the default Phase-B hidden-size search space** unless later joint evidence justifies revisiting it. **Preferred Phase-B hidden-size support: `{64, 128, 256}`.**

*H=128 reproducibility finding (exact, limited scope).* The fresh (`h128new`, this campaign) and historical (`emb128x32_seedA_lr3em4_cap25k_cal`, L.12-L.14) H=128 candidates are **exactly/deterministically reproducible** under the nominally equivalent Seed-A configuration: identical median NSE at every epoch, zero paired NSE difference (median/p25/p75 all 0.0), `frac_tied=1.0`, per-basin Pearson/Spearman correlation ≈1.0, byte-identical training loss and cumulative optimizer-update counts. This demonstrates exact computational reproducibility, not decisively — it is **not evidence of cross-seed statistical stability**; cross-seed variance remains untested and is a separate open question.

*LR×hidden-size interaction.* Deliberately untested by this campaign (LR held fixed at 3e-4 for all four candidates) and **remains unresolved, deferred to Phase-B joint HPO**. The only indirect hint (training-loss-fall rate scaling cleanly with hidden size while validation median NSE does not follow the same ordering) is consistent with, but does not prove, an LR×hidden-size interaction.

*Validation-compatible fixed 8-basin hydrograph panel v001 — frozen and accepted (`phase_a_validation_hydrograph_panel_v001`).* Separate visualization-only task (no training, no HPO, no sealed-set access), building the standing Phase-A hydrograph sanity-check panel from the canonical 400-basin development-validation screening population instead of the broader, non-validation-restricted train-pool hydrograph atlas (`config/stage1_hydrograph_atlas_selection_v001.yaml`, 24 basins — kept as a separate, independently-versioned selection). Selection: `config/stage1_hydrograph_atlas_selection_validation400_v001.yaml` + `scripts/generate_stage1_validation400_hydrograph_panel_selection.py`, reusing `src/baseline/hydrograph_atlas_selection.py`/`hydrograph_rendering.py` unmodified; candidate-independent (stratified only on the 400-basin population's pre-existing, certified-seed-run epoch009 NSE, unrelated to any hidden-size/LR-A candidate). **Frozen basin IDs (do not regenerate or replace):** `01315000, 06894200, 07165565, 07261000, 08061540, 08072300, 12210900, 14301500`. The selection driver's manifest `status` field is now `"frozen"` (was `"candidate"`) — metadata-only source change, basin membership and event windows byte-identical before/after. **Accepted as a standing Phase-A review artifact, not a CONUS-representative sample and not a second optimization objective** — the panel is geographically imbalanced (5 of 8 basins in the `plains_missouri_south_central` macro-region; 7 of 8 basins on the `west` geo_side; only 3 of 8 possible macro-regions represented at all) and must not be described as geographically representative of CONUS.

*Adopted hydrograph interpretation (accepted findings).* H=64 vs H=128 hydrographs reveal no systematic hydrological superiority of either configuration; the visual evidence is consistent with the quantitative near-tie. LR=3e-4 shows a modest/non-dominant visual edge over LR=1e-3, consistent with L.14's LR-A finding. Shared model-family limitations, not candidate-specific defects, are visible across both matched comparisons and do not overturn the numerical conclusions above: systematic underprediction of some extreme peaks; poor representation of very flashy small-basin spikes (basin 07165565); a shared double-peak prediction artifact (basin 14301500); a shared severe-failure basin (01315000, consistent with a regulated/stepped observed-discharge record) present for all four candidates identically.

*Standing Phase-A hydrograph rule (adopted, applies to future one-dimensional Phase-A milestones).* After each one-dimensional Phase-A characterization milestone: (1) identify a provisionally strongest tested configuration only if the quantitative evidence supports one; (2) render the same frozen `phase_a_validation_hydrograph_panel_v001` 8-basin panel for that configuration; (3) include a matched reference comparison where useful; (4) use the hydrographs as a scientific sanity/interpretability check, not an informal second optimization criterion; (5) preserve the same basin IDs/windows (listed above) across milestones — do not reselect.

*Evidence-manifest packaging bug fixed (untracked-script-only, no scientific data touched).* The hydrograph-panel evidence-assembly script's `find . -type f | ... | xargs sha256sum > MANIFEST_SHA256.txt` pattern truncated `MANIFEST_SHA256.txt` (via shell output redirection) before `find` enumerated the directory, so the manifest recorded a self-referential hash of its own momentarily-empty state — a benign, cosmetic checksum mismatch (72/73 real files always verified clean), not evidence of data corruption. Fixed by excluding the manifest file from `find`'s own listing (`find . -type f -not -name "MANIFEST_SHA256.txt" | ...`) in the untracked `scratch_assemble_val400_evidence.sh`. Evidence packet regenerated with identical scientific content (byte-identical selection CSV, byte-identical renders); corrected manifest reports 72/72 OK; new archive SHA256 `d88990b30b9452080acf44f46b127c8ad042bdab6b73f604f3ae173cc126d104`. Local evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/phase_a_validation_hydrograph_panel_v001/` and `.tar.gz`.

*Screening-subset caveat (reaffirmed).* The 400-basin screening-validation subset is a Phase-A convenience population for fast iteration and hydrograph rendering only — it is **not** scientifically authoritative. The full development-validation population remains the later authority for any promoted/final configuration; nothing in this entry promotes, freezes, or finalizes a Stage 1 hyperparameter.

*Not done by this entry.* No Moriah/h2o access, no Slurm submission, no training or evaluation, no sealed temporal-test/spatial-holdout access, no final Stage 1 hidden-size (or any other hyperparameter) selection, no basin reselection, no embedding-dropout implementation or training.

*Next planned stage (recorded, not started).* Embedding-dropout design survey (next Phase-A one-dimensional characterization axis). Phase B later revisits LR×hidden-size×dropout interactions jointly, per L.12's roadmap.

**L.17 — Embedding-dropout range characterization (Phase-A) design frozen (2026-08-11): one-dimensional five-candidate `embedding_dropout` range characterization at the LR-A/Hidden-size-A anchors, all five fresh (including `0.10`), dropout-specific fidelity caveat, Hidden-size-A W&B contract adopted.** Documentation-only design-freeze task, under unchanged commit `e5c6679464160e89d597363d1e1ae24d58310893` (confirmed against local `HEAD` and `origin/master` before this update; clean tracked tree), following the accepted read-only Embedding-Dropout Design Survey earlier in this session. Full decision text: `docs/decision_log.md`'s 2026-08-11 entry (topmost); candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-11 section.

*Design (frozen, not launched).* Five embedding-dropout candidates — `0.00`, `0.05`, `0.10`, `0.20`, `0.40` — all other settings frozen at L.14/L.16's LR-A/Hidden-size-A contract (`[128,32]` embedding shape/tanh activation unchanged, Seed A, `learning_rate=3e-4` fixed, `hidden_size=128` fixed, output dropout 0.25 untouched), fixed six-epoch budget, checkpoint every epoch, one uninterrupted epoch-1→6 segment, no continuation beyond epoch 6. Run_ids: `emb128x32_seedA_drop00_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop05_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop10_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop20_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop40_h128_lr3em4_cap25k_cal`; campaign `embedding_dropout_range_seedA_25k_v001`. Endpoint meaning: `0.00` no-regularization control, `0.05` light, `0.10` inherited historical default (never itself evidence-selected — confirmed by the accepted design survey), `0.20` moderate, `0.40` a deliberate high boundary intended to probe whether stronger embedding regularization becomes harmful at this Phase-A fidelity. A range characterization, explicitly not an optimized search grid, no interpolation authorized.

*All-fresh decision (binding, extends L.15's fresh-H=128 precedent).* Every candidate, including `0.10`, is trained fresh for this campaign — no reuse of any historical `embedding_dropout=0.10` run, for the same uniform campaign identity/provenance/tracking reasons L.15 gave for training H=128 fresh rather than reusing LR-A's historical run. The fresh Hidden-size-A H=128 run (`emb128x32_seedA_h128_lr3em4_cap25k_cal`, L.15/L.16, `embedding_dropout=0.10` already, trained under the mandatory tracked-W&B contract) is the closest nominally-equivalent historical run and is retained strictly as an optional, descriptive, read-only reproducibility comparator against the fresh `drop10` candidate — never a sixth campaign member, never pooled into the five-candidate comparison, never a substitute. That reproducibility question is deferred until after the fresh `drop10` run completes.

*Fidelity reuse and dropout-specific caveat (new, adopted).* Reuses L.9's 25k-update-cap/six-epoch/Seed-A fidelity unchanged — no new fidelity mechanism. New caveat for the future evidence packet's interpretation section: dropout is a regularization mechanism that can affect *optimization speed* differently than learning rate or hidden size (e.g. higher dropout may slow early convergence while remaining preferable at a longer horizon, or vice versa), so a capped six-epoch trajectory is a weaker proxy for embedding dropout's eventual effect than it was for LR (L.14) or hidden size (L.16). Full six-epoch trajectories matter more here, not less.

*Evaluation design.* Raw-space median NSE (400-basin screening subset) primary, unchanged; the subset remains an operational, non-authoritative convenience population. Official cadence epochs 3/6 unchanged; retrospective evaluation of epochs 1/2/4/5 for all five candidates via L.13's already-qualified `pilot_diagnostic_eval.py`, reused unmodified. Full epoch 1-6 trajectories required in the final evidence packet; interpretation must not rely on epochs 3/6 alone, per L.14's cadence finding, reinforced by the dropout-specific optimization-speed caveat above. No composite "winner score" and no predefined single winner-selection statistic authorized — extends L.12's "no single decision statistic" rule.

*Hydrograph rule and explicit non-goal.* The frozen `phase_a_validation_hydrograph_panel_v001` 8-basin panel (L.16) remains the standing Phase-A sanity-check artifact, to be rendered in a later, separate closure task for the provisionally strongest tested dropout value — **not rendered by this entry**. Monte-Carlo dropout, stochastic repeated inference, and any inference-time-dropout experiment are explicitly out of scope for this campaign: `embedding_dropout` is characterized here as a training-time regularization hyperparameter only; inference-time dropout behavior (normally disabled in eval mode) is untouched.

*W&B contract.* Adopts L.15's strict standard as this campaign's default: the launcher must default to the reviewed offline-enabled policy and hard-fail a real training launch on tracking failure or null-backend resolution, unless an explicit human waiver flag is passed.

*Minimum implementation plan (not implemented by this entry, 7 items).* (1) `PilotRunSpec.embedding_dropout: float | None = None` in `src/baseline/pilot_lead06_config.py`, additive, following the `hidden_size`/`learning_rate` precedent. (2) `load_pilot_policy()`'s two existing hard-equality gates (top-level `embedding_dropout`, per-profile `statics_embedding.dropout`) become override-aware — a run with an explicit override reconciles against the override, not the `0.1` profile default; every non-overridden run keeps the unchanged hard-equality check. (3) `nh_config_generation.py`: post-profile-merge `embedding_dropout` override in `build_nh_config_mapping()`; new `validate_embedding_dropout_override()` (non-numeric/bool/NaN/inf rejected, `[0,1)` bound matching `validate_statics_embedding_spec()`); `GeneratedConfigBundle.embedding_dropout`; manifest records `embedding_dropout_override`/`resolved_embedding_dropout`. No new named run profile — all five reuse `pilot_lead06_emb128x32_seedA_v001` with `learning_rate`, `hidden_size`, and `embedding_dropout` overrides applied together. (4) `pilot_tracking.build_pilot_run_identity()` gets matching `embedding_dropout_override`/`resolved_embedding_dropout` fields. (5) `pilot_orchestration.py`: new `enforce_pilot_embedding_dropout_identity()` continuation guard mirroring `enforce_pilot_hidden_size_identity()` exactly; `run_pilot()` calls it alongside the existing cap/LR/hidden-size guards. (6) New closure-splice launcher `scripts/run_stage1_embedding_dropout_range_seedA_closure.py` + `..._moriah.sbatch`, following the L.15/L.16 Hidden-size-A closure launcher precedent: fixed `max_target_epoch=6`, closed five-run_id allowlist, `REFERENCE_RUN_ID="emb128x32_seedA_h128_lr3em4_cap25k_cal"` reachable only via `--status-only`, default offline-enabled W&B policy, opt-in waiver only. (7) Minimum eight test categories: embedding-dropout validation (`[0,1)` bound, `0.00` boundary case); config generation (`drop00` resolves to `dropout: 0.0`, not skipped as falsy); identity/provenance; continuation safety; campaign allowlist (five trainable, historical comparator not trainable); single-segment contract; preparation-only structural comparison; W&B contract. **No premature generalization of the per-axis identity guards into a shared abstraction is authorized** unless a future implementation task's own inspection reveals a compelling concrete reason.

*Future roadmap (unchanged, reaffirmed).* Phase A: one-dimensional range characterizations (LR-A closed; Hidden-size-A closed; Embedding-Dropout-A this design; output dropout remains a candidate future axis, not committed). Phase B: joint multidimensional HPO per L.12's funnel, including any LR×hidden-size×embedding-dropout interaction, unchanged by this entry.

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no embedding-dropout candidate launched, no hydrograph panel rendered, no reproducibility comparison against the historical H=128/dropout=0.10 run performed, no final Stage 1 embedding-dropout selection.

**L.18 — Embedding-Dropout-A implementation complete, preparation-only validated, ready for Moriah launch review (2026-08-11): all 8 items of L.17's minimum implementation plan built and tested; two-layer real preparation-only audit confirms correct config generation for all five candidates; still not launched.** Implementation and local/preparation-only validation task, under unchanged commit `eea9f4c09bbfdb92b757ec4165b0bb61a7b466ba`, branch `master`, following L.17 immediately above. Full decision text: `docs/decision_log.md`'s topmost 2026-08-11 entry; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new implementation subsection.

*Implemented, all 8 items of L.17's plan, with one deliberate, reviewed deviation from item (2)'s wording.* (1) `PilotRunSpec.embedding_dropout: float | None = None`, additive. (2) Only `load_pilot_policy()`'s **per-profile** `statics_embedding.dropout` `0.1` gate was made override-aware; every non-overridden run keeps the unchanged check. The **top-level, policy-wide `embedding_dropout: 0.1` gate is deliberately left strict and unchanged**, not made override-aware as L.17's plan wording literally described — independent review preferred preserving this invariant as safer, since it keeps 100% of existing non-overridden policy behavior unchanged. No change to the top-level gate is needed: Embedding-Dropout-A's five new `PilotRunSpec`s are spliced into the already-loaded/validated base `PilotPolicy` in memory by the closure launcher, so the committed `stage1_lead06_pilot_v001.yaml`'s policy-wide `embedding_dropout: 0.1` is never re-validated against a dropout-varying entry; explicit candidate variation reaches the generated config solely through `PilotRunSpec.embedding_dropout` → the per-profile gate above → `build_nh_config_mapping()`'s post-merge override (item 3 below). Regression coverage: `tests/test_pilot_lead06_config.py::test_load_pilot_policy_rejects_wrong_top_level_embedding_dropout` (pre-existing, predates this campaign, untouched by this implementation) already proves the top-level gate still rejects a policy-wide `embedding_dropout` value drifted away from `0.1`. (3) `nh_config_generation.py`: post-profile-merge `embedding_dropout` override in `build_nh_config_mapping()`; `validate_embedding_dropout_override()` (non-numeric/bool/NaN/inf rejected, `[0,1)` bound); `GeneratedConfigBundle.embedding_dropout`; manifest records `embedding_dropout_override`/`resolved_embedding_dropout` — verified `0.00` recorded as an explicit `0.0`, never confused with "no override" (`None`), end-to-end through validator/mapping/run-identity/continuation-guard/standalone-CLI-audit. (4) `pilot_tracking.build_pilot_run_identity()` matching fields. (5) `pilot_orchestration.enforce_pilot_embedding_dropout_identity()`, mirroring `enforce_pilot_hidden_size_identity()` exactly; wired into `run_pilot()` alongside the cap/LR/hidden-size guards. (6) `scripts/run_stage1_embedding_dropout_range_seedA_closure.py` + `..._moriah.sbatch`: `EMBEDDING_DROPOUT_A_MAX_TARGET_EPOCH=6` fixed, `EMBEDDING_DROPOUT_A_RUN_SPECS` with exactly the five new run_ids, `REFERENCE_RUN_ID="emb128x32_seedA_h128_lr3em4_cap25k_cal"` reachable only via `--status-only`, collision guard (`_OTHER_CAMPAIGN_RESERVED_RUN_IDS`) against the real policy and all prior campaigns, default offline-enabled W&B policy. (7) Tests spanning all 8 planned categories, split across the existing focused suites plus three dedicated campaign test files (`test_run_stage1_embedding_dropout_range_seedA_closure_cli.py`, `test_embedding_dropout_range_seedA_closure_sbatch_launcher.py`, `test_embedding_dropout_range_seedA_closure_preparation.py`). No premature generalization of the per-axis identity guards was performed, per L.17's constraint.

*Preparation-only validation, two independent real layers.* (a) The pytest preparation suite calls the real, unmodified `prepare_pilot_run_only()` for all five candidates against a synthetic package covering the full 2,557-basin development/spatial-holdout union (`tests._pilot_support.build_full_union_package`) and the real committed policy/split files; confirms pairwise config diffs limited to `experiment_name`/basin-list paths/`run_dir`/`statics_embedding`, `data_dir` and basin-list contents identical, every `experiment_name`/`run_dir`/W&B identity pairwise-unique, `training_started`/`evaluation_started`/`wandb_backend_initialized` all `False` for every candidate. (b) An additional standalone, non-pytest audit script invoked the real closure-launcher CLI as a real subprocess with `--prepare-only` for all five run_ids, writing real `config.yaml`/`generation_manifest.json` files to disk (short-path root `C:\edA_prep_audit`, required to avoid a Windows `MAX_PATH` write failure encountered under the deep session-scratchpad path). All five returned `status=PREPARED_ONLY`; `drop00`/`drop40` spot-checked directly on disk: `statics_embedding.dropout` resolved to exactly `0.0`/`0.4` with `hidden_size=128`, `learning_rate=0.0003`, `output_dropout=0.25`, `seed=967139`, `seq_length=24` identical between them.

*Tests.* Focused: `test_nh_config_generation.py` 154 passed, `test_pilot_tracking.py` 42 passed, `test_pilot_orchestration.py` 102 passed/5 skipped, `test_pilot_lead06_config.py` 52 passed. Dedicated campaign files: 130 passed. Wider related-suite: 455 passed across 17 files. Full local regression suite, excluding 6 pre-existing torch/neuralhydrology-dependent collection-error files unrelated to this task (`test_nh_dataset.py`, `test_nh_evaluation_check.py`, `test_nh_full_population_structural_preflight.py`, `test_nh_register.py`, `test_nh_structural_preflight.py`, `test_run_stage1_nh_entrypoint.py`): **2070 passed, 5 skipped, 1 failed** in 1141.92s. The failure, `test_package_audit.py::test_fails_when_disk_coordinate_is_time_but_declared_schema_is_v002`, is a Windows `os.rename`/`WinError 5` file-locking race in unrelated pre-existing package-builder atomic-promotion test infrastructure — classified as a pre-existing environment-dependent flake, not a regression from this task.

**Not done by this entry.** No embedding-dropout candidate launched, no Slurm job submitted, no real NeuralHydrology training or checkpoint evaluation call, no W&B Sweep, no scientific-design change, no sealed temporal-test/spatial-holdout/California access, no full-population validation, no hydrograph panel rendered, no reproducibility comparison against the historical H=128/dropout=0.10 run performed, nothing committed.

**L.19 — Embedding-Dropout-A closed (2026-08-13): weak sensitivity over `0.00`-`0.40`, no candidate dominates, `drop10` retained as provisional anchor (not proven optimal), hydrograph sanity check clean, revised Phase-A/Phase-B roadmap supersedes L.1/L.10's sequence-length exclusion.** Documentation-only closure task recording the completed L.17/L.18 campaign (five real Moriah training runs under the strict offline-W&B contract) and the separately-executed, human-reviewed fixed 8-basin hydrograph sanity check. No training, evaluation, Slurm job, config/HPO change, or basin reselection performed by this task. Full decision text: `docs/decision_log.md`'s 2026-08-13 entry (topmost).

*Result (from the closure evidence packet `tmp/embedding_dropout_a_closure_evidence_v001/`, canonical 400-basin screening-validation subset, raw-space NSE, epochs 1-6; Moriah training jobs `drop10` 45789423/`drop00` 45790661/`drop05` 45790662/`drop20` 45790663/`drop40` 45790664, retrospective diagnostic-evaluation jobs 45790996-45791000, optimizer/update verification and paired-comparison job 45791007).* Embedding dropout is **weakly sensitive** across the tested `{0.00,0.05,0.10,0.20,0.40}` range at this fidelity. No candidate robustly dominates across epoch-6 median NSE, epochs-4-6 sustained/late-window behavior, matched-basin paired differences, or the hydrograph review. Ranking is cadence-sensitive: `drop00` leads the epoch-6 endpoint; `drop10` has the strongest late-window summary and the single best observed checkpoint in the campaign; `drop20` is among the most stable late-window candidates — differences are comparable to ordinary epoch-to-epoch variation. Higher dropout monotonically raises transformed-space training loss without producing a monotonic raw-space validation-NSE relationship. `drop40` shows **no validation-performance cliff** at this fidelity and must not be described as clearly excessive or rejected.

*Reproducibility finding (exact, limited scope, extends L.16's H=128 finding to the dropout axis).* Fresh `drop10` (this campaign) and the historical nominally-equivalent H=128/dropout=0.10 run (`emb128x32_seedA_h128_lr3em4_cap25k_cal`, L.15/L.16, commit `aec9b9cb7a6cd0b0578141c3d96d3a6df40b4b04`) are **exactly/deterministically reproducible**: identical epoch-by-epoch validation and training-loss trajectories, identical optimizer-update counts, zero paired-basin NSE difference. Demonstrates computational reproducibility only, not cross-seed statistical stability.

*Standing Phase-A hydrograph rule applied (L.16).* The frozen `phase_a_validation_hydrograph_panel_v001` 8-basin panel was rendered for `drop00` (epoch 6), `drop10` (epoch 5, its own best observed checkpoint), and `drop20` (epoch 6) — same frozen basins, event windows, target-valid timing, and shared plotting scales as prior Phase-A milestones. **Accepted finding.** Broad similarity across most basins; basin/event-specific divergences at 07261000 (`drop20` false peak), 08072300 (`drop10` false peak), and 14301500 (`drop10` damped peak) — each implicating a different candidate in a different direction, judged basin/event-specific, not a repeated candidate-specific pathology. The visual evidence does not contradict the aggregate quantitative near-tie. Evidence: `tmp/embdrop_sanity_panels_v001_evidence/drop00_drop10_drop20_v001/`, 48/48 files checksum-verified, untracked (Moriah job 45791211).

*Provisional anchor (adopted, explicitly not a final selection).* `embedding_dropout=0.10` remains the provisional working anchor — not because `0.10` was proven optimal, but because it sits safely inside the broad viable tested region, has a strong sustained late-window result, exactly reproduces the historical run, and nothing in this campaign's evidence justifies moving the anchor. Final embedding-dropout selection is deferred to Phase B joint HPO. The tested `0.00`-`0.40` region remains broadly viable at this fidelity and must not be aggressively narrowed by this result.

*Future artifact-identification infrastructure requirement (recorded, not implemented).* Reviewing this closure's hydrograph evidence exposed a portability/provenance weakness (not a defect in, and not grounds to regenerate, any completed evidence): PNG titles carry basin/checkpoint/metric identity but not necessarily campaign/candidate/run identity; generic filenames become ambiguous once separated from their parent directory; not every artifact in a bundle is individually self-identifying even though `compact_event_metrics.csv` carries `candidate_id`. Folded into the roadmap's item 1 below. Recommended future convention (design later): concise candidate/campaign/checkpoint identity in plot titles; full provenance in a rendering/evidence manifest; portable archive names containing campaign + candidate/run + checkpoint; no reliance on parent-directory context alone.

*Revised optimization roadmap (adopted, supersedes this Part's own L.1 and L.10 sequence-length exclusion — see the forward-pointing notes added there; those entries are preserved as historical, not rewritten).*
1. Reusable Phase-A/HPO Campaign Infrastructure Consolidation — consolidate LR-A/Hidden-size-A/Embedding-Dropout-A campaign machinery; support future dimensions without cloning ~500-line launchers; include the durable artifact/evidence identity requirements above; keep scientific campaign definitions explicit and auditable.
2. Sequence-Length-A — characterize `seq_length={12,24,48,72}` at the best-supported current anchor, as a bounded structural/calibratable model parameter rather than a permanently fixed 24h choice; lead time stays separate; primary comparison uses each candidate's naturally admissible samples; a lightweight common basin/timestamp-support audit will be added if practical.
3. Dynamic-input family characterization — audit whether gap-flag channels carry nonzero information in scientifically admitted samples first; define a small number of physically meaningful input families; compare at a common mature anchor; use a small standardized adaptation/rescue probe before eliminating a family rather than a full independent one-dimensional HPO per family.
4. Phase B joint HPO — LR × hidden size × embedding dropout × output dropout jointly, per L.12's funnel; carry multiple sequence lengths/input families into Phase B only on genuine near-ties from earlier characterization; not every dimension requires its own exhaustive Phase-A campaign.

*Not done by this entry.* No Moriah/h2o access, no Slurm submission, no training or evaluation, no rendering, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no final Stage 1 embedding-dropout selection.

**L.20 — Sequence-Length-A closed (2026-08-15): `seq_length=72` adopted as provisional working anchor, `seq_length=48` nearest alternative, no saturation observed by 72h, hydrograph sanity check clean, comparative-hydrograph convention frozen.** Documentation-only closure task recording the completed campaign (four real Moriah training runs, infrastructure commit `4646a55`, training jobs `45861222`-`45861225`) and the separately-executed, human-reviewed fixed 8-basin hydrograph sanity check plus a supplemental single-basin diagnostic. No new training, evaluation, or rendering performed by this task. Full decision text: `docs/decision_log.md`'s 2026-08-15 entry (topmost).

*Design/execution.* Four candidates — `seq12`/`seq24`/`seq48`/`seq72` (`seq_length` = 12/24/48/72) — all other settings frozen at the Embedding-Dropout-A-closed contract (L.19): Seed A (967139), `[128,32]` embedding (tanh, dropout 0.10), `hidden_size=128`, `learning_rate=3e-4`, `output_dropout=0.25`, lead 6h, target `qobs_mm_per_h_lead06`, `max_updates_per_epoch=25000`, six epochs, one uninterrupted segment, the fixed development-training population, the fixed ~400-basin screening population, strict offline W&B. Campaign: `seq_length_range_seedA_25k_v001`. All four completed six epochs / 25k-update cap cleanly, no resource failures, approximately comparable runtime.

*Quantitative result.* Natural-support raw-space median NSE, a new common-support-corrected evaluation (`src/baseline/common_support_audit.py`, see below), true per-basin paired comparisons, and late-window behavior all agree on the same ordering at every evaluated epoch: `seq72 > seq48 > seq24 > seq12`. Transformed-space training loss corroborates but remains diagnostic-only, never authoritative. The common-support correction did not materially change the ranking.

*Hydrograph sanity check.* The frozen `phase_a_validation_hydrograph_panel_v001` 8-basin panel rendered for `seq24` (epoch 5), `seq48` (epoch 6), `seq72` (epoch 5) — each candidate's own best observed checkpoint — with the *displayed* antecedent window widened from ~24h to 72h via the new `derive_display_window()` helper (below), so the longest tested context is visible for all three simultaneously; the frozen event/peak identity itself was never re-selected. **Result: CONSISTENT with the quantitative ranking, no repeated `seq72`-specific pathology.** A supplemental diagnostic rendered basin `06131200` alone: all three candidates fail severely at its May 2024 extreme event (NSE −6.6 to −52.3), but identically across all three sequence lengths — a shared near-zero-flow basin/model pathology and NSE-denominator sensitivity, not a `seq72`-specific defect. Evidence (untracked, checksum-verified `ALL_MATCH`): `tmp/sequence_length_a_hydrograph_sanity_v001/{frozen_panel,basin_06131200_diagnostic}/` (14 + 7 files).

*Decision (adopted).* `seq_length=72` becomes the **provisional Stage-1 working anchor** within the tested 12-72h range; `seq_length=48` is the nearest credible alternative. Performance had **not clearly saturated by 72h** — 72h is not claimed to be the final optimum; whether a longer lookback would help further is an **open question**, deferred to later higher-fidelity/integrated work. **No longer-lookback campaign is launched by this entry.**

*Comparative-hydrograph convention (new, adopted — supplements L.16's standing Phase-A hydrograph rule for small-candidate-set comparisons generally, not only sequence-length studies).* Prefer one panel per basin/event: the same frozen event/time window and observed hydrograph, shared axes/scales, all candidates overlaid — not separate per-candidate figures. Event selection stays independent of candidate performance and is never re-run once frozen (basin, peak time, peak value, `window_end` fixed); candidates clearly identified; identical observed/forcing context; no independent per-candidate autoscaling. Only for experiments that specifically vary historical context (like Sequence-Length-A) may the *displayed* antecedent window widen — presentational only, via `derive_display_window()`, never a re-selection of the frozen event/peak identity. Large candidate sets should use a representative subset or another consistent layout rather than one cluttered panel. **Not a mandate that every future panel use a 72h window** — the window widens only when the experiment itself varies antecedent context.

*Code retained.* `hydrograph_rendering.py`'s `DisplayWindow`/`derive_display_window()` — read-only, additive, backward-compatible, separate from event selection. `src/baseline/common_support_audit.py` (new) — a narrow secondary fairness audit restricting comparison to `(basin, timestamp)` positions admitted by every candidate simultaneously; reuses `nh_seed_evaluation`/`nh_raw_space_evaluation`'s metric math verbatim, adds exactly one new operation (admitted-mask intersection). Both judged in-scope and reusable, not campaign-specific scratch. 74 focused tests pass (`tests/test_hydrograph_rendering.py` + `tests/test_common_support_audit.py`).

*Moriah workaround cleaned up.* A temporary ad hoc file copy plus a narrowed tracked-file git-guard (in the untracked `seqA_hydro_sanity_moriah.sbatch`, never tracked production code) used during the hydrograph task is retired: the two files are now committed normally, Moriah re-synchronized to the committed `HEAD` via `git pull --ff-only`, and the guard-narrowing reverted to its original strict form.

*Revised roadmap (item 2 now closed; items 1/3/4 unchanged from L.19).* 1. Reusable Phase-A/HPO campaign infrastructure consolidation — still pending. 2. ~~Sequence-Length-A~~ — **closed by this entry.** 3. **Dynamic-input family characterization — next milestone**, at the new `seq_length=72` anchor. 4. Phase B joint HPO — still deferred.

*Not done by this entry.* No Moriah/h2o access beyond sync/verification, no new Slurm submission, no evaluation of any length beyond 72h, no dynamic-input-family characterization started, no Phase B started, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no final Stage 1 sequence-length selection beyond the provisional working anchor.

**L.21 — Dynamic-Input-Family-A design frozen (2026-08-16): four-family `P`/`PT`/`PTM`/`PTMW` dynamic-input hierarchy at the `seq_length=72` anchor, gap channels removed from model inputs, dewpoint deliberately omitted from the primary hierarchy, U/V wind kept paired.** Documentation-only design-freeze task, under unchanged commit `dda254b` ("Close Sequence-Length-A and adopt 72-hour working context"), following the accepted read-only Dynamic-Input Family Design Survey earlier in this session and a full-scale (2,307-basin) re-confirmation of that survey's two evidentiary audits. Full decision text: `docs/decision_log.md`'s 2026-08-16 entry (topmost); candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-16 section.

*Gap-flag audit re-confirmed at full scale (`gap_flag_channel_leakage_audit_20260815.json`).* Development-training 57,453,283 admissible windows / 0 MRMS-gap-positive / 0 RTMA-gap-positive / 0 issue-time-flag-positive; development-validation 19,528,664 admissible windows, identically all-zero; package reconciliation 136 MRMS + 2 RTMA = 138 canonical gap timestamps, symmetric difference 0. *Gap-channel decision:* `mrms_qpe_1h_mm_gap`/`rtma_gap` stay in the certified package unchanged (QC/provenance package variables); under the current hard-exclusion admission policy both are constant-zero for every admissible development window, so they are removed from the NH model's `dynamic_inputs` vector (model predictor variables) for this campaign only — not a claim they are useless under a different admission policy, and not a package change/rebuild/version-bump.

*Physical-variable audit re-confirmed at full scale (`dynamic_input_family_audit_20260815.json`).* Six physical `v001-core` variables, no sentinel/clipping/scaling pathology. Moisture redundancy: dewpoint-vs-specific-humidity Pearson ≈0.9512 (85.26M points), Spearman ≈0.9947, seasonal ≈0.925-0.963, temperature-regime ≈0.906-0.967, basin-level median ≈0.964/p95 ≈0.977. Wind components plausible and non-degenerate, no evidence either is redundant alone.

*Moisture decision (adopted, cautious).* `rtma_2sh_kgkg` adopted as the primary single moisture representation for `PTM`/`PTMW`, on grounds of strong empirical redundancy, simplicity, and being a directly sourced moisture mass-fraction — a Phase-A structural simplification, **not** proof dewpoint has zero predictive value. **Must never be justified by the historical (fixed) dewpoint lookup-key bug** — that bug is not scientific evidence against dewpoint. Both-moisture ablation remains a deferred future option.

*Wind decision.* U/V always travel together; no U-only/V-only family; wind inclusion is its own structural step (`PTMW`), not automatic once moisture is included.

*Frozen family matrix.* `P` = `mrms_qpe_1h_mm`. `PT` = `+ rtma_2t_K`. `PTM` = `+ rtma_2sh_kgkg`. `PTMW` = `+ rtma_10u_ms, rtma_10v_ms`. Intentionally 5 physical channels, not 6 — dewpoint omitted from the primary hierarchy, gap flags package-only. `v001-fullmet` (pressure/cloud/visibility/gust/ceiling) explicitly deferred, not implemented.

*Common anchor (unchanged from Sequence-Length-A's closed contract; every candidate varies only `dynamic_inputs`).* Seed A (967139), `[128,32]` embedding (tanh, dropout 0.10), `hidden_size=128`, `learning_rate=3e-4`, `output_dropout=0.25`, `seq_length=72`, lead 6h, target `qobs_mm_per_h_lead06`, `max_updates_per_epoch=25000`, six epochs, one uninterrupted segment, fixed development-training/screening populations, strict offline W&B. Campaign: `dynamic_input_family_seedA_25k_v001`.

*Evaluation design.* Raw-space median per-basin NSE (400-basin screening) primary; full epoch 1-6 trajectories; true per-basin paired comparison; late-window (4-6) behavior; transformed-space loss diagnostic-only. Frozen 8-basin hydrograph panel applies at the **standard, non-widened** display window — Sequence-Length-A's 72h display widening was specific to a context-length experiment and does not carry over to a variable-family experiment.

*Rescue policy (rule only, not exercised).* At most one standardized `hidden_size=256` capacity probe for a single weak/ambiguous non-reference family (`PT`/`PTM`/`PTMW`); `P` is the reference and is never rescued; not pre-created/pre-trained; not a second search dimension; not guaranteed to change conclusions; not part of the base four-candidate allowlist.

*Deferred.* Dewpoint/both-moisture ablation, `v001-fullmet`, longer sequence lengths, Phase B, sealed-set evaluation.

*Minimum implementation (already built this same session).* `PilotRunSpec.dynamic_inputs` override; `validate_dynamic_inputs_override()` in `nh_config_generation.py`; resolution threading in `pilot_lead06_config.py` (pre-existing `validate_dynamic_inputs()` package-integrity gate left unchanged); `dynamic_inputs_override`/`resolved_dynamic_inputs` provenance in `pilot_tracking.build_pilot_run_identity()`; new `enforce_pilot_dynamic_inputs_identity()` continuation-safety guard in `pilot_orchestration.py`, mirroring the five existing scalar-identity guards; closure-splice launcher/sbatch mirroring the Sequence-Length-A template, four trainable run_ids, no rescue candidate in the base allowlist.

*Not launched by this entry.* No dynamic-input-family candidate trained, no Slurm job submitted — design freeze only; implementation/preparation-only qualification recorded separately as this session continues.

**L.22 — Dynamic-Input-Family-A CLOSED (2026-08-16): `PT` (precipitation + temperature) adopted as the provisional Stage-1 working family; `PTM`/`PTMW` not promoted; no H256 rescue warranted.** Documentation-only closure task recording the completed base campaign (four real Moriah training runs under the L.21 contract, campaign implementation commit `a3bf51266859a8706b40cc9e862acab793ce15c7`), the true multi-candidate hydrograph-overlay review, and a dedicated 400-basin/1,200-event high-flow and event-level audit, all already executed earlier in this session. No new training, no H256 rescue, no Phase B, no Stage-1 Evaluation Framework v1 implemented by this entry. Full decision text: `docs/decision_log.md`'s 2026-08-16 CLOSED entry (topmost); candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-16 closing section.

*Whole-record result.* Raw-space median per-basin NSE (400-basin screening population), all 6 epochs, true per-basin paired comparisons: `PT` beats `P` in 63-71% of matched basins at every epoch (median gain ≈0.03-0.06) — the one robust, repeated, basin-general improvement in the campaign. `PT` epoch 3 achieves the campaign's strongest observed whole-record skill (median NSE ≈0.3726, 75,000 cumulative updates). `PTM` shows no reproducible incremental benefit over `PT` (fraction-improved oscillates around 0.5 at every epoch). `PTMW` is a near-tie with `PT` on whole-record skill (median diff within ±0.02, fraction ≈0.43-0.57 with sign flips across epochs).

*True hydrograph-overlay review.* The frozen 8-basin panel (Obs + all 4 candidates, shared axes, same frozen event windows) showed 3 of 8 basins (`07261000`, `08072300`, `14301500`) with apparently meaningful `PTM`/`PTMW` improvement during specific high-flow events — treated explicitly as illustrative and used to motivate, not substitute for, the population-level audit below.

*High-flow/event audit (400/400 basins, 1,200 deterministically selected Q95 events, top-3/basin, 72h peak separation, 24h-before/48h-after window, event-weighted + basin-balanced views).* Conditional (flow ≥ basin Q95): `PT` vs `P` clearly positive (58-64% of basins on RMSE/KGE/|PBIAS|); `PTMW` vs `PT` a small, checkpoint-robust edge (52-60%). Event-level: `PT` vs `P` remains positive on peak magnitude/volume/shape (55-61%); `PTMW` vs `PT` is essentially a near-tie on peak magnitude and volume (~50-51%), only a small positive tendency on shape — **the conditional `PTMW` edge does not translate into a broad event-level peak/volume advantage.** Peak timing is tie-dominated (~44-50%) for both comparisons and not discriminative. Severity stratification (`[Q95,Q99)` vs `≥Q99`): **no detectable increase in `PTMW` benefit observed across the severity strata represented by this selected event sample** — not a claim severity dependence is ruled out generally. Per-basin cross-check confirmed the 3 flagged overlay basins sit in the favorable tail of the 400-basin population, while `06894200`/`08061540` show the opposite (conditional gain reverses at the event level) — overlays are interpretation/sanity evidence, not model-selection votes; the 8 frozen basins are not representative of the 400-basin population.

*Final decision (adopted, provisional).* `PT` — `mrms_qpe_1h_mm` + `rtma_2t_K` — is **the provisional Stage-1 Dynamic-Input-Family-A working family**. Not "the final optimal dynamic-input family," not "globally optimal," not "permanently superior to `PTMW`," not "proof that humidity/wind do not matter." Eight-point rationale: (1) `PT` is the one robust, repeated, basin-general improvement over `P`; (2) `PT` achieves the strongest observed whole-record skill; (3) `PTM` shows no reproducible incremental benefit; (4) `PTMW` is near-tied with `PT` on whole-record skill; (5) `PTMW`'s conditional high-flow edge does not translate into a broad event-level peak/volume advantage; (6) added moisture/wind complexity has not earned promotion over `PT` at this fidelity; (7) no H256 rescue is warranted by the evidence; (8) `PTMW` remains the nearest broader credible alternative, not dismissed as useless. `PT` epoch 3 may be called **the best observed `PT` checkpoint in this specific Phase-A campaign**, not a universal training-budget rule. **This closes the predictor-family decision, not the future training-duration decision.**

*Rescue policy — not exercised.* No candidate's result was weak/ambiguous enough under the L.21 rescue policy to warrant the standardized `hidden_size=256` probe; none launched, none authorized by this entry.

*Reusable evaluation capabilities retained (committed).* `render_multi_candidate_basin_panel()` (`src/baseline/hydrograph_rendering.py`) — general-purpose true N-candidate overlay renderer, not Dynamic-Input-specific. `select_high_flow_events()` (`src/baseline/hydrograph_atlas_events.py`) — deterministic, observed-only, candidate-independent high-flow event selector, retained alongside the pre-existing `select_atlas_events()` (unchanged, distinct purpose). `src/baseline/high_flow_event_metrics.py` (new) — `basin_high_flow_threshold()`, `high_flow_conditional_metrics()`, `event_metrics()`, reusing `raw_space_metrics()` rather than reimplementing metric math. Matching test suites committed for all three. No large new evaluation architecture built for this closure.

*Known cosmetic issues (documented, not fixed).* Untracked event-audit analysis scratch had a severity-breakdown CSV missing a couple of descriptive columns and inconsistent basin-ID zero-padding in one intermediate join — cosmetic, confined to non-committed scratch, no effect on any reported metric.

*Figure pack and supervisor summary (project-local, gitignored, not committed).* `.scratch_local/moriah_evidence/dynamic_input_family_a_closure/figures_v001/` — 8 figures (learning trajectories; paired NSE effects; fraction-improved; high-flow conditional; event-level comparison [central closure figure]; general-vs-event synthesis [not a composite score]; representative hydrograph montage from the pre-existing frozen 8-basin panel; progression-diagram schematic) + `FIGURE_INDEX.md`. `SUPERVISOR_SUMMARY.md` in the parent directory. `EVIDENCE_MANIFEST.md` records source-file identity/SHA256/campaign commit/script identity/timestamp.

*Revised roadmap (item 3 now closed; items 1/4 unchanged from L.20).* 1. Reusable Phase-A/HPO campaign infrastructure consolidation — still pending. 2. ~~Sequence-Length-A~~ — closed (L.20). 3. ~~Dynamic-input family characterization~~ — **closed by this entry.** 4. **Stage-1 Evaluation Framework v1 + Phase-B Fidelity Design — next milestone**, named only, not started, not designed by this entry; unresolved questions carried forward: which metrics are routine vs. diagnostic-only; which metrics are authoritative for promoting a configuration; how to formalize the high-flow/event audit methodology into a standing protocol; how to design a higher-fidelity Phase-B protocol before the first broader HPO search; whether/when an H256 (or other capacity) probe should be revisited for `PTMW` under a future higher-fidelity protocol; whether dewpoint or a both-moisture ablation should be revisited. Phase B joint HPO itself remains deferred behind this framework/design work.

*Not done by this entry.* No new training launched, no H256 rescue run, no Phase B started, no Stage-1 Evaluation Framework v1 implemented, no sealed temporal-test/spatial-holdout/California data accessed, no generated evidence committed or staged beyond the reusable `src/`/`tests/` set and canonical documentation.

**Post-L.22 transition note (2026-08-19, documentation-only, no new Part-L entry created here).** L.22's roadmap item 4 ("Stage-1 Evaluation Framework v1 + Phase-B Fidelity Design — next milestone, named only") has been expanded into a full design/handoff document: `docs/stage1_phase_b_hpo_evaluation_plan.md`. Summary only — that document, not this note, is canonical: Phase A (LR-A, Hidden-size-A, Embedding-Dropout-A, Sequence-Length-A, Dynamic-Input-Family-A) has closed its one-dimensional structural characterizations; Phase B is now intended as **joint multidimensional HPO** (W&B Bayesian search plus a seeded random-search control, drawn from the same space); the initial Phase-B search objective is median per-basin raw-space NSE on the frozen development-validation screening population, unchanged in kind from Phase A's own evaluation metric; a parallel, richer Evaluation Framework v1 (categorical detection metrics, variable-duration observed-only event evaluation) feeds later promotion/final-interpretation decisions but does not replace the Sweep-v1 search objective. Exact Sweep-v1 search-space dimensions, medium-fidelity training/evaluation protocol, and W&B/Slurm sweep architecture remain explicitly open, to be resolved by the separate Task A/Task B design reviews named in the new document, not by this note.

**2026-08-20 update.** The canonical Phase-B plan now decides the five
Sweep-v1 axes (`learning_rate`, `hidden_size`, `embedding_dropout`,
`output_dropout`, `batch_size`), the common `max_updates_per_epoch=50,000`
medium-fidelity cap, every-epoch raw-space screening, and no
performance-based early stopping. The total epoch budget, exact trial
budgets/concurrency, and W&B/Slurm sweep architecture remain open. This update
supersedes the preceding transition note only where it describes those
Sweep-v1 details as open; the canonical Phase-B plan remains authoritative.

**2026-08-21 update.** The epoch-budget *calibration design* is now frozen: five Seed-A candidates will run one continuous 14-epoch, 50k-update-cap, per-epoch-checkpoint/no-performance-stop trajectory with every epoch eligible for canonical raw-space screening. This records no result: the common Sweep-v1 epoch budget remains open pending the later 8/10/12/14 cutoff review; `docs/stage1_phase_b_hpo_evaluation_plan.md` §7 is authoritative.

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
