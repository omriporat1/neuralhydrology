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
