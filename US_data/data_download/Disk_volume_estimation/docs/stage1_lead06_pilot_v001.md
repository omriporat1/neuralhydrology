# Stage 1 lead-6 optimization pilot (`stage1_lead06_pilot_v001`)

Status: **`emb128x64_seedA` candidate complete.** The epoch 6→15
continuation trajectory was explicitly adopted via the run-specific
SHA-256 manifest and screened to completion in job `45722908`; early
stopping fired at epoch 15, and epoch 6 is the selected checkpoint for
this one candidate configuration (not a claim about the final Stage 1
production model — see "Fifth Moriah result" below). No temporal-test or
spatial-holdout data has been accessed. See "Current status and next step"
at the end of this document for the roadmap for the remaining five runs
and the distinction between this pilot and proper HPO; see "Fifth Moriah
result: `emb128x64_seedA` candidate complete" below for the completed
candidate's own result detail. The rest of this document describes the
pilot design as originally implemented and verified locally, plus the
sequence of orchestration corrections that preceded this result.

## Purpose

First controlled architecture/optimization pilot after the Stage 1
full-population seed-run closure and the validation-and-optimization
foundation phase (`docs/stage1_validation_optimization_foundation.md`).
Compares raw static-attribute concatenation against a learned static
embedding (three shapes) under two seeds, using the seed run's own
hyperparameters as a frozen baseline, on the certified 2,307-basin
development population.

## The six run specifications (closed set, no sweep)

| run_id | static pathway | embedding hiddens | seed |
|---|---|---|---|
| `raw_seedA` | raw identity concatenation | — | 967139 |
| `raw_seedB` | raw identity concatenation | — | 1729 |
| `emb128x64_seedA` | learned FC embedding | `[128, 64]` | 967139 |
| `emb128x64_seedB` | learned FC embedding | `[128, 64]` | 1729 |
| `emb64_seedA` | learned FC embedding | `[64]` | 967139 |
| `emb128_seedA` | learned FC embedding | `[128]` | 967139 |

Seed A (967139) is the historical full-population seed run's own
NH-assigned seed, recovered read-only from its frozen `run_dir/config.yml`
(no training triggered by the recovery). Seed B is 1729. Embedding
activation is `tanh` and dropout `0.1` for all three embedding shapes —
untuned, held fixed, not varied by this pilot (consistent with the Part I
`embedded_static_cudalstm_pilot` structural-smoke profile's own values).

Everything else is frozen identically across all six runs: lead 6h,
`seq_length` 24, CudaLSTM `hidden_size` 128, `output_dropout` 0.25,
`batch_size` 256, Adam, `lr` 0.001, no scheduler, the seed run's NSE-style
loss, the 8 approved dynamic inputs, all 473 static model-input attributes,
checkpoint every epoch, `num_workers` 4. Explicitly **not** varied by this
pilot: `seq_length`, `batch_size`, `lr`, scheduler, `hidden_size`,
`output_dropout`. No EA-LSTM, no automated W&B sweep, no test-set
evaluation.

Machine-readable source of truth: `config/stage1_lead06_pilot_v001.yaml`
(portable policy only — no machine paths, credentials, or Slurm resources).
It composes, without redefining, the existing scientific baseline
(`config/stage1_scientific_baseline_v001.yaml`), early-stopping mechanics
(`config/stage1_early_stopping_policy_v001.yaml`), screening-subset policy
(`config/stage1_screening_subset_v001.yaml`), and W&B tracking policy
(`config/stage1_wandb_tracking_policy_v001.yaml`).

## Evaluation policy

Primary metric: **median per-basin raw-space NSE** on development
validation. Mean and pooled NSE are diagnostics only. Every full evaluation
retains: NSE percentiles p1/p5/p10/p25/p50/p75/p90/p95/p99; fractions with
NSE<0, NSE>0, NSE>0.5; finite-basin/admitted-sample counts; pooled and mean
NSE; KGE, RMSE, MAE, Pearson correlation, bias, PBIAS. Raw-space evaluation
preserves inverse scaling to mm/h, basin-area conversion to m³/s, correct
lead alignment, and consistent finite obs/pred masking — implemented via
`src/baseline/nh_seed_evaluation.raw_space_metrics_for_run_period`, reused
by both the screening and full-validation interfaces rather than
reimplemented.

## Early stopping (unmodified starting policy, pilot sub-cap only)

Starting policy from the seed-run closure decisions is **not tightened or
redesigned** by this pilot: save every epoch; no stopping before epoch 6;
official screening validation every 3 epochs; epoch 3 is diagnostic only
(recorded, never triggers stopping); stopping eligibility begins epoch 6;
minimum meaningful median-NSE improvement 0.005; patience 3 official
stopping-eligible validation events; best checkpoint always retained;
restart-safe persisted state.

`config/stage1_early_stopping_policy_v001.yaml` (the general policy,
unmodified) caps `max_epoch_budget` at 40. This pilot needs a stricter
36-epoch budget without editing that committed file. Resolution:
`src/baseline/pilot_early_stopping.build_effective_policy()` loads the base
policy, validates it hasn't drifted (`metric_name`,
`higher_is_better`, `min_epoch_before_stop`, `min_delta`,
`patience_events` must match what the pilot expects — raises
`PilotEarlyStoppingError` otherwise), and layers
`max_epoch_budget = min(base_max_epoch_budget, 36)` under a renamed
`policy_name` (suffix `__pilot_subcap_36`). The general policy file itself
is untouched.

Restart safety: replaying the identical last `(epoch, metric_value)` is a
no-op; any other out-of-order or contradictory replay raises
`PilotEarlyStoppingError`. State schema (persisted per run):
`schema_version`, `policy_name`, `metric_name`, `higher_is_better`,
`history` (list of `{epoch, metric_value, is_new_best}`), `best_epoch`,
`best_metric_value`, `events_since_best_improvement`, `stopped`,
`stop_reason`, `stop_epoch`.

## Screening cadence vs. full validation

Two distinct, non-interchangeable evaluation interfaces:

- **Screening** (`src/baseline/pilot_screening_eval.py`) — runs on the
  provisional ~400-basin screening subset
  (`reports/stage1_validation_optimization_foundation_v001/part_d_screening_subset/selection_v001/screening_subset_basin_ids.txt`,
  locally regenerable, not committed — `reports/**` is gitignored by repo
  convention). Tagged `scope="screening_subset_provisional"`,
  `authoritative=False`. `classify_screening_epoch_role(epoch, policy)`
  returns `"diagnostic_only"` (epoch 3, or on-cadence but before epoch 6),
  `"stopping_eligible"` (on-cadence, epoch ≥ 6), or
  `"not_a_screening_epoch"` (off-cadence). Every screening basin ID is
  re-validated as a proper subset of the 2,307-basin development population
  at evaluation time — a spatial-holdout or otherwise sealed basin id
  injected into the screening file is rejected, not silently evaluated.
- **Full validation** (`src/baseline/pilot_full_validation.py`) — a
  readiness interface only, never invoked in this implementation against
  the real certified package or a real training run. Runs the full
  2,307-basin development population, tagged
  `scope="development_full_population_validation"`,
  `authoritative=True`, with `promoted_from_run_id` passthrough. Has no
  cadence restriction of its own (unlike screening).

The screening subset is operational and provisional per the foundation
phase's Part D decision, not scientifically authoritative — see
`docs/stage1_validation_optimization_foundation.md` and
`docs/decision_log.md` for the subset's own status.

## W&B tracking (optional, disabled by default)

`config/stage1_wandb_tracking_policy_v001.yaml` ships `enabled: false`,
`mode: disabled` — confirmed the real default, and `wandb` is confirmed
**not installed** in this development environment.
`src/baseline/pilot_tracking.py` wraps
`src/baseline/wandb_tracking.py`: when tracking is disabled, no `wandb`
import is attempted at all; when enabled but `wandb` is unavailable or
`wandb.init` fails, initialization is caught and downgraded to a null sink
with `warnings.warn` rather than raising, so tracking failures can never
block or corrupt a training run. Credential-shaped keys
(`api_key`/`secret`/`password`/`token`/`credential`-fragment matches) and
sealed-scope-shaped scientific-metric keys (`test`/`holdout`/`temporal`/
`spatial` fragments) are both rejected by `wandb_tracking.py`'s key guards
before anything is logged.

## Orchestration

`src/baseline/pilot_orchestration.run_pilot()` composes the pieces above
into a restart-safe, bounded-chunk trainer driver: `chunk_epoch_targets`
produces the 36-epoch schedule in the fixed screening/stopping cadence
(`[6, 9, 12, ..., 36]`); each chunk trains via NH's own
`start_run`/`continue_run` (through a `train_chunk_fn` seam, so tests never
need real NH/torch); screening events fire on cadence within each chunk;
early stopping is evaluated after each stopping-eligible event; an evidence
bundle (`src/baseline/pilot_evidence_bundle.py`) is written after every
call, `force=True` internally regardless of the caller's `force` argument
(which only gates NH config regeneration on resume) — so re-running
`run_pilot()` against an already-stopped run neither retrains nor fails,
it simply re-derives and rewrites the same evidence record. The evidence
bundle copies only small text artifacts (`config.yaml`,
`generation_manifest.json`, a `pilot_run_evidence.json` record, and a
`checksums.json`), never checkpoint bytes; checkpoints are inventoried by
filename/size/sha256 only. Every bundle carries a
`sealed_set_non_access_statement`.

## CLI wrapper and Slurm launcher (prepared, not submitted)

`scripts/run_stage1_lead06_pilot.py` is the local/Moriah CLI entrypoint
(any of the 6 `run_id`s, never hardcoded). `scripts/run_stage1_lead06_pilot_moriah.sbatch`
is the Slurm launcher: `--cpus-per-task=8`, `--mem=128G`,
`--gres=gpu:l4:1` (the starting-policy resource defaults, deliberately not
the seed-train script's later OOM-driven `--cpus-per-task=16`/`--mem=224G`
corrections — those were an artifact of the seed run's own resource
history and are not proven necessary here). Config/evidence output
directories default under `${FLASHNH_BASE}/runs/` and
`${FLASHNH_BASE}/evidence/`, both structurally outside the tracked
`${REPO_CLONE_DIR}`/`${REPO_WORKDIR}` clone. Never passes `--force` to the
CLI wrapper on an executable line (discussed only in an explanatory
comment). `emb128x64_seedA` has since been submitted once as the
workflow-qualification run — see the next section for the result.

## Moriah workflow-qualification run and orchestration correction

The first real Moriah job for this pilot, `emb128x64_seedA` (Slurm job
`45695059`), was submitted to qualify the orchestration end-to-end before
committing to all six runs. Training itself succeeded: NH trained through
epoch 6, writing checkpoints and optimizer states 1-6 intact, peak RSS
~96.4GB. Orchestration then failed post-training with
`NHSeedEvaluationError: missing validation results pickle` at
`validation/model_epoch003/validation_results.p`.

**Root cause.** The original implementation assumed NH's in-training
`validate_every: 3` validation always persists
`validation/model_epochNNN/validation_results.p` to disk. It does not
reliably do so. This was confirmed with a separate, explicit
evaluation-only job (`45698612`) that ran NH's own `start_evaluation` for
epochs 3 and 6 against the same checkpoints and successfully produced both
result pickles (400 basins each, ~84.6MB each; job completed in 11:34
elapsed with ~1.96GB peak RSS on an L40S). This resource observation is
from one evaluation-only job on one checkpoint pair — it is not a
validated general resource requirement for evaluation and should not be
extrapolated to, e.g., the full 2,307-basin validation or other lead
times/seeds.

**Correction (this implementation).** `pilot_orchestration.py` no longer
assumes the pickle exists. Before every screening checkpoint's evaluation,
it now calls `ensure_validation_results(nh_run_dir, epoch,
evaluate_checkpoint_fn=...)`:

1. Checks the canonical path (`nh_seed_evaluation.period_results_path`,
   the single helper every caller uses — never independently
   reconstructed) for an existing `validation_results.p`.
2. If present, reuses it unchanged — no re-evaluation.
3. If absent, invokes an explicit NH evaluation for that exact
   `(nh_run_dir, epoch, period="validation")` via an injectable
   `evaluate_checkpoint_fn` (production default `default_evaluate_checkpoint`,
   which mirrors `scripts/run_stage1_nh.py`'s own `eval` subcommand:
   load `config.yml`, `setup_logging`, `start_evaluation`).
4. Re-checks the path afterward; if still absent, raises
   `PilotOrchestrationError` with no partial state persisted, so a retry is
   always safe.
5. Only then does `evaluate_screening_checkpoint()` run (unchanged — it
   remains a pure metric reader, never itself triggering evaluation; see
   its docstring) and early-stopping state update.

Resume semantics are preserved: on a re-run of `run_pilot()`/`run_pilot_chunk()`
against a run directory with checkpoints and saved result pickles already on
disk through epoch 6, orchestration does not retrain those epochs and does
not re-invoke evaluation for epochs whose pickle already exists; it resumes
training at the next chunk target (epoch 9 in this failure shape) and only
evaluates newly-reached screening epochs going forward. Epoch 3's
diagnostic-only classification and epoch 6's stopping-eligibility are
unaffected — this was purely a missing-prerequisite bug, not a change to
scheduling or stopping policy.

**Current status (superseded by the second correction below).** The pilot
run remained paused after epoch 6 pending a resumed Moriah job with this
first correction. That resume was submitted and exposed a second,
independent orchestration bug — see the next section for the current,
authoritative status.

## Second Moriah failure and continuation-nesting/epoch-semantics correction

The resumed job (`45705457`), continuing `emb128x64_seedA` from epoch 6
toward the epoch 9 chunk boundary, exposed a second, independent
orchestration bug.

**Root cause.** `continue_run` is not a bare "train N more epochs from a
checkpoint" call: NeuralHydrology sets `is_continue_training=True`
unconditionally on every call, and `BaseTrainer._create_folder_structure`
therefore *always* nests output into a new
`continue_training_from_epoch{start_epoch:03d}/` subdirectory under the run
directory it is given (raising if that exact directory already exists). The
original chunk-continuation code treated the overlay's `epochs:` key as an
absolute target epoch rather than an additive epoch count relative to the
checkpoint being resumed from. For the epoch 6→9 chunk this meant NH was
told `epochs: 9` while resuming from checkpoint 6, so it trained 9 *more*
epochs past epoch 6 (an additive 6+9=15) rather than 3 — producing
`base_run/continue_training_from_epoch006/model_epoch007.pt` through
`model_epoch015.pt`, with no epoch-9 screening validation ever run (epoch 9
was never a chunk boundary NH actually stopped at). Compounding this, no
code path resolved which physical directory owned a given checkpoint epoch
across the base directory and nested continuation directories, so
evaluation/tracking calls had no reliable way to locate epoch 9 even had it
existed correctly.

**Correction (this implementation), entirely within
`src/baseline/pilot_orchestration.py`:**

1. `TrainChunkRequest` now carries three separate, non-overloaded fields:
   `current_epoch` (the checkpoint epoch training resumes from — `None`
   only in the fully-degenerate zero-checkpoint corner),
   `additional_epochs` (the additive epoch count for this chunk, e.g. `3`
   for the 6→9 chunk), and `logical_target_epoch`
   (`current_epoch + additional_epochs`). `default_train_chunk` writes
   `epochs: additional_epochs` (plus `continue_from_epoch` when resuming) to
   the per-chunk overlay, matching NH's real additive semantics.
2. `discover_physical_checkpoints(base_run_dir)` recursively inventories
   every `model_epoch###.pt` across the base run directory and
   arbitrarily-nested `continue_training_from_epoch###/` subdirectories,
   raising `PilotOrchestrationError` ("ambiguous physical checkpoint
   inventory") if the same epoch is ever claimed by two different physical
   directories, and ignoring malformed filenames/directory names rather
   than guessing.
3. `resolve_trusted_chunk_checkpoint` / `untrusted_overshoot_epochs`
   distinguish a checkpoint this pilot's own chunk sequence produced
   (physical directory exactly matches the continuation directory NH would
   create for this exact `(previous_target_epoch, checkpoint_epoch)` pair)
   from a checkpoint that merely happens to exist on disk at the right
   epoch number but was produced under different, untrusted circumstances
   (e.g. the real `continue_training_from_epoch006/model_epoch010.pt`-
   `model_epoch015.pt` overshoot from job 45705457's bug). The pilot never
   silently trusts the latter.
4. `_advance_chunk_via_continuation` is the single shared helper for every
   non-first chunk: it resolves a trusted checkpoint if one already exists
   (idempotent no-retrain resume); else checks for untrusted checkpoints
   already occupying the target epoch range and blocks with a "manual
   review... required" reason rather than retraining over them; else checks
   whether NH's target continuation directory already exists but is
   empty/incomplete and blocks with an "already exists" reason (since
   `continue_run` would otherwise crash inside real NH trying to recreate
   it) — only then does it invoke `train_chunk_fn`.
5. `compute_pilot_status_fields(nh_run_dir, pilot_policy)` reports four
   distinct fields rather than a single conflated "current epoch":
   `highest_physical_checkpoint_epoch` (every checkpoint that exists
   anywhere on disk, trusted or not — 15 for the real job-45705457
   evidence), `highest_screened_epoch` (the highest epoch with a valid,
   trusted screening/stopping event — 6), `next_intended_screening_epoch`
   (9, per the fixed cadence — never simply "highest + 3"), and
   `overshoot_epochs` (`[10, 11, ..., 15]`), plus
   `safe_to_continue_automatically` (`False` whenever untrusted overshoot
   checkpoints exist). The Slurm launcher and evidence bundle both
   report/consume these same four fields rather than re-deriving their own
   notion of "current epoch".
6. Applied to the exact real evidence from job 45705457 (checkpoints 1-6
   flat, `continue_training_from_epoch006/model_epoch007.pt`-
   `model_epoch015.pt`, no valid epoch-9 screening result): the corrected
   orchestration trusts and screens exactly epoch 9 (never touching 10-15),
   and any further chunk attempting to continue past epoch 9 halts with a
   blocked status and manual-review reason rather than resuming from the
   wrong checkpoint or silently discarding the untrusted 10-15 checkpoints.

**Current status.** The pilot run is **not complete**. `emb128x64_seedA`
remains paused after epoch 6, now with both orchestration bugs corrected
locally. The current run is **safe for one controlled recovery
invocation**: `continue_training_from_epoch006/model_epoch009.pt` sits in
exactly the directory this pilot's own chunk sequence would produce, so it
is trusted — the corrected orchestration reuses it, runs validation
screening for epoch 9, and records that screening event without
retraining and without requiring any manual movement, archiving, or
deletion of checkpoints first. The run is **not safe to continue training
beyond epoch 9** while checkpoints 10-15 remain in the existing
continuation layout: those are preserved, untouched, scientifically-unused
artifacts, and `overshoot_epochs`/`safe_to_continue_automatically=False`
cause any attempt at a further 9→12 chunk to block rather than retrain
over or past them — a later decision is required before that continuation
is attempted. This epoch-9 recovery HAS since been executed on Moriah
(job 45718473) and confirmed scientifically correct — see the next section
for the result and a follow-on launcher defect it exposed. No scientific
hyperparameter, split, screening-membership, or early-stopping policy value
changed as part of either correction.

## Third Moriah result: real epoch-9 recovery confirmed, launcher status-propagation defect fixed

Recovery job `45718473` (partition `catfish`, one L4 GPU, elapsed 00:08:12,
Slurm `COMPLETED`, exit `0:0`) ran the continuation-nesting/additive-epoch
fix above against the real `emb128x64_seedA` artifact. The scientific
recovery was correct exactly as predicted above: no training occurred, the
existing `continue_training_from_epoch006/model_epoch009.pt` checkpoint was
reused, epoch 9 was screened and logged exactly once (median per-basin
raw-space NSE `0.18124855313577198`), epoch 6 remains best
(`0.20454161610527344`), and overshoot checkpoints 10-15 remain preserved
and unused.

However the launcher reported an internally inconsistent result:
`status: COMPLETED`, `pilot_final_status: null`, `blocked_reason: null`,
alongside a correctly computed `safe_to_continue_automatically: false` and
`overshoot_epochs: [10, 11, 12, 13, 14, 15]`. Root cause: the pilot CLI's
primary stdout JSON was unavailable when the launcher read it, so the
launcher's documented on-disk fallback (`compute_pilot_status_fields`)
engaged and correctly restored `overshoot_epochs`/
`safe_to_continue_automatically`, but never derived
`pilot_final_status`/`blocked_reason` — so the launcher's classification,
which only branched on `pilot_final_status`, fell through to `COMPLETED`.
`pilot_orchestration.run_pilot()` itself was confirmed (by direct reading
and a new end-to-end test) to already propagate a blocked chunk's
`final_status`/`blocked_reason` correctly; the loss was isolated to the
launcher's fallback classification. Fixed locally: the launcher's fallback
now also derives `pilot_final_status`/`blocked_reason` from
`safe_to_continue_automatically`/`overshoot_epochs` when the primary status
is unavailable, and the CLI now exits `1` (reusing the launcher's own
existing "needs a human" convention) for a blocked `final_status` instead of
always exiting 0. No training, evaluation, metric, stopping, checkpoint, or
overshoot logic changed. **No further Moriah job should run until this
local status-propagation fix is committed.** Full detail:
`docs/decision_log.md`'s 2026-07-30 status-propagation entry.

## Fourth Moriah result: launcher classification confirmed, rerun-idempotency defect found and fixed

Verification job `45718742` (partition `catfish`, source commit
`7c6b02a599b885682a97081a3f166d97097bd4ec`, elapsed `00:03:17`, no stderr)
confirmed the launcher status-propagation fix above: the launcher
correctly classified the final on-disk state as
`BLOCKED_MANUAL_REVIEW_REQUIRED` (`pilot_final_status:
blocked_continuation_overshoot_conflict`,
`safe_to_continue_automatically: false`, overshoot epochs 10-15, exit
code 1). **No training occurred and scientific state was not modified.**

But before reaching that clean overshoot block, the Python pilot process
crashed: `PilotEarlyStoppingError: epoch 6 is not after the last recorded
epoch 9 -- out of order`, from `run_pilot() -> run_pilot_chunk() ->
record_screening_event(epoch=6)`. Root cause: `run_pilot()` always
restarts its chunk walk from epoch 6 on every call, and
`run_pilot_chunk()`'s screening loop had no check for a screening epoch
already present in this run's persisted `pilot_orchestration_state.json`
(`logged_screening_epochs: [3, 6, 9]`), so it re-fed already-screened
epoch 6 into the early-stopping state machine after the persisted
`pilot_early_stopping_state.json` history's last entry had already
advanced to epoch 9. Fixed locally in `src/baseline/pilot_orchestration.py`:
a screening epoch already present in `logged_screening_epochs` is now
skipped outright (no re-evaluation, no re-record) instead of always
being re-processed, with a light consistency check (not broad
reconciliation) against the reloaded early-stopping history so genuinely
inconsistent state is never silently skipped. No checkpoint-discovery,
continuation, evaluation, early-stopping-policy, or launcher-
classification logic changed. **No further Moriah job should run until
this local rerun-idempotency fix is committed.** Full detail:
`docs/decision_log.md`'s 2026-07-30 rerun-idempotency entry.

## Fifth Moriah result: `emb128x64_seedA` candidate complete — continuation adopted, epochs 12 and 15 screened, early stopping fired

Job `45722908` (partition `catfish`, source commit
`af8945d04451d7699ab54b13082eaf870f04f28e`, elapsed `00:10:34`, Slurm state
`COMPLETED`, exit code `0:0`) used the production
`pilot_accepted_continuation.json` manifest (real SHA-256 hashes for the
epoch-12 and epoch-15 model+optimizer checkpoints, filename-bound to their
own key epoch per the trust-binding correction above) to adopt the existing
epoch 6→15 continuation trajectory without any retraining. **No training
occurred**; epochs 12 and 15 were evaluated sequentially from the
already-existing checkpoints.

Final screening history: epoch 3 (diagnostic only, before
`min_epoch_before_stop`); epoch 6 (median per-basin raw-space NSE
`0.20454161610527344`, new best); epoch 9 (`0.18124855313577198`, no
improvement); epoch 12 (`0.1993193615763258`, no improvement under the
`min_delta` threshold); epoch 15 (`0.17125263282608943`, no improvement).

Final early-stopping state: best epoch `6`, best metric
`0.20454161610527344`, `events_since_best_improvement = 3`, `stopped =
true`, stop reason `patience_exhausted`, stop epoch `15`.

Final orchestration state: `logged_screening_epochs = [3, 6, 9, 12, 15]`;
highest physical checkpoint epoch 15; highest screened epoch 15; no
overshoot epochs remain unresolved; no further screening epoch is intended
for this run. No further training is authorized or required for
`emb128x64_seedA`.

**Epoch 6 is the selected checkpoint for this one candidate configuration.**
It is not a claim about the final Stage 1 production model — that
determination requires comparing results across all six run specifications
in the wider optimization campaign (see "The six run specifications" above
and "What has not been done" below).

**Sealed populations untouched.** This screening run used only the
development/screening-subset population, exactly as in every prior
screening event for this pilot. No temporal-test or spatial-holdout data
was accessed.

**This closes the continuation-repair/adoption sequence** that began with
the additive-epoch overshoot bug (see "Second Moriah failure..." above):
provenance review of the real epoch 7-15 checkpoints, the explicit
run-specific adoption manifest, the manifest's trust-binding correction,
and now real Moriah adoption and screening, all without ever retraining
past the original, uninterrupted epoch 6→15 continuation. The next phase
is the wider optimization campaign: the other five run specifications,
screened and stopped under this same frozen protocol. Full detail:
`docs/decision_log.md`'s 2026-07-30 closure entry.

## Implementation modules

| Module | Task |
|---|---|
| `src/baseline/pilot_lead06_config.py` | six-run matrix, seed resolution, bundle construction, sealed-population rejection |
| `src/baseline/nh_structural_preflight.py` (extended) | structural preflight for the six pilot profiles |
| `src/baseline/pilot_screening_eval.py` | screening-subset evaluation + epoch-role classification |
| `src/baseline/pilot_early_stopping.py` | pilot sub-cap layering + restart-safe stopping state |
| `src/baseline/pilot_tracking.py` | W&B identity/hyperparameter/metric logging wrapper |
| `src/baseline/pilot_full_validation.py` | full-population validation readiness interface |
| `src/baseline/pilot_evidence_bundle.py` | compact, checkpoint-byte-free evidence bundle writer |
| `src/baseline/pilot_orchestration.py` | bounded-chunk training/screening/stopping driver |
| `scripts/run_stage1_lead06_pilot.py` | CLI wrapper |
| `scripts/run_stage1_lead06_pilot_moriah.sbatch` | Slurm launcher (not submitted) |
| `src/baseline/nh_config_generation.py` (extended) | six named run profiles (`pilot_lead06_*_v001`) |

## Tests

Eight focused pytest files (`tests/test_pilot_*.py`, plus a shared,
non-collected fixture helper `tests/_pilot_support.py`), 125 tests, all
passing (was 95 before the evaluation-prerequisite correction below, 124
after it, 125 after this correction's added logging-handler-guard test).
Coverage includes: exact six-run matrix acceptance and rejection
of any deviation (missing run, extra run, wrong profile mapping, duplicate
run_id, equal seeds); Seed A/B resolution per run; embedding shape per run
including exact-`None` (raw, identity-pathway) vs. exact-shape (embedded)
assertions — the direct regression check against a silent identity
fallback going undetected; structural rejection of malformed
`statics_embedding` specs (`tests/test_nh_config_generation.py`, extended
by an earlier milestone, re-verified here as still passing); screening
membership/count/provenance and sealed-population rejection; epoch-3
diagnostic-only vs. epoch-6 stopping-eligible behavior; patience/threshold
behavior including a real patience-exhaustion stop; restart-state
persistence, reload, and rejection of out-of-order or contradictory resume
replay (idempotent replay of the last event is accepted); W&B
disabled/enabled-but-missing/failure-non-fatal behavior; generated evidence
and config paths outside the tracked repo clone; the full-validation
readiness interface's distinct population role and lack of screening-style
cadence restriction; no test/holdout access anywhere in the fixtures or
logged keys; deterministic, idempotent evidence-bundle regeneration on
resume with no retraining.

Added for the evaluation-prerequisite correction (`tests/test_pilot_orchestration.py`):
missing result triggers an explicit evaluation call; an existing result is
reused with zero evaluator calls; a resume reproducing the exact real
qualification-run failure shape (checkpoints + pickles through epoch 6
pre-existing) retrains nothing, re-evaluates nothing already saved, and
proceeds training to epoch 9; a future screening checkpoint triggers
explicit evaluation before screening runs; an evaluator that fails to
produce the expected pickle raises loudly with no false screening/stopping
event and checkpoints untouched; epoch-3 diagnostic-only behavior is
unaffected; existing orchestration idempotency is unaffected. The fake
training callback used in tests writes checkpoint bytes only — it no
longer also fabricates validation-result pickles — so these tests exercise
the real missing-prerequisite path rather than one where a result is
always already present.

Added for the continuation-nesting/additive-epoch correction
(`tests/test_pilot_orchestration.py`, plus 6 new low-level unit tests): the
fake training callback was rewritten to reproduce NH's real nested
`continue_training_from_epoch###/` layout instead of writing checkpoints
flat, so these tests exercise the real directory-nesting behavior rather
than a simplified one. Coverage now includes: additive- (not absolute-)
epoch computation verified across two successive chunk transitions
(6→9→12); checkpoint discovery across the base directory plus one, and
then a doubly-nested, continuation directory; a loud
`PilotOrchestrationError` ("ambiguous physical checkpoint inventory") on a
duplicate physical epoch claim; malformed checkpoint filenames and
continuation-directory names are ignored rather than guessed at; a resumed
chunk that already has a trusted epoch-9 checkpoint is idempotent (zero
train/evaluate calls, correct nested `checkpoint_dir_for_target`); a chunk
blocked by untrusted checkpoints already occupying the target range
reports a "manual review... required" reason and leaves those checkpoints
and all prior logical state untouched, including across a repeated call; a
chunk blocked by an empty pre-existing (but NH-incompatible) continuation
directory reports an "already exists" reason; screening/tracking never
touches epochs 10-15 merely because they physically exist; an evaluator
failure leaves the prior logical (early-stopping) state completely
unchanged; a checkpoint reference is logged at its resolved physical path,
never a copied/base-relative one; and the exact real job-45705457 evidence
shape (checkpoints 1-6 flat, 7-15 in one nested continuation directory,
no valid epoch-9 result) is reproduced end-to-end, confirming the
corrected orchestration trusts and screens exactly epoch 9, a further
6→9→12 chunk attempt blocks rather than resuming from the wrong checkpoint,
and `compute_pilot_status_fields` reports
`highest_physical_checkpoint_epoch=15`, `highest_screened_epoch=9`,
`overshoot_epochs=[10..15]`, `safe_to_continue_automatically=False`.
`tests/_pilot_support.py` gained a `short_tmp_path` fixture (Windows-only
short-rooted temp directory) so these now-realistically-deep nested paths
do not exceed Windows' 260-character `MAX_PATH` in the local test
environment — a Windows-only local-testing accommodation, not a behavior
change; Linux (where real Moriah/h2o runs happen) has no such limit.

An adversarial self-review of this correction (cross-checked directly
against the real job-45705457 evidence file rather than relying on
paraphrase) additionally found that `default_train_chunk` — the exact
function that writes `pilot_epoch_overlay.yaml`, and therefore the function
directly responsible for both bugs in this document — had no direct test
coverage of its own, since every test here injects a fake `train_chunk_fn`.
Its overlay-dict construction was extracted into a pure,
NH/torch-free helper, `_continuation_overlay(request) -> dict` (a
same-behavior refactor), and given two direct unit tests covering the
explicit-`continue_from_epoch` case and the `current_epoch=None`
degenerate-corner case.

Current focused-suite total: 146 tests across the same eight
`tests/test_pilot_*.py` files (was 125 before this correction), all
passing.

Full pre-existing repository test suite re-run alongside these: 1173
passed (excluding 6 pre-existing collection errors in files that import
`neuralhydrology`/`torch`, not installed in this local environment by
design — those tests only run on h2o/Moriah); 1 test
(`test_package_builder.py::test_evidence_promotion_failure_after_package_success_rolls_back_both`)
failed only in the full-suite run with a Windows file-lock
`PermissionError` during atomic promotion under load, and passed cleanly in
isolation (file untouched by this work) — pre-existing flakiness unrelated
to any file touched here. Zero regressions attributable to this work.

## Explicit continuation adoption: `pilot_accepted_continuation.json` (2026-07-30)

A human review of the real `emb128x64_seedA` continuation evidence (job
45705457, confirmed by recovery/verification jobs 45718473/45718742/45721557)
judged epochs 7-15 **conditionally safe to adopt** as one valid, uninterrupted
continuation of the trusted epoch-6 checkpoint (full provenance reasoning:
`docs/decision_log.md`'s 2026-07-30 entry). The untrusted-overshoot guard
above (`resolve_trusted_chunk_checkpoint` / `untrusted_overshoot_epochs`) is
permanent and unconditional for every run — it never reinterprets a
checkpoint as trustworthy on its own. Adopting a specific pre-existing
overshoot checkpoint is only ever possible through an explicit, per-run
manifest:

- **File**: `pilot_accepted_continuation.json`, in the base NH run directory
  next to `pilot_early_stopping_state.json` — **never committed to git, never
  a general CLI override flag.** Strictly opt-in: a run without this file
  behaves exactly as documented above, with no change.
- **Schema** (`schema_version: 1`):
  ```json
  {
    "schema_version": 1,
    "run_id": "raw_seedA",
    "decision": "conditional_sequential_adoption_epoch6_to_15",
    "accepted_directory": "continue_training_from_epoch006",
    "accepted_checkpoints": {
      "12": {
        "model_path": "continue_training_from_epoch006/model_epoch012.pt",
        "model_sha256": "<sha256>",
        "optimizer_path": "continue_training_from_epoch006/optimizer_state_epoch012.pt",
        "optimizer_sha256": "<sha256>"
      },
      "15": { "...": "same shape, epoch 15" }
    },
    "provenance_basis": "job 45705457 continuation evidence, reviewed 2026-07-30"
  }
  ```
- **Validation** (`load_accepted_continuation_manifest`): `run_id` must
  exactly match the current run (else `PilotOrchestrationError`); every
  entry's `model_path`/`optimizer_path` must resolve strictly inside the run
  directory and inside the manifest's own `accepted_directory` (else raises —
  no absolute paths, no `..` escapes). SHA-256 hashes are **not** verified
  eagerly at load time — only lazily, per epoch, the moment
  `_advance_chunk_via_continuation` actually consults that epoch's entry
  (`_resolve_accepted_checkpoint`), so a bad/premature epoch-15 entry can
  never block a correctly-hashed epoch-12 adoption.
- **Sequencing**: an entry is consulted only when its epoch is the exact
  `chunk_target_epoch` a chunk call is already resolving. Epoch 12 must
  resolve before epoch 15 is ever looked at — enforced by `run_pilot()`'s
  existing chunk-by-chunk loop (no new dedicated sequencing code), which also
  already breaks out once a chunk reports `stopped`. If early stopping fires
  at epoch 12, epoch 15 stays physically present but is never scientifically
  consulted again.
- **Effect when consulted**: adopts the existing physical checkpoint
  directory as the trusted checkpoint for that epoch (no training call); the
  epoch is still evaluated/screened through the normal pipeline. Idempotent
  on rerun via the existing `logged_screening_epochs` mechanism.

**Real hashes filled in and used (superseded — see "Fifth Moriah result"
below).** The production manifest for `emb128x64_seedA` was authored with
real SHA-256 hashes for the epoch-12 and epoch-15 model+optimizer
checkpoints and successfully used, without retraining, in job `45722908` to
adopt and screen both epochs. 10 focused tests
(`test_no_manifest_preserves_block`, `test_correct_manifest_trusts_epoch_12`,
`test_epoch_12_evaluated_without_training`,
`test_epoch_15_untouched_during_epoch_12_step`,
`test_incorrect_model_hash_rejected`, `test_incorrect_optimizer_hash_rejected`,
`test_wrong_run_id_or_path_rejected`, `test_epoch_15_used_only_if_still_required`,
`test_stopping_at_12_leaves_15_unused`, `test_rerun_idempotency_with_accepted_manifest`),
plus one further trust-binding test
(`test_epoch_12_entry_pointing_to_epoch_15_files_rejected`), cover this
mechanism in `tests/test_pilot_orchestration.py` (45 passed total).

## What has not been done

`emb128x64_seedA` **is complete**: epoch 6→15 was adopted via the manifest
and screened in job `45722908`, early stopping fired at epoch 15, and epoch
6 is the selected checkpoint for this one candidate configuration (see
"Fifth Moriah result" above) — it is not a claim about the final Stage 1
production model. The other five run specifications have not been
submitted or started; screening and stopping them, under this same frozen
protocol, is the next phase (the wider optimization campaign). No
full-population evaluation. No temporal-test or spatial-holdout access. No
change to the certified Compact Scientific Package or canonical split
membership. No regeneration of the screening subset. No hydrograph atlas
generation. No automated sweep. No EA-LSTM work. The continuation-nesting
and launcher-classification corrections in this document have now been
verified against real Moriah runs (jobs `45718473`, `45718742`, and
`45722908`); one residual risk remains open and unresolved locally: the
module docstring's claim that
`continue_from_epoch` is a real, recognized NH `Config` property is not
independently verified in this codebase and cannot be checked here since
`neuralhydrology` is not installed locally — see `docs/decision_log.md`'s
2026-07-30 entry for the full analysis and why it is judged low-severity
in the pilot's own exercised code path. Nothing generated by this pilot
(configs, runtime outputs, logs, checkpoints, W&B files, evidence
directories) has been committed.

## Current status and next step (2026-08-02)

Documentation-only update, recording the roadmap decided after
`emb128x64_seedA`'s completion. Full decision text: `docs/decision_log.md`,
2026-08-02 entry; broader framing: `docs/stage1_validation_optimization_foundation.md`
Part L.

**Completed.** `emb128x64_seedA` is complete (see "Fifth Moriah result"
above): epoch 6 median raw-space screening NSE `0.20454161610527344`
(best), epoch 9 `0.18124855313577198`, epoch 12 `0.1993193615763258`,
epoch 15 `0.17125263282608943`; stopped at epoch 15 (`patience_exhausted`);
epoch 6 is the selected checkpoint for this one candidate configuration
only. This is not reopened by this section.

**Preferred next candidate (adopted direction, not a launch
authorization).** `raw_seedA` — same seed (967139) as `emb128x64_seedA`,
raw identity-concatenation static pathway instead of the learned `[128,
64]` FC embedding. It gives the cleanest same-seed contrast against the
completed candidate and directly answers this pilot's central question
(does the learned embedding beat raw concatenation at all, holding seed
fixed). Nothing in this section submits or launches that run.

**Conditional review of the remaining matrix.** The other four runs
(`emb128x64_seedB`, `emb64_seedA`, `emb128_seedA`, `raw_seedB`) remain
part of the closed six-run design (see the table above) but are not
committed to automatic sequential or simultaneous launch. Results are
reviewed between candidates, and any remaining run may be deprioritized if
its scientific value becomes redundant given earlier results (for example,
if `raw_seedA` alone already gives an unambiguous, large-margin answer to
the raw-vs-embedded question relative to plausible seed noise). Parallel
execution of `raw_seedA` and `emb128x64_seedB` remains an available
operational option — nothing in the design prevents it structurally — but
is not the current preferred workflow, which retains review between
high-information runs before deciding the next one.

**This pilot vs. proper HPO (binding distinction).** This six-run pilot is
Stage A: a structural contrast only, answering raw-vs-embedded static
pathway, approximate embedding shape, and limited two-seed robustness,
with every other hyperparameter (`seq_length`, `hidden_size`,
`output_dropout`, `batch_size`, learning rate, embedding
dropout/activation, scheduler) frozen identically across all six runs by
design. **It must not be described as a hyperparameter sweep.** Proper HPO
(Stage B) is a separate, later phase that begins only once Stage A yields
enough structural evidence to select or narrow the architecture family; its
search space, optimizer, trial budget, multi-fidelity policy, and
promotion rules are not designed or frozen here — see
`docs/stage1_validation_optimization_foundation.md` Part L for the current
roadmap-level framing (including the preferred `max_updates_per_epoch`
multi-fidelity direction and its provisional, non-binding fidelity
fractions, none of which apply to this Stage A pilot).
