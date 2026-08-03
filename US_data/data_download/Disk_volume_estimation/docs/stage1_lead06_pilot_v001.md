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

**Qualification status.** Two failure-mode gaps were confirmed and fixed:
(1) every backend call after `wandb.init` (metric logging,
checkpoint-reference logging, finish) is now wrapped so a real backend
exception is caught, warned once per operation, and recorded on the
`TrackingRun` as `degraded`/`degraded_operations`, never raised into
training/screening/early-stopping/checkpoint-selection code; (2) a
candidate that restarts across multiple bounded Slurm continuations now
keeps one stable W&B run identity — `derive_pilot_wandb_run_id`/
`resolve_pilot_wandb_run_id` compute a deterministic id from
`(pilot_policy_name, run_id, tracking_generation)`, passed to
`wandb.init(id=..., resume="allow")`, cross-checked against a small
persisted record in the NH run directory so a misidentified/reused run
directory raises a `TrackingError` instead of silently merging two
candidates' histories (`tracking_generation`, default `"g1"`, additionally
disambiguates a deliberate restart-from-scratch under the same `run_id`
from a genuine first attempt). This contract was first exercised entirely
offline, in-process, against a fake `wandb` module — never the real
package, never a network call — across `tests/test_wandb_tracking.py`,
`tests/test_pilot_tracking.py`, `tests/test_pilot_orchestration.py`, and a
dedicated `tests/test_wandb_offline_qualification.py` (15 numbered
scenarios, 140 tests total across the four files as of this update). **The
wrapper's contract has since also been exercised against the real,
installed wandb 0.28.1 package in offline mode** (two independent OS
processes reusing one stable run id, no network, no API key) via
`scripts/wandb_real_offline_qualification_smoke.py`; see
`docs/stage1_validation_optimization_foundation.md` Part L.4 for the full
record, scope limits, and one corrected assumption (offline `resume=
"allow"` does not locally continue a prior run directory — each invocation
gets a fresh local directory; reconciliation is server-side, at `wandb
sync` time only). **Online tracking remains not yet qualified** (`mode:
online` is implemented but never exercised against a live network
connection) and **sweeps remain deferred** to Stage B. The shipped default
is unchanged — `enabled: false` / `mode: disabled` — so none of this
qualification work turned tracking on for any real candidate. For what W&B
does and does not control, and how to read a tracked run once tracking is
enabled, see the project-specific `docs/stage1_wandb_user_guide.md`.

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

## Sixth Moriah result: `raw_seedA` tracking/screening-persistence failure and local repair (2026-08-02/03)

Job `45731908` (source commit `771e2bd1984f3b90a2a3d30c1b07069d6d1198df`,
candidate `raw_seedA`) trained epochs 1-6 successfully — checkpoints
`model_epoch001.pt`-`model_epoch006.pt` and matching optimizer states 1-6
exist and are usable, and epoch 3's `validation_results.p` (diagnostic-only
screening) was saved — but the pilot process died before epoch 6 finished
screening. **No epoch-6 validation result exists.** The real offline W&B
run directory for `flashnh-stage1_lead06_pilot_v001-raw_seedA-g1` exists but
was never cleanly finished.

**Root cause, part one.** `pilot_orchestration.py` called
`log_pilot_checkpoint_reference()` for epoch 6, which was routed through
`wandb_tracking.log_artifact_reference()` — a function designed for small,
generic artifacts and gated by `max_artifact_reference_bytes` (the
committed policy value is `1,048,576` bytes). The real checkpoint file is
~1.25 MB, so `log_artifact_reference()` raised `TrackingError`. That
exception was uncaught at the call site and propagated out of
`run_pilot_chunk()`, killing the pilot mid-screening — a direct violation
of this pilot's own design contract that W&B is optional telemetry and can
never stop training, validation, screening, early stopping, checkpoint
selection, or evidence generation (see "W&B tracking" above).

**Root cause, part two.** Even setting the crash aside,
`pilot_orchestration_state.json` (`logged_screening_epochs`) was only
persisted once, after the *entire* per-chunk screening loop finished — not
per epoch as each screening event was recorded. So when the crash hit
during epoch 6's telemetry call, epoch 3's already-complete, already-saved
screening result was not yet durably reflected in orchestration state
either.

**Local repair (no Moriah access, no Slurm submission, `raw_seedA` not
continued).**
1. A new `wandb_tracking.log_checkpoint_reference()` records a compact
   checkpoint reference (epoch, path, checksum, size, `checkpoint_type`) —
   never the checkpoint's own bytes — and, unlike
   `log_artifact_reference()`, never applies the "compact artifact" size
   ceiling to it (checkpoints are always large by nature) and never raises
   on any failure; any failure degrades tracking (`TrackingRun.degraded`
   plus a named entry in `degraded_operations`) instead of propagating.
   `pilot_tracking.log_pilot_checkpoint_reference()` now routes through it
   instead of the generic, size-gated function.
2. `run_pilot_chunk()` now persists `pilot_orchestration_state.json`
   immediately after each epoch's screening/early-stopping processing,
   before that epoch's W&B calls — not once after the whole chunk loop —
   so a later telemetry failure can no longer leave an already-processed
   epoch's orchestration state stale.
3. `run_pilot()` gained an optional `max_target_epoch` parameter (default
   `None`, no behavior change for ordinary callers). Passed as `6`, it
   bounds one call to the epoch-6 chunk target only, so a recovery
   invocation can reuse checkpoints 1-6, reuse the existing epoch-3 result,
   evaluate and record epoch 6, and then stop — `final_status` reports
   `"paused_at_max_target_epoch"` — without automatically training or
   screening epoch 9 within the same call. This exists solely so a human
   can review epoch 6 before any further training; ordinary
   idempotent/restart-safe orchestration alone cannot stop at a specific
   epoch mid-budget, since `run_pilot()`'s chunk walk otherwise proceeds
   through every remaining chunk target automatically. It is unrelated to,
   and does not implement, the separate (not-yet-implemented)
   `max_updates_per_epoch` per-epoch minibatch cap discussed elsewhere in
   this project.

Focused tests reproducing the real failure shape and the bounded-recovery
behavior were added to `tests/test_wandb_tracking.py`,
`tests/test_pilot_tracking.py`, and `tests/test_pilot_orchestration.py`
(see "Tests" below).

**Current state of `raw_seedA`.** Epochs 1-6 are trained and usable;
epoch 3 is validly screened (diagnostic-only); **epoch 6 is not yet validly
screened** (no saved validation result, no persisted screening/
early-stopping state for it) — `raw_seedA` is not yet a completed or
selectable candidate. Nothing has been retrained, re-evaluated, or deleted
on Moriah. The fix above has not been run against the real Moriah run
directory — see "Current status and next step" below for the prepared (not
executed) recovery sequence.

## Seventh Moriah result: `raw_seedA` bounded recovery run, second launcher-summary defect found and fixed (2026-08-03)

Job `45734071` (`raw_seedA`, `--max-target-epoch 6`, the recovery sequence
prepared in the Sixth Moriah result above) ran against the real epoch-1-6
artifact. `pilot_run_evidence.json` (the authoritative evidence bundle) and
the pilot CLI's own printed JSON both correctly recorded
`final_status: "paused_at_max_target_epoch"` — no retraining occurred, the
existing checkpoints were reused, and the process paused deliberately after
epoch 6 exactly as `max_target_epoch` is designed to do. **`raw_seedA` is
still only paused at epoch 6; it is not complete, and epoch 9 continuation
has not been attempted.**

However the launcher's own `pilot_result.json` again reported an
inconsistent summary — this time losing `pilot_final_status`,
`wandb_policy_sha256`, and every other authoritative field to `null`
outright, rather than the Third Moriah result's narrower fallback-derivation
gap. Root cause: the launcher's classification step ran `json.load()` on
the pilot subprocess's entire captured stdout
(`pilot_stdout.json.log`), but that stream also carries NeuralHydrology's
own log/progress-bar text ahead of the CLI's single final printed JSON
line for every ordinary or bounded-recovery invocation (only a
`--prepare-only` invocation's stdout is pure JSON, since that path never
imports `neuralhydrology`/`torch`). The resulting `JSONDecodeError` was
caught and silently reset to an empty mapping, discarding every
scientifically meaningful field even though the authoritative sources were
correct all along.

**Local fix (no Moriah access, no Slurm submission, `raw_seedA` not
continued).** `scripts/run_stage1_lead06_pilot_moriah.sbatch`'s
classification step no longer parses subprocess stdout at all. It instead
reads the same authoritative file `run_pilot()`/`prepare_pilot_run_only()`
already write under `--evidence-out-dir`
(`pilot_run_evidence.json`, or `pilot_preparation_result.json` for a
`--prepare-only` invocation — selected via the explicitly forwarded
`PREPARE_ONLY_USED` flag, never inferred from file presence, since a stale
preparation-result file from an earlier, unrelated `--prepare-only` call
can persist in the same fixed per-`run_id` evidence directory). A
successful bounded pause now gets its own distinct, truthful
`PAUSED_AT_MAX_TARGET_EPOCH` status (Slurm exit `0`) instead of being
folded into generic `COMPLETED`. `pilot_result.json` now also records
`wandb_run_id`, `authoritative_result_source`, and
`authoritative_result_parse_status` (`parsed_successfully` / `absent` /
`absent_pilot_failed_before_evidence_creation` / `corrupt`) and
`physical_state_fallback_used`, so a missing or corrupt authoritative file
is always visible rather than silently reduced to an empty mapping. The
on-disk physical-state fallback (`compute_pilot_status_fields`) and the
Third/Fourth Moriah results' overshoot-blocking classification are
unchanged. No training, evaluation, metric, stopping, checkpoint,
screening, or W&B policy logic changed. Focused tests were added to
`tests/test_pilot_sbatch_launcher.py`. **No further Moriah job should run
until this local fix is committed.**

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

## Per-run W&B policy override (2026-08-02, implementation-only, not launched)

`scripts/run_stage1_lead06_pilot.py` now accepts an optional `--wandb-policy-path <path>` flag (and the paired `.sbatch` launcher forwards `WANDB_POLICY_PATH=/absolute/path/to/policy.yaml` to it), plus an explicit `--tracking-generation` flag (default `"g1"`, was previously only reachable by editing code). Full operational details and rationale: `docs/stage1_wandb_user_guide.md` §15. In one sentence: the committed `config/stage1_wandb_tracking_policy_v001.yaml` stays `enabled: false`/`mode: disabled` and is never edited; an untracked, machine-local override policy (e.g. a copy with only `enabled`/`mode` flipped) can be supplied per-invocation to turn on offline W&B tracking for exactly one candidate, without touching any other pilot-policy field, scientific setting, or the shared default other candidates get. This section documents the mechanism only — it does not enable tracking for `raw_seedA` or any other candidate, and no Moriah operation was performed to implement it (no package install, no Slurm submission, no config generation, no training).

Key points that matter for anyone about to actually use this:
- The override file is validated through the exact same `load_tracking_policy` function the committed default goes through, called eagerly by the CLI before any config generation — a missing or malformed override fails immediately and loudly (`TrackingError`), not partway into a run.
- The override's checksum (not its raw path) is recorded in `run_identity["wandb_policy_sha256"]`; the raw path itself is only ever captured in the evidence bundle's `commands_used` and, on Moriah, the launcher's own `pilot_result.json`.
- `WANDB_DIR` must be outside the tracked repository clone whenever an override is supplied: production tracking code never sets `WANDB_DIR` itself, so the `.sbatch` launcher defaults it to `${FLASHNH_BASE}/wandb/offline/${RUN_ID}` (never under `${REPO_CLONE_DIR}`/`${REPO_WORKDIR}`) and exports `WANDB_MODE=offline` whenever `WANDB_POLICY_PATH` is set. Online mode is never exported by this launcher.
- `tracking_generation` stays `"g1"` by default for every ordinary invocation; nothing about its existing stable-run-identity/contradiction-check behavior (see the "W&B tracking" section above) changed — only a CLI/launcher pass-through was added where none existed before.
- No scientific, screening, early-stopping, continuation, checkpoint, or sealed-set behavior changed. Focused tests: `tests/test_run_stage1_lead06_pilot_cli.py` (new), plus additions to `tests/test_pilot_sbatch_launcher.py`, `tests/test_pilot_orchestration.py`, and `tests/test_pilot_tracking.py`.

## Preparation-only mode: `--prepare-only` / `PREPARE_ONLY=1` (2026-08-02, implementation-only, not launched)

The per-run W&B policy override above closed one gap but opened a real one: nothing in the CLI could generate a candidate's NH config/generation manifest without also, in the same call, being able to fall through into a real NH training call and (if a W&B override was supplied) a real backend initialization. `scripts/run_stage1_lead06_pilot.py` now accepts an optional `--prepare-only` flag (and the paired `.sbatch` launcher accepts `PREPARE_ONLY=1`) that exposes exactly `run_pilot()`'s own first step — `prepare_pilot_run()` plus this pilot's provenance/identity computation — and stops there. It is one narrow early-exit mode around existing preparation, not a second, generalized lifecycle framework, and it changes no scientific setting: the config it writes is the same `prepare_pilot_run()` call `run_pilot()` itself makes first.

What `--prepare-only` does, in order: loads/validates the pilot policy; applies any `--wandb-policy-path`/`--tracking-generation` override exactly as already implemented (see the section above); generates the candidate's `config.yaml`/`generation_manifest.json` via `prepare_pilot_run()` (idempotently — an already-correctly-prepared config is reused, not regenerated); computes and records the effective W&B policy checksum and `tracking_generation` via the same pure `build_pilot_run_identity()` computation `run_pilot()` uses (no NH/W&B call); and writes a `pilot_preparation_result.json` under `--evidence-out-dir` with `status: "PREPARED_ONLY"`, the generated config/manifest paths, `wandb_policy_sha256`, `tracking_generation`, and explicit `training_started`/`evaluation_started`/`wandb_backend_initialized: false` confirmation fields. It never calls `neuralhydrology.nh_run.start_run`/`continue_run`, never calls `init_pilot_tracking_run` (the one real W&B/backend-initializing call in this codebase), and never creates a checkpoint, optimizer-state, validation-result pickle, screening-event, early-stopping-state, or orchestration-state file. It never accesses the temporal-test period, any spatial-holdout basin, or any California basin — see `src.baseline.pilot_orchestration.prepare_pilot_run_only`'s docstring.

Restart-safety is intentionally stricter than `run_pilot()`'s own resume behavior, not equivalent to it: `_assert_no_prior_training_state` fails loudly (`PilotOrchestrationError`) if this `run_id` already has *any* NH run directory under `config_out_dir/runs` (a single match, not just an ambiguous multiple-match, since even one existing run directory means training has already started for real) or if `--evidence-out-dir` already contains anything other than a prior `pilot_preparation_result.json` (e.g. a real evidence bundle from an actual training invocation). `--prepare-only` never silently overwrites either kind of state, and never describes an already-trained or ambiguous candidate as a clean preparation.

The `.sbatch` launcher's `PREPARE_ONLY=1` threads `--prepare-only` through, retains the existing W&B policy-path/`WANDB_DIR`/`tracking_generation` propagation unchanged, and skips the GPU/CUDA hard-check (preparation imports no `torch`/`neuralhydrology`) — but its `#SBATCH` resource header still requests the same GPU class as an ordinary training submission by default, since `#SBATCH` directives are parsed by Slurm before the script body runs; a submitter wanting a genuinely GPU-free preparation job overrides `--partition`/`--gres`/`--cpus-per-task`/`--mem` directly at `sbatch` submission time. The launcher's own status classification checks for `stdout_result.get('status') == 'PREPARED_ONLY'` first, before its ordinary `COMPLETED`/`INTERRUPTED_RESUMABLE`/`BLOCKED_MANUAL_REVIEW_REQUIRED`/`FAILED_NO_CHECKPOINT` chain, and reports a distinct `PREPARED_ONLY` status (exit code 0, but never folded into `COMPLETED`) — this was a real classification gap: without this check, a successful preparation-only job (`pilot_status == 0`, no `final_status` key at all in its JSON output) would otherwise have fallen through and been misreported as a completed training run.

No scientific, screening, early-stopping, continuation, checkpoint, or sealed-set behavior changed, and no Moriah operation was performed to implement this (no package install, no Slurm submission, no config generation against real transferred data, no training). Focused tests: additions to `tests/test_pilot_orchestration.py`, `tests/test_run_stage1_lead06_pilot_cli.py`, and `tests/test_pilot_sbatch_launcher.py`.

## `max_updates_per_epoch` capped-update support (2026-08-03, implementation-only, not launched)

An optional `max_updates_per_epoch: int | None` field is now supported end-to-end for efficient structural-candidate screening (config generation, pilot/candidate policy, run identity, continuation/checkpoint safeguards, evidence recording). Full design framing: `docs/stage1_validation_optimization_foundation.md` Part L (L.5 adopted the direction; L.7, added alongside this section, records that the mechanism is now implemented). **This section documents the mechanism only.** It does not change `raw_seedA`, `emb128x64_seedA`, or any other named candidate — all of them keep `max_updates_per_epoch: null` (uncapped, full-fidelity) exactly as before. No numerical cap has been adopted, no capped run has been launched, and no Moriah operation was performed to implement this (no package install, no Slurm submission, no config generation against real transferred data, no training).

Key points:
- `null` (the default) means unchanged, uncapped behavior; any other value must be a positive integer — `0`, negative integers, bools, floats, and strings are all rejected before config generation.
- The cap is frozen for a candidate's whole trajectory. `enforce_pilot_cap_identity` rejects (before any training call) a continuation/resume whose freshly-resolved cap disagrees with the cap already recorded for that NH run directory — covers null→int, int→null, and int→different-int, in both directions. A capped run and an uncapped run are always distinct identities; a capped checkpoint is never a valid continuation source for an uncapped trajectory or vice versa.
- `--prepare-only` records the declared cap in the run identity without starting training, same as every other identity field it already records.
- Verified NeuralHydrology 1.13 semantics (read from the vendored source, not assumed): the cap truncates each epoch's DataLoader iteration to a deterministic index-based prefix, re-applied fresh every epoch; the scheduler still steps once per epoch, and both `model_epochNNN.pt` and `optimizer_state_epochNNN.pt` are still written unconditionally once per epoch regardless of the cap. That unconditional per-epoch optimizer-state checkpoint is what makes real actual-update evidence obtainable (`optimizer_state_epochNNN.pt`'s own persisted Adam/AdamW `state[p]['step']` counter) without any NeuralHydrology core-code change.
- The evidence bundle records the *configured* cap and, where measured, the *actual* per-epoch optimizer-update count (`actual_optimizer_updates_by_epoch`) as two distinct fields, never conflated.
- `MAX_TARGET_EPOCH`, early stopping, screening cadence, checkpoint discovery, and W&B-disabled behavior are all unchanged and structurally uncoupled from this field (confirmed by code inspection, not just testing — none of those code paths reference `max_updates_per_epoch`).

No scientific, screening, early-stopping, continuation, checkpoint, or sealed-set behavior changed. Focused tests: additions to `tests/test_pilot_lead06_config.py`, `tests/test_pilot_orchestration.py` (cap-identity safeguard plus real-optimizer-state evidence extraction), `tests/test_pilot_tracking.py`, and `tests/test_pilot_evidence_bundle.py`.

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

**W&B offline qualification (this update).** W&B tracking's wrapper
contract is fake-backend-tested, and its offline mode has now additionally
been qualified against the real, installed wandb 0.28.1 package (see "W&B
tracking" section above and
`docs/stage1_validation_optimization_foundation.md` Part L.4 for the full
record and scope limits). Online mode remains unqualified. This makes
offline tracking safe to *consider* enabling for `raw_seedA` if/when
desired — it is not itself a decision to enable it. This is documentation,
test, and a local no-network smoke-script exercise only — it does not
launch `raw_seedA`, does not enable tracking by default, and does not
change the preferred-next-candidate decision above.
