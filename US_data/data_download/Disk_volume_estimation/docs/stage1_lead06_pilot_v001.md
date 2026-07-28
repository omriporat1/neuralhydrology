# Stage 1 lead-6 optimization pilot (`stage1_lead06_pilot_v001`)

Status: **implementation and tests complete, documentation ready for review.
No Moriah job has been submitted, no training has been run, and no
temporal-test or spatial-holdout data has been accessed.** This document
describes what was built and verified locally; it makes no claim about any
run having occurred.

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
comment). **Not submitted by this implementation.** The first job this
pilot will eventually run is `emb128x64_seedA` (the workflow-qualification
run) — prepared, not launched.

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

Eight new focused pytest files (`tests/test_pilot_*.py`, plus a shared,
non-collected fixture helper `tests/_pilot_support.py`), 95 tests, all
passing. Coverage includes: exact six-run matrix acceptance and rejection
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
resume with no retraining. Full pre-existing repository test suite
re-run alongside these: 1122 passed, 3 initially-failed-but-flaky (Windows
file-lock `PermissionError` during atomic package promotion under load;
confirmed to pass in isolation, unrelated to any file touched by this
pilot), 6 pre-existing collection errors in files that import
`neuralhydrology`/`torch` (not installed in this local environment by
design — those tests only run on h2o/Moriah). Zero regressions
attributable to this work.

## What has not been done

No Moriah job submitted or started. No training run. No full-population
evaluation. No temporal-test or spatial-holdout access. No change to the
certified Compact Scientific Package or canonical split membership. No
regeneration of the screening subset. No hydrograph atlas generation. No
automated sweep. No EA-LSTM work. Nothing generated by this pilot (configs,
runtime outputs, logs, checkpoints, W&B files, evidence directories) has
been committed.
