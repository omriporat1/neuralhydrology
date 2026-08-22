# Phase-B Sweep-v1 Launch Contract

Status: **[DECIDED — implementation contract]**. This document freezes the original-domain Sweep-v1 scientific and operational contract after the five-candidate epoch-budget calibration. It authorizes neither W&B implementation nor a random manifest, Slurm submission, training, sealed-set access, or a promotion decision.

## Medium-fidelity contract

- 12 epochs; `max_updates_per_epoch=50,000`; Seed A.
- PT, `seq_length=72`, learned static embedding `[128,32]` with tanh, Adam, and lead 6h.
- Save every epoch; canonical raw-space screening is required at every epoch 1–12; performance-based early stopping is disabled.
- Objective: the best eligible authoritative median per-basin raw-space NSE observed within epochs 1–12.

The calibration completed 5/5 trajectories through epoch 14. Epoch 8 missed materially better epoch-9 checkpoints for C1/C2/C3 and had unstable ranking. Every cohort global best occurred by epoch 9; for all five, `best_score(10) == best_score(12) == best_score(14)`, with identical ranking/top-2 at 10/12/14. Thus the evidence directly supports 10 epochs for the tested cohort. The adopted 12-epoch fidelity is a deliberate precautionary margin for untested joint configurations, not a claim that the cohort required 12; epoch 14 discovered no additional best checkpoint.

## Original-domain search and budget

**[DECIDED FOR SWEEP V1]**

| Axis | Domain / prior |
|---|---|
| `learning_rate` | continuous log-uniform `1e-4`–`1e-3` |
| `hidden_size` | categorical uniform `{64,128,256}` |
| `embedding_dropout` | continuous uniform `0.0`–`0.4` |
| `output_dropout` | continuous uniform `0.0`–`0.4` |
| `batch_size` | categorical uniform `{128,256,512}` |

Adam, PT, `seq_length=72`, `[128,32]` tanh static embedding, and lead 6h are fixed. Optimizer, LR schedule, weight decay, and `initial_forget_bias` search are excluded from Sweep v1 only. Dropout’s lower bound zero is natural and model-meaningful; all other stated bounds are experimental search bounds and may later be reviewed for expansion on evidence.

The original domain contains 36 **valid** Bayesian trials and 12 frozen random-control trials. A valid Bayesian trial has workflow PASS, completes all 12 epochs, emits every required raw-space screening result, preserves the frozen package/screening/fidelity identity, and has no integrity or scientific-scope violation. Ordinary infrastructure failures do not consume a valid scientific trial.

Planning estimate only: roughly two GPU-hours per scientific trial, or about 72 GPU-hours Bayesian plus 24 GPU-hours random (roughly 95–100 total, with meaningful uncertainty). This does not predict scheduler queue or wall-clock duration. Expanded-wave cost is conditional.

## Random-control arm

Before observing any Bayesian outcome, generate 12 seeded IID draws from the frozen Sweep-v1 priors above. Use a dedicated, documented manifest RNG seed that is distinct from model Seed A; freeze and hash-pin the resulting manifest. This is ordinary random search: do not use Sobol, Latin-hypercube, or other space-filling designs, and do not reject naturally sampled duplicates merely to force uniqueness. If a random and Bayesian proposal exactly match, retain both arm-proposal provenance records; execution/reuse bookkeeping may be resolved during implementation without merging their configuration identities.

## Boundary reviews and immutable waves

At approximately 12 and 24 **valid** Bayesian results, pause issuance of new Bayesian proposals, do not kill in-flight jobs, allow bounded overshoot, and produce the required review packet. The user decides **CONTINUE**, **UNCERTAIN**, or **EXPAND**. CONTINUE resumes the same wave; UNCERTAIN keeps the domain and normally gathers more evidence to the next review; EXPAND does not silently mutate the active wave.

The domain is immutable within a versioned Bayesian wave. An approved EXPAND closes/pauses the original wave, preserves its trials/domain/provenance, and defines a new versioned domain. Valid original observations may warm-start the expanded Bayesian model only when implementation keeps provenance unambiguous. Original random trials remain controls only for the original domain; create a new frozen random manifest before inspecting expanded-wave Bayesian outcomes. Exact expanded bounds and random budget defer until an actual EXPAND decision.

Boundary pressure is a visually inspectable decision aid, never an automated score. For continuous axes evaluate the outer approximately 10% in search geometry (log coordinate for LR, ordinary coordinate for dropout); categorical boundaries are extreme values. Inspect top-quartile occupancy, proposal drift, and neighborhood support. Strong evidence normally has at least half of the top quartile near one expandable boundary plus visible drift and neighborhood support; partial alignment is moderate; little evidence is weak/none. Strong may motivate EXPAND, moderate maps naturally to UNCERTAIN, and weak/none to CONTINUE, always subject to human review. Natural bounds such as dropout zero cannot trigger expansion beyond their meaningful domain.

## Required per-trial diagnostics

Retain each complete epoch-wise raw-space trajectory plus `best_epoch`, `best_score`, `final_epoch_score`, `best_minus_final`, `best_score(10)`, `best_score(12)`, and `late_gain_10_to_12 = best_score(12)-best_score(10)`. An optional descriptive label `late_best = best_epoch >= 11` means only that the best observed checkpoint is near the medium-fidelity boundary. It neither asserts continued improvement nor alters the objective. Numeric late-bloomer and post-peak-instability thresholds defer to higher-fidelity promotion design; these diagnostics add no Bayesian bonuses or penalties.

## Execution, authority, and concurrency

Use one bounded Slurm allocation per proposal/trial, with independent evidence and failure isolation; do not use long multi-candidate allocations. W&B is the Bayesian proposal/search engine, telemetry, and live visualization convenience. Flash-NH remains authoritative for legal domain/configuration, package/split/screening identity, fidelity, raw-space objective, trial identity, sealed-set protection, evidence/provenance, boundary review, and scientific interpretation. W&B is not scientific authority.

Concurrency is operational/configurable rather than scientific: initial production should be conservative (about four concurrent scientific jobs), with a possible increase toward roughly 6–8 only after successful online-controlled production runs and a lightweight Moriah fair-share/queue review. No wall-clock completion promise is made.

Before scientific sweep launch, complete online W&B qualification on a CPU Slurm allocation—not the login node—using the exact runtime and a trivial toy/no-training objective. It must create/use a small real online sweep, receive a proposal, log a metric online, obtain a subsequent proposal, avoid silent offline fallback, and keep credentials out of logs/evidence. No GPU science smoke is required solely for this qualification.

## Visualization and fairness requirements

At both boundary reviews and original-domain closure, produce both a live W&B workspace and a durable versioned Flash-NH packet from authoritative project evidence. The packet must include a decision board with Bayesian/random best-so-far objective by valid-trial index; per-axis performance panels (LR on log x); categorical proposal/top-quartile occupancy; best-epoch and best-minus-final distributions; and a visually evidenced boundary-pressure table (STRONG/MODERATE/WEAK-NONE per axis).

The richer packet must also support same-domain Bayesian/random distributions, cumulative best and GPU-hour views, proposal distributions/boundary occupancy, parallel coordinates and supported pairwise views, explicitly non-causal parameter importance, selected epoch trajectories, best-vs-final and `late_gain_10_to_12`, runtime/score and GPU-hour views, and failures/retries. Avoid redundant figures that do not improve interpretation.

Compare arms primarily by valid trial index, overlaying only through their common completed-trial count; cumulative GPU-hours is secondary. Failures and retries are excluded from objective curves but reported as operational cost. Always expose unequal sample counts. Random search is not expected to show directional learning.

## Deferred items

The following do not block medium-fidelity implementation unless a concrete implementation dependency arises: exact late-bloomer and post-peak-instability thresholds; Seed-B finalist count; higher-fidelity promotion algorithm; expanded-wave bounds and random budget; final sealed-evaluation policy; and exact post-qualification concurrency.
