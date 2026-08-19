# Stage-1 Evaluation Framework v1 + Phase-B Bayesian HPO Design — Transition Plan

Status: **documentation-only transition/design document.** Written after Dynamic-Input-Family-A closed (`docs/decision_log.md`/`docs/FLASHNH_CURRENT_STATE.md` 2026-08-16 CLOSED entry; `docs/stage1_validation_optimization_foundation.md` Part L.22). This document does not launch HPO, does not implement W&B sweeps, does not implement an event separator, does not launch training, does not run new scientific evaluation, and does not reopen Dynamic-Input-Family-A. It exists so the many scientific decisions made about the *next* phase are captured in the repository before new ChatGPT/Claude sessions and multiple parallel workstreams begin.

Throughout this document, every substantive statement is tagged:

- **[DECIDED]** — an accepted decision, binding until explicitly reopened by the user.
- **[PROVISIONAL]** — a current working assumption, adopted for now but not scientifically final and not to be treated as though it were.
- **[OPEN]** — an explicit unresolved question. Do not silently resolve these in implementation work; they require a design task or user decision.
- **[DEFERRED]** — acknowledged as relevant, deliberately not addressed now.

Nothing in this document authorizes commit/push beyond this documentation task, HPO launch, sweep implementation, event-separator implementation, new training, new scientific evaluation, or sealed-set access. Those all require separate, explicitly authorized tasks (§17).

## 1. Current accepted state (verified against repository, 2026-08-19)

**[DECIDED]**

- Dynamic-Input-Family-A is **CLOSED**. `PT` (`mrms_qpe_1h_mm` + `rtma_2t_K`) is the provisional Stage-1 dynamic-input working family. `PTM`/`PTMW` not promoted; no H256 rescue warranted. Closure commit `205ce64` ("Close Dynamic-Input-Family-A with PT and retain event-evaluation tooling"), on top of campaign-implementation commit `a3bf512` ("Freeze Dynamic-Input-Family-A design and implement P/PT/PTM/PTMW dynamic-input override machinery"). Full text: `docs/decision_log.md` 2026-08-16 CLOSED entry; `docs/stage1_validation_optimization_foundation.md` Part L.22.
- `seq_length=72` is the provisional Stage-1 working history length (Sequence-Length-A closed, `docs/decision_log.md`/foundation-doc Part L.20, 2026-08-15). `seq_length=48` is the nearest credible alternative. Not claimed saturated at 72h.
- Static embedding architecture is `[128,32]`, tanh activation, embedding dropout 0.10 (Embedding-Dropout-A closed, Part L.19, 2026-08-13). Provisional, not proven optimal.
- Lead is `6h`; target is `qobs_mm_per_h_lead06`.
- `hidden_size=128` (Hidden-size-A closed, Part L.16, 2026-08-10). `H=64` remains a genuine near-tie and a live Phase-B alternative; `H=256` a plausible upper capacity point; `H=512` dropped from the default Phase-B search space.
- `learning_rate=3e-4` (LR-A closed, Part L.14, 2026-08-09). Useful LR region approximately `1e-4`–`1e-3`.
- `output_dropout=0.25` (inherited throughout Phase-A, never itself the subject of a dedicated one-dimensional characterization campaign).
- Roadmap item 4 already named in the L.22 closure entry as the next milestone: *"Stage-1 Evaluation Framework v1 + Phase-B Fidelity Design"* — this document is that milestone's design/handoff, refined and expanded (notably adding the Bayesian-vs-random-control HPO methodology) per direct user decisions recorded below. This is a naming refinement, not a conflict: the roadmap entry named the milestone only, without designing it.
- No conflict was found between this task's stated facts and current canonical repository policy/state. Verified against `docs/FLASHNH_CURRENT_STATE.md`, `docs/decision_log.md`, `docs/stage1_validation_optimization_foundation.md` Parts L.19–L.22, `docs/stage1_scientific_baseline_design.md` §8/§8b/§8c/§8d, `docs/stage1_wandb_user_guide.md`, `docs/repo_policy.md`, `docs/remote_operations.md`, `docs/agent_handoff_rules.md`. No stop-and-report condition was triggered.

## 2. Why two parallel tracks

**[DECIDED]**

Phase A (Dynamic-Input-Family-A, Sequence-Length-A, Embedding-Dropout-A, Hidden-size-A, LR-A) characterized individual axes one at a time. The next phase needs (a) a real joint search across multiple hyperparameters together, and (b) a richer definition of what "good" means for Flash-NH's actual purpose — hourly high-flow/flash-flood prediction — beyond a single aggregate skill number. Neither should block the other:

- **Track A — Phase-B hyperparameter optimization.** Begin joint multidimensional HPO now, using a simple, stable search objective, rather than waiting for a fully mature evaluation framework.
- **Track B — Evaluation Framework v1.** In parallel, formalize richer evaluation (categorical detection metrics, event-level diagnostics, benchmarks) that will later assess HPO shortlists, define promotion gates, challenge NSE-only winners, and support future lead-time comparisons.

> **HPO does not wait for Evaluation Framework v1 to be fully complete.**
> **Evaluation Framework v1 does not replace the primary HPO objective in Sweep v1.**

Both statements are binding **[DECIDED]** framing for this transition, not merely descriptive.

## 3. Phase-B HPO objective and data roles

**[DECIDED]**

### Training population

- Approximately **2,307** development-training basins.
- Training period: **`2020-10-14` → `2023-12-31`** (`docs/stage1_scientific_baseline_design.md` §8, approved 2026-07-06).

### Frequent HPO screening / validation population

- The frozen approximately **400-basin** screening subset — the same population used throughout Phase A.
- This is a **subset of the development-training basin population**, not a spatially unseen population. Per `docs/stage1_scientific_baseline_design.md` §8b: the validation basin list equals the development-training basin list under the current temporal split (a distinct *basin* list would only exist if the spatial split diverged validation from training basins, which it does not today).
- Evaluated over the **validation temporal period `2024-01-01` → `2024-12-31`**.

### Initial HPO search objective (binding for Sweep v1)

> **Maximize median per-basin raw-space NSE on the frozen development-validation screening population (2024 window).**

This is a **search objective**, not the final definition of hydrological model quality — Track B exists precisely because raw-space NSE alone does not capture flash-flood-relevant skill.

**Important, binding caveats:**

- NeuralHydrology's transformed-space training loss is **not** the authoritative HPO scientific metric — Flash-NH raw-space evaluation remains authoritative, consistent with every closed Phase-A campaign.
- Because the 2024 screening population/period will be repeatedly queried across many HPO trials, **it becomes part of model tuning** by construction. It must never later be described or used as an independent final test.

### Sealed sets — not accessed during HPO design/search

**[DECIDED]** Per `docs/stage1_scientific_baseline_design.md` §8/§8b/§8c (approved 2026-07-06) and reaffirmed at every Phase-A closure:

- **2025 temporal test** (`2025-01-01`–`2025-12-31`) — sealed.
- **Non-CA spatial holdout** (~10% of non-CA CONUS basins) — sealed, test-only.
- **California** — excluded entirely from Stages 1–3; reserved for the Stage 4 transfer-learning design (§8c).

These remain reserved for later generalization/authoritative evaluation stages under existing, already-approved policy. Phase-B HPO design and search must not access any of them unless the user explicitly authorizes that scope for a specific later task.

## 4. Phase-B Bayesian-search direction and authority split

**[DECIDED — direction, not implementation]**

- W&B Bayesian optimization is the intended first adaptive HPO mechanism for Sweep v1.
- Flash-NH (this repository's own code and the user) remains authoritative for: legal parameter/configuration validation; candidate identity; target/lead/split definitions; fidelity; Slurm execution; raw-space metrics; sealed-set protection; evidence/provenance; scientific promotion and final interpretation.
- W&B is allowed to: propose configurations; coordinate/search candidates; track/log results. **W&B is never described as scientific authority** — a W&B ranking or optimizer proposal is a search-process artifact, not a scientific conclusion, mirroring the existing rule already codified in `AGENTS.md` §3 and `docs/stage1_wandb_user_guide.md` §10.
- Current W&B qualification status (unchanged by this document; see `docs/stage1_wandb_user_guide.md` for full detail): wrapper contract tested against a fake backend; **real package offline mode qualified** (single-segment); **online mode not qualified**; **sweeps not implemented**; offline-to-server sync qualified for single-segment runs only; launch-contract (env/CLI policy selection) qualified locally and on Moriah. **W&B sweep support still requires implementation and qualification** before Sweep v1 can launch — this is unchanged from current repository state and is not resolved by this document.

## 5. Bayesian vs. random-search control

**[DECIDED — methodology]** / **[OPEN — exact numbers]**

The Phase-B HPO experiment will include two arms:

- **Bayesian arm.** Larger adaptive search; modest concurrency so later proposals can learn from completed trials (excessive simultaneous proposals reduce sequential adaptivity).
- **Random-search control.** Smaller, seeded, frozen cohort, drawn from the exact same allowed search space; same Seed A; same training fidelity; same evaluation objective; candidate list generated and frozen **before** looking at Bayesian outcomes; may run with greater parallelism than the Bayesian arm.

**Comparison should focus on:** best-so-far screening NSE vs. number of completed trials; best-so-far screening NSE vs. cumulative GPU-hours; score distributions; search-space coverage; whether Bayesian trials increasingly concentrate in productive regions.

**Must NOT** compare the two arms primarily by wall-clock completion time — random search is inherently more parallel, so wall-clock is not a fair axis.

**[OPEN]** Exact number of Bayesian trials; exact number of random-control trials; exact Bayesian concurrency level. None of these are frozen anywhere in the repository today.

## 6. Hyperparameter-search framing for Sweep v1

**[PROVISIONAL / OPEN mix — the final Sweep-v1 search space is NOT frozen by this document]**

### Strong current candidates for Sweep v1 dimensions **[PROVISIONAL]**

- Learning rate
- Hidden size
- Embedding dropout
- Output dropout

### Candidate fifth dimension **[OPEN]**

- Batch size — inclusion not decided.

### Parameters requiring explicit review before inclusion **[OPEN]**

- Initial forget-gate bias
- Weight decay / regularization
- Learning-rate schedule parameters
- Optimizer type — **must not be documented as permanently fixed.** Current working expectation (**[PROVISIONAL]**, not binding): keep Adam fixed in Sweep v1 unless inspection gives a concrete reason to reopen optimizer search.

### Fixed for Sweep v1 unless explicitly reopened **[DECIDED, scope-limited to "unless explicitly reopened"]**

- `PT` dynamic-input family
- `seq_length=72`
- `[128,32]` static embedding architecture, tanh — a deferred structural decision, not a claim of global optimality
- Current lead (`6h`) / target (`qobs_mm_per_h_lead06`)
- Current static attribute matrix
- Model head / output activation
- Target/split/package semantics

Static embedding architecture and sequence length are explicitly **deferred structural decisions**, carried forward from Phase A, not claims of global optimality — consistent with how their respective closure entries (Parts L.19, L.20) described them.

## 7. Fidelity — OPEN

**[OPEN]**

The Phase-A regime (25k updates/epoch cap × six epochs) was a screening/characterization fidelity. Evidence that it is probably not ideal for joint Bayesian HPO:

- Noisy trajectories across Phase-A campaigns.
- Candidate/checkpoint rankings moved across epochs (LR-A: a 3/6-only cadence would have missed the true best-observed checkpoint for all 5/5 candidates; a 2/4/6 cadence recovered it for only 2/5 — `docs/decision_log.md` 2026-08-09 LR-A closure).
- Best observed checkpoints were sometimes not at the final epoch (e.g. Dynamic-Input-Family-A's `PT` best checkpoint was epoch 3, not epoch 6).
- Phase-A comparisons repeatedly showed cadence sensitivity (Embedding-Dropout-A: "ranking is cadence-sensitive").

**Current working idea, NOT frozen policy:** raise the update cap (examples discussed: roughly `50k` updates/epoch), run more epochs (examples discussed: roughly `10`–`12`), and evaluate every epoch or substantially more densely than the 3/6 cadence.

> **Open question:** What medium-fidelity Phase-B protocol gives sufficiently stable candidate ranking without wasting HPO compute?

`50k × 12` is an example under discussion, **not** binding policy. This must be resolved (or explicitly deferred with a stated interim choice) in Task A (§17).

## 8. Evaluation Framework v1 — architecture

**[DECIDED — conceptual design]** / **[OPEN — algorithms/thresholds]**

Two layers, at minimum:

### 8.1 Exact-hour categorical / operational verification

**[DECIDED — concept]** These metrics do **not** require hydrologic event windows. For a forecast issued at time `t` and lead `L`, compare prediction for `t+L` against observation at `t+L`.

Planned metrics: POD (Probability of Detection); conditional/anticipatory POD; FAR (False Alarm Ratio); CSI (Critical Success Index); TSS where useful.

**Conditional POD, defined carefully:** for a given threshold and lead — observed `Q(t)` must be below the high-flow threshold at issue time; observed `Q(t+L)` is above threshold; candidate prediction `Qhat(t+L|t)` determines whether the future high-flow state was detected. This uses observed `Q(t)` only for **verification conditioning** — it does **not** require observed discharge to be a Flash-NH model input.

All metric APIs should be designed with lead time as an explicit parameter. **However:** current Stage-1 evaluation remains lead-6-only — this milestone does not launch or plan new lead experiments.

### 8.2 Hydrologic event evaluation

**[DECIDED — concept]** / **[OPEN — algorithm]**

Events should use a **deterministic, candidate-independent, observed-only event separator.**

**Important accepted design change [DECIDED]:** do **not** use a universal fixed event window as the canonical event definition going forward. Flashy basins span strongly different drainage areas and response times, so variable-duration observed events are scientifically preferable to a one-size window.

Note on existing tooling: `select_high_flow_events()` (`src/baseline/hydrograph_atlas_events.py`) and `src/baseline/high_flow_event_metrics.py`, committed and retained per the Dynamic-Input-Family-A closure (Part L.22), implement a **fixed-window** selector (72h peak separation, 24h-before/48h-after window) that was fit-for-purpose for that closure's retrospective diagnostic audit. That tool remains valid for the use it was built for; it is explicitly **not** to be treated as the future canonical Evaluation Framework v1 event definition, which must be variable-duration per the decision above.

The event-definition algorithm itself remains **[OPEN]** and must eventually address: observed event onset; peak; recession/end; declustering; multi-peak events; boundary conditions; minimum valid support; event matching/timing interpretation. **Predictions must never define event boundaries** — the separator is observed-only and candidate-independent.

Planned event diagnostics: peak magnitude error; peak timing error; event volume; event hydrograph/shape error.

This milestone does not implement or freeze the separator.

## 9. Evaluation hierarchy

**[DECIDED]**

- **HPO search objective:** median per-basin raw-space NSE on the frozen 2024 screening subset (§3).
- **Routine supporting diagnostics (likely):** NSE distribution/percentiles; fraction NSE > 0; fraction NSE > 0.5; KGE; RMSE; MAE; bias/PBIAS; training trajectory; resource/runtime diagnostics.
- **Rich flood-focused diagnostics:** Q90/Q95/high-flow conditional metrics; categorical detection metrics (§8.1); event peak/timing/volume/shape (§8.2).
- **Visual evaluation** (for serious/promoted candidates): true same-panel hydrograph overlays — same basin, same event/window, same observations, same axes/scales, multiple candidate predictions overlaid together — following the comparative-hydrograph convention already adopted in Part L.20.

**No composite "Flash-NH score" at this stage [DECIDED].** Preference: use richer metrics as diagnostics/promotion evidence before inventing a weighted multi-objective HPO score — consistent with the standing "no single decision statistic" rule already applied throughout Phase A.

## 10. Stratification

**[DECIDED]**

Stratification (by basin area, flashiness/response-time proxy, hydroclimate, geography, event severity, and later forecast lead) is scientifically important but is **not part of initial Phase-B calibration.**

> **The initial Bayesian HPO objective remains unstratified across the frozen screening population.**

Stratification belongs primarily to shortlist evaluation, mature-model interpretation, and later scientific/generalization analysis — not the Sweep-v1 search objective itself.

## 11. Benchmark hierarchy

**[DECIDED — plan]**

- **Immediate benchmark: persistence.** `Qhat(t+L) = Qobs(t)`. Should become an explicit Stage-1 benchmark soon. Especially valuable for hourly streamflow because streamflow is strongly autocorrelated, persistence can score well on continuous metrics, and it has limited anticipatory value before new high-flow onset — making it a useful complement to conditional POD (§8.1).
- **Existing basic reference:** NSE = 0 / mean-observation reference; KGE/basic skill references where appropriate.
- **Future external opponent benchmark: National Water Model (NWM).** Desired future operational/external benchmark where overlapping gauges/times/lead semantics allow meaningful comparison. **[DEFERRED]** — not a blocker for Phase-B HPO.
- **Later forecast-forcing reference.** When Flash-NH reaches the forecast-forcing stage, observed/QPE-forced Flash-NH should serve as a "perfect/observed forcing" reference against NWP-forced Flash-NH. **[DEFERRED]** — not claimed as the immediate next stage; current project stage numbering is unchanged by this document.

## 12. Search-monitoring / trust diagnostics

**[DECIDED — requirement]** / **[OPEN — exact figure set]**

Phase-B HPO should produce clear explanatory evidence so a human reviewer can judge whether the search was genuinely useful, not merely trust the optimizer:

- Best-so-far NSE vs. completed trials.
- Best-so-far NSE vs. cumulative GPU-hours.
- Hyperparameter-performance plots.
- Parameter-importance diagnostics where defensible.
- Search-space coverage.
- Whether winners lie on search boundaries.
- Bayesian vs. random comparison (§5).
- Training trajectories.
- Candidate score distributions.

**Interpretation rules [DECIDED]:** if top trials accumulate at a parameter boundary, consider whether the range should later be expanded; if best-so-far plateaus despite additional exploration, this supports search maturity/convergence; **the single W&B leaderboard winner is not automatically the final model** — finalists must later be challenged using richer hydrologic evidence (Track B) and seed robustness (§13).

Standing preference reaffirmed: scientific comparison closures should generate useful explanatory figures, not only markdown tables — consistent with every Phase-A closure's figure-pack convention.

## 13. Seed strategy

**[DECIDED — logic]** / **[OPEN — exact finalist count]**

- Seed A is used for Phase-B search.
- The entire HPO search is **not** multiplied across seeds.
- Seed B is reserved for a small number of promoted/integrated finalists, to determine whether apparent HPO gains exceed stochastic initialization noise.

**[OPEN]** Exact number of Seed-B finalists.

## 14. W&B + Moriah/Slurm architecture requirement

**[DECIDED — constraint]** / **[OPEN — exact implementation]**

Moriah login nodes must not perform training or substantial compute (unchanged, pre-existing rule; `docs/remote_operations.md` §2.2).

A safe Sweep design should resemble:

```text
W&B controller
  -> Slurm GPU allocation / sweep worker
  -> Flash-NH proposal validation
  -> NH training
  -> Flash-NH raw-space evaluation
  -> objective returned to W&B
```

W&B must not bypass: Slurm resource allocation; clean-tree/commit guards; configuration legality; sealed-set protections; run-identity/provenance.

**[OPEN]** Exact implementation. Likely options to inspect in Task A (§17): one W&B sweep agent per bounded Slurm GPU job (e.g. `count=1`); another queue/controller architecture if it better fits current orchestration. Not implemented here. Bayesian concurrency should be moderate rather than maximal (§5); exact concurrency remains **[OPEN]**.

## 15. Open-decision register (before Phase-B launch)

Consolidated from the sections above — none of these are resolved by this document.

**Search space**
- Exact Sweep-v1 HP dimensions.
- Batch size: yes/no.
- Forget-gate bias: yes/no.
- Weight decay: yes/no.
- Learning-rate schedule search: yes/no.
- Optimizer: fixed Adam vs. optimizer search.
- Exact parameter ranges/distributions.

**Fidelity**
- Updates-per-epoch cap.
- Epoch budget.
- Screening/evaluation cadence.
- Performance-based early stopping during Sweep v1: yes/no.

**W&B / Slurm**
- Exact sweep-agent architecture.
- Online W&B qualification (currently unqualified).
- Bayesian concurrency.
- Bayesian trial budget.
- Random-control trial budget.
- Sweep failure/retry semantics.
- Objective-reporting contract.

**Promotion**
- How many candidates advance.
- Which richer metrics can block promotion.
- When Seed B enters.
- Higher-fidelity promotion protocol.

**Evaluation Framework**
- Canonical high-flow threshold(s).
- Variable-duration observed-only event-separator algorithm.
- Event start/end, multi-peak handling, declustering, timing interpretation/tolerance.
- Categorical metric definitions.
- Routine vs. diagnostic vs. promotion-gate metric classification.

## 16. Next two planned design tasks

Neither task begins in this session. Both are read-only/design-first — no implementation until human review.

### Task A — Phase-B Bayesian HPO Launch Design Review

Inspects the actual current repository and recommends: existing override support; candidate HP dimensions; search distributions; trial fidelity; W&B/Slurm architecture; random-control design; concurrency/budget; qualification sequence.

### Task B — Evaluation Framework v1 Scientific Design

Defines: high-flow thresholds; exact-hour categorical metrics; conditional POD; persistence benchmark; variable-duration observed-only event separator; event metrics; aggregation; future lead-aware interfaces.

The two tasks may proceed as separate workstreams after this handoff commit, but their interfaces/promotional roles should be reviewed together before major implementation (Task B's event/categorical metrics are promotion-gate evidence for Task A's HPO shortlist).

## 17. Deferred work

**[DEFERRED]**

- Dewpoint / both-moisture ablation (carried forward from Dynamic-Input-Family-A).
- `v001-fullmet` (pressure/cloud/visibility/gust/ceiling) dynamic-input family.
- Longer `seq_length` testing beyond 72h.
- H256 (or other capacity) probe revisit for `PTMW` under a future higher-fidelity protocol.
- National Water Model benchmark integration (§11).
- Forecast-forcing "perfect forcing" reference (§11).
- Sealed-set (2025 temporal test, spatial holdout, California) access of any kind.
- Multi-lead-time evaluation and search.

## 18. Document history

- 2026-08-19 — Created. Documentation-only transition/design handoff following Dynamic-Input-Family-A closure (`205ce64`). No HPO, sweep, event-separator, training, or evaluation code touched. No sealed-set access.
