

# Decision Log

## 2026-08-23 — Sweep-v1 local production integration CLOSED; independently approved for commit

**[CLOSED — TECHNICAL, NOT SCIENTIFIC]** Following the prepared-execution
consumer result contract closure immediately below, Sweep-v1's local
production integration layer (`src/baseline/sweep_v1_production_adapter.py`,
`src/baseline/sweep_v1_execution.py`, `src/baseline/sweep_v1_campaign.py`,
`src/baseline/nh_config_generation.py`, the production scripts and
`.sbatch` launchers) is now CLOSED. An independent review verified the
reviewed architecture is genuinely present — `PreparedPilotExecutionResult`
consumed directly; `actual_optimizer_updates_by_epoch` as authoritative
update evidence; VALID/INVALID interpretation and the committed
`derive_trajectory_diagnostics` objective staying in Flash-NH; full
12-epoch checkpoint/screening coverage; 50,000 as `max_updates_per_epoch`
cap semantics; mature requested/evaluated/area-excluded population
accounting; INVALID/PARTIAL trials never producing a finite Bayesian
objective; Bayesian and random-control trials sharing one executor/validity
path; a thin, non-authoritative W&B bridge; early durable proposal-intake
Layer-B provenance; exact-retry identity preservation; provenance-aware
config generation with protected generated targets categorically
non-allowlistable and exact-present-`trial_id` required for same-trial
provenance coexistence (replacing the earlier unsafe `force=True`
overwrite path); and the one-allocation/one-agent/`count=1` production
launch shape for both arms — and returned final verdict: **APPROVE
PRODUCTION INTEGRATION FOR COMMIT**. The final focused safety gate passed
220 tests, 0 skipped, with torch-capable golden VALID/INVALID bridge paths
executed and `git diff --check` clean.

Committed as `a3ae86b91569e27b6e183666675c06f0e7dc89d4` ("Complete Sweep-v1
production integration"): `src/baseline/sweep_v1_production_adapter.py`,
`src/baseline/sweep_v1_execution.py`, `src/baseline/sweep_v1_campaign.py`,
`src/baseline/nh_config_generation.py`, `tests/test_sweep_v1_execution.py`,
`tests/test_sweep_v1_wandb_bridge_provenance.py`,
`tests/test_nh_config_generation.py`,
`scripts/build_sweep_v1_production_sweep_config.py`,
`scripts/run_sweep_v1_wandb_agent_moriah.sbatch`,
`scripts/run_sweep_v1_wandb_bridge.py`,
`scripts/run_sweep_v1_random_control_trial.py`,
`scripts/run_sweep_v1_random_control_moriah.sbatch`.

No scientific policy changed. No real Sweep-v1 production trial has been
consumed. No live W&B, Slurm, GPU, or training occurred during this
closure. The next operational milestone is one serialized real Bayesian
proposal/trial, which remains a separate, not-yet-authorized step.

## 2026-08-23 — Prepared-execution consumer result contract CLOSED; Phase-B Sweep-v1 production integration unblocked

**[CLOSED — TECHNICAL, NOT SCIENTIFIC]** Following the workflow entry
immediately below (Interface / Consumer Contract Gate, `docs/agent_handoff_rules.md`
§5), the prepared executor's consumer-facing result/evidence contract is now
closed. Prepared-execution mechanics remain CLOSED as before, unchanged.
`execute_prepared_pilot_run` (`src/baseline/pilot_orchestration.py`) now
returns a typed `PreparedPilotExecutionResult` — the generic, campaign-agnostic
factual execution receipt a higher-level scientific workflow such as Sweep-v1
needs. It exposes: physical checkpoint inventory (`checkpoint_inventory`, via
the existing mature `discover_physical_checkpoints`); the complete,
epoch-ordered screening history (`screening_events`, each entry the mature
`evaluate_screening_checkpoint` return shape); and stopping/state facts
(`blocked`, `blocked_reason`, `stopped`, `stop_reason`, `early_stopping_state`).
Actual optimizer-update evidence stays out of the eager receipt (torch-dependent,
not every consumer needs it) and remains available through the existing
authoritative `actual_optimizer_updates_by_epoch` helper. The type carries no
campaign-specific concept — no `VALID`/`INVALID`, `best_score`/`best_epoch`,
or Bayesian objective.

As part of this closure, resumed `execute_prepared_pilot_run`/`run_pilot`
calls now correctly reconstruct the run's full screening history from durable
state (`logged_screening_epochs`) plus the physical checkpoint inventory,
fixing a prior bug where a resumed call's evidence bundle silently carried
only that invocation's newly-processed screening epochs rather than the
run's complete history — a correctness fix to `run_pilot`'s evidence-bundle
content, not a change to its training/continuation/stopping/evaluation
behavior.

An independent read-only review applied the Interface / Consumer Contract
Gate: verified each result field's single authority, the resume
screening-history reconstruction's correctness (including correct
`owning_run_dir` use for continuation-directory checkpoints), `run_pilot`
backward compatibility, and the repaired legacy test fixture's realism. A
vertical consumer-contract test proves a generic consumer can establish
checkpoint coverage, actual optimizer-update evidence, NH evaluation
coverage, screening coverage, screening-population accounting, and the
raw-space median NSE trajectory using only the receipt plus
`actual_optimizer_updates_by_epoch` — no filesystem crawling, no reopened
predictions, no re-derived hydrologic metrics. All focused and regression
tests passed. Commit `63c31a983b2a494e3078ad18a5e97c3cf3b876ee`
(`src/baseline/pilot_orchestration.py`,
`tests/test_prepared_execution_core.py`, `tests/test_pilot_orchestration.py`).

No scientific policy, candidate set, evaluation rule, or sealed-set scope
changed. Sweep-v1 production integration, paused pending this contract (see
`docs/FLASHNH_CURRENT_STATE.md`), may now resume against this closed
contract; this entry does not itself specify or redesign that integration.

## 2026-08-23 — Workflow: adopted Interface / Consumer Contract Gate for agent handoffs

**[DECIDED — WORKFLOW/PROCESS, NOT SCIENTIFIC]** The user approved a durable
workflow lesson from the Phase-B Sweep-v1 production-adapter integration
difficulty. W&B tracking/telemetry qualification and execution-provenance
evidence had each previously been closed within their own scope (see the W&B
offline launch-contract and real-artifact qualification entries below), but
Phase-B integration work exposed that the prepared executor's
consumer-facing result/evidence contract — the structured receipt Sweep-v1
needs to determine trial validity — was a separate, unresolved interface that
neither prior closure covered. Broad claims like "tracking is closed" or
"execution is qualified" had been silently read as covering this consumer
contract too, which they did not.

Future substantial reuse/integration work (cross-component production
integration, reuse of a mature subsystem by a new consumer, reusable-component
extraction, or evidence/tracking/result plumbing where scientific validity
depends on the interface) must apply the Interface / Consumer Contract Gate
now recorded in `docs/agent_handoff_rules.md` §5: identify producer,
consumer, required inputs/outputs, fact authority, failure semantics, and a
vertical synthetic/integration test; prefer scoped closure language over
broad claims; and route missing lower-layer facts to a repaired lower-layer
contract rather than an ad hoc higher-layer reconstruction. This entry adopts
the workflow rule only; it does not change any scientific policy, and it does
not itself define the Sweep-v1 execution result contract (that remains a
separate technical task).

## 2026-08-22 — Phase-B Sweep-v1 launch contract frozen

**[DECIDED]** The completed 5/5 epoch-budget calibration freezes Sweep-v1 at
12 epochs, 50,000 updates/epoch, Seed A, every-epoch raw-space screening, and
no performance stopping; objective is best eligible median per-basin
raw-space NSE through epoch 12. Epoch 10 directly sufficed for the tested
cohort; 12 is a conservative margin for untested joint configurations, while
14 found no further cohort best checkpoint. The original domain is 36 valid
Bayesian and 12 frozen IID-random trials over the five recorded axes; its
domain is immutable within a versioned wave, with human boundary reviews at
about 12 and 24 valid Bayesian trials. Online W&B qualification remains a
CPU-Slurm prerequisite before science launch. Detailed rules, visualization
requirements, authority split, and deferred items are canonical in
`docs/stage1_phase_b_sweep_v1_launch_contract.md`. This entry implements or
launches nothing.

## 2026-08-22 — Phase-B adopted screening-subset portability repair

The accepted Stage-1 provisional operational screening subset v001 did not change: it is the epoch-9, 400-basin realization produced by `scripts/generate_stage1_screening_subset.py` (seed `42`, `stage1_screening_subset_proportional_composite_stratum_selection_v1`), with SHA-256 `d4395d93ebc567cf09e149c0121463d75cf4f7ecc02c07a7c4a7999763baa372`. Historical Phase-A launchers depended on this repo-relative report artifact as ignored state in their historical Moriah worktree. Clean canonical-clone migration exposed that operational dependency; it does not reopen any Phase-A scientific conclusion. The exact historical artifact is now promoted outside Git to the stable Flash-NH project-data location and Phase-B requires its pinned checksum, 400 unique IDs, and development-population membership. Historical campaigns identify the common accepted path/population, but not every retained run has a subset-file checksum.

## 2026-08-21 — Phase-B Track-A epoch-budget calibration core contract frozen (not launched)

**[DECIDED — CALIBRATION DESIGN]** Purpose: determine the common Sweep-v1 epoch budget among `8/10/12/14`, not select a configuration. The frozen cohort is C1 anchor (`lr=3e-4`, H128, batch256), C2 low LR (`1e-4`, H128, batch256), C3 high LR (`1e-3`, H128, batch256), C4 late H64 (`3e-4`, H64, batch256), and C5 convergence stress (`3e-4`, H256, batch128; deliberately a joint H/batch corner). All use PT, seq72, `[128,32]` tanh embedding, embedding dropout 0.10, output dropout 0.25, Adam, Seed A 967139, lead6, and 50k updates/epoch.

Each candidate is one logical uninterrupted epoch-1--14 trajectory with `save_weights_every=1`, no performance-based early stopping, and every epoch 1--14 authoritative raw-space-screening eligible. Existing NH epoch-specific evaluation and Flash-NH raw-space evaluation may run after training; continuation is technical recovery only. The common Sweep-v1 epoch budget remains **[OPEN]**: no calibration training has run and no winner is implied. This does not reopen the five axes, `{128,256,512}` batch support, 50k cap, PT, seq72, or lead6.

**[PROVISIONAL ANALYSIS RULE]** For cutoffs `k={8,10,12,14}`, inspect best score/epoch through k, late regret versus 14, ranking, top-2 membership, Spearman agreement, and rising trajectories. Working checks: no late regret above 0.01 median NSE (also report 0.005 sensitivity), stable top-2, rho >= 0.9, and no clearly still-rising candidate; a threshold-sensitive result gets conservative review, not false precision.

## 2026-08-21 — Phase-B batch-size operational qualification closed

**[DECIDED FOR SWEEP V1]** Batch-size search values are frozen as
`{128,256,512}` after operational H256/L4 eight-update qualification. This
demonstrates practical execution viability only; no winner was selected and
higher-fidelity HPO must evaluate scientific performance. Technical provenance:
`docs/phase_b_batch_size_operational_qualification.md`.

## 2026-08-20 — Phase-B Task-A decisions and `output_dropout` / `batch_size` plumbing increment

**Scientific/design decisions.** Sweep v1 searches exactly `learning_rate`,
`hidden_size`, `embedding_dropout`, `output_dropout`, and `batch_size`. Adam
is fixed. `initial_forget_bias`, weight decay/regularization, learning-rate
schedules, and optimizer search are excluded from this first joint-search
scope, not declared permanently irrelevant. Sweep-v1 medium fidelity uses the
same `max_updates_per_epoch=50,000` across batch sizes: this holds optimizer
update opportunity constant, not sample exposure, and is a Sweep-v1 policy
rather than a universal project rule. Raw-space screening occurs every epoch.
There is no performance-based scientific early stopping; every candidate
receives the complete predefined budget and the objective is its best observed
eligible raw-space screening checkpoint within that budget.

**Provisional items.** `output_dropout` is continuous-uniform `0.0`--`0.4`,
a conservative working range around inherited `0.25`, not a previously
one-dimensional-characterized optimum. Preferred `batch_size` candidates are
`{128, 256, 512}`, pending technical/operational qualification. Planning
direction is approximately 30--40 Bayesian and 10--15 frozen random-control
trials; exact counts remain open.

**Open items.** Exact epoch budget, final batch-size qualification, exact
Bayesian/random counts, Bayesian concurrency, W&B/Slurm sweep-agent
architecture, Seed-B finalist count, higher-fidelity promotion, and Task-B
Evaluation Framework scientific choices remain open.

**Implementation status.** This implementation increment adds only low-level
`output_dropout` and `batch_size` override validation, configuration
threading, identity, provenance, and continuation protection. No HPO candidate
scheme, W&B sweep/controller, random-control generator, Slurm sweep agent, or
training run has been implemented or launched; no sealed data were accessed.

Project: Flash-NH — near-real-time and forecast-aware hydrological modeling pipeline.

## 2026-08-19 — Stage 1 — Evaluation Framework v1 + Phase-B Bayesian HPO Design: documentation-only transition handoff

**Scope.** Documentation-only transition task. Follows the Dynamic-Input-Family-A closure (`205ce64`) and the agent-governance adoption (`d83fde0`), both unchanged by this entry. Records scientific decisions made about the next phase — captured now so they survive across new ChatGPT/Claude sessions and multiple parallel workstreams, rather than living only in conversation. **Does not implement HPO, W&B sweeps, or an event separator; does not launch training; does not run new scientific evaluation; does not reopen Dynamic-Input-Family-A; does not access sealed data.**

**New canonical document.** `docs/stage1_phase_b_hpo_evaluation_plan.md` — the design/handoff document for this phase. Every substantive statement in it is tagged `[DECIDED]` / `[PROVISIONAL]` / `[OPEN]` / `[DEFERRED]` so implementation agents cannot mistake a working assumption for binding policy.

**Two parallel tracks (adopted).** Track A — Phase-B joint multidimensional HPO: W&B Bayesian search as the primary adaptive mechanism, plus a seeded random-search control drawn from the same space, frozen before Bayesian outcomes are inspected; initial search objective is median per-basin raw-space NSE on the frozen development-validation screening population. Track B — Evaluation Framework v1: exact-hour categorical/operational metrics (POD, conditional/anticipatory POD, FAR, CSI, TSS) plus a deterministic, candidate-independent, **observed-only, variable-duration** hydrologic event separator (explicitly *not* a universal fixed window — flashy basins differ too much in drainage area/response time) and event diagnostics (peak magnitude/timing error, volume, shape). **HPO does not wait for Track B to be complete; Track B does not replace the Sweep-v1 HPO objective.**

**Data roles (unchanged, restated precisely).** ~2,307 development-training basins, training period `2020-10-14`–`2023-12-31`. The frozen ~400-basin screening subset is a **subset of the development population** (not spatially unseen), evaluated over `2024-01-01`–`2024-12-31`; because Sweep v1 will query it repeatedly, it becomes part of tuning by construction and must never later be called an independent final test. NeuralHydrology's transformed-space loss remains a training diagnostic only — Flash-NH raw-space evaluation is authoritative, as throughout Phase A. Sealed 2025 temporal test, non-CA spatial holdout, and California remain untouched, per existing `docs/stage1_scientific_baseline_design.md` §8/§8b/§8c policy (unchanged by this entry).

**Bayesian-vs-random-control methodology (adopted).** Comparison focuses on best-so-far NSE vs. completed trials / cumulative GPU-hours, score distributions, search-space coverage, and whether Bayesian trials concentrate in productive regions — explicitly **not** primarily by wall-clock completion time, since random search is inherently more parallel. Exact trial counts and Bayesian concurrency remain open.

**Search-space framing (adopted, not frozen).** Strong candidate Sweep-v1 dimensions: learning rate, hidden size, embedding dropout, output dropout. Batch size is a candidate fifth dimension, not yet decided. Forget-gate bias, weight decay, LR-schedule parameters, and optimizer type all require explicit review before inclusion — optimizer must not be documented as permanently fixed (current provisional expectation: keep Adam unless inspection gives concrete reason to reopen). Fixed for Sweep v1 unless explicitly reopened: `PT` dynamic-input family, `seq_length=72`, `[128,32]` tanh static embedding, current lead/target, current static attribute matrix, model head/output activation, target/split/package semantics — all explicitly deferred structural decisions, not claims of optimality.

**Fidelity (explicitly OPEN, not resolved by this entry).** The Phase-A 25k-updates × 6-epoch regime showed repeated cadence sensitivity across LR-A, Embedding-Dropout-A, and Dynamic-Input-Family-A (best checkpoints often not at the final epoch; sparse cadences missed true best-observed checkpoints). A denser/higher-fidelity Phase-B protocol (examples under discussion only: ~50k updates/epoch, ~10–12 epochs, denser screening cadence) is a plausible direction but **not frozen** — `50k`×`12` must not be treated as adopted policy anywhere downstream.

**Benchmark/evaluation-hierarchy decisions (adopted).** No composite "Flash-NH score" at this stage — richer metrics serve as diagnostics/promotion evidence first. Stratification (basin area, flashiness, hydroclimate, geography, severity, lead) is real but deferred past the initial unstratified Sweep-v1 objective. Persistence (`Qhat(t+L)=Qobs(t)`) becomes an explicit near-term benchmark. National Water Model and a future forecast-forcing "perfect forcing" reference are deferred, not blockers.

**Seed strategy (adopted).** Seed A for the full Phase-B search; the search is not multiplied across seeds. Seed B reserved for a small number of promoted finalists, to separate apparent HPO gains from stochastic initialization noise. Exact finalist count open.

**W&B/Slurm architecture (constraint adopted, implementation open).** Moriah login nodes must not train; a safe design routes W&B controller → Slurm GPU allocation/sweep worker → Flash-NH proposal validation → NH training → Flash-NH raw-space evaluation → objective returned to W&B, without W&B ever bypassing Slurm allocation, clean-tree/commit guards, configuration legality, sealed-set protection, or provenance. Exact sweep-agent architecture and concurrency remain open, to be recommended (not implemented) by Task A.

**Documentation changes made by this entry.** New: `docs/stage1_phase_b_hpo_evaluation_plan.md`. Updated: `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry + header), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (narrow roadmap-consistency note only, no Phase-A history rewritten), `docs/stage1_wandb_user_guide.md` (narrow planned-Phase-B-sweep note, clearly labeled future/unimplemented). No `src/`, `scripts/`, `tests/`, W&B config, scientific YAML, or Slurm launcher touched.

**Next (named only, not started by this entry).** Task A — Phase-B Bayesian HPO Launch Design Review (read-only/design-first). Task B — Evaluation Framework v1 Scientific Design (read-only/design-first). Full task framing: `docs/stage1_phase_b_hpo_evaluation_plan.md` §16.

**Not done by this entry.** No HPO launched, no W&B sweep implemented, no event separator implemented, no training launched, no new scientific evaluation run, Dynamic-Input-Family-A not reopened, no sealed temporal-test/spatial-holdout/California data accessed, nothing beyond documentation committed.

## 2026-08-16 — Stage 1 — Dynamic-Input-Family-A CLOSED: `PT` (precipitation + temperature) adopted as the provisional Stage-1 working family; PTM/PTMW not promoted; no H256 rescue warranted (documentation-only closure)

**Scope.** Documentation-only closure task. The full base campaign (four real Moriah training runs, `P`/`PT`/`PTM`/`PTMW`, six epochs each, at the frozen `dynamic_input_family_seedA_25k_v001` contract), the true multi-candidate hydrograph-overlay review (frozen 8-basin panel), and a dedicated 400-basin/1,200-event high-flow and event-level audit are all already complete, performed earlier in this session. This entry records the final predictor-family decision, retains and commits the supporting reusable evaluation code, produces the closure figure pack and supervisor summary (both project-local, gitignored, not part of this commit), and updates canonical documentation. **It launches no new training, launches no H256 rescue run, does not start Phase B, and does not implement Stage-1 Evaluation Framework v1.**

**Campaign identity.** `dynamic_input_family_seedA_25k_v001` — four candidates (`P`/`PT`/`PTM`/`PTMW`), all other settings frozen at the design-freeze contract below (Seed A `967139`, `[128,32]` learned static embedding, `hidden_size=128`, `learning_rate=3e-4`, `output_dropout=0.25`, `seq_length=72`, lead 6h, target `qobs_mm_per_h_lead06`, `max_updates_per_epoch=25000`, six epochs, one uninterrupted segment, the fixed development-training population, the fixed ~400-basin development-validation screening population, strict real offline-W&B contract throughout). Campaign implementation/training commit `a3bf51266859a8706b40cc9e862acab793ce15c7` ("Freeze Dynamic-Input-Family-A design and implement P/PT/PTM/PTMW dynamic-input override machinery") — the scientific campaign state against which the closure evidence and figure pack were generated. The Dynamic-Input-Family-A scientific closure and reusable evaluation tooling are recorded in a separate closure commit on top of this campaign commit.

**Whole-record result (essential finding only).** Raw-space median per-basin NSE (400-basin development-validation screening population), all 6 epochs, true per-basin paired comparisons: `PT` beats `P` in 63-71% of matched basins at every epoch (median NSE gain ≈0.03-0.06) — the one robust, repeated, basin-general improvement found anywhere in this campaign. `PT` also achieves the campaign's single strongest observed whole-record skill (epoch 3, median NSE ≈0.3726, 75,000 cumulative optimizer updates). `PTM` shows no reproducible incremental benefit over `PT` (near-zero median diff, basin-improvement fraction oscillating around the 0.5 coin-flip line across epochs). `PTMW` is a near-tie with `PT` on whole-record skill (median diff within ±0.02, fraction `PTMW`-better ≈0.43-0.57 with multiple sign flips across epochs).

**True hydrograph-overlay review.** The frozen 8-basin panel (Obs + `P` + `PT` + `PTM` + `PTMW`, shared axes, same frozen event window per basin) showed 3 of the 8 basins (`07261000`, `08072300`, `14301500`) with apparently meaningful `PTM`/`PTMW` improvement during specific high-flow events. This observation was explicitly treated as illustrative, not population-level evidence, and motivated the dedicated event audit below rather than being used directly as a selection criterion.

**High-flow/event audit (400/400 basins, 1,200 deterministically selected Q95 events, top-3/basin, 72h peak separation, 24h-before + 48h-after window, event-weighted and basin-balanced views).** Conditional analysis (flow ≥ basin Q95): `PT` vs `P` is clearly positive (58-64% of basins improve on RMSE/KGE/|PBIAS|); `PTMW` vs `PT` shows a small, checkpoint-robust edge (52-60% of basins) — real but modest. Event-level analysis: `PT` vs `P` remains positive on peak magnitude, event volume, and event shape (55-61%); `PTMW` vs `PT` is essentially a near-tie on peak magnitude and event volume (~50-51%) and only a small positive tendency on event shape — **the conditional `PTMW` edge does not translate into a broad event-level peak/volume advantage.** Peak timing is tie-dominated for both comparisons (~44-50% ties) and is not discriminative. Severity stratification (`[Q95,Q99)` vs `≥Q99`; the event-selection protocol itself skews toward severe events): **no detectable increase in `PTMW` benefit was observed across the severity strata represented by the selected event population** — this is explicitly not a claim that severity dependence has been ruled out generally, only that none was detected within this sample. Per-basin cross-check against the 8 frozen overlay basins confirmed the 3 flagged basins sit in the favorable tail of the 400-basin population, while 2 other frozen basins (`06894200`, `08061540`) show the opposite pattern (a conditional improvement that reverses into an event-level regression) — the overlays are interpretation/sanity evidence, not model-selection votes. Overlay representativeness caveat: the 8 frozen basins are not a representative sample of the 400-basin screening population and are never used as such.

**Final decision (adopted, provisional, binding for Stage 1 going forward).** `PT` — exactly `mrms_qpe_1h_mm`, `rtma_2t_K` — is **the provisional Stage-1 Dynamic-Input-Family-A working family**. It is explicitly **not** "the final optimal dynamic-input family," **not** "globally optimal," **not** "permanently superior to `PTMW`," and this closure is **not** "proof that humidity/wind do not matter." Rationale (8 points): (1) `PT` is the one robust, repeated, basin-general improvement over `P` found anywhere in the campaign; (2) `PT` achieves the campaign's strongest observed whole-record skill; (3) `PTM` shows no reproducible incremental benefit over `PT` at any epoch; (4) `PTMW` is near-tied with `PT` on general whole-record skill; (5) `PTMW`'s modest, reproducible conditional high-flow advantage does not translate into a broad event-level peak-magnitude or event-volume advantage; (6) the added moisture/wind complexity has not earned promotion over `PT` at this Stage-1 fidelity; (7) the evidence does not warrant an H256 capacity-rescue probe for any candidate; (8) `PTMW` remains documented as the nearest broader credible alternative, not dismissed as useless. `PT` epoch 3 (median NSE ≈0.3726) may be called **the best observed `PT` checkpoint in this specific Phase-A campaign** — this is explicitly not converted into a universal training-budget rule. **Dynamic-Input-Family-A closes the predictor-family decision, not the future training-duration decision.**

**Rescue policy — not exercised.** The design freeze's rescue policy (at most one standardized `hidden_size=256` capacity probe for a weak/ambiguous non-reference family) is not exercised: no candidate's result was weak or ambiguous enough to warrant it under the adopted decision above. No H256 rescue run has been launched by this entry or is authorized by it.

**Reusable evaluation capabilities retained (committed by this entry).** `render_multi_candidate_basin_panel()` (`src/baseline/hydrograph_rendering.py`) — a general-purpose true N-candidate overlay renderer (arbitrary candidate set, shared axes, observed-series consistency checks), not Dynamic-Input-specific; reviewed and judged reusable for future HPO-finalist/sequence/architecture/lead-time comparisons. `select_high_flow_events()` (`src/baseline/hydrograph_atlas_events.py`) — a deterministic, observed-only, candidate-independent high-flow event selector (explicit threshold/separation/window semantics), retained alongside the pre-existing `select_atlas_events()` (unchanged), which serves a distinct magnitude-stratum purpose and is not overwritten. `src/baseline/high_flow_event_metrics.py` (new) — `basin_high_flow_threshold()`, `high_flow_conditional_metrics()`, `event_metrics()`; small, general, documented; reuses `raw_space_metrics()` rather than reimplementing metric math. Matching test suites for all three (`tests/test_hydrograph_rendering.py`, `tests/test_hydrograph_atlas_events.py`, `tests/test_high_flow_event_metrics.py`) committed alongside. **No large new evaluation architecture was built for this closure** — these are the minimum reusable primitives the campaign already needed.

**Known cosmetic issues (documented, not fixed).** Ad-hoc/untracked analysis code produced during the event audit had a severity-breakdown CSV missing a couple of descriptive columns and inconsistent basin-ID zero-padding in one intermediate join. Both are cosmetic, confined to untracked scratch analysis code (never committed, never part of the reusable capability set above), and did not affect any reported metric. Documented here per this closure's own decision not to invest further engineering into campaign-specific scratch.

**Figure pack, evaluation evidence, and supervisor summary (project-local, gitignored, not part of this commit).** `.scratch_local/moriah_evidence/dynamic_input_family_a_closure/figures_v001/` — 8 figures (learning trajectories; paired NSE effects; fraction-of-basins-improved; high-flow conditional comparison; event-level comparison [central closure figure]; general-vs-event synthesis [explicitly not a composite score]; representative hydrograph montage drawn from the pre-existing frozen 8-basin panel; progression-diagram schematic) plus `FIGURE_INDEX.md` (question/takeaway/caveat per figure). `.scratch_local/moriah_evidence/dynamic_input_family_a_closure/SUPERVISOR_SUMMARY.md` — a plain-language, ~1-page summary (what we tested / main result / why this matters / what comes next) that does not implement or pre-empt the next milestone. `EVIDENCE_MANIFEST.md` (same directory) records source-file identity, SHA256 checksums, campaign commit, script identity, and generation timestamp for every figure input.

**Campaign-specific scratch retained untracked (not committed).** `dynfam_event_audit_runner.py`, `dynfam_event_audit.sbatch` — the event-audit driver scripts remain untracked, per this closure's decision to keep campaign-specific orchestration out of the reusable-capability commit.

**Next milestone (named only, not started, not designed by this entry).** Stage-1 Evaluation Framework v1 + Phase-B Fidelity Design. Unresolved questions carried forward unanswered: which metrics should be used routinely vs. only as diagnostics; which metrics or metric combinations should be authoritative for promoting a model configuration; how the high-flow-conditional and event-level audit methodology developed in this closure should be formalized into a standing evaluation protocol; how a higher-fidelity Phase-B training/validation protocol should be designed before the first broader hyperparameter search; whether/when an H256 (or other capacity) probe should be revisited for `PTMW` under a future higher-fidelity protocol; whether dewpoint or a both-moisture ablation should be revisited. **None of these questions are answered by this entry.**

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry + updated header), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.22), `docs/stage1_lead06_pilot_v001.md` (new 2026-08-16 closing section).

**Not done by this entry.** No new training launched, no H256 rescue run, no Phase B started, no Stage-1 Evaluation Framework v1 implemented, no sealed temporal-test/spatial-holdout/California data accessed. Generated figures/reports/evidence remain project-local and gitignored; only the reusable `src/`/`tests/` capability set and canonical documentation are committed by this entry.

## 2026-08-16 — Stage 1 — Dynamic-Input-Family-A design frozen: four-family `P`/`PT`/`PTM`/`PTMW` dynamic-input hierarchy at the `seq_length=72` anchor, gap channels removed from model inputs, dewpoint deliberately omitted from the primary hierarchy, U/V wind kept paired (documentation-only design freeze)

**Scope.** Documentation-only design-freeze task, following the accepted read-only Dynamic-Input Family Design Survey earlier in this session (full A-J report, accepted without correction) and a full-scale (2,307-basin) re-confirmation of that survey's two evidentiary audits. No Moriah/h2o access, no Slurm submission, no NeuralHydrology training or inference, no full-population validation, no experiment launched by this entry. Confirmed before any edit: clean tracked tree at `HEAD` `dda254b` ("Close Sequence-Length-A and adopt 72-hour working context").

**Purpose (adopted).** Dynamic-Input-Family-A is the next milestone on the roadmap set by the Sequence-Length-A closure (2026-08-15 entry above; `docs/stage1_validation_optimization_foundation.md` Part L.20): characterize a small number of physically meaningful dynamic-input variable families at the newly adopted `seq_length=72` working anchor. This is explicitly **not** an automated feature-selection search, **not** joint Phase-B HPO, and **not** a final choice of Stage 1's production dynamic-input set — it is a bounded, physically motivated range characterization comparing four nested variable families under an otherwise-frozen contract.

**§1A — Gap-flag audit re-confirmed at full 2,307-basin scale (`gap_flag_channel_leakage_audit_20260815.json`).** Development-training: 57,453,283 admissible windows, 0 MRMS-gap-flag-positive, 0 RTMA-gap-flag-positive, 0 issue-time-flag-positive. Development-validation: 19,528,664 admissible windows, identically all-zero. Package reconciliation: 136 canonical MRMS gap timestamps + 2 canonical RTMA gap timestamps = 138 total, symmetric difference against the package's own gap-timestamp inventory is 0.

**Gap-channel decision (adopted, binding for this campaign).** `mrms_qpe_1h_mm_gap` and `rtma_gap` remain in the certified `stage1_scientific_package_v002` package unchanged — they are **package variables**, retained for QC/provenance and for any future policy that admits gap-affected windows. Under the current hard-exclusion admission policy, both are constant-zero for every admissible development-training and development-validation input window; carrying two always-zero channels into the model's `dynamic_inputs` vector adds no information and is model-input dead weight, not a package defect. They are therefore removed from the NH model's `dynamic_inputs` vector (the **model predictor variable** set) for every Dynamic-Input-Family-A candidate. **This is not a claim that gap flags are "useless in general"** — under a different, more permissive admission policy they would carry real information; it is a claim specific to the current hard-exclusion policy's admitted-window population. **The package itself is not changed, rebuilt, or version-bumped by this decision** — this is a modeling-input choice layered on top of an unchanged package.

**§1B — Physical-variable characterization re-confirmed at full 2,307-basin scale (`dynamic_input_family_audit_20260815.json`).** Six physical `v001-core` variables (`mrms_qpe_1h_mm`, `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms`) show no sentinel, clipping, scaling, or other pathology at full scale. Moisture redundancy: global Pearson correlation between dewpoint (`rtma_2d_K`) and specific humidity (`rtma_2sh_kgkg`) ≈0.9512 (combined, 85.26M points), Spearman ≈0.9947; seasonal range ≈0.925-0.963; temperature-regime range ≈0.906-0.967; basin-level median ≈0.964, p95 ≈0.977 (development_training population). Both are directly sourced RTMA fields (neither is derived from the other). Wind: `rtma_10u_ms`/`rtma_10v_ms` are physically plausible and non-degenerate at full scale, with no evidence either component alone is redundant or uninformative.

**Moisture decision (adopted, cautious wording — binding).** `rtma_2sh_kgkg` (specific humidity) is adopted as the **primary single moisture representation** for the `PTM`/`PTMW` families. Rationale: (1) strong empirical redundancy between dewpoint and specific humidity at full scale (Pearson ≈0.95, Spearman ≈0.99, both directly sourced, neither derived); (2) simplicity — avoiding two Kelvin-valued thermal/moisture channels (`rtma_2t_K` and `rtma_2d_K`) in the smallest multi-variable families; (3) specific humidity is a directly sourced moisture mass-fraction, a natural single representative. This is a **Phase-A structural simplification, not proof dewpoint has zero predictive value** — dewpoint is not evidence-rejected, only omitted from this hierarchy's primary path. **Explicit correction, must be honored exactly going forward:** this decision must **never** be justified by treating the historical dewpoint lookup-key bug (fixed prior to this session) as a scientific disadvantage of dewpoint — the bug was fixed and is not evidence that dewpoint is inherently inferior. A both-moisture ablation (`rtma_2t_K` + `rtma_2d_K` + `rtma_2sh_kgkg` together) remains a possible, explicitly deferred, later ablation — not part of Dynamic-Input-Family-A.

**Wind decision (adopted, binding).** `rtma_10u_ms`/`rtma_10v_ms` must always travel together — no U-only or V-only family is defined or permitted. Inclusion of the wind pair is a separate, explicit structural family step (`PTMW`), not an automatic addition once moisture is included.

**Main family matrix (frozen, exact — the four Dynamic-Input-Family-A candidates).**
- **`P`** — precipitation only: `mrms_qpe_1h_mm`. Lower-bound/reference family.
- **`PT`** — precipitation + temperature: `mrms_qpe_1h_mm`, `rtma_2t_K`.
- **`PTM`** — precipitation + temperature + moisture: `mrms_qpe_1h_mm`, `rtma_2t_K`, `rtma_2sh_kgkg`.
- **`PTMW`** — precipitation + temperature + moisture + winds: `mrms_qpe_1h_mm`, `rtma_2t_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms`.

This is intentionally **5 physical channels, not 6**: dewpoint (`rtma_2d_K`) is omitted from the primary hierarchy per the moisture decision above; both gap flags are package-only/QC channels for this experiment, never a model predictor. Deferred `v001-fullmet` variables (pressure, cloud cover, visibility, gust, ceiling) are explicitly out of scope and not implemented by this campaign.

**Common anchor (frozen, unchanged from the Sequence-Length-A-closed contract — every candidate varies only `dynamic_inputs`).** Seed A (967139); `[128,32]` learned static embedding (tanh activation, embedding dropout 0.10); `hidden_size=128`; `learning_rate=3e-4`; `output_dropout=0.25`; `seq_length=72`; lead 6h; target `qobs_mm_per_h_lead06`; `max_updates_per_epoch=25000`; six training epochs; one uninterrupted segment per candidate; the fixed development-training population; the fixed ~400-basin development-validation screening population; strict real offline-W&B contract throughout. `seq_length=72` remains the provisional (not proven-optimal) working anchor established by Sequence-Length-A. Campaign identity: `dynamic_input_family_seedA_25k_v001`.

**Evaluation design (adopted, unchanged methodology from prior Phase-A axes).** Raw-space median per-basin NSE (400-basin screening population) primary. Full epoch 1-6 trajectories required for all four candidates (official cadence epochs 3/6 via the standard screening path; retrospective epochs 1/2/4/5 via the already-qualified `pilot_diagnostic_eval.py`, reused unmodified). True per-basin paired comparison (matched basin, not aggregate-only deltas) required. Late-window (epochs 4-6) behavior required, not only the epoch-6 endpoint. Transformed-space training loss remains a diagnostic only, never the official scientific model-selection metric. The frozen 8-basin `phase_a_validation_hydrograph_panel_v001` sanity check applies, using the **same** frozen event/window/scale/axes as prior Phase-A closures — **no 72h display-window widening for this campaign**: Sequence-Length-A's antecedent-window widening was specific to an experiment that itself varied historical context; Dynamic-Input-Family-A varies input *variables*, not context length, so the display convention reverts to the standard (non-widened) window.

**Rescue policy (adopted, RULE only — not exercised by this design freeze).** If a non-reference family (`PT`/`PTM`/`PTMW`) is weak or ambiguous relative to `P` at the standard `hidden_size=128` anchor, **at most one** standardized `hidden_size=256` capacity probe may be run for that specific family, as a single additional diagnostic candidate — never a second search dimension, never applied preemptively, never pre-created or pre-trained. `P` is the reference family and is **never** "rescued" — a weak result for `P` is a direct scientific finding (precipitation alone is insufficient at this fidelity), not a candidate for capacity rescue. A rescue probe is not guaranteed to change any conclusion. No rescue candidate is part of the base four-candidate allowlist; it exists only as an optional, explicitly-named future addition if a genuine ambiguity is found after the base four are evaluated.

**Deferred items (adopted, explicitly out of scope for this campaign).** Dewpoint ablation and both-moisture ablation (moisture decision above); `v001-fullmet` variables (pressure, cloud cover, visibility, gust, ceiling); longer sequence-length testing beyond the `seq_length=72` anchor; Phase B joint HPO; any sealed temporal-test/spatial-holdout/California-data evaluation.

**Minimum implementation plan (already implemented by this same session, see the implementation entry that follows this one once written).** A generic per-run `dynamic_inputs` override on `PilotRunSpec` plus a new `validate_dynamic_inputs_override()` structural validator in `nh_config_generation.py`; resolution threading through `build_pilot_bundle_with_validation_scope()`/`build_pilot_bundle()` in `pilot_lead06_config.py` (the pre-existing `validate_dynamic_inputs()` "old 8-input equality gate" is left completely unchanged as an unconditional package-integrity check, independent of any override); matching `dynamic_inputs_override`/`resolved_dynamic_inputs` provenance fields in `pilot_tracking.build_pilot_run_identity()`; a new `enforce_pilot_dynamic_inputs_identity()` always-active, W&B-independent continuation-safety guard in `pilot_orchestration.py`, mirroring the existing cap/LR/hidden-size/embedding-dropout/seq-length identity guards exactly; a closure-splice launcher/sbatch pair mirroring the Sequence-Length-A closure template, with exactly four trainable run_ids (`P`/`PT`/`PTM`/`PTMW`) and no rescue candidate in the base allowlist.

**Not launched by this entry.** No dynamic-input-family candidate has been trained, no Slurm job submitted — this entry documents the frozen design. Implementation, preparation-only qualification, and readiness assessment are recorded separately as this session's work continues.

## 2026-08-15 — Stage 1 — Sequence-Length-A closed: `seq_length=72` adopted as provisional working anchor, `seq_length=48` nearest alternative, hydrograph sanity check clean, dynamic-input family characterization next (documentation-only closure)

**Scope.** Documentation-only closure task. Sequence-Length-A execution (four real Moriah training runs, `seq_length` ∈ `{12,24,48,72}`), quantitative raw-space/common-support/paired-basin evaluation, and a dedicated hydrograph sanity check with a widened antecedent-context display window are all already complete, performed earlier in this session. This entry records closure, retains and commits the supporting evaluation/rendering code, freezes a comparative-hydrograph documentation convention, and cleans up a temporary Moriah workaround. **It launches no new training, evaluates no additional sequence length, and does not start dynamic-input-family characterization or Phase B.**

**Campaign identity.** `seq_length_range_seedA_25k_v001` — four candidates (`seq12`/`seq24`/`seq48`/`seq72`, `seq_length` = 12/24/48/72), all other settings frozen at the Embedding-Dropout-A-closed contract: Seed A (967139), `[128,32]` learned static embedding (tanh activation, embedding dropout 0.10), `hidden_size=128`, `learning_rate=3e-4`, `output_dropout=0.25`, lead 6h, target `qobs_mm_per_h_lead06`, `max_updates_per_epoch=25000`, six epochs, one uninterrupted segment, the fixed development-training population, the fixed ~400-basin development-validation screening population, strict offline W&B throughout. Campaign infrastructure committed `4646a55` ("Prepare Sequence-Length-A campaign infrastructure and Moriah launcher"); training jobs `45861222`-`45861225` (seq12/24/48/72). All four candidates completed six epochs / 25,000-update cap cleanly, with no resource failures and approximately comparable runtime across candidates.

**Quantitative result (essential finding only; full diagnostic detail lives in this session's evaluation evidence, not duplicated here).** Natural-support raw-space median NSE, a common-support-corrected evaluation, true per-basin paired comparisons, and late-window (last-epoch) behavior all show the **same ordering at every evaluated epoch: `seq72 > seq48 > seq24 > seq12`.** Transformed-space training loss shows a consistent, corroborating but diagnostic-only, non-authoritative ordering. A new secondary common-support fairness audit (`src/baseline/common_support_audit.py`, reviewed and retained by this entry) restricted the comparison to exactly the `(basin, timestamp)` positions admitted by every candidate simultaneously; the common-support correction **did not materially change** the ranking.

**Hydrograph sanity check.** The frozen `phase_a_validation_hydrograph_panel_v001` 8-basin panel (`01315000, 06894200, 07165565, 07261000, 08061540, 08072300, 12210900, 14301500`) was rendered for `seq24` (epoch 5), `seq48` (epoch 6), and `seq72` (epoch 5) — each candidate's own best observed checkpoint — with the *displayed* antecedent window widened from the panel's original frozen ~24h pre-event span to 72h, so the longest tested historical context is visible for every candidate simultaneously, via a new read-only, purely presentational `derive_display_window()` helper (`src/baseline/hydrograph_rendering.py`, reviewed and retained by this entry) that never re-selects or mutates the underlying frozen event/peak identity. MRMS QPE context shown where available. **Result: CONSISTENT with the quantitative ranking — no repeated `seq72`-specific pathology found.** A separate supplemental diagnostic rendered basin `06131200` alone (a severe outlier excluded from the main panel's aggregate story): all three candidates fail severely at this basin's May 2024 extreme event (NSE ranging −6.6 to −52.3, KGE −1.5 to −6.1 across the three candidates), but the failure is **shared across all three sequence lengths** — consistent with a near-zero-flow basin/model pathology and NSE-denominator sensitivity, not a `seq72`-specific defect. Evidence (untracked, checksum-verified `ALL_MATCH`): `tmp/sequence_length_a_hydrograph_sanity_v001/{frozen_panel,basin_06131200_diagnostic}/` (14 + 7 files), mirrored from the equivalent Moriah evidence.

**Decision (adopted, cautious — binding interpretation for the working anchor going forward).** `seq_length=72` becomes the **provisional Stage-1 working anchor** within the tested 12-72h range. `seq_length=48` remains the **nearest credible alternative**. **Performance had not clearly saturated by 72h** — 72h is explicitly **not** claimed to be the final optimum within a longer-lookback range; the upper-bound question (whether context longer than 72h would help further) remains **open**, deferred to later higher-fidelity/integrated work, not resolved or foreclosed by this closure. **No longer-lookback campaign (96/120/168h or otherwise) is authorized or launched by this entry.**

**Comparative-hydrograph convention (new, adopted for future small-candidate-set Phase-A/HPO comparisons).** For a small candidate set, prefer one panel per basin/event showing the same frozen event/time window, the same observed hydrograph, shared axes/scales per basin-event, and all candidates overlaid — not separate per-candidate figures. Rationale: direct visual comparison avoids mentally reconciling separate figures and exposes timing/magnitude/recession/false-peak/instability differences that aggregate metrics can mask. Requirements: event selection stays independent of candidate performance; the same frozen event identity (`EventWindow` — basin, peak time, peak value, `window_end`) is used for every candidate and is never re-selected; candidates are clearly identified; observed/forcing context is identical across candidates; no independent per-candidate autoscaling. For experiments that specifically vary historical context (as Sequence-Length-A did), the *displayed* antecedent window may widen to expose the longest relevant context — this widening is presentational only (via `derive_display_window()`), never a re-selection of the frozen event/peak identity. For large candidate sets, show a representative subset or another consistent layout rather than cluttering one panel. **This is a general comparison principle, not a mandate that every future panel use a 72h window** — the window widens only when the experiment itself varies antecedent context; unrelated future experiments are not required to adopt a 72h display window.

**Hydrograph provenance / marker convention (reaffirmed, applies to the Sequence-Length-A panels specifically).** The panels preserve the original frozen `phase_a_validation_hydrograph_panel_v001` event identities (basin IDs, peak times, `window_end`) exactly; only `window_start` was pulled back for display purposes. `-12h`/`-24h`/`-48h`/`-72h` markers are **nominal antecedent-context reference lines** relative to the event's physical target-valid `peak_time`, **not** the model's exact raw-input boundary — for lead=6h and sequence length `L`, the true raw input window for the sample at `peak_time` is `[peak_time − 6h − L + 1h, peak_time − 6h]`. No causal interpretation beyond this nominal framing should be drawn from the panels.

**Code retained (reviewed this closure, committed by this entry).** `src/baseline/hydrograph_rendering.py`'s `DisplayWindow`/`derive_display_window()` — read-only, additive, backward-compatible (no existing function signature changed, no default behavior changed), separate from event selection. `src/baseline/common_support_audit.py` — a narrow secondary fairness audit; reuses `nh_seed_evaluation`/`nh_raw_space_evaluation`'s metric math verbatim; adds exactly one new operation (per-basin admitted-mask intersection across candidates). Both judged in-scope, scientifically correct, and reusable — neither is campaign-specific scratch. 74 focused tests pass (`tests/test_hydrograph_rendering.py` + `tests/test_common_support_audit.py`).

**Moriah workaround cleaned up.** The hydrograph task's local `derive_display_window` diff had briefly been transferred ad hoc to Moriah (the two modified tracked files copied outside the normal commit flow) with the Moriah sbatch's tracked-file git-guard temporarily narrowed to tolerate exactly those two files. Both files are now committed and pushed normally through this entry; Moriah was re-synchronized to the committed `HEAD` via `git pull --ff-only`; the guard-narrowing lived only in an untracked sbatch script (never tracked production code) and has been reverted to its original strict form.

**Revised roadmap (unchanged from the 2026-08-13 entry's items, item 2 now closed).** (1) Reusable Phase-A/HPO campaign infrastructure consolidation — unaffected by this entry, still pending. (2) ~~Sequence-Length-A~~ — **closed by this entry.** (3) **Dynamic-input family characterization — next milestone.** First audit whether gap-flag channels carry nonzero information in scientifically admitted samples; define a small number of physically meaningful input families; compare at a common mature anchor (`seq_length=72`); use a small standardized adaptation/rescue probe before eliminating a family. (4) Phase B joint HPO, unchanged, still deferred.

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry + updated header), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.20, including the comparative-hydrograph convention), `docs/stage1_lead06_pilot_v001.md` (new 2026-08-15 closing section).

**Not done by this entry.** No Moriah/h2o access beyond the sync/verification described above, no new Slurm training submission, no evaluation of a fifth sequence length or any length beyond 72h, no dynamic-input-family characterization started, no Phase B started, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no final Stage 1 sequence-length selection beyond the provisional working anchor stated above.

## 2026-08-13 — Stage 1 — Embedding-Dropout-A closed: weak sensitivity over `0.00`–`0.40`, `drop10` retained as provisional anchor, hydrograph sanity check clean, revised Phase-A/Phase-B roadmap adopted (documentation-only closure)

**Scope.** Documentation-only closure task. Embedding-Dropout-A execution (five real Moriah training runs), quantitative analysis, reproducibility audit against the historical H=128/dropout=0.10 comparator, and the standing fixed 8-basin hydrograph sanity check are all already complete, performed earlier in this session and in the immediately preceding implementation entries (L.17/L.18 below). This entry records closure only — no training, evaluation, rendering, Moriah/h2o compute, or new analysis was performed by this task. Unchanged commit `25ec47de822265577cdd4416f1d24ffbc23bf17a` throughout (verified against local `HEAD` and `origin/master` before this update). Local evidence (untracked, gitignored, never staged): `tmp/embedding_dropout_a_closure_evidence_v001/` (quantitative packet) and `tmp/embdrop_sanity_panels_v001_evidence/drop00_drop10_drop20_v001/` (hydrograph panels, 48/48 files checksum-verified).

**Campaign identity.** `embedding_dropout_range_seedA_25k_v001` — five fresh candidates (`drop00`/`drop05`/`drop10`/`drop20`/`drop40`, `embedding_dropout` = 0.00/0.05/0.10/0.20/0.40), all other settings frozen at the LR-A/Hidden-size-A contract: `[128,32]` learned static embedding (tanh activation), Seed A (967139), `learning_rate=3e-4`, `hidden_size=128`, `output_dropout=0.25`, `seq_length=24`, lead 6h, target `qobs_mm_per_h_lead06`, `max_updates_per_epoch=25000`, six epochs, one uninterrupted segment, the fixed development-training population and the fixed ~400-basin screening-validation population. All five completed under the strict real offline-W&B contract on Moriah — training jobs `drop10` 45789423, `drop00` 45790661, `drop05` 45790662, `drop20` 45790663, `drop40` 45790664; retrospective epoch-1/2/4/5 diagnostic-evaluation jobs 45790996-45791000 (partition `catfish`, all `COMPLETED 0:0`); optimizer/update verification and paired-comparison job 45791007 (partition `glacier`, `COMPLETED 0:0`). No temporal-test, spatial-holdout, or California data accessed.

**Adopted scientific interpretation (10 points, stated cautiously — no final selection is made by this entry).**
1. Embedding dropout is weakly sensitive across `0.00`-`0.40` at this Seed-A/25k-update-cap/six-epoch Phase-A fidelity.
2. No candidate robustly dominates across epoch-6 median raw-space NSE, epochs-4-6 sustained/late-window behavior, matched-basin paired NSE differences, or the hydrograph sanity review.
3. Ranking is cadence-sensitive: `drop00` leads the epoch-6 endpoint; `drop10` has the strongest late-window summary and the best single observed checkpoint; `drop20` is among the most stable late-window candidates. Differences are comparable in size to ordinary epoch-to-epoch variation.
4. Higher embedding dropout monotonically raises transformed-space training loss, but this does **not** translate into a monotonic raw-space validation-NSE relationship.
5. `drop40` shows no validation-performance cliff at this fidelity — it must **not** be described as clearly excessive or scientifically rejected.
6. The fresh `drop10` candidate and the historical nominally-equivalent H128/dropout=0.10 run (`emb128x32_seedA_h128_lr3em4_cap25k_cal`, L.15/L.16) demonstrated exact/deterministic reproduction — identical epoch-by-epoch validation and training-loss trajectories, identical optimizer-update counts, zero paired-basin NSE difference.
7. The fixed 8-basin hydrograph sanity check (drop00 epoch 6, drop10 epoch 5, drop20 epoch 6 — identical frozen basins/event windows/target-valid timing/shared plotting scales) found broad similarity across most basins, some basin/event-specific divergences implicating different candidates in different directions (07261000 drop20 false peak; 08072300 drop10 false peak; 14301500 drop10 damped peak), and **no repeated candidate-specific hydrological pathology**. The visual evidence does not contradict the aggregate quantitative near-tie.
8. `embedding_dropout=0.10` remains the provisional working anchor. **This is not because `0.10` was proven optimal** — it remains the anchor because it lies safely inside the broad viable tested region, its sustained late-window result is strong, it exactly reproduces the nominally equivalent historical run, and the evidence gathered does not justify changing the existing anchor.
9. Final embedding-dropout selection remains deferred to joint Phase-B HPO.
10. The tested `0.00`-`0.40` region remains broadly viable at this fidelity — this result must **not** be used to aggressively narrow the future search space.

**Future infrastructure requirement (recorded, not implemented).** Reviewing the hydrograph evidence produced for this closure exposed a portability/provenance weakness, not a defect requiring regeneration of any completed evidence: PNG plot titles carry basin/checkpoint/metric identity but not necessarily campaign/candidate/run identity; generic filenames (`compact_panel.png`, `<basin_id>.png`) become ambiguous once separated from their parent directory; `compact_event_metrics.csv` carries `candidate_id` but not every artifact in a bundle is individually self-identifying. The current campaign remains fully auditable via its enclosing evidence structure, manifest, and known directory layout — nothing here is broken. This requirement is folded into roadmap item 1 below (Reusable Phase-A/HPO Campaign Infrastructure Consolidation): future portable generated artifacts/bundles must remain self-identifying when separated from their parent directory. Recommended future convention (to design later, not implemented by this entry): concise candidate/campaign/checkpoint identity visible in plot titles; full run/campaign/git/package/split/checkpoint provenance recorded in a rendering/evidence manifest; portable archive names containing campaign + candidate/run + relevant checkpoint; no reliance on parent-directory context alone for scientific provenance.

**Revised optimization roadmap (adopted, supersedes older wording that permanently excluded sequence length from Stage-B-style calibration or implied output dropout must be characterized before sequence length — see the forward-pointing notes added at `docs/stage1_validation_optimization_foundation.md` Part L.1 and Part L.10, this document's 2026-08-05 entry, `docs/FLASHNH_CURRENT_STATE.md`'s 2026-08-05 entry, and `docs/stage1_lead06_pilot_v001.md`'s sequence-length-framing passage — all preserved as historical, not rewritten).**
1. **Reusable Phase-A/HPO Campaign Infrastructure Consolidation.** Consolidate the repeated LR-A/Hidden-size-A/Embedding-Dropout-A campaign machinery so future dimensions do not require cloning ~500-line launchers; include durable artifact/evidence identity requirements (see above); keep scientific campaign definitions explicit and auditable.
2. **Sequence-Length-A.** Characterize `seq_length={12,24,48,72}` at the best-supported current anchor. Sequence length is now treated as a bounded, structural/calibratable model parameter, not permanently fixed at 24h. Lead time stays a separate axis. Primary comparison uses each candidate's naturally admissible samples; a lightweight common basin/timestamp-support audit will be added if practical, to check fairness across sequence lengths.
3. **Dynamic-input family characterization.** First audit whether gap-flag channels carry any nonzero information in scientifically admitted samples; define a small number of physically meaningful input families; compare at a common mature anchor; use a small standardized adaptation/rescue probe before eliminating a family, rather than repeating full one-dimensional HPO independently for every input family.
4. **Phase B joint HPO.** Revisit LR × hidden size × embedding dropout × output dropout jointly, per L.12's funnel. Carry multiple sequence lengths or input families into Phase B only if earlier characterization leaves genuine near-ties. Not every possible dimension requires its own exhaustive Phase-A campaign.

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry + updated header), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.19; forward-pointing correction notes added at L.1 and L.10, originals preserved), `docs/stage1_lead06_pilot_v001.md` (new 2026-08-13 closing section; forward-pointing correction note added at the existing sequence-length-framing passage, original preserved).

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no rendering, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no final Stage 1 embedding-dropout selection.

## 2026-08-11 — Stage 1 — Embedding-Dropout-A implementation and preparation-only validation complete, ready for Moriah launch review (implementation task, no launch)

**Scope.** Implementation and local/preparation-only validation of the "Minimum implementation plan" frozen by the Embedding-Dropout-A design-freeze entry immediately below. No Slurm job submitted, no real NeuralHydrology training run, no real checkpoint evaluation, no W&B Sweep, no continuation-behavior redesign, no scientific-design change, nothing committed automatically. Performed against the same unchanged commit lineage as the design freeze (`HEAD` `eea9f4c09bbfdb92b757ec4165b0bb61a7b466ba`), on branch `master`.

**Implemented, all items of the design freeze's minimum implementation plan.** (1) `PilotRunSpec.embedding_dropout: float | None = None` added to `src/baseline/pilot_lead06_config.py`, additive and default-preserving, following the `hidden_size`/`learning_rate` precedent exactly. (2) `load_pilot_policy()`: **implementation deviation from the design-freeze plan's wording, deliberate and reviewed.** Only the per-profile `statics_embedding.dropout` gate became override-aware: a run with an explicit `embedding_dropout` override reconciles that nested-profile check against the override value instead of the frozen `0.1` default; every non-overridden run keeps today's hard-equality check unchanged. The **top-level, policy-wide `embedding_dropout` gate was deliberately left strict and unchanged** — the committed `stage1_lead06_pilot_v001.yaml`'s policy-wide `embedding_dropout: 0.1` still must equal `0.1` exactly, with no override path. This is safe and requires no change because Embedding-Dropout-A's five new `PilotRunSpec`s are spliced into the already-loaded/validated base policy in memory by the closure launcher — the campaign never edits or reloads the committed policy file, so the top-level invariant is never exercised against a dropout-varying entry. Explicit candidate variation reaches the generated config only through `PilotRunSpec.embedding_dropout` → the per-profile gate above → `build_nh_config_mapping()`'s post-merge override (item 3). This preserves 100% of existing non-overridden top-level-gate behavior while enabling Embedding-Dropout-A. (3) `src/baseline/nh_config_generation.py`: `build_nh_config_mapping()` accepts an optional `embedding_dropout` override applied to the merged `statics_embedding.dropout` field after the named run-profile merge (always wins); `validate_embedding_dropout_override()` rejects non-numeric/bool/NaN/inf and enforces the `[0, 1)` bound; `GeneratedConfigBundle` gets a matching `embedding_dropout` field; `write_generated_config()`'s manifest records `embedding_dropout_override`/`resolved_embedding_dropout` — verified `0.00` is recorded as an explicit `0.0`, never confused with "no override" (`None`). No new named `_RUN_PROFILES` entry was needed; all five candidates reuse `pilot_lead06_emb128x32_seedA_v001` with `learning_rate=3e-4`/`hidden_size=128`/the candidate's `embedding_dropout` applied post-merge. (4) `src/baseline/pilot_tracking.py`: `build_pilot_run_identity()` gets matching `embedding_dropout_override`/`resolved_embedding_dropout` fields, mirroring `hidden_size_override`/`resolved_hidden_size`. (5) `src/baseline/pilot_orchestration.py`: new `enforce_pilot_embedding_dropout_identity()` continuation-safety guard (persist-on-first-call, compare-and-raise-on-mismatch, always-active, W&B-independent), following `enforce_pilot_hidden_size_identity()`'s template exactly; `run_pilot()` calls it alongside the existing cap/LR/hidden-size guards. (6) `scripts/run_stage1_embedding_dropout_range_seedA_closure.py` + `scripts/run_stage1_embedding_dropout_range_seedA_closure_moriah.sbatch`: closure-splice launcher pair mirroring the Hidden-size-A closure launcher, `EMBEDDING_DROPOUT_A_MAX_TARGET_EPOCH=6` fixed (no CLI/env override), `EMBEDDING_DROPOUT_A_RUN_SPECS` containing exactly the five new run_ids, `REFERENCE_RUN_ID="emb128x32_seedA_h128_lr3em4_cap25k_cal"` reachable only via `--status-only` and never a member of the trainable allowlist, collision guard against the real policy and all prior campaigns' run_ids (`_OTHER_CAMPAIGN_RESERVED_RUN_IDS`), default `--wandb-policy-path` pointing at the offline-enabled policy. (7) Tests across the eight planned categories: focused tests in `test_pilot_lead06_config.py`, `test_nh_config_generation.py` (33 new), `test_pilot_tracking.py` (7 new), `test_pilot_orchestration.py` (11 new identity-guard tests), plus three dedicated campaign test files — `test_run_stage1_embedding_dropout_range_seedA_closure_cli.py`, `test_embedding_dropout_range_seedA_closure_sbatch_launcher.py` (52), `test_embedding_dropout_range_seedA_closure_preparation.py`.

**Preparation-only validation (real, unmocked calls, two layers).** (a) The pytest preparation suite (`test_embedding_dropout_range_seedA_closure_preparation.py`) calls the real, unmodified `prepare_pilot_run_only()` for all five candidates against a synthetic package covering the actual full 2,557-basin development/spatial-holdout union (`tests._pilot_support.build_full_union_package`) and the real committed policy/split files; confirms pairwise config diffs are limited to `experiment_name`/basin-list paths/`run_dir`/`statics_embedding`, with `data_dir` and basin-list file contents identical across all five, and every `experiment_name`/`run_dir`/W&B run identity pairwise-unique. (b) A standalone, non-pytest audit script (`run_embedding_dropout_a_prep_audit.py`, ad hoc, not committed) additionally invoked the real `scripts/run_stage1_embedding_dropout_range_seedA_closure.py` CLI as a real subprocess with `--prepare-only` for each of the five run_ids, writing real `config.yaml`/`generation_manifest.json` files to disk. All five returned `status=PREPARED_ONLY`; spot-checked `drop00` and `drop40` configs directly: `statics_embedding.dropout` resolved to exactly `0.0` and `0.4` respectively (manifest `embedding_dropout_override`/`resolved_embedding_dropout` matching), with `hidden_size=128`, `learning_rate=0.0003`, `output_dropout=0.25`, `seed=967139`, `seq_length=24` identical across both — direct terminal evidence the frozen invariants hold and only `statics_embedding.dropout`/provenance vary.

**Explicitly confirmed unchanged from the design freeze.** No embedding-dropout candidate launched; no Slurm job submitted; no real NeuralHydrology training or checkpoint evaluation call made; no W&B Sweep/Bayesian/random-search infrastructure built; no early-stopping or continuation-behavior redesign; no scientific-design change (five dropout values, `[128,32]`, tanh activation, Seed A, `3e-4`, `hidden_size=128`, output dropout 0.25, cap 25k, six-epoch budget all unchanged); no sealed temporal-test/spatial-holdout/California access; no full-population validation; no hydrograph panel rendered; no reproducibility comparison against the historical H=128/dropout=0.10 run performed.

**Tests.** Focused suites: `test_nh_config_generation.py` (154 passed), `test_pilot_tracking.py` (42 passed), `test_pilot_orchestration.py` (102 passed, 5 skipped), `test_pilot_lead06_config.py` (52 passed). Dedicated campaign test files (130 passed) plus a wider related-suite run (455 passed across 17 files). Full local regression suite (`pytest tests/ -q`, excluding 6 pre-existing torch/neuralhydrology-dependent collection-error files unrelated to this task — `test_nh_dataset.py`, `test_nh_evaluation_check.py`, `test_nh_full_population_structural_preflight.py`, `test_nh_register.py`, `test_nh_structural_preflight.py`, `test_run_stage1_nh_entrypoint.py`): **2070 passed, 5 skipped, 1 failed** in 1141.92s. The single failure, `test_package_audit.py::test_fails_when_disk_coordinate_is_time_but_declared_schema_is_v002`, is a Windows `os.rename`/`WinError 5` file-locking race inside unrelated pre-existing package-builder atomic-promotion test infrastructure — no module touched by this task — classified as a pre-existing environment-dependent flake, not a regression.

**Git/artifact policy.** No generated config, generation manifest, preparation-result JSON, validation-result pickle, or comparison table produced by test runs is committed — all live under pytest's own `tmp_path`/`tmp_path_factory` directories or the untracked standalone-audit output directory (`C:\edA_prep_audit`, outside the repository), discarded/left untracked. Tracked changes are limited to the four modified source modules, two new scripts, five new/modified test files, and this documentation set; nothing was committed automatically.

**Final status: EMBEDDING-DROPOUT-A IMPLEMENTED — READY FOR MORIAH LAUNCH REVIEW.**

## 2026-08-11 — Stage 1 — Embedding-dropout range characterization (Phase-A) design frozen, ready for implementation (documentation-only design freeze)

**Scope.** Documentation-only design-freeze task, following the accepted read-only Embedding-Dropout Design Survey earlier in this same session. No Moriah/h2o access, no Slurm submission, no NeuralHydrology training or inference, no production Python/tests/config/policy-YAML change, no sealed temporal-test/spatial-holdout/California access, no full-population validation, no experiment launched, nothing committed automatically. Confirmed before any edit: local `HEAD` `e5c6679464160e89d597363d1e1ae24d58310893`, branch `master`, 0 commits ahead/behind `origin/master`, clean tracked tree (only the same pre-existing untracked scratch/report artifacts present, none created or touched by this task).

**Purpose (adopted).** Embedding-dropout range characterization ("Embedding-Dropout-A") is the next one-dimensional Phase-A *range characterization* on the roadmap recorded by the LR-A design freeze (2026-08-08), reaffirmed by the Hidden-size-A design freeze/closure (2026-08-09/2026-08-10): characterize the effect of the learned static embedding's `dropout` parameter at the LR-A/Hidden-size-A anchors (`learning_rate=3e-4`, `hidden_size=128`). This is explicitly **not** final embedding-dropout selection and **not** joint HPO — Phase B will later revisit embedding dropout jointly with other hyperparameters, including any interaction with learning rate or hidden size. It is explicitly **not** an optimized search grid — the five values are deliberately spaced range-characterization points, not a tuned candidate set.

**Frozen candidate set (adopted, five NEW trainable runs, all fresh — see fresh-run policy below).** Embedding dropout `0.00`, `0.05`, `0.10`, `0.20`, `0.40`; only `embedding_dropout` (the learned static embedding's `statics_embedding.dropout` field) varies. Endpoint/interior meaning, adopted for interpretation: `0.00` = no-regularization control (isolates whether embedding dropout helps at all); `0.05` = light regularization; `0.10` = the inherited historical default (origin: `_EMBEDDED_STATIC_CUDALSTM_PILOT_PROFILE`'s unremarkable first-guess FC-embedding choice, `src/baseline/nh_config_generation.py`, never itself evidence-selected — confirmed by the accepted design survey); `0.20` = moderate regularization; `0.40` = a deliberate high boundary intended to probe whether stronger embedding regularization becomes harmful at this Phase-A fidelity. This five-point set is a **range characterization, not an optimized search grid** — no interpolation or addition of further values is authorized by this entry.

**Frozen non-dropout settings (adopted, unchanged from LR-A/Hidden-size-A).** CudaLSTM, learned static embedding `[128,32]` (tanh activation — embedding *shape* and *activation* frozen; only `dropout` varies), Seed A (967139), `learning_rate=3e-4` (LR-A's provisional anchor, fixed for all five candidates, not re-tuned per dropout value), `hidden_size=128` (Hidden-size-A's provisional working anchor, fixed for all five candidates, not re-tuned per dropout value), output dropout 0.25 (unrelated regularization axis, untouched), `seq_length=24`, Adam, no scheduler, current NSE-style training loss, target `qobs_mm_per_h_lead06`, lead 6h, the fixed 2,307-basin development-training population, the fixed 400-basin development-validation screening subset, `max_updates_per_epoch=25000`, six training epochs, checkpoint every epoch, one uninterrupted `start_run()` segment epoch 1→6 per candidate (`max_target_epoch=6`, no continuation beyond epoch 6). Run_ids: `emb128x32_seedA_drop00_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop05_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop10_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop20_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop40_h128_lr3em4_cap25k_cal` (dropout value encoded as an integer percentage, no decimal point — consistent with the existing `h{H}` integer-hidden-size token; new `drop{DD}` token prepended to Hidden-size-A's fixed `h128_lr3em4_cap25k_cal` suffix, following the same "prepend the new varying dimension, carry the previously-frozen suffix forward unchanged" convention Hidden-size-A itself used when prepending `h{H}` to LR-A's `lr{X}_cap25k_cal` suffix). Campaign name: `embedding_dropout_range_seedA_25k_v001`.

**Fresh-run policy — all five candidates, including `0.10`, are fresh campaign members (adopted, binding).** Unlike LR-A's reused `1e-3` reference, **no candidate in this campaign reuses a historical run** — not even the `0.10` point, despite `0.10` being the inherited historical default already trained (with `hidden_size=128`) inside every prior campaign that never varied embedding dropout. Reasons, mirroring Hidden-size-A's fresh-H=128 rationale: (1) uniform campaign identity/provenance/tracking — all five points must share one commit, one launcher, one campaign/W&B-run-identity scheme, and one tracking contract, so the five-way comparison is not contaminated by a mixed-provenance point; (2) every prior run with `embedding_dropout=0.10` predates this campaign's own closure-splice launcher and campaign identity, even where it used the reviewed offline-enabled W&B policy (e.g. the fresh Hidden-size-A H=128 run); (3) keeping the comparison set campaign-pure (five fresh runs, one varying factor) avoids conflating this dropout range characterization with a cross-campaign reused artifact. **Historical-comparator status (adopted).** The fresh Hidden-size-A H=128 run, `emb128x32_seedA_h128_lr3em4_cap25k_cal` (`hidden_size=128`, `learning_rate=3e-4`, `embedding_dropout=0.10`, trained fresh under the mandatory tracked-W&B contract — L.15/L.16), is the closest available nominally-equivalent historical run and is retained strictly as an **optional, descriptive, read-only reproducibility comparator** against this campaign's fresh `drop10` candidate — never a sixth campaign member, never a substitute for `drop10`, never pooled into the five-candidate comparison. Its own descriptive-only reproducibility question (fresh `drop10` vs. this historical H=128 run) is explicitly deferred until after the fresh `drop10` run completes, mirroring Hidden-size-A's own deferred reproducibility question against LR-A's historical `3e-4`/H=128 run.

**Fidelity reuse and dropout-specific caveat (adopted).** Reuses the existing 25k-update-cap/six-epoch/Seed-A Phase-A fidelity unchanged (no new fidelity mechanism introduced). **Caveat, to carry into the future evidence packet's interpretation section:** poor performance at this fidelity is evidence "at this Phase-A fidelity" only, not absolute rejection of a dropout value — dropout is a regularization mechanism that can affect *optimization speed* differently than learning rate or hidden size (e.g. higher dropout may slow early convergence while still being scientifically preferable at a longer horizon, or vice versa), so a capped, six-epoch trajectory is a weaker proxy for embedding dropout's eventual effect than it was for LR or hidden size. The full six-epoch trajectory for every candidate matters more here than in prior Phase-A axes, not less.

**Evaluation design (adopted).** Raw-space median NSE (per-basin, on the 400-basin development-validation screening subset) is the primary metric, unchanged from every prior Phase-A axis. The 400-basin screening set remains an operational, non-authoritative convenience population — the full development-validation population remains the later authority for any promoted/final configuration; nothing in this entry promotes, freezes, or finalizes a Stage 1 hyperparameter. Official on-cadence screening at epochs 3 and 6 (unchanged screening policy); additionally, retrospectively evaluate epochs 1, 2, 4, 5 for all five candidates via the already-built, LR-A/Hidden-size-A-qualified `pilot_diagnostic_eval.py` (`evaluation_role="retrospective_diagnostic"`, non-authoritative, never touches early-stopping state — reused unmodified). The final evidence packet must contain full epoch 1-6 trajectories for all five dropout values; interpretation must not rely on epochs 3/6 alone, per LR-A's own cadence finding and reinforced by this entry's dropout-specific optimization-speed caveat above. Evidence to plan for (non-exhaustive, mirroring the LR-A/Hidden-size-A standard, not a fixed checklist): median and p25-p75 per-basin raw-space NSE at every epoch; `frac(NSE>0)` and similar cheap distributional diagnostics; mean training loss vs. epoch and vs. cumulative optimizer updates; exact cumulative optimizer updates by checkpoint; best-observed raw-space median NSE checkpoint per candidate; epoch-6 raw-space median NSE; the late-window (epochs 4-6) direction and median-of-medians; and true per-basin paired candidate-minus-reference NSE at every epoch against a designated in-campaign reference (median, p25/p75, fraction candidate better/reference better/tied). **No composite "winner score" and no predefined single winner-selection statistic are authorized by this entry** — consistent with the standing "no single decision statistic" rule established by LR-A and carried through Hidden-size-A.

**Standing hydrograph-panel rule (reaffirmed, unchanged, not executed by this entry).** The frozen `phase_a_validation_hydrograph_panel_v001` 8-basin panel (`01315000, 06894200, 07165565, 07261000, 08061540, 08072300, 12210900, 14301500`) remains the standing Phase-A sanity-check artifact, to be rendered after this campaign closes for the provisionally strongest tested dropout value (with a matched reference comparison where useful) — a sanity/interpretability check only, never a second optimization objective, never regenerated or reselected. This design-freeze entry does **not** render the panel; that is a later, separate closure task. **Explicitly out of scope for this entry and this campaign:** Monte-Carlo dropout, stochastic repeated inference, or any inference-time-dropout experiment — this campaign characterizes `embedding_dropout` as a *training-time* regularization hyperparameter only; inference-time behavior of dropout layers (normally disabled in eval mode) is untouched and unexamined by this design.

**W&B contract (adopted).** Adopts the Hidden-size-A standard as the campaign default: the campaign launcher must default to the reviewed offline-enabled policy (`config/stage1_wandb_tracking_policy_offline_v001.yaml`, qualified 2026-08-09) and must hard-fail a real training launch if tracking initialization fails or resolves to backend `null`/no run id, unless an explicit human waiver flag is passed. This entry only *records* the contract for this campaign; it is not implemented or launched here.

**Minimum implementation plan (adopted, not implemented by this entry).** (1) `PilotRunSpec.embedding_dropout: float | None = None` — additive, default-preserving field in `src/baseline/pilot_lead06_config.py`, following the `hidden_size`/`learning_rate` precedent exactly. (2) `src/baseline/pilot_lead06_config.py`'s `load_pilot_policy()` — the two existing hard-equality gates (top-level `embedding_dropout` check and the per-profile `statics_embedding.dropout` check) become override-aware: a run with an explicit `embedding_dropout` override reconciles validation against the override value instead of the profile default `0.1`; every other, non-overridden run keeps today's hard-equality check unchanged; threaded through `build_pilot_bundle_with_validation_scope()`/`build_pilot_bundle()`. (3) `src/baseline/nh_config_generation.py` — `build_nh_config_mapping()` accepts an optional `embedding_dropout` override, applied to the merged `statics_embedding.dropout` field after the named run-profile merge (always wins, same pattern as `learning_rate`/`hidden_size`); a new `validate_embedding_dropout_override()` rejects non-numeric/bool/NaN/inf and enforces the same `[0, 1)` bound `validate_statics_embedding_spec()` already uses; `GeneratedConfigBundle` gets a matching `embedding_dropout` field; `write_generated_config()`'s manifest records `embedding_dropout_override`/`resolved_embedding_dropout`. No new named `_RUN_PROFILES` entry is needed — all five candidates reuse the existing `pilot_lead06_emb128x32_seedA_v001` profile, applying `learning_rate=3e-4`, `hidden_size=128`, and the candidate's `embedding_dropout` override post-merge. (4) `src/baseline/pilot_tracking.py` — `build_pilot_run_identity()` gets matching `embedding_dropout_override`/`resolved_embedding_dropout` fields, mirroring the existing `hidden_size_override`/`resolved_hidden_size` fields. (5) `src/baseline/pilot_orchestration.py` — a new `enforce_pilot_embedding_dropout_identity()` continuation-safety guard plus a matching state filename, following `enforce_pilot_hidden_size_identity()`'s template exactly (persist-on-first-call, compare-and-raise-on-mismatch, always-active, W&B-independent); `run_pilot()` calls the new guard alongside the existing cap/LR/hidden-size guards. (6) New closure-splice launcher `scripts/run_stage1_embedding_dropout_range_seedA_closure.py` + matching `..._moriah.sbatch`, following the Hidden-size-A closure launcher precedent exactly: fixed `max_target_epoch=6` (no CLI/env override), a closed five-run_id allowlist, `REFERENCE_RUN_ID="emb128x32_seedA_h128_lr3em4_cap25k_cal"` (the fresh Hidden-size-A H=128 run) reachable only via `--status-only` and never trainable through this launcher, collision guard against the real policy and all prior campaigns' run_ids, default `--wandb-policy-path` pointing at the offline-enabled policy, and an explicit opt-in waiver flag to bypass `require_tracking` only with an explicit human choice. (7) Tests, minimum eight categories: embedding-dropout validation (including the `[0, 1)` bound and the `0.00` boundary case); config generation (all five candidates resolve correctly, and `drop00` correctly resolves to `dropout: 0.0` rather than being skipped as falsy); identity/provenance (manifest + run identity); continuation safety (persist/match/mismatch for embedding dropout, mirroring the existing hidden-size-identity test block); campaign allowlist (exactly five trainable run_ids, historical H=128 comparator not trainable); single-segment contract (`max_target_epoch=6`, no continuation expected); preparation-only structural comparison (five candidates differ only in `embedding_dropout` against a real synthetic package, LR/hidden_size stay fixed); W&B contract (real launcher cannot silently resolve to backend `null`). Focused tests first, then the full regression suite. **No premature generalization of the cap/LR/hidden-size/dropout identity guards into a shared abstraction is authorized by this plan** — each guard stays isolated (following the existing per-axis template) unless a future implementation task's own inspection reveals a compelling concrete reason to unify them.

**Future multidimensional HPO roadmap (unchanged, reaffirmed).** Phase A: one-dimensional range characterizations (LR-A closed; Hidden-size-A closed; Embedding-Dropout-A this entry; output dropout remains a candidate future Phase-A axis, not committed). Phase B: joint multidimensional HPO over the same axes — including any LR×hidden-size×embedding-dropout interaction — per the funnel recorded in the 2026-08-08 LR-A design-freeze entry. No change to that roadmap is made by this entry beyond adding Embedding-Dropout-A as the next concrete Phase-A instance.

**Explicit non-goals of this entry (verbatim, all observed).** No embedding-dropout candidate launched; no Slurm job submitted; no NeuralHydrology training run; no temporal-test, spatial-holdout, or California access; no full-population validation; no change to `[128,32]`, tanh activation, Seed A, `3e-4`, `hidden_size=128`, output dropout 0.25, or the six-epoch/25k-cap contract; no scheduler introduced; no LR×hidden-size×dropout joint sweep; no W&B Sweep/Bayesian/random-search implementation; no early-stopping redesign; no Monte-Carlo dropout or inference-time-dropout experiment; no hydrograph panel rendered by this entry (deferred to a later closure task); no reproducibility comparison against the historical H=128/dropout=0.10 run performed (deferred); nothing staged or committed by this task.

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.17), `docs/stage1_lead06_pilot_v001.md` (new 2026-08-11 section).

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no embedding-dropout candidate launched or run described as started, no hydrograph panel rendered, no reproducibility comparison against the historical H=128/dropout=0.10 run performed.

## 2026-08-10 — Stage 1 — Hidden-size range characterization (Phase-A) closed; validation-compatible fixed 8-basin hydrograph panel v001 frozen and accepted; standing Phase-A hydrograph rule adopted; evidence-manifest packaging bug fixed

**Scope.** Documentation and packaging-only closure task, recording results already produced by the L.15-designed hidden-size campaign (four real Moriah training runs, already executed) and a separately-executed, human-reviewed hydrograph sanity-check panel. No training, evaluation, Slurm job, config/HPO change, sealed-set access, or basin reselection performed by this task. The only code/config changes: (1) a one-line fix to an untracked evidence-assembly script's `find`/`sha256sum` manifest logic; (2) a metadata-only `status` field change (`"candidate"` → `"frozen"`) in the already-committed `scripts/generate_stage1_validation400_hydrograph_panel_selection.py`. Full technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.16; candidate-level detail unchanged from L.15/L.14 (`docs/stage1_lead06_pilot_v001.md`).

**Hidden-size Phase-A result (adopted).** Hidden size is not sharply sensitive over the tested `{64,128,256,512}` range at the current Seed-A/LR=3e-4/25k-cap/6-epoch Phase-A fidelity (epochs-4-6 median-of-medians NSE spans only ~0.255-0.278 across all four candidates, non-monotonic in hidden size). **H=128 remains the provisional working anchor** for subsequent one-dimensional Phase-A characterization — not a final winner, not an optimized value. **H=64 remains a live alternative** and should be carried into later joint Phase-B HPO (single best observed median NSE of the whole campaign, 0.2922 at epoch 6; no capacity-insufficiency signal; still-rising late trajectory). **H=256 remains a plausible upper useful capacity point.** **H=512 showed no demonstrated validation benefit at this fidelity** and is not part of the default Phase-B search space unless later joint evidence justifies revisiting it. **Preferred Phase-B hidden-size support: `{64, 128, 256}`.**

**H=128 reproducibility finding (adopted, limited interpretation).** The fresh-vs-historical H=128 audit demonstrated exact computational reproducibility (identical median NSE, byte-identical training loss and optimizer-update counts every epoch) under the nominally equivalent Seed-A configuration — but this is **not evidence of cross-seed statistical stability**; cross-seed variance remains untested and open.

**LR×hidden-size interaction.** Remains unresolved and belongs in Phase-B joint HPO; untested by this single-LR-fixed campaign.

**Validation-compatible fixed 8-basin hydrograph panel v001 — accepted (adopted).** `phase_a_validation_hydrograph_panel_v001` is accepted as a standing Phase-A review artifact after human visual review. Frozen basin IDs (do not reopen, regenerate, or replace): `01315000, 06894200, 07165565, 07261000, 08061540, 08072300, 12210900, 14301500`. Findings preserved: H=64 vs H=128 hydrographs reveal no systematic hydrological superiority of either configuration, consistent with the quantitative near-tie; LR=3e-4 shows a modest/non-dominant visual edge over LR=1e-3, consistent with LR-A; shared, not candidate-specific, model-family limitations are visible (systematic underprediction of some extreme peaks; poor representation of very flashy small-basin spikes; a shared double-peak artifact in at least one basin; shared severe-failure behavior at basin 01315000) and do not overturn the Phase-A numerical conclusions. The panel is a compact scientific sanity-check artifact, not an authoritative CONUS-representative sample and not a second optimization objective — it is geographically imbalanced (5/8 basins in `plains_missouri_south_central`, 7/8 on the `west` geo_side) and must not be described as geographically representative of CONUS. Panel status recorded as `"frozen"` in the selection driver's manifest output (was `"candidate"`); basin membership and event windows unchanged (byte-identical selection CSV before/after).

**Standing Phase-A hydrograph rule (adopted).** After each one-dimensional Phase-A characterization milestone: identify a provisionally strongest tested configuration only if the quantitative evidence supports one; render the same frozen `phase_a_validation_hydrograph_panel_v001` 8-basin panel for that configuration; include a matched reference comparison where useful; use the hydrographs as a scientific sanity/interpretability check, not an informal second optimization criterion; preserve the same basin IDs/windows across milestones.

**Evidence-manifest packaging bug (fixed, untracked script only).** Root cause: the evidence-assembly script's `find . -type f | ... | xargs sha256sum > MANIFEST_SHA256.txt` pipeline let the shell truncate/create `MANIFEST_SHA256.txt` (via `>` redirection) before `find` ran, so `find` picked up the manifest file's own momentarily-empty state and recorded a self-referential hash mismatch — cosmetic, not a real integrity failure (72/73 real content files always verified clean). Fix: exclude the manifest from its own `find` listing (`find . -type f -not -name "MANIFEST_SHA256.txt" | ...`) in `scratch_assemble_val400_evidence.sh`. Evidence packet regenerated with byte-identical scientific content; corrected manifest reports 72/72 OK; new archive SHA256 `d88990b30b9452080acf44f46b127c8ad042bdab6b73f604f3ae173cc126d104`, verified locally against `.scratch_local/moriah_evidence/phase_a_validation_hydrograph_panel_v001.tar.gz.sha256`.

**Screening-subset caveat (reaffirmed).** The 400-basin screening-validation subset used for both this campaign's fast iteration and the hydrograph panel's basin universe is a Phase-A convenience population, not scientifically authoritative. The full development-validation population remains the later authority for any promoted/final configuration.

**Not done.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no sealed temporal-test/spatial-holdout access, no final Stage 1 hyperparameter selection, no basin reselection, no embedding-dropout implementation or training.

**Next planned stage (recorded, not started).** Embedding-dropout design survey (next Phase-A one-dimensional characterization axis). Phase B later revisits LR×hidden-size×dropout interactions jointly.

## 2026-08-09 — Stage 1 — Hidden-size range characterization (Phase-A) design frozen, ready for implementation (documentation-only design freeze)

**Scope.** Documentation-only design-freeze task. No Moriah/h2o access, no Slurm submission, no NeuralHydrology training or inference, no production Python/tests/config/policy-YAML change, no sealed temporal-test/spatial-holdout/California access, no full-population validation, no experiment launched. Confirmed before any edit: local `HEAD` `785e631f0111fd352035b5b234aec4a774f4aa97`, branch `master`, 0 commits ahead/behind `origin/master`, clean tracked tree.

**Purpose (adopted).** Hidden-size-A is the next one-dimensional Phase-A *range characterization* on the roadmap recorded by the LR-A design freeze (2026-08-08 entry) and reaffirmed by the LR-A closure entry (item 12(ii), immediately below): characterize the effect of LSTM `hidden_size` at the provisional LR anchor `3e-4` adopted by LR-A. This is explicitly **not** final hidden-size selection and **not** joint HPO — Phase B will later revisit hidden size jointly with other hyperparameters, including any LR×hidden-size interaction.

**Frozen candidate set (adopted, four NEW trainable runs).** Hidden sizes `64`, `128`, `256`, `512`; only `hidden_size` varies. All other settings identical and frozen: CudaLSTM, learned static embedding `[128,32]` (tanh activation, dropout 0.1) — **not scaled with hidden size**, Seed A (967139), `seq_length=24`, output dropout 0.25, Adam, no scheduler, current NSE-style loss, target `qobs_mm_per_h_lead06`, lead 6h, the fixed 2,307-basin development-training population, the fixed 400-basin development-validation screening subset, `max_updates_per_epoch=25000`, learning rate fixed at LR-A's provisional anchor `3e-4` for every candidate (not tuned per hidden size), six training epochs, checkpoint every epoch, one uninterrupted `start_run()` segment epoch 1→6 per candidate (`max_target_epoch=6`, no continuation beyond epoch 6). Run_ids: `emb128x32_seedA_h64_lr3em4_cap25k_cal`, `emb128x32_seedA_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_h256_lr3em4_cap25k_cal`, `emb128x32_seedA_h512_lr3em4_cap25k_cal`. Campaign name: `hidden_size_range_seedA_25k_v001`.

**Fresh H=128 candidate, not a reused reference (adopted, binding — corrects an earlier proposal).** Unlike LR-A's `1e-3` reference, the campaign's H=128 point is trained fresh as `emb128x32_seedA_h128_lr3em4_cap25k_cal`, a full member of this campaign, not a reuse of the historical `emb128x32_seedA_lr3em4_cap25k_cal` (LR-A's own `3e-4` candidate, which also has `hidden_size=128`). Three reasons: (1) uniform campaign identity/provenance — all four points share one commit, one launcher, one campaign/W&B-run-identity scheme, and (critically) one tracking contract, so the four-way comparison is not contaminated by a fifth point generated under different conditions; (2) the historical run predates this campaign's mandatory offline-tracked W&B launch contract (below) and was trained with tracking disabled (backend `null`, no real run id — the LR-A closure entry's item (9) finding), so it cannot supply the same tracked evidence as the four fresh runs; (3) keeping the comparison set campaign-pure (four fresh runs, one varying factor) avoids conflating a hidden-size sweep with a cross-campaign reused artifact. The historical `emb128x32_seedA_lr3em4_cap25k_cal` run is retained strictly as a **read-only, non-pooled, non-cherry-picked reproducibility comparator** — inspected for scientific-field equivalence only, never included in the four-candidate hidden-size comparison, never relabeled as a campaign member. Its own descriptive-only reproducibility question (fresh H=128 vs. historical H=128 under nominally identical settings) is explicitly **deferred until after the fresh H=128 run completes** and is not answered by this entry or its implementation.

**W&B launch contract (new, strict requirement, adopted).** This campaign's launcher must use the reviewed offline-enabled policy (`config/stage1_wandb_tracking_policy_offline_v001.yaml`, qualified by the 2026-08-09 entry above) by default. Unlike LR-A (which silently ran with tracking disabled), a real training launch under this campaign must **hard-fail** if tracking initialization fails or resolves to backend `null`/no run id, rather than silently continuing untracked — unless an explicit human waiver flag is passed. This closes LR-A closure item (9)'s operational gap for the campaign going forward; it does not retroactively change any prior run's tracking state.

**Evaluation design (adopted).** Official on-cadence screening at epochs 3 and 6 (unchanged screening policy). Additionally, retrospectively evaluate epochs 1, 2, 4, 5 for all four candidates via the already-built, LR-A-qualified `pilot_diagnostic_eval.py` (`evaluation_role="retrospective_diagnostic"`, non-authoritative, never touches early-stopping state — reused unmodified). The final evidence packet must contain full epoch 1-6 trajectories for all four hidden sizes; interpretation must **not** rely on epochs 3/6 alone, per LR-A's own cadence finding (item 8 of the LR-A closure entry: a 3/6-only cadence missed the true best-observed checkpoint for all five LR-A candidates).

**Scientific caveats (adopted, to carry into the evidence packet's interpretation section).** (1) The `3e-4` learning rate was characterized at `hidden_size=128` only (LR-A); it may interact with hidden size, and this campaign does not test that interaction — LR is deliberately held fixed across all four candidates, not re-tuned per hidden size, and any LR×hidden-size interaction is explicitly deferred to Phase B. (2) The fixed `[128,32]` static embedding is not scaled with hidden size, so its capacity *relative to* the recurrent pathway changes across the sweep (proportionally larger relative to H=64, smaller relative to H=512); this is a deliberate simplification for a clean one-dimensional sweep, not an oversight, and must be stated as a caveat when interpreting results, especially at the extremes.

**Minimum implementation plan (adopted, not implemented by this entry).** (1) `src/baseline/pilot_lead06_config.py` — add one optional field `hidden_size: int | None = None` to the frozen `PilotRunSpec` dataclass (additive, default-preserving), following the existing `learning_rate`/`max_updates_per_epoch` precedent exactly; threaded through `build_pilot_bundle_with_validation_scope()`/`build_pilot_bundle()` and the YAML-loading path in `load_pilot_policy()`. (2) `src/baseline/nh_config_generation.py` — `build_nh_config_mapping()` accepts an optional `hidden_size` override applied after the named run-profile merge (always wins, same pattern as `learning_rate`); a new `validate_hidden_size_override()` rejects bool/non-int/zero/negative, following `validate_max_updates_per_epoch()`'s positive-int-only idiom; `GeneratedConfigBundle` gets a matching `hidden_size` field; `generate_stage1_nh_config()` threads the override through; `write_generated_config()`'s manifest records `hidden_size_override`/`resolved_hidden_size`. No new named `_RUN_PROFILES` entry is needed — all four candidates reuse the existing `pilot_lead06_emb128x32_seedA_v001` profile (LR-A's own profile), applying both the `learning_rate=3e-4` and the `hidden_size` overrides post-merge. (3) `src/baseline/pilot_tracking.py` — `build_pilot_run_identity()` gets matching `hidden_size_override`/`resolved_hidden_size` fields (mirroring the existing `learning_rate_override`/`resolved_learning_rate` fields); `init_pilot_tracking_run()` gets a new optional `require_tracking: bool = False` parameter that, when true, raises rather than silently downgrading to backend `null` (default `False` preserves 100% of existing behavior for every other caller). (4) `src/baseline/pilot_orchestration.py` — a new `enforce_pilot_hidden_size_identity()` continuation-safety guard plus `HIDDEN_SIZE_IDENTITY_STATE_FILENAME`, following the existing `enforce_pilot_cap_identity()`/`enforce_pilot_learning_rate_identity()` template exactly (persist-on-first-call, compare-and-raise-on-mismatch); `run_pilot()` calls the new guard alongside the two existing ones and threads a new `require_tracking` parameter into `init_pilot_tracking_run()`. (5) New closure-splice launcher `scripts/run_stage1_hidden_size_range_seedA_closure.py` + matching `..._moriah.sbatch`, following the LR-A closure launcher precedent exactly: `HIDDEN_SIZE_MAX_TARGET_EPOCH=6` fixed (no CLI/env override), a closed four-run_id allowlist, `REFERENCE_RUN_ID="emb128x32_seedA_lr3em4_cap25k_cal"` (the historical LR-A run) reachable only via `--status-only` and never trainable through this launcher, collision guard against the real policy and prior campaigns' run_ids, default `--wandb-policy-path` pointing at the offline-enabled policy (unlike LR-A's `None` default), and an explicit opt-in waiver flag to bypass `require_tracking` only with an explicit human choice. (6) Tests, minimum eight categories: hidden-size validation; config generation (all four candidates resolve correctly); identity/provenance (manifest + run identity); continuation safety (persist/match/mismatch for hidden size, mirroring the existing LR-identity test block); campaign allowlist (exactly four trainable run_ids, historical comparator not trainable); single-segment contract (`max_target_epoch=6`, no continuation expected); preparation-only structural comparison (four candidates differ only in `hidden_size` against a real synthetic package, LR stays exactly `3e-4`); W&B contract (real launcher cannot silently resolve to backend `null`, qualification markers testable without launching training). Focused tests first, then the full regression suite.

**Future multidimensional HPO roadmap (unchanged, reaffirmed).** Phase A: one-dimensional range characterizations (LR-A closed; Hidden-size-A this entry; embedding dropout / output dropout remain candidate future Phase-A axes, not committed). Phase B: joint multidimensional HPO over the same axes, per the funnel recorded in the 2026-08-08 LR-A design-freeze entry. No change to that roadmap is made by this entry beyond adding Hidden-size-A as the next concrete Phase-A instance.

**Explicit non-goals of this entry (verbatim, all observed).** No hidden-size candidate launched; no Slurm job submitted; no NeuralHydrology training run; no temporal-test, spatial-holdout, or California access; no full-population validation; no change to `[128,32]`, Seed A, `3e-4`, or the six-epoch/25k-cap contract; no scheduler introduced; no LR×hidden-size joint sweep; no W&B Sweep/Bayesian/random-search implementation; no early-stopping redesign; no reproducibility comparison against the historical H=128 run performed (deferred); nothing staged or committed by this task.

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.15), `docs/stage1_lead06_pilot_v001.md` (new 2026-08-09 section).

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no hidden-size candidate launched or run described as started, no reproducibility comparison against the historical H=128 run performed.

## 2026-08-09 — Stage 1 — W&B offline tracking launch-contract qualification implemented and qualified, closing item 12(i) of the LR-A closure entry below

**Scope.** Closes item (12)(i) of the LR-A closure entry immediately below: the committed W&B tracking policy stayed `enabled: false, mode: disabled` through all of LR-A because no offline-enabled policy override was ever supplied to a launcher, so every LR-A run used tracking backend `null` (item 9 of that entry). This was an operational tracking omission, not a scientific defect. This entry implements and qualifies the smallest generic, reusable fix, without touching LR-A's now-closed campaign machinery or any scientific config. Starting `HEAD`: `313c187433f4de240551db059398e56172902909`.

**(1) Reviewed offline-enabled policy override.** New `config/stage1_wandb_tracking_policy_offline_v001.yaml`: `enabled: true`, `mode: offline`, same `project`/`entity`/`tags`/`max_artifact_reference_bytes` conventions as the committed disabled default (`config/stage1_wandb_tracking_policy_v001.yaml`, left unchanged -- still what any launcher uses unless explicitly pointed elsewhere via `WANDB_POLICY_PATH`/`--wandb-policy-path`). No credentials, no machine-specific paths. `tests/test_wandb_tracking.py::test_load_tracking_policy_real_offline_config` proves it loads and resolves as an enabled offline policy matching the default's other conventions.

**(2) Generic launch-contract qualification path (deliberately not the LR-A launcher).** New `scripts/wandb_offline_launch_contract_qualification.py` + `scripts/wandb_offline_launch_contract_qualification_moriah.sbatch`, sharing no code with `scripts/run_stage1_lr_range_seedA_closure_moriah.sbatch` beyond the same commit-pin/dirty-tree/module-load boilerplate every Stage 1 Slurm job uses. This is a distinct qualification from the prior wrapper-level smoke (`scripts/wandb_real_offline_qualification_smoke.py`, `docs/stage1_wandb_user_guide.md` status item 2): that script proved the wrapper against a hand-built policy dict, never through a real launcher's own env/CLI policy-path selection. The new script: resolves the policy path via `--wandb-policy-path`/`WANDB_POLICY_PATH` (refuses to run if neither is given -- no silent fallback to the disabled default); loads it with the real `load_tracking_policy`; starts a real run via the generic, pilot-independent `init_tracking_run` (deliberately not `init_pilot_tracking_run`, which requires a full scientific `PilotPolicy` this qualification must never touch); tags the run identity unmistakably non-scientific (`qualification_kind: "wandb_offline_launch_contract"`, `launch_contract_qualification: true`); logs two tiny synthetic values; finishes cleanly; writes a checked evidence record. It never imports neuralhydrology or torch, never generates an NH config, never loads a basin/split/target package, never touches a sealed set -- nothing in its import graph is capable of any of that. `tests/test_wandb_offline_launch_contract_qualification.py` (7 tests, fake-wandb backend) and `tests/test_wandb_offline_launch_contract_qualification_sbatch_launcher.py` (23 tests, static/`bash -n`/extracted-fragment execution, mirroring `tests/test_lr_range_seedA_closure_sbatch_launcher.py`'s convention) cover both. One bug found and fixed during implementation: `TrackingRun.wandb_run_id` only echoes back a caller-*supplied* `run_id` (see `src/baseline/wandb_tracking.py`); since this script never supplies one, the real backend-generated id is read directly off the underlying `wandb` Run object instead.

**(3) Local qualification (real `wandb`, ephemeral venv).** Ran against the real, locally-installed `wandb` 0.28.1 (installed into a throwaway venv at `C:/wandb_qual_venv`, removed afterward -- the base local Python has no `wandb`). Result: `backend="wandb"`, `mode="offline"`, run id `em21le9y`, all 9 checks true (`policy_enabled_true`, `policy_mode_offline`, `backend_is_wandb`, `wandb_run_id_non_null`, `run_finished_cleanly`, `offline_run_files_created`, `no_online_dependency`, `qualification_identity_non_scientific`, `no_scientific_config_generated`). Real offline run files created under a throwaway local `WANDB_DIR`; no network call.

**(4) Commit and push.** All 6 new/changed files (the policy YAML, the qualification script, its tests, the sbatch, the sbatch's tests, the extended `tests/test_wandb_tracking.py`) committed as `fb2d6ae773993e8dd5a8cde65894fda14f5b4df7` on top of `313c187`, pushed to `origin/master`. Full local suite (excluding the pre-existing torch/`neuralhydrology`-dependent modules, which require a different environment): 1733 passed, 5 skipped, 1 unrelated pre-existing flaky Windows test (`test_package_builder.py::test_manifest_package_role_independent_of_basin_count`, a `WinError 5` file-rename race in code this entry never touched -- confirmed passing in isolated re-run).

**(5) Moriah qualification (real `wandb`, CPU-only `glacier` partition, ephemeral venv).** Moriah's clone pulled `bc8f253` -> `fb2d6ae` (fast-forward, picking up both `313c187` and this entry's commit). Slurm job `45775192` (partition `glacier`, no GRES, 2 CPU/4G/15min, `EXPECTED_COMMIT=fb2d6ae773993e8dd5a8cde65894fda14f5b4df7`) completed `COMPLETED 0:0` in 00:01:09. Commit pin verified (`HEAD` matched, tracked tree clean). Result: `backend="wandb"`, `mode="offline"`, run id `8hhayk8n`, all 9 checks true, real offline run files created under `/sci/labs/efratmorin/omripo/Flash-NH/evidence/wandb_offline_launch_contract_qualification_45775192/wandb_dir/`. The job's ephemeral venv (`wandb`+`pyyaml`, never installed into the shared `flashnh-moriah` conda env) was created and removed within the job.

**(6) Evidence transfer and checksum verification.** Compact evidence directory (720K, 6 files: `qualification_record.json`, `qualification_human.log`, and the real offline run's `.wandb`/`requirements.txt`/2 log files) transferred via `scp -O` to `.scratch_local/moriah_evidence/wandb_offline_launch_contract_qualification_45775192/` (untracked, gitignored per the existing `.scratch_local/` convention). SHA256 of all 6 files verified byte-identical between the Moriah-side `sha256sum` manifest and the local copy.

**(7) Future-launch contract (adopted, recorded, not enforced by code).** Every future real Flash-NH experiment launcher must explicitly provide the reviewed offline-enabled policy (`config/stage1_wandb_tracking_policy_offline_v001.yaml` or a later reviewed equivalent) via `WANDB_POLICY_PATH`/`--wandb-policy-path`, unless W&B tracking is explicitly waived and that waiver is documented in the launch's own record. Pre-launch qualification for any such launcher should verify: `enabled=true`, `mode=offline`, backend `wandb` (not `null`), and a non-null W&B run id -- exactly the checks this entry's qualification script already performs generically. This is a recorded expectation for future launchers to satisfy; no enforcement/gating code is added by this entry, and no existing launcher (including the now-closed LR-A launcher) is modified.

**Not done by this entry.** No hidden-size or other HPO screening started; no new training/evaluation run; LR-A's launcher/campaign machinery (`scripts/run_stage1_lr_range_seedA_closure*.py`/`.sbatch`) untouched; no scientific config generated or changed by the qualification script (structural, by import graph); no sealed temporal-test/spatial-holdout/California access; `pilot_tracking.py`'s pilot-specific composition layer untouched and unused by the new qualification path.

**Documentation changes made by this entry.** `docs/decision_log.md` (this entry), `docs/stage1_wandb_user_guide.md` (new status item 6), `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry).

## 2026-08-09 — Stage 1 — LR-A (bounded learning-rate range characterization) closed: range evidence recorded, `3e-4` adopted as provisional Phase-A working anchor, cadence and W&B findings documented (documentation-only closure)

**Scope.** Documentation-only closure task recording the completed, Moriah-executed LR-A campaign and its already-transferred, checksum-verified compact evidence packet. No training, evaluation, Slurm job, W&B sync, package generation, or new analysis was run by this task; no source code was modified by this task. Authoritative source state: design freeze commit `f300cb9` ("Freeze LR-A design: bounded learning-rate range characterization") and implementation commit `bc8f253bed9231fc4a98233ffb2b92b16af8f743` ("Implement Stage 1 LR-A range characterization"), both already merged to `master` prior to this entry.

**(1) Design and implementation complete.** The five-candidate LR-A design (frozen `f300cb9`) and its minimum implementation plan (`bc8f253`) are both complete and merged; no further code changes were required to execute the campaign.

**(2) Four new 25k runs complete; reused `1e-3` reference confirmed.** All four new candidates — `emb128x32_seedA_lr1em4_cap25k_cal` (1e-4), `emb128x32_seedA_lr3em4_cap25k_cal` (3e-4), `emb128x32_seedA_lr3em3_cap25k_cal` (3e-3), `emb128x32_seedA_lr1em2_cap25k_cal` (1e-2) — completed training on Moriah under the frozen contract: `[128,32]` learned static embedding, Seed A (967139), `hidden_size=128`, embedding dropout 0.1, output dropout 0.25, Adam, no scheduler, `seq_length=24`, target `qobs_mm_per_h_lead06`, lead 6h, canonical 2,307-basin training population, canonical 400-basin screening subset, `max_updates_per_epoch=25000`, six epochs, checkpoint every epoch. The `1e-3` candidate (`emb128x32_seedA_cap25k_cal`) is the reused historical reference adopted by the design freeze's reuse-equivalence audit — never retrained for this campaign; its package/scientific identity was re-confirmed against the same frozen commit lineage during evidence assembly.

**(3) 30/30 checkpoint-evaluation matrix complete.** All five candidates evaluated at all six epochs (5 × 6 = 30 cells), using the diagnostic-evaluation helper (`pilot_diagnostic_eval.py`) built by the implementation task. Epochs 3 and 6 carry `evaluation_role="official"` (on-cadence); epochs 1, 2, 4, 5 carry `evaluation_role="retrospective_diagnostic"` (off-cadence, non-authoritative). Early-stopping state was never touched by the retrospective evaluations (structural property of the helper, confirmed by the implementation task's own tests).

**(4) Optimizer-update evidence closed.** Exactly six epochs and 150,000 cumulative optimizer updates were verified for every candidate (25,000/epoch × 6), confirming the fixed-budget fairness rule was honored uniformly across all five candidates including the reused reference.

**(5) Scientific result — range characterization, not final LR selection (adopted interpretation).** Epoch-6 median raw-space NSE ordering across the 400-basin screening subset: `3e-4 (0.268) > 1e-4 (0.259) > 1e-3 (0.253) > 3e-3 (0.178) > 1e-2 (0.021)`. `3e-4` shows a positive median paired NSE difference against the `1e-3` reference at all six epochs and is better on approximately 55-68% of the exactly-matched 400 basins depending on epoch (all 24 paired-comparison rows in the evidence packet use exact 400/400 basin matching). `3e-3` and `1e-2` are worse than the `1e-3` reference at every epoch in matched-basin comparison (`1e-3` better on ~76-92% of basins vs. those two). Training-loss trajectories corroborate the optimization interpretation: `1e-4`, `3e-4`, and `1e-3` show normal decreasing loss; `3e-3` and especially `1e-2` show non-monotonic/elevated loss consistent with too-large step sizes. **Adopted conclusion:** the scientifically useful LR region for this model family, at this fidelity, is approximately `1e-4` to `1e-3`. `3e-4` is the strongest tested interior point and is adopted as the **provisional Phase-A working anchor** — not a final selected learning rate. `1e-4` is broadly competitive with `1e-3` but plateaus early (best_observed_epoch=2). `3e-3` is clearly too high for this model family under this fidelity; `1e-2` is decisively poor/unstable (both best_observed_epoch=1, i.e. best performance at the very first checkpoint, consistent with early divergence from too-large steps). There is no evidence from this campaign that the LR interval should be extended below `1e-4` or above `1e-2`. **This is not proof that `3e-4` is globally optimal**; Phase B will later revisit learning rate jointly with other hyperparameters rather than freezing `3e-4` permanently.

**(6)/(7) Carried-forward range and provisional anchor (adopted).** Carried-forward LR range for future Phase-A/Phase-B work: `1e-4`-`1e-3`. Provisional Phase-A working anchor: `3e-4`. Neither constitutes a final Stage 1 learning-rate selection.

**(8) Evaluation-cadence finding (adopted).** A 3/6-only evaluation cadence (the standard on-cadence screening schedule) would have missed the true best-observed checkpoint for all 5/5 candidates in this sweep (best_observed_epoch values: `lr1em4`=2, `lr3em4`=5, `ref1em3`=4, `lr3em3`=1, `lr1em2`=1 — none fall on 3 or 6). A denser 2/4/6 cadence would have recovered the true best-observed epoch for only 2/5 candidates (`lr1em4` and `ref1em3`). This does **not** imply every future HPO run must evaluate every epoch; it **does** imply a 3/6-only cadence is too sparse for short 25k-update screening trajectories whenever checkpoint localization or trajectory shape matters. Adopted recommendation for future broad HPO: use denser evaluation or a sustained-performance objective rather than relying on a single endpoint alone.

**(9) W&B operational finding — omission, not a scientific issue (adopted).** All four new LR-A runs used the committed default tracking policy (`enabled: false`, `mode: disabled`) with the launcher exporting `WANDB_MODE=offline`; no `WANDB_POLICY_PATH` offline-enabled override was supplied. The Python tracking layer therefore used backend `null` for all four runs — no real W&B run IDs were created. This did **not** affect scientific validity: all evaluation evidence was produced directly from checkpoint files and raw-space metric computation, independent of the tracking layer. This is recorded as an **operational tracking omission**, to be corrected by future policy (see item 12), not as a defect in the LR-A scientific evidence itself. No W&B change is implemented by this closure entry.

**Audit incidents (resolved, no contamination of final evidence).** (a) First diagnostic-evaluation attempt: the temporary diagnostic sbatch script did not `cd "$REPO"` before invoking Python, so the relative screening-subset basin-list path failed to resolve from Slurm's default working directory; all five jobs failed within ~9 seconds, before any GPU/NH inference began. The launcher was corrected (added `cd "$REPO"`) and resubmitted; all five diagnostic jobs then completed successfully (exit 0:0). (b) Evidence-builder bug: the evidence-assembly script initially read the per-epoch median NSE using key `"median"` instead of the actual key `"p50"` produced by `primary_metric_distribution`, yielding `None` medians and degenerate/crashing trajectory and cadence summaries. The script was corrected (`"median"` → `"p50"`), the full build+plot pipeline was re-run, and the resulting values were cross-checked against real per-epoch numbers already observed directly in job stdout logs. Both incidents are resolved workflow/audit issues internal to evidence assembly; the final scientific evidence packet reflects only the corrected pipeline.

**(10) Compact evidence packet (durable local copy, checksum-verified).** Durable local directory: `.scratch_local/moriah_evidence/lr_a_five_lr_evidence_v001/`; durable local archive: `.scratch_local/moriah_evidence/lr_a_five_lr_evidence_v001.tar.gz`; SHA256 `624c5df4e1823e00b00a303a1c577790c3a72005cc217fcee5dc3e65f186f61c` (verified against the Moriah-side archive and re-verified on the durable local copy). Manifest verification (`sha256sum -c MANIFEST_SHA256.txt` against the extracted directory): all 23 files OK. The packet is untracked and gitignored under the repository's existing `.scratch_local/` local-scratch convention (nested `.gitignore`); it is not staged or committed by this entry.

**(11) LR-A closed.** Stage 1 LR-A (bounded learning-rate range characterization) is closed as of this entry. No further LR-A runs are planned; any future learning-rate work belongs to Phase B (joint multidimensional HPO) or a later, separately-approved Phase-A range characterization.

**(12) Next planned stage (adopted, not started by this entry).** (i) A small W&B offline-tracking launch-contract fix/qualification — either an explicit reviewed offline-enabled W&B policy override, or an explicit documented waiver of W&B tracking, adopted before the next experiment launch. (ii) The next Phase-A one-dimensional range characterization, likely hidden size. (iii) Joint Phase B multidimensional HPO later, per the roadmap recorded in the 2026-08-08 design-freeze entry above. No final learning-rate freeze is made by this entry.

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.14, "Parts and status" table updated), `docs/stage1_lead06_pilot_v001.md` (new closing section with final per-candidate status table).

**Not done by this entry.** No training, evaluation, Slurm job, W&B sync, package generation, or new analysis was run; no source code was modified; no generated evidence was staged or committed; `.scratch_local/` remains gitignored and untouched by git; no final Stage 1 learning-rate selection is made; the W&B tracking-contract fix is not implemented here.

## 2026-08-08 — Stage 1 — LR-A implementation and preparation-only validation complete, ready for Moriah launch review (implementation task, no launch)

**Scope.** Implementation and local/preparation-only validation of the "Minimum implementation plan" (Task D) frozen by the LR-A design-freeze entry immediately below. No Slurm job submitted, no real NeuralHydrology training run, no real large checkpoint evaluation, no W&B Sweep, no general Bayesian/random HPO, no continuation-behavior redesign, no scientific-design change, nothing committed automatically. Performed against the same unchanged commit lineage as the design freeze (`HEAD` at task start `9b3b56f7dd68e876c9d02c8a6e5993698b0a9437`), on branch `master`.

**Implemented, all items of the design freeze's minimum implementation plan.** (1) `PilotRunSpec.learning_rate: float | None = None` added to `src/baseline/pilot_lead06_config.py`, additive and default-preserving. (2) `src/baseline/nh_config_generation.py`: `build_nh_config_mapping()` accepts an optional `learning_rate` override, applied after the named run-profile merge so it always wins; `validate_learning_rate_override()` rejects non-numeric/bool/zero/negative/NaN/inf; `write_generated_config()`'s `generation_manifest.json` now explicitly records `learning_rate_override` and `resolved_learning_rate`. (3) `src/baseline/pilot_diagnostic_eval.py`: `evaluate_diagnostic_checkpoint()`/`evaluate_all_diagnostic_checkpoints()` evaluate any epoch (1-6) via the existing epoch-agnostic `ensure_validation_results()` + `raw_space_metrics_for_run_period()` primitives, tagging off-cadence epochs `evaluation_role="retrospective_diagnostic"` (`authoritative=False`, `stopping_eligible=False`) and on-cadence epochs (3, 6) `evaluation_role="official"` via delegation to `pilot_screening_eval.evaluate_screening_checkpoint()`. Confirmed by direct test (`"record_screening_event" not in vars(pilot_diagnostic_eval)`) that this module never touches early-stopping state. (4) `scripts/run_stage1_lr_range_seedA_closure.py` + `scripts/run_stage1_lr_range_seedA_closure_moriah.sbatch`: closure-splice launcher pair mirroring `run_stage1_cap50k_closure.py`, `LR_A_MAX_TARGET_EPOCH=6` fixed (no CLI/env override), `LR_A_RUN_SPECS` containing exactly the four new run_ids, `REFERENCE_RUN_ID="emb128x32_seedA_cap25k_cal"` reachable only via `--status-only` and never a member of the trainable allowlist, collision guard against the real six-run matrix and the reference. (5) `src/baseline/checkpoint_comparison.py`: `build_n_vs_one_comparison()` (N-vs-1-reference per-(candidate,epoch) table, reusing `pilot_diagnostic_eval`'s already-certified payloads with no new metric math), `derive_trajectory_summary()` (late-window epochs 4-6 direction, best-checkpoint identification — deliberately returns no `score`/`rank`/`winner`/`composite_score`/`is_best` key, preserving the design freeze's "no single decision statistic" rule), `cadence_sensitivity_view()` (all-epoch vs. sparser-cadence comparison). (6) Tests: `tests/test_checkpoint_comparison.py` (32 tests), `tests/test_pilot_diagnostic_eval.py` (11 tests), `tests/test_run_stage1_lr_range_seedA_closure_cli.py` (46 tests), `tests/test_lr_range_seedA_closure_sbatch_launcher.py` (44 tests), `tests/test_lr_range_seedA_closure_preparation.py` (15 tests, Section 14 below) — all passing; combined with the pre-existing `test_pilot_orchestration.py`/`test_pilot_lead06_config.py`/`test_nh_config_generation.py`/`test_pilot_screening_eval.py` suites, full regression green.

**Preparation-only validation (Task D item 6's config/manifest correctness check, done via real code, no mocking).** For each of the four new candidates, called the real, unmodified `prepare_pilot_run_only()` against a real synthetic package covering the actual full 2,557-basin development/spatial-holdout union (`tests._pilot_support.build_full_union_package`) and the real committed `stage1_lead06_pilot_v001.yaml`/baseline policy/split files. Confirmed for all four: `hidden_size=128`, `output_dropout=0.25`, Adam, no scheduler, `seq_length=24`, target `qobs_mm_per_h_lead06`, `epochs=6`, `max_updates_per_epoch=25000`, seed 967139, `statics_embedding={hiddens:[128,32], activation:"tanh", dropout:0.1}`, and each candidate's own frozen `learning_rate` — with explicit LR provenance in `generation_manifest.json`'s `learning_rate_override`/`resolved_learning_rate` fields. Pairwise config-mapping diffs across all four confirm the only ever-differing keys are `learning_rate` plus unavoidable identity/path metadata (`experiment_name`, the three basin-list file paths, `run_dir`); `data_dir` (shared package_root) is identical; basin-list file *contents* (not just paths) are byte-identical across all four; `experiment_name`/`run_dir`/W&B run identity are all pairwise-unique; `training_started`/`evaluation_started`/`wandb_backend_initialized` are `False` for every candidate. One additional, locally-synthesized, clearly-labeled `PilotRunSpec` (run_id `emb128x32_seedA_lr1em3_structural_comparison_only`, `learning_rate=1e-3`, never the literal `REFERENCE_RUN_ID` string) was generated through the identical real pathway solely to confirm the local code path injects `learning_rate` as the only scientific difference between siblings — this makes **no claim** of reproducing the real, historical, Moriah-trained `emb128x32_seedA_cap25k_cal` reference itself, which remains external and read-only (see the 2026-08-08 design-freeze entry's reuse-equivalence audit and its one flagged-not-blocking caveat about that reference's own generation manifest never having been independently re-read — still true, unresolved, and non-blocking after this task).

**Explicitly confirmed unchanged from the design freeze.** No LR candidate launched; no Slurm job submitted; no real NeuralHydrology training or checkpoint evaluation call made; no W&B Sweep/Bayesian/random-search infrastructure built; no early-stopping or continuation-behavior redesign; no scientific-design change (five LR values, `[128,32]`, Seed A, cap 25k, six-epoch budget all unchanged from the design freeze); no sealed temporal-test/spatial-holdout/California access; no full-population validation; no hydrograph package generated.

**Git/artifact policy.** No generated config, generation manifest, preparation-result JSON, validation-result pickle, checkpoint, log, W&B offline directory, or comparison table produced by this task's test runs is committed — all live under pytest's own `tmp_path`/`tmp_path_factory` directories, discarded after each test run. Tracked changes are limited to the source modules, the two new scripts, the new/updated test files, and this documentation set; nothing was committed automatically — see this task's own final report for the exact reviewed diff.

**Final status: LR-A IMPLEMENTED — READY FOR MORIAH LAUNCH REVIEW.**

## 2026-08-08 — Stage 1 — LR-A (bounded learning-rate range characterization) design frozen, ready for implementation (documentation-only design freeze)

**Scope.** Documentation-only design-freeze task. No Moriah/h2o access, no Slurm submission, no NeuralHydrology training or inference, no production Python/tests/config/policy-YAML change, no sealed temporal-test/spatial-holdout/California access, no full-population validation, no experiment launched. Confirmed before any edit: local `HEAD` `9b3b56f7dd68e876c9d02c8a6e5993698b0a9437`, branch `master`, 0 commits ahead/behind `origin/master`, clean tracked tree (only pre-existing untracked scratch/report artifacts present, none created or touched by this task).

**Purpose (adopted).** LR-A is a one-dimensional learning-rate *range characterization*, not a five-candidate tournament and not final Stage 1 learning-rate optimization. Two goals: (1) characterize the useful learning-rate region around the current `lr=0.001` baseline; (2) use full six-epoch trajectories across the range to help design checkpoint cadence and a robust trajectory-summary objective for the later multidimensional HPO phase (Phase B, below). LR-A is explicitly Phase A of a two-phase roadmap; see "Future multidimensional HPO roadmap" below.

**Frozen candidate range (adopted, five values).** `1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2`. All other settings identical and frozen: CudaLSTM, learned static embedding `[128,32]` (per the 2026-08-06 entry's adopted working default), Seed A (967139), `seq_length=24`, `hidden_size=128`, embedding activation tanh, embedding dropout 0.1, output dropout 0.25, Adam, no scheduler, NSE-style loss, target `qobs_mm_per_h_lead06`, lead 6h, the fixed 2,307-basin development-training population, the fixed 400-basin development-validation screening subset, `max_updates_per_epoch=25000`, six training epochs, checkpoint every epoch. Assuming the qualified capped-update mechanism (2026-08-04 entry) behaves as already verified, this yields exactly 150,000 cumulative optimizer updates per candidate. Intended execution: one uninterrupted `start_run()` segment epoch 1→6 per candidate via `max_target_epoch=6` (the existing `chunk_epoch_targets` mechanism yields a single target `[6]` for this budget — confirmed by code read, no source change needed for this property). **Fixed-budget fairness rule (adopted, binding).** No candidate may be continued past epoch 6 within LR-A merely because its trajectory looks promising; every candidate receives exactly the same fixed six-epoch budget. Any follow-on higher-fidelity or extended-budget run for a promising candidate is a separate, later, explicitly-approved phase, not part of LR-A itself.

**All-epoch (1-6) evaluation design and audit finding (adopted design; genuine minimal gap identified, not yet implemented).** Direct code audit of `src/baseline/pilot_screening_eval.py`, `src/baseline/pilot_orchestration.py`, and `src/baseline/nh_seed_evaluation.py` establishes: the "official" screening wrapper `pilot_screening_eval.evaluate_screening_checkpoint()` (via `classify_screening_epoch_role()`) structurally rejects any epoch that is not a multiple of `screening_validation_every_n_epochs` (3) — i.e. it raises `PilotScreeningEvalError` for epochs 1, 2, 4, 5 under the standard cadence. The two lower-level primitives it is built from, `pilot_orchestration.ensure_validation_results(nh_run_dir, epoch, evaluate_checkpoint_fn=...)` and `nh_seed_evaluation.raw_space_metrics_for_run_period(...)`, are both **epoch-agnostic** — neither has a cadence gate, confirmed by direct code read. The early-stopping-state mutator `record_screening_event()` is called only immediately after `evaluate_screening_checkpoint()`, itself only reached from inside the `chunk_epoch_targets()` boundary loop (epochs 3, 6, 9, ... only) — it is never reachable for epochs 1, 2, 4, 5 in the existing flow. Therefore: (a) evaluating all six LR-A checkpoints per candidate is possible today by calling `ensure_validation_results()` + `raw_space_metrics_for_run_period()` directly for epoch in 1..6, bypassing `pilot_screening_eval`'s wrapper for the four off-cadence epochs; (b) this reuses the same scalers, basin-membership validation, and raw-space evaluation code as every official metric in the repository — it is ordinary NH inference against saved checkpoints, not a second evaluation implementation; (c) it cannot mutate early-stopping/checkpoint-selection state, because `record_screening_event()` is never in its call path; (d) no continuation boundary is required — a single uninterrupted epoch-1→6 segment (`save_weights_every=1` by profile) already produces all six checkpoints, and the evaluation loop runs entirely after training completes, against the same `nh_run_dir`. **Genuine minimal gap:** no committed, reusable function performs step (a) today — the 2026-08-05/06 retrospective evaluations of epochs 1/2/4/5 were done via untracked, Moriah-local, ad hoc scripts, never added to the repository (confirmed by local search: no `tmp/operations/` directory, no dedicated retrospective-evaluation script under `scripts/` or `src/baseline/` exists locally). This is the single required implementation item for the next milestone (see "Minimum implementation plan" below); it is a small, additive helper, not a redesign of the evaluation system.

**No single decision statistic (adopted, binding).** LR-A does not define a winner by epoch-6 median NSE alone, best-checkpoint alone, six-epoch average alone, or any newly invented composite score. The evidence packet must report, per candidate, at minimum: median and p25-p75 per-basin raw-space NSE at every epoch 1-6; `frac(NSE>0)` at every epoch (plus other already-standard cheap distributional diagnostics, e.g. `frac(NSE>0.3)`, `frac(NSE<-1)`); mean training loss vs. epoch and vs. cumulative optimizer updates; exact cumulative optimizer updates by checkpoint; best observed raw-space median NSE checkpoint; epoch-6 raw-space median NSE; median and range/variability of raw-space median NSE over epochs 4-6; the late-trajectory (epochs 4-6) direction; and, for every non-reference candidate, true per-basin paired candidate-minus-reference NSE at every epoch (median, p25/p75, fraction candidate better, fraction reference better, tied fraction), reference = `lr=1e-3`. No automatic collapse of these diagnostics into one composite score is authorized by this entry.

**Interpretation questions frozen into the future review design (not answered by this entry).** (1) any LR clearly unstable/divergent/grossly under-training; (2) a clearly poor LR region; (3) an interior LR region with consistently stronger performance; (4) whether a promising region sits at a tested boundary, implying range extension; (5) which apparent advantages persist across several checkpoints vs. a single epoch; (6) whether any candidate is still materially improving by epoch 6; (7) whether epoch-6 ordering agrees with epochs-4-6 behavior, paired per-basin behavior, and the broader trajectory; (8) whether evaluating only epochs 3/6 would have changed the interpretation; (9) whether a 2/4/6 cadence would have materially changed it; (10) whether six capped epochs suffice for later broad HPO screening; (11) what LR interval should carry into Phase B; (12) whether future automated HPO should use an endpoint metric, a late-window summary, or another sustained-performance summary — deliberately not pre-answered; to be derived from LR-A evidence once it exists.

**`lr=1e-3` reuse-equivalence audit (Task B) — REUSE ELIGIBLE (adopted finding).** `emb128x32_seedA_cap25k_cal` (trained and closed in the 2026-08-05 embedding-shape neighborhood screening; that entry documents `[128,32]` as one of "three new capped-update candidates" trained from a fresh Seed-A initialization) matches, field for field, every scientific/training specification LR-A requires of its `1e-3` reference: `[128,32]` learned static embedding, Seed A (967139), fresh initialization, `hidden_size=128`, embedding activation tanh, embedding dropout 0.1, output dropout 0.25, Adam `lr=0.001`, NSE loss, `seq_length=24`, target `qobs_mm_per_h_lead06`, lead 6h, the fixed 2,307-basin training population, the fixed 400-basin screening subset, `max_updates_per_epoch=25000`, six completed epochs with checkpoints 1-6 and no epoch 7, offline W&B (`tracking_generation=g1`), no sealed-population access. `git log --oneline 5aba586..9b3b56f` restricted to the eight config-generation/orchestration/evaluation source and config files relevant to candidate identity and evaluation returns **zero commits** — no functionally relevant drift between the commit that produced this candidate (`5aba586dc4856ecb05945b41d3ff29a34f096cb7`) and current `HEAD` (`9b3b56f7dd68e876c9d02c8a6e5993698b0a9437`). The candidate's membership in the embedding-shape experiment is incidental to its scientific identity, which is fully determined by its frozen hyperparameter/data/code-version tuple, not by which experiment commissioned it. **Reuse disposition:** `emb128x32_seedA_cap25k_cal` is adopted as the LR-A `1e-3` reference without retraining. It must appear in the LR-A comparison packet under its original `run_id` and original W&B/evidence identity, explicitly annotated as a reused reference predating the `lr_range` campaign — never relabeled or presented as trained under the new campaign. One low-risk, unverified-in-this-session item is flagged: the exact `package_root`/package-identity value recorded in that run's own generation manifest was not independently re-read this session (assumed consistent with `stage1_scientific_package_v002` by launcher-default convention); cheap to confirm before use, not a blocker.

**Candidate naming and identity (Task C, adopted).** Four new run_ids (the fifth, `1e-3`, is the reused reference above): `emb128x32_seedA_lr1em4_cap25k_cal`, `emb128x32_seedA_lr3em4_cap25k_cal`, `emb128x32_seedA_lr3em3_cap25k_cal`, `emb128x32_seedA_lr1em2_cap25k_cal` (mantissa+`e`+`m`-for-minus+exponent-digit encoding — no decimal points, no ambiguous `-` delimiter; verified to collide with none of the closed six-run matrix, the 2026-08-04/05/06 calibration-family run_ids, or each other). Shared campaign/policy name: `lr_range_seedA_25k_v001`, spliced onto the validated `stage1_lead06_pilot_v001` policy via a new closure-splice script following the `run_stage1_cap50k_closure.py` precedent (real validated `PilotPolicy` loaded via `load_pilot_policy()`, four hand-built `PilotRunSpec` entries added under the new policy name, hardcoded closed run_id allowlist, collision guard against the real six and all prior calibration-family run_ids). Generated-config, evidence-bundle, and Moriah run-directory conventions follow the existing per-`run_id` subdirectory pattern used by the 25k/50k calibration batches — no new directory scheme required. W&B run identity is a mechanical consequence of existing code: `derive_pilot_wandb_run_id("lr_range_seedA_25k_v001", run_id, "g1")` for the four new candidates; the reused `1e-3` reference keeps its original `derive_pilot_wandb_run_id("embedding_shape_neighborhood_seedA_25k_v001", "emb128x32_seedA_cap25k_cal", "g1")`-style identity unchanged.

**Minimum implementation plan (Task D, adopted, not implemented by this entry).** (1) `src/baseline/pilot_lead06_config.py` — add one optional field `learning_rate: Optional[float] = None` to the frozen `PilotRunSpec` dataclass (additive, default preserves all existing behavior); needed because none of the closed six-run matrix or prior calibration-family candidates ever varied learning rate, so no override path exists today. (2) `src/baseline/nh_config_generation.py` — after merging a named run profile's frozen dict, apply an optional `learning_rate` override from the `PilotRunSpec` if provided; smallest change, does not touch `_RUN_PROFILES` or any existing named profile. Rejected alternative: defining five near-duplicate named profiles differing only in `learning_rate` — larger, more repetitive, and against the explicit guidance to avoid disposable code where a trivial generalization is cleaner. (3) New small module/function (e.g. `evaluate_diagnostic_checkpoint()` in `pilot_screening_eval.py` or a sibling module) wrapping `ensure_validation_results()` + `raw_space_metrics_for_run_period()` directly for arbitrary epochs, tagged `authoritative=False`, `epoch_role="diagnostic_retrospective"`, structurally prevented from calling `record_screening_event()` — closes the genuine gap identified above. (4) New closure-splice launcher `scripts/run_stage1_lr_range_closure.py` (and matching `..._moriah.sbatch`, `CLOSURE_MAX_TARGET_EPOCH` fixed at 6, not 12) following the `run_stage1_cap50k_closure(_moriah.sbatch)` precedent exactly. (5) Paired-comparison tooling: the ad hoc, untracked `paired_basin_csv_join.py` (pure CSV join+describe, no pickle access) is a small, natural generalization target — extending it from one-reference-vs-one-challenger to one-reference-vs-four-challengers is a parametrization of the same join logic, not a new framework; recommended for promotion into the committed repository as part of the next milestone. (6) Tests required for all of the above: `PilotRunSpec`/config-generation override tests; diagnostic-evaluation-helper tests (epochs 1/2/4/5 acceptance, confirmation that early-stopping state file is never touched, confirmation of scaler/membership reuse); a run_id collision-guard test for the four new LR-A run_ids against the existing six-run matrix and all prior calibration-family run_ids; a generalized paired-comparison-tool correctness test on synthetic multi-candidate CSVs. None of items (1)-(6) are implemented by this documentation-only entry.

**Future multidimensional HPO roadmap recorded (Task E, adopted, documentation only).** Phase A: one-dimensional range characterization (this LR-A design is the first instance), axes learning rate / hidden size / embedding dropout / output dropout, grids not frozen beyond LR-A's own five values, non-binding on final Stage 1 hyperparameters. Phase B: joint multidimensional HPO over the same axes, funnel (1) bounded ranges taken from Phase A screens, (2) broad multidimensional Seed-A screening at capped fidelity, (3) inspection of interactions/promising regions, (4) promotion of integrated configs to higher fidelity, (5) optional narrower adaptive/Bayesian search, (6) Seed-B confirmation for strong finalists, (7) uncapped authoritative finalists with full development-validation evaluation. W&B Sweep / random / Bayesian search machinery is planned for later use as a search/controller/index layer only, once the Flash-NH training/evaluation contract is better characterized by Phase A evidence — it is explicitly **not** a scientific authority; Flash-NH retains authority for legal config bounds, sealed-set protection, training fidelity, raw-space metrics, evidence, candidate identity, and promotion/final selection. **No W&B Sweep, Bayesian, or random-search infrastructure is implemented by this entry.**

**Explicit non-goals of this entry (verbatim, all observed).** No LR candidate launched; no Slurm job submitted; no NeuralHydrology training run; no temporal-test, spatial-holdout, or California access; no full-population validation; no change to `[128,32]`; no hidden-size, dropout, or sequence-length tuning; no scheduler introduced; no W&B Sweep, Bayesian, or random-search implementation; no early-stopping redesign; no continuation-overhead fix; no generic HPO framework built; nothing staged or committed by this task.

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.12), `docs/stage1_lead06_pilot_v001.md` (new 2026-08-08 section).

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no LR-A candidate launched or run described as started.

## 2026-08-06 — Stage 1 — 50k Seed-A embedding-shape comparison closed: `[128,32]` adopted as working default, further embedding-shape exploration paused, bounded learning-rate calibration approved next (documentation-only closure)

**Scope.** Documentation-only closure task, run after a real Moriah closure comparison batch earlier in the same session (unchanged commit `a4c5456331d97af61c71167a39bf5a6a0644d1ab` throughout, confirmed both locally and against `origin/master` before any edit here). No Moriah/h2o access, no Slurm submission, no training or evaluation, and no production Python/tests/YAML-policy/config/runtime-artifact change occurred in this task. Full evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/cap50k_closure_comparison_audit_2026-08-06/`; Moriah evidence root `/sci/labs/efratmorin/omripo/Flash-NH/evidence/cap50k_closure_comparison_audit_2026-08-06/`; archive `cap50k_closure_comparison_audit_2026-08-06.tar.gz` (SHA256 `9ff1960bf7537da78ea62e5046805c28c0436bd1804395086e12c13c1a347207`, independently re-verified locally against `MANIFEST.csv` — 38/38 files, 0 mismatches, 0 missing, 0 extra).

**Closure qualification (adopted finding).** The 2026-08-05 (decision_log 2026-08-05 entry) "next approved structural phase" ran to completion exactly as designed. The existing Seed-A `[128,64]` trajectory (`emb128x64_seedA_cap_low_cal`, job 45762223, continued from its epoch-6 state) and a new Seed-A `[128,32]` trajectory (`emb128x32_seedA_cap_low_cal`, job 45762224, fresh Seed-A initialization) both reached the fixed epoch-12 closure bound cleanly: exit `0:0`, final status `PAUSED_AT_MAX_TARGET_EPOCH`, checkpoints and official screening saved through epoch 12, no overshoot, no sealed-data access, no W&B sync. Both share Seed A (967139), target `qobs_mm_per_h_lead06`, lead 6 h, `seq_length=24`, `hidden_size=128`, learned FC static embedding, tanh activation, embedding dropout 0.1, output dropout 0.25, Adam `lr=0.001`, NSE-style training loss, `max_updates_per_epoch=50000`, the fixed 2,307-basin training population, and the fixed 400-basin development-validation screening set — differing only in `statics_embedding.hiddens` (`[128,64]` vs. `[128,32]`).

**Official raw-space result (adopted finding, 400-basin screening population, median NSE).** Epoch 3: incumbent 0.2418, challenger 0.2480. Epoch 6: incumbent 0.2547, challenger 0.2541. Epoch 9: incumbent 0.2367, challenger 0.2569. Epoch 12: incumbent 0.2427, challenger 0.2464.

**True per-basin paired result (adopted finding, challenger minus incumbent, 400/400 matched basins, tie tolerance ±0.01).** Epoch 3: median +0.0136, Q25 -0.0293, Q75 +0.0650, challenger better 53.5%, incumbent better 35.0%, tied 11.5%. Epoch 6: median +0.0145, Q25 -0.0294, Q75 +0.0640, challenger better 53.25%, incumbent better 34.75%, tied 12.0%. Epoch 9: median +0.0160, Q25 -0.0330, Q75 +0.0636, challenger better 54.25%, incumbent better 33.25%, tied 12.5%. Epoch 12: median +0.0072, Q25 -0.0447, Q75 +0.0709, challenger better 48.5%, incumbent better 41.5%, tied 10.0%. Before computing this, the three pre-existing untracked helpers under `tmp/operations/` (`paired_basin_comparison.py`, `paired_basin_comparison_neighborhood.py`, `extract_per_basin_paired_nse.py`) were inspected and found unsuitable — all three recompute NSE via `raw_space_metrics_for_run_period` against `validation_results.p` pickles rather than performing pure arithmetic over the already-certified per-basin CSVs required here. A new minimal untracked helper, `paired_basin_csv_join.py` (join-and-describe only, no metric recomputation, no pickle access), was built instead and run under Slurm (job 45763464, exit `0:0`).

**Interpretation (adopted, cautious).** `[128,32]` is at least comparable to `[128,64]` and shows a small, directionally consistent paired advantage at epochs 3, 6, and 9; the advantage weakens by epoch 12 (median ΔNSE narrows to +0.0072, win-rate margin narrows to 48.5%/41.5% from roughly 53-54%/33-35% earlier). `[128,32]` is **not** described as decisively superior. The effect is modest relative to cross-basin heterogeneity — paired IQR spans roughly 0.10 at every epoch, an order of magnitude wider than the median shift. Rests on one seed only (Seed A); no independent seed replication at this fidelity. Transformed-space training-loss diagnostics point the same direction (challenger consistently lower across all 12 epochs) but remain a training diagnostic, never the official benchmark. This entry does **not** infer that static attributes are unimportant or should be removed — the comparison is between two learned-embedding widths, not between embedded and raw static pathways.

**Decision (adopted).** `[128,32]` becomes the working default embedding shape — not because it is decisively superior, but because it is at least as competitive as `[128,64]` on the official and paired evidence while being more economical (fewer static-embedding parameters). Further embedding-shape (width/depth) exploration is **paused**, not permanently closed, given the `[128,64]`/`[128,32]`/`[64,32]`/`[256,64]` evidence gathered across the 25k neighborhood screening and this 50k closure — low expected value relative to other open hyperparameters, reopenable on new evidence. No model-family switch is proposed.

**Early-stopping / closure interpretation (adopted finding).** Both trajectories share best official screening epoch 6 (incumbent 0.25474, challenger 0.25414); neither met an early-stopping condition (`stopped=false`, `stop_reason=null`) before epoch 12. Termination at epoch 12 was caused solely by the fixed `CLOSURE_MAX_TARGET_EPOCH=12` bound, not by early stopping. Both bundles' generic `continuation_status.next_intended_screening_epoch=15` fields describe what the unbounded policy would do next — never executed, not a planned continuation.

**Next approved scientific phase (adopted design, not launched).** Bounded learning-rate calibration around the current 0.001 baseline, `[128,32]` fixed, exact candidate values not yet frozen, same fixed 400-basin raw-space validation contract, staged promotion (coarse fidelity first). **Not launched by this entry.**

**Operational efficiency (adopted as a deferred engineering item, not a blocker).** Each nested NH continuation boundary (`continue_training_from_epochNNN/`) reloaded the full dataset, recalculated target standard deviations, and rebuilt lookup tables/dataloaders, adding roughly 20-40 minutes per boundary against a roughly 4-minute steady-state epoch — approximately 25-45% of total wall time across the two continuation boundaries here. Quantified from checkpoint-file mtimes (both bundles' `epoch_timing_table` is empty). Recorded as a future optimization target; **not fixed by this entry.**

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.11), `docs/stage1_lead06_pilot_v001.md` (new 2026-08-06 section).

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged to the repository (evidence archive and extracted evidence package both remain outside tracked content, under `.scratch_local/` and on Moriah only), no sealed temporal-test/spatial-holdout data accessed, no learning-rate experiment implemented or launched.

## 2026-08-05 — Stage 1 — embedding-shape neighborhood screening (25k) closed: `[128,64]`/`[128,32]` structural survivors, next 50k comparison approved, sequence length reframed as a separate model-family axis, revised hyperparameter order, learning-curve and hydrograph standards recorded (documentation-only closure)

**Scope.** Documentation-only closure task, run after a real Moriah embedding-shape neighborhood screening batch earlier in the same session (unchanged commit `5aba586dc4856ecb05945b41d3ff29a34f096cb7` throughout, confirmed both locally and against `origin/master` before any edit here). No Moriah/h2o access, no Slurm submission, no training or evaluation, and no production Python/tests/YAML-policy/config/runtime-artifact change occurred in this task. Full evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/embedding_shape_neighborhood_seedA_25k_v001/`.

**Batch qualification (adopted finding).** Three new capped-update (`max_updates_per_epoch=25000`) candidates — `emb64x32_seedA_cap25k_cal` (`[64,32]`), `emb128x32_seedA_cap25k_cal` (`[128,32]`), `emb256x64_seedA_cap25k_cal` (`[256,64]`) — were trained under Seed A (967139) and compared against the pre-existing, untouched `emb128x64_seedA_cap25k_cal` reference (`[128,64]`), all sharing identical scientific configuration otherwise (`hidden_size=128`, embedding activation tanh, embedding dropout 0.1, output dropout 0.25, Adam `lr=0.001`, NSE loss, `seq_length=24`, target `qobs_mm_per_h_lead06`, fixed 2,307-basin development-training population, fixed 400-basin development-validation screening subset). All three new candidates completed successfully, delivered exactly 25,000 optimizer updates in every one of 6 epochs (directly verified from each `optimizer_state_epochNNN.pt` Adam step counter, job 45756766: cumulative 25,000/50,000/75,000/100,000/125,000/150,000, zero drift), created checkpoints 1-6 with no epoch 7, remained offline in W&B (`tracking_generation=g1`, no sync), and accessed no sealed population. All 8 jobs submitted by this task (`sacct_full.txt`) completed exit `0:0`.

**Performance interpretation (adopted finding, provisional, coarse-screening resolution only).** Official epoch-6 median NSE (400-basin screening subset): `[128,32]` 0.25347, `[64,32]` 0.25250, `[128,64]` reference 0.24750, `[256,64]` 0.24494 — all four within a 0.0086 band. True per-basin paired comparisons (candidate minus reference, `paired_basin/` evidence, job 45756761) show: (1) **no candidate demonstrates broad, consistent superiority over `[128,64]`**; (2) `[128,32]` has the mildest positive edge — positive median paired diff at 5 of 6 epochs, paired win rate (`frac_candidate_better`) never exceeding ≈0.53 in any epoch — a plausible challenger, not a demonstrated winner; (3) `[64,32]` stays broadly close to `[128,64]` — positive median paired diff at all 6 epochs, but a one-off epoch-5 spike (+0.029) not sustained at epoch 6 (+0.0065), no stable or broad advantage; (4) `[256,64]` is the weakest tested shape — negative median paired diff vs. the reference at 4 of 6 epochs (2, 3, 4, 6) and the lowest official median NSE at both official screening epochs (3 and 6) — **provisionally rejected at the 25k structural-screening tier.**

**Cap-resolution finding (adopted).** The 25k cap remains useful for detecting divergence and rejecting a clearly weaker structural region (it cleanly separates `[256,64]` as weakest) but is **not** sufficiently precise for fine ranking among near-identical shapes (`[64,32]`/`[128,32]`/`[128,64]`), whose paired win rates all sit within roughly ±0.13 of chance (0.50) at every epoch — consistent with, and extending, the 2026-08-04 entry's "coarse rejection/triage, not fine ranking" finding to this finer shape granularity.

**Structural survivors (adopted, not a final architecture selection).** `[128,64]` as incumbent; `[128,32]` as challenger. Neither is described as superior; this is not statistical significance and not a final HPO or architecture decision.

**Next approved structural phase (adopted design, not launched).** Existing Seed-A `[128,64]` trajectory (`emb128x64_seedA_cap_low_cal`) continued to 50k vs. a new Seed-A `[128,32]` trajectory at 50k, to close the embedding-structure question at a more informative fidelity and avoid further width/depth exploration unless new evidence later justifies it. Design: `max_updates_per_epoch=50000`; target up to epoch 12; official screening at epochs 3/6/9/12; existing early-stopping policy authoritative (stopping-eligible from epoch 6, minimum improvement 0.005, patience 3 eligible screening events); every epoch saved; retrospective checkpoint evaluation usable diagnostically; no cross-fidelity checkpoint reuse; the new `[128,32]` candidate starts from the original Seed-A initialization; the existing `[128,64]` candidate may continue only within its own unchanged candidate identity and fidelity. **These runs have not started; nothing in this entry launches them.**

**Sequence length reframed as a separate temporal-context model-family axis (adopted, binding).** Sequence length is fixed at 24 for the current model family and is **not** an ordinary hyperparameter in the current near-term tuning funnel. Alternative sequence lengths represent separate temporal-context model families, since they change the scientific information available to the model, antecedent-memory assumptions, input construction, compute/memory requirements, and interpretation across basin response times. A later sequence-length study may compare alternative temporal-context model families against a mature 24-hour model, but is not part of the current hyperparameter phase. `docs/stage1_validation_optimization_foundation.md` Part L.1's Stage-B dimension list (which previously listed sequence length alongside ordinary hyperparameters) is corrected accordingly by this entry. **Further revised (2026-08-13, see this document's 2026-08-13 entry and `docs/stage1_validation_optimization_foundation.md` Part L.19):** the Embedding-Dropout-A closure schedules a dedicated Sequence-Length-A characterization (`seq_length={12,24,48,72}`), reframing sequence length as a bounded, structural/calibratable model parameter — this passage is preserved as historical, not rewritten.

**Revised hyperparameter order (adopted, within the fixed `seq_length=24` model family).** (1) Close embedding structure at 50k: `[128,32]` vs. `[128,64]`. (2) Learning rate — likely a bounded contrast around 0.001, exact values not yet authorized. (3) LSTM hidden size — bounded capacity contrast, exact candidates not yet authorized. (4) Embedding dropout. (5) Output dropout. (6) Small integration/interaction checks among independently promising settings. (7) Seed-B confirmation for only the top integrated candidates. (8) Uncapped authoritative finalists. (9) A separate, later temporal-context model-family study for sequence length. This order is a hybrid of expected scientific/optimization impact, dependency/interaction structure, experimental clarity, and operational cost; it does not encode exact future candidate grids beyond what is stated here.

**Learning-curve standard (adopted, for future serious-triage/finalist packets).** Training-optimization diagnostics: mean training loss vs. epoch, and vs. cumulative optimizer updates. Validation/scientific diagnostics: median raw-space per-basin NSE vs. epoch, a p25-p75 (or equivalent) distributional band, `frac(NSE>0)`, and explicit official-vs-retrospective evaluation markers. Transformed-space validation loss may be included only if already available or cheaply/deterministically derivable from existing predictions/targets, and only as a training diagnostic, never as the official scientific model-selection metric. **Preserved, unchanged:** NH training/validation losses may be diagnostics in transformed target space; official Flash-NH benchmark metrics remain computed after full inverse conversion to raw m³/s; raw-space screening metrics remain authoritative for candidate selection. Raw-space median NSE must not be labeled "validation loss."

**Hydrograph-demonstration standard (adopted design update; not yet implemented).** For 50k-promoted candidates, the standard compact demonstration package should include: a fixed eight-basin compact hydrograph panel; basin area (km², from the authoritative basin-area field) in every panel title; basin-average hourly MRMS QPE precipitation (mm h⁻¹) as blue bars descending from the top on a secondary right-hand axis (zero at top, increasing downward); explicit valid-time alignment (precipitation at its physical valid time, observations at physical discharge time, lead-6 predictions at the target valid time they predict, no artificial six-hour rainfall shift, stated explicitly in rendering metadata/interpretation); matched comparison scales (identical time windows, identical discharge limits, identical precipitation limits where practical, identical plot conventions across compared candidates); a compact-panel metrics table; and a short hydrograph interpretation covering peak magnitude, peak timing, false peaks, recession, baseflow, basin-specific bias, and rainfall-runoff timing where visible. The full 24-basin atlas is reserved for ambiguous cases, integrated candidates, or authoritative finalists — not required for every 50k candidate. Demonstration cadence: 25k coarse screening uses strategic metrics/learning curves only, no routine hydrograph package; 50k serious triage uses the compact eight-basin panel plus rainfall/basin-area plus compact metrics plus short interpretation plus a strategic review packet; integrated/uncapped finalists use the compact panel plus the full 24-basin atlas plus a standardized 6-8 figure package plus a comprehensive scientific summary. This standard reuses the existing fixed basin/window definitions and existing hydrograph-rendering infrastructure (`docs/stage1_validation_optimization_foundation.md` Part L.3/L.3a) — no new visualization framework is introduced.

**W&B status (already-qualified capability only; nothing new qualified by this entry).** Offline W&B remains the operational mode; all four screening runs in this batch stayed offline throughout (`tracking_generation=g1`, no `wandb sync` run for any of them). The previously-qualified controlled post-run sync (`docs/stage1_wandb_user_guide.md` §17, 2026-08-05 qualification of two unrelated single-segment runs from `cap_parallel_batch_v001`) is unaffected by and unrelated to this batch. Private project: entity `omri-porat1-huji`, project `flashnh-stage1`. W&B remains the experiment-index/comparison interface; the Flash-NH repository and its evidence remain scientifically authoritative. Online training and multi-segment offline-run reconciliation remain unqualified; no automatic sync or `--sync-all` workflow is approved. Nothing in this entry requires syncing before scientific review.

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry), `docs/decision_log.md` (this entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.10 closing the neighborhood screening and recording the 50k design; Part L.1's Stage-B dimension list corrected to remove sequence length; new Part L.3d recording the hydrograph-demonstration standard), `docs/stage1_lead06_pilot_v001.md` (new 2026-08-05 section). `docs/stage1_wandb_user_guide.md` was reviewed and left unchanged — it already documents everything referenced above (offline-mode default, the qualified single-segment sync, and the unqualified capabilities) and this task qualifies no new W&B capability.

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged to the repository, no sealed temporal-test/spatial-holdout data accessed, no run described as started.

## 2026-08-04 — Stage 1 — capped-update calibration complete: mechanism qualified, provisional fidelity workflow adopted, static embedding reopened, strategic review packet standard added (documentation-only closure)

**Scope.** Documentation-only closure task, run after three real Moriah calibration exercises earlier in the same session (unchanged commit `ac98f6b3ad9b1687a26a7509f98a02df3c06381b` throughout, confirmed both locally and against `origin/master` before any edit here). No Moriah/h2o access, no Slurm submission, no training or evaluation, and no production Python/tests/YAML-policy/config/runtime-artifact change occurred in this task. Full evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/cap_learning_diagnostics_v001/`, `.scratch_local/moriah_evidence/emb50k_architecture_diagnostic_v001/`, `.scratch_local/moriah_evidence/cap_parallel_batch_v001/`.

**Mechanism qualification (adopted finding).** The `max_updates_per_epoch` mechanism implemented in the 2026-08-03 entry below is now operationally qualified: across an uncapped reference (`raw_seedA`, 237,298 optimizer updates/epoch, measured for the first time — closes the L.5/L.8 measurement gap), two initial caps (`raw_seedA_cap_medium_cal` 100,000; `raw_seedA_cap_low_cal` 50,000), a matched raw-vs-embedded 50k pair, and a four-candidate parallel 25k/50k batch (raw and `[128,64]`-embedded pathways, Seed A 967139 and Seed B 1729), every epoch of every run enforced its configured cap exactly (verified from each run's own persisted `optimizer_state_epochNNN.pt` step counter). Run-identity isolation, per-epoch checkpointing, continuation-cap safeguards, screening, evidence-bundle generation, and offline W&B all behaved as designed.

**Performance interpretation (adopted finding).** Aggregate distributional metrics (median NSE, `frac(NSE>0)`, `frac(NSE<-1)`) stayed broadly coherent across all capped candidates and fidelities tested — no divergence or collapse. True per-basin paired differences (matched by `basin_id`, 400-basin screening subset, `comparison_evidence_v2.json`, job 45754688) are consistently wider than the aggregate effects under study: e.g. epoch-6 Seed-A cap-sensitivity (25k vs the protected 50k reference) shows a raw-pathway median paired diff of +0.0108 (56.5% favoring 50k) and an embedded-pathway median paired diff of -0.0092 (55% favoring 25k) — small and sign-inconsistent relative to each comparison's own ~0.10-wide paired p25-p75 band. **Adopted:** capped runs support coarse rejection/second-stage triage only, not fine ranking; capped performance is not evidence that fewer updates are scientifically superior; capped checkpoints must never be promoted to full-fidelity trajectories.

**Runtime interpretation (adopted finding).** Halving the per-epoch update cap (50k to 25k) roughly halved training-loop optimizer work but reduced total Slurm-elapsed time through epoch 6 by only ~12-18% in the two matched pairs measured (raw: 2625.16s vs 3213.3s; embedded `[128,64]`: 2657.5s vs 3020.2s) — validation/startup/checkpointing overhead is a large, largely fixed cost that does not shrink with the cap. **Adopted:** do not assume linear wall-time scaling with the update cap.

**Provisional fidelity workflow (adopted, explicitly not a final scientific method).** 25k = first-pass coarse rejection; 50k = second-stage triage for plausible candidates; uncapped = finalists only. Safeguards: each fidelity is a distinct run identity; a promoted candidate restarts from its original seed at full fidelity, never continuing from a capped checkpoint; capped results remain provisional; only full-fidelity finalists may support a final architecture decision.

**Retrospective per-epoch evaluation policy (adopted).** During the current diagnostic structural-calibration phase: official screening cadence (epochs 3/6/9/...) and existing early-stopping semantics are preserved unchanged; every epoch is still saved; retrospective, diagnostic-only evaluation of intermediate checkpoints (this session: epochs 1/2/4/5) may be used for close/promising/puzzling/new-family candidates, and must never feed back into authoritative early-stopping or checkpoint-selection state. For later routine broad campaigns, this every-epoch retrospective evaluation is **not** to be applied automatically to every 25k candidate — routine coarse rejection keeps the lighter official cadence only.

**Static embedding — remains unresolved (adopted framing change).** The matched 50k raw-vs-`[128,64]` comparison (Seed A, `emb50k_architecture_diagnostic_v001`) shows a modest, epoch-varying directional edge for the embedded pathway (per-basin win share favors embedded at all 6 epochs, narrowing to 44.25%/42.25% raw/embedded by epoch 6, median paired diff -0.0025) against a basin-level IQR roughly an order of magnitude wider than the aggregate gap. **Adopted:** raw and learned `[128,64]` embedding remain close; direction/magnitude vary with epoch, seed, and fidelity; this comparison is explicitly **not** resolved. Static embedding architecture is reframed as a bounded hyperparameter family — raw; `[64]`; `[128]`; `[128,64]` — rather than a settled binary choice.

**Next approved, not-yet-started structural batch.** One-layer `[64]` and one-layer `[128]` embeddings, Seed A, 25k cap, compared against the existing Seed-A raw/`[128,64]` references. Fixed: embedding activation tanh, embedding dropout 0.1, output dropout 0.25, `hidden_size` 128, learning rate 0.001, all data/split settings. Dropout/learning-rate/hidden-size/broad-HPO tuning is explicitly out of scope until this shape axis narrows. **These runs have not started; nothing in this documentation entry launches them.**

**Strategic review packet standard (new, adopted for future structural-comparison tasks only — not retroactive).** Every future structural-comparison task should assemble a local, untracked evidence packet with 7 components: (1) `PROVENANCE.json` (commit, candidate identities, config diffs, seeds, caps, Slurm jobs, evidence paths, exclusions/corrections); (2) a per-candidate/per-epoch metrics table (median NSE, p10/p25/p75/p90, `frac(NSE>0)`, `frac(NSE>0.3)`, `frac(NSE<-1)`, training loss, actual/cumulative updates, timing, official-vs-retrospective status); (3) true basin-matched paired-comparison stats (median paired diff, paired p25/p75, win fractions, tie fraction within ±0.01); (4) a runtime/updates table (cap, actual updates, training time, evaluation time, total elapsed); (5) a config-diff matrix (exact intended and unintended differences between candidates); (6) 6 compact plots (median NSE vs epoch; median NSE vs cumulative updates; IQR bands; `frac(NSE>0)`; training loss vs cumulative updates; paired median differences); (7) a `strategic_summary.md` (robust findings; ambiguous findings; rejection recommendations; promotion recommendations; next bounded batch). **The compact tables and summaries are authoritative for strategic review; plots are diagnostic aids only.** Large checkpoint files, validation pickles, W&B binaries, and full per-basin files are not required in the local packet unless a specific discrepancy needs them.

**Documentation changes made by this entry.** `docs/FLASHNH_CURRENT_STATE.md` (new top-of-file entry), `docs/stage1_validation_optimization_foundation.md` (new Part L.9, closing L.6/L.8's open calibration questions), `docs/stage1_lead06_pilot_v001.md` (new calibration section), `docs/stage1_wandb_user_guide.md` (§16 updated — capped runs have now been launched and stayed offline throughout; W&B remains telemetry, not scientific authority).

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no numerical cap adopted for production use, no generated evidence committed to the repository, no sealed temporal-test/spatial-holdout data accessed.

## 2026-08-03 — Stage 1 — `max_updates_per_epoch` capped-update screening support (implementation + tests only)

**Scope.** Implements and locally qualifies the optional
`max_updates_per_epoch` mechanism whose direction was adopted in the
2026-08-02 roadmap entry below (Part L.5). Efficiency feature only: no
training/evaluation, no Moriah/h2o access, no Slurm submission, no numerical
cap adopted, no capped run ever launched. A separate, independent Moriah
operations session was concurrently advancing the existing uncapped
`raw_seedA` trajectory through epoch 9 during this work; that trajectory,
its evidence, and the Moriah clone were not touched.

**Contract.** `max_updates_per_epoch: int | None`. `None` (default) is
exactly today's uncapped behavior; any other value must be a positive
integer (`0`, negative integers, bools, floats, strings rejected before
config generation). The cap is frozen for a candidate's entire trajectory:
`enforce_pilot_cap_identity` rejects, before any training call, a
continuation/resume whose freshly-resolved cap disagrees with the cap
already persisted for that NH run directory (null↔int and int↔different-int
in both directions). Capped and uncapped runs — and different integer caps
— are always distinct identities; a capped checkpoint is never a valid
continuation source for an uncapped trajectory or vice versa. A promoted
finalist is expected to start a new full-fidelity trajectory from its
original seed, never continue from a capped checkpoint (provisional
recommendation, per Part L.6, unchanged by this entry).

**Verified NeuralHydrology 1.13 semantics** (read from the vendored source
directly, `neuralhydrology/utils/config.py` and
`neuralhydrology/training/basetrainer.py` — not assumed): the cap is a
deterministic index-based prefix-truncation of each epoch's DataLoader
iteration order, re-applied fresh every `_train_epoch` call (the counter
resets every epoch); the scheduler still steps once per epoch; both
`model_epochNNN.pt` and `optimizer_state_epochNNN.pt` are still written
unconditionally once per epoch regardless of the cap. That last point is
why real actual-optimizer-update evidence (`optimizer_state_epochNNN.pt`'s
own persisted Adam/AdamW `state[p]['step']` counter) is obtainable with
**no NeuralHydrology core-code modification** — evidence bundles record
this actual per-epoch update count and the configured cap as two distinct,
never-conflated fields.

**Files changed.** `src/baseline/nh_config_generation.py` (validation +
optional YAML key emission), `src/baseline/pilot_lead06_config.py`
(`PilotRunSpec.max_updates_per_epoch` field), `src/baseline/pilot_tracking.py`
(`_HYPERPARAMETER_CONFIG_KEYS`, always-present `run_identity` field),
`src/baseline/pilot_orchestration.py` (`enforce_pilot_cap_identity`,
`read_actual_optimizer_updates`, `actual_optimizer_updates_by_epoch`, wired
into `run_pilot`/`prepare_pilot_run_only`), `src/baseline/pilot_evidence_bundle.py`
(verbatim recording of both cap fields). Documentation:
`docs/stage1_validation_optimization_foundation.md` (new L.7),
`docs/stage1_lead06_pilot_v001.md` (new dated section),
`docs/stage1_wandb_user_guide.md` (new §16, §6 cross-reference fix).

**Structural independence confirmed by code inspection, not just testing.**
`MAX_TARGET_EPOCH`, early stopping (`src/baseline/pilot_early_stopping.py`),
screening cadence, checkpoint discovery, and the additive-continuation
overlay have zero references to `max_updates_per_epoch` in their
implementations — no coupling was introduced.

**Not done, intentionally.** No numerical cap chosen; no real uncapped
optimizer-updates-per-epoch count measured against Moriah; no capped run
launched; no sweep/HPO machinery; no change to `raw_seedA` or
`emb128x64_seedA` (`max_updates_per_epoch` remains `null` for both, exactly
as before). A compact, execution-deferred Moriah calibration plan was
prepared alongside this entry (see the corresponding session report) but
not run.

## 2026-08-02 — Stage 1 — W&B offline tracking qualification + Flash-NH W&B user guide (documentation + tests only)

**Scope.** Documentation-and-test-only pass qualifying ordinary W&B
tracking (`src/baseline/wandb_tracking.py`, `src/baseline/pilot_tracking.py`)
before any live use on the next real structural candidate (`raw_seedA`),
per the roadmap set in the entry below (Part L.4). No training, no
evaluation, no Moriah/h2o access, no Slurm submission, no online W&B, no
sweep, no `raw_seedA` launch, and no repository staging/commit occurred.

**Two known gaps confirmed and fixed.** (1) Failure isolation after
`wandb.init` — every backend call (metric logging, checkpoint-reference
logging, finish) is now wrapped in a guard (`_guard_backend_call`) that
catches any backend exception, warns once per distinct failing operation,
and records `degraded`/`degraded_operations` on the `TrackingRun`, so a
tracking failure can never stop training, evaluation, checkpoint
selection, or early stopping. (2) Stable W&B run identity across bounded
Slurm continuations — `derive_pilot_wandb_run_id`/
`resolve_pilot_wandb_run_id` compute a deterministic id from
`(pilot_policy_name, run_id)` alone (correct on the very first call, before
any NH run directory exists), passed to `wandb.init(id=..., resume=
"allow")`; a small persisted record in the NH run directory cross-checks
this on later continuations and raises a `TrackingError` on contradiction
rather than silently merging two candidates' run histories.

**Metadata contract extended.** `build_pilot_run_identity` now also
carries `max_updates_per_epoch: None` (honest null, not an implemented
feature), `baseline_policy_sha256`, `splits_dir`, and `wandb_run_id`; the
evidence bundle's `"wandb"` block now also carries `mode`, `wandb_run_id`,
`degraded`, and `degraded_operations`.

**Offline qualification.** Exercised entirely through pytest against an
in-process fake `wandb` module (`sys.modules` monkeypatching) — never the
real `wandb` package (confirmed not installed in this environment), never
network access, nothing written outside `tmp_path`. Covers 15 numbered
scenarios (policy parsing; real init; config/hyperparameter logging;
epoch/resource metrics; screening+early-stopping logging; checkpoint-
reference logging; clean finish; continuation/same-run-id reuse;
replay-guard idempotency under simulated tracking failure; simulated log
failure; simulated finish failure; disabled-mode never imports `wandb`;
absent-package handling at both the wrapper and pilot layers; credential-
key rejection; disallowed sealed-scope-metric-key rejection) via a new
`tests/test_wandb_offline_qualification.py`, plus deeper integration
coverage added to `tests/test_wandb_tracking.py`,
`tests/test_pilot_tracking.py`, and `tests/test_pilot_orchestration.py`.
Combined four-file suite: 102 passed, 0 failed.

**Documentation.** New `docs/stage1_wandb_user_guide.md` — a concise,
Flash-NH-specific guide to reading W&B tracking once enabled (project/run/
candidate concepts, where provenance appears, comparing candidates,
reading curves, what a sweep is, run-state semantics, what to check before
approving a campaign, what W&B does and does not control, offline vs.
online mode, Slurm-continuation behavior, recognizing degraded tracking,
and how a future backfilled run would be labeled). It does not claim any
live online tracking run or sweep has occurred, because none has.
`docs/stage1_validation_optimization_foundation.md` Part L.4 and
`docs/stage1_lead06_pilot_v001.md`'s W&B section were updated to record
this qualification status.

**Status after this entry.** W&B adoption sequencing stage (1) ordinary
qualification is complete, exercised entirely against a fake `wandb`
module (see "Offline qualification" above) — despite this entry's original
title, that is a fake-backend contract test, not a real-package offline-mode
test; stage (2) real-package offline-mode testing was **not yet done** as
of this entry and is corrected by the 2026-08-02 "real-package offline
qualification" entry below. Stage (3) controlled live tracking on one
structural candidate and stage (4) sweeps remain not started. The
wrapper's shipped default is unchanged — `enabled: false` / `mode:
disabled`. `raw_seedA` remains the next scientific candidate to launch;
this entry does not authorize or perform that launch.

## 2026-08-02 — Stage 1 — W&B real-package offline qualification (Path A) + `tracking_generation` identity fix (documentation + tests + local no-network smoke only)

**Scope.** Commit-readiness review of the entry above found its "offline
qualification" claim rested entirely on an in-process fake `wandb` module,
which proves the wrapper's contract but not real W&B offline I/O,
persistence, or continuation semantics. This entry corrects that: real
package **wandb 0.28.1** was installed in an isolated `.venv` (never the
scientific NeuralHydrology environment) and exercised offline, no API key,
no network, no NH training, then uninstalled again afterward. No
`raw_seedA` launch, no Moriah/h2o access, no Slurm submission, no online
W&B, no sweep, no repository staging/commit occurred.

**Real-package smoke.** `scripts/wandb_real_offline_qualification_smoke.py`
drives this repo's actual tracking code (never a reimplementation) as two
genuinely separate OS processes reusing one stable run id, standing in for
two bounded Slurm continuations. Confirmed: real `wandb.init(mode=
"offline", id=..., resume="allow")`; config/hyperparameter and scientific-
metric logging; a checkpoint-reference record (path/checksum/size only,
never checkpoint bytes); clean finish; degradation against a real backend
exception (`wandb.errors.UsageError` on logging to a finished run — caught,
recorded `degraded`, never propagated); no network attempt. Qualification
record: `reports/wandb_real_offline_qualification_v001/qualification_record.json`
(untracked, per repo convention).

**One assumption corrected by real evidence.** Offline `resume="allow"`
does **not** make a second invocation locally continue the first's run
directory — wandb prints `` `resume` will be ignored since W&B syncing is
set to `offline`. Starting a new run with run id <id>. `` and each
invocation gets its own fresh timestamped local directory. Reconciling
same-id invocations into one logical run is server-side, at `wandb sync`
time, matched by run id + project — never a local merge. No code change
was required (the wrapper only ever passed `id=`/`resume=` through
unmodified); `docs/stage1_wandb_user_guide.md` §12 and
`docs/stage1_validation_optimization_foundation.md` Part L.4 were both
corrected to state this rather than the previously-assumed local-resume
behavior.

**Non-blocking caveat found.** An artificially long run id in an early
smoke trial silently truncated wandb-core's binary transaction-log path
past Windows' 260-char `MAX_PATH`, with no error or warning anywhere. Real
production ids (e.g. `flashnh-stage1_lead06_pilot_v001-raw_seedA-g1`, ~46
chars) are far short of this threshold — documented as a caveat, not fixed
in code.

**`tracking_generation` identity fix.** The same review found a genuine,
if narrow, identity-collision gap: NH run directories are
timestamped/prefix-matched rather than one fixed path per `run_id`, so a
deliberate operator restart-from-scratch under the same `run_id` was
indistinguishable at call time from a genuine first attempt. Fixed with
the smallest durable resolution: an explicit, manually-set
`tracking_generation` parameter (default `"g1"`, left at default for every
ordinary continuation) threaded through
`derive_pilot_wandb_run_id`/`resolve_pilot_wandb_run_id`/
`build_pilot_run_identity`/`init_pilot_tracking_run`, included in the
persisted-record contradiction check. `src/baseline/pilot_tracking.py`;
4 new tests in `tests/test_pilot_tracking.py`.

**Tests.** `tests/test_wandb_tracking.py` + `tests/test_pilot_tracking.py`
+ `tests/test_pilot_evidence_bundle.py` + `tests/test_wandb_offline_qualification.py`:
140 passed. `tests/test_pilot_orchestration.py`: 46 passed. No production
code changed other than `src/baseline/pilot_tracking.py`'s
`tracking_generation` threading.

**Status after this entry.** Stage (2) real-package offline-mode testing
is now genuinely complete, within the scope stated above (single machine,
Windows, short local runs, no GPU/NH training, no real multi-node Slurm
continuation). Stage (3) online tracking and stage (4) sweeps remain not
started/not qualified. The wrapper's shipped default is unchanged —
`enabled: false` / `mode: disabled`. `raw_seedA` remains the next
scientific candidate to launch; this entry does not authorize or perform
that launch, and does not itself decide to enable tracking for it.

## 2026-08-02 — Stage 1 lead-6 pilot — `emb128x64_seedA` real 24-basin hydrograph-atlas evaluation + rendering (jobs 45729427, 45729449): visual review adopted

**Scope.** Real Moriah execution (not documentation-only), following directly
from the roadmap entry below. Built and evaluated a disposable,
evaluation-only derivative of the completed `emb128x64_seedA` run's frozen
epoch-6 checkpoint (same weights, same fitted scaler, same config) against
the fixed 24-basin hydrograph atlas for the validation period only (job
`45729427`, PASS — build + config-diff + evaluate + integrity checks all
exit 0), then rendered the full 24-basin atlas plus a deterministic 8-basin
compact panel from the derivative's own results pickle, reusing the existing
rendering tooling unchanged (job `45729449`, PASS). No training, no scaler
refit, no sealed-set access (temporal-test, spatial-holdout, California), no
`raw_seedA` launch, no W&B activity, and no repository commit occurred. The
original run directory, checkpoint (`model_epoch006.pt`), scaler
(`train_data_scaler.yml`), config (`config.yml`), and ~400-basin screening
validation results pickle are verified byte-identical before and after
(sha256, pre/post). This derivative is diagnostic/visualization-only and is
not authoritative for checkpoint, architecture, or hyperparameter selection;
it does not replace or resample the provisional ~400-basin screening result.
This entry also records two focused safety-hardening fixes applied to the
shared evaluation-only-derivative machinery while reviewing this operation
for commit-readiness (an out-run-dir/development-run-dir path-collision
guard, and a training-entrypoint guard against the `EVALUATION_ONLY_DO_NOT_TRAIN.txt`
marker) — see `docs/stage1_validation_optimization_foundation.md` Part L.3c
for full detail.

**Evidence.** 24 individual atlas panels, one deterministic 8-basin compact
panel, per-basin metrics (`per_basin_metrics.csv`), 96 event windows
(`event_window_table.csv`), rendering manifest/summary, all reusing the
existing, unmodified raw-space conversion and metric code
(`src/baseline/nh_raw_space_evaluation.py` / `nh_seed_evaluation.py`).
Event-window selection is observed-discharge-only (unchanged from the
existing rendering tooling). Full technical/operational detail and
checksums: `docs/stage1_validation_optimization_foundation.md` Part L.3c;
full evidence write-up (untracked): `reports/stage1_validation_optimization_foundation_v001/part_l_atlas24_eval_emb128x64_seedA_v001/`.

**Adopted scientific interpretation (visual diagnostic observation, not a
formal proof; not every individual timing relationship was manually
validated).** The 24-basin atlas and compact panel show genuine hydrologic
signal — many predicted events occur in approximately the correct temporal
neighborhood of the corresponding observed event, with no obvious universal
six-hour displacement and no evidence of a global raw-space conversion
failure. Performance nonetheless remains weak and hydrologically
inconsistent: large observed peaks are commonly attenuated; some basins show
false or exaggerated predicted peaks; recession and baseflow behavior are
often poor; bias varies strongly by basin. The 24-basin atlas's aggregate
median validation-period NSE (≈0.14) is a deliberately stratified diagnostic
sample, not a representative substitute for the provisional ~400-basin
screening metric, and **must not** be presented as one. This result
supports continuing structural optimization; it does **not** establish
model adequacy, architecture superiority, full development-validation
performance, or final Stage 1 readiness, and it does not change the
selected epoch-6 checkpoint or any other model-selection decision.

**Decisions carried forward, unchanged by this entry.** `raw_seedA` remains
the preferred next Stage A structural candidate (Part L.2). W&B offline-mode
qualification remains the immediate next operational preparation before
that launch, per the existing W&B adoption sequencing (Part L.4).

**Not done by this entry.** No training, no test-set/spatial-holdout access,
no W&B operation, no `raw_seedA` launch, and no repository staging or commit
occurred.

## 2026-08-02 — Stage 1 lead-6 pilot — post-`emb128x64_seedA` roadmap: Stage A/Stage B framing, hydrograph timing, W&B sequencing, multi-fidelity direction (documentation only)

**Scope.** Documentation-only roadmap patch, written after the local,
read-only Stage 1 inspection/planning pass over the completed
`emb128x64_seedA` candidate (commit
`74ca95dbea9425c7f76a7263c97f14c4898e3f94`). No code, tests, configs,
Slurm scripts, checkpoints, manifests, or state files changed. No
Moriah/h2o access, no training, no evaluation, no hydrograph generation, no
W&B contact, no capped-update implementation, and no config generation
occurred in this task. This entry records only workflow-direction and
documentation decisions; it introduces no new binding numerical values
anywhere in the repository.

**Completed facts restated (unchanged, not reopened).** `emb128x64_seedA`
is complete: epoch 6 median raw-space screening NSE `0.20454161610527344`
(best), epoch 9 `0.18124855313577198`, epoch 12 `0.1993193615763258`,
epoch 15 `0.17125263282608943`; stopped at epoch 15
(`patience_exhausted`); epoch 6 is the selected checkpoint for this one
candidate configuration only. Sealed temporal-test and spatial-holdout
populations were not accessed. The continuation-repair/adoption history
closed by the 2026-07-30 entries below is not reopened by this entry.

**Adopted workflow direction (binding; full framing and rationale in
`docs/stage1_validation_optimization_foundation.md` Part L).**
- Stage A (this six-run structural pilot: raw-vs-learned-embedding static
  pathway, approximate embedding shape, limited two-seed robustness) and
  Stage B (proper HPO, deferred) are distinct phases; every other
  hyperparameter stays frozen across all six Stage A runs, and Stage A must
  not be described as a hyperparameter sweep (Part L.1).
- Preferred next Stage A candidate: `raw_seedA` (clean same-seed contrast
  against `emb128x64_seedA`) — a preference, not a launch authorization
  (Part L.2).
- The remaining four runs (`emb128x64_seedB`, `emb64_seedA`,
  `emb128_seedA`, `raw_seedB`) are reviewed between candidates, not
  committed to automatic or parallel launch; parallel execution remains an
  available but non-preferred option (Part L.2).
- Hydrographs (compact ~6-8-basin panel + existing 24-basin atlas) move
  earlier as an early diagnostic, not a mandatory gate after every routine
  candidate; none generated yet, and the compact-panel code does not exist
  yet (Part L.3).
- W&B adoption order: tracking qualification → offline-mode test →
  controlled live tracking → sweeps only after Stage B is frozen.
  Repository code remains authoritative over basin membership, sealed-set
  protection, metrics, early stopping, checkpoint provenance, and package
  identity; W&B is never the scientific source of truth. Not yet qualified
  or enabled; a project-specific W&B learning guide is still a required,
  unwritten artifact (Part L.4).
- Preferred first multi-fidelity mechanism for Stage B: NeuralHydrology's
  `max_updates_per_epoch`, in preference to a reduced training-basin
  subset — a direction, not an implementation; no config in this
  repository sets it (Part L.5).

**Provisional numerical ranges (explicitly not binding; full caveat in
Part L.5).** Low fidelity ≈10-15% of one confirmed full uncapped epoch's
optimizer-update count, medium ≈25-50%, full uncapped (unchanged) — starting
points for later calibration, not approved integer update caps. The real
full-epoch update count is unresolved against the real Moriah
NeuralHydrology 1.13 environment; the only local inspection done was a
rough, non-authoritative one against a differently-versioned, vendored NH
1.12 copy.

**Open technical questions (explicitly unresolved, require calibration
before Stage B screening can run; full list in Part L.6).** Epoch- vs.
cumulative-update-based screening/patience for capped trials; the patience
unit; fair cross-fidelity budget comparison; promotion thresholds;
continuation-from-checkpoint vs. restart-from-seed for promoted finalists;
learning-rate scheduling implications; and reliable cumulative-update
provenance across a resumed run (NeuralHydrology's own resumed-logger
count is not adopted as authoritative for this purpose until its
capped-epoch behavior is verified against the real environment).

**Current provisional recommendation for the first capped-update campaign
(not an immutable scientific decision; full text in Part L.6).** Capped
fidelities for screening/ranking only; restart promoted finalists from
their original seed at full fidelity rather than continuing from a
lower-fidelity checkpoint. No implementation of this recommendation exists
yet.

**Documentation changes made by this entry.** A new roadmap addendum (Part
L) was added to `docs/stage1_validation_optimization_foundation.md`; a
current-status/next-step section was added to
`docs/stage1_lead06_pilot_v001.md`; the top-of-file status pointer in
`docs/FLASHNH_CURRENT_STATE.md` was updated and two stale "remains paused
after epoch 6" passages there (2026-07-30 and 2026-07-29 historical
entries) were marked superseded in place, pointing to the existing
2026-07-30 closure entry — their original text was not deleted or
rewritten. Two equivalent stale passages in this file (below, in the
2026-07-30 continuation-nesting entry and the 2026-07-29
evaluation-prerequisite entry) are marked superseded the same way in this
same edit. `docs/stage1_scientific_baseline_design.md` was not edited.

## 2026-07-30 — Stage 1 lead-6 pilot — `emb128x64_seedA` candidate complete: continuation adopted, epochs 12 and 15 screened (job 45722908)

**Scope.** Documentation-only closure of the continuation-repair/adoption
sequence recorded in the entries below. No code, tests, configs,
launchers, checkpoints, manifests, or state files changed in this task.

**Adoption.** The production `pilot_accepted_continuation.json` manifest
for `emb128x64_seedA` was authored with real SHA-256 hashes for the
epoch-12 and epoch-15 model+optimizer checkpoints (each filename-bound to
its own key epoch per the trust-binding correction — see the entry
below), and used successfully on Moriah in job `45722908` (partition
`catfish`, source commit `af8945d04451d7699ab54b13082eaf870f04f28e`,
elapsed `00:10:34`, Slurm state `COMPLETED`, exit code `0:0`). **No
training occurred.** The existing accepted checkpoints at epoch 12 and
epoch 15 were adopted and evaluated sequentially.

**Final screening history.** Epoch 3 (diagnostic only); epoch 6 (median
per-basin raw-space NSE `0.20454161610527344`, new best); epoch 9
(`0.18124855313577198`, no improvement); epoch 12 (`0.1993193615763258`,
no improvement under the minimum-improvement threshold); epoch 15
(`0.17125263282608943`, no improvement).

**Final early-stopping state.** Best epoch `6`; best metric
`0.20454161610527344`; `events_since_best_improvement = 3`; `stopped =
true`; stop reason `patience_exhausted`; stop epoch `15`.

**Final orchestration state.** `logged_screening_epochs = [3, 6, 9, 12,
15]`; highest physical checkpoint epoch 15; highest screened epoch 15; no
overshoot epochs remain unresolved; no further screening epoch is
intended for this run. No further training is authorized or required for
`emb128x64_seedA`.

**Epoch 6 is the selected checkpoint for this one candidate configuration
only.** It is not the final Stage 1 production model — that
determination requires comparing results across all six run
specifications in the wider optimization campaign.

**Sealed populations untouched.** Screening used only the
development/screening-subset population, exactly as in every prior
screening event for this pilot. No temporal-test or spatial-holdout data
was accessed.

**Closure.** This closes the continuation-repair/adoption sequence:
provenance review of the real epoch 7-15 checkpoints, the explicit
run-specific adoption manifest, the manifest's trust-binding correction,
and now real Moriah adoption and screening — all without ever retraining
past the original, uninterrupted epoch 6→15 continuation. The next phase
is the wider optimization campaign: the other five run specifications,
screened and stopped under this same frozen protocol.

## 2026-07-30 — Stage 1 lead-6 pilot — `emb128x64_seedA` epoch 6→15 continuation: provenance review and explicit adoption mechanism

**Scope.** Two-part decision. Part 1 (inspection-only, no code changes): whether
the real `emb128x64_seedA` checkpoints epoch 7-15 — produced by the additive-
epoch overshoot bug during job `45705457`'s continuation (see the two entries
below) — form one valid, uninterrupted, configuration-consistent continuation
from the trusted epoch-6 checkpoint. Part 2 (implementation): a narrow,
run-specific mechanism to explicitly adopt that trajectory, gated by strict
epoch-12-before-epoch-15 sequencing. Neither part reopened checkpoint-discovery,
continuation, or early-stopping design, or built a general checkpoint-
governance framework.

**Provenance verdict (Part 1).** Direct inspection of
`flashnh_emb128x64_seedA_continuation_evidence_2026-07-29.txt` (job 45705457's
full `output.log` tail, git/Slurm identity, and both persisted state files)
confirms a single, unbroken `Continue training from epoch 6` invocation
producing epochs 7 through 15 with no intervening crash, config change, or
process restart, at the frozen `stage1_lead06_pilot_v001` config (source
commit `eefd9aa`). Verdict: **conditionally safe to adopt**, subject to the
sequencing constraint below — this is a provenance judgment about one existing
physical artifact, not a re-endorsement of the additive-epoch bug itself
(already fixed; see the continuation-nesting entry below).

**Adoption mechanism (Part 2).** Implemented in
`src/baseline/pilot_orchestration.py`: a new, strictly opt-in, per-run JSON
manifest, `pilot_accepted_continuation.json`, read only from the base NH run
directory (never committed, never a general CLI override — see
`docs/stage1_lead06_pilot_v001.md` for the full contract). Each entry pins one
epoch's model **and** optimizer checkpoint by relative path and SHA-256; both
hashes are verified against the real files at the moment that epoch is
consulted, and any mismatch, wrong `run_id`, or out-of-directory path raises
loudly. The manifest may list both epoch 12 and epoch 15, but
`_advance_chunk_via_continuation` only ever consults the entry for the exact
`chunk_target_epoch` a given chunk call is already resolving — epoch 15's
entry is never looked at while epoch 12 is still due, and if early stopping
fires at epoch 12, epoch 15 stays physically present but is never consulted
again. This falls out of the existing per-chunk loop structure in
`run_pilot()` (which already breaks out once a chunk reports `stopped`)
without any new dedicated sequencing code. 10 new focused tests added to
`tests/test_pilot_orchestration.py`; only `pytest tests/test_pilot_orchestration.py -q`
was run (44 passed), per this task's scope.

**Real hashes not yet available (superseded).** No Moriah access was
permitted in this task, and no SHA-256 checksums for the real
epoch-12/epoch-15 checkpoints exist anywhere in the local repository or
evidence file (which records only file sizes/timestamps). The production
`pilot_accepted_continuation.json` for `emb128x64_seedA` therefore has
**not** been authored yet — this decision records the mechanism and the
provenance verdict, not a completed adoption. Authoring it requires one
lightweight Moriah-side step: compute
`sha256sum model_epoch012.pt optimizer_state_epoch012.pt model_epoch015.pt
optimizer_state_epoch015.pt` inside
`continue_training_from_epoch006/`, and write the manifest (schema in
`docs/stage1_lead06_pilot_v001.md`) into the base run directory before the
next screening run. **This was subsequently completed — see the closure
entry above (job `45722908`).**

**Generated, not committed.** The manifest lives alongside
`pilot_early_stopping_state.json`/`pilot_orchestration_state.json` — both of
which are documented, run-directory-local, machine-specific artifacts that
are never committed to git. The manifest follows the same convention (it
would otherwise need machine-specific absolute paths or a path meaningless
outside one specific run directory). The scientific decision itself (this
entry) is the committed, reviewable record; the manifest is the mechanical,
regenerable trigger.

**Status (superseded).** No Moriah adoption or screening run using this
mechanism has occurred yet. `emb128x64_seedA` remains paused after epoch 9
(the last trusted, screened checkpoint) exactly as before this task. **See
the closure entry above: job `45722908` completed the adoption and
screening, and `emb128x64_seedA` is now complete.**

## 2026-07-30 — Stage 1 lead-6 pilot — real Moriah verification (job 45718742): launcher classification fix confirmed, rerun-idempotency defect found and fixed

**Scope.** Local-only correction to `src/baseline/pilot_orchestration.py`
(commit `7c6b02a`'s companion fix), discovered by a real Moriah verification
job. No checkpoint-discovery, continuation, evaluation, early-stopping,
launcher-classification, or scientific-policy design was reopened.

**What happened.** Slurm job `45718742` (partition `catfish`, source commit
`7c6b02a599b885682a97081a3f166d97097bd4ec`, elapsed `00:03:17`, no stderr)
confirmed the previous launcher-status fix works: the launcher correctly
classified the final on-disk state as `BLOCKED_MANUAL_REVIEW_REQUIRED`
(`pilot_final_status: blocked_continuation_overshoot_conflict`,
`safe_to_continue_automatically: false`, overshoot epochs 10-15, exit code
1). **No training occurred and scientific state was not modified.** But
before reaching that clean overshoot block, the Python pilot process
crashed with `PilotEarlyStoppingError: epoch 6 is not after the last
recorded epoch 9 -- out of order`, raised from `run_pilot() ->
run_pilot_chunk() -> record_screening_event(epoch=6)`.

**Root cause.** `run_pilot()` always restarts its chunk walk from
`target=6` on every call (`chunk_epoch_targets`), relying on each chunk's
own idempotency checks (existing checkpoint -> no retrain; existing
validation pickle -> no re-evaluation) to make a rerun a no-op. The
screening loop inside `run_pilot_chunk()` had no equivalent check: it
re-fed every cadence epoch in the chunk's range through
`record_screening_event()` regardless of whether
`pilot_orchestration_state.json`'s `logged_screening_epochs` already
recorded it. `record_official_validation_event`'s own idempotent-replay
only covers replaying the exact LAST recorded history entry, not an
earlier one a later chunk's screening has since superseded -- so once the
persisted early-stopping history's last entry was epoch 9, replaying
epoch 6 (already screened and logged) raised "out of order" instead of
being recognized as already-done.

**Fix.** In `run_pilot_chunk()`'s screening loop, an epoch already present
in `logged_screening_epochs` is now skipped outright (no re-evaluation,
no re-record) rather than always re-processed. A light consistency check
(not broad reconciliation) still guards against silently skipping
genuinely inconsistent state: a stopping-eligible epoch marked logged
must actually be present in the reloaded early-stopping history, or a
`PilotOrchestrationError` is raised. No change to checkpoint discovery,
continuation, evaluation math, early-stopping policy, or launcher
classification.

**Verification.** New end-to-end test in `tests/test_pilot_orchestration.py`,
`test_run_pilot_end_to_end_rerun_of_fully_screened_earlier_chunks_is_idempotent`,
reproduces job 45718742's exact shape (checkpoints 1-6 flat, 7-15 in
`continue_training_from_epoch006/`, `pilot_orchestration_state.json`
already listing `logged_screening_epochs: [3, 6, 9]`, early-stopping
history already ending at epoch 9 with `stopped: false`). Confirmed:
first verified to fail with the exact real-job error before the fix, then
passes after it; no train/evaluate callback invoked; both persisted state
files byte-identical before and after; `run_pilot()` proceeds straight to
`final_status: blocked_continuation_overshoot_conflict` with
`highest_screened_epoch=9`, `next_intended_screening_epoch=12`,
`overshoot_epochs=[10, 11, 12, 13, 14, 15]`,
`safe_to_continue_automatically=False`. Ran only
`pytest tests/test_pilot_orchestration.py` -- 34 passed (was 33 before this
correction's 1 new test). Full repository suite not run (binding resource
constraint for this narrow correction).

**No further Moriah run should occur until this narrow local
rerun-idempotency fix is committed.** A resubmission before that would
repeat the same crash the next time this pilot is rerun after a chunk
sequence has already been screened past its earliest chunks.

## 2026-07-30 — Stage 1 lead-6 pilot — real Moriah recovery (job 45718473): scientific recovery correct, launcher status-propagation defect found and fixed

**Scope.** Local-only correction to the two files that surface
`run_pilot()`'s result — `scripts/run_stage1_lead06_pilot.py` (CLI exit
code) and `scripts/run_stage1_lead06_pilot_moriah.sbatch` (status-fallback
classification). No change to `src/baseline/pilot_orchestration.py`: direct
code reading plus a new end-to-end test confirmed `run_pilot()` already
propagates a blocked chunk's `final_status`/`blocked_reason` correctly
through its own return value. Not a scientific-baseline change.

**What happened on Moriah.** Recovery job `45718473` (partition `catfish`,
one L4 GPU, elapsed 00:08:12, Slurm `COMPLETED`, exit `0:0`) ran the
continuation-nesting/additive-epoch fix from the entry below against the
real `emb128x64_seedA` artifact. The scientific recovery was correct: no
training occurred; the existing
`continue_training_from_epoch006/model_epoch009.pt` checkpoint was reused;
epoch 9 was screened and logged exactly once (median per-basin raw-space
NSE `0.18124855313577198`); epoch 6 remains best
(`0.20454161610527344`); early-stopping history contains eligible epochs 6
and 9 with `events_since_best_improvement: 1`, `stopped: false`; overshoot
checkpoints 10-15 remain preserved and scientifically unused. However the
launcher reported an internally inconsistent result: `status: COMPLETED`,
`pilot_exit_code: 0`, `pilot_final_status: null`, `blocked_reason: null`,
alongside a correctly computed `safe_to_continue_automatically: false` and
`overshoot_epochs: [10, 11, 12, 13, 14, 15]`.

**Root cause.** The pilot CLI's primary stdout JSON
(`pilot_stdout.json.log`) was unavailable when the launcher read it after
the job finished, so the launcher's own documented fallback path (computing
status fields directly from on-disk state via `compute_pilot_status_fields`)
engaged. That fallback correctly recomputes `overshoot_epochs` and
`safe_to_continue_automatically` from disk — which is exactly why those two
fields were populated correctly in the reported result — but it never
derived `pilot_final_status`/`blocked_reason`, leaving both `None`. The
launcher's classification block only branched on `pilot_final_status`
(`if pilot_final_status == 'blocked_continuation_overshoot_conflict':
BLOCKED_MANUAL_REVIEW_REQUIRED elif pilot_status == 0: COMPLETED`), so a
run the fallback had already determined was unsafe to continue
automatically fell through to `COMPLETED` anyway. The blocked
`run_pilot_chunk()` result was never lost inside `run_pilot()` itself — it
was lost only in the launcher's fallback-status derivation, one step removed
from where the return dict is actually built and printed.

**Fix.** Two narrow, additive changes, no checkpoint/overshoot/training
logic touched:
1. `scripts/run_stage1_lead06_pilot_moriah.sbatch`: after the existing
   on-disk fallback populates `overshoot_epochs`/`safe_to_continue_automatically`,
   a new check derives `pilot_final_status = 'blocked_continuation_overshoot_conflict'`
   and a non-null `blocked_reason` whenever `safe_to_continue_automatically
   is False` with a non-empty `overshoot_epochs` — reusing the exact status
   string `pilot_orchestration.run_pilot()` already uses for this condition
   (the "already-established convention"), not inventing a new one. Verified
   by extracting the launcher's status-classification `python -c` block and
   executing it standalone (no Slurm) against both job 45718473's exact
   on-disk shape (now classifies `BLOCKED_MANUAL_REVIEW_REQUIRED`) and an
   ordinary completed/stopped run (still classifies `COMPLETED`,
   unaffected).
2. `scripts/run_stage1_lead06_pilot.py`: now exits `1` — the launcher's own
   existing "needs a human, do not resume automatically" convention (shared
   with `FAILED_NO_CHECKPOINT`) — when `result["final_status"] ==
   "blocked_continuation_overshoot_conflict"`, instead of always exiting 0
   regardless of `final_status`. Defense-in-depth only: the launcher's
   primary classification already reads `pilot_final_status` from the JSON
   before consulting the exit code, so this does not by itself change the
   primary (non-fallback) classification path, but makes the CLI's own exit
   code meaningful for any caller (direct invocation, other tooling) that
   relies on it.

**Verification.** New end-to-end test in `tests/test_pilot_orchestration.py`,
`test_run_pilot_end_to_end_propagates_blocked_continuation_overshoot_conflict`,
reproduces job 45718473's exact on-disk shape (checkpoints 1-6 flat, 7-15 in
`continue_training_from_epoch006/`) and calls `run_pilot()` end-to-end
(not `run_pilot_chunk()` directly, unlike the existing coverage below),
confirming `final_status`/`blocked_reason`/`overshoot_epochs`/
`safe_to_continue_automatically` are all correctly non-null/populated at
the top-level return — proving the Python-level propagation was already
correct before this fix. Two new tests in
`tests/test_pilot_sbatch_launcher.py` behaviorally exercise the launcher's
status-classification snippet (extracted and run standalone, never via
Slurm/sbatch) for the blocked-fallback case and the ordinary-completed
regression case. Ran only the two directly affected focused test files:
`pytest tests/test_pilot_orchestration.py tests/test_pilot_sbatch_launcher.py`
— 52 passed (was 49 before this correction's 3 new tests: 32+1=33 in
`test_pilot_orchestration.py`, 17+2=19 in `test_pilot_sbatch_launcher.py`,
i.e. 49 = 32 + 17 before, 52 = 33 + 19 after).
Full repository suite not run (binding resource constraint for this narrow
correction).

**No further Moriah job should run until this local status-propagation fix
is committed.** A resubmission before that would repeat the same misleading
`COMPLETED` report the next time a chunk is correctly blocked.

## 2026-07-30 — Stage 1 lead-6 pilot — second qualification-run correction (continuation-nesting/additive-epoch semantics)

**Scope.** Local-only correction to `pilot_orchestration.py`, discovered by
resuming the pilot's Moriah workflow-qualification run (`emb128x64_seedA`,
Slurm job 45705457) after the 2026-07-29 evaluation-prerequisite
correction below. Not a scientific-baseline change: no hyperparameter,
split, screening-membership, or early-stopping policy value changed.

**What happened on Moriah.** The resumed job, intended to continue training
from epoch 6 to the epoch 9 chunk boundary, instead produced
`base_run/continue_training_from_epoch006/model_epoch007.pt` through
`model_epoch015.pt` with no valid epoch-9 screening result.

**Root cause.** Two compounding gaps. First, NeuralHydrology's `continue_run`
sets `is_continue_training=True` unconditionally on every call, so
`BaseTrainer._create_folder_structure` always nests output into a new
`continue_training_from_epoch{start:03d}/` subdirectory (and raises if that
directory already exists) — this nesting is not optional or an artifact of
this pilot's own config. Second, and the actual bug: the original
chunk-continuation code wrote the overlay's `epochs:` key as an absolute
target epoch rather than additive relative to the checkpoint being resumed
from. Told `epochs: 9` while resuming from checkpoint 6, NH correctly (by
its own additive semantics) trained 9 *more* epochs past 6, i.e. through 15
— an entirely correct NH response to an incorrectly-computed instruction.

**Decision — separate current/additional/logical-target epoch into three
non-overloaded `TrainChunkRequest` fields, and never trust a checkpoint's
epoch number alone.** `current_epoch` (resumed-from checkpoint),
`additional_epochs` (this chunk's additive NH `epochs:` value), and
`logical_target_epoch` (`current_epoch + additional_epochs`, used only for
the pilot's own logical scheduling, never written to the overlay). A new
`discover_physical_checkpoints()` recursively inventories every checkpoint
across the base run directory and arbitrarily-nested continuation
directories, raising `PilotOrchestrationError`
("ambiguous physical checkpoint inventory") on any duplicate epoch claim
and ignoring malformed filenames/directory names rather than guessing.
`resolve_trusted_chunk_checkpoint()` trusts a checkpoint only when its
owning physical directory exactly matches the continuation directory NH
would create for this pilot's own exact `(previous_target_epoch, epoch)`
pair; `untrusted_overshoot_epochs()` flags every other checkpoint that
merely exists at the right epoch number under different, unverified
circumstances (e.g. the real epoch 10-15 overshoot). Rejected alternative:
trusting any checkpoint found at the expected epoch number regardless of
which physical directory produced it — rejected because that is exactly
the assumption that let the additive-epoch bug silently overshoot in the
first place; distrust-by-default was chosen instead.

**Decision — never guess when it isn't safe to proceed automatically.**
The shared `_advance_chunk_via_continuation()` helper: resumes from a
trusted checkpoint idempotently if one exists; else blocks (does not
retrain) with a "manual review... required" reason if untrusted checkpoints
already occupy the target epoch range; else blocks with an "already
exists" reason if NH's target continuation directory exists but is
empty/incomplete (since `continue_run` would otherwise crash inside real
NH trying to recreate it). `compute_pilot_status_fields()` now reports four
distinct, never-conflated fields — `highest_physical_checkpoint_epoch`,
`highest_screened_epoch`, `next_intended_screening_epoch`,
`overshoot_epochs`, plus `safe_to_continue_automatically` — consumed
identically by the Slurm launcher and the evidence bundle rather than each
re-deriving its own notion of "current epoch".

**Verification.** `tests/test_pilot_orchestration.py`'s fake training
callback was rewritten to reproduce NH's real nested continuation-directory
layout (it previously wrote all checkpoints flat, which is why this class
of bug was not caught by the prior correction's tests). An adversarial
self-review noted that `default_train_chunk` itself — the exact function
that writes the `pilot_epoch_overlay.yaml` NH reads, and therefore the
function directly responsible for both the additive-epoch and
continuation-nesting bugs — had zero direct test coverage, since every test
in this file injects a fake `train_chunk_fn` instead. Its overlay-dict
construction was extracted into a pure, NH/torch-free helper,
`_continuation_overlay(request) -> dict`, and given two direct unit tests
(explicit `continue_from_epoch` case; the `current_epoch=None`
degenerate-corner case with `continue_from_epoch` correctly omitted) — a
same-behavior refactor, not a logic change. Eight pilot test
files now carry 146 tests (was 125), all passing, including: additive- (not
absolute-) epoch computation verified across two successive chunk
transitions (6→9→12); checkpoint discovery across base+one and
doubly-nested continuation directories; a loud failure on a duplicate
physical epoch claim; malformed checkpoint names/directories ignored;
idempotent resume onto an already-trusted epoch-9 checkpoint; a blocked
status (with prior checkpoints/state left untouched, including across a
repeated call) for both the untrusted-overshoot-checkpoints case and the
empty-pre-existing-continuation-directory case; screening/tracking never
touching epochs 10-15 merely because they physically exist; an evaluator
failure leaving prior logical state completely unchanged; a checkpoint
reference logged at its resolved physical path; and — the direct
regression check — the exact real job-45705457 evidence shape (checkpoints
1-6 flat, 7-15 in one nested continuation directory, no valid epoch-9
result) reproduced end-to-end, confirming the corrected orchestration
trusts/screens exactly epoch 9 and a further chunk attempt blocks rather
than resuming from the wrong checkpoint. A Windows-only `short_tmp_path`
fixture was added to `tests/_pilot_support.py` because these now-
realistically-deep nested paths, combined with pytest's own long default
tmp-dir prefix, exceeded Windows' 260-character `MAX_PATH` in the local
test environment — a local-testing accommodation only; Linux (where real
Moriah/h2o runs happen) has no such limit. Full suite re-run: 1173 passed,
the same 6 pre-existing `neuralhydrology`/`torch` import-only collection
errors as before (expected in this local environment); 1 test
(`test_package_builder.py::test_evidence_promotion_failure_after_package_success_rolls_back_both`,
untouched by this work) failed only under full-suite load with a Windows
file-lock `PermissionError` during atomic promotion, confirmed to pass
cleanly in isolation — pre-existing flakiness, not a regression.

**Current status: epoch-9 recovery is safe; further training is not.**
`emb128x64_seedA` remains paused after epoch 6. **Status (superseded):** see
the 2026-07-30 closure entry above (job `45722908`) —
`emb128x64_seedA` is now complete, epoch 6→15 was adopted without
retraining, and early stopping fired at epoch 15.
`continue_training_from_epoch006/model_epoch009.pt` sits in exactly the
directory this pilot's own chunk sequence would produce, so it is trusted:
one controlled recovery invocation of the corrected orchestration reuses
that checkpoint, runs validation screening for epoch 9, and records the
event — with no retraining, and with no manual movement or deletion of any
checkpoint required first. What remains unsafe is continuing training past
epoch 9: epochs 10-15 already physically exist (the original bug's
accidental byproduct) and stay preserved, untouched, scientifically-unused
artifacts; `overshoot_epochs`/`safe_to_continue_automatically=False` block
any attempt at a further 9→12 chunk rather than retraining over or past
them, pending a later decision on how to handle that continuation. This
epoch-9 recovery has not been executed on Moriah — it is expected behavior
of the locally tested repair only; no resume has been submitted since this
correction. Full detail: `docs/stage1_lead06_pilot_v001.md`'s "Second
Moriah failure and continuation-nesting/epoch-semantics correction"
section.

**Residual risk flagged, not resolved (requires real Moriah/NH verification,
not actionable in this local session).** The module docstring's claim that
`continue_from_epoch` is a real, recognized NH `Config` property — one that
pins `continue_run`'s start epoch, overriding NH's own default of resuming
from the highest-numbered checkpoint physically present in `run_dir` — is
not independently corroborated anywhere in this decision log (checked by
grep across this file for `continue_from_epoch`, `_get_start_epoch_number`,
`_restore_training_state`, `resume_from_epoch`, `start_epoch_number`: no
matches), and `neuralhydrology` is not installed in this local environment,
so it cannot be checked against real NH source here. In the pilot's own
exercised code path this is very likely inert either way:
`_advance_chunk_via_continuation`'s pre-flight checks only ever allow a
`train_chunk_fn` call once `untrusted_overshoot_epochs` has confirmed no
checkpoint occupies `(resume_from_epoch, chunk_target_epoch]`, which by the
pilot's own linear chunk construction means `start_dir` should contain
checkpoints only up through `resume_from_epoch` — so NH's own
highest-checkpoint default should coincide with the explicit
`continue_from_epoch` value regardless of whether NH actually honors the
latter. The one not-fully-closed corner: `untrusted_overshoot_epochs` does
not check for a checkpoint epoch *above* `chunk_target_epoch` sitting flat
in `start_dir` itself (as opposed to inside a nested continuation
directory, which is the only layout ever observed in real evidence) — such
a stray file does not exist in the real job-45705457 evidence and would
require an unrelated anomaly (e.g. a prior `start_run` overrunning its
intended epoch count) to arise. Recommended resolution: confirm
`continue_from_epoch` against real NH 1.13 `Config`/`basetrainer.py` source
on h2o/Moriah before the next real qualification-run resume; no code change
is proposed here since the exercised path is safe regardless and the
corner case is speculative.

## 2026-07-29 — Stage 1 lead-6 pilot — qualification-run integration correction (explicit evaluation prerequisite)

**Scope.** Local-only correction to `pilot_orchestration.py`, discovered by
the pilot's first real Moriah workflow-qualification run (`emb128x64_seedA`,
Slurm job 45695059). Not a scientific-baseline change: no hyperparameter,
split, screening-membership, or early-stopping policy value changed.

**What happened on Moriah.** Training succeeded through epoch 6
(checkpoints + optimizer states 1-6 intact, peak RSS ~96.4GB). Orchestration
then raised `NHSeedEvaluationError: missing validation results pickle` at
`validation/model_epoch003/validation_results.p`. A separate evaluation-only
job (45698612) confirmed NH's `start_evaluation` can produce this pickle on
demand (epochs 3 and 6, 400 basins each, ~84.6MB each, 11:34 elapsed,
~1.96GB peak RSS on an L40S — a single data point, not a validated general
requirement).

**Decision — add an explicit evaluation-prerequisite step to orchestration,
rather than assuming NH's in-training `validate_every` cadence always
persists results.** `pilot_orchestration.py` gained
`ensure_validation_results(nh_run_dir, epoch, evaluate_checkpoint_fn=...)`,
called before every screening checkpoint's `evaluate_screening_checkpoint()`.
It checks the canonical result-pickle path
(`nh_seed_evaluation.period_results_path()`, extracted as the single helper
every caller must use — never independently reconstructed), reuses an
existing result unchanged, or explicitly invokes NH evaluation through a new
injectable `evaluate_checkpoint_fn` seam (production default
`default_evaluate_checkpoint`, mirroring `scripts/run_stage1_nh.py`'s own
`eval` subcommand rather than duplicating it), then fails loudly
(`PilotOrchestrationError`, no partial state persisted) if the pickle still
doesn't exist afterward. Rejected alternative: triggering evaluation inside
`pilot_screening_eval.py` itself — kept out of scope so that module remains
a pure metric reader, per its own docstring (corrected in this same change
to stop claiming evaluation is "never...independently triggered").

**Decision — resume must not retrain or re-evaluate what's already on
disk.** Verified directly against the real failure shape: a run with
checkpoints and saved result pickles through epoch 6 resumes training at
epoch 9 without retraining epochs 1-6 or re-invoking evaluation for epochs
whose pickle already exists.

**Verification.** `tests/test_pilot_orchestration.py`'s fake training
callback no longer also fabricates validation-result pickles (it now writes
checkpoint bytes only) so tests exercise the real missing-prerequisite path;
a new, separate fake evaluation callback was added. Eight pilot test files
now carried 124 tests (was 95; see the 2026-07-29 adversarial-review entry
below for a further correction bringing this to 125), all passing, including
six new scenarios:
missing-result triggers explicit evaluation; existing result is reused
without re-invoking the evaluator; resume from the exact real
qualification-run failure shape retrains/re-evaluates nothing already
present; a future screening checkpoint (epoch 9) triggers explicit
evaluation before screening; evaluator failure is safe (no false
screening/stopping event, checkpoints untouched); epoch-3 diagnostic-only
behavior and existing orchestration idempotency are both unaffected. Full
suite re-run: 1152 passed excluding 6 pre-existing
`neuralhydrology`/`torch` import-only collection errors (expected in this
local environment); 2 tests in `test_package_builder.py`/`test_package_audit.py`
(untouched by this work) failed only under full-suite load with a Windows
file-lock `PermissionError` during atomic promotion, confirmed to pass
cleanly in isolation both with and without this change (via `git stash`) —
pre-existing flakiness, not a regression.

**Current status: not complete.** `emb128x64_seedA` remains paused after
epoch 6. No resume has been submitted to Moriah. **Status (superseded):**
see the 2026-07-30 closure entry above (job `45722908`) —
`emb128x64_seedA` is now complete, epoch 6→15 was adopted without
retraining, and early stopping fired at epoch 15. Full detail:
`docs/stage1_lead06_pilot_v001.md`'s "Moriah workflow-qualification run and
orchestration correction" section.

## 2026-07-29 — Adversarial review of the evaluation-prerequisite correction

**Scope.** Review-only pass over the above correction, before commit. No
Moriah access, no scope broadening, no scientific-baseline change.

**Finding — repeated-call logging-handler leak (fixed).** NH's
`neuralhydrology.utils.logging_utils.setup_logging` unconditionally opens a
new `FileHandler`/`StreamHandler` on every call and attaches them via
`logging.basicConfig`, which is a documented no-op once the root logger
already has handlers. `default_evaluate_checkpoint` may be called once per
screening epoch within a single long-lived orchestration process (unlike
the single-shot `scripts/run_stage1_nh.py eval` CLI this pattern was copied
from), so every call after the first leaked an open, never-attached file
descriptor against the same `output.log`. No duplicate log output resulted
(the leaked handlers are never attached), but the descriptor leak itself
was real. Fix: a new `root_logger_has_file_handler()` guard in
`pilot_orchestration.py` checks the root logger for an already-attached
`FileHandler` on the same resolved path before calling `setup_logging`;
`default_evaluate_checkpoint` now skips the redundant call when one is
already present. One new stdlib-only test added
(`test_root_logger_has_file_handler_detects_only_a_matching_filehandler`),
bringing the pilot test-file total to 125 (was 124).

**Findings — no correction needed.** (1) Production evaluation callback:
directly confirmed against the vendored NH source (`tester.py`,
`evaluate.py`, `logging_utils.py`, `nh_run.py`, `config.py`) that dataset
registration happens before `get_tester()`/`get_dataset()` are ever reached,
the callback loads the frozen `run_dir/config.yml` and never calls
`Config.dump_config` (the only on-disk-config write path in NH, called only
from config-generation helpers and the training-time `Logger`, never from
the evaluation path), evaluates exactly the requested epoch and
`period="validation"`, and never touches the `test` period or a spatial
holdout (`Tester.evaluate()`'s only period-conditional branch is
validation-only basin subsampling; the period itself is fixed by
`EvaluationRequest.period`'s `"validation"` default, never overridden).
(2) Real resume mechanics: `continue_run`'s merge/`is_continue_training`
behavior and the epoch-overlay file confirm the intended epochs 7-9
continuation; this is code-inspection-confirmed and fake-trainer-test
approximated, but NH's real resume-epoch selection (reading its own latest
checkpoint) is not exercised by any local test and needs Moriah
verification. (3) Failure atomicity: `ensure_validation_results` checks
`.is_file()` (correctly excludes a bare directory); a zero-byte or corrupt
pickle is not specially detected there, but `load_period_results`'s
`pickle.load` call raises a standard, clear exception (`EOFError` for
zero-byte, `UnpicklingError`/similar for corrupt) before any screening or
early-stopping state is touched (`evaluate_screening_checkpoint` is called,
and can raise, strictly before `record_screening_event` in
`run_pilot_chunk`'s loop) — this already satisfies "reject clearly, not
treated as success" without a code change. (4) Canonical path helper:
`period_results_path()` matches NH's real `_get_weight_file`/`_save_results`
convention exactly; no circular import (`nh_seed_evaluation.py` does not
import from `pilot_orchestration.py`). (5) Tests: all six existing
evaluation-prerequisite/resume tests were re-traced against the actual test
file content; each checklist item is satisfied by an existing assertion.

**Verification.** `tests/test_pilot_orchestration.py` re-run in isolation:
16 passed. Full repository suite re-run after this review's change: 1155
passed, 0 failed, the same 6 pre-existing `neuralhydrology`/`torch`
import-only collection errors as before (expected in this local
environment). This is consistent with the prior correction's documented
1152-passed baseline (1152 + the 2 tests that were flaky under full-suite
load in that run, which did not fail this time + 1 new test added here =
1155) -- no regression attributable to this review's change; the Windows
file-lock flakiness noted in the prior entry simply did not reproduce on
this run, which is expected of a load-dependent race and is not evidence
it was fixed.

**Correction to a prior report.** The previous final report's git-status
description was imprecise: `git status --short` entries beginning with
` M` (space-then-M) denote **unstaged** modifications, not staged ones.
Nothing was staged or committed at any point in this work.

## 2026-07-27 — Stage 1 lead-6 optimization pilot — implementation decisions

**Scope.** Local implementation of the six-run lead-6 optimization pilot
agreed after the validation-and-optimization foundation phase. Full
documentation: `docs/stage1_lead06_pilot_v001.md`. **No Moriah connection,
no Slurm submission, no training, no full-population evaluation, no
temporal-test or spatial-holdout access occurred in this increment.**

**Decision 1 — pilot epoch sub-cap implemented as a layered override, not
an edit to the committed early-stopping policy.** The pilot needs a
36-epoch budget; `config/stage1_early_stopping_policy_v001.yaml` (unmodified,
still binding for future non-pilot runs) caps at 40.
`src/baseline/pilot_early_stopping.build_effective_policy()` loads the base
policy, validates none of its core fields (`metric_name`,
`higher_is_better`, `min_epoch_before_stop`, `min_delta`,
`patience_events`) have drifted from what the pilot expects, and layers
`max_epoch_budget = min(base, 36)` under a renamed `policy_name`. Rejected
alternative: forking a second early-stopping policy file, which would
duplicate mechanics already implemented and tested.

**Decision 2 — Seed A recovered read-only, not re-derived or reused
implicitly.** The historical full-population seed run never set `seed`
explicitly; NH auto-assigned and wrote it back to the frozen run's
`run_dir/config.yml`. That value (967139) was recovered by a single
lightweight `grep` on the Moriah login node against the already-frozen
config file — no compute, no Python import, no training triggered. Seed B
fixed at 1729 (confirmed `!= 967139`, so no collision-avoidance fallback
needed).

**Decision 3 — one shared, non-collected test-fixture helper module, a
narrow deviation from the repo's per-file test convention.** No
`tests/conftest.py` exists anywhere in this repo (confirmed by search); the
established convention is self-contained fixtures per test file. Eight new
pilot test files all needed the same full-union-package, perfect-NSE
validation-results, and screening-basin-file fixtures — duplicating that
setup eight times was judged worse than one small, clearly-scoped,
non-`test_`-prefixed helper (`tests/_pilot_support.py`, never collected by
pytest). Every test file still defines its own test logic locally; only
fixture *construction* is shared.

**Decision 4 — `run_pilot()`'s evidence bundle write is unconditionally
`force=True` internally, decoupled from the caller's `force` argument.**
Found and fixed during implementation: an earlier version of
`pilot_orchestration.run_pilot()` threaded a single `force` flag to both
"regenerate NH config on resume" and "overwrite the evidence bundle
directory," which meant a safe, idempotent resume call (`force=False`, the
correct default — nothing needs regenerating) would also fail to update
the evidence bundle, or an unsafe `force=True` resume would silently
redo config generation it shouldn't. The two concerns are now separate:
`force` (caller-supplied) gates only NH config regeneration; the internal
evidence-bundle write always uses `force=True`, since the bundle is a
derived, deterministic summary that must always reflect the latest known
state on every call. Verified end-to-end by
`tests/test_pilot_orchestration.py::test_run_pilot_resume_is_idempotent_and_does_not_retrain`
and directly by
`tests/test_pilot_evidence_bundle.py::test_write_pilot_evidence_bundle_force_allows_overwrite`.

**Decision 5 — no fundamental incompatibility found; no scope escalation.**
The task's instruction was to stop and report if a broad redesign were
required. No such incompatibility was found. The only issues encountered
were two ordinary test-authoring bugs in newly-written test code itself
(a dead, non-assertive test stub; a naive substring check that
false-flagged `--force` appearing inside an explanatory comment in the
sbatch script) — both fixed in place, neither indicating a defect in the
implementation modules under test.

**Verification.** Eight new focused test files, 95 tests, all passing.
Full pre-existing repository suite re-run: 1122 passed; 3 initial failures
in `tests/test_package_builder.py`/`tests/test_package_audit.py` (files not
touched by this work) traced to a Windows-specific `PermissionError` during
atomic package-directory promotion under load, confirmed to pass cleanly
in isolation — a pre-existing local-environment flake, not a regression; 6
pre-existing collection errors from `neuralhydrology`/`torch` imports,
expected in this local Windows environment (those tests require the
h2o/Moriah environment). Two item-10 checklist phrases ("silent
identity-fallback rejection", "unexpected-embedding rejection") were
confirmed already covered — the former by this pilot's own exact-shape
assertions per run_id (raw runs assert `embedding_hiddens is None`,
i.e. no `statics_embedding` key, matching the NH 1.13 identity-fallback
behavior the Part B static-pathway audit first identified), the latter by
`tests/test_nh_config_generation.py`'s existing structural-validation
tests for malformed `statics_embedding` specs — no new test was needed for
either.

**Not done in this entry.** No Moriah job submitted. No training. No full-
population evaluation. No temporal-test or spatial-holdout access. No
change to the certified Compact Scientific Package or canonical split
membership. No screening-subset regeneration. No hydrograph atlas. No
automated sweep. No EA-LSTM work. Nothing generated by this pilot
committed.

## 2026-07-24 Commit-readiness review of the full-population NH config-generation increment — two safeguard fixes

**Scope.** Focused adversarial-validation review (no package transfer, no Moriah/Slurm work, no
training, no sweep generation) of the full-population config-generation/structural-preflight
increment documented immediately below, checking the two-bundle design's holdout-safety, scaler
provenance, basin-membership equality, two-config consistency, and minimality properties.

**Findings and fixes.** Two real, minimal gaps were found and fixed, both in how the
`spatial_holdout` bundle is distinguished from a normal trainable experiment (NH 1.13's
`Config._check_cfg_keys` rejects any config.yaml key outside its own property list, so a
`population_role`-style marker cannot live inside `config.yaml` itself — confirmed by direct
`inspect.getsource` inspection of the installed NH 1.13 `Config` class):

1. `write_generated_config`'s default `experiment_name` was previously identical for the
   `development` and `spatial_holdout` bundles (both `stage1_compact_lead06_seq24_v001` unless a
   caller explicitly overrode it). Fixed: when `population_role == "spatial_holdout_test_only"` and
   no explicit `experiment_name` is given, a `_spatial_holdout_test_only_eval` suffix is appended.
   Confirmed non-breaking: no existing test asserted the old default value.
2. Added a sibling `TEST_ONLY_DO_NOT_TRAIN.txt` file, written by `write_generated_config` only for
   the `spatial_holdout` bundle, explaining the bundle is test-only evaluation machinery and that its
   train/validation basin lists are the development population present only to satisfy NH's config
   schema. `check_generated_config_structure`'s existing `require_identical_basin_sets=False`
   (holdout-role) branch now requires this file to exist (`test_only_marker_file_present`); the
   `development`-role path (`require_identical_basin_sets=True`, the default) is untouched.

**New tests.** 9 added (4 in `tests/test_nh_full_population_config_generation.py`, 5 in
`tests/test_nh_full_population_structural_preflight.py`; full-population totals now 14 + 15 = 29),
covering: a normalization-collision duplicate (short zero-padded STAID vs. its already-8-digit
form); a California basin substituted for a development basin while preserving the exact 2,557
total count; the distinguishable default `experiment_name`; the marker file present for the holdout
bundle and absent for the development bundle; the marker-file-missing preflight error; a development
basin leaking into the holdout bundle's own `test_basins.txt` (mirroring the existing
holdout-basin-into-train test); post-generation `dynamic_inputs` order-drift detection; and a
non-empty but structurally malformed external scaler.

**Verified without new code.** Non-eight-character STAID support: the real
`development_train.txt`/`spatial_holdout_nonca.txt` union already contains 5 fifteen-character
STAIDs, exercised end-to-end by the existing real-splits happy-path test. Static-column-count edge
cases (472/474/473-with-duplicate): `validate_static_attribute_contract` (unchanged, reused as-is)
checks for internal duplicates unconditionally before the count check and uses an exact `!=`
equality comparison, so both a wrong-count and a duplicate-within-473 input are already
structurally rejected; no full-population-specific test was added for this pre-existing,
unmodified compact-package function.

**Validation.** All 29 full-population tests, all 41 pre-existing compact-package tests, and the
directly-affected NH dataset tests pass (91 total). Full suite (`pytest tests/ -q`, 889 tests): 889
passed, 0 failed, on this run — the known Windows `os.rename`-based flake in
`tests/test_package_builder.py::test_repeated_builds_produce_equivalent_manifest` did not reproduce;
it was separately re-confirmed to pass in isolation, and nothing in this increment touches
`src/baseline/package_builder.py`. Repeated the synthetic 2,557-basin dry run and this time directly
read the generated `config.yaml` files, the `TEST_ONLY_DO_NOT_TRAIN.txt` marker, and the basin-list
`.txt` artifacts (not only prior JSON summaries): development train/validation/test are each exactly
the 2,307 development basins; spatial-holdout train/validation are exactly the 2,307 development
basins with 0 overlap against its 250-basin test set; the preflight CLI end-to-end
(`--skip-dataset-construction`, placeholder NetCDFs) reports **PASS — 56 OK, 0 errors, 0 warnings**,
including the new `test_only_marker_file_present` check.

**Not done / out of scope for this review.** No package transfer, Moriah dataset construction, Slurm
preflight, training, or sweep/config-matrix work — unchanged from the increment below. Nothing was
committed as part of this review.

---

## 2026-07-24 Full-population (development + spatial-holdout) NH config-generation + structural-preflight local implementation increment

**Scope.** Local-only implementation increment (no h2o/Moriah access, no data transfer, no NH
training run, no Slurm job, no package rebuild) extending the 2026-07-22 compact-package
config-generation/structural-preflight machinery (above/below) to the certified full non-California
package (`stage1_scientific_package_v002`, 2,307 development-training + 250 spatial-holdout basins,
Gate 4 PASS). Renders and validates exactly one scientific configuration — lead 6 h, sequence length
24 h, target `qobs_mm_per_h_lead06`, the 8 approved dynamic inputs in binding order, all 473 retained
static `model_input` attributes, train 2020-10-14→2023-12-31 / validation 2024 / test 2025 — as
**two strictly separated config bundles**, not one:

- **`development`** bundle: train == validation == temporal-test, the 2,307 `development_train`
  basins (only the date period differs across the three periods).
- **`spatial_holdout`** bundle: test-only; its own train/validation basin lists are the
  *development* population (2,307 basins), never a holdout basin, so the bundle cannot be misused
  to fit or validate on holdout basins; its test list is the 250 `spatial_holdout_nonca` basins.

This mirrors the binding basin-role separation: the 250 spatial-holdout basins may only ever be
evaluated, never used for training, normalization/scaler fitting, validation, early stopping, or
checkpoint/config selection; temporal testing (development basins, 2025) and spatial-holdout
testing (holdout basins, 2025) are distinct evaluation roles despite sharing the same calendar test
period.

**New/changed code.** `src/baseline/nh_config_generation.py`: two new pinned constants
(`EXPECTED_DEVELOPMENT_BASIN_COUNT = 2307`, `EXPECTED_SPATIAL_HOLDOUT_BASIN_COUNT = 250`, not
CLI-exposed); `FullPopulationBasinMembership` and `validate_full_population_basin_membership`
(requires the package's basin set to equal *exactly* the union of the canonical
`development_train`/`spatial_holdout_nonca` splits — no subset, no extra, no California, no
dev/holdout overlap, no duplicates, exact 2,307/250 counts); `GeneratedConfigBundle` extended with
`population_role`, `train_basin_ids`, `validation_basin_ids`, `test_basin_ids` (all default `None`,
preserving old single-population behavior byte-for-byte when unused);
`generate_stage1_full_population_nh_config_bundles` (builds both bundles from one shared
policy/common-kwargs render, so both bundles share one set of period dates);
`write_generated_config` extended to honor the new per-period basin-id overrides (falling back to
`bundle.basin_ids` when unset) and to record `population_role`/`train_basin_count`/
`validation_basin_count`/`test_basin_count` in `generation_manifest.json`. `src/baseline/
nh_structural_preflight.py`: `check_generated_config_structure` extended with
`expected_train_basin_count`/`expected_validation_basin_count`/`expected_test_basin_count`,
`require_identical_basin_sets` (default `True`, preserves old behavior), `require_test_disjoint_
from_train_validation`, and `expect_generated_basins_equal_package_manifest` (default `True`);
`check_flashnh_external_scaler_test_construction` (constructs only the `test`-period
`FlashNHDataset` against the spatial-holdout bundle, reusing an externally supplied scaler
unchanged, never fitting a new one, never touching `cfg.train_dir`); `run_full_population_
structural_preflight` (composes both bundles' structural checks plus, unless skipped, real
`FlashNHDataset` construction — development train/validation/test, then spatial-holdout test-only
reusing the development scaler). Two new thin CLIs: `scripts/generate_stage1_full_population_nh_
config.py` and `scripts/check_stage1_full_population_nh_config_preflight.py` (package-root passed
via required `--package-root`, never hard-coded; Moriah path only ever appears as a caller-supplied
argument value).

**Test coverage.** 20 new tests, all passing: 10 in `tests/test_nh_full_population_config_
generation.py` (basin-membership validation against the real committed canonical splits — exact
union acceptance, missing/extra/California/duplicate/overlap rejection — plus end-to-end dual-bundle
generation and written-file contract checks) and 10 in `tests/test_nh_full_population_structural_
preflight.py` (5 real-scale Layer-1 structural tests using the real 2,307/250 split-derived basin
lists with placeholder NetCDF files, since `_find_basin_netcdf` only checks file existence; 5
tests against a tiny synthetic 4-basin fixture exercising the external-scaler test-only construction
directly — real orchestrator Layer-1 checks are pinned to the real 2,307/250 counts and are not
compatible with a tiny synthetic fixture, so full-orchestrator end-to-end coverage uses
`run_dataset_construction=False` against the real-scale fixture instead). All 41 pre-existing
compact-package tests (`tests/test_nh_config_generation.py`, `tests/test_nh_structural_
preflight.py`) continue to pass unaffected. Full-suite regression run (`pytest tests/ -q`, 880
tests): 879 passed, 1 failed — `tests/test_package_builder.py::test_repeated_builds_produce_
equivalent_manifest`, the same pre-existing non-deterministic Windows `os.rename`-based flakiness
already logged in the 2026-07-22 entry below; it predates this increment, is unrelated to any file
touched here, and passed on an isolated rerun.

**Local dry run.** Built a synthetic 2,557-basin fake package matching the real
`development_train.txt` (2,307) + `spatial_holdout_nonca.txt` (250) union, ran both new CLIs
end-to-end: the generator produced a `development` bundle with train/validation/test all equal to
the 2,307 development basins and a `spatial_holdout` bundle with train/validation equal to the
2,307 development basins and test equal to the 250 holdout basins; the preflight script (with
`--skip-dataset-construction`, since no real package or NetCDF payloads were built) reported
**PASS — 55 OK checks, 0 errors, 0 warnings**. Temporary directories were removed after the dry run;
no output was committed.

**Unaffected / not reopened.** The certified Compact Scientific Package and the full
`stage1_scientific_package_v002` package were not modified, rebuilt, or transferred. No h2o or
Moriah connection was made. No training was run. No Slurm file was written. Only the single
lead06/seq24 configuration was rendered, as two role-separated bundles — not the full 16-config
matrix. The existing `scaler={}` safeguard at train-period `get_dataset()` call sites (2026-07-22
entry, below) was preserved and reused, not redesigned.

**Not done.** Package transfer to Moriah, the real Moriah Slurm structural-preflight run, real
full-population dataset loading, model training/evaluation, and generation of the remaining 15
lead × sequence-length configurations all remain not done.

---

## 2026-07-24 Gate 4 — Full non-California Scientific Package (v002) independently audited — PASS

**Result.** The Gate 4 independent auditor (`src/baseline/package_audit.py`,
`scripts/audit_stage1_compact_scientific_package.py`, `tests/test_package_audit.py`, commit
`98b7d42f23963e76e02ad3991d7298d3ada98ee3`) was rerun for real on h2o, in full mode, against the
authoritative full non-California Stage 1 Scientific Package at
`/data42/omrip/Flash-NH/tmp/stage1_scientific_package_v002` (2,307 development-training + 250
spatial-holdout basins, build commit `61d3819deb55240652276765c6a96d12ed3ce539`). **Status: PASS —
0 errors, 1 warning, 260,870 OK checks (260,871 total).** Audit output written to
`/data42/omrip/Flash-NH/tmp/stage1_scientific_package_v002_gate4_audit/full_rerun_20260724T110557Z`.
The transferred evidence archive was independently reviewed locally; its SHA-256
(`9cc9f8e63d6c9825c2bf765106a20a58ce0560a1d733bc815ec0846f02071ed0`) was verified against the
transfer, and every generated-output checksum inside it was independently recomputed and matched
byte-exact. The build commit and the auditor commit remain intentionally distinct identities.

**This rerun follows an earlier FAILED full audit (errors=9) of the same package**, root-caused to
two auditor-side false positives (not package defects) and fixed in this commit:
1. A 1-float32-ULP tolerance (`QC_CSV_STORAGE_ULP_TOLERANCE`) for the non-authoritative
   QC-CSV-versus-NetCDF finite-value comparison only, absorbing a confirmed xarray/netcdf4
   write-path rounding artifact. Does not relax any authoritative package-content check.
2. `imputed_value_mask_basin_order` split into a strict, ERROR-severity
   `imputed_value_mask_basin_membership` check and a separate, non-blocking WARNING-severity
   `imputed_value_mask_basin_order` check — because every downstream imputation-placement check
   re-indexes the mask by basin label, so row order alone cannot affect correctness.

**The single remaining warning** is exactly `imputed_value_mask_basin_order`: exact basin-index
membership against the accepted 2,557-basin selection (zero missing, zero extra) with a differing
row order. Non-blocking by construction — `imputed_value_mask_basin_membership` reports OK, and
audit `status` is derived solely from `error_count`.

**Not done in this closure.** The package was **not** rebuilt (`build_git_commit` unchanged at
`61d3819...`); no static artifact or other source input was modified — `imputed_value_mask.parquet`
(sha256 `a22c8bf9...816437df`) and `imputed_static_attributes.parquet`
(sha256 `5be00a3b...23c91823b8bd4b6e24`) match the values already recorded above. No Moriah
transfer, no NeuralHydrology configuration generation, and no training occurred. **This closes the
production package build-and-independent-audit phase for `stage1_scientific_package_v002`; it does
not establish scientific model skill.**

## 2026-07-24 Full non-California static-attribute preparation — real h2o run PASS

**Decision.** Ran `scripts/prepare_stage1_full_static_attributes.py` for real on h2o (not a
synthetic/dry-run) against the canonical `stage1_static_attributes_v002` matrix
(`/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002.parquet`,
sha256 `4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297`), orchestrating the
existing development-only median-imputation and development-only exact zero-variance
trainability-projection primitives (`src/baseline/static_preparation.py`, see the 2026-07-20 and
2026-07-23 entries below) end to end for the first time against the real basin population.

**Populations (binding, verified directly from the returned evidence — not from a prior summary).**
Fit population: 2,307 development-training basins
(`config/stage1_baseline_splits_v001/development_train.txt`, sha256
`397ab432564c18c3abc5158a47ada2b28840bbf6f0c213d2475444fded33858f`). Applied to the full
2,557-basin non-California package population (2,307 development-training + 250 spatial holdout,
`config/stage1_baseline_splits_v001/spatial_holdout_nonca.txt`, sha256
`76d1c546e703b1b5aa8f4a3ead971327de0151dae4fcce0c90b1272da0f587b7`). Both manifests record
`fit_basin_scope: "development_training_only"`; the spatial holdout did not influence imputation
medians or zero-variance detection.

**Result.** 473 candidate `model_input` columns → **473 retained, 0 excluded**
(`zero_variance_manifest.json`: `candidate_column_count=473`, `retained_column_count=473`,
`excluded_column_count=0`, `excluded_columns=[]`). No column was entirely missing in the
2,307-basin fit population (`imputation_manifest.json`: `columns_all_nan_in_fit_population=[]`).
Zero remaining missing values after imputation across all 473 columns for all 2,557 applied basins
(`n_missing_after_apply=0` for every column, confirmed by grep over the full manifest). Because 0
columns were excluded, the retained static table is byte-identical to the imputed static table
(`retained_static_attributes.parquet` and `imputed_static_attributes.parquet` share sha256
`5be00a3b068351bffd40a3cf72991a3df888700034831123c91823b8bd4b6e24`). The full retained-column list
is in `retained_static_columns.txt` (473 lines) / `zero_variance_manifest.json`; not reproduced here.

**Modeling decision.** Use all 473 canonical static `model_input` columns for the first
full-population Stage 1 model — no run-specific static-column reduction applies. The 32-basin
compact-smoke 13-column zero-variance exclusion (2026-07-23 "Compact NeuralHydrology integration
smoke" Finding 1, below) remains compact-population-specific historical evidence only and was
confirmed **not** reused, inherited, or reopened by this run.

**Output (h2o-resident, generated evidence, not committed).**
`/data42/omrip/Flash-NH/tmp/stage1_full_static_attributes_v001/` — `imputed_static_attributes.parquet`,
`imputed_value_mask.parquet`, `imputation_manifest.json`, `retained_static_attributes.parquet`,
`zero_variance_manifest.json`, `retained_static_columns.txt`, `excluded_zero_variance_columns.txt`,
`run_summary.json`. A smaller evidence bundle (manifests, column lists, log, checksum files — not
the parquet tables) was transferred locally to
`tmp/stage1_full_static_attributes_v001_evidence/` (untracked, not committed).

**Checksums** (from `run_summary.json` / `evidence_checksums.txt` / `parquet_checksums.txt`; all
standard 64-hex-character SHA-256, cross-verified consistent across all three files):
```
input matrix (stage1_static_attributes_v002.parquet):
4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297

column manifest (stage1_static_attributes_v002_column_manifest.json):
02505eb4893e6848f7cbc4eabd2cdf40dd6aee64156d41744aebcbe4409f0e00

development_train.txt:
397ab432564c18c3abc5158a47ada2b28840bbf6f0c213d2475444fded33858f

spatial_holdout_nonca.txt:
76d1c546e703b1b5aa8f4a3ead971327de0151dae4fcce0c90b1272da0f587b7

imputation_manifest.json:
6c814ebc76d9ac1e7f2986499f9e491a3e19382af89e1a54bed9b18df8d295be

zero_variance_manifest.json:
d05d5956486e7cea23b389116f2aa2a5220ecac39f02368bfadea722fbbb6d00

retained_static_columns.txt:
4da4379eb93aee629dc4b93d54b13198f6c17937886b28cfd34fb01721726bd9

excluded_zero_variance_columns.txt (empty file — 0 excluded columns):
01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b

imputed_static_attributes.parquet / retained_static_attributes.parquet (identical):
5be00a3b068351bffd40a3cf72991a3df888700034831123c91823b8bd4b6e24

imputed_value_mask.parquet:
a22c8bf92639d46d93343380cf1eac6ced43ed7defd56c1a25f9c6ea816437df
```

**Not done in this phase.** No NetCDF package was built; `src/baseline/package_builder.py` was not
invoked; no NeuralHydrology configs were generated; no training ran; the static-preparation
script/tests (`scripts/prepare_stage1_full_static_attributes.py`,
`tests/test_prepare_stage1_full_static_attributes.py`, committed 2026-07-24 as `61d3819`) were not
modified in this pass — this entry is a docs-only closure of the real-run evidence recorded above.

## 2026-07-23 Development-population zero-variance trainability projection — mechanism implementation

**Decision.** Implemented a reusable fit/apply mechanism, in the style of
the existing development-only median-imputation primitives
(`fit_development_median_imputation` / `apply_imputation`,
`src/baseline/static_preparation.py`, 2026-07-20 2K-G-I primitives
increment), that identifies static `model_input` columns with **exactly**
zero variance over the Stage 1 development-training population (2,307
basins) after development-only imputation: `ZeroVarianceFit`,
`fit_zero_variance_projection`, `apply_zero_variance_projection`,
`build_zero_variance_manifest`, `write_zero_variance_manifest`.

**This is a run-specific trainability projection, not a package-schema
change.** The canonical static matrix and the authoritative package
contract remain **473 `model_input` columns**, unmodified —
`package_netcdf.py`, package serialization, and canonical static-matrix
construction were not touched. A future config-generation step will consume
the frozen retained-column manifest produced here; the package itself will
still carry all 473 columns.

**Method (approved).** Exact post-imputation constancy: a candidate column
is excluded when every finite post-imputation value across the
development-training fit population is identical (`nunique(dropna=False) <=
1`). No epsilon/near-zero-variance threshold — that remains a separate,
out-of-scope, possible future modeling question. Post-imputation values that
are still non-finite cause a hard failure rather than a silent zero-variance
classification. The candidate-column list's supplied order is preserved
(not alphabetized); the retained list is the canonical order with excluded
columns removed. The fit is deterministic regardless of the input matrix's
row order (fit rows are re-selected and sorted by basin id via the existing
`select_basin_rows`/`join_eligible_with_matrix` path). At least one column
must be retained, or the fit fails loud.

**Fit/apply separation (binding).** The fit reads **only** the explicitly
supplied development-training basin IDs — never validation, temporal-test,
spatial-holdout, or California basins. The frozen retained/excluded column
list is then applied, unchanged, to any other population via
`apply_zero_variance_projection` (a plain column selection against the
frozen list — it never recomputes variance on the target population, so a
column that happens to be constant only within e.g. the spatial-holdout
subset is still retained if it varied in the development-training fit).

**Relationship to the compact-smoke 13-column exclusion.** The 32-basin
compact-smoke zero-variance exclusion list (`CANALS_MAINSTEM_PCT`,
`CDL_DURUM_WHEAT`, `CDL_ORANGES`, `CDL_RICE`, `HGBC`,
`PCT_6TH_ORDER_OR_MORE`, `glc_pc_u01`, `glc_pc_u18`, `pnv_pc_u02`,
`wet_pc_u02`, `wet_pc_u03`, `wet_pc_u07`, `wet_pc_u09` — see the 2026-07-23
"Compact NeuralHydrology integration smoke" Finding 1, above) remains
**compact-smoke-specific historical evidence only**. It is not copied,
imported, asserted, or otherwise treated as an expected result anywhere in
this implementation or its tests; one test explicitly proves a column
sharing a name from that list (`CANALS_MAINSTEM_PCT`) is retained when it
varies in a synthetic development-training population, i.e. that the result
is computed from data, not hard-coded.

**Not done by this patch.** The real zero-variance list has **not** been
computed over the actual 2,307-basin development-training population (no
h2o access in this patch); no package was built; no NeuralHydrology config
was generated; no model was trained or evaluated; the certified Compact
Scientific Package and its `package_netcdf.py`/auditor code were not
touched; near-zero-variance (as opposed to exact) filtering was not
implemented; nothing was committed.

**Tests.** 18 focused tests added to `tests/test_static_preparation.py`
covering: dev-train-only fitting with no holdout leakage in either
direction, row-order independence, candidate-column order preservation,
constant/varying classification, apply-without-recompute, row-order/basin-
identity preservation on apply, missing/duplicate basin and column
rejection, non-finite-value rejection, all-columns-excluded rejection, and
manifest field/determinism checks. Full suite: 52/52 passing
(`pytest -q tests/test_static_preparation.py`).

## 2026-07-23 Versioned package schema (`date`) for future scientific packages — implementation

**Decision.** Added an explicit, versioned NetCDF package-schema contract so
that future full scientific packages can use the temporal coordinate name
`date` (required by NeuralHydrology 1.13's own loading path, see Finding 2
below) **without** ever touching the certified compact v001 package, which
remains frozen with `time` on disk. Two registered schemas now exist in
`src/baseline/package_netcdf.py`: `stage1_compact_scientific_package_v001`
(version 1, coordinate `time`, default for backward compatibility at the
low-level serializer, unchanged behavior) and
`stage1_scientific_package_v002` (version 2, coordinate `date`, for future
production packages). Basin-population size is explicitly not part of
schema identity. The package-builder CLI
(`scripts/build_stage1_baseline_nh_package.py`) now requires an explicit
`--package-schema` argument restricted to the two registered names — there
is no default and no inference from basin count, path, or output name, so a
future production build cannot silently emit a legacy `time` package by
omission.

**Provenance correction.** The pre-existing `package_schema_name` field (in
both the package manifest and `run_provenance.json`) documented the
*builder-manifest* identity, not the NetCDF package schema — a misleading
name given the new distinction. It is **preserved unchanged, marked
deprecated**, not repurposed or deleted. Five new explicit fields were added
to both the manifest and run provenance: `builder_manifest_schema_name`,
`builder_manifest_schema_version`, `netcdf_package_schema_name`,
`netcdf_package_schema_version`, `netcdf_time_coordinate`.

**Auditor independence preserved.** `src/baseline/package_audit.py` was
extended to check the declared/actual NetCDF package schema and temporal
coordinate without importing `src.baseline.package_netcdf`'s schema
registry — it independently redeclares its own accepted schema identities
and expected coordinate names, then verifies them from disk (new checks:
`netcdf_temporal_coordinate_present`,
`netcdf_temporal_coordinate_matches_declared_schema`,
`netcdf_package_schema_identity_recognized`,
`package_all_basins_same_netcdf_schema`,
`netcdf_matches_run_provenance_schema`). See
`docs/stage1_compact_package_independent_audit.md`'s 2026-07-23 addendum.

**Genuine bug found and fixed during test-writing (not part of the original
design).** `run_audit()`'s aggregate same-schema check sorted a `set` of
schema-identity tuples with a bare `sorted(...)`; when a basin's temporal
coordinate cannot be determined (both `time` and `date` present, or
neither), `audit_basin_netcdf` returns a tuple with `None` in the
coordinate-name position, and Python cannot order `None` against `str` —
this crashed with `TypeError` instead of reporting the (already-flagged)
structural error cleanly. Fixed with a string-coercing sort key. This would
have crashed the auditor on any real malformed "both/neither temporal
dimension" package; found only because the new both/neither tests were
added.

**`FlashNHDataset` compatibility (renames the Finding-2 adapter below).**
`src/baseline/nh_dataset.py`'s adapter — previously named
`_adapt_time_to_date` (see Finding 2 in the entry immediately below) — is
renamed to `_adapt_temporal_index_to_date` and now implements the full
three-way contract: a `date`-named index passes through unchanged; a
`time`-named index is renamed in memory only (on-disk files/timestamp
values never touched); both-present or neither-present fails loudly
(`FlashNHDatasetError`), in both directions (a `date` index with a stray
`time` column now also fails loudly — previously only the reverse ambiguity
was guarded). A package becoming structurally `date`-compatible with stock
NeuralHydrology does **not** mean `GenericDataset` reproduces Flash-NH's own
sample-validity filtering — `FlashNHDataset` remains required regardless of
on-disk coordinate name.

**Scope.** This is a schema-support implementation addendum — code, tests,
and documentation together, not a docs-only change. No real package was
built; h2o and Moriah were not accessed; the certified compact v001 package
artifacts were not touched. Tests: `tests/test_package_netcdf.py`,
`tests/test_package_builder.py`, `tests/test_package_audit.py`,
`tests/test_nh_dataset.py`, `tests/test_nh_structural_preflight.py` — exact
pass/fail counts recorded in the implementation report, not duplicated
here. The work is local/repository-only.

**Files changed.** `src/baseline/package_netcdf.py`,
`src/baseline/package_builder.py`,
`scripts/build_stage1_baseline_nh_package.py`,
`src/baseline/package_audit.py`, `src/baseline/nh_dataset.py`,
`tests/test_package_netcdf.py`, `tests/test_package_builder.py`,
`tests/test_package_audit.py`, `tests/test_nh_dataset.py`,
`docs/FLASHNH_CURRENT_STATE.md`, `docs/decision_log.md` (this entry),
`docs/stage1_compact_package_independent_audit.md`,
`docs/stage1_baseline_package_implementation_plan.md`,
`docs/stage1_neuralhydrology_preflight.md`,
`docs/stage1_scientific_baseline_design.md`.

---

## 2026-07-23 Compact NeuralHydrology integration smoke — CLOSED

**Decision.** The compact-package NH integration-validation effort opened by
the 2026-07-22 increment (below) is closed. Three Moriah Slurm jobs ran in
sequence against the certified 32-basin Compact Scientific Package and all
passed: CPU structural preflight (job `45624926`, `glacier` partition, 39 OK
/ 0 warnings / 0 errors; real `FlashNHDataset` construction for train,
validation, and test; finite training scaler reused unchanged by validation
and test; all admitted samples finite; admitted counts train 851,339 /
validation 274,347 / test 263,637), GPU training smoke (job `45625002`,
`catfish` partition, NVIDIA L4; target `qobs_mm_per_h_lead06`, sequence
length 24, 32 basins, 2 epochs, 460 static inputs; epoch 1 loss 0.40205,
epoch 2 loss 0.38727; run directory
`/sci/labs/efratmorin/omripo/Flash-NH/runs/stage1_nh_config_lead06_seq24_v001/runs/stage1_compact_lead06_seq24_v001_2307_135829`),
and explicit validation (2024) + test (2025) evaluation of the epoch-2
checkpoint (job `45625077`; evaluation audit 217 OK / 0 warnings / 0 errors;
metrics NSE, RMSE, KGE, Pearson-r, Beta-KGE; `validation_metrics.csv`,
`validation_results.p`, `test_metrics.csv`, `test_results.p` retained).
Metric values are not interpreted scientifically here — this was an
integration smoke, not a tuned or reportable baseline experiment. This gate
is passed and is **not** to be extended into ad hoc hyperparameter tuning;
the next work is planning the first scientifically meaningful Stage 1
baseline experiments.

**Finding 1 — compact-smoke-only zero-variance static exclusion.** Across
the 32-basin smoke population only, 13 of the 473 static `model_input`
attributes had zero standard deviation and could not be normalized by
NeuralHydrology: `CANALS_MAINSTEM_PCT`, `CDL_DURUM_WHEAT`, `CDL_ORANGES`,
`CDL_RICE`, `HGBC`, `PCT_6TH_ORDER_OR_MORE`, `glc_pc_u01`, `glc_pc_u18`,
`pnv_pc_u02`, `wet_pc_u02`, `wet_pc_u03`, `wet_pc_u07`, `wet_pc_u09`. The
GPU training smoke therefore used 460 of 473 static inputs. This is not a
package defect: the full 473-column Compact Scientific Package remains
authoritative and was not modified. **Binding rule:** this 13-column
exclusion list is compact-smoke-only and must **not** be carried forward
automatically into the full-population baseline; the full-population
baseline must independently identify zero-variance columns over its own
actual training population.

**Finding 2 — `time` vs. `date` temporal-coordinate adapter.** The
certified compact v001 NetCDFs use dimension/coordinate name `time`.
NeuralHydrology 1.13 internally requires the temporal index name `date` in
parts of its `GenericDataset` loading path. The smoke used an in-memory
`FlashNHDataset._load_basin_data` adapter (`_adapt_time_to_date` in
`src/baseline/nh_dataset.py` — **renamed `_adapt_temporal_index_to_date` by
the 2026-07-23 "Versioned package schema" entry above**, same behavior for
this `time` case, plus new explicit `date`/v002 pass-through and
both/neither fail-loud handling) that renames only the DataFrame index's
`.name` metadata from `time` to `date` after calling
`GenericDataset._load_basin_data` unchanged; timestamp values, row order,
dtypes, NaNs, and all on-disk files are untouched. **Binding rule:** v001
must not be modified in place; this adapter is an immediate compatibility
boundary, not a resolved format decision. Before the final production
package format is frozen, the on-disk temporal-coordinate convention must
be explicitly resolved, and any resulting change must be generated and
audited as a new package version rather than silently rewriting v001.

**Unaffected / not reopened.** The certified Compact Scientific Package
(Gate 4) was not modified, rebuilt, or regenerated. No scientific decision
from prior gates was reopened.

**Not done.** Final hyperparameters, final sequence length, the
full-population static-feature set, the production-package coordinate
convention, lead 1/3/12 h performance, and any spatial-holdout or
full-population scientific conclusion — none of these are established by
this closure.

---

## 2026-07-22 NH config-generation + structural-preflight local implementation increment

> **Superseded (2026-07-23):** "no NH training run" below describes
> accurately what this specific increment did. Real dataset construction,
> GPU training, and explicit validation/test evaluation have since run and
> passed — see the 2026-07-23 "Compact NeuralHydrology integration smoke —
> CLOSED" entry above.

**Scope.** Following Gate 4 certification (below), this is the first local
implementation increment for compact-package NH integration-validation.
Strictly local: no h2o/Moriah access, no data transfer, no NH training run,
no Slurm job, no W&B, no full 16-config matrix, no modification to the
certified Compact Scientific Package. Renders and validates exactly one
configuration: lead 6 h, sequence length 24 h, single target
`qobs_mm_per_h_lead06`; 8 approved dynamic inputs in the binding order
`mrms_qpe_1h_mm, rtma_2t_K, rtma_2d_K, rtma_2sh_kgkg, rtma_10u_ms,
rtma_10v_ms, mrms_qpe_1h_mm_gap, rtma_gap`; 473 static `model_input`
attributes in canonical order; the same 32 certified compact basins used
identically across train (2020-10-14→2023-12-31), validation (2024), and
test (2025) periods.

**New/changed code.** `src/baseline/nh_config_generation.py` (config
rendering; two-sided static-attribute contract; basin-list/date/dynamic-
input/target contracts; basin-list leakage safeguards across the three
periods) and `src/baseline/nh_structural_preflight.py` (two-layer
preflight: Layer 1 is file-only structural checks against a generated
config bundle and the package it was rendered against — ~18 named checks
including forbidden-key rejection, `nan_handling_method` absence, exact
seq/date/target/dynamic-input/static-count matches, basin-membership
identity across periods, output-location safety; Layer 2 is real
`FlashNHDataset` construction, train/validation/test, against synthetic
fixtures only — never the real package, which lives only on h2o and is
never transferred locally), plus thin CLIs
`scripts/generate_stage1_nh_config.py` and
`scripts/check_stage1_nh_config_preflight.py`. A read-only, synthetic-
fixture-tested dynamic-NaN inventory helper
(`inspect_dynamic_nan_inventory`) was added to `nh_structural_preflight.py`
for future real-package auditing; it has not been run against the real
package. A stale docstring in `src/baseline/nh_dataset.py` was corrected
during this increment.

**Test coverage.** 38 new tests passing: 25 in
`tests/test_nh_config_generation.py` (config-generation contracts) and 13
in `tests/test_nh_structural_preflight.py` (7 Layer-1 structural tests plus
the 6 required Layer-2 real-`FlashNHDataset` tests). The pre-existing
`tests/test_nh_dataset.py` suite continues to pass unaffected (with a
defensive fix applied there too, see below). One pre-existing, unrelated,
non-deterministic Windows `os.rename`-based flakiness in
`tests/test_package_builder.py` was observed and is out of scope — it
predates this increment and was not introduced by it.

**Notable discovery: NeuralHydrology 1.13 upstream mutable-default-argument
scaler bug (dev-tooling finding, not a scientific decision).**
`neuralhydrology.datasetzoo.basedataset.BaseDataset.__init__` declares
`scaler: Dict[...] = {}` as a mutable default argument. Python creates this
dict once, at function-definition time, and shares it across every call in
a process that omits `scaler=`. Consequence: if a single process
constructs more than one train-period dataset via
`get_dataset(cfg=cfg, is_train=True, period="train")` without passing
`scaler=` explicitly (e.g. two tests in one pytest session, or interactive
reuse), the second construction's `not scaler` check evaluates False
(because the shared dict was already populated by the first construction),
so `_setup_normalization` is skipped and the stale, unrelated scaler from
the first construction is silently reused. Because the subsequent
normalization step (`xr - center`) is an intersecting xarray operation, any
dynamic-input or target column present in the new dataset but absent from
the stale scaler is silently dropped rather than raising — this was first
observed as a spurious `"[...] not in index"` `KeyError` that only occurred
when `test_nh_dataset.py` ran before `test_nh_structural_preflight.py` in
the same pytest invocation. A real training job (one Slurm process
constructing exactly one train-period dataset) is unaffected. **Binding
local dev-tooling practice going forward:** every train-period
`get_dataset(..., is_train=True, ...)` call in this repository's code and
tests must pass `scaler={}` explicitly, never rely on the default. Applied
in `nh_structural_preflight.py::check_flashnh_dataset_construction` and
defensively at all 4 train-period call sites in `tests/test_nh_dataset.py`.
This does not affect, and is unrelated to, any of the 11 accepted mechanics
findings from the prior NH scaler/lookup-table evidence gate — it is a
distinct, purely process-lifetime artifact of NH's own `__init__` signature.

**Known documentation debt (not yet resolved).** The committed policy
config (`config/stage1_scientific_baseline_v001.yaml`) declares
`nh.dataset: generic`, documenting the underlying NH dataset family, while
`nh_config_generation.py::build_nh_config_mapping` hardcodes
`dataset: flashnh` into every generated config, selecting the registered
`FlashNHDataset` class actually used at construction time. Both values are
individually correct for their own purpose, but the discrepancy is not
currently annotated in either file. Reconciling or explicitly documenting
this is deferred to a future increment.

**Unaffected / not reopened.** No scientific decision was reopened; the 11
accepted mechanics findings from the prior evidence gate stand unchanged.
The certified Compact Scientific Package was not modified, rebuilt, or
transferred. No h2o or Moriah connection was made. No training was run. No
Slurm file was written. Only the single lead06/seq24 configuration was
rendered — the full 16-config matrix was not generated.

**Not done.** Real-package NaN inventory execution (the helper exists and
is synthetic-fixture-tested only), Moriah transfer, Smoke-run execution
against this configuration, and generation of the remaining 15 configs all
remain not done.

---

## 2026-07-22 Gate 4 — Compact Scientific Package independently certified (real h2o audit PASS)

**Result.** The Gate 4 independent auditor (`src/baseline/package_audit.py`,
commit `4b524b3851b16baa080d4237622fa7da30e05cea`) was executed for real on
h2o, in full mode, against the authoritative Compact Scientific Package at
`/data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001` (build
commit `89c4dd162f7043419b4b227de5c2bc1b3b230da6`). Execution timestamp
`2026-07-22T08:58:52Z`. **Status: PASS — 0 errors, 0 warnings, 3,114 OK
checks, full-audit exit code 0.** Audit output written to
`/data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_gate4_audit`.
The build commit and the auditor commit are intentionally distinct
identities — the auditor never imports or shares code with the builder it
checks.

**What the real run independently verified.** Exact package layout and
authoritative file membership; exact 32-basin membership and order; all 32
NetCDF timelines, schemas, dimensions, dtypes, units, and metadata; all 8
dynamic variables against their original forcing parquets; raw `qobs_m3s`
against the original qobs NetCDFs; independent m³/s→mm/h conversion;
independent 1/3/6/12-hour lead-target reconstruction; exact NaN alignment
and target-tail NaNs; static attributes against the prepared 32×473 matrix;
all 168 imputed values against the imputation mask and manifest; exact
reconstruction of 138 gap timestamps; binary and finite gap flags; exact
32-file QC evidence membership and QC-to-NetCDF stored-value agreement
(see the 2026-07-22 correction round below); checksums for all 38
authoritative package artifacts, all 32 forcing source files, and all 32
qobs source files; and policy/basin-selection/static/imputation/area/
gap-inventory/package-manifest/build-commit/auditor-commit identities.

**Evidence handling.** The transferred audit evidence bundle was
independently reviewed (by ChatGPT) and found internally consistent. The
generated evidence files (audit JSON/CSV/log outputs) remain untracked and
are not committed to this repository.

**Package status: BUILT AND INDEPENDENTLY CERTIFIED.** NeuralHydrology
configuration generation is now unblocked.

**Scope of this closure.** Documentation-only. No training, no Moriah
transfer, and no NeuralHydrology configuration were performed as part of
this closure; no audit code, package-building code, scientific policy, or
generated artifact was changed.

## 2026-07-22 Gate 4 auditor correction round (local pass, pre-h2o)

**Context.** A review of the 2026-07-21 Gate 4 auditor implementation
accepted its core architecture and scientific independence, but required a
focused correction round before the real h2o audit run. No redesign, no
broadened scope — only listed corrections, in `src/baseline/package_audit.py`,
`scripts/audit_stage1_compact_scientific_package.py`,
`tests/test_package_audit.py`, and the docs below.

**Corrections made.** (1) `--mode full` now hard-requires
`--imputation-manifest`/`--imputed-value-mask`/`--qc-evidence-root`; a
canonical full audit can no longer PASS with any of these skipped (a
`dev_allow_missing_evidence` bypass exists for isolated dev/test use only,
never set by the CLI). (2)-(3) Every forcing/qobs source file and every
authoritative package artifact/metadata file is now checksum-bound from disk
bytes, independently recomputed — never copied from any manifest under
audit. (4) Package layout is now enumerated exactly: any missing or
unexpected file (including extra top-level entries) fails the audit. (5)
`+inf`/`-inf` are now rejected in forcing/qobs source arrays as well as
package arrays; matched infinities can no longer silently pass a numeric
comparison. (6) NetCDF dimension names/sizes, per-variable dimension tuples,
the dataset-level schema name/version, and gap-flag/raw-target/lead-target
role and metadata attributes are now independently checked against the real
Gate 2 serialization contract (inspected read-only, never imported). (7)
Imputation-evidence checks now verify exact mask basin/column order, strict
boolean values, no missing cells, that every imputed cell has a manifest
fitted value (a missing fitted value is now distinguished from a
numerically-wrong one), and per-column/per-basin/total count agreement
where the manifest records them — no hard-coded basin or count. (8) QC
evidence membership is now exact (no extra CSVs, manifest count/membership
agreement) and every manifest entry's checksum/size/row-count/non-authoritative
declaration is cross-checked against the file on disk. (10) A full audit now
hard-fails (raises) if the auditor's own git commit cannot be resolved or its
working tree is not clean — distinct from, and never required to equal, the
package build commit.

**Test suite.** `tests/test_package_audit.py` grew from 16 to 37 tests (all
corrections have dedicated regression coverage); `pytest -q
tests/test_package_audit.py` and `pytest -q tests/test_package_builder.py`
both pass (one intermittent Windows directory-lock `PermissionError` recurred
and was confirmed, again, to pass cleanly in isolation — the same
pre-existing environmental flake noted in the prior entry, not a logic
defect).

**Discovered during this round, resolved by a follow-up focused correction
(same day): QC CSV vs. NetCDF float precision mismatch.** The QC CSV is
written from the builder's pre-quantization float64 table, while the
on-disk NetCDF stores the same values quantized to `float32`
(`package_netcdf.py`). Comparing the two under the CSV-text round-trip
tolerance (`rtol=1e-9`/`atol=1e-12`, designed for a float64-to-float64
self-check) would have shown relative differences of order float32 machine
epsilon (~1.2e-7) on nearly every value, unrelated to data correctness. This
was fixed, not worked around: `qc_csv_matches_netcdf` now casts each QC
CSV's finite values through the variable's actual on-disk storage dtype
(read independently from the NetCDF, not hard-coded) and requires **exact**
agreement with the stored value (`rtol=atol=0.0`), with NaN masks compared
exactly and gap-flag/integer variables kept on exact binary comparison as
before — see `compare_qc_csv_against_netcdf_storage` in
`src/baseline/package_audit.py` and the "Comparison tolerances" section of
`docs/stage1_compact_package_independent_audit.md`, which now documents all
three distinct comparison classes (CSV float64 round trip, authoritative
source-to-NetCDF at `package_float32_rtol`, and this exact QC-CSV-to-NetCDF
projection) separately. `tests/test_package_audit.py` grew from 37 to 41
tests to cover this (quantization-tolerant pass, one-ULP-precise failure,
NaN-mismatch failure, gap-flag exactness); the pre-existing deliberately
corrupted QC CSV test still fails as expected. This was a narrow, local-only
follow-up: no h2o/Moriah access, no real audit run, nothing committed, no
other behavior changed.

**Scope of this round: local implementation only.** No h2o or Moriah
access occurred; the real package and real source artifacts were not
touched or audited; the built package was not modified; nothing was
committed.

## 2026-07-21 Compact Scientific Package built on h2o; independent audit implementation (local pass)

**Context.** The 32-basin Compact Scientific Package was built and promoted
on h2o at `/data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001`
(build commit `89c4dd162f7043419b4b227de5c2bc1b3b230da6`; non-authoritative
QC evidence at `..._v001_evidence`; run logs at `..._v001_run_logs`).
Builder-level self-validation and an independent ChatGPT inspection of the
compact review bundle are complete. This entry records that state and the
follow-on work started to independently certify it.

**Package built is not the same as package certified.** Because a builder's
own checks cannot certify the builder's own output, a separate, genuinely
independent auditor was specified and implemented this session:
`src/baseline/package_audit.py`, `scripts/audit_stage1_compact_scientific_package.py`,
`tests/test_package_audit.py` (16 synthetic tests, all passing), and
`docs/stage1_compact_package_independent_audit.md`. The auditor re-derives,
from raw sources and general-purpose libraries only, package layout and
checksums, basin membership/order, NetCDF dimensions/dtypes/units/timeline,
all 8 dynamic inputs and raw `qobs_m3s` against source, the m³/s→mm/h
conversion and the 1/3/6/12 h lead-target shift (both re-expressed
independently, not imported from `src.baseline.units`/`lead_targets`), NaN
propagation, static-attribute membership/order/values/imputation placement,
independent reconstruction of the 138 MRMS+RTMA gap timestamps from the
missing-hour inventory, gap-flag validity, and QC-CSV-to-NetCDF agreement
(CSV treated as non-authoritative throughout).

**Scope of this session: local implementation only.** No h2o or Moriah
access occurred; the real package and real source artifacts were not
touched or audited; no NH config was generated; no scientific policy or
package-builder behavior changed; nothing was committed. The implementation
was validated only against synthetic fixtures built with the real builder
and then deliberately corrupted.

**Status: package built, builder-level-validated, and externally
eyeballed — but not yet independently certified.** NeuralHydrology
configuration generation remains blocked until the new auditor is run on
h2o against the real package and reports PASS, and the resulting evidence
bundle is transferred and reviewed.

> **Superseded (2026-07-22):** the auditor has since been run for real on
> h2o and reports PASS — see the 2026-07-22 "Compact Scientific Package
> independently certified" entry above. NeuralHydrology configuration
> generation is now unblocked.

## 2026-07-08 Milestone 2K-G-F-B closure — canonical h2o build/audit PASS

**Context.** Closes the "canonical build not yet produced" gap left open by
the 2026-07-07 entry below. The user ran the §11.5 commands
(`docs/stage1_static_attribute_matrix_plan.md`) directly on h2o — this
session still has no network path to h2o, so all facts below are as reported
by the user.

**Source mirror verification.** `/data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/`
contains 30 files (29 source files + `source_attributes_v001_checksums.sha256`);
`sha256sum -c source_attributes_v001_checksums.sha256` returned OK for all
29 files.

**Canonical build.** Run with `scripts/build_stage1_static_attribute_matrix.py`
against the verified source mirror, `config/stage1_initial_training_basin_manifest.csv`,
output dir `/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001`,
matrix name `stage1_static_attributes_v001`, default checksum-required path
(not bypassed).

**Canonical audit: PASS.** 0 errors, 0 warnings, 20 OK checks. Matrix shape
2,843 rows × 531 columns, 496 `model_input` columns. All Stage 1 basins
present, no extra basins, no duplicate `gauge_id`, no non-numeric or
ID/code-like `model_input` columns, `STATE`/`HUC02` excluded from
`model_input` and retained as `split_support`, `LAT_GAGE`/`LNG_GAGE` excluded
from `model_input` and retained as `diagnostic`. HydroATLAS coverage flag
matched the expected 5-basin gap exactly (`393109104464500`,
`394839104570300`, `401733105392404`, `402114105350101`, `402913084285400`),
and those basins' HydroATLAS `model_input` columns are NaN as designed.
Matrix checksum matched the provenance record.

**Canonical artifact.** `/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001.parquet`,
matrix sha256 `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`.
Output file sizes: parquet 8.8 MB; `_column_manifest.json` 58 KB;
`_provenance.json` 20 KB; `_audit_summary.md` 1.7 KB. These are h2o-resident
generated data artifacts, not git-tracked source files, per
`docs/repo_policy.md`.

**Not done.** No NH package was regenerated from this matrix; no training was
run; no NH config/Slurm scripts were modified; no Moriah mirror of the source
attributes or derived matrix has been performed. This was a docs-only
closure patch — no code, config, or generated-output files changed.

## 2026-07-07 Milestone 2K-G-F-B — static attribute source mirror + derived matrix builder/auditor

**Context.** Implements the 2K-G-F plan (`docs/stage1_static_attribute_matrix_plan.md`)
in code. Per the user, the 29-file source mirror was already copied to h2o at
`/data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/` with
a 29-line `sha256sum`-generated checksum file (~53 MB total). This session
has no network path to h2o (`ssh flashnh-h2o` fails to resolve, reconfirmed),
so the mirror itself could not be independently verified here.

**Scripts added.** `scripts/build_stage1_static_attribute_matrix.py` and
`scripts/audit_stage1_static_attribute_matrix.py` (neither writes into the
repo's tracked tree; generated matrix/manifest/provenance/audit outputs stay
under the h2o target path or repo `tmp/`, per `docs/repo_policy.md`).

**Column-classification policy implemented in code**, refining 2K-G-F's §5/§8
to exact column names:
- Duplicate `DRAIN_SQKM` (from `Bound_QA.csv`) dropped; admin free-text
  (`STANAME`, `COUNTYNAME_SITE`, `WR_REPORT_REMARKS`, `ADR_CITATION`,
  `SCREENING_COMMENTS`, `NAWQA_SUID`) dropped; admin numeric-ID columns
  (`FIPS_SITE`, `REACHCODE`, `BOUND_SOURCE`) dropped — **newly identified
  this session**: these pass a naive `pd.to_numeric` coercion check but are
  administrative IDs, not physical quantities.
- Sparse binary flags (`HCDN_2009`, `HBN36`, `OLD_HCDN`, `NSIP_SENTINEL`,
  `ACTIVE09`) encoded 0/1.
- Categorical fields deferred out of `v001-core` (raw values retained
  separately, not one-hot-encoded): the explicit list from 2K-G-F's plan,
  plus **newly identified this session**: GAGES-II `Regions.csv`'s
  `_DOM`/`_SITE` dominant/site class-code columns (only its `_PCT` columns
  are genuine continuous fractions), and HydroATLAS's `*_cl_smj`/`*_id_smj`
  numeric-coded class/admin-division columns (10 `_cl_smj` + `gad_id_smj`) —
  both groups pass the naive numeric check but are categorical, confirmed by
  direct inspection of the full HydroATLAS/Regions.csv schemas.
- `STATE`/`HUC02` → `split_support` role (excluded from `model_input`,
  retained in the matrix); `LAT_GAGE`/`LNG_GAGE` → `diagnostic_latlon` role
  (same treatment) — matches the 2K-G-F decision.
- Per-year series: `FlowRec.csv`'s `wy1900`…`wy2009` dropped outright (native
  summary columns `FLOWYRS_*`/`FLOW_PCT_EST_VALUES` already exist in the same
  file, confirmed this session — no new derivation needed);
  `Climate_Ppt_Annual`/`Climate_Tmp_Annual`'s per-year columns have no native
  summary and are reduced to computed mean/std across the 1950–2009 series.
- Dynamic near-constant (`nunique<=1`) and high-missingness (>20%) filters
  applied after the above, on the Stage 1 subset.
- Any unclassified non-numeric column causes the build to fail loud — no
  silent inclusion or drop of unreviewed fields if the source schema drifts.

**HydroATLAS 5-basin gap — resolved.** Directly verified this session
(exact/zero-padded/leading-zero-stripped match tests against the local
HydroATLAS CSV) that the 5 non-standard 15-char STAIDs are genuinely absent
under any representation — a true data gap. Policy: builder computes the
observed gap at build time and requires it to equal exactly this known
5-basin set; if it matches, those basins are retained with NaN
HydroATLAS-sourced columns plus an explicit `hydroatlas_coverage_flag`
column (0 = gap, 1 = present); if the observed gap ever differs, the build
fails loud. This is the concrete mechanism implementing 2K-G-F's mandatory
gate (option b primary, option c safety net) — no silent partial merge is
possible.

**Local dry-run validates the logic (not the canonical build).** Run against
`C:\PhD\Python\neuralhydrology\US_data\attributes` into repo `tmp/`
(gitignored): build exit 0, 2,843 rows × 531 columns (496 `model_input`, 15
dynamically excluded as near-constant — all HydroATLAS land-cover/PNV/wetland
class fractions uniformly zero for this basin set); HydroATLAS gap gate
matched the expected 5-basin set exactly; audit exit 0, 0 errors, 0 warnings,
20 OK checks including a checksum round-trip. One auditor threshold was
recalibrated during this dry-run: HydroATLAS's `gdp_ud_usu` (upstream-summed
GDP, USD) legitimately reaches ≈$1.74 trillion for the largest basins — the
numeric-range sanity bound was raised from 1e12 to 1e13 to accommodate this
real basin-integrated economic aggregate rather than flag it as an error.

**Correction.** The source mirror has **26** distinct
`attributes_gageii_*.csv` files, not 27 as stated in 2K-G-F and by the user
when describing the h2o mirror — the total file count of 29 (26 + HydroATLAS
+ NLDAS-2 + workbook) is unaffected and matches the user's own h2o-side
`find | wc -l` result.

**Not done (by design).** The canonical h2o build/audit was not executed (no
network path from this session) — user-run commands are documented in
`docs/stage1_static_attribute_matrix_plan.md` §11.5. No NH package was
regenerated; no training was run; no NH config or Slurm script was modified;
no generated matrix/manifest/provenance/audit file was committed to git.

## 2026-07-06 Milestone 2K-G-F — static attribute matrix inventory + audit plan

**Context.** 2K-G-E (revised, 2026-07-06, above) reopened static attributes
and gated them on this milestone. The existing canonical artifact
(`gagesii_v001/all_basins_merged.parquet`, 48 columns, checksum
`06a9eeda9...`) is a valid, checksum-pinned provenance artifact but draws
from only 3 of 27 available GAGES-II source tables and has no topography,
geology, land cover/vegetation, or snow fraction — insufficient as the final
Stage 1 modeling matrix. This pass inventories the richer local source and
proposes (does not build) a merge/audit policy.

**Inventory.** Local source directory
`C:\PhD\Python\neuralhydrology\US_data\attributes`: 29 CSVs (27 GAGES-II +
HydroATLAS + NLDAS-2 climate) + 1 variable-description workbook, all keyed on
`STAID`, all 9,008 rows. Cross-checked against the real Stage 1 basin
manifest (`config/stage1_initial_training_basin_manifest.csv`, 2,843 basins:
2,216 `TRAIN_CORE` + 627 `TRAIN_SOFT_KEEP`):
- 100% GAGES-II coverage after zero-padding `STAID` to 8 chars, including all
  6 non-standard-length USGS IDs (five 15-char, one 9-char).
- HydroATLAS covers 99.8% (2,838/2,843) after zero-padding; the 5-basin gap
  is exactly the 15-char non-standard IDs (HydroATLAS's raw `STAID` export is
  not zero-padded, unlike the GAGES-II CSVs). **Clarified as a mandatory
  build/audit gate, not a loose caveat** (same-day follow-up review): the
  builder/auditor must explicitly detect these 5 basins and either
  resolve/match them, retain them under a documented missing/imputation
  policy, or fail the build with a named-basin message — a silent partial
  HydroATLAS merge is not allowed.
- NLDAS-2 climate covers 100% after zero-padding.
- **Confirmed the existing 48-column canonical parquet stores `STAID` as
  `int64`** (leading zeros stripped) — already handled by the builder's
  `_norm_staid()`, but any new merge/audit script must reimplement 8-char
  zero-padding itself; do not assume any file preserves it.

**Content audit (780 non-ID columns across all sources, restricted to the
2,843 Stage 1 basins):** 758 numeric-like / 22 non-numeric. Non-numeric split
into free-text/administrative (drop: `STANAME`, `COUNTYNAME_SITE`,
`WR_REPORT_REMARKS`, `ADR_CITATION`, `SCREENING_COMMENTS`, `NAWQA_SUID`),
sparse binary membership flags (encode 0/1, not drop: `HCDN_2009`, `HBN36`,
`OLD_HCDN`, `NSIP_SENTINEL`, `ACTIVE09`), and genuine categorical class codes
(`CLASS`, `AGGECOREGION`, `HUC02`, `STATE`, `HUC10_CHECK`,
`GEOL_REEDBUSH_DOM/SITE`, `GEOL_HUNT_DOM_CODE/DESC`, `GEOL_HUNT_SITE_CODE`,
`USDA_LRR_SITE`). Only 6 of 780 columns exceed 20% missing (all sparse
membership flags); 20 near-constant columns; one duplicate column
(`DRAIN_SQKM`, appears in both `BasinID.csv` and `Bound_QA.csv`). Snow
fraction is available **only** via HydroATLAS (`snw_pc_*`), not any GAGES-II
file. Per-year time-series columns (`Climate_Ppt_Annual`/`Climate_Tmp_Annual`,
~120 cols; `FlowRec`'s `wy1900`…`wy2009`, 110 cols) are flagged as needing
reduction to summary statistics, not inclusion as raw per-year columns.
**Decided (same-day follow-up review):** `STATE`/`HUC02` are useful for split
construction, diagnostics, and reporting (CA exclusion, spatial holdout) but
are excluded outright from `v001-core` model-input features — not merely
de-prioritized pending a decision. Lat/lon are held out of `v001-core` by
default and deferred to a dedicated ablation testing whether raw coordinates
help or hurt spatial generalization.

**Filtering philosophy for `v001-core` (same-day follow-up review):**
conservative by default — any variable suspected to be problematic,
non-physical, purely administrative, weakly useful, leakage-prone,
near-constant, high-missingness, or hard to interpret is excluded from the
first modeling matrix rather than kept on the chance it helps. A smaller,
defensible first matrix is preferred over a maximal one; richer/borderline
variables can be added later only as a deliberate, documented ablation. This
applies to the per-year time-series columns and near-constant columns noted
above, and to the HydroATLAS gap-gate policy above.

**h2o/Moriah mirror status.** Not checked from this session — no network
path from the Claude Code environment to h2o/Moriah (confirmed:
`ssh flashnh-h2o` fails to resolve). Explicit user-side check/transfer/verify
commands written to `docs/stage1_static_attribute_matrix_plan.md` §6 instead
of assumed.

**Proposed canonical paths** (not yet created):
`/data42/omrip/Flash-NH/data/static_attributes/source_attributes_v001/` (h2o
source mirror), `/sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/source_attributes_v001/`
(Moriah source mirror), `/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001.parquet`
(derived modeling matrix, once built).

**Explicitly not done this session:** no final static-attribute matrix built
or written anywhere; no h2o/Moriah transfer performed; no code, config,
package, Slurm script, or training changed. The per-column audit CSV produced
during inspection is a local scratch artifact (session scratchpad), not
committed or canonical.

**Files changed:** new `docs/stage1_static_attribute_matrix_plan.md`
(inventory, content audit, mirror-status/commands, proposed paths, merge/audit
policy, audit plan for the eventual matrix); `docs/FLASHNH_CURRENT_STATE.md`;
`docs/decision_log.md` (this entry).

## 2026-07-06 Milestone 2K-G-E (revised) — scientific baseline aligned to user-approved decisions; 2K-G-F/2K-G-G gates defined

**Context.** The first 2K-G-E proposal (2026-07-03, originally recorded in this
slot) was **never committed** — user review changed several key decisions before
commit, so this entry and `docs/stage1_scientific_baseline_design.md` were revised
in place rather than layering a second entry on an uncommitted draft. This entry
replaces the 2026-07-03 text previously here.

**What changed from the 2026-07-03 draft, and why:**
1. **Static attributes reopened, not signed off.** The draft's ~16-column
   candidate list is withdrawn. The 48-column GAGES-II screening merge stays a
   valid, checksum-verified provenance artifact, but the user expects a richer
   matrix (topography, geology, land use/land cover, vegetation, snow fraction,
   climate/static hydrologic attributes) from local source material at
   `C:\PhD\Python\neuralhydrology\US_data\attributes` (~28 files, including
   `attributes_gageii_Topo.csv`, `attributes_gageii_Bas_Morph.csv`,
   `attributes_hydroATLAS.csv`, `attributes_nldas2_climate.csv`, and a
   ~350-variable description workbook `Var description_gageii.xlsx`) not yet
   confirmed mirrored to h2o/Moriah. Gated on new **Milestone 2K-G-F**.
2. **Target normalization: log-transform rejected.** User explicitly rejected it
   as poorly aligned with the project's flash-flood/high-flow emphasis. Leading
   candidate is now area-normalized/specific discharge, pending feasibility.
   Evaluation must always be reported in raw `m^3/s`. Gated on new
   **Milestone 2K-G-G**, which must inspect NH 1.13's actual installed
   normalization code on Moriah, not public docs.
3. **`seq_length` narrowed and made binding for Stage 1: only 12/24/48/72 h.**
   The draft's 336 h (Kratzert et al. 2021-grounded) proposal is withdrawn —
   168/336 h are explicitly Stage 2 (long-term antecedent modeling) territory,
   not Stage 1. This is recorded as a binding decision specifically so future
   prompts/docs stop reintroducing 168/336 h for Stage 1.
4. **Lead time added as a new, separate design axis** (the draft did not address
   it): primary benchmark lead time 6 h, secondary 12 h, 1/3 h diagnostic-only.
   Input sequence length and prediction lead time are explicitly independent
   axes.
5. **Temporal split dates revised:** train `2020-10-14`→`2023-12-31`, validation
   `2024-01-01`→`2024-12-31`, test `2025-01-01`→`2025-12-31` (was train
   ≤2022-12-31 / val 2023 / test 2024–2025) — closer to a 60/20/20 chronological
   design given the available data range. Not yet encoded in
   `scripts/build_stage1_nh_package.py`'s split constants (code change, deferred).
6. **Spatial/geographic split added:** California excluded entirely from Stages
   1–3; ~10% non-CA CONUS spatial holdout, broadly distributed, strictly
   test-only (never in training/validation/tuning/normalization/early
   stopping/model selection); official spatial-holdout evaluation uses the 2025
   test period for comparability with the temporal test set.
7. **California transfer-learning split (Stage 4) added:** CA held out through
   Stages 1–3; Stage 4 fine-tunes on CA with an internal ~90/10 split
   (fine-tune-train/CA holdout); CA-specific normalization may be refit using
   only the CA fine-tuning training subset (not the CA holdout); transfer
   benefit is quantified by comparing the non-CA-trained model and the
   fine-tuned model, both evaluated on the CA holdout.
8. **Leakage-prevention rules made explicit:** all Stage 1–3 scalers (static,
   dynamic, target, any area-based target-scaling statistic) are fit only on
   development-training basins/period — never validation, temporal test,
   spatial holdout, or CA data; Stage 4 CA normalization updates use only the CA
   fine-tuning training subset.
9. **Loss and metrics separated**, where the draft treated them as one resolved
   item. Training loss stays open (depends on target-scaling outcome, §5/2K-G-G).
   Evaluation metrics are always computed in raw `m^3/s`; raw-space NSE is
   primary; KGE+components, PBIAS, peak magnitude error, and peak timing error
   are added; detailed event/high-flow metric design is deferred to its own
   near-term discussion.
10. **Hyperparameter table reframed** from "recommend now, adopt as-is" to an
    *initial seed config only* — the official Stage 1 benchmark requires a
    controlled W&B hyperparameter sweep (candidate dimensions: `seq_length`,
    hidden size, dropout, learning rate, batch size, possibly layer count),
    not yet run. Sweep/model-selection objective for now: validation raw-space
    NSE, with high-flow/event metrics logged as secondary diagnostics.
11. **W&B logging policy expanded** beyond config/provenance to include loss and
    validation curves, learning rate, epoch timing, run duration, GPU
    type/partition/GRES, and system/resource telemetry where available.
12. **Slurm policy reaffirmed as flexible/parameterized** (not a new decision,
    but restated as binding): no permanent hard-pin to one partition/GPU;
    resources actually used are recorded in the evidence bundle; allocation may
    increase later based on telemetry.
13. **Basin-set decision (2,752-basin floor, exclude `02299472`/`04073468`)
    reconfirmed unchanged** — this item from the original draft was approved
    as-is.
14. **Two new mini-milestones defined** (not executed in this patch):
    - **2K-G-F — Static Attribute Matrix Recovery + Audit**: inventory the
      local attribute source directory; check/document h2o/Moriah mirror
      status; use `Var description_gageii.xlsx` to interpret fields; recover
      richer CAMELSH/CARAVAN/HydroATLAS/static attributes; merge with useful
      existing GAGES-II fields; drop/encode non-numeric fields; audit
      missingness/ranges/units; checksum; propose a Stage 1 attribute policy
      for sign-off.
    - **2K-G-G — Target Scaling + Gap Policy + Lead-Time Feasibility Report**:
      inspect NH 1.13's actual installed code on Moriah (not public docs or
      assumptions) for target-normalization support, `nan_handling_method`
      behavior, masked-loss support, and window/sample-exclusion feasibility;
      quantify expected sample/window loss across `seq_length`∈{12,24,48,72}
      × lead time∈{1,3,6,12h}; record an explicit RTMA-interpolation decision.

**Explicitly not done this session (by design):** no code changed; no config
written; full 2,752-basin NH package not generated; no training run; no
Moriah or California data transfer; 2K-G-F and 2K-G-G not executed — only
scoped.

**Files changed:** `docs/stage1_scientific_baseline_design.md` (revised in
place — binding-decisions section, revised §1–§12, new §8b/§8c/§8d, new "New
mini-milestones" section, updated checklist and Status),
`docs/FLASHNH_CURRENT_STATE.md`, `docs/decision_log.md` (this entry, replacing
the 2026-07-03 text in the same slot).

## 2026-07-03 Milestone 2K-G-D-A — static attribute artifact promoted out of `tmp`; h2o checksum verified

**Context.** Milestone 2K-G-D (same day, entry below) correctly identified the static
attribute file as an external, checksum-pinned generated artifact — not to be committed
to git — but left the canonical h2o copy resident under
`/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` (a scratch-space path per
`docs/repo_policy.md`, not intended for long-lived canonical inputs) and left the h2o-copy
checksum **unverified** (no h2o/Moriah shell access from that session).

**Decision 1 — promote the canonical h2o copy to a stable project data path.**
`/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` →
`/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`.
The `tmp/` path is now **historical/staged only** — retained on h2o as the
pre-promotion reference copy, not to be referenced by new work. The parquet itself
is **still not committed to git** (no change to that part of the 2K-G-D decision).

**Decision 2 — h2o checksum verification closed (user-run on h2o, reported this session).**
```
mkdir -p /data42/omrip/Flash-NH/data/static_attributes/gagesii_v001
cp /data42/omrip/Flash-NH/tmp/all_basins_merged.parquet \
   /data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet
sha256sum /data42/omrip/Flash-NH/tmp/all_basins_merged.parquet
sha256sum /data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet
```
Both paths returned `06a9eeda9e94261d0b1bb9f2c2f42cb6bf11b4c02745d7ed5867ef0e0c0ad0b1`
(`ls -lh`: `2.9M` both), matching the local repo-fixture checksum recorded at 2K-G-D.
This closes the "Evidence that must be pulled" item left open by 2K-G-D — the
tmp-vs-repo-fixture identity is now independently confirmed, not just assumed. No
further attribute-checksum verification is required before full 2,752-basin package
generation.

**Decision 3 — Moriah mirror path documented, not yet populated.**
`/sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`
is recorded as the intended Moriah-side mirror for when a Moriah build needs the
attribute file directly. Not yet copied or verified — non-blocking today because
Moriah packages so far have been transferred pre-built from h2o.

**Docs/code updated to the stable path (all four locations named `tmp/` before this
entry):** `docs/stage1_attribute_provenance.md` (canonical-path table, checksum
section, verification evidence, resolved-status note),
`reports/flashnh_basin_screening_v001/README.md` (tracked pointer),
`scripts/build_stage1_nh_package.py` (module docstring + `--attributes-csv` help
text — both now say "do NOT use the old tmp/ path"; `attributes_sha256` provenance
recording added at 2K-G-D is unchanged), `docs/FLASHNH_CURRENT_STATE.md` (current
milestone block + historical Smoke-0/1-era caveat annotated resolved, not rewritten).
Remaining historical mentions of the `tmp/` path elsewhere in
`FLASHNH_CURRENT_STATE.md` (Milestone 2K-G-B narrative) are left as period-accurate
history, with a resolved-status note added rather than being rewritten.

**Explicitly not done this session (by design):** parquet not committed to git; full
2,752-basin NH package not generated; no training run.

**Files changed:** `docs/stage1_attribute_provenance.md`,
`reports/flashnh_basin_screening_v001/README.md`,
`scripts/build_stage1_nh_package.py`, `docs/FLASHNH_CURRENT_STATE.md`,
`docs/decision_log.md` (this entry).

## 2026-07-03 Milestone 2K-G-D — attribute provenance closed + scientific baseline design gate opened

**Decision 1 — attribute source stays external/h2o-Moriah-resident, checksum-pinned;
not committed to git.**

Investigated both paths that had appeared inconsistently across docs since Milestone
2K-G-B/C: `reports/flashnh_basin_screening_v001/all_basins_merged.parquet` (repo-relative)
and `/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` (h2o-staged). Findings:
- Local file exists (2.90 MiB / 3,037,889 bytes), gitignored (`reports/**`), never
  independently tracked in the current tree (was briefly tracked at commit `905f871`,
  then untracked by `f51b34a` "Tighten generated artifact tracking policy").
- 9,008 rows × 48 columns — the full GAGES-II reference-basin universe, not scoped to the
  2,752/2,843 Flash-NH candidates (builder subsets by `STAID` at load time).
- Required columns confirmed present: `DRAIN_SQKM`, `LAT_GAGE`, `LNG_GAGE`, `BFI_AVE`.
  `STAID` present (int64); `gauge_id` absent — expected, builder normalizes via
  `_norm_staid()` (zero-pad to 8 chars; round-trip confirmed lossless).
- sha256 (local copy): `06a9eeda9e94261d0b1bb9f2c2f42cb6bf11b4c02745d7ed5867ef0e0c0ad0b1`.
- Provenance: **generated**, not source, not manually curated — a deterministic,
  unfiltered merge of local GAGES-II CSVs (`US_data/attributes/attributes_gageii_*.csv`,
  also untracked) via `scripts/flashnh_basin_screening.py`. That script reads from a
  hardcoded absolute Windows path, so the merge is **not currently reproducible on
  h2o/Moriah** — the h2o copy depends on the 2026-06-30 manual staging.

Per `docs/repo_policy.md` → "Generated artifact policy" ("Git does not track generated
data products... regardless of size"), committing the parquet — even at ~3 MB — would
contradict the policy the repo already established by deliberately untracking this exact
class of file. **Resolution: option (b) from the 2026-06-30 entry below** — document as a
canonical, checksum-pinned external artifact. Full record: `docs/stage1_attribute_provenance.md`.

**Caveat — h2o-copy checksum not yet independently verified.** This session had no
h2o/Moriah shell access. The verification command
(`ssh flashnh-h2o "sha256sum /data42/omrip/Flash-NH/tmp/all_basins_merged.parquet"`)
is documented in `docs/stage1_attribute_provenance.md` and must be run, with the result
compared against the checksum above, before full 2,752-basin package generation.
**Superseded 2K-G-D-A (2026-07-03, entry above):** verification completed (PASS) and
the canonical copy promoted off `tmp/` to a stable path.

**Decision 2 — small tracked provenance pointer added under the existing README exception.**
`reports/flashnh_basin_screening_v001/README.md` added (small, curated, points to the
full provenance doc) under the pre-existing `!reports/**/README.md` gitignore exception.
While implementing this, found that exception was **non-functional**: `reports/**`
excludes the parent directories themselves, so git never evaluated the nested negation
patterns (`!reports/**/README.md`, `!reports/**/summary.md`, `!reports/**/manifest.json`)
against files inside them — a standard gitignore gotcha ("cannot re-include a file if a
parent directory is excluded"). Fixed by adding `!reports/**/` before the file-level
negations in `.gitignore` (Disk_volume_estimation). Verified with `git check-ignore -v`:
the three intended file classes (`README.md`, `summary.md`, `manifest.json`) are now
correctly un-ignored at any depth under `reports/`, while `*.parquet`, `*.csv`, `*.png`,
etc. remain ignored via their own separate patterns. This also repairs the `manifest.json`/
`summary.md` exceptions for *other* report directories (unrelated to this milestone) —
those files were not staged or committed here; only the new README was added.

**Decision 3 — builder now records the attribute file's own checksum in provenance.**
`scripts/build_stage1_nh_package.py` `_write_provenance()` now writes `attributes_sha256`
(computed from the actual `--attributes-csv` file used) into `run_provenance.json`, so
every future package build is self-verifying regardless of which of the two documented
paths supplied the file. Docstring and `--attributes-csv` help text updated to point at
`docs/stage1_attribute_provenance.md` instead of repeating an ambiguous default path.
Syntax-checked (`python -m py_compile`); not re-run end-to-end (requires h2o data, out of
scope for this session).

**Decision 4 — scientific-baseline design gate opened; `seq_length` framing corrected.**
New scaffold doc `docs/stage1_scientific_baseline_design.md` lists what must be decided
before the first scientific-baseline training run: dynamic input set (§1–2), static
attributes (§3), target cleaning/normalization (§4–5), forcing-gap policy for training vs.
Smoke 0/1 (§6, reusing the two candidates already identified in
`docs/stage1_neuralhydrology_preflight.md` §8.2), loss/metrics (§7), train/val/test
protocol (§8), `seq_length` + conventional hyperparameters (§9), W&B policy (§10), Slurm
partition/GRES parameterization (§11), and evidence bundle conventions (§12). Most items
are explicitly **OPEN** — this is a decision scaffold, not a final spec.

Prior entries (2026-07-02, below) framed "lookback-expansion tests (seq_length
72/168/336)" as the direct next milestone after attribute cleanup. **That framing is
superseded**: `seq_length` is one hyperparameter decided inside this design gate (§9),
not the milestone driver. `FLASHNH_CURRENT_STATE.md` updated accordingly.

**Explicitly not done this session (by design):**
- Full 2,752-basin NH package was **not** generated.
- No training was run.
- No large generated file was committed; local `reports/flashnh_basin_screening_v001/`
  contents (parquets, csvs, plots) remain untracked, as before.

**Files changed:** `docs/stage1_attribute_provenance.md` (new),
`docs/stage1_scientific_baseline_design.md` (new),
`reports/flashnh_basin_screening_v001/README.md` (new, tracked),
`.gitignore` (Disk_volume_estimation — negation-pattern fix),
`scripts/build_stage1_nh_package.py` (docstring + checksum recording),
`docs/FLASHNH_CURRENT_STATE.md`, `docs/decision_log.md` (this entry).

## 2026-07-02 Smoke 1 PASS — meteorology ingestion confirmed on Moriah

**Decision:** Accept Smoke 1 as a technical meteorology-ingestion PASS. This is NOT a
scientific baseline — seq_length, epochs, and basin count are chosen for verification only.

**Evidence (Slurm job 45370873):**
- Node: `catfish-04`; State: COMPLETED; ExitCode: 0:0; Elapsed: 00:01:41; MaxRSS: 1,380,944 KB
- Preflight before submission: PASS 72 OK / 0 FAIL
- Config accepted by NH 1.13: `dataset: generic`, `seq_length: 24`, `epochs: 3`, `loss: NSE`,
  8 dynamic inputs (`mrms_qpe_1h_mm`, `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`,
  `rtma_10u_ms`, `rtma_10v_ms`, `mrms_qpe_1h_mm_gap`, `rtma_gap`)
- All 5 RTMA variables non-null for all 5 basins (confirmed in preflight)
- `rtma_2d_K` non-null confirms 2K-F-C-B dewpoint mapping fix carried through correctly
- Epoch 1: avg_loss 0.00422; Epoch 2: avg_loss 0.00360; Epoch 3: avg_loss 0.00335
  — all finite, monotonically decreasing; validation completed each epoch
- Run dir: `/sci/labs/efratmorin/omripo/Flash-NH/runs/flashnh_stage1_smoke1_0207_164941`
- Model weights: `model_epoch001/002/003.pt` (~83 KB each); optimizer states; TensorBoard events
- h2o audit (same package as Smoke 0): PASS, 0 errors, 5 expected qobs-NaN warnings

**seq_length: 24 rationale (confirmed working):**
Smoke 1 kept `seq_length: 24` (identical to Smoke 0) to isolate the dynamic-input expansion
(2 inputs → 8 inputs) from any lookback-window change. This makes failures easier to attribute.
The choice is validated: all 8 inputs load, normalize, and produce decreasing finite loss.
Lookback-expansion tests (`seq_length: 72`, `168`, `336 h`) are separate later milestones.

**Config comment discrepancy (minor):**
The config in the evidence bundle carries the stale comment
`# seq_length=72 (3 days): first step up from Smoke 0's 24 h.` — this comment was from a
build done before the comment patch (commit c3ce5df). The actual `seq_length: 24` value is
correct (confirmed by NH runtime log: `seq_length: 24` printed at training start). The
corrected builder now emits the accurate comment; any future package rebuild will be clean.

**Future Slurm improvement (deferred):**
Both sbatch templates (`run_stage1_smoke0/1_moriah.sbatch`) hard-pin `--partition=catfish
--gres=gpu:l4:1`. This prevents running on `salmon` (L40S) or `goldfish` (H200) without
editing the script. Record for future: add `PARTITION` and `GRES` variables at the top of
each sbatch so the GPU target can be changed without touching the rest of the script. Defer
until the reproducibility baseline (scientific baseline training) is established — changing
GPU hardware before that would add a confound.

**Remaining before scientific baseline:**
1. Attribute-source cleanup — `all_basins_merged.parquet` staged at h2o `tmp/`, not committed
2. Lookback-expansion smokes — seq_length 72/168/336 h (separate milestone, after attr cleanup)
3. Full 2,752-basin NH package — after attribute cleanup + lookback smoke PASS

## 2026-07-02 Smoke 1 operational corrections — preflight signature fix + seq_length policy

**Correction 1 — `load_attributes` keyword argument fix in preflight helper.**

During Smoke 1 preflight on Moriah, `scripts/check_stage1_nh_preflight.py` failed with:
```
load_attributes() got an unexpected keyword argument 'attribute_names'
```
Moriah NH 1.13's `neuralhydrology.datasetzoo.genericdataset.load_attributes` does not
accept `attribute_names=` as a positional keyword — the signature differs from what was
assumed when the script was written. Fix: use `inspect.signature(load_attributes)` to
detect whether `attribute_names` is a valid parameter. If present, pass it (forward-compat);
if absent (NH 1.13 Moriah), call `load_attributes(data_dir=pkg, basins=basins)` and verify
that all `cfg.static_attributes` appear as columns in the returned DataFrame.
Explicit column-presence check added regardless of which branch is taken.
Changed files: `scripts/check_stage1_nh_preflight.py` (source only).

**Correction 2 — Smoke 1 `seq_length` policy: keep 24 h, defer 72/168 h.**

Prior docs specified `seq_length: 72` for Smoke 1. Corrected policy: Smoke 1 keeps
`seq_length: 24` (identical to Smoke 0) so that only the dynamic-input variable set
changes between Smoke 0 and Smoke 1. This isolates the input-expansion change from
a lookback-window change, making failures easier to diagnose. Lookback-expansion
tests (`seq_length: 72`, `seq_length: 168`) are now separate named milestones after
Smoke 1 PASS, not part of Smoke 1 itself.
Changed files: `scripts/build_stage1_nh_package.py` (Smoke 1 config template),
`docs/stage1_neuralhydrology_preflight.md` (§7 seq_length note, §13 step 6),
`docs/FLASHNH_CURRENT_STATE.md` (2K-G-A milestone line).

**Note:** The Smoke 1 config currently on Moriah (generated from pre-patch builder) has
`seq_length: 72`. The package must be regenerated on h2o and re-transferred before
submitting Smoke 1 sbatch.

## 2026-07-02 Milestone 2K-G-C COMPLETE — Smoke 0 PASS on Moriah

**Decision:** Accept Smoke 0 as a technical plumbing PASS. Milestone 2K-G-C is closed.
This is NOT a scientific baseline — training parameters (2 epochs, `seq_length: 24`, 1
dynamic input) are chosen for plumbing verification only.

**Evidence (Slurm job 45370683):**
- Node: `catfish-05`; State: COMPLETED; ExitCode: 0:0; Wall time: 00:01:55
- PyTorch 2.7.0+cu128; CUDA available; NVIDIA L4 (23034 MiB)
- Package: `attributes/attributes.csv` found; 5 NC files found (h2o audit PASS, 0 errors)
- Config accepted by NH 1.13: `dataset: generic`, `head: regression`,
  `output_activation: linear`, `epochs: 2`, dates in `DD/MM/YYYY`
- Epoch 1 avg_loss 0.00577 (finite); validation PASS
- Epoch 2 avg_loss 0.00556 (finite); validation PASS
- Model weights saved: `model_epoch001.pt` + `model_epoch002.pt` (~77 KB each)
- Run dir: `/sci/labs/efratmorin/omripo/Flash-NH/runs/flashnh_stage1_smoke0_0207_153320`
- Slurm stdout ended with `=== done ===` (preflight + training + validation complete)

**h2o package audit facts (2026-07-02T11:44:43Z):**
- Package: `/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001`
- Result: PASS; Errors: 0; Warnings: 5 (qobs_m3s NaN counts per basin — expected)

**Technical plumbing PASS means:**
- NH GenericDataset loads the Flash-NH package format without error
- Forward pass, loss computation, and backward pass complete for all 5 basins
- Slurm/module/CUDA/env stack confirmed end-to-end on Moriah `catfish` partition
- Loss values are not scientifically meaningful (2 epochs, rain-only LSTM input)

**Remaining before scientific baseline training:**
1. **Smoke 1** — add 6 RTMA meteorology variables; `seq_length: 24`; confirm non-null RTMA
2. **Attribute-source cleanup** — `all_basins_merged.parquet` staged at h2o `tmp/`, not committed
3. **PyYAML on Moriah** — install in `flashnh-moriah` to enable preflight config checks
4. **Full 2,752-basin package** — after Smoke 1 PASS + attribute cleanup

## 2026-07-02 NH 1.13 compatibility patch — builder, auditor, preflight helper

**Decision:** Patch the Flash-NH Stage 1 NH pilot package generator to emit
NeuralHydrology 1.13 GenericDataset-compatible configs and package layout.
Manual Moriah-side edits were diagnostic only; the source is now the authoritative
emitter of correct configs. No generated outputs are committed.

**Root cause:** Manual Smoke 0 attempts on Moriah revealed that `build_stage1_nh_package.py`
generated several NH 1.13 incompatibilities:
- `dataset: GenericDataset` → NH 1.13 registry key is `dataset: generic`
- ISO date strings (`YYYY-MM-DD`) → NH 1.13 requires `DD/MM/YYYY` for all `_date` fields
- `num_epochs` → NH 1.13 uses `epochs`; `num_epochs` is a rejected key
- `shuffle: true`, `log_n_basins: 5` → rejected by NH 1.13
- Missing `head: regression`, `output_activation: linear` → required at train startup
- `attributes.csv` at package root → NH GenericDataset expects `data_dir/attributes/*.csv`
- Package-internal `slurm/` scripts used wrong partition (`gpu`), wrong invocation
  (`python -m neuralhydrology.training`); repo-level `scripts/run_stage1_smoke0_moriah.sbatch`
  is the correct Slurm entry point

**NH provenance note:** Local Python environment has no `neuralhydrology` installation.
Moriah NH 1.13 (installed via Slurm job `45365952`) is the sole authoritative runtime.
All compatibility targeting is based on Moriah NH 1.13 behavior confirmed interactively.

**Changes (source/scripts/docs only — no generated outputs):**
- `scripts/build_stage1_nh_package.py`: `_write_configs` now emits NH 1.13 compat configs;
  `_write_attributes` writes to canonical `attributes/attributes.csv` path; `_write_slurm`
  no longer called from `main()` (repo-level sbatch is the Slurm entry point)
- `scripts/audit_stage1_nh_package.py`: new `check_configs` section validates all NH compat
  keys; `check_structure`/`check_attributes` updated for `attributes/` canonical layout;
  `slurm/` checks removed
- `scripts/check_stage1_nh_preflight.py`: new lightweight diagnostic for Moriah post-transfer
  verification; NH-level checks (Config, load_attributes) guarded by import check; usable
  locally (structural/data only) and on Moriah (full NH checks)
- `scripts/run_stage1_smoke0_moriah.sbatch`: enhanced preflight block (`which nh-run`, date,
  SLURM_JOB_ID, Python/PyTorch versions, `attributes/attributes.csv` existence check)

**Next step:** Regenerate pilot package on h2o, re-audit, re-transfer, re-submit Smoke 0.

## 2026-07-01 Moriah env install PASS + pilot package transfer PASS

**Decision:** Accept Moriah env install (Slurm job `45365952`) and pilot package transfer
as PASS. Both are confirmed done. Smoke 0 is the next step (not done yet; 2K-G-C not yet
complete). Generated evidence (logs) remain on Moriah, not committed locally.

**Env install (job 45365952):**
- Script: `scripts/setup_flashnh_moriah_env.sbatch` (after manual module fixes, see below)
- Env prefix: `/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah`
- `nh-run` confirmed at `envs/flashnh-moriah/bin/nh-run`; `nh-run --help` lists valid modes
  (`train`, `continue_training`, `finetune`, `evaluate`)
- `neuralhydrology` imports OK; no `__version__` attribute (expected)
- Log ended with `=== done ===`. Matplotlib font cache message in stderr is harmless.

**Module fixes (binding for all future Moriah Slurm scripts):**
Initial sbatch run failed because `module` is not in PATH in non-interactive Slurm shells,
and `miniconda3` requires `spack/all` to be loaded first. Three corrections applied:
1. Source a module-system init file at job start if `module` is not already in scope.
2. `module load spack/all` before any other module.
3. Use exact module name `miniconda3/24.3.0-gcc-iqeknet` (not `miniconda3/24.3.0`).
Both `scripts/setup_flashnh_moriah_env.sbatch` and `scripts/run_stage1_smoke0_moriah.sbatch`
updated in this commit.

**Pilot package transfer (h2o → Moriah):**
- Source: `/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/`
- Destination: `/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001`
- Verified: 5 NC files under `time_series/`, `run_provenance.json` present,
  `configs/stage1_smoke0_nh.yml` present, `attributes.csv` present, size 19 MB.

## 2026-07-01 Milestone 2K-F-C: Corrected full-period curated forcing v001 PASS

**Decision:** Accept the corrected full-period curated forcing product v001 rebuild as PASS.
This closes the 2K-F-C-B schema correction loop and unblocks full 2,752-basin NH package
generation (pending Smoke 0 PASS and attribute-source cleanup — see separate gates).

**Build facts (from evidence bundle, locally at `tmp/stage1_curated_forcing_v001_corrected_fullperiod_evidence/`):**
- Product: `stage1_basin_hourly_forcings_v001`
- h2o location: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001`
- Period: 2020-10-14T00:00:00Z – 2025-12-31T23:00:00Z
- Months processed: 63 / 63; basins built: 2,752 / 2,752; 0 failed
- Rows per basin: 45,720 (full period)
- Total MRMS gap-hours: 374,272 (= 136 × 2,752 — exact match)
- Total RTMA gap-hours: 5,504 (= 2 × 2,752 — exact match)
- Wall time: 51,932.6 s (14.43 h)
- Run start: 2026-06-30T10:08:53Z; run end: 2026-07-01T00:34:26Z
- Repo commit at run: `5f07d4b`

**Audit result (full-period mode): PASS**
- 2,752 / 2,752 basins checked; 45,720 rows per basin
- MRMS gap-hours/basin = 136 ✓; RTMA gap-hours/basin = 2 ✓
- Known RTMA gap timestamps: 2020-11-12T09Z and T10Z ✓

**Sample20 diagnostic: ALL PASS**
- 20 Parquets spot-checked; each 45,720 rows
- `mrms_qpe_1h_mm` non-null = 45,584 (= 45,720 − 136 MRMS gaps) ✓
- All RTMA variables non-null = 45,718 (= 45,720 − 2 RTMA gaps) ✓
- `rtma_2d_K` populated ✓ (confirms 2K-F-C-B dewpoint mapping fix)
- `rtma_weasd_kgm2` absent ✓ (confirms 2K-F-C-B schema removal)

**Generated evidence (not committed):**
- `tmp/stage1_curated_forcing_v001_corrected_fullperiod_evidence/` (local archive)
- Includes `build_summary.md`, `audit_summary.md`, `run_provenance.json`, `manifest.json`,
  `checksums.sha256`, `dataset_config.json`, `build.log`

**What this unblocks:** Full 2,752-basin NH package generation now has a valid corrected
forcing library input. Remaining gates before full NH package: (1) Smoke 0 PASS;
(2) attribute-source cleanup (staged `all_basins_merged.parquet` not committed).

## 2026-06-30 Milestone 2K-G-C-A: Moriah GPU/Conda/Slurm preflight facts recorded

**Decision:** Record real Moriah/HURCS facts gathered via interactive `ssh`/`srun`
reconnaissance, and prepare (but not run) two Slurm templates. This is **preflight
documentation and script preparation only** — no job was run on Moriah, the env is not
installed, the pilot package was not transferred, and Smoke 0 was not attempted. 2K-G-C
is not complete; only 2K-G-C-A.

**Facts recorded (see `docs/stage1_neuralhydrology_preflight.md` §10.6 for full detail):**
1. Login node `moriah-gw-01`; lab storage `/sci/labs/efratmorin/omripo/Flash-NH` with
   subdirs `repos, envs, data, runs, logs, slurm, evidence`. Do not rely on
   `/sci/home/omripo` inside Slurm jobs.
2. Slurm partitions confirmed via `sinfo`: `catfish` (L4, 7-day limit, chosen for
   Smoke 0), `salmon` (L40S), `goldfish` (H200), `dogfish` (A100, drained at check time),
   `glacier` (CPU default).
3. Working interactive allocation:
   `srun --partition=catfish --gres=gpu:l4:1 --cpus-per-task=4 --mem=16G --time=00:10:00 --pty bash`.
4. On allocated node `catfish-05`: L4 GPU, 23034 MiB; `nvidia-smi` requires
   `module load nvidia/580.95.05` (reports driver 580.95.05 / CUDA 13.0); toolkit
   `module load cuda/12.8.1` confirmed (`nvcc` → 12.8, V12.8.93).
5. Conda is module-gated; compute allocations auto-load `miniconda3/24.3.0-gcc-iqeknet`.
   Moriah env must be a **prefix env** (`conda create -p ...`) under the Flash-NH project
   root, not a named env and not under `/sci/home`.

**Scripts prepared (not run):**
- `scripts/setup_flashnh_moriah_env.sbatch` — env install on `catfish`. Leaves the
  PyTorch CUDA wheel choice as an explicit TODO rather than guessing a wheel tag, since
  the driver-reported CUDA (13.0) and loaded toolkit (12.8.1) differ and the actual
  compatible wheel was not verified.
- `scripts/run_stage1_smoke0_moriah.sbatch` — Smoke 0 on `catfish`. Chooses
  `nh-run train --config-file ...` (upstream NH console-script entry point) as the first
  invocation to try, with `python -m neuralhydrology.nh_run train` documented as the
  fallback. Explicitly does **not** use `python -m neuralhydrology.training` — the
  invocation baked into `scripts/build_stage1_nh_package.py`'s `_write_slurm` helper —
  which is flagged as likely incorrect and unverified. Reconciling that generator
  function is a follow-up, not done in this milestone.

**Transfer procedure documented (not executed):** `scp` from h2o
`/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/` to Moriah
`/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001/`, verified by NC file count
(expect 5), `run_provenance.json` presence, and package size (~25 MB). No checksum
manifest exists for this package, so file-count/manifest-presence is the practical check.

**Not done:** h2o jobs, Moriah jobs, env install, package transfer, Smoke 0, 2K-G-C
completion.

## 2026-06-30 Milestone 2K-G-B h2o validation: NH pilot package PASS

Audit result: PASS — 0 errors, 5 warnings, 217 OK checks.
Package: `/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/`
Build time: 4.2 s; wall timestamp: 2026-06-30T12:38:35Z.

**Findings and decisions:**

**1. 5-basin corrected pilot is sufficient for package-builder validation.**
All 5 basins pass all structural, gap-count, and dewpoint-fix checks. The builder and
auditor are validated. Full 2,752-basin package generation is a separate, authorized step.

**2. Gap-fill report matches expected counts exactly.**
MRMS: 136 NaN → 0.0 mm per basin (all 5). RTMA: 2 NaN → linear interp per variable per
basin (all 10 RTMA variables, all 5 basins). No unexpected gaps introduced by reindex.

**3. qobs NaN preserved exactly; 5 warnings are expected.**
Warnings are informational: qobs NaN counts of 515, 6,751, 12,088, 3,035, 6 for the five
basins. These are the missing-discharge gaps from the target package v001. NH loss-masks
these at training time. No action needed.

**4. rtma_2d_K non-null == 45,720 confirmed per NC.**
The 2K-F-C-B dewpoint mapping fix (`d2m` → `2d` in the forcing builder) is confirmed to
have propagated correctly into the NH package. This check is retained in the auditor
permanently for any future package rebuild.

**5. Static attribute file not committed to git — cleanup gate established.**
`reports/flashnh_basin_screening_v001/all_basins_merged.parquet` is not tracked in git
(confirmed with `git ls-files` on h2o). Builder used a manually staged copy at
`/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet`. The pilot PASS is valid.
Resolution options before full-scale packaging:
  (a) Commit the parquet to the repo (adds ~1–2 MB to git history; clean path).
  (b) Document it as a canonical h2o-resident file at a fixed path, with explicit
      provenance note in `--attributes-csv` docs and run_provenance.json.
This is a cleanup gate, not a scientific blocker.

**6. Next milestone is 2K-G-C: Moriah transfer + NH environment preflight + Smoke 0.**
Package is ready. Transfer via `scp`. No scientific training; Smoke 0 is plumbing/ingestion
verification (seq_length=24, 2 epochs, 5 basins, rain-only). Smoke 0 PASS = NH environment
and package format confirmed compatible.

## 2026-06-30 Milestone 2K-G-B: NH package builder and auditor design decisions

Scripts implemented: `scripts/build_stage1_nh_package.py`, `scripts/audit_stage1_nh_package.py`.
Status: IMPLEMENTED (local syntax PASS); h2o run pending.

**1. Attribute source: `reports/flashnh_basin_screening_v001/all_basins_merged.parquet`.**
This file is committed to the repo and all 5 pilot basins confirmed present with required
columns (`DRAIN_SQKM`, `LAT_GAGE`, `LNG_GAGE`, `BFI_AVE`). `--attributes-csv` accepts
`.parquet` or `.csv`; STAID column (int64) normalized to 8-char zero-padded string.
All available columns (not just the 4 required) are written to `attributes.csv` — NH
reads only the columns listed in `static_attributes` config at runtime.

**2. All 14 NC variables: 11 forcing data + 2 gap flags + qobs_m3s.**
`rtma_sp_Pa` is included in the NC (not excluded) so it is available for Smoke 2 without
rebuilding the package. It is excluded from the `dynamic_inputs` list in `stage1_smoke1_nh.yml`.

**3. Gap flags stored as float32 (0.0/1.0), not bool.**
NH GenericDataset expects numeric input arrays. Bool xarray variables may cause issues with
NH normalization. Explicitly converting gap flags to float32 in the builder.

**4. Atomic NC writes (tmp + rename).**
Same pattern as target builder: write to `{STAID}.nc.tmp`, then rename. Avoids partial-write
files if the builder is interrupted.

**5. Auditor checks mrms_qpe_1h_mm_gap sum == 136 and rtma_gap sum == 2 per basin.**
These are the expected gap counts from the corrected v001 forcing library. A mismatch would
indicate a gap-fill bug or a wrong forcing source directory.

**6. Auditor checks rtma_2d_K non-null == 45720 explicitly.**
This check directly confirms that the 2K-F-C-B dewpoint mapping fix (`d2m` → `2d`) was
correctly carried through to the NH package NC files. If the old mapping bug recurs in a
future rebuild, this check will catch it.

## 2026-06-30 Milestone 2K-G-A corrections: Smoke design and gap-fill policy revision

Corrections to the 2K-G-A preflight design (commit `fa6754b`) before 2K-G-B implementation.
No code changes; docs-only patch.

**1. Smoke 0 `seq_length`: 336 h → 24 h; add `predict_last_n: 1`.**
Smoke 0 is a pure plumbing/ingestion test, not a scientific baseline. 14-day lookback is
unnecessary overhead for verifying that NH loads the package and produces finite loss.
24 h minimises runtime and memory before package-loading is proven. `seq_length: 336` is
reserved for later hyperparameter testing.

**2. Smoke 1 `seq_length`: 24 h (revised — see 2026-07-02 correction entry).**
First meteorology smoke keeps `seq_length: 24` to isolate input expansion from lookback
change. `seq_length: 72`/`168` are later dedicated lookback-expansion tests; 336 h is a
later hyperparameter candidate after those pass.

**3. MRMS gap-fill policy: Smoke 0/1 pilot policy only — not final scientific training policy.**
Precipitation is the primary forcing driver; silently treating archive gaps as true no-rain
must not carry into scientific baseline training. For final training, evaluate:
- window/sample exclusion (exclude training windows that intersect MRMS gap hours —
  do NOT remove rows from the NC file; the 45,720-h `date` coordinate stays aligned);
- or a deliberately tested NH `nan_handling_method`.
The 0.0 mm fill + gap flag strategy is accepted for Smoke 0/1 only.

**4. RTMA gap-fill: linear interpolation (2 hours) is accepted as pilot/package policy;
review before final scientific training.**

**5. 2K-G-B unblocked from full rebuild wait.**
The 5-basin NH pilot package builder can be implemented and tested now, using the
already-passing corrected 5-basin forcing pilot. Full-scale package generation (2,752
basins) waits for the full rebuild PASS, but the 5-basin builder and audit do not.

## 2026-06-30 Milestone 2K-G-A: NeuralHydrology Pilot Package Preflight Design

Design frozen in `docs/stage1_neuralhydrology_preflight.md` (Part I). Key decisions:

**1. Package format: GenericDataset single NC per basin.**
One NC per basin with all dynamic vars (forcings + gap flags) + `qobs_m3s` target on
a shared `date` coordinate. Matches Milestone 2G format proven with NH, avoids a
custom dataset class. Float32 values, `_FillValue=-9999.0`, no tz offset in coordinate.

**2. Gap-fill policy for NH package (binding for v001).**
- MRMS gaps (136 h / basin, 0.30%): fill with 0.0 mm (conservative no-rain assumption)
- RTMA gaps (2 h / basin, 0.004%): fill with linear interpolation (2 hours; both neighbors always available)
- Gap flags (`mrms_qpe_1h_mm_gap`, `rtma_gap`) retained as explicit dynamic inputs
- Do NOT rely on NH `nan_handling_method` as the primary strategy; pre-fill in the package builder
- Rationale: transparency — NaN in dynamic inputs is dangerous in LSTM by default; pre-fill is
  auditable in the package file; gap flags preserve the information signal

**3. Smoke levels.**
- Smoke 0 (rain-only technical): mrms_qpe_1h_mm + gap flag; 5 basins; 2 epochs; purpose is
  NH load/train verification only, not a scientific model
- Smoke 1 (minimal meteorology): + rtma_{2t,2d,2sh,10u,10v} + rtma_gap
- rtma_sp_Pa: include in NC file (for future use), exclude from Smoke 1 dynamic_inputs
  (large magnitude ~70k–101k Pa; defer normalization review to Smoke 2)

**4. Train/val/test split.**
Train: 2020-10-14 – 2022-12-31 | Val: 2023 | Test: 2024-2025
Rationale: 2024–2025 is the quasi-operational period; hold out entirely. Val is 2023
for generalization monitoring; contains varied seasonality.

**5. NH setup: clean upstream clone, no fork until specific limitation demonstrated.**
Old Flash-NH fork is abandoned. All custom logic lives in: NH YAML configs (in this repo),
package builder script, and future `src/flashnh/` custom classes. Fork only when a
config-layer workaround is exhausted.

**6. Moriah layout.**
`/sci/labs/efratmorin/omripo/Flash-NH/{repos,envs,data,runs,logs,slurm,evidence}`
Blocking unknown: GPU partition name, CUDA version — must check Moriah wiki and `sinfo`.

## 2026-06-30 Milestone 2K-F-C-B: Curated Forcing v001 Schema/Mapping Correction

Full-period build (2,752 basins × 45,720 h) structurally passed on h2o (2026-06-30,
commit `addfdd2`, 14.49 h wall). Post-build non-null check found two all-NaN variables,
triggering a schema correction before certification.

**Schema findings:**

| Variable | Non-null (5 sampled basins) | Decision |
|---|---|---|
| `rtma_2d_K` | 0 / 45,720 | **Retain** — mapping bug fixed (`d2m`→`2d`) |
| `rtma_weasd_kgm2` | 0 / 45,720 | **Remove** — `weasd` absent from all 63 source months |
| `rtma_2t_K` | 45,718 / 45,720 | Retain (normal) |
| `rtma_sp_Pa` | 45,718 / 45,720 | Retain (normal) |

**Decisions (all binding for v001):**

1. **Dewpoint retained, mapping corrected.** Source variable is `2d` (`dewpoint_temperature_2m`),
   not `d2m`. Confirmed present with `recommended_for_initial_model=True` in all 5 sampled months.
   Both builders updated: `"2d" → "rtma_2d_K"`.

2. **`rtma_weasd_kgm2` removed from v001 schema.** `weasd` is absent from all 63 monthly
   source chunks. RTMA precipitation (`ACPC01`) is not present in the RTMA CONUS source.
   Precipitation is supplied by MRMS QPE; no RTMA precip column is added. `rtma_weasd_kgm2`
   is now in `_FORBIDDEN_COLS` in the auditor — its presence in output is a FAIL.

3. **Full-period structural build is schema-superseded, not failed.** Gap counts (136 MRMS,
   2 RTMA per basin), row counts (45,720), and checksums were correct. The product correctly
   reflects the source data; the errors were a missing dewpoint (now fixed) and a spurious
   NaN column (now removed). A corrected 5-basin full-period pilot on h2o is required before
   the full 2,752-basin rebuild is authorized.

4. **Auditor non-null coverage checks added.** Full-period mode now verifies exact non-null
   counts: `mrms_qpe_1h_mm` → 45,584; each RTMA var → 45,718. Single-month mode: not-all-NaN
   guard for all data variables.

5. **`build.log` caveat.** An accidental second launch was stopped early after the first PASS.
   Post-interruption full-period audit PASS confirmed the product was not corrupted. `build.log`
   may contain aborted-rerun lines after the first complete PASS block.

**Corrected v001 schema:** 1 MRMS + 10 RTMA + 2 gap flags = 13 columns total (was 14).

**Evidence:** `tmp/stage1_curated_forcing_v001_schema_issue_evidence/` (not committed).

## 2026-05-06

- Confirmed GFS `.idx` byte-range extraction end-to-end; the acquisition path is validated and frozen except for plotting.
- Confirmed IFS 00/12 UTC MARS retrievals succeed, while 06/18 UTC remain unresolved and should stay open as an access/request issue.
- Confirmed IMERG NC4 download is valid; crop logic still needs repair and final nonzero selected-CONUS confirmation.
- Retained RTMA despite size/time cost because it is scientifically important for the Stage 1 pipeline.

## 2026-05-06 IFS Stream Investigation

- Tested `oper/fc` requests for 06 UTC and 18 UTC on 2023-01-01 with `area` subset and without `area`; all returned `MARS_EXPECTED_FIELDS` with 0 retrieved fields.
- Tested `oper/fc` with the current variable set at `step=0` and `step=0/to/24/by/1`; both area-subset and full-domain forms failed for 06 UTC and 18 UTC.
- Tested `scda` alternatives with a minimal `2T` request at `step=0`.
- `scda/type=fc` succeeded for both 06 UTC and 18 UTC.
- `scda/type=an` succeeded for both 06 UTC and 18 UTC.
- `scda/type=cf` failed with `MARS_EXPECTED_FIELDS` and 0 retrieved fields.
- The area subset `50/-126/24/-66` did not explain the oper failures.
- **Stream fix implemented**: 00/12 UTC use `oper/fc`, while 06/18 UTC use `scda/fc`.
- Provisional recommendation: use the deterministic `scda` path for historical 06/18 UTC retrievals; keep `oper` as the working path for 00/12 UTC and do not assume only 2 historical IFS cycles exist.

## 2026-05-06 IFS Resolution Comparison

- Tested 2023-01-01 historical retrievals at two grid resolutions:
  - **0.25/0.25**: current configuration
  - **0.1/0.1**: proposed higher resolution
- For cycles 00 UTC (oper/fc) and 06 UTC (scda/fc), tested both minimal requests (1 variable, 1 step) and full requests (7 variables, 25 steps).
- **Results**: Both resolutions retrieved successfully with 100% pass rate.
  - 0.25/0.25: ~17.9 MB total for both cycles (8.9 MB per cycle average)
  - 0.1/0.1: ~110.5 MB total for both cycles (55.3 MB per cycle average)
  - Ratio: 0.1/0.1 is ~6.2× larger
- **Timing**: Higher resolution added ~5–15 seconds per request but remained well within operational tolerance.
- **Recommendation**: Adopt **0.1/0.1 resolution** to align with IFS's scientific value (higher resolution than GFS).
  - Justification: Estimated annual burden ~80 GB (acceptable for 2-year window); retrieval time penalty negligible; area subset remains supported.
  - Contingency: Revert to 0.25/0.25 requires single config change if burden becomes untenable.
- **Decision**: Update `IfsMarsConfig.grid` to `0.1/0.1` and document stream logic in code comments.

## 2026-05-07 IFS 0.1-Degree Estimate Verification

- Verified estimate inputs without changing IFS retrieval logic:
  - 00/12 UTC: `oper/fc`
  - 06/18 UTC: `scda/fc`
  - `grid=0.1/0.1`, `area=50/-126/24/-66`, 7 variables, `step=0..24`, 4 cycles/day.
- Recomputed period: 2020-10-14T00:00:00 to 2025-12-31T23:59:59 (inclusive, 1,905 days; 7,620 cycles).
- Empirical sample bytes per cycle (full request, from resolution test): **54,920,250 bytes**.
- Bytes per day (4 cycles): **219,681,000 bytes** (~219.681 MB/day, ~209.505 MiB/day).
- Full-period raw download estimate: **418,492,305,000 bytes** (~418.492 GB, ~389.751 GiB).
- Full-period retained raw estimate: **418,492,305,000 bytes** (same as download estimate under current workflow assumptions).
- Derived basin-average estimate (9,000 basins; hourly; 7 vars; float32 parquet): **11,521,440,000 bytes** (~11.521 GB, ~10.730 GiB).
- Estimated acquisition time (using measured full-request mean cycle time: (54.40s + 42.36s)/2 = 48.38s):
  - ~193.52s/day (~3.23 min/day)
  - ~368,655.6s total (~102.40 h, ~4.27 days) if executed sequentially.
- Validation of prior wording: **"~80 GB/year" is approximately correct in decimal units**.
  - Recomputed value: **80.184 GB/year** (or **74.677 GiB/year** binary).

## 2026-05-07 IMERG Crop And Preview Plot Repair

- Repaired IMERG CONUS crop handling for dynamic coordinate layouts and dimension order, including `time,lat,lon`, `time,lon,lat`, `lat,lon`, and `lon,lat` forms.
- Added robust crop logging for IMERG:
  - original dims and coordinate names
  - original lon/lat bounds
  - cropped lon/lat bounds
  - cropped shape and min/max/mean/nan_pct
- Added hard failure when IMERG crop result is empty and when `selected_conus_bytes` is zero.
- Verified targeted IMERG validation on 2023-01-01 (`3B-DAY-L.MS.MRG.3IMERG.20230101-S000000-E235959.V07B.nc4`):
  - `selected_conus_bytes=624000` (nonzero)
  - crop bounds `lon=[-125.950, -66.050]`, `lat=[24.050, 49.950]`
  - crop shape `(260, 600)`
- Repaired preview plotting axes/orientation to use true lon/lat extent and north-up orientation logic.
- Added preview bounds validation logging (`preview_bounds_validation=PASS/FAIL`) and summary payloads.
- Verified preview bounds validation passed for:
  - IMERG: PASS
  - GFS: PASS
  - IFS: PASS
- Generated run artifacts under `reports/audit_2026_04_29/run_07_imerg_plot_repair/` with lightweight review bundle (no raw NC4/GRIB files).

## 2026-05-09 Final All-Source Acquisition Audit

- Executed unified 24-hour acquisition audit (2023-01-01) in dry-run mode to validate all 7 implemented datasources without large downloads.
- Orchestration script: [scripts/run_final_all_source_audit.py](../scripts/run_final_all_source_audit.py).
- Audit outputs under `reports/final_all_source_audit_2026_05/`:
  - **24-hour summaries**: `final_audit_summary.{json,csv,md,html}` + `datasource_matrix.{csv,md}`
  - **Decision-support plots**: `storage_breakdown_by_source.png`, `reduction_waterfall_by_source.png`, `download_time_vs_size.png`, `availability_timeline.png`, `crop_validation_overview.png`
  - **Previews & request specs**: organized by source in `previews/` and `request_specs/` directories
  - **Lightweight review bundle**: `review_bundle/` with representative artifacts, logs (truncated), manifests, and docs
- Validation status: All sources validated with current request logic; no architecture changes applied.
- 7-day stability audit attempted for subset (MRMS, RTMA, GFS, IFS, IMERG) but dry-run unable to cache full 7-day sample; logic implemented and ready for real-run.
- Final recommendation sections added to summary markdown with operational stack priorities:
  - **Stage 1**: MRMS + RTMA (high-priority, smallest acquisition burden)
  - **Stage 2**: ERA5-Land + GDAS + IMERG Late Daily (medium-priority, moderate burden)
  - **Stage 3**: GFS + IFS (low-priority, highest burden; IFS uses 0.1/0.1 grid with stream split logic)
- Caveats: External data provider availability, credential lifecycle, and throughput variance remain operational (not logic) issues.

## 2026-06-20 Stage 1 Forcing Throughput Optimization (Milestone 2K-D)

**D1 — Serial extraction optimization (commit `3ff4965`):**
- Pre-grouped the weight DataFrame into a `{STAID: (row_idx, col_idx, norm_w)}` dict at startup,
  eliminating 90,816 O(N) scans per RTMA-hour and shifting per-basin-hour lookup to O(1).
- Replaced 7 sequential `np.percentile` calls with one batched call (635,712 redundant sort passes
  eliminated per RTMA-hour).
- Measured result: `extraction_median_s` 91.976 s → 2.17 s/hr (**24.7× speedup**).
- Bottleneck fully shifted to S3 download. D2 process-workers judged unnecessary and **deferred**.

**Download-worker sensitivity benchmark (48h RTMA-only, 2,752 basins):**
- dw2 → dw16 scanned: individual download time increases (31 → 45 s/file) due to S3 bandwidth
  sharing, but wall-clock decreases via prefetch concurrency.
- dw16 = 570.5 s wall → 6.29 days projected (GREEN vs 14-day target, but not compelling alone).

**Outer-parallelism x2 (2 chunks × dw8, commit `cf8db74`):**
- Parent wall 736 s → 4.057 days projected — **YELLOW** (partial scaling; insufficient alone).
- Decision: do not proceed with x2.

**Outer-parallelism x3 (3 chunks × dw6, commit `a275296`):**
- Parent wall 826 s → 3.035 days projected — **USEFUL GREEN** (within acceptable range).
- All 3 chunks: `all_pass=True`, 48/48 hours, 1,453,056 rows each.
- Decision: **stop optimization here**.

**Final decisions (all binding):**
1. Full-period launch configuration: **3 concurrent chunk processes × 6 download workers each**
   (18 total S3 connections). Splits 63 months into 3 groups (~21 months each).
2. D2 process-workers: **deferred indefinitely.** Extraction is 2.17 s/hr; download dominates.
3. x4 outer-parallelism: **not recommended.** S3 contention risk; marginal gain; x3 is sufficient.
4. `run_stage1_forcing_fullperiod_h2o.sh` needs outer-parallel group support before Phase 2 launch.
5. All h2o paths remain under `/data42/omrip/Flash-NH/` (system `/tmp` prohibited).

## 2026-06-20 Stage 1 Forcing — 2K-E Pre-Launch Patch

**Goal:** Enable 3-way outer parallelism without a new launcher script.

**Changes applied (pre-launch patch, not yet run):**

- `GROUP_ID=A|B|C` env var added to `run_stage1_forcing_fullperiod_h2o.sh`; filters the 63-month
  `MONTH_LIST` to the group's sub-range before the loop. Empty `GROUP_ID` preserves original
  sequential all-63-month behaviour.
  - Group A: 2020-10 → 2022-06 (21 months)
  - Group B: 2022-07 → 2024-01 (19 months)
  - Group C: 2024-02 → 2025-12 (23 months)
- `DRY_RUN=1` mode prints the selected month list and extractor command template, then exits.
  Used to confirm group month counts before committing to a multi-day run.
- Per-group run logs: `manifests/group_{a,b,c}_run_log.txt` (independent; no write conflicts).
- Path safety guard: launcher fails immediately if `FORCING_ROOT` does not begin with
  `/data42/omrip/Flash-NH/`.
- `TMPDIR` redirected to `/data42/omrip/Flash-NH/tmp/tmpdir_flashnh`; never writes to system `/tmp`.
- `${FORCING_ROOT}/logs/` created at startup for screen `tee` targets.
- `report_stage1_forcing_progress_h2o.sh` Section 1 updated to scan all three group logs.

**Decision:** Do not launch extraction until this commit is on h2o and dry-run is confirmed PASS.

## 2026-06-24 Stage 1 Forcing — Full-Period Extraction Audit Acceptance

**Decision:** Accept the full-period MRMS+RTMA forcing extraction as **PASS_WITH_CAVEATS**.
No rerun required.

**Basis:**

- 63/63 monthly chunks `all_pass=True`; 0 failures across Groups A/B/C (PASS=21/19/23).
- 1,509,422,464 combined rows; 0 row-count formula mismatches (11 RTMA vars x n_basins x successful_hours).
- Schema: `rtma_10wdir_absent=True` and `rtma_orog_absent=True` confirmed for all 63 months.
- 138 missing hour-products across 20 months; all `not_in_s3` (permanent S3 archive absences).
  MRMS: 136 hours; RTMA: 2 hours (2020-11-12T09Z and T10Z - newly discovered in audit).
- 0 product-synchronized gaps; 0 unexpected warnings.
- MRMS 24h window impact: 949 / 45,697 possible windows (2.08%).
- Evidence: `tmp/stage1_forcing_fullperiod_evidence_20260624T060504Z.tar.gz` (local, not committed).
- Audit tables: `tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/` (local, not committed).

**Caveats recorded:**

1. **Two-commit provenance:** 2020-10 used extractor commit `194a489` (Phase 1 run);
   62 other months used `7e43760` (D1-optimized full-period extractor). Both pass all 12
   validation checks - documentation caveat only, no functional inconsistency.
2. **MRMS not_in_s3 gaps:** 136 missing MRMS hours are permanent S3 absences. Gap policy:
   preserve as NaN in raw curated product; isolated 1h gaps may be interpolated in derived
   package layers only (per `docs/stage1_forcing_fullperiod_postrun_audit_plan.md section 6`).
3. **RTMA gap discovery:** 2 RTMA hours missing in 2020-11. Not anticipated prior to audit.
   Month remains `all_pass=True`; no corrective action warranted.

**This acceptance does not authorize** curated forcing product v001 assembly (requires
visual QC gate) or NeuralHydrology package assembly or model training.

**Full result:** `docs/stage1_forcing_fullperiod_audit.md`

## 2026-06-25/28 Stage 1 Forcing — Pilot Visual QC PASS

**Decision:** Accept the pilot visual QC evidence as **PASS** for the 6-case basin-timeseries
pilot and the 2-case spatial MRMS smoke. This is a technical/rendering PASS and a scientific
QC evidence improvement. It is **not a final full forcing certification**.

**Basin-timeseries pilot (6/6 OK, 2026-06-25):**

- Cases: VQC-001, VQC-004, VQC-007, VQC-009, VQC-012, VQC-020.
- Time-series rendering, MRMS gap labeling (gray bars), RTMA gap labeling (orange shading),
  qobs hydrograph alignment, VQC-001 period-boundary clip annotation — all pass.
- h2o output: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_visual_qc_pilot_20260625T123337Z`
- Generated GIF/PNG/CSV/manifest outputs are not committed.

**Spatial MRMS smoke (VQC-009 + VQC-012, 2026-06-25/28):**

- Script: `scripts/generate_fullperiod_spatial_mrms_qc.py`
- Both cases: `basin=Y`, `gauge=Y` (basin polygon and gauge marker rendered).
- Cartopy unavailable on h2o — plain lon/lat axes with pcolormesh raster used. No basemap.
  This is sufficient for spatial placement QC; not a rendering blocker.
- MRMS lon normalization (0–360 → −180–180) applied in script; CAMELSH CRS auto-assigned
  EPSG:4326 (shapefile has no `.prj`; bounds confirmed geographic).

| Case | Observation | Interpretation |
|---|---|---|
| VQC-012 (08155541, small flashy TX) | Strong near-basin rainfall at max-hour | Consistent with sharp qobs response; no alignment failure |
| VQC-009 (09484000, SW monsoon AZ) | Patchy convective rainfall near/partly over Sabino Creek | Weak qobs response plausible (partial spatial overlap); not an extraction failure |

**h2o output directories:**
- VQC-012: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_spatial_mrms_qc_smoke_20260625T142012Z`
- VQC-009: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_spatial_mrms_qc_smoke_20260625T142332Z/VQC-009/`

**Scope of this acceptance:**
- Authorizes proceeding to curated forcing product v001 design.
- Does **not** authorize skipping the remaining 15 cases if reviewer finds the pilot evidence
  insufficient for full certification.
- Does **not** authorize NeuralHydrology package assembly or model training.
- Generated PNG/GIF/CSV/summary outputs remain under `tmp/` and must not be committed.

**Full evidence:** `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md`

## 2026-06-29 Stage 1 Forcing — Milestone 2K-F-A: Curated Product v001 Design

**Decision:** Freeze the curated forcing product v001 contract. No data is built in this
milestone. Builder and auditor implementation are deferred to Milestone 2K-F-B.

**Design decisions (all binding for v001):**

1. **Format: wide Parquet per basin.** One row per hour; one column per variable. The monthly
   extraction Parquets (long format) remain unchanged. The per-basin product is a separate
   derived format chosen for NH DataLoader compatibility.
2. **Schema: 12 data columns + 2 gap-flag columns.** 1 MRMS variable (`mrms_qpe_1h_mm`) +
   11 RTMA variables (9 dynamic + `vis` + `ceil`). Gap flags: `mrms_qpe_1h_mm_gap` (bool)
   and `rtma_gap` (bool). `10wdir` and `orog` excluded (absent from S3 in all 63 months).
3. **Gap policy: NaN preserved, no imputation, no row dropping.** Known gaps (136 MRMS hours,
   2 RTMA hours) are NaN in value columns and `True` in gap-flag columns. Every gap hour
   has a complete row in the hourly index.
4. **Smoke test month: 2020-11.** Chosen because it contains the 2 known RTMA gap hours
   (2020-11-12T09Z/T10Z) and 0 MRMS gaps — best stress test of RTMA gap handling.
5. **Product name and path confirmed:** `stage1_basin_hourly_forcings_v001` under
   `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001/`
   (full build); smoke under `tmp/stage1_basin_hourly_forcings_v001_smoke_<TIMESTAMP>/`.

**All five open choices resolved (2026-06-29 follow-up patch).**

**Full design:** `docs/stage1_curated_forcing_product_v001_design.md`

## 2026-06-29 Stage 1 Forcing — Milestone 2K-F-A: Open Choices Resolved

**OC-1 — Script naming:** Builder: `scripts/build_stage1_curated_forcing_basin_parquets.py`;
auditor: `scripts/audit_stage1_curated_forcing_basin_parquets.py`. Legacy name
`build_stage1_forcing_basin_ncs.py` is retired. Rationale: product format is wide Parquet;
the future NH-package builder (separate milestone) will create NetCDFs.

**OC-2 — Full-build output location:** First build stays under
`/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001/`.
Promotion to `/data42/hydrolab/Data/Flash-NH_data/` is a separate explicit gate after
full audit, checksums, and evidence-bundle review.

**OC-3 — RTMA gap flag granularity:** One shared `rtma_gap` boolean column for v001.
Known RTMA gaps are whole-product-hour absences (2020-11-12T09Z/T10Z), not variable-specific
decode failures. The auditor must check per-variable completeness and fail if variable-specific
missingness appears outside known product-hour gaps.

**OC-4 — `vis` and `ceil` inclusion:** Include all 11 extracted RTMA variables in the
curated product. Curated product preservation and first-model-input variable selection are
separate decisions; the first NH smoke config may use a narrower subset without changing v001.

**OC-5 — Remaining 15 VQC cases:** Not required before the 2K-F-B smoke test or the full
2,752-basin build. Gate for full build: 2K-F-B smoke PASS + no schema/gap/provenance failures.
Render 2–3 additional targeted VQC cases only if the smoke or design review reveals a concern.

## 2026-06-29 Stage 1 Forcing — Milestone 2K-F-B: Curated Forcing v001 Builder + Smoke PASS

**Decision:** Milestone 2K-F-B is COMPLETE. Builder, auditor, and h2o launcher implemented
and smoke-tested. Commit `6f4de498f1326e5e6fcd3de8157ba410ad28a6a9`.

**Smoke test result (h2o, 2026-06-29T13:27:57Z):**

| Metric | Value |
|---|---|
| Month | 2020-11 |
| Basins | 5 (`01440000`, `03021350`, `08155541`, `09484000`, `01019000`) |
| Hours per basin | 720 |
| MRMS gap-hours | 0 (correct — 2020-11 has no MRMS S3 gaps) |
| RTMA gap-hours | 10 total (2/basin at `2020-11-12T09:00:00Z` and `T10:00:00Z`) |
| Coverage fraction | 0.9972 (718 valid combined hours / 720) |
| Auditor | PASS — exit 0; all metadata, checksum, schema, and gap-flag checks passed |
| Wall time | 0.1 s |
| h2o output | `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z` |

**Gap verification:**
- `rtma_gap=True` at both known gap timestamps for all 5 basins — confirmed
- All 11 RTMA data columns NaN at gap hours — confirmed
- `mrms_qpe_1h_mm_gap=False` at RTMA-only gap hours (no false flagging) — confirmed
- SHA-256 verified for all 5 Parquets

**Prior failed explicit-basin run (same session):**
`02231000` was passed via `--staids` but is absent from the 2020-11 monthly source chunk.
Builder correctly halted with 0 basins built rather than silently skipping. Not a smoke
failure. Basin replaced by `01019000` for the passing 5-basin run.

**audit_summary.md gap:**
The auditor writes its verdict to stdout (captured in `smoke.log`). It does not write a
standalone `audit_summary.md`. For the full build (Milestone 2K-F-C), the auditor must
write `audit_summary.md` to the product directory before the build is closed.
This is a pre-build implementation requirement, not a blocker for closing 2K-F-B.

**Implementation decisions binding for 2K-F-C (full build):**
1. Metadata in JSON: `manifest.json`, `dataset_config.json`, `run_provenance.json` (not `.csv`/`.yaml`).
2. Per-basin files: flat `time_series/{STAID}.parquet` (not `{STAID}/{STAID}_hourly_forcings.parquet`).
3. Gap detection by row absence from source Parquet — consistent with `not_in_s3` semantics.
4. RTMA variable aliases: `sh2`/`2sh` → `rtma_2sh_kgkg`; `gust`/`i10fg` → `rtma_gust_ms`.
5. Path safety guard in launcher: `OUT_DIR` must begin with `/data42/omrip/Flash-NH/`.

**Authorization:** Full 2,752-basin build (Milestone 2K-F-C) requires explicit authorization.

## 2026-07-08 Stage 1 — Milestone 2K-G-G Phase A: Feasibility Report Scaffold

**Decision:** Add Phase A scaffolding only for Milestone 2K-G-G (Target Scaling + Gap
Policy + Lead-Time Feasibility Report). No Moriah code inspection was performed in this
session (no network path from this environment to Moriah); no target-scaling, gap-policy,
or lead-time implementation decision was made or finalized.

**Added:**
- `scripts/inspect_neuralhydrology_stage1_mechanics.py` — read-only inspection of the
  installed NeuralHydrology 1.13 package (Moriah). No training, no dataset package
  generation, no Slurm submission, no GPU use. Falls back gracefully (exit 0, clear
  message) when `neuralhydrology` is not importable, as confirmed by a local smoke run.
- `scripts/analyze_stage1_window_feasibility.py` — geometry-only window/sample-loss
  estimator for the `seq_length` (12/24/48/72 h) x lead-time (1/3/6/12 h) grid, with
  optional MRMS/RTMA/either-gap exclusion and target-availability layering if CSVs are
  supplied. Does not import or require NeuralHydrology. Local smoke run over the real
  Stage 1 full period (2020-10-14 to 2025-12-31) confirms `total_hours=45720`, matching
  the known full-period step count exactly.
- `docs/stage1_target_scaling_gap_leadtime_feasibility.md` — Phase A report scaffold:
  purpose, non-goals, inherited binding decisions, exact Moriah commands, expected
  evidence files, and Phase B questions. Explicitly marked
  **"status: Phase A scaffold created; Moriah evidence pending"** — no findings are
  recorded as final.

**Not done in this patch:** no NH 1.13 code inspected on Moriah; no target-scaling,
gap-policy, or lead-time decision made; no package builder, scientific NH config, or
Slurm template modified; no NH package generated; no training run.

## 2026-07-12 Stage 1 — Milestone 2K-G-G Phase A: inspection-environment wording patch

**Decision:** Wording-only clarification of
`docs/stage1_target_scaling_gap_leadtime_feasibility.md` (and the "Current milestone"
block of `docs/FLASHNH_CURRENT_STATE.md`): local inspection of a pinned
NeuralHydrology 1.13 installation is allowed and encouraged for code review, source
navigation, and preliminary interpretation. Final Phase B conclusions must still be
verified against the `flashnh-moriah` environment, because that is the runtime used
for Smoke 0/1 and future Stage 1 training; if local and Moriah installations differ,
Moriah is authoritative. This replaces earlier wording in the feasibility-report
scaffold that implied NH 1.13 code could only be inspected on Moriah.

No script was changed and no new evidence was produced by this patch — the Moriah
command block in the feasibility report is unchanged; `nh13_inspection_summary.json`
already records NH version/path so local vs. Moriah runs can be compared.

## 2026-07-12 Stage 1 — Milestone 2K-G-G Phase B: Moriah NH 1.13.0 inspection evidence

**Fact.** `scripts/inspect_neuralhydrology_stage1_mechanics.py` was run on Moriah, on
a compute node (`glacier-30`) rather than the login node, inside the `flashnh-moriah`
conda environment. Confirmed via package metadata (the package itself has no
`__version__` attribute — `pip show`/`importlib.metadata.version()` is authoritative):
- NeuralHydrology version: `1.13.0`
- Python: `3.11.15`
- Module path: `/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/lib/python3.11/site-packages/neuralhydrology`
- Evidence directory (pulled to local `tmp/`, gitignored, not committed):
  `tmp/nh13_inspection_moriah_20260712T100047Z/`
- Git commit recorded at run time: `28883a6b4c50942a9da3223b3a863a1955444d4c`
  (matches the committed Phase A scaffold commit — confirms the run used the correct,
  already-committed inspection script).

This is the first explicit, recorded confirmation of the Moriah NH version; prior docs'
"NH 1.13" label was inferred from interactive troubleshooting behavior during Smoke 0/1,
not a recorded version string, since `scripts/setup_flashnh_moriah_env.sbatch` installs
NeuralHydrology unversioned (`pip install neuralhydrology ...`).

A separate local inspection (`tmp/nh13_inspection_local_20260712T085939Z/`, NH `1.11.0`,
from Milestone 2K-G-G-L) also exists but is version-mismatched relative to Moriah's
`1.13.0` and was treated as non-authoritative background only, per the
"Inspection environment policy" in `docs/stage1_target_scaling_gap_leadtime_feasibility.md`.

**Decision/outcome.** The Moriah 1.13.0 evidence (`nh13_inspection_summary.{md,json}`,
`source_hits.txt`) was analyzed and written up as Phase B findings in
`docs/stage1_target_scaling_gap_leadtime_feasibility.md` ("Findings — Phase B"). 7 of 9
NH-mechanics questions (Q1-Q9) are answered or substantially answered: scaler
fit/save location and train-only fitting (confirms NH's existing `is_train`/passed-
`scaler` contract already satisfies the §8d leakage-prevention rule with no custom code);
area-normalized/specific-discharge feasibility (not a `GenericDataset` config flag —
`_normalize_discharge` is `lamah.py`-subclass-specific; recommend precomputing at
package-build time instead); target-transform reversibility (NH's native unscale only
undoes the z-score, via inline `tester.py` arithmetic — no `inverse_transform` API
exists in the codebase; an extra Flash-NH-side area-multiply step would be needed to
get back to raw m^3/s if area-normalization is used); dynamic-input NaN handling
(`nan_handling_method` is real, with `masked_mean` and `attention` modes confirmed in
`modelzoo/inputlayer.py`); and lead-time implementation (no native `lead_time`/`horizon`
config exists; NH's native hindcast/forecast architecture requires forecast-known future
inputs Flash-NH's historical-only `v001-core` does not have; recommend package-build-time
target shifting instead). 2 items are marked `REQUIRES TARGETED SOURCE INSPECTION` with
specific follow-up commands: the exact `tester.py` inverse-scaling arithmetic (lines
~249-260 were not captured by the keyword search), and the target-NaN masking mechanism
in `training/loss.py` (not hit by any of the 15 searched keywords). Q10/Q11
(window/sample-loss numbers against a real gap inventory) remain `PENDING` — that Moriah
run was not executed this round. No target-scaling, gap-policy, or lead-time
implementation decision was made final; no package builder, scientific NH config, or
Slurm template was modified; no training was run.

## 2026-07-12 Stage 1 — Milestone 2K-G-G Phase B follow-up: real window-loss numbers,
## leakage refinement, gap-policy framework; 3 items still blocked on Moriah access

**Fact — Moriah access blocker.** This follow-up round attempted to close the 3
remaining `REQUIRES TARGETED SOURCE INSPECTION` items from the prior Phase B entry
(exact `tester.py` inverse-scaling arithmetic, `training/loss.py` target-NaN masking,
default `nan_handling_method` behavior) using the exact `sed`/`grep` commands already
recorded in `docs/stage1_target_scaling_gap_leadtime_feasibility.md`. All three remain
unresolved: this working session has no SSH/network path to Moriah
(`ssh -o ConnectTimeout=5 -o BatchMode=yes moriah "echo CONNECTIVITY_OK"` fails with
"Could not resolve hostname moriah"). These are lightweight source-file inspections
requiring no NeuralHydrology import or compute allocation — the next concrete step is
unchanged: the user runs the 3 listed commands on Moriah directly and shares output, or
a future session with Moriah SSH access runs them.

**Fact — Q10/Q11 closed without a new Moriah run.**
`scripts/analyze_stage1_window_feasibility.py` imports no NeuralHydrology, so it was run
locally against the real gap-inventory evidence already pulled from an earlier milestone
(`tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/fullperiod_gap_inventory.csv`,
2026-06-24 full-period forcing postrun audit). No new Moriah round-trip was needed.

**Fact — real bug found and fixed in `scripts/analyze_stage1_window_feasibility.py`.**
The real gap-inventory CSV's timestamps are ISO-8601 with a `Z` suffix (tz-aware after
`pandas.to_datetime`), while the script's internal hourly period index is tz-naive;
comparing tz-aware gap-hour timestamps against the tz-naive index via `.isin()` silently
matched zero rows (no error). The first real-gap run
(`tmp/stage1_window_feasibility_real_gaps_20260712T103728Z/`, superseded, stale, not
cited as evidence) consequently reported 0% MRMS/RTMA gap-loss at every one of the 16
`seq_length` x `lead_time` combinations — implausible, since the known 136 MRMS + 2 RTMA
archive-gap hours are scattered across the full 2020-2025 period. Root-caused via a
scratch debug script isolating the internal boolean gap masks (all-`False` despite
correct gap-hour counts in the loader's own metadata). Neither prior Phase A synthetic
smoke-test CSV had a `Z` suffix, so this bug was invisible to earlier regression tests.
Fixed by adding a `_to_naive_utc()` helper
(`pd.to_datetime(series, utc=True).dt.tz_localize(None)`) applied at all 3
timestamp-parsing call sites. Verified via `python -m py_compile` (clean) and full
regression re-runs of both pre-existing synthetic fixtures (byte-identical output to
before the fix — no regression). The real-gap run was then redone with the fix applied
(`tmp/stage1_window_feasibility_real_gaps_20260712T104021Z/`), console-confirmed correct
gap detection (`mrms: 136, rtma: 2`, matching known archive-gap counts).

**Decision/outcome — Q10/Q11 answered.** Corrected either-gap window-loss fraction
ranges from ~1.3% at `seq_length=12` to ~5.6% at `seq_length=72` across the full
12/24/48/72 h x 1/3/6/12 h lead-time design space; MRMS-gap loss dominates RTMA-gap loss
by roughly two orders of magnitude at every combination (e.g. 5.44% vs. 0.16% at
`seq_length=72, lead_time=12`), tracking the known 136-vs-2 archive-gap-hour asymmetry.
No combination shows a surprisingly large loss fraction (all remain under 6%), but loss
is clearly non-negligible and grows monotonically with `seq_length` — full table recorded
in `docs/stage1_target_scaling_gap_leadtime_feasibility.md` under "Window/sample
feasibility."

**Decision/outcome — Q4 leakage finding refined (not reversed).** The original Q4
answer ("NH enforces train-dataset-only scaling, satisfying §8d with no custom code") is
now explicitly scoped: this is true for the mechanism NH's `is_train`/passed-`scaler`
contract actually protects — **temporal** leakage (fitting a scaler using
validation/test-period timesteps). It is **not** true that NH protects **spatial**
leakage (California/spatial-holdout basins) — NH has no concept of basin role; if
Flash-NH's training-basin list accidentally includes a spatial-holdout or California
basin, NH computes the scaler over it without complaint. Spatial-leakage prevention
remains entirely a Flash-NH basin-list-construction responsibility, upstream of and
unverified by this evidence.

**Decision/outcome — gap-policy decision framework added, not decided.** Compared
Policy A (leave gap hours as true NaN + explicit `nan_handling_method`: uses existing NH
machinery, no window dropped, but requires deliberate config and does not achieve clean
per-sample exclusion) against Policy B (hard-exclude gap-intersecting windows via
custom package-builder/sampler logic: scientifically cleaner for MRMS, but not native in
NH per the Q8 evidence, and costs `seq_length`-dependent sample loss per the numbers
above). Flagged that RTMA (2 gap hours) may warrant a separate, simpler policy from MRMS
(136 gap hours) given the ~100x loss asymmetry. Neither the Policy A/B choice nor the
RTMA-vs-MRMS-split question was decided in this round, per explicit instruction.

**Not done.** No final target-scaling, gap-policy, or lead-time implementation decision
was made. No package builder, scientific NH config, or Slurm template was modified. No
training was run; no NH package was generated. Nothing under `tmp/` was committed,
including both `window_feasibility_real_gaps_*` run directories produced this round (the
first stale/superseded, the second corrected/authoritative).

## 2026-07-12 Stage 1 — Milestone 2K-G-G Phase B follow-up part 2: Moriah SSH access
## restored, all 3 remaining NH-mechanics items closed

**Fact — Moriah SSH access restored.** The user fixed Moriah SSH connectivity from
local Windows/VS Code. Confirmed via `ssh moriah "hostname"` -> `moriah-gw-01`. Note:
default modern `scp`/SFTP-subsystem transfer is still unavailable on Moriah (legacy
`scp -O` required for any future file pull); plain SSH command execution works normally
and was sufficient for this round, since only inline `sed`/`grep` output over an SSH
session was needed (no file transfer).

**Commands run** (all against
`/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah/lib/python3.11/site-packages/neuralhydrology`
on the Moriah login node — lightweight source-file reads only, no compute allocation, as
instructed):
1. `ssh moriah "sed -n '240,270p' .../evaluation/tester.py"`
2. `ssh moriah "grep -n 'isnan\|nanmean\|nan_to_num\|mask\|ground_truth\[.y.\]\|prediction\[.y_hat.\]' .../training/loss.py"`
   plus `grep -n '^class' .../training/loss.py` and targeted `sed` reads of the matched
   class bodies.
3. `ssh moriah "sed -n '60,110p' .../modelzoo/inputlayer.py"`,
   `ssh moriah "sed -n '210,305p' .../modelzoo/inputlayer.py"`,
   `ssh moriah "grep -n 'nan_handling_method\|masked_mean\|attention' .../modelzoo/inputlayer.py"`,
   and `ssh moriah "grep -n 'nan_handling_method' .../utils/config.py"` plus a targeted
   `sed` read of the matched config property.

Raw output saved to `tmp/nh13_targeted_inspection_moriah_20260712T120839Z/` (3 files:
`tester_240_270.txt`, `loss_py_nan_grep.txt`, `inputlayer_nan_handling.txt`; gitignored,
not committed).

**Decision/outcome — Q2 (tester.py inverse-scaling arithmetic) confirmed.**
`tester.py:247-259`: both predictions and observations are unscaled with
`raw = scaled * feature_scale + feature_center`, the direct algebraic inverse of the
forward z-score at `basedataset.py:758`. No separate formula for predictions vs.
observations. No public `inverse_transform` API exists anywhere in the installed 1.13.0
source (confirmed zero hits in the earlier Phase B pass, now confirmed by direct
inspection of the only call site) — this is inline arithmetic private to `Tester`. The
existing caveat that this reversal only undoes the z-score (not an area-normalization
step, if Flash-NH adopts specific discharge) is unchanged and now rests on confirmed
rather than inferred arithmetic.

**Decision/outcome — Q5 (training/loss.py target-NaN masking) confirmed.**
`training/loss.py` defines `BaseLoss` and 6 concrete subclasses — `MaskedMSELoss`,
`MaskedRMSELoss`, `MaskedNSELoss`, `MaskedGMMLoss`, `MaskedCMALLoss`, `MaskedUMALLoss` —
every one of which masks target NaNs per-element inside its own `_get_loss` via
`~torch.isnan(ground_truth['y'])`-style boolean masking before the loss reduction
(exact element-wise masking, not `torch.nanmean`). This confirms `BaseDataset
._validate_samples`'s earlier-observed lack of a target-NaN sample-exclusion criterion is
intentional NH design (include the sample, mask the NaN element at loss time), not an
evidence gap. A target NaN cannot silently contaminate training through any NH-provided
loss class.

**Decision/outcome — Q6/Q7 (default `nan_handling_method` behavior) confirmed, and the
default is dangerous.** `Config.nan_handling_method` (`utils/config.py:610-613`) returns
`None` when the config key is absent. When `None`, both the `__init__` sizing branch and
the `forward()` dispatch branch in `modelzoo/inputlayer.py` fall through every
`if`/`elif` to a final unconditional `else` that performs **no NaN handling at all** —
raw (possibly NaN) dynamic-input tensors are concatenated and passed directly into an
unprotected `nn.Linear`-based embedding layer, with no masking, zeroing, or flagging.
This confirms the unset default is not a safe silent fallback: a NaN dynamic input would
propagate through the embedding and corrupt downstream activations/gradients. A third
mode, `input_replacing` (not found in the first Phase B pass), was also confirmed,
alongside the two already-known modes `masked_mean` and `attention` — 3 total real
modes, all requiring explicit configuration. Because of this finding, the gap-policy
decision framework's Policy A (leave gap hours as NaN + `nan_handling_method`) was
revised in `docs/stage1_target_scaling_gap_leadtime_feasibility.md` to state explicit
`nan_handling_method` configuration is mandatory, not merely recommended, for that policy
to be safe.

**Outcome — zero remaining `REQUIRES TARGETED SOURCE INSPECTION` items.** All Phase B
NH-mechanics questions (Q1-Q9) and window-feasibility questions (Q10-Q11) for Milestone
2K-G-G are now answered from authoritative Moriah NH 1.13.0 evidence.

**Not done.** No final target-scaling, gap-policy, or lead-time implementation decision
was made — §5/§6/§9b of `docs/stage1_scientific_baseline_design.md` remain open; this
round closed evidence gaps only. No package builder, scientific NH config, or Slurm
template was modified. No training was run; no NH package was generated. Nothing under
`tmp/` was committed, including the new `nh13_targeted_inspection_moriah_20260712T120839Z/`

## 2026-07-12 Stage 1 — Milestone 2K-G-H: Scientific Baseline Policy Sign-off

**Context.** The 2K-G-G Phase B evidence-gathering above (window-feasibility numbers,
timezone-bug fix, leakage refinement, gap-policy framework, and the three targeted NH
1.13.0 mechanics inspections) was committed at `0d0e6aa`. This entry records a
docs-only follow-up patch that converts that evidence into binding Stage 1 decisions —
no new evidence gathered, no code/config/Slurm/training files touched, no NH package
generated, no training run.

**Decision/outcome — seven policy items approved, recorded in
`docs/stage1_scientific_baseline_design.md`'s new "Binding decisions — Milestone
2K-G-H sign-off" section, with the corresponding numbered sections (§3, §5, §5a, §6,
§8b, §9b) rewritten from `STILL OPEN`/`GATED`/`REOPENED` to `APPROVED`:**

1. **Target scaling (§5).** Area-normalized discharge, internal unit mm/h equivalent
   runoff depth, computed by the Flash-NH package builder at package-build time (before
   NH sees the data); package target column named e.g. `qobs_mm_per_h_leadXX`, not raw
   `qobs_m3s`. NH's native scaler inversion (`tester.py:247-259`) only returns to mm/h;
   official evaluation requires an additional Flash-NH-side mm/h→`m^3/s` conversion
   using basin area. Binding evaluation policy: NH loss/validation curves are training
   diagnostics in transformed space unless separately proven otherwise; official
   benchmark metrics are always Flash-NH-computed in raw `m^3/s` after full inverse
   conversion.
2. **Target inversion/audit requirements (§5a, new section).** Three future
   implementation checklist items: deterministic `m^3/s -> mm/h -> m^3/s` round-trip
   unit tests using basin area; a package-audit requirement that `qobs_mm_per_h_leadXX`
   at `t` equals original `qobs_m3s` at `t+XXh` converted to mm/h on random
   basin/time samples; an evaluation-audit requirement that raw-space metric scripts
   verify units, basin area, target lead alignment, NaN masking, and the conversion
   back to `m^3/s`.
3. **Lead-time implementation (§9b).** Package-build-time target shifting confirmed as
   the only viable mechanism (no native NH `lead_time` config; native hindcast/forecast
   architecture requires forecast-known future inputs Flash-NH's purely-historical
   inputs don't have — Q9 evidence). All four lead times (1/3/6/12 h) are included in
   the first package/config/sweep design, not just 6 h/12 h — 1 h/3 h remain
   diagnostics, not the primary benchmark, but are built now because their incremental
   package/config cost is small relative to reproducing the work later. Primary
   benchmark lead 6 h, secondary 12 h, unchanged from 2026-07-06. `seq_length` and lead
   time remain separate design axes.
4. **Forcing-gap policy (§6).** Scientific baseline hard-excludes training windows
   intersecting MRMS archive-gap hours (Policy B from the feasibility doc's decision
   framework), accepted because corrected 2K-G-G real-gap window loss is modest
   (~1.3% at `seq_length=12` to ~5.6% at `seq_length=72`) and hard exclusion is
   scientifically cleaner than silent fill. RTMA (2 archive-gap hours vs. MRMS's 136)
   is worded separately: excluding RTMA-gap-intersecting windows too is acceptable if
   the implementation naturally supports a combined "either-gap" mask, but MRMS drives
   the policy either way. Silent dynamic-input NaNs are not allowed.
   `nan_handling_method` (Policy A) remains a fallback/ablation path only, not the
   baseline; if used in any ablation it must be explicitly configured (`masked_mean`,
   `attention`, or `input_replacing`) — unset/default `None` remains forbidden for
   NaN-valued dynamic inputs in any run, per the Q6 evidence already on record. Smoke
   0/1's technical fill policy remains historical/technical only, not the scientific
   baseline.
5. **Static attributes (§3).** Canonical `stage1_static_attributes_v001` (2,843 basins
   × 531 columns, 496 `model_input`; h2o canonical build/audit PASS 2026-07-08, sha256
   `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`) accepted as the
   Stage 1 v001-core static attribute matrix, replacing the earlier 48-column GAGES-II
   screening merge as the modeling matrix (that merge remains a valid provenance
   artifact, just no longer the matrix training uses). Numeric static attributes pass
   through the model's standard static-attribute pathway/embedding layer, assuming NH
   config support. No raw categorical embeddings in this first baseline —
   categorical/admin/geographic leakage-prone fields remain excluded/deferred per the
   2K-G-F/2K-G-F-B conservative column-classification policy. `STATE`/`HUC02` remain
   split-support/diagnostics only, not model inputs. `LAT_GAGE`/`LNG_GAGE` remain
   diagnostic only, deferred to a later lat/lon ablation.
6. **Spatial split and leakage (§8b).** Reproducible seeded stratified non-CA spatial
   holdout (mechanism unchanged from the 2026-07-06 ~10% rule); California remains
   excluded entirely from Stages 1–3, reserved for Stage 4 (§8c, unchanged). Spatial
   leakage prevention is a Flash-NH basin-list responsibility, not something NH enforces
   (Q4 evidence: NH's `is_train`/scaler contract protects temporal leakage automatically
   but has zero concept of basin role). Future implementation must produce explicit
   basin-list artifacts for development training, validation, temporal test, non-CA
   spatial holdout, and California Stage 4 fine-tune/holdout. Stratification must
   consider at minimum HUC02/geography, basin area, and hydroclimatic attributes such as
   aridity/climate, if available from `stage1_static_attributes_v001`.
7. **Next implementation milestone defined, not executed.** `2K-G-I — Baseline Package
   Builder + Split Config Implementation`, added as a new mini-milestone in
   `docs/stage1_scientific_baseline_design.md` with an 8-item scope checklist (target
   conversion, lead-time shifting, raw-`m^3/s` reconstruction/audit, hard MRMS-gap
   exclusion mechanism, basin-list artifact generation, `stage1_static_attributes_v001`
   adoption, baseline NH YAML/config generation, package audit updates). Checklist only
   — no code written, no milestone started, in this patch.

**Also updated (cross-references, not new decisions):**
`docs/stage1_target_scaling_gap_leadtime_feasibility.md`'s "Target-scaling decision" and
"Gap-policy decision framework" sections, and its "Status"/"Not done" sections, now point
to the binding sign-off in `docs/stage1_scientific_baseline_design.md` instead of stating
`PENDING`/"not a decision" — the evidence content in that document is unchanged, only the
decision-status framing was updated. `docs/FLASHNH_CURRENT_STATE.md`'s "Current milestone"
section was updated with a 2K-G-H summary above the existing 2K-G-G entry (kept for
history, not removed).

**Self-check during this patch.** Grepped the changed docs for any code comment or script
docstring directly contradicting the new binding policy (specifically: stale claims that
target scaling/gap policy/lead time remain undecided) — none found; the only
policy-status language lived in the four docs updated here, no `.py`/`.yaml`/`.sbatch`
file contains a stale "still open" claim.

**Not done.** No new evidence was gathered — this patch converts already-committed 2K-G-G
evidence into policy, it does not re-derive it. No package builder, NH config, or Slurm
template was modified — implementation is scoped to the new, not-yet-started 2K-G-I
milestone. No training was run; no NH package was generated; nothing under `tmp/` was
touched or committed by this patch.

## 2026-07-13 Stage 1 — Milestone 2K-G-I I-A2: spatial split generator, method finalized

**Context.** Implements the split-generator sub-milestone scoped in the 2026-07-12 entry
above (§8b basin-list artifacts). Two points needed a concrete decision before code could
be written: how to handle the 5 non-CA basins with no `ari_ix_uav` value, and how to
simplify the multi-level sparse-stratum fallback ladder drafted in
`docs/stage1_baseline_package_implementation_plan.md` §7 into something a small,
testable function set could implement without an intermediate HUC02 × area layer or a
global largest-remainder top-up pass.

**Decisions (signed off by the user, same day):**
1. **Missing-hydroclimate-stratifier basins — Option B.** All 5 basins missing
   `ari_ix_uav` stay in the 2,752-basin universe and are assigned directly to their
   population's training role (`development_train` for non-CA; the same rule applies to
   California generically, though no CA basin is missing aridity in v001) —
   `assignment_reason = missing_hydroatlas_stratifier`. They are excluded from
   stratification/sampling entirely, never enter a holdout role, and there is no
   `aridity_missing` stratum and no imputation.
2. **Sparse-stratum fallback simplified to one level.** Non-CA: stratum = HUC02 × area
   tercile × aridity tercile; strata with ≥ 10 basins are sampled directly; strata below
   that pool once into a single HUC02-level sparse pool (sampled if the pool reaches 10,
   otherwise sent to `development_train` in full). A sufficient stratum is never
   downgraded because a sibling stratum in the same HUC02 is sparse. California: the same
   shape, statewide (no HUC02 grouping key, HUC02 diagnostic only), one sparse pool.
3. **No exact global holdout count.** The 8–12% overall band
   (`nonca_holdout_fraction ± holdout_tolerance`) remains binding; the exact resulting
   basin count (e.g. 255 vs 256 non-CA, 19 vs 20 CA) is explicitly not material and is not
   pinned as a policy constant — no largest-remainder optimization is used.
4. Seed 42, tercile binning, and minimum composite stratum size 10 are unchanged from the
   2026-07-12 sign-off.

**Encoding.** `config/stage1_scientific_baseline_v001.yaml::spatial_split` gained
`missing_hydroclimate_policy`, `fallback_policy`, `california_fallback_policy`,
`exact_holdout_count_binding: false`, and `largest_remainder_optimization_used: false`;
`src/baseline/policy.py`'s validator pins the new fields; `tests/test_policy.py` gained
matching mutation tests. `docs/stage1_baseline_package_implementation_plan.md` §7 and
`docs/stage1_scientific_baseline_design.md` §8b were updated to describe the simplified
ladder and Option B instead of the earlier multi-level/largest-remainder draft text.

**Implementation.** `src/baseline/splits.py` (small, testable functions — no class
hierarchy) + `scripts/generate_stage1_baseline_splits.py` (candidate-generation CLI,
fail-fast on checksum/count/join/field/role violations) + `tests/test_splits.py`
(synthetic-fixture unit and integration tests). A candidate split was generated under
`tmp/` for machine/human review; nothing was promoted into
`config/stage1_baseline_splits_v001/` — promotion is gated on the I-A3 independent
auditor and I-A4 human QC, not yet started.

**Not done.** No independent auditor, no maps/QC figures, no promotion, no commit of this
patch's own changes (left for user review). `reports/` was not touched.

## 2026-07-16 Stage 1 — Milestone 2K-G-I I-A3/I-A4: split auditor + visual QC, both PASS

**I-A3 (independent auditor).** `src/baseline/split_audit.py` +
`scripts/audit_stage1_baseline_splits.py` + `tests/test_split_audit.py` (32 tests)
reimplement population reconstruction, tercile fitting, stratum/pool routing,
counts/fractions/HUC02 summaries, and manifest/checksum reconciliation independently of
`build_split_assignment` — it reuses only `normalize_staid`. Run against the real I-A2
candidate (`tmp/stage1_baseline_splits_v001_candidate`) and its repeat directory:
**PASS, 0 errors, 0 warnings, 146 OK checks**, byte-identical repeat confirmed. Committed
separately (I-A3 commit).

**I-A4 (human visual QC).** `scripts/generate_stage1_baseline_split_qc.py` renders four
plots from the same candidate (non-CA CONUS overview, California overview,
non-CA drainage-area ECDF, non-CA aridity ECDF) plus a `visual_qc_summary.md`. Human
review verdict: **PASS.** The non-California spatial holdout is broadly distributed
across the major CONUS basin clusters; the California holdout has reasonable
north/central/south representation for a 19-basin sample; development and holdout
drainage-area ECDFs broadly overlap; development and holdout aridity ECDFs are nearly
coincident; the five missing-aridity basins were omitted only from the aridity plot, not
imputed, and all remain in development training. No visible difference was judged severe
enough to invalidate the split. Generated plots remain under
`tmp/stage1_baseline_splits_v001_qc/` (gitignored, not committed).

**Decision.** With I-A3 and I-A4 both PASS, the split candidate is accepted for
canonical promotion. I-A5 (byte-copy promotion to `config/stage1_baseline_splits_v001/`)
is the next and final split sub-milestone; no further auditor hardening, visual QC,
statistical tests, or balance optimization is planned for this split design.

**Not done.** No split regeneration, no seed/policy change, no promotion yet, no NH
package build, no training, no h2o/Moriah commands, no push. `reports/` was not touched.

## 2026-07-16 Stage 1 — Milestone 2K-G-I I-A5: canonical split promotion, COMPLETE

**Action.** Byte-copied (via `shutil.copy2`, no regeneration/reordering/manual edits)
the accepted I-A2 candidate's 10 artifacts from `tmp/stage1_baseline_splits_v001_candidate/`
to the canonical path **`config/stage1_baseline_splits_v001/`**. Source basis: I-A3
independent audit PASS (0 errors) and I-A4 human visual QC PASS (see prior entry).

**Verification.** Pre-promotion: candidate re-audited PASS (0 errors) and reconfirmed
byte-identical to its repeat directory. Post-promotion: exact 10-file inventory matched;
every candidate/canonical file pair SHA-256-identical; the committed I-A3 auditor
(`scripts/audit_stage1_baseline_splits.py`) re-run with `--candidate-dir
config/stage1_baseline_splits_v001` and the unchanged repeat directory as repeat
evidence returned **PASS, 0 errors**, with unchanged role counts
(`development_train`/`validation`/`temporal_test` 2307, `spatial_holdout_nonca` 250,
`california_all` 195, `california_finetune_train` 176, `california_holdout` 19) and
unchanged holdout fractions (non-CA 0.09777, CA 0.09744); the five nonstandard
15-digit missing-aridity STAIDs were confirmed present unchanged in the canonical copy.

**Decision.** The Stage 1 baseline split design is now frozen for the first Stage 1
baseline. Do not reopen it absent a concrete scientific or correctness problem. Next
work is the baseline NH package-builder implementation.

**Not done.** No split regeneration, no seed/policy change, no NH package build, no
training, no h2o/Moriah commands, no push. `reports/` was not touched; only
`config/stage1_baseline_splits_v001/` and this documentation were changed.
evidence directory.

## 2026-07-20 Compact Scientific Package selection — ACCEPTED

**Context.** The fully enriched h2o run of
`scripts/generate_stage1_compact_package_selection.py` (selector commits
`71467b5`, `65af017`; see `docs/stage1_compact_package_selection.md`'s "Exact
h2o command" for the invocation) was executed by the user against the
canonical enrichment inputs: `config/stage1_baseline_splits_v001/split_assignment.csv`
(`development_train`, 2,307 basins), the canonical `stage1_static_attributes_v001`
matrix + column-role manifest, and the canonical full-period qobs/target-status
table. This closes the "two selection runs, not one" gap that document left
open (only the local split-based candidate, with enrichment columns
`not_evaluated`, had been run as of 2026-07-19).

**Acceptance checks (all PASS, as reported by the user from the h2o run).**
Count = 32. Development-pool membership PASS (all 32 confirmed
`development_train`). California exclusion PASS (`STATE != "CA"` for all 32).
Spatial-holdout leakage PASS (no overlap with `spatial_holdout_nonca`,
`validation`, `temporal_test`, or California roles). qobs enrichment and
static missingness evaluated (not `not_evaluated`) for all 32 selected
basins. Input and output artifact checksums PASS.

**Accepted characteristics.** 13 distinct HUC02s; 7 distinct macro-regions
(of the 8 defined in `config/stage1_compact_package_selection_v001.yaml`);
east/west macro-region-side split 19/13 (the hard `require_east_west_spread`
check therefore PASS by a wide margin). Area classes (canonical terciles)
high/low/middle = 12/10/10. Hydro classes (canonical aridity terciles)
high/low/middle/missing = 10/11/10/1. qobs completeness bins high/mid/low =
15/16/1. Static missingness bins none/high = 31/1 (only one basin has any
missing `model_input` static attribute).

**Designated diagnostic basins.**
- `393109104464500` — the selection's one compound edge case: satisfies all
  three reserved categories simultaneously (`unusual_identifier`, 15-char
  STAID; `hydro_stratifier_gap`, missing the aridity stratifier;
  `static_missing_value_case`), with 169 missing `model_input` static
  attributes. This is the one `static_missing_bin = high` basin above, and
  is the designated real-data stress case for the new static-imputation
  primitives (`src/baseline/static_preparation.py`, see the entry below).
- `05568800` — lowest qobs completeness in the selection, coverage fraction
  ≈0.8746 (the selection's one `qobs_completeness_bin = low` basin).

**Artifact status — two-tier, by design.** The generated evidence bundle
(canonical h2o path:
`/data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001_evidence`,
containing `compact_basin_selection.csv`, `compact_basin_ids.txt`,
`selection_summary.md/.json`, `selection_manifest.json`, `run_command.txt`)
correctly still reports `selection_manifest.json`'s `"status"` field as
`"candidate"` — this is the tool's own generated-artifact status, and per
project policy generated evidence is never hand-edited to change it.
**Project-level acceptance is recorded here and in
`docs/FLASHNH_CURRENT_STATE.md` instead** — those two documents are now the
authoritative record that this specific 32-basin selection (identified by
its artifact checksums) is the accepted Compact Scientific Package, distinct
from the generated artifact's own `candidate` self-description. The full
32-basin ID list is intentionally not pasted into this or any other
document — see `compact_basin_ids.txt` in the evidence bundle above.

**Not done.** No NH package built for the 32 basins, no `FlashNHDataset`
changes, no training, no commit/push of this acceptance beyond
documentation. `tmp/` evidence remains untracked.

## 2026-07-20 Scientific target-transformation + static-preparation primitives (Milestone 2K-G-I primitives increment)

**Context.** With the Compact Scientific Package accepted (previous entry),
the next step toward a package builder is a set of independently testable
scientific primitives: discharge-unit transforms, lead-target construction,
development-only static-attribute imputation, and a forcing-gap-timestamp
loading interface. This patch implements only those primitives — explicitly
**not** the package builder itself, not `FlashNHDataset`, not training, not
the full 2,752-basin package, and does not use Moriah.

**Reuse-first inspection (done before writing any code).** Read
`docs/stage1_scientific_baseline_design.md` in full,
`docs/stage1_baseline_package_implementation_plan.md`'s static-attribute
NaN-policy section (§15/§16 area), `config/stage1_scientific_baseline_v001.yaml`,
`src/baseline/nh_dataset.py`, `src/baseline/splits.py`,
`src/baseline/compact_selection.py`, and the existing test suites, before
deciding what (if anything) was missing.

- **`src/baseline/units.py` and `src/baseline/lead_targets.py` already fully
  satisfy the discharge-transform and lead-shift requirements** — exact
  `q_m3s <-> mm/h` conversion with strict area validation, NaN preservation,
  and `build_lead_target`/`build_lead_targets` for leads 1/3/6/12 h with
  convert-then-shift semantics, hourly-index validation, and terminal-boundary
  NaN — all already tested in `tests/test_units.py`/`tests/test_lead_targets.py`,
  including the negative-discharge arithmetic-conversion case
  (`test_negative_discharge_converts_arithmetically`, consistent with
  `docs/stage1_target_policy.md`'s cleaning-happens-upstream policy). **No new
  code was added for this part** — this entry exists to record that the
  reuse check was done, not to introduce a change.
- **`src/baseline/validity_mask.py` already implements the binding
  history/boundary validity split** (§6/§9a-9b) and needed no changes.

**New: `src/baseline/static_preparation.py`.** Implements development-train-only
median imputation for `model_input` static-attribute columns, per the
already-signed-off policy (`config/stage1_scientific_baseline_v001.yaml::static_attributes.imputation`,
signed off 2026-07-13 per `docs/stage1_baseline_package_implementation_plan.md`
§15): `strategy: median`, `fit_basin_scope: development_training_only`,
frozen fitted values applied unchanged to validation/temporal-test/spatial-holdout/
compact-selection basins, hard failure if any `model_input` column is all-NaN
over the development-training population, no missingness-indicator columns
added as model inputs. Reuses `splits.load_matrix_for_splits`,
`splits.join_eligible_with_matrix`, and `staid.normalize_staid` rather than
reimplementing matrix loading or basin-ID validation. Writes a machine-readable
imputation manifest (input/manifest checksums, fit population + count,
per-column method/fitted-value, before/after missing counts, columns with no
valid fit values, algorithm/version id) and preserves an imputed-value audit
mask. Tests cover the compound edge-case basin `393109104464500` (169 missing
`model_input` columns) via a synthetic fixture shaped to match its real
missingness profile.

**New: `src/baseline/gap_mask_io.py`.** `src/baseline/nh_dataset.py` already
expects `<data_dir>/masks/gap_timestamps.json` (a flat JSON list of ISO
timestamps), but — confirmed via `docs/stage1_compact_package_selection.md`'s
own note and a repo-wide grep — no script has ever produced that file; only
test fixtures (`tests/_nh_synthetic.py`) write a synthetic one. The real gap
inventory already exists as an uncommitted h2o audit artifact from Milestone
2K-E (`fullperiod_missing_hour_products.csv`, columns
`chunk_label,product,valid_time_utc,reason`; product values
`mrms_qpe_1h_pass1`/`rtma_conus_aws_2p5km`, per
`scripts/generate_fullperiod_audit_tables.py`). This module converts that
inventory into the canonical `gap_timestamps.json` format, validating against
an explicit hourly timeline by reusing
`validity_mask.bad_hour_mask_from_timestamps` — it does **not** decide the
gap policy (already signed off, §6: MRMS drives the policy; RTMA may be
folded into the same exclusion mask) and does not compute gap hours from raw
archive data itself, only reformats/validates an already-produced inventory.
`gap_mask_io.py` is the tested conversion primitive (17/17 tests passing);
the future compact-package builder is expected to call it directly to
produce `<data_dir>/masks/gap_timestamps.json` for each built NH package.
There is intentionally no standalone gap-conversion CLI in this increment —
conversion is a small step inside package construction, not a separate
operator-facing tool, so it is deferred to the builder itself rather than
adding orchestration ahead of a concrete caller.

**Not done.** No NH package built, no `FlashNHDataset`/`nh_register.py`/
`run_stage1_nh.py` changes, no training, no Moriah use, no full 2,752-basin
package, nothing committed or pushed pending review.

## 2026-07-20 Stage 1 — Milestone: static-attribute semantic correction (sentinel decoding + role reclassification), IMPLEMENTED, PENDING H2O REBUILD

**Context.** Before the Compact Scientific Package builder begins consuming
`stage1_static_attributes_v001` (canonical, `docs/decision_log.md` 2K-G-H
entry; 2,843 × 531, 496 `model_input`, sha256
`eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`), a
read-only semantic audit was run over all 496 `model_input` columns (script
preserved at
`C:\Users\omrip\AppData\Local\Temp\claude\...\scratchpad\audit_496.py`,
against the byte-identical local checksum-pinned parquet copy under
`tmp/stage1_baseline_split_inputs_v001/...`, not against a fresh h2o pull).
The audit checked sentinel-literal frequency (`-9999`/`-999`/`-99`/`999`/
`9999`), name-pattern screens (id/geo/leakage/percent/count-like), exact
duplicate-value columns, and `|corr|>=0.999` pairs, cross-referenced against
the GAGES-II variable-description workbook. Findings were bounded to a
specific list of columns; no defect implicated the split assignment, the
32-basin Compact Scientific Package selection, the target/lead-time/
validity-mask code, or the ~485 remaining `model_input` columns.

**Binding decisions (not to be reopened absent a new concrete problem):**
1. Eight infrastructure-distance sentinel columns (`RAW_DIS_NEAREST_DAM`,
   `RAW_AVG_DIS_ALLDAMS`, `RAW_DIS_NEAREST_MAJ_DAM`,
   `RAW_AVG_DIS_ALL_MAJ_DAMS`, `RAW_DIS_NEAREST_CANAL`,
   `RAW_AVG_DIS_ALLCANALS`, `RAW_DIS_NEAREST_MAJ_NPDES`,
   `RAW_AVG_DIS_ALL_MAJ_NPDES`): decode exact `-999`/`-999.0` to NaN before
   missingness calc; no blanket sentinel replacement elsewhere; let the
   existing `>20%` missingness rule exclude all eight; retain provenance; no
   new presence/distance derived features this increment.
2. Direct coordinates (`LAT_GAGE`, `LNG_GAGE`, `LAT_CENT`, `LONG_CENT`):
   retain as diagnostic metadata only (`diagnostic_latlon`), excluded from
   `model_input`; no other geography exclusions added.
3. Gauge-record/network metadata (`FLOWYRS_1900_2009`, `FLOWYRS_1950_2009`,
   `FLOWYRS_1990_2009`, `FLOW_PCT_EST_VALUES`, `BASIN_BOUNDARY_CONFIDENCE`,
   `ACTIVE09`, `HBN36`, `HCDN_2009`, `OLD_HCDN`, `NSIP_SENTINEL`,
   `PCT_DIFF_NWIS`, `NWIS_DRAIN_SQKM`): moved to a new
   `diagnostic_record_network_qa` role, not `model_input`.
   `NWIS_DRAIN_SQKM`/`PCT_DIFF_NWIS` exact `-9999` sentinels still decoded
   for provenance/validation despite the role change. Retained, not deleted.
4. `PERHOR`: decode exact `-9999` to NaN, retain as `model_input`, allow
   later development-only imputation, replacement count recorded.
5. `STRAHLER_MAX`: decode exact `-99` to NaN, retain as `model_input`, allow
   later imputation, replacement count recorded, no derived
   artificial-channel flag this increment.
6. `lka_pc_use`: excluded from first-baseline `model_input`, retained under
   a new `deferred_ambiguous` role, may be reconsidered once HydroATLAS
   catalog semantics are resolved.
7. Retained unchanged, no exclusion: `dor_pc_pva`, `dis_m3_pyr`,
   `run_mm_syr` — a possible future ablation is noted, not acted on.
8. Expected corrected count ≈473 `model_input` columns — explicitly
   provisional, not an acceptance criterion; the h2o rebuild is authoritative.

**Action.** `scripts/build_stage1_static_attribute_matrix.py`: added
`_SENTINEL_VALUES_BY_COLUMN` (per-column exact-match sentinel map covering
the 12 columns above) and `_decode_column_sentinels()`, invoked once per
mapped column immediately before role classification and the missingness
calculation in `_load_and_classify()`; non-numeric values in a mapped column
fail the build loud; per-column replacement counts (including legitimate
zero) are written to `provenance.json` under a new `sentinel_decoding` block
together with the sentinel map and algorithm id
(`stage1_static_sentinel_decode_v1`). Two new roles added to
`_classify_columns()`: `diagnostic_record_network_qa` (12 columns, checked
before the pre-existing binary-flag branch since 5 of the 12 overlap it) and
`deferred_ambiguous` (`lka_pc_use`); `diagnostic_latlon` extended from 2 to 4
columns. The 90%-reliably-numeric gate in the `candidate_model_input`
dispatch branch is bypassed specifically for sentinel-mapped columns (their
numeric-ness is already fail-loud-validated by the decode step, and their
post-decode missingness is expected, not schema drift) — this was required
for the 8 `RAW_*` columns to reach the intended `>20%` high-missingness
exclusion filter in `build()` rather than being rejected earlier by the
coarse gate. Critically, **the 8 `RAW_*` columns are excluded by the
pre-existing missingness mechanism, not by name** — verified directly in
provenance output. `scripts/audit_stage1_static_attribute_matrix.py`:
independently mirrors the same sentinel map and role sets (not imported)
and adds hard-fail checks for coordinates/record-network-QA/`lka_pc_use`/
`RAW_*` leaking into `model_input`, any literal mapped sentinel surviving in
`model_input`, and manifest/matrix column-role consistency, plus positive
checks that `PERHOR`/`STRAHLER_MAX`/`dor_pc_pva`/`dis_m3_pyr`/`run_mm_syr`
remain `model_input`. `src/baseline/static_preparation.py` required **no
changes** — its role-based column selection is plain string equality against
the manifest, so new role names work automatically; confirmed by inspection,
not modified, per the hard constraint against touching that file this
increment. `tests/test_static_attribute_matrix.py` (new, 19 tests): sentinel
decoding, role classification, an end-to-end synthetic-fixture build proving
the exclusion mechanism, and auditor PASS/hard-fail regressions.

**Verification.** `py_compile` clean on both changed scripts. Full local
suite: `python -m pytest tests/ -q --ignore=tests/test_nh_dataset.py
--ignore=tests/test_nh_register.py` → 448 passed (includes the 19 new
tests). A local, checksum-unverified dry-run against the same 29-file source
mirror used for the original v001 canonical build
(`C:\PhD\Python\neuralhydrology\US_data\attributes`, output to
`tmp/stage1_static_attribute_matrix_v002_dryrun/`, gitignored, not
committed) produced: 2,843 rows × 523 columns, **473 `model_input`**
columns; all 8 `RAW_*` columns present in `high_missing_excluded_model_input`
(not in any hand-authored exclusion list); 15,018 total sentinel values
replaced across the 12 mapped columns; independent-auditor result PASS, 0
errors, 0 warnings, 32 OK checks, including all new hard-fail and positive
checks; matrix sha256
`6ff9084008a2e7af8aab0ba46716650c06bb1fa7c92de815c620c0c850c734dd`
(local-mirror dry-run only — **not** a canonical checksum, since this source
mirror has not been re-verified against the h2o mirror this session).

**Decision.** The corrected classification (sentinel decoding + two new
roles + extended `diagnostic_latlon`) is accepted as the design for the
Stage 1 v002 static-attribute matrix. `stage1_static_attributes_v001` and
its checksum remain the historical record of the 2026-07-08 canonical build
and are not overwritten or deleted, but are **superseded for modeling
purposes** — the Compact Scientific Package builder and any future training
must consume the corrected matrix once it is canonically rebuilt. The
existing compact-selector output (32 basins, `71467b5`/`65af017`) and the
canonical split assignment (`config/stage1_baseline_splits_v001/`) are
**unaffected and remain valid** — selection/splitting operate on basin sets,
independent of which static-attribute columns are classified `model_input`.
The compact static-imputation artifact
(`stage1_compact_static_imputation_v001`, built from v001) is likewise
superseded pending the corrected canonical rebuild.

**Corrected canonical rebuild — exact h2o commands (not yet run).**

```bash
# 1. Rebuild the corrected canonical matrix under a new version path
python scripts/build_stage1_static_attribute_matrix.py \
  --source-dir /data42/omrip/Flash-NH/data/static_attributes/gageii_hydroatlas_source_v001 \
  --manifest   config/stage1_initial_training_basin_manifest.csv \
  --out-dir    /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002 \
  --matrix-name stage1_static_attributes_v002 \
  --force

# 2. Run the independent auditor against the new build
python scripts/audit_stage1_static_attribute_matrix.py \
  --matrix-dir /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002 \
  --matrix-name stage1_static_attributes_v002 \
  --manifest   config/stage1_initial_training_basin_manifest.csv

# 3. Create a compact evidence bundle (matrix + manifest + provenance + audit
#    summary) for local inspection, mirroring the v001 evidence convention
mkdir -p /data42/omrip/Flash-NH/tmp/stage1_static_attributes_v002_evidence
cp /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002.parquet \
   /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002_column_manifest.json \
   /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002_provenance.json \
   /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/audit_summary.md \
   /data42/omrip/Flash-NH/tmp/stage1_static_attributes_v002_evidence/
sha256sum /data42/omrip/Flash-NH/tmp/stage1_static_attributes_v002_evidence/stage1_static_attributes_v002.parquet

# 4. Pull the evidence bundle locally for inspection (run from local machine)
scp -r h2o:/data42/omrip/Flash-NH/tmp/stage1_static_attributes_v002_evidence \
  tmp/stage1_static_attributes_v002_evidence

# 5. Only after the v002 canonical matrix is reviewed and accepted: rerun the
#    compact static-imputation step against it (same 32 compact basins,
#    same split assignment, new source matrix, new out-dir — do not
#    overwrite the v001 imputation artifact)
python scripts/prepare_stage1_compact_static_attributes.py \
  --attributes-parquet /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002.parquet \
  --column-manifest    /data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/stage1_static_attributes_v002_column_manifest.json \
  --split-assignment   config/stage1_baseline_splits_v001/split_assignment.csv \
  --target-basins       /data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001_evidence/compact_basin_ids.txt \
  --out-dir             /data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002/ \
  --force

# 6. Pull the replacement compact-imputation evidence locally
scp -r h2o:/data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002 \
  tmp/stage1_compact_static_imputation_v002
```

**Not done.** No h2o or Moriah connection made this increment; no canonical
artifact generated (only a local `tmp/`-scoped dry-run against an
unverified source mirror); compact selector not rerun; canonical split
artifacts not modified; `src/baseline/compact_selection.py`,
`src/baseline/static_preparation.py`, target-conversion/lead-target/
validity-mask code not modified; no NH package built; no training run;
nothing committed or pushed pending review.

**Status update (2026-07-20, later same day): this "Not done" list is
superseded — the h2o commands above were run and their results accepted.
See the acceptance entry immediately below.**

## 2026-07-20 Stage 1 — Milestone: static-attribute matrix v002 + compact static-imputation v002 ACCEPTED

**Context.** The corrected canonical static-attribute matrix and replacement
compact static-imputation artifact, designed and implemented in the entry
immediately above, were built on h2o using the exact commands recorded
there, pulled locally, and reviewed. Both are **accepted**.

**Canonical matrix `stage1_static_attributes_v002`.** Source-checksum
verification: 29/29 files PASS. Canonical path:
`/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/`
(`stage1_static_attributes_v002.parquet`,
`stage1_static_attributes_v002_column_manifest.json`,
`stage1_static_attributes_v002_provenance.json`,
`stage1_static_attributes_v002_audit_summary.md`). Matrix shape: 2,843 rows
× 523 total columns. Column-role breakdown: **473 `model_input`**
(authoritative — the local dry-run's 473 estimate from the prior entry is
confirmed exactly, no longer provisional), 2 split-support, 4 diagnostic
lat/lon, 12 diagnostic record/network/QA, 1 deferred-ambiguous
(`lka_pc_use`), 29 categorical-deferred, 2 flag. Sentinel algorithm
`stage1_static_sentinel_decode_v1`, 15,018 total sentinel values decoded.
The 8 infrastructure-distance `RAW_*` columns excluded through the existing
`>20%` missingness mechanism (not by name); `PERHOR`/`STRAHLER_MAX` retained
`model_input` with sentinels decoded; `dor_pc_pva`/`dis_m3_pyr`/`run_mm_syr`
retained unchanged; direct-coordinate, record/network/QA, and `lka_pc_use`
exclusions all verified; HydroATLAS 5-basin gap unchanged and explicitly
handled. Independent audit
(`scripts/audit_stage1_static_attribute_matrix.py`): **PASS, 0 errors, 0
warnings, 32 OK checks.**

```
matrix (stage1_static_attributes_v002.parquet):
4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297

column manifest (stage1_static_attributes_v002_column_manifest.json):
02505eb4893e6848f7cbc4eabd2cdf40dd6aee64156d41744aebcbe4409f0e00

provenance (stage1_static_attributes_v002_provenance.json):
983b9f9ff187c4dfc2e8a6d7453929b31b006ff099d7682b3c1c7b348c55f022

audit summary (stage1_static_attributes_v002_audit_summary.md):
247cae508338cc51d18bc22dfd7d0124b459c5e4c12ebd07e848f66a88211f4a
```

**Compact static-imputation `stage1_compact_static_imputation_v002`.** Built
via `scripts/prepare_stage1_compact_static_attributes.py` (algorithm
`stage1_static_median_imputation_v1`, unchanged primitive) against the
accepted v002 matrix, per command 5 in the prior entry. Canonical generated
output path: `/data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002`.
Input matrix checksum matches the v002 canonical checksum above exactly.
Output shape 32 basins × 473 `model_input` columns; fit scope
development-training-only, fit population 2,307 basins, applied to the 32
accepted compact basins; all fit columns had valid medians; **168 total
values imputed, all on exactly one basin (`393109104464500`** — the same
designated compound-edge-case diagnostic basin from the accepted 32-basin
selection, `docs/FLASHNH_CURRENT_STATE.md`); zero remaining NaNs.

```
imputed_static_attributes.parquet:
3d476c41dda2c95481a76f7a97e288929e317b8ed0798cb4ddaa00bf4615b92e

imputed_value_mask.parquet:
61bbceb2f1643ef9184524f8c9e3c90a666396c9b44272b879c9803fcfa46796
```

**Decision.** `stage1_static_attributes_v002` is accepted as the canonical
Stage 1 baseline static-attribute matrix, superseding
`stage1_static_attributes_v001` for modeling.
`stage1_compact_static_imputation_v002` is accepted as the replacement
compact static-imputation artifact, superseding
`stage1_compact_static_imputation_v001` for modeling. Both v001 artifacts
and their checksums (matrix `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`;
imputation checksums in the 2K-G-I / target-primitives entries above) remain
preserved as historical provenance — **not deleted, not invalidated** — and
are not overwritten.

**Unaffected / not reopened by this acceptance.** The compact selector
(`scripts/generate_stage1_compact_package_selection.py`) and canonical split
artifacts (`config/stage1_baseline_splits_v001/`) were **not rerun**; the
accepted 32-basin Compact Scientific Package selection remains valid as-is
(selection is basin-set logic, independent of static-attribute column
content); the static-imputation primitive
(`src/baseline/static_preparation.py`) is unchanged code.

**Not done.** No NH package has yet been built; no training has run. This
docs-only closure updates `docs/FLASHNH_CURRENT_STATE.md`,
`docs/decision_log.md` (this entry),
`docs/stage1_static_attribute_matrix_plan.md`, and
`docs/stage1_scientific_baseline_design.md` only — no Python, tests,
configs, split artifacts, package builders, or Slurm files were touched; no
h2o/Moriah connection was made in the course of writing this entry; Compact
Scientific Package construction has not started. **Next milestone: Compact
Scientific Package builder planning and implementation.**

---

## 2026-07-20 — Stage 1 — Executable policy reconciled to static matrix v002

**Problem.** The 2026-07-20 acceptance of `stage1_static_attributes_v002`
(entry above) updated narrative documentation, but the *executable*
scientific-baseline policy — `config/stage1_scientific_baseline_v001.yaml`
and its loader/validator `src/baseline/policy.py` — still hard-pinned the
superseded v001 identity/shape/checksum
(`stage1_static_attributes_v001`, 531 total columns, 496 `model_input`,
sha256 `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`).
Any future policy-consuming builder would have failed loudly against the
now-canonical v002 artifact.

**Fix.** Reconciled the `static_attributes` block in
`config/stage1_scientific_baseline_v001.yaml` and the matching pinned
constants/`_expect(...)` calls in `src/baseline/policy.py` to the accepted
v002 values: `matrix_name: stage1_static_attributes_v002`, `expected_columns:
523`, `expected_model_input_columns: 473`,
`sha256: 4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297`,
`role_source: stage1_static_attributes_v002_column_manifest.json`. This is a
binding-value change, so `policy_version` was bumped `1 -> 2` (both in the
YAML and the loader's `_expect(data, "policy_version", 2)` check), per the
existing per-file policy-version convention already used in
`config/stage1_compact_package_selection_v001.yaml`. `tests/test_policy.py`
was updated to assert the v002 values and `policy_version == 2`, and to
mutation-test rejection of the old v001 matrix name, wrong column/model-input
counts, and the old policy version. The policy filename
(`stage1_scientific_baseline_v001.yaml`) is unchanged — it is the Stage 1
scientific-baseline policy artifact's name, not tied to the static matrix's
internal version. `scripts/prepare_stage1_compact_static_attributes.py`'s
usage-example docstring was updated from v001 to v002 paths (trivial,
operator-facing text only; the script's logic already takes all paths as CLI
arguments and was already re-run against v002 per the entry above).

**Unaffected / not reopened.** No scientific decision was reopened by this
patch — only the static-matrix identity/shape/checksum fields that already
changed with the v002 acceptance were propagated into the executable policy.
`src/baseline/split_audit.py`'s `V001_STATIC_MATRIX_SHA256` constant is
intentionally untouched: it independently verifies the matrix checksum that
actually built the already-frozen `config/stage1_baseline_splits_v001/`
artifacts (which were built from v001 and were not rerun), so it must keep
pointing at v001. `docs/stage1_baseline_package_implementation_plan.md` and
the mid-document 2K-G-H-era summary in `docs/FLASHNH_CURRENT_STATE.md` still
contain stale v001/531/496 prose; both are deferred to a later
documentation-reconciliation patch, not rewritten here.

**Not done.** No package builder, auditor, config-generator, mask, or
NeuralHydrology work was implemented. No h2o or Moriah connection was made.
No package was built and no training occurred. This is a policy/loader/test
reconciliation only.

---

## 2026-07-20 — Stage 1 — Compact Scientific Package implementation-contract docs reconciled to v002/32-basin state

**Problem.** The entry above reconciled the *executable* scientific-baseline
policy to `stage1_static_attributes_v002`, but explicitly deferred two
*documentation* artifacts: `docs/stage1_baseline_package_implementation_plan.md`
(the active Compact Scientific Package implementation plan) still prescribed
active/forward-looking work against the superseded `stage1_static_attributes_v001`
(531 total / 496 `model_input` columns) and a still-undecided "~12–15-basin"
illustrative compact subset; and the mid-document 2K-G-H-era summary in
`docs/FLASHNH_CURRENT_STATE.md` still presented the same v001/531/496 figures
without a superseded marker, alongside the correct v002 acceptance record at
the top of the same file.

**Fix (docs-only; no code/config/test/generated-artifact change).** Updated
active and forward-looking sections of
`docs/stage1_baseline_package_implementation_plan.md`: §1 binding-requirements
item 6 and the §3 gap-table static-attributes row now name
`stage1_static_attributes_v002` (523 total / 473 `model_input`, sha256
`4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297`); §15
"Static attribute integration" now names the v002 column-manifest filename
and 473-column count, and records that `attributes/attributes.csv` for the
accepted Compact Scientific Package is exactly the accepted 32×473 prepared
matrix; §21 sign-off item 2 records the updated 473-column audit scope while
preserving the original 2026-07-13 sign-off's decision text; §16 was rewritten
to replace the "~12–15 basin, chosen deliberately" illustrative proposal with
the actual accepted deterministic 32-basin Compact Scientific Package
selection (all `development_train`, no California/spatial-holdout basins;
see `docs/stage1_compact_package_selection.md` and the acceptance record in
`docs/FLASHNH_CURRENT_STATE.md`), and now states the frozen provisional
package architecture explicitly: one physical 32-basin package; every basin
NetCDF carries the eight accepted `v001-core` dynamic inputs plus the
diagnostic/provenance `qobs_m3s` series plus all four shifted mm/h lead
targets (`qobs_mm_per_h_lead01/03/06/12`); `attributes/attributes.csv` equals
the accepted 32×473 prepared static matrix; each lead-specific NH config
selects exactly one transformed target; raw `qobs_m3s` is never configured as
an NH training target. Rollout is documented as (1) one lead06/seq24
integration config first, then (2) the full 16 lead×seq config expansion
after that config's Moriah smoke passes — the plan no longer implies only
four configs exist or will ever exist. §12 gained a runtime-vs-audit-artifact
clarification, grounded in direct 2026-07-20 inspection of
`src/baseline/nh_dataset.py`, `src/baseline/gap_mask_io.py`, and
`src/baseline/validity_mask.py` (none modified): the currently *implemented*
runtime filter (`FlashNHDataset`) consumes a single flat
`masks/gap_timestamps.json` and computes `history_valid` live per instance,
materially simpler than the still-proposed 16-artifact `.npz` +
`masks_manifest.json` audit-artifact design in the same section; whether
precomputed sequence/lead validity masks or valid-timestamp arrays are still
needed as independent audit/off-by-one-verification evidence — and whether
such artifacts would live inside the transferable package, only in the
compact package's evidence/audit bundle, or both — remains an open decision
for the not-yet-started package builder/auditor increment. The approved rule
`sample_valid = history_valid_seq & target_boundary_valid_lead & split_valid`
is restated unchanged; a forcing gap at the future target timestamp is still
not by itself a basis for exclusion; basin-specific qobs NaNs remain a
separate target/loss-masking concern. §5 (basin-universe reconciliation) and
§9 (the illustrative I-0-era YAML policy-schema outline) each received a
short framing note marking their v001/531/496 content as valid, preserved
historical/illustrative record — distinct from, and not rewritten to match,
the current v002 state — since they describe the real, completed,
frozen split-generation process and the schema sketch that predates the
actual implemented `config/stage1_scientific_baseline_v001.yaml`
(`policy_version: 2`), respectively.

Also added a superseded/historical marker to the mid-document "Static
attributes (§3)" bullet in the 2026-07-12 2K-G-H sign-off summary of
`docs/FLASHNH_CURRENT_STATE.md`, pointing to the correct top-of-document
2026-07-20 v002 acceptance record without duplicating it and without altering
the historical decision text itself.

**Unaffected / not reopened.** No scientific decision was reopened. The
frozen split artifacts (`config/stage1_baseline_splits_v001/`) and the
accepted 32-basin Compact Scientific Package selection were **not**
regenerated or redesigned. No source code, YAML/config file, test, mask
artifact, or generated package was created or modified. No h2o or Moriah
connection was made. No training was run. Historical v001 provenance
(the real 2K-G-F-B build event, the real split-generation process that
genuinely used v001 fields, the real 2026-07-13 sign-off text) was preserved
as-is, not rewritten to falsely appear to have used v002.

**Not done.** Package-builder code, auditor code, the config generator, mask
artifacts (runtime or audit-form), NeuralHydrology integration code, and
Slurm files remain not implemented — this patch only reconciled planning
documentation with the already-accepted v002/32-basin state.

## 2026-07-25 Stage 1 — Milestone: initial full-population seed training profile

Following the full-population Moriah launch-readiness gate PASS (job
45634112, 2,307 development / 250 spatial-holdout basins, lead06/seq24),
work began on the first real full-population scientific seed training run.
Two small, bounded decisions were required to render a runnable config
without inventing new science; both are recorded here rather than only in
code comments.

**Decision 1 — named run-profile mechanism.** `src/baseline/nh_config_generation.py`
previously merged exactly one hardcoded technical profile
(`_COMPACT_SMOKE_RUN_PROFILE`) into every generated config, always recording
`compact_smoke_run_profile: true` in the generation manifest. This is now a
named-profile registry (`_RUN_PROFILES`, keyed by
`COMPACT_SMOKE_RUN_PROFILE_NAME` / `INITIAL_SEED_RUN_PROFILE_NAME`) selected
via a new `run_profile_name` parameter on `build_nh_config_mapping`,
`generate_stage1_nh_config`, and `generate_stage1_full_population_nh_config_bundles`
(default unchanged: `compact_smoke_v1`, so every existing caller/test is
unaffected). The generation manifest now always carries a generic
`run_profile_name` / `run_profile_values` / `run_profile_note` block; the
legacy `compact_smoke_run_profile: true` key is preserved byte-for-byte only
when that profile is actually used, and is never written (not even as
`false`) for any other profile — omission, not a false value, marks a
non-compact-smoke run.

**Decision 2 — initial seed hyperparameter values.** `docs/stage1_scientific_baseline_design.md`
Sec 9c documents the seed profile as "initial seed / first-viable-config
only," with two fields given as ranges rather than exact values. Both were
resolved narrowly, without inventing outside the documented range:
  - `dropout` (~0.2–0.3 documented) → **0.25** (range midpoint). **Correction
    (2026-07-25):** this entry originally claimed the config key name was
    plain `dropout`, citing
    `scripts/build_stage1_neuralhydrology_january_pilot.py` as NH 1.13
    precedent. That script is an unrun placeholder template (explicitly
    commented "LSTM placeholder -- update before running"), not executed
    evidence, and the key name was wrong. The real seed-training job
    (45639408) crashed immediately on Moriah with
    `ValueError: ['dropout'] are not recognized config keys.` The correct
    key, confirmed directly against the installed NH 1.13 package
    (`neuralhydrology/utils/config.py`'s `Config.output_dropout` property,
    `self._cfg.get("output_dropout", 0.0)`), is **`output_dropout`**. Fixed
    in `_INITIAL_SEED_TRAINING_PROFILE`, its manifest note, and the
    corresponding tests; job resubmitted.
  - `epochs` ("max 30–50, with early stopping" documented) → **40** (range
    midpoint), trained as a fixed epoch count with per-epoch checkpointing
    (`save_weights_every: 1`, `validate_every: 1`). NH 1.13 has no confirmed
    native early-stopping/patience config key anywhere in this repo's prior
    source inspection; this project's own Sec 9d selection rule (best epoch
    by validation raw-space NSE, chosen post-hoc after training completes)
    already **is** the early-stopping mechanism for this project — no live
    early-stopping callback was implemented or is needed.
  - `loss` (Sec 7: "likely an NSE-family loss," target-scaling-dependent and
    left open) → resolved to NH's built-in `NSE` loss, consistent with that
    steer and with this project's own prior Smoke 1 run (job 45370873,
    `loss: NSE`).
  - `validate_n_random_basins` → set to the full 2,307-basin development
    population (not a subsample), mirroring the compact profile's own
    convention of covering every available basin every epoch. This is
    required (not merely a preference): the post-hoc checkpoint-selection
    step compares validation metrics across epochs, which is only a fair
    comparison if every epoch validates the identical basin set.
  - `hidden_size` (128), `batch_size` (256), `optimizer` (Adam), `learning_rate`
    (1e-3), and `model` (cudalstm) are taken directly from Sec 9c's exact
    values, unchanged.

Neither decision met the task's stop-and-ask bar (a genuinely unresolved,
contradictory value that prevents a runnable config): both ranges had a
defensible, non-inventive midpoint resolution, and the early-stopping
mechanism question was already answered by this project's own existing
checkpoint-selection convention.

**Not done in this entry.** No training was run yet. The raw-space (m³/s)
evaluation layer, the GPU training Slurm script, and the basin-area
self-derivation approach (package NetCDFs carry no `DRAIN_SQKM` field; area
is instead derived from the algebraic identity relating each basin's
diagnostic `qobs_m3s` series to its built `qobs_mm_per_h_lead06` target) are
addressed separately as part of the same seed-run implementation increment.

**Decision 3 — the one permitted resource correction (`--mem`).** After the
`output_dropout` key fix, the first real training attempt (job 45639481)
started cleanly (CUDA detected, both guards passed, the full 2,307-basin
dataset loaded) but was killed by the Slurm cgroup OOM handler ~26 minutes
in, before completing epoch 0 or writing any checkpoint (`sacct -j 45639481`:
`OUT_OF_MEMORY`, `MaxRSS=66741204K` against the script's original
`--mem=64G`; stderr: "Detected 1 oom-kill event(s)"). This is a genuine
infrastructure sizing problem, not a model-skill or hyperparameter issue —
`--mem` is a Slurm resource request, not a training hyperparameter. Per the
task's "exactly ONE narrow, documented resource correction" allowance,
`run_stage1_full_population_lead06_seq24_seed_train_moriah.sbatch`'s
`--mem` was raised from `64G` to `128G` (catfish-04 has ~1TB RealMemory, so
ample headroom); no other setting (partition, GRES, batch size, workers,
etc.) was changed. Job resubmitted after this fix.

**Decision 4 — user-directed mid-run `--cpus-per-task`/`num_workers` increase
(2026-07-25, separate from the one `--mem` correction above).** After the
`--mem` fix, job 45640083 ran cleanly through epochs 1-3 with
`--cpus-per-task=8` / `num_workers=4`: training took ~40min (epoch 1, includes
one-time setup), ~19min (epoch 2), ~19min (epoch 3); validation (2,307 basins)
took ~20min (epoch 1), ~10min (epoch 2), ~10min (epoch 3) (measured from
on-disk checkpoint-file timestamps and log inspection; validation ran at a
flat ~2.1 basins/sec throughout, consistent with a CPU-bound per-basin
data-loading bottleneck rather than GPU compute). The user explicitly asked,
mid-run, to stop after epoch 3's validation finished (confirmed via the
appearance of the epoch 4 training progress bar in the log) specifically to
capture this before/after timing baseline, then raise `--cpus-per-task`
(8->16) and the run directory's own `config.yml` `num_workers` (4->12) and
resume, to see whether it speeds up the CPU-bound validation phase. This is a
Slurm-resource and dataloader-parallelism change only, not a model or
training hyperparameter, and was explicitly directed by the user rather than
an autonomous correction. Verified against
`neuralhydrology/nh_run.py`'s `continue_run()` (loads `run_dir/config.yml`
fresh on every resume; only overridden if an external `--config-file` is
explicitly passed, which this script's `continue` call never does), so
editing the run directory's `config.yml` directly before resubmitting is
sufficient — no risk of the resume silently reverting to the old
`num_workers` value. Job 45640083 was cancelled cleanly (not wall-time-killed)
at 2:00:04 elapsed with epoch 3's checkpoint already on disk, and resubmitted
after these changes.

**Decision 5 — user-directed follow-up `--mem` increase after a second OOM
(2026-07-25).** The job resubmitted under Decision 4's `num_workers=12`/
`--cpus-per-task=16` (job 45640233) correctly resumed from epoch 3's
checkpoint but was OOM-killed 19:30 into epoch 4 *training* (not data
loading), at `MaxRSS=133360076K` (~133.4G) against the then `--mem=128G`
limit (`sacct -j 45640233`: `OUT_OF_MEMORY`; stderr: "Detected 1 oom-kill
event(s)"). No `model_epoch004.pt` was written, so the safe resume point
remained epoch 3's checkpoint. Raising `num_workers` 3x (4->12) evidently
added enough per-dataloader-worker memory overhead to exceed the 128G ceiling
that had been sufficient at `num_workers=4` — this is itself a useful,
concrete finding about the memory/parallelism tradeoff for this dataset size.
Consulted the user on how to proceed (raise `--mem` further and keep
`num_workers=12`; dial `num_workers` back to a smaller compromise value; or
revert to the known-good `--cpus-per-task=8`/`num_workers=4`/`--mem=128G`
configuration). User chose to raise `--mem` further while keeping
`num_workers=12`/`--cpus-per-task=16` unchanged. `scontrol show node
catfish-01` showed ~849G free of ~1TB total `RealMemory` at the time, so
`--mem` was raised from `128G` to `224G` in
`run_stage1_full_population_lead06_seq24_seed_train_moriah.sbatch`; no
model/training hyperparameter was touched. Job resubmitted after this fix.

## 2026-07-26 — Stage 1 full-population seed-run closure decisions

Closure and handoff task, not a new experiment. The job resubmitted under
Decision 5's `--mem=224G` (job 45640243) was cancelled cleanly by the user
at the epoch-11 checkpoint ceiling, `MaxRSS=234046840K` (~223.2 GiB against
the 224G allocation — within ~1 GiB of a third OOM). No epoch-12 checkpoint
was produced. A complete raw-space (m³/s) development-validation evaluation
(2,307 basins, calendar year 2024) was then run for all 11 checkpoints;
every epoch admitted an identical 19,747,262 samples. Full comparison:
`reports/seed_validation_review_v001/aggregate/seed_ckpt_comparison_report_epochs1to11.{json,md}`.
Per-epoch median NSE 0.2168–0.2401, mean NSE -69.80 to -7.73, pooled NSE
0.4234–0.4651 — median (epoch 7) and pooled (epoch 6) disagree, mean (epoch
9) is outlier-dominated. No monotonic trend across epochs; training had
plateaued well before the epoch-11 cancellation.

**Decision 6 — seed-run outcome.** The full-population seed run (2,307
development basins, `qobs_mm_per_h_lead06`, `seq_length: 24`, epochs 1–11)
is closed as a **successful pipeline proof and initial optimization
baseline**. It is explicitly **not** a tuned model and **not** the official
Stage 1 benchmark. No temporal-test or spatial-holdout data was accessed at
any point in this run or its evaluation.

**Decision 7 — median per-basin raw-space NSE is the primary selection
metric.** Because the three natural cross-basin aggregations of per-basin
NSE (median, mean, pooled) disagree on the top checkpoint (epoch 7, 9, and 6
respectively) and mean NSE is dominated by extreme negative outliers on a
small subset of basins, **median per-basin raw-space NSE on development
validation** is adopted as the primary run/checkpoint-selection statistic
project-wide going forward. Mean per-basin NSE and pooled NSE are retained
as secondary diagnostics only. This resolves the gap flagged in
`docs/stage1_scientific_baseline_design.md` §7 (mean-NSE-can-hide-failure)
and formalizes the previously undocumented `selection_basis:
"development_validation_only_raw_space_median_nse"` convention already used
by `scripts/evaluate_stage1_seed_raw_space.py`.

**Decision 8 — no scientifically meaningful winner across the plateau.**
Epoch 7 (max median NSE, ≈0.2401) is recorded as the deterministic
representative checkpoint of this seed run under the Decision 7 rule, but
this is **not** a claim that epoch 7 is scientifically meaningfully
superior to nearby checkpoints (6, 9, 10 are within noise on median NSE).
NSE-sign fractions (~78–81% positive, ~12–13% > 0.5, ~19–22% negative) and
all secondary metrics are essentially flat across all 11 epochs.

**Decision 9 — required distribution reporting.** Future raw-space
evaluations must report the per-basin metric distribution
(p1/p5/p10/p25/p50/p75/p90/p95/p99) and NSE sign fractions (>0, >0.5, <0),
not just a single summary statistic — this is what surfaced the Decision 7
disagreement and must not be lost in future reporting.

**Decision 10 — early-stopping policy for future Stage 1 runs.** Save every
epoch's checkpoint. No stop before epoch 6. Run the official raw-space
validation every 2–3 epochs (not every epoch — validation throughput is the
CPU-bound bottleneck, see Decision 11 and
`docs/stage1_neuralhydrology_preflight.md`). Minimum meaningful improvement:
0.005 median NSE between successive official validation events. Patience: 3
validation events without meeting the minimum improvement. Maximum 30–40
epochs. Retain the best checkpoint by median NSE. Temporal-test and
spatial-holdout data are never used for stopping or selection decisions.

**Decision 11 — the 12-worker/224G configuration must not become the
default.** The `--cpus-per-task=16`/`num_workers=12` change tested in
Decision 4 produced no measurable training or validation speedup (validation
held a flat ~2.1 basins/sec before and after — see
`seed_ckpt_comparison_report_epochs1to11.md`) while requiring the `--mem`
ceiling to be raised to 224G (Decision 5) and running within ~1 GiB of a
third OOM at cancellation. This combination is a **known-expensive,
no-benefit configuration** and must not be used as the default for future
Stage 1 runs without first diagnosing the actual validation bottleneck
(suspected: per-basin data loading, not GPU compute or worker count).

**Decision 12 — resume integrity finding, formalized.** The epoch-3 →
epoch-4 continuation (Decisions 3–5) restores model weights and full Adam
optimizer state correctly via NeuralHydrology's `continue_run()`, but
reseeds `random`/`numpy`/`torch`/`torch.cuda` RNG to the fixed `cfg.seed`
rather than continuing the pre-interruption stream, and NeuralHydrology has
no dataloader shuffle-state serialization. **The continuation is
scientifically valid but not bitwise-equivalent** to an uninterrupted
11-epoch run — a standing caveat on any comparison spanning the epoch-3/
epoch-4 boundary, and on any future run that resumes after an interruption.

**Decision 13 — learned static representation deferred to next phase.** The
checkpoint-comparison plateau, combined with the current static-attribute
handling (raw concatenated attributes, no learned embedding), indicates a
learned static representation will likely be required to meaningfully move
past this plateau. This is **not** designed or implemented here — it is
explicitly in scope for the next phase ("Stage 1 validation and
optimization foundation").

**Not done in this entry.** No training, no re-evaluation, no temporal-test
or spatial-holdout access. This entry only records closure decisions over
already-completed, already-evidenced work.

## 2026-07-26 — Stage 1 validation and optimization foundation — Parts A-K decisions

Design/tooling/documentation foundation phase, opened immediately after the
seed-run closure above (Decision 13). **No training run was launched in
this phase; no hyperparameter sweep was run; temporal-test and
spatial-holdout data were never accessed.** Full account:
`docs/stage1_validation_optimization_foundation.md`; evidence:
`reports/stage1_validation_optimization_foundation_v001/` (untracked).

**Decision 1 — static-pathway audit confirms a learned embedding is a
genuinely novel architectural variant.** Direct inspection of
`neuralhydrology/modelzoo/cudalstm.py` and `inputlayer.py` against the
seed run's own generated configs confirms `statics_embedding` was absent
(NH 1.13 resolves this to `nn.Identity()`, i.e. raw concatenation of all
473 static attributes). This resolves Decision 13's open question: a first
embedded-static candidate is not a relabeling of the seed's own
architecture, and is therefore in scope as a design-only candidate (see
Decision 5 below).

**Decision 2 — 400-basin frozen development-validation screening subset
adopted for future per-epoch comparisons.** Selected deterministically
(seed=42) from the 2,307 development-training basins. Validated against
all 11 seed-run checkpoints: the subset's per-epoch median NSE tracks the
full-population median NSE closely (Spearman 0.90, Kendall 0.82, max
absolute difference 0.0053). Adopted as the recommended per-epoch
comparison mechanism for future training runs, reserving full-population
validation for finalist checkpoints only (see Decision 6).

**Decision 3 — early-stopping policy (Decision 10 above) is now an
executable, restart-safe state machine**, not policy prose alone:
`config/stage1_early_stopping_policy_v001.yaml` +
`src/baseline/early_stopping.py`. Idempotent replay of the last recorded
event, hard rejection of out-of-order/inconsistent replay, and best-
checkpoint retention are all enforced in code, not left to a training
script's own bookkeeping. 29 focused tests pass, including a structural
proof that no public function accepts a test/holdout-shaped parameter.

**Decision 4 — optional W&B tracking module added, disabled by default.**
`config/stage1_wandb_tracking_policy_v001.yaml` (`enabled: false`, `mode:
disabled`) + `src/baseline/wandb_tracking.py`. Every logged item is
mirrored in-memory regardless of backend; credential-shaped keys are
rejected structurally; artifact references above a small size ceiling
(1 MiB default) are refused, so large prediction/checkpoint/NetCDF/Parquet
files can never be logged through this module. This module never launches
a sweep or a training run and does not itself decide which fields a future
training harness passes to it. 30 focused tests pass.

**Decision 5 — first embedded-static CudaLSTM candidate, design/config +
structural-smoke only.** `embedded_static_cudalstm_pilot` run profile added
to `src/baseline/nh_config_generation.py`: the existing compact-smoke
technical settings (small scale, 2 epochs) plus one addition,
`statics_embedding: {type: fc, hiddens: [128, 64], activation: tanh,
dropout: 0.1}`, validated structurally by a new
`validate_statics_embedding_spec()` function. **Not trained, not compared
against the seed run, no winner declared** — this is exactly the scope
Decision 13 flagged as the next phase's responsibility, and no further than
that.

**Decision 6 — corrected operational-defaults recommendation for the next
Stage 1 training run.** An initial draft of the next-run resource
recommendation (never committed) mistakenly proposed keeping the seed run's
mid-run-escalated `--cpus-per-task=16`/`num_workers=12`/`--mem=224G`
configuration as the default. This was caught and corrected before being
committed: per Decision 11 above, that configuration produced **no
measurable speedup** and only existed because Decision 4's (2026-07-25,
seed-run) worker increase forced Decision 5's (2026-07-25, seed-run) memory
increase. The corrected recommendation reverts to the last known-good,
comfortably-margined configuration: `--cpus-per-task=8`, `num_workers=4`,
`--mem=128G` — unchanged from the seed run's own Decision 3 fix, before its
Decision 4 experiment. The new recommendation this phase actually adds is
independent of that reverted point: use the Decision 2 screening subset for
per-epoch validation cadence, reserving full-population validation for
finalist checkpoints.

**Decision 7 — two ad hoc diagnostic scripts retained, not promoted or
discarded.** `scripts/aggregate_stage1_seed_checkpoint_report.py` and
`scripts/dump_per_basin_table.py` (both already used to produce the
committed-evidence-pending `reports/seed_validation_review_v001/` report)
are recommended to be kept and committed under `scripts/` unmodified — both
are thin, structurally test/holdout-safe (`--period` hardcoded to
`"validation"`) wrappers over already-certified, already-tested evaluation
functions, not generic library code warranting promotion into
`src/baseline/`.

**Not done in this entry.** No hyperparameter sweep. No embedded-static or
EA-LSTM training run. No temporal-test or spatial-holdout evaluation. No
change to the certified Compact Scientific Package or the canonical basin
splits. No commit performed automatically — see this phase's own §19
git-status/commit-structure proposal for a human-actionable recommendation
only.

## 2026-07-27 — Commit-readiness pass for the validation-and-optimization-foundation phase

Conservative pass resolving five small ambiguities flagged in strategic
review before committing the phase above. **No training run launched, no NH
inference run, no sealed-data access, no screening-subset redesign, no
across-epoch-median skill definition introduced, no full-population
embedded-static profile created, no commit performed.** Evidence:
`reports/stage1_validation_optimization_foundation_v001/commit_readiness_epoch7_epoch9_sensitivity/`
(untracked).

**Decision 1 — epoch-7 vs. epoch-9 anchor-epoch sensitivity check: retention
conditions fail; epoch-9 artifacts neither replaced nor auto-regenerated.**
Using already-persisted per-basin NSE (no inference rerun), the existing
`compute_skill_quartile_edges`/`assign_skill_stratum` functions were applied
to epoch 7 and compared against epoch 9 for all 2,307 development basins:
75.8% same skill stratum (1,749/2,307; 516 moved by one stratum, 42 by more
than one) — below the ~90% review heuristic. Regenerating the screening
subset and hydrograph atlas at epoch 7 (identical policy/seed) gave basin
overlaps of 82/400 (Jaccard 0.114) and 3/24 (Jaccard 0.067) against the
epoch-9 candidates — both far below ~90%. The epoch-7 screening-subset
candidate's subset-vs-full validation tracking is also markedly worse than
epoch 9's (Spearman 0.482 vs. 0.900, Kendall 0.345 vs. 0.818, top-3-epoch
overlap 1 vs. 2, max abs. median-NSE diff 0.0175 vs. 0.0053). Per this
check's own explicit stopping rule, the epoch-9 screening-subset and
hydrograph-atlas artifacts are **not** regenerated or replaced, and **no**
across-epoch-median or other new skill definition is introduced. This is
recorded as an open item for user review, not resolved in this pass.

**Decision 2 — embedded-static pilot profile scope clarified, not
retuned.** `embedded_static_cudalstm_pilot`'s `hiddens=[128, 64]`,
`activation=tanh`, `dropout=0.1` are unchanged. Confirmed structurally
already correct: the manifest always records `run_profile_name` +
`run_profile_note` (STRUCTURAL-SMOKE-ONLY wording already present), and
`scripts/generate_stage1_full_population_nh_config.py`'s own
`_KNOWN_RUN_PROFILES` tuple does not include this profile name, so it cannot
become the full-population generator's default even silently. Added one
clarifying paragraph to
`reports/.../part_i_embedded_static_pilot/part_i_embedded_static_cudalstm_pilot.md`
stating explicitly that this is a structural-smoke construction choice, not
the first scientific embedded-static candidate, and that full-population
embedded-static candidates are later, separately-scoped optimization-phase
work.

**Decision 3 — early-stopping and W&B documentation wording tightened to
distinguish "implemented and tested" from "operationally wired into
training."** No code change (Parts E/F's own evidence docs already used this
framing); `docs/stage1_validation_optimization_foundation.md` and
`docs/FLASHNH_CURRENT_STATE.md` now state explicitly, at the phase-index and
current-state level, that both modules are implemented and tested and that
real training-orchestration/harness integration remains pending, so neither
document can be misread as claiming a real run is already auto-stopped or
auto-logged.

**Decision 4 — screening-subset and hydrograph-atlas wording distinguishes
"selection design/candidate" from "settled, authoritative artifact."**
`docs/stage1_validation_optimization_foundation.md`'s Parts table now
describes Part D's output as a screening-subset *candidate* "to be frozen"
(not already frozen) and Part C's output as the atlas *selection design*
plus an epoch-9 candidate basin list, explicitly noting the final
observed-vs-predicted atlas is not yet generated. This is a wording
correction only — neither Part's underlying method or epoch-9 candidate
changed.

**Decision 5 — supersedes Decision 7 (2026-07-26) on the two diagnostic
scripts: excluded from the proposed foundation commit for now, left
untracked.** `scripts/aggregate_stage1_seed_checkpoint_report.py` and
`scripts/dump_per_basin_table.py` remain useful, unmodified, and undeleted,
but — to keep this commit-readiness pass conservative and because neither
script has a dedicated CLI test — they are recommended to stay untracked
provenance/reference scripts for now rather than entering the proposed
commit. They may be tested and committed later if reused in the real
optimization harness.

**Not done in this entry.** No training run. No NH inference run. No
sealed-data (temporal-test/spatial-holdout/California) access. No
screening-subset or hydrograph-atlas redesign. No replacement of the
epoch-9 artifacts. No across-epoch-median skill definition. No real
optimization harness. No W&B/early-stopping training integration. No new
full-population embedded-static profile. No full or repeated test-suite
run beyond what Part B's (already doc-only) change required. No commit
performed automatically.

## 2026-07-27 — Stage 1 validation and optimization foundation — final status resolution and commit closure

Resolves the open item from the commit-readiness pass above and closes out
the phase for commit. **No further scientific analysis, no additional
epoch comparison, no across-epoch skill definition, no regeneration of the
screening subset or atlas selection, no algorithm change, no policy change,
no full-population embedded-static candidate, no training, no inference, no
sealed-data access.**

**Decision 6 — 400-basin screening subset accepted as `provisional
operational screening subset v001`.** The existing epoch-9-based selection
is deterministic, reproducible, and stratified by geography, physical/
hydroclimatic attributes, flow variability, and seed skill; it tracks the
full 2,307-basin population well across the existing 11 checkpoints
(Spearman ≈0.90, Kendall ≈0.82, max abs. median-NSE diff ≈0.0053); the
epoch-7 sensitivity candidate performed materially worse on every one of
those measures. Exact basin membership is sensitive to the anchor checkpoint
because the design contains many small composite strata and seeded
within-cell draws — this sensitivity does **not** invalidate the subset's
operational purpose. **Not yet permanently frozen or scientifically
authoritative.** The full 2,307-basin development-validation population
remains authoritative for final checkpoint, run, architecture, and
hyperparameter selection. **Prospective validation rule:** use the subset
for frequent feedback and early pruning; over approximately the first 3-5
materially different future model runs, compare subset-based conclusions
against full-population validation; reconsider or formally freeze the
subset only after that prospective evidence exists. No search for a
supposedly optimal anchor checkpoint was performed or is planned now.

**Decision 7 — 24-basin hydrograph-atlas selection accepted as
`deterministic provisional hydrograph-atlas selection v001`.** The existing
epoch-9-based list is reproducible; balanced by validation-skill stratum,
basin-area class, and east/west geography; and provides useful HUC02/
macroregion breadth. Its purpose is structured visual inspection and
trust-building, not statistical estimation or model selection, so basin-
identity sensitivity to the skill-anchor checkpoint is acceptable for this
use. **The deterministic selection framework is complete; the final
observed-vs-predicted hydrograph atlas is not yet generated.** When the
actual atlas is built, the selected list may be retained or revised using a
later model or a more stable cross-model skill definition, without
reopening the selection framework itself.

**Decision 8 — general interpretation, binding on both artifacts.** Exact
membership stability is not required for either artifact; reproducibility,
stratification, transparency, and fitness for purpose are the requirements.
Neither artifact is an independent test set. Neither replaces full
development validation. Neither may include temporal-test, spatial-holdout,
or California information at any point.

**Decision 9 — supersedes Decision 5 (2026-07-27, above) only in
confirming, not changing, its conclusion.** The two diagnostic scripts
(`scripts/aggregate_stage1_seed_checkpoint_report.py`,
`scripts/dump_per_basin_table.py`) remain untracked and excluded from the
foundation commit: under the current state they have no dedicated CLI
tests. Not deleted; may be tested and committed later if reused in the
optimization harness.

**Commit closure.** Two commits proposed and created from the accumulated
phase diff: (1) foundation implementation — percentile diagnostics,
hydrograph-atlas selection/event foundation, screening-subset selection and
validation tooling, early-stopping policy engine, W&B tracking wrapper,
embedded-static structural-smoke config-generation support, associated
policy YAML files, associated tests, and reusable generation/analysis
scripts; (2) documentation closure — the five status documents listed at
the top of this entry's parent phase plus this decision log. Generated
evidence under `reports/` and the two untested diagnostic scripts remain
untracked. Exact commit hashes and push status are recorded in the
session's final report, not duplicated here.

**Not done in this entry.** No training run. No NH inference run. No
sealed-data access. No further epoch comparison. No across-epoch skill
definition. No screening-subset or atlas regeneration. No selection-
algorithm change. No early-stopping or W&B policy change. No full-
population embedded-static candidate. No repeated full test-suite run
(relies on the previously reported 1094/1094-effective regression result;
no production code changed since).
