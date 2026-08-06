

# Decision Log

Project: Flash-NH — near-real-time and forecast-aware hydrological modeling pipeline.

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

**Sequence length reframed as a separate temporal-context model-family axis (adopted, binding).** Sequence length is fixed at 24 for the current model family and is **not** an ordinary hyperparameter in the current near-term tuning funnel. Alternative sequence lengths represent separate temporal-context model families, since they change the scientific information available to the model, antecedent-memory assumptions, input construction, compute/memory requirements, and interpretation across basin response times. A later sequence-length study may compare alternative temporal-context model families against a mature 24-hour model, but is not part of the current hyperparameter phase. `docs/stage1_validation_optimization_foundation.md` Part L.1's Stage-B dimension list (which previously listed sequence length alongside ordinary hyperparameters) is corrected accordingly by this entry.

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