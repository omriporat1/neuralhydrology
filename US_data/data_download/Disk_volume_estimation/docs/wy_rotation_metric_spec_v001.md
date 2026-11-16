# Water-year rotation: metric, checkpoint and second-arm specification v001

**Status: FROZEN v001 (Omri, 2026-11-16).** The choices are in §6, including the review round. **Where §6 differs from the option text in the body, §6 governs.** The gate-1 fresh-Claude review found the spec "sound with changes", and the fixes were applied. There was no second review (Omri's decision). The canonical copy is `docs/wy_rotation_metric_spec_v001.md`; see the decision log of 2026-11-16. Any change to this spec needs a new version and Omri's approval. Two choices differ from the recommendations: D7 (≥ 2/5) and D11 (λ ∈ {0.5, 2, 8}). The option text below is kept as originally written; §6 is authoritative where the two differ. Omri authorises every metric, checkpoint and loss choice below. Each item lists the options, a recommendation (labelled as one) and the strongest argument against it. The decision points are numbered **D1-D16** and collected in §6. A plain-language version is in `DECISIONS_FOR_OMRI.md`. The draft decision-log entry is in `DECISION_LOG_DRAFT.md` (not added to `docs/decision_log.md`).

**Written 2026-11-15, before any rotation score exists.** No rotation run has been launched. No training, Slurm job or model result was read or produced for this spec. Sealed: spatial holdout, California, WY2026; job 45625077's test outputs stay unopened.

**Inputs read.** `CLAUDE.md`; `docs/decision_log.md` entries of 2026-11-11, 2026-11-12, 2026-11-15; top of `docs/FLASHNH_CURRENT_STATE.md`; `.scratch_local/wy_rotation_preflight_20261022/PREFLIGHT_MEMO.md`; `journal/meetings/2026-11-11.md`; code listed in §0.

**Review gate.** This is an evaluation rule, so `docs/agent_handoff_rules.md` §3.2 gates 1 and 2 apply. After Omri's choices are filled in, a fresh-context reviewer must review the filled spec before it is frozen and before any rotation launch. This session does not review its own work.

---

## 0. What already exists and is reused (no parallel metric math)

| Need | Qualified code reused | What it does |
|---|---|---|
| Raw-space conversion | `src/baseline/nh_raw_space_evaluation.py`: `convert_period_to_raw_space`, `evaluate_basin_raw_space` | mm/h target → m³/s with the basin area taken from the training windows only (rotation sealing, 6f7310b) |
| Overall metrics | `raw_space_metrics` (same file, l.697) | NSE, KGE (+ r, α, β), RMSE, MAE, bias, PBIAS. NSE/KGE are NaN (not 0) when the observed variance is zero |
| Cross-basin summary | `aggregate_raw_space_metrics` (l.829); `percentile_diagnostics.compute_percentile_table` | median, mean, q25, q75 over basins with a finite value; full percentile grid |
| Sealed per-run scoring | `src/baseline/nh_seed_evaluation.py`: `raw_space_metrics_for_run_period` | refuses unsealed runs, checks receipts, checks every scored observation lies in the period's readable windows |
| Flood (Q98) metrics | `src/baseline/stage1_v2_12plus12_hydrological_consumer.py`: `derive_canonical_basin_q98_facts` (l.607), `compute_q98_configuration_basin_diagnostics` (l.677); constants `HIGH_FLOW_QUANTILE = 0.98`, `MIN_HIGH_FLOW_NSE_SAMPLES = 50` | the RD1 Q98 package used for P17-P19: per-basin Q98 from observed flow (linear quantile, ties at the threshold included); Q98-normalised RMSE; relative volume bias on Q98+ hours; magnitude error at the observed peak hour (earliest on ties); high-flow NSE on Q98+ hours (needs ≥ 50 hours and nonzero variance) |

**Not reused, on purpose.** `src/baseline/high_flow_event_metrics.py` (legacy, different normalisation, no NSE) and the event-window metrics. Event isolation and peak-timing alignment were deferred at the 2026-11-11 meeting to the stage after the baseline.

**Implementation gap (plan only, no code here; corrected after review).** The Q98 functions take the admitted series (dates, observed, simulated) per basin. The sealed scorer `raw_space_metrics_for_run_period` asks for the admitted arrays only when `compute_pooled` is set, and then pops them into the pooled lists (`nh_seed_evaluation.py:252-256`). The per-basin results therefore carry neither the arrays nor their dates. A new seam is needed: a per-basin variant of the sealed scorer that returns admitted dates, observations and simulations, with the same sealing checks. **Date convention:** "date" is the NH sample date, i.e. the issue hour. The observation it scores is at issue + 6 h. Peak times are reported as observation time (issue + 6 h). Ordering and ties are the same under both. `fixed_support_contract_v2` (the RD1 producer of those series) must not be used on rotation runs (decision log 2026-11-15) and is not needed.

### Scoring gates that apply to every score in this spec (recommended; part of D1)

These are qualification gates, not metric choices. A failed gate stops the comparison; it is never repaired silently.

- **G1 Population.** Every scored epoch holds all 2,307 development basins in its results. NH's `validate_n_random_basins` must not truncate (a known past defect at 1,000 basins).
- **G2 Finite predictions.** Zero non-finite predictions at admitted hours. `raw_space_metrics` would otherwise drop those hours pairwise without failing.
- **G3 Area.** The area-exclusion list is identical across the 12 epochs and across arms in the same cycle, and is reported.
- **G4 Same observations across arms.** In one cycle, both arms must have the same admitted hours and the same observed values. **Rule (Omri, review round F): bit-identical, checked first.** The arms share data, contract, scaler, areas and sequence length, so the values pass through identical arithmetic. The arms' admitted dates and observed values are compared **before any metric is computed** (score-blind). Any difference stops the comparison and is diagnosed as a technical fault; no tolerance is pre-set. The provisional RD1 envelope (`PROVENANCE_ATOL_M3S` / `PROVENANCE_RTOL`, documented as not a qualified threshold, `stage1_v2_12plus12_hydrological_consumer.py:136-156`) is not used as a gate. Q98 facts (threshold, flood hours, peak hour, eligibility) are computed once per (cycle, period) from the baseline run and reused for arm 2.

---

## 1. Baseline metric set

**Where.** Per basin, on the cycle's assessment year, raw m³/s, on the admitted hours of the frozen contract: issue and target hour in one slice (convention A), finite observation, 6 h lead. No common-support file (refused under the contract).

### 1a. Per-basin metrics (D1)

| Group | Metric | Code | Role (recommended) |
|---|---|---|---|
| Overall | **NSE** | `raw_space_metrics` | **primary overall** (continuity with all earlier work) |
| Overall | KGE and its r, α, β | `raw_space_metrics` | secondary (P19 led on KGE) |
| Overall | PBIAS (%) | `raw_space_metrics` | secondary: whole-year volume |
| Flood | **Q98-normalised RMSE** = RMSE on Q98+ hours ÷ the Q98 value | `compute_q98_configuration_basin_diagnostics` | **primary flood** (lower is better) |
| Flood | High-flow NSE on Q98+ hours (≥ 50 hours) | same | secondary flood |
| Flood | Relative volume bias on Q98+ hours (signed; < 0 = under-prediction) | same | secondary: the known "peaks too low" failure |
| Flood | Relative error at the observed annual peak hour (signed) | same | secondary: one hour per basin-year, noisy |
| Guard | **PBIAS on hours below the Q98** (the complement of the flood mask) | `raw_space_metrics` on the complement mask (new mask, no new formula) | **false-alarm / low-flow guard**, mainly for arm 2 |

The flood threshold for scoring an assessment year is that year's own Q98 per basin (same as D4 for the checkpoint year).

- **Recommendation:** the set above; NSE as the primary overall metric and Q98-normalised RMSE as the primary flood metric.
- **Strongest argument against:** every flood metric here is defined on hours where the *observed* flow is high. None penalises a false flood (a predicted peak with no observed peak); the only guard is the below-Q98 bias, which sees average over-prediction but not a single false peak. Event-based false-alarm metrics would need event isolation, which was deferred.

### 1b. Summaries per cycle and pooled (D2)

- **Per cycle (one assessment year):** median, q25, q75 and the full percentile grid of each metric over basins with a finite value; count of basins scored, undefined (e.g. zero observed variance) and flood-ineligible (§2d).
- **Pooled over 5 cycles.** Options:
  - (a) median over all basin-years (2,307 basins × 5 years, each basin-year one value);
  - (b) mean of the 5 per-cycle medians;
  - (c) per-basin median over its 5 years, then the median over basins.
- **Recommendation: (a)**, with the 5 per-cycle medians always printed beside it.
- **Against:** (a) lets a year with more eligible basins weigh more in the flood metrics; (c) treats basins exactly equally but hides year effects.

### 1c. Fixed sensitivities and flags (part of D2)

Labelled as sensitivities, never as the result:
- **S-years:** pooled over WY2021-WY2023 only. WY2024 and WY2025 influenced the configuration choice (HPO and length tests on calendar 2024), so their scores are slightly optimistic for the configuration. Every per-year table marks them.
- **S-variance:** excluding the basins in the lowest decile of observed variance in that year (the basins that broke the |ΔNSE| check on 2026-10-21). **Variance is measured in mm/h** (area-normalised, the model's target units), so the decile reflects flat hydrographs rather than basin size (Omri, review round G).
- No basin is excluded because of its score.

---

## 2. Checkpoint rules

Both picks are made per run, on the cycle's checkpoint year, over epochs 1-12, on all 2,307 development basins (decision log 2026-11-11). The checkpoint year is never the assessment year.

### 2a. The median-NSE pick (the existing rule, restated so it is frozen)

The highest median raw-space NSE over basins with finite NSE, earliest epoch on an exact tie. Gates G1-G3. No choice here unless Omri wants one; it is listed as D3 only for confirmation.

### 2b. The flood pick: which metric (D3)

| Option | Rule | Note |
|---|---|---|
| F-NSE | highest median high-flow NSE (Q98+ hours) over eligible basins | the example in the decision log |
| **F-nRMSE** | lowest median Q98-normalised RMSE over eligible basins | the RD1 "Q98 error" |
| F-guard | F-NSE or F-nRMSE, but only among epochs whose median NSE is within δ of the best epoch's | prevents a flood pick that is poor overall |

Within one basin, F-NSE and F-nRMSE rank epochs identically: both are the squared error on that basin's flood hours divided by a fixed number. They differ only in that number (the "yardstick" that makes basins comparable): the spread of flows within the flood hours (F-NSE) or the Q98 value (F-nRMSE). So the choice only matters for how basins combine in the median.

- **Recommendation: F-nRMSE, without the guard.** In one year, the flood hours of a basin often come from one to three events. Their internal spread can then be tiny (a flat high plateau, giving a very negative NSE) or large (one big event), by chance. The Q98 value is a steadier yardstick. Without the guard, the flood pick stays what the decision log describes: a separate pick, with its overall NSE reported next to it.
- **Against:** it departs from the decision-log example ("NSE on hours above the Q98"), which Omri and the supervisor already discussed. NSE has a natural zero point (as good as the mean of the flood hours); normalised RMSE has no skill reference, so it is harder to explain in a talk.

### 2c. Flood threshold (D4)

- **T1:** per basin, from that basin's admitted observed flow in the checkpoint year itself. This is how the qualified code works.
- **T2:** per basin, from the cycle's three training years. Steadier, and readable under the sealing.
- **T3:** one threshold per basin over all of WY2021-WY2025. Not allowed: it would read observations outside the cycle's readable windows.
- **Recommendation: T1** (0.98 linear quantile; hours at the threshold value count as flood hours). It uses only the year being scored, does not depend on any model, and is the same for all epochs and both arms.
- **Against:** in a dry year a basin's top 2% may not be floods at all. And "flood" then means "this year's top 2%", which differs between years and between the checkpoint and assessment years.

### 2d. Which hours count, and the minimum per basin (D5)

- **Hours.** Admitted hours of the checkpoint year (issue and target in one slice, finite observation) whose observed flow is ≥ the threshold. The prediction must be finite at every admitted hour (G2).
- **Eligibility options:**
  - (a) Q98 > 0 and at least 50 flood hours (the existing code constant). 50 flood hours need roughly ≥ 2,500 admitted hours, about 29% of a year.
  - (b) like (a), plus at most 5% of admitted hours flagged. This guards against gauges whose readings repeat the same value: a tie at the threshold then marks far more than the top 2%. When Q98 = 0 (more than 98% of hours dry), "≥ 0" would flag every hour, which (a) already excludes.
  - (c) a higher minimum, such as 100 hours.
- **Always ineligible, under every option:** a basin-year with no admitted hours. The Q98 helper raises on an empty series (`stage1_v2_12plus12_hydrological_consumer.py:625-626`), so the wrapper filters these first and lists them (review fix).
- **Ineligible basins** are left out of the flood median only. They are listed with the reason and counted per year. They stay in every overall metric. Eligibility depends only on observations, so it is the same for all 12 epochs and both arms of a cycle.
- **Minimum population (Omri, review round E):** eligibility counts per cycle and year are computed from observations alone and published before any scoring. If fewer than half of the 2,307 basins are eligible in a cycle, the flood pick **still counts as usual** and the cycle is flagged in the report. No judgement is made after scores are seen.
- **Recommendation: (b).**
- **Against:** the 5% tie cap is new. It is not in the qualified code and has never been tested on these data. It would need a unit test and might exclude a few basins in which the excluded hours are real floods.

### 2e. Ties (D6)

- **Exact:** if the aggregate is exactly equal (float64), take the earliest epoch. This matches the historical NSE rule. For arm 2's 36 (λ, epoch) candidates: earliest epoch first, then the smallest λ (review fix).
- **Tolerance:** take the earliest epoch within δ of the best.
- **Recommendation: exact**, the same for both picks.
- **Against:** a median over ~2,000 basins almost never ties exactly, so the rule is nearly inert. A later epoch can win by a margin far below noise. A tolerance rule would favour earlier, cheaper epochs, but needs a δ that nobody has measured.

### 2f. Epoch 11/12 flag (D7)

- Any pick (NSE or flood, either arm) at epoch 11 or 12 is flagged "budget edge" in the per-cycle table. The full 12-epoch curves of median NSE and the flood metric on the **checkpoint year** are always reported. Assessment-year scores are never computed per epoch.
- Nothing is retrained automatically. The budget stays at 12 epochs × 50k updates for this rotation.
- **Strength-edge flag (Omri, review round C):** an arm-2 pick at λ = 0.5 or λ = 8 is flagged "strength edge". If 2 or more of 5 cycles pick the same edge under one pick rule, the strength range is labelled "possibly too narrow" and returns to Omri. Nothing new is run automatically.
- **Recommendation:** if 3 or more of the 5 cycles of an arm are flagged under one pick, that arm's result is labelled "possibly limited by the training budget", and the budget question returns to Omri as a new decision.
- **Against:** 3/5 is an arbitrary count. A single flagged cycle may already mean the model was still improving.

---

## 3. Arm 2: magnitude-weighted NSE loss

Agreed in principle on 2026-11-11: each hour's error is weighted by its observed flow, normalised per basin, with a tunable strength. Same rotation, same metrics. Not launched before this spec is approved.

### 3a. Weight formula (D8, D9, D10)

Let q_t be the observed lead-6 target of hour t (area-normalised depth; the ratio below is the same in m³/s, since area is constant). Let Q98_b be basin b's 98th percentile of q over the cycle's **training** targets (after the leak-fix blanking, the same values the per-basin std uses).

| Option | Weight before normalisation | Problem |
|---|---|---|
| **W1 (additive ratio)** | 1 + λ · min(q_t / Q98_b, c) | none serious; see the argument against |
| W2 (power) | (q_t / mean_b)^γ | a dry hour gets weight 0, so false floods in dry periods are not penalised (ephemeral basins); huge spread |
| W3 (step) | 1 + λ · [q_t ≥ Q98_b] | ignores how far above the threshold; jumps at the threshold |

**Normalisation (per basin):** divide by M_b = the mean of the unnormalised weight over basin b's training targets. Each basin's average weight is then 1, so it contributes to the loss as much as in the baseline; only the share between its quiet and high hours changes.

Final weight: **w_t = (1 + λ · min(q_t / Q98_b, c)) / M_b**, inside the unchanged NSE loss: w_t · (ŷ − y)² / (σ_b + 0.1)².

**Dry basins (Omri, review round A):** if Q98_b over the training targets is 0, the basin gets w_t = 1 for every hour (no emphasis; M_b = 1). Such basins are counted and listed in the training receipt.

Properties: λ = 0 is exactly the baseline. Every hour keeps a weight of at least 1/M_b > 0, so over-prediction in quiet periods still costs. Only observed flow and training years enter the weight; no checkpoint or assessment observation is read.

- **D8 formula. Recommendation: W1.**
- **D9 reference flow. Recommendation: the basin's training-year Q98.** Then "λ" reads simply: before normalisation, an hour at the basin's Q98 counts (1 + λ) times a dry hour. The alternative reference, mean flow, gives flashy basins (high Q98/mean) much larger weights than smooth ones.
- **D10 cap. Recommendation: c = 5** (flows above 5 × Q98 weigh the same as 5 × Q98). Report the fraction of training hours capped per basin. Without a cap, one extreme hour can carry a weight in the hundreds and dominate a batch.
- **Strongest argument against weighting by observed flow at all:** a model trained with weights that grow with the observed flow learns to predict above the expected flow whenever it is uncertain. This is what lifts the peaks, but it also adds a systematic upward bias at moderate flows and more false rises. That is why the below-Q98 bias guard (§1a) and PBIAS are in the comparison. Against the cap specifically: it reduces emphasis on exactly the largest floods, the ones of most interest.

### 3b. Strength values and how λ is chosen without touching assessment years (D11, D12)

**Values (D11). Recommendation: λ ∈ {2, 8}.** Before normalisation, an hour at Q98 counts 3× (λ = 2) or 9× (λ = 8) a dry hour, and an hour at the cap counts 11× or 41×. The two values are a factor of 4 apart, so the pair shows whether a mild and a strong emphasis behave differently. Against: no measurement supports these values; they are chosen for their spread. A third value (e.g. 0.5) would show the mild end, at extra cost.

**Choosing λ (D12):**

| Option | How | Clean? | Extra cost over the baseline (~170 L4 GPU-h) |
|---|---|---|---|
| S1 | one λ fixed now, nothing chosen (if chosen: λ = 4, the geometric middle of {2, 8}) | yes | +~170 GPU-h |
| **S2** | both λ trained in every cycle; in each cycle the pick rule chooses jointly over the 24 (λ, epoch) checkpoints on **that cycle's checkpoint year** | yes | +~340 GPU-h |
| S3 | as S2 with three λ values | yes | +~510 GPU-h |
| S4 | try λ on one cycle, then freeze it for the others | **no**: every checkpoint year is another cycle's assessment year (e.g. cycle 4 checks on 2025, cycle 5 assesses 2025), so one assessment year would be used to choose λ | +~200 GPU-h |
| S5 | choose one λ from all five checkpoint years pooled | **no**, for the same reason, in all five cycles | as S2 |

Cost basis: preflight §(e): ~14 GPU-h training plus ~20 GPU-h per-epoch scoring on 2,307 basins, per run.

- **Recommendation: S2.** Both λ results are also reported separately as secondary rows, so the effect of strength is visible.
- **Against:** S2 doubles the arm's compute. It also gives arm 2 twice as many checkpoints to choose from as the baseline on the checkpoint year. That does not bias the assessment year (the choice is made out of sample), but on the checkpoint year arm 2 looks better by construction. S1 is half the cost; if its single λ is badly placed, though, a null result says little.

### 3c. Arm 2 metrics and checkpoint rule (D13)

- **Recommendation:** identical to §1 and §2. Each pick rule (NSE pick, flood pick) chooses among the 36 (λ, epoch) checkpoints in each cycle (D11/D12 as chosen: 3 λ × 12 epochs). Same gates G1-G4, the same Q98 facts as the baseline run of that cycle, and the same epoch-11/12 flag.
- **Against:** the arm is trained to emphasise floods, so one could argue only the flood pick is meaningful for it. Keeping both picks costs nothing and keeps the comparison symmetric.

### 3d. How it plugs into NH 1.13 (plan only, no code)

- **No loss registry in NH 1.13 (corrected after review).** `neuralhydrology/training/__init__.py:get_loss_obj` is a fixed if-chain, and it is imported by name in **two** places:
  - `BaseTrainer` (`basetrainer.py:21`, used at `:88`);
  - the evaluation `Tester` (`tester.py:25`, built at `:78`, called on every evaluation at `:496`/`:569`, including the per-epoch checkpoint-year evaluation).

  Both call sites must resolve the new loss name; patching only the trainer makes evaluation raise NotImplementedError.
- **The per-basin std must still be computed.** NH computes it only when `cfg.loss.lower() in ['nse', 'weightednse']` (`basedataset.py:749`). Under a new loss name, samples carry no `per_basin_target_stds` and the loss raises a KeyError. The Flash-NH dataset must compute it for the new loss name too. Flash-NH already extends NH explicitly at registration (`src/baseline/nh_register.py`: dataset registration, config properties, config-load checks). The new loss is added the same way: a new loss name (e.g. `flashnh_magnitude_weighted_nse`) that the patched lookup resolves, while every other loss name goes to NH unchanged.
- **Loss class.** A subclass of `MaskedNSELoss` (`loss.py:217`): same NaN mask, same per-basin std term, same eps = 0.1, same mean reduction, multiplied by w_t. **Test:** with λ = 0 it reproduces `MaskedNSELoss` exactly.
- **Where the weight comes from (two routes):**
  - **(i) Recommended: per-basin constants.** The **training** dataset computes Q98_b and M_b from the same blanked training targets as the per-basin std (NH `basedataset.py:532`, `_calculate_per_basin_std`). They are written to the run (training receipt / `train_data`). Evaluation datasets (checkpoint year) **reuse the saved training constants and never recompute them**, so no checkpoint- or assessment-year observation enters a weight. The evaluation-time loss is a training diagnostic only (review fix). It sends them with each sample, as NH already does for `per_basin_target_stds` (`basedataset.py:205`). Since the target is scaled globally, q_t / Q98_b = a_b · y_scaled + b_b with two per-basin constants. The loss computes w_t from the target it already holds, so no extra time-slicing is needed.
  - (ii) A per-hour weight tensor emitted by the dataset. It must be cut to the last `predict_last_n` step exactly like `y`, which adds a place to get wrong.
- **Config keys** (e.g. `flashnh_magnitude_weight_strength`, `flashnh_magnitude_weight_cap`): refused unless `dataset: flashnh` with a split contract, matching the existing config-load guard.
- **Receipt.** The training receipt records the λ, the cap, the per-basin Q98_b and M_b, and the fraction of capped hours, so the weights can be audited without opening scores.
- **Unchanged:** evaluation and scoring (weights act only in training), inputs, scaler, sequence length, budget, seed. NH's own validation loss will differ between arms; it stays a training diagnostic.
- **Tests and route:**
  - λ = 0 identity;
  - mean weight 1 per basin over the finite blanked training targets in the dataset's `xr`, the series M_b is computed on. Lookup-table sampling drops some hours, so the realised mean over sampled hours is approximately 1 and is reported, not tested (review fix);
  - both the trainer and the tester resolve the new loss name;
  - evaluation datasets reuse the saved training constants;
  - no observation outside the training windows read (extend `tests/test_nh_split_sealing.py`);
  - refusal outside a contract.
  - Then a code review (gate 4) and a score-blind arm-2 canary (receipts, weight statistics, timing, memory; no score opened).

---

## 4. Comparison and reporting rules (written before any score)

### 4a. What is compared

- Per cycle there are up to four chosen models: baseline NSE pick, baseline flood pick, arm-2 NSE pick, arm-2 flood pick.
- Every comparison is **paired per basin-year**: same basin, same assessment year, same admitted hours, same Q98 facts (G4).
- Seed A only in every run (preflight decision 5).

### 4b. Which comparison is primary (D14)

| Label | Comparison | Status (recommended) |
|---|---|---|
| **C1** | arm 2 vs baseline, both with the flood pick | **primary**: the arm's aim is floods, and like is compared with like |
| C2 | arm 2 vs baseline, both with the NSE pick | secondary |
| C3 | baseline flood pick vs baseline NSE pick | descriptive: what the pick rule alone changes |
| C4 | the per-λ rows: each λ (0.5, 2, 8) picks its own epoch from its 12, under both pick rules, and is compared with the baseline in the same paired way (Omri, review round D) | descriptive only; never used for the §4d claim |

**Headline of the baseline summary (also D14).** Options: the NSE pick (continuity with every earlier result), or the flood pick. **Recommendation: report both with equal weight, and name the NSE pick the reference baseline.** This keeps it comparable with the HPO, length and RD1 history.
**Against:** if C1 (flood pick) is the primary arm comparison but the reference baseline is the NSE pick, readers see two "baselines". Choosing the flood pick for both would be simpler, but breaks continuity.

### 4c. Statistics reported for each comparison

For the primary flood metric, NSE, and each secondary metric:
- the pooled median of the paired difference over basin-years;
- the fraction of basin-years improved;
- the 5 per-cycle medians of the paired difference, and in how many cycles (k/5) the sign agrees;
- a 95% interval from a basin bootstrap. Details (review fix):
  - resample the 2,307 basins with replacement, keeping all 5 years of each basin together;
  - 2,000 resamples, numpy `default_rng(20261115)`;
  - percentile interval (2.5th, 97.5th) of the resampled pooled median paired difference;
  - basin-years where either arm's metric is undefined or flood-ineligible are dropped in pairs, before resampling;
  - a resampled basin with no remaining basin-years contributes nothing.

### 4d. Decision wording, fixed in advance (D15)

**"Arm 2 improves flood skill"** only if all three hold for C1:
1. The pooled median paired difference of the primary flood metric favours arm 2, with the 95% bootstrap interval excluding 0.
2. The same sign holds in at least 4 of 5 cycles.
3. Overall guard: the pooled median paired ΔNSE ≥ **−0.02**, and the guard on quiet hours holds (Omri, review round B). For each basin-year, compute Δ = |PBIAS_below-Q98, arm 2| − |PBIAS_below-Q98, baseline|. The pooled median of Δ, over basin-years where both are defined, must be ≤ **+5 percentage points**. Basin-years with no hours below the Q98 have an undefined Δ and are left out.

Scale reference for −0.02: about one third of the L120 → L240 gain (+0.060) on 2024. **Both tolerance numbers are proposals and are Omri's to set.**

Other outcomes are worded exactly as:
- **"flood gain with an overall cost"**: 1 and 2 hold, 3 fails;
- **"no clear flood gain"**: 1 or 2 fails, and the difference is not clearly against arm 2;
- **"worse on floods"**: the interval excludes 0 against arm 2.

No outcome changes the baseline or adopts arm 2 by itself. Adoption is a separate user decision, preceded by gate 3 review (`agent_handoff_rules.md` §3.2).

- **Recommendation:** this wording.
- **Against:** with a single seed, a difference can pass rules 1-2 and still be smaller than the seed-to-seed spread. The 3-seed L120 screen of 2026-10-17 measured that spread on 2024 only and on 400 basins, so it cannot be applied directly. The report will quote it beside the result as context, not as a test.

### 4e. Report contents (fixed)

- **Tables:** a per-cycle table and a pooled table for all metrics, picks and arms.
- **Per-year marks and sensitivities:** the WY2024/WY2025 marks; the S-years and S-variance sensitivities.
- **Coverage:** per year, the counts of eligible, ineligible and undefined basins, with reasons.
- **Training curves:** the full 12-epoch curves on the **checkpoint year only**; the epoch-11/12 flags; the chosen λ per cycle. The assessment year is scored only at the picked checkpoints, and only after both picks of every arm in that cycle are fixed.
- **Figures:** hydrographs for basins chosen by a fixed rule: the basins at the 10th, 50th and 90th percentile of the C1 paired flood difference, pooled, one figure per basin, with the baseline and arm 2 overlaid on the same axes. Figures illustrate; they do not decide.

### 4f. WY2026 (out of scope here, noted so it is not forgotten)

The checkpoints that go to WY2026 must be fixed before WY2026 is opened. Recommendation for that later decision: all chosen checkpoints of both picks and both arms, each scored once, none chosen using WY2026. This does not need deciding before the rotation launches.

---

## 5. What this spec does not do

- No change to the split contract, sealing, inputs, scaler, budget or seed.
- No event-based metrics and no peak-timing alignment (deferred on 2026-11-11).
- No CMAL or asymmetric loss (options (c) and (d) of the meeting brief; not chosen).
- No code. §0's scorer gap and §3d's loss are plans to be implemented after approval, with tests and review.

## 6. Decision list (Omri fills the "Choice" column)

| # | Decision | Recommendation | Choice |
|---|---|---|---|
| D1 | Per-basin metric set and scoring gates G1-G4 (§0, §1a) | as listed; NSE primary overall, Q98-nRMSE primary flood, below-Q98 PBIAS as guard | **as recommended** |
| D2 | Pooling and sensitivities (§1b-c) | median over basin-years (a); S-years, S-variance | **as recommended** |
| D3 | Flood pick metric (§2b); confirm the NSE pick (§2a) | F-nRMSE, no NSE guard; NSE pick unchanged | **as recommended** |
| D4 | Flood threshold (§2c) | T1: that year's own Q98 per basin | **as recommended** |
| D5 | Flood eligibility (§2d) | (b): Q98 > 0, ≥ 50 flood hours, ≤ 5% of hours flagged; flag if < 50% of basins eligible | **as recommended** |
| D6 | Ties (§2e) | exact tie → earliest epoch | **as recommended** |
| D7 | Epoch 11/12 flag (§2f) | flag every pick; ≥ 3/5 cycles → "possibly budget-limited", back to Omri | **CHANGED: ≥ 2/5 cycles** → "possibly budget-limited", back to Omri |
| D8 | Arm-2 weight formula (§3a) | W1 additive ratio, normalised to mean 1 per basin | **as recommended** |
| D9 | Reference flow (§3a) | training-year Q98 per basin | **as recommended** |
| D10 | Cap (§3a) | c = 5 | **as recommended** |
| D11 | Strength values (§3b) | λ ∈ {2, 8} | **CHANGED: λ ∈ {0.5, 2, 8}** (3 arm-2 runs per cycle) |
| D12 | How λ is chosen (§3b) | S2: per cycle, on its own checkpoint year, jointly with the epoch | **per cycle, own checkpoint year, jointly with the epoch**; with D11 this is option S3 (36 candidates per pick, ≈ +510 GPU-h) |
| D13 | Arm-2 metrics and checkpoint rule (§3c) | identical to baseline, both picks | **as recommended** |
| D14 | Primary comparison and baseline headline (§4b) | C1 (flood pick) primary; NSE pick is the reference baseline, both reported | **as recommended** |
| D15 | Decision wording and tolerances (§4d) | as written; ΔNSE ≥ −0.02, below-Q98 PBIAS ≤ 5 pp worse | **as recommended** (−0.02; 5 pp) |
| D16 | NH plug-in route (§3d) | per-basin constants (i), loss registered via `nh_register` | **as recommended** |

### Review round (gate 1, fresh Claude reviewer, 2026-11-15; answers 2026-11-16)

The verdict was "sound with changes". The following were applied as fixes:
- §3d plug-in: both trainer and tester patched; per-basin std computed; constants train-only, saved and reused;
- §0 scorer gap and date convention;
- no-admitted-hours ineligibility;
- bootstrap details and the λ tie-break;
- curves on the checkpoint year only;
- the mean-weight test definition;
- body text aligned with §6;
- draft entry rule 1 ("in arm 2's favour", basin-years).

Omri's answers to the open questions:

| # | Question | Choice |
|---|---|---|
| A | Weight when the training Q98 is 0 | weight 1 for every hour (no emphasis); listed |
| B | Quiet-hour bias guard statistic | median over basin-years of the paired Δ\|PBIAS below Q98\| ≤ +5 pp |
| C | Strength-edge flag | flag λ = 0.5 / 8 picks; ≥ 2/5 cycles at the same edge → "possibly too narrow", back to Omri |
| D | Per-λ secondary rows | own 12-epoch picks (both rules), paired vs baseline, descriptive only |
| E | < 50% flood-eligible in a cycle | flood pick still counts; flagged; counts published before scoring |
| F | Cross-arm observations | bit-identical, checked score-blind before any metric; a mismatch stops the comparison; no tolerance |
| G | Low-variance sensitivity units | variance in mm/h |

Choices given by Omri on 2026-11-15 in a question-by-question walkthrough in session local_a8e15ba7 (D8+D9 and D13+D16 were asked together; D15 was asked as two tolerances). Compute implied: baseline ≈ 170 + arm 2 ≈ 510 ≈ **680 L4 GPU-h** for the rotation, before the WY2026 scoring.
