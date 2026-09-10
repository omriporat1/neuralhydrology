# Stage-1 v2 — Prospective 12+12 Scientific Review Design (RD1) — v001

**Status: FROZEN prospective design contract (RD1-C1).** User-authorized on
2026-09-09, before any substantive Bayesian Proposal 9–12 (P9–P12) or
random-control Wave-3 (R9–R12) outcome was examined in the scientific-review
lane. This document is the authoritative detailed specification for the
planned formal 12-plus-12 scientific review of the Stage-1 v2 six-axis HPO
campaign. It converts the already-approved RD1 methodology into one durable
contract so the review machinery cannot be adapted post hoc to later
outcomes.

This is a **design freeze only**. It implements no analysis code, launches no
compute, contacts no remote system, and authorizes no campaign step. The real
12+12 scientific review happens only after the execution lane declares the
valid 12+12 campaign complete and the required evidence is available.

It was preceded by the read-only RD1-A interface archaeology (see the RD1-A
completion report; `docs/FLASHNH_CURRENT_STATE.md` records its closure). RD1-A
established how this design maps onto the repository's actual producers,
artifacts, schemas, and reusable analysis code; §20 below records its central
data-contract finding.

---

## 0. Label legend

Every methodological statement in this document carries one of four bindings.

- **FROZEN** — a methodological decision approved by the user before the final
  P9–P12 review. It may not be changed by an implementing agent, by ChatGPT,
  or by a later review conversation without a fresh user authorization. Later
  P9–P12 outcomes are explicitly *not* a valid reason to change a FROZEN rule.
- **SECONDARY** — a diagnostic that is prespecified and will be produced, but
  is not the HPO objective and cannot redefine trial validity, the optimizer
  objective, configuration ranking, or promotion rules.
- **EXPLORATORY** — an analysis that may generate hypotheses but cannot
  retroactively change the review contract or any scientific conclusion about
  the campaign.
- **DOWNSTREAM / NOT AUTHORIZED** — a future experiment, range change,
  promotion, SHARED-A5 consumption, or additional compute that this document
  explicitly does **not** authorize and must not be read as authorizing.

Where a section does not label a sub-item, the section's dominant label
applies.

---

## 1. Scientific foundation

### 1.1 Common-120 is a temporal-support contract — **FROZEN**

"Common-120" names a **temporal-support contract**, not a basin count. For
this screening HPO checkpoint:

- evaluation population = the frozen **screening-400** basins (screening
  basin-ids SHA-256 `d4395d93ebc567cf09e149c0121463d75cf4f7ecc02c07a7c4a7999763baa372`,
  policy `stage1_provisional_operational_screening_subset_v001`);
- validation period = the frozen 2024 validation window;
- every configuration is evaluated on the **same** Common-120 temporal
  support (fixed-support contract SHA-256
  `cb4ebe86afa501ef3d5929ead5b455f8df06e7d38b58ebf4148f8545fe6851ef`,
  schema `flashnh_stage1_v2_fixed_support_contract` v2, 120 h look-back
  floor);
- the **official objective** for every configuration is the best eligible
  epoch's `flashnh/common120_raw_space_nse_v001` — the median of the 400
  per-basin raw-space NSE values over that fixed support;
- shorter-sequence **natural support** (each trial's own admitted set) is
  **diagnostic only** and never becomes the optimizer objective or an
  alternative ranking.

The screening-400 is **tuning / development-validation data**, not an
independent final test. The later ~2,307-basin Common-120 development-
population audit is a separate artifact and remains governed by **SHARED-A5**
(screening-400 is adequate as the economical HPO screening population but is
**not** authoritative for exact within-top-group ordering, absolute expected
full-development NSE, checkpoint selection, or promotion). **No 12+12 finding
may bypass that downstream gate. — DOWNSTREAM / NOT AUTHORIZED** for anything
SHARED-A5 reserves.

### 1.2 Unit of scientific evidence — **FROZEN**

There are **12 valid Bayesian configurations** and **12 valid random-control
configurations**.

- The 400 per-basin NSE values for one configuration characterize
  **hydrological heterogeneity across basins**. They are **not** 400
  independent HPO replicates.
- The 24 configurations are **not** 24 independent draws from one common
  sampling process: random-control configurations come from the frozen IID
  random design (manifest SHA-256
  `59be6726b60863aeed1c25e86782bd7a5e1623434ce23c956e66eb54c527c095`); later
  Bayesian configurations are adaptively conditioned on earlier Bayesian
  outcomes.
- Therefore an ordinary two-sample t-test / Mann–Whitney test **is not the
  primary evidence** for "Bayesian beats random". Algorithm-level formal
  inference would require replicated full search campaigns, which do not
  exist here.
- Only model **Seed A** (`model_seed = 967139`) is represented. **Seed
  sensitivity is not measured by this checkpoint** and must not be asserted
  or denied from it.
- The two arms are **not paired** by proposal order. Bayesian order is an
  adaptive (dependent) sequence; random order is the frozen manifest order.

Primary evidence is therefore **descriptive and distributional**: where each
arm's configuration-quality distribution sits, what the search produced over
its ordered sequence, and how candidates behave hydrologically — always
displaying the underlying observations.

---

## 2. What the checkpoint must distinguish — **FROZEN**

The review must separate at least these nine questions and never collapse
them into a single verdict:

1. **Champion discovery** — the single best objective score found.
2. **Typical search quality / yield** — the distribution of configuration
   quality per arm.
3. **Elite-candidate yield** — whether several strong configurations were
   produced, not just one.
4. **Post-incumbent return** — what evaluations *after* an early strong
   incumbent actually bought (objective, elite yield, compute).
5. **Search learning / concentration** — Bayesian proposal behaviour over the
   ordered sequence.
6. **Landscape coverage / context** — including the random arm as the
   coverage/dispersion reference.
7. **Hydrological robustness** — per-basin NSE distributions, KGE, high-flow
   (Q98) diagnostics.
8. **Training stability** — best vs final performance and trajectory
   behaviour within the frozen 12-epoch fidelity.
9. **Axis / interaction evidence** — what can actually be inferred about the
   six-dimensional search space, and with what confidence.

Constraints:

- **Trial 2 must remain visible.** Do not censor P2 or R2 because both arms'
  current incumbents were found at proposal order 2.
- Historical **4 / 8 / 12** checkpoint markers may be shown for continuity,
  but arbitrary **1–4 / 5–8 / 9–12** blocks must **not** become scientific
  pseudo-replicates.
- Do **not** invent a retrospective "bad configuration" threshold merely
  because later outcomes make one convenient.

---

## 3. Frozen references — **FROZEN**

### 3.1 Configuration-level near-incumbent anchor

The arm-neutral reference is the **best configuration known before P9 began**:

- **P2**, official pre-P9 objective **`0.4388098707961096`** (best epoch 9).

Verified from committed pre-P9 evidence (`docs/decision_log.md`,
`docs/FLASHNH_CURRENT_STATE.md`, and the retained post-P8 8+8 review table).
This is the frozen reference for descriptive strong / near-incumbent
candidate yield.

**Descriptive review margins** (NSE, absolute, relative to P2):

- primary: within **`0.01`** of P2;
- sensitivity: within **`0.005`** of P2.

These margins are:

- descriptive review margins **only**;
- **not** statistical-equivalence bounds;
- **not** promotion thresholds.

Threshold-free top-k summaries (§4) must also be reported alongside them.

### 3.2 Basin-wise anchors

For paired per-basin diagnostics, **both P2 and R2** are retained as frozen
anchors:

- **R2**, official pre-P9 objective **`0.43276757946464195`** (best epoch 6).

The intent is to avoid narrating every hydrological result relative only to
the Bayesian champion. Paired basin differences are reported against **each**
of P2 and R2 (§11).

Both P2 and R2 are preserved as frozen paired-basin reference configurations
regardless of any later P9–P12 outcome.

---

## 4. Configuration-level primary review package — **FROZEN**

For the 24 valid configurations, prospectively require:

- individual official objective vs proposal order (per arm);
- cumulative best objective vs proposal order (per arm);
- cumulative GPU-hours vs proposal order (per arm; see §8);
- objective-score distribution by arm;
- transparent distribution summaries: median, IQR, and the full set of
  underlying points (min / quartiles / max, or an equivalent transparent
  summary that displays the observations);
- **top-3 floor** = third-best configuration score (per arm);
- **top-3 mean** (per arm);
- **top-3 median** (per arm);
- count of configurations within **`0.01`** of frozen P2 (per arm and
  pooled);
- count of configurations within **`0.005`** of frozen P2 (per arm and
  pooled);
- cumulative near-incumbent count vs proposal count;
- cumulative near-incumbent count vs GPU-hours;
- **post-incumbent** objective distribution (configurations evaluated after
  each arm reached its running incumbent);
- post-incumbent top-3 / near-incumbent yield;
- post-incumbent compute expenditure;
- **unique-configuration count** as well as proposal / run count (identical
  six-axis coordinates, if any, are preserved and counted — not
  de-duplicated away);
- training-stability diagnostics (§19).

Threshold-free top-k summaries are primary; the P2-referenced margin counts
are a complementary descriptive view, not a gate.

---

## 5. Compute comparison — **FROZEN**

- **GPU-hours / configuration-equivalent compute is the primary cost basis.**
- **Wall-clock elapsed time is not the primary comparison** — random
  configurations may have been executed with different parallelism than the
  sequential Bayesian proposals.
- Require **exact per-trial GPU-hours wherever authoritative Slurm / sacct
  evidence allows it** (per-trial completion bundles retain `phase1_sacct.txt`;
  the automatic review record's `gpu_hours` field is frequently `null` and is
  not itself authoritative).
- If exact GPU-hours are missing for some configuration: **do not invent a
  value**; mark it **unavailable**; report coverage (how many of 24 have
  exact accounting); and keep exact accounting visually distinct from
  qualified campaign-level approximations.

The compute review supports: champion efficiency; strong-candidate yield per
unit compute; elite-set efficiency; post-incumbent return.

---

## 6. Frozen six-axis configuration geometry — **FROZEN**

The frozen v2 search domain is defined authoritatively in
`src/baseline/sweep_v2_six_axis_campaign.py` (`SEARCH_DOMAIN_V2`,
`SEQ_LENGTH_DOMAIN_V2`) which inherits the five v1 axes from
`src/baseline/sweep_v1_campaign.py` (`SEARCH_DOMAIN`). **This document does
not redefine the domain — the code is authoritative for exact allowed
values.** For reference only, the frozen domain is: `learning_rate`
log-uniform `[1e-4, 1e-3]`; `hidden_size` categorical `{64, 128, 256}`;
`embedding_dropout` uniform `[0.0, 0.4]`; `output_dropout` uniform
`[0.0, 0.4]`; `batch_size` categorical `{128, 256, 512}`; `seq_length`
`q_uniform` on the grid `{48, 60, 72, 84, 96, 108, 120}` (step 12).

### 6.1 Per-axis normalization to `[0, 1]`

- **learning rate** — `log10`, normalized across the frozen `[1e-4, 1e-3]`
  bounds;
- **hidden size** — `log2`, normalized across the frozen allowed range
  (`log2(64) … log2(256)`);
- **embedding dropout** — linear over `[0.0, 0.4]`;
- **output dropout** — linear over `[0.0, 0.4]`;
- **batch size** — `log2`, normalized across the frozen allowed range
  (`log2(128) … log2(512)`);
- **sequence length** — linear / ordinal over the frozen 48–120 grid.

### 6.2 Configuration distance

Over the six normalized axes `k = 1..6`:

```
d(x, y) = sqrt( mean_k( (x_k - y_k)^2 ) )
```

**Equal axis weighting is a transparent descriptive geometry. It is NOT
evidence that the six axes have equal scientific importance.** This must be
stated wherever the distance is reported.

### 6.3 Bayesian sequence quantities

For Bayesian proposal `i` (in adaptive proposal order):

- **novelty(i)** = minimum `d(config_i, config_j)` over all earlier *valid
  Bayesian* configurations `j < i`;
- **incumbent distance(i)** = `d(config_i, config_incumbent)` where
  `config_incumbent` is the best-known *Bayesian* configuration available
  **immediately before** proposal `i` (by objective score among valid
  Bayesian configs with order `< i`).

For the **random arm**, the same geometry may characterize **static coverage
/ dispersion** (e.g. nearest-neighbour spacing, spread relative to the
Bayesian cloud), but **random proposal order must not be interpreted as
optimizer learning** — there is no "incumbent distance" narrative for the
random arm.

---

## 7. Whole-sequence search-process analysis — **FROZEN**

The review **privileges the complete ordered sequence**. Early-half /
late-half segmentation is **not** a primary scientific method.

- Legacy `sweep_v1_review_analysis.proposal_drift_evidence` (hard-coded
  `half = len(ordered)//2` split plus threshold gates) and the legacy
  `fig05` / `fig06` early-vs-late rendering **may remain low-level historical
  diagnostics** but must **not** determine the RD1 scientific
  interpretation.
- Preferred sequence evidence exposes raw or minimally transformed
  quantities:
  - proposal order vs objective;
  - proposal order vs each normalized axis;
  - proposal order vs novelty;
  - proposal order vs incumbent distance;
  - boundary occupancy over the sequence;
  - cumulative best;
  - cumulative yield (near-incumbent count).
- Any rank correlation, trend summary, or similar statistic must remain
  **descriptive** and be displayed **together with the underlying
  observations**, never as a standalone number.

---

## 8. Basin-level NSE package — **FROZEN**

For **every valid configuration at its official best eligible epoch**, the
review requires the complete mapping:

```
basin_id -> Common-120 fixed-support raw-space NSE
```

over the frozen 400-basin screening population.

**Required core summary set** (per configuration):

- Q1, Q5, Q25, Q50, Q75, Q95, Q99;
- IQR (`= Q75 - Q25`);
- finite / eligible basin counts.

`P10` / `P90` and other existing `percentile_diagnostics` summaries **may
additionally be retained** but are **not substitutes** for this frozen core
set.

**Planned scientific views** (all FROZEN as part of the package):

- configuration × basin-quantile heatmap;
- basin quantiles vs configuration / proposal order;
- selected ECDF overlays;
- paired NSE-difference distributions vs **P2**;
- paired NSE-difference distributions vs **R2**;
- fraction of basins improved relative to **each** reference.

### 8.1 Basin paired-difference classification — **FROZEN**

Do **not** import the old scratch-analysis `±0.01` basin-level "tie"
convention.

For paired basin differences, with `delta = NSE_candidate - NSE_reference`:

- **improved**: `delta > tolerance`;
- **worse**: `delta < -tolerance`;
- **tied**: `abs(delta) <= tolerance`.

`tolerance` exists **only** for floating-point numerical equality. It is a
tiny, documented, machine-precision-scale value (e.g. `1e-12`, to be fixed
exactly at implementation time in RD1-C3) — **not** a scientific NSE margin.
The `0.01` / `0.005` margins from §3.1 are **configuration-level**
near-incumbent margins only and never apply to basin-level differences.

Display the **full paired-difference distribution** and robust summaries
(median, quartiles, ECDF); do not reduce the analysis to improved / worse
counts.

---

## 9. Best epoch versus all epochs — **FROZEN**

- The core 12+12 hydrological comparison is frozen at **each configuration's
  official best eligible epoch**.
- The real-data per-basin / per-timestamp consumer is therefore designed
  around **24 best-epoch validation-result products**, one per valid
  configuration.
- The core basin / KGE / Q98 review does **not** require all 12 epochs × 24
  configurations.
- The already-retained **scalar** fixed-support epoch trajectories (12 medians
  per configuration, both supports) are **sufficient** for the core stability
  / training-trajectory analysis (§19).
- **Deletion of non-best-epoch validation products is NOT authorized.** All
  already-produced validation products are preserved until the
  scientific-review workflow is safely closed (§21).

---

## 10. Natural-support trajectory — **SECONDARY**

- Natural-support results are strictly secondary.
- The main decision board and the primary objective figures use **Common-120
  fixed support**.
- Natural-support trajectories may appear only in a clearly separated
  appendix / secondary-diagnostic section, particularly to diagnose
  sequence-length / support effects.
- Every natural-support display must state, on the display: **diagnostic
  only; not the HPO objective; not an alternative configuration ranking; not
  a second optimization target.**

---

## 11. Full-support KGE diagnostics — **SECONDARY**

- For each configuration / basin at the **official best epoch**, evaluate KGE
  on the **same Common-120 fixed support** as the objective.
- Use the already-qualified repository implementation of KGE-2009 and its
  components — `src/baseline/nh_raw_space_evaluation.raw_space_metrics`
  (`kge`, `kge_r`, `kge_alpha = sigma_sim / sigma_obs`,
  `kge_beta = mu_sim / mu_obs`), as propagated per-basin by
  `fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics`. Do
  **not** create parallel KGE math.
- KGE and its components are **secondary hydrological diagnostics**. They can
  explain *why* configurations with similar NSE behave differently, but they
  **cannot** redefine trial validity, the optimizer objective, configuration
  ranking, or promotion rules.

---

## 12. Frozen Q98 high-flow methodology — **FROZEN** (package A–C, E) / **SECONDARY** (D) / **EXPLORATORY** (high-flow KGE, events)

### 12.1 Per-basin mask construction — **FROZEN**

For each basin independently:

1. take **only** that basin's frozen Common-120 admitted 2024 observed
   discharge values;
2. compute its empirical Q98 as
   `np.quantile(observed_values, 0.98, method="linear")`;
3. define the fixed high-flow mask `observed >= Q98`;
4. **retain ties** (values exactly equal to the threshold are included);
5. use this **same observed-derived mask for every configuration** — the
   mask is **candidate-independent**.

### 12.2 Prespecified high-flow timestamp-level package

**A. Q98-normalized RMSE — FROZEN**

`RMSE(high_flow) / observed_Q98_threshold`

(the denominator is the per-basin Q98 threshold value, **not** the observed
high-flow mean).

**B. Relative volume bias — FROZEN**

`sum(sim - obs) / sum(obs)` over the frozen high-flow subset.

**C. Observed-peak-time magnitude error — FROZEN**

Find the timestamp `t_obs_peak` of the **maximum observed** discharge within
the frozen admitted high-flow subset. Then compute

`[ sim(t_obs_peak) - obs_peak ] / obs_peak`.

**Do not independently search for a simulated peak timestamp for this
metric.**

**D. High-flow NSE — SECONDARY, caveated**

Included as a secondary, explicitly caveated diagnostic. Its
variance-normalization can be unstable on a conditional extreme subset.

**E. High-flow sample count — FROZEN**

Always reported.

### 12.3 Guard rails — **FROZEN**

- Minimum high-flow sample count for high-flow NSE and any KGE-like
  conditional metric: **50 timestamps**.
- Required observed variances / denominators must be **finite and nonzero**;
  otherwise the affected metric is reported as **NaN / undefined**.
- A secondary high-flow diagnostic failure **never** invalidates an otherwise
  valid HPO trial.
- **Diagnostic coverage is always reported** (how many of the 400 basins × 24
  configurations produced each metric).

### 12.4 Exploratory boundary — **EXPLORATORY**

- **High-flow KGE** is exploratory, not part of the prespecified high-flow
  core.
- Event-based peak / timing / hydrograph diagnostics are exploratory unless
  separately frozen later.

---

## 13. Legacy high-flow code — **FROZEN note**

`src/baseline/high_flow_event_metrics.py` **predates this RD1 methodology and
is NOT scientifically authoritative for the new Q98 package.** RD1-A
established concrete semantic differences:

- conditional NSE is intentionally suppressed there;
- NRMSE is normalized by the observed high-flow **mean**, not the Q98
  threshold;
- a different sample-count gate (`min_n_for_correlation = 10`, not 50);
- **independent** simulated-peak selection (`argmax(sim)`), not sim at the
  observed peak time;
- event-window semantics rather than timestamp-level conditional masking.

Reusable **low-level primitives** may later be retained (e.g. a finite-value
quantile helper; `raw_space_metrics` as an inner call), but the RD1 Q98
implementation must **not** silently inherit these old semantics. The RD1 Q98
package is implemented as a new narrow timestamp-level helper (RD1-C4).

---

## 14. `seq_length` upper-bound interpretation — **FROZEN**

The RD1 interpretation of `seq_length = 120` (the frozen upper bound) is:

> **Boundary pressure observed / measurable; expansion is not authorized;
> sufficiently strong combined evidence may motivate a separate targeted
> >120 experiment.**

Do **not** use `"natural / N/A"` as the RD1 scientific interpretation of the
120-hour upper boundary. (The current
`sweep_v2_six_axis_review_analysis` reporting-label default and its tests
label both seq_length bounds `"natural"`; that label is a lower-level
reporting artifact and is **not** the RD1 scientific interpretation. Any code
change to reconcile the two is a separate later milestone, not part of this
docs freeze.)

This decision:

- does **not** change the current search domain;
- does **not** authorize >120 trials — **DOWNSTREAM / NOT AUTHORIZED**;
- does **not** establish causality;
- does **not** by itself imply that longer context is better.

It only ensures upper-bound occupancy / performance evidence is **visible to
the scientific reviewer** rather than suppressed by a reporting label.

The **lower boundary** (`seq_length = 48`) may continue to be described
according to evidence and existing design semantics. **No symmetry
requirement is manufactured.**

---

## 15. Axis / interaction evidence matrix — **FROZEN structure**

The final review exposes **evidence, not an opaque score**. For each axis, and
for any materially supported interaction, provide fields covering:

1. coverage (where in the axis range the campaign actually sampled);
2. performance direction (descriptive association with the objective);
3. robustness across settings of the other axes;
4. random-control support (does the random arm show the same pattern?);
5. Bayesian concentration / proposal evolution;
6. boundary pressure;
7. interaction evidence;
8. failure / late-instability evidence;
9. basin-level robustness;
10. KGE robustness;
11. Q98 / high-flow robustness;
12. scientific-reviewer confidence (see §15.1 — a human field);
13. candidate action.

**Possible candidate actions:**

- keep unchanged;
- consider interior narrowing;
- retain range because evidence is weak;
- investigate a boundary with a separate targeted experiment;
- investigate a repeated problematic interaction;
- defer.

**Weak evidence normally means "keep the present search space unchanged", not
speculative narrowing.**

### 15.1 No automatic final scientific-confidence classifier — **FROZEN**

- Do **not** design an algorithm that mechanically maps the preceding columns
  to `strong` / `suggestive` / `weak`.
- Code may later compute and expose transparent **lower-level** evidence
  (occupancy fractions, direction summaries, robustness spreads, etc.).
- The final `strong / suggestive / weak` label is a **human scientific-review
  judgment by ChatGPT + user**, documented together with its supporting
  evidence.
- Existing `boundary_pressure_tier` (or any similar threshold-driven label)
  may be shown **only as low-level diagnostic evidence** and must not
  substitute for the final scientific judgment.
- **Bayesian concentration alone can never establish strong axis evidence.**

---

## 16. Training stability — **SECONDARY** (retained per configuration)

The official configuration score remains the **best eligible Common-120
objective within the frozen 12-epoch fidelity** (`mf12x50000`: 12 epochs ×
50,000 updates, save every epoch, `performance_early_stopping_enabled =
false`, Seed A `967139`).

Retain separately, per configuration:

- best epoch;
- best score;
- final-epoch score;
- best-minus-final;
- late gain (e.g. `best_score_12 - best_score_10`, `late_gain_10_to_12`);
- fixed-support epoch trajectory (12 medians);
- natural-support trajectory (SECONDARY diagnostic, §10);
- runtime;
- GPU-hours where recoverable (§5).

**A valid run with poor late stability remains a valid scientific
observation** unless it violates an existing validity rule. Do **not**
retrospectively invalidate configurations because their behaviour is
inconvenient. (The P4 / P7 `{learning_rate >= 8e-4, batch_size = 256}`
late-divergence pattern from the post-P8 checkpoint is an example: recorded,
monitored, downstream-gated — not an invalidation.)

---

## 17. Data / evidence contract discovered by RD1-A — **reference**

**Central finding.** The compact local campaign receipts (per-trial
`attempt_review_records.json`, `execution_provenance.json` /
`trajectory_and_identity_extract.json`, the post-P8
`sixteen_row_table.csv` / `sixteen_row_trajectories.csv`) retain:

- configuration / arm / proposal identities;
- the six hyperparameters;
- scalar objective results (official objective, best epoch, final, best-minus-
  final, late gain);
- best / final / stability diagnostics;
- fixed-support and natural-support **median** epoch trajectories;
- run / config / launch provenance and checksums;
- support-contract identity;
- some Slurm evidence (`phase1_sacct.txt`).

They do **not** contain the full **per-basin / per-timestamp** scientific
inputs required for:

- 400-basin NSE distributions (§8);
- P2 / R2 paired basin comparison (§8, §11);
- KGE and components (§11);
- Q98 high-flow diagnostics (§12).

Those require the authoritative NeuralHydrology
`validation/model_epochNNN/validation_results.p` products (Moriah-resident,
~84.5 MB per epoch, inventoried but **not** transferred locally) together with
the frozen Common-120 support contract (which **is** retained locally at
`.scratch_local/devpop_audit_a5/canary_inputs/stage1_v2_common120_fixed_support_contract_v001.json`).

**Three distinct operations — the design document must keep these separate:**

1. **Retrieval / re-reading** of an already-produced `validation_results.p`
   (moving or opening a file that already exists);
2. **Deterministic re-scoring** of that retained output — pure arithmetic
   over the stored obs/sim arrays via
   `fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics`
   (fixed support) or `nh_seed_evaluation.raw_space_metrics_for_run_period`
   (natural support). **This runs no NeuralHydrology inference and is not
   training.**
3. **New NeuralHydrology evaluation / inference from a checkpoint** — only
   needed if a required `validation_results.p` no longer exists.

The core analysis needs the **official best-epoch validation result for each
of the 24 valid configurations**. Whether that is operation (1)+(2) or
operation (3) depends on remote retention, which only the execution lane can
confirm.

Do **not** describe deterministic metric extraction from an already-produced
validation result as new training.

---

## 18. Preservation requirement — **reference; no remote action**

The final 12+12 scientific review depends on preservation of already-produced
validation outputs. This document **takes no remote action** and **contacts
no execution lane**. It records the dependency; the **execution lane** must
verify actual remote retention.

At minimum, preserve:

- every valid configuration's **official best-epoch `validation_results.p`**
  (24 files);
- preferably **all currently retained per-epoch validation-result products**
  until RD1 closes;
- **model checkpoints** as fallback if a validation result is missing;
- the frozen **Common-120 fixed-support contract**
  (SHA-256 `cb4ebe86af…6851ef`);
- the frozen **screening-400 identity**
  (SHA-256 `d4395d93…baa372`);
- the **per-trial completion / provenance / checksum bundles** (the only
  source of `phase1_sacct.txt`, `execution_provenance.json`, and the
  `SHA256SUMS.txt` manifests);
- **Slurm / sacct evidence** needed for exact GPU-hour accounting (§5);
- the **tracked frozen random-control manifest** (SHA-256 `59be6726…c095`).

**No preservation action, remote contact, or cleanup is performed or
authorized by this task.**

---

## 19. Planned implementation sequence after this docs freeze

Recorded for planning; **not implemented here**. Each milestone is separately
reviewable.

### RD1-C2 — Configuration-level analysis core

Pure / synthetic-tested: six-axis normalized geometry; novelty; incumbent
distance; top-3 (floor / mean / median); near-incumbent yield (P2-referenced
`0.01` / `0.005` counts + threshold-free top-k); whole-sequence descriptors;
duplicate-coordinate accounting; compute / yield derivations. Reuses
`sweep_v1_review_analysis` primitives (`cumulative_best_by_order`,
`cumulative_gpu_hours_by_order`, per-axis normalization kernels) as
building blocks only.

### RD1-C3 — Basin-analysis primitives

Pure / synthetic-tested: reuse `percentile_diagnostics.compute_percentile_table`
for the quantile grid; add IQR; ECDF; P2 / R2 paired differences;
improved / worse / numerical-tie fractions with the machine-precision
tolerance from §8.1.

### RD1-C4 — Fixed-support hydrological consumer + Q98

Highest-risk scientific-interface milestone: authoritative best-epoch
fixed-support extraction via
`fixed_support_contract_v2.evaluate_fixed_support_raw_space_metrics`;
per-basin NSE; KGE and components; admitted obs / sim / timestamp arrays; the
frozen Q98 diagnostics (§12); provenance binding; fail-closed behaviour; one
vertical synthetic integration test. **Strong candidate for independent Codex
review after implementation.**

### RD1-C5 — Transparent axis / interaction evidence derivations

Generate the §15 evidence columns only. **No automated final
strong/suggestive/weak verdict** (§15.1).

### RD1-C6 — Rendering and final review packet

Only after the numerical / data contracts are stable. A new v2 review-
rendering module (no v2 rendering module exists today; `sweep_v1_review_rendering`
is 5-axis and renders the early/late split RD1 avoids). Reuse only the
save / style / synthetic-banner scaffolding.

**The real 12+12 final scientific review occurs only after the execution lane
declares the valid 12+12 campaign complete and the required evidence is
available.**

---

## 20. What this document does NOT authorize — **DOWNSTREAM / NOT AUTHORIZED**

For the avoidance of any doubt, nothing in this document authorizes:

- `seq_length > 120`, or any other search-range change;
- narrowed search ranges;
- continued Bayesian search (P13+);
- additional random controls (R13+);
- full-development (~2,307-basin) evaluation, or any SHARED-A5 consumption
  beyond its standing conclusions;
- model or hyperparameter promotion / selection / winner declaration;
- new W&B / controller / sweep activity;
- any new training or NeuralHydrology inference;
- any new compute beyond deterministic re-scoring of already-produced
  validation results;
- sealed / temporal-test / spatial-holdout / California access.

Those remain separate user decisions.

---

## 21. Provenance of this freeze

- Authorized by the user on **2026-09-09** as task **RD1-C1**, immediately
  after the read-only RD1-A archaeology.
- **No substantive P9–P12 / R9–R12 outcome was inspected or used** to form
  any decision in this document. The bundle
  `.scratch_local/v2_p9_p12_wave3_5a09bce/` was not opened or enumerated.
- Pre-P9 evidence used to freeze the references: P2 `0.4388098707961096`
  (best epoch 9) and R2 `0.43276757946464195` (best epoch 6), from committed
  `docs/decision_log.md` and `docs/FLASHNH_CURRENT_STATE.md` and the retained
  post-P8 8+8 review table.
- Repository state at freeze: branch `master`, HEAD
  `5a09bcee034748d85ed13fea8d0b098326e8988c` ("Close unattended launcher
  hardening").
- The matching `docs/decision_log.md` and `docs/FLASHNH_CURRENT_STATE.md`
  entries record the freeze; this document is the authoritative detailed
  specification.
