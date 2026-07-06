# Stage 1 scientific baseline — design gate

Milestone: 2K-G-D (Attribute Provenance + Modeling Design Gate), opened 2026-07-03.
Milestone 2K-G-E (Scientific Baseline Design Resolution): first proposed 2026-07-03,
**revised 2026-07-06** after user review changed several key decisions (basin
splits, target scaling, lead time, static attributes, hyperparameter framing). This
revision replaces the 2026-07-03 draft in place — that draft was never committed
(see `docs/decision_log.md` for what changed and why).

This document now distinguishes three kinds of items:
- **Approved** — a binding decision, recorded below, not open for re-litigation
  absent a new explicit decision.
- **Still open, needs sign-off** — a scientific/methodological choice with a
  recommended default, but not yet decided.
- **Gated on a mini-milestone** — cannot be responsibly decided from documentation
  or assumptions alone; requires inspecting the actual NeuralHydrology 1.13 code
  installed on Moriah (§"New mini-milestones" below), or recovering source data
  that does not yet live at its canonical path.

## What this document is

A **decision scaffold**, not a locked model spec. Smoke 0 and Smoke 1 (both
PASS on Moriah, 2026-07-02) proved the technical pipeline — package format,
NH 1.13 compatibility, MRMS + core RTMA ingestion — works end to end at
`seq_length=24`, 5 basins, a handful of epochs. None of that constitutes a
scientific baseline. This doc lists what must be explicitly decided before
the first full scientific-baseline training run, so those decisions are made
once, deliberately, and recorded — rather than inherited by default from
whatever Smoke 0/1 happened to use.

**`seq_length` is one hyperparameter among many decided here — it is not the
milestone driver.** It is now a **binding, narrow** choice for Stage 1: only
12/24/48/72 hours are Stage-1-eligible (§9). Longer lookback (168/336 h and
beyond) is explicitly **Stage 2 / long-term antecedent-modeling territory, not
Stage 1** — do not reintroduce it here in future revisions of this document.

**Stage 1 excludes California entirely** (§8c). California is reserved for a
Stage 4 transfer-learning experiment; Stages 1–3 are non-CA CONUS only.

## Purpose and non-goals

**Purpose:** define the first scientific-baseline NeuralHydrology training
configuration for Flash-NH Stage 1 — a run whose results are meant to be
scientifically interpretable (skill scores worth reporting), not just a
plumbing check.

**Non-goals of this document:**
- Does not generate the full 2,752-basin NH package (separate execution step,
  gated on §2 attribute provenance being closed — see
  `docs/stage1_attribute_provenance.md` — **and** on the new 2K-G-F / 2K-G-G
  gates below).
- Does not run training.
- Does not perform hyperparameter search / sweeps — it defines the sweep
  *policy* (§9) so a future sweep has a documented protocol to follow.
- Does not finalize every value below as immutable — items marked **still
  open** are decisions still needed, via explicit user sign-off; items marked
  **gated** require a technical report before they can even be proposed
  responsibly.
- Does not create the richer static-attribute matrix or the gap-policy /
  target-scaling feasibility report — both are scoped as separate
  mini-milestones (2K-G-F, 2K-G-G) and are **not done in this patch**.

---

## Binding decisions (user-approved, 2026-07-06)

Quick-reference list. Details and rationale are in the numbered sections
below; this list exists so future prompts/docs do not keep re-proposing
values already settled here.

1. **Basin set:** conservative 2,752-basin floor; exclude `02299472` and
   `04073468`. (§4)
2. **Stage 1 dynamic inputs:** `v001-core` — the 8 already-confirmed Smoke 0/1
   variables. The extra 5 RTMA variables are deferred to a later
   `v001-fullmet` ablation, **not** a default extension after a smoke test.
   (§1)
3. **`seq_length`:** Stage 1 candidates are **only** 12 / 24 / 48 / 72 hours.
   Default smoke/preflight value stays 24. 168/336 h are **not** Stage 1
   candidates — they belong to Stage 2. (§9)
4. **Lead time:** primary benchmark lead time **6 h**; secondary **12 h**; 1 h
   and 3 h are diagnostic/sanity-check only, not the primary benchmark.
   Input sequence length and prediction lead time are **separate design
   axes** — do not conflate them. (§9)
5. **Temporal split:** train `2020-10-14`→`2023-12-31`, validation
   `2024-01-01`→`2024-12-31`, test `2025-01-01`→`2025-12-31`. (§8)
6. **Spatial split:** California excluded entirely from Stages 1–3. Within
   non-CA CONUS, ~10% spatial-holdout basins, broadly distributed, test-only
   (never used for training/validation/tuning/normalization/early
   stopping/model selection). Official spatial-holdout evaluation uses the
   2025 test period. (§8b)
7. **California transfer learning (Stage 4):** CA held out through Stages
   1–3; used in Stage 4 for fine-tuning with an internal ~90/10 CA split
   (fine-tune-train / CA holdout). CA-specific normalization may be refit
   using only the CA fine-tuning training subset. (§8c)
8. **Leakage prevention:** scaling/normalization statistics for Stages 1–3
   are fit only on the development training basins and training period —
   never on validation, temporal test, spatial holdout, or CA data. (§8d)
9. **Target scaling:** log-transform is **rejected** as the default (poorly
   aligned with flash-flood/high-flow emphasis). Leading candidate is
   area-normalized / specific discharge, pending NH/package feasibility —
   gated on 2K-G-G. (§5)
10. **Loss vs. metrics:** training loss is separate from evaluation metrics.
    Evaluation is always reported in raw `m^3/s` after inverse conversion;
    raw-space NSE is the primary evaluation metric. (§7)
11. **Hyperparameters:** the conventional table is an **initial seed config
    only**, not the official Stage 1 benchmark. The official benchmark
    requires a controlled W&B hyperparameter sweep (not run yet). (§9)
12. **Sweep objective:** validation raw-space NSE is the primary
    model-selection objective for now; high-flow/event metrics are logged as
    secondary diagnostics; composite objectives are a later discussion. (§9)
13. **W&B logging:** expanded beyond config/provenance to include loss/validation
    curves, learning rate, epoch timing, run duration, hyperparameters, final
    metrics, Slurm job ID/node/partition/GRES/GPU type, git commit, package
    provenance, and system/resource telemetry where available. (§10)
14. **Slurm/resources:** kept flexible and parameterized, not permanently
    hard-pinned to one partition/GPU; actual resources used are recorded in
    the evidence bundle; allocation may be increased later if telemetry shows
    training is resource-limited. (§11)

---

## 1. Candidate dynamic inputs for the first baseline — `v001-core` (APPROVED)

Confirmed working end-to-end in Smoke 1 (`docs/stage1_neuralhydrology_preflight.md`
§2, §7):

| Variable | Units | Status |
|---|---|---|
| `mrms_qpe_1h_mm` | mm | Smoke 0+1 PASS |
| `rtma_2t_K` | K | Smoke 1 PASS |
| `rtma_2d_K` | K | Smoke 1 PASS (dewpoint mapping fix confirmed, 2K-F-C-B) |
| `rtma_2sh_kgkg` | kg/kg | Smoke 1 PASS |
| `rtma_10u_ms` | m/s | Smoke 1 PASS |
| `rtma_10v_ms` | m/s | Smoke 1 PASS |
| `mrms_qpe_1h_mm_gap` | bool→float32 | Smoke 0+1 PASS (gap flag, not a physical forcing) |
| `rtma_gap` | bool→float32 | Smoke 1 PASS (gap flag) |

**APPROVED (2026-07-06):** these 8 variables are `v001-core` — the input set
for the **first Stage 1 scientific benchmark**. No further review needed.

**Deferred, not a default extension:**

| Variable | Units | Note |
|---|---|---|
| `rtma_sp_Pa` | Pa | Surface pressure |
| `rtma_tcc_pct` | % | Total cloud cover |
| `rtma_vis_m` | m | Visibility |
| `rtma_gust_ms` | m/s | Wind gust |
| `rtma_ceil_m` | m | Cloud ceiling height |

These 5 variables define a later **`v001-fullmet`** ablation experiment, run
*after* the `v001-core` benchmark exists, not automatically appended to it
once a smoke test passes. **Correction from the 2026-07-03 draft:** that
draft implied the first benchmark should default to all 13 variables after a
"Smoke 2"-style check; that implication is removed. `v001-core` (8 variables)
is the first Stage 1 benchmark input set, full stop — `v001-fullmet` is a
separate, later experiment.

## 2. Deferred dynamic inputs (explicitly out of scope for first baseline)

- Any variable outside the 13-column curated v001 forcing schema (no other
  source has been ingested/audited at full-period, full-basin scale).
- URMA-derived precipitation (`docs/stage1_rtma_urma_mrms_diagnostic.md`) —
  diagnostic-only exploration, not integrated into the curated product.
- Forecast/NWP-derived features — out of scope for Stage 1 (historical
  reanalysis-driven baseline only; forecast-aware modeling is a later
  project phase per the project's stated purpose).

## 3. Static attributes — REOPENED, gated on Milestone 2K-G-F

The 2026-07-03 draft proposed signing off a curated ~16-column subset of the
48-column GAGES-II screening merge. **That proposal is withdrawn — do not
sign off any column list from the current screening merge.**

`/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`
(48 columns, checksum-pinned per `docs/stage1_attribute_provenance.md`)
remains a **valid, checksum-verified provenance artifact**, but it is likely
**not sufficient** as the final Stage 1 modeling static-attribute matrix. The
user has richer local source material at
`C:\PhD\Python\neuralhydrology\US_data\attributes` — reportedly ~28 attribute
files, mostly GAGES-II / related sources, going well beyond the 48-column
screening merge, including at minimum:
- `attributes_gageii_Topo.csv`
- `attributes_gageii_Bas_Morph.csv`
- `attributes_hydroATLAS.csv`
- `attributes_nldas2_climate.csv`
- `Var description_gageii.xlsx` (variable-description reference, reportedly
  ~350 described static variables)

The user expects the final Stage 1 static-attribute matrix to include
topography, geology, land use/land cover, vegetation, snow fraction, and
climate/static hydrologic attributes — none of which exist in the current
48-column screening merge (confirmed in `docs/stage1_attribute_provenance.md`:
that merge has no `ELEV`, `SLOPE`, `FOREST_PCT`, or climate-normal fields).

**Resolution (2026-07-06):** static attributes are **gated on Milestone
2K-G-F** (Static Attribute Matrix Recovery + Audit — see "New mini-milestones"
below). No column list — curated or otherwise — is proposed in this document
until that milestone produces an audited, checksummed candidate matrix.

## 4. Target variable and target cleaning (APPROVED — basin set)

Target: `qobs_m3s` (streamflow, m³/s). Cleaning policy already decided and
applied by the target package builder (`docs/stage1_target_policy.md`,
Milestone 2J-A/2J-B) — **reuse as-is, do not redefine here**:
- Negative qobs → NaN (no other transformation of positive observations).
- NaN preserved exactly; no interpolation, gap-filling, or imputation.
- `TARGET_OPERATIONAL_REVIEW` (89 basins, late-period gaps) already excluded
  by policy — do not override for the first baseline.

**APPROVED (2026-07-06):** exclude both `02299472` and `04073468` →
2,752-basin floor. This is the conservative basin set for Stage 1–3 (before
the §8b spatial split and §8c CA exclusion are applied on top of it).

## 5. Target normalization / transformation policy — STILL OPEN, gated on 2K-G-G

**Log-transform is rejected as the recommended default (2026-07-06).** The
2026-07-03 draft proposed `log(qobs + eps)` as the leading candidate; the
user explicitly rejected this as poorly aligned with the project's
flash-flood / high-flow emphasis (log-compression de-emphasizes exactly the
peak events Flash-NH cares most about).

**Current candidates, still open:**
- **(a) Area-normalized / specific discharge** (e.g. runoff-depth-equivalent,
  discharge divided by `DRAIN_SQKM` or similar) — **leading candidate**,
  pending confirmation that NH / the package builder can support this
  transform cleanly and invert it back to raw `m^3/s` for evaluation.
- **(b) Raw `qobs_m3s` with NH default z-score normalization per basin** —
  simplest alternative, still under consideration.
- **(c) Per-basin standardization** (basin-specific mean/std, not global) —
  alternative still under consideration.

Regardless of which is chosen, **evaluation/reporting metrics must always be
computed in raw discharge units (`m^3/s`)** after inverse-transforming model
output — see §7.

**Gated on Milestone 2K-G-G:** before locking any target-scaling policy, NH
1.13's actual target-normalization behavior (as installed on Moriah) must be
inspected directly in code — not inferred from public docs or assumptions.
This determines which of (a)/(b)/(c) is actually implementable without
custom code, and whether inverse-transform-for-evaluation is handled natively
or needs a custom step.

## 6. Forcing gap policy — scientific baseline vs. Smoke 0/1 — HIGH PRIORITY, gated on 2K-G-G

Smoke 0/1 used a **technical-only** gap-fill policy (documented in
`docs/stage1_neuralhydrology_preflight.md` §8.2, and flagged with an explicit
`WARNING` in `scripts/build_stage1_nh_package.py`'s docstring): MRMS gaps
(136 h/basin) filled with 0.0 mm, RTMA gaps (2 h/basin) linearly
interpolated. This is explicitly **not** approved for scientific training,
and **must not be silently carried forward** — this is now treated as a
high-priority unresolved technical/scientific issue, not a routine cleanup
item.

Two candidate approaches were already identified in §8.2 of the preflight
doc:
- **(a) Window/sample exclusion.** Keep the full 45,720-hour aligned `date`
  coordinate in the NC (do not remove rows), but exclude any training
  window whose input sequence or prediction horizon contains a gap hour,
  at NH batch-sampling time.
- **(b) `nan_handling_method`.** Use a tested NH-native NaN-handling method
  (e.g. `masked_mean`) instead of pre-filling, so the model/loss sees the
  gap structure directly rather than an imputed value.

RTMA's 2 h/basin (0.004%) interpolation likely remains acceptable as-is, but
this must be **explicitly recorded as a decision**, not assumed by omission.

**Gated on Milestone 2K-G-G:** a required **gap-policy technical report**
must examine, by reading the actual NeuralHydrology 1.13 code installed on
Moriah (not public docs, not assumptions):
- support for `nan_handling_method` and its exact behavior,
- masked-loss support for dynamic-input NaNs,
- whether window/sample exclusion based on `mrms_qpe_1h_mm_gap` is natively
  supported by the `generic` dataset's batch sampler, or requires a custom
  sampler/package mask,
- quantified expected sample/window loss for each combination of
  `seq_length` ∈ {12, 24, 48, 72} × lead time ∈ {1, 3, 6, 12} h — this
  matters because window exclusion cost scales with both axes,
- an explicit, recorded decision on the RTMA interpolation policy.

No forcing-gap policy is proposed as final in this document — this section
is gated, not merely "needs sign-off."

## 7. Loss and metrics — training loss vs. evaluation metrics (separated)

**Training loss** (still under design, not finalized): likely an NSE-family
loss computed on whichever scaled target §5 resolves to. The exact loss
formulation depends on the target-scaling outcome (§5, gated on 2K-G-G) —
an NSE-family loss on area-normalized discharge is not identical to NSE on
raw or log discharge.

**Evaluation / reporting metrics** (separate from training loss, always
computed from predicted and observed hydrographs in **raw `m^3/s`**, after
inverse-transforming model output regardless of the internal training
scaling):
- **Raw-space NSE — primary evaluation metric** for the benchmark.
- KGE and its components (correlation, bias ratio, variability ratio).
- Percent bias (PBIAS).
- Peak magnitude error.
- Peak timing error.
- Event/high-flow metrics — noted as important but **deferred as its own
  near-term discussion/milestone**, not designed in detail here.
- Per-basin metric distributions (not just a mean across basins) — mean NSE
  across 2,752 basins can hide systematic failure on a subset.

**Correction from the 2026-07-03 draft:** that draft treated "loss" and
"metrics" as one resolved item (both "recommended now," NSE for both). This
revision separates them explicitly: the training loss is still open pending
§5, while raw-space NSE + the metric list above is the recommended,
lower-risk **evaluation** default (adopt now — it's a reporting choice with
no training-time consequence).

## 8. Train / validation / test protocol — temporal split (APPROVED, revised dates)

**APPROVED (2026-07-06) — replaces the 2026-07-03 dates:**

| Split | Period |
|---|---|
| Train | 2020-10-14 → 2023-12-31 |
| Validation | 2024-01-01 → 2024-12-31 |
| Test | 2025-01-01 → 2025-12-31 |

Rationale: closer to an intended 60/20/20 chronological design, adjusted for
the data actually available (full period starts 2020-10-14). This **replaces**
the previous split (train ≤2022-12-31 / val 2023 / test 2024–2025) that was
encoded as constants in `scripts/build_stage1_nh_package.py`
(`_TRAIN_END`, `_VAL_START/_END`, `_TEST_START/_END`) — those constants are
**not yet updated** (that would be a code change, out of scope for this
documentation-only patch); updating them is a follow-up implementation step,
not authorized here.

This is a **temporal split** (same basins across all three periods) — see
§8b for the orthogonal spatial-holdout axis, which is layered on top of this,
not a replacement for it.

## 8b. Spatial / geographic split (NEW, APPROVED)

Two rules, approved 2026-07-06:

1. **California exclusion.** Stages 1–3 exclude California basins
   **completely** — not a holdout-within-training-eligible-pool, but fully
   out of scope until Stage 4 (§8c).
2. **Non-CA spatial holdout.** Within non-CA CONUS, define an approximately
   **10% spatial-holdout basin set**, selected to be broadly / stratifiably
   distributed across the continent (not clustered in one region). These
   basins are **test-only**: never used for training, validation,
   hyperparameter tuning, normalization/scaler fitting, early stopping, or
   model selection, at any point.
3. The remaining ~90% non-CA CONUS basins form the **development pool** —
   this is the set the §8 temporal train/validation/test split is applied
   within.
4. **Official spatial-holdout evaluation uses the 2025 test period** (same
   period as the temporal test set) for apples-to-apples comparison between
   temporal-test skill and spatial-holdout skill. All-period diagnostics on
   the spatial-holdout set may be an optional later addition, not required
   for the first benchmark.

The exact ~10% spatial-holdout basin list is **not selected in this
document** — selecting it (with a documented, reproducible stratification
method) is a follow-up step, not authorized here.

## 8c. California transfer-learning split (Stage 4) (NEW, APPROVED)

1. California is held out completely from Stages 1–3 (§8b).
2. In **Stage 4**, California is used for a transfer-learning / fine-tuning
   experiment: take the model trained on non-CA CONUS (Stages 1–3) and
   fine-tune it on California data.
3. Within California, use an internal split similar in spirit to §8b:
   approximately **90% CA basins** for fine-tuning/development, approximately
   **10% CA basins** as a never-seen CA holdout.
4. **Normalization exception for Stage 4 only:** during Stage 4 fine-tuning,
   it is acceptable to fit/update normalization statistics using **only the
   allowed CA fine-tuning training subset**, because those basins are then
   explicitly part of retraining — this is *not* a leakage violation because
   the CA holdout is still excluded from any statistic-fitting (§8d).
5. **Quantifying transfer-learning benefit:** compare (i) the original
   non-CA-trained model and (ii) the fine-tuned model, both evaluated on the
   10% CA holdout basins — the gap between them is the measured
   transfer-learning benefit.

## 8d. Leakage prevention (NEW, APPROVED — binding rules)

Explicit, binding rules tying together §8/§8b/§8c:

1. For **Stages 1–3**, all scaling/normalization statistics (static-attribute
   scalers, dynamic-input scalers, target scalers, and any basin-area
   target-scaling statistics per §5) must be fit **only** on the allowed
   development **training** basins and **training period** — never on
   validation, temporal test, spatial holdout, or California data, for any
   reason.
2. For **Stage 4 fine-tuning**, any CA-specific normalization update must use
   **only** the CA fine-tuning training subset — never the CA holdout (§8c
   item 4).
3. These rules apply uniformly to every scaler type — there is no exception
   for "just the static attributes" or "just the target" — all of them are
   development-training-only for Stages 1–3.
4. Violating these rules (e.g. fitting a global scaler across all basins
   including spatial holdout or CA before splitting) would silently leak
   test-set information into the model and invalidate the benchmark's skill
   scores — this is why the rule is stated explicitly here rather than left
   implicit in the package-builder code.

## 9. `seq_length`, lead time, and hyperparameters

### 9a. `seq_length` — Stage 1 binding decision

**APPROVED, binding (2026-07-06):** Stage 1 `seq_length` candidates are
**only** 12, 24, 48, or 72 hours. Default smoke/preflight value remains 24 h
(unchanged from Smoke 0/1). **168 h and 336 h are explicitly not Stage 1
candidates** — they belong to Stage 2 / long-term antecedent-moisture
modeling. **This is a binding design decision specifically to stop future
prompts/docs from reintroducing 168/336 h for Stage 1** — the 2026-07-03
draft proposed 336 h (citing Kratzert et al. 2021) as the Stage 1 candidate;
that proposal is **withdrawn**. Final `seq_length` selection within
{12,24,48,72} happens via the W&B sweep (§9c), not a unilateral pre-sweep
choice.

### 9b. Lead time — new, separate design axis

**APPROVED (2026-07-06):** input sequence length (`seq_length`, "how much
history the model sees") and prediction lead time ("how far ahead the model
predicts") are **separate design axes** — the 2026-07-03 draft did not
address lead time at all, and future revisions should not conflate the two.

- **Primary Stage 1 benchmark lead time: 6 hours.**
- **Secondary lead time: 12 hours.**
- 1 h and 3 h lead times may be used as diagnostic/sanity checks (e.g.
  confirming the model performs reasonably at near-nowcast horizons), but are
  **not** the primary benchmark.

### 9c. Hyperparameters — initial seed config, not the final benchmark

**Reframed (2026-07-06):** the hyperparameter table below is an **initial
seed / first-viable-config only**. It is explicitly **not** the official
Stage 1 benchmark. The 2026-07-03 draft framed a similar table as
"recommend now, adopt as-is" — that framing is corrected: a single
non-tuned config is useful only to confirm the pipeline trains sensibly
end-to-end, not to report as the benchmark result.

**The official Stage 1 benchmark requires a controlled W&B hyperparameter
sweep** (not run in this patch or milestone). Candidate sweep dimensions,
at minimum:
- `seq_length` ∈ {12, 24, 48, 72} (§9a)
- hidden size
- dropout
- learning rate
- batch size
- possibly number of LSTM layers

**Initial seed config** (for pipeline verification only, not benchmark
results):

| Hyperparameter | Seed value |
|---|---|
| Model | `cudalstm` (same as Smoke 0/1) |
| Hidden size | 128 |
| LSTM layers | 1 |
| Dropout | ~0.2–0.3 |
| Batch size | 256 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | max 30–50, with early stopping |
| Early stopping / model selection | Validation raw-space NSE (§9d) |

Careful sweep design (search strategy, budget, parallelism) is deferred to a
later, dedicated discussion — **not designed or run now.**

### 9d. Sweep objective — distinct from training loss

**APPROVED (2026-07-06):** distinguish two different objectives that are
easy to conflate:
1. **Training objective/loss** for a single model run (§7 — still open,
   pending §5).
2. **Sweep / model-selection objective** for choosing among hyperparameter
   configurations once a sweep runs.

For now, use **validation raw-space NSE** as the primary sweep/
model-selection objective. High-flow/event metrics are logged as secondary
diagnostics during the sweep, not part of the objective yet. Composite
objectives (e.g. `NSE_raw` + a high-flow NSE term) can be discussed later,
after event metrics (§7) mature — not designed here.

## 10. W&B logging / sweep policy — expanded

**Expanded (2026-07-06)** beyond the 2026-07-03 draft's config/provenance-only
scope. Every run should log:
- Full resolved NH config, `run_provenance.json` contents (including
  `attributes_sha256`).
- **Training loss curves, validation curves, learning rate (schedule),
  epoch timing, total run duration.**
- Final metrics (raw-space NSE + the §7 metric set).
- Hyperparameters actually used.
- **Slurm job ID, node, partition, GRES, GPU type.**
- Git commit hash.
- Package provenance.
- **System/resource telemetry where available** (e.g. GPU utilization/memory
  from `nvidia-smi`, if captured) — enough to diagnose whether a run is
  compute- or I/O-bound, feeding into the §11 Slurm-resource decision.

Policy (unchanged from the 2026-07-03 draft, still recommended now):
- One W&B project per major package version (e.g.
  `flashnh-stage1-scientific-baseline-v001`), not per-run.
- Sweeps (§9c/§9d) should be defined declaratively (W&B sweep YAML or
  equivalent) and committed to `config/`, not run ad hoc from the CLI.
  **W&B sweeps are expected for the tuned Stage 1 benchmark, but are not yet
  run** — this document only records the policy.
- Credentials: W&B API key follows the same rule as all other credentials in
  `docs/repo_policy.md` — never committed, never logged.

## 11. Slurm partition / GRES parameterization policy — kept flexible

Both existing sbatch templates (`run_stage1_smoke0/1_moriah.sbatch`)
hard-pin `--partition=catfish --gres=gpu:l4:1`. **Policy (reaffirmed
2026-07-06): keep Slurm resources flexible and parameterized — do not
permanently hard-pin scientific baseline runs to one partition/GPU.**
Parameterize `PARTITION` and `GRES` as variables at the top of the sbatch
scripts (so future runs can target `salmon`/L40S or `goldfish`/H200 without
editing the script body); this is a code change and remains **deferred** —
out of scope for this documentation-only patch.

Record the actual resources used (partition, GRES, node, GPU type) in the
evidence bundle for every run (§12). If W&B/Slurm/`nvidia-smi` telemetry
(§10) shows training is too slow or resource-limited on the currently-used
GPU class, resource allocation may be increased later — that is a decision
to make from evidence, not in advance.

## 12. Evidence bundle conventions for Moriah runs

Reuse the existing conventions rather than invent new ones:
- `docs/repo_policy.md` → "h2o/Moriah Remote Run Evidence Policy": pull logs,
  manifests, config, checksums, and `run_provenance.json` to a local
  `tmp/` evidence bundle before any conclusion is documented or committed.
- `docs/repo_policy.md` → "Post-h2o Run Export Policy" table (include:
  summary JSON/MD, audit CSVs, main log, shard manifests; exclude: canonical
  NetCDF/Parquet, per-station logs, GRIB/large binaries).
- For the baseline run specifically, also pull: the resolved NH YAML config
  as actually consumed by NH (not just the source template), the W&B run
  URL/ID (§10), and the Slurm `sacct` record (node, partition, GRES, GPU
  type, elapsed, exit code) — same fields already captured for Smoke 0/1 in
  `docs/decision_log.md`, now extended per §10's expanded W&B/telemetry list.

---

## New mini-milestones (required gates before full package generation)

Both milestones below are **design/technical-preflight milestones**. Neither
is executed in this patch — this document only defines their scope and why
they are required. Both explicitly require inspecting the **actual installed
code** (NeuralHydrology 1.13 on Moriah; the local attribute-source directory)
rather than relying on public documentation or assumptions.

### 2K-G-F — Static Attribute Matrix Recovery + Audit

Required because the current 48-column GAGES-II screening merge is a valid
provenance artifact but likely insufficient as the final Stage 1 static
attribute matrix (§3). Scope:

a. Inventory the local attribute source directory
   `C:\PhD\Python\neuralhydrology\US_data\attributes` (reportedly ~28
   attribute files).
b. Check whether a corresponding attribute-source directory exists on h2o
   and Moriah.
c. If missing on h2o/Moriah, document that the source files must be
   mirrored there before reproducible package generation (do not assume
   this can be skipped).
d. Reference source files named by the user as examples:
   `attributes_gageii_Topo.csv`, `attributes_gageii_Bas_Morph.csv`,
   `attributes_hydroATLAS.csv`, `attributes_nldas2_climate.csv`,
   `Var description_gageii.xlsx`.
e. Use `Var description_gageii.xlsx` (reportedly ~350 described variables)
   as the variable-description reference for interpreting GAGES-II fields.
f. Locate/recover richer CAMELSH/CARAVAN/HydroATLAS/static attributes
   (topography, geology, land use/land cover, vegetation, snow fraction,
   climate/static hydrologic attributes).
g. Merge with useful existing GAGES-II screening fields (the current
   48-column merge is not necessarily discarded — some fields may still be
   useful).
h. Remove or explicitly encode non-numeric/categorical/id fields.
i. Audit missingness/distributions/ranges/units.
j. Checksum the resulting matrix.
k. Propose the Stage 1 static-attribute policy (a column list, this time
   backed by the richer source) — for user sign-off, not a unilateral lock.

**Not done in this patch:** no new matrix is created; this section only
documents the requirement and scope.

### 2K-G-G — Target Scaling + Gap Policy + Lead-Time Feasibility Report

Required because target-scaling (§5) and forcing-gap policy (§6) cannot be
responsibly decided from documentation or assumptions — both require
inspecting NH 1.13's actual installed behavior on Moriah. Scope:

- Inspect NH 1.13 target-normalization code paths actually installed on
  Moriah: what transforms are natively supported, how inverse-transform for
  evaluation is handled, and whether area-normalized/specific-discharge
  scaling (§5 leading candidate) is implementable without custom code.
- Inspect NH 1.13 support for: `nan_handling_method` (exact behavior, not
  just its existence), masked losses, dynamic-input NaN handling, and
  whether window/sample exclusion keyed on `mrms_qpe_1h_mm_gap` is natively
  supported by the `generic` dataset's batch sampler or requires a custom
  sampler/package mask.
- Quantify expected sample/window loss under gap-exclusion for every
  combination of `seq_length` ∈ {12, 24, 48, 72} (§9a) × lead time ∈
  {1, 3, 6, 12} h (§9b) — this is a small matrix (16 cells), not a single
  number, because exclusion cost scales with both axes.
- Explicitly record a decision on RTMA's 2 h/basin interpolation policy
  (likely acceptable as-is, but must be recorded, not assumed).
- This report **must** be based on reading the actual NeuralHydrology code
  installed on Moriah — not public docs, not assumptions carried over from
  the upstream NH README or changelog.

**Not done in this patch:** no code inspected yet (would require Moriah
shell access from a session with that access); this section only documents
the requirement and scope.

---

## Checklist: before full 2,752-basin NH package generation

- [x] Attribute provenance (48-column screening merge) closed — canonical
      path, checksum verified (Milestone 2K-G-D-A). **Note:** this closes
      provenance for the *current* merge only — it does not close §3/2K-G-F.
- [ ] **2K-G-F** — richer static-attribute matrix recovered, audited,
      checksummed, and a column-list policy proposed for sign-off.
- [ ] **2K-G-G** — target-scaling and forcing-gap-policy feasibility report
      completed via actual NH 1.13 code inspection on Moriah.
- [ ] Target normalization policy signed off (§5), informed by 2K-G-G.
- [ ] Forcing gap policy signed off (§6), informed by 2K-G-G.
- [ ] Non-CA spatial-holdout basin list selected (~10%, §8b) with a
      documented, reproducible stratification method.
- [ ] California basin list identified and excluded from Stages 1–3 (§8c).
- [ ] `seq_length` and lead-time combination selected within the approved
      Stage 1 candidates (§9a/§9b), via the W&B sweep (§9c/§9d) once it can
      be run.
- [ ] W&B sweep executed for the official Stage 1 benchmark (§9c) — the
      seed config (§9c table) is not a substitute.
- [ ] `config/stage1_scientific_baseline_v001.yaml` + NH YAML encode the
      resolved policy, including the revised temporal split (§8) —
      **not written yet** (would require updating
      `scripts/build_stage1_nh_package.py`'s `_TRAIN_END`/`_VAL_START/_END`/
      `_TEST_START/_END` constants, a code change out of scope here).
- [ ] Slurm `PARTITION`/`GRES` parameterized in the sbatch template used for
      the baseline run (code change, deferred until the config exists).
- [ ] Full 2,752-basin package generated on h2o using the resolved static
      attribute matrix and split policy — **not done, this milestone is
      documentation only.**
- [ ] Training run launched on Moriah — **not done, this milestone is
      documentation only.**
- [ ] California data transferred/prepared for Stage 4 — **not done, out of
      scope until Stage 4.**

---

## Status

**DESIGN GATE — v001 REVISED (2K-G-E revision, 2026-07-06).** 14 binding
decisions are now recorded (see "Binding decisions" above), replacing the
2026-07-03 draft's proposals where the user's review changed them (target
normalization, `seq_length` range, lead time, temporal split dates, spatial/CA
splits, static attributes, hyperparameter framing, loss/metric separation,
W&B scope, Slurm policy). Two new mini-milestones (2K-G-F, 2K-G-G) are
required — both gate real scientific decisions (static attributes, target
scaling, gap policy) behind actual code/data inspection rather than
documentation assumptions.

This document does **not** authorize full 2,752-basin package generation,
training, or any Moriah/California data transfer. All remain separate,
explicitly gated steps — now gated on 2K-G-F and 2K-G-G in addition to the
existing sign-off items.
