# Stage 1 scientific baseline — design gate

Milestone: 2K-G-D (Attribute Provenance + Modeling Design Gate), opened 2026-07-03.

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
milestone driver.** Earlier docs (`FLASHNH_CURRENT_STATE.md`,
`decision_log.md`) framed the next step as "lookback-expansion tests
(seq_length 72/168/336)"; that framing is superseded by this design gate.
Lookback is Section 9 below, one line item among ~12.

## Purpose and non-goals

**Purpose:** define the first scientific-baseline NeuralHydrology training
configuration for Flash-NH Stage 1 — a run whose results are meant to be
scientifically interpretable (skill scores worth reporting), not just a
plumbing check.

**Non-goals of this document:**
- Does not generate the full 2,752-basin NH package (separate execution step,
  gated on §2 attribute provenance being closed — see
  `docs/stage1_attribute_provenance.md`).
- Does not run training.
- Does not perform hyperparameter search / sweeps — it defines the sweep
  *policy* (§8) so a future sweep has a documented protocol to follow.
- Does not finalize every value below as immutable — items marked **OPEN**
  are decisions still needed, likely via `AskUserQuestion` or explicit user
  sign-off, before the baseline run is launched.

---

## 1. Candidate dynamic inputs for the first baseline

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

**OPEN — not yet smoke-tested, present in the curated forcing schema:**

| Variable | Units | Note |
|---|---|---|
| `rtma_sp_Pa` | Pa | Surface pressure |
| `rtma_tcc_pct` | % | Total cloud cover |
| `rtma_vis_m` | m | Visibility |
| `rtma_gust_ms` | m/s | Wind gust |
| `rtma_ceil_m` | m | Cloud ceiling height |

**Decision needed:** include all 11 physical forcing variables + 2 gap flags
(13 total, matching the full curated schema), or a physically-motivated
subset for the first baseline? Recommendation: start with the full 11 +
2 gap-flag set — the curated forcing library already produced and audited
all of them (Milestone 2K-E, PASS_WITH_CAVEATS), so there is no acquisition
cost to including them, and dropping variables should be a deliberate
ablation *after* a full-input baseline exists, not a default.

## 2. Deferred dynamic inputs (explicitly out of scope for first baseline)

- Any variable outside the 13-column curated v001 forcing schema (no other
  source has been ingested/audited at full-period, full-basin scale).
- URMA-derived precipitation (`docs/stage1_rtma_urma_mrms_diagnostic.md`) —
  diagnostic-only exploration, not integrated into the curated product.
- Forecast/NWP-derived features — out of scope for Stage 1 (historical
  reanalysis-driven baseline only; forecast-aware modeling is a later
  project phase per the project's stated purpose).

## 3. Static attributes

Source: `docs/stage1_attribute_provenance.md` (this milestone). 48 GAGES-II
columns available per basin; 4 are structurally required by the builder
(`DRAIN_SQKM`, `LAT_GAGE`, `LNG_GAGE`, `BFI_AVE`) and confirmed present.

**OPEN — attribute subset for the baseline:**
- (a) All 48 columns (minus non-numeric/id columns like `STANAME`, `STATE`,
  `COUNTYNAME_SITE`, `REACHCODE`) fed to NH as static inputs.
- (b) A curated hydrologically-motivated subset (e.g. CAMELS-style: drainage
  area, BFI, climate/water-balance fields, stream-order fractions) excluding
  fields likely to be redundant or noisy at this basin count.
Recommendation: (b) for the first baseline — fewer, hydrologically
interpretable static inputs make skill-score attribution easier to reason
about; a full-attribute run is a natural ablation, not the starting point.
Exact column list is an **OPEN** decision, not fixed by this doc.

## 4. Target variable and target cleaning

Target: `qobs_m3s` (streamflow, m³/s). Cleaning policy already decided and
applied by the target package builder (`docs/stage1_target_policy.md`,
Milestone 2J-A/2J-B) — **reuse as-is, do not redefine here**:
- Negative qobs → NaN (no other transformation of positive observations).
- NaN preserved exactly; no interpolation, gap-filling, or imputation.
- `02299472` / `04073468` carry `review_required` status (large negative-qobs
  counts) — **OPEN**: include or exclude from the first baseline. Excluding
  both drops the candidate set from 2,754 to 2,752 (the "conservative
  first-package floor" already used as the working basin count throughout
  Milestone 2K-G).
- `TARGET_OPERATIONAL_REVIEW` (89 basins, late-period gaps) already excluded
  by policy — do not override for the first baseline.

## 5. Target normalization / transformation policy

**OPEN — not yet decided anywhere in prior docs.** NH applies its own
normalization pipeline; candidates:
- (a) Raw `qobs_m3s` with NH default z-score normalization per basin.
- (b) Log-transform (`log(qobs + eps)`) before normalization — standard in
  CAMELS-style LSTM literature to handle streamflow's heavy right skew and
  multi-order-of-magnitude range across the 1–1000 km² drainage-area range
  in this candidate set (screening range documented in
  `reports/flashnh_basin_screening_v001/`).
- (c) Basin-normalized discharge (divide by `DRAIN_SQKM` or similar) before
  the model, as some hydrology-LSTM papers do, to reduce cross-basin scale
  variance.
Recommendation: (b), consistent with common practice for small, flashy
basins (this candidate set is explicitly BFI/area-screened for flashiness);
final choice is still **OPEN** pending explicit sign-off.

## 6. Forcing gap policy — scientific baseline vs. Smoke 0/1

Smoke 0/1 used a **technical-only** gap-fill policy (documented in
`docs/stage1_neuralhydrology_preflight.md` §8.2, and flagged with an explicit
`WARNING` in `scripts/build_stage1_nh_package.py`'s docstring): MRMS gaps
(136 h/basin) filled with 0.0 mm, RTMA gaps (2 h/basin) linearly
interpolated. This is explicitly **not** approved for scientific training.

Two candidate approaches were already identified in §8.2 of the preflight
doc and are restated here as the decision to close before baseline training:

- **(a) Window/sample exclusion.** Keep the full 45,720-hour aligned `date`
  coordinate in the NC (do not remove rows), but exclude any training
  window whose input sequence or prediction horizon contains a gap hour,
  at NH batch-sampling time.
- **(b) `nan_handling_method`.** Use a tested NH-native NaN-handling method
  (e.g. `masked_mean`) instead of pre-filling, so the model/loss sees the
  gap structure directly rather than an imputed value.

**OPEN:** which of (a)/(b), or a hybrid (e.g. (a) for MRMS given its 0.30%
gap rate is large enough to matter, (b) or leave RTMA's near-negligible
0.004% gap rate pre-filled) is used for the baseline. Whichever is chosen,
gap flags (`mrms_qpe_1h_mm_gap`, `rtma_gap`) stay in the package as auxiliary
dynamic inputs regardless (§8.3 of the preflight doc already covers their
normalization).

## 7. Loss and metrics

Smoke 0/1 used `loss: NSE` (NH's Nash-Sutcliffe-efficiency loss) — this is a
reasonable **default carry-forward** for the baseline (it is the standard
loss for LSTM rainfall-runoff literature and already confirmed to train
stably). **OPEN:** confirm NSE vs. an alternative (e.g. KGE-based loss) as
final; if NSE is kept, no further decision needed here.

Metrics to report for baseline evaluation (**OPEN** — propose as default,
not yet confirmed):
- NSE (primary)
- KGE and its components (correlation, bias ratio, variability ratio)
- Percent bias (PBIAS)
- Peak-flow error (relevant given this is a flashy, BFI-screened basin set)
Per-basin metric distributions (not just a mean across basins) should be
reported — mean NSE across 2,752 basins can hide systematic failure on a
subset.

## 8. Train / validation / test protocol

A period split is already defined and encoded as constants in
`scripts/build_stage1_nh_package.py` (`_TRAIN_END`, `_VAL_START/_END`,
`_TEST_START/_END`) and referenced in `docs/stage1_target_policy.md`:

| Split | Period |
|---|---|
| Train | 2020-10-14 → 2022-12-31 |
| Validation | 2023-01-01 → 2023-12-31 |
| Test | 2024-01-01 → 2025-12-31 |

This is a **temporal split** (same basins across all three periods), not a
basin-holdout split. **OPEN:** confirm this is the intended baseline
protocol, or whether a basin-holdout (spatial generalization) split, or a
combined spatiotemporal split, is wanted for the first scientific baseline.
Recommendation: keep the temporal split already built into the target
package for the *first* baseline (it is already implemented and audited);
treat spatial-holdout evaluation as a documented follow-up experiment.

## 9. `seq_length` and other conventional hyperparameters

`seq_length` is **one** hyperparameter, decided alongside — not ahead of —
the others in this document. Smoke 1 confirmed `seq_length=24` trains
without error; that says nothing about whether 24, 72, 168, or 336 hours is
scientifically appropriate. **OPEN**, to be decided together with:
- `seq_length` (candidates: 24, 72, 168, 336 h — per-candidate smoke tests
  are a valid *verification* step, but the final choice should be motivated
  by basin response-time characteristics, not just "what trains without
  error")
- Hidden size, number of LSTM layers, dropout
- Batch size, learning rate (schedule), optimizer, epoch count
- Early stopping criterion (on which validation metric)
None of these are fixed by this document. They belong in a versioned NH
YAML config (`configs/stage1_scientific_baseline_v001.yml` or similar) once
decided, following the existing config-versioning convention used for
`config/stage1_target_policy*.yaml`.

## 10. W&B logging / sweep policy

**OPEN — no prior convention exists in this repo.** Proposed defaults for
discussion:
- One W&B project per major package version (e.g.
  `flashnh-stage1-scientific-baseline-v001`), not per-run, so runs are
  comparable within a project.
- Every run logs: full resolved NH config, `run_provenance.json` contents
  (including the new `attributes_sha256` field), Slurm job ID, node,
  git commit hash — enough to reproduce the run from the evidence bundle
  alone (`docs/repo_policy.md` → evidence bundle conventions).
- Sweeps (if used) should be defined declaratively (W&B sweep YAML or
  equivalent) and committed to `config/`, not run ad hoc from the CLI —
  consistent with the project's "config drives behavior, not hard-coded
  script edits" convention already used for target policy.
- Credentials: W&B API key follows the same rule as all other credentials
  in `docs/repo_policy.md` — never committed, never logged.

## 11. Slurm partition / GRES parameterization policy

Both existing sbatch templates (`run_stage1_smoke0/1_moriah.sbatch`)
hard-pin `--partition=catfish --gres=gpu:l4:1`. This was **explicitly
deferred** in the 2026-07-02 decision log entry ("Future Slurm improvement")
until the reproducibility baseline is established, specifically to avoid
confounding GPU hardware changes with the first scientific comparison.

**Decision for this milestone:** parameterize `PARTITION` and `GRES` as
variables at the top of the sbatch scripts *before* the first scientific
baseline run — not after — so the baseline run's evidence bundle records
which GPU it used, and future runs can target `salmon` (L40S) or `goldfish`
(H200) without editing the script body. This is a small, low-risk change
that removes a reproducibility gap right when it starts to matter (the
Smoke 0/1 runs did not need this since they were plumbing checks only).

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
  URL/ID (§10), and the Slurm `sacct` record (node, partition, GRES,
  elapsed, exit code) — same fields already captured for Smoke 0/1 in
  `docs/decision_log.md`.

---

## Status

**DESIGN GATE OPEN.** No values in §1–§11 marked **OPEN** are decided. This
document exists to make those decisions explicit and reviewable in one place
before the first scientific-baseline training run is configured or
launched. Next step: resolve the **OPEN** items (likely via targeted
`AskUserQuestion` rounds or explicit user sign-off per section), then encode
the resolved policy into a versioned config
(`config/stage1_scientific_baseline_v001.yaml` + NH YAML), following the
same "config drives behavior" pattern as `config/stage1_target_policy.yaml`.

This document does **not** authorize full 2,752-basin package generation or
training. Both remain separate, explicitly gated steps.
