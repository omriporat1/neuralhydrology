# Flash-NH Basin Screening Decision Memo

**Date**: 2026-05-27 (updated after final pre-training basin selection; prior: 2026-05-19)
**Status**: Final pre-training selection complete — flashnh_final_basin_selection_v001
**Author**: Flash-NH research team

---

## 1. Executive Recommendation

**Initial training set: 2,843 basins (TRAIN_CORE + TRAIN_SOFT_KEEP, from flashnh_final_basin_selection_v001). Held for secondary review: 156 basins (HOLDOUT_REVIEW). Hard-excluded: 35 basins (EXCLUDE_TRAINING).**

| Final status | Count | % of 3,034 | Notes |
|---|---|---|---|
| TRAIN_CORE | 2,216 | 73.0% | No risk flags; passes all checks |
| TRAIN_SOFT_KEEP | 627 | 20.7% | One moderate risk flag; included in training, tracked as risk stratum |
| HOLDOUT_REVIEW | 156 | 5.1% | Compound regulation/lentic risk; withheld pending residual analysis |
| EXCLUDE_TRAINING | 35 | 1.2% | Manual EXCLUDE label or rule_A override |
| **Initial training set** | **2,843** | **93.7%** | TRAIN_CORE + TRAIN_SOFT_KEEP |

The main training universe of 3,034 main_training_candidate basins was established in two automated screening passes:
1. **Streamflow hard-QC**: 279 basins excluded for data-quality failures (completeness, negative flow, near-zero median, missing RBI) → 3,045 hard-QC-passing basins.
2. **USGS site-type metadata audit** (completed 2026-05-19): 11 additional basins excluded from the hard-QC-passing set because their USGS monitoring-location `site_type_code` identifies them as tidal streams or lake/reservoir sites → **3,034 main_training_candidate basins**.

A subsequent two-pass manual hydrograph review (148 basins total: 73 in pass-1, 75 in pass-2) calibrated risk rules against human judgment and produced the final pre-training basin status. See Section 7 for the final selection details and Section 9 for the manual-review execution record.

Do not restrict model training or evaluation to Pilot-100 or Pilot-150. Those subsets are useful only for debugging and rapid iteration. The candidate classes (FLASHY_CORE, FLASHY_MODERATE, FLASHY_POSSIBLE, LOW_FLASHINESS_CONTROL, MANUAL_REVIEW_CONTEXT) should be retained as stratification metadata and diagnostic strata — not as strict inclusion filters.

**The screening phase is complete.** The next step is to begin meteorological forcing preprocessing for the initial training set of 2,843 basins (`reports/flashnh_final_basin_selection_v001/tables/training_basin_list_initial.csv`).

---

## 2. Screening-Stage Lineage

The following table summarizes basin counts at each filtering stage.

| Stage | Approximate count | Notes |
|---|---|---|
| GAGES-II / CAMELSH total | 9,008 | Full USGS-monitored basin catalog |
| Area-filtered (1–1,000 km²) | ~5,836 | Headwater to mid-size; excludes tiny hillslopes and large routed rivers |
| BFI exploration (BFI ≤ 40) | ~2,130 | Exploratory only; USGS audit was run on all area-filtered basins |
| USGS coverage-eligible (ELIGIBLE_SCREENING_WY) | ~3,647 | Metadata confirms parameter 00060 overlaps WY2024 window |
| WY2024 streamflow metrics computed | 3,324 | Successful hourly discharge retrieval and RBI calculation |
| Streamflow hard-QC excluded | 279 (8.4%) | See Section 4 for exclusion reasons |
| Hard-QC-passing (pre-metadata) | 3,045 (91.6%) | All basins passing streamflow quality thresholds |
| Metadata hard-exclusions (site-type audit) | 19 total; 11 from hard-QC-passing | 18 tidal streams (ST-TS) + 1 lake/reservoir (LK); see Section 3 |
| **main_training_candidate universe** | **3,034** | Hard-QC-pass AND metadata policy = ACCEPT |
| Manual review pass-1 | 73 basins reviewed | Stratified sample enriched for suspicious cases |
| Manual review pass-2 | 75 basins reviewed | Targeted sample from HOLDOUT_REVIEW_PRELIM + per-rule tiers |
| **EXCLUDE_TRAINING (final)** | **35** | Manual EXCLUDE (both passes) + rule_A override |
| **HOLDOUT_REVIEW (final)** | **156** | CDEJ compound risk ≥ 2; manual UNSURE decisions |
| **Initial training set (TRAIN_CORE + TRAIN_SOFT_KEEP)** | **2,843** | 93.7% of main_training_candidate; see Section 7 |

Note: 8 of the 19 metadata hard-exclusions were already in the streamflow EXCLUDE_HARD_QC class (they failed both streamflow QC and metadata QC). The 11 that are new exclusions were previously in the hard-QC-passing set and would have been silently included without the metadata audit.

### Candidate class breakdown (hard-QC-passing pre-metadata, n=3,045)

| Class | Count | Share | Median RBI |
|---|---|---|---|
| FLASHY_CORE | 397 | 13.0% | 0.177 |
| FLASHY_MODERATE | 503 | 16.5% | 0.066 |
| FLASHY_POSSIBLE | 2,082 | 68.4% | 0.021 |
| LOW_FLASHINESS_CONTROL | 5 | 0.2% | 0.010 |
| MANUAL_REVIEW_CONTEXT | 58 | 1.9% | 0.082 |

RBI class boundaries are: CORE ≥ 0.10, MODERATE 0.05–0.10, POSSIBLE 0.001–0.05, CONTROL < 0.05 by overlapping definition, MANUAL_REVIEW assigned for specific flag combinations regardless of RBI value.

### Geographic coverage

Basins span 48 states and all 19 HUC02 regions. Largest concentrations: CA (226), TX (187), PA (162), FL (150), NY (137). HUC02 regions with most coverage: 02 Mid-Atlantic (524), 03 South Atlantic (521), 17 Pacific Northwest (306), 05 Ohio (280).

---

## 3. USGS Site-Type Metadata Audit

**Completed**: 2026-05-19. Script: `scripts/build_usgs_site_metadata_audit.py`. Outputs: `reports/flashnh_usgs_site_metadata_v001/`.

### Motivation

USGS monitoring-location records include a closed `site_type_code` field that identifies the physical character of the monitoring site (stream, tidal stream, lake, estuary, well, etc.). During manual hydrograph review, station 02247222 (Pellicer Creek Near Espanola, FL) was identified as tide-dominated and unsuitable for rainfall-runoff / flash-flood modeling. Its USGS `site_type_code` is `ST-TS` (Tidal stream). A systematic metadata audit was run to identify all such stations using this closed field rather than relying on hydrograph shape alone.

### Policy buckets

| Bucket | Criteria | Count in WY2024 set |
|---|---|---|
| ACCEPT | ST (Stream) | 3,305 |
| REVIEW | ST-CA, ST-DCH, SP, LA-SNK, LA-PLY, LA-SR | 0 |
| HARD_EXCLUDE | ST-TS, ES, OC, OC-CO, LK, GW-\*, SB-\*, FA-\*, LA-\*, AT, GL, WE, AG, AS, AW | 19 |
| MISSING_METADATA | site_type_code absent or retrieval failed | 0 |

All 3,324 basins in the WY2024 metrics table received metadata (0 missing).

### Hard-excluded sites (n=19)

All 19 metadata hard-exclusions are either tidal streams (ST-TS, n=18) or a lake/reservoir (LK, n=1). Geographic breakdown: FL (9 sites), MA (6 sites), TX (1 site), RI (1 site), plus 1 lake in FL. The 11 that were previously in the hard-QC-passing set include stations such as:

- 01105585 TOWN BROOK AT QUINCY, MA (FLASHY_CORE, ST-TS)
- 01103025 ALEWIFE BROOK NEAR ARLINGTON, MA (FLASHY_MODERATE, ST-TS)
- 08074000 Buffalo Bayou at Houston, TX (FLASHY_POSSIBLE, ST-TS)
- 02300042 WARD LAKE NEAR BRADENTON FL (FLASHY_POSSIBLE, LK)
- 01102345 SAUGUS RIVER AT SAUGUS IRONWORKS AT SAUGUS, MA (FLASHY_POSSIBLE, ST-TS)

### Why metadata exclusions matter

Streamflow hard-QC metrics (completeness, median flow, RBI) measure data availability and signal character, not site physics. A tidal stream with high data completeness and a plausible RBI will pass streamflow QC because its hydrograph superficially resembles a flashy stream — tidal oscillations produce repeated rapid rises and falls. The metadata audit uses the authoritative closed site_type_code to remove these sites before training, preventing the model from learning tidal dynamics as if they were rainfall-runoff responses.

### Training candidate flags

- **main_training_candidate**: hard_qc_pass AND metadata_policy_bucket == ACCEPT → **3,034 basins**
- **inclusive_training_candidate**: hard_qc_pass AND metadata_policy_bucket in {ACCEPT, REVIEW} → **3,034 basins** (identical, because no REVIEW-type sites were found in the WY2024 set)

The `site_type_code`, `site_type_group`, USGS drainage area, contributing area, and HUC8 code retrieved during this audit are stored in `reports/flashnh_usgs_site_metadata_v001/tables/static_metadata_attributes_candidates.csv` as candidate static attributes for the model. These closed physical/geographic fields are appropriate model inputs. Do not use `metadata_policy_bucket`, `metadata_exclusion_reason`, or QC labels as model inputs.

---

## 4. Streamflow Hard Exclusions vs. Soft Context Flags

These are fundamentally different in meaning and should not be conflated.

### Streamflow hard exclusions (n=279, permanent removal)

These basins fail objective data-quality thresholds and are excluded from all downstream analysis:

| Exclusion flag | Count | Meaning |
|---|---|---|
| HARD_LOW_COMPLETENESS_LT90 | 232 | Hourly completeness < 90% over WY2024 |
| HARD_NEGATIVE_FLOW_SEVERE | 25 | Implausible negative discharge values |
| HARD_Q50_ZERO_OR_NEAR_ZERO | 25 | Median hourly flow at or near zero; RBI undefined |
| HARD_NO_RBI | 18 | RBI could not be computed (zero total flow or missing) |

Hard exclusions are not reversible without re-downloading and reprocessing data for a different water year. They should be treated as permanent for WY2024 screening purposes.

### Context flags (informational, n up to ~1,042 basins with ≥1 flag)

Context flags are diagnostic annotations. They do not exclude a basin. A context-flagged basin that passes hard QC is still in the usable universe. The flags signal that human inspection is warranted before high-confidence scientific interpretation.

| Context flag | Count | Concern |
|---|---|---|
| CONTEXT_SUSPICIOUS_SPIKE_SEVERE | 1,239 | max_jump/Q50 ≥ 20; possible data artifact or extreme event |
| CONTEXT_HIGH_NORMALIZED_JUMP | 830 | Moderate spike pattern |
| CONTEXT_HIGH_BFI | 642 | Baseflow-dominated; low flashiness; included as context, not exclusion |
| CONTEXT_HIGH_SPECIFIC_PEAK | 522 | Q_max/km² ≥ 1.0; regulation or rare event possible |
| CONTEXT_ZERO_FLOW_SOME | 280 | Zero-flow fraction ≥ 5%; intermittent character |
| CONTEXT_LOW_SPECIFIC_FLOW | 241 | Very low flow regime |
| CONTEXT_INTERMITTENT_LIKE | 231 | Zero-flow fraction ≥ 10% |
| CONTEXT_SMALL_BASIN | 79 | Drainage area < 10 km²; sparse forcing grids |
| CONTEXT_LOW_BFI | 32 | Very low BFI; potential for artifact spike misclassification |

A single basin can carry multiple flags. The 1,042-basin figure in `manual_review_priority.csv` reflects basins with at least one moderate-to-severe context flag, not 1,042 independent problems.

---

## 5. Why Pilot-100 Is Useful for Debugging but Too Small for Model Assessment

The current Pilot-100 is composed of 92 FLASHY_CORE and 8 FLASHY_MODERATE basins (median RBI 0.180). This is by design — the scoring function heavily weights FLASHY_CORE — but it creates several problems for model assessment:

- **Class imbalance**: 92% FLASHY_CORE basins is not representative of the 3,034-basin universe (13% FLASHY_CORE). A model evaluated only on Pilot-100 will be assessed almost entirely on high-RBI basins.
- **No representation of FLASHY_POSSIBLE**: 68% of usable basins are FLASHY_POSSIBLE (RBI 0.001–0.050). Pilot-100 contains none. Model behavior on moderate-response basins is unknown.
- **Geographic clustering risk**: 100 basins are insufficient to sample 48 states and 19 HUC02 regions without strong geographic concentration.
- **No controls**: Only 5 LOW_FLASHINESS_CONTROL basins exist; Pilot-100 may include none or very few.

Pilot-100 (or Pilot-150) remains appropriate for:
- Initial dataset pipeline testing (forcing extraction, NeuralHydrology dataset generation)
- Training runs with rapid feedback (<1 hour)
- Debugging model architecture, loss functions, and preprocessing

For scientific model evaluation — generalization across basin types, geographic regions, and flashiness regimes — the full 3,034-basin universe is required.

---

## 6. Recommended Carry-Forward: main_training_candidate Universe

**Decision**: Use all 3,034 main_training_candidate basins as the screening universe. The final pre-training selection is documented in Section 7.

These basins pass both:
1. Streamflow hard-QC (no HARD_* flag)
2. USGS site-type metadata audit (metadata_policy_bucket == ACCEPT, i.e., site_type_code == ST)

**How to use candidate classes**:

- **As metadata and stratification strata**: Report model performance broken out by candidate class (CORE, MODERATE, POSSIBLE, CONTROL).
- **As sampling weights for pilot subsets**: When compute constraints require a smaller training set initially, sample proportionally by class rather than selecting only CORE/MODERATE.
- **Not as strict inclusion filters**: Do not exclude FLASHY_POSSIBLE basins from training. The model will need to generalize across the full flashiness spectrum.

**MANUAL_REVIEW_CONTEXT (n=58)**: Included in training unless specifically excluded by manual review label. These are basins with moderate-to-high RBI that also carry context flags.

**EXCLUDE_HARD_QC (n=279)**: Do not include. These fail objective streamflow data quality criteria.

**Metadata HARD_EXCLUDE (n=19, of which 11 overlap with hard-QC-passing)**: Do not include. These are non-stream monitoring locations (tidal streams, lake/reservoir) unsuitable for rainfall-runoff model training.

---

## 7. Final Pre-Training Basin Selection (flashnh_final_basin_selection_v001)

**Completed**: 2026-05-27. Scripts: `scripts/analyze_combined_manual_review_results.py`,
`scripts/build_final_basin_training_status.py`. Outputs: `reports/flashnh_final_basin_selection_v001/`.

### 7.1 Policy Summary

The final pre-training basin status was determined by combining automated risk rules with
human review labels from two manual inspection passes. The policy applies in priority order:

| Final status | Trigger | Count |
|---|---|---|
| EXCLUDE_TRAINING | manual EXCLUDE, or rule_A (previous EXCLUDE designation) | 35 |
| HOLDOUT_REVIEW | CDEJ compound risk ≥ 2, or manual UNSURE | 156 |
| TRAIN_SOFT_KEEP | single CDEJ/G/H/F/I rule flag, or manual KEEP_LOW_CONFIDENCE | 627 |
| TRAIN_CORE | no flags trigger | 2,216 |

CDEJ = rule_C (HYDRO_DISTURB_INDX) + rule_D (lake/reservoir area %) +
rule_E (degree of regulation) + rule_J (canals/DOR). The ≥ 2 threshold was calibrated
against the combined review set (exclude-or-unsure rate 0.45–0.69 for CDEJ≥2 combinations).

### 7.2 Key Interpretation

- **TRAIN_CORE + TRAIN_SOFT_KEEP (2,843 basins)** form the initial training set. TRAIN_SOFT_KEEP
  basins are fully included but tracked as a risk stratum for post-training residual analysis.

- **HOLDOUT_REVIEW (156 basins)** are not permanently excluded. They are withheld from the
  first training run and should be revisited after post-training residual analysis or targeted
  secondary review. Roughly 30–50% of CDEJ≥2 basins reviewed as part of pass-2 were labeled
  KEEP on visual inspection, so the holdout pool contains legitimate training candidates.

- **EXCLUDE_TRAINING (35 basins)** are hard exclusions. All 35 reflect either manual EXCLUDE
  labels (where the reviewer identified regulation artifacts, sensor problems, or lake-dominated
  hydrology) or the rule_A override from the pass-1 analysis. Zero basins were excluded purely
  by automated thresholds without human confirmation.

- **Rule_G (mostly-zero / ephemeral) and rule_H (extreme hourly jumps) do not trigger holdout
  or exclusion when appearing alone.** Both were confirmed by the combined review to reflect
  real hydrological signals (intermittent streams and genuine flashy response, respectively).
  Their exclude-or-unsure rates in the review set were 0.22 and 0.15 — below the 0.35 threshold
  used to recommend holdout-level action.

### 7.3 Evidence from Combined Manual Review

- 148 basins reviewed total (73 pass-1 + 75 pass-2); no overlapping STAIDs.
- Combined exclude rate 23.6%; unsure rate 10.1%; total concern rate 33.8%.
- **Regulation/lentic risk** is the dominant exclusion driver. Reviewer notes mentioning
  "regulated/managed" (n=28, exclude-or-unsure rate 0.82) and "dam/reservoir" (n=26, rate 0.77)
  strongly predict exclusion decisions.
- **Ephemeral/zero-flow mentions** (n=28) have exclude-or-unsure rate 0.04 — confirming these
  should remain in training.

### 7.4 Key Output Files

| File | Contents |
|---|---|
| `tables/final_basin_training_status.csv` | 3,034 rows; final_training_status, reasons, all rule flags |
| `tables/training_basin_list_initial.csv` | 2,843-row STAID list for initial training run |
| `tables/holdout_basin_list.csv` | 156-row STAID list for secondary review |
| `tables/excluded_basin_list.csv` | 35-row STAID list of hard exclusions |
| `summaries/final_basin_selection_summary.md` | Narrative policy documentation |

---

## 8. Remaining Concerns

### 7.1 Outlier-sensitive plots and metric distributions

Several metrics exhibit extreme right skew that will distort visualizations and summary statistics:

| Metric | Median | Mean | P95 | Max |
|---|---|---|---|---|
| max_hourly_rise_per_km² (m³/s/km²/hr) | 0.056 | 1.485 | 0.891 | **1,670** |
| max_abs_jump / Q50 | 10.8 | 8,520 | 517 | **~25,000,000** |
| Q95/Q50 ratio | 6.8 | 17.0 | 46 | **4,123** |
| Q99/Q50 ratio | 17.3 | 68.8 | 172 | **20,827** |

The max values are orders of magnitude above the 95th percentile. Before finalizing plots or reporting derived statistics:

- Apply log scale or log-transform for all four metrics in any figure.
- Report medians and percentiles (P10, P25, P50, P75, P90, P95), not means, for skewed metrics.
- Flag basins at or above P99 in each metric for manual inspection.

### 7.2 High skew in RBI, Q95/Q50, and max rise

Even within FLASHY_CORE, RBI ranges from 0.100 to 1.520 (median 0.177). EXCLUDE_HARD_QC basins have RBI up to 2.07 — the highest values in the dataset, which reflects that high-RBI basins are also more likely to have severe spikes or completeness problems. The overall RBI distribution (median 0.031, P95 0.238, max 2.07) is strongly right-skewed; log-RBI is the preferred axis for exploratory plots.

### 7.3 Possible artifacts or regulated basins

CONTEXT_SUSPICIOUS_SPIKE_SEVERE (n=1,239) and CONTEXT_HIGH_NORMALIZED_JUMP (n=830) flag basins where a single hourly change is implausibly large relative to background flow. Likely sources:

- Sensor calibration errors or rating curve step changes
- Regulated releases (dam operations, controlled spills)
- Genuine extreme flash floods (cannot be dismissed a priori)

Manual review of the top ~30 worst-case basins by max_abs_jump/Q50 ratio will resolve most ambiguous cases.

### 7.4 Zero-flow basins are not automatically bad

CONTEXT_INTERMITTENT_LIKE (n=231) and CONTEXT_ZERO_FLOW_SOME (n=280) flag basins with non-trivial zero-flow fractions. These should not be excluded by default. Many are genuine ephemeral or intermittent streams — scientifically important for flash-flood modeling in arid and semi-arid regions. The HARD_Q50_ZERO_OR_NEAR_ZERO exclusion already removes the extreme case (median flow at zero). Basins with seasonal or partial zero flow but a non-zero median remain in the usable universe and should be treated as valid training samples with appropriate model handling of zero-flow states.

---

## 9. Manual Review Execution Record (Completed)

Manual hydrograph review was completed across two passes. The strategy originally described
in this section (Tier 1 / Tier 2 / Tier 3) was executed as pass-1 and pass-2.

### Pass 1 — Tier 1 + Tier 2 stratified sample (73 basins, v004 review set)

Enriched for suspicious cases: extreme RBI, extreme rise rates per km², high Q-ratio,
context-flagged basins, and candidate-class reference strata. Labels locked to
`manual_review_labels_pass1_locked.csv`. Results:

- KEEP: 40 (54.8%) | KEEP_LOW_CONFIDENCE: 12 (16.4%) | UNSURE: 5 (6.8%) | EXCLUDE: 16 (21.9%)

### Pass 2 — Rule validation sample (75 basins, v005 review set)

Targeted sample stratified by risk rule tier: 25 basins with compound risk (CDEGHJ ≥ 2),
8 basins per per-rule tier-2 sample (rules C, D, E, J), 5 each for rules G and H,
4 near-threshold controls, 4 clean CORE controls. Labels in `manual_review_labels_pass2.csv`. Results:

- KEEP: 34 (45.3%) | KEEP_LOW_CONFIDENCE: 12 (16.0%) | UNSURE: 10 (13.3%) | EXCLUDE: 19 (25.3%)

### Combined result

148 total reviewed basins; no overlapping STAIDs. Combined exclude-or-unsure rate: 33.8%.
Scripts: `scripts/analyze_combined_manual_review_results.py`.
Full analysis: `reports/flashnh_combined_manual_review_analysis_v001/`.

Remaining ~900 context-flagged basins not individually reviewed will be handled via:
- Model residual analysis post-training (outlier basins will show anomalously high loss)
- Targeted secondary review of HOLDOUT_REVIEW basins after the first training run

---

## 10. Next Recommended Milestones

### Milestone 1 — Manual review and final basin selection ✓ COMPLETE

Basin selection is complete. Final status assigned to all 3,034 main_training_candidate basins.
Initial training set: 2,843 basins. See Section 7 and `reports/flashnh_final_basin_selection_v001/`.

### Milestone 2 — Meteorological preprocessing (current milestone)

**Goal**: Prepare ERA5/MRMS/RTMA basin-average forcing for the initial training set of 2,843 basins.

- Use `reports/flashnh_final_basin_selection_v001/tables/training_basin_list_initial.csv` as the
  authoritative basin list for forcing extraction.
- Use Pilot-100 or Pilot-150 as the debugging subset for initial pipeline validation only.
- The 156 HOLDOUT_REVIEW basins may optionally have forcing prepared in parallel (they are not
  excluded permanently and may enter training after secondary review).

**Rationale**: Forcing preprocessing is the longest-lead-time task in the pipeline.

### Milestone 3 — Stage 1 model training and evaluation

- Train Flash-NH on the 2,843-basin initial training set.
- Report performance stratified by candidate class, area bin, BFI bin, HUC02, and
  final_training_status (TRAIN_CORE vs. TRAIN_SOFT_KEEP).
- Use TRAIN_SOFT_KEEP basins as an embedded risk stratum: identify whether they show
  systematically elevated residuals relative to TRAIN_CORE.
- Use Pilot-100/150 results as debugging benchmarks, not as primary evaluation targets.

### Milestone 4 — Post-training residual analysis and HOLDOUT_REVIEW secondary decision

- Rank all training basins by validation-period NSE/KGE. Identify the bottom decile.
- Cross-check bottom-decile basins against final_training_status and rule flags.
- For HOLDOUT_REVIEW basins: review those with the highest model-predicted confidence
  of clean hydrology. Add confirmed-clean basins to training for a second model run.
- For confirmed-problematic TRAIN_SOFT_KEEP or TRAIN_CORE basins: move to EXCLUDE_TRAINING
  and document the data-driven justification in flashnh_final_basin_selection_v002.

---

## References

| Document | Path |
|---|---|
| Basin screening strategy | `docs/basin_screening_strategy.md` |
| Pilot selection strategy | `docs/pilot_selection_strategy.md` |
| RBI screening strategy | `docs/usgs_rbi_screening_strategy.md` |
| Coverage eligibility strategy | `docs/usgs_coverage_eligibility_strategy.md` |
| Pilot selection summary (MD) | `reports/flashnh_wy2024_pilot_selection_v001/summaries/pilot_selection_summary.md` |
| Pilot selection summary (JSON) | `reports/flashnh_wy2024_pilot_selection_v001/summaries/pilot_selection_summary.json` |
| Streamflow metrics matrix | `docs/wy2024_streamflow_metrics_matrix.md` |
| Agent handoff rules | `docs/agent_handoff_rules.md` |
| Site metadata audit script | `scripts/build_usgs_site_metadata_audit.py` |
| Site metadata audit summary (MD) | `reports/flashnh_usgs_site_metadata_v001/summaries/usgs_site_metadata_audit_summary.md` |
| Site metadata audit summary (JSON) | `reports/flashnh_usgs_site_metadata_v001/summaries/usgs_site_metadata_audit_summary.json` |
| Site metadata joined table | `reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv` |
| Static metadata attribute candidates | `reports/flashnh_usgs_site_metadata_v001/tables/static_metadata_attributes_candidates.csv` |
| Manual review filter rule methodology | `docs/manual_review_filter_rule_methodology.md` |
| Rule analysis script | `scripts/analyze_manual_review_filter_rules.py` |
| Rule analysis outputs | `reports/flashnh_manual_review_rule_analysis_v001/` |
| Combined review analysis script | `scripts/analyze_combined_manual_review_results.py` |
| Combined review analysis outputs | `reports/flashnh_combined_manual_review_analysis_v001/` |
| Final basin selection script | `scripts/build_final_basin_training_status.py` |
| Final basin selection outputs | `reports/flashnh_final_basin_selection_v001/` |
| Final basin selection summary (MD) | `reports/flashnh_final_basin_selection_v001/summaries/final_basin_selection_summary.md` |
| Initial training basin list | `reports/flashnh_final_basin_selection_v001/tables/training_basin_list_initial.csv` |
| Full final training status table | `reports/flashnh_final_basin_selection_v001/tables/final_basin_training_status.csv` |
