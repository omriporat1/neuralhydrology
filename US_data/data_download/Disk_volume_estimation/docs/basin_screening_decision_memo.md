# Flash-NH Basin Screening Decision Memo

**Date**: 2026-05-13
**Status**: Decision-ready — screening phase complete
**Author**: Flash-NH research team

---

## 1. Executive Recommendation

**Carry forward all 3,045 hard-QC-passing basins as the Flash-NH usable basin universe.**

Do not restrict model training or evaluation to Pilot-100 or Pilot-150. Those subsets are useful only for debugging and rapid iteration. The candidate classes (FLASHY_CORE, FLASHY_MODERATE, FLASHY_POSSIBLE, LOW_FLASHINESS_CONTROL, MANUAL_REVIEW_CONTEXT) should be retained as stratification metadata and diagnostic strata — not as strict inclusion filters.

The screening phase is effectively complete. The remaining work before meteorological preprocessing is:

1. Improve summary visualizations for human-interpretable QC.
2. Spot-check a prioritized sample of suspicious basins — not all 1,000+ context-flagged basins.
3. Begin meteorological forcing preprocessing for the full 3,045-basin universe in parallel with targeted manual review.

---

## 2. Screening-Stage Lineage

The following table summarizes the approximate basin counts at each filtering stage.

| Stage | Approximate count | Notes |
|---|---|---|
| GAGES-II / CAMELSH total | 9,008 | Full USGS-monitored basin catalog |
| Area-filtered (1–1,000 km²) | ~5,836 | Headwater to mid-size; excludes tiny hillslopes and large routed rivers |
| BFI exploration (BFI ≤ 40) | ~2,130 | Exploratory only; USGS audit was run on all area-filtered basins |
| USGS coverage-eligible (ELIGIBLE_SCREENING_WY) | ~3,647 | Metadata confirms parameter 00060 overlaps WY2024 window |
| WY2024 streamflow metrics computed | 3,324 | Successful hourly discharge retrieval and RBI calculation |
| Hard-QC excluded | 279 (8.4%) | See Section 3 for exclusion reasons |
| **Usable basin universe** | **3,045 (91.6%)** | All hard-QC-passing basins; recommended carry-forward set |

### Candidate class breakdown (usable universe, n=3,045)

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

## 3. Hard Exclusions vs. Soft Context Flags

These are fundamentally different in meaning and should not be conflated.

### Hard exclusions (n=279, permanent removal)

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

## 4. Why Pilot-100 Is Useful for Debugging but Too Small for Model Assessment

The current Pilot-100 is composed of 92 FLASHY_CORE and 8 FLASHY_MODERATE basins (median RBI 0.180). This is by design — the scoring function heavily weights FLASHY_CORE — but it creates several problems for model assessment:

- **Class imbalance**: 92% FLASHY_CORE basins is not representative of the 3,045-basin universe (13% FLASHY_CORE). A model evaluated only on Pilot-100 will be assessed almost entirely on high-RBI basins.
- **No representation of FLASHY_POSSIBLE**: 68% of usable basins are FLASHY_POSSIBLE (RBI 0.001–0.050). Pilot-100 contains none. Model behavior on moderate-response basins is unknown.
- **Geographic clustering risk**: 100 basins are insufficient to sample 48 states and 19 HUC02 regions without strong geographic concentration.
- **No controls**: Only 5 LOW_FLASHINESS_CONTROL basins exist; Pilot-100 may include none or very few.

Pilot-100 (or Pilot-150) remains appropriate for:
- Initial dataset pipeline testing (forcing extraction, NeuralHydrology dataset generation)
- Training runs with rapid feedback (<1 hour)
- Debugging model architecture, loss functions, and preprocessing

For scientific model evaluation — generalization across basin types, geographic regions, and flashiness regimes — the full 3,045-basin universe is required.

---

## 5. Recommended Carry-Forward: Broad Hard-QC-Passing Universe

**Decision**: Use all 3,045 hard-QC-passing basins as the training and evaluation universe.

**How to use candidate classes**:

- **As metadata and stratification strata**: Report model performance broken out by candidate class (CORE, MODERATE, POSSIBLE, CONTROL).
- **As sampling weights for pilot subsets**: When compute constraints require a smaller training set initially, sample proportionally by class rather than selecting only CORE/MODERATE.
- **Not as strict inclusion filters**: Do not exclude FLASHY_POSSIBLE basins from training. The model will need to generalize across the full flashiness spectrum.

**MANUAL_REVIEW_CONTEXT (n=58)**: Include in training for now. These are basins with moderate-to-high RBI that also carry context flags. Their scientific value is real; the flags indicate inspection priority, not exclusion.

**EXCLUDE_HARD_QC (n=279)**: Do not include. These fail objective data quality criteria.

---

## 6. Remaining Concerns

### 6.1 Outlier-sensitive plots and metric distributions

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

### 6.2 High skew in RBI, Q95/Q50, and max rise

Even within FLASHY_CORE, RBI ranges from 0.100 to 1.520 (median 0.177). EXCLUDE_HARD_QC basins have RBI up to 2.07 — the highest values in the dataset, which reflects that high-RBI basins are also more likely to have severe spikes or completeness problems. The overall RBI distribution (median 0.031, P95 0.238, max 2.07) is strongly right-skewed; log-RBI is the preferred axis for exploratory plots.

### 6.3 Possible artifacts or regulated basins

CONTEXT_SUSPICIOUS_SPIKE_SEVERE (n=1,239) and CONTEXT_HIGH_NORMALIZED_JUMP (n=830) flag basins where a single hourly change is implausibly large relative to background flow. Likely sources:

- Sensor calibration errors or rating curve step changes
- Regulated releases (dam operations, controlled spills)
- Genuine extreme flash floods (cannot be dismissed a priori)

Manual review of the top ~30 worst-case basins by max_abs_jump/Q50 ratio will resolve most ambiguous cases.

### 6.4 Zero-flow basins are not automatically bad

CONTEXT_INTERMITTENT_LIKE (n=231) and CONTEXT_ZERO_FLOW_SOME (n=280) flag basins with non-trivial zero-flow fractions. These should not be excluded by default. Many are genuine ephemeral or intermittent streams — scientifically important for flash-flood modeling in arid and semi-arid regions. The HARD_Q50_ZERO_OR_NEAR_ZERO exclusion already removes the extreme case (median flow at zero). Basins with seasonal or partial zero flow but a non-zero median remain in the usable universe and should be treated as valid training samples with appropriate model handling of zero-flow states.

---

## 7. Efficient Manual-Review Strategy

There are ~1,042 basins in `manual_review_priority.csv`. Reviewing all of them before proceeding is not feasible and is not necessary. The following prioritized strategy covers the scientifically important cases with a tractable workload.

### Tier 1 — Review immediately (~30–50 basins)

These are high-risk basins where artifacts could substantially bias model training or reported performance:

- Top 30 basins by `max_abs_hourly_jump_over_Q50` (max value ~25M; clearly non-physical)
- Top 20 basins by `max_hourly_rise_per_km2` (max 1,670; verify against neighboring sites)
- All 58 MANUAL_REVIEW_CONTEXT basins (already small; sort by pilot_score and inspect top 20)

For each: plot the WY2024 hourly hydrograph. If spikes are isolated single-point outliers, consider whether to despike or exclude. If the pattern reflects genuine regulation or natural extreme events, document and retain.

### Tier 2 — Stratified sample before final reporting (~50–80 basins)

Before publishing or submitting model results, sample and inspect:

- 5 basins per candidate class × 3 area bins × 2 BFI bins = up to 30 stratification cells
- Top 10 basins per HUC02 region by RBI (geographic representativeness check)
- 10 basins with zero_flow_fraction between 0.10 and 0.50 (intermittent system check)
- 10 FLASHY_POSSIBLE basins with the lowest RBI (close to threshold; confirm they are truly low-response)

### Tier 3 — Deferred (remaining ~900 context-flagged basins)

Do not review before proceeding to meteorological preprocessing. These basins carry flags but are not in the extreme tails. They will be handled via:

- Automated outlier clipping in the data pipeline (e.g., cap extreme spikes at P99.9)
- Model residual analysis post-training (outlier basins will show anomalously high loss)
- Targeted manual review only if specific basins surface as problematic during training

---

## 8. Next Recommended Milestones

### Milestone 1 — Visualization and reporting improvements (before meteorological preprocessing)

**Goal**: Make the 3,045-basin universe human-interpretable for ongoing QC.

- Generate log-scale histograms and maps for: RBI, Q95/Q50, max_hourly_rise_per_km2, zero_flow_fraction
- Generate class-by-class comparison plots (box plots of RBI, completeness, area, BFI by candidate class)
- Produce a geographic map of all 3,045 usable basins colored by candidate class and RBI
- Complete Tier 1 manual review (~30–50 basins) and document decisions

These are lightweight post-processing tasks; they do not require new USGS downloads.

### Milestone 2 — Meteorological preprocessing for the full usable basin set (main workstream)

**Goal**: Prepare ERA5/MRMS/RTMA basin-average forcing for all 3,045 usable basins.

- Begin forcing extraction pipeline for all 3,045 hard-QC-passing basins
- Use Pilot-100 or Pilot-150 as the debugging subset for initial pipeline validation only
- Do not wait for Tier 2/3 manual review to complete before starting forcing extraction; those reviews are deferred

**Rationale**: Forcing preprocessing is the longest-lead-time task in the pipeline. Starting it now, while Tier 1 review and visualization improvements are in progress, is the fastest path to Stage 1 model runs with a scientifically defensible basin set.

### Milestone 3 — Stage 1 model training and evaluation

- Train Flash-NH on the full 3,045-basin set (or the maximum subset for which forcing data are ready)
- Report performance stratified by candidate class, area bin, BFI bin, and HUC02
- Use Pilot-100/150 results as debugging benchmarks, not as primary evaluation targets

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
