# Manual Review Filter Rule Methodology — Flash-NH Basin Screening

**Version**: 2.0
**Status**: Final — two-pass manual review completed; final basin selection complete (flashnh_final_basin_selection_v001).

---

## 1. Why Manual Review Was Performed

The Flash-NH project uses WY2024 hourly streamflow data from USGS NWIS for 3,034
main-training-candidate basins selected from the GAGES-II/CAMELSH catalog. Automated
streamflow QC (hard exclusions) and USGS site-type metadata audit already removed
~300 basins. However, automated QC cannot distinguish all problematic basins from
legitimately extreme hydrology.

Two manual review passes were completed. **Pass 1** drew a stratified random sample of
73 basins enriched for suspicious cases (extreme RBI, extreme rise rates, high Q-ratio,
context-flagged basins) to calibrate whether automated flags reflect real problems or
real flashy hydrology. **Pass 2** drew 75 basins targeted at the HOLDOUT_REVIEW_PRELIM
pool, stratified by risk rule tier (compound-risk CDEGHJ≥2, per-rule tier-2 samples,
near-threshold controls, and clean CORE controls). Together, **148 basins** were reviewed
with no overlapping STAIDs between passes.

---

## 2. Separation of QC Layers

Three independent screening layers apply to the candidate universe:

| Layer | Applied at | What it checks | Outputs |
|---|---|---|---|
| Streamflow hard-QC | Build time | Data completeness (<90%), severe negatives, zero-Q50, no-RBI | EXCLUDE_HARD_QC class |
| USGS site-type metadata | Post-QC | Site type classification (tidal streams, lakes, etc.) | metadata_policy_bucket |
| Manual review + risk rules | Post-metadata | Hydrograph quality, regulation signals, sensor artifacts | preliminary_training_status |

These layers are **independent** and applied in sequence. The manual review layer does
not revisit basins already excluded by hard-QC or metadata audit. It operates only on
the 3,034 `main_training_candidate == True` basins.

---

## 3. Manual Labels as Calibration Data, Not Classifier Labels

The 148 reviewed basins (73 pass-1 + 75 pass-2) are **not** a representative random
sample of the 3,034 main candidates. They are **stratified/enriched** for suspicious
cases. This means:

- EXCLUDE and UNSURE rates in the review set are artificially inflated relative to
  the full population.
- The labels cannot be used to train or calibrate a statistical model for basin
  selection without correcting for the sampling design.
- Any rule derived from the review labels has uncertain precision in the full
  population until validated on a broader, unbiased sample.

The labels are used as **calibration evidence** only: they tell us whether a given
automated risk rule (e.g., high HYDRO_DISTURB_INDX) tends to agree with human
judgment on the flagged basins. A rule with high `reviewed_exclude_or_unsure_rate`
has stronger calibration support than one with low rate.

---

## 4. Decision Taxonomy

### Preliminary labels (from rule analysis)

Used during the analysis phase to propose status for all 3,034 candidates:

| Preliminary label | Meaning | Policy proposal |
|---|---|---|
| TRAIN_CORE_PRELIM | No risk flags; passes all checks | Proposed: include in training data |
| TRAIN_SOFT_KEEP_PRELIM | One moderate risk flag triggers (F, I, C, D, E) | Proposed: include but track as risk stratum |
| HOLDOUT_REVIEW_PRELIM | Compound risk (≥2 CDEGHJ flags), or reviewed UNSURE | Proposed: hold out pending second-pass review |
| EXCLUDE_TRAINING_PRELIM | Reviewed EXCLUDE (manual override) or metadata HARD_EXCLUDE | Proposed: exclude; document reason |

### Final labels (from flashnh_final_basin_selection_v001)

Applied after both review passes; these are the accepted pre-training assignments:

| Final label | Count | Trigger |
|---|---|---|
| TRAIN_CORE | 2,216 | No flags trigger; reviewed KEEP with no compound risk |
| TRAIN_SOFT_KEEP | 627 | Single CDEJ/G/H/F/I rule, or reviewed KEEP_LOW_CONFIDENCE |
| HOLDOUT_REVIEW | 156 | CDEJ compound risk ≥ 2, or reviewed UNSURE |
| EXCLUDE_TRAINING | 35 | Manual EXCLUDE label, or rule_A override |

**Priority order**: EXCLUDE > HOLDOUT > SOFT_KEEP > CORE.

HOLDOUT_REVIEW is not a permanent exclusion. These basins are withheld from the
initial training run and should be revisited after post-training residual analysis.
See `docs/basin_screening_decision_memo.md` Section 10 for next steps.

---

## 5. Candidate Rule Philosophy

### 5.1 High Precision Over Broad Recall

The rules are designed to flag basins that are *likely* problematic with high
confidence, not to cast a wide net. A rule that catches 10 real problems and
5 false positives is preferable to one that catches 15 real problems and 50 false
positives — because false exclusions remove real hydrological signal from training.

Thresholds are set at the p95 or p99 of the full candidate distribution to ensure
only the most extreme cases are flagged. See `tables/candidate_rule_screening.csv`
for the exact threshold values used.

### 5.2 Reversible Risk Flags

Rules C through I are reversible soft flags. A basin that triggers rule F (large,
slow, low-flashiness) is flagged as TRAIN_SOFT_KEEP_PRELIM but is NOT excluded.
Post-training residual analysis may reveal whether that basin degrades model
performance — at which point it can be removed with data-driven justification.

### 5.3 Do Not Exclude Valid Hydrologic Regimes

Intermittent and ephemeral streams are real hydrological phenomena, especially in
arid and semi-arid regions. Rule G (mostly-zero suspicious) targets only the most
extreme zero-flow cases (zero_flow_fraction >= p95 AND Q50 <= 0.001 m3/s) and
assigns HOLDOUT_REVIEW rather than EXCLUDE. These basins should be reviewed by
region and season before exclusion.

Similarly, large basins with low flashiness (rule F) may represent important model
training contexts. They are flagged as SOFT_KEEP, not excluded.

### 5.4 Proxy Variables and Substitutions

GAGES-II does not include ARTIFPATH_PCT, ARTIFPATH_MAINSTEM_PCT, or HIRES_LENTIC_PCT.
The following proxies are used:

| Requested variable | Proxy used | Source | Limitation |
|---|---|---|---|
| ARTIFPATH_PCT | CANALS_PCT | GAGES-II HydroMod_Other | Canals only; excludes other artificial paths |
| HIRES_LENTIC_PCT | WATERNLCD06 | GAGES-II LC06_Basin | Basin-wide open water; includes natural lakes |
| -- | lka_pc_use | HydroATLAS | Lake area %; 99.84% coverage (3,029/3,034) after join-key fix (see note) |
| -- | dor_pc_pva | HydroATLAS | Degree of regulation; same 99.84% coverage after fix |

All proxy substitutions are documented in `tables/input_join_audit.csv` and the
analysis summary.

**HydroATLAS join-key correction**: HydroATLAS stores USGS station IDs without leading zeros
(e.g. `3144816` instead of `03144816`). An initial run using raw string matching yielded only
553/3,034 matches (18.2%). The fix applies `str.zfill(8)` normalization to all STAID columns
before joining, raising coverage to 3,029/3,034 (99.84%). The 5 unmatched candidates are
15-digit coordinate-based USGS site IDs (e.g. `401733105392404`) that are outside the
HydroATLAS coverage universe; their lka_pc_use and dor_pc_pva values remain NaN, and they
are not penalized by rules C or D (NaN comparisons evaluate to False, not True).

---

## 6. Reproducible Inputs and Outputs

### 6.1 Inputs (exact paths)

#### Rule analysis (pass-1 calibration)

| Input | Path | Row count at analysis time |
|---|---|---|
| Pass-1 labels (locked) | `reports/flashnh_hydrograph_review_cards_v004_main_training_candidate/manual_review_labels_pass1_locked.csv` | 73 |
| Pass-1 review template | `reports/flashnh_hydrograph_review_cards_v004_main_training_candidate/tables/human_review_template.csv` | 73 |
| WY2024 metrics + metadata | `reports/flashnh_usgs_site_metadata_v001/tables/wy2024_metrics_with_site_metadata.csv` | 3,324 total; 3,034 main_training_candidate |
| GAGES-II attributes | `C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_gageii_*.csv` | 9,008 rows each |
| HydroATLAS | `C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_hydroATLAS.csv` | 9,008 rows; 3,029/3,034 match after join-key fix |
| NLDAS-2 climate | `C:/PhD/Python/neuralhydrology/US_data/attributes/attributes_nldas2_climate.csv` | 9,008 rows |

#### Combined analysis (both passes)

| Input | Path | Row count |
|---|---|---|
| Pass-2 labels | `reports/flashnh_hydrograph_review_cards_v005_second_pass_rules/manual_review_labels_pass2.csv` | 75 |
| Pass-2 review template | `reports/flashnh_hydrograph_review_cards_v005_second_pass_rules/tables/human_review_template.csv` | 75 |
| Full candidate rule matrix | `reports/flashnh_manual_review_rule_analysis_v001/tables/full_candidate_rule_matrix.csv` | 3,034 |

### 6.2 Key Scripts

| Script | Purpose |
|---|---|
| `scripts/analyze_manual_review_filter_rules.py` | Pass-1 rule calibration; generates rule matrix and preliminary status |
| `scripts/analyze_combined_manual_review_results.py` | Combines both passes; rule performance tables; proposed policy |
| `scripts/build_final_basin_training_status.py` | Encodes accepted policy; writes final training/holdout/excluded lists |

All scripts run with no arguments. Deterministic output (no random seeds in rule computation).

### 6.3 Outputs

All outputs go to `reports/flashnh_manual_review_rule_analysis_v001/` (gitignored).

| File | Purpose |
|---|---|
| `tables/input_join_audit.csv` | Row counts, match rates, proxy notes per attribute file |
| `tables/manual_review_label_summary.csv` | Decision/behavior/artifact counts and percentages |
| `tables/candidate_rule_screening.csv` | 10 rules with thresholds, reviewed evidence, full-dataset impact |
| `tables/reviewed_basin_rule_matrix.csv` | 73 reviewed basins with all rule flags and attributes |
| `tables/full_candidate_rule_matrix.csv` | 3,034 candidates with preliminary_training_status |
| `summaries/manual_review_rule_analysis_summary.md` | Narrative summary and rule table |
| `summaries/manual_review_rule_analysis_summary.json` | Machine-readable key counts and thresholds |

---

## 7. How to Report This in a Publication

The following framing is consistent with the completed methodology:

> "We applied a multi-stage screening workflow to identify high-quality streamflow
> records for model training. After automated data-quality exclusions (data completeness
> <90%, severe negative flow, zero median flow) and USGS site-type metadata filtering
> (removing tidal stream and lake sites), we conducted manual hydrograph inspection
> across two stratified review passes totaling 148 basins. Pass 1 (73 basins) was
> enriched for suspicious extreme-flashiness and context-flagged cases. Pass 2 (75 basins)
> targeted basins flagged by compound regulation/lentic risk rules. Review labels were
> used as calibration evidence to evaluate candidate risk rules based on streamflow
> metrics and basin attributes (GAGES-II, HydroATLAS, NLDAS-2). Rules were conservative,
> using p95/p99 quantile thresholds. Regulation/lentic/artificial-flow compound risk
> (CDEJ rules active ≥ 2) was the strongest exclusion signal (exclude-or-unsure rate
> 0.45–0.69 in the reviewed sample). Zero-flow and extreme-jump flags alone were not
> treated as exclusion criteria, consistent with their low exclude-or-unsure rates in
> the reviewed sample (0.22 and 0.15 respectively). The final training set comprised
> 2,843 basins (TRAIN_CORE + TRAIN_SOFT_KEEP); 156 basins with compound regulation risk
> were withheld for secondary review; 35 basins were excluded based on manual EXCLUDE
> labels or previous manual override flags."

---

## 8. Completed Steps and Final Outputs

All steps listed in the original "Next Steps" section have been completed.

1. ✓ **Rule screening table inspected** (`tables/candidate_rule_screening.csv`):
   Rules C, D, E, J have strong calibration support (exclude-or-unsure rate 0.41–0.57 in
   reviewed sample). Rules G and H have weak support (0.22 and 0.15) and are not
   treated as holdout triggers when appearing alone.

2. ✓ **Accepted rule policy**: CDEJ compound risk (≥ 2 active) triggers HOLDOUT_REVIEW.
   Single CDEJ/G/H/F/I rule triggers TRAIN_SOFT_KEEP. Manual EXCLUDE or rule_A triggers
   EXCLUDE_TRAINING.

3. ✓ **Final basin training status script created and run**:
   `scripts/build_final_basin_training_status.py`. Output:
   `reports/flashnh_final_basin_selection_v001/`.

4. ✓ **Manual review expanded** to 148 basins across two passes. Combined analysis at
   `scripts/analyze_combined_manual_review_results.py` and
   `reports/flashnh_combined_manual_review_analysis_v001/`.

### Next action

Begin meteorological forcing preprocessing for the initial training set:
`reports/flashnh_final_basin_selection_v001/tables/training_basin_list_initial.csv`
(2,843 basins). See `docs/basin_screening_decision_memo.md` Section 10 for the
full milestone plan.

---

*This document describes methodology and final outcomes. For quantitative results, see:*
- *`reports/flashnh_manual_review_rule_analysis_v001/summaries/manual_review_rule_analysis_summary.md`*
- *`reports/flashnh_combined_manual_review_analysis_v001/summaries/combined_manual_review_analysis_summary.md`*
- *`reports/flashnh_final_basin_selection_v001/summaries/final_basin_selection_summary.md`*
