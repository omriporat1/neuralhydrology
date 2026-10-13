# RD1 P1–P24 / R1–R12 development-screening evidence stage — closure and evidence index

**Status: CLOSED as an evidence stage (2026-10-13).** This document closes the evidence stage built from P1–P24 (Bayesian) and R1–R12 (random-control). That covers Session A, which recovered and evaluated P17/P18/P19, and Session B, which built the 36-trial candidate dashboard v2. It is the tracked index of what was evaluated, which artifacts are authoritative, where they are retained, and which narrative statements in those artifacts are corrected (§7 Errata).

It is **not** a winner, promotion, or selection decision. It does not authorize any search-design, objective, P19-selection, sequence-length, cross-validation or P25+ work. All of those remain open user decisions (§9).

## 1. Scope and evidence tier

- Population: 36 trials (P1–P24 and R1–R12) × the same 400 Common-120 screening basins = **14,400 trial-basin rows**.
- Fixed conditions for every trial:
  - Seed A (`model_seed=967139`)
  - fixed fidelity `mf12x50000`
  - development-validation-2024 window
  - Common-120 fixed-support contract `stage1_v2_common120_fixed_support_contract_v001.json`, sha256 `cb4ebe86afa501ef3d5929ead5b455f8df06e7d38b58ebf4148f8545fe6851ef`
- Metrics are raw-space metrics on the official best epoch. The official objective is the median per-basin raw-space NSE over the 400 basins.
- **Evidence tier:** development screening only. Results are **single-seed**. There is no sealed-scope access and no generalization claim. Under SHARED-A5, screening-400 is not authoritative for ordering, so nothing here is a ranking or promotion basis.
- Every trial now has basin-level evidence:
  - The `qualification_status` is `qualified_basin_level_evidenced` for all 36.
  - P17/P18/P19 moved from scalar/circumstantial-only to directly evidenced via Session A (§3).

## 2. Controlled-comparison boundary

- **P1–P12 vs R1–R12 is the only controlled Bayesian-vs-random comparison.** It follows the frozen 12+12 RD1-C1 design (`docs/stage1_v2_12plus12_review_design_v001.md`). Its configuration-level facts are in `.scratch_local/rd1_supervisor_meeting_c2/SUPERVISOR_BRIEF.md` (sha256 `88585378a6041e15fb556797adfe76d846f1a4e605963fcc7d272f5cf1508d5b`).
- This closure neither re-derives nor extends that comparison.
- **P13–P24 are continuation proposals of the same Bayesian controller and have no random-control counterpart.** No P13–P24-vs-R1–R12 or P1–P24-vs-R1–R12 statement in this stage is a controlled comparison.
- Where R labels appear below (win credit, sensitivity ranks, Pareto fronts), they are population context only.

## 3. Authoritative evidence, locations, hashes

All of the following are gitignored, project-local evidence. Paths are relative to the project root.

| Artifact | Location | sha256 |
|---|---|---|
| Session A final handoff manifest v3 (authoritative for P17/P18/P19) | `.scratch_local/rd1_p17p19_recovery_eval_20261008T071622Z/11_final_handoff_manifest_v3.json` | `d3ccfe5d2dd9fff334024a17f1b5829584935571b98609b3631c80ec900119e2` |
| P17/P18/P19 recovered evaluations (job `46340136`, **authoritative**) | `…/09_recovered_evaluation_outputs_v2_q98fix/P17_recovered_evaluation.json`, `P18_…`, `P19_…`, `rd1_p17p19_recovered_evaluation_manifest.json` | `ff92d95fa7cc281917ab184874815245b1f07cd5e5cc83ed32fdd1fa500b9abe`, `fca80457c54d270dc3b19da6ac270c933b44fd37555f753581ad1376c638db87`, `33c701389ffc4080f34a6e5bf050c3eab1539bc5843ff6ae0b0c201ca54e2300`, `5c7162e5e6e433c242196236d84985857b37c6ce734ee31d308a97f6010c0004` |
| Superseded P17–P19 evaluation (job `46339847`, 0/400 Q98 — historical only, **do not cite**) | `…/09_recovered_evaluation_outputs/` | listed in v3 under the superseded outputs |
| Session A wrong-root correction addendum | `…/07_correction_addendum_wrong_root_bug.md` | `62e4f7284e6f06a202891e6d17a870ed4ccc7000f43db4eaeca0396e4b348902` |
| Retained-originals manifest | `…/10_retained_originals/retained_originals_manifest.json` | `e4b4b33073b7612be600ee9231a139249bef9addc4b9a808ebe4650cdea6331e` (not strict JSON — E19) |
| Dashboard v2 bundle manifest (61 files) | `.scratch_local/rd1_candidate_dashboard_v2_20261008T093605Z/09_manifest.json` | `230d8a465799ef0a5f30155370ee46b861dd150e428135dd907dcdeeb30ac9aa` |
| Standardized 36×400 long table (in that manifest) | `…/tables/standardized_results_long.csv` | `10564f5e004a5bdd56ca000c476a081366deba86a686bbe6a92128620ee9ca81` |

**Upstream per-basin inputs of the long table.** Each long-table row carries the hash of its source file. Each hash below was re-verified against the retained file on 2026-10-13.

| Source tag | Trials | Directory | `per_basin_metrics.csv` / `q98_diagnostics.csv` sha256 |
|---|---|---|---|
| `orig24` (job `46183269`) | P1–P12, R1–R12 | `.scratch_local/rd1_c4_e_rerun_46183269_evidence/` | `277d0998…6e18` / `efea390c…170a` |
| `cont8` (job `46331427`) | P13–P15, P20–P24 | `.scratch_local/rd1_p13p24_hydrological_review_20261007T144327Z/job_output_8trial/` | `a416a9ae…187f` / `f8bec60b…61b7` |
| `p16` (job `46331662`) | P16 | `.scratch_local/rd1_p9_p21_p22_scientific_comparison_20261007T165507Z/job_output_p16/` | `8a35ea8f…83e6` / `621a65c7…53ab` |
| `p17p19_recovered_v3_q98fix` (job `46340136`) | P17–P19 | Session A v2_q98fix directory (above) | per-trial JSON hashes above |

The dashboard builder also read configuration and W&B identity tables:

- `.scratch_local/rd1_supervisor_meeting_c2/tables/output_A_configuration_table_24row.csv`, sha256 `87a51c25…af5c`
- `.scratch_local/rd1_p13p24_scientific_review_checkpoint_20261007T131024Z/wandb_cpu_job/job_output_46328087/wandb_evidence_result.json`, sha256 `97e9d356…532b`

Full hashes are recorded in the closure-audit results file (§10).

**Retention verified 2026-10-13:**

- **Dashboard v2 bundle.** All 61 `09_manifest.json` entries match by sha256 and size locally. The only unmanifested file is the manifest itself. The retained archive `.scratch_local/rd1_candidate_dashboard_v2_20261008T093605Z (3).tar` (sha256 `5e4948b0…1e41`) contains a byte-identical `09_manifest.json`.
  - The other dashboard-v2 tars are older or empty snapshots: `(2).tar` is an older state, and the unsuffixed `.tar` is an 11 KB empty-manifest stub. Neither is authoritative.
- **Session A bundle, locally:** complete, plus `rd1_p17p19_recovery_eval_20261008T071622Z (3).tar` (sha256 `2ae23c92…a8f0`).
- **Session A on Moriah.** Under `/sci/labs/efratmorin/omripo/Flash-NH/repos/flash-nh/US_data/data_download/Disk_volume_estimation/.scratch_local/rd1_p17p19_recovery_eval_20261008T071622Z/`:
  - The four authoritative v2_q98fix JSONs hash-match.
  - All 12 retained best-epoch originals match their manifest sha256: config, model, optimizer state, and `validation_results.p`, for each of P17/P18/P19.
  - All 12 original source files at the Flash-NH repo-root `source_run_dir`s also still exist and match. These are under `…/flash-nh/.scratch_local/p17_launch_2026-09-24_20260924T134251Z/…` for P17 and `…/rd1_continuation_chain_20260925T123812Z/…` for P18/P19.
  - This check used login-node `stat` + `sha256sum` only.
- **Second copy of local-only evidence (2026-10-13T15:40Z, moriah-gw-02).**
  - Before this, several items existed only locally: the dashboard-v2 bundle, its erratum pointer, the `orig24` upstream directory, Session A's top-level documents (including the v3 manifest and the wrong-root addendum), and this closure-audit directory.
  - A tar was built containing all of these. It covers the whole Session A bundle except `source_evidence/`, which is itself packed in the bundle's `rd1_p17p19_source_evidence_20261008T095449Z.tar.gz`.
  - Tar: `.scratch_local/rd1_p1p24_stage_closure_retained_copy_20261013.tar`, 123 files, sha256 `e8bc1c585376ccf40b5a2d28372131d4d1241e0acb932d4eeddb74b77d141e72`. The per-file list is `….sha256`, sha256 `6699f2b7…dc35`.
  - The tar was copied to Moriah and extracted into `.scratch_local/rd1_p1p24_stage_closure_audit_20261013/retained_copy/`.
  - All 123 files pass `sha256sum -c`. The log is `a06_moriah_retained_copy.out`, in the same directory on Moriah.

## 4. Trial identity and headline medians (400 basins, official best epoch)

The flood-error columns are lower-is-better. "|vol. bias|" is the median **absolute** relative volume bias over the high-flow mask. "|err. at obs. peak time|" is the median **absolute** error at the observed peak's timestamp, `(sim[t_obs_peak] − obs_peak) / obs_peak`. That column measures magnitude at one fixed time; it is **not** a peak-timing error.

| Trial | configuration_id | best epoch | objective (median NSE) | median KGE | median Q98-norm. RMSE | median \|vol. bias\| | median \|err. at obs. peak time\| | source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| P1 | `sweep_v2_cfg_26daf2a8f293007dd60b` | 1 | 0.366218 | 0.3947 | 1.0661 | 0.5219 | 0.6925 | orig24 |
| P2 | `sweep_v2_cfg_dd9f7b847381f637439d` | 9 | 0.438810 | 0.4609 | 0.9971 | 0.4727 | 0.6580 | orig24 |
| P3 | `sweep_v2_cfg_00be6b4182e764f24f15` | 9 | 0.375309 | 0.3880 | 1.1099 | 0.5221 | 0.6657 | orig24 |
| P4 | `sweep_v2_cfg_5697a85648162f8ecbd8` | 5 | 0.421200 | 0.4572 | 1.0093 | 0.4463 | 0.6393 | orig24 |
| P5 | `sweep_v2_cfg_2be7714001f238085245` | 9 | 0.431368 | 0.4617 | 1.0001 | 0.4260 | 0.6416 | orig24 |
| P6 | `sweep_v2_cfg_3a5677ba0bf8c3e25425` | 10 | 0.418348 | 0.4462 | 0.9855 | 0.4504 | 0.6541 | orig24 |
| P7 | `sweep_v2_cfg_0c846d03199b14bf496b` | 8 | 0.427811 | 0.4507 | 0.9982 | 0.4200 | 0.6011 | orig24 |
| P8 | `sweep_v2_cfg_f183a7388ec063d9e223` | 9 | 0.412785 | 0.4029 | 1.0242 | 0.4980 | 0.6618 | orig24 |
| P9 | `sweep_v2_cfg_0fbf9ece824647158bc9` | 11 | 0.443063 | 0.4685 | 0.9790 | 0.4044 | 0.6004 | orig24 |
| P10 | `sweep_v2_cfg_0fb5d746c60f2872801f` | 9 | 0.433144 | 0.4568 | 0.9906 | 0.4527 | 0.6334 | orig24 |
| P11 | `sweep_v2_cfg_5853a2c9c24812e2d98e` | 4 | 0.425788 | 0.4601 | 0.9690 | 0.4452 | 0.6385 | orig24 |
| P12 | `sweep_v2_cfg_c9191df0471c3d4d6a6f` | 5 | 0.403236 | 0.4244 | 1.0300 | 0.4736 | 0.6666 | orig24 |
| P13 | `sweep_v2_cfg_8021f3b88f95d38920ae` | 7 | 0.436284 | 0.4602 | 0.9879 | 0.4427 | 0.6225 | cont8 |
| P14 | `sweep_v2_cfg_76dbe8b07495ca790dd1` | 4 | 0.442700 | 0.4787 | 0.9733 | 0.4283 | 0.6154 | cont8 |
| P15 | `sweep_v2_cfg_6f51b08ded29e01178d8` | 3 | 0.438641 | 0.4980 | 0.9757 | 0.3911 | 0.5962 | cont8 |
| P16 | `sweep_v2_cfg_15bffef18d682c6ff956` | 4 | 0.435144 | 0.4538 | 0.9839 | 0.4655 | 0.6551 | p16 |
| P17 | `sweep_v2_cfg_224202d59859c8e969d1` | 4 | 0.421067 | 0.4438 | 1.0139 | 0.4660 | 0.6480 | recovered |
| P18 | `sweep_v2_cfg_ecfc86475926fac9cf25` | 5 | 0.454178 | 0.4801 | 0.9980 | 0.4208 | 0.6108 | recovered |
| P19 | `sweep_v2_cfg_bfcf674895deb7f57f88` | 7 | 0.436464 | 0.5106 | 0.9609 | 0.3973 | 0.5950 | recovered |
| P20 | `sweep_v2_cfg_58e981d1e24e6e810152` | 10 | 0.435543 | 0.4859 | 0.9903 | 0.4221 | 0.6274 | cont8 |
| P21 | `sweep_v2_cfg_bda8ecc3930aeed26274` | 6 | 0.443769 | 0.4639 | 0.9838 | 0.4403 | 0.5894 | cont8 |
| P22 | `sweep_v2_cfg_c9c4824cc8657f86abcd` | 6 | 0.453207 | 0.4651 | 0.9964 | 0.4649 | 0.6455 | cont8 |
| P23 | `sweep_v2_cfg_cb29732b6066af175cd5` | 4 | 0.438252 | 0.4749 | 0.9467 | 0.4134 | 0.5972 | cont8 |
| P24 | `sweep_v2_cfg_fc7d136c70d7fd1a8ecd` | 9 | 0.427024 | 0.4916 | 0.9763 | 0.4036 | 0.6041 | cont8 |
| R1 | `sweep_v2_cfg_a476ab2b0f9fb7b42863` | 10 | 0.369898 | 0.4151 | 1.0677 | 0.4792 | 0.6532 | orig24 |
| R2 | `sweep_v2_cfg_5f9f8c5442fb1200d68e` | 6 | 0.432768 | 0.4455 | 1.0047 | 0.4542 | 0.6582 | orig24 |
| R3 | `sweep_v2_cfg_54476936ed984e3a4d64` | 5 | 0.371762 | 0.3894 | 1.0313 | 0.4985 | 0.6777 | orig24 |
| R4 | `sweep_v2_cfg_54d75cc3479d174bfa7e` | 5 | 0.427700 | 0.4618 | 0.9865 | 0.4283 | 0.6249 | orig24 |
| R5 | `sweep_v2_cfg_a1a077c559a271065c1b` | 3 | 0.357893 | 0.3949 | 1.0515 | 0.5012 | 0.6736 | orig24 |
| R6 | `sweep_v2_cfg_c41f1463e32650938ad4` | 4 | 0.416652 | 0.4570 | 1.0116 | 0.4499 | 0.6591 | orig24 |
| R7 | `sweep_v2_cfg_777203e76a1c04610304` | 11 | 0.368567 | 0.3940 | 1.0564 | 0.5057 | 0.6694 | orig24 |
| R8 | `sweep_v2_cfg_04ae5be16d148b3b23db` | 2 | 0.318546 | 0.3386 | 1.1149 | 0.5411 | 0.6939 | orig24 |
| R9 | `sweep_v2_cfg_395eef2f67c97d86c415` | 4 | 0.443493 | 0.4743 | 0.9946 | 0.4276 | 0.6271 | orig24 |
| R10 | `sweep_v2_cfg_f99bddcbf63152ce2ebb` | 5 | 0.418782 | 0.4252 | 1.0028 | 0.4884 | 0.6611 | orig24 |
| R11 | `sweep_v2_cfg_3ae1f024a3ddd895dd21` | 6 | 0.429866 | 0.4576 | 0.9850 | 0.4565 | 0.6177 | orig24 |
| R12 | `sweep_v2_cfg_22536659ebbdf2e05de7` | 12 | 0.413590 | 0.4522 | 1.0079 | 0.4601 | 0.6164 | orig24 |

P17/P18/P19 W&B runs are `mwxip3q4` / `x2dv6l4b` / `onbe4e1c`. Their recorded and recomputed objectives agree exactly (absolute difference 0.0), and each has 400/400 basins with 400/400 Q98 diagnostics and 0 failures. The full-precision objectives are P17 `0.42106726019345797`, P18 `0.4541781146131809`, and P19 `0.4364637928204671`. The CSV long table stores these values to ≤2.8e-14 round-trip precision.

**How the P17–P19 recovery actually happened (corrects earlier wording):**

- **What failed first.** Rounds 1/2 of the investigation searched the nested project `.scratch_local` for the original run directories. The directories are actually at the Flash-NH repo-root `.scratch_local`. This was a **search-path error in the investigation, not an evaluation-runner bug**. The originals were never lost.
- **No retraining.** Session A then evaluated the original, untouched best-epoch artifacts with a new standalone runner that directly composes qualified primitives. No `execution_provenance.json` receipt exists for these continuation-chain trials.
- **The runner's own Q98 bug.** The first run, job `46339847`, passed the trial's own observed series as `obs_m3s` and produced 0/400 Q98 diagnostics. The fixed run, job `46340136`, passes `facts.canonical_obs_m3s` and is authoritative.
- **No pre-existing per-basin values.** No per-basin values existed for these trials before this evaluation. They were computed fresh from the original `validation_results.p`.

## 5. Audited decisive findings

All numbers below were recomputed independently from the long table by `scripts/rd1_p1p24_stage_closure_recompute.py` (34/34 consistency checks pass). Win credit gives each basin's single best candidate the credit; ties share it equally. Exclusion subsets are **post-hoc sensitivities only**: the primary population is all 400 basins.

- **P18 is the scalar/NSE leader only.**
  - Its medians are worse than P9, P15, P19 and P23 on all three flood metrics.
  - Its win-credit share is the lowest of the six (P18, P9, P15, P19, P22, P23) on NSE (2.25%), KGE (2.0%) and Q98-RMSE (1.5%). It is not the lowest on |vol. bias| (2.0%; P22 has 1.5%) or |err. at obs. peak time| (3.0%; P22 has 1.25%).
  - Its win-credit rank among all 36 trials is 21/20/26/21/13 on NSE/KGE/Q98/|vb|/|pk|.
- **P18 under the exclusion sensitivities.** NSE rank 1 with all 400 basins. **Rank 2** after excluding the 28 basins where every candidate has negative NSE (top-1 becomes P2). **Rank 10** after excluding the 131 basins where any candidate has negative NSE (top-1 becomes P16).
- **P19 vs P9 win credit: P19 is below P9 on all five metrics.**

  | Metric | P19 | P9 |
  |---|---:|---:|
  | NSE | 4.0% | 8.25% |
  | KGE | 5.25% | 7.25% |
  | Q98-RMSE | 6.5% | 9.25% |
  | \|vb\| | 6.75% | 7.5% |
  | \|pk\| | 3.5% | 7.75% |

  - Same-basin joint improvement of P19 over P9 (higher NSE **and** lower flood error): 138, 132 and 111 of 400 basins for Q98, |vb| and |pk|.
  - P19 has the highest median KGE (rank 1 in all three subsets) and the lowest median NSE of the six.
  - On the three flood metrics, P19's median ranks are 2/2/2 (all 400 basins), 3/2/3 (excl-28) and 1/2/2 (excl-131).
- **Pareto points added by P13–P24.** These are 2-metric fronts on medians (NSE vs each flood metric). R1–R12 lie on no front, so the P1–P24 and all-36 fronts are identical.

  | Flood metric | Front, P1–P12 only | Front, P1–P24 | Added | Displaced |
  |---|---|---|---|---|
  | Q98 | {P9, P11} | {P9, P14, P18, P21, P22, P23} | P14, P18, P21, P22, P23 | P11 |
  | \|vb\| | {P9} | {P9, P15, P18} | P15, P18 | none |
  | \|pk\| | {P9} | {P18, P21} | P18, P21 | P9 |

  - P18 lies on every front only because it has the highest median NSE. A point with the maximum NSE cannot be dominated in a 2-metric test.
  - **Excluding P18, P13–P24 added 4 new points on the Q98 front (P14, P21, P22, P23). Across the three fronts they added 5 distinct candidates (P14, P15, P21, P22, P23).**
- **Joint-improvement maxima vs P9:** P24 164/400 on Q98, P24 140/400 on |vb|, P15 134/400 on |pk|. No candidate reaches a same-basin majority on any flood metric.
- **Shared-failure categories (36 trials):**
  - 28 basins where every candidate has negative NSE (29 with the 33 non-recovered trials). Basin `08276300` left this set because P19 is non-negative there.
  - 131 basins where at least one candidate has negative NSE (129 with 33 trials). Two basins joined: `05416900` through P18 and `05593900` through P17.
  - 297 basins where every candidate underpredicts at the observed-peak timestamp. This set is unchanged.
  - Median cross-candidate NSE standard deviation is 2.97 inside the 28-basin set and 0.069 elsewhere.
  - Pearson r(log area, median NSE) = −0.056, which is negligible.

## 6. Limitations

- Single seed, single 2024 development window, 400 screening basins, a single fidelity. There is no sealed or holdout evidence, and between-trial differences of ~0.01 in median NSE have no seed-variance estimate.
- The P17–P19 evaluations used direct composition of qualified primitives, not the receipt-bound production entry point. They were run at a HEAD that differed from the frozen payload commit `dabd2ca`. Session A records that the primitives used are unaffected; this closure did not re-audit that source-level claim.
- Win credit, Pareto fronts and exclusion sensitivities are descriptive screening diagnostics, not an approved selection rule.
- Dashboard-v2 narrative files carry the defects listed in §7; its tables and figures are numerically reproduced.

## 7. Errata to retained (checksummed) evidence — read before citing dashboard v2 or Session A narrative

The files below are covered by `09_manifest.json` or the v3 manifest. **They are left byte-unchanged on purpose.** This section is the correction of record. A sibling pointer, `.scratch_local/rd1_candidate_dashboard_v2_20261008T093605Z.ERRATUM_20261013.md`, points here. Unless a row below says otherwise, the bundle's **tables and figures are numerically confirmed**.

| # | File (dashboard v2 unless noted) | As written | Correct |
|---|---|---|---|
| E1 | `08_…brief…PROVISIONAL.md` Q1; `06_…addendum_v2.md` §7 | P18 "lowest of these six on every one of the 5 main metrics (1.5–3.0%, vs P9's 7.5–9.25%)" | P18 is lowest of the six on NSE/KGE/Q98 only (2.25/2.0/1.5%). P22 is lower on \|vb\| (1.5% vs 2.0%) and \|pk\| (1.25% vs 3.0%). P9's range is 7.25–9.25% (its KGE share is 7.25%) |
| E2 | brief Q1 | P18 "not on the 400-basin NSE top-5 under either exclusion check" | P18 is NSE **rank 2** under excl-28; outside top-5 (rank 10) only under excl-131 |
| E3 | brief Q1 | R9's flood metrics "better than P18's on all three" | R9 is better only on Q98-RMSE (0.9946 vs 0.9980); worse on \|vb\| (0.4276 vs 0.4208) and \|pk\| (0.6271 vs 0.6108) |
| E4 | brief Q1/Q2 | P19 "top-2 (usually top-1) on KGE, \|vb\|, Q98 across subsets"; "top-2 on 3 of 4 non-NSE metrics in all three subsets" | Q98 rank is 3 under excl-28; there P19 is top-2 on only 2 of 4 non-NSE metrics. Top-1 in 4 of 9 KGE/\|vb\|/Q98 cells, not "usually" |
| E5 | brief Q2 | P19 win shares "all competitive with or above P9's" | P19 is **below P9 on all five metrics** (§5) |
| E6 | brief Q2 | P19's joint-improvement counts "comparable to or above most other continuation candidates" | Among P13–P24, P19 ranks 11/12 (Q98), 4/12 (\|vb\|) and 9/12 (\|pk\|); the claim holds only for \|vb\| |
| E7 | brief Q3 | P21/P22/P23 win shares "4.5–6.0% on NSE and 3.75–6.5% on the flood metrics" | NSE 4.5–4.75%; flood 1.25–4.5% |
| E8 | brief Q6 vs Q3 | Q6 "bought 3 genuinely new non-dominated points" vs Q3's 4 | 4 on the Q98 front (P14/P21/P22/P23, excluding P18); 5 distinct candidates across all three fronts (adds P15). Q6 "P21–P24 still produced … new frontier points": P24 is on no front |
| E9 | brief, captions | "\|peak-time error\|" | Means \|error at observed peak time\| (magnitude at the observed-peak timestamp), not a timing error |
| E10 | brief Q1 table | P18's Q98 (0.998) and \|vb\| (0.421) shown in bold as if best | Neither is best among the six; the bold has no ranking meaning |
| E11 | `07_basin_difficulty_diagnostic_writeup.md` | "one basin moved each way net" | One basin left the all-negative set (29→28, `08276300` via P19); **two** joined the any-negative set (129→131) |
| E12 | `fig10_…caption.json`, `dashboard_v2.html` | r = −0.056 is "a weak-to-moderate association" | Negligible association |
| E13 | `dashboard_v2.html` banner, `fig12` caption, `06_correction_addendum_v2.md` §7, `10_change_log…` follow-up, brief header | Recovery was a "wrong-root evaluation error" / "wrong-root path bug and Q98-argument bug in the evaluation runner"; per-basin values "recorded by the original trials are unchanged… before the wrong-root bug silently produced no basin-level output" | See §4: the wrong root was in the investigation's search path; the Q98 bug was in Session A's own new runner (fixed before the authoritative run); per-basin values were computed fresh — none existed before |
| E14 | brief header; `06_…addendum_v2.md` §7; Session B decision-log draft | Objective agreement "to 1e-6" | Exact agreement (abs diff 0.0) per v3; 1e-6 was only the builder's assertion tolerance |
| E15 | Status labels | Filename/banner say PROVISIONAL; header says "full evidence, still screening-level"; change log says "non-provisional" | Tables/figures: development-screening evidence, numerically confirmed. Narrative: authoritative only as corrected here |
| E16 | `10_change_log_v1_to_v2.md`, all sections before "2026-10-08 (same day, follow-up pass)" | 33-candidate statements (29/129, "exclude-29/129", "33-candidate pool", "P18 flood-evidence gap remains open") | **Historical (pre-P17–P19-import) only**; final state is the follow-up section and §5 here |
| E17 | `06_…addendum_v2.md` header, §1, §5 | 33-candidate-era framing; §5 "P18 status … not yet resolved" | §1's P15/P24 counts are re-verified unchanged at 36 trials; §5 is superseded by §7 (and E13) |
| E18 | `tables/exclusion_sensitivity_analysis.json` keys `excl29`/`excl129`, `scripts/03_…` subset names; `tables/shared_failure_categories.json` definition "any of the 33 candidates" | Labels refer to the 33-trial counts | Values are correct for 36 trials: `excl29` excludes 28 basins (n=372), `excl129` excludes 131 basins (n=269); the definition applies to all 36 |
| E19 | Session A `10_retained_originals/retained_originals_manifest.json` | `"best_epoch": 004` | Not valid strict JSON (leading zeros); a lenient parse is needed. All hashes inside were verified on Moriah 2026-10-13 |

## 8. What remains provisional

- The recommendations and framings in the dashboard-v2 brief Q6 ("evidence favoring pausing / continuing") are agent framing, not adopted positions.
- Interpretive adjectives in that brief are not audited claims. Examples are P19 "most balanced", "robust", and "no qualitative gain".
- The audited numbers are §5 above.

## 9. Open user decisions (not started by this closure)

- Whether flood-aware or multi-metric criteria should enter screening/selection, and whether P18's scalar lead or P19's KGE/flood profile should drive any next step. **No P19 selection is made.**
- Another unchanged 12-proposal Bayesian block (P25–P36) vs a search-design/screening decision first.
- Seed replication, sequence-length experiments, or year-based cross-validation before any promotion-path work.
- Authorization of the unfixed dataclass-default bug fix in `rd1_continuation_trial_authentication.py` / `rd1_continuation_results_runner.py` (decision-log 2026-10-07).

## 10. Reproduce / re-audit

- `python scripts/rd1_p1p24_stage_closure_recompute.py` recomputes §5 and checks the long table against its upstream hashes, the v3 recovery outputs and the dashboard's own tables. It writes `.scratch_local/rd1_p1p24_stage_closure_audit_20261013/closure_recompute_results.json` and exits 1 on any failed check.
- `scripts/rd1_p17p19_recovery/` holds byte-identical tracked copies of the Session A runner and sbatch:
  - runner sha256 `2f3896ed02f545a030eba69f5a210e5c4a9d76bda8945688be1e5fb0720e4a41`
  - sbatch sha256 `59d726402739d76e94dfa95267378d3a4d38a4ed16641346c109c80884e5ae7e`
  - **The sbatch as run invokes the runner at the project root.** These copies are the provenance record. Re-running would require an explicit runner path and is not authorized by this closure.
- The dashboard-v2 builder scripts (`00`–`07`) are not promoted into `scripts/`. They are bundle-relative one-off builders and are retained by hash in `09_manifest.json`.
- Closure-audit working files are in `.scratch_local/rd1_p1p24_stage_closure_audit_20261013/`: the independent recompute drafts, results JSON, the Moriah retention-check script and output, and the retained-copy manifest.
