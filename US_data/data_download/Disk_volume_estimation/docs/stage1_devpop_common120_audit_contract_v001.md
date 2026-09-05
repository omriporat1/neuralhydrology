# Stage-1 Development-Population Common-120 Audit — Scientific-Entry and Producer Contract (v001)

**`[SHARED-A3]` — CLOSED as a documentation-only design freeze (2026-09-05).** This document freezes the scientific question, the checkpoint-selection/manifest contract, the canonical support/producer/consumer boundaries, the seven-row aggregation contract, and the pre-result interpretation rubric for the seven-configuration screening-400 vs full-2,307-basin development-population audit. It authorizes no execution, no remote access, no W&B contact, no real checkpoint/artifact materialization, and no promotion. It supersedes nothing in `docs/decision_log.md`'s 2026-09-02/2026-09-03 entries; it closes the gap those entries explicitly left open ("the audit's implementation, checkpoint-selection rule, compute plan, and acceptance threshold are deliberately not predefined... they require a new inspected planning milestone").

Builds on the already-committed `[SHARED-A1]` contract (`src/baseline/devpop_common120_audit_contract.py`) and `[SHARED-A2]` evaluator/provenance seam (`src/baseline/devpop_common120_audit_evaluator.py`), both unmodified by this document.

## 1. Primary scientific audit question

The first seven-configuration development-population audit isolates **population transfer only**. For each of the seven already-valid v2 configurations (three Bayesian: Proposal 1/2/3; four frozen IID random-control: Wave 1 rows 0-3):

- use the exact checkpoint corresponding to that configuration's screening-400 v2 `best_epoch` (as already recorded in `docs/decision_log.md`'s 2026-09-02 entry);
- hold that checkpoint fixed;
- change only the evaluation population from screening-400 to the canonical 2,307-basin development-validation population;
- evaluate on the canonical Common-120 raw-space NSE audit contract (`common120_raw_space_nse_devpop_audit_v001`, `objective_scope="devpop_audit"`).

This answers: **how stable are the screening-400 conclusions when the same seven screening-selected checkpoints are evaluated on the full 2,307-basin development population?**

It does **not** answer: which epoch full-population validation itself would select; which configuration is ultimately best; whether any configuration should be promoted; or any causal hyperparameter effect. The audit's diagnostic objective scope (`devpop_audit`) stays explicitly separate from the v2 optimizer objective (`common120_raw_space_nse_v001`, `flashnh/`-prefixed) — this separation is already structurally enforced at import time in `devpop_common120_audit_contract.py`. No winner or promotion is authorized by SHARED-A3.

## 2. Checkpoint selection versus checkpoint audit — frozen distinction

**For this population-transfer audit:** the screening-selected `best_epoch` is frozen as the audit subject. This is a comparison-rule choice, not a declaration that the screening-selected epoch is scientifically final.

**For eventual promotion:** the screening-selected `best_epoch` is **not automatically authoritative**. Before any eventual model/configuration promotion decision, checkpoint selection must be grounded in authoritative full-development-population validation (per CLAUDE.md §2/§3, unchanged by this document). A broader multi-epoch full-population checkpoint-selection/stability study therefore becomes **mandatory before promotion**, and may additionally be escalated earlier if the population-transfer audit reveals meaningful instability (see §7). These two questions must never be conflated.

## 3. Seven-entry frozen checkpoint-selection manifest

One atomic, machine-readable, hash-pinned artifact covering all seven valid v2 configurations — not seven independent manifests, because the audit itself is one seven-configuration frozen comparison set. The manifest **records** an already-made decision; it must contain no logic that recomputes `best_epoch`, and W&B must never be a required runtime authority/dependency for the production audit.

Required fields per entry, with authority:

| Field | Authority |
|---|---|
| `campaign_arm` (`bayesian` / `random_control`) | `docs/decision_log.md` 2026-09-02 entry |
| `proposal_order` / stable trial-row identity | `docs/decision_log.md` 2026-09-02 entry |
| `trial_id` | W&B run/trial identity recorded at screening time (historical, immutable) |
| `configuration_id` | Campaign layer (`sweep_v2_six_axis_campaign` identity scheme) |
| `screening_objective_id` | `OBJECTIVE_ID_V2` / `common120_raw_space_nse_v001` — recorded, never re-derived |
| `screening_objective_score` | `docs/decision_log.md` 2026-09-02 entry (already recorded per configuration) |
| `screening_best_epoch` | `docs/decision_log.md` 2026-09-02 entry (already recorded per configuration) |
| `best_epoch_source` | Durable local/Moriah campaign evidence (`execution_provenance.json` / `review_records.json` per trial) — **not yet a structured, checksummed artifact**; see §11 |
| `source_training_run_location` | Moriah run_dir path — **not populated in this document**; recovery is SHARED-A5 scope |
| `checkpoint_filename` | NH's `weight_stem(epoch)` convention (`nh_seed_evaluation.py`) |
| `checkpoint_sha256` | Computed once, at manifest-freeze time, over actual checkpoint bytes — **not computed in this document**; SHARED-A5 scope |
| `selected_for_audit_reason` / policy id | This document — record as `"screening_best_epoch_fixed_v001"` |

A single seven-entry immutable/hash-pinned manifest is the frozen structure. **This document does not populate real source-run paths or checkpoint hashes.** Those facts remain to be recovered/verified later from durable local campaign evidence (the `.scratch_local/...` completion bundles already referenced in `docs/decision_log.md`) or explicitly authorized remote (Moriah) verification — SHARED-A5 scope.

## 4. Canonical 2,307-basin Common-120 support

- `common120_support_builder.build_common120_support_for_development_population` is the frozen canonical builder entry point.
- The real canonical Common-120 development-population support artifact/contract is built **once** and reused unchanged across all seven configurations — no configuration-specific regeneration is permitted to alter scientific support silently.
- Its `checksum_sha256` and population identity (the canonical 2,307-basin membership hash) become part of the comparison provenance every audit row must carry.
- An existing matching artifact (same identity, same checksum) may be reused. Same nominal identity with a **mismatching** checksum must fail closed — never silently overwritten.
- **This document does not materialize the real artifact.** That is SHARED-A5 scope.

## 5. Production producer -> A2 consumer boundary

Frozen future vertical path:

```
frozen checkpoint-selection manifest (§3)
  -> audit-specific evaluation-run preparation (new, additive)
  -> full 2,307-basin NH validation config
  -> evaluate-only NH result (validation_results.p)
  -> real provenance receipt (build_devpop_audit_provenance_receipt, unmodified)
  -> existing SHARED-A2 consumer (evaluate_devpop_common120_audit_row, unmodified)
  -> one canonical audit row
```

The existing A1/A2 contract/evaluator remain the consumer foundation, unmodified.

**Refinement on producer reuse (supersedes this document's earlier planning-pass framing).** `pilot_full_validation.build_pilot_full_validation_bundle` must **not** be frozen as the new audit's semantic authority merely because it is currently convenient — it was built for a different purpose (post-screening promotion-readiness full-population metrics, untested against any real run, with silent per-basin area-derivation exclusion, and no Common-120 masking or A1/A2 integration). Likewise, `nh_seed_evaluation.prepare_development_population_eval_run_dir` must not be reused unchanged, because its committed marker text explicitly describes a **narrower, non-authoritative diagnostic** scope ("NOT authoritative for full-population validation, NOT usable for checkpoint or architecture selection") that would misdescribe the canonical 2,307-basin audit's actual scope if left in place.

Instead, SHARED-A4 should:

- reuse tested lower-level configuration-building machinery where appropriate — `build_pilot_bundle_with_validation_scope(...)` (the shared, population-agnostic builder both of the above already sit on top of) or another neutral shared primitive is the cleaner basis, chosen at implementation time for the smallest reuse path that preserves correct full-development-population target/lead/period semantics without inheriting stale "pilot" or otherwise-scoped semantics;
- create an audit-specific additive producer/helper for the checkpoint/scaler byte-copy-and-verify step, or safely factor out its population-agnostic primitives, rather than reusing either existing helper's marker/manifest contract as-is;
- have the producer **explicitly cross-check** target variable, lead hours, evaluation period, population identity, selected checkpoint identity, scaler/config provenance, and the resulting NH artifact's identity against the frozen manifest (§3) and canonical contract (§4) — by assertion, not by convention. No scientifically meaningful fact should have to be reconstructed ad hoc by the A2 consumer or a campaign collector.

**This document does not implement the producer.** SHARED-A4 scope.

## 6. Seven-row aggregation contract

A thin collector/comparator must require, before comparing any results:

- exactly the seven expected `(trial_id, configuration_id)` identities from the frozen manifest (§3) — no duplicates, no missing entries;
- an identical canonical Common-120 contract checksum (§4) across all seven rows;
- an identical canonical development-population identity across all seven rows;
- `canonical_completeness == True` for every row (already emitted by the existing A1 completeness gate);
- verified provenance for every row (already enforced inside `evaluate_devpop_common120_audit_row` before a row is ever returned);
- `screening_objective_score`/`screening_best_epoch` joined in from the frozen selection manifest (§3) for every row — never reconstructed from the audit row itself (the audit contract's `devpop_audit` scope deliberately does not carry the screening/optimizer score).

May be implemented in SHARED-A4 only if it remains genuinely thin (assertion-and-join over already-validated rows); otherwise it moves to SHARED-A5 without changing this document's scientific contract.

## 7. Pre-result interpretation contract

Avoiding arbitrary NSE PASS/FAIL thresholds does not license defining "small" or "large" transfer shifts only after seeing the seven full-population results — that would make the acceptance framework partly post hoc. The evidence dimensions and escalation logic are frozen **now**, before any full-population number is produced.

**Required minimum reporting, per configuration and in aggregate:**

- screening-400 median NSE and full-2,307 median NSE, every configuration;
- signed and absolute score delta, every configuration;
- screening rank and full-population rank;
- Spearman rank correlation across the seven;
- Kendall rank correlation across the seven;
- pairwise rank reversals;
- top-group membership stability, using **top 3** as the primary compact decision group;
- basin-level paired diagnostics, when aggregate results require investigation.

**Interpretation tiers:**

**A. Materially stable.** Evidence supports continued use of screening-400 as the economical working screening population when: top-3 membership is unchanged; there is no other conspicuous cross-population result that would change the scientific decision being made; score shifts and rank movements are reported transparently rather than hidden behind a binary threshold. This does **not** make screening authoritative for final promotion (§2).

**B. Investigate / uncertain.** Escalate for diagnostic investigation when, for example: top-3 membership is unchanged but internal ordering changes; near-tied configurations swap rank; one or more configurations show noticeably different screening/full behavior without changing the substantive top-group conclusion; global rank statistics and individual movements give a mixed picture. Use basin-level paired diagnostics to understand the source before making a stronger statement.

**C. Materially misleading for the next selection decision.** Treat screening-400 as insufficient for the next relevant selection decision if **top-3 membership changes** between screening and full-population evaluation, or another result clearly changes the substantive scientific conclusion. This is an escalation rule, not an automatic winner-selection rule — a top-3 membership change triggers deeper full-population analysis before relying on the screening ranking for further selection; it does not itself select a winner.

**No absolute NSE-delta threshold is defined in SHARED-A3.** Spearman/Kendall values are quantitative evidence to report and reason from, not independently hard-coded PASS/FAIL gates, unless later scientific evidence prospectively justifies such a threshold in a future closure. Historical screening-vs-full observations already on record (e.g. fixed/natural-support divergence figures in `docs/decision_log.md`) remain context, not acceptance thresholds.

## 8. Scientific figures

Compact default figure set:

1. screening-400 vs full-2,307 median NSE scatter, one point per configuration, with an identity line;
2. screening rank -> full-population rank slopegraph;
3. per-configuration signed median-NSE delta (bar chart);
4. targeted basin-level paired distribution/diagnostic plots, only for configurations landing in tier B or C above.

Figures are explanatory aids alongside the quantitative comparison (§7), not alternative objective definitions.

## 9. Failure/restart/overwrite semantics

Fail-closed principles binding for SHARED-A4/A5 implementation:

- no partial canonical 2,307-basin row is ever emitted;
- no silent basin exclusion;
- admitted non-finite simulations fail the canonical evaluation;
- checkpoint/result provenance mismatch fails before a row is accepted;
- copied checkpoint/scaler/config provenance must be verified (byte-identity check), not assumed;
- no silent overwrite of a conflicting output identity;
- a verified-complete existing result may later be eligible for reuse (idempotent re-runs should not repeat NH inference unnecessarily);
- a partial, corrupt, or mismatched output must not be resumed into or silently overwritten — it requires a fresh, distinctly-identified output;
- a changed checkpoint, population, or Common-120 contract identity requires a distinct valid production output identity.

Implementation of these principles is SHARED-A4/A5 work; this document freezes the principles only.

## 10. Revised milestone boundary

### SHARED-A4 — Local production-audit orchestration implementation

Implementation + local/synthetic verification only. Expected scope: seven-checkpoint manifest schema/loader/validator; the audit-specific evaluation-run producer/preparer; safe reuse/factoring of existing checkpoint/scaler byte-copy verification; full-development-population config construction through the cleanest existing shared primitive (§5); explicit target/lead/period/population cross-checks; connecting producer outputs to the existing provenance receipt and A2 consumer; one vertical synthetic/local integration test; optionally the thin seven-row collector (§6) if genuinely small.

A4 must **not**: contact Moriah/h2o/W&B; populate real checkpoint locations/hashes using remote evidence; materialize the real production 2,307-basin Common-120 artifact; run a real seven-config or one-config production NH audit; execute Slurm jobs.

### SHARED-A5 — Production evidence, preflight, and execution

Begins only after SHARED-A4 is reviewed/closed and remote activity is explicitly authorized. Expected scope: recover/verify the seven real source-training-run identities; verify/freeze the seven exact checkpoint files and SHA-256 hashes; populate the real seven-entry manifest; materialize and checksum the real canonical 2,307-basin Common-120 contract once; establish the exact Moriah/Slurm resource and run plan; run one production canary configuration first; verify producer -> receipt -> A2 canonical row; if clean, execute the remaining six; collect all seven rows; produce the frozen comparison statistics/figures (§7/§8); interpret using this document's prospectively frozen rubric (§7).

No Proposal 4, Wave 2, winner selection, promotion, or sealed-scope access is implied by either A4 or A5.

## 11. Open items carried forward (not decided here)

- `screening_best_epoch`'s durable machine-readable provenance chain (which local evidence bundle/field authoritatively fixes it, and the Moriah checkpoint path/hash) is not yet a structured artifact — populating it is SHARED-A5 scope, contingent on inspecting the already-referenced `.scratch_local/...` completion bundles or an explicitly authorized fresh Moriah lookup.
- Canonical target/lead/period constants currently exist independently in `devpop_common120_audit_contract.py` and in the `pilot_lead06_config`/policy-driven resolution path with no automatic cross-check between them; SHARED-A4's producer must assert their equality explicitly (§5) rather than rely on convention.
