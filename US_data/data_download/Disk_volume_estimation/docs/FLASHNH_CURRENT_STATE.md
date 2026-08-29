# Flash-NH Current State

## V2 Rehearsal Preparation A/B CLOSED; registration seam Linux/offline-qualified; real disposable registration remains separately gated (2026-08-29)

**[CLOSED — IMPLEMENTATION + LINUX/OFFLINE QUALIFICATION ONLY]** The V2 Common-120 fixed-support artifact identity closure below is unchanged. Building on it, the disposable CPU W&B rehearsal preparation milestone it named is now split into two closed increments:

- **PREPARATION A — CLOSED** (`ba1f130274f82c6b76506f5b5c99d9155b4c4c1d`, "Add v2 rehearsal sweep config serializer"): offline v2 six-axis rehearsal sweep config serialization, no W&B contact.
- **PREPARATION B — CLOSED** (`aa0a2bd93c1163a348893a20b72acfc85a737456`, "Add v2 rehearsal sweep registration seam"): the registration seam (`scripts/create_sweep_v2_six_axis_wandb_bridge_rehearsal_sweep.py`, `src/baseline/sweep_v2_six_axis_wandb_bridge_manifest.py`). It validates the one committed descriptor (`config/stage1_v2_common120_fixed_support_artifact_identity_v001.json`); prospectively assembles and validates a strict rehearsal manifest before any external call; runs the runtime/Git/interpreter contract before registration; confines all real W&B interaction to a single `wandb.sweep(...)` call (`_call_wandb_sweep`, the module's sole online boundary); on a post-registration failure (unsafe returned sweep id, or a manifest write failure after a real sweep was created) preserves the returned identity in a `partial_failure` receipt and never retries automatically; and publishes the manifest via a genuinely atomic no-clobber writer (`os.link`, never `os.replace`). The manifest's `fixed_support_contract_sha256` binds to the descriptor's *internal canonical* contract checksum, not the external serialized-file checksum — the two are deliberately distinct identities (see the Common-120 entry below).

**LINUX/OFFLINE QUALIFICATION — CLOSED (2026-08-29).** A focused, fully offline Linux qualification of Preparation B's registration seam ran on Moriah at commit `aa0a2bd…` (Slurm job `45972985`, `v2_prepb_linux_qual`, partition `glacier`, host `glacier-31`, `COMPLETED`, exit `0:0`, elapsed `00:00:41`): the exact six-file focused pytest suite (123 passed, 0 skipped/failed/errors, 29.61s); a direct Moriah-filesystem `os.link` probe proving both the same-device no-clobber-failure case (existing destination raises `FileExistsError`, both files' bytes unchanged) and the successful-independent-link case (link survives removal of the source pathname); and an offline `--preflight-only` run with a poisoned `wandb.py` first on `PYTHONPATH` (import raises immediately), exit 0, empty stderr — proving `run_preflight()` never imports or contacts W&B. The preflight used the rehearsal placeholder metric and the real six-axis identity, confirmed `stop_before_training=true`/`max_agents=1`, confirmed the intended manifest absent both before and after, and used `proposal_order=execution_generation=2147483647` strictly as an in-memory preflight sentinel — never a reservation, never written to disk or sent externally. Local evidence, independently checksum-verified: `.scratch_local/v2_rehearsal_preparation_b_linux_qual_aa0a2bd_20260829T111017Z/` (23/23 files) and a follow-up scheduler/Git corroboration bundle `.scratch_local/v2_rehearsal_preparation_b_linux_qual_followup_aa0a2bd_20260829T112702Z/` (9/9 files, including raw `sacct` accounting for job `45972985` and current Git identity/reflog/ancestry evidence). Accepted, non-blocking evidence-scope limitations (e.g., no captured pre-sync Git transcript in the original bundle; external `output_root` non-creation inferred from the inspected code path rather than a direct filesystem probe) are recorded in the follow-up bundle and do not contradict the qualification. See `docs/decision_log.md`'s 2026-08-29 entry (top) for full detail.

**Status, precisely.** PREPARATION A — CLOSED. PREPARATION B — CLOSED. LINUX/OFFLINE QUALIFICATION — CLOSED. REAL DISPOSABLE REGISTRATION — NOT PERFORMED / SEPARATELY GATED (no sweep, run, manifest, proposal, or agent has been created; the poisoned import guard proved that no W&B import or contact occurred). REAL CPU AGENT REHEARSAL — NOT PERFORMED / SEPARATELY GATED, and remains a distinct later gate after real registration evidence is inspected — registration alone does not authorize agent execution. PRODUCTION SWEEP/TRAINING — NOT AUTHORIZED. No agent, Common-120 runtime access, training, evaluation, or objective publication occurred as part of this closure.

## V2 Common-120 fixed-support artifact identity recorded; disposable CPU W&B rehearsal preparation remains separately gated (2026-08-28)

**[CLOSED — ARTIFACT IDENTITY RECORDING ONLY]** The independently qualified authoritative v2 Common-120 fixed-support contract is external and untracked at `data/fixed_support_contracts/stage1_v2_common120_fixed_support_contract_v001.json`; its compact tracked identity record is [`config/stage1_v2_common120_fixed_support_artifact_identity_v001.json`](../config/stage1_v2_common120_fixed_support_artifact_identity_v001.json). The 130,593,808-byte serialized file SHA-256 (`99fdc4f44768779661a735d25974e25b64a5d7f502ec8ffa4c5d2bab42bc34d6`) and internal canonical-contract SHA-256 (`cb4ebe86afa501ef3d5929ead5b455f8df06e7d38b58ebf4148f8545fe6851ef`) are distinct identities. It was generated once and independently qualified (`QUALIFIED`) against all 400 memberships and the full packaged MRMS+RTMA gap mask.

The approximately 98.04% figure is mean retained finite-qobs membership divided by the 8,537 already-history-valid Common-120 issue times, not a 72h-to-120h sequence-length retention ratio. No v2 disposable W&B sweep exists; no real v2 rehearsal, production sweep, training, objective publication, production proposal request, or proposal-order ledger exists. The next milestone is **V2 DISPOSABLE CPU W&B REHEARSAL PREPARATION**, separately gated; this record does not authorize W&B contact.

## Sweep-v1 attempt005 CLOSED as the countable Proposal-1 result (1/36); STARTED-stage provenance-clobbering root cause repaired; objective-recovery mechanism operationally qualified (2026-08-26)

**[CLOSED — TECHNICAL AUDIT + REPAIR, ONE SCIENTIFIC RESULT ACCEPTED]** attempt005 (Slurm job `45948021`, `execution_generation=5`, `retry_of_trial_id=attempt001`, W&B run `ardib08c`/sweep `4x3btz2s`, `proposal_order=1`) is adjudicated **VALID** after a completion audit independently recomputed all 12 epochs' raw-space NSE from the original checkpoints/validation pickles via the repository's own qualified evaluation helper — every value matches the persisted result exactly, 400/400 basins every epoch, 0 exclusions, 0 non-finite. `best_epoch=1`, `objective_score=0.391678449944578`, unchanged and un-republished. **NO CONTRADICTION FOUND — VALID stands.** attempt005 is Proposal 1's sole countable trial (1/36); attempts 1–4 remain non-countable (attempt001 INVALID; attempt002/003 operational failures; attempt004 a disposable rehearsal). The audit's own log-scan step required a correction first: the 58MB Slurm stdout is tqdm-carriage-return-heavy, and a naive line-based scan silently covers almost none of it — fixed by CR-normalization, verified against independent positive controls, then re-confirmed clean (0 errors/OOM/NaN/exceptions across the full file).

The audit also found (no scientific impact) that attempt005's terminal `execution_provenance.json` is missing `retry_history`/`executor_mode`, because `execute_prepared_trial` wrote a fresh provenance-dict literal at both its STARTED and terminal stages instead of merging onto the already-accumulated durable envelope. Root-cause repaired prospectively (does not and cannot retroactively touch attempt005's immutable original record): `enrich_layer_b_provenance` now merges onto any existing on-disk record with identity-conflict detection; `execute_prepared_trial`/`run_prepared_trial_in_production` now thread an `executor_mode` field through both writes. 20 new regression tests (`tests/test_sweep_v1_provenance_lifecycle_repair.py`) walk the real intake→prepared→STARTED→terminal lifecycle reproducing attempt005's exact shape.

Separately, the objective-recovery mechanism (`src/baseline/sweep_v1_objective_recovery.py`) — previously enforcing only 3 of 8 required rejection properties — was completed (5 new rejection checks: non-VALID, incomplete, missing source hash, non-finite objective, changed-objective idempotency; `assert_matches_pinned_identity` now also checks `wandb_run_id`) and **operationally qualified via a real, disposable, online Moriah exercise** (CPU-only `glacier` partition, no GPU/training, never `wandb.agent()`, hard-refusing any match against production sweep `4x3btz2s`/run `ardib08c`). Two real bugs were found and fixed by the first two qualification attempts — a genuine `run.sweepId` → `run.sweep.id` production fix (the local unit-test fake had baked in the same bug, so only the real online run caught it) and a harness-only stale-`wandb.Api()`-cache readback bug — before job `45950255` passed cleanly: all 8 negative cases correctly rejected, one successful publication, proven idempotent repeat (with `wandb` poisoned out of `sys.modules`), clean W&B finish, disposable-only identities throughout. Evidence independently verified locally (SHA-256, 10/10 files byte-identical to the Moriah-computed checksums).

Commits `f8fc878`/`aaa630c`/`7fe4a2b`, pushed and Moriah-synced. attempt005's original evidence, objective, checkpoints, and config are unmodified; proposal 2 remains unlaunched; the controller remains paused. See `docs/decision_log.md`'s 2026-08-26 entry (top) for full detail.

## Production exact-retry launcher migrated to manifest-driven entry point; attempt004 rehearsal durably reserved; attempt005 next permissible retry (2026-08-25)

**[CLOSED, implementation only]** `scripts/run_sweep_v1_exact_retry_moriah.sbatch` now invokes `main_from_manifest` (one positional launch-manifest argument) instead of the legacy multi-flag CLI — the same manifest-driven entry point, runtime contract (`run_full_runtime_contract`), identity resolution, W&B init helper, config preparation, and executor selector already exercised by the disposable rehearsal launcher; they differ only in Slurm resources and the manifest's own `mode` field. The legacy `main()` CLI route can no longer reach the real production sweep (`4x3btz2s`): `_execute_retry` now hard-fails before any durable-intake write or `wandb` import unless `runtime_contract_verified=True`, which only `main_from_manifest` sets. **attempt004** (Slurm job `45942761`, disposable W&B sweep `bgx4yovw`/run `6ikc3xgh`) is now durably reserved in the project-local prior-attempts record as a disposable, zero-training startup rehearsal — non-scientific, permanently reserved, not countable toward the 36-trial budget, same status class as generations 2–3. **attempt005** (`execution_generation=5`, `retry_of_trial_id=attempt001`) is the next permissible production retry, carrying the complete attempts-2–4 operational history; proposal 2 remains untouched. See `docs/decision_log.md`'s 2026-08-25 entry (top).

## Sweep-v1 trial-1 (attempt001) INVALID; monolithic-execution repair committed and qualified (2026-08-24)

**[CLOSED — TECHNICAL, NOT SCIENTIFIC]** The first real Sweep-v1 Bayesian launch (proposal_order=1, Slurm job `45935704`) hit `blocked_continuation_overshoot_conflict`: the old bounded-chunk executor's second `continue_run` call found epoch 2 already trained flat (Sweep-v1's generated config trains the full 12-epoch budget in one `start_run`). **attempt001 is adjudicated INVALID — not a valid scientific trial, no proposal consumed, no objective.** Repair: new `pilot_orchestration.execute_prepared_pilot_run_monolithic` trains once through `target_epoch`, requires all 12 checkpoints, then screens every epoch post hoc via the existing raw-space screening path; wired into `sweep_v1_execution.run_prepared_trial_in_production` in place of the bounded-chunk executor. Also fixed: `review_records.json.operations.slurm_job_id` (previously always `null`) now populated from the live `SLURM_JOB_ID`. Committed `2d750db`, pushed, Moriah canonical clone fast-forwarded. Non-authoritatively qualified against attempt001's own real checkpoints (copied to a separate untracked path, never mutating attempt001): `execution_status=VALID`, all 12 epochs screened PASS, exactly 50,000 optimizer updates/epoch (cap-compliant), 400/400 basins every epoch, `best_epoch=1`/`objective_score=0.391678449944578` — never published to W&B, not counted toward the 36-trial budget. attempt001's real evidence verified byte-unchanged after qualification. Exact `attempt002` retry contract (same proposal/configuration/hyperparameters/seed, fresh `execution_generation=2`, `retry_of_trial_id` set, no proposal 2 consumed, new W&B run under the same sweep rather than resuming attempt001's crashed run) is specified but **not launched**. See `docs/decision_log.md`'s 2026-08-24 entry (top).

## Sweep-v1 launch-command seam repaired — no real trial consumed (2026-08-24)

**[CLOSED, implementation only]** A launch-readiness inspection found the production W&B sweep config had no explicit `command`; W&B's default would have appended the five swept hyperparameters as CLI flags the bridge's argparse does not accept, failing before `wandb.init()`/proposal intake. `build_production_sweep_config` now declares `"command": ["${interpreter}", "${program}"]` (no `${args}`, no `${env}`, no hardcoded paths). Because W&B itself now constructs the bridge's argv, the four operational inputs (package root, screening basin ids, output root, proposal order) are supplied via `FLASHNH_SWEEP_V1_*` environment variables exported by `scripts/run_sweep_v1_wandb_agent_moriah.sbatch` and resolved by the bridge with strict CLI/environment precedence (agreement/one-supplied accepted; contradiction/absence hard-fails before proposal intake). The sweep-config builder now refuses to silently overwrite `--output` without `--force`. New `tests/test_sweep_v1_launch_command_contract.py` (20 tests) includes a real OS-level subprocess test of the exact W&B-constructed argv. Full focused Sweep-v1 suite: 97 passed, 14 skipped (no local torch), 0 failed. No sweep/run/proposal created, no Slurm submitted, no training run, no production trial consumed. See `docs/decision_log.md`'s 2026-08-24 entry. Next operational milestone remains one serialized real Bayesian proposal/trial (proposal order 1).

## Sweep-v1 local production integration CLOSED — independently approved for commit (2026-08-23)

**[CLOSED]** Building on the prepared-execution consumer contract closure
immediately below, Sweep-v1's local production integration layer is now
CLOSED and independently reviewed as APPROVE PRODUCTION INTEGRATION FOR
COMMIT (commit `a3ae86b91569e27b6e183666675c06f0e7dc89d4`). Production
execution consumes `PreparedPilotExecutionResult` directly, with
`actual_optimizer_updates_by_epoch` as the authoritative update evidence;
VALID/INVALID scientific interpretation, the committed
`derive_trajectory_diagnostics` objective, and the 50,000-update
`max_updates_per_epoch` cap semantics stay authoritative in Flash-NH.
Bayesian and frozen random-control trials share one prepare/execute/validity
path and one-allocation/one-agent/`count=1` production launch shape. W&B is
proposal/telemetry only — never validity or objective authority. Proposal-
intake Layer-B provenance is written durably before any preparation/config
failure point, so an exact retry recovers identical scientific proposal/
config identity while only its attempt identity changes.

Config generation (`write_generated_config` /
`write_prepared_proposal(..., allow_layer_b_provenance=True)`) no longer
uses the earlier unsafe `force=True` escape hatch: protected generated
targets (`train_basins.txt`, `validation_basins.txt`, `test_basins.txt`,
`config.yaml`, `generation_manifest.json`, the holdout marker file) can
never be allowlisted as pre-existing, an allowlisted pre-existing name must
be a regular file, and same-trial `execution_provenance.json` coexistence
requires an exact present `trial_id` match (a missing/null/mismatched
`trial_id` hard-fails before any write). The final focused safety gate
passed 220 tests, 0 skipped, including torch-capable golden VALID/INVALID
bridge paths. No real Sweep-v1 production trial has been consumed yet; the
next operational milestone is one serialized real Bayesian proposal/trial.
See `docs/decision_log.md`'s 2026-08-23 closure entry.

## Prepared-execution consumer result contract CLOSED — Phase-B Sweep-v1 production integration unblocked (2026-08-23)

**[CLOSED]** The gap identified by the entry immediately below is resolved.
Prepared-execution mechanics remain CLOSED (unchanged). The prepared
executor's consumer-facing result/evidence contract is now also CLOSED:
`execute_prepared_pilot_run` (`src/baseline/pilot_orchestration.py`) returns
a typed `PreparedPilotExecutionResult` — a generic, campaign-agnostic
factual execution receipt exposing physical checkpoint inventory (via
`discover_physical_checkpoints`), the complete epoch-ordered screening
history (`screening_events`), and stopping/state facts
(`stopped`/`stop_reason`/`early_stopping_state`/`blocked`/`blocked_reason`).
Actual optimizer-update evidence remains available through the existing
authoritative `actual_optimizer_updates_by_epoch` helper rather than being
folded into the eager receipt. As part of this closure, full screening
history is now correctly reconstructed across resumed
`execute_prepared_pilot_run`/`run_pilot` calls (previously a resumed call's
evidence bundle silently carried only that invocation's newly-processed
screening epochs, not the run's full history). An independent review
(Interface / Consumer Contract Gate, `docs/agent_handoff_rules.md` §5)
verified field authority, resume-history correctness, `run_pilot` backward
compatibility, and a vertical consumer-contract test proving a generic
consumer can establish checkpoint/optimizer-update/NH-evaluation/screening
coverage and the raw-space NSE trajectory from the receipt alone, with no
filesystem archaeology or re-derived metrics; all tests passed
(commit `63c31a983b2a494e3078ad18a5e97c3cf3b876ee`). No scientific policy
changed. Sweep-v1 production integration may now resume against this
closed contract; see `docs/decision_log.md`'s 2026-08-23 closure entry.

## Phase-B Sweep-v1 production integration PAUSED before real launch (2026-08-23; superseded by the closure above)

Prepared-execution mechanics (the generic prepared executor and its
qualification evidence) are committed and reusable. Sweep-v1 production
integration — wiring that executor as the W&B Bayesian/random-search
consumer — is **paused before any real launch**: the generic prepared
executor does not yet expose the complete structured authoritative execution
receipt that Sweep-v1 needs to determine trial validity (see the 2026-08-23
workflow entry in `docs/decision_log.md` and the Interface / Consumer
Contract Gate in `docs/agent_handoff_rules.md` §5). No Bayesian or random
scientific trial has been consumed because of this gap. The next technical
step is to define, review, and implement that generic execution result
contract before resuming the W&B/Sweep-v1 bridge; the contract itself is not
specified by this entry. The frozen Sweep-v1 launch design below is
unaffected.

## Phase-B Sweep-v1 launch contract frozen (2026-08-22; not launched)

**[DECIDED]** The completed five-candidate epoch-budget calibration freezes
Sweep-v1 medium fidelity at 12 epochs, 50,000 updates/epoch, Seed A,
every-epoch authoritative raw-space screening, and no performance stopping.
The objective is best eligible median per-basin raw-space NSE through epoch
12. The cohort directly supported epoch 10 for its tested configurations;
12 is the deliberate precautionary margin for untested joint configurations.
Sweep v1 has 36 valid Bayesian and 12 frozen IID-random trials over LR
log-uniform `1e-4`--`1e-3`, H `{64,128,256}`, both dropouts uniform
`0.0`--`0.4`, and batch `{128,256,512}`. The detailed frozen contract,
boundary reviews, visualization requirements, online-W&B qualification gate,
and deferred items are canonical in
`docs/stage1_phase_b_sweep_v1_launch_contract.md`. No W&B implementation,
random manifest, or Sweep-v1 launch has occurred.

## Historical Phase-B Track-A epoch-budget calibration design (2026-08-21; superseded by completed closure)

**[HISTORICAL — CALIBRATION DESIGN]** The five-candidate Seed-A calibration was designed to determine whether the common Sweep-v1 epoch budget should be 8, 10, 12, or 14. It used one logical continuous trajectory through epoch 14 per candidate, at 50,000 updates/epoch, checkpointed every epoch, with performance stopping disabled and authoritative raw-space screening eligible at every epoch 1--14. NH's existing epoch-specific evaluation and Flash-NH's existing raw-space path could operate post-training; continuation was recovery-only. Cohort: C1 `3e-4/H128/B256`; C2 `1e-4/H128/B256`; C3 `1e-3/H128/B256`; C4 `3e-4/H64/B256`; C5 `3e-4/H256/B128` (joint convergence-stress corner). Shared PT/seq72/[128,32]-tanh/0.10 embedding-dropout/0.25 output-dropout/Adam/Seed-A/lead6 contract remained frozen. Its completed closure and current Sweep-v1 decision are recorded above and in `docs/stage1_phase_b_sweep_v1_launch_contract.md`.

## Phase-B Task-A design review and first `output_dropout` / `batch_size` plumbing increment (2026-08-20)

**Sweep-v1 decisions.** The five search axes are `learning_rate`,
`hidden_size`, `embedding_dropout`, `output_dropout`, and `batch_size`.
Adam is fixed for Sweep v1. `initial_forget_bias`, weight decay,
learning-rate schedules, and optimizer search are excluded from this first
joint-search scope only; they are not declared permanently irrelevant.

**Medium-fidelity policy.** Sweep-v1 candidates use the same
`max_updates_per_epoch=50,000` across batch sizes, deliberately holding
optimizer-update opportunity constant rather than sample exposure. Authoritative
raw-space screening occurs every epoch, and there is no performance-based
scientific early stopping: each candidate receives the complete predefined
budget and is scored from its best observed eligible screening checkpoint.

**[DECIDED FOR SWEEP V1]** `batch_size={128,256,512}` is operationally
qualified under the reviewed H256/L4 eight-update envelope; no batch-size
winner was selected. **Provisional and open items.** `output_dropout` has a continuous-uniform
working range of `0.0`--`0.4` (not previously characterized one-dimensionally).
The preferred `batch_size` set is `{128, 256, 512}`, pending technical and
operational qualification. The epoch budget, exact Bayesian/random trial
counts, and Bayesian concurrency remain open. This implementation increment
adds `output_dropout`/`batch_size` configuration, identity, and
provenance plumbing; it creates no HPO controller, W&B sweep, or training run.

Last updated: 2026-08-20 (Phase-B Task-A design review and first local
`output_dropout` / `batch_size` plumbing increment; no HPO/sweep/training
launched). Historical 2026-08-19 transition context follows.

## Historical transition context (2026-08-19)

Stage-1 Evaluation Framework v1 + Phase-B
Bayesian HPO Design — documentation-only transition handoff following
Dynamic-Input-Family-A's closure. Records the accepted scientific
motivation, data roles, HPO objective, Bayesian-vs-random-control
methodology, hyperparameter-search framing, fidelity open question,
Evaluation Framework v1 architecture, evaluation hierarchy, benchmark
plan, seed strategy, and W&B/Slurm architecture requirement for the next
phase, all as **decisions already made about design**, not as HPO/sweep/
event-separator implementation or new training/evaluation. Canonical
document: `docs/stage1_phase_b_hpo_evaluation_plan.md`. See the section
immediately below for detail; see further below for the
Dynamic-Input-Family-A closure this transition follows, and the earlier
Dynamic-Input-Family-A design freeze, Sequence-Length-A closure,
implementation task, design freeze, Embedding-Dropout-A closure,
implementation task, design freeze, hidden-size closure, hidden-size
design freeze, W&B launch-contract qualification, LR-A closure, 50k
embedding-shape closure, 25k neighborhood screening, `max_updates_per_epoch`
calibration, prior W&B qualification, `emb128x64_seedA` hydrograph-atlas
evaluation, and post-`emb128x64_seedA` roadmap entries it builds on.)

## Stage 1 — Evaluation Framework v1 + Phase-B Bayesian HPO Design: documentation-only transition handoff (2026-08-19)

Documentation-only transition task recording scientific decisions made about the next phase after Dynamic-Input-Family-A's closure, so they are captured before new ChatGPT/Claude sessions and multiple parallel workstreams begin. Full text: `docs/stage1_phase_b_hpo_evaluation_plan.md` (new canonical design/handoff document); decision text: `docs/decision_log.md`'s 2026-08-19 entry (topmost).

**Two parallel tracks adopted.** Track A (Phase-B joint multidimensional HPO, W&B Bayesian search + seeded random-search control, initial objective = median per-basin raw-space NSE on the frozen ~400-basin 2024 development-validation screening subset) and Track B (Evaluation Framework v1: exact-hour categorical/operational metrics including conditional POD, plus a deterministic variable-duration observed-only hydrologic event separator and event diagnostics) proceed in parallel. HPO does not wait for Track B to complete; Track B does not replace the Sweep-v1 HPO objective.

**Data roles, sealed sets unchanged.** ~2,307 development-training basins (`2020-10-14`–`2023-12-31`); the ~400-basin screening subset is a subset of the development population, evaluated on `2024-01-01`–`2024-12-31` — repeated HPO querying makes this population/year part of tuning, not an independent test. 2025 temporal test, non-CA spatial holdout, and California remain sealed and unaccessed by this document, per existing `docs/stage1_scientific_baseline_design.md` §8/§8b/§8c policy (unchanged).

**Not decided by this entry, explicitly open.** Exact Sweep-v1 search-space dimensions/ranges (batch size, forget-gate bias, weight decay, LR schedule, optimizer-fixed-vs-searched all under review); medium-fidelity Phase-B training/evaluation protocol (update cap, epoch budget, cadence — `50k`×`12` is an example under discussion, not frozen); exact W&B sweep-agent/Slurm architecture, Bayesian concurrency, and trial budgets; exact Seed-B finalist count; canonical high-flow threshold(s) and event-separator algorithm. Full open-question register in the new document §15.

**Not done by this entry.** No HPO launched, no W&B sweep implemented, no event separator implemented, no training launched, no new scientific evaluation run, Dynamic-Input-Family-A not reopened, no sealed-set access. `docs/stage1_validation_optimization_foundation.md` (roadmap note only) and `docs/stage1_wandb_user_guide.md` (planned-Phase-B-sweep note, clearly labeled future/unimplemented) received narrow consistency updates; no other source/test/config files touched.

**Next.** Two separate read-only/design-first tasks, not started by this entry: Task A (Phase-B Bayesian HPO Launch Design Review) and Task B (Evaluation Framework v1 Scientific Design). See `docs/stage1_phase_b_hpo_evaluation_plan.md` §16.

## Stage 1 — Dynamic-Input-Family-A CLOSED: `PT` (precipitation + temperature) adopted as the provisional Stage-1 working family; PTM/PTMW not promoted; no H256 rescue warranted (2026-08-16)

Final scientific closure of the Dynamic-Input-Family-A campaign, following the design freeze below, the four-candidate base campaign (whole-record raw-space evaluation across all 6 epochs), a true multi-candidate hydrograph-overlay review (8 frozen basin/event panels, Obs + P + PT + PTM + PTMW on shared axes), and a dedicated population-level high-flow/event audit (400-basin conditional Q90/Q95 analysis plus a deterministic 1,200-event peak/volume/shape/timing audit). This entry closes the **predictor-family decision only** — it does not launch Phase B, does not implement Stage-1 Evaluation Framework v1, and does not launch any new training. Full decision text: `docs/decision_log.md`'s 2026-08-16 CLOSED entry (topmost); technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.22; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-16 closing section. Campaign implementation/training commit `a3bf51266859a8706b40cc9e862acab793ce15c7` — the scientific campaign state against which the closure figure/evidence pack was generated. The scientific closure decision and reusable evaluation tooling are recorded in a separate closure commit on top of this campaign commit.

**Whole-record result (400-basin dev-validation screening population, all 6 epochs).** PT beats P in 63-71% of matched basins at every epoch (median NSE gain ≈0.03-0.06 per epoch); this is the one robust, repeated, basin-general improvement found anywhere in the campaign. PT also achieves the campaign's single strongest observed whole-record skill (epoch 3, median NSE ≈0.3726). PTM shows no reproducible incremental benefit over PT (near-zero median diff, basin-improvement fraction oscillating around the 0.5 coin-flip line across epochs). PTMW is a near-tie with PT on whole-record skill (median diff within ±0.02, fraction PTMW-better ≈0.43-0.57 with multiple sign flips across epochs) — "no reproducible whole-population PTMW advantage over PT" at the whole-record level.

**True hydrograph-overlay review.** The frozen 8-basin panel (Obs + all 4 candidates, shared axes, same event window) showed 3 of the 8 basins (`07261000`, `08072300`, `14301500`) with apparently meaningful PTM/PTMW improvement during specific high-flow events — illustrative only, not population-level evidence, and this observation motivated the population-level event audit below.

**High-flow/event audit (400/400 basins, 1,200 deterministically selected Q95 events, top-3/basin, 72h peak separation, 24h-before+48h-after window, event-weighted and basin-balanced views).** Conditional analysis (flow ≥ basin Q95): PT vs P is clearly positive (58-64% of basins improve on RMSE/KGE/|PBIAS|); PTMW vs PT shows a small-but-checkpoint-robust edge (52-60% of basins) — "real but modest." Event-level analysis: PT vs P remains positive on peak magnitude, event volume, and event shape (55-61%); PTMW vs PT is essentially a near-tie on peak magnitude and event volume (~50-51%) and only a small positive tendency on event shape — the conditional PTMW edge does **not** translate into a broad event-level peak/volume advantage. Peak timing is tie-dominated for both comparisons (~44-50% ties) and is not discriminative. Severity stratification (high `[Q95,Q99)` vs extreme `≥Q99`, note the event-selection protocol itself skews toward severe events): no detectable increase in PTMW benefit was observed across the severity strata represented by the selected event population — this is **not** a claim that severity dependence has been ruled out generally, only that none was detected within this sample. Per-basin cross-check against the 8 frozen overlay basins confirmed the 3 flagged basins (`07261000`, `08072300`, `14301500`) sit in the favorable tail of the 400-basin population, while 2 other frozen basins (`06894200`, `08061540`) show the opposite pattern — a small conditional improvement that reverses into an event-level regression — validating that the overlays are interpretation/sanity evidence, not model-selection votes.

**Final decision (binding, provisional).** `PT` — exactly `mrms_qpe_1h_mm` + `rtma_2t_K` — is adopted as the **provisional Stage-1 Dynamic-Input-Family-A working family**. It is explicitly **not** called "the final optimal dynamic-input family," "globally optimal," "permanently superior to PTMW," or "proof that humidity/wind do not matter." Rationale: (1) PT is the one robust, repeated, basin-general improvement over P; (2) PT achieves the strongest observed whole-record skill; (3) PTM shows no reproducible incremental benefit; (4) PTMW is near-tied with PT on general whole-record skill; (5) PTMW's modest reproducible conditional high-flow advantage does not translate into a broad event-level peak-magnitude or event-volume advantage; (6) the added moisture/wind complexity has not earned promotion over PT at this Stage-1 fidelity; (7) no H256 rescue is warranted by the evidence; (8) PTMW remains documented as the nearest broader credible alternative, not dismissed as useless. PT epoch 3 (median NSE ≈0.3726, 75,000 cumulative optimizer updates) is identified as the **best observed PT checkpoint in this specific Phase-A campaign** — this is not converted into a universal training-budget rule; Dynamic-Input-Family-A closes the predictor-family decision, not the future training-duration decision.

**Reusable evaluation capabilities retained (committed by this entry).** `render_multi_candidate_basin_panel()` (`src/baseline/hydrograph_rendering.py`) — general-purpose true N-candidate overlay renderer (arbitrary candidate set, shared axes, observed-series consistency checks), not Dynamic-Input-specific. `select_high_flow_events()` (`src/baseline/hydrograph_atlas_events.py`) — deterministic, observed-only, candidate-independent high-flow event selector (explicit threshold/separation/window semantics); retained alongside the pre-existing `select_atlas_events()`, which serves a distinct magnitude-stratum purpose. `src/baseline/high_flow_event_metrics.py` (new) — `basin_high_flow_threshold()`, `high_flow_conditional_metrics()`, `event_metrics()`, reusing `raw_space_metrics()` rather than reimplementing metric math. All three, plus their test suites, judged general enough for future HPO-finalist/sequence/architecture/lead-time evaluation work, not campaign scratch.

**Not done by this entry.** No new training launched, no H256 rescue run, no Phase B started, no Stage-1 Evaluation Framework v1 implemented, no sealed temporal-test/spatial-holdout/California data accessed. Campaign-specific drivers (`dynfam_event_audit_runner.py`, `dynfam_event_audit.sbatch`) remain untracked scratch, not committed. Generated figures/reports/evidence remain project-local and gitignored under `.scratch_local/moriah_evidence/dynamic_input_family_a_closure/`.

## Stage 1 — Dynamic-Input-Family-A design frozen, not launched: four-family P/PT/PTM/PTMW dynamic-input hierarchy at the `seq_length=72` anchor, gap channels removed from model inputs, dewpoint deliberately omitted, U/V wind kept paired (2026-08-16)

Documentation-only design-freeze task, immediately followed in the same session by the campaign's minimum implementation (`PilotRunSpec.dynamic_inputs` override machinery, config-gen/manifest/identity threading, continuation-safety guard, and the `dynamic_input_family_seedA_25k_v001` campaign closure script) — see the implementation entry that follows this one once that work is separately reported closed. Full decision text: `docs/decision_log.md`'s 2026-08-16 entry (topmost); technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.21; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-16 section.

**Evidence re-confirmed at full 2,307-basin development-training/19,528-basin-window development-validation scale.** Gap-flag audit (`gap_flag_channel_leakage_audit_20260815.json`): 0 MRMS/RTMA flag-positive admissible windows anywhere; package reconciliation clean (138 canonical gap timestamps, symmetric difference 0). Physical-variable audit (`dynamic_input_family_audit_20260815.json`): 6 physical `v001-core` variables, no sentinel/clipping/scaling pathology; dewpoint-vs-specific-humidity Pearson ≈0.9512 globally (Spearman ≈0.9947), consistently strong across seasons/temperature regimes/basins; both directly sourced RTMA fields.

**Gap-channel decision.** `_gap` channels remain in the certified package for QC/provenance (package variables) but leave the NH model `dynamic_inputs` vector for this campaign (model predictor variables) — they are constant-zero for every admissible development input window under current hard exclusion policy. The package itself is not changed or version-bumped for this modeling-input decision.

**Moisture decision (cautious wording, binding).** `rtma_2sh_kgkg` (specific humidity) is the primary single moisture representation. This is a Phase-A structural-simplification choice — strong empirical redundancy with dewpoint, simplicity, avoiding two Kelvin-valued thermal/moisture channels in the smallest family — **not** evidence that dewpoint is inherently inferior, and **not** justified by the historical (fixed) dewpoint lookup-key bug. Both-moisture use remains a possible later ablation.

**Wind decision.** U/V (`rtma_10u_ms`/`rtma_10v_ms`) physically plausible, non-degenerate; always travel together (no U-only/V-only); a separate structural family step, not automatic inclusion.

**Frozen family matrix (the four Dynamic-Input-Family-A candidates — 5 physical channels total, not 6; dewpoint omitted from the hierarchy; both gap flags package-only for this experiment).** P: `mrms_qpe_1h_mm`. PT: + `rtma_2t_K`. PTM: + `rtma_2sh_kgkg`. PTMW: + `rtma_10u_ms`, `rtma_10v_ms`.

**Common anchor (unchanged from the Sequence-Length-A-closed contract).** Seed A (967139), `[128,32]` learned-FC embedding (tanh), `hidden_size=128`, `learning_rate=3e-4`, `output_dropout=0.25`, `embedding_dropout=0.10` (profile default, unset), `seq_length=72`, lead 6h, target `qobs_mm_per_h_lead06`, `max_updates_per_epoch=25000`, six epochs, one uninterrupted segment, the fixed development-training population, the fixed ~400-basin development-validation screening population. Every candidate varies only `dynamic_inputs`.

**Rescue policy (rule only, not exercised by this design freeze).** At most one standardized H256 capacity probe if a non-reference family is weak/ambiguous; `P` is reference and is never "rescued"; no pre-created/trained H256 variants; not a second search dimension.

**Deferred.** Dewpoint/both-moisture ablation, `v001-fullmet` (pressure/cloud/visibility/gust/ceiling), longer `seq_length` testing, Phase B, sealed-set evaluation.

**Not done by this entry.** No candidate launched, no Slurm job submitted, no real NeuralHydrology training or checkpoint evaluation, no W&B run, no hydrograph panel rendered, no package rebuild, nothing committed.

## Stage 1 — Sequence-Length-A closed: `seq_length=72` adopted as provisional working anchor, `seq_length=48` nearest alternative, hydrograph sanity check clean, dynamic-input family characterization next (2026-08-15)

Documentation-only closure task recording the completed campaign (four real Moriah training runs, infrastructure commit `4646a55`, training jobs `45861222`-`45861225`) and the separately-executed, human-reviewed fixed 8-basin hydrograph sanity check plus a supplemental single-basin diagnostic. No new training, evaluation, or rendering performed by this task. Full decision text: `docs/decision_log.md`'s 2026-08-15 entry (topmost); technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.20; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-15 closing section.

**Result.** Four candidates (`seq_length` = 12/24/48/72), all other settings frozen at the Embedding-Dropout-A-closed contract (Seed A, `[128,32]` embedding, `hidden_size=128`, `learning_rate=3e-4`, `output_dropout=0.25`, 25k-update cap, six epochs). Natural-support raw-space median NSE, a new common-support-corrected evaluation, true per-basin paired comparisons, and late-window behavior all agree on the same ordering at every evaluated epoch: `seq72 > seq48 > seq24 > seq12`. The common-support correction did not materially change the ranking.

**Hydrograph sanity check.** Frozen 8-basin panel rendered for `seq24`/epoch5, `seq48`/epoch6, `seq72`/epoch5, with the displayed antecedent window widened to 72h (presentational only; frozen event/peak identity never re-selected) so all three candidates' longest tested context is visible together. Consistent with the quantitative ranking; no repeated `seq72`-specific pathology. A supplemental diagnostic confirmed basin `06131200`'s severe extreme-event failure is shared identically across all three rendered sequence lengths — a near-zero-flow basin/model pathology, not a `seq72`-specific defect.

**Decision.** `seq_length=72` becomes the provisional Stage-1 working anchor within the tested 12-72h range — the strongest-performing tested value, not proven to be the final optimum. `seq_length=48` remains the nearest credible alternative. Performance had not clearly saturated by 72h; whether longer context would help further is an open question, deferred to later work. No longer-lookback campaign launched by this entry.

**Comparative-hydrograph convention (new).** For small candidate sets: same frozen event/time window and observed hydrograph, shared axes/scales, all candidates overlaid on one panel; event selection independent of candidate performance and never re-run once frozen; only the *displayed* antecedent window may widen, and only for experiments that specifically vary historical context. Not a mandate that every future panel use a 72h window.

**Revised roadmap.** (1) Reusable Phase-A/HPO campaign infrastructure consolidation — unaffected, still pending. (2) Sequence-Length-A — closed by this entry. (3) Dynamic-input family characterization — next milestone, at the new `seq_length=72` anchor. (4) Phase B joint HPO — still deferred.

**Not done by this entry.** No Moriah/h2o access beyond sync/verification of a previously identified temporary workaround, no new Slurm submission, no evaluation of any sequence length beyond 72h, no dynamic-input-family characterization started, no Phase B started, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no final Stage 1 sequence-length selection beyond the provisional working anchor.

## Stage 1 — Embedding-Dropout-A closed: weak sensitivity over `0.00`-`0.40`, `drop10` retained as provisional anchor, hydrograph sanity check clean, revised Phase-A/Phase-B roadmap adopted (2026-08-13)

Documentation-only closure task recording the completed L.17/L.18 campaign (five real Moriah training runs — `drop10` job 45789423, `drop00` job 45790661, `drop05` job 45790662, `drop20` job 45790663, `drop40` job 45790664 — plus retrospective diagnostic-evaluation jobs 45790996-45791000 and optimizer/update verification job 45791007) and the separately-executed, human-reviewed fixed 8-basin hydrograph sanity check (job 45791211). No training, evaluation, rendering, Moriah/h2o compute, or new analysis performed by this task. Full decision text: `docs/decision_log.md`'s 2026-08-13 entry (topmost); technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.19; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-13 closing section.

**Result.** Embedding dropout is weakly sensitive over `{0.00,0.05,0.10,0.20,0.40}` at this Seed-A/25k-cap/6-epoch fidelity; no candidate robustly dominates across epoch-6 median NSE, late-window behavior, paired-basin differences, or the hydrograph review; ranking is cadence-sensitive (`drop00` leads epoch 6, `drop10` has the strongest late-window summary and best single checkpoint, `drop20` is among the most stable late-window candidates); `drop40` shows no validation-performance cliff. Fresh `drop10` and the historical H=128/dropout=0.10 comparator are exactly/deterministically reproducible. `embedding_dropout=0.10` remains the provisional working anchor — not proven optimal, retained because it sits safely inside the broad viable region, has a strong late-window result, and exactly reproduces the historical run. Final selection deferred to Phase B; the tested range must not be aggressively narrowed.

**Hydrograph sanity check.** Frozen 8-basin panel rendered for `drop00`/epoch6, `drop10`/epoch5, `drop20`/epoch6; broad similarity across most basins, basin/event-specific divergences at 3 of 8 basins each implicating a different candidate, no repeated candidate-specific pathology, no contradiction of the quantitative near-tie.

**Revised roadmap (supersedes older wording that permanently excluded sequence length from calibration — see the forward-pointing notes added at this document's 2026-08-05 entry, `docs/decision_log.md`'s 2026-08-05 entry, and `docs/stage1_validation_optimization_foundation.md` Part L.1/L.10; all preserved as historical).** (1) Reusable Phase-A/HPO campaign infrastructure consolidation, including a new durable artifact/evidence self-identification requirement discovered while reviewing this closure's hydrograph evidence (generic filenames/titles become ambiguous once separated from their parent directory — not a defect in, and no regeneration required of, existing evidence). (2) Sequence-Length-A: `seq_length={12,24,48,72}` at the best-supported anchor, now a bounded/calibratable structural parameter. (3) Dynamic-input family characterization. (4) Phase B joint HPO (LR × hidden size × embedding dropout × output dropout).

**Not done by this entry.** No Moriah/h2o access, no Slurm submission, no training or evaluation, no rendering, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout/California data accessed, no final Stage 1 embedding-dropout selection.

## Stage 1 — Embedding-Dropout-A implementation complete, preparation-only validated, ready for Moriah launch review (2026-08-11)

Implementation and local/preparation-only validation task, under unchanged commit `eea9f4c09bbfdb92b757ec4165b0bb61a7b466ba` (branch `master`), following the design freeze immediately below. Full decision text: `docs/decision_log.md`'s topmost 2026-08-11 entry; technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.18; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new implementation subsection.

**Implemented, all 8 items of the design freeze's minimum implementation plan.** `PilotRunSpec.embedding_dropout` override field (additive, default-preserving); `load_pilot_policy()`'s per-profile `statics_embedding.dropout` `0.1` gate made override-aware — **deliberate deviation from the design-freeze plan's wording:** the top-level, policy-wide `embedding_dropout: 0.1` gate remains strict and unchanged (not made override-aware), preferred on independent review as safer since it keeps all non-overridden policy behavior identical; explicit candidate variation reaches the generated config only through `PilotRunSpec.embedding_dropout`, and Embedding-Dropout-A's five new run specs are spliced into the already-loaded/validated base policy in memory, so the committed policy-wide default is never exercised against a dropout-varying entry; `validate_embedding_dropout_override()` + `build_nh_config_mapping()` threading (`[0,1)` bound, `0.00` recorded as explicit `0.0`); `GeneratedConfigBundle`/manifest provenance (`embedding_dropout_override`/`resolved_embedding_dropout`); `build_pilot_run_identity()` extension; `enforce_pilot_embedding_dropout_identity()` continuation-safety guard (persist-on-first-call, compare-and-raise-on-mismatch), following `enforce_pilot_hidden_size_identity()`'s template; closure-splice launcher `scripts/run_stage1_embedding_dropout_range_seedA_closure.py` + `..._moriah.sbatch` (`EMBEDDING_DROPOUT_A_MAX_TARGET_EPOCH=6` fixed, exactly five trainable run_ids, `REFERENCE_RUN_ID` reachable only via `--status-only`); tests across all 8 planned categories.

**Preparation-only validation (real, unmocked, two layers).** The pytest suite's real `prepare_pilot_run_only()` calls for all five candidates against a synthetic package covering the full 2,557-basin development/spatial-holdout union, plus a standalone non-pytest CLI-subprocess audit invoking the real closure-launcher script with `--prepare-only` for all five run_ids. Both confirm: `resolved_embedding_dropout` exactly matches each candidate (`0.0/0.05/0.1/0.2/0.4`); `hidden_size=128`, `learning_rate=3e-4`, `output_dropout=0.25`, seed 967139, `seq_length=24` identical across all five; pairwise config diffs limited to `experiment_name`/basin-list paths/`run_dir`/`statics_embedding`; `training_started`/`evaluation_started`/`wandb_backend_initialized` all `False`.

**Tests.** Focused suites (`test_nh_config_generation.py` 154, `test_pilot_tracking.py` 42, `test_pilot_orchestration.py` 102/5 skipped, `test_pilot_lead06_config.py` 52), 3 dedicated campaign test files (130 passed), wider related-suite (455 passed/17 files), full local regression suite excluding 6 pre-existing torch/neuralhydrology collection-error files: 2070 passed, 5 skipped, 1 pre-existing unrelated Windows file-locking flake (`test_package_audit.py`, `os.rename`/`WinError 5`).

**Not done by this entry.** No embedding-dropout candidate launched, no Slurm job submitted, no real NeuralHydrology training or checkpoint evaluation, no W&B Sweep, no scientific-design change, no hydrograph panel rendered, nothing committed.

## Stage 1 — Embedding-dropout range characterization (Phase-A) design frozen, ready for implementation (2026-08-11)

Documentation-only design-freeze task, under unchanged commit `e5c6679464160e89d597363d1e1ae24d58310893` (verified against local `HEAD` and `origin/master` before this update; clean tracked tree), following the accepted read-only Embedding-Dropout Design Survey earlier in this session. Full decision text: `docs/decision_log.md`'s 2026-08-11 entry (topmost); technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.17; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-11 section.

**Design frozen.** Five new run_ids — `emb128x32_seedA_drop00_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop05_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop10_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop20_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_drop40_h128_lr3em4_cap25k_cal` — varying only `embedding_dropout` (`{0.00,0.05,0.10,0.20,0.40}`). Endpoint meaning: `0.00` = no-regularization control; `0.05` = light; `0.10` = inherited historical default (never itself evidence-selected); `0.20` = moderate; `0.40` = a deliberate high boundary intended to probe whether stronger embedding regularization becomes harmful at this Phase-A fidelity — a range characterization, **not** an optimized search grid. Everything else frozen at the LR-A/Hidden-size-A contract: `[128,32]` embedding (tanh activation, shape unchanged), Seed A (967139), `learning_rate=3e-4` fixed, `hidden_size=128` fixed, output dropout 0.25 (untouched), `seq_length=24`, Adam, no scheduler, target `qobs_mm_per_h_lead06`, lead 6h, the fixed 2,307-basin training population, the fixed 400-basin screening subset, `max_updates_per_epoch=25000`, six epochs, one uninterrupted segment, no continuation beyond epoch 6. Campaign: `embedding_dropout_range_seedA_25k_v001`.

**All five candidates fresh, including `0.10` (binding).** Unlike LR-A's reused `1e-3` reference, no candidate here reuses a historical run — the inherited `0.10` default is retrained fresh for uniform campaign identity/provenance/tracking. The fresh Hidden-size-A H=128 run (`emb128x32_seedA_h128_lr3em4_cap25k_cal`, `embedding_dropout=0.10` already, trained under the tracked-W&B contract) is retained strictly as an optional, read-only reproducibility comparator against the fresh `drop10` candidate — never a sixth member, never a substitute.

**Fidelity reuse + dropout-specific caveat.** Reuses the existing 25k-cap/6-epoch/Seed-A fidelity unchanged. New caveat: dropout can affect optimization speed differently than LR/hidden size, so poor performance at this fidelity is evidence "at this fidelity" only, not absolute rejection — full six-epoch trajectories matter more here than in prior Phase-A axes.

**Evaluation design.** Raw-space median NSE (400-basin screening subset) primary, unchanged; subset remains non-authoritative. Official cadence epochs 3/6; retrospective epochs 1/2/4/5 via the already-qualified `pilot_diagnostic_eval.py`, reused unmodified. Full epoch 1-6 trajectories required for all five candidates. No composite "winner score" and no predefined single winner-selection statistic — consistent with the standing "no single decision statistic" rule.

**Hydrograph rule + explicit non-goal.** The frozen 8-basin `phase_a_validation_hydrograph_panel_v001` panel remains the standing Phase-A sanity check, to be rendered in a later closure task for the provisionally strongest dropout value — not rendered by this entry. Monte-Carlo dropout, stochastic repeated inference, and inference-time-dropout experiments are explicitly out of scope: this campaign characterizes `embedding_dropout` as a training-time hyperparameter only.

**W&B contract.** Adopts the Hidden-size-A standard: default to the reviewed offline-enabled policy, hard-fail a real launch on tracking failure or null-backend resolution, unless explicitly waived.

**Not launched by this entry.** No dropout candidate has been trained, no Slurm job submitted, no code changed — design freeze only. The `embedding_dropout` override plumbing (including making `load_pilot_policy()`'s two hard-equality gates override-aware), continuation-identity guard, and closure-splice launcher/sbatch remain to be implemented in a later, separate task.

## Stage 1 — Hidden-size range characterization (Phase-A) closed; validation-compatible fixed 8-basin hydrograph panel v001 frozen and accepted (2026-08-10)

Documentation-only closure task recording the completed Moriah-executed hidden-size campaign (design freeze `785e631`, four real training runs) and a separately-executed, human-reviewed hydrograph sanity-check panel. No training, evaluation, Slurm job, config/HPO change, or basin reselection performed by this task; only change is a one-line fix to an untracked evidence-assembly script plus a metadata-only `status` field update in an already-committed selection driver script. Full decision text: `docs/decision_log.md`'s 2026-08-10 entry (topmost); technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.16; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s 2026-08-10 closing sections.

**Result.** Hidden size is not sharply sensitive over `{64,128,256,512}` at this Seed-A/LR=3e-4/25k-cap/6-epoch fidelity (epochs-4-6 median-of-medians NSE spans only ~0.255-0.278, non-monotonic in hidden size). H=128 is the provisional working anchor (not a final winner); H=64 is a genuine near-tie and a live Phase-B alternative (single best observed median NSE in the campaign, 0.2922 at epoch 6); H=256 a plausible upper useful capacity point; H=512 shows no demonstrated validation benefit and is dropped from the default Phase-B search space. Preferred Phase-B hidden-size support: `{64,128,256}`. Fresh-vs-historical H=128 audit: exact/deterministic reproducibility under the nominally equivalent Seed-A config (not cross-seed evidence). LR×hidden-size interaction unresolved, deferred to Phase B.

**Hydrograph panel.** `phase_a_validation_hydrograph_panel_v001` — 8 frozen basins (`01315000, 06894200, 07165565, 07261000, 08061540, 08072300, 12210900, 14301500`), built from the canonical 400-basin screening-validation population, distinct from the broader train-pool hydrograph atlas. Accepted findings: no systematic H64-vs-H128 hydrological superiority (consistent with the near-tie); modest LR=3e-4 edge over LR=1e-3; shared (not candidate-specific) limitations at basins 01315000/07165565/14301500 and systematic extreme-peak underprediction — none overturn the numerical conclusions. Geographically imbalanced (5/8 basins in `plains_missouri_south_central`), **not** CONUS-representative, not a second optimization objective. Selection status now `"frozen"` (was `"candidate"`); membership/windows unchanged. Standing rule adopted: render this same panel after each future one-dimensional Phase-A milestone as a sanity check only.

**Evidence-manifest bug fixed.** Self-referential checksum entry in `MANIFEST_SHA256.txt` (cosmetic; 72/73 real files always verified clean) fixed by excluding the manifest from its own `find` listing. Corrected packet: 72/72 OK, archive SHA256 `d88990b30b9452080acf44f46b127c8ad042bdab6b73f604f3ae173cc126d104`. Evidence (untracked, gitignored): `.scratch_local/moriah_evidence/phase_a_validation_hydrograph_panel_v001/` and `.tar.gz`.

**Not done.** No sealed-set access, no final Stage 1 hyperparameter selection, no basin reselection, no embedding-dropout implementation or training. **Next:** embedding-dropout design survey; Phase B later revisits LR×hidden-size×dropout jointly.

## Stage 1 — Hidden-size range characterization (Phase-A) design frozen, ready for implementation (2026-08-09)

Documentation-only design-freeze task, under unchanged commit `785e631f0111fd352035b5b234aec4a774f4aa97` (verified against local `HEAD` and `origin/master` before this update; clean tracked tree). Full decision text: `docs/decision_log.md`'s 2026-08-09 entry (the newest, above the W&B qualification entry); technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.15; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s new 2026-08-09 section.

**Design frozen.** Four new run_ids — `emb128x32_seedA_h64_lr3em4_cap25k_cal`, `emb128x32_seedA_h128_lr3em4_cap25k_cal`, `emb128x32_seedA_h256_lr3em4_cap25k_cal`, `emb128x32_seedA_h512_lr3em4_cap25k_cal` — varying only `hidden_size` (`{64,128,256,512}`). Everything else frozen at LR-A's contract: `[128,32]` embedding (not scaled with hidden size), Seed A (967139), `learning_rate=3e-4` fixed for all four (not re-tuned per hidden size), `seq_length=24`, output dropout 0.25, Adam, no scheduler, target `qobs_mm_per_h_lead06`, lead 6h, the fixed 2,307-basin training population, the fixed 400-basin screening subset, `max_updates_per_epoch=25000`, six epochs, one uninterrupted segment, no continuation beyond epoch 6. Campaign: `hidden_size_range_seedA_25k_v001`.

**Fresh H=128, not reused (corrects an earlier proposal).** The campaign trains a fresh `emb128x32_seedA_h128_lr3em4_cap25k_cal` rather than reusing the historical LR-A `3e-4` candidate (which also has `hidden_size=128`) — for uniform campaign identity/provenance/tracking, since the historical run predates this campaign's mandatory tracked-W&B contract and ran with tracking disabled. The historical run is retained read-only as a non-pooled reproducibility comparator only; that comparison is explicitly deferred until after the fresh H=128 run completes.

**W&B contract (new, strict).** The campaign launcher must default to the reviewed offline-enabled policy and must hard-fail a real launch if tracking initialization fails or resolves to backend `null`, unless an explicit human waiver is given — closing LR-A's item (9) operational gap for this campaign going forward.

**Evaluation design.** Official cadence epochs 3/6; retrospective epochs 1/2/4/5 via LR-A's already-qualified `pilot_diagnostic_eval.py`, reused unmodified. Full epoch 1-6 trajectories required in the final packet; no 3/6-only interpretation.

**Caveats recorded.** LR×hidden-size interaction is deliberately untested and deferred to Phase B (LR held fixed, not tuned per hidden size); the fixed `[128,32]` embedding's capacity relative to the recurrent pathway changes across the sweep (not scaled with H) — a deliberate simplification, not an oversight.

**Not launched by this entry.** No hidden-size candidate has been trained, no Slurm job submitted, no code changed — design freeze only. The `hidden_size` override plumbing, continuation-identity guard, `require_tracking` hard-fail contract, and closure-splice launcher/sbatch remain to be implemented in a later, separate task.

## Stage 1 — W&B offline tracking launch-contract qualification implemented and qualified, closing item 12(i) of the LR-A closure entry (2026-08-09)

Implements and qualifies the smallest generic, reusable fix for the LR-A closure entry's item (9)/(12)(i) W&B finding (all four LR-A runs used the disabled default policy, backend `null`, no real run IDs — an operational omission, not a scientific defect). Full decision text: `docs/decision_log.md`'s 2026-08-09 entry (the one above the LR-A closure entry); user-facing detail: `docs/stage1_wandb_user_guide.md` status item 6.

**Built.** `config/stage1_wandb_tracking_policy_offline_v001.yaml` (`enabled: true`, `mode: offline`; committed disabled default unchanged). `scripts/wandb_offline_launch_contract_qualification.py` + `..._moriah.sbatch`: exercises the real `--wandb-policy-path`/`WANDB_POLICY_PATH` launcher contract → real `load_tracking_policy` → real generic `init_tracking_run`, tagged unmistakably non-scientific (`launch_contract_qualification: true`); shares no code with the now-closed LR-A launcher beyond common boilerplate; never imports neuralhydrology/torch, never touches a sealed set. 30 new focused tests (7 script + 23 sbatch structural) plus 1 policy-load test extending `tests/test_wandb_tracking.py`.

**Qualified, real `wandb`, both locally and on Moriah.** Local (Windows, ephemeral venv, `wandb` 0.28.1): run id `em21le9y`, all 9 checks true. Moriah (`glacier` partition, CPU-only, Slurm job `45775192`, commit `fb2d6ae773993e8dd5a8cde65894fda14f5b4df7`, ephemeral venv, commit pin verified): `COMPLETED 0:0` in 00:01:09, run id `8hhayk8n`, all 9 checks true. Both: `backend="wandb"`, `mode="offline"`, non-null run id, real local offline run files created, no network dependency.

**Evidence.** Moriah evidence (720K, 6 files) transferred via `scp -O` to `.scratch_local/moriah_evidence/wandb_offline_launch_contract_qualification_45775192/` (untracked, gitignored), SHA256 verified byte-identical 6/6.

**Future-launch contract (recorded, not enforced by code).** Future real experiment launchers must explicitly provide the reviewed offline policy (or an explicit documented waiver); pre-launch qualification should verify `enabled=true`, `mode=offline`, backend `wandb`, non-null run id.

**Not done.** No hidden-size or other HPO screening; no new training/evaluation; LR-A's launcher/campaign machinery untouched; no scientific config generated by the qualification script.

## Stage 1 — LR-A (bounded learning-rate range characterization) closed: range evidence recorded, `3e-4` adopted as provisional Phase-A working anchor (2026-08-09)

Documentation-only closure task recording the completed, Moriah-executed LR-A campaign (design freeze `f300cb9`, implementation `bc8f253bed9231fc4a98233ffb2b92b16af8f743`, both already merged). No training, evaluation, Slurm job, W&B sync, package generation, or new analysis was run by this task; no source code was modified. Full decision text: `docs/decision_log.md`'s 2026-08-09 entry; technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.14; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s final 2026-08-09 closing section.

**Runs complete.** Four new candidates (`1e-4`, `3e-4`, `3e-3`, `1e-2`) trained on Moriah under the frozen contract (`[128,32]` embedding, Seed A, `max_updates_per_epoch=25000`, six epochs, 150,000 cumulative optimizer updates each); the `1e-3` candidate is the reused historical reference (`emb128x32_seedA_cap25k_cal`), never retrained. All five evaluated at all six epochs (30/30 cells).

**Result (range characterization, not final selection).** Epoch-6 median NSE ordering: `3e-4 (0.268) > 1e-4 (0.259) > 1e-3 (0.253) > 3e-3 (0.178) > 1e-2 (0.021)`. Useful LR region approximately `1e-4`-`1e-3`; `3e-4` adopted as the **provisional Phase-A working anchor** (not a final selected learning rate); `3e-3`/`1e-2` clearly too high for this model family at this fidelity. This is not proof `3e-4` is globally optimal — Phase B will revisit learning rate jointly with other hyperparameters.

**Cadence finding.** A 3/6-only evaluation cadence would have missed the true best-observed checkpoint for all 5/5 candidates; a 2/4/6 cadence recovered it for only 2/5. Future broad HPO should use denser evaluation or a sustained-performance objective, not a single endpoint.

**W&B finding.** All four new runs ran with tracking disabled (default policy, no offline-enabled override supplied) — an operational tracking omission with no effect on scientific validity. Fix planned as the next small increment, not implemented here.

**Evidence.** Durable local copy (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/lr_a_five_lr_evidence_v001/` and `lr_a_five_lr_evidence_v001.tar.gz` (SHA256 `624c5df4e1823e00b00a303a1c577790c3a72005cc217fcee5dc3e65f186f61c`); manifest verification 23/23 files OK.

**Not done by this entry.** No training/evaluation/Slurm/W&B/package-generation work; no source-code change; no generated evidence staged or committed; no final Stage 1 learning-rate selection.

## Stage 1 — LR-A implementation and preparation-only validation complete, ready for Moriah launch review (2026-08-08)

Implementation task following the design freeze immediately below: no Slurm job submitted, no real NeuralHydrology training or checkpoint evaluation, no W&B Sweep, no scientific-design change, nothing committed automatically. Full detail: `docs/decision_log.md`'s "LR-A implementation and preparation-only validation complete" 2026-08-08 entry; technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.13; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s "LR-A implementation complete" 2026-08-08 section.

**Built.** `learning_rate` override on `PilotRunSpec`/`nh_config_generation.py` with explicit manifest provenance; `pilot_diagnostic_eval.py` (all six checkpoints evaluable, off-cadence epochs tagged `retrospective_diagnostic`/non-authoritative, on-cadence epochs tagged `official`, early-stopping state never touched); `scripts/run_stage1_lr_range_seedA_closure.py` + matching `.sbatch` launcher (four new run_ids only, reused `1e-3` reference reachable read-only via `--status-only`); `checkpoint_comparison.py` (N-vs-1-reference table, late-window trajectory summary, cadence-sensitivity view — no composite score, matching the design freeze's "no single decision statistic" rule).

**Validated (preparation-only, real code, no mocking).** All four new candidates' generated configs and generation manifests confirmed to match the frozen LR-A contract exactly and to differ from each other only in `learning_rate` plus unavoidable identity/path metadata — checked against a real synthetic package covering the actual full basin population, not a byte-level diff against the historical `1e-3` reference (which remains external/read-only; see the reuse-equivalence audit's one flagged-not-blocking caveat, still open).

**Not done.** No LR candidate launched, no Slurm job submitted, no real training/evaluation call, no full-population validation, no hydrograph package.

## Stage 1 — LR-A (bounded learning-rate range characterization) design frozen: five-candidate range, all-epoch evaluation gap identified and planned, `1e-3` reuse from existing candidate, multidimensional HPO roadmap recorded (2026-08-08)

Documentation-only design-freeze task (no Moriah access, no Slurm submission, no training/evaluation, no experiment launch in this task), performed under unchanged commit `9b3b56f7dd68e876c9d02c8a6e5993698b0a9437` (verified against local `HEAD` and `origin/master` before this update; clean tracked tree, only pre-existing untracked scratch/report artifacts present). Full technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.12; decision text: `docs/decision_log.md`'s 2026-08-08 entry; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s 2026-08-08 section.

**Design frozen.** Five learning-rate candidates (`1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2`), `[128,32]` embedding fixed, Seed A, `max_updates_per_epoch=25000`, fixed six-epoch budget per candidate regardless of trajectory shape, checkpoint every epoch, evaluated at every epoch 1-6 on the existing fixed 400-basin screening set. Purpose: characterize the useful LR region around the current 0.001 baseline and inform checkpoint-cadence/objective design for a later multidimensional HPO phase — explicitly not final LR optimization and not a five-candidate tournament decided by any single statistic.

**Evaluation-path audit.** The official screening wrapper rejects off-cadence epochs (1, 2, 4, 5) by design; the lower-level evaluation primitives it is built from are epoch-agnostic and safely reusable for all six epochs without touching early-stopping state. This is a real but small implementation gap — a diagnostic-evaluation helper — planned but not built by this entry.

**Reuse decision.** The existing `emb128x32_seedA_cap25k_cal` candidate (2026-08-05 embedding-shape neighborhood screening) is reused as the LR-A `1e-3` reference without retraining, on the strength of a field-by-field scientific-equivalence audit and confirmed zero code/config drift since it was generated.

**Not launched by this entry.** No LR candidate has been trained, no Slurm job submitted, no code changed — this is a design freeze only. The four new candidates, the diagnostic-evaluation helper, the closure-splice launcher, and the generalized paired-comparison tool remain to be implemented and then executed in a later, separate milestone.

## Stage 1 — 50k Seed-A embedding-shape comparison closed: `[128,32]` adopted as working default, further embedding-shape exploration paused, bounded learning-rate calibration approved next (2026-08-06)

Documentation-only closure (no Moriah access, no Slurm submission, no training/evaluation, no new experiment launch in this task) recording a completed real Moriah closure run under unchanged commit `a4c5456331d97af61c71167a39bf5a6a0644d1ab` (verified against local `HEAD` before this update; clean tree before editing). Full technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.11; decision text: `docs/decision_log.md`'s 2026-08-06 entry; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s 2026-08-06 section. Evidence: Moriah `/sci/labs/efratmorin/omripo/Flash-NH/evidence/cap50k_closure_comparison_audit_2026-08-06/` and archive `cap50k_closure_comparison_audit_2026-08-06.tar.gz` (SHA256 `9ff1960bf7537da78ea62e5046805c28c0436bd1804395086e12c13c1a347207`); local extracted copy (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/cap50k_closure_comparison_audit_2026-08-06/`.

**Closure completed.** The 2026-08-05 (L.10) "next approved structural phase" ran to completion exactly as designed: the existing Seed-A `[128,64]` trajectory (`emb128x64_seedA_cap_low_cal`, job 45762223, continued from its epoch-6 state) and a new Seed-A `[128,32]` trajectory (`emb128x32_seedA_cap_low_cal`, job 45762224, fresh Seed-A initialization) both reached the fixed epoch-12 closure bound cleanly — exit `0:0`, final status `PAUSED_AT_MAX_TARGET_EPOCH`, physical checkpoints and official screening through epoch 12, no overshoot, no sealed-data access, no W&B sync. Both share Seed A (967139), target `qobs_mm_per_h_lead06`, lead 6 h, `seq_length=24`, `hidden_size=128`, learned FC static embedding, tanh activation, embedding dropout 0.1, output dropout 0.25, Adam `lr=0.001`, NSE-style training loss, `max_updates_per_epoch=50000`, the fixed 2,307-basin training population, and the fixed 400-basin development-validation screening set — differing only in `statics_embedding.hiddens`.

**Official raw-space result (400-basin screening population, median NSE).**

| Epoch | Incumbent `[128,64]` | Challenger `[128,32]` |
|---|---|---|
| 3 | 0.2418 | 0.2480 |
| 6 | 0.2547 | 0.2541 |
| 9 | 0.2367 | 0.2569 |
| 12 | 0.2427 | 0.2464 |

**True per-basin paired result (challenger minus incumbent, 400/400 matched basins, tie tolerance ±0.01).**

| Epoch | Median ΔNSE | Q25 | Q75 | Challenger better | Incumbent better | Tied |
|---|---|---|---|---|---|---|
| 3 | +0.0136 | -0.0293 | +0.0650 | 53.5% | 35.0% | 11.5% |
| 6 | +0.0145 | -0.0294 | +0.0640 | 53.25% | 34.75% | 12.0% |
| 9 | +0.0160 | -0.0330 | +0.0636 | 54.25% | 33.25% | 12.5% |
| 12 | +0.0072 | -0.0447 | +0.0709 | 48.5% | 41.5% | 10.0% |

**Adopted interpretation (cautious).** `[128,32]` is at least comparable to `[128,64]` and shows a small, directionally consistent paired advantage at epochs 3, 6, and 9; the advantage weakens by epoch 12 (median ΔNSE narrows to +0.0072 and the win-rate margin narrows to 48.5%/41.5%, down from roughly 53-54%/33-35% at the earlier epochs). The effect is modest relative to cross-basin heterogeneity — the paired IQR (Q25-Q75) spans roughly 0.10 at every epoch, an order of magnitude wider than the median shift. The comparison rests on one seed only (Seed A, 967139); no independent seed replication was run at this fidelity. Transformed-space training-loss diagnostics (`docs/stage1_lead06_pilot_v001.md`'s 2026-08-06 section) point in the same direction — challenger's training loss is consistently lower than incumbent's across all 12 epochs — but per standing policy this is a training diagnostic only, never the official scientific benchmark.

**Adopted decision: `[128,32]` becomes the working default embedding shape.** Not because it is decisively superior — it is not — but because it is at least as competitive as `[128,64]` on the official raw-space and paired evidence while being more economical (fewer static-embedding parameters). This decision does not imply that static attributes are unimportant, and does not imply that static attributes should be removed — the comparison is between two learned-embedding widths, not between the embedded and raw static pathways (that separate, still-open question is unaffected by this entry).

**Further embedding-shape (width/depth) exploration is paused.** Given the `[128,64]`/`[128,32]`/`[64,32]`/`[256,64]` evidence gathered across the 25k neighborhood screening and this 50k closure, further exploration of this axis now has low expected value relative to other open hyperparameters. This is a pause, not a permanent close — new evidence could reopen it. No model-family switch is proposed by this entry.

**Early-stopping / closure interpretation.** Both trajectories share best official screening epoch 6 under the current early-stopping history (incumbent best value 0.25474, challenger 0.25414); neither met an early-stopping condition (`stopped=false`, `stop_reason=null`) before epoch 12 in either trajectory. Termination at epoch 12 was caused solely by the fixed `CLOSURE_MAX_TARGET_EPOCH=12` closure bound, not by early stopping. Both evidence bundles' generic `continuation_status.next_intended_screening_epoch=15` / `safe_to_continue_automatically=true` fields describe what the unbounded 36-epoch policy would do next — they were never executed and must not be read as a planned or approved continuation.

**Next approved scientific phase (design only, not launched): bounded learning-rate calibration.** `[128,32]` fixed as the working embedding; current baseline learning rate 0.001; exact candidate values not yet frozen (no existing approved document specifies them); same fixed 400-basin raw-space validation contract; staged promotion (coarse fidelity first) rather than running every candidate to epoch 12 automatically. **Not launched by this entry.**

**Operational efficiency item (deferred engineering work, not a blocker).** Each nested NH continuation boundary (`continue_training_from_epochNNN/`) reloaded the full dataset, recalculated target standard deviations, and rebuilt lookup tables and dataloaders, adding roughly 20-40 minutes per boundary against a roughly 4-minute steady-state epoch — approximately 25-45% of total wall time across the two continuation boundaries in this comparison. Quantified from checkpoint-file mtimes (both evidence bundles' `epoch_timing_table` is empty). Recorded as a future optimization target; **not fixed by this entry.**

**Not done by this update.** No Moriah access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout access, no learning-rate experiment implemented or launched.

## Stage 1 — 25k Seed-A embedding-shape neighborhood screening closed: `[128,64]`/`[128,32]` structural survivors, next 50k comparison approved, sequence length reframed as a separate model-family axis, learning-curve/hydrograph standards revised (2026-08-05)

Documentation-only closure (no Moriah access, no Slurm submission, no training/evaluation in this task) recording a completed real Moriah screening batch run earlier in this same session under unchanged commit `5aba586dc4856ecb05945b41d3ff29a34f096cb7` (verified against local `HEAD` and `origin/master` before this update; both identical, clean tree before editing). Full technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.10; decision text: `docs/decision_log.md`'s 2026-08-05 entry; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s 2026-08-05 section. Local evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/embedding_shape_neighborhood_seedA_25k_v001/`.

**Screening closed.** Three new capped-update (`max_updates_per_epoch=25000`) candidates — `emb64x32_seedA_cap25k_cal` (`[64,32]`), `emb128x32_seedA_cap25k_cal` (`[128,32]`), `emb256x64_seedA_cap25k_cal` (`[256,64]`) — were trained and compared against the pre-existing, untouched `emb128x64_seedA_cap25k_cal` reference (`[128,64]`). All four share Seed A (967139), `hidden_size` 128, embedding activation tanh, embedding dropout 0.1, output dropout 0.25, Adam `lr` 0.001, NSE loss, `seq_length` 24, `qobs_mm_per_h_lead06`, the fixed 2,307-basin development-training population, and the fixed 400-basin development-validation screening subset — differing only in `statics_embedding.hiddens`. All three new candidates completed successfully, performed exactly 25,000 optimizer updates in every one of 6 epochs (directly verified from each `optimizer_state_epochNNN.pt` Adam step counter — 25,000/50,000/75,000/100,000/125,000/150,000 cumulative, zero drift), reached 150,000 cumulative updates at epoch 6, created checkpoints 1-6 and no epoch 7, stayed offline in W&B throughout, and accessed no sealed population.

**Provisional interpretation (coarse-screening resolution only).**
1. **No tested candidate shows broad, consistent superiority over the `[128,64]` reference.**
2. **`[128,32]` shows the mildest positive edge**: positive median paired per-basin difference vs. `[128,64]` at 5 of 6 epochs (only epoch 2 slightly negative), but its paired win rate against the reference never exceeds ≈0.53 in any epoch — a plausible challenger, not a demonstrated winner.
3. **`[64,32]` stays broadly close to `[128,64]`**: positive median paired difference at all 6 epochs, but no stable or broad advantage (a one-off epoch-5 spike is not sustained at epoch 6) — no compelling reason to prioritize it further.
4. **`[256,64]` is the weakest tested shape**: negative median paired difference vs. the reference in 4 of 6 epochs, and the lowest official median NSE at both official screening epochs (3 and 6) — provisionally rejected at this 25k structural-screening tier.
5. **The 25k cap remains useful for divergence detection and rejecting clearly weak regions** (it separates `[256,64]` as weakest) **but is not precise enough for fine ranking** among `[64,32]`/`[128,32]`/`[128,64]`, whose paired win rates all sit close to chance.

**Structural survivors for the next phase: `[128,64]` (incumbent) and `[128,32]` (challenger).** Neither is described as the final architecture.

**Next approved structural phase (not yet started): existing Seed-A `[128,64]` trajectory continued to 50k vs. new Seed-A `[128,32]` trajectory at 50k.** Design: `max_updates_per_epoch=50000`, target up to epoch 12, official screening at epochs 3/6/9/12, existing early-stopping policy authoritative (stopping-eligible from epoch 6, minimum improvement 0.005, patience 3 eligible screening events), every epoch saved, retrospective checkpoint evaluation usable diagnostically, no cross-fidelity checkpoint reuse. The new `[128,32]` 50k candidate starts from the original Seed-A initialization; the existing `[128,64]` 50k candidate (`emb128x64_seedA_cap_low_cal`) may continue only within its own unchanged candidate identity and fidelity. **Not launched by this entry.**

**Sequence length reframed (adopted, binding for the current model family).** Sequence length is fixed at 24 for the current model family and is **not** an ordinary near-term tuning-funnel hyperparameter — alternative sequence lengths define separate temporal-context model families (different scientific information, antecedent-memory assumptions, input construction, compute/memory cost, and cross-basin response-time interpretation). A later sequence-length study may compare alternative model families against a mature 24-hour model, but that is not part of the current hyperparameter phase. `docs/stage1_validation_optimization_foundation.md` Part L.1's Stage B dimension list is corrected accordingly. **Further revised (2026-08-13, see this document's 2026-08-13 entry and `docs/stage1_validation_optimization_foundation.md` Part L.19):** the Embedding-Dropout-A closure schedules a dedicated Sequence-Length-A characterization (`seq_length={12,24,48,72}`), reframing sequence length as a bounded, structural/calibratable model parameter — this passage is preserved as historical, not rewritten.

**Revised hyperparameter order within the fixed `seq_length=24` model family:** (1) close embedding structure at 50k (`[128,32]` vs. `[128,64]`); (2) learning rate (bounded contrast around 0.001, exact values not yet authorized); (3) LSTM hidden size (bounded capacity contrast, exact candidates not yet authorized); (4) embedding dropout; (5) output dropout; (6) small integration/interaction checks among independently promising settings; (7) Seed-B confirmation for only the top integrated candidates; (8) uncapped authoritative finalists; (9) a separate, later temporal-context model-family study for sequence length. Ordering rationale: expected scientific/optimization impact, dependency/interaction structure, experimental clarity, and operational cost.

**Learning-curve standard (adopted for future serious-triage/finalist packets).** Training diagnostics: mean training loss vs. epoch and vs. cumulative optimizer updates. Validation/scientific diagnostics: median raw-space per-basin NSE vs. epoch, a p25-p75 (or equivalent) distributional band, `frac(NSE>0)`, and explicit official-vs-retrospective evaluation markers. Transformed-space validation loss may be included only if already available or cheaply/deterministically derivable, and only as a training diagnostic — **never as the official scientific model-selection metric.** The existing scientific rule is preserved: NH training/validation losses are diagnostics in transformed target space; official Flash-NH benchmark metrics remain computed after full inverse conversion to raw m³/s; raw-space screening metrics remain authoritative for candidate selection. Raw-space median NSE must not be labeled "validation loss."

**Hydrograph-demonstration standard (design update; not yet implemented — see `docs/stage1_validation_optimization_foundation.md` Part L.3d).** For future 50k-promoted candidates: a fixed eight-basin compact panel with basin area (km², authoritative basin-area field) in every title; basin-average hourly MRMS QPE precipitation (mm h⁻¹) as blue bars descending from a secondary right-hand axis (zero at top, increasing downward); rainfall plotted at its physical valid time, observations at physical discharge time, lead-6 predictions at their target valid time (no artificial six-hour shift); matched time windows/discharge limits/precipitation limits/plot conventions across compared candidates; a compact-panel metrics table; and a short interpretation covering peak magnitude/timing, false peaks, recession, baseflow, basin-specific bias, and rainfall-runoff timing. The full 24-basin atlas remains reserved for ambiguous cases, integrated candidates, or authoritative finalists — not required for every 50k candidate. Cadence: 25k coarse screening uses strategic metrics/learning curves only (no routine hydrograph package); 50k serious triage uses the compact panel + compact metrics + short interpretation; integrated/uncapped finalists use the compact panel + full 24-basin atlas + standardized 6-8 figure package + comprehensive summary.

**W&B status (already-qualified capability only; nothing new qualified by this entry).** Offline W&B remains the operational mode for all four screening runs above (`tracking_generation=g1`, no `wandb sync` run for any of them); the previously-qualified controlled post-run sync (`docs/stage1_wandb_user_guide.md` §17) covers two other, unrelated single-segment runs and is unaffected by this entry; the private project remains entity `omri-porat1-huji`, project `flashnh-stage1`; W&B stays the experiment-index/comparison interface, not scientific authority; online training and multi-segment offline-run reconciliation remain unqualified; no automatic sync or `--sync-all` workflow is approved.

**Not done by this update.** No Moriah access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no generated evidence committed or staged, no sealed temporal-test/spatial-holdout access. The next approved 50k comparison (`[128,64]` continuation vs. new `[128,32]`) is approved in design only and has not been launched.

## Stage 1 — `max_updates_per_epoch` capped-update calibration complete: mechanism qualified, provisional fidelity workflow adopted, static embedding reopened as a bounded hyperparameter family (2026-08-04)

Documentation-only closure (no Moriah access, no Slurm submission, no training/evaluation in this task) consolidating three real Moriah calibration exercises run earlier in this same session under unchanged commit `ac98f6b3ad9b1687a26a7509f98a02df3c06381b`: (1) an uncapped-reference/two-cap calibration (`cap_learning_diagnostics_v001`), (2) a matched raw-vs-embedded 50k comparison (`emb50k_architecture_diagnostic_v001`), and (3) a four-candidate parallel batch at 25k/50k caps across both static pathways and both seeds (`cap_parallel_batch_v001`). Full technical detail: `docs/stage1_validation_optimization_foundation.md` Part L.9; decision text: `docs/decision_log.md`'s 2026-08-04 entry; candidate-level detail: `docs/stage1_lead06_pilot_v001.md`'s 2026-08-04 calibration section. Local evidence (untracked, gitignored, never staged): `.scratch_local/moriah_evidence/{cap_learning_diagnostics_v001,emb50k_architecture_diagnostic_v001,cap_parallel_batch_v001}/`.

**Mechanism: operationally qualified.** Across all runs above (uncapped reference, 100k/50k initial caps, 25k/50k parallel batch, raw and learned-embedding static pathways, Seed A 967139 and Seed B 1729, sequential and parallel GPU execution), the configured cap matched the actual per-epoch optimizer-update count exactly, every epoch, verified from each run's own persisted `optimizer_state_epochNNN.pt` step counter (never inferred from wall time or logs). Run-identity isolation, per-epoch checkpointing, continuation-cap safeguards, screening, evidence-bundle generation, and offline W&B all behaved as designed; no scientific hyperparameter, split, early-stopping, or sealed-population access was affected. The true full-population uncapped optimizer-updates-per-epoch count is now measured: **237,298** (Seed A, raw pathway, real Moriah NeuralHydrology 1.13) — this closes the measurement gap previously left open in Part L.5/L.8.

**Performance interpretation: coarse screening only, not fine ranking.** Median per-basin raw-space NSE, `frac(NSE>0)`, and `frac(NSE<-1)` stayed broadly coherent across every capped candidate and every fidelity tested (25k/50k/100k, both pathways, both seeds) — no divergence, collapse, or runaway gain at any epoch. But true per-basin paired differences (matched by basin, not epoch-aggregate deltas) are consistently wider than the aggregate/architecture/seed effects under study: e.g. the four-candidate batch's epoch-6 Seed-A cap-sensitivity comparison (25k vs the protected 50k reference) shows a raw-pathway median paired diff of +0.0108 (400/400 finite pairs, p25/p75 -0.037/+0.064, 56.5% of basins favoring the 50k reference) and an embedded-pathway median paired diff of -0.0092 (55% favoring 25k) — small, sign-inconsistent aggregate effects sitting well inside a roughly 0.10-wide paired IQR. **Capped runs are therefore useful for coarse rejection/triage, not fine ranking; capped performance must not be read as proof that fewer updates are scientifically superior, and capped checkpoints must not be promoted to authoritative full-fidelity trajectories.**

**Runtime interpretation: fixed costs dominate; do not assume linear wall-time scaling.** Halving the per-epoch update budget (50k reference to 25k) roughly halved training-loop optimizer work, but total Slurm-elapsed time through epoch 6 improved by only ~12-18% in the two matched pairs available (raw: 2625.16s vs 3213.3s, 18.3% faster; embedded `[128,64]`: 2657.5s vs 3020.2s, 12.0% faster) — validation, startup, and checkpointing overhead are a large, largely fixed share of total elapsed time and do not shrink with the update cap.

**Provisional fidelity workflow (adopted, not binding as a final scientific method).** 25k = first-pass coarse rejection; 50k = second-stage triage for plausible candidates; uncapped/full fidelity = finalists only. Safeguards: each fidelity is a distinct run identity; a promoted candidate restarts from its original seed at full fidelity, never continuing from a capped checkpoint; capped results remain provisional; only full-fidelity finalists may support a final architecture decision. **This is a provisional workflow decision, not a final scientific result.**

**Evaluation cadence (adopted for the current diagnostic phase only).** Official screening at epochs 3/6/9/etc. and current early-stopping semantics are preserved; every epoch is still saved; retrospective, diagnostic-only per-epoch (1/2/4/5) evaluation is used selectively for close/promising/puzzling/new-family candidates during this structural-calibration phase, and must never feed back into authoritative early-stopping or checkpoint-selection state. For later routine broad campaigns, every-epoch retrospective evaluation is **not** automatic for every 25k candidate — routine coarse rejection keeps the lighter official cadence only.

**Static embedding: still open, reframed as a bounded hyperparameter family.** The matched 50k raw-vs-`[128,64]`-embedded comparison (Seed A, identical settings otherwise) shows the embedded pathway with a modest directional edge across most of epochs 1-6 (per-basin win share favors embedded at every epoch, narrowing by epoch 6 to 44.25%/42.25% raw/embedded with a median paired diff of only -0.0025) — but basin-level spread (IQR ~0.35-0.37) is an order of magnitude larger than this aggregate gap. **Raw and learned `[128,64]` embedding remain close; direction and magnitude vary with epoch, seed, and fidelity; this is not a resolved comparison.** Static embedding architecture is now treated as a bounded hyperparameter family (raw; `[64]`; `[128]`; `[128,64]`), not a settled raw-vs-embedded question. **Next approved, not-yet-started structural batch:** one-layer `[64]` and one-layer `[128]` embeddings, Seed A, 25k cap, compared against the existing Seed-A raw/`[128,64]` references — embedding activation (tanh), embedding dropout (0.1), output dropout (0.25), `hidden_size` (128), learning rate (0.001), and all data/split settings held fixed. No dropout/learning-rate/hidden-size/broad-HPO tuning is in scope until this shape axis narrows. **These runs have not started.**

**Strategic review packet standard.** A 7-component compact-evidence-bundle standard for *future* structural-comparison tasks was added to `docs/decision_log.md` (not applied retroactively to the runs above): `PROVENANCE.json`, a per-candidate/per-epoch metrics table, true basin-paired comparison stats, a runtime/updates table, a config-diff matrix, 6 compact plots, and a `strategic_summary.md`. Compact tables/summaries are authoritative for strategic review; plots are diagnostic aids; large checkpoint/pickle/W&B-binary files are excluded from the local packet by default.

**Not done by this update.** No Moriah access, no Slurm submission, no training or evaluation, no production code/tests/config/policy-YAML change, no numerical cap adopted for production use, no generated evidence committed, no sealed temporal-test/spatial-holdout access. The next embedding-shape batch (`[64]`/`[128]`, Seed A, 25k) is approved in direction only and has not been launched.

## Stage 1 — W&B tracking qualification status (documentation + tests + local no-network smoke only) (2026-08-02)

W&B tracking (`src/baseline/wandb_tracking.py`,
`src/baseline/pilot_tracking.py`) ahead of possible live use on
`raw_seedA`, per the adoption sequencing in
`docs/stage1_validation_optimization_foundation.md` Part L.4. No training,
evaluation, Moriah/h2o access, Slurm submission, online W&B use, sweep,
`raw_seedA` launch, or repository staging/commit occurred.

**Status, precisely.** Stage (1) wrapper contract — fake-backend pytest
only (`tests/test_wandb_offline_qualification.py` + integration coverage
in the tracking/orchestration test files) — **qualified**. Stage (2) real
package, offline mode — real, installed **wandb 0.28.1**, two independent
OS processes, no network, no API key
(`scripts/wandb_real_offline_qualification_smoke.py`) — **qualified**.
Stage (3) online mode — **not qualified**. Stage (4) sweeps — **not
implemented**. Two failure-mode gaps (backend-call failure isolation;
stable run identity across bounded Slurm continuations, including a
`tracking_generation` fix for a deliberate restart-from-scratch) were
found and fixed in both wrapper modules. One prior assumption was
corrected by the real-package evidence: offline `resume="allow"` does not
locally continue a prior run directory; each invocation gets a fresh local
directory, and reconciliation happens only server-side at `wandb sync`
time. Wrapper default is unchanged: `enabled: false` / `mode: disabled`.
`raw_seedA` remains the next scientific candidate; this update does not
launch it and does not itself decide to enable tracking for it.

Full technical detail and scope limits:
`docs/stage1_validation_optimization_foundation.md` Part L.4. Concise
decision/rationale: `docs/decision_log.md`'s two 2026-08-02 W&B entries
(top of file). User-facing guide: `docs/stage1_wandb_user_guide.md`.
Candidate-specific integration status: `docs/stage1_lead06_pilot_v001.md`'s
"W&B tracking" section.

**Next step:** none launched by this update. `raw_seedA` remains the
preferred next Stage A candidate, pending separate review/launch decision.

## Stage 1 lead-6 pilot — `emb128x64_seedA` real 24-basin hydrograph-atlas evaluation + rendering complete, visual review adopted (2026-08-02)

Real Moriah execution (not documentation-only): a disposable, evaluation-only
derivative of `emb128x64_seedA`'s selected epoch-6 checkpoint (same weights,
same fitted scaler, same config, never retrained) was built and evaluated
against the fixed 24-basin hydrograph atlas for the validation period only
(job `45729427`, `PASS`), then rendered into 24 individual atlas panels plus
a deterministic 8-basin compact panel (job `45729449`, `PASS`), reusing the
existing rendering tooling and raw-space conversion/metric code unchanged.
Original run directory, checkpoint, scaler, config, and the ~400-basin
screening validation results pickle are verified byte-identical before and
after. **Diagnostic/visualization only** — not authoritative for checkpoint,
architecture, or hyperparameter selection, and not a replacement for the
provisional ~400-basin screening result.

**Adopted visual interpretation (diagnostic observation, not formal proof).**
Genuine hydrologic signal, with many predicted events in approximately the
correct temporal neighborhood and no obvious universal six-hour displacement
or global raw-space conversion failure — but performance remains weak and
hydrologically inconsistent: large observed peaks are commonly attenuated,
some basins show false/exaggerated predicted peaks, recession/baseflow
behavior is often poor, and bias varies strongly by basin. Supports
continuing structural optimization; does **not** establish model adequacy,
architecture superiority, full development-validation performance, or final
Stage 1 readiness. The atlas's aggregate median NSE (≈0.14) is not
representative of the ~400-basin screening metric.

**No sealed-set access, no full-population evaluation, and no
model-selection change resulted from this update.** `raw_seedA` remains the
preferred next Stage A structural candidate; W&B offline-mode qualification
remains the immediate next operational preparation before that launch.
Full technical/operational detail: `docs/stage1_validation_optimization_foundation.md`
Part L.3c; full decision text: `docs/decision_log.md`'s 2026-08-02 entry
(same date, above the roadmap entry below); full evidence write-up
(untracked): `reports/stage1_validation_optimization_foundation_v001/part_l_atlas24_eval_emb128x64_seedA_v001/`.

**Next step:** none launched by this update. `raw_seedA` remains the
preferred next Stage A candidate, pending separate review/launch decision.

## Stage 1 lead-6 pilot — post-`emb128x64_seedA` roadmap: Stage A/Stage B, hydrograph timing, W&B sequencing, multi-fidelity direction (2026-08-02)

Documentation-only roadmap patch, written after `emb128x64_seedA`'s
completion below. Full text: `docs/decision_log.md`'s 2026-08-02 entry;
full framing: `docs/stage1_validation_optimization_foundation.md` Part L;
pilot-specific next step: `docs/stage1_lead06_pilot_v001.md`'s "Current
status and next step" section.

**Adopted, binding.** Stage A (this six-run structural pilot) is distinct
from Stage B (proper HPO, deferred) and is not a hyperparameter sweep.
Preferred next candidate: `raw_seedA` — not a launch authorization; the
remaining four runs are reviewed between candidates, not auto-launched.
Hydrographs move earlier as an early diagnostic, not a mandatory gate.
W&B adoption is sequenced (tracking → offline test → live → sweeps after
Stage B freezes); repository code stays authoritative, never W&B.
Preferred multi-fidelity mechanism: NeuralHydrology's
`max_updates_per_epoch`. Full framing and rationale:
`docs/stage1_validation_optimization_foundation.md` Part L (linked above).

**Provisional, not binding.** Fidelity fractions (low ≈10-15%, medium
≈25-50%, full uncapped) and the screening-only/restart-from-seed
capped-update recommendation are starting points only — the real
full-epoch update count is unresolved against Moriah NeuralHydrology 1.13,
and capped-update stopping/promotion semantics remain an open design item.
Full caveats and open questions: Part L.5-L.6 of the same document.

**Not done by this update.** No run launched or reviewed as a result of
this entry; no hydrograph generated; no W&B tracking enabled; no
`max_updates_per_epoch` value set anywhere in the repository; no code,
config, or Slurm script changed.

## Stage 1 lead-6 optimization pilot — `emb128x64_seedA` candidate complete: continuation adopted, epochs 12 and 15 screened (job 45722908) (2026-07-30)

The production `pilot_accepted_continuation.json` manifest (real SHA-256
hashes, filename-bound to each entry's own key epoch per the trust-binding
correction) was used in job `45722908` (partition `catfish`, source commit
`af8945d04451d7699ab54b13082eaf870f04f28e`, elapsed `00:10:34`, `COMPLETED`,
exit `0:0`) to adopt the existing epoch 6→15 continuation with **no
retraining**; epochs 12 and 15 were evaluated sequentially. Final screening:
epoch 6 median NSE `0.20454161610527344` (best), epoch 9
`0.18124855313577198`, epoch 12 `0.1993193615763258`, epoch 15
`0.17125263282608943` — no improvement after epoch 6, so early stopping
fired at epoch 15 (`patience_exhausted`). **Epoch 6 is the selected
checkpoint for the `emb128x64_seedA` candidate only**, not the final Stage 1
production model — that requires comparing all six run specifications in
the wider optimization campaign, which is the next phase. Sealed
temporal-test and spatial-holdout populations were not accessed. This
closes the continuation-repair/adoption sequence described in the entry
below. Full detail: `docs/decision_log.md`'s 2026-07-30 closure entry;
`docs/stage1_lead06_pilot_v001.md`'s "Fifth Moriah result" section.

## Stage 1 lead-6 optimization pilot — `emb128x64_seedA` epoch 6→15 continuation: provenance review + explicit adoption mechanism (2026-07-30)

Direct inspection of the real job-45705457 continuation evidence
(`flashnh_emb128x64_seedA_continuation_evidence_2026-07-29.txt`) confirmed
epochs 7-15 come from one single, uninterrupted `Continue training from
epoch 6` invocation at the frozen pilot config — **conditionally safe to
adopt**. A narrow, run-specific adoption mechanism was then implemented in
`src/baseline/pilot_orchestration.py`: a strictly opt-in
`pilot_accepted_continuation.json` manifest (base run directory, never
committed, no general CLI override) pins one epoch's model+optimizer
checkpoint by SHA-256; `_advance_chunk_via_continuation` consults an entry
only when that epoch is the exact chunk currently being resolved, so an
accepted epoch 15 is never consulted while epoch 12 is still due, and stays
unused if early stopping fires at epoch 12. 10 new focused tests added
(`tests/test_pilot_orchestration.py`, 44 passed total). See
`docs/decision_log.md`'s 2026-07-30 entry for the full provenance verdict and
manifest contract, and `docs/stage1_lead06_pilot_v001.md` for the schema.
**Superseded:** the real epoch-12/epoch-15 SHA-256 hashes were subsequently
filled in and the manifest used in a real Moriah adoption/screening run —
see the closure entry above, "`emb128x64_seedA` candidate complete."

## Stage 1 lead-6 optimization pilot — real Moriah verification (job 45718742): launcher classification fix confirmed, rerun-idempotency defect found and fixed (2026-07-30)

Slurm job `45718742` (partition `catfish`, source commit
`7c6b02a599b885682a97081a3f166d97097bd4ec`, elapsed `00:03:17`, no stderr)
confirmed the previous launcher-status fix works: the launcher correctly
classified the run as `BLOCKED_MANUAL_REVIEW_REQUIRED`
(`pilot_final_status: blocked_continuation_overshoot_conflict`,
`safe_to_continue_automatically: false`, overshoot epochs 10-15, exit code
1). **No training occurred; scientific state was not modified.** But
before reaching that clean overshoot block, the Python pilot process
crashed: `PilotEarlyStoppingError: epoch 6 is not after the last recorded
epoch 9 -- out of order`, from `run_pilot() -> run_pilot_chunk() ->
record_screening_event(epoch=6)`. Root cause: `run_pilot()` always
restarts its chunk walk from epoch 6 on every call, and the screening
loop had no check for an epoch already present in this run's persisted
`pilot_orchestration_state.json` (`logged_screening_epochs: [3, 6, 9]`),
so it re-fed already-screened epoch 6 into the early-stopping state
machine after the persisted history's last entry had already advanced to
epoch 9. Fixed narrowly in `src/baseline/pilot_orchestration.py`: an
already-logged screening epoch is now skipped outright (no
re-evaluation, no re-record), with a light consistency check against the
reloaded early-stopping history so genuinely inconsistent state is never
silently skipped. New focused test
`test_run_pilot_end_to_end_rerun_of_fully_screened_earlier_chunks_is_idempotent`
reproduces job 45718742's exact shape; verified to fail with the real
job's exact error before the fix and pass after it, with both persisted
state files unchanged. **No further Moriah run should occur until this
narrow local rerun-idempotency fix is committed.**

## Stage 1 lead-6 optimization pilot — real Moriah recovery (job 45718473): correct scientific recovery, launcher status-propagation defect found and fixed (2026-07-30)

Recovery job `45718473` (partition `catfish`, one L4 GPU, elapsed 00:08:12,
`COMPLETED`/exit 0:0) ran the continuation-nesting/additive-epoch fix
described below against the real `emb128x64_seedA` artifact. Scientifically
correct: no training occurred, the existing
`continue_training_from_epoch006/model_epoch009.pt` checkpoint was reused,
epoch 9 was screened and logged exactly once (median per-basin raw-space
NSE `0.18124855313577198`), epoch 6 remains best (`0.20454161610527344`),
`events_since_best_improvement: 1`, `stopped: false`, and overshoot
checkpoints 10-15 remain preserved and untouched.

However the launcher (`scripts/run_stage1_lead06_pilot_moriah.sbatch`)
reported an internally inconsistent result: `status: COMPLETED`,
`pilot_final_status: null`, `blocked_reason: null`, alongside a correctly
computed `safe_to_continue_automatically: false` and
`overshoot_epochs: [10, 11, 12, 13, 14, 15]`. Root cause: the pilot CLI's
primary stdout JSON was unavailable when the launcher read it, so the
launcher's own documented fallback path (computing status fields directly
from on-disk state via `compute_pilot_status_fields`) correctly restored
`overshoot_epochs`/`safe_to_continue_automatically`, but left
`pilot_final_status`/`blocked_reason` as `None` — and the launcher's
classification only branched on `pilot_final_status`, so it fell through to
`elif pilot_status == 0: COMPLETED`. Confirmed by direct code reading that
`pilot_orchestration.run_pilot()` itself already propagates a blocked
chunk's `final_status`/`blocked_reason` correctly through its own return
value (new end-to-end test,
`test_run_pilot_end_to_end_propagates_blocked_continuation_overshoot_conflict`
in `tests/test_pilot_orchestration.py`) — the loss was isolated to the
launcher's fallback classification, not to `run_pilot()`.

Fix (local only, not yet re-run on Moriah): the launcher's fallback now also
derives `pilot_final_status = 'blocked_continuation_overshoot_conflict'` and
a non-null `blocked_reason` whenever the fallback-computed
`safe_to_continue_automatically` is `False` with a non-empty
`overshoot_epochs`, reusing the exact status string
`pilot_orchestration.run_pilot()` already uses for this condition; the pilot
CLI (`scripts/run_stage1_lead06_pilot.py`) now also exits `1` (the
launcher's own existing "needs a human" convention) when `final_status`
is `blocked_continuation_overshoot_conflict`, rather than always exiting 0.
No training, evaluation, metric, stopping, checkpoint, or overshoot logic
changed. Two new behavioral tests in `tests/test_pilot_sbatch_launcher.py`
extract and execute the launcher's status-classification snippet standalone
(no Slurm) against job 45718473's exact on-disk shape, confirming it now
reports `BLOCKED_MANUAL_REVIEW_REQUIRED` (not `COMPLETED`), and that an
ordinary completed/stopped run is unaffected. **No further Moriah job
should run until this local status-propagation fix is committed** — a
resubmission before that would repeat the same misleading `COMPLETED`
report on the next blocked chunk. Full detail: `docs/decision_log.md`'s
2026-07-30 status-propagation entry.

## Stage 1 lead-6 optimization pilot — second qualification-run failure, continuation-nesting/epoch-semantics corrected (2026-07-30)

The resumed Moriah job (`45705457`), continuing `emb128x64_seedA` from
epoch 6 toward the epoch 9 chunk boundary, exposed a second, independent
orchestration bug on top of the one below. Root cause: NH's `continue_run`
unconditionally nests output into a new
`continue_training_from_epoch{start:03d}/` subdirectory on every call, but
the original chunk-continuation code treated the overlay's `epochs:` key as
an absolute target epoch rather than additive relative to the resumed
checkpoint. Told `epochs: 9` while resuming from checkpoint 6, NH trained 9
*more* epochs (additive 6+9=15), leaving
`continue_training_from_epoch006/model_epoch007.pt`-`model_epoch015.pt` on
disk with no valid epoch-9 screening result.

Fix (local only, not yet re-run on Moriah), entirely within
`src/baseline/pilot_orchestration.py`: `TrainChunkRequest` now separates
`current_epoch` (checkpoint resumed from), `additional_epochs` (additive
count for this chunk), and `logical_target_epoch`
(`current_epoch + additional_epochs`); a new
`discover_physical_checkpoints()` recursively inventories checkpoints
across the base run directory and arbitrarily-nested continuation
directories, failing loudly on any duplicate epoch claim;
`resolve_trusted_chunk_checkpoint()`/`untrusted_overshoot_epochs()`
distinguish a checkpoint this pilot's own chunk sequence produced from one
that merely exists on disk at the right epoch under untrusted
circumstances; the shared `_advance_chunk_via_continuation()` helper
resumes from a trusted checkpoint idempotently, blocks with a
manual-review reason if untrusted checkpoints already occupy the target
range, or blocks with an already-exists reason if NH's target continuation
directory exists but is empty/incomplete — never silently retraining over
existing files or guessing which checkpoint is authoritative;
`compute_pilot_status_fields()` now reports four distinct fields
(`highest_physical_checkpoint_epoch`, `highest_screened_epoch`,
`next_intended_screening_epoch`, `overshoot_epochs`,
`safe_to_continue_automatically`) consumed identically by the Slurm
launcher and evidence bundle. Applied to the exact real job-45705457
evidence, the corrected orchestration safely recovers and screens exactly
epoch 9 (no retraining, no manual checkpoint movement required) and halts
any further automatic continuation past epoch 9 with a blocked status
rather than resuming from the wrong checkpoint or retraining over the
preserved 10-15 checkpoints. No scientific
hyperparameter, split, screening-membership, or early-stopping policy
changed. Eight pilot test files now carry 146 tests (was 125), all passing
— including 6 new low-level checkpoint-discovery/trust unit tests, 2 new
direct unit tests of `_continuation_overlay` (the overlay-dict helper
extracted from `default_train_chunk` during self-review, since that
function had no direct test coverage of its own before), and a full
end-to-end reproduction of the real job-45705457 evidence shape; full
suite re-run: 1173 passed, same 6 pre-existing `neuralhydrology`/`torch`
import-only collection errors as before (expected in this local
environment), 1 Windows file-lock flake in an unrelated, untouched
package-builder test confirmed to pass in isolation — zero regressions
attributable to this work. One residual risk flagged but not resolved (the
`continue_from_epoch` NH-Config-property claim is unverified locally; see
`docs/decision_log.md`'s 2026-07-30 entry). Full detail: `docs/stage1_lead06_pilot_v001.md`'s
"Second Moriah failure and continuation-nesting/epoch-semantics correction"
section and `docs/decision_log.md`.

**Current status: epoch-9 recovery is safe; further training is not.**
`emb128x64_seedA` remains paused after epoch 6. **Superseded:** see the
"`emb128x64_seedA` candidate complete" entry above (job `45722908`) —
`emb128x64_seedA` is now complete, epoch 6→15 was adopted without
retraining, and early stopping fired at epoch 15.
`continue_training_from_epoch006/model_epoch009.pt` sits in exactly the
directory this pilot's own chunk sequence would produce, so it is trusted:
one controlled recovery invocation of the corrected orchestration reuses
that checkpoint, screens epoch 9, and records the event — no retraining,
no manual checkpoint movement/deletion required first. Continuing training
past epoch 9 is not yet safe while checkpoints 10-15 remain in the
existing continuation layout — those stay preserved, untouched,
scientifically-unused artifacts, and `safe_to_continue_automatically=False`
blocks any further 9→12 chunk attempt pending a later decision. This
epoch-9 recovery has not been executed on Moriah; it is expected behavior
of the locally tested repair only, and no resume has been submitted since
this correction. The other five pilot runs have not started.

## Stage 1 lead-6 optimization pilot — qualification run paused, orchestration corrected (2026-07-29)

The pilot's first real Moriah job, `emb128x64_seedA` (Slurm job 45695059),
trained successfully through epoch 6 (checkpoints + optimizer states 1-6
intact, peak RSS ~96.4GB) but then failed post-training with
`NHSeedEvaluationError: missing validation results pickle`. Root cause:
NH's in-training `validate_every: 3` validation does not reliably persist
`validation/model_epochNNN/validation_results.p`, but
`pilot_orchestration.py` assumed it always did. Confirmed by a separate
evaluation-only job (45698612) that explicitly produced both epoch-3 and
epoch-6 result pickles (400 basins each, ~84.6MB each; 11:34 elapsed,
~1.96GB peak RSS on an L40S — a single observation, not a general resource
requirement).

Fix (local only, not yet re-run on Moriah): `pilot_orchestration.py` now
calls a new `ensure_validation_results()` before every screening
checkpoint, which reuses an existing result pickle unchanged or explicitly
invokes NH evaluation (`default_evaluate_checkpoint`, mirroring
`scripts/run_stage1_nh.py`'s `eval` subcommand) via an injectable
`evaluate_checkpoint_fn` seam, then fails loudly if the pickle still isn't
produced. `nh_seed_evaluation.period_results_path()` is now the single
canonical helper for the result-pickle path. No scientific hyperparameter,
split, screening-membership, or early-stopping policy changed. Eight pilot
test files now carry 125 tests (was 95, then 124, then 125 after a
pre-commit adversarial review added one more focused test for a
repeated-call logging-handler leak found and fixed during that review),
all passing; full suite re-run after the adversarial review: 1155 passed,
0 failed, same 6 pre-existing `neuralhydrology`/`torch` import-only
collection errors as before (expected in this local environment) -- zero
regressions attributable to this work; the 2 Windows file-lock flakes in
unrelated, untouched package-builder tests seen in the prior run did not
reproduce this time (load-dependent race, not evidence of a fix). Full
detail:
`docs/stage1_lead06_pilot_v001.md`'s "Moriah workflow-qualification run and
orchestration correction" section and `docs/decision_log.md`'s adversarial-
review entry.

**Current status: not complete.** `emb128x64_seedA` remains paused after
epoch 6 pending a resumed Moriah job with the corrected orchestration; no
resume has been submitted. The other five pilot runs have not started.
**Superseded:** see the "`emb128x64_seedA` candidate complete" entry above
(job `45722908`) — `emb128x64_seedA` is now complete, epoch 6→15 was
adopted without retraining, and early stopping fired at epoch 15.

## Stage 1 lead-6 optimization pilot — implementation and tests complete (2026-07-27)

Local implementation-only increment, opened immediately after the
validation-and-optimization foundation phase below. **No Moriah job has
been submitted, no training has been run, and no temporal-test or
spatial-holdout data has been accessed.** Full documentation:
`docs/stage1_lead06_pilot_v001.md`; machine-readable run matrix:
`config/stage1_lead06_pilot_v001.yaml`.

Delivered: a closed six-run pilot matrix (raw static concatenation vs. a
learned FC static embedding in three shapes `[128,64]`/`[64]`/`[128]`,
crossed with two seeds — historical Seed A 967139 recovered read-only from
the frozen seed run's own config, and Seed B 1729) built entirely on the
seed run's frozen hyperparameters, with no seq_length/batch_size/lr/
scheduler/hidden_size/dropout variation and no EA-LSTM or automated sweep;
a pilot-specific 36-epoch early-stopping sub-cap layered on top of the
unmodified, still-binding general early-stopping policy (`max_epoch_budget`
40) via `src/baseline/pilot_early_stopping.py`, restart-safe with
idempotent-replay/contradictory-replay handling; a provisional-screening-
subset evaluation interface with epoch-3-diagnostic / epoch-6-stopping-
eligible cadence classification and sealed-population rejection; a
full-population validation readiness interface (not yet exercised against
a real run); an optional, disabled-by-default W&B tracking wrapper with a
confirmed-safe failure-downgrade path; a compact, checkpoint-byte-free
evidence bundle writer; a bounded-chunk restart-safe orchestration driver;
a CLI wrapper and a Slurm sbatch launcher (both prepared, neither
submitted/executed — the launcher's correctness was checked only by static
text inspection and a `bash -n` syntax check). Eight new focused test
files, 95 tests, all passing; full pre-existing repository suite re-run
alongside them with zero regressions attributable to this work (3
Windows file-lock flakes in unrelated, untouched package-builder tests
confirmed to pass in isolation; 6 pre-existing `neuralhydrology`/`torch`
import-only collection errors, expected in this local environment).

**Not done in this increment:** no Moriah connection, no Slurm submission,
no training, no full-population evaluation, no temporal-test or
spatial-holdout access, no change to the certified Compact Scientific
Package or canonical split membership, no screening-subset regeneration,
no hydrograph atlas, no automated sweep, no EA-LSTM work, nothing generated
by this pilot committed. The first job this pilot will eventually run is
`emb128x64_seedA` (the workflow-qualification run) — prepared, not
launched.

## Stage 1 validation and optimization foundation — Parts A-K complete (2026-07-26)

Design/tooling/documentation foundation phase, opened immediately after the
seed-run closure below. **No training run was launched in this phase.**
Full index: `docs/stage1_validation_optimization_foundation.md`; evidence:
`reports/stage1_validation_optimization_foundation_v001/` (untracked).

Delivered: seed percentile-diagnostic closure — the center of the NSE
distribution (p25-p99) was effectively flat across all 11 epochs, while
lower-tail percentiles (p1/p5) were unstable and non-monotonic; supports the
adopted early-stopping policy without yet justifying making it more
aggressive from one raw-static run (Part A); confirmation that the seed run
used raw static-attribute concatenation, not a learned embedding (Part B,
static-pathway audit); a **deterministic provisional hydrograph-atlas
selection v001** — selection/event-design tooling complete, 24-basin
realization balanced by skill stratum/area class/east-west geography; the
final observed-vs-predicted atlas is not yet generated and the realization
may be revised when it is, without reopening the selection framework (Part
C); a **provisional operational screening subset v001** — 400 basins,
deterministic, reproducible, multiply stratified, tracking the full
2,307-basin population well across all 11 seed-run epochs (Spearman 0.90,
Kendall 0.82, max abs. diff 0.0053); accepted for operational use, not yet
permanently frozen or scientifically authoritative; full population remains
authoritative for final run/checkpoint/architecture/hyperparameter
selection; prospective subset-vs-full comparison planned over the first ~3-5
materially different future optimization runs (Part D); an early-stopping
policy engine, implemented and tested, with real training-orchestration
integration still pending — reviewed after ~3-5 future optimization runs,
not tightened now (Part E, `src/baseline/early_stopping.py`); an optional,
disabled-by-default W&B tracking wrapper, implemented and tested, with
integration into a real training/validation harness still pending — future
runs are not yet automatically logged (Part F,
`src/baseline/wandb_tracking.py`); evidence-grounded next-run operational
defaults (Part G); an architecture-strategy status note declaring no winner
(Part H); a first embedded-static CudaLSTM candidate profile,
`embedded_static_cudalstm_pilot` — a **structural-smoke-only** config
profile; `[128, 64]`/tanh/dropout 0.1 are untuned construction values;
embedding dimension, depth, activation, and dropout remain future
architecture-specific hyperparameters; based deliberately on compact-smoke
settings; not authorized for full-population scientific training; the first
real full-population embedded-static candidates will be created in the
optimization phase, not now (Part I, extends
`src/baseline/nh_config_generation.py`); disposition of two retained
diagnostic utility scripts, left untracked pending dedicated tests (Part J,
revised 2026-07-27); and full test verification (Part K: 189 new/extended
focused tests, full-repository regression suite effectively 1094/1094
passing).

**Commit-readiness pass and final status resolution (2026-07-27):** an
epoch-7 vs. epoch-9 anchor-epoch sensitivity check for Parts C/D's skill
stratification found exact basin membership is sensitive to the anchor
checkpoint (75.8% stratum retention, 11.4%/6.7% basin-overlap Jaccard) —
expected given the selection design's many small composite strata and
seeded within-cell draws. This does not invalidate either artifact's
operational purpose; both are reclassified as provisional (above) rather
than settled, and the epoch-9 candidates are retained unchanged (no
regeneration, no replacement, no new skill definition). See
`docs/stage1_validation_optimization_foundation.md` ("Commit-readiness
pass" section), `docs/decision_log.md` (2026-07-27 final status resolution
entry), and
`reports/stage1_validation_optimization_foundation_v001/commit_readiness_epoch7_epoch9_sensitivity/`.

**Not done in this phase** (unchanged, next phase's scope): no hyperparameter
sweep, no embedded-static or EA-LSTM training run, no temporal-test or
spatial-holdout evaluation, no change to the certified Compact Scientific
Package or canonical basin splits.

## Stage 1 full-population seed training run — CLOSED (2026-07-25 training / 2026-07-26 evidence closure)

**Scope.** First full-population CudaLSTM training run on the certified 2,307
development-training basins (Gate 4 package, non-California). Target
`qobs_mm_per_h_lead06`, `seq_length: 24`. Training ran epochs 1–11 (epochs
1–3 as the initial job, epochs 4–11 as a resumed continuation after two OOM
kills — see `docs/decision_log.md` Decisions 3–5) and was then stopped
(clean cancellation at the epoch-11 checkpoint, not a crash). A complete
raw-space (m³/s) development-validation evaluation (calendar year 2024) was
run for **all 11 checkpoints**: 2,307 basins evaluated, 0 area-excluded,
**19,747,262 admitted samples for every single checkpoint** (identical
denominator across all 11 epochs — confirms no silent sample-set drift
between checkpoints). **No temporal-test or spatial-holdout data was
accessed at any point.** Full evidence bundle:
`reports/seed_validation_review_v001/` (local, untracked; see
`docs/decision_log.md` for the closure decisions and checksums).

**Checkpoint comparison result: a broad validation plateau, no scientifically
clear winning epoch.** Per-basin raw-space NSE (development validation,
2024):
- Epoch **7** has the maximum **median** per-basin NSE (≈0.2401) — adopted
  as the primary run-level selection statistic (see decision below).
- Epoch **6** has the maximum **pooled** (sample-concatenated) NSE (≈0.4651).
- Epoch **9** has the least-negative **mean** per-basin NSE, but mean NSE is
  dominated by a handful of extreme negative outliers and is not used as the
  primary selection criterion.
- All three aggregation methods disagree on the top epoch.
- Roughly 78–81% of basins have NSE > 0, ~12–13% have NSE > 0.5, ~19–22%
  have NSE < 0, essentially flat across all 11 epochs with no improving
  trend — consistent with training having already plateaued well before the
  epoch-11 cancellation.

**Decisions adopted at closure (full text: `docs/decision_log.md`,
2026-07-26 entries):**
1. **Median per-basin raw-space NSE** on development validation is the
   primary run/checkpoint-selection metric going forward.
2. Mean per-basin NSE and pooled NSE are retained as **diagnostics**, not
   primary selection metrics.
3. Future evaluations report the full distribution (p1/p5/p10/p25/p50/p75/
   p90/p95/p99) plus NSE sign fractions, not just a single summary number.
4. Early-stopping policy for future runs: save every epoch; no stop before
   epoch 6; official validation every 2–3 epochs; minimum meaningful
   improvement 0.005 median NSE; patience 3 validation events; max 30–40
   epochs; best checkpoint retained; temporal-test/spatial-holdout data is
   never used for stopping or selection.
5. Test sets (temporal-test, spatial-holdout) remain sealed during
   optimization.
6. **This seed run is a successful pipeline proof and an initial
   optimization baseline — it is not a tuned model and not the official
   Stage 1 benchmark.**
7. Epoch 7 may be recorded as the deterministic representative of the
   plateau under the adopted median-NSE rule, but is **not** scientifically
   meaningfully superior to the nearby checkpoints (6, 9, 10 are all within
   noise on median NSE).

**Operational findings** (full detail: `docs/stage1_neuralhydrology_preflight.md`):
raising the training job to `--cpus-per-task=16`/`num_workers=12` (from
`8`/`4`) produced no measurable training or validation speedup (validation
held a flat ~2.1 basins/sec before and after) while pushing peak memory to
~223 GiB against a 224G allocation — **this configuration must not become
the default** for future Stage 1 seed runs. The epoch-3→epoch-4 resume
restored model weights and full Adam optimizer state correctly but reseeds
RNG to the fixed `cfg.seed` rather than continuing the pre-interruption
stream (NeuralHydrology has no dataloader shuffle-state serialization) — the
continuation is scientifically valid but not bitwise-equivalent to an
uninterrupted run.

**Not established by this run:** final hyperparameters, sequence length,
lead time, the full architecture (in particular a learned static
representation, not yet designed), temporal-test or spatial-holdout
performance, or W&B tracking. **Next phase: "Stage 1 validation and
optimization foundation"** — percentile diagnostics, a deterministic
hydrograph atlas, a learned static representation, a screening validation
subset, W&B tracking, and the first embedded-static CudaLSTM candidate.

## Commit-readiness review of the full-population NH config-generation increment — two safeguard fixes (2026-07-24)

Review-only pass (no transfer/Moriah/Slurm/training) over the increment directly below. Found and
fixed two gaps in distinguishing the `spatial_holdout` bundle from a trainable experiment (a
custom marker cannot live inside `config.yaml` itself — NH 1.13's `Config._check_cfg_keys` rejects
unrecognized keys): (1) the holdout bundle's default `experiment_name` now gets a
`_spatial_holdout_test_only_eval` suffix instead of colliding with the development bundle's name;
(2) `write_generated_config` now writes a sibling `TEST_ONLY_DO_NOT_TRAIN.txt` for the holdout
bundle only, and `check_generated_config_structure`'s holdout-role path requires it to exist. Added
9 tests (full-population totals now 14 + 15 = 29 passing). Re-ran the synthetic dry run and this
time directly read the generated `config.yaml`/marker/basin-list files; ran the preflight CLI
end-to-end (`--skip-dataset-construction`) — PASS, 56 OK, 0 errors. Full suite: 889 passed, 0 failed
(known Windows `os.rename` flake in `test_package_builder.py` did not reproduce this run; separately
re-confirmed passing in isolation, unrelated to this patch). Nothing committed. Full detail:
`docs/decision_log.md` (2026-07-24 "Commit-readiness review..." entry).

## Full-population (development + spatial-holdout) NH config-generation + structural-preflight local implementation increment (2026-07-24)

**Local-only** (no h2o/Moriah access, no package rebuild, no training). Extends the 2026-07-22
compact-package config-generation/structural-preflight machinery (below) to the certified full
non-California package (Gate 4, below: 2,307 development-training + 250 spatial-holdout basins).
Renders the single lead06/seq24 scientific configuration as **two strictly separated bundles**:
a `development` bundle (train == validation == temporal-test, the 2,307 development basins across
different date periods) and a test-only `spatial_holdout` bundle (train/validation lists are the
development population, never a holdout basin; test list is the 250 holdout basins). New basin-
membership validation requires the package's basin set to equal exactly the
`development_train` ∪ `spatial_holdout_nonca` union (no California, no overlap, no missing/extra,
exact 2,307/250 counts). New `src/baseline/nh_structural_preflight.py` check
(`check_flashnh_external_scaler_test_construction`) constructs only the holdout `test`-period
dataset, reusing the development-fitted scaler unchanged and never touching the holdout config's
`train_dir`. Two new CLIs: `scripts/generate_stage1_full_population_nh_config.py` and
`scripts/check_stage1_full_population_nh_config_preflight.py` (package root always supplied via
`--package-root`, never hard-coded). 20 new tests passing (10 in `tests/test_nh_full_population_
config_generation.py`, 10 in `tests/test_nh_full_population_structural_preflight.py`); all 41
pre-existing compact-package tests unaffected. Verified via a local dry run against a synthetic
2,557-basin fake package (matching the real split union): generator produced the expected
development/holdout basin-count contracts; preflight (`--skip-dataset-construction`) reported PASS,
55 OK checks, 0 errors. **Not done:** package transfer, real Moriah Slurm preflight, real
full-population dataset loading, training, or the remaining 15 lead × sequence-length
configurations. Full detail: `docs/decision_log.md` (2026-07-24 entry, same title).

## Gate 4 — Full non-California Scientific Package (v002) independently audited — PASS (2026-07-24)

The Gate 4 independent auditor (`src/baseline/package_audit.py`, commit
`98b7d42f23963e76e02ad3991d7298d3ada98ee3`) reran for real on h2o, in full mode, against
`/data42/omrip/Flash-NH/tmp/stage1_scientific_package_v002` (2,307 development-training + 250
spatial-holdout non-California basins, build commit `61d3819deb55240652276765c6a96d12ed3ce539`).

**Result: PASS — 0 errors, 1 warning, 260,870 OK checks.** Audit output:
`/data42/omrip/Flash-NH/tmp/stage1_scientific_package_v002_gate4_audit/full_rerun_20260724T110557Z`.
Evidence archive SHA-256 `9cc9f8e63d6c9825c2bf765106a20a58ce0560a1d733bc815ec0846f02071ed0`,
transferred locally and independently verified (checksums, arithmetic tally, provenance).

The single warning, `imputed_value_mask_basin_order`, is non-blocking: exact mask basin-index
membership passed with zero missing/extra basins (`imputed_value_mask_basin_membership`: OK); only
row order differs, and every downstream imputation-placement check re-indexes the mask by basin
label, so order alone cannot affect correctness.

This rerun follows and fixes an earlier FAILED full audit (errors=9) of the same package. The
auditor patch (same commit) introduced: (1) a 1-float32-ULP tolerance for the non-authoritative
QC-CSV-versus-NetCDF finite comparison only, absorbing a confirmed xarray/netcdf4 write-path
rounding artifact; (2) separate strict mask-membership (ERROR) and non-blocking mask-order
(WARNING) checks, replacing a single check that conflated the two.

**The package itself was not rebuilt** (`build_git_commit` unchanged), and no static artifact or
other source input was modified. No Moriah transfer, no NeuralHydrology configuration generation,
and no training occurred. **This closes the production package build-and-independent-audit phase
for `stage1_scientific_package_v002`; it does not establish scientific model skill.** Full detail:
`docs/decision_log.md` (2026-07-24 Gate 4 entry).

## Full non-California static-attribute preparation — real h2o run PASS (2026-07-24)

Real (not synthetic/dry-run) execution of `scripts/prepare_stage1_full_static_attributes.py`
on h2o against the canonical `stage1_static_attributes_v002` matrix. Development-only median
imputation (fit on the 2,307 development-training basins only) followed by a development-only
exact post-imputation zero-variance trainability projection, both frozen and applied unchanged
to the full 2,557-basin non-California package population (2,307 development-training + 250
spatial holdout). The spatial holdout did not influence either fit.

**Result: 473 candidate `model_input` columns → 473 retained, 0 excluded.** All 473 canonical
static model-input columns will be used by the first full-population Stage 1 model — the
canonical package contract is unchanged at 473 columns. No column was entirely missing in the
2,307-basin fit population, and zero missing values remain anywhere after imputation. Because 0
columns were excluded, the retained static table is byte-identical to the imputed static table.
The 32-basin compact-smoke 13-column zero-variance exclusion (2026-07-23 "Compact
NeuralHydrology integration smoke" Finding 1, below) is compact-population-specific historical
evidence only and was confirmed **not** reused, inherited, or reopened by this run.

Output (h2o-resident, generated evidence, **not committed**):
`/data42/omrip/Flash-NH/tmp/stage1_full_static_attributes_v001/` (`imputed_static_attributes.parquet`,
`imputed_value_mask.parquet`, `retained_static_attributes.parquet`, manifests, column lists,
`run_summary.json`).

Key checksums (SHA-256):
```
input matrix (stage1_static_attributes_v002.parquet):
4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297

imputed_static_attributes.parquet / retained_static_attributes.parquet (identical):
5be00a3b068351bffd40a3cf72991a3df888700034831123c91823b8bd4b6e24
```

Full counts, per-artifact checksums, and the modeling decision: `docs/decision_log.md`
(2026-07-24 entry). **No NetCDF package was built by this run.** Next step: full 2,557-basin
v002 package build and independent audit — see `docs/stage1_baseline_package_implementation_plan.md`.

## Development-population zero-variance trainability projection — mechanism implementation (2026-07-23)

Added a reusable fit/apply mechanism in `src/baseline/static_preparation.py`
(`ZeroVarianceFit`, `fit_zero_variance_projection`,
`apply_zero_variance_projection`, `build_zero_variance_manifest`,
`write_zero_variance_manifest`) that identifies static `model_input` columns
with exactly zero variance over the Stage 1 development-training population
(2,307 basins) **after** development-only median imputation
(`fit_development_median_imputation`, above/below). This is a **run-specific
trainability projection, not a package-schema change**: the canonical static
matrix and Compact/full package contract remain **473 `model_input`
columns**, unmodified. The fit uses exact post-imputation constancy (no
near-zero-variance threshold), is fitted once on the 2,307-basin
development-training population, and its frozen retained/excluded column
list is meant to be applied unchanged — never recomputed — to validation,
temporal-test, and spatial-holdout populations. The compact-smoke 13-column
exclusion list (`docs/decision_log.md` "Finding 1", 2026-07-23) is historical
evidence for the 32-basin smoke population only and is explicitly not
reused, imported, or asserted here. **This patch implements the mechanism
only: it does not compute the real 2,307-basin excluded-column list (no
h2o access), does not build a package, and does not generate NeuralHydrology
configs.** 18 focused tests added in `tests/test_static_preparation.py`
(52/52 passing). Full detail: `docs/decision_log.md` (2026-07-23 entry).

## Versioned package schema (`date`) for future scientific packages (2026-07-23)

**Schema-support implementation addendum** (code, tests, and documentation
together — not a docs-only change). Added an explicit,
versioned NetCDF package-schema registry (`src/baseline/package_netcdf.py`)
so future full scientific packages can use temporal coordinate `date`
(`stage1_scientific_package_v002`, version 2) while the certified compact
package below **remains exactly as built and certified: frozen, on disk,
with coordinate `time`** (`stage1_compact_scientific_package_v001`, version
1, unchanged default at the low-level serializer). The package-builder CLI
now requires an explicit `--package-schema` choice — no default, no
inference from basin count/path/output name — so a future production build
cannot silently emit a legacy `time` package by omission. Provenance now
records both the (corrected, deprecated-but-preserved)
`package_schema_name` builder-manifest identity and five new explicit
fields (`builder_manifest_schema_name`, `builder_manifest_schema_version`,
`netcdf_package_schema_name`, `netcdf_package_schema_version`,
`netcdf_time_coordinate`). The independent auditor
(`src/baseline/package_audit.py`) now checks the declared/actual NetCDF
schema and coordinate from disk without importing the schema registry it
audits. `FlashNHDataset`'s `time`→`date` compatibility adapter
(`src/baseline/nh_dataset.py`) is renamed `_adapt_temporal_index_to_date`
and now handles all four coordinate-presence combinations explicitly
(pass through `date`; rename `time`; fail loudly on both or neither, in
either direction). Structural NH compatibility with a `date`-coordinate
package does not imply stock `GenericDataset` reproduces Flash-NH's own
sample-validity filtering — `FlashNHDataset` remains required either way.
**No real package was built by this patch; h2o/Moriah were not accessed;
the certified compact v001 package was not touched.** Full detail:
`docs/decision_log.md` (2026-07-23 "Versioned package schema" entry) and
`docs/stage1_compact_package_independent_audit.md` (2026-07-23 addendum).

---

## Current milestone

**Compact NeuralHydrology integration smoke — CLOSED (CPU preflight + GPU
training + explicit validation/test evaluation all PASS, 2026-07-23).** This
closes the compact-package NH integration-validation effort opened by the
2026-07-22 config-generation/structural-preflight increment (below). Three
Moriah Slurm jobs, run in sequence against the certified 32-basin Compact
Scientific Package (Gate 4, below), all passed:

- **CPU structural preflight — job `45624926`** (Moriah `glacier`
  partition/CPU node class). 39 checks OK, 0 warnings, 0 errors. Real
  `FlashNHDataset` construction succeeded for train, validation, and test;
  the training scaler was finite; validation and test reused the training
  scaler unchanged; every admitted sample inspected by the preflight was
  finite. Admitted sample counts: train 851,339; validation 274,347; test
  263,637.
- **GPU training smoke — job `45625002`** (Moriah `catfish` partition,
  NVIDIA L4). Target `qobs_mm_per_h_lead06`, sequence length 24, 32 basins,
  2 epochs, 460 static inputs (see exclusion note below). Epoch 1 average
  loss 0.40205; epoch 2 average loss 0.38727. Run directory:
  `/sci/labs/efratmorin/omripo/Flash-NH/runs/stage1_nh_config_lead06_seq24_v001/runs/stage1_compact_lead06_seq24_v001_2307_135829`.
  Retained artifacts: `config.yml`, `model_epoch001.pt`, `model_epoch002.pt`,
  `optimizer_state_epoch001.pt`, `optimizer_state_epoch002.pt`,
  `train_data/train_data_scaler.yml`, `output.log`, TensorBoard event file.
- **Explicit validation + test evaluation — job `45625077`.** Evaluated the
  epoch-2 checkpoint from the run above: validation period calendar-year
  2024, test period calendar-year 2025. Evaluation audit: 217 OK, 0
  warnings, 0 errors. Metrics produced: NSE, RMSE, KGE, Pearson-r, Beta-KGE.
  Retained outputs: `validation/model_epoch002/validation_metrics.csv`,
  `validation/model_epoch002/validation_results.p`,
  `test/model_epoch002/test_metrics.csv`,
  `test/model_epoch002/test_results.p`. Metric values are not interpreted
  scientifically here — this was an integration smoke, not a tuned or
  reportable baseline experiment.

**What this proves.** The Stage 1 package-to-NeuralHydrology pipeline can,
end to end: (1) construct real datasets from the certified Compact
Scientific Package; (2) apply filtering and reuse a single training scaler
across periods without leakage; (3) train on Moriah GPU; (4) save and
reload checkpoints; (5) evaluate held-out validation and test periods; (6)
retain metrics and prediction artifacts on disk.

**What this does not prove.** Final model skill; final hyperparameters;
final sequence length; final static-feature set for the full basin
population; final production-package temporal-coordinate convention;
performance at lead 1 h, 3 h, or 12 h; any spatial-holdout or
full-population scientific conclusion. This closure is an integration gate,
not a scientific result, and is not itself grounds to begin ad hoc
hyperparameter tuning.

**Two findings recorded, not resolved, by this closure** — see
`docs/decision_log.md` (2026-07-23 entry) for full detail:

1. *Compact-smoke-only zero-variance static exclusion.* Across the 32-basin
   smoke population only (not the full package), 13 static attributes had
   zero standard deviation and were excluded for this smoke only (460 of 473
   used): `CANALS_MAINSTEM_PCT`, `CDL_DURUM_WHEAT`, `CDL_ORANGES`,
   `CDL_RICE`, `HGBC`, `PCT_6TH_ORDER_OR_MORE`, `glc_pc_u01`, `glc_pc_u18`,
   `pnv_pc_u02`, `wet_pc_u02`, `wet_pc_u03`, `wet_pc_u07`, `wet_pc_u09`. The
   full 473-column Compact Scientific Package remains authoritative and
   unchanged; this exclusion list must **not** be inherited automatically by
   the full-population baseline, which must independently identify
   zero-variance columns over its own training population.
2. *`time` vs. `date` temporal-coordinate adapter.* The certified compact
   v001 NetCDFs use dimension/coordinate name `time`; NeuralHydrology 1.13
   requires `date` internally. `FlashNHDataset` applies an in-memory
   index-name-only adapter (`src/baseline/nh_dataset.py`); on-disk v001
   files are unchanged. The final on-disk temporal-coordinate convention
   must still be explicitly resolved before the production package format is
   frozen; any such change belongs in a new package version, not a silent
   rewrite of v001.

**Next phase.** Planning the first scientifically meaningful Stage 1
baseline experiments (hyperparameters, sequence-length/lead sweep, the
static-feature set for the full population, spatial-holdout evaluation) —
not implied or started by this closure.

---

### Predecessor: NH config-generation + structural-preflight local implementation increment (2026-07-22)

> **Superseded (2026-07-23):** the "no training" scope below describes
> accurately what this specific increment did. Real dataset construction,
> GPU training, and explicit validation/test evaluation have since run and
> passed — see "Current milestone" above.

**NH config-generation + structural-preflight local implementation increment
(2026-07-22, local-only, no h2o/Moriah access, no training).** Following
Gate 4 certification (below), this increment implements the first local
foundation for compact-package NH integration-validation, strictly scoped
to the first configuration only (lead 6 h, sequence length 24 h, target
`qobs_mm_per_h_lead06`, 8 approved dynamic inputs in binding order, 473
static `model_input` attributes, temporal split train 2020-10-14→2023-12-31
/ validation 2024 / test 2025, same 32 certified compact basins in all three
periods). Scope: `src/baseline/nh_config_generation.py` (config rendering +
basin/date/static-list contracts), `src/baseline/nh_structural_preflight.py`
(two-layer preflight: Layer 1 file-only structural checks against a
generated config bundle; Layer 2 real `FlashNHDataset` construction —
train/validation/test — against synthetic fixtures only, never the real
package), `scripts/generate_stage1_nh_config.py`,
`scripts/check_stage1_nh_config_preflight.py`. Test coverage: 38 tests
passing (25 in `tests/test_nh_config_generation.py`, 13 in
`tests/test_nh_structural_preflight.py`), plus the pre-existing
`tests/test_nh_dataset.py` suite unaffected. No h2o/Moriah access, no data
transfer, no NH training, no Slurm job, no W&B, and only this single
configuration was rendered — not the full 16-config matrix. The certified
Compact Scientific Package itself was not modified or rebuilt.

**Notable discovery: NeuralHydrology 1.13 upstream mutable-default-argument
scaler bug.** `neuralhydrology.datasetzoo.basedataset.BaseDataset.__init__`
declares `scaler: Dict[...] = {}` as a mutable default argument, shared by
Python across every call site in a process that omits `scaler=`. A second
train-period `get_dataset(..., is_train=True, ...)` call in the same
process (no explicit `scaler=`) inherits the first call's already-populated
dict, so its own `not scaler` check is False, `_setup_normalization` is
skipped, and NH silently reuses a stale, unrelated scaler — whose
intersecting xarray arithmetic (`xr - center`) can silently drop dynamic
input/target columns absent from that stale scaler. This only manifests
when a single Python process constructs more than one train-period NH
dataset without passing `scaler={}` explicitly (e.g. a shared pytest
session, or interactive/dev usage) — a real training job (one Slurm
process, one train-dataset construction) is unaffected. Fix applied:
`scaler={}` is now passed explicitly at every train-period `get_dataset`
call site in `nh_structural_preflight.py` and `tests/test_nh_dataset.py`.
This is an NH-mechanics/dev-tooling finding, not a Stage 1 scientific
decision.

**Known documentation debt (not yet resolved):** the committed policy
config declares `nh.dataset: generic`, while the task-mandated / generated
config's `build_nh_config_mapping` hardcodes `dataset: flashnh`. Both are
intentional for their respective purposes (the policy YAML documents the
underlying NH dataset family; the generated config selects the registered
`FlashNHDataset` class) but the discrepancy is not yet called out in-line
in either file and should be reconciled or explicitly annotated in a future
increment.

---

## Compact Scientific Package — Gate 4 certification (2026-07-22)

**Compact Scientific Package — built and independently certified
(2026-07-22).** The 32-basin Compact Scientific Package (built via
`scripts/build_stage1_baseline_nh_package.py`, commit
`89c4dd162f7043419b4b227de5c2bc1b3b230da6`) has been built and promoted on
h2o at `/data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001`
(non-authoritative QC evidence at `..._v001_evidence`, run logs at
`..._v001_run_logs`). Builder-level self-validation and an independent
ChatGPT inspection of its compact review bundle are complete. Package facts:
32 per-basin NetCDF files; 45,720 hourly rows/basin; period 2020-10-14
00:00 through 2025-12-31 23:00; 8 approved dynamic inputs; diagnostic raw
`qobs_m3s`; 4 lead targets (1/3/6/12 h); 473 static model-input columns; 138
global gap timestamps (136 MRMS + 2 RTMA); one 15-character basin ID
`393109104464500`.

**Gate 4 independent audit: PASS.** The genuinely independent auditor
(`src/baseline/package_audit.py`,
`scripts/audit_stage1_compact_scientific_package.py`,
`tests/test_package_audit.py`, `docs/stage1_compact_package_independent_audit.md`
— committed `4b524b3851b16baa080d4237622fa7da30e05cea`) was run for real on
h2o against the real package and real source artifacts, in full mode, at
`2026-07-22T08:58:52Z`. Result: **status PASS, 0 errors, 0 warnings, 3,114 OK
checks, exit code 0.** Audit output:
`/data42/omrip/Flash-NH/tmp/stage1_compact_scientific_package_v001_gate4_audit`.
The auditor re-derives every scientific/structural claim from raw
sources — it does not import
`package_builder`/`package_assembly`/`package_netcdf`/`units`/
`lead_targets`/`gap_mask_io`, so a shared bug cannot pass both the build and
the audit. The build commit (`89c4dd162...`) and the auditor commit
(`4b524b385...`) are intentionally distinct identities. The transferred
audit evidence bundle was independently reviewed by ChatGPT and found
internally consistent; the generated evidence files remain untracked and
are not committed to this repository.

**The package is built and independently certified.** NeuralHydrology
configuration generation is now unblocked. See
`docs/decision_log.md`'s 2026-07-22 certification entry and
`docs/stage1_compact_package_independent_audit.md`'s Status section for
full detail.

---

**Static-attribute matrix v002 — ACCEPTED as canonical Stage 1 baseline
(2026-07-20).** `stage1_static_attributes_v002` (source-checksum-verified
29/29 PASS build via `scripts/build_stage1_static_attribute_matrix.py`,
independently audited PASS by
`scripts/audit_stage1_static_attribute_matrix.py`) is accepted as the
canonical Stage 1 v001-core static-attribute matrix, superseding
`stage1_static_attributes_v001` for modeling. Canonical path:
`/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v002/`
(`stage1_static_attributes_v002.parquet`,
`stage1_static_attributes_v002_column_manifest.json`,
`stage1_static_attributes_v002_provenance.json`,
`stage1_static_attributes_v002_audit_summary.md`). Matrix: 2,843 rows × 523
total columns — **473 `model_input`** (authoritative, no longer provisional),
2 split-support, 4 diagnostic lat/lon, 12 diagnostic record/network/QA, 1
deferred-ambiguous (`lka_pc_use`), 29 categorical-deferred, 2 flag. Sentinel
decoding (`stage1_static_sentinel_decode_v1`) replaced 15,018 values across
the 12 mapped columns; the 8 infrastructure-distance `RAW_*` columns are
excluded via the pre-existing `>20%` high-missingness mechanism, not by
name; `PERHOR`/`STRAHLER_MAX` retained `model_input` with sentinels decoded;
`dor_pc_pva`/`dis_m3_pyr`/`run_mm_syr` retained unchanged; direct-coordinate,
record/network/QA, and `lka_pc_use` exclusions all verified; the HydroATLAS
5-basin gap is unchanged and explicitly handled. Independent audit: PASS, 0
errors, 0 warnings, 32 OK checks. Canonical matrix sha256
`4954a320d9e720dfaef29c05f77a505183e10bae4891cf06161958e17cdb2297`;
companion checksums (column manifest, provenance, audit summary) recorded in
`docs/decision_log.md` (2026-07-20 acceptance entry). The full 473-column
`model_input` list is not duplicated in prose — see the canonical column
manifest.

**Compact static-imputation v002 — ACCEPTED (2026-07-20).** Rebuilt against
the accepted v002 matrix via `scripts/prepare_stage1_compact_static_attributes.py`
(algorithm `stage1_static_median_imputation_v1`, unchanged primitive).
Canonical generated output path:
`/data42/omrip/Flash-NH/tmp/stage1_compact_static_imputation_v002`. Input
matrix checksum matches canonical v002 exactly. Output: 32 basins × 473
`model_input` columns; fit scope development-training-only (2,307-basin fit
population); applied to the 32 accepted compact basins; all fit columns had
valid medians; 168 total values imputed, all on a single basin
(`393109104464500`, the same designated compound-edge-case diagnostic basin
noted below); zero remaining NaNs. Output checksums (`imputed_static_attributes.parquet`,
`imputed_value_mask.parquet`) recorded in `docs/decision_log.md`.

**Both v001 artifacts (`stage1_static_attributes_v001` matrix and
`stage1_compact_static_imputation_v001`) remain preserved as historical
provenance — not deleted, not invalid — but are superseded for modeling.**
The selector and canonical split artifacts were **not** rerun; the accepted
32-basin Compact Scientific Package selection (below) remains valid as-is;
the static-imputation primitive (`src/baseline/static_preparation.py`) is
unchanged. No NH package has yet been built; no training has run. Full
detail: `docs/stage1_static_attribute_matrix_plan.md` §12,
`docs/stage1_scientific_baseline_design.md` §3, `docs/decision_log.md`
(2026-07-20 acceptance entry). **Next milestone: Compact Scientific Package
builder planning and implementation (not yet started).**

**Compact Scientific Package selection — ACCEPTED (2026-07-20).** The fully
enriched h2o run of `scripts/generate_stage1_compact_package_selection.py`
(selector commits `71467b5`, `65af017`) is accepted as the project's Compact
Scientific Package basin list. Enriched run inputs: canonical
`split_assignment.csv` (`development_train`, 2,307 basins), canonical
`stage1_static_attributes_v001` matrix + column-role manifest, canonical
`stage1_static_attributes_v001` full-period qobs/target-status table. All
acceptance checks PASS: count=32; development-pool membership PASS;
California exclusion PASS; spatial-holdout leakage PASS; qobs enrichment and
static missingness evaluated for all 32 selected basins; input/artifact
checksums PASS. Accepted characteristics: 13 distinct HUC02s, 7 macro-regions,
east/west split 19/13; area classes high/low/middle = 12/10/10; hydro classes
high/low/middle/missing = 10/11/10/1; qobs completeness high/mid/low =
15/16/1; static missingness none/high = 31/1. Two designated diagnostic
basins: `393109104464500` (compound edge case — `unusual_identifier` +
`hydro_stratifier_gap` + `static_missing_value_case`, 169 missing
`model_input` static attributes) and `05568800` (lowest qobs completeness in
the selection, coverage fraction ≈0.8746). Canonical evidence path on h2o:
`/data42/omrip/Flash-NH/tmp/stage1_compact_package_selection_v001_evidence`
(generated artifact; its `selection_manifest.json` correctly still reports
`"status": "candidate"` per the tool's own generated-artifact convention —
per policy, generated evidence is never hand-edited; **project-level
acceptance is recorded here and in `docs/decision_log.md` instead**). Full
32-basin ID list is not duplicated in this document — see
`compact_basin_ids.txt` in the evidence bundle above, or the local
split-based candidate run described in
`docs/stage1_compact_package_selection.md`. Building the 32-basin NH package
is a separate, not-yet-started step.

> **Superseded (2026-07-21):** the 32-basin NH package has since been built
> and promoted on h2o (build commit `89c4dd162f7043419b4b227de5c2bc1b3b230da6`);
> see the current-state block at the top of this document and
> `docs/stage1_compact_package_independent_audit.md`. It is not yet
> independently certified.
>
> **Further update (2026-07-22):** the package has since been independently
> certified — Gate 4 PASS, 3,114 OK / 0 errors / 0 warnings; see the
> current-state block at the top of this document.

**Scientific target-transformation + static-preparation primitives increment
(2026-07-20).** Reviewed existing code before writing anything new (reuse-first):
`src/baseline/units.py` (m³/s↔mm/h conversion) and `src/baseline/lead_targets.py`
(1/3/6/12 h lead-target shifting) already fully satisfy the discharge-transform
and lead-semantics requirements, with existing test coverage in
`tests/test_units.py`/`tests/test_lead_targets.py` — no new code needed there.
`src/baseline/validity_mask.py` already implements the history/boundary
validity split needed for forcing-gap awareness. Two genuine gaps were found
and filled: (1) `src/baseline/static_preparation.py` — development-train-only
median imputation for `model_input` static-attribute columns, per the
already-signed-off policy (`config/stage1_scientific_baseline_v001.yaml::static_attributes.imputation`,
`docs/stage1_baseline_package_implementation_plan.md` §15); (2)
`src/baseline/gap_mask_io.py` — a loader/writer converting the Milestone 2K-E
forcing-audit's missing-hour-product inventory into the flat
`masks/gap_timestamps.json` format `src/baseline/nh_dataset.py` already
expects but that no script previously produced. Neither changes any signed
scientific decision. **Not done in this increment:** no NH package built, no
`FlashNHDataset`/NH-registration/launcher changes, no training, no Moriah use,
no full 2,752-basin package. See `docs/decision_log.md` for full detail.

> **Superseded (2026-07-21):** the 32-basin NH package has since been built
> (see the current-state block at the top of this document); the
> `FlashNHDataset`/NH-registration/launcher/training/Moriah/full-population
> items in this sentence remain not done.

**2K-G-I I-A1-I-A5 (spatial/temporal split generation through canonical
promotion) COMPLETE (2026-07-16).** Seeded stratified split candidate
(I-A2) passed an independent auditor (I-A3: PASS, 0 errors,
`scripts/audit_stage1_baseline_splits.py`) and human visual QC (I-A4:
PASS, no clustering/imbalance found; see `docs/decision_log.md` for
accepted findings) and was byte-copy-promoted (I-A5) to the canonical
path **`config/stage1_baseline_splits_v001/`** (10 artifacts;
`development_train`/`validation`/`temporal_test` 2307 each,
`spatial_holdout_nonca` 250, `california_finetune_train` 176,
`california_holdout` 19). **The split design is now frozen; do not
reopen it absent a concrete scientific or correctness problem.** Next
work: baseline NH package-builder implementation (remaining 2K-G-I
checklist items below).

**2K-G-H Scientific Baseline Policy Sign-off COMPLETE (2026-07-12) —
docs-only.** Converts the 2K-G-G Phase B evidence (committed at `0d0e6aa`)
into binding Stage 1 decisions in `docs/stage1_scientific_baseline_design.md`.
No new evidence gathered in this patch; no code, config, Slurm script, or NH
package changed. Seven decisions recorded (full detail and rationale in the
design doc's "Binding decisions — Milestone 2K-G-H sign-off" section):
1. **Target scaling (§5):** area-normalized discharge, internal unit mm/h
   equivalent runoff depth, computed by the package builder at
   package-build time; package target column e.g. `qobs_mm_per_h_leadXX`.
   NH's native scaler inversion only returns to mm/h; official evaluation
   requires an additional Flash-NH-side mm/h→`m^3/s` conversion using basin
   area. NH loss/validation curves are training diagnostics in transformed
   space; official benchmark metrics are always Flash-NH-computed raw-space
   `m^3/s` after full inverse conversion.
2. **Target inversion/audit requirements (§5a, new):** deterministic
   `m^3/s -> mm/h -> m^3/s` round-trip unit tests; a package audit
   requirement (`qobs_mm_per_h_leadXX` at `t` == `qobs_m3s` at `t+XXh`
   converted to mm/h, on random basin/time samples); an evaluation audit
   requirement (raw-space metric scripts verify units, basin area, lead
   alignment, NaN masking, and conversion back to `m^3/s`).
3. **Lead-time implementation (§9b):** package-build-time target shifting
   (no native NH `lead_time` config, per Q9 evidence). All four lead times
   — 1/3/6/12 h — included in the first package/config/sweep design (not
   just 6 h/12 h); primary benchmark lead 6 h, secondary 12 h, 1 h/3 h
   diagnostics included now for low incremental cost. `seq_length` and lead
   time remain separate axes.
4. **Forcing-gap policy (§6):** scientific baseline hard-excludes training
   windows intersecting MRMS archive-gap hours (Policy B), accepted because
   corrected real-gap window loss is modest (~1.3% at `seq_length=12` to
   ~5.6% at `seq_length=72`). RTMA (2 gap hours vs. MRMS's 136) may be
   folded into the same exclusion mask if that's free, but MRMS drives the
   policy. `nan_handling_method` (Policy A) remains a fallback/ablation path
   only, not the baseline; unset/default `None` remains forbidden in any
   run per Q6 evidence.
5. **Static attributes (§3) — HISTORICAL, SUPERSEDED (see 2026-07-20 state at
   top of this document).** At the time of this 2K-G-H sign-off
   (2026-07-12), canonical `stage1_static_attributes_v001`
   (2,843 × 531 cols, 496 `model_input`, h2o canonical PASS 2026-07-08) was
   accepted as the Stage 1 baseline static matrix, replacing the earlier
   48-column merge. Numeric attributes pass through NH's standard
   static-attribute pathway; no categorical embeddings in this first
   baseline. `STATE`/`HUC02` remain split-support/diagnostics only;
   `LAT_GAGE`/`LNG_GAGE` remain diagnostic only, deferred to a later
   ablation. **This v001 matrix was superseded on 2026-07-20 by the accepted
   `stage1_static_attributes_v002` (2,843 × 523 cols, 473 `model_input`) —
   see the acceptance record at the top of this document and
   `docs/decision_log.md` for the current binding static-attribute state.**
6. **Spatial split and leakage (§8b):** reproducible seeded stratified
   non-CA spatial holdout (mechanism unchanged), stratifying on at least
   HUC02/geography, basin area, and hydroclimatic/aridity attributes from
   `stage1_static_attributes_v001`. California excluded from Stages 1–3
   (unchanged, §8c). Explicit basin-list artifacts required for
   development-training/validation/temporal-test/non-CA-spatial-holdout/
   California-Stage-4 splits — none exist yet, spatial leakage prevention
   remains a Flash-NH basin-list responsibility (Q4 evidence).
7. **Next milestone defined, not started:** `2K-G-I — Baseline Package
   Builder + Split Config Implementation` — a checklist (target conversion,
   lead-time shifting, raw-`m^3/s` audit, MRMS-gap exclusion, basin-list
   artifacts, `stage1_static_attributes_v001` adoption, baseline NH
   YAML/config, package audit updates), not code, not executed in this
   patch.

No scientific/methodological item remains open pending evidence in
`docs/stage1_scientific_baseline_design.md`; what remains is 2K-G-I's
implementation work.

**2K-G-G Phase B evidence-gathering COMPLETE (2026-07-12) — Target
Scaling + Gap Policy + Lead-Time Feasibility Report.** All 9 NH-mechanics
questions (Q1-Q9) and the window-feasibility questions (Q10-Q11) are now
answered from authoritative Moriah NH 1.13.0 evidence; zero items remain
`REQUIRES TARGETED SOURCE INSPECTION`. Sequence across two follow-up
rounds this date:
- **Part 1 (Moriah SSH still broken from this session):** Closed Q10/Q11
  (window/sample-loss numbers) using the real `fullperiod_gap_inventory.csv`
  (from the 2026-06-24 full-period forcing postrun audit) — no Moriah run
  needed, since `scripts/analyze_stage1_window_feasibility.py` imports no
  NeuralHydrology. Either-gap window loss ranges from ~1.3%
  (`seq_length=12`) to ~5.6% (`seq_length=72`) across the full 12/24/48/72 h
  x 1/3/6/12 h design space; MRMS-gap loss dominates RTMA-gap loss by ~2
  orders of magnitude (136 vs. 2 archive-gap hours). Also found and fixed
  a real timezone-handling bug in that script (real gap-inventory
  timestamps are `Z`-suffixed/tz-aware, the internal hourly index was
  tz-naive, so the first real-gap run silently reported 0% gap-loss
  everywhere) via a `_to_naive_utc()` helper, regression-tested clean.
  Refined the Q4 leakage finding to explicitly distinguish temporal
  leakage (NH's `is_train`/passed-`scaler` contract protects this
  automatically) from spatial leakage (California/spatial-holdout basins —
  NH provides zero automatic protection; Flash-NH's basin-list
  construction upstream of NH is solely responsible). Added a gap-policy
  decision framework (Policy A: NaN + `nan_handling_method`, vs. Policy B:
  hard window exclusion) informed by the real loss numbers, without
  selecting between them. The 3 remaining NH-mechanics items were blocked
  this part because this working session had no SSH/network path to
  Moriah.
- **Part 2 (Moriah SSH access restored):** the user fixed Moriah SSH
  access from local Windows/VS Code (`ssh moriah "hostname"` ->
  `moriah-gw-01`; note default `scp`/SFTP is still broken on Moriah, plain
  SSH command execution works — legacy `scp -O` needed for any future file
  transfer, not used this round since only inline `sed`/`grep` output was
  needed). All 3 remaining items were closed by inspecting the Moriah
  1.13.0 source directly: (1) **Q2 confirmed** — `tester.py:247-259`'s
  exact inverse-scaling arithmetic is `raw = scaled * feature_scale +
  feature_center` for both predictions and observations, inline, no
  public `inverse_transform` API exists; (2) **Q5 confirmed** —
  `training/loss.py` masks target NaNs per-element in every one of 6
  `Masked*Loss` classes (`MaskedMSELoss`, `MaskedRMSELoss`,
  `MaskedNSELoss`, `MaskedGMMLoss`, `MaskedCMALLoss`, `MaskedUMALLoss`) via
  `~torch.isnan(ground_truth['y'])`-style masking before the loss
  reduction — a target NaN cannot silently contaminate training through
  any NH-provided loss class; (3) **Q6/Q7 confirmed, and the default is
  dangerous** — `nan_handling_method` defaults to `None` when unset
  (`utils/config.py:610-613`), and the unset case falls through to a final
  `else` branch in `modelzoo/inputlayer.py` that performs **no NaN
  handling at all**, passing raw (possibly NaN) dynamic inputs straight
  into an unprotected `nn.Linear` embedding — explicit configuration
  (`masked_mean`, `attention`, or `input_replacing`) is mandatory, not
  optional, for Flash-NH's gap-policy Policy A to be safe. Raw command
  output saved to
  `tmp/nh13_targeted_inspection_moriah_20260712T120839Z/` (gitignored, not
  committed).

No target-scaling, gap-policy, or lead-time implementation decision has
been made final; no NH package generated; no training run; no package
builder/config/Slurm template modified.

**2K-G-F-B COMPLETE — canonical h2o PASS (2026-07-08) — static attribute
source mirror + derived matrix builder/auditor.** Implements
the 2K-G-F plan (`docs/stage1_static_attribute_matrix_plan.md`) in code:
- `scripts/build_stage1_static_attribute_matrix.py` — merges the 29-file
  GAGES-II/HydroATLAS/NLDAS-2 source mirror into
  `stage1_static_attributes_v001.parquet` for the 2,843-basin Stage 1
  universe, applying the conservative column-classification policy (admin/
  duplicate/binary-flag/categorical-deferred/split-support/diagnostic-lat-lon
  handling, per-year-series reduction, dynamic near-constant/high-missingness
  exclusion) and the mandatory HydroATLAS 5-basin-gap gate (fail loud unless
  the observed gap exactly matches the known 5 non-standard-ID basins).
- `scripts/audit_stage1_static_attribute_matrix.py` — independently verifies
  the output (coverage, duplicates, missingness, ranges, constant/duplicate
  columns, categorical/ID-name leakage, `STATE`/`HUC02`/lat-lon exclusion,
  HydroATLAS gap handling, checksum).
- **Local dry-run PASS** against `C:\PhD\Python\neuralhydrology\US_data\attributes`
  into repo `tmp/` (gitignored, not committed): build exit 0 (2,843 rows ×
  531 cols, 496 `model_input`), audit exit 0 (0 errors, 0 warnings, 20 OK
  checks). This validated the build/audit *logic* only.
- **Canonical h2o build/audit: PASS (2026-07-08).** Run by the user directly
  on h2o (no network path exists from this environment to h2o). Source mirror
  verified (30 files = 29 source files + checksum file, all 29 `sha256sum -c`
  OK). Canonical build + audit: 0 errors, 0 warnings, 20 OK checks, matrix
  2,843 rows × 531 columns / 496 `model_input`, HydroATLAS 5-basin gap matched
  exactly, checksum verified. Canonical artifact:
  `/data42/omrip/Flash-NH/data/static_attributes/stage1_static_attributes_v001/stage1_static_attributes_v001.parquet`,
  sha256 `eb17aaa07c786a25291ceaf69e770bd54bda4bc22fbd1216a81734fa6882f464`.
  Full detail in `docs/stage1_static_attribute_matrix_plan.md` §11.6.
- Minor correction carried over from 2K-G-F: the source mirror has **26**
  `attributes_gageii_*.csv` files, not 27 as previously stated (29 total
  files is unaffected: 26 + HydroATLAS + NLDAS-2 + workbook).

No NH package was regenerated from this matrix; no training was run; no NH
config/Slurm scripts were modified — all explicitly out of scope for this
milestone.

**2K-G-F DONE (2026-07-06) — static attribute matrix recovery + audit
plan.** Inventoried the full local source directory
(`C:\PhD\Python\neuralhydrology\US_data\attributes`: 29 CSVs + 1 variable
description workbook, all keyed on `STAID`, all 9,008 rows) and cross-checked
it against the real Stage 1 basin manifest (`config/stage1_initial_training_basin_manifest.csv`,
2,843 basins). Key findings, recorded in the new
`docs/stage1_static_attribute_matrix_plan.md`:
- **100% GAGES-II source coverage** of the Stage 1 basin set once `STAID` is
  zero-padded to 8 characters, including all 6 non-standard-length USGS IDs
  (five 15-char, one 9-char coordinate-based station numbers).
- **HydroATLAS covers 99.8%** (2,838/2,843) after zero-padding — the 5-basin
  gap is exactly the 15-char non-standard IDs (HydroATLAS's own `STAID` export
  is not zero-padded, unlike the GAGES-II CSVs). **Clarified as a mandatory
  build/audit gate** (same-day follow-up review): the builder/auditor must
  detect these 5 basins and either resolve/match them, retain them under a
  documented missing-value policy, or fail the build — no silent partial
  HydroATLAS merge is allowed.
- **Confirmed the existing canonical 48-column parquet stores `STAID` as
  `int64`** (leading zeros stripped) — already handled by the builder's
  `_norm_staid()`, but any new merge/audit script must reimplement 8-char
  zero-padding independently; do not assume any source/intermediate file
  preserves it.
- **The existing 48-column merge draws from only 3 of the 27 GAGES-II source
  files** — no topography, geology, land cover/vegetation, or snow fraction.
  The richer sources (Topo, Geology, 5× land-cover buffers, Soils, Climate,
  HydroATLAS, NLDAS-2) are cataloged and ready to merge; **snow fraction is
  only available via HydroATLAS** (`snw_pc_*`), not any GAGES-II file.
- Full audit of all 780 non-ID source columns restricted to the Stage 1
  basins: 758 numeric-like / 22 non-numeric (classified into
  drop/binary-flag/categorical groups); only 6 columns >20% missing; 20
  near-constant columns; one duplicate column (`DRAIN_SQKM`).
- Proposed canonical paths (h2o source mirror, Moriah source mirror, derived
  matrix path — see plan doc §7) and a merge/audit policy (§8–9). h2o/Moriah
  mirror status **not checked from this session** (no network path from this
  environment) — explicit user-side check/transfer commands documented in
  the plan doc §6 instead.
- **Filtering philosophy decided (same-day follow-up review): conservative by
  default.** Ambiguous/borderline variables (problematic, non-physical,
  administrative, weakly useful, leakage-prone, near-constant, high-missingness,
  hard to interpret) are excluded from `v001-core` by default, not kept on
  the chance the model learns something — a defensible small first matrix is
  preferred over a maximal one. Under this policy, `STATE`/`HUC02` are
  excluded outright from `v001-core` model inputs (kept only for split
  construction/diagnostics/reporting); lat/lon are held out of `v001-core`
  by default and deferred to a dedicated ablation on spatial generalization.

**No final static-attribute matrix was built.** No code, config, package,
Slurm script, or training changed; no h2o/Moriah transfer performed. The
per-column audit CSV produced during inspection is a local scratch artifact
only, outside the repo.

**2K-G-E REVISED (2026-07-06) — scientific baseline design aligned to
user-approved decisions; two new gating mini-milestones defined.**

The first 2K-G-E proposal (2026-07-03) was never committed; user review
changed several key decisions before commit, so `docs/stage1_scientific_baseline_design.md`
was revised in place rather than amended. 14 binding decisions are now
recorded there ("Binding decisions" section). Highlights of what changed from
the 2026-07-03 draft:
- **Static attributes reopened:** the draft ~16-column sign-off candidate is
  **withdrawn**. The 48-column GAGES-II screening merge remains a valid,
  checksum-verified provenance artifact but is likely insufficient as the
  final modeling matrix — richer source material exists locally
  (`US_data/attributes`, ~28 files, ~350-variable description workbook) and
  is not yet mirrored to h2o/Moriah. Gated on new **Milestone 2K-G-F**.
- **Target normalization:** log-transform **rejected** (poorly aligned with
  flash-flood/high-flow emphasis). Leading candidate is now area-normalized/
  specific discharge, pending feasibility. Gated on new **Milestone 2K-G-G**.
- **`seq_length`:** narrowed and made **binding** — Stage 1 candidates are
  only 12/24/48/72 h; 168/336 h explicitly belong to Stage 2, not Stage 1
  (withdraws the draft's 336 h literature-based proposal).
- **Lead time:** new design axis added — primary 6 h, secondary 12 h, 1/3 h
  diagnostic-only; explicitly separate from `seq_length`.
- **Temporal split dates revised:** train 2020-10-14→2023-12-31, validation
  2024, test 2025 (was train ≤2022-12-31 / val 2023 / test 2024–2025).
- **Spatial split added:** California excluded entirely from Stages 1–3;
  ~10% non-CA spatial holdout, test-only, evaluated on the 2025 test period.
- **California transfer learning (Stage 4) added:** ~90/10 CA split,
  CA-only normalization-refit exception for fine-tuning, compare
  original-vs-fine-tuned on CA holdout.
- **Leakage-prevention rules made explicit:** all Stage 1–3 scalers fit only
  on development-training data/period; Stage 4 CA scalers fit only on the CA
  fine-tuning subset.
- **Loss vs. metrics separated:** training loss still open (depends on target
  scaling); evaluation always in raw `m^3/s`, raw-space NSE primary.
- **Hyperparameters reframed:** the conventional table is now an *initial
  seed config* only — the official benchmark requires a W&B sweep, not yet
  run.
- **W&B policy expanded:** loss/validation curves, LR, epoch timing, run
  duration, GPU type, resource telemetry, in addition to config/provenance.
- **Slurm policy:** stays flexible/parameterized, not hard-pinned; resources
  may be increased later based on telemetry.

A "Before full 2,752-basin NH package generation" checklist was updated to
include the two new mini-milestones. **No code was changed** — this remains
documentation-only; no config written, no package generated, no training run,
no Moriah/California data transfer.

**2K-G-D-A COMPLETE (2026-07-03) — canonical attribute artifact promoted off `tmp`;
h2o checksum verification PASS.**

2K-G-D (same day) identified the static attribute file as an external,
checksum-pinned artifact (per `docs/repo_policy.md` generated-artifact policy — a
generated data product is not committed regardless of its small size) but left it
resident under `/data42/omrip/Flash-NH/tmp/`, and left the h2o-copy checksum
unverified pending h2o shell access. 2K-G-D-A closes both:
- **Promoted** the canonical h2o-resident copy from
  `/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` (now historical/staged only)
  to the stable project data path
  `/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`.
- **Verified** (user-run on h2o): sha256 of both the `tmp/` copy and the newly
  promoted stable-path copy is
  `06a9eeda9e94261d0b1bb9f2c2f42cb6bf11b4c02745d7ed5867ef0e0c0ad0b1` (`ls -lh`: 2.9M
  both) — identical, matching the local repo-fixture checksum recorded at 2K-G-D.
  Full evidence: `docs/stage1_attribute_provenance.md`.
- The parquet itself is still **not committed to git** — only the checksum,
  path, and provenance are documented. `attributes_sha256` continues to be
  written into every package's `run_provenance.json`
  (`scripts/build_stage1_nh_package.py`).
- Remaining open item (non-blocking): the Moriah mirror path
  (`/sci/labs/efratmorin/omripo/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`)
  is documented but not yet populated or verified.

A design-gate scaffold for the first scientific baseline is now at
`docs/stage1_scientific_baseline_design.md`: purpose/non-goals, dynamic-input set,
static-attribute subset, target cleaning/normalization, forcing-gap policy, loss/metrics,
train/val/test protocol, W&B policy, Slurm partition/GRES parameterization, and evidence
bundle conventions. Most items are explicitly marked **OPEN** — this is a decision
scaffold, not a locked spec. **Correction to prior framing:** earlier entries below
described "lookback-expansion tests (seq_length 72/168/336)" as the next milestone —
`seq_length` is one hyperparameter decided inside the design gate (§9), not the
milestone driver.

**Smoke 1 PASS (2026-07-02) — meteorology ingestion confirmed (retained for reference).**

**This is a technical meteorology-ingestion PASS, not a scientific baseline.**
6 RTMA vars + MRMS QPE + 2 gap flags, `seq_length=24` (same as Smoke 0 — isolates input
expansion from lookback change), 5 basins, 3 epochs, loss NSE.
Purpose: confirm RTMA meteorology loads, normalizes, and trains without error.

**Smoke 1 facts (Slurm job 45370873, 2026-07-02):**
- Node: `catfish-04` (NVIDIA L4, `catfish` partition); wall time: 00:01:41; exit 0:0
- MaxRSS: 1,380,944 KB (~1.35 GB batch step)
- Same package as Smoke 0 (h2o audit 2026-07-02T11:44:43Z — PASS, 0 errors)
- Config: `seq_length: 24`, `epochs: 3`, `loss: NSE`, 8 dynamic inputs — all source-built
- `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms` all non-null ✓
- `rtma_2d_K` non-null confirms 2K-F-C-B dewpoint fix carried through correctly
- Epoch 1: 0.00422 (finite ✓); Epoch 2: 0.00360 (finite ✓); Epoch 3: 0.00335 (finite ✓)
- All 3 epochs show monotonically decreasing loss; validation completed each epoch
- Run dir: `/sci/labs/efratmorin/omripo/Flash-NH/runs/flashnh_stage1_smoke1_0207_164941`
- Artefacts: `model_epoch001/002/003.pt` (~83 KB each), optimizer states, TensorBoard events

**[Historical — this "Next"/"Remaining gates" block predates Milestones
2K-G-F-B, 2K-G-G, and 2K-G-H and is superseded by them; see the 2K-G-H
block at the top of this file. Retained for reference only, not current
guidance.]**

**Next: close out attribute-policy sign-off (2K-G-F/2K-G-F-B close-out) now
that the canonical h2o build/audit has PASSed. In parallel, run Milestone
2K-G-G (target scaling + gap policy +
lead-time feasibility report — requires actual NH 1.13 code inspection on
Moriah), then close the remaining sign-off items in
`docs/stage1_scientific_baseline_design.md`, select the non-CA
spatial-holdout basin list, encode the resolved policy into
`config/stage1_scientific_baseline_v001.yaml` + NH YAML, run the W&B
hyperparameter sweep, then generate the full 2,752-basin NH package.**

**Remaining gates before full 2,752-basin NH package + scientific baseline:**
- ~~Attribute provenance / checksum verification~~ — **CLOSED 2K-G-D-A (2026-07-03)**.
  Canonical path promoted off `tmp`; h2o checksum verified PASS (48-column
  screening merge only — see next item for the modeling-matrix gate).
- **Milestone 2K-G-F (Static Attribute Matrix Recovery + Audit): plan done
  2026-07-06**; **2K-G-F-B (builder/auditor + local dry-run + canonical h2o
  build/audit) COMPLETE 2026-07-08** (`docs/stage1_static_attribute_matrix_plan.md`
  §11) — source inventory, coverage cross-check, column-classification
  policy, builder/auditor scripts, local dry-run, and canonical h2o
  build/audit (PASS) are all complete. **Not done:** Moriah mirror of the
  source attributes and derived matrix, attribute-policy final sign-off.
- **NEW — Milestone 2K-G-G (Target Scaling + Gap Policy + Lead-Time
  Feasibility Report):** not started. Requires reading actual NH 1.13 code
  on Moriah (not docs/assumptions) to resolve target-normalization
  feasibility (§5) and forcing-gap-policy feasibility (§6), and to quantify
  sample/window loss across `seq_length`×lead-time combinations.
- Scientific-baseline design gate: **REVISED into 14 binding decisions
  2K-G-E (2026-07-06)** — see `docs/stage1_scientific_baseline_design.md` →
  "Binding decisions." Still open pending 2K-G-F/2K-G-G: target
  normalization, forcing-gap policy, static-attribute column list. Also
  still open: non-CA spatial-holdout basin selection (~10%), California
  basin identification for Stage 4, `seq_length`/lead-time final selection
  (via W&B sweep, not yet run).
- Slurm templates (smoke0/1 sbatch) are hard-pinned to `catfish/L4`. Policy
  is to keep this parameterized/flexible (§11) — sbatch edit itself still
  deferred to when the baseline config is assembled.
- Moriah mirror of the attribute file (documented path, not yet populated/verified) —
  only needed if a Moriah-side build reads the attribute file directly.
- Revised temporal split dates (§8) and California exclusion are **not yet**
  encoded in `scripts/build_stage1_nh_package.py`'s split constants — that is
  a code change, out of scope for documentation-only milestones.

---

**Smoke 0 PASS (2026-07-02) — technical plumbing confirmed (retained for reference).**

Rain-only (`mrms_qpe_1h_mm` + `mrms_qpe_1h_mm_gap`), `seq_length=24`, 5 basins, 2 epochs.

**Smoke 0 facts (Slurm job 45370683, 2026-07-02):**
- Node: `catfish-05` (NVIDIA L4, `catfish` partition); wall time: 00:01:55; exit 0:0
- Package regenerated on h2o (2026-07-02T11:43:53Z) with patched builder; h2o audit PASS
- Config: `dataset: generic`, DD/MM/YYYY dates, `epochs: 2`, `head: regression`,
  `output_activation: linear` — all from source (not manual edits)
- `attributes/attributes.csv: OK` (new canonical layout)
- Epoch 1 avg loss: 0.00577 (finite ✓); Epoch 2 avg loss: 0.00556 (finite ✓); validation completed
- Run dir: `/sci/labs/efratmorin/omripo/Flash-NH/runs/flashnh_stage1_smoke0_0207_153320`
- Artefacts: `model_epoch001.pt`, `model_epoch002.pt` (~77 KB each), optimizer states, TensorBoard events

---

**NH 1.13 compatibility patch applied 2026-07-02 (commits 5e8a334 + 60fce38).**

Manual Smoke 0 diagnostic attempts revealed NH 1.13 config/layout incompatibilities in
the original builder. Source corrected; regenerated package passed h2o audit and Moriah Smoke 0:
- `dataset: generic` (was `GenericDataset`)
- All `_date` fields: `DD/MM/YYYY` (was ISO `YYYY-MM-DD`)
- `epochs` key (was `num_epochs`; rejected by NH 1.13)
- `head: regression`, `output_activation: linear` (were absent)
- `shuffle`, `log_n_basins` removed (rejected by NH 1.13)
- `attributes/attributes.csv` canonical layout (was root-level `attributes.csv`)
- Package-internal `slurm/` no longer generated; repo-level sbatch is the Slurm entry point

---

Pre-conditions completed 2026-07-01:
- Moriah `flashnh-moriah` env installed (Slurm job `45365952` PASS)
- Corrected full-period curated forcing v001 built on h2o (PASS — see below)

---

**Moriah env install PASS (2026-07-01): Slurm job `45365952`.**

- Env prefix: `/sci/labs/efratmorin/omripo/Flash-NH/envs/flashnh-moriah`
- `torch==2.7.0+cu128` installed; `nh-run` present at `envs/flashnh-moriah/bin/nh-run`
- `nh-run --help` confirmed: valid modes `train`, `continue_training`, `finetune`, `evaluate`
- `neuralhydrology` import OK (no `__version__` attribute — expected)
- Log ended with `=== done ===`

**Module fixes applied to both Moriah sbatch scripts** (initial non-interactive shell failure resolved):
1. Source module-system init file if `module` not in PATH
2. `module load spack/all` before any other module
3. Exact module name `miniconda3/24.3.0-gcc-iqeknet` (not `miniconda3/24.3.0`)

**Pilot package transfer to Moriah PASS (2026-07-01):**
- Path: `/sci/labs/efratmorin/omripo/Flash-NH/data/stage1_pilot_v001`
- Verified: 5 NC files, `run_provenance.json`, `configs/stage1_smoke0_nh.yml`,
  `attributes.csv` — all present; size 19 MB.

---

**Corrected full-period curated forcing v001 PASS (2026-07-01): 2,752 basins × 45,720 h.**

Build facts (evidence bundle: `tmp/stage1_curated_forcing_v001_corrected_fullperiod_evidence/`):
- h2o path: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/stage1_basin_hourly_forcings_v001`
- 63/63 months; 2,752/2,752 basins; 45,720 rows/basin; 0 failures
- MRMS gap-hours: 374,272 (= 136/basin × 2,752 — exact); RTMA gap-hours: 5,504 (= 2/basin × 2,752 — exact)
- Wall time: 14.43 h (2026-06-30T10:09Z → 2026-07-01T00:34Z); repo commit at run: `5f07d4b`

Audit (full-period mode): **PASS** — 2,752/2,752 basins checked; all row/gap counts exact.
Sample20 diagnostic: **ALL PASS** — `rtma_2d_K` populated ✓; `rtma_weasd_kgm2` absent ✓.
Generated evidence not committed (local: `tmp/stage1_curated_forcing_v001_corrected_fullperiod_evidence/`).

This closes the 2K-F-C-B corrected-rebuild loop. Full 2,752-basin NH package generation
is unblocked from the forcing side; remaining gates: Smoke 0 PASS + attribute-source cleanup.

---

**Milestone 2K-G-C-A COMPLETE (2026-06-30): Moriah GPU/Conda/Slurm preflight documented.**

Facts confirmed via `ssh`/`srun` reconnaissance on 2026-06-30:
- Login `moriah-gw-01`; project root `/sci/labs/efratmorin/omripo/Flash-NH`
- Partitions: `catfish` (L4, `gpu:l4:8`, 7-day) — chosen for Smoke 0; `salmon` (L40S);
  `goldfish` (H200); `dogfish` (A100, drained); `glacier` (CPU default)
- GPU node `catfish-05`: NVIDIA L4, 23034 MiB; driver 580.95.05 / CUDA 13.0;
  toolkit `cuda/12.8.1` (`nvcc` → 12.8.93)
- Conda module-gated; prefix env pattern under Flash-NH project root confirmed
- Two Slurm templates prepared (`setup_flashnh_moriah_env.sbatch`, `run_stage1_smoke0_moriah.sbatch`)

Full detail: `docs/stage1_neuralhydrology_preflight.md` §10.6.

---

**Milestone 2K-G-B COMPLETE (2026-06-30): NeuralHydrology pilot package built and audited on h2o.**

h2o audit result: **PASS** — 0 errors, 5 warnings, 217 OK checks.
Build time: 4.2 s. Audit timestamp: 2026-06-30T12:38:40Z.
Package: `/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/`
Evidence: `tmp/stage1_nh_pilot_v001_evidence/` (not committed)

**5-basin audit summary (all pass):**

| Basin | Rows | MRMS gap | RTMA gap | qobs NaN | qobs coverage |
|---|---|---|---|---|---|
| 01019000 | 45,720 | 136 | 2 | 515 | 98.87% |
| 01022500 | 45,720 | 136 | 2 | 6,751 | 85.23% |
| 01033000 | 45,720 | 136 | 2 | 12,088 | 73.56% |
| 01038000 | 45,720 | 136 | 2 | 3,035 | 93.36% |
| 01049500 | 45,720 | 136 | 2 | 6 | 99.99% |

**5 warnings (all expected):** one per basin — qobs NaN counts logged (normal; NH loss-masks missing targets).
No forcing NaN warnings. All forcing variables non-null after gap-fill.

**Key checks confirmed:**
- All 14 variables present per NC (11 forcing + 2 gap flags + qobs_m3s)
- `rtma_weasd_kgm2` absent (forbidden — confirmed)
- `rtma_2d_K` non-null == 45,720 (confirms 2K-F-C-B dewpoint mapping fix carried through)
- `mrms_qpe_1h_mm_gap sum == 136` per basin; `rtma_gap sum == 2` per basin
- Gap fill: MRMS 136 NaN → 0.0 mm/basin; RTMA 2 NaN → linear interp per variable/basin

**Static attribute caveat (cleanup required before full-scale package) — RESOLVED 2K-G-D-A
(2026-07-03), see top of this document and `docs/stage1_attribute_provenance.md`:**
`reports/flashnh_basin_screening_v001/all_basins_merged.parquet` is **not tracked in git**
(verified with `git ls-files` on h2o). The h2o builder used a manually staged copy at
`/data42/omrip/Flash-NH/tmp/all_basins_merged.parquet` (this path is now historical/staged
only — promoted to `/data42/omrip/Flash-NH/data/static_attributes/gagesii_v001/all_basins_merged.parquet`,
checksum-verified, at 2K-G-D-A).
The 5-basin pilot PASS is valid. Before full 2,752-basin NH package generation, this file
must be made canonical: committed to the repo or documented as a stable h2o-resident input
with explicit provenance. This is a cleanup gate, not a blocker for Moriah transfer.

**Package structure (on h2o):**
```
/data42/omrip/Flash-NH/tmp/stage1_nh_pilot_v001/
  time_series/{STAID}.nc     # 5 NCs; 14 vars; 45,720 rows; float32; _FillValue=-9999.0
  attributes.csv             # 5 basins × 47 cols
  basins/smoke{0,1}_{train,val,test}.txt
  configs/stage1_smoke{0,1}_nh.yml
  slurm/smoke{0,1}.sh        # Moriah Slurm job templates
  manifests/                 # dataset_manifest.json + variable_schema.csv + gap_fill_report.csv + per_basin_summary.csv
  run_provenance.json + README.md + audit_summary.md
```

**Next: 2K-G-C — Moriah transfer + environment preflight + Smoke 0.**
Transfer pilot package (`scp`), confirm NH conda env on GPU node, run Smoke 0 (seq_length=24, 2 epochs).
No NH training has run yet. Full 2,752-basin NH package generation waits for:
(1) corrected full forcing rebuild PASS on h2o; (2) attribute-source cleanup
(**resolved 2K-G-D-A, 2026-07-03** — see top of document).

---

**Milestone 2K-G-A COMPLETE (2026-06-30): NeuralHydrology pilot package preflight design + corrections.**

Design frozen in `docs/stage1_neuralhydrology_preflight.md` (Part I), with corrections applied
after initial commit `fa6754b`:
- NH package format: GenericDataset single-NC-per-basin, `date` coord, float32, `_FillValue=-9999.0`
- Smoke 0: rain-only (mrms_qpe_1h_mm + gap flag, 5 basins, 1–2 epochs); `seq_length: 24`, `predict_last_n: 1`
- Smoke 1: minimal meteorology (6 forcings: mrms + rtma_{2t,2d,2sh,10u,10v}); `seq_length: 24`
  (same as Smoke 0 — isolate input expansion from lookback change; 72/168 h are later separate tests)
- Gap-fill policy (Smoke 0/1 pilot policy only): MRMS gaps → 0.0 mm; RTMA gaps → linear interp; gap flags retained
- Final training gap policy: window-exclusion preferred over silent fill; to be decided after Smoke 1
- Moriah layout: `/sci/labs/efratmorin/omripo/Flash-NH/{repos,envs,data,runs,logs,slurm,evidence}`
- NH setup: clean upstream `neuralhydrology` clone; no fork until specific limitation demonstrated

---

**Milestone 2K-F-C-B COMPLETE (2026-06-30): Curated forcing schema/mapping correction.**

Full-period build structurally PASS on h2o (2026-06-30, 2,752 basins, 45,720 h, 14.49 h wall),
but post-build non-null check found two all-NaN variables. Build is **schema-superseded**;
corrected rebuild required before final certification.

**Schema issues found and corrected in code:**
- `rtma_2d_K` (dewpoint): all-NaN because builder mapped source `d2m` → `rtma_2d_K`, but
  actual source variable is `2d`. Fixed: `"2d" → "rtma_2d_K"` in both builders.
- `rtma_weasd_kgm2`: all-NaN because `weasd` is absent from all 63 monthly source chunks.
  RTMA precipitation (`ACPC01`) is not present. Removed from schema entirely.
- `rtma_2d_K` is **retained** (source `2d` confirmed present in all sampled months with
  `variable_standard_name=dewpoint_temperature_2m`).

**Corrected v001 schema:** 1 MRMS variable + 10 RTMA variables + 2 gap flags = 13 columns.

**Full-period structural build evidence (schema-superseded, not committed):**
- Period: 2020-10-14T00Z – 2025-12-31T23Z
- 63/63 months, 2,752 basins, 45,720 rows/basin, 374,272 MRMS gap-hrs, 5,504 RTMA gap-hrs
- Full-period audit: PASS (structural); wall time 14.49 h; commit at run `addfdd2`
- Note: accidental second launch was stopped early; post-interruption audit PASS confirmed
  product not corrupted. `build.log` may contain aborted-rerun lines after first PASS.
- Evidence under `tmp/stage1_curated_forcing_v001_schema_issue_evidence/` (not committed)

**Corrected full-period rebuild:** PASS (2026-07-01) — see current milestone block above.
**Design doc:** `docs/stage1_curated_forcing_product_v001_design.md`

---

**Milestone 2K-F-B COMPLETE (2026-06-29): Curated forcing product v001 builder + smoke test — PASS.**

Builder (`build_stage1_curated_forcing_basin_parquets.py`), auditor
(`audit_stage1_curated_forcing_basin_parquets.py`), and h2o launcher implemented and committed
(`6f4de49`). 5-basin / 2020-11 smoke test run on h2o: all 9 acceptance criteria PASS.
- 5/5 basins (`01440000`, `03021350`, `08155541`, `09484000`, `01019000`); 720 h each
- 0 MRMS gap-hours; 10 RTMA gap-hours (2/basin) at 2020-11-12T09Z/T10Z
- Coverage 0.9972; `rtma_gap=True` confirmed at both known timestamps; MRMS not falsely flagged
- Auditor exit 0; SHA-256 checksums verified; commit at run `6f4de498`
- Note: `02231000` attempted but absent from 2020-11 source; builder correctly halted; not a failure
- h2o output: `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z`

---

**Milestone 2K-E COMPLETE (2026-06-24): Full-period forcing extraction audit — PASS_WITH_CAVEATS.**

Full-period MRMS+RTMA basin-average forcing extraction (63 months, 2020-10 → 2025-12,
2,752 basins) is complete on h2o. Post-run audit finished locally.

**Audit result summary:**
- 63/63 months `all_pass=True`, 0 failures
- 1,509,422,464 combined rows (125,447,168 MRMS + 1,383,975,296 RTMA); 0 row-count mismatches
- 11 RTMA variables, uniform; `rtma_10wdir_absent` and `rtma_orog_absent` confirmed all months
- 138 missing hour-products across 20 months (136 MRMS + 2 RTMA), all `not_in_s3`
- MRMS 24h window impact: 949 / 45,697 windows (2.08%); RTMA: 25 / 45,697 (0.05%)
- 0 basin×product pairs incomplete across all months
- 0 unexpected warnings
- Caveat: two-commit provenance (2020-10 → `194a489`; 2020-11 → 2025-12 → `7e43760`); documentation only
- **No rerun required**

**Full audit result:** `docs/stage1_forcing_fullperiod_audit.md`  
**Audit plan:** `docs/stage1_forcing_fullperiod_postrun_audit_plan.md`  
**Generated audit tables (not committed):** `tmp/stage1_forcing_fullperiod_postrun_audit_20260624T060504Z/`

**Next step:** Milestone 2K-F-B — curated forcing product v001 builder + 5-basin smoke test on h2o.
Design frozen in `docs/stage1_curated_forcing_product_v001_design.md`. Not model training yet.

**Pilot visual QC PASS (2026-06-25/28):**
- Basin-timeseries pilot: 6/6 cases OK (VQC-001, -004, -007, -009, -012, -020).
  Time-series rendering, gap labeling, VQC-001 boundary clip, and qobs alignment all pass.
  h2o output: `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_visual_qc_pilot_20260625T123337Z`
- Spatial MRMS smoke (VQC-012, VQC-009): basin=Y, gauge=Y. Raster placement consistent
  with observed qobs responses. No extraction or alignment failures detected.
  h2o output (VQC-012): `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod_spatial_mrms_qc_smoke_20260625T142012Z`
- This is a technical/rendering PASS and scientific QC evidence improvement.
  It is **not a final full forcing certification** — 15 of 21 cases not yet animated.
- Generated outputs (PNG/GIF/CSV/summary) remain under `tmp/` and are not committed.
- See `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md` for full evidence.

---

Stage 1 full 2,843-basin USGS IV target acquisition structurally complete (2026-06-13).
Target policy configured (`config/stage1_target_policy.yaml`, 2026-06-15).
h2o preprocessing environment installed and smoke-tested (`flashnh-stage1`, 2026-06-15).
Target package builder + auditor implemented, smoke-tested, and h2o policy-smoke PASS (2026-06-15).
**v001 target package (2,752 basins) built and audited on h2o (2026-06-16): PASS — 0 errors, 0 warnings.**
**Milestone 2K-A COMPLETE (2026-06-18): v001 basin-weight tables built on h2o — 2,752/2,752 basins, PASS.**
**Milestone 2K-B COMPLETE (2026-06-18): forcing extraction smoke test — PASS. RTMA 48/48 h; MRMS 27/48 h (21 `not_in_s3`, expected early archive gap).**
**Milestone 2K-C COMPLETE (2026-06-18): October 2020 one-month run — PASS.
432h, 2,752 basins, 396/432 MRMS, 432/432 RTMA, 14,167,296 rows, 15h 05m wall.
Full-period extraction PAUSED — 66.5-day projected wall time requires 2K-D optimization.**
**Milestone 2K-D COMPLETE (2026-06-20): D1 serial optimization → 24.7× speedup
(91.9 s → 2.17 s/hr, commit `3ff4965`). Outer-parallelism x3×dw6 → 3.04 days projected
(commit `a275296`). D2 deferred. x4 not recommended.
Decision: full-period launch — 3 concurrent chunks × 6 download workers.**
**Milestone 2K-E pre-launch patch COMPLETE (2026-06-20): `GROUP_ID=A/B/C` and `DRY_RUN=1`
added to fullperiod launcher; path safety guard and per-group logs; reporter updated.
Dry-run validation pending on h2o. Full-period extraction NOT yet launched.**

See `docs/stage1_hpc_transition_preflight.md` for the full audit summary and
`docs/stage1_target_policy.md` for target-policy rationale.

### Quick summary

- 2,843 canonical hourly NetCDF files on h2o at `/data42/omrip/Flash-NH/tmp/stage1_full_2843/`
- Coverage 0.9652 overall; 2,754 basins with `historical_training_utility_flag=True`
- 89 basins with late-period gaps (`TARGET_OPERATIONAL_REVIEW`) — hold out of first package
- 18 basins with negative qobs in the acquisition audit — set to NaN during package build
  (2 heavily-negative special-review basins excluded from v001; 16 basins cleaned, 235 values neg→NaN)
- `TARGET_QUALITY_REVIEW` (1,375 basins): eligible for training; spike flag is advisory only
- No systematic offset issues (0 basins)

### h2o / Moriah operating plan (as of 2026-06-15)

Key policy clarifications from PI:
- h2o is **storage, downloads, preprocessing, and assembly** — not training
- h2o has **no usable GPU** (`nvidia-smi` not found; PI confirms)
- No scheduler by design; `screen` is the agreed background job manager
- CPU compute allowed with etiquette: ≤50–60% CPU; start 16–32 workers; notify before long jobs
- `/data42/omrip` is not auto-deleted; `/data42` is not backed up
- `/data42/hydrolab/Data/Flash-NH_data/` subfolders allowed with reproducibility provenance
- **NeuralHydrology training → Moriah cluster** (`/sci/labs/efratmorin/omripo/PhD`)

See `docs/stage1_h2o_operations_preflight.md` for full gate status.

### h2o environment status (as of 2026-06-18)

- **Prefix:** `/data42/omrip/Flash-NH/envs/flashnh-stage1`
- **Python:** `3.11.15` | **Size:** `7.0 G`
- **Smoke test:** ALL PASS — core, geospatial, dask, cfgrib/eccodes, NetCDF, Parquet, neuralhydrology
- **Log:** `/data42/omrip/Flash-NH/tmp/env_smoke_20260615T120918Z/env_smoke.log`
- **Activation on h2o:** `source /opt/conda/etc/profile.d/conda.sh && conda activate /data42/omrip/Flash-NH/envs/flashnh-stage1`
- **Activation caveat:** The shell prompt may show `(flashnh-stage1)` while `which python` still
  points to `/opt/conda/envs/iacpy3_2025/bin/python`. Always run the explicit `source` + `conda activate`
  sequence and verify with `which python` before running any job. Observed during 2K-A (2026-06-18);
  clean reactivation resolved it.
- **py7zr added (2026-06-18):** Installed `py7zr` into `flashnh-stage1` using the standard h2o workaround:
  `CONDA_PKGS_DIRS=/home/omrip/.conda/pkgs conda install --solver classic py7zr`.
- **Caveat:** `neuralhydrology` pip-pulled CUDA torch (2.12.0+cu130); env is 7.0 G vs lean CPU intent.
  `cuda_available=False` on h2o — functionally harmless. Future spec revision to use `--no-deps` or CPU torch.
- **h2o is not for NeuralHydrology training.** Training remains designated for Moriah cluster.

See `docs/stage1_environment.md` for full install notes, workaround, and CUDA caveat details.

### Target package builder status (as of 2026-06-16)

Milestone 2J-B: **COMPLETE** — scripts implemented, smoke-tested locally and on h2o, full v001 build PASS.

- **Builder:** `scripts/build_stage1_target_package.py`
- **Auditor:** `scripts/audit_stage1_target_package.py`
- **Launcher:** `scripts/run_stage1_target_package_v001_h2o.sh` (commit `3ac51ff`)
- **Doc:** `docs/stage1_target_package_builder.md`
- **Local smoke result (2026-06-15):** 5/5 PASS — 0 errors, 0 warnings
- **h2o policy smoke (2026-06-15):** PASS — 4 basins, 01135300 excluded (hist_util=False),
  08010000 cleaned 95 neg→NaN; audit 0 errors/0 warnings; 02299472 halt confirmed (EXIT 1)
  - `canonical_merged` confirmed: 2,843 flat NCs, 2,843 unique STAIDs, 0 recursive duplicates
- **Full h2o v001 build (2026-06-16): PASS — 0 errors, 0 warnings**
  - Input: 2,843 NCs from `canonical_merged`
  - Excluded: 2 (`--exclude-staids`) + 89 (policy: `hist_util=False`) = 91 total excluded
  - Built: **2,752 basins**, 0 failed
  - Cleaned: 235 neg→NaN across 16 basins; NaN 3,880,507 → 3,880,742; valid hours 121,940,698
  - Audit: 2,752/2,752 checksums OK; 89 held-out absent; SR basins absent; 1,373 TQR advisory
  - Audit runtime: 18.8 s
  - policy_sha256: `449165686d033b9cdbd395ad70e64a3bfa82d01757021e62059f254a2a30d691`
  - Evidence bundle: `tmp/stage1_target_package_v001_evidence/` (not committed)
  - Full result: `docs/stage1_target_package_v001_result.md`
- **Special-review 02299472/04073468:** excluded from v001; disposition open for future v002

See `docs/stage1_target_package_builder.md` for full commands and acceptance criteria.

### Stage 1 forcing — Milestone 2K-A (completed 2026-06-18)

Input preflight and v001 basin-weight table build on h2o. **PASS — 2,752/2,752 basins.**

**Input preflight (`verify_stage1_forcing_inputs_h2o.sh`):** 10/10 PASS, 0 WARN, 0 FAIL.

**Key input locations on h2o:**

| Item | Path | Notes |
|---|---|---|
| v001 basin list CSV | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv` | 2,752 rows excl. header |
| CAMELSH shapefile | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/02_basin_geometries/camelsh/shapefiles/CAMELSH_shapefile.shp` | 2,752 polygons; no `.prj`, EPSG:4326 |
| MRMS grid def | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions/mrms_grid_definition.json` | v001 flat layout (not pilot path) |
| RTMA grid def | `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions/rtma_grid_definition.json` | same |

**Weight Parquets (output):**

| File | Size | Basins |
|---|---|---|
| `02_basin_geometries/weights/mrms/v001_2752_mrms_weights.parquet` | 37 MB | 2,752/2,752 |
| `02_basin_geometries/weights/rtma/v001_2752_rtma_weights.parquet` | 12 MB | 2,752/2,752 |

All paths relative to `/data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/`.

**Clean build command:**

```bash
python scripts/build_stage1_basin_weights.py \
    --config configs/stage1_forcing_fullperiod.yaml \
    --data-root /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod \
    --basin-list /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/v001_basin_list.csv \
    --out-tag v001_2752 \
    --grid-def-dir /data42/omrip/Flash-NH/tmp/stage1_forcing_fullperiod/grid_definitions \
    --skip-qc-plots
```

Fatal validation: all PASS. `--skip-qc-plots` used because the h2o CAMELSH shapefile lacks
`LNG_GAGE`, `LAT_GAGE`, `DRAIN_SQKM` columns (schema: `LAYER, MAP_NAME, AREA, PERIMETER, GAGE_ID, geometry`).
QC plotting is advisory; the fix is in commit `026c363`.

**Operational lessons from 2K-A:**

- **Activation caveat:** Shell prompt can show `(flashnh-stage1)` while `which python` points to
  the wrong env. Always verify with `which python` after activation.
- **py7zr:** Added to `flashnh-stage1` on h2o using `CONDA_PKGS_DIRS` + `--solver classic` workaround.
- **PS1 helper broken:** `scripts/prepare_stage1_forcing_inputs_h2o.ps1` fails to parse on
  Windows PowerShell 5.1 (8 AST parse errors). It is not needed for 2K-B (grid JSONs and
  CAMELSH shapefile are already in place). Fix in a separate commit before relying on it again.
- **Stale verifier message:** `verify_stage1_forcing_inputs_h2o.sh` still prints
  "Ready to proceed to Milestone 2K-A" even after weights are built. Minor stale message;
  not a blocker. Clean up in a later small commit.
- **Grid-def path:** `build_stage1_basin_weights.py` now supports `--grid-def-dir` with 3-level
  auto-discovery (explicit → v001 flat → pilot legacy). Pass it explicitly to avoid ambiguity.

### Stage 1 forcing — Milestone 2K-B (completed 2026-06-18)

Forcing extraction smoke test on h2o. **PASS — all 12 validation checks passed.**

**Evidence:** Compact evidence bundle inspected locally from `tmp/stage1_forcing_smoke_evidence/`
(not committed). Evidence files: `smoke_manifest.json`, `smoke_summary.md`,
`smoke_live_run.log`, `smoke_hourly_runtime_and_volume.csv`, `smoke_missing_files.csv`.

**Original smoke was run via direct extractor invocation.** The launcher (`scripts/run_stage1_forcing_smoke_h2o.sh`)
raised `CondaError: Run 'conda init' before 'conda activate'` when invoked as `bash script.sh`
in a non-interactive shell, even after the PATH-prepend patch in `43af035`. The launcher
activation block was subsequently patched (commit `ccb2631`) to source `conda.sh` unconditionally
and make `conda activate` non-fatal.

**Launcher activation fix verified on h2o (2026-06-18):** After pulling `ccb2631`,
`bash scripts/run_stage1_forcing_smoke_h2o.sh` completed end-to-end via the launcher
wrapper. Python resolved correctly to `/data42/omrip/Flash-NH/envs/flashnh-stage1/bin/python (Python 3.11.15)`.
This was a cached/resume rerun (0.0 B downloaded, ~1m 12s elapsed); output row counts and
PASS status matched the original uncached run. **The launcher activation bug is resolved.**
Download and runtime estimates for 2K-C should be taken from the original uncached run (10m 13s, ~3.2 GB), not this verification rerun.

**Smoke results:**

| Metric | Value |
|---|---|
| Period | 2020-10-14T00:00:00Z – 2020-10-15T23:00:00Z |
| Basins | 10 |
| MRMS hours extracted | 27/48 |
| MRMS missing | 21 (`not_in_s3`, 2020-10-14T00Z–20Z — see note below) |
| RTMA hours extracted | 48/48 |
| RTMA missing | 0 |
| `mrms_smoke.parquet` rows | 270 (27 h × 10 basins) |
| `rtma_smoke.parquet` rows | 5,280 (48 h × 10 basins × 11 vars) |
| `combined_smoke.parquet` rows | 5,550 |
| Wall clock | 10m 13s |
| Downloaded | ~3.2 GB (RTMA `selected_messages`, 4 workers) |
| `all_pass` (manifest) | `true` |
| Git commit at run time | `43af035d` |

**MRMS 21-hour early archive gap (expected):** `noaa-mrms-pds` QPE 1h Pass1 coverage for
2020-10-14 begins at 21:00Z, not midnight. The first 21 hours (00Z–20Z) are genuinely
absent from S3 — this is a permanent upstream archive gap, not a pipeline error.
The full-period first chunk (`2020-10`) will carry the same 21-hour gap in its
`missing_files.csv`. All subsequent months have complete MRMS coverage.

**Validation checks (all PASS):**
`mrms_extracted_hours_gt_zero` · `mrms_N_basins_per_ok_hour` · `mrms_no_all_null_weighted_mean`
· `mrms_valid_weight_fraction_ok` · `mrms_parquet_written` · `rtma_extracted_hours_gt_zero`
· `rtma_10wdir_absent` · `rtma_orog_absent` · `rtma_at_least_8_variables`
· `rtma_no_all_null_weighted_mean` · `rtma_parquet_written` · `combined_parquet_written`

**Performance notes:**
- RTMA `selected_messages` download: median ~42 s/file at 4 workers → ~33–40 h total at 16 workers.
- MRMS download: ~0.3–1.3 s/file (cfgrib cold start on first file only). Negligible vs RTMA.
- Estimated full-period RTMA raw: ~3.2 TB (`selected_messages`); MRMS raw: ~0.5 TB.

### Stage 1 forcing — Milestone 2K-C (completed 2026-06-18)

October 2020 one-month forcing extraction on h2o. **PASS — all 12 extractor validation checks passed.**

**Evidence:** Compact bundle in `tmp/stage1_evidence_exports/2020-10/` (not committed).

| Metric | Value |
|---|---|
| Period | 2020-10-14T00Z – 2020-10-31T23Z |
| Scheduled hours | 432 |
| Basins | 2,752 |
| MRMS extracted | 396/432 |
| MRMS not_in_s3 | 36 (3 clusters — see below) |
| RTMA extracted | 432/432 |
| RTMA variables | 11 (incl. diagnostic `ceil`, `vis`; `10wdir`/`orog` absent, confirmed) |
| Combined rows | 14,167,296 (1,089,792 MRMS + 13,077,504 RTMA) |
| MRMS raw | 207 MB |
| RTMA raw | 30.7 GB |
| Wall clock | 15h 04m 57s (`download_workers=8`) |
| `all_pass` | `true` |
| Git commit at run | `194a489` |

**MRMS 36-hour gap (permanent S3 gaps — not pipeline errors):**

| Cluster | Hours | Timestamps |
|---|---|---|
| Archive-start | 21 h | 2020-10-14T00Z–20Z |
| Oct 25–26 outage | 14 h | 2020-10-25T23Z; 2020-10-26T00Z–11Z, 15Z |
| Oct 29 spot | 1 h | 2020-10-29T23Z |

**Throughput and full-period projection:**

- Actual throughput: 125.7 s/hr (serial, extraction-dominated)
- Full-period projection at current serial code: **66.5 days** (45,720 h × 125.7 s / 86400)
- Primary bottleneck: `extract_basin_statistics` in `src/pipeline/extraction.py:396`
  — `weights_df.loc[weights_df["STAID"] == staid]` O(N) scan, 30,272 calls per RTMA hour
- The 20.2-day figure from `scaling_estimates.json` was computed from RTMA download time only
  (download is pipelined/prefetched and is NOT on the serial critical path)

**Full-period extraction was PAUSED at 2K-C completion.** 2K-D is now COMPLETE — see section below.

### Stage 1 forcing — Milestone 2K-D (completed 2026-06-20)

Extraction optimization and outer-parallelism throughput benchmark.
**COMPLETE — effective full-period projection 3.04 days (3 concurrent chunks × 6 download workers).**

#### D1: Serial extraction optimization (commit `3ff4965`)

Two targeted changes to `src/pipeline/extraction.py` and `scripts/extract_stage1_forcing_chunk.py`:

1. **Pre-grouped weight lookup** — `_build_basin_cells()` pre-groups the weight DataFrame by
   STAID into a `{STAID: (row_idx, col_idx, norm_w)}` dict at startup. Each per-basin-hour call
   becomes an O(1) dict lookup instead of an O(N) boolean scan over the 2,752-row weight table
   (90,816 scans/RTMA-hour eliminated).
2. **Batched percentile computation** — 7 sequential `np.percentile` calls replaced with one
   batched call, eliminating 635,712 redundant sort passes per RTMA-hour.

**Measured result:** `extraction_median_s` 91.976 s → 2.17 s/hr (**24.7× speedup**).
Bottleneck fully shifted from extraction CPU to S3 download. D2 process-workers not needed.

#### Download-worker sensitivity benchmark (48h RTMA-only, 2,752 basins)

Commit `3ff4965`; RTMA `selected_messages`; Oct 2020 period; all runs `all_pass=True`.

| Workers | Wall (s) | Proj. days | dl_median (s) | ext_median (s) |
|---|---|---|---|---|
| 2  | 1157.7 | 12.76 | 31.3 | 2.21 |
| 4  | 804.8  | 8.87  | 31.3 | 2.19 |
| 8  | 642.9  | 7.09  | 35.9 | 2.18 |
| 16 | 570.5  | **6.29** | 44.9 | 2.17 |

Individual download time increases with worker count (S3 bandwidth sharing) but wall-clock improves
via prefetch concurrency. dw16 projects 6.29 days. D2 process-workers deferred; outer parallelism
is the lever for sub-4-day throughput.

#### Outer-parallelism benchmarks (RTMA-only, 48h per chunk, 2,752 basins)

All chunks `all_pass=True`, `successful_hours=48/48`, `actual_rows=1,453,056`.

**x2 — 2 chunks × dw8 (16 total S3 connections):**
Commits `cf8db74`; evidence `tmp/stage1_2kd_evidence/outer_parallel_rtma_48h_dw8_x2/`.

| Chunk | Chunk wall (s) | dl_median (s) | ext_median (s) |
|---|---|---|---|
| outer-x2-a | 735.4 | 47.2 | 2.195 |
| outer-x2-b | 720.0 | 43.1 | 2.291 |
| **Parent wall** | **736 s** | | |

Projection: 45720 × 736 / (2 × 48) / 86400 = **4.057 days — YELLOW (partial scaling).**

**x3 — 3 chunks × dw6 (18 total S3 connections):**
Commit `a275296`; evidence `tmp/stage1_2kd_evidence/outer_parallel_rtma_48h_dw6_x3/`.

| Chunk | Chunk wall (s) | dl_median (s) | ext_median (s) |
|---|---|---|---|
| outer-x3-a | 825.9 | 45.9 | 2.233 |
| outer-x3-b | 801.1 | 43.9 | 2.206 |
| outer-x3-c | 801.2 | 42.5 | 2.204 |
| **Parent wall** | **826 s** | | |

Projection: 45720 × 826 / (3 × 48) / 86400 = **3.035 days — USEFUL GREEN.**

#### Decisions (all binding)

- **Stop performance optimization.** 3.04 days projected is within the acceptable range.
- **D2 process-workers: deferred indefinitely.** Extraction is 2.17 s/hr; download (43–46 s/file)
  dominates. Process parallelism within a single chunk would not improve end-to-end throughput.
- **x4 outer-parallelism: not recommended.** x3 achieves 3.04 days; x4 would push total S3
  concurrency to 24 workers, increasing contention and operational risk for marginal gain.
  RTMA-only benchmark may understate MRMS+RTMA mixed-product overhead.
- **Full-period launch recommendation:** 3 concurrent chunk processes × 6 download workers each.
  All outputs under `/data42/omrip/Flash-NH/`. Mechanism: 3 independent screen sessions covering
  non-overlapping month groups (~21 months each), or a new parallel launcher.
  See updated `docs/stage1_forcing_fullperiod_launch_plan.md` for Phase 2 outer-parallel details.

### Immediate next steps

The v001 target package is **streamflow-only**. Full NeuralHydrology training requires
forcing data and package assembly on h2o before any Moriah transfer.

1. ~~**Push 2K-E pre-launch patch and pull on h2o**~~ — **COMPLETE (2026-06-20).**
2. ~~**Stage 1 forcing acquisition plan + weight build (2K-A)**~~ — **COMPLETE (2026-06-18).**
3. ~~**Milestone 2K-B — forcing extraction smoke test**~~ — **COMPLETE (2026-06-18): PASS.**
4. ~~**Milestone 2K-C — October 2020 one-month run**~~ — **COMPLETE (2026-06-18): PASS.**
4b. ~~**Milestone 2K-D — extraction optimization + h2o CPU-parallel benchmark**~~ — **COMPLETE (2026-06-20): PASS.**
4c. ~~**Milestone 2K-E — full-period forcing extraction**~~ — **COMPLETE and AUDITED (2026-06-24): PASS_WITH_CAVEATS.**
    63/63 months, 1.51B rows, 0 failures. See `docs/stage1_forcing_fullperiod_audit.md`.
5. ~~**Visual / event QC case selection + pilot animation + spatial MRMS QC**~~ — **PILOT VISUAL QC PASS (2026-06-25/28).**
   21 cases generated (seed=42). Basin-timeseries pilot 6/6 OK. Spatial MRMS smoke VQC-009/VQC-012 PASS (basin=Y, gauge=Y).
   Case selection: `docs/stage1_forcing_fullperiod_visual_qc_selection.md`.
   Animation plan and evidence: `docs/stage1_forcing_fullperiod_visual_qc_animation_plan.md`.
   Outputs under `tmp/` (not committed). 15 remaining cases not yet animated — not a final certification.
6. ~~**Curated forcing product v001 design (Milestone 2K-F-A)**~~ — **COMPLETE (2026-06-29).**
   Product contract frozen: wide-format per-basin Parquet, gap-flag columns, manifest, provenance.
   Design doc: `docs/stage1_curated_forcing_product_v001_design.md`.
7. ~~**Curated forcing product v001 — builder + smoke test (Milestone 2K-F-B)**~~ — **COMPLETE (2026-06-29): PASS.**
   5/5 basins, 720 h, 0 MRMS gaps, 10 RTMA gap-hours (coverage 0.9972). Scripts: commit `6f4de49`.
   h2o output: `/data42/omrip/Flash-NH/tmp/stage1_curated_forcing_smoke_20260629T132757Z/`.
8. **Curated forcing product v001 — corrected schema build (Milestone 2K-F-C)** — schema
   corrected in 2K-F-C-B (2026-06-30): dewpoint mapping fixed, `rtma_weasd_kgm2` removed.
   Next: corrected 5-basin full-period pilot on h2o (`--max-basins 5 --overwrite`), then
   full 2,752-basin rebuild authorization. Full rebuild NOT yet authorized.
9. **Milestone 2K-G-B — NH pilot package builder** — implement `scripts/build_stage1_nh_package.py`
   on h2o: merge corrected forcing Parquets + target NCs into 5-basin GenericDataset NCs,
   apply gap-fill policy (MRMS→0.0, RTMA→interp), write `attributes.csv` and basin lists.
   Transfer pilot package (~25 MB) to Moriah.
9a. **Milestone 2K-G-C — Moriah environment + Smoke 0** — install `flashnh-moriah` conda env
    (PyTorch+CUDA, NH), run Smoke 0 Slurm job (5 basins, 2 epochs, mrms_qpe_1h_mm only),
    confirm finite loss and checkpoint.
9b. **Milestone 2K-G-D — Smoke 1** — add RTMA meteorology, confirm `rtma_2d_K` non-null.
    Preflight design: `docs/stage1_neuralhydrology_preflight.md`.
10. **Moriah transfer layout and checksum-verified transfer** — define directory structure
    and `rsync`/`scp` transfer procedure; verify checksums on arrival before training.
11. **Moriah training environment and config** — only after the assembled package passes
    audit on Moriah. NeuralHydrology training remains designated for Moriah cluster.

#### 2K-C pre-launch checklist and caution

Before any 2K-C run, confirm all of the following:

**Launcher verification (new requirement):**
- Pull latest commits on h2o: `git pull --ff-only`
- Run a dry activation test: `bash scripts/run_stage1_forcing_smoke_h2o.sh --help` or check that
  the launcher reaches the Python version line without error.
- The launcher activation bug (CondaError in non-interactive shells) is patched in the current commit.
  **Verify the fix is working on h2o before launching 2K-C.**

**One-month dry run before full 63-month launch:**
- Run 2020-10 alone first (`screen -S flashnh-2020-10 bash scripts/run_stage1_forcing_fullperiod_h2o.sh`
  with the month list reduced to a single entry, or via direct extractor for 2020-10-14T21Z – 2020-10-31T23Z).
- Confirm the 2020-10 chunk manifest is written, `missing_files.csv` contains exactly 21 MRMS
  `not_in_s3` entries for 2020-10-14T00Z–20Z, and Parquet row counts are consistent.
- Pull the 2020-10 evidence bundle locally before enabling the full loop.

**Expected 2020-10 MRMS 21-hour gap:**
- 2020-10-14T00Z–20Z will appear as `not_in_s3` in `missing_files.csv` for the first chunk.
- This is a documented upstream archive gap, not a pipeline error. Do not treat as a blocker.
- All hours from 2020-10-14T21Z onward and all subsequent months have complete MRMS coverage.

**PI notification:**
- Notify PI/machine owner before starting the full 63-month extraction loop.
- Check `uptime` before launch; hold if 1-min load > 0.7 × nproc.
- Target ≤ 50–60% CPU; start with 16 workers; increase only after monitoring a full chunk.

**Storage and raw GRIB2 deletion policy:**
- Raw MRMS + RTMA GRIB2 cache accumulates to ~3.7 TB over the full period.
- After each quarter's monthly chunk Parquets are written and checksummed, delete the
  corresponding raw GRIB2 cache to free space. Do not delete until Parquets are verified.
- Monthly chunk Parquets + per-basin forcing NCs are the curated products; raw GRIB2 is reproducible.
- Do not exceed ~20 TB total across all Flash-NH data on `/data42`.

**Evidence-bundle pull policy:**
- After every quarter (roughly every 3 months of chunks), transfer compact evidence bundles
  locally: chunk manifests (`*_manifest.json`) and missing-file CSVs (`*_missing_files.csv`).
- Do not transfer raw GRIB2, staging Parquets, or combined chunk Parquets unless needed for debugging.
- Document each quarterly bundle in `docs/FLASHNH_CURRENT_STATE.md` before proceeding.

**Progress monitoring:**
- Attach to the screen session with `screen -r flashnh-fullperiod` to check live log output.
- Each monthly chunk writes a progress log to `{FORCING_ROOT}/manifests/{chunk_label}_live_run.log`.
- Check `uptime` and `df -h /data42` periodically (once per few hours).
- A per-month completion summary will be logged; each month's manifest is the checkpoint.

**Stop and resume procedure:**
- To stop cleanly: `Ctrl-C` inside the screen session; the current hour's staging Parquet may be incomplete.
- To resume: re-run the launcher with `--resume`; already-written staging Parquets for completed hours
  are skipped automatically.
- Each completed monthly chunk is independent; re-running a month re-uses cached raw files and
  skips already-extracted hours.

**Special-review disposition (02299472/04073468)** — open for future v002, not a blocker
for steps 3–8 above. 02299472: 2,605 neg; 04073468: 2,054 neg.

The following require additional confirmation before proceeding:

- Promotion of curated data to shared lab storage — gate G4 CONDITIONALLY UNBLOCKED
  (confirm write access to `/data42/hydrolab/Data/Flash-NH_data/` before first promotion).
- NeuralHydrology training — gate G3 NOT PLANNED ON h2o; blocked on Moriah scheduler
  confirmation and env setup.

**Do not run TB-scale spatial downloads without smoke-test sign-off under etiquette rules.**

---

## Milestone 2G — NeuralHydrology NetCDF builder + preflight auditor (completed 2026-06-09)

NeuralHydrology-compatible January 2023 pilot package built and audited.
Full documentation: `docs/stage1_neuralhydrology_preflight.md`

**Scripts:**
- `scripts/build_stage1_neuralhydrology_january_pilot.py` — builder (~8s)
- `scripts/audit_stage1_neuralhydrology_january_pilot.py` — auditor (~20s)

**Package:** `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/package/` (gitignored)

**Audit result:** PASS — 0 errors, 1 warning

**Package summary:**
- 50 per-basin NetCDF files; `date` coordinate; 744 hourly UTC steps; January 2023
- 11 variables per basin (10 dynamic forcings + `qobs_m3s` target)
- Smoke dynamic inputs: `mrms_qpe_1h_mm`, `rtma_2t_K`, `rtma_2d_K`, `rtma_2sh_kgkg`, `rtma_10u_ms`, `rtma_10v_ms`
- `attributes_full.csv`: 50 rows × 238 cols (237 attribute cols + `gauge_id`)
  - Manifest records 237, counting only attribute cols; both are correct
- Full HydroATLAS integration: 50/50 pilot match; 193 new columns
- Streamflow: 20 full, 8 partial, 22 all-NaN (CAMELSH files missing locally)
- Audit warning (expected, S2): nulls in `max_abs_hourly_jump_over_Q50` (1), `q95_q50_ratio` (1), `wet_cl_smj` (14) — NaN preserved, no imputation

**No model training run. No generated files committed.**

---

## Milestone 2F — NeuralHydrology package design (completed 2026-06-08)

Design and decision documentation for the NeuralHydrology package.
Full documentation: `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/design/`

Key decisions: V1 (both rtma_2d_K and rtma_2sh_kgkg in smoke), V2 (rtma_sp_Pa in wide only),
V3 (rtma_tcc_pct in wide only), S1 (22 missing CAMELSH → all-NaN qobs_m3s, 2H blocker),
S2 (preserve NaN, no imputation), S3 (full HydroATLAS 50/50), S4 (seed=42, streamflow-only split).

---

## Milestone 2E — Event animation pipeline (completed 2026-06-07)

Pilot animations (R02, R06, R09, R11) generated and approved in v2.1-stable design.
Pipeline cleanup completed.

**Stable scripts:**
- `scripts/generate_january_event_animations.py` — main animation generator
- `scripts/audit_rtma_spatial_alignment.py` — RTMA spatial audit gate
- `scripts/audit_january_event_animation_sync.py` — MRMS sync audit gate

**Audits confirmed:**
- RTMA spatial audit: 8/8 PASS, 0.0000% diff (2t, 10u, 10v)
- MRMS sync audit: 10/10 PASS, 0.0000% diff

**Key v2.1 design notes:**
- MRMS lat DECREASES with row (row 0 = 54.995 N)
- RTMA lat INCREASES with row (row 0 = 19.229 N)
- RTMA 10m wind quiver is qualitative context only — not storm-steering validation

**All-12 command** (not yet executed; run after explicit approval):
```bash
python scripts/audit_rtma_spatial_alignment.py
python scripts/audit_january_event_animation_sync.py
python scripts/generate_january_event_animations.py --all
```
Output: `tmp/stage1_pilot_dryrun/10_animations/stage1_pilot/pilot/`
Estimated runtime: ~27 min (local, GIF mode).

---

## RTMA/URMA-family precipitation diagnostic (completed 2026-06-08)

Diagnostic-only follow-up to Milestone 2E. Confirmed RTMA/URMA grid, weight,
and timestamp consistency against MRMS. **Did not modify Stage 1 model inputs.**

Full documentation: `docs/stage1_rtma_urma_mrms_diagnostic.md`

**Key findings:**

- Regular RTMA Stage 1 files have no precipitation field.
- URMA QPE `pcp_01h.wexp.grb2` contains `tp` (Total Precipitation, kg m**-2 = mm).
- URMA and RTMA share the same 1597 x 2345 LCC 2.5 km CONUS grid exactly.
- Existing `pilot_rtma_weights.parquet` reused without modification. No new weights.
- Timestamp convention A confirmed (filename HH = end of accumulation):
  r = 0.961 on R02; shifted alternatives much worse; peak at Jan 29 08Z for both.

**Pilot metrics (Convention A):**

| Candidate | r | RMSE (mm) | Note |
|---|---|---|---|
| R02 (AR, STRONG_WET) | 0.963 | 1.12 | URMA smooths peak vs MRMS |
| R06 (MN, MOD_COLD) | 0.913 | 0.70 | URMA higher; snow/mixed-precip context |
| R11 (MA, OFFSET) | 0.944 | 0.39 | Strong agreement |

**Scripts (committed):**
- `scripts/discover_rtma_urma_precip_january2023.py`
- `scripts/urma_mrms_timestamp_and_pilot.py`

**Diagnostic outputs (untracked):**
`tmp/stage1_pilot_dryrun/11_rtma_urma_mrms_diagnostics/`

---

## Completed extraction state

January 2023 pilot extraction for 50 basins:
- MRMS: 744/744 hours, 37,200 rows
- RTMA: 744/744 hours, 409,200 rows
- Combined: 446,400 rows
- valid_weight_fraction = 1.0

Streamflow: CAMELSH hourly NetCDF, 28/50 pilot basins have January 2023 data.

Refined event candidates: R01–R12 (R03 usable-with-gap).
Pilot animations: R02, R06, R09, R11 — reviewed and approved.

---

## Standing cautions

- Do not generate all 12 animations until explicitly instructed.
- Do not start model training yet.
- Do not commit generated MP4/GIF/PNG/Parquet/GRIB/NetCDF/log outputs.
- Keep local-to-HPC transition in mind.
- RTMA 10m wind vectors are qualitative context only — not storm-steering validation.
- URMA precipitation is diagnostic-only — do not add to Stage 1 model inputs.

---

## Historical note: Milestone 2H — Streamflow recovery for 22 missing CAMELSH basins

> This section is superseded for full-period target-package construction, which is now
> complete (v001, 2026-06-16). The recovery work below applied to the January 2023 pilot
> package (Milestone 2G) and is retained for reference. The current top-level next step
> is Moriah transfer layout design (see Immediate next steps above).

Recovery was needed because the January 2023 pilot package built from CAMELSH files
had 22 basins with all-NaN `qobs_m3s`. Those basins were recovered from USGS IV
(Milestones 2H–2H-D) and are fully represented in the full-period v001 package.

Recovery plan (historical): `tmp/stage1_pilot_dryrun/12_neuralhydrology_january_pilot_dataset/design/streamflow_recovery_plan.md`

**Original pending tasks (now completed or superseded):**

1. Milestone 2H: CAMELSH streamflow recovery for 22 missing basins.
2. Decide on all-12 animation run (2E follow-up).
3. Event QC conclusions: finalize which of R01–R12 are included in Stage 1 training.
4. HPC transfer planning.
5. Stage 1 model configuration and first training run.
