# Stage 1 W&B user guide (Flash-NH lead-6 pilot)

Project-specific notes for reading this pilot's Weights & Biases tracking,
once it is actually turned on for a real candidate. This is not a general
W&B tutorial (see W&B's own docs for that) -- it only covers what is
specific to how `src/baseline/wandb_tracking.py` and
`src/baseline/pilot_tracking.py` use W&B for Flash-NH Stage 1.

**Status as of this writing -- read this distinction carefully, it is the
whole basis for what "qualified" means below:**

1. **Wrapper contract, tested against a fake backend.** `tests/
   test_wandb_tracking.py`, `tests/test_pilot_tracking.py`, and `tests/
   test_wandb_offline_qualification.py` monkeypatch an in-process fake
   `wandb` module into `sys.modules` and exercise every call shape, guard,
   and failure-isolation path this wrapper makes. This proves the
   *wrapper's* contract -- it never imports or touches the real `wandb`
   package, and by itself proves nothing about real W&B I/O.
2. **Real package, offline mode: qualified.** `scripts/
   wandb_real_offline_qualification_smoke.py` drives this repo's actual
   `init_tracking_run`/`init_pilot_tracking_run`/`log_*`/`finish_*` code
   (never a reimplementation) against the real, locally-installed `wandb`
   **0.28.1** package, `mode="offline"`, no API key, no network call. It
   ran as two genuinely separate OS processes reusing the same stable run
   id (simulating two bounded Slurm jobs continuing one candidate) and
   confirmed: config/hyperparameter logging, scientific/resource metric
   logging, a compact checkpoint-reference (never checkpoint bytes), a
   clean finish, and degradation handling against a *real* backend
   exception (`wandb.errors.UsageError`, not a synthetic one -- see §13).
   The qualification record (commands, version, directory inventory,
   findings) is at `reports/wandb_real_offline_qualification_v001/
   qualification_record.json` (untracked, not part of this patch). Scope
   limits that still apply: single machine, one short local run each,
   Windows only, no GPU/NeuralHydrology training involved, and no real
   multi-node Slurm continuation (two subprocesses on one machine
   standing in for it) -- see §12 for the one continuation-semantics
   assumption this smoke run corrected.
3. **Real package, online mode: not qualified.** No live network-connected
   run (i.e. training itself running with `mode: "online"`) has ever been
   executed for this project. Treat `mode: "online"` as implemented and
   policy-selectable but unqualified until it is separately exercised
   against a live network connection during training.
4. **Sweeps: not implemented.** See §7.
5. **Offline-to-server sync: qualified for single-segment runs.** Two
   already-completed offline candidates were synced to the real hosted
   `flashnh-stage1` project on 2026-08-05 and independently verified via
   `wandb.Api()`. See §17. This did **not** exercise the multi-segment
   reconciliation case described in §12 (each synced run had exactly one
   local offline directory) -- that remains unverified until a
   multi-Slurm-job candidate is synced.

Every screen this guide describes below is a description of what W&B will
show once tracking is actually enabled for a real candidate -- not a
report of runs that already exist for a real Flash-NH structural
candidate (`raw_seedA` and its siblings remain untracked so far).

## 1. Project / run / candidate concepts

- **W&B project**: `flashnh-stage1` (`config/stage1_wandb_tracking_policy_v001.yaml`'s `project` field). All Flash-NH Stage 1 tracking, across every candidate, lives in this one project.
- **W&B run**: one entry in that project's run table. For this pilot, one W&B run corresponds to one **Flash-NH candidate** -- i.e. one `run_id` from `config/stage1_lead06_pilot_v001.yaml`'s matrix (e.g. `raw_seedA`, `emb128x64_seedA`), not one Slurm job. A candidate that trains across several bounded Slurm continuations still has exactly one W&B run (see §12).
- **Candidate**: the scientific unit of comparison -- one architecture/static-pathway/seed combination, defined by `PilotRunSpec` and logged as `run_identity["run_id"]`/`run_identity["run_profile_name"]`.

## 2. Where config and provenance appear

Every candidate's W&B run config (the "Config" panel/table) is populated once at init from `pilot_tracking.build_pilot_run_identity` plus `build_pilot_hyperparameters`. Look there for: `git_commit` / `git_dirty` (exact code state), `package_manifest_identity` / `package_root` / `package_type` (which data package trained this candidate), `baseline_policy_sha256` / `splits_dir` (which scientific policy and basin splits), `target_variable` / `lead_hours` / `seq_length`, `seed`, `static_pathway` / `embedding_hiddens`, `effective_early_stopping_policy_name` and its thresholds, and `wandb_run_id` (this run's own stable id, see §12). This is the same information already written to the repository-authoritative `pilot_run_evidence.json` for that candidate -- W&B is a second, browsable view of it, never a different source of truth (see §10).

## 3. Comparing candidates

Because every candidate logs into the same project with a consistent config schema, W&B's run-comparison table (select multiple runs -> "Compare") is the fastest way to see, side by side: config differences (e.g. `static_pathway`, `seed`, `hidden_size`), final `screening/primary_metric_median`, `early_stopping/stop_reason`, and total epochs trained. For this pilot's closed 6-run matrix, this replaces manually diffing six `pilot_run_evidence.json` files.

## 4. Inspecting training/screening curves

Two distinct metric families are logged per epoch, both plottable against the epoch step:
- **Training/resource metrics** (`log_pilot_epoch_training_metrics`, every epoch): `training_loss`, `learning_rate`, `optimizer_steps`, `wall_time_s`, etc. -- ordinary training-health curves.
- **Screening metrics** (`log_pilot_screening_event`, only at screening-cadence epochs -- see `pilot_policy.screening_validation_every_n_epochs`): `screening/primary_metric_median` (the raw-space NSE percentile this pilot's stopping decision is based on), `screening/nse_p*` percentiles, and `early_stopping/best_epoch` / `early_stopping/events_since_best_improvement` / `early_stopping/stopped` / `early_stopping/stop_reason`. These only appear at screening epochs, so the curve will have gaps between cadence points -- that is expected, not missing data.

## 5. Secondary metrics and hydrograph media

Secondary raw-space metrics (bias, KGE components, pooled statistics, etc., wherever the screening result carries them) are logged as additional `screening/*_median` / `screening/*_mean` / `screening/pooled_*` scalars alongside the primary metric -- same panel, same epoch axis. **Hydrograph plots/media are not logged to W&B at all** -- rendered hydrographs (see the atlas evaluation workflow) are a repository artifact under `reports/`, inspected directly, never uploaded here. If W&B ever gains hydrograph media logging, this line will change; it has not.

## 6. Filtering/grouping by architecture, seed, fidelity

Use W&B's run filter/group-by on config fields, not custom tags, since the config already carries the structural axes: group by `static_pathway` or `embedding_hiddens` to compare architectures, by `seed_name`/`seed` to compare seeds, by `run_profile_name` for the exact candidate label. The `max_updates_per_epoch` config field shows `null` for this pilot's own six full-fidelity structural candidates (see §16) -- every one of those trains at full fidelity. Separate capped calibration/screening candidates with a non-null value now exist (see §16 and `docs/stage1_lead06_pilot_v001.md`'s 2026-08-04 calibration section); grouping by that field distinguishes those from this pilot's structural candidates.

## 7. What a sweep is (and why none exists yet)

A W&B "sweep" is an automated hyperparameter search: W&B itself proposes and launches new runs against a search space. **No sweep has been created for this project.** This pilot's six runs are a fixed, hand-specified structural matrix (`config/stage1_lead06_pilot_v001.yaml`), not a sweep's output -- each was/will be launched individually. Real HPO sweeps are explicitly deferred to Stage B (see the foundation doc's adoption sequencing) and are out of scope until Stage B design is frozen.

## 8. Running / failed / stopped / completed state differences

W&B's own run-state column (visible in the runs table) reflects only whether the **process that called `wandb.init`/`finish`** stayed alive -- it does not know Flash-NH's own stopping semantics. Read it alongside, never instead of, the repository evidence bundle's `run_status` field:
- W&B "running": a Slurm job currently holds this run open (or a prior job crashed without calling `finish`, i.e. a genuinely orphaned "running" state -- check the repository evidence bundle's timestamp before assuming activity).
- W&B "finished" + evidence bundle `run_status: stopped_patience_exhausted` / `blocked_*`: the scientifically meaningful outcome is in the evidence bundle's `run_status`, not just the fact that `finish()` was called.
- A **"finished" W&B run with `degraded: true`** in its logged wandb summary means tracking itself had failures (see §13) -- it does NOT mean training or screening failed.

## 9. What to inspect before approving a campaign

Before trusting a candidate's W&B-visible summary to approve or reject it: (1) confirm `git_commit`/`git_dirty` match the code you expect; (2) confirm `package_manifest_identity`/`splits_dir` match the intended data package and split policy; (3) check `wandb.degraded` is `false` (or, if `true`, that `degraded_operations` only lists telemetry calls, never anything that reads as scientific); (4) cross-check the final `screening/primary_metric_median` and `early_stopping/stop_reason` against the repository's own `pilot_run_evidence.json` for the same run -- W&B is a convenience view, the repository evidence bundle is authoritative (§10); (5) never approve a campaign from W&B's dashboard alone without opening that evidence bundle at least once per candidate.

## 10. What W&B does not control

W&B is TRACKING ONLY. It never decides, and this codebase never lets it decide: basin membership or sealed-set protection, metric computation, raw-space conversion, early-stopping evaluation, checkpoint trust/adoption, package identity, or any final scientific/campaign decision. Every one of those stays entirely inside the repository-authoritative code path (`pilot_orchestration.py`, `pilot_early_stopping.py`, `pilot_evidence_bundle.py`) and would produce the exact same scientific outcome with W&B tracking fully disabled. A W&B run id is a label on a repository-defined run, never the other way around.

## 11. Offline vs. online mode

`config/stage1_wandb_tracking_policy_v001.yaml`'s `mode` field controls this, three values: `disabled` (the shipped default -- no `wandb` import, no run at all, everything above is a local no-op), `offline` (a real W&B run is created and written to local disk only -- no network call, nothing appears on the hosted W&B dashboard until someone later runs `wandb sync` on that local run directory), `online` (a real network-connected run, visible on the dashboard immediately, requires `WANDB_API_KEY` already present in the operator's shell -- this module never reads or stores that key itself). **Only `disabled` and `offline` have been exercised in this project so far.** `online` mode is implemented and policy-selectable, but has not been used or separately qualified against a live network connection -- treat it as unqualified until it has been.

## 12. Behavior across Slurm continuation

A candidate that trains across multiple bounded Slurm jobs (this pilot's normal restart pattern) reuses one stable W&B run id, not a new one per job: `pilot_tracking.derive_pilot_wandb_run_id` computes a deterministic id from `(pilot_policy_name, run_id, tracking_generation)`, and every `init_pilot_tracking_run` call passes it as W&B's own `id=` with `resume="allow"`. `tracking_generation` defaults to `"g1"` and stays there for every ordinary continuation; an operator only bumps it (e.g. to `"g2"`) to deliberately mark a genuine restart-from-scratch of the same `run_id` after abandoning its prior NH run directory -- see the module docstring in `src/baseline/pilot_tracking.py` for why that case cannot be told apart from a first-ever attempt any other way (NH run directories are timestamped, not fixed per `run_id`).

**Corrected by the real-package qualification (§ status item 2 above; do not assume the older, offline-untested description that used to be here):** in *offline* mode, "same `id=`, `resume="allow"`" does **not** make job 2 append to job 1's local run directory. Each process invocation gets its own fresh, timestamped local `offline-run-<timestamp>-<id>/` directory, and wandb itself says so out loud: `` WARNING `resume` will be ignored since W&B syncing is set to `offline`. Starting a new run with run id <id>. `` Reconciling every invocation that shares the same run id into one logical run is a **server-side, sync-time** operation (`wandb sync`, matched by run id + project) -- it never happens locally. Two single-segment offline runs have now been synced (§17), but no multi-segment candidate (one whose local `wandb/offline/<run_id>/` holds more than one `offline-run-<timestamp>-<id>/` directory) has been synced yet, so the reconciliation behavior described in this paragraph is still unverified in practice for this project. Until sync happens, a Slurm-continued candidate's tracking history exists as N separate local offline-run directories sharing one id, not one merged directory; W&B's own dashboard/history view only becomes a single coherent run for that id after all N are synced. (`online` mode's `resume="allow"` behaves as originally described -- a live server-side run is genuinely reconnected to in real time -- but `online` mode remains unqualified per § status item 3.)

A small `pilot_wandb_run_identity.json` file is also written into the NH run directory once it exists, cross-checking `(pilot_policy_name, run_id, tracking_generation)` on every call, purely so a stale run directory accidentally reused for a *different* candidate, or reused under a different `tracking_generation` without bumping it, is caught loudly (a `TrackingError`) rather than silently mixing two attempts' histories into one W&B run id. This id is derived, never handed to you to invent -- you should never need to look it up manually except as `wandb_run_id` in a run's own config or evidence bundle.

## 13. Recognizing degraded or incomplete tracking

After init, every individual W&B call (metric log, checkpoint reference, finish) is best-effort: a real backend failure is caught, warns once, and is recorded rather than raised -- it can never stop training, screening, early stopping, or checkpoint selection. To tell whether a run's tracking is complete: check its logged `wandb` summary (mirrored into the repository evidence bundle's `"wandb"` block) for `degraded: true` and a non-empty `degraded_operations` list. A `true` value means some telemetry calls silently failed for that run -- inspect `degraded_operations` to see which kind (e.g. `log_scientific_metrics`, `finish_tracking_run`); it never means the underlying scientific run itself is incomplete or untrustworthy. A run whose `backend` is `"null"` was never tracked at all (disabled policy, or a graceful init-failure downgrade) -- that is a normal, expected state for the vast majority of runs today (tracking defaults to disabled), not a degradation.

**Real incident (job `45731908`, 2026-08-02, `raw_seedA`) and local fix
(2026-08-03).** Before this fix, checkpoint references were routed through
`log_artifact_reference`, which enforces `max_artifact_reference_bytes`
(committed value `1,048,576` bytes) against the referenced file's own size
-- appropriate for small generic artifacts, but wrong for a checkpoint
reference, which is always well over that ceiling and was never meant to
carry the file's bytes in the first place. Logging a real ~1.25 MB
checkpoint's reference raised `TrackingError` **uncaught**, which killed
the whole pilot process mid-screening -- a direct violation of the "never
raised" contract stated at the top of this section. Checkpoint references
are now logged through a separate `log_checkpoint_reference` function that
never applies that size ceiling and never raises; any failure degrades
tracking (recorded under `degraded_operations` as
`"log_checkpoint_reference"`) exactly like any other telemetry call. See
`docs/stage1_lead06_pilot_v001.md`'s "Sixth Moriah result" section for the
full incident and the accompanying orchestration-state persistence-ordering
fix; this local repair has not yet been re-verified against a real Moriah
run.

## 14. How a historical backfilled run would be labeled, if one is ever added

No completed candidate has been backfilled into W&B. If this is ever done later (e.g. to make an already-completed pre-tracking candidate like `emb128x64_seedA` visible alongside newly-tracked ones), it must be clearly distinguishable from a live-tracked run, not indistinguishable retroactive history: at minimum, a `backfilled: true` config field and a note in the run's notes/description naming the source evidence bundle it was reconstructed from, with any metrics that cannot be reconstructed byte-for-byte from that evidence bundle left absent rather than approximated. This convention is documented here as a requirement for if/when backfilling happens -- **it has not been implemented, and no backfilled run exists yet.**

## 15. Per-run W&B policy override (`--wandb-policy-path`)

The committed `config/stage1_wandb_tracking_policy_v001.yaml` ships `enabled: false` / `mode: "disabled"` and stays that way -- this is the safe default every candidate uses unless a submitter explicitly opts in for that one invocation. `scripts/run_stage1_lead06_pilot.py` accepts an optional `--wandb-policy-path <path>` flag (propagated by the `.sbatch` launcher via `WANDB_POLICY_PATH=/absolute/path/to/policy.yaml`) that replaces only `PilotPolicy.wandb_policy_path` for that single run -- every other pilot-policy field (basins, hyperparameters, early-stopping policy, seed, static pathway, etc.) is untouched. The override:

- Is **machine-local and untracked**: a typical override is a copy of the committed policy with only `enabled`/`mode` flipped to `true`/`"offline"`, kept outside git (e.g. under a runtime-only directory such as `${FLASHNH_BASE}/runtime_policies/`). It is never committed, and the committed policy file itself is never edited.
- Is **validated exactly like the committed default**: the CLI calls the same `load_tracking_policy` validator eagerly, before any config generation or training starts, so a missing or malformed override fails loudly and immediately (`TrackingError`), not partway through a run.
- Is **recorded, not just used**: the override's checksum lands in `run_identity["wandb_policy_sha256"]` (mirroring `pilot_policy_sha256`/`baseline_policy_sha256`), and the literal `--wandb-policy-path <path>` flag is captured verbatim in the evidence bundle's `commands_used`. The raw path is machine-local, so only the checksum belongs in the portable `run_identity`; the path itself is only ever in `commands_used` and (on Moriah) the `.sbatch` launcher's own `pilot_result.json`.
- Leaves the **stable W&B run id and `tracking_generation` mechanism (§12) untouched**: switching the policy file does not change `derive_pilot_wandb_run_id`'s inputs, so a candidate resumed with and without an override still targets the same logical W&B run for a given `tracking_generation`.
- Requires an **explicit `WANDB_DIR` outside the repository whenever it is supplied on Moriah**: production tracking code never sets `WANDB_DIR` itself (W&B's own default would otherwise write local run state beneath the tracked repository clone). The `.sbatch` launcher defaults `WANDB_DIR` to `${FLASHNH_BASE}/wandb/offline/${RUN_ID}` (never under `REPO_CLONE_DIR`/`REPO_WORKDIR`) whenever `WANDB_POLICY_PATH` is set, and exports `WANDB_MODE=offline` alongside it -- online mode is never exported by this launcher, matching § status item 3 above. A caller may still override `WANDB_DIR` directly if a different location outside the repo is wanted.
- Does not generalize into a runtime-override framework: this is the one optional path (plus the pre-existing `--tracking-generation`, default `"g1"`, see §12) the CLI/launcher accept -- no other pilot-policy field has, or needs, an equivalent override point.

## 16. `max_updates_per_epoch` in the config panel (2026-08-03 implementation; calibration runs added 2026-08-04)

Capped-update screening support (an optional bounded-optimizer-update mechanism for cheap early candidate screening, see `docs/stage1_validation_optimization_foundation.md` Part L) is implemented and has now been exercised by real capped calibration/screening candidates (`raw_seedA_cap_medium_cal`, `raw_seedA_cap_low_cal`, `emb128x64_seedA_cap_low_cal`, `raw_seedA_cap25k_cal`, `emb128x64_seedA_cap25k_cal`, `raw_seedB_cap50k_cal`, `emb128x64_seedB_cap50k_cal` -- see `docs/stage1_lead06_pilot_v001.md`'s 2026-08-04 calibration section and `docs/decision_log.md`'s 2026-08-04 entry for full evidence). **No numerical cap has been adopted for this pilot's own six structural candidates** (`raw_seedA` and `emb128x64_seedA` still show `null`), and no cap has been adopted for general production use. All capped runs above stayed in **offline** W&B mode throughout, per the same offline-mode qualification described elsewhere in this guide -- none were synced to a live server. What this means for what you see in W&B:

- `run_identity["max_updates_per_epoch"]` is always present in the config panel -- `null` for every uncapped run (`raw_seedA`, `emb128x64_seedA`, and this pilot's other full-fidelity candidates), or the exact configured positive integer for a capped run. It reaches the real W&B config the same way every other `run_identity` field does, via `init_tracking_run`'s `wandb.init(..., config=dict(run_identity), ...)` -- there is no separate wiring to remember.
- A capped run is a **distinct identity** from an uncapped one: Flash-NH rejects (before any training call) a resumed/continued run whose freshly-resolved cap contradicts a previously persisted cap for the same NH run directory, so you will never see a run's `max_updates_per_epoch` silently change value across a continuation in this project's evidence or W&B history.
- Capped-run results are provisional screening evidence, not full-fidelity confirmation -- treat any run with a non-null `max_updates_per_epoch` accordingly when comparing metrics against full-fidelity runs. **W&B's own logged metrics are never the authority for this comparison** -- the scientific numbers documented in `docs/decision_log.md` and `docs/stage1_validation_optimization_foundation.md` come from the repository's own evidence-bundle and retrospective-evaluation pipeline (real `validation_results.p` pickles and `optimizer_state_epochNNN.pt` counters), not from reading W&B's dashboard.
- The evidence bundle (`pilot_run_evidence.json`) separately records the *actual* number of optimizer updates completed per epoch (`actual_optimizer_updates_by_epoch`, read from NH's own persisted optimizer checkpoint state, never inferred from wall time or logs) alongside the *configured* cap -- W&B does not currently get this second field; consult the evidence bundle directly if you need it.

An invocation with no `--wandb-policy-path` (the ordinary case for every run today) is completely unaffected by any of this: it loads and uses the committed disabled policy exactly as before this flag existed.

## 17. Real offline-to-server sync (2026-08-05 qualification)

Two already-completed capped calibration candidates (see §16),
`raw_seedA_cap25k_cal` and `emb128x64_seedA_cap25k_cal` -- both Seed A
(967139), both `max_updates_per_epoch: 25000`, distinct only in
`static_pathway`/`embedding_hiddens` -- were synced from their Moriah
local offline directories to the real hosted `flashnh-stage1` project and
independently verified via `wandb.Api()`. This is the first time any run
has been synced to a live W&B server for this project; it qualifies the
sync workflow itself, not any new training or online-mode behavior (§
status items 3 and 5 above still apply).

**Workflow used, on the Moriah login node, from `$FLASHNH_BASE`:**

```bash
ENVBIN=$FLASHNH_BASE/envs/flashnh-moriah/bin

# One-time (or after any credential doubt): interactive re-login, verified
# against the W&B server. Never print $WANDB_API_KEY.
env -u WANDB_API_KEY "$ENVBIN/wandb" login --cloud --relogin --verify

# Sync exactly one local offline run directory. --legacy is required: this
# wandb version's default ("beta") sync mode failed with
# "ERROR user is not logged in" even immediately after a verified login.
env -u WANDB_API_KEY "$ENVBIN/wandb" sync --legacy -p flashnh-stage1 \
  "$FLASHNH_BASE/wandb/offline/<run_id>/wandb/offline-run-<timestamp>-<full-run-name>"
```

No `--entity` was passed; the verified account's `api.default_entity`
(`omri-porat1-huji`) resolved correctly on its own. Each run's local
directory holds exactly one `offline-run-<timestamp>-...` segment, so this
exercised single-segment sync only -- see §12's updated note on
multi-segment reconciliation, still unverified.

**A first sync attempt (before the `--relogin --verify` step) reported
`done.` but never actually landed the run under the verified account** --
`wandb.Api()` and a direct GraphQL `viewer` query both showed zero
projects existed. This was caught by verifying run 1 before syncing run 2
(the same discipline this section recommends for any future sync), never
attributed with false confidence, and resolved simply by re-running
`wandb login --cloud --relogin --verify` before re-attempting the sync.
Anyone syncing a run and not seeing it appear under `wandb.Api()` should
suspect stale/mismatched local credential state first, before assuming
the sync itself failed silently.

**Verification performed for each run**, read-only via `wandb.Api()` with
`WANDB_API_KEY` unset:

- `run.state == "finished"` for both.
- Config matched the source evidence bundle exactly: `run_id`,
  `static_pathway` (`raw_identity_concatenation` vs.
  `learned_fc_embedding`), `embedding_hiddens` (`null` vs. `[128, 64]`),
  `seed` (967139 for both), `max_updates_per_epoch` (25000 for both),
  `git_commit`, `package_manifest_identity`, `splits_dir` (identical
  across both, as expected for two candidates from the same package/split
  policy).
- Summary contained only metadata-only `checkpoint_ref/epoch_NNN` dict
  entries (path/checksum/size_bytes/epoch, never file bytes) plus scalar
  screening/early-stopping fields -- no `wandb.Artifact`, no
  `add_reference`/`add_file` call exists anywhere in
  `src/baseline/wandb_tracking.py` (confirmed by source read, not just by
  what happened to be logged), so no checkpoint, optimizer-state,
  validation-pickle, NetCDF, parquet, or evidence-bundle file was ever a
  candidate for upload in the first place.
- `run.files()` listed only small text/JSON files that `wandb.init`/
  `wandb.log` always write (`config.yaml`, `output.log`,
  `requirements.txt`, `wandb-metadata.json`, `wandb-summary.json`) --
  nothing else.
- `api.runs("omri-porat1-huji/flashnh-stage1")` listed exactly these two
  runs, both `finished`, confirming they appear together in one project
  and that the earlier failed attempt left no stray run behind.

**Resulting run identifiers** (entity `omri-porat1-huji`, project
`flashnh-stage1`):

- `flashnh-stage1_lead06_pilot_cap_parallel_batch_v001-raw_seedA_cap25k_cal-g1`
- `flashnh-stage1_lead06_pilot_cap_parallel_batch_v001-emb128x64_seedA_cap25k_cal-g1`

**To compare them in the W&B web UI**: open the `flashnh-stage1` project
under the `omri-porat1-huji` entity, select both run rows in the runs
table, and use W&B's built-in "Compare" action -- this is a native table
feature, not anything custom built for Flash-NH (§ hard boundary: no
custom dashboard exists or is planned here). Group by `static_pathway` or
`embedding_hiddens` (§6) to see the two candidates' structural difference
line up against their `screening/primary_metric_median` curves.

**Scope of this qualification**: two runs, both already-completed and
non-degraded, both single-segment, synced manually from an interactive
Moriah login-node shell. It does not qualify: syncing a run still being
written to (a "running" W&B state) mid-training, syncing a multi-segment
Slurm-continued candidate, `wandb sync --sync-all` or any bulk sync of
this project's other (still-local, unsynced) offline runs, or any
automation of the login/sync step itself. Every other capped-calibration
and structural-matrix run in `docs/stage1_lead06_pilot_v001.md` remains
local-only (`mode: "offline"`, never synced) unless a future entry in
this section says otherwise.
