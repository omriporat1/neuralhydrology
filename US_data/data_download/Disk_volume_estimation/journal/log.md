# Daily log (newest first)

Format: date, 3-5 lines (did / output links / tomorrow). Written by `/eod`.

## 2026-11-17 (Tue)
- Brief for Wed 11-18 drafted (scheduled run): [meetings/2026-11-18.md](meetings/2026-11-18.md); cycle-1 and talk markers filled in the evening, the other [edit] markers need Omri.
- Committed and pushed the G1 erratum + W&B stage A (ec16ec4), the WY2026 legacy clip (5db8e38) and the W&B config-upload fix (3b5fc61). Lab talk given ([meetings/2026-11-17_lab_meeting_talk.md](meetings/2026-11-17_lab_meeting_talk.md)).
- Baseline cycle 1 trained: job 46762920 COMPLETED in 4 h 40 min; 2,289 basins + 18 explained missing, score-blind check passed, W&B loss = log for all 12 epochs, no scores. Evidence `.scratch_local/wy_rotation_runs_v001/cycle1_evidence_46762920/`. No picks yet: they need the scorer seam (gate 4).
- Decided: parallel sessions in one shared checkout with a daily control session (decision log 11-17, `agent_handoff_rules.md` §3.4, skills /plan-day and /close-day).
- Tomorrow (Wed 11-18): supervisor meeting (brief ready; review the [edit] markers before it), then /debrief; afterwards start the deferred runner fixes. Scheduled: seminar 11:00-12:00, supervisor meeting.

## 2026-11-16 (Mon) (reconstructed Tue morning)
- Task 4 done, gate 1 passed: rotation metric, flood-aware checkpoint rule and the magnitude-weighted arm-2 spec frozen (commit d63497d, not pushed). The cycle-1 canary (job 46731318, Sun) failed only its key-count check (2,289 of 2,307 basins: 18 with no usable target in WY2022). The verifier now accepts explained-missing basins (commit 80a8d99), and the CPU re-verify passed (job 46748201). Evidence: `.scratch_local/wy_rotation_runs_v001/canary_evidence/CANARY_RESULT.md`.
- W&B stage-A tracking written and reviewed, all fixes applied, 148 tests pass, not committed (packet `.scratch_local/review_packets/20261116_wy_rotation_wandb_stage_a/`). WY2026 legacy audit now clips at 2025-09-30, reviewed as "ready with fixes", not committed (packet `.scratch_local/review_packets/20261116_wy2026_q1_legacy_clip/`). Lab-talk deck built (`.scratch_local/lab_talk_20261117/`).
- Waiting on decisions: the G1 erratum, because the frozen spec says "all 2,307 basins", which fails every cycle-1 run (draft `.scratch_local/wy_rotation_metric_spec_v001/ERRATUM_G1_DRAFT.md`); W&B decision-log draft + commit + CPU smoke; WY2026 clip fixes + Q-a/Q-b.
- Tomorrow (Tue 11-17): clear the three decisions and commit; shorten the talk to about 7 min and update its status slide; after the talk, launch the W&B smoke, then cycle 1.
- Scheduled: lab meeting 14:00.

## 2026-11-15 (Sun) (written Mon morning)
- Loader sealing finished and through its review gate (3 independent reviews, fixes applied): under a rotation cycle the loader and raw-space scorer read observations only inside that cycle's windows and write sealing receipts. Commit 6f7310b; packet `.scratch_local/review_packets/20261112_loader_sealing_v001/`; decision-log entry 2026-11-15. All three pre-launch gates now passed.
- Rotation run plumbing, score-blind: 5 cycle configs from P19 120 h, per-epoch checkpoint-year evaluation on all 2,307 basins, a runner with receipts, 41 tests (commit f5a186f). No rotation job launched yet.
- WY2026 data: forcing done; on h2o the curated build (2,752/2,752 basins, auditor PASS, Oct-Dec 2025 identical to v001) and the USGS structural-only audit (2,843/2,843 NCs on the hourly grid) passed. Code committed Mon 08:27 as a17002e (not pushed; earlier commits are pushed).
- Waiting: week-plan task 4 (metric/checkpoint rule + weighted-loss arm spec) must be approved before the canary. `docs/FLASHNH_CURRENT_STATE.md` edits uncommitted and partly stale (says 6f7310b unpushed, run builder not done).
- Tomorrow (Mon 11-16): prepare the lab-meeting status talk (Tue 14:00); start the task-4 spec. Scheduled: busy 15:30-17:30 and 18:30-20:00.

## 2026-11-12 (Thu)
- Task 2 done, gate 2 passed: rotation split contract v001 frozen (`src/baseline/wy_rotation_split.py`, checksum 125b071b…), decision log entries on boundary convention and single-year blanking. Commit dec4f03 (not pushed yet).
- Task 3 done and launched: WY2026 plan `docs/wy2026_data_plan.md` + `--wy-chunks derived`, forcing wrapper, status reporter (commit 8629428, running on h2o). USGS fetch finished: 2,843 NCs, all 4 shards return code 0 (`/data42/omrip/Flash-NH/tmp/stage1_usgs_iv_wy2026_v001`; 33 WY requests with no data, 429 monthly-fallback failures). Forcing: 6/9 months PASS (Jan, Feb, Apr, May, Jul, Aug); Mar, Jun, Sep running (about 4 h each, expected done overnight).
- Started (stretch): loader sealing for rotation runs in `nh_dataset.py`, `nh_raw_space_evaluation.py`, `nh_seed_evaluation.py`, `nh_register.py` + `tests/test_nh_split_sealing.py`. Uncommitted, not reviewed.
- Waiting: the WY2026 post-download steps (USGS flat copy L3, curated builder C1, structural-only auditor C3) wait for the forcing to finish.
- Tomorrow (Sun 11-15, next work day): check that all forcing months PASS, then L3 copy; finish and test the loader sealing and send it to review. Task 4 (weighted-loss + metric spec) not started.
- Scheduled: Fri busy 09:30-12:30 (non-work day); Sun working remotely 09:00-17:00.

## 2026-11-11 (Wed)
- Meeting with Efrat held: [meetings/2026-11-11.md](meetings/2026-11-11.md) (outcome and week plan).
- Task 1 done: water-year rotation recorded in `docs/decision_log.md` (uncommitted). IGS 2027 abstract deadline (Dec 31) added to the week plan.
- Task 2, first half: training-leak fix in `src/baseline/nh_dataset.py`. Training targets that point into excluded years are now blanked and those samples dropped; tests in `tests/test_nh_dataset_multislice.py`; 126 tests pass. Uncommitted.
- Open: keep the blanking for single-year training too (the scaler moves by about 6 hours' worth; old runs no longer reproduce bit-for-bit)? Recommended yes.
- Tomorrow: task 2, second half: rotation split code, then the review packet for the split-design gate.
- Scheduled: busy 08:00-11:00; evening 17:30-18:30.

## 2026-11-10 (Tue)
- Back after the poster and time off. Reviewed the state of the project from the docs and git.
- Set up this journal (`journal/`) and the `/eod`, `/brief`, `/debrief` commands.
- Tomorrow: meeting with Efrat, using brief `meetings/2026-11-11.md`.
