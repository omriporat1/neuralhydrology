# Daily log (newest first)

Format: date, 3-5 lines (did / output links / tomorrow). Written by `/eod`.

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
