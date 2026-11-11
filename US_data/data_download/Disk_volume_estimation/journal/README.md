# PhD work journal

A light weekly system anchored on the Wednesday meeting with Efrat. Claude drafts everything; Omri corrects. Missing a day breaks nothing.

## The three commands

| When | Command | What Claude does | What you do |
|---|---|---|---|
| End of a work day (optional) | `/eod` | Drafts 3-5 lines in `log.md` from the day's commits, new output folders and anything you say | Fix the draft, say what is scheduled tomorrow (~2 min) |
| Tuesday, day before the meeting | `/brief` | Builds `meetings/<meeting-date>.md`: what was done, what to show, decisions needed, proposed goals. Rebuilds missing days from git and output folders | Read and edit (~15 min) |
| Wednesday, after the meeting | `/debrief` | Reads the transcript, records what was agreed, turns it into next week's tasks and a Gantt | Drop the transcript in, give the week's constraints (conferences, days off, meetings), confirm (~10 min) |

A scheduled task drafts the brief automatically on Tuesday afternoon, so it is waiting for you.

## Files

```
journal/
  README.md                 this page
  log.md                    daily entries, newest first
  meetings/YYYY-MM-DD.md    one file per meeting: brief -> outcome -> week plan + Gantt
.scratch_local/meetings/YYYY-MM-DD/   transcript / recording / notes (gitignored, never committed)
```

## Rules that keep it alive

1. **The weekly loop is the core.** Daily entries are a bonus. Skipping them is fine: `/brief` reconstructs the week.
2. **Never a blank page.** Claude always writes the first draft.
3. **Short beats complete.** A daily entry is at most 5 lines. A brief fits on two screens. A Gantt has at most ~6 bars.
4. **Honest carry-over.** Each brief marks last week's tasks as done, partly done or not done, without judgement. Not done is information, not failure.
5. **Supervisor language.** Briefs are written for Efrat: plain words, what a number means, no internal gate jargon.
6. **Decisions go to their home.** Agreed scientific decisions are proposed as entries for `docs/decision_log.md` and added only after Omri confirms.
7. **Four-week check** (first one: 2026-12-09): keep what was used, drop what was not.

## Settings (change freely)

- Meeting day: Wednesday (Zoom). Working days: Sunday-Thursday.
- Transcript source: Zoom transcript (`.vtt`/`.txt`), Zoom AI summary, or typed notes. Audio alone cannot be read by Claude: export a transcript.
- Commits: journal files are small markdown and may be committed, but only when Omri asks (repo policy §11).
- Calendar: Google Calendar is connected (read-only). `/eod` and `/debrief` use it for tomorrow's and the week's blocked time, and record only times and generic labels, never personal event names.
- Review gates: at a design freeze, before a new campaign, or before recording a scientific conclusion, run `/review-packet` for an independent review (fresh Claude reviewer and/or ChatGPT). See `docs/agent_handoff_rules.md` §3.2.
