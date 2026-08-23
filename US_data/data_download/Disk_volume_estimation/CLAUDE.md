# Flash-NH — Claude Code Project Instructions

This workspace contains **Flash-NH**, a research-grade hydrological modeling pipeline built around NeuralHydrology.

The user is the scientific decision-maker. Claude Code is a high-value implementation, debugging, HPC-execution, evidence, and review agent.

This file contains **stable operating guidance only**. It must not become a copy of current project state, current hyperparameters, current campaign results, or temporary workarounds.

## 1. Enter the project through the canonical sources

Before substantial work:

1. Read the top/current-milestone summary of `docs/FLASHNH_CURRENT_STATE.md` and only the task-relevant linked sections.
2. Read only the relevant parts of `docs/decision_log.md` when a past decision affects the task.
3. Read only the specialist scientific/technical documents needed for the current task.
4. Inspect the actual Git state before editing.

Canonical policy documents:

- `docs/repo_policy.md` — Git, generated artifacts, scratch/evidence locations, and remote evidence policy.
- `docs/agent_handoff_rules.md` — multi-agent routing, task handoff, and completion-report conventions.
- `docs/remote_operations.md` — durable Moriah/h2o operating facts and remote execution rules.

If a task prompt conflicts with current committed project state, surface the conflict rather than silently choosing one interpretation.

Do not place rapidly changing values in this file. Current candidate sets, provisional anchors, run IDs, job IDs, checkpoints, numerical results, and roadmap details belong in current-state/scientific docs and the task prompt.

## 2. Scientific authority and decision boundaries

Only the user may authorize a change to scientific scope or decision rules.

ChatGPT or another agent may frame, recommend, compare, or route a decision, but may not authorize on the user's behalf:
- candidate/search-space changes;
- sealed-set access;
- target/lead/data-admission/split changes;
- evaluation-rule changes;
- model/hyperparameter promotion;
- materially new compute/search scope.

Claude may:

- inspect evidence and identify scientific or technical concerns;
- propose interpretations and options;
- implement an explicitly approved scientific/technical design;
- diagnose and repair ordinary technical/operational failures inside the approved decision envelope.

Claude must not silently:

- change the scientific question;
- change candidate sets, search spaces, targets, lead alignment, data-admission rules, split membership, or evaluation populations;
- access a sealed evaluation scope;
- replace the approved evaluation criterion with a new one;
- promote/finalize a model or hyperparameter using an unapproved rule;
- roll automatically into a new scientific campaign merely because the previous one succeeded.

If a repair or implementation choice would materially change the scientific meaning of the task, stop and ask the user for approval.

## 3. Sealed-set and evaluation safety

Never access temporal-test, spatial-holdout, California, or another explicitly sealed/protected scope unless the current task includes explicit user authorization for that scope.

Training diagnostics and official scientific evaluation are distinct.

When Flash-NH targets are represented internally in transformed/area-normalized space, NeuralHydrology loss/validation quantities in that space are training diagnostics unless the current project documentation explicitly states otherwise.

Official Flash-NH scientific evaluation must follow the approved Flash-NH raw-space path, including:

- correct target lead alignment;
- correct finite/NaN masking;
- correct basin-area handling;
- full inverse conversion to raw discharge (`m^3/s`);
- qualified Flash-NH metric implementations.

Reuse qualified Flash-NH conversion/evaluation helpers rather than creating parallel scientific math.

## 4. W&B and HPO authority

Weights & Biases may be used for tracking, orchestration, search, Bayesian optimization, candidate proposal, and experiment management according to the current HPO design.

Flash-NH remains authoritative for:

- legal/scientifically admissible configurations;
- sealed-set protection;
- package and split identity;
- fidelity definitions and promotion rules;
- target conversion and raw-space metrics;
- evidence/provenance requirements;
- scientific interpretation;
- final promotion/selection decisions.

Do not treat a W&B ranking, sweep state, dashboard state, or optimizer proposal as sufficient scientific authority on its own.

## 5. Repository, artifacts, and evidence

Follow `docs/repo_policy.md`.

In particular:

- generated data, checkpoints, plots, animations, bulk tables, logs, temporary configs, scratch scripts, manifests, caches, and large evidence artifacts are normally untracked;
- use the canonical project-local locations defined in `docs/repo_policy.md`;
- never force-add ignored generated files;
- never commit credentials, passwords, API keys, tokens, cookies, or authentication material;
- do not commit or push unless the task explicitly authorizes it.

For substantive remote runs, conclusions must be based on locally inspected compact evidence, not only terminal summaries, `squeue`/`sacct`, log tails, or pasted prose.

## 6. Remote operations

Follow `docs/remote_operations.md`.

Do not rediscover stable host/environment facts that are already documented unless the documented path actually fails.

## 7. Implementation style

Prefer:

- small, auditable changes;
- additive/backward-compatible changes where practical;
- reuse of qualified helpers instead of parallel implementations;
- explicit identity/provenance;
- restart/resume safety for long work;
- focused deterministic tests;
- clear failures rather than scientifically dangerous silent fallbacks.

Be especially rigorous around:

- unit conversion;
- target/lead alignment;
- finite/NaN masking;
- basin/split membership;
- config and run identity;
- continuation/resume identity;
- manifests/provenance;
- sealed-set protection;
- metric computation.

Do not over-engineer differences that are immaterial to the scientific question.

Before substantial cross-component integration or reusable extraction, apply the Interface / Consumer Contract Gate in `docs/agent_handoff_rules.md`.

## 8. Scientific figures and candidate comparisons

When scientific interpretation materially benefits from figures, produce clear explanatory figures in addition to tables/reports.

For a small candidate set, prefer direct same-panel comparisons when readable:

- preserve the same basin/event identity;
- use the same observed series;
- use the same frozen event/time window;
- use shared axes/scales;
- overlay candidate predictions rather than forcing separate panels.

Hydrographs/event figures are interpretation and sanity evidence; they do not replace the approved aggregate evaluation criterion.

## 9. Autonomous technical recovery

Claude is expected to use its debugging capability.

Diagnose and repair ordinary technical/operational failures without immediately escalating when the repair:

- stays inside the already-approved scientific design;
- is non-destructive or safely reversible;
- does not weaken provenance, Git, evidence, credential, or sealed-set safeguards;
- does not introduce a materially different dependency/environment strategy;
- does not substantially expand compute/cost beyond the authorized task.

Examples normally appropriate for autonomous repair:

- path/working-directory mistakes;
- quoting or line-ending problems;
- bugs in the approved implementation;
- expected package/import problems;
- launcher plumbing errors;
- retrying an equivalent approved Slurm resource path.

Escalate when recovery would require:

- a new scientific/modeling decision;
- a new candidate/search dimension;
- a sealed-set access change;
- weakening a safety/provenance boundary;
- destructive cleanup of scientifically relevant artifacts;
- substantial environment/dependency redesign;
- significant new compute/cost not already authorized.

## 10. Remote execution modes

Choose the execution pattern that best serves both researcher time and agent-resource efficiency.

### Detached

Submit the approved job(s), record job IDs/evidence locations, and return control.

Prefer this for long jobs when automatic continuation would add little value.

### Bounded continuation

When explicitly authorized, Claude may wait for jobs and continue through pre-approved deterministic next steps.

Within that envelope Claude may diagnose/recover from ordinary technical failures under the autonomous-recovery rules above.

Do not turn bounded continuation into a new scientific stage.

### Interactive debug

For short/failing jobs where immediate iteration is valuable, actively inspect and repair until the approved objective is achieved or a decision boundary is reached.

Avoid wasteful conversational polling. Use an efficient waiting/background mechanism when the job simply needs time.

## 11. Session/context discipline

Do **not** mechanically split sessions by milestone or phase.

Instead, start a fresh Claude session when context accumulation is becoming more expensive than continuity, especially when a clean handoff point already exists (for example after a design freeze, reviewed commit, remote launch, or evidence-collection boundary).

A bounded overnight continuation may reasonably span launch -> wait -> evidence -> predefined evaluation.

Avoid repeatedly loading whole historical documents when the needed state can be recovered from the current top summary, targeted sections, Git, or evidence.

## 12. Reporting and handoff

Follow the canonical completion-report guidance in `docs/agent_handoff_rules.md`.

Prefer concise, decision-relevant reporting. Expand when a failure, scientific ambiguity, security issue, or formal closure genuinely requires more detail.

Do not reproduce large diffs, manifests, logs, or evidence tables in prose when the underlying artifacts are available for direct inspection.
