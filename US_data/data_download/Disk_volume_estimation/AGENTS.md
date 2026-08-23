# Flash-NH — Agent Instructions (Codex and other coding/review agents)

This directory contains **Flash-NH**, a research-grade hydrological modeling pipeline built around NeuralHydrology.

The user is the scientific decision-maker.

This file contains stable project-entry guidance for Codex and other coding/review agents. Current scientific state, current campaign values, numerical results, and temporary workarounds belong in the project documentation and task prompt.

## 1. Establish current state before acting

Before substantive work:

1. Read the top/current-milestone summary of `docs/FLASHNH_CURRENT_STATE.md` and only the task-relevant linked sections.
2. Inspect the exact Git state or commit named by the task.
3. Consult only the relevant parts of `docs/decision_log.md` when prior decisions matter.
4. Read only the specialist docs needed for the requested work.

Canonical policy documents:

- `docs/repo_policy.md` — Git, generated artifacts, scratch/evidence locations, remote evidence.
- `docs/agent_handoff_rules.md` — multi-agent routing, handoff, completion-report conventions.
- `docs/remote_operations.md` — stable Moriah/h2o operating facts.

If the task conflicts with current committed state, report the conflict rather than silently choosing an interpretation.

## 2. Scientific authority and boundaries

Only the user may authorize a change to scientific scope or decision rules.

ChatGPT or another agent may frame, recommend, compare, or route a decision, but may not authorize on the user's behalf:
- candidate/search-space changes;
- sealed-set access;
- target/lead/data-admission/split changes;
- evaluation-rule changes;
- model/hyperparameter promotion;
- materially new compute/search scope.

Do not silently change:

- the scientific question;
- experiment candidate/search spaces;
- split membership or data-admission policy;
- target definitions or lead alignment;
- evaluation metrics;
- fidelity/promotion rules;
- sealed evaluation scope.

Never access temporal-test, spatial-holdout, California, or another sealed scope unless the task includes explicit user authorization for that scope.

If implementation exposes a genuine scientific ambiguity, return it to the user for a decision rather than inventing a scientific interpretation.

When transformed/area-normalized targets are used, NeuralHydrology transformed-space loss/validation quantities are training diagnostics unless current project documentation explicitly states otherwise.

Official Flash-NH scientific evaluation must follow the approved raw-space path, including basin-area conversion, lead alignment, masking, and qualified project metric implementations.

Reuse qualified scientific helpers instead of reimplementing the math.

## 3. W&B / optimization authority

W&B may be used for tracking, orchestration, search, Bayesian optimization, and candidate proposal according to the current HPO design.

Flash-NH remains authoritative for:

- legal/scientifically admissible configurations;
- sealed-set protection;
- package/split identity;
- fidelity and promotion rules;
- target conversion and raw-space scientific metrics;
- evidence/provenance;
- final scientific interpretation and selection.

A W&B ranking, sweep state, or optimizer proposal alone is not scientific authority.

## 4. Repository and generated artifacts

Follow `docs/repo_policy.md`.

Do not force-add ignored generated artifacts.

Never commit credentials or authentication material.

Do not automatically commit or push unless the task explicitly authorizes it.

## 5. Evidence and reproducibility

Scientific conclusions from substantive remote runs must be grounded in compact inspected evidence rather than terminal summaries alone.

Preserve the identities required by the relevant task, including:

- Git/code state;
- run/config identity;
- package/split identity;
- command/job provenance;
- checksums/manifests when required;
- scientifically relevant result/audit tables.

Large canonical data products normally remain remote.

## 6. Role in multi-agent work

Treat the repository, Git commit, tests, project docs, and evidence as the synchronization layer between agents.

Do not assume access to another agent's conversation.

### Git/project scope

Flash-NH lives under this project-relative path inside the enclosing repository:

`US_data/data_download/Disk_volume_estimation/`

Handoffs and reviews should identify both:
- the exact commit or working-tree state;
- the Flash-NH project-relative scope/diff being reviewed.

If Git access fails because of environment-level trust/ownership protection (for example Git's `safe.directory` / "dubious ownership" check), report the issue. Do **not** mutate global Git configuration merely to make the task proceed.

A per-command, narrowly scoped read-only trust override is acceptable when needed for inspection, provided it does not alter persistent/global configuration.

### When implementing

- work from the explicit task contract and exact Git state;
- keep scope bounded;
- use focused tests;
- surface scientific decisions instead of inventing them.

### When reviewing another agent's work

- inspect the exact commit/diff requested;
- check the task contract, scientific boundaries, tests, and provenance;
- distinguish correctness/safety issues from stylistic preferences;
- do not rewrite an acceptable implementation merely because you prefer a different style.

A clean Git commit is the preferred cross-agent handoff boundary when practical.

## 7. Initial Codex role

Until intentionally broadened, Codex is primarily used for:

- bounded implementation tasks;
- focused tests;
- code archaeology;
- mechanical/refactoring work with a frozen design;
- independent review of Claude-generated changes;
- overflow engineering when Claude allocation is scarce.

HPC execution is **not** Codex's default role at first.

If a Codex task touches remote/HPC code or scripts, follow `docs/remote_operations.md` and do not perform remote execution unless the task explicitly authorizes it.

## 8. Implementation style

Prefer:

- small auditable patches;
- additive/backward-compatible changes where practical;
- reuse over parallel scientific implementations;
- focused deterministic tests;
- explicit provenance and continuation identity;
- clear failures over scientifically dangerous silent fallback.

Prioritize tests around:

- unit conversion;
- target/lead alignment;
- masking;
- split membership;
- config/run identity;
- continuation safety;
- manifests/provenance;
- sealed-set protection;
- scientific metric calculations.

Do not over-engineer immaterial differences.

Before substantial cross-component integration or reusable extraction, apply the Interface / Consumer Contract Gate in `docs/agent_handoff_rules.md`.

## 9. Autonomous repair

Diagnose and fix ordinary technical failures when the repair stays inside the approved task/scientific envelope and does not weaken safety/provenance or materially expand environment/compute scope.

Escalate rather than silently changing:

- scientific design;
- sealed-set access;
- candidate/search dimensions;
- evidence/safety requirements;
- major dependency/environment strategy;
- substantial compute/cost commitments.

## 10. Reporting

Follow the completion-report convention in `docs/agent_handoff_rules.md`.

Keep reports compact and decision-relevant unless the task/anomaly warrants more detail.
