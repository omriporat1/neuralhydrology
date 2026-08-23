# Agent Handoff Rules — Flash-NH

This file is the canonical policy for **multi-agent routing, task handoff, and completion reporting** in Flash-NH.

It applies to ChatGPT, Claude Code, Codex, Copilot, and any later coding/review agent.

Scientific policy lives in the scientific/current-state docs. Git/artifact/evidence policy lives in `docs/repo_policy.md`. Stable remote host facts live in `docs/remote_operations.md`. Agent-native entrypoints (`CLAUDE.md`, `AGENTS.md`) should point here rather than duplicating this policy.

## 1. Scientific authority

Only the user may authorize a change to scientific scope or decision rules.

ChatGPT may help frame, recommend, compare, or route decisions, but it does not authorize scientific changes on the user's behalf.

Explicit user approval is required for changes such as:
- candidate/search-space expansion or alteration;
- sealed-set access;
- target/lead/data-admission/split changes;
- evaluation-rule changes;
- model/hyperparameter promotion decisions;
- materially new compute/search scope.

Task prompts should state the approved decision envelope clearly enough that an implementation agent can distinguish ordinary technical execution from a new scientific decision.

## 2. General handoff principle

Agents should synchronize through concrete project state, not through assumptions about another agent's conversation.

Preferred handoff objects are:

- exact Git commit or clearly identified working-tree state;
- task objective and scientific scope;
- relevant current-state/scientific docs;
- validation/test results;
- compact evidence paths;
- unresolved questions requiring a decision.

A clean Git commit is the preferred cross-agent implementation/review boundary when practical, but a dirty-tree review is allowed when explicitly intended and clearly scoped.

## 3. Preferred agent roles

| Agent | Preferred role |
|---|---|
| **ChatGPT** | Scientific strategy, experiment design, interpretation, workflow/routing control, prompt design, cross-agent review |
| **Claude Code** | Difficult repo integration, multi-file implementation, complex debugging, Slurm/HPC execution, remote evidence workflows |
| **Codex** | Bounded implementation, focused tests, code archaeology, independent code review, mechanical/refactoring work under a frozen design, overflow engineering |
| **Copilot** | Micro-edits, autocomplete, tiny local refactors/fixes |

These are defaults, not rigid capability boundaries.

Use the agent that best matches the task while minimizing duplicated expensive work.

Cross-model review is most useful when it adds independent value, for example:

- Claude implementation -> Codex review of the exact commit/diff;
- Codex implementation -> Claude review of risky integration/HPC aspects;
- ChatGPT reviews scientific framing/interpretation regardless of implementation agent.

Do not ask a second agent to reimplement work merely for redundancy.

## 4. Bounded tasks and decision envelopes

A task is suitable for bounded implementation/review when its scientific objective and acceptance criteria are already frozen enough that the agent does not need to invent a new scientific choice.

A task handoff should make clear, when relevant:
- allowed data/split scope;
- whether candidate/search-space changes are forbidden;
- whether HPO/search launch is authorized;
- whether promotion/selection decisions are authorized;
- allowed compute budget/fidelity;
- remote execution mode.

If execution reveals a genuine scientific ambiguity, return it to the user rather than silently expanding the task.

## 5. Interface / Consumer Contract Gate

Before substantial cross-component production integration, reuse of a mature
subsystem by a new consumer, extraction of a reusable component, or
evidence/tracking/result plumbing where scientific validity depends on the
interface, the task must identify, when relevant:

1. Producer.
2. Intended consumer.
3. Required inputs.
4. Required outputs / success receipt.
5. Authority for each scientifically or operationally meaningful fact.
6. Failure/incomplete semantics.
7. One vertical synthetic/integration test proving the consumer can use the
   producer contract.

**Core rule.** "The information exists internally" is not sufficient. The
intended consumer must be able to obtain every authoritative fact it needs
through the defined interface or another explicitly-authoritative shared
artifact/helper. A higher layer must not silently reconstruct lower-layer
scientific/execution facts using an ad hoc parallel interpretation merely
because the lower layer does not expose them. If the missing information
belongs to the lower layer's authority, prefer repairing/exposing the generic
lower-level result contract over working around it in the higher layer.

**Reusable-extraction review.** When reviewing an extracted/reusable
component, do not only verify code motion, backward compatibility, and
sufficient inputs. Also verify whether the stated new consumer receives all
outputs/evidence it needs to use the component safely. A reusable extraction
may be behaviorally correct for its old caller while still having an
incomplete consumer-facing result contract.

**Scoped closure language.** Avoid broad closure claims such as "tracking is
closed," "evidence is settled," or "execution is qualified" when multiple
distinct contracts exist. Prefer scoped status language, for example: W&B
telemetry — CLOSED; execution provenance — CLOSED; consumer result contract —
OPEN. This is not a heavy formal status system; it exists only to prevent
confidence in one contract from being silently transferred to another.

**Facts vs. interpretation.** Lower layers should expose authoritative facts;
higher layers should interpret those facts according to their scientific
contract. For example: an execution layer reports what physically executed; a
campaign/scientific layer decides whether that execution is scientifically
valid and what objective it implies; a telemetry layer reports/displays the
result. This is a general project rule, not specific to any one subsystem.

## 6. Commit and push authority

Generated-artifact policy is defined in `docs/repo_policy.md`.

Agents must not push to GitHub or another remote unless the user explicitly authorizes the push.

Agents must not commit automatically unless the task explicitly authorizes a commit.

Completing an implementation does not itself authorize either commit or push.

## 7. Expensive downloads and compute

Agents must not initiate substantial new external downloads, large data acquisition, or materially new compute commitments unless the task explicitly authorizes them.

Once an approved run is launched, an agent may continue through a pre-authorized bounded continuation envelope when specified by the task.

Ordinary technical recovery inside the approved design is allowed; new scientific scope or material new compute/cost requires escalation.

## 8. Output/scratch locations

Canonical location rules are defined in `docs/repo_policy.md`.

Do not invent a new output convention inside a task prompt unless the task genuinely requires one.

## 9. Completion-report convention

Task-completion replies should normally include:

1. **Files changed** — created, modified, or deleted tracked/source files.
2. **Validation commands run** — exact relevant commands and concise results.
3. **Output/evidence paths** — repo-relative or full paths to generated outputs/evidence needed for review.
4. **Git status** — concise, preferably scoped to the Flash-NH project; summarize pre-existing unrelated untracked clutter rather than dumping it.
5. **Commit hash** — only when a commit was actually made.
6. **Anomalies / decisions needed** — include when there is anything unexpected, unresolved, or requiring scientific/user judgment.

For substantial integration tasks (see §5), also explicitly surface:

7. **Consumer contract status** — whether the consumer result/evidence contract is CLOSED or still PARTIAL.
8. **Implicit/reconstructed facts** — any authoritative facts the consumer still has to infer rather than obtain from the interface.
9. **Vertical synthetic/integration test result** — pass/fail and where to inspect it.

This is a default reporting contract, not a ceremonial requirement.

For a tiny read-only review, report only what is relevant. For a complex failure, security issue, or formal scientific closure, expand the report as needed.

Do not paste large diffs, full logs, or large tables into the completion message when the underlying artifact can be inspected directly.

## 10. Prompt/handoff template

Use/adapt this lightweight structure when helpful:

```text
Project: Flash-NH
Task: <one-sentence objective>

Current state:
- Git commit / working-tree state: <...>
- Current milestone/source of truth: <...>

Read first:
- <only relevant files/docs>

User-approved decision envelope:
- Data/split scope: <...>
- Candidate/search-space changes: <allowed/forbidden>
- HPO/search launch: <allowed/forbidden>
- Promotion/selection decisions: <allowed/forbidden>
- Compute/fidelity budget: <...>
- Remote execution mode: <local only / detached / bounded continuation / interactive debug / etc.>

Acceptance:
- <tests/evidence/result required>

Integration contract (optional; use for substantial cross-component tasks — see §5):
- Producer:
- Consumer:
- Required inputs:
- Required output/success receipt:
- Authority boundary:
- Failure semantics:
- Vertical synthetic test:

When done:
- follow docs/agent_handoff_rules.md completion-report convention
```

Do not repeat stable project rules that already live in `CLAUDE.md`, `AGENTS.md`, `docs/repo_policy.md`, or `docs/remote_operations.md`.
