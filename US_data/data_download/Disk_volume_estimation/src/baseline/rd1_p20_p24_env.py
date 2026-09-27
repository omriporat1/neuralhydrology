"""Environment sanitization for the RD1 P20-P24 fixed chain.

Root-cause context (see the RD1 P20 forensic finding this replacement is
built from): the retired self-propagating controller ran a Slurm transition
job that called ``wandb.Api()`` (which can start a local wandb-core
background service and record its address in ``WANDB_SERVICE``), then
``sbatch``'d the next agent job with an implicit ``--export=ALL``, so the
child job inherited ``WANDB_SERVICE`` pointing at a socket that no longer
existed by the time the child ran -- ``wandb agent`` then failed with
"Failed to connect to service on socket" before creating any run.

The fixed chain's structural fix is "no job submits another job" (each
proposal's fixed Slurm dependency is created once, up front, from a clean
login shell -- see ``scripts/submit_rd1_p20_p24_chain.sh``). This module is
the defense-in-depth layer inside each job: it strips every inherited
``WANDB_*`` variable unconditionally before the in-job ``wandb agent``
subprocess ever starts, so this is safe even if a future submission path
accidentally forwards more than intended.

Correction (Phase A.1 review, 2026-09-27): this module previously also
defined ``resolve_wandb_api_key()`` / ``CredentialError``, a from-scratch
``.netrc`` credential resolver. It was never called from any execution path
(dead code) and duplicated logic the maintained agent launcher already
performs itself. Removed rather than wired in. The one live credential
dependency for this chain is:
``scripts/run_sweep_v2_six_axis_wandb_agent_moriah.sbatch`` (the
``AGENT_LAUNCHER`` this chain's job driver invokes as a subprocess): if
``WANDB_API_KEY`` is not already non-empty in the launcher's own
environment, it resolves the key itself by shelling out to
``python -c 'import netrc; netrc.netrc().authenticators("api.wandb.ai")'``
against ``$HOME/.netrc``, and refuses with a FATAL message before any W&B
contact if that also fails (see that file's own comments immediately above
its ``WANDB_API_KEY`` resolution block). Do not reimplement this here;
``sanitize_environment`` below guarantees no stale ``WANDB_API_KEY`` (or any
other ``WANDB_*`` value) reaches that launcher from this job's own
submitting environment, so every invocation always exercises that same,
already-working resolution path fresh.
"""
from __future__ import annotations

from typing import Mapping

__all__ = [
    "sanitize_environment",
]


def sanitize_environment(env: Mapping[str, str], *, keep: "frozenset[str]" = frozenset()) -> dict[str, str]:
    """Return a copy of ``env`` with every ``WANDB``-prefixed variable
    removed, except names explicitly listed in ``keep``.

    ``keep`` defaults to empty: today there is no stable W&B identity
    setting that must survive from the submitting shell into the job (see
    module docstring). It exists as an explicit, reviewed override point
    rather than a silent blanket strip, so a future genuinely-stable
    setting can be added here deliberately instead of by weakening this
    function's default behavior.

    Matches on the variable name only (case-sensitive, exact prefix
    ``WANDB`` or ``WANDB_...``) -- never inspects values, so this is safe to
    call on an environment that might otherwise contain a credential.
    """
    cleaned = {}
    for name, value in env.items():
        if name in keep:
            cleaned[name] = value
            continue
        if name == "WANDB" or name.startswith("WANDB_"):
            continue
        cleaned[name] = value
    return cleaned
