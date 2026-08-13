"""Minimal, execution-facing campaign declaration (Stage 1, Scope C of the
Sequence-Length-A minimum-viable-infrastructure task).

:class:`CampaignSpec` is deliberately narrow: only the fields a prospective
range-characterization campaign (starting with Sequence-Length-A) actually
needs in order to be preparation-ready. It is NOT a general-purpose HPO
framework -- no multi-axis grids, no dynamic-input profiles, no arbitrary
config-patch mechanism, no generic scientific-analysis configuration. The
scientific rationale for a campaign's design belongs in documentation/tests
(docs/decision_log.md and the campaign's own closure script/tests), never
hidden inside this dataclass or inferred at runtime.

Constructing a ``CampaignSpec`` reserves its candidates' run_ids in
:mod:`campaign_registry` immediately (via
``register_prospective_campaign_run_ids``) -- a collision with any
historical or other prospective campaign's run_ids raises
:class:`campaign_registry.CampaignRegistryError` at construction time,
before any config is ever generated.
"""

from dataclasses import dataclass, field

from .campaign_registry import register_prospective_campaign_run_ids
from .pilot_lead06_config import PilotRunSpec

__all__ = [
    "CampaignSpecError",
    "CampaignSpec",
]


class CampaignSpecError(Exception):
    """Raised for a structurally invalid CampaignSpec."""


@dataclass(frozen=True)
class CampaignSpec:
    # Short, stable identity used both as this campaign's human-readable
    # label and as its campaign_registry.register_prospective_campaign_
    # run_ids "source" key (e.g. "Sequence-Length-A"). Never reused across
    # two structurally different campaigns.
    name: str
    # Free-form version string for this campaign's declaration (e.g.
    # "v001") -- bumped if the candidate set or fixed hyperparameters ever
    # change, mirroring the pilot policy YAML's own policy_name convention.
    version: str
    # Name of the single scalar axis this campaign varies across its
    # candidates (e.g. "seq_length") -- documentation/provenance only, never
    # used to infer which PilotRunSpec field to read; each candidate's
    # PilotRunSpec already carries its own explicit override value.
    varied_axis: str
    # Explicit candidate run_id -> PilotRunSpec mapping. No Cartesian
    # expansion, no generation from a grid -- exactly the agreed candidates,
    # spelled out by hand, exactly like every existing closure script's
    # *_RUN_SPECS mapping (e.g. EMBEDDING_DROPOUT_A_RUN_SPECS).
    candidates: "dict[str, PilotRunSpec]"
    # This campaign's approved, fixed target epoch (mirrors e.g.
    # EMBEDDING_DROPOUT_A_MAX_TARGET_EPOCH) -- never a caller-supplied value
    # at launch time.
    max_target_epoch: int
    # W&B tracking policy contract: whether a real training launch under
    # this campaign must hard-fail if tracking initialization fails/resolves
    # to a null sink (see pilot_tracking.init_pilot_tracking_run's
    # require_tracking parameter), unless an explicit human waiver is given
    # by the caller. True (strict) is this campaign family's established
    # convention (Hidden-size-A, Embedding-Dropout-A).
    require_tracking: bool = True
    # Reviewed offline W&B policy path this campaign defaults to, or None to
    # defer entirely to the caller (e.g. a CLI's own default). Never a
    # bare/unreviewed policy.
    wandb_policy_path: "str | None" = None
    # Run_ids of already-existing, already-trained/evidenced runs (from this
    # or another campaign) kept as read-only, status-only reproducibility
    # comparators -- never a member of `candidates`, never trainable or
    # reconfigurable through this campaign. Empty by default.
    comparator_run_ids: "tuple[str, ...]" = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.candidates:
            raise CampaignSpecError(f"CampaignSpec {self.name!r} declares no candidates")
        for run_id, run_spec in self.candidates.items():
            if not isinstance(run_spec, PilotRunSpec):
                raise CampaignSpecError(
                    f"CampaignSpec {self.name!r}: candidate {run_id!r} is not a PilotRunSpec "
                    f"(got {type(run_spec).__name__})"
                )
            if run_spec.run_id != run_id:
                raise CampaignSpecError(
                    f"CampaignSpec {self.name!r}: candidates key {run_id!r} does not match its "
                    f"PilotRunSpec.run_id {run_spec.run_id!r}"
                )
        if not isinstance(self.max_target_epoch, int) or isinstance(self.max_target_epoch, bool) \
                or self.max_target_epoch <= 0:
            raise CampaignSpecError(
                f"CampaignSpec {self.name!r}: max_target_epoch must be a positive int, "
                f"got {self.max_target_epoch!r}"
            )
        overlap = set(self.candidates) & set(self.comparator_run_ids)
        if overlap:
            raise CampaignSpecError(
                f"CampaignSpec {self.name!r}: comparator_run_ids overlaps with candidates: "
                f"{sorted(overlap)} -- a comparator must never also be a trainable candidate"
            )
        # Reserve this campaign's own run_ids against every historical and
        # other prospective campaign's run_ids -- fails loudly on collision.
        # comparator_run_ids are NOT registered here: they reference
        # already-reserved run_ids belonging to another (historical)
        # campaign by design, so registering them again under this
        # campaign's name would itself look like a false-positive collision.
        register_prospective_campaign_run_ids(self.name, tuple(self.candidates.keys()))
