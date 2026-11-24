"""Internal knowledge models; the harness boundary remains defined in harness/contract.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID


@dataclass(frozen=True)
class EntityReference:
    """A source-bound mention of another canonical entity."""

    entity_id: str | None
    label: str
    quote: str


@dataclass(frozen=True)
class PendingStatement:
    """Source-quoted candidate retained when it cannot become a Fact yet."""

    id: UUID
    evidence_id: UUID
    entity: str
    attribute: str
    qualifiers: dict[str, Any]
    value: str
    unit: str | None
    quote: str
    issues: tuple[str, ...]
    table_header: str = ""
    heading_path: tuple[str, ...] = ()
    raw_date_notes: tuple[str, ...] = ()
    effective_from: date | None = None
    effective_to: date | None = None
    date_quote: str | None = None
    identity_signals: tuple[tuple[str, str, str], ...] = ()
    entity_refs: tuple[EntityReference, ...] = ()


@dataclass
class Fact:
    """Stable identity of a fact, independent of its value, time, and source.

    key is derived from normalized entity, attribute, and qualifiers.
    Qualifiers distinguish terms, balance tiers, model years, and other conditions.
    """

    id: UUID
    key: str
    entity: str
    attribute: str
    qualifiers: dict[str, Any]


@dataclass
class FactObservation:
    """One source's observed value for a fact.

    Observation dates come from the associated harness Snapshot.taken_on,
    reached through evidence_id and Evidence.snapshot_id.
    Effective dates are populated only when supported by the source; a later
    observation does not automatically establish the prior value's end date.
    Snapshot membership is obtained through evidence_id. entity_refs are
    occurrence-specific, source-quoted references and never change Fact.key.
    context stores occurrence-specific source structure that helps interpret
    the observation; it is not part of the stable Fact identity.
    """

    id: UUID
    fact_id: UUID
    value: str
    unit: str | None
    effective_from: date | None
    effective_to: date | None
    evidence_id: UUID
    entity_refs: tuple[EntityReference, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalResolution:
    fact: Fact
    observations: tuple[FactObservation, ...]
    status: str  # resolved / conflicting / ambiguous / not_found
    snapshot_id: str | None
    reason: str
    # Applicability of returned observations, independent of value agreement.
    # latest_supported means a dated assertion is the latest available source
    # assertion for the selected snapshot/date, but no end date is stated.
    temporal_certainty: Literal[
        "explicit", "latest_supported", "snapshot_only", "uncertain"
    ] | None


@dataclass(frozen=True)
class FactBuildAudit:
    """Outcome of building a candidate, including reuse of existing records.

    Audit metadata does not participate in fact or observation identity.
    A skipped candidate has no record IDs and must retain a reason in issues.
    """

    candidate_id: str | None
    evidence_id: UUID
    fact_id: UUID | None
    observation_id: UUID | None
    issues: tuple[str, ...]
    status: Literal["built", "skipped"]

    def __post_init__(self) -> None:
        if self.status == "built":
            if self.fact_id is None or self.observation_id is None:
                raise ValueError("Built audit requires fact_id and observation_id")
        elif self.status == "skipped":
            if self.fact_id is not None or self.observation_id is not None:
                raise ValueError("Skipped audit must not reference built records")
            if not self.issues:
                raise ValueError("Skipped audit requires a reason in issues")
        else:
            raise ValueError(f"Unknown fact build status: {self.status}")


@dataclass
class Evidence:
    """A supporting content block appearing at one URL in one snapshot.

    text retains the context needed to interpret a fact, including table headers
    and relevant footnotes. content_hash hashes normalized text for deduplication;
    matching hashes do not merge distinct source or snapshot appearances.
    source_type describes authority, e.g. official_rate_table or calculator.
    source_title preserves the original Page.title as source metadata.
    """

    id: UUID
    snapshot_id: str
    source_url: str
    source_title: str
    source_type: str
    in_recommendations: bool
    text: str
    content_hash: str
    fetched_at: datetime
