"""Internal query-time grouping and selection models, independent of stored facts.

These models describe proposals and validation outcomes. They do not establish
semantic entailment, source precedence, or observation eligibility themselves.
"""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SelectionStatus = Literal['preferred', 'distinct', 'ambiguous', 'insufficient']
GroupRelation = Literal['competing', 'distinct', 'complementary']


class SupportingQuote(BaseModel):
    model_config = ConfigDict(extra='forbid')
    evidence_id: str
    quote: str = Field(min_length=1)


@dataclass(frozen=True)
class EvidenceGroup:
    """Input candidates for one requested field; IDs must later be scope-checked.

    Prose-only groups can have no observations, but must supply Evidence IDs.
    'competing' includes potential competition; it does not assert a conflict.
    """
    group_id: str
    requested_field: str
    observation_ids: tuple[str, ...]
    relation: GroupRelation
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.group_id.strip() or not self.requested_field.strip():
            raise ValueError('Group identity and requested field must be nonempty')
        if self.relation not in {'competing', 'distinct', 'complementary'}:
            raise ValueError('Unsupported group relation')
        if not self.observation_ids and not self.evidence_ids:
            raise ValueError('A group requires Observation or Evidence candidates')


class GroupSelection(BaseModel):
    """Model proposal, never consumed by the renderer without validation."""
    model_config = ConfigDict(extra='forbid')
    group_id: str = Field(min_length=1, pattern=r'\S')
    status: SelectionStatus
    observation_ids: list[str] = Field(default_factory=list)
    supporting_quotes: list[SupportingQuote] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def require_selection_support(self):
        if self.status in {'preferred', 'distinct'} and not self.supporting_quotes:
            raise ValueError('A proposed answer requires supporting quotes')
        return self

    @property
    def evidence_ids(self) -> list[str]:
        # Derived rather than independently emitted by the model.
        return list(dict.fromkeys(q.evidence_id for q in self.supporting_quotes))


@dataclass(frozen=True)
class ValidatedSelection:
    """Validator outcome; valid ambiguous/insufficient results remain non-renderable.

    Only a separate validator may mark a proposal valid. The shape and can_render
    property do not substitute for semantic and candidate-membership checks.
    """
    group_id: str
    status: SelectionStatus
    observation_ids: tuple[str, ...] = ()
    valid: bool = False
    reasons: tuple[str, ...] = ()
    supporting_quotes: tuple[SupportingQuote, ...] = ()

    def __post_init__(self):
        if not self.group_id.strip():
            raise ValueError('Group identity must be nonempty')
        if self.status not in {'preferred', 'distinct', 'ambiguous', 'insufficient'}:
            raise ValueError('Unsupported selection status')

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(q.evidence_id for q in self.supporting_quotes))

    @property
    def can_render(self) -> bool:
        return self.valid and self.status in {'preferred', 'distinct'} and bool(self.supporting_quotes)
