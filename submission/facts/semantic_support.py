"""Evidence-bound support for one semantic FactObservation.

Legacy quote fields are implicitly bound to the Observation's primary Evidence.
Additional citations identify their Evidence explicitly. Rule-node bindings map
each Boolean-tree node to one or more exact source citations without changing
the FactObservation's single primary evidence_id.
"""
from dataclasses import dataclass
from typing import Literal
from uuid import UUID


@dataclass(frozen=True)
class EvidenceQuote:
    evidence_id: UUID
    quote: str
    role: Literal["claim", "subject", "date", "context", "rule"] = "context"


@dataclass(frozen=True)
class RuleNodeBinding:
    """Citations for a node, addressed by child indices from the root."""

    node_path: tuple[int, ...]
    evidence_quotes: tuple[EvidenceQuote, ...]


@dataclass(frozen=True)
class SemanticSupport:
    claim_quote: str
    subject_quotes: tuple[str, ...]
    date_quote: str | None = None
    context_quotes: tuple[str, ...] = ()
    evidence_quotes: tuple[EvidenceQuote, ...] = ()
    rule_node_bindings: tuple[RuleNodeBinding, ...] = ()

    @property
    def quotes(self):
        """Legacy primary-Evidence quote view."""
        return tuple(dict.fromkeys((self.claim_quote, *self.subject_quotes, *self.context_quotes,
                                   *((self.date_quote,) if self.date_quote else ()))))

    def citations(self, primary_evidence_id: UUID) -> tuple[EvidenceQuote, ...]:
        """Return all citations with their owning Evidence IDs, deduplicated."""
        primary = [EvidenceQuote(primary_evidence_id, self.claim_quote, "claim")]
        primary.extend(EvidenceQuote(primary_evidence_id, q, "subject")
                       for q in self.subject_quotes)
        primary.extend(EvidenceQuote(primary_evidence_id, q, "context")
                       for q in self.context_quotes)
        if self.date_quote:
            primary.append(EvidenceQuote(primary_evidence_id, self.date_quote, "date"))
        primary.extend(self.evidence_quotes)
        for binding in self.rule_node_bindings:
            primary.extend(binding.evidence_quotes)
        result, seen = [], set()
        for citation in primary:
            key = (citation.evidence_id, citation.quote)
            if key not in seen:
                seen.add(key)
                result.append(citation)
        return tuple(result)


def bind_rule_tree(value: str, evidence_id: UUID) -> tuple[RuleNodeBinding, ...]:
    """Bind each rule-tree node to its exact quote in the primary Evidence.

    Cross-Evidence callers can add more citations to a node explicitly; this
    helper only records the candidate's source and never discovers related
    Evidence or infers a relationship.
    """
    import json

    try:
        root = json.loads(value)
    except (ValueError, TypeError):
        return ()
    result = []

    def visit(node, path=()):
        if not isinstance(node, dict) or not isinstance(node.get('quote'), str):
            return
        result.append(RuleNodeBinding(
            path, (EvidenceQuote(evidence_id, node['quote'], 'rule'),),
        ))
        children = node.get('children')
        if isinstance(children, list):
            for index, child in enumerate(children):
                visit(child, path + (index,))

    visit(root)
    return tuple(result)
