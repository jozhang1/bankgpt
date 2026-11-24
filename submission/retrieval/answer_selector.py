"""Route a query through the existing typed selectors and return selected candidates.

This layer owns candidate discovery, filtering and route choice. Domain-specific
matching remains in FactSelector and LocationIndex;
AnswerRenderer owns the shared formatting entry point.
"""
from dataclasses import dataclass, replace
from datetime import date
from typing import Literal

from facts.fact_matching import MatchStatus
from retrieval.query_scope import ResultShape, parse_result_shape


@dataclass(frozen=True)
class SnapshotScope:
    snapshot: object
    reference_date: date


def snapshot_scope(store, as_of: str | date | None = None) -> SnapshotScope | None:
    """Return the latest snapshot eligible for the query and its date context."""
    cutoff = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    cutoff = cutoff or date.max
    eligible = [snapshot for snapshot in store.snapshots.values()
                if date.fromisoformat(snapshot.taken_on) <= cutoff]
    if not eligible:
        return None
    snapshot = max(eligible, key=lambda item: (date.fromisoformat(item.taken_on),
                                               item.snapshot_id))
    reference = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    return SnapshotScope(snapshot, reference or date.today())


def evidence_is_eligible(store, evidence, snapshot_id: str, *,
                         include_recommendations: bool = False) -> bool:
    """Require exact stored provenance, the selected snapshot, and an allowed region."""
    return (store.evidence.get(evidence.id) == evidence
            and evidence.snapshot_id == snapshot_id
            and (include_recommendations or not evidence.in_recommendations))


def retain_non_mismatches(candidates, status_of=lambda item: item.status):
    """Keep MATCH and UNKNOWN candidates; discard only explicit MISMATCH."""
    result = []
    for candidate in candidates:
        status = status_of(candidate)
        if getattr(status, "value", status) != "mismatch":
            result.append(candidate)
    return result


def stable_unique(candidates, key):
    """Keep the first candidate for each key without changing input order."""
    result, seen = [], set()
    for candidate in candidates:
        marker = key(candidate)
        if marker not in seen:
            seen.add(marker)
            result.append(candidate)
    return result


@dataclass(frozen=True)
class AnswerSelection:
    route: Literal["location", "fact", "retrieved"]
    location: object | None = None
    matches: tuple | None = None
    hits: tuple = ()
    result_shape: ResultShape = ResultShape()
    value_existence: bool | None = None


class AnswerSelector:
    """Apply snapshot-aware candidate selection before answer rendering."""

    def __init__(self, *, retriever, fact_selector, location_index):
        self.retriever = retriever
        self.fact_selector = fact_selector
        self.location_index = location_index

    def retrieve(self, question: str, *, as_of: str | None = None, top_k: int = 10):
        # EvidenceRetriever fuses lexical and vector Evidence rankings.
        return self.retriever.search(
            question, as_of=as_of, top_k=top_k,
        )

    def _scope_evidence(self, hits, as_of):
        scope = snapshot_scope(self.retriever.store, as_of)
        if scope is None:
            return []
        snapshot_id = scope.snapshot.snapshot_id
        return [hit for hit in hits
                if evidence_is_eligible(self.retriever.store, hit.evidence, snapshot_id)]

    def _filter_location(self, lookup):
        if not lookup.records:
            return lookup
        # A not-found result may retain nearby identity records as diagnostic
        # citations. They are not answer candidates, but preserve that context.
        if lookup.status == "not_found":
            return lookup
        query = lookup.query
        records = []
        for record in lookup.records:
            value = (record.address if query.field == "address" else
                     record.lobby_hours if query.field == "lobby_hours" else
                     record.drive_up_hours if query.field == "drive_up_hours" else
                     (record.lobby_hours or record.drive_up_hours))
            evidence = self.retriever.store.evidence.get(record.evidence_id)
            if (value and lookup.snapshot_id is not None and evidence is not None
                    and evidence_is_eligible(self.retriever.store, evidence,
                                             lookup.snapshot_id)):
                records.append(record)
        if len(records) == len(lookup.records):
            return lookup
        status = lookup.status if records else "not_found"
        return replace(lookup, status=status, records=tuple(records))

    def select(self, question: str, *, as_of: str | None = None) -> AnswerSelection:
        result_shape = parse_result_shape(question)
        location = self.location_index.lookup(question, as_of=as_of)
        if location is not None:
            location = self._filter_location(location)
            return AnswerSelection("location", location=location,
                                   result_shape=getattr(location.query, 'result_shape', result_shape))

        matches = self.fact_selector.select_matches(question, as_of=as_of)
        if matches is not None:
            matches = [match for match in matches
                       if match.status is not MatchStatus.MISMATCH]
            scope = self.fact_selector.last_scope
            value_existence = None
            if (scope is not None and scope.result_shape.kind == "existence"
                    and scope.requested_value is not None):
                exact_matches = [
                    match for match in matches
                    if dict(match.checks).get("requested_value") is MatchStatus.MATCH
                ]
                value_existence = bool(exact_matches)
            return AnswerSelection("fact", matches=tuple(matches),
                                   result_shape=result_shape,
                                   value_existence=value_existence)

        hits = self.retrieve(question, as_of=as_of, top_k=10)
        hits = self._scope_evidence(hits, as_of)
        return AnswerSelection("retrieved", hits=tuple(hits), result_shape=result_shape)
