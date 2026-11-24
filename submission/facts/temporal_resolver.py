"""Resolve a Fact's observations inside the latest eligible complete snapshot."""

from datetime import date
from itertools import combinations

from facts.attribute_normalizer import SEMANTIC_ATTRIBUTES
from facts.conflict_detector import ConflictDetector
from foundation.contract import Fact, FactObservation, TemporalResolution
from foundation.store import KnowledgeStore


class TemporalResolver:
    """Separate snapshot selection, value agreement and temporal applicability.

    None bounds remain unknown. Agreement can resolve a snapshot value even if
    every applicable-period candidate is unknown; it does not prove validity on
    the query date. Never fall back to an older snapshot when a Fact disappears.
    """

    def __init__(self, store: KnowledgeStore):
        self.store = store

    @staticmethod
    def _applicability(observation: FactObservation, query_date: date) -> str:
        start, end = observation.effective_from, observation.effective_to
        if start is not None and end is not None and start > end:
            raise ValueError('Invalid effective date interval')
        if (start is not None and start > query_date) or (end is not None and end < query_date):
            return 'excluded'
        # An open-ended assertion proves its effective start, but not
        # uninterrupted validity after that date. Keep that distinction
        # explicit so answer selection can report it as the latest supported
        # assertion without presenting it as a guaranteed active rate.
        if start is not None and end is not None:
            return 'applicable'
        if start is not None:
            return 'applicable' if start == query_date else 'latest_supported'
        return 'unknown'

    @staticmethod
    def _value_relation(fact: Fact, a: FactObservation, b: FactObservation) -> str:
        if fact.attribute in SEMANTIC_ATTRIBUTES:
            # Semantic values may be identifiers, dates or prose. Never parse
            # them as numeric intervals (e.g. routing numbers with leading zeros).
            if a.unit != b.unit:
                return 'different_units'
            return 'equal' if a.value == b.value else 'different'
        return ConflictDetector._value_relation(a, b)

    def resolve(self, fact: Fact, *, as_of: str | None = None) -> TemporalResolution:
        """Resolve snapshot knowledge, not reporting/event-time query intent.

        Select the desired reporting/ranking-year Fact upstream. A date-valued
        semantic observation is not an effective bound; a scheduled event does
        not become completed merely because the query date passes its value.
        """
        cutoff = date.fromisoformat(as_of) if as_of is not None else date.max
        eligible = [(date.fromisoformat(snapshot.taken_on), snapshot.snapshot_id, snapshot)
                    for snapshot in self.store.snapshots.values()
                    if date.fromisoformat(snapshot.taken_on) <= cutoff]
        if not eligible:
            return TemporalResolution(fact, (), 'not_found', None,
                                      'No snapshot exists at or before the requested date.', None)
        taken_on, snapshot_id, _ = max(eligible, key=lambda item: (item[0], item[1]))
        query_date = cutoff if as_of is not None else taken_on
        stored_fact = self.store.facts.get(fact.id)
        if stored_fact is None:
            return TemporalResolution(fact, (), 'not_found', snapshot_id, 'Fact is not stored.', None)
        if stored_fact != fact:
            raise ValueError('Fact does not match the stored identity')

        candidates = []
        applicability = {}
        selected_count = 0
        excluded_count = 0
        for observation_id in self.store.obs_by_fact.get(fact.id, ()):
            observation = self.store.observations[observation_id]
            if observation.fact_id != fact.id:
                raise ValueError('Observation index references another Fact')
            evidence = self.store.evidence.get(observation.evidence_id)
            if evidence is None or evidence.snapshot_id not in self.store.snapshots:
                raise ValueError('Observation references unknown Evidence/snapshot')
            if evidence.snapshot_id != snapshot_id:
                continue
            selected_count += 1
            relation = self._applicability(observation, query_date)
            if relation == 'excluded':
                excluded_count += 1
                continue
            candidates.append(observation)
            applicability[observation.id] = relation
        candidates.sort(key=lambda o: (self.store.evidence[o.evidence_id].source_url,
                                       o.value, o.unit or '',
                                       str(o.evidence_id), str(o.id)))
        observations = tuple(candidates)
        if not observations:
            reason = ('No observation of this Fact exists in the selected snapshot.'
                      if selected_count == 0 else
                      'All observations in the selected snapshot explicitly exclude the query date.')
            return TemporalResolution(stored_fact, (), 'not_found', snapshot_id, reason, None)

        relations = [(a, b, self._value_relation(stored_fact, a, b))
                     for a, b in combinations(observations, 2)]
        known_count = sum(value == 'applicable' for value in applicability.values())
        latest_count = sum(value == 'latest_supported' for value in applicability.values())
        unknown_count = sum(value == 'unknown' for value in applicability.values())
        certainty = (
            'explicit' if unknown_count == 0 and latest_count == 0 else
            'latest_supported' if unknown_count == 0 and latest_count > 0 else
            'snapshot_only' if known_count == 0 and latest_count == 0 else
            'uncertain'
        )
        context = (f'Snapshot {snapshot_id}; query date {query_date.isoformat()}; '
                   f'{known_count} explicitly applicable, {latest_count} latest-supported, '
                   f'{unknown_count} unknown, '
                   f'{excluded_count} explicitly excluded observations. ')
        if all(relation == 'equal' for _, _, relation in relations):
            reason = ('Remaining values agree; the source provides an effective start but '
                      'no end date, so this is the latest supported assertion.'
                      if certainty == 'latest_supported' else
                      'Remaining snapshot values agree; temporal applicability is unknown, '
                      'not assumed valid.' if known_count == 0 else
                      'Remaining snapshot values agree; unknown candidates do not disagree '
                      'with explicitly applicable observations.')
            return TemporalResolution(stored_fact, observations, 'resolved', snapshot_id, context + reason, certainty)
        confirmed_pairs = sum(
            relation == 'different' and applicability[a.id] == applicability[b.id] == 'applicable'
            for a, b, relation in relations
        )
        if confirmed_pairs:
            return TemporalResolution(
                stored_fact, observations, 'conflicting', snapshot_id,
                context + f'{confirmed_pairs} differing-value pairs explicitly apply on the query date; '
                'no winner selected.', certainty,
            )
        return TemporalResolution(
            stored_fact, observations, 'ambiguous', snapshot_id,
            context + 'Values differ, but simultaneous applicability or value incompatibility '
            'is not established; unknown candidates remain in the result.', certainty,
        )
