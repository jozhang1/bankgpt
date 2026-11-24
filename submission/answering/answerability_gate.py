"""Conservative Fact admission based on source support and temporal scope."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from retrieval.evidence_selection import GroupSelection, SupportingQuote, ValidatedSelection
from retrieval.selection_validator import SelectionValidator


@dataclass(frozen=True)
class AnswerabilityDecision:
    group_id: str
    requested_field: str
    status: Literal['direct', 'needs_reasoning', 'insufficient', 'excluded']
    reasons: tuple[str, ...]
    selection: ValidatedSelection | None = None


class AnswerabilityGate:
    """Route each supplied group independently and retain its proof of admission."""

    def __init__(self):
        self.validator = SelectionValidator()

    def assess(self, bundle: dict | None) -> tuple[AnswerabilityDecision, ...]:
        if bundle is None:
            return ()
        return tuple(self._assess_group(group, bundle) for group in bundle.get('groups', ()))

    def _assess_group(self, group, bundle):
        def decision(status, *reasons, selection=None):
            return AnswerabilityDecision(group['group_id'], group['requested_field'], status,
                                         tuple(reasons), selection)
        evidence = {e['id']: e for e in bundle.get('evidence', ())}
        sources = [evidence[eid] for eid in group.get('evidence_ids', ())
                   if eid in evidence
                   and evidence[eid].get('snapshot_id') == bundle.get('snapshot_id')
                   and str(evidence[eid].get('text') or '').strip()
                   and not evidence[eid].get('in_recommendations', False)]
        if not sources:
            return decision('insufficient', 'No nonempty source in the selected snapshot.')
        candidates = {c['observation']['id']: c for c in bundle.get('candidates', ())}
        observation_ids = list(group.get('observation_ids', ()))
        if any(oid not in candidates for oid in observation_ids):
            return decision('insufficient', 'Group references a missing observation.')

        rows = [candidates[oid] for oid in observation_ids]
        if self._is_out_of_scope_unknown(rows, bundle):
            return decision(
                'excluded',
                'This candidate does not establish an explicitly requested '
                'condition, while another candidate for the same field does. '
                'It remains available for audit but is not rendered as an answer.',
            )
        eligible, unresolved = [], []
        snapshot_only = []
        latest_supported = []
        historical_query = bundle.get('as_of') is not None
        for row in rows:
            match = row.get('match') or {}
            temporal = row.get('temporal') or {}
            match_status = str(match.get('status', '')).lower()
            applicability = str(temporal.get('applicability', '')).lower()
            if match_status == 'mismatch' or applicability == 'excluded':
                continue
            if match_status != 'match' or temporal.get('status') != 'resolved':
                unresolved.append(row)
                continue
            if applicability in {'applicable', 'included'}:
                eligible.append(row)
                continue
            if applicability == 'latest_supported':
                eligible.append(row)
                latest_supported.append(row)
                continue
            # An undated observation can answer only an unqualified current
            # query. A dated, open-ended assertion is handled separately as
            # latest-supported and remains qualified in the rendered answer.
            if applicability == 'unknown' and not historical_query:
                eligible.append(row)
                snapshot_only.append(row)
                continue
            unresolved.append(row)

        if not eligible and not unresolved:
            return decision('insufficient',
                            'All observations are explicitly incompatible or temporally excluded.')
        if unresolved:
            return decision('needs_reasoning',
                            f'{len(unresolved)} observations have unresolved query or temporal applicability.')

        source_ids = {source['id'] for source in sources}
        if any(row['observation'].get('evidence_id') not in source_ids for row in eligible):
            return decision('needs_reasoning',
                            'An observation lacks a usable primary source in this evidence group.')
        if (group.get('relation') == 'competing'
                and not self._consistent_competing_values(eligible)):
            return decision('needs_reasoning',
                            'Competing observations are not deterministically consistent.')

        quotes = self._collect_supporting_quotes(eligible, source_ids)
        if not quotes:
            return decision('needs_reasoning', 'No usable source-bound supporting quotes.')
        proposed = GroupSelection(group_id=group['group_id'], status='preferred',
                                  observation_ids=[row['observation']['id'] for row in eligible],
                                  supporting_quotes=quotes)
        validated = self.validator.validate(proposed, group, bundle)
        if validated.can_render:
            temporal_note = (
                ' Effective validity is not stated; the value is reported only '
                'as present in the selected latest snapshot.'
                if snapshot_only else ''
            )
            if latest_supported:
                starts = sorted({
                    row['observation'].get('effective_from')
                    for row in latest_supported
                    if row['observation'].get('effective_from')
                })
                start_note = (
                    f' Source effective date: {", ".join(starts)}.' if starts else ''
                )
                temporal_note += (
                    start_note + ' No end date is stated; returned as the latest '
                    'source-supported assertion, not a guarantee of uninterrupted validity.'
                )
            return decision('direct', 'Query conditions, temporal resolution, value consistency and '
                            'source row binding passed.' + temporal_note, selection=validated)
        return decision('needs_reasoning', *validated.reasons)

    @staticmethod
    def _consistent_competing_values(rows):
        """Treat only identical Fact identities and observations as consistent."""
        if not rows:
            return False

        def signature(row):
            observation = row['observation']
            fact = row.get('fact') or {}
            value = str(observation.get('value', '')).strip()
            try:
                value_key = ('numeric', str(Decimal(value.replace(',', ''))))
            except (InvalidOperation, ValueError):
                value_key = ('text', value)
            return (
                fact.get('id') or observation.get('fact_id'), value_key,
                observation.get('unit'), observation.get('effective_from'),
                observation.get('effective_to'),
            )

        return len({signature(row) for row in rows}) == 1

    @staticmethod
    def _collect_supporting_quotes(rows, allowed_source_ids):
        """Collect current and legacy support bindings; validator checks every quote."""
        quotes, seen = [], set()

        def add(evidence_id, quote):
            if evidence_id not in allowed_source_ids or not isinstance(quote, str) or not quote.strip():
                return
            key = (evidence_id, quote)
            if key not in seen:
                seen.add(key)
                quotes.append(SupportingQuote(evidence_id=evidence_id, quote=quote))

        for row in rows:
            observation = row['observation']
            primary_id = observation.get('evidence_id')
            support = row.get('source_support') or {}
            for key in ('table_header', 'row_quote', 'claim_quote', 'subject_quote', 'value_quote'):
                add(primary_id, support.get(key))
            for key in ('subject_quotes', 'context_quotes', 'date_quotes'):
                values = support.get(key) or ()
                if isinstance(values, str):
                    values = (values,)
                for quote in values:
                    add(primary_id, quote)
            if support.get('date_quote'):
                add(primary_id, support['date_quote'])

            citations = support.get('citations') or ()
            for citation in citations:
                if isinstance(citation, dict):
                    add(citation.get('evidence_id', primary_id), citation.get('quote'))

            for key in ('bindings', 'rule_node_bindings'):
                for binding in support.get(key) or ():
                    if not isinstance(binding, dict):
                        continue
                    spans = binding.get('evidence_quotes') or ()
                    if isinstance(spans, dict):
                        spans = (spans,)
                    for span in spans:
                        if isinstance(span, dict):
                            add(span.get('evidence_id', primary_id),
                                span.get('quote', span.get('span')))
                    span = binding.get('span') or {}
                    if isinstance(span, dict):
                        add(span.get('evidence_id'), span.get('quote'))
        return quotes

    @staticmethod
    def _is_out_of_scope_unknown(rows, bundle) -> bool:
        """Suppress condition-unknown rows only when a same-field match exists.

        A condition such as a requested term must not be inferred for a row
        that omits it. An unknown row should not clutter the answer when
        another candidate for the same field explicitly matches the requested
        scope. If no matching row exists, ordinary UNKNOWN handling remains.
        """
        if not rows:
            return False
        target_attributes = {row['fact']['attribute'] for row in rows}
        if len(target_attributes) != 1:
            return False

        condition_keys = {
            key
            for candidate in bundle.get('candidates', ())
            for key in candidate.get('match', {}).get('checks', {})
            if key not in {'product', 'attribute'}
        }
        if not condition_keys:
            return False

        # Exclude only groups made entirely of UNKNOWN candidates whose
        # uncertainty touches an explicit query condition.
        if not all(row['match']['status'] == 'unknown' for row in rows):
            return False
        if not all(
            any(row.get('match', {}).get('checks', {}).get(key) == 'unknown'
                for key in condition_keys)
            for row in rows
        ):
            return False

        return any(
            candidate['fact']['attribute'] in target_attributes
            and candidate.get('match', {}).get('status') == 'match'
            and candidate.get('temporal', {}).get('applicability') != 'excluded'
            for candidate in bundle.get('candidates', ())
            if candidate['observation']['id'] not in {
                row['observation']['id'] for row in rows
            }
        )
