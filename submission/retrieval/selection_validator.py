"""Validate query-time evidence selections without trusting model adjudication.

Supports independently separated tiers and corroborating observations. Explicit
supersession requires a separately verified relation and is not inferred here.
"""
from itertools import combinations
from types import SimpleNamespace

from facts.conflict_detector import ConflictDetector
from retrieval.evidence_selection import GroupSelection, ValidatedSelection
from facts.fact_matching import MatchStatus
from facts.qualifier_schema import compare_fact_qualifiers


class SelectionValidator:

    @staticmethod
    def _different(a: dict, b: dict) -> bool:
        left, right = a['observation'], b['observation']
        if left['unit'] != right['unit']:
            return True
        return ConflictDetector._value_relation(
            SimpleNamespace(value=left['value'], unit=left['unit']),
            SimpleNamespace(value=right['value'], unit=right['unit']),
        ) != 'equal'

    @staticmethod
    def _separated(a: dict, b: dict) -> bool:
        """Only explicitly disjoint qualifiers prove separation.

        Missing qualifiers and different parent/child names do NOT prove it.
        """
        status, _ = compare_fact_qualifiers(
            a['fact']['qualifiers'], b['fact']['qualifiers'],
        )
        return status is MatchStatus.MISMATCH

    @staticmethod
    def _source_bound(candidate: dict, quotes) -> bool:
        support = candidate.get('source_support') or {}
        if support.get('kind') == 'semantic':
            citations = support.get('citations')
            if citations is None:
                primary_id = candidate['observation']['evidence_id']
                citations = [{'evidence_id': primary_id, 'quote': span}
                             for span in (support.get('claim_quote'),
                                          *support.get('subject_quotes', ()),
                                          *support.get('context_quotes', ()),
                                          *((support['date_quote'],)
                                            if support.get('date_quote') else ()))
                             if span]
            cited = {(q.evidence_id, q.quote) for q in quotes}
            required = {(citation['evidence_id'], citation['quote'])
                        for citation in citations if citation.get('quote')}
            for binding in support.get('rule_node_bindings', ()):
                required.update((citation['evidence_id'], citation['quote'])
                                for citation in binding.get('evidence_quotes', ())
                                if citation.get('quote'))
            return (bool(support.get('subject_quotes')) and bool(required)
                    and required <= cited)
        row, header = support.get('row_quote'), support.get('table_header')
        if not row or not header:
            return False
        source_id = candidate['observation']['evidence_id']
        own_quotes = [quote.quote for quote in quotes
                      if quote.evidence_id == source_id]
        # Header and row may be cited separately, but both must be present.
        return (any(row in quote for quote in own_quotes)
                and any(header in quote for quote in own_quotes)
                and all(quote in own_quotes
                        for quote in support.get('context_quotes', ())))

    def validate(self, selection: GroupSelection, group, bundle) -> ValidatedSelection:
        gid = group['group_id']
        reasons = []
        if selection.group_id != gid:
            reasons.append('Selection/group identity mismatch.')

        sources = {source['id']: source for source in bundle.get('evidence', ())}
        candidates = {c['observation']['id']: c for c in bundle.get('candidates', ())}
        allowed_obs = set(group['observation_ids'])
        allowed_evidence = set(group['evidence_ids'])
        chosen_ids = selection.observation_ids
        chosen = set(chosen_ids)

        if len(chosen) != len(chosen_ids):
            reasons.append('Duplicate selected observations.')
        if not chosen <= allowed_obs:
            reasons.append('Selected observation is outside this group.')
        if any(oid not in candidates for oid in allowed_obs):
            reasons.append('Group references an unknown observation.')
        for quote in selection.supporting_quotes:
            source = sources.get(quote.evidence_id)
            if (source is None or quote.evidence_id not in allowed_evidence
                    or source.get('snapshot_id') != bundle['snapshot_id']
                    or not quote.quote.strip()
                    or quote.quote not in source.get('text', '')):
                reasons.append('Citation is outside group/snapshot or is not exact source text.')

        if reasons:
            return ValidatedSelection(gid, selection.status,
                                      reasons=tuple(dict.fromkeys(reasons)))
        if selection.status in {'ambiguous', 'insufficient'}:
            return ValidatedSelection(gid, selection.status, valid=True,
                                      reasons=tuple(selection.unresolved))

        if not selection.supporting_quotes:
            reasons.append('Answer selection requires supporting quotes.')

        if not allowed_obs:
            if chosen:
                reasons.append('Prose group cannot select observations.')
            # Exact quote membership does not prove semantic entailment.
            if reasons:
                return ValidatedSelection(gid, selection.status,
                                          reasons=tuple(dict.fromkeys(reasons)))
            return ValidatedSelection(
                gid, selection.status, valid=True,
                reasons=('Source quotes verified; semantic completeness not established.',),
                supporting_quotes=tuple(selection.supporting_quotes),
            )

        if not chosen:
            reasons.append('Fact selection requires observations.')
        rows = [candidates[oid] for oid in sorted(allowed_obs) if oid in candidates]
        selected_rows = [candidates[oid] for oid in chosen_ids if oid in candidates]

        for row in selected_rows:
            if row['match']['status'] != 'match':
                reasons.append('Selected observation has unproven query applicability.')
            if row['temporal']['applicability'] == 'excluded':
                reasons.append('Selected observation is excluded on the query date.')
            if not self._source_bound(row, selection.supporting_quotes):
                reasons.append('Citation does not bind selected observation to its source row and metric header.')

        # UNKNOWN competitors cannot be silently discarded. A definite MISMATCH
        # or explicitly excluded candidate is not an active competitor.
        active = [row for row in rows
                  if row['match']['status'] != 'mismatch'
                  and row['temporal']['applicability'] != 'excluded']
        for a, b in combinations(active, 2):
            if not self._different(a, b):
                continue
            if self._separated(a, b):
                # Distinct tiers can coexist, but 'preferred' cannot silently
                # omit a tier that is still relevant to the broad query.
                if selection.status == 'preferred':
                    a_id, b_id = a['observation']['id'], b['observation']['id']
                    if (a_id in chosen) != (b_id in chosen):
                        reasons.append('Preferred selection omits a distinct applicable tier.')
                continue
            reasons.append('Unresolved competing values; no verified separation or replacement.')

        if selection.status == 'distinct':
            # A distinct selection may include multiple tiers, but every chosen
            # item must be independently grounded. Identical observations are
            # allowed as corroboration.
            for row in active:
                if row['observation']['id'] not in chosen:
                    if not any(not self._different(row, selected)
                               for selected in selected_rows):
                        reasons.append('Distinct selection omits an unresolved value.')

        if reasons:
            return ValidatedSelection(gid, selection.status,
                                      reasons=tuple(dict.fromkeys(reasons)))
        return ValidatedSelection(
            gid, selection.status, tuple(chosen_ids), valid=True,
            supporting_quotes=tuple(selection.supporting_quotes),
        )
