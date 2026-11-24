"""Find source-backed disagreements within and across snapshots."""

from collections import defaultdict
from decimal import Decimal, InvalidOperation
from itertools import combinations, product
import json
import logging
import os
import re
from types import SimpleNamespace

from contract import Conflict, Source
from facts.fact_matching import MatchStatus
from facts.qualifier_schema import QUALIFIER_SCHEMA, compare_fact_qualifiers
from foundation.contract import FactObservation, PendingStatement
from foundation.entities import entity_label
from foundation.products import PRODUCTS
from foundation.store import KnowledgeStore

logger = logging.getLogger(__name__)


def _report_version_changes() -> bool:
    """Report explicitly dated version changes of one Fact as conflicts.

    On by default; set BANKGPT_REPORT_VERSION_CHANGES=0 to restore the older
    behaviour of leaving them out, so the two can be compared on real data.
    """
    return os.environ.get('BANKGPT_REPORT_VERSION_CHANGES', '1').strip() != '0'


_NON_CONFLICT_ATTRIBUTES = frozenset({'application_url', 'rates_url'})
_AUTHORITY_RATE_TABLE = 3
# Lower rank = less authoritative for a current rate/fee. Narrative pages
# (news, blog, press releases, PDFs) describe offers at one point in time and are
# routinely left online after the rate moved; calculators are derived.
_AUTHORITY_LABEL = {0: 'narrative page', 1: 'calculator', _AUTHORITY_RATE_TABLE: 'official_rate_table'}
_NARRATIVE_URL = re.compile(
    r'(?:^|[/_-])(?:news|newsroom|blog|press|press-release|article|articles|stories|story|archive|archives)(?:[/_.-]|$)'
    r'|\.pdf(?:$|[?#])', re.I)


class ConflictDetector:
    """Report conflicting observations with evidence scoped to the differing values.

    Same-Fact comparisons preserve version disagreements. Different Fact
    identities are compared for the same entity, or for a catalog parent and
    child when the parent-side observation explicitly names the child in its
    structured source context. In either case the metric and a schema-defined
    numeric condition must explicitly overlap; disjoint conditions and
    categorical mismatches remain separate.
    """

    def __init__(self, store: KnowledgeStore):
        self.store = store
        self._products = {item.id: item for item in PRODUCTS}

    def _parent_child(self, left, right):
        """Return (parent, child) only for a catalog-backed product lineage."""
        for parent, child in ((left, right), (right, left)):
            current = self._products.get(child.entity)
            while current is not None and current.parent_id is not None:
                if current.parent_id == parent.entity:
                    return parent, child
                current = self._products.get(current.parent_id)
        return None

    def _parent_observation_names_child(self, observation, child) -> bool:
        """Require an exact product label in source-local identity context."""
        product = self._products.get(child.entity)
        if product is None:
            return False
        evidence = self.store.evidence.get(observation.evidence_id)
        if evidence is None or evidence.in_recommendations:
            return False

        expected = {entity_label(label) for label in (product.name, *product.aliases)}
        signals = observation.context.get("identity_signals", ())
        for signal in signals:
            if not isinstance(signal, dict):
                continue
            kind = signal.get("kind")
            if kind not in {"heading", "table_label", "product_column"}:
                continue
            label = signal.get("label")
            if isinstance(label, str):
                normalized = entity_label(label)
                if normalized in expected:
                    return True
                # Headings often add descriptive wording around a product
                # name (for example, "Auto Loans For Every Ride"). Accept
                # only a whole-token phrase inside a heading, not prose.
                if kind == "heading" and any(
                    f" {alias} " in f" {normalized} " for alias in expected if alias
                ):
                    return True

        # Heading paths are also source-local structure retained by ingestion.
        return any(
            isinstance(heading, str)
            and any(
                f" {alias} " in f" {entity_label(heading)} "
                for alias in expected if alias
            )
            for heading in observation.context.get("heading_path", ())
        )

    @staticmethod
    def _temporal_relation(a: FactObservation, b: FactObservation) -> str:
        # Publication dates are provenance, not effective intervals. Only
        # explicit effective dates can establish temporal separation here.
        for obs in (a, b):
            if (obs.effective_from is not None and obs.effective_to is not None
                    and obs.effective_from > obs.effective_to):
                # One malformed interval must not abort the whole report; treat
                # its temporal relation as unknown (reported as uncertain).
                return 'unknown'
        if ((a.effective_to is not None and b.effective_from is not None
             and a.effective_to < b.effective_from)
                or (b.effective_to is not None and a.effective_from is not None
                    and b.effective_to < a.effective_from)):
            return 'disjoint'
        # Rate sheets commonly publish a new effective date without an
        # explicit end date for the prior rate. For the same Fact, a later
        # effective start supersedes the earlier open-ended assertion.
        if (a.effective_from is not None and b.effective_from is not None
                and a.effective_from != b.effective_from
                and a.effective_to is None and b.effective_to is None):
            return 'disjoint'
        if all(x is not None for x in (a.effective_from, a.effective_to,
                                        b.effective_from, b.effective_to)):
            return 'overlap'
        if a.effective_from is not None and a.effective_from == b.effective_from:
            return 'overlap'
        return 'unknown'

    @staticmethod
    def _value_relation(a: FactObservation, b: FactObservation) -> str:
        if a.value == b.value and a.unit == b.unit:
            return 'equal'
        if a.unit != b.unit:
            return 'different_units'

        def interval(value):
            try:
                parts = value.replace('—', '–').split('–')
                if len(parts) not in (1, 2):
                    return None
                bounds = [Decimal(part.strip().rstrip('%')) for part in parts]
                if not all(bound.is_finite() for bound in bounds):
                    return None
                if len(bounds) == 2 and bounds[0] > bounds[1]:
                    return None
                return bounds[0], bounds[-1]
            except (InvalidOperation, ValueError):
                return None

        left, right = interval(a.value), interval(b.value)
        if left is None or right is None:
            return 'different'
        if left == right:
            return 'equal'
        if left[0] <= right[1] and right[0] <= left[1]:
            return 'intersecting_ranges'
        return 'different'

    @staticmethod
    def _display_value(observation: FactObservation) -> str:
        if observation.unit == 'percent':
            return observation.value + '%'
        return observation.value + (f' {observation.unit}' if observation.unit else '')

    @staticmethod
    def _source_quote(observation: FactObservation, evidence, store) -> str:
        """Prefer the actual table row or claim over the whole evidence chunk."""
        quote = observation.context.get('row_quote')
        if quote and quote in evidence.text:
            return quote
        support = store.semantic_support.get(observation.id)
        if support:
            quote = getattr(support, 'claim_quote', None)
            if quote and quote in evidence.text:
                return quote
        return evidence.text

    def _effective_date_resolution(self, observations):
        """Select a uniquely latest source-dated assertion when periods roll over."""
        if any(observation.effective_from is None for observation in observations):
            return None, (), 'Competing observations do not all have an effective start date.'

        latest_date = max(observation.effective_from for observation in observations)
        winners = tuple(observation for observation in observations
                        if observation.effective_from == latest_date)
        winner = winners[0]
        if any(self._value_relation(winner, other) != 'equal' for other in winners[1:]):
            return None, (), 'The latest effective date has multiple differing values.'

        for observation in observations:
            if self._value_relation(winner, observation) == 'equal':
                continue
            if (observation.effective_to is not None
                    and observation.effective_to >= latest_date):
                return None, (), (
                    'An older differing observation is explicitly effective through '
                    'the latest start date.'
                )

        value = self._display_value(winner)
        prior_dates = sorted({
            observation.effective_from.isoformat()
            for observation in observations
            if observation.effective_from < latest_date
            and self._value_relation(winner, observation) != 'equal'
        })
        reason = (
            f'Believed winner: {value}. Its source-supported effective start '
            f'{latest_date.isoformat()} is later than the competing assertion(s) '
            f'effective {", ".join(prior_dates)}; no differing older interval '
            'is explicitly valid through that start date.'
        )
        return value, winners, reason

    @staticmethod
    def _authority(evidence):
        """Authority rank of a source for a current value; None when unknown."""
        # Narrative pages are checked first: a press release or PDF that happens
        # to contain a rate table is still a dated description, not the schedule.
        if _NARRATIVE_URL.search(evidence.source_url or ''):
            return 0
        if evidence.source_type == 'calculator':
            return 1
        if evidence.source_type == 'official_rate_table':
            return _AUTHORITY_RATE_TABLE
        return None

    def _latest_snapshot_resolution(self, fact_ids, observations):
        """Choose a winner only if the latest full-site snapshot supports it.

        Snapshot order identifies the currently captured website state; it is
        not interpreted as an assertion's effective date. The harness defines
        Snapshot.pages as the full website, so an older assertion absent from
        the latest snapshot does not compete with its current value.
        """
        if not self.store.snapshots:
            return None, (), 'No snapshot is available to establish a current value.'

        latest_snapshot_id = max(
            self.store.snapshots.values(),
            key=lambda snapshot: (snapshot.taken_on, snapshot.snapshot_id),
        ).snapshot_id
        latest_observations = []
        fact_ids = set(fact_ids)
        for observation in self.store.observations.values():
            if observation.fact_id not in fact_ids:
                continue
            evidence = self.store.evidence.get(observation.evidence_id)
            if evidence is None or evidence.snapshot_id != latest_snapshot_id:
                continue
            latest_observations.append(observation)

        if not latest_observations:
            return None, (), 'The latest full-site snapshot has no observation for this Fact.'

        winner = latest_observations[0]
        if any(self._value_relation(winner, other) != 'equal'
               for other in latest_observations[1:]):
            effective_value, effective_winners, effective_reason = (
                self._effective_date_resolution(latest_observations)
            )
            if effective_value is not None:
                return effective_value, effective_winners, effective_reason

            ranked = [
                (observation, self._authority(self.store.evidence[observation.evidence_id]))
                for observation in latest_observations
            ]
            known = [(observation, rank) for observation, rank in ranked
                     if rank is not None]
            if len({rank for _, rank in known}) >= 2:
                highest = max(rank for _, rank in known)
                preferred = tuple(observation for observation, rank in known
                                  if rank == highest)
                preferred_value = preferred[0]
                # Every observation that disagrees with the preferred value must
                # be a source we positively know to be lower-authority. An
                # unranked source that disagrees keeps the conflict unresolved;
                # an unranked source that agrees is harmless.
                lower = tuple(
                    observation for observation, rank in ranked
                    if self._value_relation(preferred_value, observation) != 'equal'
                )
                rank_of = {observation.id: rank for observation, rank in ranked}
                if (lower
                        and all(self._value_relation(preferred_value, other) == 'equal'
                                for other in preferred[1:])
                        and all(rank_of[observation.id] is not None
                                and rank_of[observation.id] < highest
                                for observation in lower)):
                    value = self._display_value(preferred_value)
                    preferred_label = _AUTHORITY_LABEL[highest]
                    lower_types = sorted({
                        _AUTHORITY_LABEL[rank_of[observation.id]]
                        for observation in lower
                    })
                    basis = (
                        'The published rate table is the direct quote schedule; '
                        'calculator and narrative (news/blog/press/PDF) content '
                        'is derived or dated and may lag it.'
                        if highest == _AUTHORITY_RATE_TABLE
                        else 'The selected source role has higher authority for this value.'
                    )
                    reason = (
                        f'Believed winner: {value}. {basis} The latest full-site '
                        f'snapshot supports it in {preferred_label}; the competing '
                        f'value is from lower-authority source class(es): '
                        f'{", ".join(lower_types)}.'
                    )
                    return value, preferred, reason
            return None, (), (
                'The latest full-site snapshot still contains differing values, '
                f'{effective_reason} The available source types also do not '
                'establish a unique higher-authority value.'
            )
        if all(self._value_relation(winner, old) == 'equal' for old in observations):
            return None, (), 'The latest snapshot does not resolve the reported disagreement.'

        value = self._display_value(winner)
        reason = (
            f'Believed winner: {value}. All observations of this Fact in the '
            f'latest full-site snapshot ({latest_snapshot_id}) agree on this '
            'value; the differing older assertion is absent from that snapshot. '
            'Snapshot order identifies the captured site version, not an '
            'effective date.'
        )
        return value, tuple(latest_observations), reason

    def _make_conflict(self, fact, snapshot_id, observations, reason, *,
                       resolution=None, topic=None):
        observations = {obs.id: obs for obs in observations}
        sources_by_id = {}
        for observation in observations.values():
            evidence = self.store.evidence[observation.evidence_id]
            key = (evidence.source_url, evidence.snapshot_id, self._source_quote(observation, evidence, self.store))
            sources_by_id[key] = Source(url=evidence.source_url,
                                        snapshot_id=evidence.snapshot_id, quote=key[2])
        values = sorted({self._display_value(obs) for obs in observations.values()})
        return Conflict(
            topic=topic or self._topic(fact, snapshot_id),
            values=values,
            sources=sorted(sources_by_id.values(), key=lambda source: (source.url, source.quote or '')),
            resolution=resolution,
            reason=reason,
        )

    def _statement_conflicts(self, *, snapshot_id=None):
        """Compare source-bound pending claims with Facts and one another.

        This is deliberately separate from canonical Fact conflict handling:
        pending claims never receive an inferred winner, and uncertainty in
        identity, conditions, or effective dates keeps the result potential.
        """
        assertions = []
        for observation in self.store.observations.values():
            fact = self.store.facts.get(observation.fact_id)
            evidence = self.store.evidence.get(observation.evidence_id)
            if fact is None or evidence is None:
                continue
            if snapshot_id is not None and evidence.snapshot_id != snapshot_id:
                continue
            assertions.append({
                'kind': 'fact', 'id': observation.id,
                'entity': fact.entity, 'attribute': fact.attribute,
                'qualifiers': fact.qualifiers, 'value': observation.value,
                'unit': observation.unit,
                'effective_from': observation.effective_from,
                'effective_to': observation.effective_to,
                'evidence': evidence, 'quote': self._source_quote(
                    observation, evidence, self.store,
                ),
                'issues': (),
            })

        for statement in self.store.pending_statements.values():
            evidence = self.store.evidence.get(statement.evidence_id)
            if evidence is None or evidence.in_recommendations:
                continue
            if snapshot_id is not None and evidence.snapshot_id != snapshot_id:
                continue
            assertions.append({
                'kind': 'pending', 'id': statement.id,
                'entity': statement.entity, 'attribute': statement.attribute,
                'qualifiers': statement.qualifiers, 'value': statement.value,
                'unit': statement.unit,
                'effective_from': statement.effective_from,
                'effective_to': statement.effective_to,
                'evidence': evidence, 'quote': statement.quote,
                'issues': statement.issues,
            })

        # Bucket first so unrelated entities/attributes are never compared.
        buckets = defaultdict(list)
        for assertion in assertions:
            entity = entity_label(assertion['entity'])
            attribute = assertion['attribute'].strip().casefold()
            if (not entity or not attribute
                    or attribute in _NON_CONFLICT_ATTRIBUTES):
                continue
            buckets[(entity, attribute)].append(assertion)

        grouped = {}
        for (entity, attribute), records in buckets.items():
            for left, right in combinations(records, 2):
                if left['kind'] == right['kind'] == 'fact':
                    continue
                if not self._pending_value_is_comparable(left) or not self._pending_value_is_comparable(right):
                    continue

                qualifier_status, qualifier_reasons = compare_fact_qualifiers(
                    left['qualifiers'], right['qualifiers'],
                )
                if qualifier_status is MatchStatus.MISMATCH:
                    continue

                temporal = self._temporal_relation(
                    SimpleNamespace(
                        effective_from=left['effective_from'],
                        effective_to=left['effective_to'],
                    ),
                    SimpleNamespace(
                        effective_from=right['effective_from'],
                        effective_to=right['effective_to'],
                    ),
                )
                if temporal == 'disjoint':
                    continue

                value_relation = self._value_relation(
                    SimpleNamespace(value=left['value'], unit=left['unit']),
                    SimpleNamespace(value=right['value'], unit=right['unit']),
                )
                if value_relation in {'equal', 'intersecting_ranges'}:
                    continue

                # Unit mismatches are review candidates, never proof that the
                # underlying quantities disagree.
                # At least one side is pending. Its source support or identity
                # admission has not completed, so this pass must not promote a
                # disagreement to a confirmed website contradiction.
                status = 'potential_conflict'
                uncertainties = []
                if qualifier_status is MatchStatus.UNKNOWN:
                    uncertainties.append(
                        'condition scope is not fully established'
                        + (f' ({", ".join(qualifier_reasons)})'
                           if qualifier_reasons else '')
                    )
                if temporal == 'unknown':
                    uncertainties.append('effective-date overlap is unknown')
                if value_relation == 'different_units':
                    uncertainties.append('values use different or unverified units')

                left_sig = self._statement_signature(left)
                right_sig = self._statement_signature(right)
                ordered = tuple(sorted(
                    (left_sig, right_sig),
                    key=lambda item: json.dumps(item, ensure_ascii=False, default=str),
                ))
                key = (entity, attribute, ordered, status)
                entry = grouped.setdefault(key, {
                    'records': {}, 'uncertainties': set(),
                    'pending_issues': set(),
                })
                entry['records'][(left['evidence'].source_url,
                                  left['evidence'].snapshot_id,
                                  left['quote'])] = left
                entry['records'][(right['evidence'].source_url,
                                  right['evidence'].snapshot_id,
                                  right['quote'])] = right
                entry['uncertainties'].update(uncertainties)
                entry['pending_issues'].update(left['issues'])
                entry['pending_issues'].update(right['issues'])

        conflicts = []
        for (entity, attribute, signatures, status), group in grouped.items():
            records = list(group['records'].values())
            values = sorted({self._format_statement_value(record) for record in records})
            sources = sorted(
                (Source(url=record['evidence'].source_url,
                        snapshot_id=record['evidence'].snapshot_id,
                        quote=record['quote']) for record in records),
                key=lambda source: (source.url, source.snapshot_id or '', source.quote or ''),
            )
            scope = snapshot_id or 'all snapshots'
            conditions = ' <> '.join(
                json.dumps(signature[5], sort_keys=True, ensure_ascii=False,
                           separators=(',', ':'))
                for signature in signatures
            )
            topic = f'{entity}.{attribute} statements {conditions} [{scope}]'
            reason_parts = [
                f'{status}: a pending source-quoted claim differs from another '
                'claim for the same exact entity and attribute.'
            ]
            if group['uncertainties']:
                reason_parts.append(
                    'Review required: ' + '; '.join(sorted(group['uncertainties'])) + '.'
                )
            if group['pending_issues']:
                reason_parts.append(
                    'Pending statement admission issues: '
                    + ', '.join(sorted(group['pending_issues'])) + '.'
                )
            reason_parts.append(
                'No winner was selected; a pending statement is not treated as a '
                'canonical Fact and source recency alone does not resolve it.'
            )
            conflicts.append(Conflict(
                topic=topic, values=values, sources=sources,
                resolution=None, reason=' '.join(reason_parts),
            ))
        return conflicts

    @staticmethod
    def _pending_value_is_comparable(assertion):
        """Keep rejected claims for review, but do not compare unparsed values."""
        if assertion['kind'] != 'pending':
            return True
        return not any(
            issue == 'value_not_normalized'
            or (issue.startswith('unsupported_') and issue.endswith('_value'))
            or issue.endswith(':unsupported_citation:value')
            for issue in assertion['issues']
        )

    @staticmethod
    def _format_statement_value(assertion):
        value = assertion['value'].strip()
        unit = assertion['unit']
        if not unit:
            return value
        normalized = unit.casefold()
        folded_value = value.casefold()
        if normalized == 'percent':
            return value if '%' in value else value + '%'
        if normalized == 'usd' and ('$' in value or 'usd' in folded_value):
            return value
        return value if normalized in folded_value else f'{value} {unit}'

    @staticmethod
    def _statement_signature(assertion):
        """Stable comparison key excluding provenance and the quoted wording."""
        return (
            assertion['kind'], assertion['value'], assertion['unit'],
            assertion['effective_from'].isoformat()
            if assertion['effective_from'] else None,
            assertion['effective_to'].isoformat()
            if assertion['effective_to'] else None,
            json.dumps(assertion['qualifiers'], sort_keys=True,
                       ensure_ascii=False, separators=(',', ':'), default=str),
        )

    @staticmethod
    def _topic(fact, snapshot_id):
        qualifiers = json.dumps(fact.qualifiers, sort_keys=True, ensure_ascii=False,
                                separators=(',', ':'))
        return f'{fact.entity}.{fact.attribute} {qualifiers} [{snapshot_id}]'

    @staticmethod
    def _overlapping_interval_reason(left, right):
        """Return a schema interval with distinct, overlapping Fact scopes."""
        status, reasons = compare_fact_qualifiers(left.qualifiers, right.qualifiers)
        if status is MatchStatus.MISMATCH:
            return None
        reason_set = set(reasons)
        for spec in QUALIFIER_SCHEMA:
            if spec.comparator not in {'interval', 'year_interval'}:
                continue
            keys = (spec.lower_key, spec.upper_key)
            left_bounds = tuple(left.qualifiers.get(key) for key in keys)
            right_bounds = tuple(right.qualifiers.get(key) for key in keys)
            # A one-sided cap/floor may be a tier whose missing opposite bound
            # was not extracted. Do not treat that missing bound as infinity
            # when deciding whether two different Facts form a real conflict.
            if (left_bounds != right_bounds
                    and all(value is not None for value in (*left_bounds, *right_bounds))
                    and f'overlapping_interval:{spec.key}' in reason_set):
                return spec.key
        return None

    @staticmethod
    def _pair_topic(left, right, snapshot_id):
        left_q = json.dumps(left.qualifiers, sort_keys=True, ensure_ascii=False,
                            separators=(',', ':'))
        right_q = json.dumps(right.qualifiers, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'))
        entities = (left.entity if left.entity == right.entity
                    else f'{left.entity} <> {right.entity}')
        return (f'{entities}.{left.attribute} scopes {left_q} <> {right_q} '
                f'[{snapshot_id}]')

    def _detect_overlapping_scope_conflicts(self, *, snapshot_id=None):
        """Report differing Facts whose schema-defined numeric scopes overlap."""
        facts = sorted(self.store.facts.values(), key=lambda fact: str(fact.id))
        observations_by_fact = defaultdict(list)
        for observation in self.store.observations.values():
            evidence = self.store.evidence.get(observation.evidence_id)
            if evidence is None:
                continue
            if snapshot_id is None or evidence.snapshot_id == snapshot_id:
                observations_by_fact[observation.fact_id].append(observation)

        # Only same-attribute pairs can ever conflict, so bucket first instead
        # of walking every pair of Facts (quadratic in the whole fact count).
        facts_by_attribute = defaultdict(list)
        for fact in facts:
            if fact.attribute not in _NON_CONFLICT_ATTRIBUTES:
                facts_by_attribute[fact.attribute].append(fact)
        fact_pairs = [pair for attribute in sorted(facts_by_attribute)
                      for pair in combinations(facts_by_attribute[attribute], 2)]

        conflicts = []
        for left, right in fact_pairs:
            parent_child = None if left.entity == right.entity else self._parent_child(left, right)
            if left.entity != right.entity and parent_child is None:
                continue
            interval = self._overlapping_interval_reason(left, right)
            if interval is None:
                continue

            if parent_child is not None:
                parent, child = parent_child
                parent_observations = observations_by_fact.get(parent.id, ())
                if not any(
                    self._parent_observation_names_child(observation, child)
                    for observation in parent_observations
                ):
                    continue

            def assertions(observations):
                grouped = defaultdict(list)
                for observation in observations:
                    evidence = self.store.evidence[observation.evidence_id]
                    grouped[(
                        evidence.snapshot_id, observation.value, observation.unit,
                        observation.effective_from, observation.effective_to,
                    )].append(observation)
                return tuple(grouped.items())

            left_assertions = assertions(observations_by_fact.get(left.id, ()))
            right_assertions = assertions(observations_by_fact.get(right.id, ()))
            if parent_child is not None:
                parent, child = parent_child
                parent_assertions = assertions(
                    observation
                    for observation in observations_by_fact.get(parent.id, ())
                    if self._parent_observation_names_child(observation, child)
                )
                child_assertions = assertions(observations_by_fact.get(child.id, ()))
                if parent.id == left.id:
                    left_assertions, right_assertions = parent_assertions, child_assertions
                else:
                    left_assertions, right_assertions = child_assertions, parent_assertions
            differing = {}
            within_confirmed = within_uncertain = 0
            cross_confirmed = cross_uncertain = 0
            superseded_pairs = 0
            for (left_key, left_group), (right_key, right_group) in product(
                left_assertions, right_assertions,
            ):
                a, b = left_group[0], right_group[0]
                relation = self._value_relation(a, b)
                temporal = self._temporal_relation(a, b)
                if relation == 'equal':
                    continue
                if temporal == 'disjoint':
                    # A later explicit effective start is a version change,
                    # but still report it when the Fact scopes overlap and
                    # expose the later value as the resolution.
                    if (a.effective_from is not None and b.effective_from is not None
                            and a.effective_from != b.effective_from
                            and a.effective_to is None and b.effective_to is None):
                        superseded_pairs += 1
                        differing.update(
                            (item.id, item) for item in (*left_group, *right_group)
                        )
                    continue
                a_snapshot, b_snapshot = left_key[0], right_key[0]
                confirmed = relation == 'different' and temporal == 'overlap'
                if a_snapshot == b_snapshot:
                    if confirmed:
                        within_confirmed += 1
                    else:
                        within_uncertain += 1
                elif confirmed:
                    cross_confirmed += 1
                else:
                    cross_uncertain += 1
                differing.update((item.id, item) for item in (*left_group, *right_group))

            if not differing:
                continue
            resolution = None
            if snapshot_id is None:
                resolution, winners, winner_reason = self._latest_snapshot_resolution(
                    {left.id, right.id}, differing.values(),
                )
                differing.update((observation.id, observation) for observation in winners)
            else:
                winner_reason = (
                    'No winner selected for a snapshot-scoped conflict; '
                    'future source versions are outside this comparison.'
                )
            confirmed_count = within_confirmed + cross_confirmed
            status = (
                'confirmed_conflict' if confirmed_count else
                'potential_conflict' if within_uncertain + cross_uncertain else
                'resolved_version_difference'
            )
            reason = (
                f'{status}: different Facts for the same metric have '
                f'overlapping {interval} scopes; '
                f'within-snapshot pairs: {within_confirmed} confirmed, '
                f'{within_uncertain} uncertain; across-snapshot pairs: '
                f'{cross_confirmed} confirmed, {cross_uncertain} uncertain; '
                f'explicitly superseded effective-date pairs: {superseded_pairs}. '
                f'{winner_reason}'
            )
            if parent_child is not None:
                reason += ' Parent/child product comparison is supported by an exact product label in the parent-side evidence context.'
            conflicts.append(self._make_conflict(
                left, snapshot_id or 'all snapshots', differing.values(), reason,
                resolution=resolution,
                topic=self._pair_topic(left, right, snapshot_id or 'all snapshots'),
            ))
        return conflicts

    def detect(self, *, snapshot_id: str | None = None) -> list[Conflict]:
        if snapshot_id is not None and snapshot_id not in self.store.snapshots:
            raise ValueError(f'Unknown snapshot: {snapshot_id}')
        by_fact = defaultdict(list)
        for observation in self.store.observations.values():
            # A dangling reference skips that one observation instead of
            # aborting the whole report (a raise here would zero the pillar).
            if observation.fact_id not in self.store.facts:
                continue
            evidence = self.store.evidence.get(observation.evidence_id)
            if evidence is None or evidence.snapshot_id not in self.store.snapshots:
                continue
            if snapshot_id is None or evidence.snapshot_id == snapshot_id:
                by_fact[observation.fact_id].append((evidence.snapshot_id, observation))

        conflicts = []
        for fact_id, records in by_fact.items():
            fact = self.store.facts[fact_id]
            if fact.attribute in _NON_CONFLICT_ATTRIBUTES:
                continue
            # Repeated copies of the same assertion within one snapshot should
            # preserve their provenance without multiplying conflict counts.
            assertions = defaultdict(list)
            for record_snapshot, observation in records:
                assertion = (
                    record_snapshot,
                    observation.value,
                    observation.unit,
                    observation.effective_from,
                    observation.effective_to,
                )
                assertions[assertion].append(observation)
            assertions = sorted(
                assertions.items(),
                key=lambda item: (
                    self.store.snapshots[item[0][0]].taken_on,
                    item[0][0],
                    str(item[0][1]),
                    str(item[0][2]),
                    str(item[0][3]),
                    str(item[0][4]),
                ),
            )
            differing_observations = {}
            within_confirmed = within_uncertain = 0
            cross_confirmed = cross_uncertain = 0
            superseded_pairs = 0
            for (left_key, left_observations), (right_key, right_observations) in combinations(assertions, 2):
                left_snapshot = left_key[0]
                right_snapshot = right_key[0]
                left = left_observations[0]
                right = right_observations[0]
                relation = self._value_relation(left, right)
                temporal = self._temporal_relation(left, right)
                if relation == 'equal':
                    continue
                if temporal == 'disjoint':
                    # Explicitly dated, non-overlapping versions of one Fact
                    # (a rate sheet re-issued with a new effective date) are a
                    # version change between snapshots. They are reported with
                    # the later value as the resolution, as the overlapping-
                    # scope detector already does, instead of being dropped.
                    if _report_version_changes():
                        superseded_pairs += 1
                        differing_observations.update(
                            (observation.id, observation)
                            for observation in (*left_observations, *right_observations)
                        )
                    continue
                is_confirmed = relation == 'different' and temporal == 'overlap'
                if left_snapshot == right_snapshot:
                    if is_confirmed:
                        within_confirmed += 1
                    else:
                        within_uncertain += 1
                else:
                    if is_confirmed:
                        cross_confirmed += 1
                    else:
                        cross_uncertain += 1
                differing_observations.update(
                    (observation.id, observation)
                    for observation in (*left_observations, *right_observations)
                )
            if differing_observations:
                confirmed = within_confirmed + cross_confirmed
                status = (
                    'confirmed_conflict' if confirmed else
                    'potential_conflict' if within_uncertain + cross_uncertain else
                    'resolved_version_difference'
                )
                scope = snapshot_id or 'all snapshots'
                reason = (
                    f'{status}: same Fact; '
                    f'within-snapshot pairs: {within_confirmed} confirmed, '
                    f'{within_uncertain} uncertain; '
                    f'across-snapshot pairs: {cross_confirmed} confirmed, '
                    f'{cross_uncertain} uncertain; '
                    f'explicitly superseded effective-date pairs: {superseded_pairs}. '
                )
                resolution = None
                # A snapshot-scoped audit must not use evidence from a later
                # snapshot to resolve a disagreement in the requested one.
                if snapshot_id is None:
                    resolution, winner_observations, resolution_reason = self._latest_snapshot_resolution(
                        {fact_id}, differing_observations.values(),
                    )
                    differing_observations.update(
                        (observation.id, observation)
                        for observation in winner_observations
                    )
                else:
                    resolution_reason = (
                        'No winner selected for a snapshot-scoped conflict; '
                        'future source versions are outside this comparison.'
                    )
                reason += resolution_reason
                conflicts.append(self._make_conflict(
                    fact, scope, differing_observations.values(), reason,
                    resolution=resolution,
                ))

        # Different Fact identities carry different qualifiers or product
        # conditions. Do not promote them to conflicts based on fuzzy qualifier
        # compatibility; conflicts require the same Fact and conditions.
        # Each secondary detector is isolated: one failing must not discard the
        # conflicts already found by the others.
        for secondary in (self._detect_overlapping_scope_conflicts,
                          self._statement_conflicts):
            try:
                conflicts.extend(secondary(snapshot_id=snapshot_id))
            except Exception:
                logger.warning('%s failed; continuing without it', secondary.__name__,
                               exc_info=True)
        conflicts = self._dedupe(conflicts)
        return sorted(conflicts, key=lambda conflict: (conflict.topic, tuple(conflict.values),
            tuple((source.url, source.quote or '') for source in conflict.sources)))

    @staticmethod
    def _dedupe(conflicts):
        """Drop repeats of the same contradiction reported by several detectors.

        The scorer matches each reference conflict to one submitted conflict, so
        extra copies only count as false positives. Identity is the competing
        values plus the pages/snapshots they come from; the copy that carries a
        resolution wins.
        """
        best = {}
        order = []
        for conflict in conflicts:
            key = (tuple(sorted(conflict.values)),
                   frozenset((source.url, source.snapshot_id)
                             for source in conflict.sources))
            if key not in best:
                best[key] = conflict
                order.append(key)
            elif best[key].resolution is None and conflict.resolution is not None:
                best[key] = conflict
        return [best[key] for key in order]
