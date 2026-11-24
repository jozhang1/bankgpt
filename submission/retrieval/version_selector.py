"""Select supported versions of explicitly dated rate assertions.

The same-Fact rate path uses explicit source dates and matched query conditions;
it never infers an end date or withdrawal, and never changes stored observations.
"""
from dataclasses import dataclass, replace
from datetime import date, datetime
import re
from uuid import UUID

from facts.fact_matching import MatchStatus
from facts.temporal_resolver import TemporalResolver
from foundation.products import PRODUCTS


_RATE_ATTRIBUTES = frozenset({'apr', 'apy', 'dividend_rate', 'interest_rate'})
_PRODUCT_BY_ID = {product.id: product for product in PRODUCTS}
_EFFECTIVE_DATE_MARKER = re.compile(
    r'\b(?:rates?|apr|apy|interest rates?)\s+(?:are\s+)?effective\s+(?:as of|on)\s+'
    r'(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VersionSelection:
    fact_id: UUID
    status: str  # preferred / unresolved
    selected: tuple[UUID, ...]
    suppressed: tuple[UUID, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()
    reason: str = ''
    policy_id: str | None = None


@dataclass(frozen=True)
class QuoteScheduleSelection:
    """Audit a latest-effective schedule preference across overlapping Facts."""

    selected_fact_ids: tuple[UUID, ...]
    suppressed_fact_ids: tuple[UUID, ...]
    selected_observation_ids: tuple[UUID, ...]
    suppressed_observation_ids: tuple[UUID, ...]
    policy_id: str
    reason: str


def _parse_explicit_date(value: str) -> date | None:
    formats = ('%Y-%m-%d', '%m/%d/%Y', '%m/%d/%y', '%m-%d-%Y', '%m-%d-%y')
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


class VersionSelector:
    def __init__(self, store):
        self.store = store
        self.resolver = TemporalResolver(store)

    def _has_explicit_effective_date(self, observation) -> bool:
        """Require the observation's normalized start date in its cited text."""
        source = self.store.evidence[observation.evidence_id]
        if source.in_recommendations or observation.effective_from is None:
            return False
        # Markdown italic underscores are word characters to the regex engine;
        # remove that formatting before checking the source's date statement.
        date_text = source.text.replace('_', ' ')
        dates = {
            parsed
            for raw in _EFFECTIVE_DATE_MARKER.findall(date_text)
            if (parsed := _parse_explicit_date(raw)) is not None
        }
        return observation.effective_from in dates

    def _latest_effective_quote(self, resolution, match, *, as_of=None):
        """Prefer the latest explicitly effective same-Fact rate assertion.

        This orders dated published rate assertions for the requested snapshot;
        it does not infer that an earlier assertion's offer was withdrawn or
        add an effective_to bound.
        """
        fact = resolution.fact
        original = tuple(resolution.observations)

        def unchanged(reason):
            return resolution, VersionSelection(
                fact.id, 'unresolved', tuple(o.id for o in original), reason=reason,
                policy_id='latest_effective_rate_assertion_v1',
            )

        product = _PRODUCT_BY_ID.get(fact.entity)
        if (product is None or fact.attribute not in _RATE_ATTRIBUTES or match.fact != fact
                or match.status is not MatchStatus.MATCH):
            return unchanged('Rate scope is not established for latest-effective selection.')
        if resolution != self.resolver.resolve(fact, as_of=as_of):
            return unchanged('Input is not the complete stored temporal resolution.')
        if resolution.status not in {'ambiguous', 'conflicting'} or len(original) < 2:
            return unchanged('No differing dated rate assertions require version selection.')
        if any(not self._has_explicit_effective_date(o) for o in original):
            return unchanged('Every competing rate needs a source-verified effective date.')
        if len({o.unit for o in original}) != 1:
            return unchanged('Competing rates use different units.')

        latest_date = max(o.effective_from for o in original)
        latest = tuple(o for o in original if o.effective_from == latest_date)
        if any(self.resolver._value_relation(fact, latest[0], o) != 'equal'
               for o in latest[1:]):
            return unchanged('Latest-effective rows disagree; no quote is preferred.')
        if len(latest) == len(original):
            return unchanged('All competing values have the same effective date.')

        # A bounded older offer that still covers the query date is explicit
        # competing applicability, so an effective-date ranking cannot hide it.
        reference = date.fromisoformat(
            as_of or self.store.snapshots[resolution.snapshot_id].taken_on
        )
        older = tuple(o for o in original if o.effective_from != latest_date)
        if any(o.effective_to is not None and o.effective_to >= reference for o in older):
            return unchanged('An older quote is explicitly valid on the query date.')

        reason = (
            f'Latest source-verified rate assertion effective {latest_date.isoformat()} '
            'preferred for this snapshot. Earlier observations remain stored; no '
            'effective end date or withdrawal is inferred.'
        )
        audit = VersionSelection(
            fact.id, 'preferred', tuple(o.id for o in latest),
            tuple(o.id for o in older),
            tuple(sorted({o.evidence_id for o in latest}, key=str)),
            reason, 'latest_effective_rate_assertion_v1',
        )
        projected = replace(
            resolution, observations=latest, status='resolved', reason=reason,
            temporal_certainty='snapshot_only',
        )
        return projected, audit

    def select(self, resolution, match, *, as_of=None):
        """Return an ephemeral resolution projection and its verifiable audit."""
        return self._latest_effective_quote(resolution, match, as_of=as_of)

    def select_all(self, matches, resolutions, *, as_of=None):
        by_fact = {m.fact.id: m for m in matches}
        projected, audits = [], []
        for resolution in resolutions:
            result, audit = self.select(resolution, by_fact[resolution.fact.id], as_of=as_of)
            projected.append(result)
            audits.append(audit)
        return projected, tuple(audits)

    @staticmethod
    def _related_products(left: str, right: str) -> bool:
        parents = {product.id: product.parent_id for product in PRODUCTS}

        def ancestors(entity):
            found, current = set(), parents.get(entity)
            while current is not None and current not in found:
                found.add(current)
                current = parents.get(current)
            return found

        return left == right or left in ancestors(right) or right in ancestors(left)

    @staticmethod
    def _schedule_scope_compatible(left, right, left_match, right_match) -> bool:
        """Compare offer dimensions, permitting only queried model-year rollover."""
        left_q, right_q = left.fact.qualifiers, right.fact.qualifiers
        year_keys = {'model_year_from', 'model_year_to'}
        ignored = set()
        if any(left_q.get(key) != right_q.get(key) for key in year_keys):
            left_checks, right_checks = dict(left_match.checks), dict(right_match.checks)
            if (left_checks.get('model_year') is not MatchStatus.MATCH
                    or right_checks.get('model_year') is not MatchStatus.MATCH):
                return False
            ignored.update(year_keys)

        if left_q.get('rate_basis') != right_q.get('rate_basis'):
            # A missing source label may be enriched in a later schedule, but
            # two different explicit bases remain separate offer conditions.
            if left_q.get('rate_basis') is not None and right_q.get('rate_basis') is not None:
                return False
            left_checks, right_checks = dict(left_match.checks), dict(right_match.checks)
            if ('model_year' not in left_checks or 'model_year' not in right_checks
                    or left_checks['model_year'] is not MatchStatus.MATCH
                    or right_checks['model_year'] is not MatchStatus.MATCH):
                return False
            ignored.add('rate_basis')

        all_keys = (set(left_q) | set(right_q)) - ignored
        return all(left_q.get(key) == right_q.get(key) for key in all_keys)

    @staticmethod
    def _normalized_phrase(value: str) -> str:
        tokens = re.findall(r'[a-z0-9]+', value.casefold())
        singular = {'loans': 'loan', 'accounts': 'account', 'rates': 'rate',
                    'certificates': 'certificate', 'cards': 'card'}
        return ' '.join(singular.get(token, token) for token in tokens)

    def _family_product_supported(self, resolution, family_id, requested_product_ids):
        products = {product.id: product for product in PRODUCTS}
        family = products.get(family_id)
        if family is None or family.kind != 'family':
            return False
        children = [product for product in PRODUCTS
                    if product.parent_id == family_id and product.id in requested_product_ids]
        for observation in resolution.observations:
            evidence = self.store.evidence[observation.evidence_id]
            if evidence.in_recommendations:
                continue
            headings = [evidence.source_title]
            headings.extend(re.findall(r'^#{1,6}\s+(.+?)\s*#*\s*$', evidence.text, re.M))
            for child in children:
                for label in (child.name, *child.aliases):
                    phrase = self._normalized_phrase(label)
                    if phrase and any(re.search(
                        rf'(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])',
                        self._normalized_phrase(heading),
                    ) for heading in headings):
                        return True
        return False

    def select_overlapping_schedule_quotes(
        self, matches, resolutions, *, requested_product_ids=(),
    ):
        """Select newer dated offers only for the same proven query scope.

        Exact condition rows can be versioned across parent/child product
        records. Model-year buckets can roll over only when the question names
        a year and both intervals match that year. Amount tiers, loan purposes,
        terms, phases and other explicit conditions remain independent.
        """
        requested_product_ids = set(requested_product_ids)
        match_by_id = {match.fact.id: match for match in matches}
        candidates = []
        for resolution in resolutions:
            match = match_by_id.get(resolution.fact.id)
            if (match is None or resolution.status != 'resolved' or not resolution.observations
                    or resolution.fact.attribute not in _RATE_ATTRIBUTES
                    or (_PRODUCT_BY_ID.get(resolution.fact.entity) is None)
                    or _PRODUCT_BY_ID[resolution.fact.entity].category != 'loan'
                    or not all(self._has_explicit_effective_date(o)
                               for o in resolution.observations)):
                continue
            if match.status is MatchStatus.MATCH:
                candidates.append(resolution)
                continue
            unknown = {key for key, status in match.checks if status is MatchStatus.UNKNOWN}
            if (unknown != {'product'}
                    or any(status is MatchStatus.MISMATCH for _, status in match.checks)
                    or not self._family_product_supported(
                        resolution, match.fact.entity, requested_product_ids,
                    )):
                continue
            # Catalog lineage alone is insufficient. Bind a broad family quote
            # to the requested child only when this source names that child.
            match_by_id[match.fact.id] = replace(
                match,
                checks=tuple((key, MatchStatus.MATCH if key == 'product' else status)
                             for key, status in match.checks),
            )
            candidates.append(resolution)

        # Deterministic clique grouping prevents a merely transitive overlap
        # chain from joining incompatible loan/product scopes.
        groups = []
        for candidate in sorted(candidates, key=lambda r: str(r.fact.id)):
            placed = False
            for group in groups:
                if all(self._schedule_pair_compatible(
                    candidate, member, match_by_id,
                ) for member in group):
                    group.append(candidate)
                    placed = True
                    break
            if not placed:
                groups.append([candidate])

        suppressed = set()
        decisions = []
        for group in groups:
            if len(group) < 2:
                continue
            dates = [max(o.effective_from for o in item.observations) for item in group]
            if len(set(dates)) < 2:
                continue
            latest_date = max(dates)
            winners = [item for item, day in zip(group, dates) if day == latest_date]
            losers = [item for item, day in zip(group, dates) if day < latest_date]
            winner_values = {
                (o.value, o.unit)
                for item in winners for o in item.observations
                if o.effective_from == latest_date
            }
            if len(winner_values) != 1:
                continue
            reference = max(
                date.fromisoformat(self.store.snapshots[item.snapshot_id].taken_on)
                for item in group if item.snapshot_id is not None
            )
            if any(o.effective_to is not None and o.effective_to >= reference
                   for item in losers for o in item.observations):
                continue
            if any((o.value, o.unit) != next(iter(winner_values))
                   for item in losers for o in item.observations):
                suppressed.update(item.fact.id for item in losers)
                decisions.append(QuoteScheduleSelection(
                    selected_fact_ids=tuple(sorted((item.fact.id for item in winners), key=str)),
                    suppressed_fact_ids=tuple(sorted((item.fact.id for item in losers), key=str)),
                    selected_observation_ids=tuple(sorted(
                        (o.id for item in winners for o in item.observations
                         if o.effective_from == latest_date), key=str)),
                    suppressed_observation_ids=tuple(sorted(
                        (o.id for item in losers for o in item.observations), key=str)),
                    policy_id='latest_effective_overlapping_rate_schedule_v1',
                    reason=(f'Latest source-verified schedule effective {latest_date.isoformat()} '
                            'matches the requested conditions. Older observations remain stored; '
                            'no end date or withdrawal is inferred.'),
                ))
        kept_matches = [match_by_id.get(m.fact.id, m) for m in matches
                        if m.fact.id not in suppressed]
        kept_resolutions = [r for r in resolutions if r.fact.id not in suppressed]
        return kept_matches, kept_resolutions, tuple(decisions)

    def _schedule_pair_compatible(self, left, right, match_by_id):
        if (left.fact.attribute != right.fact.attribute
                or left.snapshot_id != right.snapshot_id
                or not self._related_products(left.fact.entity, right.fact.entity)):
            return False
        left_match, right_match = match_by_id[left.fact.id], match_by_id[right.fact.id]
        return self._schedule_scope_compatible(left, right, left_match, right_match)
