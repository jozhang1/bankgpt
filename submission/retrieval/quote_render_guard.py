"""Withhold cross-Fact competing quotes until an explicit selection is made.

This is an output gate, not an adjudicator: it never chooses a newer value,
merges Fact identities, asserts supersession, or changes effective intervals.
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import re
from uuid import UUID

from facts.conflict_detector import ConflictDetector
from facts.fact_matching import MatchStatus
from foundation.products import PRODUCTS
from facts.qualifier_schema import compare_fact_qualifiers
from facts.qualifier_schema import QUALIFIER_SCHEMA

_RATE_ATTRIBUTES = frozenset({'apr', 'apy', 'interest_rate', 'dividend_rate'})
_QUALIFIER_SPECS = {spec.key: spec for spec in QUALIFIER_SCHEMA}


def _segment_neutral_product_name(product):
    tokens = re.findall(r'[a-z0-9]+', product.name.casefold())
    return ' '.join(token for token in tokens if token != product.segment.casefold())


def contextual_fact_quotes(primary_fact, primary_observations, explicit_checks,
                           store, snapshot_id):
    """Find source-bound sibling values that clarify an explicitly scoped Fact.

    Returned values are context only. A sibling must share entity, metric,
    source occurrence and snapshot, differ on a categorical qualifier explicitly
    matched by the query, and have a source-bound row and disclosure quote.
    """
    explicit_keys = {
        key for key, status in explicit_checks.items()
        if getattr(status, 'value', status) == 'match'
        and key in _QUALIFIER_SPECS
        and _QUALIFIER_SPECS[key].comparator == 'exact'
        and key in primary_fact.qualifiers
    }
    if not explicit_keys:
        return ()

    primary_source_ids = {
        observation.evidence_id for observation in primary_observations
        if (evidence := store.evidence.get(observation.evidence_id)) is not None
        and evidence.snapshot_id == snapshot_id
        and not evidence.in_recommendations
    }
    if not primary_source_ids:
        return ()

    results = []
    sibling_ids = store.facts_by_entity_attribute.get(
        (primary_fact.entity, primary_fact.attribute), (),
    )
    for fact_id in sibling_ids:
        if fact_id == primary_fact.id:
            continue
        fact = store.facts.get(fact_id)
        if fact is None:
            continue
        differing = tuple(
            key for key in sorted(explicit_keys)
            if key in fact.qualifiers
            and fact.qualifiers[key] != primary_fact.qualifiers[key]
        )
        if not differing:
            continue
        for observation_id in store.obs_by_fact.get(fact.id, ()):
            observation = store.observations.get(observation_id)
            if observation is None or observation.evidence_id not in primary_source_ids:
                continue
            evidence = store.evidence.get(observation.evidence_id)
            support = store.semantic_support.get(observation.id)
            if evidence is None or support is None:
                continue
            row_quote = support.claim_quote
            if not row_quote or row_quote not in evidence.text:
                continue
            qualifier_terms = tuple(
                str(fact.qualifiers[key]).replace('_', ' ').casefold()
                for key in differing
            )
            disclosure = next((quote for quote in support.context_quotes
                               if quote and quote in evidence.text
                               and observation.value.strip() in quote
                               and all(term in quote.casefold()
                                       for term in qualifier_terms)), None)
            if disclosure is None:
                continue
            results.append((fact, observation, evidence, row_quote, disclosure,
                            differing))
    return tuple(results)




@dataclass(frozen=True)
class FactAssociationRule:
    """Describe how one Fact may add a related Fact to a rendered answer.

    The association never changes either Fact's identity. By default, Facts
    must share an entity and exact qualifier set. use_entity_refs enables a
    cross-entity lookup through source-bound canonical entity references on an
    Observation. Snapshot eligibility and citations are checked at render.
    """
    primary_attribute: str
    related_attribute: str
    label: str
    separator: str = ": "
    use_entity_refs: bool = False
    match_qualifiers: tuple[str, ...] = ()


# Presentation relationships are data, not branching logic in the selector or
# renderer. Add domain relationships here (or pass a different rule set).
DEFAULT_FACT_ASSOCIATIONS = (
    # Loan tables commonly publish the note rate and APR as adjacent metrics
    # for the same quote. Keep the requested rate as the selected Fact, then
    # attach APR only when both Facts share an exact identity and evidence.
    FactAssociationRule('interest_rate', 'apr', 'APR'),
    FactAssociationRule('apy', 'minimum_earning_balance', 'minimum to earn'),
    FactAssociationRule('supported_payment_wallet', 'default_payment_option',
                        'default card selection', match_qualifiers=(
                            'statement_topic', 'card_type',
                        )),
    FactAssociationRule('eligibility_requirement', 'geographic_scope',
                        'geographic charter area', separator=':\n',
                        use_entity_refs=True),
)


def find_associated_facts(
    primary_candidates, store, rules=DEFAULT_FACT_ASSOCIATIONS, *,
    snapshot_by_primary=None,
):
    """Associate related Facts independently of query matching or selection.

    The returned mapping is context only. Related Facts are not turned into
    query candidates; the renderer may consult them only after selecting a
    primary Fact. A cross-entity lookup requires a source-bound, uniquely
    normalized EntityReference on one of the primary Fact's observations.
    """
    facts = {fact.id: fact for fact in primary_candidates}
    related_by_primary = {}
    for primary in facts.values():
        related = []
        for rule in rules:
            if primary.attribute != rule.primary_attribute:
                continue
            if not rule.use_entity_refs:
                related.extend(
                    fact for fact in store.facts.values()
                    if fact.attribute == rule.related_attribute
                    and fact.entity == primary.entity
                    and (
                        all(
                            key in primary.qualifiers and key in fact.qualifiers
                            and primary.qualifiers[key] == fact.qualifiers[key]
                            for key in rule.match_qualifiers
                        )
                        if rule.match_qualifiers
                        else fact.qualifiers == primary.qualifiers
                    )
                )
            else:
                refs = {
                    reference.entity_id
                    for observation_id in store.obs_by_fact.get(primary.id, ())
                    if (snapshot_by_primary is None
                        or store.evidence[store.observations[observation_id].evidence_id].snapshot_id
                        == snapshot_by_primary.get(primary.id))
                    for reference in store.observations[observation_id].entity_refs
                    if reference.entity_id
                }
                related.extend(
                    fact for fact in store.facts.values()
                    if fact.attribute == rule.related_attribute
                    and fact.entity in refs
                    and all(
                        key in primary.qualifiers and key in fact.qualifiers
                        and primary.qualifiers[key] == fact.qualifiers[key]
                        for key in rule.match_qualifiers
                    )
                )
        if related:
            related_by_primary[primary.id] = tuple({fact.id: fact for fact in related}.values())
    return related_by_primary


def render_associated_values(observations, rule: FactAssociationRule) -> str:
    """Render related values using their stored units."""
    def format_value(observation):
        value = observation.value.strip()
        unit = (observation.unit or '').casefold()
        if unit == 'usd':
            try:
                from decimal import Decimal, InvalidOperation
                amount = Decimal(value.replace(',', '').replace('$', ''))
            except InvalidOperation:
                return value if value.startswith('$') else f'${value}'
            if amount == amount.to_integral_value():
                return f'${amount:,.0f}'
            return f'${amount:,.2f}'.rstrip('0').rstrip('.')
        if unit == 'percent':
            return value + '%'
        return value + (f' {observation.unit}' if observation.unit else '')

    values = sorted({
        format_value(observation)
        for observation in observations
    })
    if not values:
        return ''
    return f"{rule.label}{rule.separator}{' / '.join(values)}"


@dataclass(frozen=True)
class QuoteRelation:
    left_fact_id: UUID
    right_fact_id: UUID
    relation: str  # distinct / consistent / uncertain / competing
    reasons: tuple[str, ...]

    def as_dict(self):
        return {'left_fact_id': str(self.left_fact_id), 'right_fact_id': str(self.right_fact_id),
                'relation': self.relation, 'reasons': list(self.reasons)}


def classify_quote_relations(matches, resolutions) -> tuple[QuoteRelation, ...]:
    """Classify pairs; grouping is not a value, precedence or version decision.

    Missing categorical scope is uncertain, not proven independence. Such an
    offer remains subject to its own query-match checks and does not acquire the
    explicit variant of a neighbour. Aligned variants with unknown/overlapping
    numeric scope retain potential competition. Parentage permits comparison,
    but never proves that a parent's price applies to the requested child.
    """
    permitted = {m.fact.id for m in matches if m.status is not MatchStatus.MISMATCH}
    candidates = sorted((r for r in resolutions if r.fact.id in permitted
                         and r.status != 'not_found' and r.observations
                         and r.fact.attribute in _RATE_ATTRIBUTES), key=lambda r: str(r.fact.id))
    match_by_id = {match.fact.id: match for match in matches}
    parent = {p.id: p.parent_id for p in PRODUCTS}

    def ancestors(entity):
        result, current = set(), parent.get(entity)
        while current is not None and current not in result:
            result.add(current)
            current = parent.get(current)
        return result

    result = []
    for a, b in combinations(candidates, 2):
        fa, fb = a.fact, b.fact
        if fa.id == fb.id:
            continue
        qa, qb = fa.qualifiers, fb.qualifiers
        reasons = []
        if fa.attribute != fb.attribute:
            relation, reasons = 'distinct', ['different_metric']
        elif a.snapshot_id != b.snapshot_id:
            relation, reasons = 'distinct', ['different_snapshot']
        elif not (fa.entity == fb.entity or fa.entity in ancestors(fb.entity)
                  or fb.entity in ancestors(fa.entity)):
            relation, reasons = 'distinct', ['unrelated_or_sibling_product']
        else:
            condition_status, condition_reasons = compare_fact_qualifiers(qa, qb)
            live_pairs = [(oa, ob) for oa in a.observations for ob in b.observations
                          if ConflictDetector._temporal_relation(oa, ob) != 'disjoint']
            if condition_status is MatchStatus.MISMATCH:
                relation = 'distinct'
                reasons = list(condition_reasons)
            elif not live_pairs:
                relation, reasons = 'distinct', ['disjoint_effective_intervals']
            elif all(ConflictDetector._value_relation(oa, ob) == 'equal' for oa, ob in live_pairs):
                relation, reasons = 'consistent', ['equal_values_do_not_establish_same_fact_identity']
            elif condition_status is MatchStatus.UNKNOWN:
                relation = 'uncertain'
                reasons = list(condition_reasons)
                # A broad rates request should list distinct condition tiers.
                # Require quote selection only when the query asks about the
                # unresolved dimension, or product ownership itself is unknown.
                unresolved_dimensions = {
                    reason.split(':', 1)[1]
                    for reason in condition_reasons
                    if ':' in reason
                }
                requested_dimensions = {
                    key
                    for fact_id in (fa.id, fb.id)
                    for key, _ in match_by_id[fact_id].checks
                    if key not in {'product', 'attribute'}
                }
                product_scope_unknown = any(
                    dict(match_by_id[fact_id].checks).get('product') is MatchStatus.UNKNOWN
                    for fact_id in (fa.id, fb.id)
                )
                if unresolved_dimensions & requested_dimensions or product_scope_unknown:
                    reasons.append('quote_selection_required')
            else:
                relation = 'competing'
                reasons = ['different_values_with_potentially_overlapping_conditions']
                if fa.entity != fb.entity:
                    reasons.append('product_lineage_does_not_establish_price_equivalence')
                if any(ConflictDetector._temporal_relation(oa, ob) == 'unknown' for oa, ob in live_pairs):
                    reasons.append('effective_overlap_unknown')
        result.append(QuoteRelation(fa.id, fb.id, relation, tuple(reasons)))
    return tuple(result)


def competitors_from_relations(pairs) -> dict[UUID, tuple[UUID, ...]]:
    relations = defaultdict(set)
    for pair in pairs:
        if pair.relation == 'competing' or 'quote_selection_required' in pair.reasons:
            relations[pair.left_fact_id].add(pair.right_fact_id)
            relations[pair.right_fact_id].add(pair.left_fact_id)
    return {fact_id: tuple(sorted(others, key=str))
            for fact_id, others in sorted(relations.items(), key=lambda pair: str(pair[0]))}


def deduplicate_consistent_facts(
    matches, resolutions, accepted_observation_ids, *, segment=None,
):
    """Group independently stored Facts that can share one rendered assertion.

    This is presentation deduplication only. It requires the same selected
    snapshot, catalog-equivalent entity identity and qualifiers, a resolved query match,
    equal values/units, and all observations admitted by the answerability
    gate. The Fact and Observation records remain untouched; callers should
    continue attaching provenance from every member of each group.
    """
    accepted = {str(value) for value in accepted_observation_ids}
    match_by_id = {match.fact.id: match for match in matches}
    resolution_by_id = {resolution.fact.id: resolution for resolution in resolutions}
    parent = {product.id: product.parent_id for product in PRODUCTS}

    def related_entities(left, right):
        if left == right:
            return True
        def ancestors(entity):
            found, current = set(), parent.get(entity)
            while current is not None and current not in found:
                found.add(current)
                current = parent.get(current)
            return found
        if left in ancestors(right) or right in ancestors(left):
            return True
        left_product, right_product = products.get(left), products.get(right)
        if (left_product is None or right_product is None
                or left_product.category != right_product.category
                or left_product.segment == right_product.segment
                or 'shared' in {left_product.segment, right_product.segment}
                or segment is not None
                or _segment_neutral_product_name(left_product)
                != _segment_neutral_product_name(right_product)):
            return False
        return True

    products = {product.id: product for product in PRODUCTS}

    eligible = {}
    for fact_id, match in match_by_id.items():
        resolution = resolution_by_id.get(fact_id)
        if (resolution is None or match.status is not MatchStatus.MATCH
                or resolution.status != 'resolved' or not resolution.snapshot_id
                or not resolution.observations
                or any(str(obs.id) not in accepted for obs in resolution.observations)):
            continue
        eligible[fact_id] = resolution

    def equivalent(left, right):
        a, b = left.fact, right.fact
        condition_status, _ = compare_fact_qualifiers(a.qualifiers, b.qualifiers)
        if (a.attribute != b.attribute or condition_status is not MatchStatus.MATCH
                or left.snapshot_id != right.snapshot_id
                or not related_entities(a.entity, b.entity)):
            return False
        return all(
            ConflictDetector._value_relation(oa, ob) == 'equal'
            and ConflictDetector._temporal_relation(oa, ob) != 'disjoint'
            for oa in left.observations for ob in right.observations
        )

    # Build deterministic cliques so a merely transitive chain cannot merge
    # two members whose conditions or values disagree with each other.
    groups = []
    remaining = sorted(eligible, key=str)
    while remaining:
        first = remaining.pop(0)
        group = [first]
        for candidate in tuple(remaining):
            if all(equivalent(eligible[candidate], eligible[member]) for member in group):
                group.append(candidate)
                remaining.remove(candidate)
        if len(group) > 1:
            groups.append(tuple(group))
    return tuple(groups)


def grouped_entity_label(facts, names) -> str:
    """Choose one concise label for facts grouped for display.

    Product peers with matching catalog identity across segments use one
    shared label. Other groups keep their explicit member names.
    """
    labels = sorted({names.get(fact.entity, fact.entity) for fact in facts},
                    key=str.casefold)
    if len(labels) <= 1:
        return labels[0] if labels else ""

    products_by_id = {product.id: product for product in PRODUCTS}
    products = [products_by_id.get(fact.entity) for fact in facts]
    if (all(product is not None for product in products)
            and len({product.category for product in products}) == 1
            and len({_segment_neutral_product_name(product) for product in products}) == 1
            and len({product.segment for product in products}) > 1):
        shared_label = min(labels, key=lambda label: (len(label), label.casefold()))
        segments = ' and '.join(sorted({product.segment for product in products}))
        return f'{shared_label} ({segments})'

    return ' / '.join(labels)


def pending_quote_competitors(matches, resolutions) -> dict[UUID, tuple[UUID, ...]]:
    """Gate competing and possibly overlapping quotes without choosing values."""
    return competitors_from_relations(classify_quote_relations(matches, resolutions))
