"""Conservative question-to-Fact selection; no value or date inference."""
import re
from decimal import Decimal, InvalidOperation

from foundation.contract import Fact
from facts.fact_matching import FactMatch, MatchStatus
from foundation.products import CATEGORY_QUERY_TERMS, PRODUCT_CATEGORY_GROUPS, PRODUCTS
from foundation.store import KnowledgeStore
from retrieval.semantic_query import SemanticQueryMatcher, selected_snapshot
from foundation.entities import DUPACO_ENTITIES
from retrieval.query_scope import QueryScopeParser, match_named_records
from facts.qualifier_schema import compare_condition, compare_interval_condition
from facts.semantic_fact_profile import DETERMINISTIC_FIELD_ATTRIBUTES
from retrieval.answer_selector import retain_non_mismatches


def _label(text: str) -> str:
    return QueryScopeParser.normalize_query_label(text)


class FactSelector:
    def __init__(self, store: KnowledgeStore, *, entities=DUPACO_ENTITIES):
        self.store = store
        self.scope_parser = QueryScopeParser()
        self.last_scope = None
        self.last_semantic_attributes = set()
        self.semantic = SemanticQueryMatcher(store, entities=entities)

    @staticmethod
    def select_resolved_extreme(matches, resolutions, shape, *, as_of=None):
        """Select extrema only from completed temporal resolutions.

        If any candidate for an entity/attribute has unknown query conditions,
        conflicting values, unsupported temporal applicability, or a non-scalar
        value, keep that entire comparison set for adjudication. A dated,
        open-ended latest-supported assertion may participate with a caveat.
        """
        if shape.kind != 'extreme' or not matches:
            return matches, resolutions
        resolution_by_id = {result.fact.id: result for result in resolutions}
        grouped = {}
        blocked = set()
        for match in matches:
            fact = match.fact
            identity = (fact.entity, fact.attribute)
            resolution = resolution_by_id.get(fact.id)
            # Only facts with an actual temporal resolution participate.
            # Missing observations have already been classified as not_found.
            if resolution is None:
                continue
            if (match.status is MatchStatus.UNKNOWN or resolution.status != 'resolved'
                    or (as_of is not None and resolution.temporal_certainty not in {
                        'explicit', 'latest_supported'
                    })):
                blocked.add(identity)
                continue
            values = set()
            units = {observation.unit for observation in resolution.observations}
            if len(units) == 1:
                for observation in resolution.observations:
                    try:
                        values.add(Decimal(observation.value.replace(',', '').strip()))
                    except (InvalidOperation, AttributeError):
                        values.clear()
                        break
            if len(values) != 1 or len(units) != 1:
                blocked.add(identity)
                continue
            unit = next(iter(units))
            grouped.setdefault((*identity, unit), []).append(
                (match, resolution, next(iter(values)))
            )

        kept_ids = set()
        for key, group in grouped.items():
            if key[:2] in blocked:
                kept_ids.update(match.fact.id for match, _, _ in group)
                continue
            values = [value for _, _, value in group]
            target = max(values) if shape.direction == 'maximum' else min(values)
            kept_ids.update(match.fact.id for match, _, value in group if value == target)
        # Preserve every candidate in an uncertain comparison set, including
        # candidates that could not be assigned a numeric value or unit.
        kept_ids.update(
            match.fact.id for match in matches
            if match.fact.id in resolution_by_id
            and (match.fact.entity, match.fact.attribute) in blocked
        )
        # Only temporally available facts continue to the answer-selection
        # stage. Excluded/not-found observations remain in the store and audit.
        selected_matches = [match for match in matches
                            if match.fact.id in resolution_by_id
                            and match.fact.id in kept_ids]
        selected_ids = {match.fact.id for match in selected_matches}
        selected_resolutions = [result for result in resolutions if result.fact.id in selected_ids]
        return selected_matches, selected_resolutions

    def select(self, question: str) -> list[Fact] | None:
        """Compatibility view: return only definite matches, never unknowns."""
        matches = self.select_matches(question)
        return None if matches is None else [m.fact for m in matches if m.status is MatchStatus.MATCH]

    def select_matches(self, question: str, *, as_of: str | None = None) -> list[FactMatch] | None:
        """Retain MATCH/UNKNOWN candidates and exclude explicit mismatches.

        None defers unsupported query shapes to Evidence retrieval; [] means
        no compatible candidate. Missing requested qualifiers are UNKNOWN.
        Unspecified conditions do not create checks or unknowns.
        """
        self.last_scope = None
        semantic_attributes = self.scope_parser.semantic_attributes(question)
        self.last_semantic_attributes = semantic_attributes
        semantic = self.semantic.select_matches(
            question, as_of=as_of, attributes=semantic_attributes,
        )
        if semantic is not None:
            attrs = semantic_attributes
            # A recognized field alone does not establish the requested owner
            # or conditions. Preserve the existing Evidence/link route when
            # the scalar rule cannot answer this query.
            if attrs <= DETERMINISTIC_FIELD_ATTRIBUTES and not any(
                m.status is MatchStatus.MATCH for m in semantic
            ):
                return None
            if any(self.store.facts[self.store.observations[oid].fact_id].attribute in attrs
                   for oid in self.store.semantic_support):
                self.last_scope = self.scope_parser.parse(
                    question, attributes=attrs, as_of=as_of,
                )
                return semantic
        text = _label(question)
        if re.search(r'\b(link|url|where|how|change|changed|between may|compare)\b', text):
            return None
        attributes = self.scope_parser.metric_attributes(text)
        if not attributes:
            if not self.scope_parser.requests_rate(text):
                return None
        matches = match_named_records(question, PRODUCTS, normalize=_label)
        family = None
        if re.search(r'\b(?:mortgage|home loan)\b', text):
            family = 'mortgage'
        elif re.search(r'\bchecking\b', text):
            family = 'checking'
        elif re.search(r'\bcredit card\b', text):
            family = 'credit_card'
        elif re.search(r'\b(?:certificate|cd|cds)\b', text):
            family = 'certificate'
        category_groups = {
            group for group, terms in CATEGORY_QUERY_TERMS.items()
            if any(re.search(r'\b' + re.escape(term) + r'\b', text) for term in terms)
        }
        if matches:
            products = {product.id for product in matches}
        elif family == 'mortgage':
            products = {p.id for p in PRODUCTS if p.id in {
                'personal_home_loans', 'personal_home_purchase', 'personal_home_refinance',
                'personal_fixed_mortgage', 'personal_adjustable_mortgage'}}
        elif family:
            products = {p.id for p in PRODUCTS if p.category == family}
        elif category_groups:
            categories_in_groups = set().union(
                *(PRODUCT_CATEGORY_GROUPS[group] for group in category_groups)
            )
            products = {
                p.id for p in PRODUCTS
                if p.category in categories_in_groups and p.kind == 'product'
            }
        else:
            return None
        categories = {p.category for p in PRODUCTS if p.id in products}
        # Generic rates retain source metric distinctions. Deposit "interest"
        # is a colloquial request: return APY/dividend, never relabel either.
        if not attributes:
            if family == 'mortgage':
                attributes = {'interest_rate', 'apr'}
            elif categories <= {
                'checking', 'certificate', 'retirement_instrument', 'savings',
                'money_market', 'health_savings',
            }:
                attributes = {'apy', 'dividend_rate'}
            elif categories == {'credit_card'}:
                attributes = {'apr'}
            else:
                return None
        elif attributes == {'interest_rate'} and categories <= {
            'checking', 'certificate', 'retirement_instrument', 'savings',
            'money_market', 'health_savings',
        }:
            attributes = {'apy', 'dividend_rate'}
        segment = 'business' if re.search(r'\bbusiness\b', text) else (
            'personal' if re.search(r'\bpersonal\b', text) else None)
        if segment:
            products = {p.id for p in PRODUCTS if p.id in products and p.segment in (segment, 'shared')}
        # A source may publish a rate at family level. Preserve those broader
        # candidates with their original entity; never relabel them as the child.
        ancestors = set(products)
        while True:
            expanded = ancestors | {p.parent_id for p in PRODUCTS
                                    if p.id in ancestors and p.parent_id is not None}
            if expanded == ancestors:
                break
            ancestors = expanded
        families = {p.id for p in PRODUCTS if p.kind == 'family' and p.id in ancestors}
        # Include registered child products of an explicitly requested family.
        while True:
            expanded = products | {p.id for p in PRODUCTS if p.parent_id in products}
            if expanded == products:
                break
            products = expanded
        broader_only = families - products
        products |= families
        if family == 'certificate':
            if re.search(r'\bspecial\b', text):
                products = {pid for pid in products if 'special' in pid}
            elif re.search(r'\bregular\b', text):
                products = {pid for pid in products if 'regular' in pid}
            if re.search(r'\bira\b', text):
                products = {p.id for p in PRODUCTS if 'ira' in p.id and 'certificate' in p.id}
        scope = self.scope_parser.parse(question, product_ids=products, attributes=attributes,
                                        family=family, categories=categories, as_of=as_of)
        self.last_scope = scope
        if not scope.supported:
            return None
        selected = selected_snapshot(self.store, as_of) if scope.requested_value is not None else None
        if scope.requested_value is not None and selected is None:
            return None
        broad_category_value_query = (
            bool(category_groups) and not matches and scope.requested_value is not None
        )
        category_value_fact_ids: set[object] = set()
        exact_value_fact_ids: set[object] = set()
        if broad_category_value_query:
            # For a category-level existence question, retain each product's
            # top observed tier as context, plus every exact requested-value
            # tier. Product identity stays separate, so a Money Market rate
            # cannot be presented as a Share Savings rate.
            rates_by_entity: dict[str, list[tuple[Decimal, Fact]]] = {}
            categories_in_groups = set().union(
                *(PRODUCT_CATEGORY_GROUPS[group] for group in category_groups)
            )
            for fact in self.store.facts.values():
                if (fact.entity not in products or fact.attribute not in attributes
                        or not any(p.id == fact.entity and p.category in categories_in_groups
                                   for p in PRODUCTS)):
                    continue
                for observation_id in self.store.obs_by_fact.get(fact.id, ()):
                    observation = self.store.observations.get(observation_id)
                    evidence = (self.store.evidence.get(observation.evidence_id)
                                if observation is not None else None)
                    if (observation is None or evidence is None
                            or evidence.snapshot_id != selected.snapshot_id
                            or evidence.in_recommendations
                            or observation.unit != scope.requested_unit):
                        continue
                    try:
                        value = Decimal(observation.value.replace(',', '').strip())
                    except (InvalidOperation, AttributeError):
                        continue
                    rates_by_entity.setdefault(fact.entity, []).append((value, fact))
                    if value == scope.requested_value:
                        exact_value_fact_ids.add(fact.id)
            for entity_rates in rates_by_entity.values():
                maximum = max(value for value, _ in entity_rates)
                category_value_fact_ids.update(
                    fact.id for value, fact in entity_rates if value == maximum
                )
            category_value_fact_ids.update(exact_value_fact_ids)
            if (scope.result_shape.kind == 'existence'
                    and exact_value_fact_ids):
                # A positive existence query needs only exact hits. If there
                # are no hits, retain the scope's maxima as concise evidence
                # for the nearest supported answer instead of returning an
                # empty result with no useful context.
                category_value_fact_ids = set(exact_value_fact_ids)
        requested = dict(scope.conditions)
        if scope.term_months is not None:
            requested['term_months'] = scope.term_months
        if scope.model_year is not None:
            requested['model_year'] = scope.model_year
        if scope.amount is not None and scope.amount_kind is not None:
            requested[scope.amount_kind] = scope.amount
        range_key = scope.amount_kind if scope.amount_min is not None else None
        result = []
        for fact in self.store.facts.values():
            if fact.entity not in products or fact.attribute not in attributes:
                continue
            if scope.requested_value is not None and not broad_category_value_query:
                matching_observation = False
                for observation_id in self.store.obs_by_fact.get(fact.id, ()):
                    observation = self.store.observations.get(observation_id)
                    evidence = (self.store.evidence.get(observation.evidence_id)
                                if observation is not None else None)
                    if (observation is None or evidence is None
                            or evidence.snapshot_id != selected.snapshot_id
                            or evidence.in_recommendations
                            or observation.unit != scope.requested_unit):
                        continue
                    try:
                        value = Decimal(observation.value.replace(',', '').strip())
                    except (InvalidOperation, AttributeError):
                        continue
                    if value == scope.requested_value:
                        matching_observation = True
                        break
                if not matching_observation:
                    continue
            elif broad_category_value_query and fact.id not in category_value_fact_ids:
                continue
            checks: list[tuple[str, MatchStatus]] = [
                ('product', MatchStatus.UNKNOWN if fact.entity in broader_only else MatchStatus.MATCH),
                ('attribute', MatchStatus.MATCH),
            ]
            q = fact.qualifiers
            checks.extend((key, compare_condition(q, key, value))
                          for key, value in sorted(requested.items()))
            if (scope.requested_value is not None
                    and (not broad_category_value_query or fact.id in exact_value_fact_ids)):
                checks.append(('requested_value', MatchStatus.MATCH))
            if range_key is not None:
                checks.append((
                    range_key + '_range',
                    compare_interval_condition(
                        q, range_key, scope.amount_min, scope.amount_max,
                        lower_inclusive=scope.amount_min_inclusive,
                        upper_inclusive=scope.amount_max_inclusive,
                    ),
                ))
            match = FactMatch(fact, tuple(checks))
            result.append(match)
        result = retain_non_mismatches(result)
        return sorted(result, key=lambda m: (m.fact.entity, m.fact.attribute, m.fact.key))
