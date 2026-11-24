"""Explicit rate-query constraints, separate from source eligibility/version choice."""
from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Callable, Iterable, Literal
from foundation.entities import entity_label
from facts.qualifier_schema import parse_declared_conditions


def match_named_records(
    question: str,
    records: Iterable[object],
    *,
    normalize: Callable[[str], str] = entity_label,
    include: Callable[[object], bool] | None = None,
) -> tuple[object, ...]:
    """Return longest exact registered-label matches; never fuzzy-match."""
    text = normalize(question)
    matches = [
        (len(label.split()), record)
        for record in records
        if include is None or include(record)
        for alias in (getattr(record, 'name', ''), *getattr(record, 'aliases', ()))
        if (label := normalize(alias)) and f' {label} ' in f' {text} '
    ]
    if not matches:
        return ()
    specificity = max(length for length, _ in matches)
    return tuple({record.id: record for length, record in matches
                  if length == specificity}.values())


@dataclass(frozen=True)
class ResultShape:
    """Requested answer cardinality, interpreted before rendering.

    ``direction`` is populated only for extrema.  A representative comparison
    is a set of distinct, source-backed representatives, not a complete list.
    """

    kind: Literal[
        'single_value', 'extreme', 'full_collection', 'representative_comparison',
        'existence',
    ] = 'single_value'
    direction: Literal['maximum', 'minimum'] | None = None


def parse_result_shape(question: str) -> ResultShape:
    """Parse broad answer-shape intent without deciding which records win."""
    text = ' '.join(re.findall(r'[a-z0-9]+', question.casefold()))
    if re.search(r'\b(?:highest|maximum|max)\b', text):
        return ResultShape('extreme', 'maximum')
    if re.search(r'\b(?:lowest|minimum|min|least)\b', text):
        return ResultShape('extreme', 'minimum')
    if re.search(r'\b(?:compare|same|different|vary|varies|comparison)\b', text):
        return ResultShape('representative_comparison')
    if re.search(
        r'\b(?:do|does|did)\s+any\b|'
        r'\b(?:is|are|was|were)\s+(?:there\s+)?any\b|'
        r'\bwhether\s+any\b',
        text,
    ):
        return ResultShape('existence')
    if re.search(r'\b(?:all|every|each|list|complete list|full list)\b', text):
        return ResultShape('full_collection')
    return ResultShape('single_value')


@dataclass(frozen=True)
class QueryScope:
    product_ids: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    segment: str | None = None
    family: str | None = None
    term_months: int | None = None
    model_year: int | None = None
    amount: Decimal | None = None
    amount_kind: str | None = None
    amount_min: Decimal | None = None
    amount_max: Decimal | None = None
    amount_min_inclusive: bool = True
    amount_max_inclusive: bool = True
    requested_value: Decimal | None = None
    requested_unit: str | None = None
    result_shape: ResultShape = ResultShape()
    conditions: tuple[tuple[str, str | bool], ...] = ()
    unsupported: tuple[str, ...] = ()
    as_of: str | None = None

    @property
    def supported(self):
        return not self.unsupported


class QueryScopeParser:
    @staticmethod
    def normalize_query_label(text):
        tokens = re.findall(r'[a-z0-9]+', text.casefold())
        singular = {'loans': 'loan', 'accounts': 'account', 'certificates': 'certificate',
                    'cards': 'card', 'rates': 'rate', 'aprs': 'apr', 'apys': 'apy'}
        return ' '.join(singular.get(token, token) for token in tokens)

    @staticmethod
    def semantic_attributes(question):
        """Recognize the supported semantic Fact fields in a question."""
        text = entity_label(re.sub(r"[’']s\b", "", question))
        attrs = set()
        count = bool(re.search(r'\b(?:how many|number of|count|total)\b', text))
        if count and re.search(r'\bbranches\b', text): attrs.add('branch_count')
        if (re.search(r'\b(?:mobile|digital) wallets?\b',text)
                and re.search(r'\b(?:cards?|debit|credit)\b',text)
                and re.search(r'\b(?:can|compatible|support|add|use|work)\b',text)
                and not re.search(r'\b(?:how|fee|cost|lost|safe|security)\b',text)):
            attrs.add('supported_payment_wallet')
        if (re.search(r'\b(?:eligible|eligibility|requirements|who can)\b',text)
                and re.search(r'\b(?:join|membership|member)\b',text)):
            attrs.add('eligibility_requirement')
        if count and re.search(
            r'\b(?:employee count|number of employees|total employees|workforce|'
            r'how many employees (?:does|do|are))\b', text,
        ):
            attrs.add('employee_count')
        if re.search(r'\bvolunteer\w*\b', text) and re.search(r'\bhours\b', text):
            attrs.add('volunteer_hours')
        if re.search(r'\b(?:recognition|recognized|recognised|how many times)\b', text):
            attrs.add('recognition_count')
        if re.search(r'\b(?:rank|ranked|ranking|position)\b', text): attrs.add('ranking_position')
        if re.search(r'\b(?:grant|assistance)\b', text) and re.search(
                r'\b(?:how much|maximum|amount|up to)\b', text): attrs.add('maximum_assistance_amount')
        if re.search(r'\b(?:when|date)\b', text):
            if re.search(r'\b(?:open|opening|opened)\b', text): attrs.add('opening_date')
            if re.search(r'\b(?:close|closing|closed|closure)\b', text): attrs.add('closure_date')
        for attr in ('origination_fee', 'prepayment_fee', 'routing_number', 'phone_number', 'email_address'):
            if attr.replace('_', ' ') in text: attrs.add(attr)
        if re.search(r'\b(?:link|url)\b', text):
            if re.search(r'\b(?:apply|application)\b', text): attrs.add('application_url')
            if re.search(r'\brates?\b', text): attrs.add('rates_url')
            if re.search(r'\blocator\b', text): attrs.add('locator_url')
        return attrs

    @staticmethod
    def metric_attributes(normalized_text):
        """Recognize explicit rate and balance metrics from normalized query text."""
        patterns = {
            'apy': r'\b(?:apy|annual percentage yield)\b',
            'apr': r'\b(?:apr|annual percentage rate)\b',
            'dividend_rate': r'\bdividend rate\b',
            'interest_rate': r'\binterest rate\b',
            'minimum_opening_balance': r'\bminimum (?:opening balance|to open)\b',
            'minimum_earning_balance': r'\bminimum (?:earning balance|to earn)\b',
        }
        return {
            attribute for attribute, pattern in patterns.items()
            if re.search(pattern, normalized_text)
        }

    @staticmethod
    def requests_rate(normalized_text):
        return bool(re.search(r'\brate\b', normalized_text))

    def parse(self, question, *, product_ids=(), attributes=(), family=None, categories=(), as_of=None):
        text = ' '.join(re.findall(r'[a-z0-9]+', question.casefold()))
        result_shape = parse_result_shape(question)
        mortgage = family == 'mortgage' or bool(re.search(r'\b(?:mortgage|home loans?|cash out refinance)\b', text))
        card = set(categories) == {'credit_card'} or bool(re.search(r'\b(?:visa|credit cards?)\b', text))
        auto = bool(re.search(r'\b(?:auto|car|vehicle|motorcycle)\b', text))
        issues, conditions = [], {}
        requested_value = None
        requested_unit = None
        if set(attributes).intersection(
            {'apy', 'apr', 'dividend_rate', 'interest_rate', 'rate'}
        ):
            percentages = re.findall(r'(?<![\w.])(\d+(?:\.\d+)?)\s*%', question)
            if len(percentages) == 1:
                requested_value = Decimal(percentages[0])
                requested_unit = 'percent'
            elif len(percentages) > 1:
                issues.append('multiple_requested_metric_values')
        contexts = set()
        if mortgage:
            contexts.add('mortgage')
        if card:
            contexts.add('credit_card')
        if auto:
            contexts.add('auto')
        conditions, condition_issues = parse_declared_conditions(question, contexts)
        issues.extend(condition_issues)

        def condition(key, value):
            if key in conditions and conditions[key] != value:
                issues.append('conflicting_' + key)
            conditions[key] = value

        if re.search(r'\b\d+\s+(?:and|or|to)\s+\d+\s+(?:month|year)', text) or re.search(
                r'\b\d+\s*[-–—]\s*\d+\s*[- ]?\s*(?:month|year)', question, re.I):
            issues.append('multiple_or_range_terms')
        terms = {int(n) * (12 if u.startswith('year') else 1)
                 for n, u in re.findall(r'\b(\d{1,3})[ -]*(months?|years?)\b', text)}
        if len(terms) > 1 or 0 in terms:
            issues.append('ambiguous_term')
        term = next(iter(terms)) if len(terms) == 1 and 0 not in terms else None
        all_years = set(re.findall(r'\b(?:19|20)\d{2}\b', text)) if auto else set()
        linked = set(re.findall(r'\b((?:19|20)\d{2})\s+(?:model(?: year)?|vehicle|auto|car|motorcycle)\b', text))
        linked.update(re.findall(r'\bmodel year\s+((?:19|20)\d{2})\b', text))
        years = {int(y) for y in linked} if auto else set()
        if auto and all_years - linked:
            issues.append('calendar_year_or_model_year_not_disambiguated')
        if len(years) > 1 or (auto and re.search(r'\b(?:19|20)\d{2}\s*[-–—]\s*\d{2,4}\b', question)):
            issues.append('ambiguous_model_year')
        year = next(iter(years)) if len(years) == 1 else None
        money = re.findall(r'\$\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)(?!\w|,\d|\.\d)', question)
        if question.count('$') != len(money) or re.search(r'-\s*\$', question):
            issues.append('invalid_or_unsupported_amount')
        range_match = re.search(
            r'\bbetween\s+\$\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)'
            r'\s+and\s+\$\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)',
            question, re.I,
        )
        amount_min = amount_max = None
        amount = None
        if range_match:
            amount_min = Decimal(range_match.group(1).replace(',', ''))
            amount_max = Decimal(range_match.group(2).replace(',', ''))
            if amount_min > amount_max:
                issues.append('reversed_amount_range')
        elif len(money) > 1:
            issues.append('multiple_amounts')
        elif len(money) == 1:
            amount = Decimal(money[0].replace(',', ''))
        amount_kind = ('loan_amount' if mortgage or auto or 'loan' in text.split() else 'balance') if (
            amount is not None or amount_min is not None
        ) else None
        if amount is None and amount_min is None and re.search(r'\b(?:balance|amount)\b', re.sub(r'\bbalance transfer\b', '', text)):
            issues.append('amount_not_parsed')
        if (amount is not None or amount_min is not None) and re.search(r'\b(?:down payment|income|closing costs?)\b', text):
            issues.append('monetary_role_not_loan_or_deposit_amount')
        if (amount is not None or amount_min is not None) and not (mortgage or auto or re.search(r'\b(?:balance|amount|loan|deposit|tier|checking|certificate|savings|cd)\b', text)):
            issues.append('amount_scope_not_supported')
        if re.search(r'\b(?:over|under|above|below|at least|more than|less than)\b', text):
            issues.append('comparison_not_supported')
        if re.search(r'\b(?:not|except|excluding)\b', text):
            issues.append('negated_constraint_not_supported')
        if re.search(r'\bchanged?\b', text) or (
            re.search(r'\bbetween\b', text) and amount_min is None
        ):
            issues.append('comparison_or_history_query')
        if mortgage:
            points = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:discount\s+)?points?\b', question, re.I)
            if re.search(r'\b(?:no|zero|without)\s+(?:discount\s+)?points?\b', text): points.append('0')
            if len(set(points)) > 1:
                issues.append('ambiguous_mortgage_points')
            elif points: condition('mortgage_points', format(Decimal(points[0]).normalize(), 'f'))
            elif re.search(r'\bpoints?\b', text): issues.append('mortgage_points_not_parsed')
        if card and term is not None:
            introductory_duration = re.search(
                r'\b(?:first|introductory|intro|promotional)\b.{0,35}\b\d+\s+(?:month|year)|'
                r'\b\d+\s+(?:month|year)\s+(?:introductory|intro|promotional)\b', text)
            if introductory_duration:
                condition('intro_months', str(term))
                if re.search(r'\bafter (?:the )?first\s+\d+\s+(?:month|year)', text):
                    condition('phase', 'post_promotion')
                elif re.search(r'\bfirst\s+\d+\s+(?:month|year)', text):
                    condition('phase', 'introductory')
                term = None
            else:
                issues.append('credit_card_duration_scope_unknown')
        if mortgage and term is not None and re.search(r'\b(?:first|initial)\s+\d+\s+(?:month|year)', text):
            condition('initial_rate_months', str(term))
            term = None
        return QueryScope(
            product_ids=tuple(sorted(product_ids)), attributes=tuple(sorted(attributes)),
            segment=('business' if re.search(r'\bbusiness\b', text) else
                     'personal' if re.search(r'\bpersonal\b', text) else None),
            family=family, term_months=term, model_year=year, amount=amount,
            amount_kind=amount_kind, amount_min=amount_min, amount_max=amount_max,
            requested_value=requested_value, requested_unit=requested_unit,
            result_shape=result_shape,
            conditions=tuple(sorted(conditions.items())),
            unsupported=tuple(dict.fromkeys(issues)), as_of=as_of,
        )
