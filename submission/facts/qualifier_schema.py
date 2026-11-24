"""Declarative query-condition vocabulary and three-state Fact matching."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping

from facts.fact_matching import MatchStatus


@dataclass(frozen=True)
class QualifierSpec:
    key: str
    aliases: tuple[str, ...] = ()
    comparator: str = 'exact'  # exact / numeric / interval / year_interval
    lower_key: str | None = None
    upper_key: str | None = None
    query_patterns: tuple[tuple[str, str | bool], ...] = ()
    contexts: frozenset[str] = frozenset()


# Query wording is data in this registry. New qualifier support should add a
# declaration here, rather than another branch in FactSelector.
QUALIFIER_SCHEMA = (
    QualifierSpec('term_months', ('term', 'loan term', 'duration'), 'numeric'),
    QualifierSpec('model_year', ('vehicle model year',), 'year_interval',
                  'model_year_from', 'model_year_to'),
    QualifierSpec('loan_amount', ('principal', 'loan principal'), 'interval',
                  'loan_amount_min', 'loan_amount_max'),
    QualifierSpec('balance', ('account balance', 'deposit balance'), 'interval',
                  'balance_min', 'balance_max'),
    QualifierSpec('mortgage_points', ('discount points', 'points'), 'numeric'),
    QualifierSpec('intro_months', ('promotional duration',), 'numeric'),
    QualifierSpec('initial_rate_months', ('initial rate duration',), 'numeric'),
    QualifierSpec('jumbo', query_patterns=((r'\bnon[ -]?jumbo\b', 'false'),
                                           (r'\bjumbo\b', 'true'))),
    QualifierSpec('loan_purpose', query_patterns=(
        (r'\bcash[ -]?out\s+(?:re)?financ(?:e|ing)\b', 'cash_out'),
        (r'\b(?:purchase|buying|buy|buy a home)\b', 'purchase'),
        (r'\b(?:refinance|refinancing|refi)\b', 'refinance'),
    ), contexts=frozenset({'mortgage'})),
    QualifierSpec('rate_type', query_patterns=((r'\bfixed\b', 'fixed'),
                                               (r'\b(?:adjustable|arm|variable)\b', 'variable')),
                  contexts=frozenset({'mortgage'})),
    QualifierSpec('loan_program', query_patterns=((r'\bconventional\b', 'conventional'),
                                                  (r'\bfha\b', 'fha'),
                                                  (r'\bva\b', 'va'),
                                                  (r'\bmortgage rate relief\b', 'mortgage_rate_relief')),
                  contexts=frozenset({'mortgage'})),
    QualifierSpec('borrower_eligibility', query_patterns=((r'\bfirst time (?:buyer|homebuyer)\b',
                                                            'first_time_buyer'),),
                  contexts=frozenset({'mortgage'})),
    QualifierSpec('property_use', query_patterns=((r'\binvestment property\b', 'investment'),
                                                  (r'\bprimary residence\b', 'primary_residence'),
                                                  (r'\bsecond home\b', 'second_home')),
                  contexts=frozenset({'mortgage'})),
    QualifierSpec('phase', query_patterns=((r'\b(?:introductory|intro|promotional)\b', 'introductory'),
                                           (r'\bstandard\b', 'standard'),
                                           (r'\bafter (?:the )?(?:introductory|promotional) period\b',
                                            'post_promotion')),
                  contexts=frozenset({'credit_card'})),
    QualifierSpec('transaction_type', query_patterns=((r'\bbalance transfers?\b', 'balance_transfer'),
                                                       (r'\bcash advances?\b', 'cash_advance'),
                                                       (r'\bpurchases?\b', 'purchase')),
                  contexts=frozenset({'credit_card'})),
    QualifierSpec('ira_type', query_patterns=((r'\broth\b', 'roth'),
                                              (r'\btraditional\b', 'traditional'))),
    QualifierSpec('vehicle_condition', query_patterns=((r'\bnew\b', 'new'),
                                                       (r'\bused\b', 'used')),
                  contexts=frozenset({'auto'})),
    QualifierSpec('arm_notation', query_patterns=((r'\b(\d{1,2})/(\d{1,2})\s*(?:arm|adjustable)',
                                                   '__capture__'),),
                  contexts=frozenset({'mortgage'})),
)

_SPECS = {spec.key: spec for spec in QUALIFIER_SCHEMA}


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value).replace(',', '').replace('$', '').strip())
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def parse_declared_conditions(question: str, contexts: set[str]):
    """Parse declared categorical conditions; return values and ambiguity flags."""
    text = re.sub(r'[^a-z0-9/]+', ' ', question.casefold())
    values: dict[str, set[str | bool]] = {}
    issues = []
    for spec in QUALIFIER_SCHEMA:
        if spec.contexts and not (spec.contexts & contexts):
            continue
        for pattern, value in spec.query_patterns:
            match = re.search(pattern, text, re.I)
            if not match:
                continue
            if value == '__capture__':
                parsed = f'{match.group(1)}/{match.group(2)}'
                values.setdefault(spec.key, set()).add(parsed)
            elif spec.key == 'jumbo':
                # Avoid also reading the embedded positive token in “non-jumbo”.
                if value == 'true' and re.search(r'\bnon[ -]?jumbo\b', text):
                    continue
                values.setdefault(spec.key, set()).add(value == 'true')
            else:
                values.setdefault(spec.key, set()).add(value)

    # A cash-out refinance is one purpose, not two conflicting phrases.
    if 'cash_out' in values.get('loan_purpose', set()):
        values['loan_purpose'] = {'cash_out'}
    # “After the introductory period” identifies the post-promotion phase.
    if 'post_promotion' in values.get('phase', set()):
        values['phase'] = {'post_promotion'}

    conditions = {}
    for key, found in values.items():
        if len(found) == 1:
            conditions[key] = next(iter(found))
        elif len(found) > 1:
            issues.append('conflicting_' + key)
    return conditions, issues


def compare_condition(qualifiers: Mapping[str, Any], key: str, expected: Any) -> MatchStatus:
    """Compare a requested canonical qualifier with a Fact using three states."""
    spec = _SPECS.get(key)
    if spec is None:
        actual = qualifiers.get(key)
        if actual is None:
            return MatchStatus.UNKNOWN
        return MatchStatus.MATCH if str(actual).strip().casefold() == str(expected).strip().casefold() else MatchStatus.MISMATCH

    if spec.comparator == 'numeric':
        actual_number, expected_number = _decimal(qualifiers.get(key)), _decimal(expected)
        if actual_number is None or expected_number is None:
            return MatchStatus.UNKNOWN
        return MatchStatus.MATCH if actual_number == expected_number else MatchStatus.MISMATCH

    if spec.comparator in {'interval', 'year_interval'}:
        value = _decimal(expected)
        low_raw, high_raw = qualifiers.get(spec.lower_key), qualifiers.get(spec.upper_key)
        low, high = _decimal(low_raw), _decimal(high_raw)
        if value is None or (low_raw is not None and low is None) or (high_raw is not None and high is None):
            return MatchStatus.UNKNOWN
        if low is None and high is None:
            return MatchStatus.UNKNOWN
        if low is not None and high is not None and low > high:
            return MatchStatus.UNKNOWN
        if spec.comparator == 'year_interval' and any(
            bound is not None and bound != bound.to_integral_value()
            for bound in (low, high)
        ):
            return MatchStatus.UNKNOWN
        if low is not None:
            inclusive = qualifiers.get(f'{spec.lower_key}_inclusive', True)
            if value < low or (value == low and inclusive is False):
                return MatchStatus.MISMATCH
        if high is not None:
            inclusive = qualifiers.get(f'{spec.upper_key}_inclusive', True)
            if value > high or (value == high and inclusive is False):
                return MatchStatus.MISMATCH
        if spec.comparator == 'year_interval' and (low is None or high is None):
            return MatchStatus.UNKNOWN
        return MatchStatus.MATCH

    actual = qualifiers.get(key)
    if actual is None:
        return MatchStatus.UNKNOWN
    if isinstance(actual, bool) or isinstance(expected, bool):
        return MatchStatus.MATCH if actual is expected else MatchStatus.MISMATCH
    return (MatchStatus.MATCH if str(actual).strip().casefold() == str(expected).strip().casefold()
            else MatchStatus.MISMATCH)


def compare_interval_condition(
    qualifiers: Mapping[str, Any], key: str, lower: Any, upper: Any,
    *, lower_inclusive: bool = True, upper_inclusive: bool = True,
) -> MatchStatus:
    """Compare a requested bounded interval with a fact's declared interval.

    MATCH means the fact's interval covers the whole requested interval;
    MISMATCH means they are disjoint; partial or unbounded overlap is UNKNOWN.
    """
    spec = _SPECS.get(key)
    if spec is None or spec.comparator not in {'interval', 'year_interval'}:
        return MatchStatus.UNKNOWN
    qlow, qhigh = _decimal(lower), _decimal(upper)
    flow_raw, fhigh_raw = qualifiers.get(spec.lower_key), qualifiers.get(spec.upper_key)
    flow = _decimal(flow_raw) if flow_raw is not None else None
    fhigh = _decimal(fhigh_raw) if fhigh_raw is not None else None
    if qlow is None or qhigh is None or qlow > qhigh:
        return MatchStatus.UNKNOWN
    if (flow_raw is not None and flow is None) or (fhigh_raw is not None and fhigh is None):
        return MatchStatus.UNKNOWN
    if flow is None and fhigh is None:
        return MatchStatus.UNKNOWN
    if flow is not None and fhigh is not None and flow > fhigh:
        return MatchStatus.UNKNOWN

    fact_low_inclusive = qualifiers.get(f'{spec.lower_key}_inclusive', True)
    fact_high_inclusive = qualifiers.get(f'{spec.upper_key}_inclusive', True)
    if (fhigh is not None and (fhigh < qlow or
            (fhigh == qlow and (fact_high_inclusive is False or lower_inclusive is False)))):
        return MatchStatus.MISMATCH
    if (flow is not None and (flow > qhigh or
            (flow == qhigh and (fact_low_inclusive is False or upper_inclusive is False)))):
        return MatchStatus.MISMATCH

    covers_low = flow is None or flow < qlow or (
        flow == qlow and (fact_low_inclusive is not False or lower_inclusive is False)
    )
    covers_high = fhigh is None or fhigh > qhigh or (
        fhigh == qhigh and (fact_high_inclusive is not False or upper_inclusive is False)
    )
    return MatchStatus.MATCH if covers_low and covers_high else MatchStatus.UNKNOWN


def compare_fact_qualifiers(left: Mapping[str, Any], right: Mapping[str, Any]):
    """Compare two Facts' conditions using the same schema as query matching.

    MISMATCH means a proven disjoint condition, UNKNOWN means scopes could
    overlap but are not represented equally, and MATCH means all represented
    condition dimensions agree. Unknown qualifier keys use conservative exact
    comparison so newly added fields participate without special-case code.
    """
    unknown_keys = []
    interval_keys = set()

    for spec in QUALIFIER_SCHEMA:
        if spec.comparator not in {'interval', 'year_interval'}:
            continue
        lower, upper = spec.lower_key, spec.upper_key
        interval_keys.update((lower, upper, f'{lower}_inclusive', f'{upper}_inclusive'))
        left_low_raw, left_high_raw = left.get(lower), left.get(upper)
        right_low_raw, right_high_raw = right.get(lower), right.get(upper)
        left_has = left_low_raw is not None or left_high_raw is not None
        right_has = right_low_raw is not None or right_high_raw is not None
        if not left_has and not right_has:
            continue
        if not left_has or not right_has:
            unknown_keys.append(f'overlapping_interval:{spec.key}')
            continue
        bounds = [_decimal(raw) if raw is not None else None for raw in
                  (left_low_raw, left_high_raw, right_low_raw, right_high_raw)]
        if any(raw is not None and value is None for raw, value in zip(
                (left_low_raw, left_high_raw, right_low_raw, right_high_raw), bounds)):
            unknown_keys.append(spec.key)
            continue
        left_low, left_high, right_low, right_high = bounds
        if ((left_low is not None and left_high is not None and left_low > left_high)
                or (right_low is not None and right_high is not None and right_low > right_high)):
            unknown_keys.append(spec.key)
            continue
        separated = False
        if left_high is not None and right_low is not None:
            separated = (left_high < right_low or (left_high == right_low and
                         (left.get(f'{upper}_inclusive', True) is False
                          or right.get(f'{lower}_inclusive', True) is False)))
        if right_high is not None and left_low is not None:
            separated = separated or (right_high < left_low or (right_high == left_low and
                         (right.get(f'{upper}_inclusive', True) is False
                          or left.get(f'{lower}_inclusive', True) is False)))
        if separated:
            return MatchStatus.MISMATCH, (f'disjoint_interval:{spec.key}',)
        same_bounds = all(
            a == b for a, b in ((left_low, right_low), (left_high, right_high),
                                (left.get(f'{lower}_inclusive', True), right.get(f'{lower}_inclusive', True)),
                                (left.get(f'{upper}_inclusive', True), right.get(f'{upper}_inclusive', True)))
        )
        if not same_bounds:
            unknown_keys.append(f'overlapping_interval:{spec.key}')

    all_keys = (set(left) | set(right)) - interval_keys
    for key in sorted(all_keys):
        a, b = left.get(key), right.get(key)
        if a is None and b is None:
            continue
        if a is None or b is None:
            spec = _SPECS.get(key)
            if spec is not None and spec.comparator == 'numeric':
                unknown_keys.append(f'overlapping_condition:{key}')
            else:
                unknown_keys.append(key)
            continue
        spec = _SPECS.get(key)
        if spec is not None and spec.comparator == 'numeric':
            a_number, b_number = _decimal(a), _decimal(b)
            if a_number is None or b_number is None:
                unknown_keys.append(key)
            elif a_number != b_number:
                return MatchStatus.MISMATCH, (f'different_condition:{key}',)
        elif isinstance(a, bool) or isinstance(b, bool):
            if a is not b:
                return MatchStatus.MISMATCH, (f'different_condition:{key}',)
        elif str(a).strip().casefold() != str(b).strip().casefold():
            return MatchStatus.MISMATCH, (f'different_condition:{key}',)

    if unknown_keys:
        reasons = tuple(sorted({
            key if key.startswith(('overlapping_interval:', 'overlapping_condition:'))
            else f'condition_scope_unproven:{key}'
            for key in unknown_keys
        }))
        return MatchStatus.UNKNOWN, reasons
    return MatchStatus.MATCH, ()
