"""The representable subset of semantic candidates, not the whole vocabulary."""
from datetime import date
import re
from urllib.parse import urlsplit

from facts.qualifier_normalizer import SEMANTIC_REQUIRED_QUALIFIERS

_ALL_KINDS = ('organization', 'product', 'branch', 'atm', 'program', 'service')
COUNTS = frozenset({'branch_count', 'employee_count', 'vto_user_count',
                    'recognition_count', 'ranking_position'})
DATES = frozenset({'opening_date', 'closure_date'})
FEES = frozenset({'origination_fee', 'prepayment_fee'})
URLS = frozenset({'application_url', 'rates_url', 'locator_url'})

# These profiles require scalar values and a known entity type. Open-ended prose
# cannot enter this path merely by choosing a broad canonical attribute.
SEMANTIC_FACT_PROFILES = {
    **{a: {'value_kind': 'integer', 'units': [None], 'entity_kinds': ['organization']}
       for a in COUNTS},
    **{a: {'value_kind': 'iso_date', 'units': [None], 'entity_kinds': ['branch', 'atm']}
       for a in DATES},
    **{a: {'value_kind': 'nonnegative_decimal', 'units': ['USD', 'percent', None],
           'entity_kinds': ['product']} for a in FEES},
    'maximum_assistance_amount': {'value_kind': 'nonnegative_decimal', 'units': ['USD'],
                                  'entity_kinds': ['program', 'product']},
    'volunteer_hours': {'value_kind': 'nonnegative_decimal', 'units': ['hours'],
                        'entity_kinds': ['organization']},
    'supported_payment_wallet': {'value_kind': 'support_status', 'units': [None],
                                 'entity_kinds': ['organization','product']},
    'default_payment_option': {'value_kind': 'source_text', 'units': [None],
                               'entity_kinds': ['organization','product']},
    'eligibility_requirement': {'value_kind': 'rule_expression', 'units': [None],
                                'entity_kinds': ['organization','program','product']},
    'geographic_scope': {'value_kind': 'source_text', 'units': [None],
                         'entity_kinds': ['geographic_area']},
    'routing_number': {'value_kind': 'routing_number', 'units': [None],
                       'entity_kinds': ['organization']},
    'phone_number': {'value_kind': 'phone_number', 'units': [None], 'entity_kinds': list(_ALL_KINDS)},
    'email_address': {'value_kind': 'email_address', 'units': [None], 'entity_kinds': list(_ALL_KINDS)},
    **{a: {'value_kind': 'url', 'units': [None], 'entity_kinds': list(_ALL_KINDS)} for a in URLS},
}

SEMANTIC_FACT_PROFILES['vto_user_count']['meaning'] = 'People explicitly stated to have used VTO; never all volunteers.'
SEMANTIC_FACT_PROFILES['employee_count']['meaning'] = 'Total workforce, not training or volunteer participants.'
SEMANTIC_FACT_PROFILES['eligibility_requirement']['allowed_qualifiers'] = [
    'statement_topic', 'rule_scope', 'rule_clause',
]
SEMANTIC_FACT_PROFILES['eligibility_requirement']['identity_only_qualifiers'] = [
    'statement_topic', 'rule_scope', 'rule_clause',
]
SEMANTIC_FACT_PROFILES['supported_payment_wallet']['allowed_qualifiers'] = ['statement_topic','wallet_name','card_type','eligibility']
SEMANTIC_FACT_PROFILES['default_payment_option']['allowed_qualifiers'] = ['statement_topic','card_type']
SEMANTIC_FACT_PROFILES['eligibility_requirement']['meaning'] = 'A source-quoted rule statement; not a claim of complete membership policy coverage.'
SEMANTIC_FACT_PROFILES['geographic_scope']['meaning'] = (
    'An explicitly stated geographic area, kept separate from any eligibility rule that references it.'
)
SEMANTIC_FACT_PROFILES['branch_count']['meaning'] = 'Branches, not a combined count of branches and ATMs.'
for _attribute, _profile in SEMANTIC_FACT_PROFILES.items():
    _profile['required_qualifiers'] = list(SEMANTIC_REQUIRED_QUALIFIERS.get(_attribute, ()))
EXTRACTABLE_SEMANTIC_ATTRIBUTES = frozenset(SEMANTIC_FACT_PROFILES)
DETERMINISTIC_FIELD_ATTRIBUTES = frozenset({
    'routing_number', 'phone_number', 'email_address', *URLS,
})
LLM_EXTRACTABLE_SEMANTIC_ATTRIBUTES = EXTRACTABLE_SEMANTIC_ATTRIBUTES - DETERMINISTIC_FIELD_ATTRIBUTES


def semantic_value_issue(attribute, value, unit):
    profile = SEMANTIC_FACT_PROFILES.get(attribute)
    if profile is None:
        return 'unrepresentable_semantic_attribute'
    if unit not in profile['units']:
        return 'unsupported_semantic_unit'
    kind = profile['value_kind']
    valid = False
    if kind == 'integer':
        valid = bool(re.fullmatch(r'0|[1-9][0-9]*', value))
        if attribute == 'ranking_position' and value == '0':
            valid = False
    elif kind == 'nonnegative_decimal':
        valid = bool(re.fullmatch(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?', value))
        if attribute in FEES and unit is None and value != '0':
            valid = False  # Only an explicitly absent fee is unit-independent.
    elif kind == 'iso_date':
        try:
            valid = date.fromisoformat(value).isoformat() == value
        except ValueError:
            pass
    elif kind == 'routing_number':
        valid = bool(re.fullmatch(r'[0-9]{9}', value))
    elif kind == 'phone_number':
        valid = bool(re.fullmatch(r'\+?[0-9() .-]+(?:\s*(?:x|ext\.?)\s*[0-9]+)?', value, re.I)) and sum(c.isdigit() for c in value) >= 7
    elif kind == 'email_address':
        valid = bool(re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value))
    elif kind == 'support_status':
        valid = value in {'supported', 'not_supported', 'conditional'}
    elif kind == 'rule_expression':
        from ingestion.membership_rules import rule_issue
        valid = rule_issue(value, allow_discontiguous=True) is None
    elif kind == 'url':
        try:
            url = urlsplit(value)
            valid = url.scheme in {'http', 'https'} and bool(url.netloc) and not re.search(r'\s', value)
        except ValueError:
            pass
    elif kind == 'source_text':
        valid = bool(value.strip())
    return None if valid else 'non_atomic_semantic_value'


def geographic_scope_benefit(text):
    """Route only an explicit named geographic-area enumeration to extraction."""
    plain = re.sub(r'[*_]', '', text)
    if not re.search(r'\bgeographic\s+(?:charter|service)\s+area\b', plain, re.I):
        return None
    labels = re.findall(
        r"^\s*(?:[-+*]\s*)?([A-Z][A-Za-z.'’ -]{1,40}):\s*\S",
        plain,
        re.M,
    )
    return 'explicit_geographic_scope' if len(set(labels)) >= 2 else None


def semantic_benefit(text, context=''):
    """Positive use cases, never an ambiguity/parse-failure fallback.

    Mere numbers, product keywords, contact formats or unknown qualifiers are
    insufficient. The value/conditions must still fit a scalar Fact profile.
    """
    if re.search(r'\b(?:overall|industry|category)\b', text, re.I) and re.search(
        r'(?:#\s*\d+|rank(?:ed|s)?\s+(?:no\.?\s*)?\d+)', text, re.I):
        return 'scoped_ranking'
    if re.search(r'\b(?:recogniz\w*|forbes)\b', text, re.I) and re.search(
        r'\b(?:\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+time', text, re.I):
        return 'recognition_count_attribution'
    plain = re.sub(r'[*_]', '', text)
    wallet_context = plain + ' ' + context
    wallet_subject = re.search(
        r'\b(?:debit|credit|payment)(?:\s+(?:and|or)\s+(?:debit|credit))?\s+cards?\b',
        wallet_context, re.I,
    )
    wallet_topic = re.search(r'\b(?:mobile|digital) wallets?\b', wallet_context, re.I)
    wallet_relation = re.search(
        r'\b(?:compatible|support(?:s|ed)?|add|use|work)\b', plain, re.I,
    )
    if (wallet_subject and wallet_topic and wallet_relation
            and not re.search(r'\b(?:lost|stolen|breach|deactivate|safe|security)\b',
                              plain, re.I)):
        return 'wallet_compatibility_statement'
    if (re.search(r'\b(?:volunteer time off|vto)\b', plain, re.I)
            and re.search(r'\b(?:used|volunteered|contributed|logged|participated|utilized)\b', plain, re.I)
            and re.search(r'\b\d[\d,]*\s+(?:Dupaco\s+)?(?:employees|hours)\b', plain, re.I)):
        return 'volunteer_metric_attribution'
    full_date = re.search(
        r'\b(?:20\d{2}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/20\d{2}|'
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
        r'\s+\d{1,2},?\s+20\d{2})\b', plain, re.I)
    if (re.search(r'\b(?:branch|atm|location)\b', plain + ' ' + context, re.I)
            and re.search(r'\b(?:clos(?:e|es|ed|ing|ure)|open(?:ed|ing))\b', plain, re.I)
            and full_date
            and re.search(r'\b(?:will|scheduled|plans?|planned|effective|permanently|closed|opened)\b', plain, re.I)):
        return 'location_event_date_association'
    if (re.search(r'\b(?:grant|home\$tart|down.payment assistance)\b', plain + ' ' + context, re.I)
            and re.search(r'\b(?:up to|maximum(?: of)?|capped at)\s+\$\s*\d', plain, re.I)
            and re.search(r'\b(?:eligible|qualif\w*|first.time|whose|who|hardship|depending|if|only)\b', plain, re.I)
            and not re.search(r'\b(?:valued|reimbursement|insurance)\b', plain, re.I)):
        return 'assistance_cap_and_conditions'
    if (re.search(r'\b(?:origination|prepayment)\b', text, re.I)
            and re.search(r'\b(?:if|when|unless|only|except|depending)\b', text, re.I)
            and re.search(r'\b(?:no|zero)\b|\$\s*\d|\d\s*%', text, re.I)):
        return 'conditional_product_fee'
    if (re.search(r'\bvolunteer\w*\b', plain, re.I)
            and re.search(r'\b\d[\d,]*(?:\.\d+)?\s+hours\b', plain, re.I)):
        return 'volunteer_hours_attribution'
    if (re.search(r'\b(?:origination|prepayment)\s+(?:fees?|penalt(?:y|ies))\b', plain, re.I)
            and re.search(r'\b(?:no|zero)\b|\$\s*\d|\d\s*%', plain, re.I)):
        return 'explicit_product_fee'
    return None
