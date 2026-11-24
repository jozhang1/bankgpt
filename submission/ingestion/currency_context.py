"""Auditable currency proof: explicit notation or a narrowly scoped issuer policy.

The policy is trusted ingestion configuration, not an LLM assertion and not a
claim that a bare dollar sign literally says USD. Unknown issuers/programs retain
independent model review. Currency never follows merely from a website suffix.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from urllib.parse import urlsplit
from ingestion.semantic_context import bound_context_quotes
from foundation.entities import DUPACO_ENTITIES
from ingestion.subject_scope import contains_label


@dataclass(frozen=True)
class CurrencyProof:
    status: str  # verified / conflicting / unresolved
    basis: str
    quote: str | None = None
    policy_id: str | None = None


# Dataset-specific domestic grant policy. Scope is canonical program + predicate
# + official source host, never every dollar-denominated claim on dupaco.com.
# Nonprofit program's eligibility lists Iowa/Illinois/Wisconsin in
# non-profit-grant-8d385b.md; Hardship Grant is the issuer's domestic assistance
# program described in hardship-grant-73e414.md and community-c481b3.md.
# These are explicit configured interpretation policies, not literal USD quotes.
PROGRAM_USD_POLICIES = {
    'program:dupaco_foundation_nonprofit_grant': 'dupaco-domestic-nonprofit-grant-usd-v1',
    'program:dupaco_foundation_hardship_grant': 'dupaco-domestic-hardship-grant-usd-v1',
}
_OFFICIAL_HOSTS = frozenset({'www.dupaco.com', 'dupaco.com'})
_NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
_MONEY = re.compile(r'(?<![\w$])(?P<prefix>US\s*\$|USD\s*\$?|\$)\s*(?P<amount>' + _NUMBER + r')'
                    r'(?!\d|,\d|\.\d)', re.I)
_FOREIGN = re.compile(r'\b(?:CAD|AUD|NZD|SGD|HKD|Canadian dollars?|Australian dollars?|'
                      r'New Zealand dollars?|Singapore dollars?|Hong Kong dollars?)\b|'
                      r'(?<![A-Za-z])(?:CA|C|AU|A|NZ|SG|HK)\s*\$', re.I)


def currency_proof(candidate, evidence):
    if candidate.unit != 'USD':
        return CurrencyProof('unresolved', 'not_usd')
    if candidate.evidence_id != evidence.id or not candidate.quote or candidate.quote not in evidence.text:
        return CurrencyProof('unresolved', 'unbound_amount_quote')
    # Conflicting explicit currency is never silently overridden by a policy.
    if _FOREIGN.search(candidate.quote) or any(_FOREIGN.search(q) for q in bound_context_quotes(candidate, evidence)):
        return CurrencyProof('conflicting', 'explicit_other_currency', candidate.quote)
    try:
        value = Decimal(candidate.value)
    except InvalidOperation:
        return CurrencyProof('unresolved', 'invalid_amount')
    text = re.sub(r'[*_]', '', candidate.quote)
    matches = [m for m in _MONEY.finditer(text)
               if Decimal(m['amount'].replace(',', '')) == value]
    if not matches:
        return CurrencyProof('unresolved', 'no_matching_currency_amount')
    if any(m['prefix'].upper().replace(' ', '').startswith(('USD', 'US$')) for m in matches):
        return CurrencyProof('verified', 'explicit_usd_notation', candidate.quote)
    if re.search(r'\b(?:U\.?S\.?|United States) dollars?\b', text, re.I):
        return CurrencyProof('verified', 'explicit_us_dollar_language', candidate.quote)
    source = urlsplit(evidence.source_url)
    policy = PROGRAM_USD_POLICIES.get(candidate.entity)
    if (policy and candidate.attribute == 'maximum_assistance_amount'
            and source.scheme == 'https' and source.hostname in _OFFICIAL_HOSTS
            and source.username is None and source.password is None):
        return CurrencyProof('verified', 'configured_program_currency', candidate.quote, policy)
    return CurrencyProof('unresolved', 'currency_context_not_established')


def extraction_currency_policies(evidence):
    """Only expose policies whose named program appears on its official source."""
    source = urlsplit(evidence.source_url)
    if (source.scheme != 'https' or source.hostname not in _OFFICIAL_HOSTS
            or source.username is not None or source.password is not None):
        return ()
    result = []
    for record in DUPACO_ENTITIES:
        policy = PROGRAM_USD_POLICIES.get(record.id)
        labels = (record.name, *record.aliases)
        if policy and any(contains_label(evidence.text, label) for label in labels):
            result.append({'policy_id': policy, 'program_labels': labels,
                           'attribute': 'maximum_assistance_amount', 'currency': 'USD',
                           'requires_literal_matching_dollar_amount': True})
    return tuple(result)
