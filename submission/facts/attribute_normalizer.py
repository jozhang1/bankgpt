"""Exact attribute vocabulary; identity, value, conditions and time stay separate.

A recognized label is not proof that Evidence entails it. Semantic candidates
still require source validation and later value/qualifier/builder support.
"""
from __future__ import annotations

import re
from types import MappingProxyType
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.extractor import CandidateFact

RATE_ATTRIBUTES = frozenset({'apy', 'apr', 'dividend_rate', 'interest_rate'})
AMOUNT_ATTRIBUTES = frozenset({'minimum_opening_balance', 'minimum_earning_balance'})
UNRESOLVED_ATTRIBUTE = 'rate'

# First semantic vocabulary, shared across sources rather than question IDs.
# Generic multi-valued attributes require identity qualifiers in a later step.
SEMANTIC_ATTRIBUTES = frozenset({
    'closure_date', 'opening_date',
    'branch_count', 'employee_count', 'vto_user_count', 'volunteer_hours',
    'recognition_count', 'ranking_position', 'award_name',
    'product_description', 'product_feature', 'repayment_option',
    'eligibility_requirement', 'application_step', 'alternative_service_location',
    'geographic_scope',
    'program_description', 'program_provider', 'maximum_assistance_amount',
    'supported_payment_wallet', 'default_payment_option',
    'origination_fee', 'prepayment_fee',
    'routing_number', 'phone_number', 'email_address',
    'application_url', 'rates_url', 'locator_url',
})
CANONICAL_ATTRIBUTES = RATE_ATTRIBUTES | AMOUNT_ATTRIBUTES | SEMANTIC_ATTRIBUTES


def attribute_key(value: str) -> str:
    """Case/spacing/punctuation only; never guess from a substring."""
    return '_'.join(re.findall(r'[a-z0-9]+', unicodedata.normalize('NFKC', value).casefold()))


# Only equivalent labels. Do not collapse scheduled vs actual closure,
# volunteer counts vs VTO users, or generic counts/fees/URLs into specific fields.
_EXTRA_ALIASES = {
    'apy': ('Annual Percentage Yield', 'Annual Percentage Yield (APY)'),
    'apr': ('Annual Percentage Rate', 'Annual Percentage Rate (APR)'),
    'dividend_rate': ('Dividend Rate for 1st 6 Months', 'Dividend Rate After Promotional Period'),
    'minimum_opening_balance': ('Minimum to Open',),
    'minimum_earning_balance': ('Minimum to Earn',),
    'branch_count': ('Number of Branches', 'Branch Office Count'),
    'employee_count': ('Number of Employees', 'Total Employee Count'),
    'vto_user_count': ('Volunteer Time Off User Count', 'Number of VTO Users'),
    'volunteer_hours': ('Volunteer Service Hours', 'Total Volunteer Hours'),
    'recognition_count': ('Times Recognized', 'Number of Times Recognized'),
    'ranking_position': ('Rank Position', 'Ranking Position'),
    'award_name': ('Award Title',),
    'alternative_service_location': ('Alternative Banking Location',),
    'maximum_assistance_amount': ('Maximum Assistance',),
    'supported_payment_wallet': ('Supported Mobile Wallet',),
    'default_payment_option': ('Default Card Selection',),
    'phone_number': ('Contact Phone Number',),
    'email_address': ('Contact Email Address',),
    'application_url': ('Application Link',),
    'rates_url': ('Rates Link',),
    'locator_url': ('Locator Link',),
}


def _alias_map():
    result = {}
    for canonical in sorted(CANONICAL_ATTRIBUTES):
        for alias in (canonical, *_EXTRA_ALIASES.get(canonical, ())):
            key = attribute_key(alias)
            if key in result and result[key] != canonical:
                raise ValueError(f'Conflicting canonical attribute alias: {alias}')
            result[key] = canonical
    return MappingProxyType(result)


ATTRIBUTE_ALIASES = _alias_map()


class AttributeNormalizer:
    def normalize(self, candidate: CandidateFact) -> str | None:
        """Return a known canonical name, unresolved 'rate', or None.

        Unknown semantic labels are not turned into a known predicate. The
        caller retains the raw label and records unresolved_attribute.
        """
        label = attribute_key(candidate.attribute)
        if label in {'rate', 'rates'}:
            if label == 'rate' and self._explicit_mortgage_interest_rate(candidate):
                return 'interest_rate'
            return UNRESOLVED_ATTRIBUTE
        return ATTRIBUTE_ALIASES.get(label)

    @staticmethod
    def is_rate_context(candidate: CandidateFact) -> bool:
        """Legacy rate diagnostics only; never establishes APR/APY meaning."""
        if attribute_key(candidate.attribute) in {'rate', 'rates'}:
            return True
        if any(signal.kind == 'subject' for signal in candidate.identity_signals):
            return False
        return bool(re.search(r'\b(?:apr|apy|rate|rates|dividend|yield)\b',
                              candidate.table_header, re.I))

    @staticmethod
    def _explicit_mortgage_interest_rate(candidate: CandidateFact) -> bool:
        # Preserve the original source-verified schema rule, not a product-name guess.
        headers = tuple(attribute_key(h) for h in candidate.table_header.strip().strip('|').split('|'))
        known_schema = (len(headers) == 5 and headers[0] in {'term', 'program'}
                        and headers[1:] == ('rate', 'points', 'apr', 'payment_example'))
        program = next((value.strip() for key, value in candidate.qualifiers.items()
                        if attribute_key(key) in {'column_1_term', 'column_1_program'}
                        and isinstance(value, str)), '')
        loan_program = re.match(
            r'^(?:purchase\s*:|refinance\s*:|mortgage rate relief program\b)', program, re.I)
        rate_column = next(
            (value for key, value in candidate.qualifiers.items()
             if attribute_key(key) == 'rate_column' and isinstance(value, str)),
            '',
        )
        # Equivalent spellings must not hide conflicting column labels.
        if any(attribute_key(value) != attribute_key(rate_column)
               for key, value in candidate.qualifiers.items()
               if attribute_key(key) == 'rate_column' and isinstance(value, str)):
            return False
        return bool(known_schema and loan_program and isinstance(rate_column, str)
                    and attribute_key(rate_column) == 'rate')
