"""Explicit non-product identities supplied by verified source processing.

This module does not invent a site-wide registry or infer aliases from names.
Branch/ATM IDs should come from verified location identity (e.g. full address),
not a name-only hash. Definitions/aliases are trusted inputs, not LLM proposals.
"""
from dataclasses import dataclass
import re
from typing import Literal
import unicodedata

EntityKind = Literal['product', 'branch', 'atm', 'organization', 'program', 'service', 'geographic_area']
ENTITY_KINDS = frozenset({'product', 'branch', 'atm', 'organization', 'program', 'service', 'geographic_area'})


def entity_label(value: str) -> str:
    """Typography-only comparison. Do not strip accents or infer synonyms."""
    value = unicodedata.normalize('NFKC', value).casefold()
    return ' '.join(re.findall(r'[^\W_]+', value))


@dataclass(frozen=True)
class EntityDefinition:
    id: str
    name: str
    kind: EntityKind
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.id.strip() or self.id != self.id.strip():
            raise ValueError('Entity ID must be nonempty without surrounding whitespace')
        if not entity_label(self.name):
            raise ValueError('Entity name must contain letters or numbers')
        if self.kind not in ENTITY_KINDS - {'product'}:
            raise ValueError('Use existing Product definitions for product identities')
        if not isinstance(self.aliases, tuple) or any(not entity_label(a) for a in self.aliases):
            raise ValueError('Entity aliases must be a tuple of nonempty labels')


# Verified in baseline source newsroom-dupaco-ranked-top-workplace-usa-5f2414.md:
# "Dupaco Community Credit Union ..." and "Dupaco has ..." refer to this issuer.
# Volunteer Week articles (2024/2025), *giving-b-62f656.md and
# *giving-b-bbbfc4.md, also name this issuer "Dupaco Credit Union".
# All entries below are manually source-verified, never registered from LLM output.
DUPACO_ENTITIES = (
    EntityDefinition('organization:dupaco', 'Dupaco Community Credit Union',
                     'organization', ('Dupaco', 'Dupaco Credit Union')),
    # community headings and non-profit-grant-8d385b.md describe the same program.
    EntityDefinition('program:dupaco_foundation_nonprofit_grant',
                     'Dupaco Foundation Nonprofit Grant', 'program',
                     ('Nonprofit Grant Program', 'Dupaco Foundation Nonprofit Grant program')),
    # community heading plus hardship-grant-73e414.md, not the credit union itself.
    EntityDefinition('program:dupaco_foundation_hardship_grant',
                     'Dupaco Foundation Hardship Grant', 'program',
                     ('Hardship Grant', 'Hardship Grant Program')),
    # newsroom-first-time-homebuyers-get-a-boost-dupaco-offers-fhlb-hometart-grants-ec654a.md.
    EntityDefinition('program:hometart', 'Home$tart', 'program',
                     ('Home$tart program', 'FHLB Home$tart grants')),
    # The grant eligibility page explicitly calls this the credit union's
    # geographic charter area; the membership page refers to it as "our charter area".
    EntityDefinition('geographic_area:dupaco_charter',
                     'Dupaco Credit Union Geographic Charter Area', 'geographic_area',
                     ("credit union's geographic charter area", 'geographic charter area',
                      'charter area')),
)

