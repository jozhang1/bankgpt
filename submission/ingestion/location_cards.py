"""Recognize verified locator records, preserving source status verbatim."""
import re
from typing import TYPE_CHECKING

from ingestion.atomic_groups import AtomicGroup

if TYPE_CHECKING:
    from ingestion.chunker import Block

_TYPES = {'Dupaco Branch': 'branch', 'Surcharge-Free ATM': 'atm'}
_ADDRESS = re.compile(r'^\d[^\n]*,\s*[^,\n]+,\s*[A-Z]{2},\s*\d{5}(?:-\d{4})?$', re.I)
# Only the status formats verified in the snapshots. These are not schedules.
_STATUS = re.compile(r'(?:Closed|Open · Closes at \d{1,2}(?::\d{2})?(?:AM|PM))')
_SERVICE = r'(?:Drive Up Window|Live Video Teller|24-hour ATM)'
_SERVICES = re.compile(_SERVICE + r'(?: · ' + _SERVICE + r')*')


def recognize_location_card(
    blocks: list['Block'], start: int, limit: int,
) -> AtomicGroup | None:
    """Consume consecutive fields; never scan past unknown content or boundaries."""
    if not 0 <= start < limit <= len(blocks):
        raise ValueError('Invalid candidate window')
    if start + 3 > limit:
        return None
    name, kind, address = blocks[start:start + 3]
    if not (name.kind == 'heading' and name.heading_level == 3 and name.text.strip()
            and kind.kind == 'paragraph' and kind.text.strip() in _TYPES
            and address.kind == 'paragraph' and _ADDRESS.fullmatch(address.text.strip())):
        return None
    region = name.in_recommendations
    if any(b.in_recommendations != region for b in (kind, address)):
        return None
    end = start + 3

    def matches(pattern: re.Pattern) -> bool:
        return (end < limit and blocks[end].kind == 'paragraph'
                and blocks[end].in_recommendations == region
                and bool(pattern.fullmatch(blocks[end].text.strip())))

    if matches(_STATUS):
        end += 1
    if matches(_SERVICES):
        end += 1
    return AtomicGroup('location_card', _TYPES[kind.text.strip()], start, end)


def consume_location_card(blocks: list['Block'], start: int) -> int | None:
    if not 0 <= start < len(blocks):
        return None
    match = recognize_location_card(blocks, start, len(blocks))
    return match.end if match else None
