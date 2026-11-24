"""Conservative personnel-card recognition over existing source blocks."""
import re
from typing import TYPE_CHECKING

from ingestion.atomic_groups import AtomicGroup

if TYPE_CHECKING:
    from ingestion.chunker import Block


ROLE = re.compile(
    r'\b(?:NMLS|consultant|insurance|administrative|operations manager|'
    r'chief|president|director|secretary|treasurer|vice chair|chair of the board)\b', re.I,
)
APPLY = re.compile(r'\[Apply\]\(https://homeloans\.dupaco\.com/[^\n]+\)', re.I)
PHONE = re.compile(r'\b\d{3}[- .]\d{3}[- .]\d{4}\b')


def _portrait(block: 'Block') -> bool:
    return block.kind == 'paragraph' and block.text.startswith('Image description:')


def _header_end(blocks: list['Block'], start: int, limit: int) -> int | None:
    i = start + int(_portrait(blocks[start]))
    if i >= limit:
        return None
    name = blocks[i]
    # The verified directory layouts use H4 names, including linked name/title.
    if name.kind != 'heading' or name.heading_level != 4:
        return None
    if 'mailto:' in name.text and ROLE.search(name.text):
        return i + 1
    if i + 1 < limit:
        role = blocks[i + 1]
        if role.kind == 'heading' and role.heading_level == 6 and ROLE.search(role.text):
            return i + 2
    return None


class PersonCardBoundaryDetector:
    @staticmethod
    def detect_start(blocks: list['Block'], start: int) -> bool:
        if not 0 <= start < len(blocks):
            return False
        if not _portrait(blocks[start]) and start and _portrait(blocks[start - 1]):
            return False
        return _header_end(blocks, start, len(blocks)) is not None

    @classmethod
    def detect_next_start(cls, blocks: list['Block'], start: int) -> int:
        region = blocks[start].in_recommendations
        header_end = _header_end(blocks, start, len(blocks))
        for i in range(start + 1, len(blocks)):
            b = blocks[i]
            if b.in_recommendations != region:
                return i
            if i < (header_end or start):
                continue
            if cls.detect_start(blocks, i):
                return i
            if (b.kind in {'horizontal_rule', 'table', 'code'} or _portrait(b)
                    or (b.kind == 'heading' and (b.heading_level or 0) <= 5)):
                return i
        return len(blocks)


def recognize_application(blocks: list['Block'], start: int, limit: int) -> AtomicGroup | None:
    body = _header_end(blocks, start, limit)
    if body is None or not any('NMLS' in b.text for b in blocks[start:body]):
        return None
    for i in range(body, limit):
        if blocks[i].kind != 'paragraph':
            return None
        if APPLY.fullmatch(blocks[i].text.strip()):
            return AtomicGroup('person_card', 'application', start, i + 1)
    return None


def recognize_contact(blocks: list['Block'], start: int, limit: int) -> AtomicGroup | None:
    body = _header_end(blocks, start, limit)
    if body is None:
        return None
    email_end = None
    phone = False
    for i in range(body, limit):
        b = blocks[i]
        if b.kind not in {'paragraph', 'heading'} or (b.kind == 'heading' and b.heading_level != 6):
            break
        phone |= bool(PHONE.search(b.text))
        if 'mailto:' in b.text:
            email_end = i + 1
        elif email_end is not None and not PHONE.search(b.text):
            # Unknown prose after contacts belongs to ordinary grouping.
            break
        if email_end is not None:
            email_end = i + 1
    return AtomicGroup('person_card', 'contact', start, email_end) if email_end is not None and phone else None


def recognize_biography(blocks: list['Block'], start: int, limit: int) -> AtomicGroup | None:
    body = _header_end(blocks, start, limit)
    if body is None or body >= limit:
        return None
    # Only verified leadership roles support open-ended biography/affiliation text.
    role = blocks[body - 1]
    if role.kind != 'heading' or not re.search(
        r'\b(?:chief|president|director|secretary|treasurer|vice chair|chair of the board)\b', role.text, re.I,
    ):
        return None
    if not all(b.kind == 'paragraph' and 'mailto:' not in b.text for b in blocks[body:limit]):
        return None
    return AtomicGroup('person_card', 'biography', start, limit)


def recognize_person_card(blocks: list['Block'], start: int, limit: int) -> AtomicGroup | None:
    if not 0 <= start < limit <= len(blocks):
        raise ValueError('Invalid candidate window')
    if not PersonCardBoundaryDetector.detect_start(blocks, start):
        return None
    if any(b.in_recommendations != blocks[start].in_recommendations for b in blocks[start:limit]):
        return None
    for recognizer in (recognize_application, recognize_contact, recognize_biography):
        match = recognizer(blocks, start, limit)
        if match is not None:
            if match.start != start or not start < match.end <= limit:
                raise ValueError('Invalid person card interval')
            return match
    return None


def consume_person_card(blocks: list['Block'], start: int) -> int | None:
    if not PersonCardBoundaryDetector.detect_start(blocks, start):
        return None
    limit = PersonCardBoundaryDetector.detect_next_start(blocks, start)
    match = recognize_person_card(blocks, start, limit)
    return match.end if match else None
