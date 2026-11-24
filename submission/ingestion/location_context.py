"""Conservative, section-local association of branch announcements and hours."""
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.chunker import Block


@dataclass(frozen=True)
class LocationCandidate:
    quote: str
    source_block: int


@dataclass(frozen=True)
class HoursRegion:
    start: int
    end: int


@dataclass(frozen=True)
class LocationContext:
    quote: str
    source_block: int
    start: int
    end: int  # exclusive; hours range in this section's retrieval blocks


class LocationRecognizer:
    @staticmethod
    def recognize(blocks: list['Block']) -> list[LocationCandidate]:
        return [LocationCandidate(b.text, i) for i, b in enumerate(blocks)
                if b.kind == 'paragraph' and not b.in_recommendations
                and re.search(r'\b(?:branch|Learning Lab)\b', b.text, re.I)
                and re.search(r'\blocated at \d', b.text)
                and '\n' not in b.text]


class BranchHoursRecognizer:
    _LABEL = re.compile(r'^\*\*(?:Lobby|Drive-up)\*\*(?:\s+(.+))?$')
    _DAY = re.compile(
        r'^(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)'
        r'(?:[-–](?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))?'
        r':\s*(?:\d[^\n]*|Closed|ATM only)$'
    )

    @classmethod
    def recognize_at(cls, blocks: list['Block'], start: int) -> HoursRegion | None:
        if not 0 <= start < len(blocks):
            return None
        intro = blocks[start]
        if (intro.kind != 'paragraph' or intro.in_recommendations
                or not intro.text.endswith('Branch hours are:')):
            return None
        i, labels, rows = start + 1, 0, 0
        while i < len(blocks):
            b = blocks[i]
            if b.kind != 'paragraph' or b.in_recommendations != intro.in_recommendations:
                break
            label = cls._LABEL.fullmatch(b.text)
            if label:
                if labels and not rows:
                    return None
                labels += 1
                rows = 0
                if label.group(1):
                    if not cls._DAY.fullmatch(label.group(1)):
                        return None
                    rows = 1
            elif labels and cls._DAY.fullmatch(b.text):
                rows += 1
            else:
                break
            i += 1
        return HoursRegion(start, i) if labels and rows else None

    @classmethod
    def recognize(cls, blocks: list['Block']) -> list[HoursRegion]:
        return [region for i in range(len(blocks))
                if (region := cls.recognize_at(blocks, i)) is not None]


class LocationMatcher:
    def match(self, blocks: list['Block']) -> list[LocationContext]:
        locations = LocationRecognizer().recognize(blocks)
        hours = BranchHoursRecognizer.recognize(blocks)
        # First version supports only the verified single-location announcement.
        if len(locations) != 1 or len(hours) != 1:
            return []
        location, region = locations[0], hours[0]
        if location.source_block > region.start:
            return []
        if any(b.kind in {'heading', 'horizontal_rule'} or b.in_recommendations
               for b in blocks[location.source_block:region.start]):
            return []
        # An explicit location switch/extra address defeats uniqueness, even if
        # it does not fit the primary "branch ... located at" candidate grammar.
        street = re.compile(r'\b\d+\s+[^\n,;]*\b(?:Road|Rd|Street|St|Blvd|Avenue|Ave|Highway)\b', re.I)
        if len(re.findall(r'\blocated at \d', location.quote)) != 1:
            return []
        if len(street.findall(location.quote)) > 1:
            return []
        if any(i != location.source_block and b.kind == 'paragraph'
               and (re.search(r'\blocated at \d', b.text)
                    or street.search(b.text)
                    or re.search(r'\b(?:another|other|second|different) branch\b', b.text, re.I))
               for i, b in enumerate(blocks) if not b.in_recommendations):
            return []
        return [LocationContext(location.quote, location.source_block, region.start, region.end)]
