"""Recognize bounded, verified HTML calculator heading scopes."""
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from ingestion.chunker import Block


@dataclass(frozen=True)
class LocalHeadingScope:
    start: int
    end: int  # exclusive, page-level block indices

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError('Invalid local heading scope')


class LocalHeadingScopeDetector:
    def detect(self, blocks: list['Block'], *, source_url: str | None) -> list[LocalHeadingScope]:
        url = urlsplit(source_url or '')
        if (url.hostname != 'www.dupaco.com' or url.path.lower().endswith('.pdf')
                or not url.scheme in {'http', 'https'}):
            return []
        first_h1 = next((i for i, b in enumerate(blocks)
                         if b.kind == 'heading' and b.heading_level == 1
                         and not b.in_recommendations), None)
        if first_h1 is None:
            return []
        scopes = []
        signatures = {'Calculators': ('HOME LOANS', 'Mortgage Payment'),
                      'Explore at Your Own Pace': ('HOME LOANS', 'Mortgage Payment'),
                      'Crunch the Numbers': ('PERSONAL LOANS', 'Personal Loan')}
        for i in range(first_h1 + 1, len(blocks) - 4):
            heading = blocks[i]
            if (heading.kind != 'heading' or heading.heading_level != 1
                    or heading.in_recommendations or heading.text not in signatures):
                continue
            category, title, description, link = blocks[i + 1:i + 5]
            expected_category, expected_title = signatures[heading.text]
            if not (category.kind == 'paragraph' and category.text == expected_category
                    and title.kind == 'heading' and title.heading_level == 4
                    and title.text == expected_title
                    and description.kind == 'paragraph'
                    and link.kind == 'paragraph'
                    and re.fullmatch(r'\[Calculate\]\(https://www\.dupaco\.com/calculators/[^\s)]+\)', link.text)
                    and all(not b.in_recommendations for b in blocks[i + 1:i + 5])):
                continue
            end = next((j for j in range(i + 1, len(blocks))
                        if blocks[j].in_recommendations
                        or (blocks[j].kind == 'heading' and blocks[j].heading_level in (1, 2))), len(blocks))
            if end >= i + 5:
                scopes.append(LocalHeadingScope(i, end))
        return scopes
