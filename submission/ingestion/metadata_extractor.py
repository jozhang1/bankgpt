"""Document metadata model and extraction from parsed source blocks."""
from dataclasses import dataclass
from datetime import date, datetime
import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from ingestion.chunker import Block


@dataclass(frozen=True)
class DocumentMetadata:
    """Source-level metadata shared by all chunks of a document.

    Publication time is distinct from snapshot and fetch time. Preserve its
    supporting source text; unknown publication dates remain ``None``.
    """

    page_title: str
    source_url: str
    published_on: date | None = None
    publication_quote: str | None = None


class MetadataExtractor:
    # Modern byline immediately before the article's first H1.
    _BYLINE = re.compile(
        r'\[[^\]\n]+\]\(https://www\.dupaco\.com/(?P<section>newsroom|learn/blog)'
        r'/author/[^()/\s]+/?\)\n'
        r'(?P<date>(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December) \d{1,2}, \d{4})'
    )
    # Legacy HTML export: linked H1, followed immediately by this paragraph.
    _WRITTEN_BY = re.compile(
        r'Written by \[[^\]\n]+\]\(https?://www\.dupaco\.com/'
        r'(?P<section>newsroom|learn/blog|business/resources)/author/[^()/\s]+/?\)'
        r' \\?\| (?P<date>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4})'
        r',? (?P<time>\d{1,2}:\d{2}:\d{2} (?:AM|PM))'
    )
    _LINKED_TITLE = re.compile(
        r'\[(?:\\.|[^\]\\\n])+\]\((?P<url>https?://www\.dupaco\.com/[^)\s]+)\)'
    )
    # Exact source aliases verified in the snapshots; do not accept arbitrary
    # pages merely because they embed a linked article header.
    _ARTICLE_ALIASES = {
        '/protection/identity-protection/online-safety':
            '/learn/blog/how-to-protect-your-identity-from-fraudsters',
        '/2018/09/25/understand-insurance-deductibles':
            '/learn/blog/understand-insurance-deductibles',
    }

    def extract(
        self, blocks: list['Block'], *, page_title: str, source_url: str,
    ) -> DocumentMetadata:
        empty = DocumentMetadata(page_title=page_title, source_url=source_url)
        url = urlsplit(source_url)
        if url.hostname != 'www.dupaco.com':
            return empty
        first_h1 = next((i for i, b in enumerate(blocks)
                         if b.kind == 'heading' and b.heading_level == 1), None)
        if first_h1 is None or blocks[first_h1].in_recommendations:
            return empty
        matches = []
        for i, block in enumerate(blocks[:first_h1]):
            if block.kind == 'paragraph' and not block.in_recommendations:
                match = self._BYLINE.fullmatch(block.text.strip())
                if match:
                    matches.append((i, match, 'modern'))
        if first_h1 + 1 < len(blocks):
            block = blocks[first_h1 + 1]
            if block.kind == 'paragraph' and not block.in_recommendations:
                match = self._WRITTEN_BY.fullmatch(block.text.strip())
                if match:
                    matches.append((first_h1 + 1, match, 'legacy'))
        if len(matches) != 1:
            return empty
        index, match, style = matches[0]
        if style == 'modern':
            article = re.fullmatch(r'/(newsroom|learn/blog)/[^/]+/?', url.path)
            if (article is None or match.group('section') != article.group(1)
                    or index != first_h1 - 1):
                return empty
        else:
            title = self._LINKED_TITLE.fullmatch(blocks[first_h1].text)
            if title is None:
                return empty
            canonical_path = urlsplit(title.group('url')).path.rstrip('/')
            section = match.group('section')
            prefix = '/' + section + '/'
            if not canonical_path.startswith(prefix):
                return empty
            parts = canonical_path[len(prefix):].split('/')
            if not all(parts) or any(part in {'author', 'tag', 'page'} for part in parts):
                return empty
            source_path = url.path.rstrip('/')
            if (source_path != canonical_path
                    and self._ARTICLE_ALIASES.get(source_path) != canonical_path):
                return empty
        try:
            if style == 'modern':
                published_on = datetime.strptime(match.group('date'), '%B %d, %Y').date()
            else:
                # Validate the observed timestamp, but retain only its date.
                # No timezone is supplied; never invent one or convert to UTC.
                published_on = datetime.strptime(
                    match.group('date') + ' ' + match.group('time'), '%b %d, %Y %I:%M:%S %p',
                ).date()
        except ValueError:
            return empty
        return DocumentMetadata(page_title=page_title, source_url=source_url,
                                published_on=published_on, publication_quote=blocks[index].text)
