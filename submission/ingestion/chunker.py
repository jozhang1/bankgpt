from datetime import datetime
from dataclasses import dataclass
from functools import cache
from typing import Literal
import re

from ingestion.person_cards import consume_person_card
from ingestion.location_cards import consume_location_card
from ingestion.location_context import BranchHoursRecognizer, LocationContext, LocationMatcher
from ingestion.local_heading_scopes import LocalHeadingScope, LocalHeadingScopeDetector
from ingestion.metadata_extractor import DocumentMetadata
from ingestion.cleaner import PageCleaner
from ingestion.markdown_tables import TABLE_DELIMITER_CELL, is_table_start, table_end
from foundation.products import PRODUCTS


BlockKind = Literal["heading", "paragraph", "table", "list", "code", "horizontal_rule"]


@dataclass
class Block:
    kind: BlockKind
    text: str
    heading_level: int | None = None
    in_recommendations: bool = False


def _validate_heading_context(path: tuple[str, ...], levels: tuple[int, ...]) -> None:
    if len(path) != len(levels):
        raise ValueError("Heading path and levels must align")
    if any(type(level) is not int or not 1 <= level <= 6 for level in levels):
        raise ValueError("Heading levels must be between 1 and 6")


def _normalize_label(text: str) -> str:
    text = text.strip().strip("*_").strip()
    return " ".join(text.casefold().split()).rstrip(":.!?")


@cache
def _normalized_labels(labels: frozenset[str]) -> frozenset[str]:
    return frozenset(_normalize_label(label) for label in labels)


def _registered_product_labels(*, category: str | None = None) -> frozenset[str]:
    return frozenset(
        label
        for product in PRODUCTS
        if product.kind in {"product", "family"}
        and (category is None or product.category == category)
        for label in (product.name, *product.aliases)
        if label.strip()
    )


REGISTERED_PRODUCT_LABELS = _registered_product_labels()
REGISTERED_CHECKING_LABELS = _registered_product_labels(category="checking")


@dataclass
class Section:
    heading_path: tuple[str, ...]
    heading_levels: tuple[int, ...]
    blocks: list[Block]
    in_recommendations: bool = False
    # Verified source lead-in retained in blocks, not inherited as an ancestor.
    eyebrow: Block | None = None

    def __post_init__(self) -> None:
        _validate_heading_context(self.heading_path, self.heading_levels)


@dataclass
class Chunk:
    heading_path: tuple[str, ...]
    heading_levels: tuple[int, ...]
    blocks: list[Block]
    in_recommendations: bool = False
    # Verified source lead-in retained in blocks, not inherited as an ancestor.
    eyebrow: Block | None = None

    location_context: LocationContext | None = None

    def __post_init__(self) -> None:
        _validate_heading_context(self.heading_path, self.heading_levels)


@dataclass(frozen=True)
class TableContext:
    scanned_end: int
    accepted_note_indices: tuple[int, ...]
    bridge_indices: tuple[int, ...]


class EvidenceChunker:
    # Exact H5 + H2 pairs observed in the supplied snapshots. These are UI
    # lead-ins, not an invitation to move arbitrary deep headings forward.
    EYEBROW_PAIRS = {
        "who can join": frozenset({"You're Eligible to Join Dupaco If ...",
                                  "You're Eligible to Join If Any of These Sound Like You"}),
        "rates": frozenset({"More Money in Your Pocket", "Business Credit Card Rates",
            "Get on the road with low interest rates", "Compare Rates", "Boat & RV Loan Rates",
            "ATV, Jet Ski & Snowmobile Loans", "Launchpad Savings Rates", "IRA Account Rates",
            "Savings Account Rates", "Bright Savings Rates", "Checking Account Rates", "Home Rates",
            "Earn on your Bright Savings when you enroll in eStatements"}),
        "changeup savings": frozenset({"Save Every Time You Spend"}),
        "money snapshot": frozenset({"Easily Track Your Money, Accounts and Credit—All in One Place"}),
        "set goals": frozenset({"Big or Small, Your Savings and Spending Goals Feel More Possible When You Can See Your Progress"}),
        "access money": frozenset({"Easily move your money, open accounts and personalize your digital banking"}),
        "advanced security": frozenset({"Built-in Security Features Work Behind the Scenes to Protect Your Money"}),
        "faqs": frozenset({"Have questions? We have answers."}),
        "business savings rates": frozenset({"Higher rates. Guaranteed earnings."}),
        "business checking rates": frozenset({"Make day-to-day money management the least of your business worries"}),
    }

    @classmethod
    def _is_eyebrow_pair(cls, label: Block, title: str, level: int) -> bool:
        if label.kind != "heading" or label.heading_level != 5 or level != 2:
            return False
        normalize = lambda value: " ".join(value.casefold().split())
        titles = cls.EYEBROW_PAIRS.get(normalize(label.text), ())
        return normalize(title) in {normalize(value) for value in titles}

    @classmethod
    def _leading_eyebrow(cls, section: Section | Chunk) -> Block | None:
        if (section.eyebrow is not None and section.blocks
                and section.eyebrow == section.blocks[0]
                and section.heading_path and section.heading_levels
                and section.blocks[0].in_recommendations == section.in_recommendations
                and cls._is_eyebrow_pair(section.blocks[0], section.heading_path[-1],
                                        section.heading_levels[-1])):
            return section.blocks[0]
        return None

    def chunk(
        self,
        markdown: str,
        max_chars: int = 2400,
        *,
        page_title: str | None = None,
        source_url: str | None = None,
    ) -> list[Chunk]:
        """Parse, section, and pack cleaned Markdown, retaining chunk metadata."""
        return self.chunk_blocks(
            self.parse_blocks(markdown), max_chars=max_chars,
            page_title=page_title, source_url=source_url,
        )

    def chunk_blocks(
        self, blocks: list[Block], max_chars: int = 2400, *,
        page_title: str | None = None, source_url: str | None = None,
        document_metadata: DocumentMetadata | None = None,
    ) -> list[Chunk]:
        """Section and pack already-parsed blocks with optional document context."""
        blocks = self._certificate_group_boundaries(blocks, source_url)
        blocks = self._checking_product_boundaries(blocks, source_url)
        if source_url in {"https://www.dupaco.com/rates", "https://www.dupaco.com/rates/"}:
            blocks = self._rates_product_boundaries(blocks)
        scopes = LocalHeadingScopeDetector().detect(blocks, source_url=source_url)
        sections = self.build_sections(blocks, local_scopes=scopes)
        return self.chunk_sections(
            sections, max_chars=max_chars, page_title=page_title,
            document_metadata=document_metadata,
        )

    CHECKING_CONTEXT_PAGES = frozenset({
        "https://www.dupaco.com/business-rates",
        "https://www.dupaco.com/business/checking-accounts",
        "https://www.dupaco.com/business/business-checking-accounts",
        "https://www.dupaco.com/checking-offer-lp",
        "https://www.dupaco.com/tax-checking-offer-lp",
    })
    CHECKING_PRODUCT_LABELS = frozenset(
        label
        for label in REGISTERED_CHECKING_LABELS
        if _normalize_label(label) != "business free checking"
    )

    @classmethod
    def _checking_product_boundaries(
        cls, blocks: list[Block], source_url: str | None,
    ) -> list[Block]:
        """Restore exact product labels spanning descriptions and feature tables.

        Only verified pages and standalone labels are repaired. Product headings
        carry context across chunks; successive products always start sections.
        Feature tables and rate tables remain separate blocks and atomic units.
        """
        if (source_url or "").rstrip("/") not in cls.CHECKING_CONTEXT_PAGES:
            return blocks
        return [
            Block("heading", b.text, heading_level=2,
                  in_recommendations=b.in_recommendations)
            if (b.kind == "paragraph" and not b.in_recommendations
                and _normalize_label(b.text)
                in _normalized_labels(cls.CHECKING_PRODUCT_LABELS))
            else b for b in blocks
        ]

    CERTIFICATE_GROUPS = frozenset({
        "Term-Share Certificate Rates", "Business Term-Share Certificate Rates",
    })
    CERTIFICATE_CONTEXT_PAGES = frozenset({
        "https://www.dupaco.com/rates", "https://www.dupaco.com/business-rates",
        "https://www.dupaco.com/personal/savings-account",
        "https://www.dupaco.com/business/savings-accounts",
    })
    _RATE_GROUP_LINK = re.compile(
        r"- \[([^\]\n]+)\]\((https://www\.dupaco\.com/[^\s()]*#)\)"
    )

    @classmethod
    def _certificate_group_boundaries(
        cls, blocks: list[Block], source_url: str | None,
    ) -> list[Block]:
        """Restore verified certificate group/child hierarchy before splitting.

        Keep the original group link; an explicit parent heading carries its
        visible label into continuation chunks. Other links never open a group.
        """
        if (source_url or "").rstrip("/") not in cls.CERTIFICATE_CONTEXT_PAGES:
            return blocks
        result: list[Block] = []
        active = False
        region = False
        page_title = next((b.text for b in blocks if b.kind == "heading"
                           and b.heading_level in {1, 2}), None)
        for block in blocks:
            if block.in_recommendations != region:
                active = False
                region = block.in_recommendations
            match = cls._RATE_GROUP_LINK.fullmatch(block.text.strip()) if block.kind == "list" else None
            if match and match.group(2)[:-1].rstrip("/") == (source_url or "").rstrip("/") and not region:
                label = match.group(1)
                is_certificate = label in cls.CERTIFICATE_GROUPS
                if active or is_certificate or label == "Health Savings Account Rates":
                    result.append(Block("heading", label, heading_level=2,
                                        in_recommendations=region))
                active = is_certificate
                result.append(block)
                continue
            if active and block.kind == "horizontal_rule":
                # End certificate inheritance at an explicit structural break.
                if page_title:
                    result.append(Block("heading", page_title, heading_level=1,
                                        in_recommendations=region))
                active = False
            if block.kind == "heading" and block.heading_level is not None and block.heading_level <= 2:
                active = False
            if (active and block.kind == "paragraph" and block.text in {"Special", "Regular"}):
                result.append(Block("heading", block.text, heading_level=3,
                                    in_recommendations=region))
            else:
                result.append(block)
        return result

    # Product labels come from the shared registry; these remaining labels
    # provide product-family context or identify rate-sheet sections.
    RATES_PRODUCT_LABELS = REGISTERED_PRODUCT_LABELS | frozenset({
        "Certificates", "Checking Accounts", "Credit Cards", "Home Equity",
        "Money Market Accounts", "Regular", "Retirement / IRAs",
        "Savings Accounts", "Special",
    })

    @classmethod
    def _rates_product_boundaries(cls, blocks: list[Block]) -> list[Block]:
        """Restore product boundaries missing heading markup on /rates only.

        Retain every label, including duplicates, without mutating source blocks.
        H2 boundaries keep preceding CTAs with their own product and carry the
        product label into continuation chunks.
        """
        return [
            Block("heading", block.text, heading_level=2,
                  in_recommendations=block.in_recommendations)
            if (block.kind == "paragraph"
                and not block.in_recommendations
                and _normalize_label(block.text)
                in _normalized_labels(cls.RATES_PRODUCT_LABELS))
            else block
            for block in blocks
        ]

    @classmethod
    def _is_captcha_table(cls, block: Block) -> bool:
        """Defensive fallback for callers bypassing PageCleaner."""
        return block.kind == "table" and PageCleaner.is_captcha_table(block.text)

    COUNTRY_DIAL_OPTION = re.compile(
        r"^[-+*]\s+[\U0001F1E6-\U0001F1FF]{2}\s+.+?\s+\+\d{1,4}\s*$"
    )
    COUNTRY_FLAG = re.compile(r"[\U0001F1E6-\U0001F1FF]{2}")

    @classmethod
    def _retrieval_blocks(cls, blocks: list[Block]) -> list[Block]:
        """Hide verified expanded phone selectors without changing source blocks.

        Require a contiguous run of at least ten flag/name/dial-code options;
        isolated country references and ordinary business form text stay intact.
        """
        output: list[Block] = []
        i = 0
        while i < len(blocks):
            start = i
            region = blocks[i].in_recommendations
            while (i < len(blocks) and blocks[i].kind == "list"
                   and blocks[i].in_recommendations == region
                   and cls.COUNTRY_DIAL_OPTION.fullmatch(blocks[i].text.strip())):
                i += 1
            if i - start >= 10:
                if (output and output[-1].kind == "paragraph"
                        and output[-1].in_recommendations == region
                        and cls.COUNTRY_FLAG.fullmatch(output[-1].text.strip())):
                    output.pop()
                continue
            if i > start:
                output.extend(blocks[start:i])
                continue
            if not cls._is_captcha_table(blocks[i]):
                output.append(blocks[i])
            i += 1
        return output

    BOILERPLATE_BOUNDARY = re.compile(
        r"^(?:recaptcha\b|third party plugins\b"
        r"|heads up!\s*this link leads to a different website\.?)",
        re.IGNORECASE,
    )
    TABLE_NOTE = re.compile(
        r"^(?:notes?\s*:|\\?\*(?!\*)|[†‡]|\(\d+\)|\[\d+\]"
        r"|rates?\s+(?:are\s+)?(?:subject\s+to|effective\b|assume\b)"
        r"|rate\s+quotes?\b|(?:apy|apr)\s*(?:=|:|\b(?:is|assumes|based)\b)"
        r"|savings rates:\s*dividends and annual percentage yields\s*\(apy\)"
        r"|high-yield savings:\s*dividends and apy accurate\b"
        r"|subject\s+to\s+change\b"
        r"|to earn dividends\b|this account is available\b"
        r"|there is no annual fee\.\s*other fees include\s*:"
        r"|fees could reduce (?:account earnings|earnings on the account)\b"
        r"|loan annual percentage rates?\s*\(apr\)"
        r"|closing costs may range\b"
        r"|intro rate available only for qualifying borrowers\b"
        r"|dividends and annual percentage yields\s*\(apy\)"
        r"|(?:home improvement|home equity|boat\s*&\s*rv) loan rates\s*:"
        r"|atv,\s*snowmobile,?\s*&\s*jet ski loan rates\s*:"
        r"|a minimum of \$25,000 of new money is required to open an achieve money market account\b"
        r"|business accounts are excluded, and public funds are not accepted in the achieve money market account\b"
        r"|a loan is considered jumbo if the amount of the mortgage exceeds loan-servicing limits\b"
        r"|\d+\s*(?:auto loan rates|personal loan rates|savings rates"
        r"|money market rates|high-yield savings|term share certificate rates"
        r"|checking rates\s*:|(?:variable|fixed) ira rates\s*:"
        r"|launchpad savings rate\s*:|health savings account rates\s*:"
        r"|home equity line of credit\s*\(heloc\)\s*:"
        r"|unlimited atm access at all dupaco-owned atms\b"
        r"|before you open a business analysis checking account\b"
        r"|rates are effective as of\b|__apy\s*="
        r"|fixed\b[^:\n]*rate\s*:|apy\s*="
        r"|(?:business\s+)?(?:platinum|rewards|visa)\b[^:\n]*rate\s*:))",
        re.IGNORECASE,
    )

    @classmethod
    def _is_table_note(cls, text: str) -> bool:
        """Accept numeric prefixes only when the remaining text is a known note."""
        text = text.strip()
        candidates = [text]
        stripped = re.sub(r"^\d{1,2}(?=[A-Za-z_])", "", text)
        if stripped != text:
            # Observed extraction variant: 1__APY = ... . Do not strip
            # arbitrary stars, which could turn emphasized prose into a note.
            if stripped.startswith("__"):
                stripped = stripped[2:]
            elif stripped.startswith("_"):
                stripped = stripped[1:]
            candidates.append(stripped.strip())
        return any(cls.TABLE_NOTE.match(value) for value in candidates)

    def _consume_table_notes(
        self,
        blocks: list[Block],
        start: int,
        *,
        in_recommendations: bool,
        max_notes: int = 5,
    ) -> int:
        """Compatibility wrapper returning the accepted contiguous context span."""
        return self.collect_table_context(
            blocks, start, in_recommendations=in_recommendations, max_notes=max_notes,
        ).scanned_end

    TABLE_PAYMENT_EXAMPLE = re.compile(
        r"^(?:auto loan|personal loan|home improvement loan|home equity loan|boat\s*&\s*rv loan|atv loan)"
        r" payment example for illustrative purposes only\s*:", re.IGNORECASE,
    )
    TABLE_APPLICATION_LINK = re.compile(
        r"\[(?:apply today|apply now|request an application)\]" + PageCleaner._LINK_DESTINATION,
        re.IGNORECASE,
    )

    def collect_table_context(
        self, blocks: list[Block], start: int, *, in_recommendations: bool,
        max_notes: int = 5, max_blocks: int = 10, max_chars: int = 4000,
        max_bridges: int = 2,
    ) -> TableContext:
        """Accept notes, cross verified bridges, and stop at unknown content.

        Trailing bridges without a subsequent note are not included. Scanned
        text remains contiguous for chunking; extraction uses note indices only.
        """
        notes: list[int] = []
        bridges: list[int] = []
        end = start
        size = 0
        for i in range(start, min(len(blocks), start + max_blocks)):
            if len(notes) >= max_notes:
                break
            block = blocks[i]
            if block.kind not in {"paragraph", "list"}:
                break
            if block.in_recommendations != in_recommendations:
                break
            text = block.text.strip()
            size += len(text)
            if not text or size > max_chars or self.HORIZONTAL_RULE.fullmatch(text):
                break
            if self._is_faq_end(block):
                break
            if block.kind == "list":
                text = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", text, count=1)
            for wrapper in ("**", "__", "_", "*"):
                if (text.startswith(wrapper) and text.endswith(wrapper)
                        and len(text) > 2 * len(wrapper)):
                    text = text[len(wrapper):-len(wrapper)].strip()
                    break
            if self._is_table_note(text):
                notes.append(i)
                end = i + 1
            elif (self.TABLE_PAYMENT_EXAMPLE.match(text)
                  or self.TABLE_APPLICATION_LINK.fullmatch(text)):
                if len(bridges) >= max_bridges:
                    break
                bridges.append(i)
            else:
                break
        return TableContext(
            scanned_end=end, accepted_note_indices=tuple(notes),
            bridge_indices=tuple(i for i in bridges if i < end),
        )

    FAQ_END_LABELS = {
        "contact us",
        "have questions? need help?",
        "have questions? needhelp?",
        "wanna talk? we're here to help!",
        "business loan calculator",
    }

    def __init__(self):
        self._faq_end_labels = frozenset(
            self._normalize_label(label) for label in self.FAQ_END_LABELS
        )

    @staticmethod
    def _normalize_label(text: str) -> str:
        return _normalize_label(text)

    def _is_faq_end_label(self, block: Block) -> bool:
        if block.kind not in {"paragraph", "heading"}:
            return False
        text = block.text.strip()
        if "\n" in text or len(text) > 80:
            return False
        return self._normalize_label(text) in self._faq_end_labels

    def _is_boilerplate_boundary(self, block: Block) -> bool:
        if block.kind not in {"paragraph", "heading"}:
            return False
        text = block.text.strip()
        if not text:
            return False
        return bool(
            self.BOILERPLATE_BOUNDARY.match(
                self._normalize_label(text)
            )
        )

    def _is_faq_end(self, block: Block) -> bool:
        return (
            self._is_faq_end_label(block)
            or self._is_boilerplate_boundary(block)
        )

    HEADING = re.compile(
        r"^ {0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$"
    )
    LIST_ITEM = re.compile(
        r"^( *)(?:[-+*]|\d+[.)])[ \t]+\S"
    )
    FENCE = re.compile(
        r"^ {0,3}(`{3,}|~{3,})(.*)$"
    )
    TABLE_DELIMITER_CELL = TABLE_DELIMITER_CELL
    HORIZONTAL_RULE = re.compile(
        r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$"
    )

    MARKERS = {
        PageCleaner.RECOMMENDATIONS_START,
        PageCleaner.RECOMMENDATIONS_END,
    }

    @classmethod
    def _is_table_start(cls, lines: list[str], i: int) -> bool:
        return is_table_start(lines, i)

    @classmethod
    def _fence(cls, line: str):
        return cls.FENCE.match(line)

    @classmethod
    def _is_ui_marker(cls, line: str) -> bool:
        """Recognize whole-line UI labels, never substrings of body prose."""
        normalized = cls._normalize_label(line)
        return (
            any(normalized == cls._normalize_label(label) for label in cls.FAQ_END_LABELS)
            or bool(cls.BOILERPLATE_BOUNDARY.fullmatch(normalized))
        )

    @classmethod
    def _is_boundary(cls, lines: list[str], i: int) -> bool:
        line = lines[i]

        return (
            not line.strip()
            or bool(cls.HORIZONTAL_RULE.fullmatch(line))
            or line.strip() in cls.MARKERS
            or bool(cls.HEADING.match(line))
            or bool(cls._fence(line))
            or bool(cls.LIST_ITEM.match(line))
            or cls._is_table_start(lines, i)
            or cls._is_ui_marker(line)
        )

    def parse_blocks(self, markdown: str) -> list[Block]:
        """Parse cleaned Markdown into structural blocks.

        Headings exclude Markdown prefixes.
        Other blocks preserve their original Markdown.
        Recommendation markers become metadata.
        """
        lines = markdown.splitlines()
        blocks: list[Block] = []
        in_recommendations = False
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 1. Empty lines
            if not stripped:
                i += 1
                continue

            if self.HORIZONTAL_RULE.fullmatch(line):
                blocks.append(Block(
                    kind="horizontal_rule", text=line,
                    in_recommendations=in_recommendations,
                ))
                i += 1
                continue

            # 2. Recommendation boundaries
            if stripped == PageCleaner.RECOMMENDATIONS_START:
                in_recommendations = True
                i += 1
                continue

            if stripped == PageCleaner.RECOMMENDATIONS_END:
                in_recommendations = False
                i += 1
                continue

            # 3. Headings
            heading = self.HEADING.match(line)
            if heading:
                blocks.append(
                    Block(
                        kind="heading",
                        text=heading.group(2),
                        heading_level=len(heading.group(1)),
                        in_recommendations=in_recommendations,
                    )
                )
                i += 1
                continue

            start = i

            # 4. Fenced code
            fence = self._fence(line)
            if fence:
                kind: BlockKind = "code"
                opening = fence.group(1)
                fence_char = opening[0]
                fence_length = len(opening)

                i += 1

                while i < len(lines):
                    closing = self._fence(lines[i])
                    i += 1

                    if (
                        closing
                        and closing.group(1)[0] == fence_char
                        and len(closing.group(1)) >= fence_length
                        and not closing.group(2).strip()
                    ):
                        break

            # 5. Markdown tables
            elif self._is_table_start(lines, i):
                kind = "table"
                i = table_end(lines, i)

            # Whole-line UI labels must stand apart even without blank lines.
            elif self._is_ui_marker(line):
                kind = "paragraph"
                i += 1

            # 6. Lists
            elif item := self.LIST_ITEM.match(line):
                kind = "list"
                base_indent = len(item.group(1))
                i += 1

                while i < len(lines):
                    current = lines[i]

                    if not current.strip():
                        # Preserve a blank line only when followed
                        # by an indented continuation.
                        if (
                            i + 1 < len(lines)
                            and lines[i + 1].strip()
                            and self._indent(lines[i + 1]) > base_indent
                        ):
                            i += 1
                            continue
                        break

                    if (
                        current.strip() in self.MARKERS
                        or self.HORIZONTAL_RULE.fullmatch(current)
                        or self._is_ui_marker(current)
                        or self.HEADING.match(current)
                        or self._fence(current)
                        or self._is_table_start(lines, i)
                    ):
                        break

                    next_item = self.LIST_ITEM.match(current)

                    if next_item:
                        next_indent = len(next_item.group(1))

                        if next_indent <= base_indent:
                            break

                        i += 1
                        continue

                    if self._indent(current) > base_indent:
                        i += 1
                        continue

                    break

            # 7. Paragraphs
            else:
                kind = "paragraph"
                i += 1

                while i < len(lines) and not self._is_boundary(lines, i):
                    i += 1

            blocks.append(
                Block(
                    kind=kind,
                    text="\n".join(lines[start:i]).rstrip(),
                    in_recommendations=in_recommendations,
                )
            )

        return blocks

    def build_sections(
        self, blocks: list[Block], *, local_scopes: list[LocalHeadingScope] | None = None,
    ) -> list[Section]:
        """Group blocks by H2, preserving H3-H6 within each section.

        Verified local scopes save/restore the complete enclosing heading stack.
        H1 supplies page context. Recommendation boundaries are hard
        section boundaries. A title replaced before receiving content becomes
        a heading-only Section, preserving its level and boundary. Titles that
        remain inherited context need no duplicate Section.
        """
        scopes = sorted(local_scopes or [], key=lambda scope: scope.start)
        previous_end = 0
        for scope in scopes:
            if (scope.start < previous_end or scope.end > len(blocks)
                    or blocks[scope.start].kind != "heading"
                    or blocks[scope.start].heading_level != 1
                    or any(b.in_recommendations != blocks[scope.start].in_recommendations
                           for b in blocks[scope.start:scope.end])):
                raise ValueError("Invalid, overlapping or cross-region local heading scopes")
            previous_end = scope.end
        scopes_by_start = {scope.start: scope for scope in scopes}
        saved_headings: list[tuple[int, str]] | None = None
        scope_end: int | None = None
        sections: list[Section] = []
        headings: list[tuple[int, str]] = []
        pending: list[Block] = []
        boundary_heading: Block | None = None
        section_eyebrow: Block | None = None
        in_recommendations = False

        def flush(*, preserve_empty: bool = False) -> None:
            nonlocal pending, boundary_heading, section_eyebrow
            if pending and not (
                preserve_empty and boundary_heading is not None
                and all(b.kind == "horizontal_rule" for b in pending)
            ):
                sections.append(
                    Section(
                        heading_path=tuple(title for _, title in headings),
                        heading_levels=tuple(level for level, _ in headings),
                        eyebrow=section_eyebrow,
                        blocks=pending,
                        in_recommendations=in_recommendations,
                    )
                )
                pending = []
            elif preserve_empty and boundary_heading is not None:
                # The title is a block here, so exclude it from inherited
                # context: render it once, using its original heading level.
                sections.append(Section(
                    heading_path=tuple(title for _, title in headings[:-1]),
                    heading_levels=tuple(level for level, _ in headings[:-1]),
                    blocks=[boundary_heading, *pending],
                    in_recommendations=in_recommendations,
                ))
                pending = []
            boundary_heading = None
            section_eyebrow = None

        for index, block in enumerate(blocks):
            if scope_end == index:
                flush(preserve_empty=True)
                headings = saved_headings if saved_headings is not None else []
                saved_headings = None
                scope_end = None
            if index in scopes_by_start:
                flush(preserve_empty=True)
                saved_headings = list(headings)
                scope_end = scopes_by_start[index].end
                headings = []
            # Recommendation region changes are hard boundaries.
            if block.in_recommendations != in_recommendations:
                flush(preserve_empty=True)

                # Reset to page-level context when crossing regions.
                headings = [
                    (level, title)
                    for level, title in headings
                    if level == 1
                ]
                in_recommendations = block.in_recommendations

            if block.kind != "heading":
                pending.append(block)
                continue

            level = block.heading_level
            if level is None or not 1 <= level <= 6:
                raise ValueError(
                    "Heading blocks require a level between 1 and 6"
                )

            if level <= 2:
                eyebrow = None
                if (pending and pending[-1].in_recommendations == block.in_recommendations
                        and self._is_eyebrow_pair(pending[-1], block.text, level)):
                    eyebrow = pending.pop()
                flush(preserve_empty=(
                    boundary_heading is not None
                    and level <= boundary_heading.heading_level
                ))

                while headings and headings[-1][0] >= level:
                    headings.pop()

                headings.append((level, block.text))
                boundary_heading = block
                if eyebrow is not None:
                    section_eyebrow = eyebrow
                    pending.append(eyebrow)
            else:
                # H3-H6 are internal section structure.
                pending.append(block)

        flush(preserve_empty=True)
        return sections

    @classmethod
    def render_chunk(
        cls, chunk: Chunk, page_title: str | None = None, *,
        document_metadata: DocumentMetadata | None = None,
    ) -> str:
        """Render a chunk into self-contained Markdown for retrieval.

        Include inherited heading context and restore heading block prefixes.
        Non-heading Markdown and recommendation metadata remain unchanged.
        Verified eyebrow labels precede their H2 title once and do not become
        inherited ancestors. The chunk builder keeps path context separate from block headings.
        Explicit document publication context is repeated on body continuations,
        excluding recommendations and chunks already containing the source byline.
        An optional page title supplies context only when the path is empty.
        Page.title is passed by the caller and stored separately as
        Evidence.source_title; rendering never writes it into the chunk.
        """
        parts: list[str] = []
        if (chunk.location_context is not None and not chunk.in_recommendations
                and not any(b.text == chunk.location_context.quote for b in chunk.blocks)):
            parts.append("Location source:\n" + chunk.location_context.quote)
        if (document_metadata is not None and document_metadata.published_on is not None
                and document_metadata.publication_quote
                and not chunk.in_recommendations
                and not any(b.text == document_metadata.publication_quote for b in chunk.blocks)):
            parts.append(f"Document published on: {document_metadata.published_on.isoformat()}")
            parts.append("Publication source:\n" + document_metadata.publication_quote)
        if not chunk.heading_path and page_title and page_title.strip():
            parts.append(f"# {page_title.strip()}")
        levels = chunk.heading_levels
        _validate_heading_context(chunk.heading_path, levels)
        eyebrow = cls._leading_eyebrow(chunk)
        for index, (level, title) in enumerate(zip(levels, chunk.heading_path)):
            if eyebrow is not None and index == len(levels) - 1:
                parts.append(f"{'#' * eyebrow.heading_level} {eyebrow.text}")
            title = title.strip()
            if title:
                parts.append(f"{'#' * level} {title}")

        for block in chunk.blocks:
            if block is eyebrow:
                continue
            if block.kind == "heading":
                level = block.heading_level
                if level is None or not 1 <= level <= 6:
                    raise ValueError("Heading blocks require a level between 1 and 6")
                parts.append(f"{'#' * level} {block.text}")
            else:
                parts.append(block.text)

        return "\n\n".join(parts).strip()

    @classmethod
    def _chunk_size(
        cls,
        heading_path: tuple[str, ...],
        blocks: list[Block],
        page_title: str | None = None,
        *,
        heading_levels: tuple[int, ...],
        eyebrow: Block | None = None,
        document_metadata: DocumentMetadata | None = None,
        in_recommendations: bool = False,
        location_context: LocationContext | None = None,
    ) -> int:
        """Measure the actual rendered character count without changing blocks."""
        return len(cls.render_chunk(
            Chunk(heading_path=heading_path, blocks=blocks, heading_levels=heading_levels,
                  eyebrow=eyebrow, in_recommendations=in_recommendations,
                  location_context=location_context),
            page_title=page_title, document_metadata=document_metadata,
        ))

    def chunk_sections(
        self,
        sections: list[Section],
        max_chars: int = 2400,
        *,
        page_title: str | None = None,
        document_metadata: DocumentMetadata | None = None,
    ) -> list[Chunk]:
        """Pack atomic units into retrieval chunks.

        - Never cross Section boundaries.
        - Never split atomic units.
        - Preserve H3-H6 context across chunk boundaries.
        - FAQ entry/exit always starts a new chunk; other H3s pack by size.
        - Allow oversized atomic units.
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")

        chunks: list[Chunk] = []

        for section in sections:
            # Keep source sections unchanged; filter only the retrieval view.
            retrieval_section = Section(
                heading_path=section.heading_path,
                heading_levels=section.heading_levels,
                eyebrow=section.eyebrow,
                blocks=self._retrieval_blocks(section.blocks),
                in_recommendations=section.in_recommendations,
            )
            contexts = LocationMatcher().match(retrieval_section.blocks)
            context_by_block = {
                id(retrieval_section.blocks[i]): context
                for context in contexts for i in range(context.start, context.end)
            }
            units = self._section_units(retrieval_section)
            eyebrow = self._leading_eyebrow(retrieval_section)
            if not units:
                continue

            pending: list[Block] = []
            pending_path: tuple[str, ...] = section.heading_path
            section_levels = section.heading_levels
            pending_levels = section_levels
            pending_location: LocationContext | None = None
            active: list[tuple[int, str]] = []
            faq_level: int | None = (
                2 if any(self._is_faq_heading(t) for t in section.heading_path) else None
            )

            def flush() -> None:
                nonlocal pending, pending_path
                if not pending:
                    return
                chunks.append(Chunk(
                    heading_path=pending_path,
                    heading_levels=pending_levels,
                    location_context=pending_location,
                    eyebrow=eyebrow if pending[0] is eyebrow else None,
                    blocks=pending,
                    in_recommendations=section.in_recommendations,
                ))
                pending = []

            for unit in units:
                if not unit:
                    continue

                if unit[0].kind == "horizontal_rule":
                    flush()
                    active = []
                    continue

                unit_location = next((context_by_block[id(b)] for b in unit
                                      if id(b) in context_by_block), None)
                if pending and unit_location != pending_location:
                    flush()
                content_unit = [block for block in unit if block is not eyebrow]
                is_card = bool(content_unit) and (
                    self._consume_article_card(content_unit, 0) == len(content_unit)
                    or consume_person_card(content_unit, 0) == len(content_unit)
                    or consume_location_card(content_unit, 0) == len(content_unit)
                )
                if is_card:
                    # Card headings are local; they cannot inherit a previous
                    # person's heading or change subsequent heading context.
                    if active:
                        flush()
                    active = []

                # Unit construction keeps FAQ answers atomic. Detect region
                # changes before packing so body text never shares a FAQ chunk.
                previous_faq_level = faq_level
                faq_entry = False
                faq_exit = False
                faq_exit_via_label = False
                for block in ([] if is_card else content_unit):
                    if block.kind == "heading":
                        level = block.heading_level
                        if level is None:
                            raise ValueError("Heading block missing level")
                        if faq_level is not None and level <= faq_level:
                            faq_level = None
                            faq_exit = True
                        if self._is_faq_heading(block.text):
                            faq_level = level
                            faq_entry = True
                        elif faq_level is not None and self._is_faq_end(block):
                            faq_level = None
                            faq_exit = True
                            faq_exit_via_label = True
                    elif faq_level is not None and self._is_faq_end(block):
                        faq_level = None
                        faq_exit = True
                        faq_exit_via_label = True

                if pending and (faq_entry or faq_exit):
                    flush()
                if faq_exit_via_label and previous_faq_level is not None:
                    active = [(level, title) for level, title in active
                              if level < previous_faq_level]

                if (
                    pending
                    and len(pending_path) > len(section.heading_path)
                    and unit[0].kind == "heading"
                ):
                    flush()

                context_before = tuple(active)
                first_heading = next(
                    (b for b in unit if b.kind == "heading"), None,
                )
                if unit[0].kind == "heading" and first_heading is not None:
                    level = first_heading.heading_level
                    context_before = tuple(
                        (heading_level, title) for heading_level, title in active
                        if level is not None and heading_level < level
                    )

                unit_path = section.heading_path + tuple(title for _, title in context_before)
                unit_levels = section_levels + tuple(level for level, _ in context_before)
                if pending and self._chunk_size(
                    pending_path,
                    pending + unit,
                    page_title=page_title,
                    heading_levels=pending_levels,
                    location_context=pending_location,
                    eyebrow=eyebrow if pending[0] is eyebrow else None,
                    document_metadata=document_metadata,
                    in_recommendations=section.in_recommendations,
                ) > max_chars:
                    flush()

                if not pending:
                    pending_path = unit_path
                    pending_levels = unit_levels
                    pending_location = unit_location
                pending.extend(unit)

                for block in ([] if is_card else content_unit):
                    if block.kind != "heading":
                        continue
                    level = block.heading_level
                    if level is None:
                        raise ValueError("Heading block missing level")
                    while active and active[-1][0] >= level:
                        active.pop()
                    active.append((level, block.text))

            flush()

        return chunks

    @staticmethod
    def _is_faq_heading(title: str) -> bool:
        normalized = " ".join(
            title.casefold().strip().rstrip(":.!?").split()
        )

        return (
            normalized in {"faq", "faqs"}
            or normalized.startswith("frequently asked questions")
            or normalized == "have questions? we have answers"
        )


    @classmethod
    def _is_question(cls, block: Block) -> bool:
        """Recognize a list item containing a question."""
        if block.kind != "list":
            return False

        first_line = block.text.splitlines()[0]
        first_line = re.sub(r"^ *(?:[-+*]|\d+[.)])[ \t]+", "", first_line, count=1)
        first_line = re.sub(
            r"\[([^\]]+)\]\([^)]*\)",
            r"\1",
            first_line,
        )

        return first_line.strip().rstrip("*_ ").endswith("?")


    @staticmethod
    def _is_question_heading(block: Block) -> bool:
        return (
            block.kind == "heading"
            and block.text.strip().rstrip("*_ ").endswith("?")
        )


    # Non-product table titles, calculator outputs, and form schedules.
    # Product names and aliases come from the shared product registry.
    TABLE_INTRO_CONTEXT_LABELS = frozenset({
        "Balance", "Balance Over Time", "Balance at retirement", "Breakdown",
        "Certificates", "Checking Accounts", "Comparison",
        "Cost of Buying vs Leasing", "Credit Cards", "EXPENSES",
        "For Sale by Dupaco", "Home Equity", "Life insurance coverage",
        "MISCELLANEOUS FEES AND CHARGES", "Money Market Accounts",
        "Money Market Rates", "Monthly Breakdown", "Monthly Income",
        "Monthly Payment", "Monthly Payment Breakdown",
        "Monthly Payment Comparison", "Payment", "Payoff Scenarios",
        "Regular", "Remaining Balance", "Rent Roll and Operating Expenses",
        "Retirement / IRAs",
        "SCHEDULE A – REAL ESTATE", "SCHEDULE B – STOCKS AND BONDS",
        "SCHEDULE B –STOCKS AND BONDS (401K, IRA, Mutual Funds, etc.)",
        "SCHEDULE C – DEBTS OWED TO BANKS AND CREDIT UNIONS (Including Auto Loans, Personal Loans, Credit Cards and Lines of Credit)",
        "SCHEDULE C – DEBTS OWED TO BANKS, CREDIT UNIONS (Including Credit Cards and Lines of Credit)",
        "SCHEDULE D – DEBTS AND ACCOUNTS OWED TO OTHERS (Itemize Debts over $500; Combine Debts under $500)",
        "SCHEDULE D – LIFE INSURANCE", "SCHEDULE E – PARTNERSHIPS",
        "SCHEDULE F – LIFE INSURANCE", "Saving Account Rates", "Savings Accounts",
        "Savings breakdown", "Savings over time", "Schedule", "Special",
        "Total", "Total Interest", "Total Interest Comparison", "Total Payment",
        "Total cost", "Yearly",
    })
    # These labels did not bind a following table in the supplied snapshots.
    # Some still serve other roles, such as certificate or rate headings.
    TABLE_INTRO_UNBOUND_LABELS = frozenset({
        "1st Rate Checking Account", "Business 1st Rate Checking",
        "Certificates", "Checking Accounts", "Credit Cards",
        "Free Checking Account", "Health Savings (HSA)", "Home Equity",
        "Launchpad Savings Rates", "Money Market Accounts", "Recreational Loans",
        "Regular", "Retirement / IRAs",
        "SCHEDULE C – DEBTS OWED TO BANKS AND CREDIT UNIONS (Including Auto Loans, Personal Loans, Credit Cards and Lines of Credit)",
        "Savings Accounts", "Special",
    })
    TABLE_INTRO_LABELS = (
        REGISTERED_PRODUCT_LABELS | TABLE_INTRO_CONTEXT_LABELS
    ) - TABLE_INTRO_UNBOUND_LABELS

    @classmethod
    def _is_table_intro(cls, block: Block) -> bool:
        if block.kind != "paragraph" or "\n" in block.text:
            return False
        text = block.text.strip()
        # Links (including linked images) and image descriptions are not labels.
        if re.search(r"!?\[[^\]]*\]\(", text) or text.startswith("Image description:"):
            return False
        return cls._normalize_label(text) in _normalized_labels(cls.TABLE_INTRO_LABELS)

    _CARD_IMAGE = re.compile(
        r"\[!\[[^\]\n]*\]\([^\n]+\)\]\((https?://[^\s()]+)\)"
    )
    _CARD_LINK = re.compile(
        r'\[([^\]\n]+)\]\((https?://[^\s()]+)(?:\s+"[^"\n]*")?\)'
    )

    @staticmethod
    def _is_publication_date(block: Block) -> bool:
        if block.kind != "paragraph":
            return False
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                datetime.strptime(block.text.strip(), fmt)
                return True
            except ValueError:
                pass
        return False

    def _consume_article_card(
        self, blocks: list[Block], start: int,
    ) -> int | None:
        """Recognize a bounded, URL-verified image/date/title/excerpt card.

        Return the exclusive end only when a matching closing link is present.
        Incomplete or unfamiliar layouts retain the ordinary grouping behavior.
        """
        if start >= len(blocks) or blocks[start].kind != "paragraph":
            return None
        image = self._CARD_IMAGE.fullmatch(blocks[start].text.strip())
        if image is None:
            return None
        url = image.group(1)
        region = blocks[start].in_recommendations
        limit = min(len(blocks), start + 12)
        i = start + 1
        if i < limit and blocks[i].kind == "paragraph" and re.fullmatch(
            r"\d+ min read", blocks[i].text.strip(), re.IGNORECASE,
        ):
            i += 1
        if i >= limit or not self._is_publication_date(blocks[i]):
            return None
        i += 1
        if i >= limit:
            return None
        title = blocks[i]
        link = self._CARD_LINK.fullmatch(title.text.strip())
        if (title.kind != "heading" or title.heading_level is None
                or title.heading_level < 3 or link is None
                or link.group(2) != url):
            return None
        i += 1
        while i < limit:
            block = blocks[i]
            if block.kind != "paragraph" or self._CARD_IMAGE.fullmatch(block.text.strip()):
                return None
            closing = self._CARD_LINK.fullmatch(block.text.strip())
            if closing and closing.group(1).strip("*_ ").casefold() in {
                "read more", "keep reading",
            }:
                if closing.group(2) != url:
                    return None
                end = i + 1
                return end if all(
                    b.in_recommendations == region for b in blocks[start:end]
                ) else None
            i += 1
        return None

    STEP_ITEM = re.compile(
        r"^(?P<indent> *)[-+*]\s+(?:Step\s+(?P<plain>\d{1,2}):"
        r"|\*\*Step\s+(?P<bold>\d{1,2}):\*\*)\s+\S",
        re.IGNORECASE,
    )

    @classmethod
    def _consume_step_sequence(cls, blocks: list[Block], start: int) -> int | None:
        """Keep a contiguous, explicit Step 1..N list together.

        Only the plain and bold Step-label formats observed in the corpus are
        recognized. Gaps, restarts, indentation or region changes reject the
        sequence rather than presenting a partial procedure as a complete unit.
        """
        if start >= len(blocks) or blocks[start].kind != "list":
            return None
        first = cls.STEP_ITEM.match(blocks[start].text)
        if first is None or int(first.group("plain") or first.group("bold")) != 1:
            return None
        indent = first.group("indent")
        region = blocks[start].in_recommendations
        end = start
        expected = 1
        while end < len(blocks):
            block = blocks[end]
            match = cls.STEP_ITEM.match(block.text) if block.kind == "list" else None
            if match is None:
                break
            number = int(match.group("plain") or match.group("bold"))
            if (number != expected or match.group("indent") != indent
                    or block.in_recommendations != region):
                return None
            expected += 1
            end += 1
        return end if end - start >= 2 else None

    def _section_units(self, section: Section) -> list[list[Block]]:
        """Build indivisible content units without duplicating blocks.

        - Heading + associated content stay together.
        - FAQ question + complete answer stay together.
        - Consecutive explicit Step 1..N items stay together.
        - Tables retain adjacent introductory text and notes.
        """
        eyebrow = self._leading_eyebrow(section)
        blocks = section.blocks[1:] if eyebrow is not None else section.blocks
        units: list[list[Block]] = []
        i = 0

        # H2 FAQ sections have their heading in section.heading_path.
        faq_level: int | None = (
            2
            if any(
                self._is_faq_heading(title)
                for title in section.heading_path
            )
            else None
        )

        while i < len(blocks):
            unit: list[Block] = []
            hours = BranchHoursRecognizer.recognize_at(blocks, i)
            if hours is not None:
                units.append(blocks[i:hours.end])
                i = hours.end
                continue
            card_end = consume_location_card(blocks, i)
            if card_end is None:
                card_end = consume_person_card(blocks, i)
            if card_end is None:
                card_end = self._consume_article_card(blocks, i)
            if card_end is not None:
                units.append(blocks[i:card_end])
                i = card_end
                continue

            if blocks[i].kind == "horizontal_rule":
                units.append([blocks[i]])
                i += 1
                continue

            # Collect consecutive headings.
            while (i < len(blocks) and blocks[i].kind == "heading"
                   and consume_person_card(blocks, i) is None
                   and consume_location_card(blocks, i) is None):
                heading = blocks[i]
                level = heading.heading_level

                if level is None or not 1 <= level <= 6:
                    raise ValueError(
                        "Heading blocks require a level between 1 and 6"
                    )

                if faq_level is not None and level <= faq_level:
                    faq_level = None

                if self._is_faq_heading(heading.text):
                    faq_level = level
                elif faq_level is not None and self._is_faq_end(heading):
                    faq_level = None

                unit.append(heading)
                i += 1

            if i >= len(blocks):
                # Preserve trailing headings instead of dropping them.
                if unit:
                    units.append(unit)
                break

            if blocks[i].kind == "horizontal_rule":
                if unit:
                    units.append(unit)
                continue

            # Do not attach a following card image to preceding headings.
            if unit and (self._consume_article_card(blocks, i) is not None
                         or consume_person_card(blocks, i) is not None
                         or consume_location_card(blocks, i) is not None):
                units.append(unit)
                continue

            block = blocks[i]
            if faq_level is not None and self._is_faq_end(block):
                faq_level = None
            unit.append(block)
            i += 1

            # Case 1: FAQ question expressed as a heading.
            question_heading = next(
                (
                    h for h in reversed(unit)
                    if h.kind == "heading"
                    and self._is_question_heading(h)
                ),
                None,
            )

            if faq_level is not None and question_heading is not None:
                question_level = question_heading.heading_level

                # Consume the full answer until the next heading
                # at the question's level or above.
                while i < len(blocks):
                    next_block = blocks[i]
                    if (self._consume_article_card(blocks, i) is not None
                            or consume_person_card(blocks, i) is not None
                         or consume_location_card(blocks, i) is not None):
                        break

                    if next_block.kind == "horizontal_rule":
                        break

                    if self._is_faq_end(next_block):
                        faq_level = None
                        break

                    if next_block.kind == "heading":
                        next_level = next_block.heading_level
                        if (
                            next_level is not None
                            and next_level <= question_level
                        ):
                            break

                        if self._is_question_heading(next_block):
                            break

                    if self._is_question(next_block):
                        break

                    unit.append(next_block)
                    i += 1

            # Case 2: FAQ question expressed as a list item.
            elif faq_level is not None and self._is_question(block):
                while i < len(blocks):
                    next_block = blocks[i]
                    if (self._consume_article_card(blocks, i) is not None
                            or consume_person_card(blocks, i) is not None
                         or consume_location_card(blocks, i) is not None):
                        break

                    if next_block.kind == "horizontal_rule":
                        break

                    if self._is_faq_end(next_block):
                        faq_level = None
                        break

                    if next_block.kind == "heading":
                        break

                    if self._is_question(next_block):
                        break

                    unit.append(next_block)
                    i += 1

            # Case 3: An explicit procedure is indivisible, even if oversized.
            elif block.kind == "list" and (
                step_end := self._consume_step_sequence(blocks, i - 1)
            ) is not None:
                unit.extend(blocks[i:step_end])
                i = step_end

            # Case 4: Table with surrounding context.
            elif block.kind == "table":
                # Only a verified standalone label can introduce this table. Absorbing
                # a prior table's notes would recursively merge unrelated tables.
                # A heading already collected here is also an explicit boundary.
                previous = units[-1][0] if units and len(units[-1]) == 1 else None

                if (
                    len(unit) == 1
                    and previous is not None
                    and self._is_table_intro(previous)
                    and previous.in_recommendations == block.in_recommendations
                ):
                    unit = units.pop() + unit

                end = self._consume_table_notes(
                    blocks, i, in_recommendations=block.in_recommendations
                )
                unit.extend(blocks[i:end])
                i = end

            units.append(unit)

        if eyebrow is not None:
            if units and units[0][0].kind != "horizontal_rule":
                units[0] = [eyebrow, *units[0]]
            else:
                units.insert(0, [eyebrow])
        return units

    @staticmethod
    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))
