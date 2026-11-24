"""Deterministic candidate extraction.

Extraction, normalization, and adjudication are separate responsibilities.
"""

from dataclasses import dataclass, field
from datetime import date
import re
from typing import Literal
from uuid import UUID

from ingestion.chunker import Block, EvidenceChunker
from foundation.contract import EntityReference, Evidence


SignalKind = Literal[
    "product_column", "table_header", "table_label", "parent_label", "heading", "source_url",
    "product_link", "subject",
]


@dataclass(frozen=True)
class IdentitySignal:
    kind: SignalKind
    label: str
    quote: str


@dataclass
class CandidateFact:
    entity: str
    attribute: str
    qualifiers: dict[str, str | bool]
    value: str
    unit: str | None
    evidence_id: UUID
    quote: str
    table_header: str
    heading_path: tuple[str, ...]
    raw_date_notes: tuple[str, ...] = ()
    effective_from: date | None = None
    effective_to: date | None = None
    date_quote: str | None = None
    issues: list[str] = field(default_factory=list)
    identity_signals: tuple[IdentitySignal, ...] = ()
    entity_refs: tuple[EntityReference, ...] = ()


@dataclass(frozen=True)
class DateNote:
    value: date
    quote: str
    scope: str | None
    ambiguous: bool


class FactExtractor:
    """Extract rate cells and explicit minimum-balance requirements from tables.

    - Preserve raw values, row conditions, and source wording.
    - Exclude recommendation content.
    - Read only accepted table notes, never bridge content.
    - Resolve dates only when applicability is sufficiently explicit.
    - Do not normalize product identity or adjudicate conflicting facts.
    """

    METRIC = re.compile(
        r"\b(?:apr|apy|rate|rates|yield|yields)\b",
        re.IGNORECASE,
    )

    CELL_METRIC = re.compile(
        r"\b(APR|APY)(?=\b|\d)",
        re.IGNORECASE,
    )

    PERCENT = re.compile(
        r"\d[\d.,]*\s*%",
    )

    EFFECTIVE_DATE = re.compile(
        r"\brates?\s+(?:are\s+)?effective\s+(?:as\s+of\s+)?"
        r"(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?!\d)",
        re.IGNORECASE,
    )

    PRODUCT_HEADERS = frozenset({
        "product",
        "account",
        "account type",
        "loan type",
    })

    ROW_LABEL_HEADERS = frozenset({
        "term",
        "certificate",
        "certificate type",
        "product",
        "product type",
        "account",
        "account type",
        "loan type",
        "loan program",
        "program",
        "description",
    })

    @staticmethod
    def _minimum_attribute(header: str) -> str | None:
        # Numeric suffixes are source footnote markers, not different metrics.
        label = header.strip().strip("*_ ").casefold()
        label = re.sub(r"\d{1,2}$", "", label).strip()
        return {
            "min. to open": "minimum_opening_balance",
            "min. to earn": "minimum_earning_balance",
        }.get(label)

    def __init__(self):
        self._parser = EvidenceChunker()

    @staticmethod
    def _cells(line: str) -> list[str]:
        """Split Markdown table cells on unescaped pipes."""
        return [
            cell.strip()
            for cell in re.split(
                r"(?<!\\)\|",
                line.strip().strip("|"),
            )
        ]

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(
            re.findall(r"[a-z0-9]+", value.casefold())
        )

    @classmethod
    def _scope_matches(
        cls,
        scope: str,
        entity: str,
        conditions: dict[str, str],
    ) -> bool:
        """Require an exact normalized label match.

        No arbitrary substring matching or inferred aliases.
        """
        target = cls._normalize_name(scope)

        if not target:
            return False

        labels = {
            cls._normalize_name(entity),
            *(
                cls._normalize_name(value)
                for value in conditions.values()
            ),
        }

        return target in labels

    @staticmethod
    def _date_scope(
        note: str,
        match: re.Match[str],
    ) -> tuple[str | None, bool]:
        """Return (scope, ambiguous).

        (None, False): explicitly table-wide.
        (str, False): explicit suffix product scope.
        (str, True): possible prefix product scope.
        (None, True): unresolved contextual wording.

        Prefix scopes must be verified against row labels later.
        """
        before = note[:match.start()].split("\n")[-1]
        before = re.sub(
            r"^\s*\d{1,2}\s*",
            "",
            before,
        )
        before = before.strip("*_ \t")

        after = note[match.end():]

        # Example:
        # "Rates effective 08/31/2026 for Bright Savings."
        suffix = re.match(
            r"\s+for\s+([^.;\n]+)",
            after,
            re.IGNORECASE,
        )

        if suffix:
            scope = suffix.group(1).strip("*_ \t")

            if not scope or before:
                return None, True

            return scope, False

        # Example:
        # "Bright Savings rates effective 08/31/2026."
        #
        # The prefix is a candidate scope, not proof that
        # the text names a product.
        if before:
            return before, True

        trailing = after.strip("*_ \t")

        # Example:
        # "Rates effective 08/31/2026."
        if not trailing or trailing.startswith((".", ",", ";")):
            return None, False

        # Example:
        # "Rates effective 08/31/2026 and are subject to change."
        if re.fullmatch(
            r"and (?:are )?subject to change(?: without notice)?\.?",
            trailing,
            re.IGNORECASE,
        ):
            return None, False

        return None, True

    @classmethod
    def _parse_date_notes(
        cls,
        notes: list[str],
    ) -> tuple[list[DateNote], tuple[str, ...]]:
        """Parse dates without discarding their original statements."""
        parsed: list[DateNote] = []
        raw: list[str] = []

        for note in notes:
            # Strip only a matched whole-note underscore wrapper for parsing.
            # Match offsets and scope checks must use the same parsing text;
            # stored notes and date quotes retain the original Markdown.
            parsing_note = note.strip()
            for wrapper in ("__", "_"):
                if (parsing_note.startswith(wrapper) and parsing_note.endswith(wrapper)
                        and len(parsing_note) > 2 * len(wrapper)):
                    parsing_note = parsing_note[len(wrapper):-len(wrapper)].strip()
                    break
            matches = list(cls.EFFECTIVE_DATE.finditer(parsing_note))

            if not matches:
                continue

            raw.append(note)

            for match in matches:
                month, day, year = map(int, match.groups())

                if year < 100:
                    year += 2000

                try:
                    parsed_date = date(year, month, day)
                except ValueError:
                    continue

                scope, ambiguous = cls._date_scope(
                    parsing_note,
                    match,
                )

                parsed.append(
                    DateNote(
                        value=parsed_date,
                        quote=note,
                        scope=scope,
                        ambiguous=ambiguous,
                    )
                )

        return parsed, tuple(raw)

    @classmethod
    def _resolve_date(
        cls,
        date_notes: list[DateNote],
        entity: str,
        conditions: dict[str, str],
    ) -> tuple[date | None, str | None, list[str]]:
        """Resolve an applicable date conservatively.

        Priority:
        1. Exactly matched product-specific dates.
        2. Unambiguous table-wide dates.
        3. Otherwise unresolved.

        An unresolved statement blocks a table-wide fallback,
        but does not invalidate a verified product-specific date.
        """
        matched_scoped: dict[date, str] = {}
        table_wide: dict[date, str] = {}
        unresolved: list[str] = []

        for item in date_notes:
            if item.scope is None and not item.ambiguous:
                table_wide[item.value] = item.quote
                continue

            if (
                item.scope is not None
                and cls._scope_matches(
                    item.scope,
                    entity,
                    conditions,
                )
            ):
                matched_scoped[item.value] = item.quote
                continue

            if item.ambiguous:
                unresolved.append(item.quote)

        # Product-specific statements override table-wide dates.
        dates = (
            matched_scoped
            if matched_scoped
            else table_wide
        )

        issues: list[str] = []

        if len(dates) > 1:
            issues.append(
                "multiple_effective_dates_in_table_notes"
            )

        scope_uncertain = (
            (bool(unresolved) and not bool(matched_scoped))
            or (bool(date_notes) and not bool(dates))
        )

        if scope_uncertain:
            issues.append(
                "effective_date_scope_uncertain"
            )

        if len(dates) != 1 or scope_uncertain:
            return None, None, issues

        effective = next(iter(dates))

        return (
            effective,
            dates[effective],
            issues,
        )

    @classmethod
    def _header_metric(
        cls,
        header: str,
    ) -> str | None:
        normalized = header.casefold()

        if "annual percentage yield" in normalized:
            return "APY"

        if "annual percentage rate" in normalized:
            return "APR"

        match = cls.CELL_METRIC.search(header)

        return match.group(1).upper() if match else None

    @classmethod
    def _extract_attribute(
        cls,
        cell: str,
        header: str,
    ) -> tuple[str, list[str]]:
        issues: list[str] = []

        explicit = cls.CELL_METRIC.search(cell)
        header_metric = cls._header_metric(header)

        if explicit:
            attribute = explicit.group(1).upper()
        elif header_metric:
            attribute = header_metric
        else:
            attribute = header
            issues.append("unresolved_rate_metric")

        if (
            explicit
            and header_metric
            and explicit.group(1).upper() != header_metric
        ):
            issues.append(
                "cell_metric_disagrees_with_header"
            )

        return attribute, issues

    @staticmethod
    def _is_condition_value(value: str) -> bool:
        """Reject recognizable row conditions, without resolving product IDs."""
        text = value.strip().strip("*_ ")
        if not text or text in {"-", "–", "—"}:
            return True
        if re.search(r"model year", text, re.IGNORECASE):
            return True
        if re.match(r"^\$\s*\d", text):
            return True
        # Bare numbers/ranges are tiers or years, not product names. A named
        # product such as 1st Rate Checking must not be rejected by its digit.
        if re.fullmatch(
            r"\d[\d,]*(?:\.\d+)?(?:\s*[-–—]\s*\$?\s*\d[\d,]*(?:\.\d+)?)?\s*\+?",
            text,
        ):
            return True
        return bool(re.fullmatch(
            r"\d+\s*(?:months?|mo\.?|years?|yrs?\.?)", text, re.IGNORECASE,
        ))

    @classmethod
    def _extract_entity(
        cls,
        source_title: str,
        headers: list[str],
        cells: list[str],
    ) -> str:
        """Keep a raw product label, excluding recognizable row conditions.

        Canonical identity and signal arbitration belong to EntityNormalizer.
        Rejected conditions remain unchanged in qualifiers and IdentitySignals.
        """
        entity = source_title

        for j, header in enumerate(headers):
            if header.strip().casefold() not in cls.PRODUCT_HEADERS:
                continue

            value = cells[j].strip()

            if not cls._is_condition_value(value):
                entity = value
                break

        return entity

    IDENTITY_COLUMN_HEADERS = frozenset({
        "product", "product name", "account", "account type", "card type",
        "loan type", "loan program",
    })
    # Explicit catalog grouping labels, not inferred canonical product IDs.
    PARENT_LABELS = frozenset({
        "certificates", "term-share certificates", "business certificates",
        "savings accounts", "money market accounts", "checking accounts",
        "retirement / iras", "iras", "ira certificates", "ira account rates",
        "business savings", "business money market", "business credit cards",
    })

    _MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^\s()]+)\)")

    def _table_product_links(
        self, blocks: list[Block], table_index: int,
    ) -> tuple[IdentitySignal, ...]:
        """Read a table's trailing detail links without crossing unknown prose.

        Only verified adjacent notes/examples may intervene. Application links
        in the same paragraph are retained in the quote but do not select scope.
        """
        table = blocks[table_index]
        context = self._parser.collect_table_context(
            blocks, table_index + 1, in_recommendations=table.in_recommendations,
            max_notes=10, max_blocks=10, max_chars=6000,
        )
        index = context.scanned_end
        if index >= len(blocks):
            return ()
        block = blocks[index]
        if block.kind != "paragraph" or block.in_recommendations != table.in_recommendations:
            return ()
        text = block.text.strip()
        links = list(self._MARKDOWN_LINK.finditer(text))
        # A link embedded in narrative is not a verified product-detail block.
        if not links or self._MARKDOWN_LINK.sub("", text).strip():
            return ()
        return tuple(
            IdentitySignal("product_link", link.group(2), block.text)
            for link in links if link.group(1).strip().casefold() in {"explore", "see details"}
        )

    def _table_identity_signals(
        self, blocks: list[Block], table_index: int,
        headings: list[tuple[int, str]], evidence: Evidence,
    ) -> tuple[IdentitySignal, ...]:
        signals = [IdentitySignal("heading", title, title) for _, title in headings]
        # The chunker restored only verified certificate group headings. These
        # survive splitting and explicitly bind Special/Regular to their parent.
        certificate_parent = next((title for _, title in reversed(headings)
                                   if title in self._parser.CERTIFICATE_GROUPS), None)
        if certificate_parent:
            signals.append(IdentitySignal("parent_label", certificate_parent, certificate_parent))
            child = next((title for _, title in reversed(headings)
                          if title in {"Special", "Regular"}), None)
            if child:
                signals.append(IdentitySignal("table_label", child, child))
        block = blocks[table_index]
        # Only directly preceding verified labels belong to this table. Never
        # walk past CTAs, notes, prior tables, rules, or recommendation boundaries.
        if table_index:
            label = blocks[table_index - 1]
            if (label.in_recommendations == block.in_recommendations
                    and self._parser._is_table_intro(label)):
                signals.append(IdentitySignal("table_label", label.text, label.text))
                if table_index > 1:
                    parent = blocks[table_index - 2]
                    text = parent.text.strip().strip("*_ ").casefold()
                    if (parent.kind in {"paragraph", "heading"}
                            and parent.in_recommendations == block.in_recommendations
                            and text in self.PARENT_LABELS):
                        signals.append(IdentitySignal("parent_label", parent.text, parent.text))
        signals.extend(self._table_product_links(blocks, table_index))
        signals.append(IdentitySignal("source_url", evidence.source_url, evidence.source_url))
        return tuple(signals)

    def _row_identity_signals(
        self, headers: list[str], cells: list[str], row: str,
    ) -> tuple[IdentitySignal, ...]:
        # These are raw column signals, not resolved identities. A misleading
        # Loan Type containing an amount/year remains visible for validation.
        return tuple(
            IdentitySignal("product_column", cells[j], row)
            for j, header in enumerate(headers)
            if header.strip().casefold() in self.IDENTITY_COLUMN_HEADERS
            and cells[j].strip()
        )

    def extract(
        self,
        evidence: Evidence,
    ) -> list[CandidateFact]:
        if evidence.in_recommendations:
            return []

        blocks = self._parser.parse_blocks(evidence.text)

        candidates: list[CandidateFact] = []
        headings: list[tuple[int, str]] = []

        for i, block in enumerate(blocks):
            if block.in_recommendations:
                continue

            if block.kind == "heading":
                level = block.heading_level

                if level is None:
                    continue

                while (
                    headings
                    and headings[-1][0] >= level
                ):
                    headings.pop()

                headings.append((level, block.text))
                continue

            if block.kind != "table":
                continue

            lines = block.text.splitlines()

            if len(lines) < 3:
                continue

            headers = self._cells(lines[0])

            rate_columns = [
                j
                for j, header in enumerate(headers)
                if self.METRIC.search(header)
            ]
            minimum_columns = {
                j: attribute for j, header in enumerate(headers)
                if (attribute := self._minimum_attribute(header)) is not None
            }

            if not rate_columns and not minimum_columns:
                continue

            # A missing "jumbo" token is normally unknown. Set jumbo=False
            # only for a row paired with an otherwise identical Jumbo row in
            # this same table, with distinct row-linked footnotes that name
            # the two certificate variants.
            ordinary_jumbo_rows = self._proven_regular_certificate_rows(
                evidence, headers, lines[2:], rate_columns, minimum_columns,
            )

            context = self._parser.collect_table_context(
                blocks,
                i + 1,
                in_recommendations=block.in_recommendations,
            )

            # Only accepted notes are eligible for date extraction.
            # Bridges remain in the retrieval chunk but are not notes.
            notes = [
                blocks[index].text
                for index in context.accepted_note_indices
            ]

            date_notes, raw_date_notes = (
                self._parse_date_notes(notes)
            )

            heading_path = tuple(
                title for _, title in headings
            )

            # Some source tables put the product name in a header cell instead
            # of a Product/Card Type column. Preserve non-metric headers as raw
            # signals; only the normalizer decides whether an exact alias fits.
            header_signals = tuple(
                IdentitySignal("table_header", header, lines[0])
                for j, header in enumerate(headers)
                if j not in rate_columns and header.strip()
            )
            table_signals = header_signals + self._table_identity_signals(
                blocks, i, headings, evidence,
            )

            for row in lines[2:]:
                cells = self._cells(row)

                if len(cells) != len(headers):
                    continue

                conditions = {
                    f"column_{j + 1}: {header}": cells[j]
                    for j, header in enumerate(headers)
                    if j not in rate_columns
                }
                if row in ordinary_jumbo_rows:
                    conditions["jumbo"] = False

                entity = self._extract_entity(
                    evidence.source_title,
                    headers,
                    cells,
                )

                effective, date_quote, date_issues = (
                    self._resolve_date(
                        date_notes,
                        entity,
                        conditions,
                    )
                )

                identity_signals = self._row_identity_signals(headers, cells, row) + table_signals

                for j in rate_columns:
                    cell = cells[j]

                    if not self.PERCENT.search(cell):
                        continue

                    attribute, metric_issues = (
                        self._extract_attribute(
                            cell,
                            headers[j],
                        )
                    )

                    qualifiers = dict(conditions)

                    # Raw source metadata, not a normalized
                    # fact-identity qualifier.
                    qualifiers["rate_column"] = headers[j]

                    candidates.append(
                        CandidateFact(
                            entity=entity,
                            attribute=attribute,
                            qualifiers=qualifiers,
                            value=cell,
                            unit="%",
                            evidence_id=evidence.id,
                            quote=row,
                            table_header=lines[0],
                            heading_path=heading_path,
                            identity_signals=identity_signals,
                            raw_date_notes=raw_date_notes,
                            effective_from=effective,
                            effective_to=None,
                            date_quote=date_quote,
                            issues=[
                                *metric_issues,
                                *date_issues,
                            ],
                        )
                    )

                # Extract once per row/column, independently of how many rate
                # columns exist. Rate-effective dates do not establish when an
                # opening/earning requirement became effective.
                for j, attribute in minimum_columns.items():
                    cell = cells[j].strip()
                    if cell in {"", "—", "–", "-"}:
                        continue
                    candidates.append(CandidateFact(
                        entity=entity,
                        attribute=attribute,
                        qualifiers=dict(conditions),
                        value=cell,
                        unit="USD",
                        evidence_id=evidence.id,
                        quote=row,
                        table_header=lines[0],
                        heading_path=heading_path,
                        identity_signals=identity_signals,
                        raw_date_notes=raw_date_notes,
                        issues=(["effective_date_scope_uncertain"] if raw_date_notes else []),
                    ))

        return candidates

    def _proven_regular_certificate_rows(
        self,
        evidence: Evidence,
        headers: list[str],
        rows: list[str],
        rate_columns: list[int],
        minimum_columns: dict[int, str],
    ) -> set[str]:
        """Find ordinary certificate rows explicitly or structurally proven.

        Missing "Jumbo" alone is not proof. A regular row is accepted when it
        explicitly says Non-Jumbo, or when its linked special-certificate note
        is paired with a similarly labeled Jumbo row and Jumbo-specific note.
        """
        label_columns = [
            index for index, header in enumerate(headers)
            if header.strip().casefold() in self.ROW_LABEL_HEADERS
            and index not in rate_columns
            and index not in minimum_columns
        ]
        if not label_columns:
            return set()

        rows_by_base: dict[str, list[dict[str, str | bool]]] = {}
        for row in rows:
            cells = self._cells(row)
            if len(cells) != len(headers):
                continue
            label = " ".join(cells[index] for index in label_columns).strip()
            if not label:
                continue
            explicit_non_jumbo = bool(
                re.search(r"\bnon[- ]jumbo\b", label, re.I)
            )
            is_jumbo = (
                not explicit_non_jumbo
                and bool(re.search(r"\bjumbo\b", label, re.I))
            )

            base_label = re.sub(
                r"\b(?:non[- ]jumbo|jumbo)\b",
                "",
                label,
                flags=re.I,
            )
            base_label = self._normalize_name(base_label)
            if not base_label:
                continue

            linked_notes = self.rate_disclosure_quotes(evidence, row)[1:]
            note_labels = [
                self._rate_note_body(note).split(":", 1)[0].casefold()
                for note in linked_notes
            ]
            special_labels = [
                note_label for note_label in note_labels
                if "certificate" in note_label and "special" in note_label
            ]
            has_jumbo_note = any(
                re.search(r"\bjumbo\b", note_label)
                and not re.search(r"\bnon[- ]jumbo\b", note_label)
                for note_label in special_labels
            )
            has_regular_note = any(
                not re.search(r"\bjumbo\b", note_label)
                for note_label in special_labels
            )
            rows_by_base.setdefault(base_label, []).append({
                "row": row,
                "is_jumbo": is_jumbo,
                "explicit_non_jumbo": explicit_non_jumbo,
                "has_jumbo_note": has_jumbo_note,
                "has_regular_note": has_regular_note,
            })

        proven: set[str] = set()
        for group in rows_by_base.values():
            has_verified_jumbo = any(
                item["is_jumbo"] and item["has_jumbo_note"]
                for item in group
            )
            for item in group:
                if item["explicit_non_jumbo"]:
                    proven.add(str(item["row"]))
                elif (
                    not item["is_jumbo"]
                    and item["has_regular_note"]
                    and has_verified_jumbo
                ):
                    proven.add(str(item["row"]))

        return proven

    def rate_disclosure_quotes(
        self, evidence: Evidence, row_quote: str,
    ) -> tuple[str, ...]:
        """Return one rate row and only the footnotes explicitly linked to it.

        Rate rows can carry numeric, bracketed, parenthesized, or typographic
        footnote markers. Unmarked table notes are intentionally not broadcast
        to every row: they may still be used by other extraction stages, but
        they do not prove a row-specific transaction or promotional condition.
        """
        row_markers = self._rate_row_markers(row_quote)
        if not row_markers:
            return ()

        blocks = self._parser.parse_blocks(evidence.text)
        table_index = next((i for i, block in enumerate(blocks)
                            if block.kind == "table" and row_quote in block.text.splitlines()), None)
        if table_index is None:
            return ()
        table = blocks[table_index]
        table_rates = {
            re.sub(r"\s+", "", value).casefold()
            for value in re.findall(r"\d[\d,]*(?:\.\d+)?\s*%", table.text)
        }
        linked = []
        end = min(len(blocks), table_index + 11)
        for index in range(table_index + 1, end):
            block = blocks[index]
            if (block.kind not in {"paragraph", "list"}
                    or block.in_recommendations != table.in_recommendations):
                break
            note = blocks[index].text
            marker = self._rate_note_marker(note)
            if marker in row_markers:
                linked.append(note)
                continue
            # Other marked footnotes and known table-wide disclosures may sit
            # between this row's marker and its note. They are traversed but
            # never attached unless their marker matches the row.
            plain = note.strip()
            for wrapper in ("**", "__", "_"):
                if plain.startswith(wrapper) and plain.endswith(wrapper) and len(plain) > 2 * len(wrapper):
                    plain = plain[len(wrapper):-len(wrapper)].strip()
                    break
            plain = re.sub(
                r"^\s*(?:\d{1,2}|\[\d{1,2}\]|\(\d{1,2}\)|(?:\\?\*){1,3}|[†‡¹²³⁴⁵⁶⁷⁸⁹])",
                "", plain,
            ).strip()
            if marker is not None or self._parser._is_table_note(plain):
                continue
            # Some exports emit an unmarked copy immediately before its
            # numbered footnote. Treat only an exact adjacent duplicate as a
            # bridge; the marker-bearing copy remains the actual support.
            next_index = index + 1
            if next_index < end:
                following = blocks[next_index]
                if (following.kind in {"paragraph", "list"}
                        and following.in_recommendations == table.in_recommendations
                        and self._rate_note_marker(following.text) is not None
                        and self._rate_note_body(note) == self._rate_note_body(following.text)):
                    continue
            if (self._parser.TABLE_PAYMENT_EXAMPLE.match(note.strip())
                    or self._parser.TABLE_APPLICATION_LINK.fullmatch(note.strip())):
                continue
            if (re.match(r"^(?:if you borrow\b|\*?payment example\b)", note.strip(), re.I)
                    and any(re.sub(r"\s+", "", rate).casefold() in
                            re.sub(r"\s+", "", note).casefold() for rate in table_rates)):
                # Traverse an explicit payment example only when it repeats a
                # rate already present in this table. It is a bridge, never a
                # supporting disclosure for the candidate.
                continue
            break
        return (row_quote, *dict.fromkeys(linked)) if linked else ()

    @staticmethod
    def _rate_row_markers(row: str) -> frozenset[str]:
        """Read only explicit trailing footnote syntax from a Markdown row."""
        markers: set[str] = set()
        cells = re.split(r"(?<!\\)\|", row.strip().strip("|"))
        for cell in cells:
            value = cell.strip()
            for pattern in (
                r"(?<=%)\s*(\d{1,2})\s*$",
                r"(?<=[A-Za-z)])(\d{1,2})\s*$",
                r"\[(\d{1,2})\]\s*$",
                r"\((\d{1,2})\)\s*$",
                r"((?:\\\*){1,3}|\*{1,3}|[†‡])\s*$",
                r"([¹²³⁴⁵⁶⁷⁸⁹])\s*$",
            ):
                match = re.search(pattern, value)
                if match:
                    marker = match.group(1).replace("\\", "")
                    markers.add(marker.translate(str.maketrans("¹²³⁴⁵⁶⁷⁸⁹", "123456789")))
                    break
        return frozenset(markers)

    @staticmethod
    def _rate_note_marker(note: str) -> str | None:
        text = note.strip()
        for wrapper in ("__", "_", "**"):
            if text.startswith(wrapper) and text.endswith(wrapper) and len(text) > 2 * len(wrapper):
                text = text[len(wrapper):-len(wrapper)].strip()
                break
        text = re.sub(r"\\(\*)", r"\1", text)
        for pattern in (
            r"^(\d{1,2})(?=\D)",
            r"^\[(\d{1,2})\]",
            r"^\((\d{1,2})\)",
            r"^(\*{1,3}|[†‡])(?=\S)",
            r"^([¹²³⁴⁵⁶⁷⁸⁹])",
        ):
            match = re.match(pattern, text)
            if match:
                return match.group(1).translate(str.maketrans("¹²³⁴⁵⁶⁷⁸⁹", "123456789"))
        return None

    @staticmethod
    def _rate_note_body(note: str) -> str:
        text = note.strip()
        for wrapper in ("__", "_", "**"):
            if text.startswith(wrapper) and text.endswith(wrapper) and len(text) > 2 * len(wrapper):
                text = text[len(wrapper):-len(wrapper)].strip()
                break
        text = re.sub(r"\\(\*)", r"\1", text)
        text = re.sub(
            r"^(?:\d{1,2}|\[\d{1,2}\]|\(\d{1,2}\)|\*{1,3}|[†‡¹²³⁴⁵⁶⁷⁸⁹])",
            "", text,
        )
        return " ".join(text.casefold().split())
