"""Snapshot-scoped, source-grounded location and branch-hours retrieval.

This index retrieves entity-specific evidence. It does not infer that a newer
announcement supersedes an older one, or that examples establish universal hours.
"""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import re
from typing import Literal
from uuid import UUID

from ingestion.chunker import EvidenceChunker
from ingestion.location_cards import recognize_location_card
from ingestion.location_context import BranchHoursRecognizer, LocationRecognizer
from ingestion.metadata_extractor import MetadataExtractor
from foundation.store import KnowledgeStore
from retrieval.answer_selector import snapshot_scope, stable_unique
from retrieval.query_scope import ResultShape, parse_result_shape


Field = Literal['address', 'lobby_hours', 'drive_up_hours', 'hours']
Scope = Literal['specific', 'general', 'all', 'comparison']


@dataclass(frozen=True)
class LocationRecord:
    evidence_id: UUID
    name: str
    kind: Literal['branch', 'atm']
    address: str | None
    city: str | None
    state: str | None
    postal_code: str | None
    location_quote: str
    lobby_hours: tuple[str, ...] = ()
    drive_up_hours: tuple[str, ...] = ()
    published_on: date | None = None
    # A stable identity is supplied only when supported by structured source data.
    location_id: str | None = None


@dataclass(frozen=True)
class LocationQuery:
    label: str | None
    field: Field
    kind: Literal['branch', 'atm'] | None
    scope: Scope = 'specific'
    result_shape: ResultShape = ResultShape()


@dataclass(frozen=True)
class LocationLookup:
    query: LocationQuery
    status: Literal['match', 'ambiguous', 'not_found']
    records: tuple[LocationRecord, ...]
    snapshot_id: str | None
    reason: str


_ABBREVIATIONS = {
    'ln': 'lane', 'rd': 'road', 'ave': 'avenue', 'blvd': 'boulevard',
    'st': 'street', 'dr': 'drive', 'e': 'east', 'w': 'west',
    'n': 'north', 's': 'south', 'illinois': 'il', 'iowa': 'ia',
    'wisconsin': 'wi',
}
_GENERIC = {
    'dupaco', 'all', 'your', 'my', 'our', 'this', 'that',
    'closest', 'nearest', 'near me', 'all branches', 'each',
    'every', 'branches', 'branch', 'atm',
}


def _normalize(text: str) -> str:
    return ' '.join(_ABBREVIATIONS.get(t, t)
                    for t in re.findall(r'[a-z0-9]+', text.casefold()))


def _entity_id(kind: str, *, address: str) -> str:
    """A full structured locator address establishes this source identity.

    Announcement names/cities/partial streets never enter this function.
    """
    return f'{kind}:address:{_normalize(address)}'


def _field_value(record: LocationRecord, field: Field):
    if field == 'address':
        return _normalize(record.address) if record.address else None
    if field == 'lobby_hours':
        return record.lobby_hours or None
    if field == 'drive_up_hours':
        return record.drive_up_hours or None
    return (record.lobby_hours, record.drive_up_hours) if (
        record.lobby_hours or record.drive_up_hours) else None


def _collection_value_key(value):
    """Canonicalize returned values for duplicate reduction in broad queries."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return tuple(_collection_value_key(item) for item in value)
    if isinstance(value, str):
        return re.sub(r'\s+', ' ', re.sub(r'\s*[-–—]\s*', '-', value.casefold())).strip()
    return value


def _select_collection_representatives(query: LocationQuery, records):
    """Reduce broad, unlabelled collections to the latest example per value.

    Explicitly named comparisons and full-list requests retain entity-level
    rows because dropping equal-valued entities would hide requested coverage.
    """
    if (query.result_shape.kind != 'representative_comparison'
            or query.label is not None):
        return records
    return stable_unique(
        records,
        key=lambda row: _collection_value_key(_field_value(row, query.field)),
    )


def _matches_label(record: LocationRecord, label: str) -> bool:
    needle = _normalize(label)
    if not needle:
        return False
    # Match only structured entity attributes, never arbitrary announcement prose.
    aliases = (record.name, record.city or '', record.address or '')
    return any(f' {needle} ' in f' {_normalize(alias)} ' for alias in aliases)


class LocationIndex:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self._by_snapshot: dict[str, list[LocationRecord]] = {}
        self._built = False

    def identity_records(self) -> tuple[LocationRecord, ...]:
        """Verified address-backed identities across all indexed snapshots.

        This historical view establishes identity only, never current availability.
        """
        if not self._built:
            self.build_index()
        return tuple(r for records in self._by_snapshot.values() for r in records
                     if r.location_id and r.address)

    def build_index(self) -> None:
        staged: dict[str, list[LocationRecord]] = defaultdict(list)
        parser = EvidenceChunker()
        metadata_extractor = MetadataExtractor()
        location_recognizer = LocationRecognizer()

        for evidence in self.store.evidence.values():
            if evidence.in_recommendations:
                continue
            text = evidence.text
            if not any(marker in text for marker in (
                'Dupaco Branch', 'Surcharge-Free ATM', 'Branch hours are:'
            )):
                continue
            blocks = parser.parse_blocks(text)
            publication = re.search(
                r'^Document published on: (\d{4}-\d{2}-\d{2})$', text, re.M)
            if publication:
                published_on = date.fromisoformat(publication.group(1))
            else:
                published_on = metadata_extractor.extract(
                    blocks, page_title=evidence.source_title,
                    source_url=evidence.source_url).published_on

            i = 0
            while i < len(blocks):
                card = recognize_location_card(blocks, i, len(blocks))
                if card is None:
                    i += 1
                    continue
                name, _, address = blocks[i:i + 3]
                parts = [p.strip() for p in address.text.rsplit(',', 3)]
                if len(parts) != 4:
                    i = card.end
                    continue
                _, city, state, postal = parts
                staged[evidence.snapshot_id].append(LocationRecord(
                    evidence_id=evidence.id,
                    name=name.text,
                    kind=card.subtype,
                    address=address.text,
                    city=city,
                    state=state,
                    postal_code=postal,
                    location_quote='\n\n'.join(b.text for b in blocks[i:card.end]),
                    published_on=published_on,
                    location_id=_entity_id(card.subtype, address=address.text),
                ))
                i = card.end

            # Context markers are only accepted if unique; do not bind a
            # schedule to an arbitrary location mentioned elsewhere in a page.
            context_sources = [
                b.text[len('Location source:\n'):]
                for b in blocks
                if b.kind == 'paragraph'
                and b.text.startswith('Location source:\n')
            ]
            recognized = location_recognizer.recognize(blocks)
            regions = BranchHoursRecognizer.recognize(blocks)
            for region in regions:
                sources = (context_sources if len(regions) == 1 else []) or [
                    c.quote for c in recognized if c.source_block == region.start
                ]
                if len(sources) != 1:
                    continue
                quote = sources[0]
                city_match = re.match(r'^([A-Z][A-Z ]+),\s*(Iowa|Wis\.)[–—]', quote)
                city = city_match.group(1).title() if city_match else None
                state = ({'Iowa': 'IA', 'Wis.': 'WI'}.get(city_match.group(2))
                         if city_match else None)
                named = re.match(r'^The (.+?) Dupaco branch is located at ', quote)
                name = city or (named.group(1).strip() if named else '')
                # Extract only from the uniquely bound location quote. Failure
                # to parse an address/dateline must not discard its schedule.
                street_match = re.search(
                    r"\blocated at (\d[\w .’'#-]*?\b(?:Highway\s+\d+|Road|Rd\.?|Street|St\.?|"
                    r"Avenue|Ave\.?|Lane|Ln\.?|Boulevard|Blvd\.?|Drive|Dr\.?|Court|Ct\.?))(?=\s|[,.;]|$)",
                    quote)
                address = street_match.group(1).strip() if street_match else None
                schedules: dict[str, list[str]] = {
                    'lobby': [], 'drive_up': []}
                active: str | None = None
                for block in blocks[region.start + 1:region.end]:
                    label = BranchHoursRecognizer._LABEL.fullmatch(block.text)
                    if label:
                        if re.search(r'\*\*Lobby\*\*', block.text, re.I):
                            active = 'lobby'
                        elif re.search(r'\*\*Drive-up\*\*', block.text, re.I):
                            active = 'drive_up'
                        else:
                            active = None
                        if active and label.group(1):
                            schedules[active].append(label.group(1))
                    elif active:
                        schedules[active].append(block.text)

                if not (schedules['lobby'] or schedules['drive_up']):
                    continue
                # City-only evidence can represent multiple branches. Leave
                # identity unresolved rather than pretending city is unique.
                staged[evidence.snapshot_id].append(LocationRecord(
                    evidence_id=evidence.id,
                    name=name,
                    kind='branch',
                    address=address,
                    city=city,
                    state=state,
                    postal_code=None,
                    location_quote=quote,
                    lobby_hours=tuple(schedules['lobby']),
                    drive_up_hours=tuple(schedules['drive_up']),
                    published_on=published_on,
                    location_id=None,
                ))

        self._by_snapshot = dict(staged)
        self._built = True

    @staticmethod
    def parse_query(question: str) -> LocationQuery | None:
        text = question.strip().rstrip('?.! ').replace("Dupaco's", 'Dupaco').replace("dupaco's", 'dupaco')
        lowered = text.casefold()
        if re.search(r'\b(address|located|where)\b', lowered):
            field: Field = 'address'
        elif re.search(r'\bhours?\b', lowered) or (
            'what time' in lowered and re.search(r'\b(open|close)\b', lowered)
        ):
            field = ('drive_up_hours' if re.search(r'drive[- ](?:up|through)', lowered)
                     else 'lobby_hours' if 'lobby' in lowered else 'hours')
        else:
            return None
        kind = ('atm' if re.search(r'\batms?\b', lowered)
                else 'branch' if re.search(r'\bbranch(?:es)?\b', lowered) else None)
        comparison = bool(re.search(r'\b(same|different|vary|varies|consistent|identical|uniform|compare)\b', lowered))
        all_requested = bool(re.search(r'\b(all|every|each)\b', lowered))
        result_shape = parse_result_shape(question)
        # Fully generic templates contain no residual entity constraint.
        generic = re.fullmatch(
            r"(?:what are|what is|tell me|show me|give me) (?:dupaco |your |the |our )?"
            r"(?:general |typical )?(?:lobby |drive[- ]up |branch |atm )?hours", lowered)
        generic_comparison = re.fullmatch(
            r'(?:do|are|does) (?:all |every |each )?(?:dupaco )?(?:branches|atms) '
            r'(?:keep |have |use )?(?:the )?(?:same|different|identical|uniform) '
            r'(?:lobby |drive[- ]up )?hours', lowered)
        if generic or generic_comparison:
            scope = ('comparison' if comparison else
                     'all' if all_requested else 'general')
            if scope in {'general', 'comparison'} and result_shape.kind == 'single_value':
                result_shape = ResultShape('representative_comparison')
            return LocationQuery(None, field, kind or 'branch', scope, result_shape)
        # Extract named scope before considering a collective fallback. Unknown
        # names still stay specific; they never become a different branch.
        patterns = (
            r'^(?:are|do|does|compare)\s+(?:the\s+)?(.+?)\s+(?:branches|atms)\b',
            r'\b(?:for|of|at|about)\s+(?:the\s+)?(.+?)\s+(?:branch|atm)\b',
            r"\b(?:where is|where's|find|locate)\s+(?:the\s+)?(.+?)\s+(?:branch|atm)\b",
            r'\bwhat time\s+(?:does|do|is)\s+(?:the\s+)?(.+?)\s+(?:branch|atm)\b',
            r'^(?:what are|what is|tell me|show me|give me|compare)\s+(?:the\s+)?(.+?)\s+(?:(?:branch|atm)\s+)?(?:lobby\s+|drive[- ]up\s+|opening\s+)?hours\b',
            r'^(?:the\s+)?(?!what\b|when\b|how\b|is\b|do\b|does\b|can\b|where\b)(.+?)\s+(?:branch|atm)\b',
            r'\b(?:hours|address)\b.*?\b(?:in|at|for)\s+(?:the\s+)?([^?]+)$',
        )
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if not match:
                continue
            label = match.group(1).strip()
            label = re.sub(r'^(?:(?:the|all|every|each|general)\s+)+', '', label, flags=re.I)
            label = re.sub(r'\s*(?:branches|branch|atms|atm)$', '', label, flags=re.I).strip()
            if not label or _normalize(label) in _GENERIC or label.casefold() in {'general', 'typically', 'usually'}:
                continue
            if re.search(r'\b(?:nearest|closest|near me)\b', label, re.I):
                return None  # No proximity resolver; do not broaden scope.
            if comparison:
                scope: Scope = 'comparison'
            elif all_requested:
                scope = 'all'
            else:
                scope = 'specific'
            return LocationQuery(label, field, kind, scope, result_shape)
        addresses = re.fullmatch(
            r'(?:where are|show me|list) (?:all |every |each )(?:dupaco )?(?:branches|atms)(?: located| addresses)?', lowered)
        if addresses:
            return LocationQuery(None, field, kind or 'branch', 'all',
                                 ResultShape('full_collection'))
        return None

    @staticmethod
    def _labels(query: LocationQuery) -> tuple[str, ...]:
        if query.label is None:
            return ()
        # Multiple explicitly named locations are retained as a query subset.
        return tuple(part.strip() for part in re.split(r'\s+and\s+|,\s*', query.label, flags=re.I)
                     if part.strip()) if query.scope == 'comparison' else (query.label,)

    def _snapshot(self, as_of: str | None):
        scope = snapshot_scope(self.store, as_of)
        if scope is None:
            return None, None
        # Snapshot observation time is not the publication/effective date.
        return scope.snapshot, scope.reference_date

    def _eligible_records(self, snapshot_id: str, reference: date):
        return [r for r in self._by_snapshot.get(snapshot_id, ())
                if r.published_on is None or r.published_on <= reference]


    def _lookup_collection(self, query: LocationQuery, *, as_of: str | None = None) -> LocationLookup:
        if not self._built:
            self.build_index()
        snapshot, reference = self._snapshot(as_of)
        if snapshot is None:
            return LocationLookup(query, 'not_found', (), None, 'No eligible snapshot.')
        rows = [r for r in self._eligible_records(snapshot.snapshot_id, reference)
                if (query.kind is None or r.kind == query.kind) and _field_value(r, query.field) is not None]
        labels = self._labels(query)
        if labels:
            missing = [label for label in labels if not any(_matches_label(r, label) for r in rows)]
            rows = [r for r in rows if any(_matches_label(r, label) for label in labels)]
            if missing:
                return LocationLookup(query, 'not_found', tuple(rows), snapshot.snapshot_id,
                                      'No requested-field evidence for: ' + ', '.join(missing))
        rows.sort(key=lambda r: (r.published_on or date.min,
                                 self.store.evidence[r.evidence_id].source_url,
                                 r.location_quote, str(r.evidence_id)), reverse=True)
        # Preserve every source occurrence and historical disagreement.
        unique = stable_unique(
            rows,
            key=lambda row: (row.evidence_id, row.location_quote, _field_value(row, query.field)),
        )
        # Rows are sorted newest-first above. Broad, unlabelled collections
        # retain one representative per distinct returned value; explicit
        # comparisons and full-list requests retain their requested entities.
        unique = _select_collection_representatives(query, unique)
        return LocationLookup(query, 'match' if unique else 'not_found', tuple(unique), snapshot.snapshot_id,
                              'Source-specific records only; broad collections retain the newest published '
                              'example per distinct value. This does not establish current validity, '
                              'completeness, simultaneous applicability, or supersession.')

    def lookup(self, question: str, *, as_of: str | None = None) -> LocationLookup | None:
        query = self.parse_query(question)
        if query is None:
            return None
        if query.scope != 'specific':
            return self._lookup_collection(query, as_of=as_of)
        if not self._built:
            self.build_index()
        snapshot, reference = self._snapshot(as_of)
        if snapshot is None:
            return LocationLookup(query, 'not_found', (), None, 'No eligible snapshot.')

        candidates = [r for r in self._eligible_records(snapshot.snapshot_id, reference)
                      if (query.kind is None or r.kind == query.kind)
                      and query.label is not None and _matches_label(r, query.label)]
        relevant = [r for r in candidates if _field_value(r, query.field) is not None]
        relevant.sort(key=lambda r: (
            self.store.evidence[r.evidence_id].source_url, str(r.evidence_id)))
        if not relevant:
            return LocationLookup(query, 'not_found', tuple(candidates), snapshot.snapshot_id,
                                  'No matching entity evidence establishes the requested field.')

        # Address-backed records are identified by address. City-only schedules
        # remain provisional: they cannot be equated with an address-backed
        # branch or with each other just because the city matches.
        ids = {r.location_id for r in relevant if r.location_id is not None}
        unresolved = [r for r in relevant if r.location_id is None]
        if len(ids) > 1:
            return LocationLookup(query, 'ambiguous', tuple(relevant), snapshot.snapshot_id,
                                  'Multiple distinct source-backed locations match.')
        if unresolved and (ids or len({r.evidence_id for r in unresolved}) > 1):
            return LocationLookup(query, 'ambiguous', tuple(relevant), snapshot.snapshot_id,
                                  'Cannot prove city-only schedule belongs to a unique branch.')
        if not ids and unresolved:
            # One source can establish its own hours, but not a globally unique
            # branch identity. The caller must retain the source qualification.
            values = {_field_value(r, query.field) for r in relevant}
            return LocationLookup(query, 'match' if len(values) == 1 else 'ambiguous',
                                  tuple(relevant), snapshot.snapshot_id,
                                  'Source-specific city schedule; branch identity not fully verified.')

        values = {_field_value(r, query.field) for r in relevant}
        status = 'match' if len(values) == 1 else 'ambiguous'
        return LocationLookup(query, status, tuple(relevant), snapshot.snapshot_id,
                              'Source-backed entity and field match.' if status == 'match'
                              else 'Sources disagree; no supersession inferred.')
