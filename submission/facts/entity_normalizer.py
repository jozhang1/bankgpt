"""Resolve product context and source-bound semantic subjects conservatively."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
import re
from typing import Literal
from urllib.parse import urlsplit

from foundation.contract import Evidence

from ingestion.extractor import CandidateFact, IdentitySignal
from foundation.products import Product
from foundation.entities import EntityDefinition, EntityKind, ENTITY_KINDS, entity_label
from ingestion.subject_scope import contains_label, subject_scope_issue
from facts.semantic_fact_profile import SEMANTIC_FACT_PROFILES


def normalize_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


@dataclass(frozen=True)
class EntityResolution:
    """Canonical identity for products and non-product semantic subjects alike."""
    entity_id: str | None
    status: Literal["resolved", "ambiguous", "unresolved"]
    candidate_ids: tuple[str, ...] = ()
    source: str | None = None
    matched_label: str | None = None
    segment: str | None = None
    reason: str | None = None
    entity_kind: EntityKind | None = None


class EntityNormalizer:
    def __init__(self, products: Iterable[Product], *, entities: Iterable[EntityDefinition] = ()):
        self.by_alias: dict[str, list[Product]] = defaultdict(list)
        self.products: dict[str, Product] = {}
        seen_ids: set[str] = set()
        for product in products:
            if product.id in seen_ids:
                raise ValueError(f"Duplicate product ID: {product.id}")
            seen_ids.add(product.id)
            self.products[product.id] = product
            # Different aliases can normalize to the same key. Index each
            # product once per key so typography does not create ambiguity.
            keys = {normalize_label(a) for a in (product.name, *product.aliases)}
            for key in sorted(keys):
                if key:
                    self.by_alias[key].append(product)
        for matches in self.by_alias.values():
            matches.sort(key=lambda product: product.id)

        self.entities: dict[str, EntityDefinition] = {}
        self.by_entity_alias: dict[str, list[EntityDefinition]] = defaultdict(list)
        for entity in entities:
            if entity.id in self.products or entity.id in self.entities:
                raise ValueError(f"Duplicate entity/catalog ID: {entity.id}")
            self.entities[entity.id] = entity
            for key in {entity_label(label) for label in (entity.name, *entity.aliases)}:
                self.by_entity_alias[key].append(entity)
        for matches in self.by_entity_alias.values():
            matches.sort(key=lambda entity: entity.id)

    def _lookup(
        self, label: str, segment: str | None = None,
    ) -> list[Product]:
        matches = self.by_alias.get(normalize_label(label), [])
        return [p for p in matches if segment is None or p.segment in {segment, "shared"}]

    @staticmethod
    def _url_segment(source_url: str | None) -> str | None:
        path = urlsplit(source_url or "").path.rstrip("/")
        if path == "/business-rates" or path == "/business" or path.startswith("/business/"):
            return "business"
        if path == "/personal" or path.startswith("/personal/"):
            return "personal"
        return None

    @classmethod
    def _product_link_segment(cls, url: str) -> str | None:
        """Catalog attribution only; it does not describe customer eligibility."""
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname != "www.dupaco.com":
            return None
        return cls._url_segment(url)

    def _ancestor(self, parent: str, child: str) -> bool:
        visited: set[str] = set()
        while child in self.products and child not in visited:
            if child == parent:
                return True
            visited.add(child)
            child = self.products[child].parent_id
        return False

    def _scoped_table_matches(
        self, label: str, parent_labels: tuple[str, ...],
    ) -> list[Product]:
        """Resolve bare Regular/Special only inside a verified certificate family.

        Certificates is a contextual family label, never a global product alias.
        Terms or rates cannot supply missing parent context.
        """
        key = normalize_label(label)
        if key not in {"regular", "special"}:
            return []
        parents: set[str] = set()
        for parent in parent_labels:
            parent_key = normalize_label(parent)
            if parent_key == "term share certificate rates":
                if "personal_certificates" in self.products:
                    parents.add("personal_certificates")
            elif parent_key == "business term share certificate rates":
                if "business_certificates" in self.products:
                    parents.add("business_certificates")
            elif parent_key == "certificates":
                parents.update(pid for pid in ("personal_certificates", "business_certificates")
                               if pid in self.products)
            else:
                parents.update(p.id for p in self._lookup(parent) if p.kind == "family")
        return [p for p in self.products.values()
                if p.parent_id in parents and normalize_label(p.name).split()[0] == key
                and p.category == "certificate"]

    def normalize(
        self, candidate: CandidateFact, *, segment: str | None = None,
        evidence: Evidence | None = None, source_url: str | None = None,
        table_label: str | None = None, entity_kind: EntityKind | None = None,
        use_profile_entity_kind: bool = False,
    ) -> EntityResolution:
        """Keep legacy table/product resolution; explicitly resolve semantic subjects.

        A subject IdentitySignal selects the semantic path, never a heading-based
        fallback. Without a subject signal, an exact supplied non-product identity
        or an explicit entity_kind also selects it. Unknown names are preserved.
        """
        if entity_kind is not None and entity_kind not in ENTITY_KINDS:
            raise ValueError("Unknown entity kind")
        if segment not in {None, "personal", "business", "shared"}:
            raise ValueError("Unknown product segment")
        # Fact ingestion may use a uniquely typed predicate to disambiguate an
        # exact label shared across entity kinds (e.g. an organization name
        # repeated as the name on branch cards). Keep the general resolver
        # conservative by making this opt-in, and only apply it when the exact
        # label actually collides across allowed and disallowed kinds. Multiple
        # same-kind matches and multi-kind profiles remain ambiguous.
        if use_profile_entity_kind and entity_kind is None:
            profile = SEMANTIC_FACT_PROFILES.get(candidate.attribute)
            allowed_kinds = tuple(profile.get("entity_kinds", ())) if profile else ()
            if len(allowed_kinds) == 1:
                expected_kind = allowed_kinds[0]
                labels = [candidate.entity]
                labels.extend(signal.label for signal in candidate.identity_signals
                              if signal.kind == "subject")
                for label in dict.fromkeys(labels):
                    matches = self._subject_matches(label, None)
                    kinds = {self._entity_kind(record) for record in matches}
                    if expected_kind in kinds and kinds - {expected_kind}:
                        entity_kind = expected_kind
                        break
        if evidence is not None:
            if evidence.id != candidate.evidence_id:
                raise ValueError("Candidate and Evidence IDs differ")
            if source_url is not None and source_url != evidence.source_url:
                raise ValueError("source_url disagrees with Evidence")
        semantic = (entity_kind is not None
                    or any(s.kind == "subject" for s in candidate.identity_signals)
                    or candidate.entity in self.entities
                    or bool(self.by_entity_alias.get(entity_label(candidate.entity))))
        if semantic:
            return self._normalize_subject(candidate, evidence=evidence, entity_kind=entity_kind,
                                           segment=segment, source_url=source_url)
        return self._normalize_product(candidate, segment=segment, evidence=evidence,
                                       source_url=source_url, table_label=table_label)

    def _subject_matches(self, label: str, kind: EntityKind | None):
        # IDs are accepted only if already present in the supplied definitions.
        named = ([self.entities[label]] if label in self.entities
                 else self.by_entity_alias.get(entity_label(label), []))
        catalog = ([self.products[label]] if label in self.products else self._lookup(label))
        return [record for record in (*named, *catalog)
                if kind is None or self._entity_kind(record) == kind]

    def resolve_reference(self, label: str, quote: str, evidence: Evidence) -> str | None:
        """Resolve a source-quoted entity mention to one known canonical ID."""
        if (not label.strip() or not quote.strip() or quote not in evidence.text
                or not contains_label(quote, label)):
            return None
        matches = self._subject_matches(label, None)
        bound = [record.id for record in matches
                 if any(contains_label(quote, alias)
                        for alias in (record.name, *record.aliases))]
        unique = tuple(sorted(set(bound)))
        return unique[0] if len(unique) == 1 else None

    @staticmethod
    def _entity_kind(record) -> EntityKind:
        if isinstance(record, Product):
            return record.kind if record.kind in {"service", "program"} else "product"
        return record.kind

    def _normalize_subject(
        self,
        candidate,
        *,
        evidence,
        entity_kind,
        segment,
        source_url,
    ):
        def result(
            status,
            reason,
            records=(),
            chosen=None,
            label=None,
            source="subject",
        ):
            return EntityResolution(
                chosen.id if chosen else None,
                status,
                tuple(sorted({r.id for r in records})),
                source,
                label,
                segment=None,
                reason=reason,
                entity_kind=(
                    self._entity_kind(chosen)
                    if chosen is not None
                    else entity_kind
                ),
            )

        # ---------------------------------------------------------
        # 1. Validate source evidence
        # ---------------------------------------------------------

        if evidence is None:
            return result("unresolved", "subject_evidence_required")

        if evidence.in_recommendations:
            return result(
                "unresolved",
                "recommendation_subject_not_admitted",
            )

        if (
            not candidate.quote.strip()
            or candidate.quote not in evidence.text
        ):
            return result(
                "unresolved",
                "candidate_quote_not_in_evidence",
            )

        signals = [
            signal
            for signal in candidate.identity_signals
            if signal.kind == "subject"
        ]

        # Every explicit subject signal must be source-backed.
        for signal in signals:
            if (
                not signal.quote.strip()
                or signal.quote not in evidence.text
            ):
                return result(
                    "unresolved",
                    "subject_signal_not_source_bound",
                )

            matches = self._subject_matches(
                signal.label,
                entity_kind,
            )

            labels = {signal.label}

            for record in matches:
                labels.update((record.name, *record.aliases))

            if not any(
                contains_label(signal.quote, label)
                for label in labels
            ):
                return result(
                    "unresolved",
                    "subject_signal_not_source_bound",
                )

        # ---------------------------------------------------------
        # 2. Validate source URL consistency
        # ---------------------------------------------------------

        urls = {
            signal.label
            for signal in candidate.identity_signals
            if signal.kind == "source_url"
        }

        if urls - {evidence.source_url}:
            return result(
                "ambiguous",
                "source_url_conflict",
            )

        # ---------------------------------------------------------
        # 3. Resolve explicit subject labels
        # ---------------------------------------------------------

        labels = [
            (signal.label, "subject")
            for signal in signals
        ]

        if not labels:
            labels = [(candidate.entity, "entity")]

        elif self._subject_matches(candidate.entity, None):
            labels.append((candidate.entity, "entity"))

        pools = []
        records = {}

        for label, source in labels:
            matches = self._subject_matches(
                label,
                entity_kind,
            )

            if not matches:
                return result(
                    "unresolved",
                    "no_exact_entity_match",
                    records.values(),
                    label=label,
                    source=source,
                )

            # A catalog match is insufficient unless its name
            # or alias is supported by the supplied source quotes.
            quotes = [
                candidate.quote,
                *(signal.quote for signal in signals),
            ]

            bound = []

            for record in matches:
                aliases = {
                    entity_label(name)
                    for name in (record.name, *record.aliases)
                }

                if any(
                    contains_label(quote, alias)
                    for alias in aliases
                    for quote in quotes
                ):
                    bound.append(record)

            if not bound:
                return result(
                    "unresolved",
                    "entity_label_not_in_subject_quotes",
                    matches,
                    label=label,
                    source=source,
                )

            pools.append({record.id for record in bound})

            records.update(
                (record.id, record)
                for record in bound
            )

        if not pools:
            return result(
                "unresolved",
                "subject_identity_missing",
            )

        # ---------------------------------------------------------
        # 4. Product-specific resolution
        # ---------------------------------------------------------

        if records and all(
            isinstance(record, Product)
            for record in records.values()
        ):
            # Preserve the existing product resolver because it
            # handles personal/business homonyms and product-family
            # compatibility.
            #
            # Product-column signals are synthetic compatibility
            # inputs, not proof of table-column provenance.
            # The original subject signals remain available to
            # _check_subject_scope() for locality validation.

            product_signals = tuple(
                IdentitySignal(
                    "product_column",
                    (
                        self.products[label].name
                        if label in self.products
                        else label
                    ),
                    candidate.quote,
                )
                for label, _ in labels
            )

            product_signals += tuple(
                signal
                for signal in candidate.identity_signals
                if signal.kind in {
                    "source_url",
                    "product_link",
                }
            )

            product_candidate = replace(
                candidate,
                entity="",
                identity_signals=product_signals,
            )

            resolved = self._normalize_product(
                product_candidate,
                segment=segment,
                evidence=evidence,
                source_url=source_url,
            )

            if resolved.entity_id is None:
                return replace(
                    resolved,
                    entity_kind=entity_kind,
                    source=labels[0][1],
                    matched_label=labels[0][0],
                )

            chosen = self.products[resolved.entity_id]

            if (
                entity_kind is not None
                and self._entity_kind(chosen) != entity_kind
            ):
                return result(
                    "ambiguous",
                    "entity_kind_context_conflict",
                    records.values(),
                )

            resolved = replace(
                resolved,
                entity_kind=self._entity_kind(chosen),
                source=labels[0][1],
                matched_label=labels[0][0],
            )

            return self._check_subject_scope(
                resolved,
                chosen,
                candidate,
                evidence,
                signals,
            )

        # ---------------------------------------------------------
        # 5. Resolve non-product identity
        # ---------------------------------------------------------

        viable = set.intersection(*pools)

        if not viable:
            return result(
                "ambiguous",
                "subject_identity_conflict",
                records.values(),
            )

        if len(viable) != 1:
            return result(
                "ambiguous",
                "multiple_exact_entity_matches",
                [
                    records[identity]
                    for identity in viable
                ],
            )

        chosen = records[next(iter(viable))]

        resolved = result(
            "resolved",
            None,
            (chosen,),
            chosen,
            labels[0][0],
            labels[0][1],
        )

        # ---------------------------------------------------------
        # 6. Validate subject locality
        # ---------------------------------------------------------

        return self._check_subject_scope(
            resolved,
            chosen,
            candidate,
            evidence,
            signals,
        )
    
    def _check_subject_scope(self, resolution, chosen, candidate, evidence, signals):
        keys = (chosen.name, *chosen.aliases)
        claim = entity_label(candidate.quote)
        anaphoric = bool(re.match(
            r"^(?:(?:this|that|these|those|it|they|its|their)\b|"
            r"the (?:branch|location|atm|account|loan|card|program|initiative|service|organization)\b)", claim))
        if not anaphoric:
            embedded = re.search(
                r"(?<=[.!?])\s+(?P<reference>(?:(?:this|that|the) "
                r"(?:branch|location|atm|account|loan|card|program|initiative|service|organization)\b|"
                r"(?:it|they|its|their)\b))", candidate.quote, re.I)
            if embedded:
                # A whole-paragraph quote must not hide a later ambiguous
                # reference merely because the selected name appears earlier.
                start = embedded.start("reference")
                prefix = candidate.quote[:start].rstrip()
                anchors = list(signals)
                if any(contains_label(prefix, key) for key in keys):
                    anchors.append(IdentitySignal("subject", chosen.name, prefix))
                return self._check_subject_scope(
                    resolution, chosen, replace(candidate, quote=candidate.quote[start:]), evidence, anchors)
            if any(contains_label(candidate.quote, key) for key in keys):
                return resolution
        # Typed references compare compatible entity kinds. Generic pronouns
        # retain all kinds: they can refer to an organization as well as a product.
        kind = self._entity_kind(chosen)
        reference_types = {
            "branch": {"branch"}, "atm": {"atm"}, "location": {"branch", "atm"},
            "organization": {"organization"}, "cooperative": {"organization"},
            "company": {"organization"}, "institution": {"organization"},
            "credit union": {"organization"}, "program": {"program"},
            "initiative": {"program"},
            "product": {"product"}, "loan": {"product"}, "account": {"product"},
            "card": {"product"}, "certificate": {"product"},
            "service": {"service"}, "wallet": {"service"},
        }
        typed = re.match(r"^(?:this|that|the) (" + "|".join(reference_types) + r")\b", claim)
        if typed and kind not in reference_types[typed.group(1)]:
            return replace(resolution, entity_id=None, status="ambiguous",
                           reason="subject_reference_kind_incompatible")
        generic = bool(re.match(r"^(?:it|they|its|their|this|that|these|those)\b", claim)) and not typed
        kinds = None if generic else (reference_types[typed.group(1)] if typed else {kind})
        aliases = {}
        for record in (*self.entities.values(), *self.products.values()):
            if kinds is not None and self._entity_kind(record) not in kinds:
                continue
            if (isinstance(record, Product) and resolution.segment is not None
                    and record.segment not in {resolution.segment, "shared"}):
                continue
            aliases[record.id] = (record.name, *record.aliases)
        issue = subject_scope_issue(
            evidence.text,
            candidate.quote,
            [signal.quote for signal in signals],
            chosen.id,
            aliases,
            reference_kind=self._entity_kind(chosen),
            claim_block_index=getattr(candidate, "source_block_index", None),
        )
        if issue is None:
            return resolution
        reason, candidate_ids = issue
        return replace(resolution, entity_id=None, status="ambiguous",
                       reason=reason, candidate_ids=candidate_ids)

    def _normalize_product(
        self, candidate: CandidateFact, *, segment: str | None = None,
        evidence: Evidence | None = None, source_url: str | None = None,
        table_label: str | None = None,
    ) -> EntityResolution:
        """Resolve identity from exact labels, then validate all known signals.

        Structured identity_signals are authoritative for extraction context.
        Evidence is optional and used only to validate provenance and source URL;
        its Markdown is never reparsed. Candidates without signals retain legacy
        qualifier/heading handling. Explicit table_label supplies caller-verified
        context and is checked alongside extracted labels, never replacing them.
        """
        if segment not in {None, "personal", "business", "shared"}:
            raise ValueError("Unknown product segment")
        if evidence is not None:
            if evidence.id != candidate.evidence_id:
                raise ValueError("Candidate and Evidence IDs differ")
            if source_url is not None and source_url != evidence.source_url:
                raise ValueError("source_url disagrees with Evidence")
            source_url = evidence.source_url

        # Gather strong-to-weak signals before returning any match.
        signals: list[tuple[str, str, list[Product]]] = []
        def add(label: str, source: str) -> bool:
            matches = self._lookup(label)
            if matches:
                signals.append((label, source, matches))
            return bool(matches)

        structured = candidate.identity_signals
        if structured:
            for signal in structured:
                if signal.kind == "product_column":
                    add(signal.label, "product_column")
            for signal in structured:
                if signal.kind == "table_header":
                    add(signal.label, "table_header")
            table_labels = tuple(s.label for s in structured if s.kind == "table_label")
            heading_labels = tuple(s.label for s in structured if s.kind == "heading")
            parent_labels = tuple(s.label for s in structured if s.kind == "parent_label")
            signal_urls = {s.label for s in structured if s.kind == "source_url"}
            catalog_segments = {self._product_link_segment(s.label)
                                for s in structured if s.kind == "product_link"} - {None}
        else:
            # Compatibility for manually constructed, pre-signal candidates.
            for key, value in candidate.qualifiers.items():
                if re.fullmatch(
                    r"column \d+ (product|product name|account|account type|card type|loan type|loan program)",
                    normalize_label(key),
                ):
                    add(value, "product_column")
            table_labels = ()
            heading_labels = candidate.heading_path
            parent_labels = ()
            signal_urls = set()
            catalog_segments = set()
        if table_label is not None:
            table_labels = (*table_labels, table_label)
        for label in dict.fromkeys(table_labels):
            if not add(label, "table_label"):
                scoped = self._scoped_table_matches(label, (*parent_labels, *heading_labels))
                if scoped:
                    signals.append((label, "table_label", scoped))
        for heading in reversed(heading_labels):
            if add(heading, "heading"):
                break
        for parent in dict.fromkeys(parent_labels):
            add(parent, "parent_label")
        add(candidate.entity, "entity")

        # A label bound to this table supersedes inherited heading/entity scope.
        # Product columns remain independent strong evidence and must agree.
        has_table_identity = any(source in {"table_label", "table_header"}
                                 for _, source, _ in signals)
        if has_table_identity:
            signals = [s for s in signals if s[1] not in {"heading", "entity"}]
        elif any(normalize_label(label) in {"regular", "special"} for label in table_labels):
            # Missing family context must not silently resolve a generic table
            # label to the nearest family or page title as a specific product.
            signals = [s for s in signals if s[1] == "product_column"]

        if source_url is not None:
            signal_urls.add(source_url)
        url_segments = {self._url_segment(url) for url in signal_urls} - {None}
        context_segments = set()
        # Exact segment markers from catalog structure; no keyword guessing.
        markers = {
            "business rates": "business", "business savings": "business",
            "business money market": "business", "business checking rates": "business",
            "business savings rates": "business", "business credit cards": "business",
            "personal rates": "personal", "personal savings": "personal",
            "term share certificate rates": "personal",
            "business term share certificate rates": "business",
        }
        for label in (*table_labels, *parent_labels, *heading_labels):
            if normalize_label(label) in markers:
                context_segments.add(markers[normalize_label(label)])
        explicit_segments = ({segment} if segment is not None else set()) | url_segments
        # Eligibility wording (individual/small business/etc.) is deliberately
        # absent here. Product links select catalog records, not eligible users.
        explicit_segments.update(context_segments)
        explicit_segments.update(catalog_segments)
        ids = tuple(sorted({p.id for _, _, ps in signals for p in ps}))
        def ambiguous(reason: str, candidates: tuple[str, ...] = ids) -> EntityResolution:
            return EntityResolution(
                entity_id=None, status="ambiguous", candidate_ids=candidates,
                source=signals[0][1] if signals else None,
                matched_label=signals[0][0] if signals else None,
                segment=selected_segment, reason=reason,
            )

        selected_segment = next(iter(explicit_segments)) if len(explicit_segments) == 1 else None
        if len(signal_urls) > 1:
            return ambiguous("source_url_conflict")
        if len(catalog_segments) > 1:
            return ambiguous("product_link_segment_conflict")
        if len(explicit_segments) > 1:
            return ambiguous("segment_context_conflict")

        # On mixed pages, an exact unambiguous product signal can establish
        # segment. Intersect per-signal possibilities rather than pick a default.
        if selected_segment is None and signals:
            possibilities = None
            for _, _, matches in signals:
                options = {p.segment for p in matches if p.segment != "shared"}
                if options:
                    possibilities = options if possibilities is None else possibilities & options
            if possibilities is not None and len(possibilities) == 1:
                selected_segment = next(iter(possibilities))

        filtered = []
        for label, source, matches in signals:
            compatible = [p for p in matches if selected_segment is None
                          or p.segment in {selected_segment, "shared"}]
            if not compatible:
                return ambiguous("segment_incompatible")
            filtered.append((label, source, compatible))
        if not filtered:
            return EntityResolution(None, "unresolved", segment=selected_segment,
                                    reason="no_exact_product_match")

        # Product-column and table-label signals must describe the same product
        # or a compatible ancestor/descendant. A consistent child refines a family.
        # An unmatched child is never invented; unrelated identities stay ambiguous.
        pool = {p.id for _, _, ps in filtered for p in ps}
        viable = sorted(pid for pid in pool if all(
            any(self._ancestor(p.id, pid) for p in ps) for _, _, ps in filtered
        ))
        if len(viable) == 1:
            pid = viable[0]
            label, source, _ = next(
                (s for s in filtered if any(p.id == pid for p in s[2])), filtered[0],
            )
            return EntityResolution(pid, "resolved", (pid,), source, label,
                                    selected_segment, entity_kind=self._entity_kind(self.products[pid]))
        candidates = tuple(viable) if viable else ids
        return ambiguous("multiple_exact_matches" if viable else "identity_signal_conflict",
                         candidates)
