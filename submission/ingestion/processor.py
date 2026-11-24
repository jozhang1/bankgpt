"""Snapshot processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import asdict, replace
from collections import Counter
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from uuid import NAMESPACE_URL, uuid4, uuid5
from urllib.parse import urlsplit
import re
import sys
from time import monotonic

from ingestion.rate_lines import rate_lines_to_tables
from ingestion.cleaner import PageCleaner
from ingestion.chunker import Block, Chunk, EvidenceChunker
from ingestion.metadata_extractor import DocumentMetadata, MetadataExtractor
from foundation.contract import Evidence, FactBuildAudit, PendingStatement
from ingestion.extractor import FactExtractor
from facts.entity_normalizer import EntityNormalizer
from foundation.products import PRODUCTS
from ingestion.semantic_extraction_gate import SemanticExtractionGate
from ingestion.deterministic_coverage import DeterministicCoverage
from ingestion.deterministic_fields import DeterministicFieldExtractor, DeterministicFieldEntityNormalizer
from ingestion.semantic_extractor import SemanticExtractor
from ingestion.normalizer import FactNormalizer
from facts.fact_builder import FactBuilder
from foundation.store import KnowledgeStore
from facts.semantic_support import SemanticSupport, bind_rule_tree
from ingestion.semantic_context import bound_context_quotes
from ingestion.rate_relation_extractor import RateRelationExtractor
from ingestion.ingestion_calls import IngestionModelMeter, new_call_counts
from facts.attribute_normalizer import RATE_ATTRIBUTES

if TYPE_CHECKING:
    from contract import Page, Snapshot


def _verbose_ingest(*parts, **kwargs) -> None:
    """Per-fact dumps are noisy; print them only when BANKGPT_VERBOSE_INGEST=1."""
    if os.environ.get('BANKGPT_VERBOSE_INGEST', '').strip() == '1':
        print(*parts, **kwargs)


class KnowledgeProcessor:
    def __init__(
        self, *, semantic_extractor: SemanticExtractor | None = None,
        entities=(), semantic_validator=None, deterministic_coverage=None,
    ):
        self.store = KnowledgeStore()
        self.cleaner = PageCleaner()
        self.chunker = EvidenceChunker()
        self.metadata_extractor = MetadataExtractor()
        self.extractor = FactExtractor()
        entities = tuple(entities)
        self.normalizer = FactNormalizer(
            entity_normalizer=EntityNormalizer(PRODUCTS, entities=entities),
        )
        self.fact_builder = FactBuilder(entities=entities, semantic_validator=semantic_validator)
        self.field_extractor = DeterministicFieldExtractor(entities=entities)
        field_entities = DeterministicFieldEntityNormalizer(self.field_extractor)
        self.field_normalizer = FactNormalizer(entity_normalizer=field_entities)
        self.field_builder = FactBuilder(entity_normalizer=field_entities,
                                        semantic_validator=self.field_extractor.validate)
        self.field_rejections = {}
        self.model_call_counts = new_call_counts()
        self.ingestion_call_counts = {}
        for component in (semantic_extractor, semantic_validator):
            if component is not None and getattr(component, 'caller', None) is not None:
                component.caller = IngestionModelMeter(component.caller, self.model_call_counts)
        self.rate_extractor = (RateRelationExtractor(semantic_extractor.caller,
            cache_dir=semantic_extractor.cache.directory,
            cache_namespace=semantic_extractor.cache.namespace) if semantic_extractor else None)
        self.rate_validator = semantic_validator
        self.rate_relation_rejections = {}
        self.rate_transition_audits = {}
        self.semantic_extractor = semantic_extractor
        self.semantic_gate = SemanticExtractionGate()
        self.deterministic_coverage = (deterministic_coverage if deterministic_coverage is not None
                                       else DeterministicCoverage())
        # Retain unresolved/skipped proposals outside canonical Fact admission.
        # Published only after the store commits the corresponding Evidence.
        self.semantic_candidates = {}
        self.semantic_decisions = {}
        self.semantic_rejections = {}

    @staticmethod
    def _pending_statement(candidate, audit: FactBuildAudit, evidence: Evidence):
        """Retain a skipped candidate only when its claim quote is source-bound."""
        if (audit.status != 'skipped' or evidence.in_recommendations
                or not candidate.quote.strip() or candidate.quote not in evidence.text):
            return None

        identity = {
            'evidence_id': str(evidence.id),
            'entity': candidate.entity,
            'attribute': candidate.attribute,
            'qualifiers': candidate.qualifiers,
            'value': candidate.value,
            'unit': candidate.unit,
            'quote': candidate.quote,
        }
        stable = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str)
        return PendingStatement(
            id=uuid5(NAMESPACE_URL, f'bankgpt:pending-statement:v1:{stable}'),
            evidence_id=evidence.id,
            entity=candidate.entity,
            attribute=candidate.attribute,
            qualifiers=dict(candidate.qualifiers),
            value=candidate.value,
            unit=candidate.unit,
            quote=candidate.quote,
            issues=tuple(dict.fromkeys(audit.issues)),
            table_header=candidate.table_header,
            heading_path=tuple(candidate.heading_path),
            raw_date_notes=tuple(candidate.raw_date_notes),
            effective_from=candidate.effective_from,
            effective_to=candidate.effective_to,
            date_quote=candidate.date_quote,
            identity_signals=tuple(
                (signal.kind, signal.label, signal.quote)
                for signal in candidate.identity_signals
            ),
            entity_refs=tuple(candidate.entity_refs),
        )

    @staticmethod
    def _classify_source_type(page_title: str, markdown: str) -> str:
        """Classify a source by its published structure, not its URL alone."""
        title = page_title.casefold()
        body = markdown.casefold()

        calculator_fields = (
            'loan amount', 'loan term', 'interest rate', 'monthly payment',
            'down payment', 'purchase information', 'sales tax',
            'trade in value', 'balance transfer amount', 'desired payment',
        )
        calculator_label = 'calculator' in title or bool(
            re.search(r'^#{1,6}[^\n]*\bcalculator\b', body, re.MULTILINE)
        )
        if calculator_label and sum(field in body for field in calculator_fields) >= 3:
            return 'calculator'

        for line in markdown.splitlines():
            if not line.strip().startswith('|'):
                continue
            cells = [cell.strip().casefold() for cell in line.strip().strip('|').split('|')]
            has_rate_metric = any(re.search(
                r'\b(?:apr|apy|annual percentage (?:rate|yield)|interest rate|dividend rate)\b',
                cell,
            ) for cell in cells)
            has_scope_column = any(re.search(
                r'\b(?:term|model year|balance|product|account|loan program|minimum)\b',
                cell,
            ) for cell in cells)
            if has_rate_metric and has_scope_column:
                return 'official_rate_table'

        return 'unknown'

    @staticmethod
    def chunk_to_evidence(
        text: str,
        page: Page,
        snapshot_id: str,
        *,
        source_type: str,
        in_recommendations: bool,
    ) -> Evidence:
        """Create one source occurrence without deduplicating or storing it.

        Hash normalized line endings and outer whitespace; retain the supplied
        text verbatim. Source classification is supplied by the caller.
        """
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        dt = datetime.fromisoformat(
            page.fetched_at.strip().replace("Z", "+00:00")
        )
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        fetched_at = dt.astimezone(timezone.utc)
        return Evidence(
            id=uuid4(),
            snapshot_id=snapshot_id,
            source_url=page.url,
            source_title=page.title,
            source_type=source_type,
            in_recommendations=in_recommendations,
            text=text,
            content_hash=sha256(normalized.encode("utf-8")).hexdigest(),
            fetched_at=fetched_at,
        )

    def process(self, snapshot: Snapshot) -> None:
        if snapshot.snapshot_id in self.store.snapshots:
            raise ValueError(f"Duplicate snapshot: {snapshot.snapshot_id}")
        calls_before = dict(self.model_call_counts)
        evidence_records: list[Evidence] = []
        for page in snapshot.pages:
            cleaned = rate_lines_to_tables(self.cleaner.clean(page.markdown))
            source_type = self._classify_source_type(page.title, cleaned)
            blocks = self.chunker.parse_blocks(cleaned)
            metadata = self.metadata_extractor.extract(
                blocks, page_title=page.title, source_url=page.url,
            )
            chunks = self._chunk_page(cleaned, page, blocks=blocks, document_metadata=metadata)
            for chunk in chunks:
                text = self.chunker.render_chunk(
                    chunk, page_title=page.title, document_metadata=metadata,
                )
                evidence_records.append(self.chunk_to_evidence(
                    text, page, snapshot.snapshot_id, source_type=source_type,
                    in_recommendations=chunk.in_recommendations,
                ))

        # Stage locator identities from current evidence and committed history.
        # Local collaborators keep failed ingestion from publishing catalog state.
        from copy import copy
        from ingestion.location_entities import location_normalizer
        entity_normalizer = location_normalizer(
            self.normalizer.entity_normalizer, self.store, evidence_records)
        normalizer = copy(self.normalizer)
        normalizer.entity_normalizer = entity_normalizer
        fact_builder = copy(self.fact_builder)
        fact_builder.entity_normalizer = entity_normalizer
        if hasattr(fact_builder.semantic_validator, 'entity_normalizer'):
            fact_builder.semantic_validator = copy(fact_builder.semantic_validator)
            fact_builder.semantic_validator.entity_normalizer = entity_normalizer

        build_results = []
        semantic_results = []
        semantic_support = {}
        pending_statements = {}
        semantic_candidates = {}
        semantic_decisions = {}
        semantic_rejections = {}
        field_rejections = {}
        rate_rejections = {}
        rate_transition_audits = {}
        for evidence_index, evidence in enumerate(evidence_records, start=1):
            deterministic_candidates = self.extractor.extract(evidence)
            deterministic_results = []
            for candidate in deterministic_candidates:
                normalized = normalizer.normalize(candidate, evidence=evidence)
                result = fact_builder.build(normalized)
                build_results.append(result)
                deterministic_results.append(result)
                pending = self._pending_statement(normalized, result[2], evidence)
                if pending is not None:
                    pending_statements[pending.id] = pending
            # Contact/navigation rules never invoke a model. Do not mark the
            # entire block covered: it may also contain a statistic or fee.
            field_candidates = self.field_extractor.extract(evidence)
            field_rejections[evidence.id] = self.field_extractor.last_rejections
            for candidate in field_candidates:
                normalized = self.field_normalizer.normalize(candidate, evidence=evidence)
                result = self.field_builder.build(normalized, evidence=evidence)
                build_results.append(result)
                pending = self._pending_statement(normalized, result[2], evidence)
                if pending is not None:
                    pending_statements[pending.id] = pending
                if result[2].status == 'built':
                    semantic_support[result[1].id] = SemanticSupport(
                        normalized.quote, tuple(s.quote for s in normalized.identity_signals),
                        normalized.date_quote,
                    )
                else:
                    field_rejections[evidence.id] += ({'attribute': candidate.attribute,
                                                       'reason': result[2].issues},)
            if self.semantic_extractor is None:
                continue
            # Coverage is computed per whole block by the default cell-aware
            # adapter, or by an explicitly injected deterministic processor.
            covered_quotes = (self.deterministic_coverage(
                evidence, tuple(deterministic_candidates), tuple(deterministic_results),
            ) if self.deterministic_coverage is not None else ())
            decision = self.semantic_gate.decide(evidence, covered_quotes=covered_quotes)
            semantic_decisions[evidence.id] = decision
            if not decision.should_extract:
                continue
            rate_indices = tuple(i for i, benefit in decision.benefits if benefit == 'rate_disclosure_binding')
            semantic_indices = tuple(i for i in decision.uncovered_block_indices if i not in rate_indices)
            if rate_indices:
                rate_started = monotonic()
                blocks = self.chunker.parse_blocks(evidence.text)
                rate_rows = [
                    c for c, built in zip(deterministic_candidates, deterministic_results)
                    if built[2].status == 'built' and built[0].attribute in RATE_ATTRIBUTES
                ]
                rate_support = {}
                bases_list = []
                for candidate in rate_rows:
                    row_quotes = self.extractor.rate_disclosure_quotes(evidence, candidate.quote)
                    # A marker-less note is not evidence for a row-specific
                    # transaction or phase. Only send candidates with an
                    # explicit row-to-footnote association to the model.
                    if len(row_quotes) < 2:
                        continue
                    index = len(bases_list)
                    bases_list.append(normalizer.normalize(candidate, evidence=evidence))
                    rate_support[index] = row_quotes
                bases = tuple(bases_list)
                rate_progress = f"[rate-relations {snapshot.snapshot_id}] evidence {evidence_index}/{len(evidence_records)}"
                if bases:
                    print(f"{rate_progress}: binding {len(bases)} existing rate cells",file=sys.stderr,flush=True)
                bindings = self.rate_extractor.extract(
                    evidence, bases, supporting_quotes_by_candidate=rate_support,
                )
                rate_rejections[evidence.id] = list(self.rate_extractor.last_rejections)
                for binding in bindings:
                    normalized = normalizer.normalize(binding.candidate, evidence=evidence)
                    provisional = replace(normalized, issues=[issue for issue in normalized.issues
                                                              if issue != 'llm_rate_review_required'])
                    trial = fact_builder.build(provisional)
                    accepted = False
                    if trial[2].status == 'built' and hasattr(self.rate_validator, 'validate'):
                        review = self.rate_validator.validate(normalized, evidence,
                                                             support_quotes=binding.supporting_quotes)
                        accepted = review.valid
                        if not accepted:rate_rejections[evidence.id].extend(review.reasons)
                    if accepted:
                        build_results.append(trial)
                        semantic_results.append(trial)
                        semantic_support[trial[1].id] = SemanticSupport(normalized.quote,
                            tuple(s.quote for s in normalized.identity_signals if s.kind != 'source_url'
                                  and s.quote and s.quote in evidence.text), normalized.date_quote,
                            tuple(dict.fromkeys((*bound_context_quotes(normalized,evidence),
                                                 *binding.supporting_quotes))))
                        for transition in binding.transitions:
                            source_index = transition.source_candidate_index
                            target_index = transition.target_candidate_index
                            rate_transition_audits.setdefault(evidence.id, []).append({
                                'relation': transition.relation,
                                'source_value': bases[source_index].value,
                                'target_value': bases[target_index].value,
                                'entity': normalized.entity,
                                'attribute': normalized.attribute,
                                'supporting_quote': transition.supporting_quote,
                            })
                    else:
                        result = fact_builder.build(normalized)
                        build_results.append(result)
                        pending = self._pending_statement(normalized, result[2], evidence)
                        if pending is not None:
                            pending_statements[pending.id] = pending
                if bases:
                    print(f"{rate_progress}: proposals={len(bindings)}, rejections={len(rate_rejections[evidence.id])}, "
                          f"cache_hit={self.rate_extractor.last_cache_hit}, elapsed={monotonic()-rate_started:.1f}s",
                          file=sys.stderr,flush=True)
            if not semantic_indices:
                continue
            started = monotonic()
            progress = (f"[semantic {snapshot.snapshot_id}] evidence "
                        f"{evidence_index}/{len(evidence_records)} hash={evidence.content_hash[:12]}")
            print(f"{progress}: extracting {len(decision.uncovered_block_indices)} blocks; "
                  f"benefits={dict(Counter(reason for _, reason in decision.benefits))}",
                  file=sys.stderr, flush=True)
            candidates = self.semantic_extractor.extract(
                evidence, uncovered_block_indices=semantic_indices,
            )
            semantic_candidates[evidence.id] = tuple(candidates)
            semantic_rejections[evidence.id] = tuple(self.semantic_extractor.last_rejections)
            rejection_counts = Counter(r['reason'] for r in self.semantic_extractor.last_rejections)
            if rejection_counts:
                print(f"{progress}: proposal_rejections="
                      + json.dumps(dict(rejection_counts), ensure_ascii=False),
                      file=sys.stderr, flush=True)
            built_semantic_count = 0
            for candidate_index, candidate in enumerate(candidates):
                normalized = normalizer.normalize(candidate, evidence=evidence)
                result = fact_builder.build(normalized, evidence=evidence)
                build_results.append(result)
                pending = self._pending_statement(normalized, result[2], evidence)
                if pending is not None:
                    pending_statements[pending.id] = pending
                if result[2].status == "built":
                    semantic_results.append(result)
                    semantic_support[result[1].id] = SemanticSupport(
                        normalized.quote,
                        tuple(s.quote for s in normalized.identity_signals if s.kind == 'subject'),
                        normalized.date_quote,
                        bound_context_quotes(normalized, evidence),
                        rule_node_bindings=(bind_rule_tree(normalized.value, evidence.id)
                                            if normalized.attribute == 'eligibility_requirement'
                                            else ()),
                    )
                    built_semantic_count += 1
                    success = {
                        'status': 'built_pending_commit',
                        'candidate_index': candidate_index,
                        'source_url': evidence.source_url,
                        'fact': asdict(result[0]),
                        'observation': asdict(result[1]),
                    }
                    _verbose_ingest(f"{progress}: success=" + json.dumps(success, ensure_ascii=False, default=str),
                          file=sys.stderr, flush=True)
                else:
                    detail = {'candidate_index': candidate_index, 'entity': normalized.entity,
                              'attribute': normalized.attribute, 'value': normalized.value, 'unit': normalized.unit,
                              'qualifiers': normalized.qualifiers, 'issues': result[2].issues}
                    _verbose_ingest(f"{progress}: skipped_candidate=" + json.dumps(detail, ensure_ascii=False),
                          file=sys.stderr, flush=True)
            print(f"{progress}: candidates={len(candidates)}, "
                  f"built={built_semantic_count}, skipped={len(candidates) - built_semantic_count}, "
                  f"rejected_proposals={len(self.semantic_extractor.last_rejections)}, "
                  f"extraction_cache_hit={self.semantic_extractor.last_cache_hit}, "
                  f"elapsed={monotonic() - started:.1f}s", file=sys.stderr, flush=True)
        # Publish Evidence and constructed records together, only after every
        # processing stage and repository validation has succeeded.
        self.store.commit_ingestion(snapshot, evidence_records, build_results=build_results,
                                    semantic_support=semantic_support,
                                    pending_statements=pending_statements.values())
        self.rate_transition_audits.update(rate_transition_audits)
        self.field_rejections.update({k: v for k, v in field_rejections.items() if v})
        self.rate_relation_rejections.update({k:v for k,v in rate_rejections.items() if v})
        self.ingestion_call_counts[snapshot.snapshot_id] = {
            k:self.model_call_counts[k]-calls_before.get(k,0) for k in self.model_call_counts}
        if self.semantic_extractor is not None:
            print(f"[ingestion {snapshot.snapshot_id}] model_calls="
                  + json.dumps(self.ingestion_call_counts[snapshot.snapshot_id]),file=sys.stderr,flush=True)
        self.semantic_candidates.update(semantic_candidates)
        self.semantic_decisions.update(semantic_decisions)
        self.semantic_rejections.update(semantic_rejections)
        for fact, observation, _ in semantic_results:
            _verbose_ingest("Semantic Fact [committed]:", fact, flush=True)
            _verbose_ingest("Semantic FactObservation [committed]:", observation, flush=True)

    def _chunk_page(
        self, cleaned: str, page: Page, *, blocks: list[Block] | None = None,
        document_metadata: DocumentMetadata | None = None,
    ) -> list[Chunk]:
        """Isolate a verified scrambled travel-notice panel in one PDF export.

        Keep extracted wording/order and source metadata. Do not infer steps
        from the misplaced numbers or inherit the adjacent savings title.
        """
        path = urlsplit(page.url).path.casefold()
        if path == "/hubfs/downloadables/personal/owners-manual-summer2026.pdf":
            panel = re.search(
                r"(?m)^Go to My Settings,\nthen Card Security\.\n"
                r"Select Submit\na Travel Notice\.[\s\S]*?"
                r"^### Mobile Banking\. 1\s*\n(?=\s*---\s*$)",
                cleaned,
            )
            if panel and all(value in panel.group(0) for value in (
                "# 2", "# 3", "# 4", "Tell us when and where", "Log in to Shine",
            )):
                fragment = re.sub(r"(?m)^#{1,6}\s+", "", panel.group(0))
                chunks: list[Chunk] = []
                for part in (cleaned[:panel.start()], fragment, cleaned[panel.end():]):
                    chunks.extend(self.chunker.chunk(part, page_title=page.title))
                return chunks
        if blocks is None:
            blocks = self.chunker.parse_blocks(cleaned)
        return self.chunker.chunk_blocks(
            blocks, page_title=page.title, source_url=page.url,
            document_metadata=document_metadata,
        )
