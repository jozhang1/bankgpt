"""Harness-facing interfaces for the knowledge system."""

from __future__ import annotations

import os
import sys
from pathlib import Path

harness_root = next(
    (
        parent / "harness"
        for parent in Path(__file__).resolve().parents
        if (parent / "harness" / "contract.py").is_file()
    ),
    None,
)
if harness_root is not None:
    sys.path.insert(0, str(harness_root))

from contract import Answer, Conflict, KnowledgeSystem, Snapshot, Source
from ingestion.processor import KnowledgeProcessor
from ingestion.semantic_extractor import SemanticExtractor
from ingestion.semantic_source_validator import SemanticSourceValidator
from foundation.entities import DUPACO_ENTITIES
from retrieval.retrieval import EvidenceRetriever
from retrieval.answer_selector import AnswerSelector
from answering.answer_renderer import AnswerRenderer
from facts.conflict_detector import ConflictDetector
from facts.temporal_resolver import TemporalResolver
from retrieval.version_selector import VersionSelector
from facts.fact_selector import FactSelector
from retrieval.quote_render_guard import DEFAULT_FACT_ASSOCIATIONS
from answering.llm_evidence_reasoner import LLMEvidenceReasoner
from answering.answerability_gate import AnswerabilityGate
from answering.knowledge_answer_flow import KnowledgeAnswerFlow
from retrieval.location_index import LocationIndex


class Knowledge(KnowledgeAnswerFlow, KnowledgeSystem):
    def __init__(
        self,
        llm_caller=None,
        *,
        ingestion_caller=None,
        semantic_entities=(),
        semantic_cache_dir=None,
        fact_association_rules=DEFAULT_FACT_ASSOCIATIONS,
    ):
        self.reasoner = LLMEvidenceReasoner(llm_caller)
        self.answerability_gate = AnswerabilityGate()
        self.last_gate_decisions = ()
        self.last_version_selections = ()
        self.last_quote_schedule_selections = ()
        self.fact_association_rules = tuple(fact_association_rules)
        semantic_entities = tuple(semantic_entities)
        self.processor = KnowledgeProcessor(
            entities=semantic_entities,
            semantic_extractor=(
                SemanticExtractor(ingestion_caller, cache_dir=semantic_cache_dir)
                if ingestion_caller is not None else None
            ),
            semantic_validator=SemanticSourceValidator(
                ingestion_caller, entities=semantic_entities, cache_dir=semantic_cache_dir,
            ) if ingestion_caller is not None else None,
        )
        self.fact_selector = FactSelector(self.processor.store, entities=semantic_entities)
        self.retriever = EvidenceRetriever(
            self.processor.store,
        )
        self.conflict_detector = ConflictDetector(self.processor.store)
        self.temporal_resolver = TemporalResolver(self.processor.store)
        self.version_selector = VersionSelector(self.processor.store)
        self.location_index = LocationIndex(self.processor.store)
        self.answer_selector = AnswerSelector(
            retriever=self.retriever,
            fact_selector=self.fact_selector,
            location_index=self.location_index,
        )
        self.answer_renderer = AnswerRenderer({
            'location': self._render_location_route,
            'fact': self._render_fact_route,
            'retrieved': self._render_retrieved_route,
        })

    def ingest(self, snapshot: Snapshot) -> None:
        self.processor.process(snapshot)
        # Index construction belongs to ingestion, outside answer latency timing.
        self.retriever.build_bm25_index()
        self.location_index.build_index()

    def answer(self, question: str, as_of: str | None = None) -> Answer:
        """Select candidates, then render through the shared answer interface."""
        self.last_gate_decisions = ()
        self.last_version_selections = ()
        self.last_quote_schedule_selections = ()
        selection = self.answer_selector.select(question, as_of=as_of)
        return self.answer_renderer.render(selection, question=question, as_of=as_of)

    def conflicts(self) -> list[Conflict]:
        """Expose source disagreements using the harness's module identity."""
        return [Conflict(
            topic=conflict.topic, values=list(conflict.values),
            sources=[Source(url=source.url, snapshot_id=source.snapshot_id, quote=source.quote)
                     for source in conflict.sources],
            resolution=conflict.resolution, reason=conflict.reason,
        ) for conflict in self.conflict_detector.detect()]

def build() -> KnowledgeSystem:
    from answering.openai_caller import (
        caller_from_environment,
        load_local_environment,
        retrieval_caller_from_environment,
    )
    load_local_environment()
    ingestion_caller = caller_from_environment()
    retrieval_caller = retrieval_caller_from_environment()
    return Knowledge(
        llm_caller=retrieval_caller,
        ingestion_caller=ingestion_caller,
        semantic_entities=(
            DUPACO_ENTITIES
            if ingestion_caller is not None
            else ()
        ),
        semantic_cache_dir=os.environ.get(
            'BANKGPT_SEMANTIC_CACHE_DIR',
            str(Path(__file__).resolve().parent.parent / '.cache' / 'semantic'),
        ),
    )
