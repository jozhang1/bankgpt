"""In-memory knowledge records and lookup indexes."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Iterable
from uuid import UUID

from foundation.contract import (
    Evidence, Fact, FactBuildAudit, FactObservation, PendingStatement,
)
from facts.semantic_support import SemanticSupport
from facts.attribute_normalizer import SEMANTIC_ATTRIBUTES, RATE_ATTRIBUTES

if TYPE_CHECKING:
    from contract import Snapshot


class KnowledgeStore:
    def __init__(self):
        self.snapshots: dict[str, Snapshot] = {}

        self.facts: dict[UUID, Fact] = {}
        self.observations: dict[UUID, FactObservation] = {}
        self.evidence: dict[UUID, Evidence] = {}
        self.pending_statements: dict[UUID, PendingStatement] = {}
        # One audit per repository save attempt, including skips and ID reuse.
        self.fact_build_audits: list[FactBuildAudit] = []
        self.semantic_support: dict[UUID, SemanticSupport] = {}

        self.fact_by_key: dict[str, UUID] = {}
        self.obs_by_fact: dict[UUID, list[UUID]] = defaultdict(list)
        self.facts_by_entity_attribute: dict[
            tuple[str, str], list[UUID]
        ] = defaultdict(list)

        # Snapshot IDs in ascending taken_on order, without duplicates.
        self.snapshot_order: list[str] = []

    def add_snapshot(self, snapshot: Snapshot) -> None:
        """Register a snapshot once, in chronological order."""
        self._validate_snapshot(snapshot)
        self.snapshots[snapshot.snapshot_id] = snapshot
        self.snapshot_order.append(snapshot.snapshot_id)

    def _validate_snapshot(self, snapshot: Snapshot) -> None:
        snapshot_id = snapshot.snapshot_id

        if snapshot_id in self.snapshots:
            raise ValueError(f"Duplicate snapshot: {snapshot_id}")

        if self.snapshot_order:
            latest = self.snapshots[self.snapshot_order[-1]]
            if date.fromisoformat(snapshot.taken_on) < date.fromisoformat(latest.taken_on):
                raise ValueError(f"Snapshot out of order: {snapshot_id}")

    def commit_ingestion(
        self,
        snapshot: Snapshot,
        evidence: list[Evidence],
        *,
        build_results: Iterable[tuple[Fact | None, FactObservation | None, FactBuildAudit]] = (),
        semantic_support: dict[UUID, SemanticSupport] | None = None,
        pending_statements: Iterable[PendingStatement] = (),
    ) -> None:
        """Publish a validated batch with no partial changes on validation failure.

        This synchronous in-memory operation assumes a single writer. It is not
        a persistence transaction or a thread-safety guarantee.
        """
        self._validate_snapshot(snapshot)
        batch: dict[UUID, Evidence] = {}
        for record in evidence:
            if record.snapshot_id != snapshot.snapshot_id:
                raise ValueError(f"Evidence snapshot mismatch: {record.id}")
            if record.id in batch or record.id in self.evidence:
                raise ValueError(f"Duplicate evidence: {record.id}")
            batch[record.id] = record

        pending_batch = {}
        for statement in pending_statements:
            source = batch.get(statement.evidence_id)
            if source is None:
                raise ValueError(
                    f"Pending statement is outside ingestion batch: {statement.id}"
                )
            if (source.in_recommendations or not statement.quote.strip()
                    or statement.quote not in source.text):
                raise ValueError(
                    f"Pending statement must cite literal non-recommendation Evidence: {statement.id}"
                )
            if statement.id in pending_batch or statement.id in self.pending_statements:
                raise ValueError(f"Duplicate pending statement: {statement.id}")
            pending_batch[statement.id] = statement

        # The repository validates/writes only the staged state. Keep this store
        # object stable for retrievers/detectors that already reference it.
        from facts.fact_repository import FactRepository

        staged = KnowledgeStore()
        staged.snapshots = {**self.snapshots, snapshot.snapshot_id: snapshot}
        staged.snapshot_order = [*self.snapshot_order, snapshot.snapshot_id]
        staged.evidence = {**self.evidence, **batch}
        staged.pending_statements = {**self.pending_statements, **pending_batch}
        staged.facts = dict(self.facts)
        staged.observations = dict(self.observations)
        staged.fact_by_key = dict(self.fact_by_key)
        staged.obs_by_fact = defaultdict(list, {k: list(v) for k, v in self.obs_by_fact.items()})
        staged.facts_by_entity_attribute = defaultdict(
            list, {k: list(v) for k, v in self.facts_by_entity_attribute.items()},
        )
        staged.fact_build_audits = list(self.fact_build_audits)
        repository = FactRepository(staged)
        for fact, observation, audit in build_results:
            if audit.evidence_id not in batch:
                raise ValueError(f"Build result is outside ingestion batch: {audit.evidence_id}")
            repository.save(fact, observation, audit)

        staged.semantic_support = dict(self.semantic_support)
        for oid, support in (semantic_support or {}).items():
            observation = staged.observations.get(oid)
            if (observation is None or observation.evidence_id not in batch
                    or not any(a.status == 'built' and a.observation_id == oid
                               and a.fact_id == observation.fact_id
                               and a.evidence_id == observation.evidence_id
                               for a in staged.fact_build_audits)):
                raise ValueError('Semantic support requires a built batch observation')
            source = batch[observation.evidence_id]
            if (staged.facts[observation.fact_id].attribute not in SEMANTIC_ATTRIBUTES | RATE_ATTRIBUTES
                    or source.in_recommendations or not support.subject_quotes
                    or any(not q.strip() or q not in source.text for q in support.quotes)):
                raise ValueError('Semantic support must bind literal source quotes')
            citations = support.citations(observation.evidence_id)
            from foundation.entities import entity_label
            referenced_entities = {
                reference.entity_id for reference in observation.entity_refs
                if reference.entity_id is not None and reference.label.strip()
                and reference.quote.strip() and reference.quote in source.text
                and entity_label(reference.label)
                in f" {entity_label(reference.quote)} "
            }
            for citation in citations:
                cited_source = staged.evidence.get(citation.evidence_id)
                if (cited_source is None
                        or cited_source.snapshot_id != source.snapshot_id
                        or cited_source.in_recommendations
                        or not citation.quote.strip()
                        or citation.quote not in cited_source.text):
                    raise ValueError(
                        'Semantic support citations must bind exact text in non-recommendation '
                        'Evidence from the Observation snapshot'
                    )
                if citation.evidence_id != observation.evidence_id:
                    linked = False
                    for related in staged.observations.values():
                        if (related.evidence_id != citation.evidence_id
                                or staged.facts[related.fact_id].entity not in referenced_entities):
                            continue
                        related_support = staged.semantic_support.get(related.id)
                        target_quotes = [related.value]
                        if related_support is not None:
                            target_quotes.extend(related_support.quotes)
                        if any(quote and (quote in citation.quote or citation.quote in quote)
                               for quote in target_quotes):
                            linked = True
                            break
                    if not linked:
                        raise ValueError(
                            'Cross-Evidence support requires an Observation EntityReference '
                            'that links to a Fact in the cited Evidence'
                        )

            fact = staged.facts[observation.fact_id]
            if fact.attribute == 'eligibility_requirement':
                from ingestion.membership_rules import rule_issue, rule_nodes

                issue = rule_issue(observation.value, allow_discontiguous=True)
                nodes = dict(rule_nodes(observation.value))
                bindings = {binding.node_path: binding
                            for binding in support.rule_node_bindings}
                if issue or len(bindings) != len(support.rule_node_bindings):
                    raise ValueError('Rule-tree support has invalid or duplicate node bindings')
                if set(bindings) != set(nodes):
                    raise ValueError('Every rule-tree node requires an explicit source binding')
                for path, node in nodes.items():
                    binding = bindings[path]
                    if not any(citation.quote == node['quote']
                               for citation in binding.evidence_quotes):
                        raise ValueError('Rule-tree node binding must cite its exact node quote')
                    if not binding.evidence_quotes:
                        raise ValueError('Rule-tree node binding requires at least one Evidence quote')
            elif support.rule_node_bindings:
                raise ValueError('Rule-node bindings are only valid for rule-expression observations')
            staged.semantic_support[oid] = support


        (self.snapshots, self.snapshot_order, self.evidence,
         self.facts, self.observations, self.fact_by_key,
         self.obs_by_fact, self.facts_by_entity_attribute, self.fact_build_audits,
         self.semantic_support, self.pending_statements) = (
            staged.snapshots, staged.snapshot_order, staged.evidence,
            staged.facts, staged.observations, staged.fact_by_key,
            staged.obs_by_fact, staged.facts_by_entity_attribute, staged.fact_build_audits,
            staged.semantic_support, staged.pending_statements,
        )
