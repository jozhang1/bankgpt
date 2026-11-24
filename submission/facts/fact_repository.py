"""Persist constructed facts, observations and audits in a KnowledgeStore."""

from copy import deepcopy

from foundation.contract import Fact, FactBuildAudit, FactObservation
from foundation.store import KnowledgeStore


class FactRepository:
    """Save FactBuilder output; validate before changing the single-writer store.

    Identical IDs reuse records and index entries. Distinct Evidence occurrences
    retain distinct observations. Every successful save call appends its audit,
    including skipped candidates and reuse of previously saved records.
    """

    def __init__(self, store: KnowledgeStore):
        self.store = store

    def save(
        self,
        fact: Fact | None,
        observation: FactObservation | None,
        audit: FactBuildAudit,
    ) -> tuple[Fact | None, FactObservation | None, FactBuildAudit]:
        """Save a validated build result; conflicts raise without partial writes.

        Evidence must already exist. This is an in-memory, synchronous operation,
        not a persistence transaction or a concurrent-writer guarantee.
        """
        if audit.evidence_id not in self.store.evidence:
            raise ValueError(f"Unknown evidence: {audit.evidence_id}")
        if audit.status == "skipped":
            if fact is not None or observation is not None:
                raise ValueError("Skipped audit must not include built records")
            self.store.fact_build_audits.append(audit)
            return None, None, audit
        if fact is None or observation is None:
            raise ValueError("Built audit requires a Fact and FactObservation")
        if (audit.fact_id != fact.id or audit.observation_id != observation.id
                or observation.fact_id != fact.id
                or audit.evidence_id != observation.evidence_id):
            raise ValueError("Fact, observation and audit references disagree")

        existing_fact = self.store.facts.get(fact.id)
        indexed_id = self.store.fact_by_key.get(fact.key)
        if indexed_id is not None and indexed_id != fact.id:
            raise ValueError(f"Fact key already belongs to another ID: {fact.key}")
        if existing_fact is not None:
            if existing_fact != fact:
                raise ValueError(f"Conflicting Fact for ID: {fact.id}")
            if indexed_id != fact.id:
                raise ValueError(f"Missing or inconsistent key index for Fact: {fact.id}")
            if fact.id not in self.store.facts_by_entity_attribute.get((fact.entity, fact.attribute), ()):
                raise ValueError(f"Missing entity/attribute index for Fact: {fact.id}")
        elif indexed_id is not None:
            raise ValueError(f"Key index references a missing Fact: {fact.id}")

        existing_observation = self.store.observations.get(observation.id)
        if existing_observation is not None:
            if existing_observation != observation:
                raise ValueError(f"Conflicting observation for ID: {observation.id}")
            if existing_fact is None:
                raise ValueError(f"Observation references a missing Fact: {fact.id}")
            if observation.id not in self.store.obs_by_fact.get(fact.id, ()):
                raise ValueError(f"Missing observation index for ID: {observation.id}")

        # Stage copies and replacement index lists before publishing. Do not keep
        # caller-owned mutable Fact qualifiers as store state.
        stored_fact = existing_fact if existing_fact is not None else deepcopy(fact)
        stored_observation = (existing_observation if existing_observation is not None
                              else deepcopy(observation))
        entity_key = (fact.entity, fact.attribute)
        fact_ids = None
        obs_ids = None
        if existing_fact is None:
            fact_ids = [*self.store.facts_by_entity_attribute.get(entity_key, ()), fact.id]
        if existing_observation is None:
            obs_ids = [*self.store.obs_by_fact.get(fact.id, ()), observation.id]

        if existing_fact is None:
            self.store.facts[fact.id] = stored_fact
            self.store.fact_by_key[fact.key] = fact.id
            self.store.facts_by_entity_attribute[entity_key] = fact_ids
        if existing_observation is None:
            self.store.observations[observation.id] = stored_observation
            self.store.obs_by_fact[fact.id] = obs_ids
        self.store.fact_build_audits.append(audit)
        return stored_fact, stored_observation, audit
