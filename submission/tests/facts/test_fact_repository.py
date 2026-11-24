import copy
from dataclasses import replace
import unittest
from uuid import uuid4

from contract import Page, Snapshot
from ingestion.extractor import CandidateFact
from facts.fact_builder import FactBuilder
from facts.fact_repository import FactRepository
from ingestion.normalizer import FactNormalizer
from ingestion.processor import KnowledgeProcessor
from foundation.store import KnowledgeStore


class FactRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.store = KnowledgeStore()
        self.evidence = []
        for snapshot_id, taken_on in (('may', '2026-05-12'), ('sep', '2026-09-28')):
            page = Page('https://example.com/auto', 'Auto Loans', 'same source',
                        taken_on + 'T00:00:00Z')
            record = KnowledgeProcessor.chunk_to_evidence(
                page.markdown, page, snapshot_id, source_type='unknown', in_recommendations=False)
            self.store.commit_ingestion(Snapshot(snapshot_id, taken_on, [page]), [record])
            self.evidence.append(record)
        self.repository = FactRepository(self.store)

    def candidate(self, evidence=None, term='60 Months'):
        evidence = evidence or self.evidence[0]
        return FactNormalizer().normalize(CandidateFact(
            entity='Auto Loans', attribute='APR', qualifiers={'column_1: Term': term},
            value='5.39%', unit='%', evidence_id=evidence.id,
            quote='source row', table_header='| Term | APR |', heading_path=('Auto Loans',)))

    def state(self):
        return copy.deepcopy(vars(self.store))

    def test_repeated_save_reuses_records_and_indexes(self):
        built = FactBuilder().build(self.candidate())
        first = self.repository.save(*built)
        second = FactRepository(self.store).save(*built)
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        fact, observation, audit = first
        self.assertEqual(self.store.fact_by_key, {fact.key: fact.id})
        self.assertEqual(self.store.obs_by_fact[fact.id], [observation.id])
        self.assertEqual(self.store.facts_by_entity_attribute[(fact.entity, fact.attribute)], [fact.id])
        self.assertEqual(len(self.store.facts), 1)
        self.assertEqual(len(self.store.observations), 1)
        self.assertEqual(self.store.fact_build_audits, [audit, audit])

    def test_history_and_cross_url_provenance_are_not_content_deduplicated(self):
        self.assertEqual(self.evidence[0].content_hash, self.evidence[1].content_hash)
        a = self.repository.save(*FactBuilder().build(self.candidate(self.evidence[0])))
        b = self.repository.save(*FactBuilder().build(self.candidate(self.evidence[1])))
        self.assertEqual(a[0].id, b[0].id)
        self.assertNotEqual(a[1].id, b[1].id)
        self.assertEqual(len(self.store.obs_by_fact[a[0].id]), 2)
        self.assertEqual({o.evidence_id for o in self.store.observations.values()},
                         {e.id for e in self.evidence})
        self.assertIsNone(a[1].effective_from)
        self.assertIsNone(a[1].effective_to)


    def test_validation_failures_do_not_partially_write(self):
        original = FactBuilder().build(self.candidate())
        self.repository.save(*original)
        fact, obs, audit = original
        new_fact, new_obs, new_audit = FactBuilder().build(self.candidate(term='72 Months'))
        collision_id = uuid4()
        failures = (
            (fact, obs, replace(audit, evidence_id=uuid4())),
            (fact, replace(obs, evidence_id=self.evidence[1].id), audit),
            (None, obs, audit),
            (replace(fact, qualifiers={'term_months': '72'}), obs, audit),
            (replace(fact, id=collision_id), replace(obs, fact_id=collision_id),
             replace(audit, fact_id=collision_id)),
            (fact, replace(obs, value='7.00'), audit),
            (new_fact, replace(new_obs, id=obs.id), replace(new_audit, observation_id=obs.id)),
        )
        before = self.state()
        for result in failures:
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.repository.save(*result)
            self.assertEqual(vars(self.store), before)

