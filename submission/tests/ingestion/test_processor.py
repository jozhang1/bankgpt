import unittest

from contract import Page, Snapshot
from ingestion.processor import KnowledgeProcessor


class KnowledgeProcessorTests(unittest.TestCase):


    def test_process_stores_snapshot_and_evidence(self):
        processor = KnowledgeProcessor()
        page = Page("https://example.com", "Student Loans", "## Benefits\nFlexible repayment.", "2026-09-28T00:00:00Z")
        snapshot = Snapshot("latest", "2026-09-28", [page])
        processor.process(snapshot)
        self.assertIs(processor.store.snapshots["latest"], snapshot)
        self.assertEqual(len(processor.store.evidence), 1)
        evidence = next(iter(processor.store.evidence.values()))
        self.assertEqual(evidence.source_title, page.title)
        self.assertEqual(evidence.snapshot_id, "latest")
        self.assertIn("Flexible repayment.", evidence.text)
        with self.assertRaises(ValueError):
            processor.process(snapshot)
        self.assertEqual(len(processor.store.evidence), 1)

    def test_failed_processing_does_not_register_snapshot(self):
        processor = KnowledgeProcessor()
        page = Page("https://example.com", "Title", "Body", "invalid-date")
        with self.assertRaises(ValueError):
            processor.process(Snapshot("bad", "2026-09-28", [page]))
        self.assertEqual(processor.store.snapshots, {})
        self.assertEqual(processor.store.evidence, {})


    def rate_page(self, url, value='5.39%'):
        return Page(url, 'Auto Loans',
                    '# Auto Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | ' + value +
                    ' |\n\nRates effective 05/01/2026.', '2026-05-12T00:00:00Z')

    def test_process_builds_and_saves_facts_observations_and_audits(self):
        processor = KnowledgeProcessor()
        store = processor.store
        first = Snapshot('may', '2026-05-12', [self.rate_page('https://example.com/auto')])
        processor.process(first)
        self.assertIs(processor.store, store)
        fact = next(iter(store.facts.values()))
        obs = next(iter(store.observations.values()))
        self.assertEqual((fact.entity, fact.attribute, fact.qualifiers),
                         ('personal_auto', 'apr', {'term_months': '60'}))
        self.assertEqual((obs.value, obs.unit), ('5.39', 'percent'))
        self.assertIn(obs.evidence_id, store.evidence)
        self.assertEqual(store.fact_by_key[fact.key], fact.id)
        self.assertEqual(store.obs_by_fact[fact.id], [obs.id])
        self.assertEqual(len(store.fact_build_audits), 1)
        processor.process(Snapshot('sep', '2026-09-28', [self.rate_page('https://example.com/auto')]))
        self.assertEqual(len(store.facts), 1)
        self.assertEqual(len(store.observations), 2)
        self.assertEqual(len(store.fact_build_audits), 2)
        self.assertEqual({store.evidence[o.evidence_id].snapshot_id for o in store.observations.values()},
                         {'may', 'sep'})
        import copy
        before = copy.deepcopy(vars(store))
        with self.assertRaises(ValueError):
            processor.process(first)
        self.assertEqual(vars(store), before)


