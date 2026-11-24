from datetime import date
import unittest

from contract import Page, Snapshot
from ingestion.extractor import CandidateFact
from facts.fact_builder import FactBuilder
from facts.fact_repository import FactRepository
from ingestion.processor import KnowledgeProcessor
from foundation.store import KnowledgeStore
from facts.temporal_resolver import TemporalResolver


class TemporalResolverTests(unittest.TestCase):
    def setUp(self):
        self.store = KnowledgeStore()
        self.store.add_snapshot(Snapshot('may', '2026-05-12', []))
        self.repository = FactRepository(self.store)
        self.resolver = TemporalResolver(self.store)

    def add(self, value='5.39', *, sid='may', start=None, end=None):
        page = Page(f'https://example.com/{len(self.store.evidence)}', 'Auto Loans',
                    'APR: ' + value, '2026-05-12T00:00:00Z')
        evidence = KnowledgeProcessor.chunk_to_evidence(
            page.markdown, page, sid, source_type='unknown', in_recommendations=False)
        self.store.evidence[evidence.id] = evidence
        candidate = CandidateFact('personal_auto', 'apr', {'term_months': '60'}, value,
                                  'percent', evidence.id, evidence.text, '| Term | APR |',
                                  ('Auto Loans',), effective_from=start, effective_to=end)
        return self.repository.save(*FactBuilder().build(candidate))


    def test_equal_explicitly_applicable_values_are_resolved(self):
        fact, _, _ = self.add(start=date(2026, 5, 1), end=date(2026, 5, 31))
        self.add(start=date(2026, 5, 2), end=date(2026, 5, 20))
        result = self.resolver.resolve(fact)
        self.assertEqual(result.status, 'resolved')
        self.assertEqual(result.temporal_certainty, 'explicit')
        self.assertIn('2 explicitly applicable', result.reason)

    def test_different_values_at_the_query_date_are_conflicting(self):
        fact, _, _ = self.add(start=date(2026, 5, 1), end=date(2026, 5, 31))
        self.add('6.25', start=date(2026, 5, 2), end=date(2026, 5, 20))
        result = self.resolver.resolve(fact)
        self.assertEqual(result.status, 'conflicting')
        self.assertEqual(result.temporal_certainty, 'explicit')
        self.assertEqual({o.value for o in result.observations}, {'5.39', '6.25'})

    def test_different_unknown_values_are_ambiguous(self):
        fact, _, _ = self.add()
        self.add('6.25')
        self.assertEqual(self.resolver.resolve(fact).status, 'ambiguous')


    def test_selects_latest_complete_snapshot_without_fallback(self):
        fact, _, _ = self.add()
        self.store.add_snapshot(Snapshot('sep', '2026-09-28', []))
        self.assertEqual(self.resolver.resolve(fact).status, 'not_found')
        self.assertEqual(self.resolver.resolve(fact).snapshot_id, 'sep')
        historical = self.resolver.resolve(fact, as_of='2026-08-01')
        self.assertEqual((historical.status, historical.snapshot_id), ('resolved', 'may'))
        self.add('6.25', sid='sep')
        current = self.resolver.resolve(fact)
        self.assertEqual(current.status, 'resolved')
        self.assertEqual([o.value for o in current.observations], ['6.25'])


