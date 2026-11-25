import unittest
from unittest.mock import patch

from solution import Knowledge, Conflict, Source, Snapshot
from retrieval.answer_selector import AnswerSelection
from contract import Page


class KnowledgeIntegrationTests(unittest.TestCase):

    def test_missing_fact_falls_back_to_evidence_only_when_evidence_exists(self):
        knowledge = Knowledge()
        selection = AnswerSelection('fact', matches=())
        hits = (object(),)
        with patch.object(knowledge.answer_selector, 'retrieve', return_value=hits), \
             patch.object(knowledge.answer_selector, '_scope_evidence', return_value=hits), \
             patch.object(knowledge, '_render_retrieved_route', return_value='evidence answer') as render:
            self.assertEqual(
                knowledge._render_fact_route(selection, 'VTO users last year?', None),
                'evidence answer',
            )
            self.assertEqual(render.call_count, 1)
        with patch.object(knowledge.answer_selector, 'retrieve', return_value=()), \
             patch.object(knowledge.answer_selector, '_scope_evidence', return_value=()), \
             patch.object(knowledge, '_render_retrieved_route') as render:
            self.assertIn(
                'couldn’t find information',
                knowledge._render_fact_route(selection, 'Unpublished loan rate?', None).text,
            )
            render.assert_not_called()

    def test_answer_failure_returns_an_unconfident_answer_instead_of_raising(self):
        knowledge = Knowledge()
        with patch.object(knowledge.answer_selector, 'select', side_effect=RuntimeError('boom')):
            with self.assertLogs(level='ERROR'):
                answer = knowledge.answer('Anything?')
        self.assertFalse(answer.confident)
        self.assertFalse(answer.sources)
        self.assertIn('couldn’t find information', answer.text)


    def test_ingest_and_conflicts_use_the_harness_module_identity(self):
        knowledge = Knowledge()
        pages = [Page(f'https://example.com/{i}', 'Auto Loans',
                      '# Auto Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | ' + value +
                      ' |\n\nRates effective 05/01/2026.', '2026-05-12T00:00:00Z')
                 for i, value in enumerate(('5.39%', '6.25%'))]
        knowledge.ingest(Snapshot('may', '2026-05-12', pages))
        self.assertIs(knowledge.conflict_detector.store, knowledge.processor.store)
        self.assertIs(knowledge.retriever.store, knowledge.processor.store)
        self.assertEqual(len(knowledge.processor.store.facts), 1)
        self.assertEqual(len(knowledge.processor.store.observations), 2)
        conflicts = knowledge.conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertIsInstance(conflicts[0], Conflict)
        self.assertTrue(all(isinstance(source, Source) for source in conflicts[0].sources))
        self.assertEqual(conflicts[0].values, ['5.39%', '6.25%'])
        self.assertTrue(conflicts[0].reason.startswith('confirmed_conflict:'))
        self.assertEqual(conflicts, knowledge.conflicts())


    def test_current_snapshot_fact_and_explicit_historical_date(self):
        knowledge = Knowledge()
        def page(value, fetched):
            return Page('https://example.com/auto', 'Auto Loans',
                        '# Auto Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | ' + value + ' |', fetched)
        knowledge.ingest(Snapshot('may', '2026-05-12', [page('5.39%', '2026-05-12T00:00:00Z')]))
        knowledge.ingest(Snapshot('sep', '2026-09-28', [page('6.25%', '2026-09-28T00:00:00Z')]))
        query = 'What is the APR for a 60-month auto loan?'
        current = knowledge.answer(query)
        self.assertIn('6.25%', current.text)
        self.assertNotIn('5.39%', current.text)
        self.assertTrue(all(s.snapshot_id == 'sep' for s in current.sources))
        historical = knowledge.answer(query, as_of='2026-05-31')
        # The May snapshot is the latest one before this date, but its APR has
        # no explicit effective period and therefore is not asserted for May 31.
        self.assertNotIn('5.39%', historical.text)
        self.assertFalse(historical.confident)
        self.assertTrue(all(s.snapshot_id == 'may' for s in historical.sources))
        missing = knowledge.answer('What is the APR for a 72-month auto loan?')
        self.assertFalse(missing.confident)
        self.assertFalse(missing.sources)

    def test_answer_does_not_pick_a_winner_for_conflicting_values(self):
        knowledge = Knowledge()
        pages = [Page(f'https://example.com/{i}', 'Auto Loans',
                      '# Auto Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | ' + value + ' |',
                      '2026-05-12T00:00:00Z') for i, value in enumerate(('5.39%', '6.25%'))]
        knowledge.ingest(Snapshot('may', '2026-05-12', pages))
        # Two undated pages disagree, so no value is admitted by the fact
        # route; the answer falls back to evidence instead of picking a side.
        hits = (object(),)
        with patch.object(knowledge.answer_selector, 'retrieve', return_value=hits), \
             patch.object(knowledge.answer_selector, '_scope_evidence', return_value=hits), \
             patch.object(type(knowledge), '_render_retrieved_route',
                          return_value='evidence answer') as render:
            answer = knowledge.answer('APR for a 60-month auto loan?')
        self.assertEqual(answer, 'evidence answer')
        self.assertEqual(render.call_count, 1)
        self.assertEqual(len(knowledge.processor.store.observations), 2)

