"""Regression tests for behaviours that earlier evaluation runs got wrong."""

import unittest
from datetime import date

from solution import Knowledge, Snapshot
from contract import Page
from retrieval.answer_selector import snapshot_scope

QUERY = 'What is the APR for a 60-month auto loan?'


def _page(url, value, title='Auto Loans', fetched='2026-09-28T00:00:00Z'):
    return Page(url, title,
                f'# {title}\n\n| Term | APR |\n| --- | --- |\n| 60 Months | {value} |', fetched)


class AnswerRegressionTests(unittest.TestCase):

    def test_value_stated_on_several_pages_cites_each_page(self):
        # dup-017: the same 5.59% appears on the product page and on /rates.
        # Which page the validator happened to quote used to vary run to run.
        knowledge = Knowledge()
        knowledge.ingest(Snapshot('sep', '2026-09-28', [
            _page('https://example.com/personal/loans/auto-loans', '5.39%'),
            _page('https://example.com/rates', '5.39%'),
        ]))
        answer = knowledge.answer(QUERY)
        self.assertIn('5.39%', answer.text)
        cited = {source.url for source in answer.sources}
        self.assertEqual(cited, {'https://example.com/personal/loans/auto-loans',
                                 'https://example.com/rates'})

    def test_page_missing_from_the_latest_snapshot_is_not_answered_from_an_older_one(self):
        # The latest snapshot is the whole website: a page it no longer
        # contains must not resurrect its old value as the current answer.
        knowledge = Knowledge()
        knowledge.ingest(Snapshot('may', '2026-05-12', [
            _page('https://example.com/auto', '5.39%', fetched='2026-05-12T00:00:00Z'),
            _page('https://example.com/other', '7.00%', title='Other Loans',
                  fetched='2026-05-12T00:00:00Z'),
        ]))
        knowledge.ingest(Snapshot('sep', '2026-09-28', [
            _page('https://example.com/other', '7.00%', title='Other Loans'),
        ]))
        answer = knowledge.answer(QUERY)
        self.assertNotIn('5.39%', answer.text)
        self.assertFalse(any(source.snapshot_id == 'may' for source in answer.sources))

    def test_future_as_of_is_answered_from_the_latest_snapshot(self):
        # An as_of after every snapshot selects the newest one, never an older
        # value. Undated rates are not asserted for an explicit date, so the
        # answer may be unconfident, but it must not carry the stale value.
        knowledge = Knowledge()
        knowledge.ingest(Snapshot('may', '2026-05-12',
                                  [_page('https://example.com/auto', '5.39%',
                                         fetched='2026-05-12T00:00:00Z')]))
        knowledge.ingest(Snapshot('sep', '2026-09-28',
                                  [_page('https://example.com/auto', '6.25%')]))
        answer = knowledge.answer(QUERY, as_of='2099-01-01')
        self.assertNotIn('5.39%', answer.text)
        self.assertTrue(all(source.snapshot_id == 'sep' for source in answer.sources))

    def test_current_means_the_snapshot_date_not_the_wall_clock(self):
        knowledge = Knowledge()
        knowledge.ingest(Snapshot('sep', '2026-09-28', [_page('https://example.com/auto', '6.25%')]))
        scope = snapshot_scope(knowledge.processor.store, None)
        self.assertEqual(scope.reference_date, date(2026, 9, 28))
        self.assertEqual(scope.snapshot.snapshot_id, 'sep')


if __name__ == '__main__':
    unittest.main()
