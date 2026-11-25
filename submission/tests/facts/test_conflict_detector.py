import unittest

from solution import Knowledge, Snapshot
from facts.conflict_detector import ConflictDetector
from contract import Conflict, Page, Source


def _page(url, title, value, when='2026-09-28T00:00:00Z', extra=''):
    return Page(url, title,
                f'# {title}\n\n{extra}| Term | APR |\n| --- | --- |\n| 60 Months | {value} |',
                when)


class ConflictDetectorTests(unittest.TestCase):

    def _conflict_for(self, knowledge, *values):
        for conflict in knowledge.conflicts():
            if all(v in conflict.values for v in values):
                return conflict
        return None

    def test_rate_page_beats_narrative_page_in_latest_snapshot(self):
        knowledge = Knowledge()
        knowledge.ingest(Snapshot('sep', '2026-09-28', [
            _page('https://www.example.com/personal/loans/auto-loans/', 'Auto Loans', '5.39%'),
            _page('https://www.example.com/newsroom/old-auto-promo', 'Auto Loans', '5.10%'),
        ]))
        conflict = self._conflict_for(knowledge, '5.39%', '5.10%')
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.resolution, '5.39%')
        self.assertIn('narrative page', conflict.reason)

    def test_pdf_rate_sheet_is_lower_authority_than_rate_page(self):
        knowledge = Knowledge()
        knowledge.ingest(Snapshot('sep', '2026-09-28', [
            _page('https://www.example.com/personal/loans/auto-loans/', 'Auto Loans', '5.39%'),
            _page('https://www.example.com/docs/auto-rates.pdf', 'Auto Loans', '5.20%'),
        ]))
        conflict = self._conflict_for(knowledge, '5.39%', '5.20%')
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.resolution, '5.39%')

    def test_two_equally_ranked_pages_stay_unresolved(self):
        knowledge = Knowledge()
        knowledge.ingest(Snapshot('sep', '2026-09-28', [
            _page('https://www.example.com/a/', 'Auto Loans', '5.39%'),
            _page('https://www.example.com/b/', 'Auto Loans', '6.25%'),
        ]))
        conflict = self._conflict_for(knowledge, '5.39%', '6.25%')
        self.assertIsNotNone(conflict)
        self.assertIsNone(conflict.resolution)

    def test_same_page_change_across_snapshots_still_reported_with_latest_winner(self):
        knowledge = Knowledge()
        url = 'https://www.example.com/personal/loans/auto-loans/'
        knowledge.ingest(Snapshot('may', '2026-05-12',
                                  [_page(url, 'Auto Loans', '5.39%', '2026-05-12T00:00:00Z')]))
        knowledge.ingest(Snapshot('sep', '2026-09-28',
                                  [_page(url, 'Auto Loans', '6.25%')]))
        conflict = self._conflict_for(knowledge, '5.39%', '6.25%')
        self.assertIsNotNone(conflict)
        self.assertEqual(conflict.resolution, '6.25%')

    def test_dated_version_change_is_reported_with_later_value(self):
        # Two dated, non-overlapping versions of one Fact must not be dropped.
        from unittest.mock import patch
        knowledge = Knowledge()
        url = 'https://www.example.com/personal/loans/auto-loans/'
        knowledge.ingest(Snapshot('may', '2026-05-12',
                                  [_page(url, 'Auto Loans', '3.90%', '2026-05-12T00:00:00Z')]))
        knowledge.ingest(Snapshot('sep', '2026-09-28',
                                  [_page(url, 'Auto Loans', '4.00%')]))
        with patch.object(ConflictDetector, '_temporal_relation', return_value='disjoint'):
            conflict = self._conflict_for(knowledge, '3.90%', '4.00%')
            self.assertIsNotNone(conflict)
            self.assertEqual(conflict.resolution, '4.00%')
            self.assertIn('resolved_version_difference', conflict.reason)
            with patch.dict('os.environ', {'BANKGPT_REPORT_VERSION_CHANGES': '0'}):
                self.assertIsNone(self._conflict_for(knowledge, '3.90%', '4.00%'))

    def test_assertion_not_yet_in_effect_is_not_the_current_value(self):
        from datetime import date
        from types import SimpleNamespace as NS
        current = NS(value='3.90', unit='percent', effective_from=date(2026, 9, 1), effective_to=None)
        upcoming = NS(value='4.00', unit='percent', effective_from=date(2026, 11, 1), effective_to=None)
        detector = ConflictDetector.__new__(ConflictDetector)
        value, winners, _ = detector._effective_date_resolution(
            [current, upcoming], reference=date(2026, 9, 28))
        self.assertEqual(value, '3.90%')
        self.assertEqual(winners, (current,))
        # Without a reference date the old behaviour (latest start wins) remains.
        value, _, _ = detector._effective_date_resolution([current, upcoming])
        self.assertEqual(value, '4.00%')
        # Nothing in effect yet: no winner is invented.
        value, winners, reason = detector._effective_date_resolution(
            [upcoming], reference=date(2026, 9, 28))
        self.assertIsNone(value)
        self.assertIn('none is in effect yet', reason)

    def test_authority_ranks(self):
        class Evidence:
            def __init__(self, url, source_type):
                self.source_url, self.source_type = url, source_type
        rank = ConflictDetector._authority
        self.assertEqual(rank(Evidence('https://x.com/newsroom/a', 'official_rate_table')), 0)
        self.assertEqual(rank(Evidence('https://x.com/a.pdf', 'unknown')), 0)
        self.assertEqual(rank(Evidence('https://x.com/calc', 'calculator')), 1)
        self.assertEqual(rank(Evidence('https://x.com/rates', 'official_rate_table')), 3)
        self.assertIsNone(rank(Evidence('https://x.com/about', 'unknown')))

    def test_dedupe_keeps_resolved_copy_once(self):
        sources = [Source(url='https://x.com/a', snapshot_id='s1'),
                   Source(url='https://x.com/b', snapshot_id='s1')]
        unresolved = Conflict(topic='t1', values=['1%', '2%'], sources=sources)
        resolved = Conflict(topic='t2', values=['2%', '1%'], sources=list(reversed(sources)),
                            resolution='2%', reason='r')
        other = Conflict(topic='t3', values=['3%', '4%'], sources=sources)
        result = ConflictDetector._dedupe([unresolved, resolved, other])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].resolution, '2%')

    def test_malformed_effective_interval_does_not_raise(self):
        from datetime import date
        from types import SimpleNamespace as NS
        bad = NS(effective_from=date(2026, 9, 1), effective_to=date(2026, 8, 1))
        ok = NS(effective_from=None, effective_to=None)
        self.assertEqual(ConflictDetector._temporal_relation(bad, ok), 'unknown')


if __name__ == '__main__':
    unittest.main()
