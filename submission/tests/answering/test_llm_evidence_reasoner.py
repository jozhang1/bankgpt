import unittest
from retrieval.retrieval import RetrievalHit

from solution import Knowledge, Snapshot
from contract import Page


class ReasonerTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def caller(request, timeout):
            self.calls.append(request)
            bundle = request['input']
            evidence = {e['id']: e for e in bundle['evidence']}
            return {'selections': [
                {'group_id': group['group_id'], 'status': 'preferred',
                 'supporting_quotes': [{'evidence_id': eid, 'quote': evidence[eid]['text']}
                                       for eid in group['evidence_ids']],
                 'observation_ids': group['observation_ids']}
                for group in bundle['groups']]}
        self.knowledge = Knowledge(caller)
        self.knowledge.ingest(Snapshot('may', '2026-05-12', [
            Page('https://example.com/auto', 'Auto Loans',
                 '# Auto Loans\n\n| Term | APR |\n| --- | --- |\n| 60 Months | 5.39% |',
                 '2026-05-12T00:00:00Z')]))


    def _prose_hits(self):
        evidence = next(iter(self.knowledge.processor.store.evidence.values()))
        self.knowledge.retriever.search = lambda *args, **kwargs: [RetrievalHit(evidence=evidence, score=1.0)]


    def test_invalid_quote_rejected(self):
        self.knowledge.reasoner.caller = lambda *_: {
            'selections': [{'group_id': 'answer', 'status': 'preferred', 'supporting_quotes': [
                {'evidence_id': 'invented', 'quote': 'fabricated'}]}]}
        self._prose_hits()
        answer = self.knowledge.answer('Tell me about auto loans')
        self.assertNotIn('fabricated', answer.text)
        self.assertFalse(answer.confident)
        self.assertIn('failed citation validation', answer.conflict_note)




if __name__ == '__main__':
    unittest.main()
