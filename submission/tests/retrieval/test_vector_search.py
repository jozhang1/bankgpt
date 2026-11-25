import unittest

from retrieval.retrieval import EvidenceRetriever
from solution import Knowledge, Snapshot
from contract import Page


class FakeEncoder:
    def encode(self, texts, **_kwargs):
        vectors = []
        for text in texts:
            value = text.casefold()
            if value.startswith('how can i amend my member profile'):
                vectors.append([1.0, 0.0])
            elif 'account help center' in value:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class VectorSearchTests(unittest.TestCase):
    def test_vector_recall_finds_semantic_candidate_without_lexical_overlap(self):
        knowledge = Knowledge()
        knowledge.ingest(Snapshot('latest', '2026-09-28', [
            Page('https://example.com/member-help', 'Member help',
                 '# Member help\n\nUse the account help center to update contact details.',
                 '2026-09-28T00:00:00Z'),
            Page('https://example.com/business', 'Business lending',
                 '# Business lending\n\nContact our office about financing options.',
                 '2026-09-28T00:00:00Z'),
        ]))

        retriever = EvidenceRetriever(
            knowledge.processor.store,
            vector_encoder=FakeEncoder(),
        )
        retriever.build_bm25_index()
        hits = retriever.search('How can I amend my member profile?', top_k=2)

        self.assertTrue(hits)
        self.assertEqual(hits[0].evidence.source_url,
                         'https://example.com/member-help')

    def test_vector_index_is_limited_to_selected_snapshot(self):
        knowledge = Knowledge()
        page = Page('https://example.com/member-help', 'Member help',
                    '# Member help\n\nUse the account help center to update contact details.',
                    '2026-05-12T00:00:00Z')
        knowledge.ingest(Snapshot('may', '2026-05-12', [page]))
        knowledge.ingest(Snapshot('latest', '2026-09-28', [
            Page('https://example.com/new', 'New page',
                 '# New page\n\nUnrelated current content.',
                 '2026-09-28T00:00:00Z'),
        ]))

        retriever = EvidenceRetriever(
            knowledge.processor.store,
            vector_encoder=FakeEncoder(),
        )
        retriever.build_bm25_index()
        latest_hits = retriever.search('How can I amend my member profile?')
        historical_hits = retriever.search(
            'How can I amend my member profile?', as_of='2026-05-31',
        )

        self.assertNotIn('https://example.com/member-help',
                         [hit.evidence.source_url for hit in latest_hits])
        self.assertEqual(historical_hits[0].evidence.source_url,
                         'https://example.com/member-help')


if __name__ == '__main__':
    unittest.main()
