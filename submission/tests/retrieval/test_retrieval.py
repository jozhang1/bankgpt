import unittest

from contract import Page, Snapshot
from ingestion.processor import KnowledgeProcessor
from retrieval.retrieval import EvidenceRetriever


class EvidenceRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.processor = KnowledgeProcessor()
        for sid, day, texts in (
            ("may", "2026-05-19", ["certificate oldrate", "routing unchanged"]),
            ("sep", "2026-09-28", ["certificate newrate", "routing unchanged", "routing unchanged"]),
        ):
            pages = [Page(f"https://example.com/{i}", "", text, day + "T00:00:00Z") for i, text in enumerate(texts)]
            self.processor.process(Snapshot(sid, day, pages))
        self.retriever = EvidenceRetriever(self.processor.store)
        self.retriever.build_bm25_index()

    def test_index_deduplicates_hashes_and_returns_eligible_sources(self):
        self.assertEqual(len(self.retriever._hashes), 3)
        hits = self.retriever.search("routing")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].evidence.snapshot_id, "sep")
        historical = self.retriever.search("routing", as_of="2026-05-31")
        self.assertEqual(historical[0].evidence.snapshot_id, "may")


    def test_empty_and_invalid_queries(self):
        self.assertEqual(self.retriever.search(""), [])
        self.assertEqual(self.retriever.search("unmatchedword"), [])
        self.assertEqual(self.retriever.search("routing", top_k=0), [])
        with self.assertRaises(ValueError):
            self.retriever.search("routing", as_of="not-a-date")
        lazy = EvidenceRetriever(self.processor.store)
        self.assertTrue(lazy.search("routing"))
        self.assertTrue(lazy._built)

