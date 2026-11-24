import unittest

from ingestion.chunker import EvidenceChunker


class EvidenceChunkerTests(unittest.TestCase):
    def setUp(self):
        self.chunker = EvidenceChunker()


    def test_sections_follow_heading_hierarchy(self):
        blocks = self.chunker.parse_blocks(
            "# Student Loans\n## Benefits\n### Flexible Repayment\n"
            "Defer payments.\n### Fees\nNo origination fees.\n## FAQ\nAnswers."
        )
        sections = self.chunker.build_sections(blocks)
        self.assertEqual([s.heading_path for s in sections], [
            ("Student Loans", "Benefits"),
            ("Student Loans", "FAQ"),
        ])

        self.assertEqual([b.kind for b in sections[0].blocks], [
            "heading", "paragraph", "heading", "paragraph",
        ])
        self.assertEqual(sections[0].blocks[0].text, "Flexible Repayment")


    def test_long_faq_keeps_answers_with_questions(self):
        sections = self.chunker.build_sections(self.chunker.parse_blocks(
            "## Frequently Asked Questions\n"
            "- [Can I defer payments?](https://example.com/#)\n\n"
            "Yes, while enrolled.\n\n- Who is eligible?\n\nEnrolled students."
        ))
        chunks = self.chunker.chunk_sections(sections, max_chars=40)
        self.assertEqual(len(chunks), 2)
        self.assertEqual([b.kind for b in chunks[0].blocks], ["list", "paragraph"])
        self.assertIn("Enrolled students.", chunks[1].blocks[-1].text)

    def test_large_table_retains_intro_and_notes(self):
        sections = self.chunker.build_sections(self.chunker.parse_blocks(
            "## Rates\nHigh-Yield Savings\n\n"
            "| Term | APY |\n| --- | --- |\n| 10 | 4% |\n\n"
            "Rates effective 09/01/2026.\n\nSubject to change."
        ))
        chunks = self.chunker.chunk_sections(sections, max_chars=30)
        self.assertEqual(len(chunks), 1)
        self.assertEqual([b.kind for b in chunks[0].blocks], [
            "paragraph", "table", "paragraph", "paragraph",
        ])


