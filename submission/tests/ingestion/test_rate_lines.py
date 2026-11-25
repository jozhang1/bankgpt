import unittest

from ingestion.rate_lines import rate_lines_to_tables


class RateLinesTests(unittest.TestCase):

    def test_whole_line_rates_become_one_table(self):
        out = rate_lines_to_tables(
            '# C\n\n6-month certificate APY: 3.50%\n\n12-month certificate APY: 3.10%\n')
        self.assertIn('| Term | APY |', out)
        self.assertIn('| 6 Months | 3.50% |', out)
        self.assertIn('| 12 Months | 3.10% |', out)

    def test_prose_mentions_are_untouched(self):
        text = 'See our 6-month certificate APY: 2.95% today.\n'
        self.assertEqual(rate_lines_to_tables(text), text)

    def test_code_fences_are_untouched(self):
        text = '```\n6-month certificate APY: 3.50%\n```\n'
        self.assertEqual(rate_lines_to_tables(text), text)


if __name__ == '__main__':
    unittest.main()
