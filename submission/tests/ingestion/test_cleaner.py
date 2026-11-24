import unittest

from ingestion.cleaner import PageCleaner


class PageCleanerTests(unittest.TestCase):
    def setUp(self):
        self.cleaner = PageCleaner()


    def test_preserves_links_tables_and_dates(self):
        raw = (
            "[Join Dupaco](https://join.dupaco.com/)\n\n"
            "| Term | APY |\n| --- | --- |\n| 10 | 4.00% |\n\n"
            "Rates are effective as of 09/01/2026."
        )
        self.assertEqual(self.cleaner.clean(raw), raw)


if __name__ == "__main__":
    unittest.main()
