import unittest
from ...solution import Knowledge, Snapshot
from contract import Page


def locator(addresses):
    cards = []
    for address, kind, name in addresses:
        cards.append(f'### {name}\n\n{kind}\n\n{address}\n\nClosed\n\n24-hour ATM')
    return Page('https://www.dupaco.com/contact', 'Contact',
                '# Contact\n\n## Find a Location Near You\n\n' + '\n\n'.join(cards),
                '2026-09-28T00:00:00Z')


def announcement():
    return Page('https://www.dupaco.com/newsroom/waukee', 'Waukee announcement',
                '[Emily](https://www.dupaco.com/newsroom/author/emily)\nApril 26, 2024\n\n'
                '# Branch opening\n\nWAUKEE, Iowa—Dupaco opened a new branch, located at 1090 E. Hickman Road in Waukee.\n\n'
                'Background information.\n\nBranch hours are:\n\n**Lobby**\n\nMonday-Friday: 8:30 a.m.-5:30 p.m.\n\n'
                'Sunday: Closed\n\n**Drive-up**\n\nMonday-Friday: 8 a.m.-6 p.m.\n\nSunday: ATM only\n\n'
                '## Handpicked for You\n\nOther locations.', '2026-09-28T00:00:00Z')


class LocationIndexTests(unittest.TestCase):
    def setUp(self):
        self.knowledge = Knowledge()
        self.knowledge.ingest(Snapshot('latest', '2026-09-28', [locator([
            ('11375 Oldenburg Ln., Galena, IL, 61036', 'Dupaco Branch', 'Dupaco Credit Union'),
            ('3299 Hillcrest Rd., Dubuque, IA, 52001', 'Dupaco Branch', 'Dupaco Credit Union'),
            ('3999 Pennsylvania Ave., Dubuque, IA, 52002', 'Dupaco Branch', 'Dupaco Credit Union'),
            ('1000 Jackson St, Dubuque, IA, 52001', 'Surcharge-Free ATM', 'Dupaco Operations Center'),
        ]), announcement()]))


    def test_same_city_requires_disambiguation(self):
        result = self.knowledge.location_index.lookup('Address of the Dubuque branch?')
        self.assertEqual(result.status, 'ambiguous')
        self.assertEqual(len(result.records), 2)
        self.assertFalse(self.knowledge.answer('Address of the Dubuque branch?').confident)

    def test_atm_is_not_substituted_for_branch(self):
        self.assertEqual(self.knowledge.location_index.lookup('Address of the Operations Center branch?').status, 'not_found')
        self.assertEqual(self.knowledge.location_index.lookup('Address of the Operations Center ATM?').status, 'match')


