import unittest

from foundation.entities import DUPACO_ENTITIES
from foundation.products import PRODUCTS
from retrieval.query_scope import (
    QueryScopeParser,
    match_named_records,
)


class SharedQueryParsingTests(unittest.TestCase):
    def test_product_alias_matching_preserves_equal_specificity_ties(self):
        matches = match_named_records(
            'business share savings account',
            PRODUCTS,
        )

        self.assertEqual(
            {record.id for record in matches},
            {'personal_share_savings', 'business_share_savings'},
        )

    def test_semantic_entity_matching_uses_registered_labels(self):
        matches = match_named_records(
            'Dupaco community credit union',
            DUPACO_ENTITIES,
        )

        self.assertEqual([record.id for record in matches], ['organization:dupaco'])

    def test_segment_is_shared_across_query_paths(self):
        parser = QueryScopeParser()
        self.assertEqual(parser.parse('business checking rates').segment, 'business')
        self.assertEqual(parser.parse('personal auto loan').segment, 'personal')
        self.assertIsNone(parser.parse('latest rates').segment)

    def test_rate_scope_still_parses_term_and_model_year(self):
        scope = QueryScopeParser().parse(
            'What is the APR for a 2024 auto loan for 60 months?',
            family='auto',
            attributes=('apr',),
        )

        self.assertEqual(scope.term_months, 60)
        self.assertEqual(scope.model_year, 2024)
        self.assertTrue(scope.supported, scope.unsupported)


if __name__ == '__main__':
    unittest.main()
