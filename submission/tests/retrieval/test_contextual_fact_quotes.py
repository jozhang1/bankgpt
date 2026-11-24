import unittest
from datetime import datetime, timezone
from uuid import uuid4

from facts.fact_matching import MatchStatus
from facts.semantic_support import SemanticSupport
from foundation.contract import Evidence, Fact, FactObservation
from foundation.store import KnowledgeStore
from retrieval.quote_render_guard import contextual_fact_quotes


class ContextualFactQuoteTests(unittest.TestCase):
    def setUp(self):
        self.store = KnowledgeStore()
        self.evidence = Evidence(
            uuid4(), 'snapshot', 'https://example.test/rates', 'Rates',
            'official_rate_table', False,
            '| Introductory Balance Transfer Rate | 2.90%1 |\n'
            '| Platinum Visa Rate | 12.50%–19.50%2 |\n'
            '1 The 2.90% introductory APR is valid for transfers posted during '
            'the first 90 days following the opening of the account. This '
            'applies to balance transfer transactions.',
            'hash', datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        self.store.evidence[self.evidence.id] = self.evidence
        self.primary = Fact(uuid4(), 'primary', 'platinum_visa', 'apr', {
            'phase': 'standard', 'transaction_type': 'purchase',
        })
        self.sibling = Fact(uuid4(), 'sibling', 'platinum_visa', 'apr', {
            'phase': 'introductory', 'transaction_type': 'balance_transfer',
        })
        self.store.facts[self.primary.id] = self.primary
        self.store.facts[self.sibling.id] = self.sibling
        self.store.facts_by_entity_attribute[('platinum_visa', 'apr')].extend(
            (self.primary.id, self.sibling.id),
        )
        primary_obs = FactObservation(
            uuid4(), self.primary.id, '12.50–19.50', 'percent', None, None,
            self.evidence.id,
        )
        sibling_obs = FactObservation(
            uuid4(), self.sibling.id, '2.90', 'percent', None, None,
            self.evidence.id,
        )
        self.store.observations[primary_obs.id] = primary_obs
        self.store.observations[sibling_obs.id] = sibling_obs
        self.store.obs_by_fact[self.primary.id].append(primary_obs.id)
        self.store.obs_by_fact[self.sibling.id].append(sibling_obs.id)
        self.store.semantic_support[sibling_obs.id] = SemanticSupport(
            claim_quote='| Introductory Balance Transfer Rate | 2.90%1 |',
            subject_quotes=(),
            context_quotes=(
                '1 The 2.90% introductory APR is valid for transfers posted '
                'during the first 90 days following the opening of the account. '
                'This applies to balance transfer transactions.',
            ),
        )
        self.primary_obs = primary_obs

    def test_returns_only_same_source_sibling_with_explicit_condition_difference(self):
        result = contextual_fact_quotes(
            self.primary, (self.primary_obs,),
            {'phase': MatchStatus.MATCH,
             'transaction_type': MatchStatus.MATCH},
            self.store, 'snapshot',
        )
        self.assertEqual(len(result), 1)
        fact, observation, evidence, _, disclosure, differing = result[0]
        self.assertEqual(fact.id, self.sibling.id)
        self.assertEqual(observation.value, '2.90')
        self.assertEqual(evidence.id, self.evidence.id)
        self.assertIn('first 90 days', disclosure)
        self.assertEqual(differing, ('phase', 'transaction_type'))

    def test_does_not_return_sibling_without_explicit_query_scope(self):
        self.assertEqual(contextual_fact_quotes(
            self.primary, (self.primary_obs,), {}, self.store, 'snapshot',
        ), ())


if __name__ == '__main__':
    unittest.main()
