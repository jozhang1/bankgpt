"""Parse explicit semantic query scope for Fact selection.

This is a bounded query parser, not an entailment classifier. Unsupported prose
can use Evidence retrieval. Reporting years never become as_of dates.
"""
from datetime import date
import re

from foundation.entities import DUPACO_ENTITIES, entity_label
from facts.fact_matching import FactMatch, MatchStatus
from foundation.products import PRODUCTS, Product
from facts.semantic_fact_profile import SEMANTIC_FACT_PROFILES
from retrieval.answer_selector import snapshot_scope, retain_non_mismatches
from retrieval.query_scope import QueryScopeParser, match_named_records


ATTRIBUTE_TERMS = {
    'branch_count': 'number branches branch offices office count',
    'supported_payment_wallet': 'mobile digital wallet card cards debit credit compatibility compatible payment supported support add use',
    'eligibility_requirement': 'membership member eligibility eligible join requirements charter area',
    'employee_count': 'number employees workforce employee count',
    'volunteer_hours': 'volunteer hours contributed logged',
    'recognition_count': 'recognition recognized award times',
    'ranking_position': 'ranking rank position',
    'maximum_assistance_amount': 'maximum grant assistance amount up to',
    'opening_date': 'branch location opening opened date',
    'closure_date': 'branch location closure closing closed date',
    'origination_fee': 'origination fee',
    'prepayment_fee': 'prepayment fee',
    'routing_number': 'routing number',
    'phone_number': 'phone number telephone',
    'email_address': 'email address',
    'application_url': 'application link URL apply',
    'rates_url': 'rates link URL',
    'locator_url': 'branch ATM locator link URL',
}


def selected_snapshot(store, as_of=None):
    scope = snapshot_scope(store, as_of)
    return scope.snapshot if scope is not None else None


def contains_wallet_request(text, facts):
    return any(f.attribute == 'supported_payment_wallet' and f.qualifiers.get('wallet_name')
               and f" {entity_label(str(f.qualifiers['wallet_name']))} " in f" {text} " for f in facts)


class SemanticQueryMatcher:
    def __init__(self, store, *, entities=DUPACO_ENTITIES):
        self.store = store
        self.records = {r.id: r for r in (*PRODUCTS, *DUPACO_ENTITIES, *tuple(entities))}

    def descriptor(self, fact):
        record = self.records.get(fact.entity)
        names = (record.name, *record.aliases) if record else (fact.entity,)
        return ' '.join((*names, fact.attribute.replace('_', ' '),
                         ATTRIBUTE_TERMS.get(fact.attribute, ''),
                         *(f'{k.replace("_", " ")} {v}' for k, v in sorted(fact.qualifiers.items()))))

    @staticmethod
    def attributes(question):
        return QueryScopeParser.semantic_attributes(question)

    def select_matches(self, question, *, as_of=None, reference_date=None,
                       relax_reporting_year=False, attributes=None):
        attrs = self.attributes(question) if attributes is None else set(attributes)
        text = entity_label(re.sub(r"[’']s\b", "", question))
        if not attrs or re.search(r'\b(?:compare|changed|difference|between)\b', text):
            return None
        kinds = {kind for attr in attrs for kind in SEMANTIC_FACT_PROFILES.get(attr,{}).get('entity_kinds',())}
        def subject_kind(record):
            return (record.kind if record.kind in {'service','program'} else 'product') if isinstance(record,Product) else record.kind
        named_records = match_named_records(
            re.sub(r"[’']s\b", "", question),
            self.records.values(),
            include=lambda record: not kinds or subject_kind(record) in kinds,
        )
        identities = {record.id for record in named_records}
        segment = 'business' if 'business' in text.split() else (
            'personal' if 'personal' in text.split() else None)
        if segment:
            identities = {identity for identity in identities
                          if getattr(self.records[identity], 'segment', segment) in {segment, 'shared'}}
        years = set(re.findall(r'\b(?:19|20)\d{2}\b', text))
        snapshot = selected_snapshot(self.store, as_of)
        relative = bool(re.search(r'\b(?:last|previous|this) year\b', text))
        provider = (re.search(r'\b(?:from|by) (.+?)(?=\s+(?:in|for|during)\b|$)', text)
                    if attrs.intersection({'recognition_count', 'ranking_position'}) else None)
        reference_value = as_of or reference_date or date.today().isoformat()
        if relative:
            reference = date.fromisoformat(str(reference_value))
            years.add(str(reference.year - (0 if 'this year' in text else 1)))
        if len(years) > 1:
            return None
        year = next(iter(years), None)
        result = []
        requested_location_kind = (
            'branch' if re.search(r'\bbranch(?:es)?\b', text) else
            'atm' if re.search(r'\batm(?:s)?\b', text) else None
        )
        # Only successfully admitted Semantic Observations participate. A raw
        # candidate or skipped audit is never a retrieval/answer assertion.
        admitted = {self.store.observations[oid].fact_id for oid in self.store.semantic_support
                    if oid in self.store.observations}
        for fact in self.store.facts.values():
            if fact.id not in admitted or fact.attribute not in attrs:
                continue
            if requested_location_kind and fact.attribute in {'opening_date', 'closure_date'}:
                record = self.records.get(fact.entity)
                actual_kind = getattr(record, 'kind', None)
                if actual_kind is None and ':' in fact.entity:
                    actual_kind = fact.entity.split(':', 1)[0]
                if actual_kind != requested_location_kind:
                    continue
            if identities and fact.entity not in identities:
                continue
            checks = [('entity', MatchStatus.MATCH if len(identities) == 1 else MatchStatus.UNKNOWN),
                      ('attribute', MatchStatus.MATCH)]
            q = fact.qualifiers
            if fact.attribute == 'supported_payment_wallet':
                debit,credit = 'debit' in text.split(),'credit' in text.split()
                requested = 'debit_and_credit' if debit and credit else 'debit' if debit else 'credit' if credit else None
                actual=q.get('card_type')
                if requested:
                    status=MatchStatus.MATCH if actual in {requested,'debit_and_credit'} else MatchStatus.UNKNOWN if actual is None else MatchStatus.MISMATCH
                    checks.append(('card_type',status))
                if q.get('wallet_name') and contains_wallet_request(text,self.store.facts.values()):
                    checks.append(('wallet_name',MatchStatus.MATCH if entity_label(str(q['wallet_name'])) in text else MatchStatus.MISMATCH))
            if fact.attribute == 'eligibility_requirement' and q.get('statement_topic') != 'membership':
                checks.append(('statement_topic',MatchStatus.UNKNOWN))
            if year is not None:
                key = 'ranking_year' if fact.attribute == 'ranking_position' else 'reporting_year'
                actual = q.get(key)
                if not relax_reporting_year:
                    checks.append((key, MatchStatus.UNKNOWN if actual is None else
                                   MatchStatus.MATCH if str(actual) == year else MatchStatus.MISMATCH))
            elif relative:
                checks.append(('reporting_year', MatchStatus.UNKNOWN))
            if year is None and re.search(r'\b(?:current|currently|today|now|latest)\b', text):
                key = 'ranking_year' if fact.attribute == 'ranking_position' else 'reporting_year'
                if key in q and snapshot is not None:
                    reference_year = str(date.fromisoformat(str(reference_value)).year)
                    if str(q[key]) != reference_year:
                        checks.append(('current_reporting_period', MatchStatus.UNKNOWN))
            if fact.attribute == 'branch_count' and q.get('location_scope') and not re.search(r'\b(?:in|within|outside)\b',text):
                checks.append(('location_scope',MatchStatus.UNKNOWN))
            geography = re.search(r'\b(?:in|within|outside) ([a-z]+(?: [a-z]+)?)$', text)
            if geography and geography.group(1) not in {'total', 'all'}:
                requested = geography.group(1)
                actual = q.get('location_scope')
                checks.append(('location_scope', MatchStatus.UNKNOWN if actual is None else
                               MatchStatus.MATCH if requested in entity_label(str(actual))
                               else MatchStatus.MISMATCH))
            # Explicit provider/award names must be supported by identity
            # conditions or by this fact's reviewed claim, not unrelated body.
            if provider and fact.attribute in {'recognition_count', 'ranking_position'}:
                spans = [str(q.get(k, '')) for k in ('award_name', 'ranking_name')]
                spans.extend(self.store.semantic_support[oid].claim_quote
                             for oid in self.store.obs_by_fact.get(fact.id, ())
                             if oid in self.store.semantic_support and snapshot is not None
                             and self.store.evidence[self.store.observations[oid].evidence_id].snapshot_id
                             == snapshot.snapshot_id)
                checks.append(('recognition_provider', MatchStatus.MATCH if any(
                    f' {provider.group(1)} ' in f' {entity_label(s)} ' for s in spans)
                    else MatchStatus.UNKNOWN))
            for word, key in (('overall', 'ranking_scope'), ('scheduled', 'event_status'),
                              ('planned', 'event_status'), ('actual', 'event_status')):
                if re.search(r'\b' + word + r'\b', text):
                    expected = 'scheduled' if word == 'planned' else word
                    actual = q.get(key)
                    checks.append((key, MatchStatus.UNKNOWN if actual is None else
                                   MatchStatus.MATCH if expected in entity_label(str(actual))
                                   else MatchStatus.MISMATCH))
            # Do not silently accept unparsed quantitative/geographic conditions.
            remainder = re.sub(r'\b(?:19|20)\d{2}\b', '', text)
            if re.search(r'\d|\b(?:only|except|excluding|within|outside)\b', remainder):
                checks.append(('unparsed_query_condition', MatchStatus.UNKNOWN))
            # A parser that recognizes "employee count" must not silently
            # discard "remote", "part-time", or another unmodelled restriction.
            grammar = set('what which who how many much is are was were do does did has have had '
                          'the a an of for from to in on at by with and or me tell give can '
                          'you your i my our we it its received receive used use reported '
                          'report provide provides available get become last previous this '
                          'year currently current today now latest overall scheduled planned '
                          'actual repeat'.split())
            if provider:
                grammar.update(provider.group(1).split())
            known = set(entity_label(self.descriptor(fact)).split()) | grammar
            if set(remainder.split()) - known:
                checks.append(('unparsed_query_scope', MatchStatus.UNKNOWN))
            match = FactMatch(fact, tuple(checks))
            if match.status is not MatchStatus.MISMATCH:
                result.append(match)
        # A fact about the wrong location kind must not suppress the ordinary
        # evidence path for a branch/ATM-specific event query.
        if requested_location_kind and not result:
            return None
        result = retain_non_mismatches(result)
        return sorted(result, key=lambda m: (m.fact.entity, m.fact.attribute, m.fact.key))
