"""Explicit scalar contact/navigation rules, never a fallback to a model.

Unproven identity is retained as an extraction rejection. A trusted local
validator replays the rule and checks every field before FactBuilder admission.
"""
from copy import deepcopy
from html import unescape
import re
from urllib.parse import urljoin, urlsplit

from ingestion.chunker import EvidenceChunker
from foundation.entities import DUPACO_ENTITIES, entity_label
from facts.entity_normalizer import EntityNormalizer, EntityResolution
from ingestion.extractor import CandidateFact, IdentitySignal
from ingestion.cleaner import PageCleaner
from foundation.products import PRODUCTS
from facts.semantic_fact_profile import semantic_value_issue, DETERMINISTIC_FIELD_ATTRIBUTES
from foundation.source_urls import page_url


DETERMINISTIC_ATTRIBUTES = DETERMINISTIC_FIELD_ATTRIBUTES
# Explicit dataset issuer metadata. It binds institutional labels only; it
# must never assign arbitrary statistics, people or grant claims to the owner.
SOURCE_OWNERS = {'www.dupaco.com': 'organization:dupaco', 'dupaco.com': 'organization:dupaco'}


class DeterministicFieldExtractor:
    LINK = re.compile(r'(?<!!)\[(?P<label>[^\]\n]+)\](?P<destination>'
                      + PageCleaner._LINK_DESTINATION + r')')
    CONTACT_HEADINGS = frozenset({'contact', 'contact us', 'contact information',
                                  'we re here to help', 'general contact information'})
    SUBHEADINGS = frozenset({'contact', 'contact us', 'contact information',
                             'apply', 'apply now', 'rates', 'account rates', 'loan rates'})

    def __init__(self, *, entities=(), source_owners=None):
        self.entities = tuple({r.id: r for r in (*DUPACO_ENTITIES, *tuple(entities))}.values())
        self.records = {r.id: r for r in (*PRODUCTS, *self.entities)}
        self.owners = dict(SOURCE_OWNERS if source_owners is None else source_owners)
        self.parser = EvidenceChunker()
        self.last_rejections = ()
        self._key, self._candidates = None, ()

    @staticmethod
    def plain(text):
        text = re.sub(r'!?\[([^\]\n]*)\]\([^\n]*?\)', r'\1', text)
        return re.sub(r'[*_]', '', text).strip()

    @staticmethod
    def fingerprint(c):
        return (c.entity, c.attribute, tuple(sorted(c.qualifiers.items())), c.value,
                c.unit, c.evidence_id, c.quote, c.table_header, c.heading_path,
                c.effective_from, c.effective_to, c.date_quote, c.raw_date_notes,
                tuple(c.identity_signals))

    def _binding(self, headings, evidence, *, routing=False):
        owner = self.owners.get(urlsplit(evidence.source_url).hostname)
        if routing:
            if owner in self.records and self.records[owner].kind == 'organization':
                return owner, None
            return None
        segment = EntityNormalizer._url_segment(evidence.source_url)
        # Exact product heading, with only known neutral subheadings after it.
        for i in range(len(headings) - 1, -1, -1):
            label = entity_label(headings[i].text)
            matches = [p for p in PRODUCTS if label in {
                entity_label(a) for a in (p.name, *p.aliases)}
                and (segment is None or p.segment in {segment, 'shared'})]
            if matches:
                if (len(matches) != 1 or any(entity_label(h.text) not in self.SUBHEADINGS
                                             for h in headings[i + 1:])):
                    return None
                return matches[0].id, IdentitySignal('subject', matches[0].id, headings[i].text)
        owner = self.owners.get(urlsplit(evidence.source_url).hostname)
        if owner not in self.records or self.records[owner].kind != 'organization':
            return None
        institutional = (len(headings) == 1 and
                         entity_label(headings[0].text) in self.CONTACT_HEADINGS)
        if not (routing or institutional):
            return None
        # The claim quote is supplied later. This is a verified host-owner
        # binding, not a fabricated subject name inside the source wording.
        return owner, None

    def extract(self, evidence):
        key = (evidence.id, evidence.text, evidence.source_url, evidence.in_recommendations)
        if key != self._key:
            self._key = key
            self._candidates, self.last_rejections = self._extract(evidence)
        return deepcopy(list(self._candidates))

    def validate(self, candidate, evidence):
        if evidence.id != candidate.evidence_id or evidence.in_recommendations:
            return False
        return any(self.fingerprint(c) == self.fingerprint(candidate) for c in self.extract(evidence))

    def _extract(self, evidence):
        candidates, rejected, headings = [], [], []
        if evidence.in_recommendations:
            return (), ()
        for i, block in enumerate(self.parser.parse_blocks(evidence.text)):
            if block.in_recommendations:
                continue
            if block.kind == 'heading':
                while headings and headings[-1].heading_level >= block.heading_level:
                    headings.pop()
                headings.append(block)
                continue
            if block.kind not in {'paragraph', 'list'}:
                continue
            text = self.plain(block.text)

            def emit(attribute, value, *, routing=False):
                binding = self._binding(headings, evidence, routing=routing)
                issue = semantic_value_issue(attribute, value, None)
                if issue or binding is None:
                    rejected.append({'block_index': i, 'attribute': attribute,
                                     'reason': issue or 'subject_scope_unresolved', 'quote': block.text})
                    return
                identity, signal = binding
                if signal is None:
                    signal = IdentitySignal('source_url', evidence.source_url, block.text)
                candidates.append(CandidateFact(identity, attribute, {}, value, None,
                    evidence.id, block.text, '', tuple(h.text for h in headings),
                    identity_signals=(signal,)))

            # Anchored labels and explicit issuer "our" statements. A number
            # elsewhere, a recipient's bank, or a quoted example is not proof.
            routing = re.fullmatch(
                r'(?:[-+]\s*)?(?:Our\s+)?Routing Number\s*:\s*'
                r'(?:Provide the payer with our ABA number\s*)?\(?(\d{9})\)?\.?', text, re.I)
            incoming = re.fullmatch(
                r'[-+]\s*Simple incoming:\s*Receive funds by sharing our routing number '
                r'\((\d{9})\) and your account info\.', text, re.I)
            if routing or incoming:
                emit('routing_number', (routing or incoming)[1], routing=True)
            phone = re.fullmatch(r'(?:Phone|Telephone)\s*:\s*(.+)', text, re.I)
            email = re.fullmatch(r'(?:Email|Email address)\s*:\s*([^\s]+@[^\s]+)', text, re.I)
            if phone:
                value = phone[1].replace(',', '')
                # Remove punctuation only, preserving explicit country prefixes
                # and extensions. Never infer a country from a ten-digit number.
                value = re.sub(r'[(). -]', '', value) if not re.search(r'\b(?:ext|x)', value, re.I) else value
                emit('phone_number', value)
            if email:
                emit('email_address', email[1])
            for link in self.LINK.finditer(block.text):
                label, raw = link.group('label'), unescape(link.group('destination')[1:-1].strip())
                if '[' in label or not raw or re.search(r'\s', raw):
                    continue
                if raw.casefold().startswith('tel:') and re.fullmatch(r'(?:call(?: us)?|phone|telephone|[+\d(). -]+)', label, re.I):
                    # Labelled fields were already handled above; avoid two
                    # observations with formatting-only value differences.
                    if not phone:
                        emit('phone_number', re.sub(r'[(). -]', '', raw[4:]))
                    continue
                if raw.casefold().startswith('mailto:'):
                    if not email and re.fullmatch(r'(?:email(?: us)?|[^\s]+@[^\s]+)', label, re.I):
                        emit('email_address', raw[7:])
                    continue
                url = urljoin(evidence.source_url, raw)
                if page_url(url) is None:
                    continue
                if re.search(r'\b(?:rates|apr|apy)\b', label, re.I) and not re.search(r'\b(?:open|apply|application)\b', label, re.I):
                    emit('rates_url', url)
                elif re.search(r'\b(?:apply|application|open (?:my|an?|your))\b', label, re.I):
                    emit('application_url', url)
                elif re.fullmatch(r'join(?: dupaco)?', label.strip(), re.I):
                    emit('application_url', url)
                elif re.fullmatch(r'(?:branch or ATM|branch locator|ATM locator|find a branch|find an ATM)', label.strip(), re.I):
                    emit('locator_url', url)

        unique = {self.fingerprint(c): c for c in candidates}
        return tuple(unique.values()), tuple(rejected)


class DeterministicFieldEntityNormalizer(EntityNormalizer):
    """Source-rule identity proof is scoped to the deterministic builder only."""
    def __init__(self, rules):
        super().__init__(PRODUCTS, entities=rules.entities)
        self.rules = rules

    def normalize(self, candidate, *, evidence=None, **kwargs):
        if evidence is not None and self.rules.validate(candidate, evidence):
            record = self.rules.records[candidate.entity]
            return EntityResolution(candidate.entity, 'resolved', (candidate.entity,),
                'deterministic_field_rule', candidate.entity, entity_kind=self._entity_kind(record))
        return super().normalize(candidate, evidence=evidence, **kwargs)
