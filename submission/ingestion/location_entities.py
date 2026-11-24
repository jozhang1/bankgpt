"""Verified locator identities for semantic subjects; never register LLM names."""
from foundation.entities import EntityDefinition, entity_label
from facts.entity_normalizer import EntityNormalizer
from retrieval.location_index import LocationIndex, _normalize
from foundation.store import KnowledgeStore
from ingestion.subject_scope import contains_label


class LocationEntityNormalizer(EntityNormalizer):
    def __init__(self, products, *, entities=(), records=()):
        self.location_records = tuple(r for r in records if r.location_id and r.address)
        grouped = {}
        for record in self.location_records:
            grouped.setdefault(record.location_id, []).append(record)
        definitions = {e.id: e for e in entities}
        for identity, occurrences in grouped.items():
            names = sorted({r.name for r in occurrences})
            existing = definitions.get(identity)
            if existing is not None and existing.kind != occurrences[0].kind:
                raise ValueError('Conflicting location entity kind')
            definitions[identity] = EntityDefinition(identity, names[0], occurrences[0].kind,
                tuple(sorted(set(names[1:] + [r.address for r in occurrences]))))
        super().__init__(products, entities=definitions.values())

    def _subject_matches(self, label, kind):
        exact = super()._subject_matches(label, kind)
        if exact:
            return exact
        # Require a complete verified locator name (not city alone, street
        # fragments, or a fuzzy similarity score). City/address constrain it.
        named = [
            r for r in self.location_records
            if (kind is None or r.kind == kind)
            and contains_label(label, r.name)
            # Some locator cards use only a city for the ATM display name
            # (for example, name="Dubuque", city="Dubuque"). A city mention
            # in a separate announcement is not enough to identify that ATM.
            and not (r.city and _normalize(r.name) == _normalize(r.city)
                     and not (r.address and _normalize(r.address) in _normalize(label)))
        ]
        mentioned_cities = {entity_label(r.city) for r in self.location_records
                            if r.city and contains_label(label, r.city)}
        if mentioned_cities:
            named = [r for r in named if entity_label(r.city or '') in mentioned_cities]
        # An explicit city following "in" must match even when absent from
        # the known city vocabulary. Do not silently ignore unknown geography.
        import re
        suffix = re.search(r'\bin\s+([^,.;]+)\s*[.;]?$', label, re.I)
        if suffix:
            named = [r for r in named if _normalize(suffix.group(1)) == _normalize(r.city or '')]
        return [self.entities[identity] for identity in sorted({r.location_id for r in named})]


def location_normalizer(base, store, evidence_records):
    """Stage current evidence plus committed history without mutating the store."""
    staged = KnowledgeStore()
    staged.evidence = {**store.evidence, **{e.id: e for e in evidence_records}}
    index = LocationIndex(staged)
    index.build_index()
    return LocationEntityNormalizer(base.products.values(), entities=base.entities.values(),
                                    records=index.identity_records())
