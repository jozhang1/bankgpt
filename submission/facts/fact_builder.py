"""Deterministic construction from normalized candidates; no store mutations."""

from collections.abc import Callable, Iterable
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
import re
import unicodedata
from uuid import NAMESPACE_URL, uuid5

from foundation.contract import Evidence, Fact, FactBuildAudit, FactObservation
from foundation.entities import EntityDefinition
from facts.semantic_fact_profile import SEMANTIC_FACT_PROFILES, semantic_value_issue
from facts.entity_normalizer import EntityNormalizer
from facts.attribute_normalizer import SEMANTIC_ATTRIBUTES
from ingestion.extractor import CandidateFact
from ingestion.normalizer import AMOUNT_ATTRIBUTES, RATE_ATTRIBUTES
from foundation.products import PRODUCTS
from facts.qualifier_normalizer import CANONICAL_QUALIFIERS, SEMANTIC_QUALIFIERS, SEMANTIC_REQUIRED_QUALIFIERS, _SEMANTIC_ALIASES, _SEMANTIC_ENUMS
from facts.attribute_normalizer import attribute_key


# Keep these namespaces stable: changing either changes persisted identities.
FACT_NAMESPACE = uuid5(NAMESPACE_URL, "bankgpt:fact:v1")
OBSERVATION_NAMESPACE = uuid5(NAMESPACE_URL, "bankgpt:observation:v1")
_PRODUCT_IDS = frozenset(p.id for p in PRODUCTS)
_ALLOWED_ISSUES = frozenset({
    "effective_date_scope_uncertain",
    "cell_metric_disagrees_with_header",
})
_BOOLEAN_QUALIFIERS = frozenset({
    "jumbo", "balance_min_inclusive", "balance_max_inclusive",
    "loan_amount_min_inclusive", "loan_amount_max_inclusive",
})
_INTEGER_QUALIFIERS = frozenset({
    "term_months", "intro_months", "model_year_from", "model_year_to",
})
_DECIMAL_QUALIFIERS = frozenset({
    "balance_min", "balance_max", "loan_amount_min", "loan_amount_max",
    "mortgage_points",
})


def _serialize(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


class FactBuilder:
    """Build records from FactNormalizer output, without merging or persisting.

    build() returns (Fact, FactObservation, Audit), or (None, None, skipped Audit).
    Equal normalized identities produce equal Fact IDs; distinct Evidence IDs
    always produce distinct Observation IDs. A repository can reuse these IDs.
    Raw columns remain in the candidate/Evidence and never enter a Fact key.
    """

    def __init__(
        self, *, entities: Iterable[EntityDefinition] = (),
        semantic_validator: Callable[[CandidateFact, Evidence], bool] | None = None,
        entity_normalizer: EntityNormalizer | None = None,
    ):
        # Validator is a trusted ingestion component, never an LLM-provided flag.
        self.entity_normalizer = (entity_normalizer if entity_normalizer is not None
                                  else EntityNormalizer(PRODUCTS, entities=entities))
        self.semantic_validator = semantic_validator

    def build(
        self, candidate: CandidateFact, *, candidate_id: str | None = None,
        evidence: Evidence | None = None,
    ) -> tuple[Fact | None, FactObservation | None, FactBuildAudit]:
        issues = list(dict.fromkeys(candidate.issues))
        semantic = candidate.attribute in SEMANTIC_ATTRIBUTES
        reasons = (self._semantic_skip_reasons(candidate, evidence) if semantic
                   else self._skip_reasons(candidate))
        if reasons:
            return None, None, FactBuildAudit(
                candidate_id=candidate_id, evidence_id=candidate.evidence_id,
                fact_id=None, observation_id=None,
                issues=tuple(dict.fromkeys([*issues, *reasons])), status="skipped",
            )

        if semantic:
            issues = [issue for issue in issues if issue != "semantic_normalization_pending"]
        vocabulary = SEMANTIC_QUALIFIERS if semantic else CANONICAL_QUALIFIERS
        qualifiers = {k: candidate.qualifiers[k]
                      for k in sorted(vocabulary.intersection(candidate.qualifiers))}
        context = self._source_context(candidate)
        payload = {"entity": candidate.entity, "attribute": candidate.attribute,
                   "qualifiers": qualifiers}
        key = sha256(_serialize(payload).encode("utf-8")).hexdigest()
        fact_id = uuid5(FACT_NAMESPACE, key)
        observation_payload = {
            "fact_key": key,
            "evidence_id": str(candidate.evidence_id),
            "value": candidate.value,
            "unit": candidate.unit,
            "effective_from": candidate.effective_from.isoformat() if candidate.effective_from else None,
            "effective_to": candidate.effective_to.isoformat() if candidate.effective_to else None,
            "entity_refs": [
                {"entity_id": reference.entity_id, "label": reference.label,
                 "quote": reference.quote}
                for reference in candidate.entity_refs
            ],
            "context": context,
        }
        observation_id = uuid5(OBSERVATION_NAMESPACE, _serialize(observation_payload))
        return (
            Fact(id=fact_id, key=key, entity=candidate.entity,
                 attribute=candidate.attribute, qualifiers=qualifiers),
            FactObservation(id=observation_id, fact_id=fact_id,
                            value=candidate.value, unit=candidate.unit,
                            effective_from=candidate.effective_from,
                            effective_to=candidate.effective_to,
                            evidence_id=candidate.evidence_id,
                            entity_refs=tuple(candidate.entity_refs),
                            context=context),
            FactBuildAudit(candidate_id=candidate_id, evidence_id=candidate.evidence_id,
                           fact_id=fact_id, observation_id=observation_id,
                           issues=tuple(issues), status="built"),
        )

    @staticmethod
    def _source_context(candidate: CandidateFact) -> dict:
        """Keep source-local structure on the observation, never on Fact.key."""
        context = {}
        if candidate.heading_path:
            context["heading_path"] = tuple(candidate.heading_path)
        if candidate.table_header:
            context["table_header"] = candidate.table_header
        if candidate.quote:
            context["row_quote"] = candidate.quote
        if candidate.raw_date_notes:
            context["date_notes"] = tuple(candidate.raw_date_notes)
        if candidate.date_quote:
            context["date_quote"] = candidate.date_quote
        if candidate.identity_signals:
            context["identity_signals"] = tuple(
                {"kind": signal.kind, "label": signal.label, "quote": signal.quote}
                for signal in candidate.identity_signals
            )
        return context

    def _semantic_skip_reasons(
        self, candidate: CandidateFact, evidence: Evidence | None,
    ) -> list[str]:
        reasons = [issue for issue in candidate.issues
                   if issue not in _ALLOWED_ISSUES | {"semantic_normalization_pending"}]
        profile = SEMANTIC_FACT_PROFILES.get(candidate.attribute)
        if evidence is None or evidence.id != candidate.evidence_id:
            reasons.append("semantic_evidence_required")
        else:
            if evidence.in_recommendations:
                reasons.append("recommendation_evidence")
            if not candidate.quote.strip() or candidate.quote not in evidence.text:
                reasons.append("quote_not_in_evidence")
            if any(not reference.label.strip() or not reference.quote.strip()
                   or reference.quote not in evidence.text
                   for reference in candidate.entity_refs):
                reasons.append("entity_reference_not_source_bound")
            if any(reference.entity_id is not None
                   and self.entity_normalizer.resolve_reference(
                       reference.label, reference.quote, evidence,
                   ) != reference.entity_id
                   for reference in candidate.entity_refs):
                reasons.append("entity_reference_not_canonical")
            resolution = self.entity_normalizer.normalize(
                candidate, evidence=evidence, use_profile_entity_kind=True,
            )
            if resolution.status != "resolved" or resolution.entity_id != candidate.entity:
                reasons.append("entity_not_canonical")
            if (candidate.attribute == 'eligibility_requirement'
                    and candidate.qualifiers.get('rule_clause') != candidate.quote.strip()):
                reasons.append('rule_clause_must_equal_quote')
            if profile and profile['value_kind'] in {'rule_expression','support_status'}:
                issue = semantic_value_issue(candidate.attribute,candidate.value,candidate.unit)
                if issue:reasons.append(issue)
                if profile['value_kind'] == 'rule_expression':
                    from ingestion.membership_rules import rule_issue
                    issue = rule_issue(candidate.value, evidence_text=evidence.text,
                                       claim_quote=candidate.quote,
                                       allow_discontiguous=True)
                    if issue:reasons.append(issue)
                elif candidate.value == 'conditional' and not candidate.qualifiers.get('eligibility'):
                    reasons.append('conditional_support_requires_condition')
            if profile and resolution.status == 'resolved' and resolution.entity_kind not in profile['entity_kinds']:
                reasons.append('entity_kind_incompatible_with_attribute')
            # Quote presence is necessary, but does not prove predicate/value entailment.
            if self.semantic_validator is None:
                reasons.append("semantic_source_not_validated")
        for key, value in candidate.qualifiers.items():
            canonical = _SEMANTIC_ALIASES.get(attribute_key(key))
            if canonical is None:
                reasons.append(f"unknown_qualifier:{key}")
                continue
            normalized = " ".join(unicodedata.normalize("NFKC", value).split()) if isinstance(value, str) else None
            if normalized and canonical in _SEMANTIC_ENUMS:
                normalized = _SEMANTIC_ENUMS[canonical].get(attribute_key(normalized))
            if not normalized or candidate.qualifiers.get(canonical) != normalized:
                reasons.append(f"invalid_qualifier:{key}")
            if canonical in {"ranking_year", "reporting_year"} and (
                not normalized or not re.fullmatch(r"[1-9][0-9]{3}", normalized)
            ):
                reasons.append(f"invalid_qualifier:{canonical}")
        required = SEMANTIC_REQUIRED_QUALIFIERS.get(candidate.attribute, ())
        for key in required:
            if key not in candidate.qualifiers:
                reasons.append(f"missing_identity_qualifier:{key}")
        valid = isinstance(candidate.value, str) and bool(candidate.value.strip())
        if profile and profile['value_kind'] in {'integer', 'iso_date'}:
            valid = semantic_value_issue(
                candidate.attribute, candidate.value, candidate.unit,
            ) is None
        if not valid:
            reasons.append("value_not_normalized")
        for field in ("effective_from", "effective_to"):
            value = getattr(candidate, field)
            if value is not None and type(value) is not date:
                reasons.append(f"invalid_{field}")
        if (type(candidate.effective_from) is date and type(candidate.effective_to) is date
                and candidate.effective_from > candidate.effective_to):
            reasons.append("invalid_effective_date_range")
        if not reasons and self.semantic_validator(candidate, evidence) is not True:
            reasons.append("semantic_source_not_validated")
            validation = getattr(self.semantic_validator, 'last_result', None)
            validation_reasons = getattr(validation, 'reasons', ())
            if isinstance(validation_reasons, (tuple, list)):
                reasons.extend(f"semantic_validation:{reason}" for reason in validation_reasons)
        return reasons

    @staticmethod
    def _skip_reasons(candidate: CandidateFact) -> list[str]:
        reasons = [issue for issue in candidate.issues if issue not in _ALLOWED_ISSUES]
        if candidate.entity not in _PRODUCT_IDS:
            reasons.append("entity_not_canonical")
        if candidate.attribute not in RATE_ATTRIBUTES | AMOUNT_ATTRIBUTES:
            reasons.append("attribute_not_canonical")
        for key, value in candidate.qualifiers.items():
            if key not in CANONICAL_QUALIFIERS:
                if key != "rate_column" and not re.fullmatch(r"column_\d+:\s*.+", key):
                    reasons.append(f"unknown_qualifier:{key}")
                continue
            valid = isinstance(value, str) and bool(value.strip())
            if key in _BOOLEAN_QUALIFIERS:
                valid = type(value) is bool
            elif key in _INTEGER_QUALIFIERS:
                valid = isinstance(value, str) and bool(re.fullmatch(r"[1-9]\d*", value))
            elif key in _DECIMAL_QUALIFIERS:
                valid = isinstance(value, str) and bool(re.fullmatch(r"\d+(?:\.\d+)?", value))
            if not valid:
                reasons.append(f"invalid_qualifier:{key}")
        for prefix in ("balance", "loan_amount"):
            for side in ("min", "max"):
                bound, inclusive = f"{prefix}_{side}", f"{prefix}_{side}_inclusive"
                if (bound in candidate.qualifiers) != (inclusive in candidate.qualifiers):
                    reasons.append(f"incomplete_qualifier:{bound}")
            low, high = (candidate.qualifiers.get(f"{prefix}_{side}") for side in ("min", "max"))
            if (isinstance(low, str) and isinstance(high, str)
                    and re.fullmatch(r"\d+(?:\.\d+)?", low)
                    and re.fullmatch(r"\d+(?:\.\d+)?", high)
                    and Decimal(low) > Decimal(high)):
                reasons.append(f"invalid_qualifier:{prefix}_bounds")
        if candidate.attribute in RATE_ATTRIBUTES:
            valid = candidate.unit == "percent" and bool(re.fullmatch(
                r"\d+\.\d{2,}(?:–\d+\.\d{2,})?", candidate.value))
            if valid and "–" in candidate.value:
                low, high = map(Decimal, candidate.value.split("–"))
                valid = low <= high
            if not valid:
                reasons.append("value_not_normalized")
        elif candidate.attribute in AMOUNT_ATTRIBUTES:
            if candidate.unit != "USD" or not re.fullmatch(r"\d+(?:\.\d+)?", candidate.value):
                reasons.append("value_not_normalized")
        for field in ("effective_from", "effective_to"):
            value = getattr(candidate, field)
            if value is not None and type(value) is not date:
                reasons.append(f"invalid_{field}")
        if (type(candidate.effective_from) is date and type(candidate.effective_to) is date
                and candidate.effective_from > candidate.effective_to):
            reasons.append("invalid_effective_date_range")
        return reasons
