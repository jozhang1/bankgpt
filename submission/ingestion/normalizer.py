"""Normalize extracted candidates while preserving their supporting evidence."""

import copy
import json
from dataclasses import replace
from decimal import Decimal
import re

from ingestion.extractor import CandidateFact, FactExtractor
from foundation.contract import EntityReference, Evidence
from facts.entity_normalizer import EntityNormalizer, EntityResolution
from foundation.products import PRODUCTS
from facts.qualifier_normalizer import CANONICAL_QUALIFIERS, QualifierNormalizer


# Re-export the existing numeric constants for current callers/builders.
from facts.attribute_normalizer import (
    AttributeNormalizer, RATE_ATTRIBUTES, AMOUNT_ATTRIBUTES, UNRESOLVED_ATTRIBUTE,
    SEMANTIC_ATTRIBUTES,
)


_SINGLE_PERCENT = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*%\s*(?:(?:APR|APY)\s*)?"
    r"(?:\d{1,2}|[¹²³†‡]|\\?\*)?",
    re.IGNORECASE,
)


_PERCENT_NUMBER = re.compile(r"(?P<value>\d+(?:\.\d+)?)")

_PERCENT_RANGE = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*%\s*[-–—]\s*"
    r"(?P<high>\d+(?:\.\d+)?)\s*%\s*(?:APR\s*)?"
    r"(?:\d{1,2}|[¹²³†‡]|\\?\*)?",
    re.IGNORECASE,
)
_CANONICAL_PERCENT_RANGE = re.compile(
    r"(?P<low>\d+(?:\.\d+)?)\s*[-–—]\s*(?P<high>\d+(?:\.\d+)?)"
)


def _percent_value(value: str) -> str:
    """Use at least two decimals, preserving any additional significant digits."""
    number = Decimal(value).normalize()
    return format(number, ".2f") if number.as_tuple().exponent > -2 else format(number, "f")


MIN_TO_OPEN_COLUMN = re.compile(
    r"column_\d+:\s*min\.?\s+to\s+open(?:\d{1,2})?",
    re.I,
)


def normalize_no_opening_minimum(
    candidate: CandidateFact,
    result: CandidateFact,
) -> CandidateFact | None:
    """Normalize explicit, mutually consistent None opening-minimum columns."""
    if result.attribute != "minimum_opening_balance":
        return None

    if result.value.strip().casefold() != "none":
        return None

    matching_values = [
        raw.strip().casefold()
        for key, raw in candidate.qualifiers.items()
        if MIN_TO_OPEN_COLUMN.fullmatch(key.strip())
        and isinstance(raw, str)
    ]

    if not matching_values:
        return None

    if any(value != "none" for value in matching_values):
        return None

    return replace(
        result,
        value="0",
        unit="USD",
        issues=[
            issue
            for issue in result.issues
            if issue != "unsupported_amount_value"
        ],
    )


class FactNormalizer:
    def __init__(
        self,
        entity_normalizer: EntityNormalizer | None = None,
        qualifier_normalizer: QualifierNormalizer | None = None,
        attribute_normalizer: AttributeNormalizer | None = None,
    ):
        self.attribute_normalizer = attribute_normalizer if attribute_normalizer is not None else AttributeNormalizer()
        self.entity_normalizer = (entity_normalizer if entity_normalizer is not None
                                  else EntityNormalizer(PRODUCTS))
        self.qualifier_normalizer = (
            qualifier_normalizer if qualifier_normalizer is not None
            else QualifierNormalizer()
        )

    def normalize(
        self, candidate: CandidateFact, *, evidence: Evidence | None = None,
    ) -> CandidateFact:
        """Return a normalized copy, preserving provenance.

        Values stay strings to match CandidateFact; Decimal canonicalizes scalar
        percentages and ranges without converting them to fractions. Unsupported values remain
        unchanged and flagged. Semantic labels are canonicalized but their values
        and qualifiers remain unchanged and marked pending. Raw input is not mutated.
        No key is generated here.
        """
        result = copy.deepcopy(candidate)
        resolution = self._entity(candidate, evidence=evidence)
        if evidence is not None:
            result.entity_refs = tuple(
                replace(reference, entity_id=self.entity_normalizer.resolve_reference(
                    reference.label, reference.quote, evidence,
                ))
                for reference in candidate.entity_refs
            )
            if (candidate.attribute in SEMANTIC_ATTRIBUTES
                    and resolution.status == "resolved"):
                result.entity_refs = self._add_explicit_entity_references(
                    result.entity_refs, candidate.quote, evidence,
                    subject_entity=resolution.entity_id,
                )
        if resolution.status == "resolved" and resolution.entity_id is not None:
            result.entity = resolution.entity_id
        else:
            # Keep the raw label and all IdentitySignals for later review.
            issue = f"entity_{resolution.status}"
            if issue not in result.issues:
                result.issues.append(issue)
        attribute = self.attribute_normalizer.normalize(candidate)
        if attribute is None and self.attribute_normalizer.is_rate_context(candidate):
            attribute = UNRESOLVED_ATTRIBUTE  # Preserve existing numeric fallback.
        if attribute is None:
            # Unknown non-rate predicates remain raw and explicitly unresolved.
            if "unresolved_attribute" not in result.issues:
                result.issues.append("unresolved_attribute")
            return result
        result.attribute = attribute
        result.issues = [issue for issue in result.issues if issue != "unresolved_attribute"]
        if attribute in RATE_ATTRIBUTES | AMOUNT_ATTRIBUTES | SEMANTIC_ATTRIBUTES:
            result.issues = [issue for issue in result.issues if issue != "unresolved_rate_metric"]
        elif "unresolved_rate_metric" not in result.issues:
            result.issues.append("unresolved_rate_metric")
        if attribute in SEMANTIC_ATTRIBUTES:
            if resolution.status != "resolved" and resolution.reason:
                detail = f"entity_resolution:{resolution.reason}"
                if detail not in result.issues:
                    result.issues.append(detail)
            result.qualifiers = self.qualifier_normalizer.normalize_semantic(
                candidate, issues=result.issues,
            )
            if attribute == 'eligibility_requirement':
                from ingestion.membership_rules import rule_issue
                if rule_issue(result.value, allow_discontiguous=True) is None:
                    result.value = json.dumps(json.loads(result.value), sort_keys=True,
                                              separators=(',', ':'), ensure_ascii=False)
            # Values and semantic Fact admission remain pending; no rate parsing.
            if "semantic_normalization_pending" not in result.issues:
                result.issues.append("semantic_normalization_pending")
            return result
        result.qualifiers = self._qualifiers(candidate, issues=result.issues)

        # Use only a footnote explicitly linked to this rate row by its marker.
        # That source-bound disclosure can establish categorical rate_type
        # without an LLM, while conflicting statements remain unresolved.
        if evidence is not None and result.attribute in RATE_ATTRIBUTES:
            disclosures = FactExtractor().rate_disclosure_quotes(
                evidence, candidate.quote,
            )[1:]
            declared_types = set()
            for disclosure in disclosures:
                plain = re.sub(r"[*_]", "", disclosure)
                if re.search(
                    r"\b(?:variable\s+(?:annual percentage rate|apr|interest rate|rate)|"
                    r"(?:annual percentage rate|apr|interest rate|rate)\s+is\s+variable)\b",
                    plain, re.I,
                ):
                    declared_types.add("variable")
                if re.search(
                    r"\b(?:fixed\s+(?:annual percentage rate|apr|interest rate|rate)|"
                    r"(?:annual percentage rate|apr|interest rate|rate)\s+is\s+fixed)\b",
                    plain, re.I,
                ):
                    declared_types.add("fixed")
            if len(declared_types) == 1:
                declared_type = next(iter(declared_types))
                existing = result.qualifiers.get("rate_type")
                if existing is None:
                    result.qualifiers["rate_type"] = declared_type
                elif existing != declared_type:
                    issue = "qualifier_conflict:rate_type"
                    if issue not in result.issues:
                        result.issues.append(issue)
                    result.qualifiers.pop("rate_type", None)
            elif len(declared_types) > 1:
                issue = "qualifier_conflict:rate_type"
                if issue not in result.issues:
                    result.issues.append(issue)
                result.qualifiers.pop("rate_type", None)

        value = candidate.value
        unit = candidate.unit
        issues = result.issues
        text = value.strip()
        if result.attribute in AMOUNT_ATTRIBUTES:
            normalized = normalize_no_opening_minimum(candidate, result)
            if normalized is not None:
                return normalized
            # '+' explicitly marks a minimum threshold in these columns.
            # Other prose and placeholders are not converted to zero.
            amount = re.fullmatch(
                r"\$\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\+?", text,
            )
            if amount is None and unit == "USD":
                amount = re.fullmatch(r"(\d+(?:\.\d+)?)", text)
            if amount is not None:
                result.value = format(Decimal(amount.group(1).replace(",", "")).normalize(), "f")
                result.unit = "USD"
            elif "unsupported_amount_value" not in issues:
                issues.append("unsupported_amount_value")
            return result
        interval = _PERCENT_RANGE.fullmatch(text) if unit in {"%", "percent"} else None
        if interval is None and unit == "percent":
            interval = _CANONICAL_PERCENT_RANGE.fullmatch(text)
        if interval is not None and Decimal(interval["low"]) <= Decimal(interval["high"]):
            result.value = f"{_percent_value(interval['low'])}–{_percent_value(interval['high'])}"
            result.unit = "percent"
            result.issues = [issue for issue in issues if issue != "unsupported_rate_value"]
            return result
        match = _SINGLE_PERCENT.fullmatch(text) if unit in {"%", "percent"} else None
        if match is None and unit == "percent":
            # Accept our own canonical output, but still validate the entire
            # numeric string. Bare values with unknown/% units remain unsupported.
            match = _PERCENT_NUMBER.fullmatch(text)
        if match is not None:
            value = _percent_value(match["value"])
            result.issues = [issue for issue in issues if issue != "unsupported_rate_value"]
            unit = "percent"
        elif "unsupported_rate_value" not in issues:
            issues.append("unsupported_rate_value")

        result.value = value
        result.unit = unit
        return result

    def _add_explicit_entity_references(
        self, references, quote: str, evidence: Evidence, *, subject_entity: str | None,
    ):
        """Bind exact, source-quoted mentions of known entities as references.

        This only records a relation when the referenced canonical alias occurs
        in the claim itself. It does not infer a target from page ownership or
        from a nearby paragraph. The primary entity is excluded.
        """
        import re

        result = list(references)
        seen = {(item.entity_id, item.quote) for item in result}
        records = (*self.entity_normalizer.entities.values(),
                   *self.entity_normalizer.products.values())
        for record in records:
            if record.id == subject_entity:
                continue
            aliases = sorted({record.name, *record.aliases}, key=lambda item: (-len(item), item.casefold()))
            for alias in aliases:
                words = re.findall(r"[^\W_]+", alias, re.UNICODE)
                if not words:
                    continue
                pattern = (r"(?<!\w)" + r"[\W_]*".join(re.escape(word) for word in words)
                           + r"(?!\w)")
                match = re.search(pattern, quote, re.IGNORECASE | re.UNICODE)
                if not match:
                    continue
                source_quote = match.group(0)
                # Resolve from the exact span, so aliases that collide remain
                # unresolved instead of creating an unsafe cross-Fact join.
                entity_id = self.entity_normalizer.resolve_reference(
                    alias, source_quote, evidence,
                )
                if entity_id is None:
                    continue
                key = (entity_id, source_quote)
                if key not in seen:
                    result.append(EntityReference(entity_id, alias, source_quote))
                    seen.add(key)
                break
        return tuple(result)

    def _entity(
        self, candidate: CandidateFact, *, evidence: Evidence | None = None,
    ) -> EntityResolution:
        return self.entity_normalizer.normalize(
            candidate, evidence=evidence, use_profile_entity_kind=True,
        )

    def _qualifiers(
        self, candidate: CandidateFact, *, issues: list[str] | None = None,
    ) -> dict[str, str | bool]:
        return self.qualifier_normalizer.normalize(candidate, issues=issues)
