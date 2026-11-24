"""Conservative semantic qualifiers alongside unchanged source-column metadata."""

from collections import defaultdict
from decimal import Decimal
import re
import unicodedata
from foundation.entities import entity_label

from facts.attribute_normalizer import attribute_key

from ingestion.extractor import CandidateFact


CANONICAL_QUALIFIERS = frozenset({
    "term_months", "model_year_from", "model_year_to",
    "loan_amount_min", "loan_amount_max",
    "loan_amount_min_inclusive", "loan_amount_max_inclusive",
    "balance_min", "balance_max", "balance_min_inclusive", "balance_max_inclusive",
    "rate_basis", "rate_type", "phase", "intro_months",
    "jumbo", "loan_purpose", "loan_program", "ira_type", "transaction_type",
    "mortgage_points",
})

# Separate vocabulary until semantic FactBuilder/key support is implemented.
SEMANTIC_QUALIFIERS = frozenset({
    "reporting_year", "ranking_year", "ranking_name", "ranking_scope",
    "award_name", "event_status", "channel", "eligibility",
    "fee_basis", "service_type", "location_scope", "product_variant", "statement_topic",
    "wallet_name", "card_type", "rule_scope", "rule_clause",
})
# Shared by extraction instructions and Builder admission; never infer missing
# identity dimensions just to satisfy these requirements.
SEMANTIC_REQUIRED_QUALIFIERS = {
    **{attribute: ("reporting_year",) for attribute in (
        "employee_count", "vto_user_count", "volunteer_hours",
    )},
    "ranking_position": ("ranking_name", "ranking_year", "ranking_scope"),
    "recognition_count": ("award_name", "reporting_year"),
    **{attribute: ("statement_topic",) for attribute in (
        "product_feature", "eligibility_requirement", "application_step",
        "repayment_option", "alternative_service_location", "supported_payment_wallet",
        "default_payment_option",
    )},
}

SEMANTIC_REQUIRED_QUALIFIERS["eligibility_requirement"] = (
    "statement_topic", "rule_scope", "rule_clause",
)
SEMANTIC_REQUIRED_QUALIFIERS["supported_payment_wallet"] = ("statement_topic", "wallet_name", "card_type")
SEMANTIC_REQUIRED_QUALIFIERS["default_payment_option"] = ("statement_topic", "card_type")

_SEMANTIC_ALIASES = {
    **{key: key for key in SEMANTIC_QUALIFIERS},
    "report_year": "reporting_year",
    "award_title": "award_name",
    "ranking_category": "ranking_scope",
    "application_channel": "channel",
}
_SEMANTIC_ENUMS = {
    'card_type': {'debit':'debit', 'credit':'credit', 'debit_and_credit':'debit_and_credit'},
    'rule_scope': {'quoted_statement':'quoted_statement'},
    "event_status": {"scheduled": "scheduled", "planned": "planned",
                     "actual": "actual", "completed": "completed",
                     "cancelled": "cancelled", "canceled": "cancelled"},
    "channel": {"online": "online", "in_person": "in_person",
                "phone": "phone", "mobile": "mobile"},
}


_NUMBER = r"\d[\d,]*(?:\.\d+)?"
_MONEY = rf"\$?\s*({_NUMBER})"
_TERM = re.compile(r"\b(\d+)\s*[- ]?\s*(months?|mo\.?|years?|yrs?\.?)\b", re.I)
_YEAR = re.compile(
    r"(?:Model Year\s+)?(\d{4})(?:\s*[-–—]\s*(\d{4}|\d{2}))?"
    r"\s*(?:Model Year)?\s*(\(and older\))?", re.I,
)


def _decimal(value: str) -> str:
    return format(Decimal(value.replace(",", "")).normalize(), "f")


def _bounds(text: str, prefix: str) -> dict[str, str | bool] | None:
    """Parse only complete amount expressions; no endpoints from nearby numbers."""
    text = text.strip()
    match = re.fullmatch(rf"{_MONEY}\s*[-–—]\s*{_MONEY}", text)
    if match:
        lower, upper = map(_decimal, match.groups())
        if Decimal(lower) > Decimal(upper):
            return None
        return {f"{prefix}_min": lower, f"{prefix}_max": upper,
                f"{prefix}_min_inclusive": True, f"{prefix}_max_inclusive": True}
    patterns = (
        (rf"{_MONEY}\s*\+", "min", True),
        (rf"{_MONEY}\s+(?:or more|and above)", "min", True),
        (rf"{_MONEY}\s+(?:or less|and below)", "max", True),
        (rf"(?:at least)\s+{_MONEY}", "min", True),
        (rf"(?:over|more than)\s+{_MONEY}", "min", False),
        (rf"(?:up to|at most)\s+{_MONEY}", "max", True),
        (rf"(?:under|less than)\s+{_MONEY}", "max", False),
    )
    for pattern, side, inclusive in patterns:
        match = re.fullmatch(pattern, text, re.I)
        if match:
            return {f"{prefix}_{side}": _decimal(match.group(1)),
                    f"{prefix}_{side}_inclusive": inclusive}
    return None


class QualifierNormalizer:
    def normalize_semantic(
        self, candidate: CandidateFact, *, issues: list[str] | None = None,
    ) -> dict[str, str | bool]:
        """Normalize explicit conditions, retaining raw keys and unknown meanings.

        No inference from quote, value, URL or current year. Event dates and
        reporting/ranking years never establish effective dates. Original keys
        remain available for auditing even when aliases conflict.
        """
        if issues is None:
            issues = []
        result = dict(candidate.qualifiers)
        proposals: dict[str, set[str]] = defaultdict(set)

        def flag(issue: str) -> None:
            if issue not in issues:
                issues.append(issue)

        for raw_key, raw in candidate.qualifiers.items():
            key = _SEMANTIC_ALIASES.get(attribute_key(raw_key))
            if key is None:
                flag(f"unparsed_qualifier:{raw_key}")
                continue
            if not isinstance(raw, str) or not raw.strip():
                flag(f"unparsed_qualifier:{key}")
                continue
            value = " ".join(unicodedata.normalize("NFKC", raw).split())
            if key in {"reporting_year", "ranking_year"}:
                if not re.fullmatch(r"[1-9][0-9]{3}", value):
                    flag(f"unparsed_qualifier:{key}")
                    continue
            elif key == 'wallet_name':
                value = entity_label(value)
            elif key in _SEMANTIC_ENUMS:
                normalized = _SEMANTIC_ENUMS[key].get(attribute_key(value))
                if normalized is None:
                    flag(f"unparsed_qualifier:{key}")
                    continue
                value = normalized
            # Names, scopes and eligibility expressions retain case and meaning.
            proposals[key].add(value)
        for key, values in proposals.items():
            if len(values) == 1:
                result[key] = next(iter(values))
            else:
                flag(f"qualifier_conflict:{key}")
        return result

    @staticmethod
    def _transaction_type(label: str) -> str | None:
        """Recognize explicit transaction labels, not loan purchase purposes."""
        match = re.fullmatch(
            r"(?:introductory\s+|standard\s+)?"
            r"(balance[ -]transfers?|purchases?|cash[ -]advances?)"
            r"(?:\s+(?:rate|apr))?",
            label.strip("*_ ").casefold(),
        )
        if match is None:
            return None
        label = match.group(1).replace("-", " ")
        if label.startswith("balance"):
            return "balance_transfer"
        if label.startswith("cash"):
            return "cash_advance"
        return "purchase"

    def normalize(
        self, candidate: CandidateFact, *, issues: list[str] | None = None,
    ) -> dict[str, str | bool]:
        """Add canonical conditions without discarding original column information.

        Monetary bounds use decimal strings, counts use integer strings, flags use
        booleans. Absent or uncertain conditions are omitted, not defaulted. Only
        CANONICAL_QUALIFIERS should be considered by future identity-key construction.
        """
        if issues is None:
            issues = []

        result: dict[str, str | bool] = {
            key: " ".join(value.split()) if isinstance(value, str) else value
            for key, value in candidate.qualifiers.items()
        }
        proposals: dict[str, set[str | bool]] = defaultdict(set)

        def flag(issue: str) -> None:
            if issue not in issues:
                issues.append(issue)

        def propose(key: str, value: str | bool) -> None:
            proposals[key].add(value)

        for key in CANONICAL_QUALIFIERS:
            if key in result:
                propose(key, result.pop(key))

        for key, raw in candidate.qualifiers.items():
            column = re.fullmatch(r"column_\d+:\s*(.+)", key)
            if column is None or not isinstance(raw, str):
                continue
            header = column.group(1).strip().casefold()
            text = " ".join(raw.split()).strip("*_ ")
            if header in {
                "transaction type", "card type", "platinum credit card",
                "rewards credit card", "business platinum visa", "business rewards visa",
            }:
                transaction_type = self._transaction_type(text)
                if transaction_type is not None:
                    propose("transaction_type", transaction_type)
                    if text.casefold().startswith("introductory "):
                        propose("phase", "introductory")
                elif header == "transaction type":
                    flag("unparsed_qualifier:transaction_type")
            if header == "points" and candidate.unit in {"%", "percent"}:
                # Points are a quote condition, not another rate metric.
                points = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%?", text)
                if points:
                    propose("mortgage_points", _decimal(points.group(1)))
                else:
                    flag("unparsed_qualifier:mortgage_points")
            # Tax treatment must be explicit; product names alone do not imply it.
            ira_prefix = re.match(r"^(traditional|roth):\s*", text, re.I)
            if ira_prefix:
                propose("ira_type", ira_prefix.group(1).casefold())
                text = text[ira_prefix.end():]
                # Verified IRA Type cells combine tax treatment with either
                # a deposit tier or a certificate term/variant. Reuse the
                # existing parsers while retaining the original source column.
                if header == "ira type" and text:
                    if text.startswith("$"):
                        header = "balance"
                    elif _TERM.search(text):
                        header = "term"
                    else:
                        flag("unparsed_qualifier:ira_conditions")
            elif header == "ira type":
                ira_type = re.fullmatch(r"(traditional|roth)(?:\s+IRA)?", text, re.I)
                if ira_type:
                    propose("ira_type", ira_type.group(1).casefold())
                else:
                    flag("unparsed_qualifier:ira_type")

            if "model year" in text.casefold():
                match = _YEAR.fullmatch(text)
                if match:
                    first = int(match.group(1))
                    end = match.group(2)
                    if end is None:
                        last = first
                    elif len(end) == 4:
                        last = int(end)
                    else:
                        last = first // 100 * 100 + int(end)
                    if match.group(3):
                        if last != first:
                            flag("unparsed_qualifier:model_year")
                        else:
                            propose("model_year_to", str(first))
                    else:
                        # Verified source layout also uses descending 2026-23.
                        propose("model_year_from", str(min(first, last)))
                        propose("model_year_to", str(max(first, last)))
                else:
                    flag("unparsed_qualifier:model_year")
                continue

            term_headers = {"term", "certificate", "program", "loan program", "product", "loan type", "account type", "balance tier"}
            terms = list(_TERM.finditer(text)) if header in term_headers else []
            if terms:
                if len(terms) != 1 or re.search(r"\d+\s*[-–—]\s*\d+\s*(?:months?|years?)", text, re.I):
                    flag("unparsed_qualifier:term_months")
                else:
                    match = terms[0]
                    months = int(match.group(1)) * (12 if match.group(2).casefold().startswith("y") else 1)
                    if months > 0:
                        propose("term_months", str(months))
                    else:
                        flag("unparsed_qualifier:term_months")

            if header in {"balance", "balance tier"} and not terms:
                if text not in {"", "—", "–", "-"}:
                    bounds = _bounds(text, "balance")
                    if bounds:
                        for field, value in bounds.items():propose(field, value)
                    else:
                        flag("unparsed_qualifier:balance")
            elif header in {"min. balance", "max. balance"}:
                amount = re.fullmatch(_MONEY, text)
                if amount:
                    side = "min" if header.startswith("min") else "max"
                    propose(f"balance_{side}", _decimal(amount.group(1)))
                    propose(f"balance_{side}_inclusive", True)
                else:
                    flag("unparsed_qualifier:balance")
            elif header == "loan amount" or (header == "loan type" and text.startswith("$")):
                bounds = _bounds(text, "loan_amount")
                if bounds:
                    for field, value in bounds.items():propose(field, value)
                else:
                    flag("unparsed_qualifier:loan_amount")

            if header in term_headers:
                if re.search(r"\bnon[- ]jumbo\b", text, re.I):propose("jumbo", False)
                elif re.search(r"\bjumbo\b", text, re.I):propose("jumbo", True)
                if re.search(r"\bfixed\b", text, re.I):propose("rate_type", "fixed")
                if re.search(r"\bvariable\b", text, re.I):propose("rate_type", "variable")
                if re.search(r"\bcash[- ]out\b", text, re.I):propose("loan_purpose", "cash_out")
                elif re.match(r"^purchase\s*:", text, re.I):propose("loan_purpose", "purchase")
                elif re.match(r"^refinance\s*:", text, re.I):propose("loan_purpose", "refinance")
                if re.search(r"\bconventional\b", text, re.I):propose("loan_program", "conventional")
                if re.search(r"\bmortgage rate relief program\b", text, re.I):propose("loan_program", "mortgage_rate_relief")
                if re.fullmatch(r"HELOC Intro Rate", text, re.I):propose("phase", "introductory")
                if re.match(r"^(?:purchase|refinance)\s*:", text, re.I):
                    amount = re.search(rf"(\$\s*{_NUMBER}\s+(?:or less|or more))\s*$", text, re.I)
                    if amount:
                        for field, value in (_bounds(amount.group(1), "loan_amount") or {}).items():propose(field, value)
            if re.fullmatch(r"Variable Interest Rate", text, re.I):propose("rate_type", "variable")

        rate_column = candidate.qualifiers.get("rate_column", candidate.attribute)
        if isinstance(rate_column, str):
            transaction_type = self._transaction_type(rate_column)
            if transaction_type is not None:
                propose("transaction_type", transaction_type)
                if rate_column.strip().casefold().startswith("introductory "):
                    propose("phase", "introductory")
            if re.search(r"\bas low as\b", rate_column, re.I):propose("rate_basis", "as_low_as")
            intro = re.search(r"\bfor\s+(?:the\s+)?(?:1st|first)\s+(\d+)\s+months?\b", rate_column, re.I)
            if intro:
                propose("phase", "introductory")
                propose("intro_months", str(int(intro.group(1))))
            if re.search(r"\bafter promotional period\b", rate_column, re.I):propose("phase", "post_promotion")

        for field, values in proposals.items():
            if len(values) == 1:
                result[field] = next(iter(values))
            else:
                flag(f"qualifier_conflict:{field}")
        for prefix in ("balance", "loan_amount"):
            low, high = result.get(f"{prefix}_min"), result.get(f"{prefix}_max")
            if isinstance(low, str) and isinstance(high, str) and Decimal(low) > Decimal(high):
                for key in (f"{prefix}_min", f"{prefix}_max", f"{prefix}_min_inclusive", f"{prefix}_max_inclusive"):
                    result.pop(key, None)
                flag(f"qualifier_conflict:{prefix}_bounds")
        return result
