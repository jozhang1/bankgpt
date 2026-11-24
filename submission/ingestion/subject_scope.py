
"""Bounded source-context validation for semantic fact subjects.

This is not general coreference resolution.

Accepted identity scopes:
1. A direct subject mention in the claim.
2. A preceding subject anchor in the same block.
3. An anchor in the immediately preceding heading or paragraph.
4. A uniquely named program/product H1 covering its article body.
5. A verified program card with one Application period bridge.
6. A uniquely named H1 anchoring a bounded descendant heading section.

All contextual scopes are checked for competing referents.
Semantic entailment remains the responsibility of the source validator.
"""

import re

from ingestion.chunker import EvidenceChunker
from ingestion.cleaner import PageCleaner
from foundation.entities import entity_label


def _visible(text: str) -> str:
    """Normalize visible text without treating link destinations as labels."""
    text = re.sub(r"!?\[([^\]\n]*)\]\([^)\n]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    return entity_label(text)


def contains_label(text: str, label: str) -> bool:
    key = entity_label(label)
    return bool(key and f" {key} " in f" {_visible(text)} ")


def _mentions(text, aliases):
    """Find maximal alias spans; preserve equal-span homonyms."""
    visible = " " + _visible(text) + " "
    spans = []

    for identity, labels in aliases.items():
        for key in {entity_label(label) for label in labels} - {""}:
            pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
            for match in re.finditer(pattern, visible):
                spans.append((match.start(), match.end(), identity))

    return {
        identity
        for start, end, identity in spans
        if not any(
            a <= start
            and end <= b
            and b - a > end - start
            for a, b, _ in spans
        )
    }


_NOUN_PATTERNS = {
    "branch": r"branches|branch|locations?",
    "atm": r"atms?|locations?",
    "location": r"branches|branch|atms?|locations?",
    "organization": (
        r"organizations?|cooperatives?|companies|company|"
        r"institutions?|credit unions?"
    ),
    "product": r"loans?|accounts?|cards?|certificates?|products?",
    "program": r"programs?",
    "service": r"services?|wallets?",
}


def _block_spans(text, blocks):
    """Map parsed blocks back to unique, ordered source occurrences."""
    spans = []
    cursor = 0

    for block in blocks:
        offset = text.find(block.text, cursor)

        if offset < 0:
            return None

        start = offset

        if block.kind == "heading":
            line_start = text.rfind("\n", 0, offset) + 1
            if re.fullmatch(r"\s*#{1,6}\s+", text[line_start:offset]):
                start = line_start

        spans.append((start, offset + len(block.text), block))
        cursor = offset + len(block.text)

    return spans


def _block_containing(spans, start, end):
    return next(
        (
            index
            for index, (block_start, block_end, _) in enumerate(spans)
            if block_start <= start and end <= block_end
        ),
        None,
    )


def _is_program_introduction(text):
    """Explicit launch or application-opening relation; not a bare mention."""
    return bool(re.search(
        r"\b(?:launch|introduc(?:e|es|ed|ing|tion))\b|"
        r"\bApplications?\s+(?:are|is)\s+(?:now\s+)?(?:being\s+)?accepted\s+for\b",
        text, re.I,
    ))


def _application_period_bridge(block) -> bool:
    if block.kind != "paragraph" or len(block.text) > 100:
        return False

    plain = block.text.replace("**", "").strip()

    return bool(
        re.fullmatch(
            r"Application period:\s*"
            r"(?:Open now|Closed|Opens [A-Za-z]+ \d{1,2}, \d{4})\.?",
            plain,
            re.I,
        )
    )


def _organization_heading_scope(
    text, spans, anchor_index, claim_index, claim_start, subject_id, aliases,
):
    """Build a bounded local scope under a uniquely named organization H1.

    The H1 identifies the document subject; only the current nested heading
    section and its content are checked for competing referents. This avoids
    treating every paragraph in a long page as one organization claim.
    """
    anchor = spans[anchor_index][2]
    claim = spans[claim_index][2]
    if (anchor.kind != "heading" or anchor.heading_level != 1
            or _mentions(anchor.text, aliases) != {subject_id}):
        return None

    intervening = [block for _, _, block in spans[anchor_index + 1:claim_index]]
    if any(block.kind == "heading" and block.heading_level == 1
           for block in intervening):
        return None
    if any(block.in_recommendations != claim.in_recommendations
           for block in [anchor, *intervening]):
        return None

    active_headings = []
    section_start = anchor_index
    for index in range(anchor_index + 1, claim_index):
        block = spans[index][2]
        if block.kind != "heading":
            continue
        level = block.heading_level or 6
        while active_headings and spans[active_headings[-1]][2].heading_level >= level:
            active_headings.pop()
        active_headings.append(index)
        if level <= 2:
            section_start = index

    # Carry the explicit organization title and active structural path, then
    # inspect only content in the current H1/H2 subsection before the claim.
    heading_context = [anchor.text]
    heading_context.extend(spans[index][2].text for index in active_headings)
    local_start = spans[section_start][0] if section_start > anchor_index else spans[anchor_index][1]
    local_content = text[local_start:claim_start].strip()
    if local_content:
        heading_context.append(local_content)
    return "\n".join(heading_context)


def _heading_scope_issue(
    text, spans, claim_index, subject_id, aliases, nouns,
):
    """Validate a heading's local subject scope using document structure.

    A heading may scope its descendant content regardless of the fact
    attribute or heading wording. It must have a uniquely identified H1
    ancestor, and its scope ends at a horizontal rule, recommendation boundary,
    or the next same/higher-level heading. Content outside that subtree is
    intentionally excluded from subject binding.
    """
    heading = spans[claim_index][2]
    fallback = (subject_id,)
    if heading.in_recommendations:
        return "subject_recommendation_scope_unverified", fallback

    level = heading.heading_level or 6
    has_content = False
    for _, _, block in spans[claim_index + 1:]:
        if block.in_recommendations != heading.in_recommendations:
            break
        if block.kind == "horizontal_rule":
            break
        if (block.kind == "heading" and block.heading_level is not None
                and block.heading_level <= level):
            break
        if block.kind in {"paragraph", "list", "table"} and block.text.strip():
            has_content = True

    # A title without descendant source content cannot establish a claim scope.
    if not has_content:
        return "subject_claim_scope_unverified", fallback

    preceding_h1 = [
        index for index, (_, _, block) in enumerate(spans[:claim_index])
        if block.kind == "heading" and block.heading_level == 1
    ]
    if not preceding_h1:
        return "subject_claim_scope_unverified", fallback
    anchor_index = preceding_h1[-1]
    anchor = spans[anchor_index][2]
    if _mentions(anchor.text, aliases) != {subject_id}:
        return "subject_claim_scope_unverified", fallback

    intervening = [block for _, _, block in spans[anchor_index + 1:claim_index + 1]]
    if any(block.in_recommendations != anchor.in_recommendations
           for block in intervening):
        return "subject_claim_scope_unverified", fallback
    # A new H1 would have become the nearest anchor above. Explicit visual
    # separators are allowed here because the verified H1 is the page identity;
    # recommendation boundaries remain a hard stop.
    if any(marker in text[spans[anchor_index][1]:spans[claim_index][0]]
           for marker in (PageCleaner.RECOMMENDATIONS_START,
                          PageCleaner.RECOMMENDATIONS_END)):
        return "subject_claim_scope_unverified", fallback

    heading_stack = [(1, anchor.text)]
    for _, _, block in spans[anchor_index + 1:claim_index + 1]:
        if block.kind != "heading":
            continue
        current_level = block.heading_level or 6
        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()
        heading_stack.append((current_level, block.text))
    context = "\n".join(title for _, title in heading_stack)
    issue = _context_issue(context, heading.text, subject_id, aliases, nouns)
    return issue


def _crosses_boundary(text, start, end):
    if start > end:
        return True

    between = text[start:end]

    if any(
        marker in between
        for marker in (
            PageCleaner.RECOMMENDATIONS_START,
            PageCleaner.RECOMMENDATIONS_END,
        )
    ):
        return True

    return any(
        EvidenceChunker.HORIZONTAL_RULE.fullmatch(line.strip())
        for line in between.splitlines()
    )


def _named_unregistered_subject(text, aliases, nouns):
    """Detect an unregistered named entity used as a sentence subject.

    Mentions inside descriptions, attributions, and affiliation phrases are
    not competing antecedents. Only a named phrase at the start of a local
    sentence/clause is strong enough to make the current subject ambiguous.
    """
    pattern = (
        r"\b(?:[A-Z][\w’'&-]*[ \t]+){1,5}"
        r"(?i:" + nouns + r")\b"
    )

    generic = {
        "the " + entity_label(noun)
        for noun in (
            "branch", "location", "atm", "loan", "account",
            "card", "program", "service", "organization",
        )
    }

    for match in re.finditer(pattern, text):
        label = match.group(0)

        if entity_label(label) in generic:
            continue

        if _mentions(label, aliases):
            continue

        # A named organization mentioned as an object, affiliation, or
        # descriptive complement does not compete with the current subject.
        # Require sentence/clause-initial position (allowing a conjunction)
        # before treating an unknown name as a possible antecedent.
        prefix = re.sub(
            r"!?\[([^\]\n]*)\]\([^)]*\)",
            r"\1",
            text[:match.start()],
        )
        prefix = re.sub(r"https?://\S+", "", prefix)
        local_prefix = re.split(r"[.!?;:\n]", prefix)[-1].strip()
        local_prefix = re.sub(r"^(?:and|but|or)\s+", "", local_prefix,
                              flags=re.IGNORECASE)
        if not local_prefix:
            return True

    return False


def _context_issue(context, claim_quote, subject_id, aliases, nouns):
    """Check competing referents within a proposed binding scope."""
    found = _mentions(context, aliases)

    if found != {subject_id}:
        return (
            "subject_antecedent_not_unique",
            tuple(sorted(found | {subject_id})),
        )

    visible = _visible(context)

    if re.search(
        r"\b(?:another|other|different|second)\s+(?:"
        + nouns + r")\b",
        visible,
    ):
        return (
            "subject_antecedent_not_unique",
            tuple(sorted(found)),
        )

    if _named_unregistered_subject(context, aliases, nouns):
        return (
            "subject_unregistered_antecedent",
            tuple(sorted(found)),
        )

    claim_mentions = _mentions(claim_quote, aliases)

    if claim_mentions - {subject_id}:
        return (
            "subject_antecedent_not_unique",
            tuple(sorted(claim_mentions | {subject_id})),
        )

    return None


def subject_scope_issue(
    text,
    claim_quote,
    anchor_quotes,
    subject_id,
    aliases,
    *,
    reference_kind=None,
    claim_block_index=None,
):
    """Return None if source-local identity is supported.

    Otherwise return (reason, candidate_entity_ids).

    `aliases` must already be restricted by the EntityNormalizer to the
    relevant entity kind and segment.

    A heading claim is validated against its own descendant content and
    heading hierarchy; no attribute-specific opt-in is required.

    `claim_block_index` is optional for compatibility with existing callers.
    When provided, it refers to EvidenceChunker.parse_blocks(text).
    """
    fallback = (subject_id,)

    nouns = _NOUN_PATTERNS.get(
        reference_kind,
        "|".join(_NOUN_PATTERNS.values()),
    )

    blocks = EvidenceChunker().parse_blocks(text)
    spans = _block_spans(text, blocks)

    if spans is None:
        return "subject_local_structure_unverified", fallback

    # Locate the claim. A known block index resolves duplicate claim text
    # without guessing which occurrence the extractor intended.
    if claim_block_index is not None:
        if (
            type(claim_block_index) is not int
            or not 0 <= claim_block_index < len(spans)
        ):
            return "subject_claim_scope_unverified", fallback

        claim_index = claim_block_index
        start, end, claim_block = spans[claim_index]

        if claim_quote not in claim_block.text:
            return "subject_claim_scope_unverified", fallback

        if claim_block.text.count(claim_quote) != 1:
            return "subject_claim_occurrence_ambiguous", fallback

        claim_start = text.find(claim_quote, start, end)
        if claim_start < 0:
            return "subject_claim_scope_unverified", fallback

    else:
        if not claim_quote or text.count(claim_quote) != 1:
            return "subject_claim_occurrence_ambiguous", fallback

        claim_start = text.index(claim_quote)
        claim_end = claim_start + len(claim_quote)

        claim_index = _block_containing(
            spans,
            claim_start,
            claim_end,
        )

        if claim_index is None:
            return "subject_claim_scope_unverified", fallback

        claim_block = spans[claim_index][2]

    claim_end = claim_start + len(claim_quote)

    if claim_block.kind == "heading":
        return _heading_scope_issue(
            text, spans, claim_index, subject_id, aliases, nouns,
        )

    if claim_block.kind not in {"paragraph", "list"}:
        return "subject_claim_scope_unverified", fallback

    if claim_block.in_recommendations:
        return "subject_recommendation_scope_unverified", fallback

    # Directly named subjects do not require antecedent resolution.
    # However, competing names make the identity ambiguous.
    claim_mentions = _mentions(claim_quote, aliases)

    starts_with_reference = bool(re.match(
        r"^(?:(?:this|that|the)\s+(?:" + nouns + r")\b|(?:it|they|its|their)\b)",
        _visible(claim_quote), re.I,
    ))
    if subject_id in claim_mentions and not starts_with_reference:
        if claim_mentions == {subject_id}:
            return None

        return (
            "subject_antecedent_not_unique",
            tuple(sorted(claim_mentions)),
        )

    contexts = []
    verified_anchors = list(dict.fromkeys(anchor_quotes))
    if reference_kind == "organization":
        # The source's own H1 is an independently visible document identity
        # anchor; it need not be repeated in a model-proposed subject_quote.
        verified_anchors.extend(
            block.text
            for _, _, block in spans[:claim_index]
            if block.kind == "heading"
            and block.heading_level == 1
            and _mentions(block.text, aliases) == {subject_id}
        )

    for quote in dict.fromkeys(verified_anchors):
        if not quote:
            continue

        # Keep anchors unambiguous. The optional claim block index
        # does not authorize choosing an arbitrary repeated anchor.
        if text.count(quote) != 1:
            continue

        anchor_start = text.index(quote)
        anchor_end = anchor_start + len(quote)

        # An antecedent must precede the claim.
        if anchor_end > claim_start:
            continue

        anchor_index = _block_containing(
            spans,
            anchor_start,
            anchor_end,
        )

        if anchor_index is None:
            continue

        block_start, _, anchor_block = spans[anchor_index]

        crosses_boundary = _crosses_boundary(text, anchor_end, claim_start)
        is_organization_document_anchor = (
            reference_kind == "organization"
            and anchor_block.kind == "heading"
            and anchor_block.heading_level == 1
            and _mentions(anchor_block.text, aliases) == {subject_id}
        )
        if crosses_boundary:
            # A verified organization H1 is document identity, not a local
            # antecedent. Markdown horizontal rules may separate page regions;
            # the actual claim check remains limited to its current subsection.
            recommendation_markers = (
                PageCleaner.RECOMMENDATIONS_START,
                PageCleaner.RECOMMENDATIONS_END,
            )
            between = text[anchor_end:claim_start]
            if (not is_organization_document_anchor
                    or any(marker in between for marker in recommendation_markers)):
                continue

        if anchor_block.kind not in {
            "heading", "paragraph", "list"
        }:
            continue

        if (
            anchor_block.in_recommendations
            != claim_block.in_recommendations
        ):
            continue

        allowed_local = anchor_index in {
            claim_index,
            claim_index - 1,
        }

        # A uniquely named program/product H1 may scope its article body.
        if (
            reference_kind in {"program", "product"}
            and anchor_index < claim_index
            and anchor_block.kind == "heading"
            and anchor_block.heading_level == 1
            and _mentions(anchor_block.text, aliases) == {subject_id}
        ):
            intervening = [
                block
                for _, _, block in spans[
                    anchor_index + 1:claim_index
                ]
            ]

            if all(
                block.kind != "heading"
                and block.in_recommendations
                == claim_block.in_recommendations
                for block in intervening
            ):
                allowed_local = True

        # An explicitly named organization H1 may identify the current
        # nested section. Unlike program/product article scope, this does not
        # cover the entire document: only the active heading path and local
        # subsection are checked for competing subjects.
        organization_context = None
        if reference_kind == "organization":
            organization_context = _organization_heading_scope(
                text, spans, anchor_index, claim_index, claim_start,
                subject_id, aliases,
            )
            if organization_context is not None:
                allowed_local = True

        # Explicit program-card structure:
        # program heading -> Application period -> claim.
        if (
            reference_kind == "program"
            and anchor_index == claim_index - 2
            and anchor_block.kind == "heading"
            and _mentions(anchor_block.text, aliases) == {subject_id}
        ):
            bridge = spans[claim_index - 1][2]

            if (
                bridge.in_recommendations
                == claim_block.in_recommendations
                and _application_period_bridge(bridge)
            ):
                allowed_local = True

        # An explicit launch/introduction paragraph may introduce one program
        # whose description continues through a short list and quotation.
        # Never let a generic mention establish article-wide scope.
        if (
            reference_kind == "program"
            and anchor_index < claim_index <= anchor_index + 8
            and anchor_block.kind == "paragraph"
            and _mentions(anchor_block.text, aliases) == {subject_id}
            and _is_program_introduction(anchor_block.text)
            and all(block.kind in {"paragraph", "list"}
                    and block.in_recommendations == claim_block.in_recommendations
                    for _, _, block in spans[anchor_index + 1:claim_index])
        ):
            allowed_local = True

        if not allowed_local:
            continue

        # Inspect the entire antecedent block and all intervening
        # source content, not only the short selected anchor quote.
        contexts.append(
            organization_context
            if organization_context is not None
            else text[block_start:claim_start]
        )

    if not contexts:
        return "subject_antecedent_not_local", fallback

    failures = set()

    for context in contexts:
        issue = _context_issue(
            context,
            claim_quote,
            subject_id,
            aliases,
            nouns,
        )

        if issue is None:
            return None

        reason, identities = issue

        if reason == "subject_unregistered_antecedent":
            return issue

        failures.update(identities)

    return (
        "subject_antecedent_not_unique",
        tuple(sorted(failures | {subject_id})),
    )
