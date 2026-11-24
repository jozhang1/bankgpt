"""Conservative ingestion routing; no model calls, source edits or Fact admission."""
from dataclasses import dataclass
import re

from ingestion.chunker import EvidenceChunker
from foundation.contract import Evidence
from foundation.entities import entity_label
from ingestion.extractor import FactExtractor
from facts.semantic_fact_profile import geographic_scope_benefit, semantic_benefit
from ingestion.membership_rules import eligibility_benefit, rate_benefit


@dataclass(frozen=True)
class SemanticExtractionDecision:
    should_extract: bool
    reason: str
    uncovered_block_indices: tuple[int, ...] = ()
    benefits: tuple[tuple[int, str], ...] = ()


class SemanticExtractionGate:
    """Route uncovered target facts, retaining excluded content in Evidence.

    This is a targeted extraction view, not a no-information-loss claim. Unknown
    prose remains retrievable even when it is outside the supported predicates.
    Coverage is explicit, whole-block and supplied by deterministic processing.
    """
    _LINK_ONLY = re.compile(r"\[([^\]]+)\]\(([^\n]+)\)")
    _CTA = frozenset({'apply', 'apply now', 'read more', 'learn more', 'contact us',
                      'join', 'join now', 'back to top', 'continue', 'cancel'})

    @staticmethod
    def _is_permission_matrix(text):
        """Verified role/action layout, outside current semantic Fact vocabulary.

        This is not deterministic Fact coverage. The intact table remains in
        Evidence; unknown layouts still receive ordinary target screening.
        """
        lines = text.strip().splitlines()
        if len(lines) < 3:
            return False
        headers = tuple(entity_label(cell) for cell in FactExtractor._cells(lines[0]))
        if headers != ('privileges', 'primary member', 'joint secondary owner'):
            return False
        for row in lines[2:]:
            cells = FactExtractor._cells(row)
            if len(cells) != 3 or not cells[0].strip():
                return False
            if any(not re.fullmatch(r'(?:Yes|No)(?:\s*\(.*\))?', cell.strip(), re.I)
                   for cell in cells[1:]):
                return False
        return True

    def _is_target(self, block, headings):
        text = block.text.strip()
        if text.casefold().startswith('image description:'):
            return False
        if block.kind == 'table' and self._is_permission_matrix(text):
            return False
        link = self._LINK_ONLY.fullmatch(text)
        if link:
            return None  # Contacts/navigation need deterministic handling, not model reasoning.
        return (geographic_scope_benefit(text)
                or eligibility_benefit(text, ' '.join(headings))
                or rate_benefit(text) or semantic_benefit(text, ' '.join(headings)))

    def __init__(self):
        self.parser = EvidenceChunker()

    def decide(
        self, evidence: Evidence, *, covered_quotes: tuple[str, ...] = (),
    ) -> SemanticExtractionDecision:
        if evidence.in_recommendations:
            return SemanticExtractionDecision(False, 'recommendation_region')
        uncovered = []
        benefits = []
        substantive = False
        headings = []
        seen_uncovered = False
        for i, block in enumerate(self.parser.parse_blocks(evidence.text)):
            if block.in_recommendations:
                continue
            if block.kind == 'heading':
                level = block.heading_level or 6
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, block.text))
                continue
            if block.kind == 'horizontal_rule':
                continue
            text = block.text.strip()
            if not text:
                continue
            link = self._LINK_ONLY.fullmatch(text)
            label = link.group(1) if link else text
            if label.casefold().strip(' .!') in self._CTA:
                continue
            substantive = True
            # Explicit full block coverage only, with a nonempty source quote.
            if any(text == quote.strip() for quote in covered_quotes if quote.strip()):
                continue
            seen_uncovered = True
            # Cell extraction remains deterministic. Explicit multi-phase
            # disclosures can request semantic binding to successfully built
            # cells; parsing failure alone still never earns a model call.
            if block.kind == 'table':
                benefit = rate_benefit(text)
                if benefit:
                    uncovered.append(i)
                    benefits.append((i, benefit))
                continue
            benefit = self._is_target(block, [title for _, title in headings])
            if benefit:
                uncovered.append(i)
                benefits.append((i, benefit))
        if uncovered:
            return SemanticExtractionDecision(True, 'positive_semantic_benefit', tuple(uncovered), tuple(benefits))
        reason = ('no_positive_semantic_benefit' if seen_uncovered else
                  'fully_covered' if substantive else 'no_substantive_content')
        return SemanticExtractionDecision(False, reason)
