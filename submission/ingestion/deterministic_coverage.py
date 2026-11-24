"""Whole-block coverage backed by successful deterministic cell construction."""
from ingestion.chunker import EvidenceChunker
from ingestion.extractor import FactExtractor


class DeterministicCoverage:
    # Other columns may contain unextracted business facts; don't hide them.
    CONDITION_HEADERS = frozenset({
        'product', 'product name', 'account', 'account type', 'card type',
        'loan type', 'loan program', 'term', 'certificate', 'program',
        'balance', 'balance tier', 'min. balance', 'max. balance', 'loan amount',
        'ira type', 'transaction type', 'points', 'model year',
        'platinum credit card', 'rewards credit card', 'business platinum visa',
        'business rewards visa',
    })

    def __init__(self):
        self.parser = EvidenceChunker()
        self.extractor = FactExtractor()

    def __call__(self, evidence, candidates, results):
        if len(candidates) != len(results):
            raise ValueError('Coverage requires one build result per candidate')
        successful = [c for c, result in zip(candidates, results)
                      if result[2].status == 'built' and c.evidence_id == evidence.id]
        covered = []
        for block in self.parser.parse_blocks(evidence.text):
            if block.kind != 'table' or block.in_recommendations:
                continue
            lines = block.text.strip().splitlines()
            if len(lines) < 3:
                continue
            headers = self.extractor._cells(lines[0])
            metrics = {j for j, h in enumerate(headers)
                       if self.extractor.METRIC.search(h) or self.extractor._minimum_attribute(h)}
            if not metrics or any(h.strip().casefold() not in self.CONDITION_HEADERS
                                  for j, h in enumerate(headers) if j not in metrics):
                continue
            complete = True
            expected = 0
            for row in lines[2:]:
                cells = self.extractor._cells(row)
                if len(cells) != len(headers):
                    complete = False
                    break
                for j in metrics:
                    raw = cells[j].strip()
                    # Empty cells/placeholders are not proof of a constructed value.
                    if not raw or raw in {'-', '–', '—'}:
                        complete = False
                        break
                    minimum = self.extractor._minimum_attribute(headers[j])
                    matches = [c for c in successful
                               if c.table_header == lines[0] and c.quote == row
                               and c.value.strip() == raw and (
                                   c.attribute == minimum if minimum else
                                   c.qualifiers.get('rate_column') == headers[j])]
                    if not matches:
                        complete = False
                        break
                    expected += 1
                if not complete:
                    break
            if complete and expected:
                covered.append(block.text.strip())
        return tuple(covered)
