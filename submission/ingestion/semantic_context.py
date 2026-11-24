"""Literal context within verified local block boundaries; never a new claim."""
import re
from ingestion.chunker import EvidenceChunker
from ingestion.subject_scope import _block_spans, _crosses_boundary, _is_program_introduction


def bound_context_quotes(candidate, evidence):
    """Keep the claim block, active headings, subject anchors and local context.

    No cross-Evidence joins, forward search, recommendation crossing, or date
    inference. Uniquely located source spans only; missing context stays missing.
    The reviewer must still prove each condition applies to the selected claim.
    """
    text = evidence.text
    blocks = EvidenceChunker().parse_blocks(text)
    spans = _block_spans(text, blocks)
    if not spans or not candidate.quote or text.count(candidate.quote) != 1:
        return ()
    offset = text.index(candidate.quote)
    index = next((i for i, (a, b, _) in enumerate(spans)
                  if a <= offset and offset + len(candidate.quote) <= b), None)
    if index is None or blocks[index].in_recommendations:
        return ()
    selected = {index}
    active = []
    boundary = index
    for i, block in enumerate(blocks[:index + 1]):
        if i and _crosses_boundary(text, spans[i - 1][1], spans[i][0]):
            active = []
            boundary = i
        if block.in_recommendations != blocks[index].in_recommendations:
            active = []
            boundary = i + 1
            continue
        if block.kind == 'heading':
            while active and blocks[active[-1]].heading_level >= block.heading_level:
                active.pop()
            active.append(i)
            boundary = i + 1
    selected.update(active)
    # A heading/paragraph immediately before the claim may establish a year,
    # award identity or condition. A list can inherit a year-only introduction.
    if index > boundary:
        selected.add(index - 1)
    if blocks[index].kind == 'list':
        i = index - 1
        while i >= boundary and blocks[i].kind == 'list':
            i -= 1
        if i >= boundary and re.fullmatch(r'\s*(?:In\s+)?(?:19|20)\d{2}(?:\s+alone)?\s*[:.]?\s*',
                                         blocks[i].text, re.I):
            selected.add(i)
    # Identity anchors already undergo EntityNormalizer locality validation.
    # Include their complete source block, never unrelated neighbouring blocks.
    for signal in candidate.identity_signals:
        if signal.kind != 'subject' or not signal.quote or text.count(signal.quote) != 1:
            continue
        start = text.index(signal.quote)
        anchor = next((i for i, (a, b, _) in enumerate(spans)
                       if a <= start and start + len(signal.quote) <= b), None)
        if anchor is not None and anchor <= index and not _crosses_boundary(text, spans[anchor][1], offset):
            if blocks[anchor].in_recommendations == blocks[index].in_recommendations:
                selected.add(anchor)
                if (anchor < index <= anchor + 8 and blocks[anchor].kind == 'paragraph'
                        and _is_program_introduction(blocks[anchor].text)
                        and all(b.kind in {'paragraph', 'list'} for b in blocks[anchor + 1:index])):
                    selected.update(range(anchor, index))
    return tuple(dict.fromkeys(blocks[i].text for i in sorted(selected)))
