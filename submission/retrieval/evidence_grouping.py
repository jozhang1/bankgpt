"""Deterministic groups retain same-Fact and cross-Fact competition edges."""
from hashlib import sha256
from retrieval.evidence_selection import EvidenceGroup


def build_groups(bundle: dict) -> tuple[EvidenceGroup, ...]:
    candidates = bundle['candidates']
    if not candidates:
        ids = tuple(sorted(e['id'] for e in bundle['evidence']))
        return (EvidenceGroup('answer', 'answer', (), 'complementary', ids),) if ids else ()
    by_fact = {}
    for candidate in candidates:
        by_fact.setdefault(candidate['fact']['id'], []).append(candidate)
    edges = {fid: set() for fid in by_fact}
    for fid, rows in by_fact.items():
        for row in rows:
            for other in row.get('competing_fact_ids', ()):
                if other in by_fact:
                    edges[fid].add(other)
                    edges[other].add(fid)
    # Preserve explicit pair edges rather than treating a connected component
    # as a claim that every pair within it competes.
    for pair in bundle.get('quote_relations', ()):
        a, b = pair['left_fact_id'], pair['right_fact_id']
        if (pair['relation'] == 'competing' or 'quote_selection_required' in pair.get('reasons', ())) \
                and a in edges and b in edges:
            edges[a].add(b)
            edges[b].add(a)
    groups, visited = [], set()
    for fid in sorted(by_fact):
        if fid in visited:
            continue
        component, todo = set(), [fid]
        while todo:
            current = todo.pop()
            if current in component:
                continue
            component.add(current)
            todo.extend(edges[current] - component)
        visited.update(component)
        rows = [row for key in sorted(component) for row in by_fact[key]]
        observation_ids = tuple(sorted(row['observation']['id'] for row in rows))
        evidence_ids = tuple(sorted({
            evidence_id
            for row in rows
            for evidence_id in (
                row['observation']['evidence_id'],
                *(citation['evidence_id']
                  for citation in (row.get('source_support') or {}).get('citations', ())),
            )
        }))
        uncertain = len(component) > 1 or any(row['temporal']['status'] != 'resolved' or row['match']['status'] == 'unknown' for row in rows)
        identity = sha256('|'.join(sorted(component)).encode()).hexdigest()[:24]
        fields = sorted({row['fact']['attribute'] for row in rows})
        groups.append(EvidenceGroup(identity, ','.join(fields), observation_ids,
                                    'competing' if uncertain else 'complementary', evidence_ids))
    return tuple(groups)
