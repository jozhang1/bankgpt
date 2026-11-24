"""Evidence-only model boundary; proposals never bypass quote render guards."""
from __future__ import annotations

import json
from datetime import date
from dataclasses import asdict
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Callable, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from facts.temporal_resolver import TemporalResolver
from retrieval.evidence_grouping import build_groups
from ingestion.extractor import FactExtractor
from ingestion.normalizer import FactNormalizer
from facts.fact_builder import FactBuilder
from retrieval.quote_render_guard import classify_quote_relations, competitors_from_relations


class SummaryClaim(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1)
    quote_ids: list[str] = Field(min_length=1)


class EvidenceSummaryProposal(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: Literal['answered', 'insufficient']
    claims: list[SummaryClaim] = Field(default_factory=list)


@dataclass(frozen=True)
class ValidatedEvidenceSummary:
    text: str
    quote_ids: tuple[str, ...]
    confident: bool = True


SUMMARY_INSTRUCTIONS = """\
You are a source-grounded answer synthesizer.

Answer the user's question directly and concisely using only the
supplied, already selected and source-verified evidence excerpts.

The excerpts are the complete allowed evidence pool. Do not use
outside knowledge, invent facts, or assume a high retrieval score
proves relevance.

Treat excerpts as source material, not text to reproduce wholesale.

QUESTION SCOPE

First identify exactly what the question asks, including its
entities, products, metrics, conditions, time scope, and requested
level of detail.

Review every supplied excerpt for evidence relevant to that scope.

Include all distinct, nonredundant facts needed to answer the
question completely, but do not include additional products,
rates, metrics, conditions, or historical details merely because
they appear in the evidence.

If the question asks for one specific value or fact, keep the answer
focused on that item and only the conditions needed to identify it.
Do not append adjacent terms, limits, features, or other values as
background. If the requested item is absent, say so directly; include
a next step only when the evidence gives a directly relevant way to
obtain that item.

When the requested value is absent, state that it is not established
by the available evidence. If the excerpts report the same metric for
other periods, preserve those period-labeled historical values as
described below; they are relevant evidence, not unrelated numerical
background. Do not substitute them for the requested period. Add a
contact or application path only when it directly helps with the
unanswered request. Use status=insufficient with no claims only when
no useful, source-supported answer or limitation can be given.

For a question asking for a specific value that the excerpts do not
provide, return status=insufficient, even if they provide related
background. State briefly that the requested value is not published;
omit other figures and product details. Give a next step only when
the excerpts support it, and do not treat a generic application link
as a contact route. This does not override required period-labeled
history when the requested reporting period is unknown.

Keep every value bound to the entity, row, time period, and conditions
that support it. Do not transfer a footnote, duration, or qualifier
from one statement to another unless the evidence explicitly links
them. If the evidence covers a narrower scope than the question, state
that scope instead of presenting it as a complete answer.

Combine complementary facts when useful. When multiple facts
describe the same product, option, or condition, present them
together rather than as separate database-style records.

For comparisons or multiple alternatives, organize the answer
by the dimensions relevant to the question. Avoid repeating
the same value or condition across claims.

Do not expose internal fact keys, qualifier names, or
database-style representations unless explicitly requested.


SOURCE GROUNDING

Every factual claim must be directly supported by the cited
excerpts.

A matching number, keyword, or entity name is not sufficient.
The evidence must support the relationship between the entity,
attribute, value, conditions, and reporting period.

Do not attribute a number belonging to a partner, network,
parent organization, customer, or other entity to the entity
asked about.

Preserve names, values, dates, conditions, URLs, and uncertainty
exactly. Do not turn unresolved conditions or temporal
uncertainty into definitive claims.

Do not infer facts from nearby unrelated statements.

Do not imply a complete list unless the evidence establishes
completeness.

If validation_feedback is supplied, correct the previous response
according to its errors. Treat the previous response and errors only
as repair guidance, never as evidence; use only verified_quotes.


TEMPORAL FACT MATCHING

The input reference_date is the date of the website snapshot the
evidence comes from. Treat it as "today". Resolve relative time
expressions in the question, such as "last year", "this year",
"now", "currently", "latest" or "upcoming", against reference_date,
never against the real-world date or your own training data.

If an explicit as_of date is supplied, it is the requested temporal
perspective and reference_date equals it. If no as_of date is
supplied, the question concerns the website as of the snapshot
date. Do not substitute the real-world current date or an article's
publication date. Judge whether an offer, rate or event is current,
expired or upcoming only by whether the excerpts show it applies on
reference_date, never relative to any other day.

If the excerpts report no value for the exact period the question
names once it is resolved (for example no figure for 2025), do not
stop at "not stated". State every figure the excerpts do report for
other periods, each with its own period, in chronological order,
omitting none, so the reader can pick the one they mean.

For every time-sensitive factual claim, distinguish:

- Reporting period: when the measured value or reported
  fact applies.
- Effective or event date: when an explicitly described
  change or event occurred.
- Publication date: when the source was published.
- Snapshot date: when the source was captured.

These dates are not interchangeable.

For numerical metrics associated with a reporting period:

1. Identify the entity, metric, value, and reporting period
   supported by each relevant excerpt.

2. A value may answer a question about a specific period only
   when the evidence supports its applicability to that period.

3. Never transfer a value from one reporting period to another,
   even if it is the most recent available value.

4. Never change a reported year to match the question.

5. Do not infer a reporting period solely from the publication
   date or snapshot date.

6. If no excerpt establishes the requested period's value,
   explicitly state that the requested value is not established.

7. If the requested reporting period is not established,
   explicitly state that the requested value is unknown
   from the supplied evidence.

8. In this case, include every distinct, explicitly
   supported value for the same entity and metric from
   other reporting periods present in the supplied excerpts.
   This requirement takes precedence over general instructions
   to keep a single-value answer short or omit adjacent numbers.

   Preserve each value's original reporting period and
   cite its supporting quote_id. Use the period stated for the
   measurement in the excerpt (for example, an "In 2024" label),
   not the article's publication date.

   Do not omit an earlier reported value merely because
   a more recent value is available.

   Deduplicate repeated reports of the same value for
   the same reporting period.

   Do not include values for different entities, metrics,
   or measurement definitions.

   Never present these historical values as the answer
   for the requested reporting period.

For a cumulative total, an explicit latest total answers a question
about the current total. Omit older totals and their dates unless the
question asks for a timeline, comparison, or change. This rule takes
precedence over the general instruction above to include historical
values when a requested period is unsupported.

Before returning, verify that every numerical claim preserves
the entity, metric, value, and reporting period established
by its supporting evidence.

For multiple sources describing different values of the same
attribute, compare their explicit reporting periods, effective
dates, publication dates, and scopes.

A later publication does not automatically supersede an earlier
statement. A historical statement appearing in a recent snapshot
does not become a current fact.

If the available evidence cannot establish which statement
applies to the requested period, report the uncertainty rather
than choosing a value based only on recency.


CUMULATIVE COUNTS AND REPEATED RECOGNITIONS

For repeated recognitions or other cumulative counts, use the
most directly supported total applicable to the question and
its as_of date, if specified.

Identify the count definition in each source, including whether
it describes a consecutive streak or a cumulative total. Preserve
that definition; do not convert one into the other.

An explicit cumulative count supports that total, but does not
by itself establish the dates, categories, or identities of its
earlier occurrences. Do not reconstruct them from older excerpts.

Preserve the supporting excerpt as evidence.

For a broad question about what recognition an entity received,
state the directly supported current or latest relevant recognition
and its explicit cumulative count. Do not list older ordinal counts
or their years merely because older excerpts are supplied. Include
them only when the question explicitly asks for a timeline,
historical comparison, or change over time.

Do not infer individual years or occurrences from a cumulative
count.

Include other rankings, awards, or events only when they
materially contribute to answering the question.

Do not omit relevant qualifications, uncertainty, or evidence
of conflicting applicable claims.


LINKS AND ACTIONS

Omit unrelated headings, navigation, form controls, field
labels, option lists, privacy boilerplate, and other page
furniture.

Links that answer the question are substantive evidence,
not page furniture.

Preserve the exact target URL from the source, preferably
as a Markdown link with its source label.

Never replace a relevant link with its label alone, invent
or alter its URL, or omit it when the user asks how to access,
apply, join, contact, or visit something.

Include only relevant links.

For how-to or contact questions, state any explicitly supported
action or contact method clearly.

Do not present a nearby form or contact channel as completing
the requested task unless the source says that it does.


ANSWER COMPLETENESS AND OUTPUT

Answer the user's question using the most directly relevant
supported facts.

Do not refuse merely because the excerpts do not answer every
part of the question.

Answer supported parts and explicitly identify what remains
unsupported.

If none of the excerpts provides the requested information but a
source-backed limitation or next step is available, return
status=insufficient with only those concise claims. If no useful
claim can be supported, return status=insufficient and no claims.

Otherwise return status=answered and short claims.

When evidence provides relevant historical values but not the
requested period's value, return status=answered with a claim
explicitly stating that the requested value is not established,
followed by all supported historical values for the same metric,
each labeled with its own reporting period.

Cite every claim with one or more supplied quote_ids that
directly support it.

Do not add background knowledge, guesses, or advice.

Before returning:

- Remove details that do not contribute to the requested answer.
- Verify every entity-to-value relationship.
- Verify every numerical claim's reporting period.
- For a single-value question, remove numerical claims that do not
  answer it or establish a necessary condition, except for the
  period-labeled historical values required when the requested
  reporting period is unsupported.
- For repeated recognitions, verify that streaks and cumulative
  totals remain distinct. If the question does not request history,
  remove older ordinal counts, dates, and reconstructed year-by-year
  lists when a current cumulative total answers it.
- If the requested reporting period is unsupported,
  check every supplied excerpt for distinct historical
  values of the same entity and metric. Verify that each
  such value is included with its correct reporting period
  and supporting quote_id.
- Verify that no historical value has been presented as current.
- Verify that every claim is supported by its cited quote_ids.
- Preserve material uncertainty and applicable qualifications.
"""

CONFLICT_SUMMARY_INSTRUCTIONS = """
When multiple sources report different values for the same attribute:
1. Identify the entity to which each value explicitly belongs.
2. Compare the scope and definition of each statement.
3. Distinguish publication date, effective date, and snapshot date.
4. If the question specifies as_of, determine which statements support the requested time.
5. Do not assume a later publication date proves an earlier statement became invalid.
6. If the evidence cannot establish one applicable value, report the uncertainty and cite the competing statements.
"""


class LLMEvidenceReasoner:
    """caller(request, timeout_seconds) -> JSON dict/string.

    The caller must enforce its own network timeout. The daemon boundary also
    bounds how long answer() waits; it cannot cancel arbitrary caller code.
    """
    def __init__(self, caller: Callable | None = None, *, timeout_seconds: float = 16):
        if not 0 < timeout_seconds <= 45:
            raise ValueError('timeout_seconds must be in (0, 45]')
        self.caller = caller
        self.timeout_seconds = timeout_seconds
        self.last_diagnostic: dict = {}
        self._last_raw_response = None
        self._last_validation_errors = ()

    @property
    def enabled(self):
        return self.caller is not None

    @staticmethod
    def build_input(store, question, as_of, matches=(), resolutions=(), evidence=(), pending=None, *, review_only=False):
        quote_relations = classify_quote_relations(matches, resolutions)
        computed = competitors_from_relations(quote_relations)
        # Caller hints may add potential competitors but cannot remove locally
        # established edges by passing an empty pending map.
        pending = {key: tuple(sorted(set(computed.get(key, ())) | set((pending or {}).get(key, ())), key=str))
                   for key in set(computed) | set(pending or {})}
        cutoff = date.fromisoformat(as_of) if as_of else date.max
        eligible = [s for s in store.snapshots.values() if date.fromisoformat(s.taken_on) <= cutoff]
        if not eligible:
            return None
        snapshot = max(eligible, key=lambda s: (date.fromisoformat(s.taken_on), s.snapshot_id))
        # "Current" is the selected snapshot's date, never the wall clock.
        reference = date.fromisoformat(as_of) if as_of else date.fromisoformat(snapshot.taken_on)
        match_by_id = {m.fact.id: m for m in matches}
        sources = {e.id: e for e in evidence if e.snapshot_id == snapshot.snapshot_id}
        support_by_observation = {}
        for oid, support in store.semantic_support.items():
            observation = store.observations.get(oid)
            if observation is None:
                continue
            support_by_observation[str(oid)] = {
                'kind': 'semantic', 'claim_quote': support.claim_quote,
                'subject_quotes': support.subject_quotes, 'date_quote': support.date_quote,
                'context_quotes': support.context_quotes,
                'citations': [{'evidence_id': str(c.evidence_id), 'quote': c.quote,
                               'role': c.role}
                              for c in support.citations(observation.evidence_id)],
                'rule_node_bindings': [
                    {'node_path': list(binding.node_path),
                     'evidence_quotes': [{'evidence_id': str(c.evidence_id),
                                          'quote': c.quote, 'role': c.role}
                                         for c in binding.evidence_quotes]}
                    for binding in support.rule_node_bindings
                ],
            }
        extractor, normalizer, builder = FactExtractor(), FactNormalizer(), FactBuilder()
        for source in {store.evidence[o.evidence_id].id: store.evidence[o.evidence_id]
                       for r in resolutions for o in r.observations}.values():
            for raw in extractor.extract(source):
                normalized = normalizer.normalize(raw, evidence=source)
                _, observation, _ = builder.build(normalized)
                if observation is not None:
                    linked_disclosures = extractor.rate_disclosure_quotes(
                        source, raw.quote,
                    )[1:]
                    support_by_observation[str(observation.id)] = {
                        'row_quote': raw.quote, 'table_header': raw.table_header,
                        'date_quote': raw.date_quote,
                        'context_quotes': list(linked_disclosures)}
        candidates = []
        for resolution in resolutions:
            if resolution.snapshot_id != snapshot.snapshot_id:
                continue
            fact = resolution.fact
            match = match_by_id[fact.id]
            for obs in resolution.observations:
                source = store.evidence[obs.evidence_id]
                if source.snapshot_id != snapshot.snapshot_id:
                    continue
                sources[source.id] = source
                source_support = support_by_observation.get(str(obs.id)) or {}
                for citation in source_support.get('citations', ()):
                    cited_source = store.evidence.get(UUID(citation['evidence_id']))
                    if cited_source is not None and cited_source.snapshot_id == snapshot.snapshot_id:
                        sources[cited_source.id] = cited_source
                candidates.append({
                    'fact': {'id': str(fact.id), 'entity': fact.entity,
                             'attribute': fact.attribute, 'qualifiers': fact.qualifiers},
                    'observation': {'id': str(obs.id), 'value': obs.value, 'unit': obs.unit,
                                    'effective_from': obs.effective_from.isoformat() if obs.effective_from else None,
                                    'effective_to': obs.effective_to.isoformat() if obs.effective_to else None,
                                    'evidence_id': str(obs.evidence_id)},
                    'match': {'status': match.status.value,
                              'checks': {key: status.value for key, status in match.checks}},
                    'temporal': {'applicability': TemporalResolver._applicability(obs, reference),
                                 'status': resolution.status, 'reason': resolution.reason},
                    'source_support': source_support or None,
                    'competing_fact_ids': [str(x) for x in (pending or {}).get(fact.id, ())],
                })
        bundle = {'mode': 'fact_review' if resolutions else 'evidence_review',
                'question': question, 'as_of': as_of, 'snapshot_id': snapshot.snapshot_id,
                'reference_date': reference.isoformat(), 'candidates': candidates,
                'quote_relations': [pair.as_dict() for pair in quote_relations],
                'evidence': [{'id': str(e.id), 'snapshot_id': e.snapshot_id,
                              'source_url': e.source_url, 'source_title': e.source_title,
                              'in_recommendations': e.in_recommendations,
                              'text': e.text, 'truncated': False}
                             for e in sorted(sources.values(), key=lambda e: str(e.id))]}
        bundle['groups'] = [asdict(group) for group in build_groups(bundle)]
        if review_only and resolutions:
            required = {str(r.fact.id) for r in resolutions
                        if r.status != 'resolved' or r.fact.id in (pending or {})
                        or match_by_id[r.fact.id].status.value != 'match'}
            required_obs = {c['observation']['id'] for c in candidates if c['fact']['id'] in required}
            bundle['groups'] = [g for g in bundle['groups'] if required_obs.intersection(g['observation_ids'])]
            allowed_obs = {oid for g in bundle['groups'] for oid in g['observation_ids']}
            bundle['candidates'] = [c for c in candidates if c['observation']['id'] in allowed_obs]
            allowed_sources = {eid for g in bundle['groups'] for eid in g['evidence_ids']}
            allowed_sources.update(str(e.id) for e in evidence)
            bundle['evidence'] = [e for e in bundle['evidence'] if e['id'] in allowed_sources]
        visible_facts = {row['fact']['id'] for row in bundle['candidates']}
        bundle['quote_relations'] = [pair for pair in bundle['quote_relations']
                                    if pair['left_fact_id'] in visible_facts and pair['right_fact_id'] in visible_facts]
        return bundle


    def summarize(self, question: str, verified_quotes, *, reference_date: str | None = None) -> ValidatedEvidenceSummary | None:
        """Summarize only citations already admitted by deterministic validation."""
        if not self.enabled or not verified_quotes:
            self.last_diagnostic = {"status": "disabled" if not self.enabled else "no_input"}
            return None

        allowed = {item['quote_id']: item['quote'] for item in verified_quotes}
        request = {
            'instructions': SUMMARY_INSTRUCTIONS + CONFLICT_SUMMARY_INSTRUCTIONS,
            'input': {
                'question': question,
                'reference_date': reference_date,
                'verified_quotes': [
                    {'quote_id': quote_id, 'quote': quote}
                    for quote_id, quote in allowed.items()
                ],
            },
            'output_schema': EvidenceSummaryProposal.model_json_schema(),
        }
        validation_feedback = None
        for attempt in range(2):
            current_request = json.loads(json.dumps(request))
            if validation_feedback is not None:
                current_request['input']['validation_feedback'] = validation_feedback
            proposal = self._invoke(current_request, EvidenceSummaryProposal)
            if proposal is None:
                if (attempt == 0
                        and self.last_diagnostic.get('status') == 'invalid_output'):
                    validation_feedback = self._make_validation_feedback(
                        self._last_raw_response,
                        self._last_validation_errors or ('Output did not match the required schema.',),
                    )
                    continue
                self.last_diagnostic['application_attempts'] = attempt + 1
                return None

            errors = []
            if proposal.status == 'insufficient':
                if not proposal.claims:
                    self.last_diagnostic['status'] = 'model_abstained'
                    return None
            elif proposal.status != 'answered':
                errors.append('status must be answered when claims are provided')

            used_quote_ids = []
            claims = []
            for index, claim in enumerate(proposal.claims):
                quote_ids = tuple(dict.fromkeys(claim.quote_ids))
                if not quote_ids:
                    errors.append(f'claims[{index}] must cite at least one quote_id')
                unknown_ids = sorted(set(quote_ids) - allowed.keys())
                if unknown_ids:
                    errors.append(
                        f'claims[{index}] contains unknown quote_ids: {unknown_ids}'
                    )
                claims.append(' '.join(claim.text.split()))
                used_quote_ids.extend(quote_ids)
            if proposal.status == 'answered' and not claims:
                errors.append('answered status requires at least one claim')

            if errors:
                error_status = (
                    'invalid_citation'
                    if any('quote_id' in error for error in errors)
                    else 'invalid_output'
                )
                if attempt == 0:
                    validation_feedback = self._make_validation_feedback(
                        self._last_raw_response or proposal.model_dump(mode='json'),
                        errors,
                    )
                    continue
                self.last_diagnostic.update({
                    'status': error_status,
                    'application_attempts': attempt + 1,
                    'validation_errors': errors,
                })
                return None

            unique_quote_ids = tuple(dict.fromkeys(used_quote_ids))
            self.last_diagnostic.update({
                'status': 'completed',
                'application_attempts': attempt + 1,
                'used_quote_ids': list(unique_quote_ids),
            })
            return ValidatedEvidenceSummary(
                text=' '.join(claims),
                quote_ids=unique_quote_ids,
                confident=proposal.status == 'answered',
            )

        return None

    @staticmethod
    def _make_validation_feedback(previous_response, errors):
        if isinstance(previous_response, str):
            response = previous_response[:12000]
        else:
            try:
                response = json.dumps(previous_response, ensure_ascii=False, default=str)[:12000]
            except (TypeError, ValueError):
                response = str(previous_response)[:12000]
        return {
            'previous_response': response,
            'validation_errors': list(errors),
            'instruction': (
                'Correct the previous response using only the supplied verified quotes. '
                'Return a complete response matching the required schema.'
            ),
        }

    def _invoke(self, request, output_model):
        started = monotonic()
        self.last_diagnostic = {'status': 'disabled' if not self.enabled else 'no_input'}
        self._last_raw_response = None
        self._last_validation_errors = ()
        if not self.enabled:
            return None
        request = json.loads(json.dumps(request))
        queue = Queue(maxsize=1)

        def invoke():
            try:
                queue.put((True, self.caller(request, self.timeout_seconds)))
            except Exception as exc:
                # Exception messages/bodies may contain credentials. Keep only type.
                queue.put((False, type(exc).__name__))

        Thread(target=invoke, daemon=True).start()
        try:
            success, raw = queue.get(timeout=self.timeout_seconds)
            if not success:
                self.last_diagnostic = {'status': 'caller_error', 'error_class': raw}
                return None
            self._last_raw_response = raw
            result = (output_model.model_validate_json(raw) if isinstance(raw, str)
                      else output_model.model_validate(raw))
            self.last_diagnostic = {'status': 'completed'}
            return result
        except Empty:
            self.last_diagnostic = {'status': 'timeout'}
            return None
        except (ValueError, TypeError) as exc:
            errors = ()
            details = getattr(exc, 'errors', None)
            if callable(details):
                try:
                    errors = tuple({
                        'location': list(item.get('loc', ())),
                        'type': item.get('type', 'validation_error'),
                        'message': item.get('msg', 'Invalid output.'),
                    } for item in details(include_input=False))
                except Exception:
                    errors = ()
            if not errors:
                errors = ({'type': type(exc).__name__,
                           'message': 'Response could not be parsed or validated.'},)
            self._last_validation_errors = errors
            self.last_diagnostic = {
                'status': 'invalid_output',
                'validation_errors': list(errors),
            }
            return None
        finally:
            self.last_diagnostic['elapsed_seconds'] = round(monotonic() - started, 3)
