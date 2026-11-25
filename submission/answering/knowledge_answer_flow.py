"""Answer-flow methods mixed into the harness-facing Knowledge class."""

from __future__ import annotations

from datetime import date
from uuid import UUID

from contract import Source
from answering.answer_renderer import FactAnswerRow, RenderBlock
from foundation.contract import Fact, TemporalResolution
from foundation.products import PRODUCTS
from facts.fact_matching import FactMatch, MatchStatus
from facts.semantic_fact_profile import SEMANTIC_FACT_PROFILES
from retrieval.quote_render_guard import (
    contextual_fact_quotes,
    deduplicate_consistent_facts,
    find_associated_facts,
    grouped_entity_label,
    pending_quote_competitors,
    render_associated_values,
)
from retrieval.answer_selector import AnswerSelection, snapshot_scope


class KnowledgeAnswerFlow:
    def _resolve_fact(
        self, fact: Fact, *, as_of: str | None = None,
    ) -> TemporalResolution:
        """Resolve a selected Fact within the answer flow."""
        return self.temporal_resolver.resolve(fact, as_of=as_of)

    def _selection_reason_note(self, selected_ids, suppressed_ids, reason):
        """Render an audit note with the compared values and selection basis."""
        observations = self.processor.store.observations

        def describe(ids):
            details = set()
            for observation_id in ids:
                observation = observations.get(observation_id)
                if observation is None:
                    continue
                fact = self.processor.store.facts.get(observation.fact_id)
                value = self.answer_renderer.display_fact_value(
                    fact.attribute if fact else '', observation.value,
                )
                if observation.unit == 'percent':
                    value += '%'
                elif observation.unit:
                    value += f' {observation.unit}'
                if observation.effective_from:
                    value += f' (effective {observation.effective_from.isoformat()}'
                    if observation.effective_to:
                        value += f' through {observation.effective_to.isoformat()}'
                    value += ')'
                details.add(value)
            return tuple(sorted(details))

        selected = describe(selected_ids)
        suppressed = describe(suppressed_ids)
        if not selected or not suppressed:
            return None
        return (
            f"Selected {'; '.join(selected)} over {'; '.join(suppressed)}. "
            f"{reason}"
        )

    def _no_matching_fact_answer(self):
        return self.answer_renderer.compose([
            RenderBlock(
                "I couldn’t find information matching your question in the available sources.",
                "",
            )
        ])

    def _fallback_to_evidence(self, question, as_of):
        hits = self.answer_selector.retrieve(question, as_of=as_of, top_k=10)
        hits = self.answer_selector._scope_evidence(hits, as_of)
        if not hits:
            return self._no_matching_fact_answer()
        return self._render_retrieved_route(
            AnswerSelection('retrieved', hits=tuple(hits)), question, as_of,
        )

    def _render_location_route(self, selection, question, as_of):
        return self.answer_renderer.render_location(
            selection.location,
            self.processor.store.evidence,
            collection=selection.location.query.scope != 'specific',
        )

    def _cite_observation_page(self, observation, sources, seen_sources):
        """Add the page an observation was read from, once per page and snapshot."""
        store = self.processor.store
        evidence = store.evidence[observation.evidence_id]
        if any(existing.url == evidence.source_url
               and existing.snapshot_id == evidence.snapshot_id for existing in sources):
            return
        support = store.semantic_support.get(observation.id)
        quote = (support.claim_quote
                 if support and support.claim_quote and support.claim_quote in evidence.text
                 else evidence.text[:500])
        source = Source(url=evidence.source_url, snapshot_id=evidence.snapshot_id, quote=quote)
        seen_sources.add(self.answer_renderer.source_display_key(source))
        sources.append(source)

    def _render_fact_route(self, selection, question, as_of):
        matches = selection.matches
        if matches is None:
            return self._fallback_to_evidence(question, as_of)
        resolutions = [self._resolve_fact(match.fact, as_of=as_of) for match in matches]
        # Entity-reference discovery is independent of query matching, but
        # references must come from the same selected snapshot as the primary.
        related_by_primary = find_associated_facts(
            (match.fact for match in matches),
            self.processor.store,
            self.fact_association_rules,
            snapshot_by_primary={r.fact.id: r.snapshot_id for r in resolutions},
        )
        available = [r for r in resolutions if r.status != 'not_found']
        if not available:
            return self._fallback_to_evidence(question, as_of)
        available, self.last_version_selections = self.version_selector.select_all(
            matches, available, as_of=as_of)
        matches, available, self.last_quote_schedule_selections = (
            self.version_selector.select_overlapping_schedule_quotes(
                matches, available,
                requested_product_ids=(self.fact_selector.last_scope.product_ids
                                       if self.fact_selector.last_scope else ()),
            )
        )
        matches, available = self._suppress_shadowed_rate_matches(matches, available)
        # Extreme selection happens only after snapshot/time resolution,
        # explicit version policy, and quote-schedule adjudication.
        matches, available = self.fact_selector.select_resolved_extreme(
            matches, available, selection.result_shape, as_of=as_of,
        )
        if not available:
            return self._fallback_to_evidence(question, as_of)
        match_by_fact = {m.fact.id: m for m in matches}
        # Schema-UNKNOWN qualifier scopes remain gated as possible quote
        # competitors; only proven mismatches are treated as distinct.
        pending = pending_quote_competitors(matches, available)
        bundle = self.reasoner.build_input(
            self.processor.store, question, as_of, matches, available, pending=pending)
        decisions = self.answerability_gate.assess(bundle)
        self.last_gate_decisions = decisions
        direct = tuple(d.selection for d in decisions if d.status == 'direct')
        # Query-time LLM use is limited to summarizing evidence after the
        # deterministic gate has admitted its source-bound citations.
        validated = direct
        accepted_ids = {oid for selection in validated if selection.can_render
                        for oid in selection.observation_ids}
        if not accepted_ids:
            return self._fallback_to_evidence(question, as_of)
        accepted_by_fact = {
            result.fact.id: tuple(
                observation for observation in result.observations
                if str(observation.id) in accepted_ids
            )
            for result in available
        }
        verified_value_existence = selection.value_existence
        if (verified_value_existence is False
                and not any(accepted_by_fact.values())):
            # A filter-level absence/presence result is only renderable when
            # at least one scoped source observation passed temporal and quote
            # validation.
            verified_value_existence = None
        associated_by_primary = {}
        associated_sources_by_primary = {}
        for primary in available:
            primary_observations = accepted_by_fact.get(primary.fact.id, ())
            if not primary_observations:
                continue
            primary_evidence_ids = {obs.evidence_id for obs in primary_observations}
            related_facts = related_by_primary.get(primary.fact.id, ())
            rendered = []
            associated_sources = []
            for rule in self.fact_association_rules:
                if primary.fact.attribute != rule.primary_attribute:
                    continue
                related_observations = []
                for related_fact in related_facts:
                    if related_fact.attribute != rule.related_attribute:
                        continue
                    related_resolution = self._resolve_fact(related_fact, as_of=as_of)
                    # Apply the same query-aware version policy to a
                    # companion metric before rendering it. A related APR
                    # may have older, conflicting observations even when
                    # the selected rate row has a verified newer schedule.
                    related_match = FactMatch(
                        related_fact, match_by_fact[primary.fact.id].checks,
                    )
                    related_resolution, _ = self.version_selector.select(
                        related_resolution, related_match, as_of=as_of,
                    )
                    if (related_resolution.status != 'resolved'
                            or related_resolution.snapshot_id != primary.snapshot_id):
                        continue
                    for observation in related_resolution.observations:
                        evidence = self.processor.store.evidence[observation.evidence_id]
                        same_evidence = observation.evidence_id in primary_evidence_ids
                        allows_cross_evidence = rule.use_entity_refs
                        if not allows_cross_evidence and not same_evidence:
                            continue
                        related_observations.append(observation)
                        support = self.processor.store.semantic_support.get(observation.id)
                        quote = (support.claim_quote if support and support.claim_quote
                                 and support.claim_quote in evidence.text else evidence.text[:500])
                        associated_sources.append(Source(
                            url=evidence.source_url,
                            snapshot_id=evidence.snapshot_id,
                            quote=quote,
                        ))
                if related_observations:
                    rendered.append(render_associated_values(related_observations, rule))
            if rendered:
                associated_by_primary[primary.fact.id] = tuple(dict.fromkeys(rendered))
                associated_sources_by_primary[primary.fact.id] = tuple(associated_sources)
        # Presentation dedup uses the same schema comparator and requires
        # a MATCH scope plus answerability admission for every observation.
        scope = self.fact_selector.last_scope
        duplicate_groups = deduplicate_consistent_facts(
            matches,
            available,
            accepted_ids,
            segment=scope.segment if scope else None,
        )
        duplicate_group_by_id = {
            fact_id: group for group in duplicate_groups for fact_id in group
        }
        insufficient_ids = {oid for group in bundle['groups']
                            if any(d.group_id == group['group_id'] and d.status == 'insufficient'
                                   for d in decisions)
                            for oid in group['observation_ids']} if bundle else set()
        excluded_ids = {oid for group in bundle['groups']
                        if any(d.group_id == group['group_id'] and d.status == 'excluded'
                               for d in decisions)
                        for oid in group['observation_ids']} if bundle else set()
        names = {product.id: product.name for product in PRODUCTS}
        names.update({r.id: r.name for r in self.fact_selector.semantic.records.values()})
        rows, sources, notes = [], [], []
        contextual_rows_added = set()
        notes.extend(
            d.reasons[0]
            for d in decisions if d.status == 'excluded' and d.reasons
        )
        seen_sources = set()
        preferred_versions = [
            version for version in self.last_version_selections
            if version.status == 'preferred'
        ]
        for version in preferred_versions:
            note = self._selection_reason_note(
                version.selected, version.suppressed, version.reason,
            )
            if note:
                notes.append(note)
            else:
                notes.append(version.reason)
        latest_supported_notes = []
        for result in available:
            if result.temporal_certainty != 'latest_supported':
                continue
            starts = sorted({
                observation.effective_from.isoformat()
                for observation in result.observations
                if observation.effective_from is not None
            })
            date_note = f" effective {', '.join(starts)}" if starts else ''
            latest_supported_notes.append(
                'Latest source-supported assertion' + date_note
                + '; no end date is stated, so uninterrupted validity is not guaranteed.'
            )
        notes.extend(dict.fromkeys(latest_supported_notes))
        for schedule in self.last_quote_schedule_selections:
            note = self._selection_reason_note(
                schedule.selected_observation_ids,
                schedule.suppressed_observation_ids,
                schedule.reason,
            )
            if note:
                notes.append(note)
            else:
                notes.append(schedule.reason)
        for result in available:
            observations = tuple(
                o for o in result.observations if str(o.id) not in excluded_ids
            )
            if not observations:
                continue
            fact = result.fact
            hidden_qualifiers = set(SEMANTIC_FACT_PROFILES.get(fact.attribute, {}).get(
                'identity_only_qualifiers', ()))
            conditions = ', '.join(f'{key}={value}' for key, value in sorted(fact.qualifiers.items())
                                   if key not in hidden_qualifiers)
            label = (f"{names.get(fact.entity, fact.entity)} — {fact.attribute}"
                     f"{f' ({conditions})' if conditions else ''}")
            match = match_by_fact[fact.id]
            admitted = tuple(o for o in observations if str(o.id) in accepted_ids)
            if admitted:
                # Render only stored, validated observations, never model
                # reasoning or a competing raw source value.
                observations = admitted
                duplicate_group = duplicate_group_by_id.get(fact.id, (fact.id,))
                if fact.id == duplicate_group[0]:
                    grouped_facts = [self.processor.store.facts[fact_id]
                                     for fact_id in duplicate_group]
                    grouped_label = grouped_entity_label(grouped_facts, names)
                    label = (f"{grouped_label} — {fact.attribute}"
                             f"{f' ({conditions})' if conditions else ''}")
                    values = sorted({self.answer_renderer.display_fact_value(fact.attribute, o.value)
                                     + ('%' if o.unit == 'percent' else
                                     ' ' + o.unit if o.unit else '')
                                     for o in admitted})
                    prefix = ''
                    suffix = ''
                    if associated_by_primary.get(fact.id):
                        suffix = ' (' + '; '.join(associated_by_primary[fact.id]) + ')'
                    # Add a concise, source-bound contrast only when the user
                    # explicitly requested a categorical condition and another
                    # value for the same metric is linked to the same source.
                    explicit_checks = {
                        key: status for key, status in match.checks
                    }
                    contextual = contextual_fact_quotes(
                        fact, admitted, explicit_checks, self.processor.store,
                        result.snapshot_id,
                    )
                    for other_fact, other_obs, other_evidence, row_quote, disclosure, _differing in contextual:
                        # Render a source-verified sibling as an ordinary Fact
                        # row. Keep the qualifying source note verbatim instead
                        # of synthesizing domain-specific prose from it.
                        row_key = (other_fact.id, other_obs.id, disclosure)
                        if row_key not in contextual_rows_added:
                            contextual_rows_added.add(row_key)
                            other_profile = SEMANTIC_FACT_PROFILES.get(
                                other_fact.attribute, {},
                            )
                            other_hidden = set(other_profile.get(
                                'identity_only_qualifiers', (),
                            ))
                            other_conditions = tuple(sorted(
                                (key, str(value))
                                for key, value in other_fact.qualifiers.items()
                                if key not in other_hidden
                            ))
                            other_entity = names.get(
                                other_fact.entity, other_fact.entity,
                            )
                            other_value = (
                                self.answer_renderer.display_fact_value(
                                    other_fact.attribute, other_obs.value,
                                )
                                + ('%' if other_obs.unit == 'percent' else
                                   ' ' + other_obs.unit if other_obs.unit else '')
                            )
                            other_condition_text = ', '.join(
                                f'{key}={value}'
                                for key, value in other_conditions
                            )
                            rows.append(FactAnswerRow(
                                label=(f'{other_entity} — {other_fact.attribute}'
                                       f'{f" ({other_condition_text})" if other_condition_text else ""}'),
                                state='value',
                                value=other_value,
                                suffix=f' Source note: {disclosure}',
                                entity=other_entity,
                                attribute=other_fact.attribute,
                                conditions=other_conditions,
                            ))
                        for quote in (row_quote, disclosure):
                            contextual_source = Source(
                                url=other_evidence.source_url,
                                snapshot_id=other_evidence.snapshot_id,
                                quote=quote,
                            )
                            source_key = self.answer_renderer.source_display_key(contextual_source)
                            if source_key not in seen_sources:
                                seen_sources.add(source_key)
                                sources.append(contextual_source)
                    rows.append(FactAnswerRow(
                        label=label,
                        state='value',
                        value=' / '.join(values),
                        prefix=prefix,
                        suffix=suffix,
                        entity=grouped_label,
                        attribute=fact.attribute,
                        conditions=tuple(sorted(
                            (key, str(value)) for key, value in fact.qualifiers.items()
                            if key not in hidden_qualifiers
                        )),
                    ))
            elif any(str(o.id) in insufficient_ids for o in observations):
                rows.append(FactAnswerRow(label=label, state='insufficient'))
                notes.append('Missing source support or incompatible query conditions.')
            elif fact.id in pending:
                rows.append(FactAnswerRow(label=label, state='pending'))
                notes.append(f'{fact.entity}: potentially competing values remain unresolved.')
                if match.status is MatchStatus.UNKNOWN:
                    unknown = ', '.join(key for key, status in match.checks if status is MatchStatus.UNKNOWN)
                    notes.append(f'{fact.entity}: missing or unproven query conditions: {unknown}.')
                if result.status != 'resolved':
                    notes.append(result.reason)
            elif match.status is MatchStatus.UNKNOWN:
                unknown = ', '.join(key for key, status in match.checks if status is MatchStatus.UNKNOWN)
                rows.append(FactAnswerRow(label=label, state='unknown'))
                notes.append(f'{fact.entity}: missing or unproven query conditions: {unknown}.')
            elif result.status == 'resolved' and match.status is MatchStatus.MATCH:
                # A failed source-binding check is not permission to print a value.
                rows.append(FactAnswerRow(label=label, state='unconfirmed'))
                notes.append('The value could not be verified against its source row and metric header.')
            else:
                # Competing values stay in the evidence/audit, not in the
                # asserted answer. A later start date alone proves neither
                # replacement nor a prior observation's end date.
                rows.append(FactAnswerRow(label=label, state='unable', status=result.status))
                notes.append(result.reason)
            for observation in observations:
                evidence = self.processor.store.evidence[observation.evidence_id]
                cited = [q for selection in validated if selection.can_render
                         and str(observation.id) in selection.observation_ids
                         for q in selection.supporting_quotes]
                source_quotes = (
                    [(self.processor.store.evidence[UUID(q.evidence_id)], q.quote)
                     for q in cited if UUID(q.evidence_id) in self.processor.store.evidence]
                    if admitted else [(evidence, evidence.text[:500])]
                )
                for cited_evidence, quote in source_quotes:
                    source_key = self.answer_renderer.source_display_key(Source(
                        url=cited_evidence.source_url,
                        snapshot_id=cited_evidence.snapshot_id,
                        quote=quote,
                    ))
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        sources.append(Source(url=cited_evidence.source_url,
                                              snapshot_id=cited_evidence.snapshot_id,
                                              quote=quote))
                if admitted:
                    # Every page that states the rendered value supports it.
                    # Which of them the validator happened to quote varies run
                    # to run, so cite each supporting page deterministically.
                    self._cite_observation_page(observation, sources, seen_sources)
            if admitted:
                # Facts merged into one displayed row (identical values for
                # the same product on different pages) are all cited.
                group_ids = duplicate_group_by_id.get(fact.id, (fact.id,))
                if fact.id == group_ids[0]:
                    for other in available:
                        if other.fact.id == fact.id or other.fact.id not in group_ids:
                            continue
                        for other_obs in other.observations:
                            if str(other_obs.id) in accepted_ids:
                                self._cite_observation_page(other_obs, sources, seen_sources)
                for source in associated_sources_by_primary.get(fact.id, ()):
                    source_key = self.answer_renderer.source_display_key(source)
                    if source_key not in seen_sources:
                        seen_sources.add(source_key)
                        sources.append(source)
        broad_family_result = bool(
            scope
            and scope.family
            and len(rows) > 1
            and not scope.conditions
            and scope.term_months is None
            and scope.model_year is None
            and scope.amount is None
            and scope.amount_min is None
        )
        return self.answer_renderer.render_fact_rows(
            rows,
            sources=sources,
            notes=notes,
            confident=bool(decisions) and all(
                          d.status in {'direct', 'excluded'} for d in decisions
                      ) and any(d.status == 'direct' for d in decisions),
            value_existence=verified_value_existence,
            full_collection=bool(
                (scope and scope.result_shape.kind == 'full_collection')
                or broad_family_result
            ),
        )

    def _render_retrieved_route(self, selection, question, as_of):
        hits = selection.hits
        if not hits:
            self.last_gate_decisions = ()
            return self.answer_renderer.compose([RenderBlock("I don't know.", "")])
        # Every retrieved candidate is rebound to stored, eligible Evidence.
        # needs_reasoning is not a prerequisite for concise answer formatting.
        summary_hits = hits[:5]
        records = self.answer_renderer.verified_candidate_records(
            summary_hits, self.processor.store.evidence,
        )
        if records:
            # "Current" is the selected snapshot's date, never the wall clock.
            scope = snapshot_scope(self.processor.store, as_of)
            reference_date = as_of or (
                scope.snapshot.taken_on if scope is not None else None)
            summary = self.reasoner.summarize(
                question, records, reference_date=reference_date,
            )
            if summary is not None:
                used = set(summary.quote_ids)
                used_records = [record for record in records
                                if record['quote_id'] in used]
                return self.answer_renderer.render_evidence_summary(
                    summary, used_records,
                )
            if self.reasoner.last_diagnostic.get('status') == 'model_abstained':
                return self.answer_renderer.compose([
                    RenderBlock("I don't know based on the available sources.", "")
                ])
        # On disabled, failed, or rejected summaries, return the original top hit.
        evidence = hits[0].evidence
        stored = self.processor.store.evidence.get(evidence.id)
        if stored is None or stored != evidence or stored.in_recommendations:
            return self.answer_renderer.compose([RenderBlock("I don't know.", "")])
        return self.answer_renderer.compose(
            [RenderBlock(evidence.text, "")],
            sources=[Source(url=evidence.source_url, snapshot_id=evidence.snapshot_id,
                            quote=evidence.text[:500])],
            conflict_note='Summary unavailable or failed citation validation; returned the original retrieved excerpt.',
        )

    def _suppress_shadowed_rate_matches(self, matches, available):
        """Hide duplicate base APR observations only in a matching answer.

        A less-qualified APR is suppressed only when a reviewed, more-qualified
        APR matches every missing query condition and cites the exact same
        Evidence occurrence and value. Both Facts remain stored for audit.
        """
        resolution_by_id = {resolution.fact.id: resolution for resolution in available}
        shadowed = set()

        for broad in matches:
            if broad.fact.attribute != 'apr' or broad.status is not MatchStatus.UNKNOWN:
                continue
            broad_resolution = resolution_by_id.get(broad.fact.id)
            if (broad_resolution is None or broad_resolution.status != 'resolved'
                    or not broad_resolution.observations):
                continue
            missing = {key for key, status in broad.checks
                       if status is MatchStatus.UNKNOWN}
            if not missing or not missing <= {'phase', 'transaction_type'}:
                continue

            broad_qualifiers = broad.fact.qualifiers
            for specific in matches:
                if (specific.status is not MatchStatus.MATCH
                        or specific.fact.id == broad.fact.id
                        or specific.fact.entity != broad.fact.entity
                        or specific.fact.attribute != broad.fact.attribute):
                    continue
                specific_qualifiers = specific.fact.qualifiers
                if (len(specific_qualifiers) <= len(broad_qualifiers)
                        or not missing <= specific_qualifiers.keys()
                        or any(specific_qualifiers.get(key) != value
                               for key, value in broad_qualifiers.items())):
                    continue
                specific_resolution = resolution_by_id.get(specific.fact.id)
                if (specific_resolution is None or specific_resolution.status != 'resolved'
                        or specific_resolution.snapshot_id != broad_resolution.snapshot_id):
                    continue
                supported = []
                for observation in specific_resolution.observations:
                    support = self.processor.store.semantic_support.get(observation.id)
                    if support and getattr(support, 'context_quotes', ()):
                        supported.append(observation)
                if not supported:
                    continue

                signatures = {(observation.evidence_id, observation.value, observation.unit)
                              for observation in supported}
                if all((observation.evidence_id, observation.value, observation.unit) in signatures
                       for observation in broad_resolution.observations):
                    shadowed.add(broad.fact.id)
                    break

        if not shadowed:
            return matches, available
        return ([match for match in matches if match.fact.id not in shadowed],
                [resolution for resolution in available if resolution.fact.id not in shadowed])
