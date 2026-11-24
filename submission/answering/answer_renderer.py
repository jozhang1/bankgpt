"""Common rendering boundary for all selected answer routes."""
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import re
from decimal import Decimal, InvalidOperation

try:  # Harness imports the submission as a top-level package.
    from contract import Answer, Source
except ModuleNotFoundError:  # Unit tests may import the harness as a package.
    from contract import Answer, Source

from retrieval.answer_selector import AnswerSelection
from facts.semantic_fact_profile import SEMANTIC_FACT_PROFILES


RenderHandler = Callable[[AnswerSelection, str, str | None], Answer]


@dataclass(frozen=True)
class FactAnswerRow:
    """Presentation-ready fact row, separate from fact selection/adjudication."""

    label: str
    state: str
    value: str | None = None
    status: str | None = None
    prefix: str = ""
    suffix: str = ""
    entity: str | None = None
    attribute: str | None = None
    conditions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RenderBlock:
    """A content unit; the renderer owns how adjacent units are separated."""

    text: str
    separator_before: str = "\n\n"


class AnswerRenderer:
    """Shared answer assembly and route-specific data-to-block formatters."""

    def __init__(self, handlers: Mapping[str, RenderHandler]):
        self._handlers = dict(handlers)

    @staticmethod
    def source_display_key(source: Source) -> tuple[str, str, str]:
        """Collapse repeated displayed citations without merging stored sources."""
        quote = ' '.join((source.quote or '').split()).casefold()
        url = '' if quote else source.url
        return source.snapshot_id, quote, url

    @staticmethod
    def display_fact_value(attribute: str, value: str) -> str:
        """Render structured semantic values in readable source-level form."""
        profile = SEMANTIC_FACT_PROFILES.get(attribute, {})
        if profile.get('value_kind') == 'rule_expression':
            try:
                root = json.loads(value)
            except (TypeError, ValueError, AttributeError):
                root = None
            if isinstance(root, dict):
                if root.get('operator') == 'condition' and not root.get('children'):
                    quote = root.get('quote')
                    if isinstance(quote, str) and quote.strip():
                        return quote

                def render(node, depth=0, is_root=False):
                    quote = node.get('quote', '').strip()
                    quote = re.sub(r'^\s*(?:[-+*]|\d+[.)])\s+', '', quote)
                    marker = '' if is_root else '- '
                    lines = ['  ' * depth + marker + quote] if quote else []
                    for child in node.get('children', ()):
                        if isinstance(child, dict):
                            lines.extend(render(child, depth + 1))
                    return lines

                rendered = render(root, is_root=True)
                if rendered:
                    return '\n'.join(rendered)
        return value

    @staticmethod
    def _attribute_label(attribute: str) -> str:
        labels = {
            'apy': 'APY',
            'apr': 'APR',
            'dividend_rate': 'dividend rate',
            'interest_rate': 'interest rate',
            'minimum_opening_balance': 'minimum opening balance',
            'minimum_earning_balance': 'minimum balance to earn dividends',
            'branch_count': 'number of branches',
            'employee_count': 'number of employees',
            'vto_user_count': 'employees who used Volunteer Time Off',
            'volunteer_hours': 'volunteer hours',
            'routing_number': 'routing number',
            'closure_date': 'closure date',
            'opening_date': 'opening date',
            'eligibility_requirement': 'eligibility',
            'supported_payment_wallet': 'supported digital wallet',
        }
        return labels.get(attribute, attribute.replace('_', ' '))

    @staticmethod
    def _condition_phrases(conditions: tuple[tuple[str, str], ...]) -> list[str]:
        """Turn canonical Fact qualifiers into concise reader-facing phrases."""
        values = dict(conditions)
        phrases = []

        def money(value: str) -> str:
            try:
                number = Decimal(value)
            except (InvalidOperation, TypeError, ValueError):
                return value
            rendered = format(number, ',f')
            return rendered.rstrip('0').rstrip('.') if '.' in rendered else rendered

        if 'term_months' in values:
            raw_months = values.pop('term_months')
            try:
                months = int(float(raw_months))
                term = (f'{months // 12}-year' if months and months % 12 == 0
                        else f'{months}-month')
                phrases.append(f'{term} term')
            except (TypeError, ValueError):
                phrases.append(f'{raw_months} month term')
        if 'jumbo' in values:
            raw = values.pop('jumbo').casefold()
            if raw in {'true', '1'}:
                phrases.append('Jumbo')
            elif raw in {'false', '0'}:
                phrases.append('standard (non-Jumbo)')
        if 'rate_type' in values:
            rate_type = values.pop('rate_type').replace('_', ' ')
            phrases.append(f'{rate_type}-rate' if rate_type in {'fixed', 'variable'} else rate_type)
        phase = values.pop('phase', None)
        intro_months = values.pop('intro_months', None)
        initial_months = values.pop('initial_rate_months', None)
        if phase == 'introductory':
            phrases.append('introductory rate' + (f' for {intro_months} months' if intro_months else ''))
        elif phase:
            phrases.append(phase.replace('_', ' ') + ' rate')
        elif intro_months:
            phrases.append(f'introductory period of {intro_months} months')
        if initial_months:
            phrases.append(f'initial rate for {initial_months} months')
        for key in ('loan_purpose', 'transaction_type', 'vehicle_condition', 'ira_type'):
            if key not in values:
                continue
            raw = values.pop(key).replace('_', ' ')
            if key in {'loan_purpose', 'transaction_type'}:
                phrases.append(f'for {raw}')
            else:
                phrases.append(raw)

        if 'rate_basis' in values:
            basis = values.pop('rate_basis').replace('_', ' ')
            phrases.append('as low as' if basis == 'as low as' else f'{basis} rate basis')
        if 'loan_program' in values:
            phrases.append(values.pop('loan_program').replace('_', ' '))
        if 'mortgage_points' in values:
            points = values.pop('mortgage_points')
            phrases.append('no discount points' if points in {'0', '0.0'}
                           else f'{points} discount points')

        for lower, upper, label in (
            ('balance_min', 'balance_max', 'balance'),
            ('loan_amount_min', 'loan_amount_max', 'loan amount'),
            ('model_year_from', 'model_year_to', 'model year'),
        ):
            low, high = values.pop(lower, None), values.pop(upper, None)
            values.pop(lower + '_inclusive', None)
            values.pop(upper + '_inclusive', None)
            if low is not None and high is not None:
                phrases.append(f'{label} ${money(low)}–${money(high)}' if label != 'model year'
                               else f'model years {low}–{high}')
            elif low is not None:
                phrases.append(f'{label} of at least ${money(low)}' if label != 'model year'
                               else f'model year {low} or newer')
            elif high is not None:
                phrases.append(f'{label} up to ${money(high)}' if label != 'model year'
                               else f'model year {high} or older')

        for key, value in sorted(values.items()):
            readable_key = key.replace('_', ' ')
            readable_value = value.replace('_', ' ')
            phrases.append(f'{readable_key}: {readable_value}')
        return phrases

    @staticmethod
    def _merge_conditions(left, right):
        """Merge compatible qualifiers for duplicate values without erasing scope."""
        merged = dict(left)
        for key, value in right.items():
            if key in merged and merged[key] != value:
                return None
            merged[key] = value
        return merged

    @classmethod
    def _render_full_collection(cls, rows: list[FactAnswerRow]) -> str:
        """Render complete selected collections as compact prose, not a Markdown grid."""
        # Collapse only equal values for the same entity and attribute. If their
        # qualifiers are compatible, retain the union so the line stays scoped.
        facts = []
        for row in rows:
            record = {
                'entity': row.entity or row.label,
                'attribute': row.attribute or 'value',
                'value': row.value or '',
                'prefix': row.prefix,
                'suffix': row.suffix,
                'conditions': dict(row.conditions),
            }
            merged = False
            for existing in facts:
                if (existing['entity'] != record['entity']
                        or existing['attribute'] != record['attribute']
                        or existing['value'] != record['value']
                        or existing['prefix'] != record['prefix']
                        or existing['suffix'] != record['suffix']):
                    continue
                combined = cls._merge_conditions(existing['conditions'], record['conditions'])
                if combined is not None:
                    existing['conditions'] = combined
                    merged = True
                    break
            if not merged:
                facts.append(record)

        grouped: dict[str, dict[tuple[tuple[str, str], ...], dict[str, list[str]]]] = {}
        for fact in facts:
            path = tuple(sorted((key, str(value)) for key, value in fact['conditions'].items()))
            values = grouped.setdefault(fact['entity'], {}).setdefault(path, {})
            display = fact['prefix'] + fact['value'] + fact['suffix']
            bucket = values.setdefault(fact['attribute'], [])
            if display not in bucket:
                bucket.append(display)

        blocks = []
        for entity in sorted(grouped, key=str.casefold):
            clauses = []
            def scope_sort_key(item):
                conditions = dict(item[0])
                phase_order = {'standard': 0, 'introductory': 1}.get(
                    conditions.get('phase'), 2,
                )
                term = conditions.get('term_months')
                try:
                    term_order = float(term) if term is not None else float('inf')
                except (TypeError, ValueError):
                    term_order = float('inf')
                return (phase_order, term_order,
                        cls._condition_phrases(item[0]), item[0])

            for conditions, attributes in sorted(
                grouped[entity].items(),
                key=scope_sort_key,
            ):
                scope = cls._condition_phrases(conditions)
                parts = []
                for attribute in sorted(attributes):
                    label = cls._attribute_label(attribute)
                    values = ' / '.join(attributes[attribute])
                    if attribute in {'apy', 'apr', 'dividend_rate', 'interest_rate'}:
                        parts.append(f'{label} is {values}')
                    else:
                        parts.append(f'{label}: {values}')
                clause = '; '.join(parts)
                if scope:
                    clause = f"{', '.join(scope)}: {clause}"
                clauses.append('- ' + clause + '.')
            blocks.append(f"{entity}:\n" + '\n'.join(clauses))
        return '\n'.join(blocks)

    @classmethod
    def _fact_row_text(cls, row: FactAnswerRow) -> str:
        if not row.entity or not row.attribute:
            return f'{row.prefix}{row.label}: {row.value or ""}{row.suffix}'
        subject = row.entity
        conditions = cls._condition_phrases(row.conditions)
        if conditions:
            subject += ' (' + ', '.join(conditions) + ')'
        attribute = cls._attribute_label(row.attribute)
        value = row.value or ''
        if row.attribute in {'apy', 'apr', 'dividend_rate', 'interest_rate'}:
            text = f'{subject}: {attribute} is {value}.'
        else:
            text = f'{subject}: {attribute} — {value}.'
        return f'{row.prefix}{text}{row.suffix}'

    def render(
        self,
        selection: AnswerSelection,
        *,
        question: str,
        as_of: str | None = None,
    ) -> Answer:
        handler = self._handlers.get(selection.route)
        if handler is None:
            raise ValueError(f"No answer renderer registered for route: {selection.route}")
        return handler(selection, question, as_of)

    @classmethod
    def compose(
        cls,
        blocks: list[RenderBlock] | tuple[RenderBlock, ...],
        *,
        sources: list[Source] | tuple[Source, ...] = (),
        confident: bool = False,
        conflict_note: str | None = None,
    ) -> Answer:
        """Apply shared empty-block, spacing, citation, and Answer rules."""
        rendered: list[str] = []
        for block in blocks:
            text = block.text
            if not text.strip():
                continue
            if rendered:
                rendered.append(block.separator_before)
            rendered.append(text)

        unique_sources: list[Source] = []
        seen_sources = set()
        for source in sources:
            key = (source.url, source.snapshot_id, source.quote)
            if key in seen_sources:
                continue
            seen_sources.add(key)
            unique_sources.append(source)

        note = ' '.join(dict.fromkeys((conflict_note or '').splitlines())).strip() or None
        return Answer(text=''.join(rendered), sources=unique_sources,
                      confident=confident, conflict_note=note)

    @staticmethod
    def render_fact_rows(
        rows: tuple[FactAnswerRow, ...] | list[FactAnswerRow],
        *,
        sources: list,
        notes: list[str],
        confident: bool,
        value_existence: bool | None = None,
        full_collection: bool = False,
    ) -> Answer:
        """Render structured Fact outcomes without deciding which facts qualify."""
        templates = {
            "insufficient": "no supported applicable observation [insufficient]",
            "pending": "quote selection is unresolved [pending quote selection]",
            "unknown": "query applicability is unconfirmed [unknown match]",
            "unconfirmed": "source support requires review [unconfirmed evidence]",
        }
        blocks = []
        collection_rows = [
            row for row in rows
            if row.state == "value" and row.entity and row.attribute
        ]
        if value_existence is False:
            blocks.append(RenderBlock(
                'No.',
                separator_before='',
            ))
        # A full collection remains complete but is rendered as concise prose.
        if full_collection and collection_rows and len(collection_rows) == len(rows):
            blocks.append(RenderBlock(
                AnswerRenderer._render_full_collection(collection_rows),
                separator_before="",
            ))

        for row in rows:
            if full_collection and collection_rows and len(collection_rows) == len(rows):
                break
            if row.state == "value":
                rendered = AnswerRenderer._fact_row_text(row)
            elif row.state == "unable":
                rendered = f"{row.label}: unable to determine a single applicable value [{row.status}]"
            else:
                rendered = f"{row.label}: {templates[row.state]}"
            blocks.append(RenderBlock(rendered, separator_before='\n'))
        return AnswerRenderer.compose(blocks, sources=sources, confident=confident,
                                      conflict_note=' '.join(notes))


    @staticmethod
    def verified_candidate_records(hits, evidence_by_id) -> list[dict[str, str]]:
        """Bind retrieved candidates to stored source text before model review."""
        records = []
        seen = set()
        for hit in hits:
            candidate = hit.evidence
            evidence = evidence_by_id.get(candidate.id)
            if evidence is None:
                evidence = evidence_by_id.get(str(candidate.id))
            if evidence is None or evidence.in_recommendations:
                continue
            if (evidence.id != candidate.id
                    or evidence.snapshot_id != candidate.snapshot_id
                    or evidence.source_url != candidate.source_url
                    or evidence.text != candidate.text
                    or not evidence.text.strip()):
                continue
            key = (str(evidence.id), evidence.text)
            if key in seen:
                continue
            seen.add(key)
            records.append({
                'quote_id': f'q{len(records) + 1}',
                'evidence_id': str(evidence.id),
                'quote': evidence.text,
                'source_url': evidence.source_url,
                'snapshot_id': evidence.snapshot_id,
            })
        return records

    @staticmethod
    def render_evidence_summary(summary, quote_records, *, unresolved=False) -> Answer:
        """Render a summary with provenance limited to its verified quote set."""
        if not summary or not summary.text.strip():
            raise ValueError('A nonempty validated summary is required.')
        sources = []
        seen = set()
        for record in quote_records:
            key = (record['snapshot_id'], record['source_url'], record['quote'])
            if key in seen:
                continue
            seen.add(key)
            sources.append(Source(
                url=record['source_url'],
                snapshot_id=record['snapshot_id'],
                quote=record['quote'],
            ))
        note = 'Some retrieved fields remain unresolved.' if unresolved else None
        return AnswerRenderer.compose(
            [RenderBlock(summary.text, '')], sources=sources,
            confident=summary.confident and not unresolved,
            conflict_note=note,
        )

    @staticmethod
    def render_location(result, evidence_by_id: Mapping, *, collection: bool = False) -> Answer:
        """Render specific or collection location records using one location template."""
        if collection:
            if result.status == 'not_found' or not result.records:
                return AnswerRenderer.compose([
                    RenderBlock("I don't have enough source-backed records for this location query. "
                                + result.reason, "")
                ])
            field = result.query.field
            parts = []
            sources, seen = [], set()
            for record in result.records:
                evidence = evidence_by_id[record.evidence_id]
                lines = []
                if field == 'address':
                    lines.append(record.location_quote)
                    lines.append(f'{record.name}: {record.address}')
                elif record.city or record.name:
                    lines.append(record.city or record.name)
                if field in {'lobby_hours', 'hours'} and record.lobby_hours:
                    lines.append('Lobby hours:\n' + '\n'.join(record.lobby_hours))
                if field in {'drive_up_hours', 'hours'} and record.drive_up_hours:
                    lines.append('Drive up hours:\n' + '\n'.join(record.drive_up_hours))
                parts.append('\n'.join(lines))
                if evidence.id not in seen:
                    seen.add(evidence.id)
                    sources.append(Source(url=evidence.source_url,
                                          snapshot_id=evidence.snapshot_id,
                                          quote=evidence.text))
            return AnswerRenderer.compose(
                [RenderBlock(part) for part in parts], sources=sources,
                confident=result.status == 'match',
            )

        records = result.records
        sources, seen = [], set()
        for record in records:
            if record.evidence_id in seen:
                continue
            seen.add(record.evidence_id)
            evidence = evidence_by_id[record.evidence_id]
            sources.append(Source(url=evidence.source_url, snapshot_id=evidence.snapshot_id,
                                  quote=evidence.text if (record.lobby_hours or record.drive_up_hours)
                                  else record.location_quote))
        if result.status == 'not_found':
            return AnswerRenderer.compose([
                RenderBlock("I don't have grounded " + result.query.field.replace('_', ' ')
                            + f" for {result.query.label} in the available sources.", "")
            ], sources=sources)
        if result.status == 'ambiguous':
            return AnswerRenderer.compose([
                RenderBlock('Multiple locations or differing records match. Please specify the city or street.', "")
            ], sources=sources, conflict_note=result.reason)
        record = records[0]
        if result.query.field == 'address':
            if record.address is not None:
                text = f'{record.name}: {record.address}'
                if record.postal_code is None:
                    if record.published_on:
                        text += f'\nAnnouncement published on {record.published_on.isoformat()}.'
                    return AnswerRenderer.compose(
                        [RenderBlock(text, "")], sources=sources, confident=True,
                    )
                return AnswerRenderer.compose([RenderBlock(text, "")], sources=sources, confident=True)
            return AnswerRenderer.compose([RenderBlock(record.location_quote, "")], sources=sources)
        parts = []
        if record.city or record.name:
            parts.append(record.city or record.name)
        if result.query.field in {'lobby_hours', 'hours'} and record.lobby_hours:
            parts.append('Lobby:\n' + '\n'.join(record.lobby_hours))
        if result.query.field in {'drive_up_hours', 'hours'} and record.drive_up_hours:
            parts.append('Drive-up:\n' + '\n'.join(record.drive_up_hours))
        if len(parts) == 1 and not (record.lobby_hours or record.drive_up_hours):
            parts = [record.location_quote]
        return AnswerRenderer.compose(
            [RenderBlock(part) for part in parts], sources=sources,
            confident=(result.status == 'match'
                       and bool(record.lobby_hours or record.drive_up_hours)),
        )
