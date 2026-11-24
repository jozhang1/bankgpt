"""Turn standalone ``N-month certificate APY: X%`` lines into a Markdown table.

The deterministic extractor reads tables only.  A page that states a term rate
as a plain labelled line carries the same fact, so it is rewritten into the
table form before chunking.  Only whole-line matches are rewritten; prose that
merely mentions a rate is left alone.
"""

import re

_LINE = re.compile(
    r'^\s*(?P<n>\d{1,3})[- ](?P<unit>month|year)s?\s+'
    r'(?:[A-Za-z][A-Za-z ]{0,40}?\s+)?'
    r'(?P<metric>APY|APR|dividend rate)\s*:\s*'
    r'(?P<value>\d{1,2}(?:\.\d{1,3})?%)\s*$',
    re.IGNORECASE,
)


def rate_lines_to_tables(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith(('```', '~~~')):
            fence = not fence
        match = None if fence else _LINE.match(line)
        if match is None:
            output.append(line)
            i += 1
            continue
        metric = match['metric'].upper() if match['metric'].lower() != 'dividend rate' \
            else 'Dividend Rate'
        rows = []
        j = i
        while j < len(lines):
            if not lines[j].strip():
                j += 1
                continue
            m = _LINE.match(lines[j])
            if m is None or m['metric'].lower() != match['metric'].lower():
                break
            unit = 'Months' if m['unit'].lower() == 'month' else 'Years'
            rows.append(f"| {m['n']} {unit} | {m['value']} |")
            j += 1
        output += ['', f'| Term | {metric} |', '| --- | --- |', *rows, '']
        i = j
    return '\n'.join(output) + ('\n' if markdown.endswith('\n') else '')
