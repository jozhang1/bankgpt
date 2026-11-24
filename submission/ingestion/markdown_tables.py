"""Shared recognition of Markdown tables present in the supplied snapshots."""

import re


TABLE_DELIMITER_CELL = re.compile(r":?-{3,}:?")
_STRUCTURAL_BOUNDARY = re.compile(r"^ {0,3}(?:#{1,6}\s|`{3,}|~{3,}|<!--)")


def is_table_start(lines: list[str], i: int) -> bool:
    if i + 1 >= len(lines):
        return False
    header = lines[i].strip()
    delimiter = lines[i + 1].strip()
    if "|" not in header or "|" not in delimiter:
        return False
    cells = delimiter.strip("|").split("|")
    return bool(cells) and all(
        TABLE_DELIMITER_CELL.fullmatch(cell.strip()) for cell in cells
    )


def table_end(lines: list[str], start: int) -> int:
    """Return the exclusive end; caller must first verify is_table_start()."""
    end = start + 2
    while end < len(lines):
        line = lines[end]
        if not line.strip() or "|" not in line or _STRUCTURAL_BOUNDARY.match(line):
            break
        end += 1
    return end
