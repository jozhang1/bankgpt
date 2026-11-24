"""Three-state comparison of explicitly requested Fact conditions."""
from dataclasses import dataclass
from enum import Enum

from foundation.contract import Fact
from typing import Iterable


class MatchStatus(Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


def aggregate_match_status(statuses: Iterable[MatchStatus]) -> MatchStatus:
    """Combine checked conditions: MISMATCH > UNKNOWN > MATCH.

    Unspecified query conditions must not be included. An empty iterable means
    no checked condition prevents a match; it does not validate the query itself.
    """
    result = MatchStatus.MATCH
    for status in statuses:
        if status is MatchStatus.MISMATCH:
            return MatchStatus.MISMATCH
        if status is MatchStatus.UNKNOWN:
            result = MatchStatus.UNKNOWN
    return result


@dataclass(frozen=True)
class FactMatch:
    """Internal query-match evidence, independent of temporal resolution."""

    fact: Fact
    checks: tuple[tuple[str, MatchStatus], ...]

    @property
    def status(self) -> MatchStatus:
        return aggregate_match_status(status for _, status in self.checks)
