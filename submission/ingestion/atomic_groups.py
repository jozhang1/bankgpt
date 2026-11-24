"""Section-local ranges consumed whole by the retrieval chunker."""
from dataclasses import dataclass
from typing import Literal

GroupKind = Literal["person_card", "location_card"]
GroupSubtype = Literal["application", "contact", "biography", "branch", "atm"]


@dataclass(frozen=True)
class AtomicGroup:
    kind: GroupKind
    subtype: GroupSubtype
    start: int
    end: int  # exclusive, section-local block indices

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Invalid AtomicGroup range")
        allowed = {
            "person_card": {"application", "contact", "biography"},
            "location_card": {"branch", "atm"},
        }
        if self.subtype not in allowed.get(self.kind, set()):
            raise ValueError("Invalid AtomicGroup kind/subtype")
