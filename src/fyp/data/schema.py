"""Explicit schema for the official FLAWEDFICTIONS dataset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


REQUIRED_FIELDS: tuple[str, ...] = (
    "story",
    "cont_error",
    "cont_error_expl",
    "cont_error_lines",
    "contradicted_lines",
    "example_id",
)

INCONSISTENCY_CONTINUITY_ERROR = "continuity_error"
INCONSISTENCY_NONE = "none"
VALID_INCONSISTENCY_TYPES = frozenset(
    {INCONSISTENCY_CONTINUITY_ERROR, INCONSISTENCY_NONE}
)


class FlawedFictionsSplit(str, Enum):
    """Official Hugging Face split names for `kahuja/flawed-fictions`."""

    FLAWED_FICTIONS = "flawed_fictions"
    FLAWED_FICTIONS_LONG = "flawed_fictions_long"
    FLAWED_FICTIONS_ERROR_RESOLVED_NEGS = "flawed_fictions_error_resolved_negs"
    FLAWED_FICTIONS_CF_NEGS = "flawed_fictions_cf_negs"

    @classmethod
    def parse(cls, value: str | "FlawedFictionsSplit") -> "FlawedFictionsSplit":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(split.value for split in cls)
            raise ValueError(f"Unknown FLAWEDFICTIONS split {value!r}; expected one of: {allowed}") from exc


@dataclass(frozen=True)
class SplitSpec:
    """Expected immutable facts from official dataset metadata."""

    split: FlawedFictionsSplit
    filename: str
    expected_examples: int
    expected_positive: int
    expected_negative: int
    notes: str


SPLIT_SPECS: dict[FlawedFictionsSplit, SplitSpec] = {
    FlawedFictionsSplit.FLAWED_FICTIONS: SplitSpec(
        split=FlawedFictionsSplit.FLAWED_FICTIONS,
        filename="flawed_fictions.parquet",
        expected_examples=414,
        expected_positive=207,
        expected_negative=207,
        notes="Main human-verified FLAWEDFICTIONS benchmark split.",
    ),
    FlawedFictionsSplit.FLAWED_FICTIONS_LONG: SplitSpec(
        split=FlawedFictionsSplit.FLAWED_FICTIONS_LONG,
        filename="flawed_fictions_long.parquet",
        expected_examples=200,
        expected_positive=97,
        expected_negative=103,
        notes="Long-story benchmark split.",
    ),
    FlawedFictionsSplit.FLAWED_FICTIONS_ERROR_RESOLVED_NEGS: SplitSpec(
        split=FlawedFictionsSplit.FLAWED_FICTIONS_ERROR_RESOLVED_NEGS,
        filename="flawed_fictions_error_resolved_negs.parquet",
        expected_examples=414,
        expected_positive=207,
        expected_negative=207,
        notes="Experimental alternate split with error-resolved negative examples.",
    ),
    FlawedFictionsSplit.FLAWED_FICTIONS_CF_NEGS: SplitSpec(
        split=FlawedFictionsSplit.FLAWED_FICTIONS_CF_NEGS,
        filename="flawed_fictions_cf_negs.parquet",
        expected_examples=414,
        expected_positive=207,
        expected_negative=207,
        notes="Experimental alternate split with counterfactual negative examples.",
    ),
}


@dataclass(frozen=True)
class FlawedFictionsExample:
    """Normalized in-memory representation of one official dataset row.

    The original fields are preserved either directly or in `raw_record`. Parsed line
    tuples are deterministic convenience fields for prompting and evaluation.
    """

    split: FlawedFictionsSplit
    example_id: str
    story: str
    has_continuity_error: bool
    cont_error_expl: str
    cont_error_lines_raw: str
    contradicted_lines_raw: str
    cont_error_lines: tuple[str, ...]
    contradicted_lines: tuple[str, ...]
    inconsistency_type: str
    raw_label: float
    raw_record: Mapping[str, Any]

    @property
    def prompt_story(self) -> str:
        """Story text exactly as supplied by the official dataset row."""

        return self.story

    @property
    def cont_error(self) -> float:
        """Official-compatible numeric label."""

        return 1.0 if self.has_continuity_error else 0.0

    def to_official_prompt_record(self) -> dict[str, Any]:
        """Return the field shape expected by the official prompting code."""

        return {
            "story": self.story,
            "cont_error": self.cont_error,
            "cont_error_expl": self.cont_error_expl,
            "cont_error_lines": self.cont_error_lines_raw,
            "contradicted_lines": self.contradicted_lines_raw,
            "example_id": self.example_id,
        }
