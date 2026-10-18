"""Serializable inference records for future benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from fyp.prompting.parser import ParsedPrediction


@dataclass(frozen=True)
class InferenceRecord:
    """Minimum record needed to preserve raw benchmark outputs."""

    example_id: str
    subset: str
    model: str
    strategy: str
    prompt_version: str
    prompt_hash: str
    raw_response: str
    parsed_prediction: ParsedPrediction
    thinking: str | None = None
    intermediate_stage_outputs: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
