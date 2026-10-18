"""Validation for normalized FLAWEDFICTIONS examples."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from fyp.data.schema import (
    REQUIRED_FIELDS,
    SPLIT_SPECS,
    VALID_INCONSISTENCY_TYPES,
    FlawedFictionsExample,
    FlawedFictionsSplit,
    SplitSpec,
)


class ValidationError(RuntimeError):
    """Raised when a dataset split does not match official expectations."""


@dataclass(frozen=True)
class SplitValidationSummary:
    split: FlawedFictionsSplit
    examples: int
    positive: int
    negative: int
    duplicate_ids: tuple[str, ...]
    unexpected_fields: tuple[str, ...]


@dataclass(frozen=True)
class DatasetValidationReport:
    splits: tuple[SplitValidationSummary, ...]

    @property
    def total_examples(self) -> int:
        return sum(split.examples for split in self.splits)


def validate_examples(
    examples: Sequence[FlawedFictionsExample],
    split: FlawedFictionsSplit | str,
    expected: SplitSpec | None = None,
) -> SplitValidationSummary:
    """Validate one normalized official split.

    `expected` should be the official split spec during production validation.
    Unit tests can pass a smaller spec to exercise the same checks offline.
    """

    split = FlawedFictionsSplit.parse(split)
    expected = expected if expected is not None else SPLIT_SPECS.get(split)
    errors: list[str] = []

    ids = [example.example_id for example in examples]
    duplicates = tuple(sorted(item for item, count in Counter(ids).items() if count > 1))
    unexpected_fields = _collect_unexpected_fields(examples)

    positive = sum(1 for example in examples if example.has_continuity_error)
    negative = len(examples) - positive

    if expected is not None:
        if len(examples) != expected.expected_examples:
            errors.append(
                f"{split.value}: expected {expected.expected_examples} examples, observed {len(examples)}"
            )
        if positive != expected.expected_positive:
            errors.append(
                f"{split.value}: expected {expected.expected_positive} positive examples, observed {positive}"
            )
        if negative != expected.expected_negative:
            errors.append(
                f"{split.value}: expected {expected.expected_negative} negative examples, observed {negative}"
            )

    if duplicates:
        errors.append(f"{split.value}: duplicate example_id values: {', '.join(duplicates)}")

    for row_idx, example in enumerate(examples):
        _validate_example(example, split=split, row_idx=row_idx, errors=errors)

    if errors:
        raise ValidationError("\n".join(errors))

    return SplitValidationSummary(
        split=split,
        examples=len(examples),
        positive=positive,
        negative=negative,
        duplicate_ids=duplicates,
        unexpected_fields=unexpected_fields,
    )


def validate_dataset(
    dataset: dict[FlawedFictionsSplit, Sequence[FlawedFictionsExample]],
) -> DatasetValidationReport:
    """Validate every loaded official split against expected metadata."""

    summaries = []
    for split, examples in dataset.items():
        summaries.append(validate_examples(examples, split=split, expected=SPLIT_SPECS[split]))
    return DatasetValidationReport(splits=tuple(summaries))


def _validate_example(
    example: FlawedFictionsExample,
    split: FlawedFictionsSplit,
    row_idx: int,
    errors: list[str],
) -> None:
    prefix = f"{split.value}[{row_idx}] example_id={example.example_id!r}"

    if example.split != split:
        errors.append(f"{prefix}: example split is {example.split.value!r}")
    if not example.example_id.strip():
        errors.append(f"{prefix}: empty example_id")
    if not example.story.strip():
        errors.append(f"{prefix}: empty story")
    if example.raw_label not in (0.0, 1.0):
        errors.append(f"{prefix}: invalid raw_label {example.raw_label!r}")
    if example.inconsistency_type not in VALID_INCONSISTENCY_TYPES:
        errors.append(f"{prefix}: invalid inconsistency_type {example.inconsistency_type!r}")

    for field in REQUIRED_FIELDS:
        if field not in example.raw_record:
            errors.append(f"{prefix}: raw_record missing {field!r}")
        elif example.raw_record[field] is None:
            errors.append(f"{prefix}: raw_record field {field!r} is null")

    if example.has_continuity_error:
        if not example.cont_error_expl.strip():
            errors.append(f"{prefix}: positive example has empty cont_error_expl")
        if not example.cont_error_lines:
            errors.append(f"{prefix}: positive example has no cont_error_lines")
        if not example.contradicted_lines:
            errors.append(f"{prefix}: positive example has no contradicted_lines")
    else:
        if example.cont_error_lines:
            errors.append(f"{prefix}: negative example has cont_error_lines")
        if example.contradicted_lines:
            errors.append(f"{prefix}: negative example has contradicted_lines")


def _collect_unexpected_fields(
    examples: Iterable[FlawedFictionsExample],
) -> tuple[str, ...]:
    unexpected = set()
    required = set(REQUIRED_FIELDS)
    for example in examples:
        unexpected.update(set(example.raw_record) - required)
    return tuple(sorted(unexpected))
