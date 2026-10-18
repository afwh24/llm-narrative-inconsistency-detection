from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fyp.data.loader import load_split, normalize_gold_lines, normalize_record
from fyp.data.schema import SPLIT_SPECS, FlawedFictionsSplit, SplitSpec
from fyp.data.validation import ValidationError, validate_dataset, validate_examples


def _raw_record(**overrides):
    record = {
        "story": "The lamp was blue. Later, the lamp was red.",
        "cont_error": 1.0,
        "cont_error_expl": "The lamp color changes.",
        "cont_error_lines": '<li>"Later, the lamp was red."</li>',
        "contradicted_lines": '<li>"The lamp was blue."</li>',
        "example_id": "example-1",
    }
    record.update(overrides)
    return record


def _small_spec(count: int, positive: int, negative: int) -> SplitSpec:
    return SplitSpec(
        split=FlawedFictionsSplit.FLAWED_FICTIONS,
        filename="flawed_fictions.parquet",
        expected_examples=count,
        expected_positive=positive,
        expected_negative=negative,
        notes="unit-test fixture",
    )


def test_load_split_reads_official_shape_from_parquet(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = [
        _raw_record(example_id="positive-1"),
        _raw_record(
            story="A consistent story.",
            cont_error=0.0,
            cont_error_expl="No error",
            cont_error_lines="NA",
            contradicted_lines="NA",
            example_id="negative-1",
        ),
    ]
    pq.write_table(pa.Table.from_pylist(rows), raw_dir / "flawed_fictions.parquet")

    examples = load_split(FlawedFictionsSplit.FLAWED_FICTIONS, raw_data_dir=raw_dir)

    assert [example.example_id for example in examples] == ["positive-1", "negative-1"]
    assert examples[0].has_continuity_error is True
    assert examples[1].has_continuity_error is False
    assert examples[1].cont_error_lines == ()
    assert examples[1].contradicted_lines == ()


def test_validation_accepts_expected_small_fixture() -> None:
    examples = [
        normalize_record(_raw_record(example_id="positive-1"), FlawedFictionsSplit.FLAWED_FICTIONS),
        normalize_record(
            _raw_record(
                story="A consistent story.",
                cont_error=0.0,
                cont_error_expl="No error",
                cont_error_lines="NA",
                contradicted_lines="NA",
                example_id="negative-1",
            ),
            FlawedFictionsSplit.FLAWED_FICTIONS,
        ),
    ]

    summary = validate_examples(
        examples,
        FlawedFictionsSplit.FLAWED_FICTIONS,
        expected=_small_spec(count=2, positive=1, negative=1),
    )

    assert summary.examples == 2
    assert summary.positive == 1
    assert summary.negative == 1


def test_duplicate_ids_are_rejected() -> None:
    examples = [
        normalize_record(_raw_record(example_id="same"), FlawedFictionsSplit.FLAWED_FICTIONS),
        normalize_record(_raw_record(example_id="same"), FlawedFictionsSplit.FLAWED_FICTIONS),
    ]

    with pytest.raises(ValidationError, match="duplicate example_id"):
        validate_examples(
            examples,
            FlawedFictionsSplit.FLAWED_FICTIONS,
            expected=_small_spec(count=2, positive=2, negative=0),
        )


def test_invalid_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid cont_error label value"):
        normalize_record(_raw_record(cont_error=0.5), FlawedFictionsSplit.FLAWED_FICTIONS)


def test_malformed_record_missing_required_field_is_rejected() -> None:
    raw = _raw_record()
    raw.pop("story")

    with pytest.raises(ValueError, match="Missing required field"):
        normalize_record(raw, FlawedFictionsSplit.FLAWED_FICTIONS)


def test_normalization_is_deterministic_and_preserves_raw_fields() -> None:
    raw = _raw_record(
        cont_error_lines=' - <li>"Later, the lamp was red."</li>\n* NA',
        contradicted_lines="NA",
    )

    first = normalize_record(raw, FlawedFictionsSplit.FLAWED_FICTIONS)
    second = normalize_record(raw, FlawedFictionsSplit.FLAWED_FICTIONS)

    assert first == second
    assert first.raw_record == raw
    assert first.cont_error_lines == ('<li>"Later, the lamp was red."',)
    assert first.contradicted_lines == ()


def test_gold_line_markers_normalize_to_empty_tuple() -> None:
    assert normalize_gold_lines("NA") == ()
    assert normalize_gold_lines("n/a") == ()
    assert normalize_gold_lines("") == ()


def test_local_official_raw_dataset_statistics_when_available() -> None:
    raw_dir = Path("data/raw/flawed-fictions")
    if not all((raw_dir / spec.filename).exists() for spec in SPLIT_SPECS.values()):
        pytest.skip("official raw FLAWEDFICTIONS Parquet files are not present locally")

    dataset = {
        split: load_split(split, raw_data_dir=raw_dir)
        for split in SPLIT_SPECS
    }

    report = validate_dataset(dataset)

    assert report.total_examples == 1442
