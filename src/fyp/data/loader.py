"""Load official FLAWEDFICTIONS raw Parquet files without mutating them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from fyp.data.schema import (
    INCONSISTENCY_CONTINUITY_ERROR,
    INCONSISTENCY_NONE,
    REQUIRED_FIELDS,
    SPLIT_SPECS,
    FlawedFictionsExample,
    FlawedFictionsSplit,
)


DEFAULT_RAW_DATA_DIR = Path("data/raw/flawed-fictions")
NO_GOLD_LINE_MARKERS = frozenset({"", "na", "n/a", "none", "null"})


class DatasetLoadError(RuntimeError):
    """Raised when raw FLAWEDFICTIONS files cannot be read."""


def parse_label(value: Any) -> bool:
    """Convert the official float label into a boolean continuity-error label."""

    if isinstance(value, bool):
        numeric = 1.0 if value else 0.0
    elif isinstance(value, (int, float)):
        numeric = float(value)
    else:
        raise ValueError(f"Invalid cont_error label type: {type(value).__name__}")

    if numeric == 1.0:
        return True
    if numeric == 0.0:
        return False
    raise ValueError(f"Invalid cont_error label value: {value!r}")


def normalize_gold_lines(lines: Any) -> tuple[str, ...]:
    """Parse official line fields using the same broad rules as official code.

    Official code treats line fields as strings, splits either on `</li>` or
    newlines, removes leading list markers `-` and `*`, strips whitespace, and
    drops empty entries. This function keeps that behavior deterministic.
    """

    if lines is None:
        return ()
    if not isinstance(lines, str):
        raise ValueError(f"Gold line field must be a string, got {type(lines).__name__}")

    if lines.strip().lower() in NO_GOLD_LINE_MARKERS:
        return ()

    if "<li>" in lines:
        parts = lines.split("</li>")
    else:
        parts = lines.split("\n")
    cleaned = []
    for part in parts:
        line = part.strip().replace("-", "").replace("*", "").strip()
        if line and line.lower() not in NO_GOLD_LINE_MARKERS:
            cleaned.append(line)
    return tuple(cleaned)


def normalize_record(
    raw_record: Mapping[str, Any],
    split: FlawedFictionsSplit | str,
) -> FlawedFictionsExample:
    """Normalize one official dataset row into a typed example."""

    split = FlawedFictionsSplit.parse(split)
    missing = [field for field in REQUIRED_FIELDS if field not in raw_record]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    story = _require_string(raw_record["story"], "story")
    example_id = _require_string(raw_record["example_id"], "example_id")
    explanation = _require_string(raw_record["cont_error_expl"], "cont_error_expl")
    cont_error_lines_raw = _require_string(
        raw_record["cont_error_lines"], "cont_error_lines"
    )
    contradicted_lines_raw = _require_string(
        raw_record["contradicted_lines"], "contradicted_lines"
    )
    has_error = parse_label(raw_record["cont_error"])
    raw_label = 1.0 if has_error else 0.0
    inconsistency_type = (
        INCONSISTENCY_CONTINUITY_ERROR if has_error else INCONSISTENCY_NONE
    )

    return FlawedFictionsExample(
        split=split,
        example_id=example_id,
        story=story,
        has_continuity_error=has_error,
        cont_error_expl=explanation,
        cont_error_lines_raw=cont_error_lines_raw,
        contradicted_lines_raw=contradicted_lines_raw,
        cont_error_lines=normalize_gold_lines(cont_error_lines_raw),
        contradicted_lines=normalize_gold_lines(contradicted_lines_raw),
        inconsistency_type=inconsistency_type,
        raw_label=raw_label,
        raw_record=dict(raw_record),
    )


def load_split(
    split: FlawedFictionsSplit | str,
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
) -> list[FlawedFictionsExample]:
    """Load one official split from local raw Parquet data."""

    split = FlawedFictionsSplit.parse(split)
    path = Path(raw_data_dir) / SPLIT_SPECS[split].filename
    records = read_parquet_records(path)
    return [normalize_record(record, split) for record in records]


def load_all_splits(
    splits: Sequence[FlawedFictionsSplit | str] | None = None,
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
) -> dict[FlawedFictionsSplit, list[FlawedFictionsExample]]:
    """Load multiple splits keyed by official split enum."""

    selected: Iterable[FlawedFictionsSplit | str] = splits or tuple(SPLIT_SPECS)
    loaded = {}
    for split in selected:
        parsed = FlawedFictionsSplit.parse(split)
        loaded[parsed] = load_split(parsed, raw_data_dir=raw_data_dir)
    return loaded


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    """Read a local Parquet file as Python dictionaries."""

    if not path.exists():
        raise DatasetLoadError(f"Raw dataset file not found: {path}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise DatasetLoadError(
            "pyarrow is required to read official FLAWEDFICTIONS Parquet files. "
            "Install the project dependencies in .venv first."
        ) from exc

    table = pq.read_table(path)
    return [dict(record) for record in table.to_pylist()]


def _require_string(value: Any, field: str) -> str:
    if value is None:
        raise ValueError(f"Field {field!r} is null")
    if not isinstance(value, str):
        raise ValueError(f"Field {field!r} must be a string, got {type(value).__name__}")
    return value
