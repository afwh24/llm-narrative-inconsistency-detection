"""Fixed manifest support for pilot and final scoped experiments."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fyp.data.loader import DEFAULT_RAW_DATA_DIR, load_all_splits
from fyp.data.schema import FlawedFictionsExample, FlawedFictionsSplit


DEFAULT_PILOT_DIR = Path("data/pilot")
DEFAULT_FINAL_DIR = Path("data/final")
PILOT_MANIFESTS = {
    "pilot_20": DEFAULT_PILOT_DIR / "pilot_20.json",
    "final_flawed_fictions_minus_pilot": DEFAULT_FINAL_DIR
    / "final_flawed_fictions_minus_pilot.json",
    "final_flawed_fictions_long_minus_pilot": DEFAULT_FINAL_DIR
    / "final_flawed_fictions_long_minus_pilot.json",
}


class PilotManifestError(ValueError):
    """Raised when a pilot manifest is malformed or inconsistent."""


@dataclass(frozen=True)
class PilotRecord:
    """One manifest row pointing to an official dataset example."""

    example_id: str
    subset: FlawedFictionsSplit
    cont_error: float
    selection_hash: str

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "subset": self.subset.value,
            "cont_error": self.cont_error,
            "selection_hash": self.selection_hash,
        }


@dataclass(frozen=True)
class PilotManifest:
    """Validated fixed manifest without duplicated story text."""

    name: str
    path: Path
    records: tuple[PilotRecord, ...]
    raw: Mapping[str, Any]

    @property
    def role(self) -> str:
        value = self.raw.get("scope_role", self.raw.get("manifest_role"))
        if isinstance(value, str) and value:
            return value
        return "pilot" if self.name.startswith("pilot") else "fixed_scope"

    @property
    def subsets(self) -> tuple[FlawedFictionsSplit, ...]:
        seen: list[FlawedFictionsSplit] = []
        for record in self.records:
            if record.subset not in seen:
                seen.append(record.subset)
        return tuple(seen)

    @property
    def manifest_hash(self) -> str:
        return stable_file_hash(self.path)


def resolve_pilot_manifest(name_or_path: str | Path) -> Path:
    """Resolve a named fixed manifest or an explicit manifest path."""

    value = Path(name_or_path)
    if value.exists():
        return value
    key = str(name_or_path)
    if key in PILOT_MANIFESTS:
        return PILOT_MANIFESTS[key]
    raise PilotManifestError(f"Unknown pilot manifest {name_or_path!r}")


def load_pilot_manifest(path: str | Path) -> PilotManifest:
    """Load and validate a fixed manifest."""

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PilotManifestError(f"Pilot manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PilotManifestError(f"Pilot manifest is invalid JSON: {path}") from exc
    if not isinstance(raw, Mapping):
        raise PilotManifestError("Pilot manifest root must be an object")
    name = _require_string(raw, "name")
    records_raw = raw.get("records")
    if not isinstance(records_raw, list):
        raise PilotManifestError("Pilot manifest must contain a records list")

    records = []
    seen_ids: set[tuple[FlawedFictionsSplit, str]] = set()
    for idx, item in enumerate(records_raw):
        if not isinstance(item, Mapping):
            raise PilotManifestError(f"Manifest record {idx} must be an object")
        subset = FlawedFictionsSplit.parse(_require_string(item, "subset"))
        example_id = _require_string(item, "example_id")
        cont_error = _require_label(item.get("cont_error"), idx)
        selection_hash = _require_string(item, "selection_hash")
        expected_hash = stable_example_id_hash(example_id)
        if selection_hash != expected_hash:
            raise PilotManifestError(
                f"Manifest record {idx} selection_hash mismatch for {example_id}"
            )
        key = (subset, example_id)
        if key in seen_ids:
            raise PilotManifestError(f"Duplicate pilot example: {subset.value}/{example_id}")
        seen_ids.add(key)
        records.append(
            PilotRecord(
                example_id=example_id,
                subset=subset,
                cont_error=cont_error,
                selection_hash=selection_hash,
            )
        )
    return PilotManifest(name=name, path=path, records=tuple(records), raw=raw)


def load_pilot_examples(
    manifest: PilotManifest,
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
) -> list[FlawedFictionsExample]:
    """Load official dataset examples referenced by a validated manifest."""

    dataset = load_all_splits(manifest.subsets, raw_data_dir=raw_data_dir)
    index = {
        (example.split, example.example_id): example
        for examples in dataset.values()
        for example in examples
    }
    loaded = []
    for record in manifest.records:
        example = index.get((record.subset, record.example_id))
        if example is None:
            raise PilotManifestError(
                f"Pilot example not found in official data: {record.subset.value}/{record.example_id}"
            )
        if example.cont_error != record.cont_error:
            raise PilotManifestError(
                f"Pilot label mismatch for {record.subset.value}/{record.example_id}: "
                f"manifest={record.cont_error}, official={example.cont_error}"
            )
        loaded.append(example)
    return loaded


def generate_pilot_20_records(
    dataset: Mapping[FlawedFictionsSplit, Sequence[FlawedFictionsExample]] | None = None,
) -> tuple[PilotRecord, ...]:
    """Generate the deterministic 20-example pilot record list."""

    splits = (
        FlawedFictionsSplit.FLAWED_FICTIONS,
        FlawedFictionsSplit.FLAWED_FICTIONS_LONG,
    )
    dataset = dataset or load_all_splits(splits)
    records = []
    for split in splits:
        strata: dict[float, list[FlawedFictionsExample]] = defaultdict(list)
        for example in dataset[split]:
            strata[example.cont_error].append(example)
        for label in (0.0, 1.0):
            selected = sorted(
                strata[label],
                key=lambda example: (stable_example_id_hash(example.example_id), example.example_id),
            )[:5]
            if len(selected) != 5:
                raise PilotManifestError(f"Not enough examples for {split.value} label {label}")
            records.extend(
                PilotRecord(
                    example_id=example.example_id,
                    subset=split,
                    cont_error=example.cont_error,
                    selection_hash=stable_example_id_hash(example.example_id),
                )
                for example in selected
            )
    return tuple(records)


def stable_example_id_hash(example_id: str) -> str:
    """Return stable SHA-256 hash used for deterministic pilot selection."""

    return hashlib.sha256(example_id.encode("utf-8")).hexdigest().upper()


def stable_file_hash(path: Path) -> str:
    """Return byte-level SHA-256 for a manifest file."""

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _require_string(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise PilotManifestError(f"Missing or invalid string field {field!r}")
    return value


def _require_label(value: object, idx: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotManifestError(f"Manifest record {idx} has invalid cont_error")
    label = float(value)
    if label not in (0.0, 1.0):
        raise PilotManifestError(f"Manifest record {idx} has invalid cont_error")
    return label
