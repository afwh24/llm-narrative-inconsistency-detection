"""Create pilot-excluded final FLAWEDFICTIONS manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fyp.data.loader import load_all_splits
from fyp.data.pilot import load_pilot_manifest, stable_example_id_hash, stable_file_hash
from fyp.data.schema import FlawedFictionsSplit


OUTPUT_DIR = Path("data/final")
PILOT_PATH = Path("data/pilot/pilot_20.json")
DATASET_IDENTITY = "kahuja/flawed-fictions"
DATASET_VERSION = "733717281afdb715538b753f0f85ef7d55568b85"

FINAL_SCOPES = {
    "final_flawed_fictions_minus_pilot": FlawedFictionsSplit.FLAWED_FICTIONS,
    "final_flawed_fictions_long_minus_pilot": FlawedFictionsSplit.FLAWED_FICTIONS_LONG,
}


def main() -> int:
    pilot = load_pilot_manifest(PILOT_PATH)
    pilot_ids = {(record.subset, record.example_id) for record in pilot.records}
    dataset = load_all_splits(tuple(FINAL_SCOPES.values()))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for name, split in FINAL_SCOPES.items():
        examples = [
            example
            for example in dataset[split]
            if (split, example.example_id) not in pilot_ids
        ]
        labels = Counter(example.cont_error for example in examples)
        raw = {
            "name": name,
            "purpose": (
                "Pilot-excluded final confirmatory FLAWEDFICTIONS scope. "
                "This is not an official train/dev/test split."
            ),
            "scope_role": "final_confirmatory",
            "selection_date": "2026-09-15",
            "selection_method": (
                "Load the official released benchmark variant in source order and exclude "
                "the exact example IDs listed in data/pilot/pilot_20.json."
            ),
            "dataset": {
                "identity": DATASET_IDENTITY,
                "version": DATASET_VERSION,
                "source_subset": split.value,
                "official_example_count": len(dataset[split]),
                "pilot_manifest": str(PILOT_PATH),
                "pilot_manifest_hash": stable_file_hash(PILOT_PATH),
                "pilot_examples_excluded": len(dataset[split]) - len(examples),
                "final_example_count": len(examples),
                "label_counts": {
                    "0.0": labels[0.0],
                    "1.0": labels[1.0],
                },
            },
            "records": [
                {
                    "example_id": example.example_id,
                    "subset": split.value,
                    "cont_error": example.cont_error,
                    "selection_hash": stable_example_id_hash(example.example_id),
                }
                for example in examples
            ],
        }
        path = OUTPUT_DIR / f"{name}.json"
        path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        written.append(path)

    for path in written:
        manifest = load_pilot_manifest(path)
        print(
            f"{path}: {len(manifest.records)} examples; "
            f"sha256={manifest.manifest_hash}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
