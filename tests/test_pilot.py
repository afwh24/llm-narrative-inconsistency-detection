from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from fyp.data.loader import load_all_splits
from fyp.data.pilot import generate_pilot_20_records, load_pilot_examples, load_pilot_manifest
from fyp.data.schema import FlawedFictionsSplit
from fyp.experiments.runner import RunConfig, run_experiment
from fyp.models.base import ModelConfig, ModelRequest, ModelResponse


PILOT_PATH = Path("data/pilot/pilot_20.json")
FINAL_MAIN_PATH = Path("data/final/final_flawed_fictions_minus_pilot.json")
FINAL_LONG_PATH = Path("data/final/final_flawed_fictions_long_minus_pilot.json")


class FakeClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def generate(self, request: ModelRequest | str) -> ModelResponse:
        return ModelResponse(
            model=self.config.model,
            response=(
                "<response>\n"
                "<explanation>No issue.</explanation>\n"
                "<error_lines></error_lines>\n"
                "<contradicted_lines></contradicted_lines>\n"
                "<decision>No continuity error found</decision>\n"
                "</response>"
            ),
            thinking=None,
            raw={"model": self.config.model, "response": "redacted", "done": True},
            effective_config=self.config.effective_ollama_request_fields(),
        )


@pytest.fixture(autouse=True)
def _avoid_live_ollama_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fyp.experiments.runner._ollama_version", lambda config: "not_checked")


def test_pilot_manifest_has_exact_fixed_composition() -> None:
    manifest = load_pilot_manifest(PILOT_PATH)
    records = manifest.records

    assert len(records) == 20
    assert all("story" not in record for record in manifest.raw["records"])
    assert len({(record.subset, record.example_id) for record in records}) == 20

    subset_counts = Counter(record.subset for record in records)
    label_counts = Counter(record.cont_error for record in records)
    strata_counts = Counter((record.subset, record.cont_error) for record in records)

    assert subset_counts[FlawedFictionsSplit.FLAWED_FICTIONS] == 10
    assert subset_counts[FlawedFictionsSplit.FLAWED_FICTIONS_LONG] == 10
    assert label_counts[0.0] == 10
    assert label_counts[1.0] == 10
    assert strata_counts[(FlawedFictionsSplit.FLAWED_FICTIONS, 0.0)] == 5
    assert strata_counts[(FlawedFictionsSplit.FLAWED_FICTIONS, 1.0)] == 5
    assert strata_counts[(FlawedFictionsSplit.FLAWED_FICTIONS_LONG, 0.0)] == 5
    assert strata_counts[(FlawedFictionsSplit.FLAWED_FICTIONS_LONG, 1.0)] == 5


def test_pilot_manifest_ids_exist_and_labels_match_official_data() -> None:
    manifest = load_pilot_manifest(PILOT_PATH)

    examples = load_pilot_examples(manifest)

    assert len(examples) == 20
    assert [example.example_id for example in examples] == [
        record.example_id for record in manifest.records
    ]
    assert [example.cont_error for example in examples] == [
        record.cont_error for record in manifest.records
    ]


def test_pilot_selection_is_deterministic_and_manifest_matches_generator() -> None:
    manifest = load_pilot_manifest(PILOT_PATH)
    dataset = load_all_splits(
        [
            FlawedFictionsSplit.FLAWED_FICTIONS,
            FlawedFictionsSplit.FLAWED_FICTIONS_LONG,
        ]
    )

    first = generate_pilot_20_records(dataset)
    second = generate_pilot_20_records(dataset)

    assert first == second
    assert [record.to_manifest_dict() for record in first] == [
        record.to_manifest_dict() for record in manifest.records
    ]


def test_experiment_runner_can_use_pilot_manifest_without_limit(tmp_path: Path) -> None:
    manifest = load_pilot_manifest(PILOT_PATH)
    config = RunConfig(
        model="qwen3:4b",
        strategy="vanilla",
        subsets=manifest.subsets,
        run_id="pilot-mock-run",
        run_root=tmp_path,
        manifest_path=PILOT_PATH,
    )

    result = run_experiment(config, client_factory=FakeClient)

    assert result.expected_examples == 20
    assert result.completed_examples == 20
    metadata = (result.run_dir / "metadata.json").read_text(encoding="utf-8")
    assert "pilot_20" in metadata
    assert "pilot_manifest_hash" in metadata
    summary = (result.run_dir / "run_summary.md").read_text(encoding="utf-8")
    assert "PILOT RESULTS \u2014 N=20" in summary
    assert "EXPLORATORY ONLY" in summary
    assert "NOT FULL FLAWEDFICTIONS BENCHMARK RESULTS" in summary


def test_final_manifests_exclude_pilot_ids_and_are_not_pilot_labeled(tmp_path: Path) -> None:
    pilot = load_pilot_manifest(PILOT_PATH)
    pilot_ids = {(record.subset, record.example_id) for record in pilot.records}

    main = load_pilot_manifest(FINAL_MAIN_PATH)
    long = load_pilot_manifest(FINAL_LONG_PATH)

    assert main.role == "final_confirmatory"
    assert long.role == "final_confirmatory"
    assert len(main.records) == 404
    assert len(long.records) == 190
    assert not ({(record.subset, record.example_id) for record in main.records} & pilot_ids)
    assert not ({(record.subset, record.example_id) for record in long.records} & pilot_ids)

    result = run_experiment(
        RunConfig(
            model="qwen3:4b",
            strategy="vanilla",
            subsets=main.subsets,
            run_id="final-manifest-mock-run",
            run_root=tmp_path,
            manifest_path=FINAL_MAIN_PATH,
            max_new_examples=1,
        ),
        client_factory=FakeClient,
    )

    metadata = (result.run_dir / "metadata.json").read_text(encoding="utf-8")
    summary = (result.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "fixed_manifest_name" in metadata
    assert "pilot_name" not in metadata
    assert "PRE-FLIGHT PARTIAL RUN" in summary
    assert "PILOT RESULTS" not in summary
