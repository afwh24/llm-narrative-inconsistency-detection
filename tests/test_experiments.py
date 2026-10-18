from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fyp.data.schema import FlawedFictionsExample, FlawedFictionsSplit
from fyp.experiments.runner import ExperimentRunError, RunConfig, run_experiment
from fyp.models.base import ModelConfig, ModelRequest, ModelResponse
from fyp.models.ollama_client import OllamaInterruptedError


class FakeClient:
    def __init__(self, config: ModelConfig, failures: int = 0) -> None:
        self.config = config
        self.failures = failures
        self.prompts: list[str] = []

    def generate(self, request: ModelRequest | str) -> ModelResponse:
        prompt = request.prompt if isinstance(request, ModelRequest) else request
        self.prompts.append(prompt)
        if self.failures > 0:
            self.failures -= 1
            raise TimeoutError("synthetic timeout")
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


def _example(example_id: str) -> FlawedFictionsExample:
    split = FlawedFictionsSplit.FLAWED_FICTIONS
    story = f"{example_id} packed a map. {example_id} followed the map home."
    return FlawedFictionsExample(
        split=split,
        example_id=example_id,
        story=story,
        has_continuity_error=False,
        cont_error_expl="No error",
        cont_error_lines_raw="NA",
        contradicted_lines_raw="NA",
        cont_error_lines=(),
        contradicted_lines=(),
        inconsistency_type="none",
        raw_label=0.0,
        raw_record={
            "story": story,
            "cont_error": 0.0,
            "cont_error_expl": "No error",
            "cont_error_lines": "NA",
            "contradicted_lines": "NA",
            "example_id": example_id,
        },
    )


def _dataset() -> dict[FlawedFictionsSplit, list[FlawedFictionsExample]]:
    return {FlawedFictionsSplit.FLAWED_FICTIONS: [_example("ex-1"), _example("ex-2")]}


def _config(tmp_path: Path, **overrides: Any) -> RunConfig:
    values = {
        "model": "qwen3:4b",
        "strategy": "vanilla",
        "subsets": (FlawedFictionsSplit.FLAWED_FICTIONS,),
        "limit": 2,
        "run_id": "tiny-run",
        "run_root": tmp_path,
    }
    values.update(overrides)
    return RunConfig(**values)


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(autouse=True)
def _avoid_live_ollama_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fyp.experiments.runner._ollama_version", lambda config: "not_checked")


def test_run_writes_required_artifacts_and_metadata(tmp_path: Path) -> None:
    client = FakeClient

    result = run_experiment(_config(tmp_path), dataset=_dataset(), client_factory=client)

    assert result.completed_examples == 2
    assert result.failed_examples == 0
    for name in (
        "config.yaml",
        "metadata.json",
        "predictions.jsonl",
        "errors.jsonl",
        "metrics.json",
        "summary.md",
    ):
        assert (result.run_dir / name).exists()
    metadata = json.loads((result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    records = _records(result.run_dir / "predictions.jsonl")
    assert metadata["model"] == "qwen3:4b"
    assert metadata["benchmark_strategy"] == "vanilla"
    assert metadata["dataset"]["subsets"] == ["flawed_fictions"]
    assert metadata["model_configuration"]["generation"]["temperature"] == 0.5
    assert metadata["model_configuration"]["generation"]["num_predict"] == 4096
    assert all(isinstance(record["latency_seconds"], float) for record in records)


def test_prompt_version_override_is_recorded_and_resume_is_version_checked(tmp_path: Path) -> None:
    version2 = run_experiment(
        _config(tmp_path, strategy="evidence_first", prompt_version="evidence_first_v2"),
        dataset=_dataset(), client_factory=FakeClient,
    )
    rows = _records(version2.run_dir / "predictions.jsonl")
    assert all(row["prompt_version"] == "evidence_first_v2" for row in rows)
    resumed = run_experiment(
        _config(tmp_path, strategy="evidence_first", prompt_version="evidence_first_v2", resume=True),
        dataset=_dataset(), client_factory=FakeClient,
    )
    assert resumed.skipped_completed == 2
    with pytest.raises(ExperimentRunError, match="incompatible"):
        run_experiment(
            _config(tmp_path, strategy="evidence_first", prompt_version="evidence_first_v1", resume=True),
            dataset=_dataset(), client_factory=FakeClient,
        )


def test_interrupted_inference_is_not_written_as_terminal_failure(tmp_path: Path) -> None:
    class InterruptedClient(FakeClient):
        def generate(self, request):
            raise OllamaInterruptedError("synthetic interruption")
    with pytest.raises(OllamaInterruptedError):
        run_experiment(_config(tmp_path), dataset=_dataset(), client_factory=InterruptedClient)
    assert _records(tmp_path / "tiny-run" / "predictions.jsonl") == []
    assert _records(tmp_path / "tiny-run" / "errors.jsonl") == []
    metadata = json.loads((tmp_path / "tiny-run" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["failed_example_count"] == 0
def test_run_can_override_native_thinking_mode(tmp_path: Path) -> None:
    result = run_experiment(
        _config(tmp_path, think_override=True),
        dataset=_dataset(),
        client_factory=FakeClient,
    )

    metadata = json.loads((result.run_dir / "metadata.json").read_text(encoding="utf-8"))
    records = _records(result.run_dir / "predictions.jsonl")
    assert metadata["thinking_mode"] is True
    assert metadata["model_configuration"]["generation"]["think"] is True
    assert all(record["thinking_mode"] is True for record in records)
    assert all(record["effective_config"]["think"] is True for record in records)


def test_resume_skips_completed_without_duplicates(tmp_path: Path) -> None:
    first_client = FakeClient
    first = run_experiment(_config(tmp_path), dataset=_dataset(), client_factory=first_client)

    second_client = FakeClient
    second = run_experiment(
        _config(tmp_path, resume=True),
        dataset=_dataset(),
        client_factory=second_client,
    )

    records = _records(first.run_dir / "predictions.jsonl")
    assert second.skipped_completed == 2
    assert len(records) == 2
    assert len({record["cache_key"] for record in records}) == 2


def test_max_new_examples_creates_resumable_partial_run(tmp_path: Path) -> None:
    first = run_experiment(
        _config(tmp_path, max_new_examples=1),
        dataset=_dataset(),
        client_factory=FakeClient,
    )

    assert first.expected_examples == 2
    assert first.completed_examples == 1
    partial_summary = (first.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "PRE-FLIGHT PARTIAL RUN" in partial_summary
    assert "NOT FINAL RESULTS" in partial_summary

    resumed = run_experiment(
        _config(tmp_path, resume=True),
        dataset=_dataset(),
        client_factory=FakeClient,
    )

    records = _records(resumed.run_dir / "predictions.jsonl")
    assert resumed.expected_examples == 2
    assert resumed.completed_examples == 2
    assert resumed.skipped_completed == 1
    assert len(records) == 2
    assert len({record["cache_key"] for record in records}) == 2


def test_resume_rejects_incompatible_configuration(tmp_path: Path) -> None:
    run_experiment(_config(tmp_path), dataset=_dataset(), client_factory=FakeClient)

    with pytest.raises(ExperimentRunError, match="incompatible"):
        run_experiment(
            _config(tmp_path, strategy="cot", resume=True),
            dataset=_dataset(),
            client_factory=FakeClient,
        )


def test_failed_records_remain_identifiable_and_retryable(tmp_path: Path) -> None:
    failing = FakeClient

    result = run_experiment(
        _config(tmp_path, limit=1),
        dataset=_dataset(),
        client_factory=lambda config: failing(config, failures=1),
    )

    assert result.failed_examples == 1
    failed_records = _records(result.run_dir / "errors.jsonl")
    assert failed_records[0]["target_inference_failure"] is True

    resumed = run_experiment(
        _config(tmp_path, limit=1, resume=True, retry_failures=True),
        dataset=_dataset(),
        client_factory=FakeClient,
    )

    prediction_records = _records(resumed.run_dir / "predictions.jsonl")
    assert resumed.completed_examples == 1
    assert resumed.failed_examples == 0
    assert len(prediction_records) == 1
    assert prediction_records[0]["target_inference_failure"] is False
    assert len(_records(resumed.run_dir / "errors.jsonl")) == 1


def test_cache_key_isolates_prompt_strategy(tmp_path: Path) -> None:
    vanilla = run_experiment(
        _config(tmp_path, run_id="vanilla-run", strategy="vanilla", limit=1),
        dataset=_dataset(),
        client_factory=FakeClient,
    )
    cot = run_experiment(
        _config(tmp_path, run_id="cot-run", strategy="cot", limit=1),
        dataset=_dataset(),
        client_factory=FakeClient,
    )

    vanilla_key = _records(vanilla.run_dir / "predictions.jsonl")[0]["cache_key"]
    cot_key = _records(cot.run_dir / "predictions.jsonl")[0]["cache_key"]
    assert vanilla_key != cot_key


def test_cache_key_isolates_native_thinking_mode(tmp_path: Path) -> None:
    no_think = run_experiment(
        _config(tmp_path, run_id="no-think-run", limit=1, think_override=False),
        dataset=_dataset(),
        client_factory=FakeClient,
    )
    think = run_experiment(
        _config(tmp_path, run_id="think-run", limit=1, think_override=True),
        dataset=_dataset(),
        client_factory=FakeClient,
    )

    no_think_key = _records(no_think.run_dir / "predictions.jsonl")[0]["cache_key"]
    think_key = _records(think.run_dir / "predictions.jsonl")[0]["cache_key"]
    assert no_think_key != think_key


def test_resume_rejects_native_thinking_mismatch(tmp_path: Path) -> None:
    run_experiment(
        _config(tmp_path, think_override=False),
        dataset=_dataset(),
        client_factory=FakeClient,
    )

    with pytest.raises(ExperimentRunError, match="incompatible"):
        run_experiment(
            _config(tmp_path, resume=True, think_override=True),
            dataset=_dataset(),
            client_factory=FakeClient,
        )


def test_resume_rejects_manifest_mismatch(tmp_path: Path) -> None:
    run_experiment(_config(tmp_path), dataset=_dataset(), client_factory=FakeClient)

    with pytest.raises(ExperimentRunError, match="incompatible"):
        run_experiment(
            _config(
                tmp_path,
                resume=True,
                limit=None,
                manifest_path=Path("data/pilot/pilot_20.json"),
                subsets=(
                    FlawedFictionsSplit.FLAWED_FICTIONS,
                    FlawedFictionsSplit.FLAWED_FICTIONS_LONG,
                ),
            ),
            client_factory=FakeClient,
        )
