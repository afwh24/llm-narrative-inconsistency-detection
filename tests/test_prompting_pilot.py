from __future__ import annotations

import json
from pathlib import Path
from string import Formatter

import pytest

from fyp.experiments import prompting_pilot
from fyp.experiments.runner import ExperimentRunError
from fyp.prompting.strategies import get_benchmark_strategy


@pytest.mark.parametrize("strategy_name", ["evidence_first", "structured_check", "verify_revise"])
def test_frozen_prompts_are_single_call_final_only_and_story_only(strategy_name) -> None:
    settings, base = prompting_pilot.load_settings()
    entry = settings["strategies"][strategy_name]
    strategy = get_benchmark_strategy(strategy_name, version=entry["version"])
    text = strategy.template_path.read_text(encoding="utf-8")
    assert [field for _, field, _, _ in Formatter().parse(text) if field is not None] == ["story"]
    result = strategy.build_prompt({"story": "SYNTHETIC_STORY", "cont_error": 1, "cont_error_expl": "SECRET_GOLD"})
    assert strategy.num_model_calls == len(result.stages) == 1
    assert result.template_hash == entry["hash"]
    assert result.prompt.count("SYNTHETIC_STORY") == 1 and "SECRET_GOLD" not in result.prompt
    assert "Return only one final <response> block" in text
    assert "Do not expose a scratchpad" in text
    assert "real-world implausibility alone" in text
    for tag in ("response", "explanation", "error_lines", "contradicted_lines", "decision"):
        assert f"<{tag}>" in text and f"</{tag}>" in text
    assert "No continuity error found" in text
    assert "<scratchpad>" not in text and "{examples}" not in text


def test_baselines_remain_method_compatible_without_rewriting_artifacts() -> None:
    settings, base = prompting_pilot.load_settings()
    manifest, examples = prompting_pilot.validate_scope(settings, base)
    for model in settings["models"]:
        result = prompting_pilot.validate_baseline(model, settings, base, manifest, examples)
        assert result["status"] == "REUSABLE_COMPLETE" and result["records"] == 10


def test_preparation_freezes_matrix_and_never_generates(tmp_path, monkeypatch) -> None:
    settings, base = prompting_pilot.load_settings()
    settings["preparation_dir"] = str(tmp_path / "preparation")
    settings["run_root"] = str(tmp_path / "runs")
    baseline_meta = json.loads((Path(base["preparation_dir"]) / "metadata.json").read_text(encoding="utf-8"))
    inspection = {"models": baseline_meta["provenance"]["models"],
                  "ollama_version": baseline_meta["provenance"]["ollama_version"]}
    monkeypatch.setattr(prompting_pilot, "load_settings", lambda: (settings, base))
    monkeypatch.setattr("fyp.experiments.baseline_pilot.inspect_models", lambda *_: inspection)
    monkeypatch.setattr("fyp.experiments.baseline_pilot._local_resources", lambda: {})
    def unexpected_inference(*args, **kwargs):
        pytest.fail("Preparation must not invoke model generation")
    monkeypatch.setattr("fyp.models.ollama_client.OllamaClient.generate", unexpected_inference)
    monkeypatch.setattr(prompting_pilot, "run_experiment", unexpected_inference)
    metadata = prompting_pilot.prepare_prompting_pilot()
    assert metadata["status"] == "PREPARED_NO_INFERENCE"
    assert metadata["new_generations_executed"] == 0
    assert metadata["matrix_records"] == 120 and metadata["new_records_planned"] == 90
    assert len(metadata["matrix"]) == 12
    assert sum(row["status"] == "REUSE_BASELINE" for row in metadata["matrix"]) == 3
    assert not list(tmp_path.rglob("predictions.jsonl"))
    reference = prompting_pilot.run_prompting_pilot("qwen3:8b", "vanilla")
    assert reference["status"] == "REUSABLE_COMPLETE"


def test_prompting_resume_blocks_changed_freeze_before_inference(tmp_path, monkeypatch) -> None:
    settings, base = prompting_pilot.load_settings()
    settings["preparation_dir"] = str(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({"provenance": {"source_hashes": {}}}), encoding="utf-8")
    monkeypatch.setattr(prompting_pilot, "load_settings", lambda: (settings, base))
    monkeypatch.setattr(prompting_pilot, "run_experiment", lambda *_: pytest.fail("Must refuse changed freeze"))
    with pytest.raises(ExperimentRunError, match="changed since freeze"):
        prompting_pilot.run_prompting_pilot("qwen3:8b", "evidence_first")


def test_final_experiment_default_prompt_versions_are_unchanged() -> None:
    assert get_benchmark_strategy("evidence_first").prompt_version == "evidence_first_v1"
    assert get_benchmark_strategy("verify_revise").prompt_version == "verify_revise_v1"
    assert get_benchmark_strategy("vanilla").prompt_version == "vanilla_v1"
