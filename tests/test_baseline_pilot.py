from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from fyp.data.pilot import load_pilot_examples, load_pilot_manifest
from fyp.experiments import baseline_pilot
from fyp.experiments.runner import ExperimentRunError, RunConfig, run_experiment
from fyp.models.base import ModelResponse


def test_scope_reuses_exact_main_ids_original_text_and_hashes() -> None:
    settings = baseline_pilot.load_settings()
    manifest = baseline_pilot.build_manifest(settings)
    parent = load_pilot_manifest(settings["parent_manifest"])
    expected_ids = [r.example_id for r in parent.records if r.subset.value == "flawed_fictions"]
    assert [r["example_id"] for r in manifest["records"]] == expected_ids
    assert manifest["class_counts"] == {"negative": 5, "positive": 5}
    originals = {e.example_id: e for e in load_pilot_examples(parent)}
    for record in manifest["records"]:
        assert record["subset"] == "flawed_fictions"
        assert record["story_hash"] == baseline_pilot._hash_text(originals[record["example_id"]].raw_record["story"])


def test_scope_does_not_replace_missing_or_changed_authoritative_manifest(tmp_path) -> None:
    settings = baseline_pilot.load_settings()
    settings["parent_manifest"] = str(tmp_path / "missing.json")
    with pytest.raises(ValueError, match="not found"):
        baseline_pilot.build_manifest(settings)
    settings["parent_manifest"] = "data/pilot/pilot_20.json"
    settings["parent_manifest_hash"] = "changed"
    with pytest.raises(ExperimentRunError, match="hash changed"):
        baseline_pilot.build_manifest(settings)


def test_context_includes_prompt_template_and_output_and_rejects_overflow() -> None:
    settings = baseline_pilot.load_settings()
    records = baseline_pilot.build_manifest(settings)["records"]
    show = {
        "model_info": {"general.architecture": "llama", "tokenizer.ggml.model": "llama",
                       "llama.context_length": 32768, "llama.block_count": 32,
                       "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
                       "llama.embedding_length": 4096},
        "template": "[INST] {{ .Prompt }} [/INST]",
    }
    budget = baseline_pilot.context_budget(settings, records, show)
    assert budget["required_token_upper_bound"] < 16384
    assert budget["output_allowance"] == 4096
    show["model_info"]["llama.context_length"] = 8192
    with pytest.raises(ExperimentRunError, match="Context insufficient"):
        baseline_pilot.context_budget(settings, records, show)


def test_preparation_never_generates(tmp_path, monkeypatch) -> None:
    settings = baseline_pilot.load_settings()
    settings["manifest"] = str(tmp_path / "manifest.json")
    settings["preparation_dir"] = str(tmp_path / "preparation")
    monkeypatch.setattr(baseline_pilot, "load_settings", lambda: settings)
    monkeypatch.setattr(baseline_pilot, "inspect_models", lambda *_: {
        "ollama_version": "mock", "models": {name: {"available": False} for name in settings["models"]},
    })
    monkeypatch.setattr(baseline_pilot, "_local_resources", lambda: {})
    def unexpected_generation(*args, **kwargs):
        pytest.fail("Preparation must never run inference")
    monkeypatch.setattr("fyp.models.ollama_client.OllamaClient.generate", unexpected_generation)
    metadata = baseline_pilot.prepare_pilot()
    assert metadata["status"] == "BLOCKED_MODELS_UNAVAILABLE"
    assert metadata["official_examples_executed"] == 0
    assert not list(tmp_path.rglob("predictions.jsonl"))


def test_new_pilot_resume_preserves_failures_and_raw_records(tmp_path, monkeypatch) -> None:
    settings = baseline_pilot.load_settings()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(baseline_pilot.build_manifest(settings)), encoding="utf-8")
    monkeypatch.setattr("fyp.experiments.runner._ollama_version", lambda _: "mock")
    calls = []
    class Client:
        def __init__(self, config):
            self.config = config
        def generate(self, request):
            calls.append(request.prompt)
            if len(calls) == 1:
                raise TimeoutError("synthetic only")
            return ModelResponse(
                model=self.config.model,
                response="<response><decision>No continuity error found</decision></response>",
                thinking=None, raw={"done": True},
                effective_config=self.config.effective_ollama_request_fields(),
            )
    config = RunConfig(
        model="llama3.1:8b", strategy="vanilla", subsets=load_pilot_manifest(manifest_path).subsets,
        run_id="mock-baseline", run_root=tmp_path / "runs", manifest_path=manifest_path,
        model_config_path=Path(settings["models"]["llama3.1:8b"]["config"]),
        provenance={"model_digest": "mock"}, max_new_examples=2,
    )
    first = run_experiment(config, client_factory=Client)
    before = (first.run_dir / "predictions.jsonl").read_bytes()
    resumed = run_experiment(replace(config, resume=True, max_new_examples=None), client_factory=Client)
    assert len(calls) == 10
    assert resumed.skipped_completed == 1 and resumed.skipped_failed == 1
    assert resumed.failed_examples == 1 and resumed.completed_examples == 9
    assert (first.run_dir / "predictions.jsonl").read_bytes().startswith(before)
    assert "N=10" in (first.run_dir / "summary.md").read_text(encoding="utf-8")
    assert "N=10" in (first.run_dir / "run_summary.md").read_text(encoding="utf-8")
    with pytest.raises(ExperimentRunError, match="incompatible"):
        run_experiment(replace(config, resume=True, provenance={"model_digest": "changed"}), client_factory=Client)

    settings["manifest"] = str(manifest_path)
    settings["run_root"] = str(config.run_root)
    settings["export_dir"] = str(tmp_path / "exports")
    settings["models"]["llama3.1:8b"]["run_id"] = config.run_id
    monkeypatch.setattr(baseline_pilot, "load_settings", lambda: settings)
    exported = baseline_pilot.export_baseline("llama3.1:8b")
    assert exported.name == "llama3_1_8b_baseline.json"
    raw_copy = exported.with_suffix(".jsonl")
    assert raw_copy.read_bytes() == (first.run_dir / "predictions.jsonl").read_bytes()
    simple = json.loads(exported.read_text(encoding="utf-8"))
    assert list(simple) == ["model", "prompt", "total", "failed", "metrics", "results"]
    assert simple["prompt"] == "baseline" and simple["total"] == 10 and simple["failed"] == 1
    assert simple["metrics"] == {"accuracy": 0.4, "precision": 0.0, "recall": 0.0,
                                 "f1": 0.0, "ceeval_full": 0.4, "ceeval_pos": 0.0}
    originals = load_pilot_examples(load_pilot_manifest(manifest_path))
    assert [r["story_id"] for r in simple["results"]] == [e.example_id for e in originals]
    assert all(list(r) == ["story_id", "story", "status", "prediction"] for r in simple["results"])
    assert [r["story"] for r in simple["results"]] == [e.raw_record["story"] for e in originals]
    assert simple["results"][0]["status"] == "timeout" and simple["results"][0]["prediction"] is None
    assert all(r["status"] == "ok" for r in simple["results"][1:])
    assert all(r["prediction"]["has_error"] is False and r["prediction"]["error_sentences"] == []
               and r["prediction"]["contradicted_sentences"] == [] for r in simple["results"][1:])
    lines = (first.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    (first.run_dir / "predictions.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ExperimentRunError, match="incomplete"):
        baseline_pilot.export_baseline("llama3.1:8b")


def test_run_blocks_changed_source_before_inference(tmp_path, monkeypatch) -> None:
    settings = baseline_pilot.load_settings()
    settings["manifest"] = str(tmp_path / "manifest.json")
    settings["preparation_dir"] = str(tmp_path / "prep")
    manifest = baseline_pilot.build_manifest(settings)
    Path(settings["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    Path(settings["preparation_dir"]).mkdir()
    (Path(settings["preparation_dir"]) / "metadata.json").write_text(json.dumps({
        "status": "PREPARED_NO_INFERENCE", "provenance": {"source_hashes": {}},
    }), encoding="utf-8")
    monkeypatch.setattr(baseline_pilot, "load_settings", lambda: settings)
    monkeypatch.setattr(baseline_pilot, "run_experiment", lambda *_: pytest.fail("Must block before inference"))
    with pytest.raises(ExperimentRunError, match="Source/configuration changed"):
        baseline_pilot.run_pilot("qwen3:8b")


@pytest.mark.parametrize("model", ["qwen3:8b", "llama3.1:8b", "mistral:7b"])
def test_guarded_entry_point_runs_once_resumes_and_rejects_identity_drift(tmp_path, monkeypatch, model) -> None:
    settings = baseline_pilot.load_settings()
    settings.update(manifest=str(tmp_path / "manifest.json"),
                    preparation_dir=str(tmp_path / "preparation"),
                    run_root=str(tmp_path / "runs"), export_dir=str(tmp_path / "exports"))
    monkeypatch.setattr(baseline_pilot, "load_settings", lambda: settings)
    inspection = {"ollama_version": "mock", "models": {
        name: {"available": True, "identity": {"digest": name + "-mock"}}
        for name in settings["models"]
    }}
    monkeypatch.setattr(baseline_pilot, "inspect_models", lambda *_: inspection)
    monkeypatch.setattr(baseline_pilot, "_local_resources", lambda: {})
    monkeypatch.setattr("fyp.experiments.runner._ollama_version", lambda _: "mock")
    calls = []
    class Client:
        def __init__(self, config):
            self.config = config
        def generate(self, request):
            calls.append(request.prompt)
            return ModelResponse(
                model=self.config.model,
                response="<response><decision>No continuity error found</decision></response>",
                thinking=None, raw={"done": True},
                effective_config=self.config.effective_ollama_request_fields(),
            )
    monkeypatch.setattr("fyp.experiments.runner.build_model_client", Client)
    baseline_pilot.prepare_pilot()
    first = baseline_pilot.run_pilot(model)
    assert first.expected_examples == 10
    assert first.completed_examples == 10
    saved = (first.run_dir / "predictions.jsonl").read_bytes()
    resumed = baseline_pilot.run_pilot(model)
    assert resumed.skipped_completed == 10
    assert len(calls) == 10
    assert saved == (first.run_dir / "predictions.jsonl").read_bytes()
    records = [json.loads(line) for line in saved.splitlines()]
    assert all(record["subset"] == "flawed_fictions" for record in records)
    run_metadata = json.loads((first.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert run_metadata["pilot_provenance"]["models"][model]["identity"]["digest"] == model + "-mock"
    inspection["models"][model]["identity"]["digest"] = "changed"
    with pytest.raises(ExperimentRunError, match="model identities"):
        baseline_pilot.run_pilot(model)
    assert len(calls) == 10


def test_simple_export_preserves_positive_evidence_explanation_and_manifest_order(tmp_path, monkeypatch) -> None:
    settings = baseline_pilot.load_settings()
    settings.update(manifest=str(tmp_path / "manifest.json"),
                    run_root=str(tmp_path / "runs"), export_dir=str(tmp_path / "exports"))
    manifest = baseline_pilot.build_manifest(settings)
    Path(settings["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    run_dir = Path(settings["run_root"]) / settings["models"]["qwen3:8b"]["run_id"]
    run_dir.mkdir(parents=True)
    records = [{
        "example_id": row["example_id"], "subset": "flawed_fictions", "model": "qwen3:8b",
        "benchmark_strategy": "vanilla", "input_hash": row["story_hash"],
        "prompt_hash": row["prompt_hash"], "prompt_template_hash": settings["prompt_template_hash"],
        "target_inference_failure": False,
        "parsed_prediction": {"cont_error": 1.0, "cont_error_lines": ["synthetic error quote"],
                              "contradicted_lines": ["synthetic earlier quote"],
                              "cont_error_expl": "synthetic explanation", "parsing_failures": ["synthetic warning"]},
    } for row in manifest["records"]]
    (run_dir / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in reversed(records)) + "\n", encoding="utf-8")
    for name in ("metadata.json", "metrics.json", "evaluation_examples.jsonl"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(baseline_pilot, "load_settings", lambda: settings)
    output = baseline_pilot.export_baseline("qwen3:8b")
    simple = json.loads(output.read_text(encoding="utf-8"))
    assert [r["story_id"] for r in simple["results"]] == [r["example_id"] for r in manifest["records"]]
    assert simple["failed"] == 0
    assert simple["metrics"]["accuracy"] == 0.5
    assert simple["metrics"]["precision"] == 0.5 and simple["metrics"]["recall"] == 1.0
    assert all(r["status"] == "ok" and r["prediction"] == {
        "has_error": True, "error_sentences": ["synthetic error quote"],
        "contradicted_sentences": ["synthetic earlier quote"], "explanation": "synthetic explanation",
    } for r in simple["results"])
