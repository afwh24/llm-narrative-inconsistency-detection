import importlib.util
import json
from pathlib import Path

import pytest

from fyp.experiments.prompting_outputs import export_prompting
from fyp.experiments.prompting_pilot import _run_id, load_settings, validate_scope
from fyp.experiments.runner import ExperimentRunError


spec = importlib.util.spec_from_file_location("structured_execution", Path("scripts/run_structured_check_main10.py"))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.mark.parametrize("model,stem", [("qwen3:8b", "qwen3_8b"), ("llama3.1:8b", "llama3_1_8b"), ("mistral:7b", "mistral_7b")])
def test_structured_progress_and_export_do_not_touch_other_conditions(tmp_path, monkeypatch, model, stem):
    settings, base = load_settings()
    manifest, examples = validate_scope(settings, base)
    settings["run_root"] = str(tmp_path / "runs")
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    protected = tmp_path / "evidence_first_progress.json"
    protected.write_text('{"sentinel": true}', encoding="utf-8")
    directory = Path(settings["run_root"]) / _run_id(settings, model, "structured_check")
    directory.mkdir(parents=True)
    rows = [{"model": model, "benchmark_strategy": "structured_check", "subset": "flawed_fictions",
             "example_id": row["example_id"], "input_hash": row["story_hash"],
             "target_inference_failure": index == 0,
             "inference_error": {"type": "OllamaTimeoutError", "message": "synthetic only"} if index == 0 else None,
             "parsed_prediction": None if index == 0 else {"cont_error": 0.0, "cont_error_lines": ["synthetic"],
                 "contradicted_lines": ["synthetic"], "cont_error_expl": "synthetic", "parsing_failures": ["synthetic"]}}
            for index, row in enumerate(manifest.raw["records"])]
    checkpoint = directory / "predictions.jsonl"
    checkpoint.write_text("\n".join(json.dumps(row) for row in rows[:3]), encoding="utf-8")
    progress_path = adapter.update_progress(settings)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    state = next(row for row in progress["models"] if row["model"] == model)
    assert state["completed"] == 3 and state["pending"] == 7 and state["terminal_failed"] == 1
    assert state["status"] == "PENDING" and state["output_path"] is None
    assert state["resume_command"].endswith(f"run --model {model}")
    with pytest.raises(ExperimentRunError, match="Cannot export pending"):
        export_prompting(model, "structured_check", settings, directory)
    checkpoint.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = export_prompting(model, "structured_check", settings, directory)
    assert output.name == stem + "_structured_check.json"
    exported = json.loads(output.read_text(encoding="utf-8"))
    assert exported["total"] == 10 and exported["failed"] == 1 and exported["prompt"] == "structured_check"
    assert [row["story"] for row in exported["results"]] == [e.raw_record["story"] for e in examples]
    assert exported["results"][0]["prediction"] is None and exported["results"][0]["status"] == "timeout"
    assert all(row["prediction"]["error_sentences"] == [] for row in exported["results"][1:])
    adapter.update_progress(settings)
    state = next(row for row in json.loads(progress_path.read_text(encoding="utf-8"))["models"] if row["model"] == model)
    assert state["status"] == "COMPLETE" and state["pending"] == 0
    assert len(state["historical_request_logging_gaps"]) == 10
    assert protected.read_text(encoding="utf-8") == '{"sentinel": true}'
