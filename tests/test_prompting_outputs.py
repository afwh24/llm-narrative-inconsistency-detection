from __future__ import annotations

import json
from pathlib import Path

import pytest

from fyp.experiments.prompting_pilot import load_settings, validate_scope, _run_id
from fyp.experiments.prompting_outputs import export_prompting, update_evidence_progress
from fyp.experiments.runner import ExperimentRunError


@pytest.mark.parametrize("model,filename", [
    ("qwen3:8b", "qwen3_8b_evidence_first.json"),
    ("llama3.1:8b", "llama3_1_8b_evidence_first.json"),
    ("mistral:7b", "mistral_7b_evidence_first.json"),
])
def test_export_and_progress_are_complete_only_for_ten_records(tmp_path, model, filename):
    settings, base = load_settings()
    manifest, examples = validate_scope(settings, base)
    settings["run_root"] = str(tmp_path / "runs")
    run_dir = Path(settings["run_root"]) / _run_id(settings, model, "evidence_first")
    run_dir.mkdir(parents=True)
    rows = [{
        "model": model, "benchmark_strategy": "evidence_first", "subset": "flawed_fictions",
        "example_id": item["example_id"], "input_hash": item["story_hash"],
        "target_inference_failure": index == 0,
        "inference_error": {"type": "OllamaTimeoutError", "message": "synthetic only"} if index == 0 else None,
        "parsed_prediction": None if index == 0 else {
            "cont_error": 0.0, "cont_error_lines": ["placeholder"], "contradicted_lines": ["placeholder"],
            "cont_error_expl": "synthetic explanation", "parsing_failures": ["synthetic warning"],
        },
    } for index, item in enumerate(manifest.raw["records"])]
    path = run_dir / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows[:5]) + "\n", encoding="utf-8")
    progress_path = update_evidence_progress(settings)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    actual = next(row for row in progress["models"] if row["model"] == model)
    assert actual["completed"] == 5 and actual["pending"] == 5 and actual["terminal_failed"] == 1
    assert actual["status"] == "PENDING" and actual["output_path"] is None
    with pytest.raises(ExperimentRunError, match="Cannot export pending"):
        export_prompting(model, "evidence_first", settings, run_dir)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    exported = export_prompting(model, "evidence_first", settings, run_dir)
    assert exported.name == filename
    output = json.loads(exported.read_text(encoding="utf-8"))
    assert list(output) == ["model", "prompt", "total", "failed", "metrics", "results"]
    assert output["prompt"] == "evidence_first" and output["total"] == 10 and output["failed"] == 1
    assert output["metrics"]["accuracy"] == 0.4 and output["metrics"]["precision"] == 0.0
    assert [row["story"] for row in output["results"]] == [example.raw_record["story"] for example in examples]
    assert [row["story_id"] for row in output["results"]] == [example.example_id for example in examples]
    assert all(list(row) == ["story_id", "story", "status", "prediction"] for row in output["results"])
    assert output["results"][0]["status"] == "timeout" and output["results"][0]["prediction"] is None
    assert all(row["status"] == "ok" and row["prediction"]["error_sentences"] == [] and row["prediction"]["contradicted_sentences"] == [] for row in output["results"][1:])
    update_evidence_progress(settings)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    actual = next(row for row in progress["models"] if row["model"] == model)
    assert actual["status"] == "COMPLETE" and actual["pending"] == 0
    assert len(actual["historical_request_logging_gaps"]) == 10
