"""Read-only completion audit; never invokes inference or rewrites predictions."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from fyp.data.pilot import stable_file_hash
from fyp.evaluation.evaluator import _extract_prediction, evaluate_predictions
from fyp.experiments.prompting_pilot import (
    _run_id, _source_hashes, load_settings, validate_baseline, validate_scope,
)
from fyp.experiments.runner import RunConfig, _fingerprint, _parsed_to_record, _research_config, _to_simple_yaml
from fyp.models.registry import load_model_config
from fyp.prompting.parser import parse_official_response
from fyp.prompting.strategies import get_benchmark_strategy


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    settings, base = load_settings()
    manifest, examples = validate_scope(settings, base)
    assert len(examples) == 10
    assert Counter(e.cont_error for e in examples) == {0.0: 5, 1.0: 5}
    prepared = read(Path(settings["preparation_dir"]) / "metadata.json")
    protected = read(Path(settings["run_root"]).parent / "verify_revise_protected_artifacts.json")
    assert all(stable_file_hash(Path(row["path"])) == row["hash"] for row in protected)
    execution = read(Path(settings["run_root"]).parent / "verify_revise_execution_freeze.json")
    assert stable_file_hash(Path(execution["script"])) == execution["script_hash"]
    provenance = {**prepared["provenance"], "execution_adapter": execution}
    assert provenance["source_hashes"] == _source_hashes(settings, base)
    strategy = get_benchmark_strategy("verify_revise", version="verify_revise_v2")
    template = strategy.template_path.read_text(encoding="utf-8")
    template_hash = stable_file_hash(strategy.template_path)
    ids = [e.example_id for e in examples]
    audited = []
    for model in settings["models"]:
        assert validate_baseline(model, settings, base, manifest, examples) == provenance["baseline_references"][model]
        directory = Path(settings["run_root"]) / _run_id(settings, model, "verify_revise")
        records = rows(directory / "predictions.jsonl")
        assert len(records) == 10 and [r["example_id"] for r in records] == ids
        assert len(set(ids)) == 10
        metadata = read(directory / "metadata.json")
        assert metadata["end_timestamp"] and metadata["expected_example_count"] == 10
        assert metadata["pilot_provenance"] == provenance
        config = load_model_config(base["models"][model]["config"])
        research = _research_config(RunConfig(
            model=model, strategy="verify_revise", prompt_version="verify_revise_v2",
            subsets=manifest.subsets, experiment_type="exploratory_prompting_pilot",
            manifest_path=manifest.path, model_config_path=Path(base["models"][model]["config"]),
            run_id=metadata["run_id"], run_root=Path(settings["run_root"]), provenance=provenance,
        ), config, metadata["run_id"], manifest=manifest)
        assert metadata["configuration_fingerprint"] == _fingerprint(research)
        assert (directory / "config.yaml").read_text(encoding="utf-8") == _to_simple_yaml(research)
        request_fields = config.effective_ollama_request_fields()
        events = rows(directory / "requests.jsonl")
        starts = {e["request_id"]: e for e in events if e["event"] == "started"}
        finishes = {e["request_id"]: e for e in events if e["event"] == "finished"}
        assert len(starts) + len(finishes) == len(events)
        assert set(starts) == set(finishes)
        for record, example in zip(records, examples):
            prompt = strategy.build_prompt(example)
            assert record["model"] == model and record["benchmark_strategy"] == "verify_revise"
            assert record["subset"] == "flawed_fictions" and record["prompt_version"] == "verify_revise_v2"
            assert record["prompt_template_hash"] == template_hash and record["prompt_hash"] == prompt.prompt_hash
            assert record["input_hash"] == provenance["story_hashes"][example.example_id]
            assert record["model_configuration"]["generation"] == config.generation.snapshot()
            assert record["model_configuration"]["runtime"] == settings["runtime"]
            history = record["attempt_history"]
            assert 1 <= len(history) <= config.runtime.max_retries + 1
            assert [a["attempt"] for a in history] == list(range(1, len(history) + 1))
            assert len([e for e in starts.values() if e["cache_key"] == record["cache_key"]]) == len(history)
            for attempt in history:
                start, finish = starts[attempt["request_id"]], finishes[attempt["request_id"]]
                assert start["example_id"] == example.example_id and start["model"] == model
                assert start["run_id"] == metadata["run_id"] and start["benchmark_strategy"] == "verify_revise"
                assert start["prompt_template"] == template and start["prompt_template_hash"] == template_hash
                assert start["model_digest"] == provenance["models"][model]["identity"]["digest"]
                request = start["application_request"]
                assert request["method"] == "POST" and request["url"].endswith("/api/generate")
                assert json.loads(request["body_utf8"]) == request["body"]
                assert hashlib.sha256(request["body_utf8"].encode("utf-8")).hexdigest().upper() == request["body_hash"]
                assert request["body"] == {**request_fields, "prompt": prompt.prompt}
                assert (request["body"].get("think") is False) if model == "qwen3:8b" else ("think" not in request["body"])
                assert finish["status"] == attempt["status"] and finish["latency_seconds"] == attempt["latency_seconds"]
            if not record["target_inference_failure"]:
                assert record["effective_config"] == request_fields
                assert record["parsed_prediction"] == _parsed_to_record(parse_official_response(record["raw_response"]))
                last = finishes[history[-1]["request_id"]]["raw_response"]
                assert last["response"] == record["raw_response"]
                assert last.get("thinking") == record["thinking"]
        report = evaluate_predictions(records, dataset={manifest.subsets[0]: examples})
        assert len(report.examples) == 10 and sum(e.ground_truth_cont_error == 1 for e in report.examples) == 5
        assert read(directory / "metrics.json") == report.metrics
        output_path = directory.parent.parent / "exports" / (settings["models"][model].replace("-", "_") + "_verify_revise.json")
        output = read(output_path)
        assert list(output) == ["model", "prompt", "total", "failed", "metrics", "results"]
        assert output["model"] == model and output["prompt"] == "verify_revise" and output["total"] == 10
        failures = report.metrics["failure_counts"]["target_inference_failure"]
        assert output["failed"] == failures == metadata["failed_example_count"]
        assert metadata["completed_example_count"] + failures == 10
        metrics = {**report.metrics["classification_metrics"], **{k: report.metrics["localization_metrics"][k] for k in ("ceeval_full", "ceeval_pos")}}
        assert output["metrics"] == metrics
        assert set(metrics) == {"accuracy", "precision", "recall", "f1", "ceeval_full", "ceeval_pos"}
        assert all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values())
        assert len(output["results"]) == 10
        for result, example, record in zip(output["results"], examples, records):
            assert list(result) == ["story_id", "story", "status", "prediction"]
            assert type(result["story_id"]) is type(example.raw_record["example_id"])
            assert result["story_id"] == example.raw_record["example_id"] and result["story"] == example.raw_record["story"]
            parsed, flags = _extract_prediction(record)
            if flags["target_inference_failure"]:
                assert result["prediction"] is None
                assert result["status"] == ("timeout" if "timeout" in record["inference_error"]["type"].lower() else "inference_error")
            else:
                positive = bool(parsed.cont_error)
                assert result["status"] == "ok"
                assert list(result["prediction"]) == ["has_error", "error_sentences", "contradicted_sentences", "explanation"]
                assert result["prediction"] == {"has_error": positive, "error_sentences": list(parsed.cont_error_lines) if positive else [], "contradicted_sentences": list(parsed.contradicted_lines) if positive else [], "explanation": parsed.cont_error_expl}
        audited.append({"model": model, "completed": 10, "terminal_failed": failures, "attempts": len(starts), "parser_warning_records": sum(bool(r["parsed_prediction"] and r["parsed_prediction"]["parsing_failures"]) for r in records), "output": str(output_path.resolve())})
    progress = read(Path(settings["run_root"]).parent / "verify_revise_progress.json")
    assert progress["status"] == "COMPLETE" and progress["completed"] == 30 and progress["pending"] == 0
    assert all(not row["historical_request_logging_gaps"] for row in progress["models"])
    assert {p.parent.name for p in Path(settings["run_root"]).glob("*/metadata.json")} == {
        _run_id(settings, model, strategy_name) for model in settings["models"]
        for strategy_name in ("evidence_first", "structured_check", "verify_revise")
    }
    matrix = []
    for model in settings["models"]:
        matrix_config = load_model_config(base["models"][model]["config"])
        for method in ("vanilla", "evidence_first", "structured_check", "verify_revise"):
            matrix_strategy = get_benchmark_strategy(method, version=settings["strategies"][method]["version"])
            if method == "vanilla":
                entry = base["models"][model]
                directory = Path(base["run_root"]) / entry["run_id"]
                output_path = Path(base["export_dir"]) / (entry["export_stem"] + ".json")
            else:
                directory = Path(settings["run_root"]) / _run_id(settings, model, method)
                output_path = directory.parent.parent / "exports" / (settings["models"][model].replace("-", "_") + "_" + method + ".json")
            records = rows(directory / "predictions.jsonl")
            output = read(output_path)
            assert len(records) == 10 and [r["example_id"] for r in records] == ids
            assert list(output) == ["model", "prompt", "total", "failed", "metrics", "results"]
            assert output["model"] == model and output["total"] == 10
            assert output["prompt"] == ("baseline" if method == "vanilla" else method)
            assert len(output["results"]) == 10
            raw_records = [{k: v for k, v in r.items() if k != "parsed_prediction"} for r in records]
            report = evaluate_predictions(raw_records, dataset={manifest.subsets[0]: examples})
            assert len(report.examples) == 10 and sum(e.ground_truth_cont_error == 1 for e in report.examples) == 5
            metrics = {**report.metrics["classification_metrics"], **{k: report.metrics["localization_metrics"][k] for k in ("ceeval_full", "ceeval_pos")}}
            assert output["metrics"] == metrics and read(directory / "metrics.json") == report.metrics
            assert all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values())
            failures = report.metrics["failure_counts"]["target_inference_failure"]
            assert output["failed"] == failures
            for result, example, record in zip(output["results"], examples, records):
                assert record["model"] == model and record["benchmark_strategy"] == method and record["subset"] == "flawed_fictions"
                assert record["model_configuration"]["generation"] == matrix_config.generation.snapshot()
                assert record["model_configuration"]["runtime"] == settings["runtime"]
                assert record["prompt_hash"] == matrix_strategy.build_prompt(example).prompt_hash
                assert record["prompt_template_hash"] == settings["strategies"][method]["hash"]
                assert record["input_hash"] == provenance["story_hashes"][example.example_id]
                assert list(result) == ["story_id", "story", "status", "prediction"]
                assert result["story_id"] == example.raw_record["example_id"] and type(result["story_id"]) is type(example.raw_record["example_id"])
                assert result["story"] == example.raw_record["story"]
                parsed, flags = _extract_prediction(record)
                if flags["target_inference_failure"]:
                    assert result["prediction"] is None
                    assert result["status"] == ("timeout" if "timeout" in record["inference_error"]["type"].lower() else "inference_error")
                else:
                    assert record["effective_config"] == matrix_config.effective_ollama_request_fields()
                    assert record["parsed_prediction"] == _parsed_to_record(parse_official_response(record["raw_response"]))
                    positive = bool(parsed.cont_error)
                    assert result["status"] == "ok"
                    assert list(result["prediction"]) == ["has_error", "error_sentences", "contradicted_sentences", "explanation"]
                    assert type(result["prediction"]["has_error"]) is bool
                    assert result["prediction"] == {"has_error": positive, "error_sentences": list(parsed.cont_error_lines) if positive else [], "contradicted_sentences": list(parsed.contradicted_lines) if positive else [], "explanation": parsed.cont_error_expl}
            matrix.append({"model": model, "strategy": method, "status": "COMPLETE_VERIFIED", "records": 10,
                           "terminal_failed": failures, "output": str(output_path.resolve())})
    print(json.dumps({"status": "PASS", "protected_artifacts_unchanged": len(protected), "models": audited,
                      "matrix_available": True, "matrix_records": sum(r["records"] for r in matrix), "matrix": matrix}, indent=2))


if __name__ == "__main__":
    main()
