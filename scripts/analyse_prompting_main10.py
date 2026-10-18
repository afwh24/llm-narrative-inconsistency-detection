"""Offline audit and paired descriptive analysis; never invokes inference."""

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import statistics

from fyp.data.pilot import stable_file_hash
from fyp.evaluation.evaluator import _extract_prediction, evaluate_predictions
from fyp.evaluation.localization import extract_all_full_lines, listify_lines, preprocess_sent, sentence_tokenize
from fyp.experiments.prompting_pilot import _run_id, _source_hashes, load_settings, validate_baseline, validate_scope
from fyp.experiments.prompting_outputs import _read_rows, _write_json
from fyp.experiments.runner import RunConfig, _fingerprint, _parsed_to_record, _research_config, _to_simple_yaml
from fyp.models.registry import load_model_config
from fyp.prompting.parser import parse_official_response
from fyp.prompting.strategies import get_benchmark_strategy


ROOT = Path("experiments/three_model_main10_prompting")
METHODS = ("vanilla", "evidence_first", "structured_check", "verify_revise")
METRICS = ("accuracy", "precision", "recall", "f1", "ceeval_full", "ceeval_pos")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def paired(before, after):
    if len(before) != len(after):
        raise ValueError("Paired outcomes must use identical denominators")
    return {"improved": sum(b > a for a, b in zip(before, after)),
            "regressed": sum(b < a for a, b in zip(before, after)),
            "unchanged": sum(b == a for a, b in zip(before, after)),
            "unchanged_correct": sum(a == b == 1 for a, b in zip(before, after)),
            "unchanged_incorrect": sum(a == b == 0 for a, b in zip(before, after))}


def empty_fragments(lines):
    return [fragment for line in lines for fragment in sentence_tokenize(line) if not preprocess_sent(fragment)]


def audit_condition(model, method, settings, base, manifest, examples, preparation):
    config = load_model_config(base["models"][model]["config"])
    strategy = get_benchmark_strategy(method, version=settings["strategies"][method]["version"])
    if method == "vanilla":
        validate_baseline(model, settings, base, manifest, examples)
        entry = base["models"][model]
        directory = Path(base["run_root"]) / entry["run_id"]
        output_path = Path(base["export_dir"]) / (entry["export_stem"] + ".json")
    else:
        directory = Path(settings["run_root"]) / _run_id(settings, model, method)
        output_path = ROOT / "exports" / (settings["models"][model].replace("-", "_") + "_" + method + ".json")
    records = _read_rows(directory / "predictions.jsonl")
    metadata = read(directory / "metadata.json")
    output = read(output_path)
    ids = [e.example_id for e in examples]
    assert len(records) == 10 and len(set(ids)) == 10 and [r["example_id"] for r in records] == ids
    assert metadata["end_timestamp"] and metadata["expected_example_count"] == 10
    assert metadata["thinking_mode"] == config.generation.think
    assert metadata["prompt_version"] == settings["strategies"][method]["version"]
    assert metadata["prompt_template_hash"] == settings["strategies"][method]["hash"]
    assert metadata["dependencies"] == {key: importlib.metadata.version(key) for key in metadata["dependencies"]}
    provenance = metadata["pilot_provenance"]
    assert provenance["models"][model]["identity"] == preparation["provenance"]["models"][model]["identity"]
    assert metadata["ollama_version"] == preparation["provenance"]["ollama_version"]
    if method != "vanilla":
        expected_provenance = dict(preparation["provenance"])
        if method in ("structured_check", "verify_revise"):
            execution = read(ROOT / (method + "_execution_freeze.json"))
            assert stable_file_hash(Path(execution["script"])) == execution["script_hash"]
            expected_provenance["execution_adapter"] = execution
        assert provenance == expected_provenance
        provenance = expected_provenance
    else:
        expected_provenance = read(Path(base["preparation_dir"]) / "metadata.json")["provenance"]
        assert provenance == expected_provenance
        provenance = expected_provenance
    research = _research_config(RunConfig(
        model=model, strategy=method, prompt_version=None if method == "vanilla" else metadata["prompt_version"],
        subsets=manifest.subsets, experiment_type=metadata["experiment_type"], run_id=metadata["run_id"],
        run_root=directory.parent, manifest_path=manifest.path, model_config_path=Path(base["models"][model]["config"]),
        provenance=provenance,
    ), config, metadata["run_id"], manifest=manifest)
    assert metadata["configuration_fingerprint"] == _fingerprint(research)
    assert (directory / "config.yaml").read_text(encoding="utf-8") == _to_simple_yaml(research)
    events = _read_rows(directory / "requests.jsonl")
    starts = {e["request_id"]: e for e in events if e["event"] == "started"}
    finishes = {e["request_id"]: e for e in events if e["event"] == "finished"}
    if method != "vanilla":
        assert len(events) == len(starts) + len(finishes) and set(starts) == set(finishes)
    for record, example in zip(records, examples):
        prompt = strategy.build_prompt(example)
        assert record["model"] == model and record["benchmark_strategy"] == method and record["subset"] == "flawed_fictions"
        assert record["input_hash"] == provenance["story_hashes"][example.example_id]
        assert record["prompt_hash"] == prompt.prompt_hash and record["prompt_template_hash"] == settings["strategies"][method]["hash"]
        assert record["model_configuration"]["generation"] == config.generation.snapshot()
        assert record["model_configuration"]["runtime"] == settings["runtime"]
        history = record["attempt_history"]
        assert 1 <= len(history) <= config.runtime.max_retries + 1
        assert [a["attempt"] for a in history] == list(range(1, len(history) + 1))
        if not record["target_inference_failure"]:
            assert record["effective_config"] == config.effective_ollama_request_fields()
            assert record["parsed_prediction"] == _parsed_to_record(parse_official_response(record["raw_response"]))
        if method != "vanilla":
            assert len([e for e in starts.values() if e["cache_key"] == record["cache_key"]]) == len(history)
            for attempt in history:
                start, finish = starts[attempt["request_id"]], finishes[attempt["request_id"]]
                assert start["model"] == model and start["run_id"] == metadata["run_id"] and start["example_id"] == example.example_id
                assert start["benchmark_strategy"] == method and start["prompt_version"] == metadata["prompt_version"]
                assert start["prompt_template"] == strategy.template_path.read_text(encoding="utf-8")
                assert start["prompt_template_hash"] == settings["strategies"][method]["hash"]
                assert start["model_digest"] == provenance["models"][model]["identity"]["digest"]
                application = start["application_request"]
                assert application["method"] == "POST" and application["url"].endswith("/api/generate")
                assert application["body"] == {**config.effective_ollama_request_fields(), "prompt": prompt.prompt}
                assert json.loads(application["body_utf8"]) == application["body"]
                assert hashlib.sha256(application["body_utf8"].encode("utf-8")).hexdigest().upper() == application["body_hash"]
                assert finish["status"] == attempt["status"] and finish["latency_seconds"] == attempt["latency_seconds"]
            if not record["target_inference_failure"]:
                response = finishes[history[-1]["request_id"]]["raw_response"]
                assert response["response"] == record["raw_response"] and response.get("thinking") == record["thinking"]
    raw_records = [{k: v for k, v in row.items() if k != "parsed_prediction"} for row in records]
    report = evaluate_predictions(raw_records, dataset={manifest.subsets[0]: examples})
    assert len(report.examples) == 10 and sum(e.ground_truth_cont_error == 1 for e in report.examples) == 5
    assert read(directory / "metrics.json") == report.metrics
    metrics = {**report.metrics["classification_metrics"], **{k: report.metrics["localization_metrics"][k] for k in ("ceeval_full", "ceeval_pos")}}
    assert output["metrics"] == metrics and output["total"] == 10 and output["failed"] == report.metrics["failure_counts"]["target_inference_failure"]
    assert set(metrics) == set(METRICS)
    assert all(type(value) in (int, float) and math.isfinite(value) for value in metrics.values())
    assert metadata["failed_example_count"] == output["failed"] and metadata["completed_example_count"] + output["failed"] == 10
    assert list(output) == ["model", "prompt", "total", "failed", "metrics", "results"]
    assert output["model"] == model and output["prompt"] == ("baseline" if method == "vanilla" else method)
    assert len(output["results"]) == 10
    for result, record, example in zip(output["results"], records, examples):
        assert list(result) == ["story_id", "story", "status", "prediction"]
        assert result["story_id"] == example.raw_record["example_id"] and type(result["story_id"]) is type(example.raw_record["example_id"])
        assert result["story"] == example.raw_record["story"]
        prediction, flags = _extract_prediction(record)
        if flags["target_inference_failure"]:
            assert result["prediction"] is None and result["status"] == ("timeout" if "timeout" in record["inference_error"]["type"].lower() else "inference_error")
        else:
            positive = bool(prediction.cont_error)
            assert result["status"] == "ok" and type(result["prediction"]["has_error"]) is bool
            assert list(result["prediction"]) == ["has_error", "error_sentences", "contradicted_sentences", "explanation"]
            assert result["prediction"] == {"has_error": positive, "error_sentences": list(prediction.cont_error_lines) if positive else [], "contradicted_sentences": list(prediction.contradicted_lines) if positive else [], "explanation": prediction.cont_error_expl}
    facts = []
    warnings = Counter()
    for record, scored in zip(records, report.examples):
        parsed = parse_official_response(record["raw_response"]) if not record["target_inference_failure"] else None
        warnings.update(parsed.parsing_failures if parsed else ())
        facts.append({"story_id": scored.example_id, "gold": scored.ground_truth_cont_error,
                      "predicted": scored.predicted_cont_error, "classification_correct": scored.classification_correct,
                      "ceeval_full": scored.ceeval_full, "error_side_correct": scored.cont_error_lines_correct,
                      "contradicted_side_correct": scored.contradicted_lines_correct,
                      "warnings": list(parsed.parsing_failures) if parsed else [], "decision_text": parsed.cont_error_expl_short if parsed else None,
                      "empty_normalised_error_fragments": empty_fragments(parsed.cont_error_lines) if parsed else [],
                      "empty_normalised_contradicted_fragments": empty_fragments(parsed.contradicted_lines) if parsed else [],
                      "explanation": parsed.cont_error_expl if parsed else None,
                      "error_sentences": list(parsed.cont_error_lines) if parsed else [],
                      "contradicted_sentences": list(parsed.contradicted_lines) if parsed else []})
    confusion = {"tp": sum(r["gold"] == 1 and r["predicted"] == 1 for r in facts),
                 "fn": sum(r["gold"] == 1 and r["predicted"] != 1 for r in facts),
                 "tn": sum(r["gold"] == 0 and r["predicted"] == 0 for r in facts),
                 "fp": sum(r["gold"] == 0 and r["predicted"] == 1 for r in facts)}
    times = [r["latency_seconds"] for r in records]
    durations = [r["ollama_raw"]["total_duration"] / 1e9 for r in records if (r.get("ollama_raw") or {}).get("total_duration") is not None]
    counts = [r["ollama_raw"]["eval_count"] for r in records if (r.get("ollama_raw") or {}).get("eval_count") is not None]
    done_reasons = Counter((r.get("ollama_raw") or {}).get("done_reason", "not_recorded") for r in records)
    load_seconds = sum((r.get("ollama_raw") or {}).get("load_duration", 0) / 1e9 for r in records)
    return {"model": model, "strategy": method, "status": "COMPLETE_VERIFIED", "total": 10, "failed": output["failed"],
            "metrics": metrics, "confusion": confusion, "positive_localised": sum(r["ceeval_full"] for r in facts if r["gold"] == 1),
            "negative_correct": confusion["tn"], "warnings": dict(warnings), "warning_records": sum(bool(r["warnings"]) for r in facts),
            "attempts": sum(len(r["attempt_history"]) for r in records), "request_journal": bool(events),
            "runtime": {"sum_seconds": sum(times), "mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times),
                        "min_seconds": min(times), "max_seconds": max(times), "ollama_sum_seconds": sum(durations),
                        "mean_output_tokens": statistics.mean(counts) if counts else None, "load_seconds": load_seconds,
                        "done_reasons": dict(done_reasons)},
            "run_id": metadata["run_id"], "output": str(output_path.resolve()), "run_directory": str(directory.resolve()), "stories": facts}


def main():
    settings, base = load_settings()
    manifest, examples = validate_scope(settings, base)
    assert len(examples) == 10 and Counter(e.cont_error for e in examples) == {0.0: 5, 1.0: 5}
    prepared = read(Path(settings["preparation_dir"]) / "metadata.json")
    assert prepared["provenance"]["source_hashes"] == _source_hashes(settings, base)
    protected = read(ROOT / "verify_revise_protected_artifacts.json")
    assert all(stable_file_hash(Path(row["path"])) == row["hash"] for row in protected)
    conditions = [audit_condition(model, method, settings, base, manifest, examples, prepared)
                  for model in settings["models"] for method in METHODS]
    comparisons = []
    for model in settings["models"]:
        baseline = next(c for c in conditions if c["model"] == model and c["strategy"] == "vanilla")
        for condition in (c for c in conditions if c["model"] == model and c["strategy"] != "vanilla"):
            before, after = baseline["stories"], condition["stories"]
            comparisons.append({"model": model, "strategy": condition["strategy"],
                "delta_percentage_points": {k: (condition["metrics"][k] - baseline["metrics"][k]) * 100 for k in METRICS},
                "classification": paired([r["classification_correct"] for r in before], [r["classification_correct"] for r in after]),
                "localisation_full": paired([r["ceeval_full"] for r in before], [r["ceeval_full"] for r in after]),
                "localisation_positive": paired([r["ceeval_full"] for r in before if r["gold"] == 1], [r["ceeval_full"] for r in after if r["gold"] == 1]),
                "changed_story_ids": {field: {"improved": [a["story_id"] for a, b in zip(before, after) if b[field] > a[field]],
                                             "regressed": [a["story_id"] for a, b in zip(before, after) if b[field] < a[field]]}
                                      for field in ("classification_correct", "ceeval_full")}})
    tokenizer = "nltk.sent_tokenize"
    try:
        from nltk import sent_tokenize
        sent_tokenize("First sentence. Second sentence.")
    except Exception as error:
        tokenizer = "deterministic regex fallback: " + type(error).__name__
    gold = [{"story_id": e.example_id, "has_error": e.has_continuity_error, "story": e.story,
             "gold_explanation": e.cont_error_expl,
             "gold_error_sentences": extract_all_full_lines(e.story, listify_lines(e.cont_error_lines_raw)) if e.has_continuity_error else [],
             "gold_contradicted_sentences": extract_all_full_lines(e.story, listify_lines(e.contradicted_lines_raw)) if e.has_continuity_error else [],
             "empty_normalised_gold_fragments": empty_fragments(listify_lines(e.cont_error_lines_raw) + listify_lines(e.contradicted_lines_raw)) if e.has_continuity_error else []} for e in examples]
    result = {"audit": "PASS", "records": sum(c["total"] for c in conditions), "protected_artifacts_unchanged": len(protected),
              "manifest_hash": manifest.manifest_hash, "tokenizer": tokenizer, "conditions": conditions, "comparisons": comparisons, "gold_analysis_only": gold}
    paths = [Path(c["output"]) for c in conditions] + [p for c in conditions for p in Path(c["run_directory"]).iterdir() if p.is_file()]
    result["audited_artifact_hashes"] = {str(p): stable_file_hash(p) for p in paths}
    _write_json(ROOT / "prompting_pilot_analysis.json", result)
    print(json.dumps({"audit": result["audit"], "records": result["records"], "tokenizer": tokenizer,
                      "conditions": [{k: v for k, v in c.items() if k != "stories"} for c in conditions], "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
