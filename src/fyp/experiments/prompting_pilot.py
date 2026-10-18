"""Frozen single-call prompting study over the existing three-model main10 scope."""

from __future__ import annotations

import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path
from string import Formatter
from typing import Any

from fyp.data.pilot import load_pilot_examples, load_pilot_manifest, stable_file_hash
from fyp.data.schema import FlawedFictionsSplit
from fyp.evaluation.evaluator import evaluate_predictions
from fyp.experiments import baseline_pilot
from fyp.experiments.runner import (
    ExperimentRunError, RunConfig, RunResult, _fingerprint, _git_metadata,
    _parsed_to_record, _research_config, _to_simple_yaml, _write_config_yaml, run_experiment,
)
from fyp.models.registry import load_model_config
from fyp.prompting.parser import parse_official_response
from fyp.prompting.strategies import get_benchmark_strategy


DEFAULT_CONFIG = Path("configs/experiments/three_model_main10_prompting.json")


def _load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_settings() -> tuple[dict[str, Any], dict[str, Any]]:
    settings = _load_json(DEFAULT_CONFIG)
    base = baseline_pilot.load_settings()
    if Path(settings["baseline_configuration"]) != baseline_pilot.DEFAULT_CONFIG:
        raise ExperimentRunError("Unexpected baseline configuration")
    if settings["manifest"] != base["manifest"] or stable_file_hash(Path(settings["manifest"])) != settings["manifest_hash"]:
        raise ExperimentRunError("The prompting study must use the unchanged baseline manifest")
    if settings["generation"] != base["generation"] or settings["runtime"] != base["runtime"]:
        raise ExperimentRunError("Prompting settings differ from the verified baseline")
    if set(settings["models"]) != set(base["models"]):
        raise ExperimentRunError("Prompting models differ from the baseline matrix")
    if set(settings["strategies"]) != {"vanilla", "evidence_first", "structured_check", "verify_revise"}:
        raise ExperimentRunError("Unexpected prompting conditions")
    for key in ("run_root", "preparation_dir"):
        if not Path(settings[key]).resolve().is_relative_to(Path("experiments/three_model_main10_prompting").resolve()):
            raise ExperimentRunError("Prompting artifacts must use their isolated namespace")
    for name, entry in settings["strategies"].items():
        strategy = get_benchmark_strategy(name, version=entry["version"])
        if strategy.template_path != Path(entry["path"]) or stable_file_hash(strategy.template_path) != entry["hash"]:
            raise ExperimentRunError(f"Frozen prompt changed: {name}")
        template = strategy.template_path.read_text(encoding="utf-8")
        fields = [field for _, field, _, _ in Formatter().parse(template) if field is not None]
        if fields != ["story"] or strategy.examples_path is not None or strategy.num_model_calls != 1:
            raise ExperimentRunError("Prompt conditions must be single-call and insert only the original story")
    return settings, base


def validate_scope(settings: dict[str, Any], base: dict[str, Any]):
    manifest = load_pilot_manifest(settings["manifest"])
    if manifest.raw != baseline_pilot.build_manifest(base):
        raise ExperimentRunError("Shared scope differs from the authoritative inspected pilot")
    examples = load_pilot_examples(manifest)
    return manifest, examples


def validate_baseline(
    model: str, settings: dict[str, Any], base: dict[str, Any], manifest, examples,
    live_inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read-only reuse check; registration changes do not change baseline semantics."""

    entry = base["models"][model]
    run_dir = Path(base["run_root"]) / entry["run_id"]
    metadata = _load_json(run_dir / "metadata.json")
    records = _records(run_dir / "predictions.jsonl")
    original_provenance = _load_json(Path(base["preparation_dir"]) / "metadata.json")["provenance"]
    if metadata["pilot_provenance"] != original_provenance:
        raise ExperimentRunError("Baseline run provenance differs from its preparation snapshot")
    for source in settings["baseline_contract_sources"]:
        if stable_file_hash(Path(source)) != original_provenance["source_hashes"].get(source):
            raise ExperimentRunError(f"Baseline method changed: {source}")
    if metadata["dependencies"] != {key: importlib.metadata.version(key) for key in metadata["dependencies"]}:
        raise ExperimentRunError("Baseline evaluation dependencies changed")
    config = load_model_config(entry["config"])
    research = _research_config(RunConfig(
        model=model, strategy="vanilla", subsets=manifest.subsets,
        experiment_type="exploratory_baseline_pilot", run_id=entry["run_id"],
        run_root=Path(base["run_root"]), manifest_path=manifest.path,
        model_config_path=Path(entry["config"]), provenance=original_provenance,
    ), config, entry["run_id"], manifest=manifest)
    if metadata["configuration_fingerprint"] != _fingerprint(research):
        raise ExperimentRunError("Baseline configuration is incompatible")
    if (run_dir / "config.yaml").read_text(encoding="utf-8") != _to_simple_yaml(research):
        raise ExperimentRunError("Baseline snapshot differs from its recorded configuration")
    expected_ids = [example.example_id for example in examples]
    if len(records) != 10 or [row["example_id"] for row in records] != expected_ids or len(set(expected_ids)) != 10:
        raise ExperimentRunError("Baseline is incomplete, duplicated or has unexpected story IDs/order")
    strategy = get_benchmark_strategy("vanilla")
    raw_records = []
    for row, example in zip(records, examples):
        prompt = strategy.build_prompt(example)
        if (
            row["model"] != model or row["subset"] != "flawed_fictions"
            or row["input_hash"] != baseline_pilot._hash_text(example.story)
            or row["prompt_hash"] != prompt.prompt_hash
            or row["prompt_template_hash"] != settings["strategies"]["vanilla"]["hash"]
            or row["model_configuration"]["generation"] != config.generation.snapshot()
            or row["model_configuration"]["runtime"] != settings["runtime"]
        ):
            raise ExperimentRunError("Baseline story/prompt/settings mismatch")
        if not row["target_inference_failure"]:
            if row["effective_config"] != config.effective_ollama_request_fields():
                raise ExperimentRunError("Baseline effective request fields differ")
            if row["parsed_prediction"] != _parsed_to_record(parse_official_response(row["raw_response"])):
                raise ExperimentRunError("Stored baseline parsing differs from the preserved parser")
        raw_records.append({key: value for key, value in row.items() if key != "parsed_prediction"})
    report = evaluate_predictions(raw_records, dataset={FlawedFictionsSplit.FLAWED_FICTIONS: examples})
    if _load_json(run_dir / "metrics.json") != report.metrics:
        raise ExperimentRunError("Baseline metrics differ from fresh evaluation")
    if metadata["completed_example_count"] + metadata["failed_example_count"] != 10 or not metadata["end_timestamp"]:
        raise ExperimentRunError("Baseline completion bookkeeping is inconsistent")
    if metadata["failed_example_count"] != report.metrics["failure_counts"]["target_inference_failure"]:
        raise ExperimentRunError("Baseline terminal failure count differs")
    identity = original_provenance["models"][model]["identity"]
    if live_inspection is not None:
        live = live_inspection["models"].get(model, {})
        if live.get("available") and (live["identity"] != identity or live_inspection["ollama_version"] != metadata["ollama_version"]):
            raise ExperimentRunError("Installed model identity/runtime no longer matches the baseline")
    exported_path = Path(base["export_dir"]) / (entry["export_stem"] + ".json")
    exported = _load_json(exported_path)
    if exported["total"] != 10 or exported["failed"] != report.metrics["failure_counts"]["target_inference_failure"]:
        raise ExperimentRunError("Baseline export counts differ from authoritative records")
    expected_metrics = dict(report.metrics["classification_metrics"])
    expected_metrics.update({key: report.metrics["localization_metrics"][key] for key in ("ceeval_full", "ceeval_pos")})
    if exported["metrics"] != expected_metrics:
        raise ExperimentRunError("Baseline export metrics differ")
    if [row["story_id"] for row in exported["results"]] != expected_ids or [row["story"] for row in exported["results"]] != [e.story for e in examples]:
        raise ExperimentRunError("Baseline export text or order differs")
    for row, result in zip(records, exported["results"]):
        if row["target_inference_failure"]:
            status = "timeout" if "timeout" in row["inference_error"]["type"].lower() else "inference_error"
            if result["status"] != status or result["prediction"] is not None:
                raise ExperimentRunError("Baseline failure export differs")
        else:
            parsed = row["parsed_prediction"]
            positive = bool(parsed["cont_error"])
            expected_prediction = {
                "has_error": positive,
                "error_sentences": parsed["cont_error_lines"] if positive else [],
                "contradicted_sentences": parsed["contradicted_lines"] if positive else [],
                "explanation": parsed["cont_error_expl"],
            }
            if result["status"] != "ok" or result["prediction"] != expected_prediction:
                raise ExperimentRunError("Baseline parsed export differs")
    paths = list(run_dir.glob("*")) + [exported_path]
    return {
        "status": "REUSABLE_COMPLETE", "run_id": entry["run_id"], "run_dir": str(run_dir),
        "records": 10, "failed": report.metrics["failure_counts"]["target_inference_failure"],
        "configuration_fingerprint": metadata["configuration_fingerprint"],
        "identity": identity,
        "artifact_hashes": {path.as_posix(): stable_file_hash(path) for path in paths if path.is_file()},
        "reason": "Same original stories/order, vanilla rendering, model identity, options, retry policy and unchanged parser/evaluator contracts; source additions only register/orchestrate new conditions.",
    }


def _source_hashes(settings: dict[str, Any], base: dict[str, Any]) -> dict[str, str]:
    paths = list(Path("src/fyp").rglob("*.py"))
    paths += [DEFAULT_CONFIG, Path(settings["baseline_configuration"]), Path(settings["manifest"]),
              Path(base["parent_manifest"]), Path(base["raw_data_path"])]
    paths += [Path(entry["config"]) for entry in base["models"].values()]
    paths += [Path(entry["path"]) for entry in settings["strategies"].values()]
    return {path.as_posix(): stable_file_hash(path) for path in sorted(paths)}


def _run_id(settings: dict[str, Any], model: str, strategy: str) -> str:
    return f"exploratory-{settings['models'][model]}-{strategy.replace('_', '-')}-main10-prompting-v1"


def prepare_prompting_pilot() -> dict[str, Any]:
    settings, base = load_settings()
    manifest, examples = validate_scope(settings, base)
    rendered = {}
    capacity_records = []
    for name, entry in settings["strategies"].items():
        strategy = get_benchmark_strategy(name, version=entry["version"])
        rendered[name] = []
        for example in examples:
            prompt = strategy.build_prompt(example)
            row = {"example_id": example.example_id, "story_hash": baseline_pilot._hash_text(example.story),
                   "prompt_hash": prompt.prompt_hash, "prompt_utf8_bytes": len(prompt.prompt.encode("utf-8"))}
            rendered[name].append(row)
            capacity_records.append(row)
    inspection = baseline_pilot.inspect_models(base, capacity_records)
    baselines = {model: validate_baseline(model, settings, base, manifest, examples, inspection)
                 for model in settings["models"]}
    budgets = {}
    matrix = []
    for model in settings["models"]:
        live = inspection["models"].get(model, {})
        budgets[model] = {}
        for name, entry in settings["strategies"].items():
            if live.get("available"):
                identity = live["identity"]
                budgets[model][name] = baseline_pilot.context_budget(base, rendered[name], identity)
            matrix.append({
                "model": model, "strategy": name, "prompt_version": entry["version"],
                "prompt_hash": entry["hash"], "expected_records": 10,
                "status": "REUSE_BASELINE" if name == "vanilla" else "PREPARED_NOT_RUN",
                "run_id": baselines[model]["run_id"] if name == "vanilla" else _run_id(settings, model, name),
                "run_dir": baselines[model]["run_dir"] if name == "vanilla" else str(Path(settings["run_root"]) / _run_id(settings, model, name)),
            })
    provenance = {
        "experiment_name": settings["name"], "role": settings["role"],
        "dataset": manifest.raw["dataset"], "manifest_hash": manifest.manifest_hash,
        "story_hashes": {r["example_id"]: r["story_hash"] for r in manifest.raw["records"]},
        "strategies": settings["strategies"], "rendered_prompts": rendered,
        "models": inspection["models"], "ollama_version": inspection["ollama_version"],
        "baseline_references": baselines, "source_hashes": _source_hashes(settings, base),
        "git": _git_metadata(),
    }
    prep_dir = Path(settings["preparation_dir"])
    if (prep_dir / "metadata.json").exists():
        prior = _load_json(prep_dir / "metadata.json")
        if prior["provenance"] != provenance and any(Path(settings["run_root"]).glob("*/metadata.json")):
            raise ExperimentRunError("Cannot replace the prompting freeze after execution starts")
    available = all(inspection["models"].get(model, {}).get("available") for model in settings["models"])
    metadata = {
        "status": "PREPARED_NO_INFERENCE" if available else "BLOCKED_MODELS_UNAVAILABLE",
        "prepared_at": datetime.now(timezone.utc).isoformat(), "new_generations_executed": 0,
        "class_counts": manifest.raw["class_counts"], "matrix": matrix,
        "matrix_records": 120, "reused_baseline_records": 30, "new_records_planned": 90,
        "provenance": provenance, "context_budgets": budgets,
        "local_resources": baseline_pilot._local_resources(),
    }
    prep_dir.mkdir(parents=True, exist_ok=True)
    _write_config_yaml(prep_dir / "config.yaml", settings)
    (prep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def run_prompting_pilot(model: str, strategy: str) -> RunResult | dict[str, Any]:
    """Future execution only; vanilla is a read-only reference, never regenerated."""

    settings, base = load_settings()
    manifest, examples = validate_scope(settings, base)
    metadata = _load_json(Path(settings["preparation_dir"]) / "metadata.json")
    provenance = metadata["provenance"]
    if provenance["source_hashes"] != _source_hashes(settings, base):
        raise ExperimentRunError("Prompting source/configuration changed since freeze")
    capacity = [row for rows in provenance["rendered_prompts"].values() for row in rows]
    inspection = baseline_pilot.inspect_models(base, capacity)
    if metadata["status"] != "PREPARED_NO_INFERENCE" or inspection["models"] != provenance["models"] or inspection["ollama_version"] != provenance["ollama_version"]:
        raise ExperimentRunError("Prompting models/runtime differ from the prepared freeze")
    for name in settings["models"]:
        verified = validate_baseline(name, settings, base, manifest, examples, inspection)
        if verified != provenance["baseline_references"][name]:
            raise ExperimentRunError("Referenced baseline artifacts changed after freeze")
    if strategy == "vanilla":
        return provenance["baseline_references"][model]
    entry = settings["strategies"][strategy]
    run_id = _run_id(settings, model, strategy)
    run_dir = Path(settings["run_root"]) / run_id
    if strategy == "evidence_first":
        from fyp.experiments.prompting_outputs import export_prompting, update_evidence_progress
        update_evidence_progress(settings)
    error = None
    try:
        result = run_experiment(RunConfig(
            model=model, strategy=strategy, prompt_version=entry["version"],
            subsets=manifest.subsets, experiment_type="exploratory_prompting_pilot",
            manifest_path=manifest.path, model_config_path=Path(base["models"][model]["config"]),
            run_id=run_id, run_root=Path(settings["run_root"]), resume=run_dir.exists(), provenance=provenance,
            journal_requests=strategy == "evidence_first",
            on_checkpoint=(lambda _: update_evidence_progress(settings)) if strategy == "evidence_first" else None,
        ))
        if strategy == "evidence_first":
            export_prompting(model, strategy, settings, run_dir)
        return result
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if strategy == "evidence_first":
            update_evidence_progress(settings, last_error=error)
