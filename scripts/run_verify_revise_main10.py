"""Execute only the frozen verify-revise condition using the existing runner."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from fyp.data.pilot import load_pilot_manifest, stable_file_hash
from fyp.experiments import baseline_pilot
from fyp.experiments.prompting_outputs import _output_path, _read_rows, _write_json, export_prompting
from fyp.experiments.prompting_pilot import _run_id, _source_hashes, load_settings, validate_baseline, validate_scope
from fyp.experiments.runner import ExperimentRunError, RunConfig, run_experiment


STRATEGY = "verify_revise"
SCRIPT = Path("scripts/run_verify_revise_main10.py")
ROOT = Path("experiments/three_model_main10_prompting")
FREEZE = ROOT / "verify_revise_execution_freeze.json"
PROTECTED = ROOT / "verify_revise_protected_artifacts.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_protected():
    changed = [row["path"] for row in read(PROTECTED)
               if not Path(row["path"]).is_file() or stable_file_hash(Path(row["path"])) != row["hash"]]
    if changed:
        raise ExperimentRunError(f"Protected reference/final artifacts changed: {changed}")


def load_frozen():
    settings, base = load_settings()
    manifest, examples = validate_scope(settings, base)
    prepared = read(Path(settings["preparation_dir"]) / "metadata.json")
    provenance = prepared["provenance"]
    if provenance["source_hashes"] != _source_hashes(settings, base):
        raise ExperimentRunError("Prepared source/configuration freeze changed")
    if len(examples) != 10 or Counter(e.cont_error for e in examples) != {0.0: 5, 1.0: 5}:
        raise ExperimentRunError("Expected exactly ten main-subset stories, five per class")
    capacity = [r for records in provenance["rendered_prompts"].values() for r in records]
    inspection = baseline_pilot.inspect_models(base, capacity)
    if inspection["models"] != provenance["models"] or inspection["ollama_version"] != provenance["ollama_version"]:
        raise ExperimentRunError("Model identity/availability/runtime differs from the prepared freeze")
    for model in settings["models"]:
        reference = validate_baseline(model, settings, base, manifest, examples, inspection)
        if reference != provenance["baseline_references"][model]:
            raise ExperimentRunError(f"Baseline reference changed: {model}; no baseline was run")
        budget = prepared["context_budgets"][model][STRATEGY]
        if budget["required_token_upper_bound"] > budget["configured_context"]:
            raise ExperimentRunError("Complete input and output allowance exceed the frozen context")
    return settings, base, manifest, examples, prepared


def prepare_execution():
    settings, _, _, examples, prepared = load_frozen()
    # Preserve the shared preparation and completed conditions; freeze only this adapter separately.
    if not PROTECTED.exists():
        if any((Path(settings["run_root"]) / _run_id(settings, model, STRATEGY)).exists()
               for model in settings["models"]):
            raise ExperimentRunError("Existing verify-revise work has no adapter freeze; inspect before proceeding")
        prior = read(ROOT / "execution_protected_artifact_hashes.json")
        indexed = {row["path"]: row for row in prior}
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and path not in (PROTECTED, FREEZE):
                indexed[str(path)] = {"path": str(path), "hash": stable_file_hash(path)}
        _write_json(PROTECTED, list(indexed.values()))
    check_protected()
    execution = {"strategy": STRATEGY, "script": SCRIPT.as_posix(), "script_hash": stable_file_hash(SCRIPT),
                 "preparation_hash": stable_file_hash(Path(settings["preparation_dir"]) / "metadata.json"),
                 "protected_hash": stable_file_hash(PROTECTED),
                 "prompt": settings["strategies"][STRATEGY], "manifest_hash": settings["manifest_hash"]}
    if FREEZE.exists():
        if read(FREEZE) != execution:
            raise ExperimentRunError("Verify-revise execution adapter/freeze changed; refusing to resume")
    else:
        _write_json(FREEZE, execution)
    update_progress(settings)
    return {"status": "READY_NO_INFERENCE", "examples": len(examples), "class_counts": prepared["class_counts"],
            "execution": execution, "references": "Baselines verified; evidence-first/structured-check artifacts protected read-only"}


def update_progress(settings, *, last_error=None):
    expected_ids = [row.example_id for row in load_pilot_manifest(settings["manifest"]).records]
    models = []
    for model in settings["models"]:
        run_id = _run_id(settings, model, STRATEGY)
        directory = Path(settings["run_root"]) / run_id
        records = _read_rows(directory / "predictions.jsonl")
        ids = [row["example_id"] for row in records]
        if ids != expected_ids[:len(ids)] or len(ids) != len(set(ids)) or len(ids) > 10:
            raise ExperimentRunError("Verify-revise checkpoint has duplicate/unexpected/out-of-order IDs")
        events = _read_rows(directory / "requests.jsonl")
        starts = [e for e in events if e["event"] == "started"]
        finishes = {e["request_id"] for e in events if e["event"] == "finished"}
        gaps = []
        for record in records:
            logged = [e for e in starts if e["cache_key"] == record.get("cache_key")]
            history = record.get("attempt_history", [])
            if not logged or len(logged) != len(history) or any(e["request_id"] not in finishes for e in logged):
                gaps.append(record["example_id"])
        failed = sum(bool(row.get("target_inference_failure") or row.get("inference_error")) for row in records)
        output = _output_path(settings, model, STRATEGY)
        status = "PENDING" if len(records) < 10 else "RESULTS_COMPLETE_EXPORT_PENDING"
        if len(records) == 10 and output.exists():
            exported = read(output)
            if exported["total"] == 10 and exported["failed"] == failed and [r["story_id"] for r in exported["results"]] == expected_ids:
                status = "COMPLETE"
        models.append({"model": model, "run_id": run_id, "status": status, "completed": len(records),
                       "pending": 10 - len(records), "successful": len(records) - failed, "terminal_failed": failed,
                       "request_attempts_logged": len(starts), "historical_request_logging_gaps": gaps,
                       "output_path": str(output) if status == "COMPLETE" else None,
                       "resume_command": f".\\.venv\\Scripts\\python.exe scripts\\run_verify_revise_main10.py run --model {model}"})
    progress = {"strategy": STRATEGY, "updated_at": datetime.now(timezone.utc).isoformat(),
                "status": "COMPLETE" if all(r["status"] == "COMPLETE" for r in models) else "PENDING",
                "expected": 30, "completed": sum(r["completed"] for r in models),
                "pending": sum(r["pending"] for r in models), "last_error": last_error, "models": models}
    path = ROOT / "verify_revise_progress.json"
    _write_json(path, progress)
    return path


def run_condition(model):
    settings, base, manifest, _, prepared = load_frozen()
    if not FREEZE.exists() or not PROTECTED.exists():
        raise ExperimentRunError("Run prepare first; no inference started")
    execution = read(FREEZE)
    if execution["script_hash"] != stable_file_hash(SCRIPT) or execution["protected_hash"] != stable_file_hash(PROTECTED):
        raise ExperimentRunError("Execution adapter/protection snapshot differs from freeze")
    check_protected()
    provenance = {**prepared["provenance"], "execution_adapter": execution}
    directory = Path(settings["run_root"]) / _run_id(settings, model, STRATEGY)
    error = None
    try:
        result = run_experiment(RunConfig(
            model=model, strategy=STRATEGY, prompt_version=settings["strategies"][STRATEGY]["version"],
            subsets=manifest.subsets, experiment_type="exploratory_prompting_pilot",
            manifest_path=manifest.path, model_config_path=Path(base["models"][model]["config"]),
            run_id=directory.name, run_root=directory.parent, resume=directory.exists(),
            provenance=provenance, journal_requests=True, max_new_examples=10,
            on_checkpoint=lambda _: update_progress(settings),
        ))
        export_prompting(model, STRATEGY, settings, directory)
        return {"model": model, "completed": result.completed_examples, "terminal_failed": result.failed_examples,
                "skipped_completed": result.skipped_completed, "skipped_failed": result.skipped_failed}
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        update_progress(settings, last_error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "progress"))
    parser.add_argument("--model", choices=("qwen3:8b", "llama3.1:8b", "mistral:7b"))
    args = parser.parse_args()
    if args.action == "prepare":
        output = prepare_execution()
    elif args.action == "progress":
        settings, _ = load_settings()
        output = read(update_progress(settings))
    else:
        if args.model is None:
            parser.error("run requires --model; conditions must be executed sequentially")
        output = run_condition(args.model)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
