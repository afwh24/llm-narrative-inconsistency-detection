"""Simple prompting exports and durable, truthful evidence-first progress."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fyp.data.pilot import load_pilot_examples, load_pilot_manifest
from fyp.evaluation.evaluator import _extract_prediction, evaluate_predictions
from fyp.experiments.runner import ExperimentRunError


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _output_path(settings: dict[str, Any], model: str, strategy: str) -> Path:
    stem = settings["models"][model].replace("-", "_")
    return Path(settings["run_root"]).parent / "exports" / f"{stem}_{strategy}.json"


def export_prompting(model: str, strategy: str, settings: dict[str, Any], run_dir: Path) -> Path:
    manifest = load_pilot_manifest(settings["manifest"])
    examples = load_pilot_examples(manifest)
    records = _read_rows(run_dir / "predictions.jsonl")
    expected = {row["example_id"]: row for row in manifest.raw["records"]}
    if len(records) != 10 or {row["example_id"] for row in records} != set(expected):
        raise ExperimentRunError("Cannot export pending, duplicated or unexpected results as a complete pilot")
    indexed = {row["example_id"]: row for row in records}
    ordered = [indexed[example.example_id] for example in examples]
    for row in ordered:
        if row["model"] != model or row["benchmark_strategy"] != strategy or row["subset"] != "flawed_fictions" or row["input_hash"] != expected[row["example_id"]]["story_hash"]:
            raise ExperimentRunError("Export record identity/story mismatch")
    report = evaluate_predictions(ordered, dataset={manifest.subsets[0]: examples})
    results = []
    for example, row in zip(examples, ordered):
        parsed, flags = _extract_prediction(row)
        if flags["target_inference_failure"]:
            kind = (row.get("inference_error") or {}).get("type", "").lower()
            status = "timeout" if "timeout" in kind else "inference_error"
            prediction = None
        else:
            if parsed is None:
                raise ExperimentRunError("Successful inference record lacks a usable official-style prediction")
            status = "ok"
            positive = bool(parsed.cont_error)
            prediction = {
                "has_error": positive,
                "error_sentences": list(parsed.cont_error_lines) if positive else [],
                "contradicted_sentences": list(parsed.contradicted_lines) if positive else [],
                "explanation": parsed.cont_error_expl,
            }
        results.append({"story_id": example.raw_record["example_id"], "story": example.raw_record["story"],
                        "status": status, "prediction": prediction})
    output = {
        "model": model, "prompt": strategy, "total": len(results),
        "failed": report.metrics["failure_counts"]["target_inference_failure"],
        "metrics": {**report.metrics["classification_metrics"],
                    **{key: report.metrics["localization_metrics"][key] for key in ("ceeval_full", "ceeval_pos")}},
        "results": results,
    }
    path = _output_path(settings, model, strategy)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != output:
        raise ExperimentRunError("Existing export differs from authoritative saved predictions")
    _write_json(path, output)
    return path


def update_evidence_progress(settings: dict[str, Any], *, last_error: str | None = None) -> Path:
    from fyp.experiments.prompting_pilot import _run_id

    rows = []
    for model in settings["models"]:
        run_id = _run_id(settings, model, "evidence_first")
        run_dir = Path(settings["run_root"]) / run_id
        records = _read_rows(run_dir / "predictions.jsonl")
        ids = [row["example_id"] for row in records]
        manifest_ids = {row.example_id for row in load_pilot_manifest(settings["manifest"]).records}
        if len(ids) != len(set(ids)) or not set(ids).issubset(manifest_ids):
            raise ExperimentRunError("Progress checkpoint has duplicate/unexpected IDs")
        journal = _read_rows(run_dir / "requests.jsonl")
        starts = [event for event in journal if event.get("event") == "started"]
        gaps = []
        for record in records:
            logged = [event for event in starts if event.get("cache_key") == record.get("cache_key")]
            history = record.get("attempt_history", [])
            if not logged or len(logged) != len(history) or any(row.get("status") == "unfinished" or row.get("latency_seconds") is None for row in history):
                gaps.append(record["example_id"])
        failed = sum(bool(row.get("target_inference_failure")) for row in records)
        output_path = _output_path(settings, model, "evidence_first")
        status = "PENDING"
        if len(records) == 10:
            status = "RESULTS_COMPLETE_EXPORT_PENDING"
            if output_path.exists():
                exported = json.loads(output_path.read_text(encoding="utf-8"))
                if exported["total"] == 10 and exported["failed"] == failed:
                    status = "COMPLETE"
        rows.append({
            "model": model, "run_id": run_id, "status": status,
            "completed": len(records), "successful": len(records) - failed,
            "terminal_failed": failed, "pending": 10 - len(records),
            "request_attempts_logged": len(starts), "historical_request_logging_gaps": gaps,
            "output_path": str(output_path) if status == "COMPLETE" else None,
            "resume_command": f".\\.venv\\Scripts\\python.exe -m fyp.cli prompt-pilot run --model {model} --strategy evidence_first",
        })
    progress = {
        "strategy": "evidence_first", "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE" if all(row["status"] == "COMPLETE" for row in rows) else "PENDING",
        "expected": 30, "completed": sum(row["completed"] for row in rows),
        "pending": sum(row["pending"] for row in rows), "last_error": last_error, "models": rows,
    }
    path = Path(settings["run_root"]).parent / "evidence_first_progress.json"
    _write_json(path, progress)
    return path
