"""Preparation and guarded execution of the separate three-model main-story pilot."""

from __future__ import annotations

import csv
import ctypes
import hashlib
import io
import json
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fyp.data.pilot import load_pilot_examples, load_pilot_manifest, stable_file_hash
from fyp.data.schema import FlawedFictionsSplit
from fyp.evaluation.evaluator import _extract_prediction, evaluate_predictions
from fyp.experiments.runner import (
    DATASET_IDENTITY,
    DATASET_VERSION,
    ExperimentRunError,
    RunConfig,
    RunResult,
    _git_metadata,
    _write_config_yaml,
    run_experiment,
)
from fyp.models.ollama_client import OllamaClient, OllamaError
from fyp.models.registry import load_model_config
from fyp.prompting.strategies import get_benchmark_strategy


DEFAULT_CONFIG = Path("configs/experiments/three_model_main10.json")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def load_settings() -> dict[str, Any]:
    settings = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if settings["subset"] != "flawed_fictions" or settings["expected_examples"] != 10:
        raise ExperimentRunError("This exploratory pilot must use exactly 10 main-subset stories")
    if set(settings["models"]) != {"qwen3:8b", "llama3.1:8b", "mistral:7b"}:
        raise ExperimentRunError("Unexpected model matrix")
    for key in ("run_root", "preparation_dir", "export_dir"):
        if not Path(settings[key]).resolve().is_relative_to(
            Path("experiments/three_model_main10").resolve()
        ):
            raise ExperimentRunError(f"{key} must remain inside the separate pilot namespace")
    if Path(settings["manifest"]).resolve() != Path("data/pilot/three_model_main10.json").resolve():
        raise ExperimentRunError("Unexpected derived manifest path")
    strategy = get_benchmark_strategy(settings["strategy"])
    if settings["strategy"] != "vanilla" or strategy.prompt_version != settings["prompt_version"]:
        raise ExperimentRunError("This baseline must use the existing vanilla prompt")
    if stable_file_hash(strategy.template_path) != settings["prompt_template_hash"]:
        raise ExperimentRunError("Vanilla prompt hash changed")
    for name, entry in settings["models"].items():
        if Path(entry["run_id"]).name != entry["run_id"] or Path(entry["export_stem"]).name != entry["export_stem"]:
            raise ExperimentRunError("Run IDs and export stems must be plain filenames")
        config = load_model_config(entry["config"])
        generation = config.generation.snapshot()
        expected_think = False if name == "qwen3:8b" else None
        if (
            config.model != name
            or generation.pop("think") is not expected_think
            or generation != settings["generation"]
            or asdict(config.runtime) != settings["runtime"]
        ):
            raise ExperimentRunError(f"Mismatched frozen settings for {name}")
    return settings


def build_manifest(settings: dict[str, Any]) -> dict[str, Any]:
    """Filter the authoritative manifest, never select replacement examples."""

    parent = load_pilot_manifest(settings["parent_manifest"])
    if parent.manifest_hash != settings["parent_manifest_hash"]:
        raise ExperimentRunError("Authoritative pilot manifest hash changed")
    if stable_file_hash(Path(settings["raw_data_path"])) != settings["raw_data_hash"]:
        raise ExperimentRunError("Official main-subset data hash changed")
    if parent.raw.get("dataset") != {"identity": DATASET_IDENTITY, "version": DATASET_VERSION}:
        raise ExperimentRunError("Unexpected pilot dataset identity")
    selected = [r for r in parent.records if r.subset.value == settings["subset"]]
    if len(selected) != 10 or len({r.example_id for r in selected}) != 10:
        raise ExperimentRunError("Missing or duplicate main-subset pilot IDs")
    counts = {"negative": sum(r.cont_error == 0 for r in selected),
              "positive": sum(r.cont_error == 1 for r in selected)}
    if counts != {"negative": settings["expected_negative"], "positive": settings["expected_positive"]}:
        raise ExperimentRunError("Pilot class counts changed")
    examples = {e.example_id: e for e in load_pilot_examples(replace(parent, records=tuple(selected)))}
    strategy = get_benchmark_strategy(settings["strategy"])
    records = []
    for record in selected:
        example = examples[record.example_id]
        if example.story != example.raw_record["story"]:
            raise ExperimentRunError(f"Story text was altered: {example.example_id}")
        prompt = strategy.build_prompt(example)
        records.append({
            **record.to_manifest_dict(),
            "story_hash": _hash_text(example.story),
            "story_utf8_bytes": len(example.story.encode("utf-8")),
            "prompt_hash": prompt.prompt_hash,
            "prompt_utf8_bytes": len(prompt.prompt.encode("utf-8")),
        })
    return {
        "name": settings["name"],
        "scope_role": "pilot",
        "purpose": "Previously inspected exploratory baseline pilot; not an unseen or final test set.",
        "parent_manifest": settings["parent_manifest"],
        "parent_manifest_hash": parent.manifest_hash,
        "selection_method": "Reuse exactly the main-subset rows of pilot_20 in their existing order.",
        "dataset": parent.raw["dataset"],
        "raw_data_hash": settings["raw_data_hash"],
        "class_counts": counts,
        "records": records,
    }


def context_budget(
    settings: dict[str, Any], records: list[dict[str, Any]], show: dict[str, Any]
) -> dict[str, Any]:
    """Use a conservative byte-token bound, including native template overhead."""

    info = show["model_info"]
    architecture = info["general.architecture"]
    supported = info[f"{architecture}.context_length"]
    if info.get("tokenizer.ggml.model") not in {"gpt2", "llama"}:
        raise ExperimentRunError("Unverified tokenizer; context capacity needs a new audit")
    # Byte BPE / byte-fallback tokenizers need at most one token per input byte.
    # Counting the full template source overestimates this single-user wrapper.
    wrapper = len(show.get("template", "").encode("utf-8"))
    wrapper += len(show.get("system", "").encode("utf-8"))
    wrapper += settings["template_token_margin"]
    largest = max(r["prompt_utf8_bytes"] for r in records)
    needed = largest + wrapper + settings["generation"]["num_predict"]
    chosen = settings["generation"]["num_ctx"]
    if needed > chosen or chosen > supported:
        raise ExperimentRunError(f"Context insufficient: need {needed}, configured {chosen}, supported {supported}")
    heads = info[f"{architecture}.attention.head_count"]
    dimension = info.get(f"{architecture}.attention.key_length", info[f"{architecture}.embedding_length"] // heads)
    value_dimension = info.get(f"{architecture}.attention.value_length", dimension)
    kv_bytes = (
        info[f"{architecture}.block_count"]
        * info[f"{architecture}.attention.head_count_kv"]
        * (dimension + value_dimension) * 2 * chosen
    )
    return {
        "configured_context": chosen,
        "supported_context": supported,
        "largest_prompt_utf8_bytes": largest,
        "wrapper_and_special_token_bound": wrapper,
        "output_allowance": settings["generation"]["num_predict"],
        "required_token_upper_bound": needed,
        "method": "UTF-8 byte bound plus full template/system bytes and special-token margin; not measured tokenization",
        "estimated_f16_kv_cache_bytes": kv_bytes,
    }


def inspect_models(settings: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    first = next(iter(settings["models"].values()))
    inspector = OllamaClient(load_model_config(first["config"]))
    try:
        installed = {m["name"]: m for m in inspector.list_model_details()}
        version = inspector.get_version()
    except OllamaError as exc:
        return {"ollama_version": None, "error": str(exc), "models": {}}
    result: dict[str, Any] = {"ollama_version": version, "models": {}}
    for name, entry in settings["models"].items():
        if name not in installed:
            result["models"][name] = {"available": False, "manual_pull": f"ollama pull {name}"}
            continue
        show = OllamaClient(load_model_config(entry["config"])).show_model()
        budget = context_budget(settings, records, show)
        thinking_supported = "thinking" in show.get("capabilities", [])
        if thinking_supported != (name == "qwen3:8b"):
            raise ExperimentRunError(f"Thinking capabilities changed for {name}")
        result["models"][name] = {
            "available": True,
            "identity": {
                "tag": name,
                "digest": installed[name]["digest"],
                "size_bytes": installed[name]["size"],
                "details": show["details"],
                "capabilities": show.get("capabilities", []),
                "model_info": show["model_info"],
                "template": show.get("template", ""),
                "template_hash": _hash_text(show.get("template", "")),
                "system": show.get("system", ""),
                "parameters": show.get("parameters", ""),
            },
            "thinking": "disabled" if thinking_supported else "not_applicable_option_omitted",
            "context_budget": budget,
        }
    return result


def _source_hashes(settings: dict[str, Any]) -> dict[str, str]:
    paths = list(Path("src/fyp").rglob("*.py"))
    paths += [DEFAULT_CONFIG, Path(settings["parent_manifest"]), Path(settings["raw_data_path"])]
    paths += [Path(e["config"]) for e in settings["models"].values()]
    paths.append(get_benchmark_strategy(settings["strategy"]).template_path)
    return {p.as_posix(): stable_file_hash(p) for p in sorted(paths)}


def _local_resources() -> dict[str, Any]:
    result: dict[str, Any] = {"loaded_model_memory_test": "not performed; no inference authorized"}
    if hasattr(ctypes, "windll"):
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total_physical", "available_physical", "total_pagefile", "available_pagefile",
                    "total_virtual", "available_virtual", "available_extended",
                )
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result["physical_memory_bytes"] = status.total_physical
            result["available_physical_memory_bytes"] = status.available_physical
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        result["gpus"] = [
            {"name": row[0].strip(), "total_bytes": int(row[1]) * 1048576, "free_bytes": int(row[2]) * 1048576}
            for row in csv.reader(io.StringIO(gpu.stdout))
        ]
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result["gpu_inspection_error"] = str(exc)
    result["execution_policy"] = "Serial; possible CPU offload. Static inspection does not guarantee loading or latency."
    return result


def prepare_pilot() -> dict[str, Any]:
    """Write only scope/configuration metadata; never call generate."""

    settings = load_settings()
    manifest = build_manifest(settings)
    manifest_path = Path(settings["manifest"])
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise ExperimentRunError("Existing derived pilot manifest differs; refusing to overwrite")
    manifest_path.write_text(manifest_text, encoding="utf-8")
    inspection = inspect_models(settings, manifest["records"])
    provenance = {
        "experiment_name": settings["name"],
        "role": settings["role"],
        "dataset_identity": DATASET_IDENTITY,
        "dataset_version": DATASET_VERSION,
        "manifest_hash": stable_file_hash(manifest_path),
        "prompt_version": settings["prompt_version"],
        "prompt_template_hash": settings["prompt_template_hash"],
        "source_hashes": _source_hashes(settings),
        "git": _git_metadata(),
        "ollama_version": inspection["ollama_version"],
        "models": inspection["models"],
        "story_hashes": {r["example_id"]: r["story_hash"] for r in manifest["records"]},
        "prompt_hashes": {r["example_id"]: r["prompt_hash"] for r in manifest["records"]},
    }
    prep_dir = Path(settings["preparation_dir"])
    metadata_path = prep_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing["provenance"] != provenance and any(Path(settings["run_root"]).glob("*/metadata.json")):
            raise ExperimentRunError("Preparation changed after runs began; refusing to replace the freeze")
    ready = all(inspection["models"].get(name, {}).get("available") for name in settings["models"])
    metadata = {
        "status": "PREPARED_NO_INFERENCE" if ready else "BLOCKED_MODELS_UNAVAILABLE",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "official_examples_executed": 0,
        "expected_examples_per_model": 10,
        "class_counts": manifest["class_counts"],
        "provenance": provenance,
        "local_resources": _local_resources(),
        "inspection_error": inspection.get("error"),
    }
    prep_dir.mkdir(parents=True, exist_ok=True)
    _write_config_yaml(prep_dir / "config.yaml", settings)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def run_pilot(model: str) -> RunResult:
    """Next-stage entry point: verify the freeze, then delegate to the existing runner."""

    settings = load_settings()
    manifest = build_manifest(settings)
    actual = json.loads(Path(settings["manifest"]).read_text(encoding="utf-8"))
    if actual != manifest:
        raise ExperimentRunError("Derived manifest differs from the authoritative pilot scope")
    metadata_path = Path(settings["preparation_dir"]) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    provenance = metadata["provenance"]
    if metadata["status"] != "PREPARED_NO_INFERENCE":
        raise ExperimentRunError("Preparation is blocked; install missing models and prepare again")
    if provenance["source_hashes"] != _source_hashes(settings):
        raise ExperimentRunError("Source/configuration changed since preparation; audit and prepare again")
    if provenance["manifest_hash"] != stable_file_hash(Path(settings["manifest"])):
        raise ExperimentRunError("Prepared manifest hash changed")
    inspection = inspect_models(settings, manifest["records"])
    if inspection["models"] != provenance["models"] or inspection["ollama_version"] != provenance["ollama_version"]:
        raise ExperimentRunError("Installed model identities or Ollama version changed since preparation")
    entry = settings["models"][model]
    run_dir = Path(settings["run_root"]) / entry["run_id"]
    result = run_experiment(RunConfig(
        model=model,
        strategy=settings["strategy"],
        subsets=(FlawedFictionsSplit.FLAWED_FICTIONS,),
        experiment_type="exploratory_baseline_pilot",
        manifest_path=Path(settings["manifest"]),
        model_config_path=Path(entry["config"]),
        run_id=entry["run_id"],
        run_root=Path(settings["run_root"]),
        resume=run_dir.exists(),
        provenance=provenance,
    ))
    export_baseline(model)
    return result


def export_baseline(model: str) -> Path:
    """Export the simple JSON view alongside unmodified internal record copies."""

    settings = load_settings()
    entry = settings["models"][model]
    run_dir = Path(settings["run_root"]) / entry["run_id"]
    records = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    manifest = load_pilot_manifest(settings["manifest"])
    expected = {r["example_id"]: r for r in manifest.raw["records"]}
    if len(records) != 10 or {r["example_id"] for r in records} != set(expected):
        raise ExperimentRunError("Cannot export incomplete, duplicate or unexpected pilot IDs")
    for record in records:
        gold = expected[record["example_id"]]
        if (
            record["model"] != model or record["subset"] != "flawed_fictions"
            or record["benchmark_strategy"] != "vanilla"
            or record["input_hash"] != gold["story_hash"]
            or record["prompt_hash"] != gold["prompt_hash"]
            or record["prompt_template_hash"] != settings["prompt_template_hash"]
        ):
            raise ExperimentRunError("Export record identity/text/prompt mismatch")
    examples = load_pilot_examples(manifest)
    records_by_id = {record["example_id"]: record for record in records}
    ordered_records = [records_by_id[example.example_id] for example in examples]
    report = evaluate_predictions(ordered_records, dataset={FlawedFictionsSplit.FLAWED_FICTIONS: examples})
    results = []
    for example, record in zip(examples, ordered_records):
        parsed, flags = _extract_prediction(record)
        if flags["target_inference_failure"]:
            error_type = (record.get("inference_error") or {}).get("type", "")
            status = "timeout" if "timeout" in error_type.lower() else "inference_error"
            prediction = None
        else:
            if parsed is None:
                raise ExperimentRunError("Successful inference has no usable official-style prediction")
            status = "ok"
            has_error = bool(parsed.cont_error)
            prediction = {
                "has_error": has_error,
                "error_sentences": list(parsed.cont_error_lines) if has_error else [],
                "contradicted_sentences": list(parsed.contradicted_lines) if has_error else [],
                "explanation": parsed.cont_error_expl,
            }
        results.append({
            "story_id": example.raw_record["example_id"],
            "story": example.raw_record["story"],
            "status": status,
            "prediction": prediction,
        })
    simple_output = {
        "model": model,
        "prompt": "baseline",
        "total": report.metrics["num_examples"],
        "failed": report.metrics["failure_counts"]["target_inference_failure"],
        "metrics": {
            **report.metrics["classification_metrics"],
            "ceeval_full": report.metrics["localization_metrics"]["ceeval_full"],
            "ceeval_pos": report.metrics["localization_metrics"]["ceeval_pos"],
        },
        "results": results,
    }
    output_dir = Path(settings["export_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, suffix in (("predictions.jsonl", ".jsonl"), ("metadata.json", "_metadata.json"),
                           ("metrics.json", "_metrics.json"), ("evaluation_examples.jsonl", "_evaluation.jsonl")):
        destination = output_dir / (entry["export_stem"] + suffix)
        content = (run_dir / source).read_bytes()
        if suffix == ".jsonl" and destination.exists() and destination.read_bytes() != content:
            raise ExperimentRunError(f"Existing export differs: {destination}")
        destination.write_bytes(content)
    destination = output_dir / (entry["export_stem"] + ".json")
    content = json.dumps(simple_output, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != content:
        raise ExperimentRunError(f"Existing simple export differs: {destination}")
    destination.write_text(content, encoding="utf-8")
    return destination
