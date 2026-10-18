"""Run-directory, checkpointing, and inference orchestration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fyp.data.loader import DEFAULT_RAW_DATA_DIR, load_all_splits
from fyp.data.pilot import PilotManifest, load_pilot_examples, load_pilot_manifest
from fyp.data.schema import SPLIT_SPECS, FlawedFictionsExample, FlawedFictionsSplit
from fyp.evaluation.evaluator import evaluate_run_directory
from fyp.models.base import ModelClient, ModelConfig, ModelRequest
from fyp.models.ollama_client import OllamaClient, OllamaError, OllamaInterruptedError
from fyp.models.registry import build_model_client, get_model_config, load_model_config
from fyp.prompting.parser import ParsedPrediction, parse_official_response
from fyp.prompting.strategies import get_benchmark_strategy


DEFAULT_RUN_ROOT = Path("experiments/runs")
DATASET_IDENTITY = "kahuja/flawed-fictions"
DATASET_VERSION = "733717281afdb715538b753f0f85ef7d55568b85"


class ExperimentRunError(RuntimeError):
    """Raised when a run cannot be created, resumed, or completed."""


@dataclass(frozen=True)
class RunConfig:
    """Research-relevant configuration for one target-model run."""

    model: str
    strategy: str
    subsets: tuple[FlawedFictionsSplit, ...]
    limit: int | None = None
    experiment_type: str = "baseline"
    run_id: str | None = None
    run_root: Path = DEFAULT_RUN_ROOT
    raw_data_dir: Path = DEFAULT_RAW_DATA_DIR
    manifest_path: Path | None = None
    resume: bool = False
    retry_failures: bool = False
    think_override: bool | str | None = None
    max_new_examples: int | None = None
    model_config_path: Path | None = None
    provenance: Mapping[str, Any] | None = None
    prompt_version: str | None = None
    journal_requests: bool = False
    on_checkpoint: Callable[[Path], None] | None = None

    def resolved_run_id(self) -> str:
        if self.run_id:
            return self.run_id
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        subset_label = "_".join(split.value for split in self.subsets)
        model_label = self.model.replace(":", "_").replace("/", "_")
        return f"{timestamp}_{model_label}_{self.strategy}_{subset_label}_{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunResult:
    """Summary of a completed or resumed run invocation."""

    run_dir: Path
    run_id: str
    expected_examples: int
    completed_examples: int
    failed_examples: int
    skipped_completed: int
    skipped_failed: int
    evaluated: bool


def run_experiment(
    config: RunConfig,
    *,
    dataset: Mapping[FlawedFictionsSplit, Sequence[FlawedFictionsExample]] | None = None,
    client_factory: Callable[[ModelConfig], ModelClient] | None = None,
    now: Callable[[], datetime] | None = None,
) -> RunResult:
    """Run or resume a small, append-only FLAWEDFICTIONS inference job."""

    model_config = _apply_run_overrides(
        load_model_config(config.model_config_path)
        if config.model_config_path is not None
        else get_model_config(config.model),
        config,
    )
    if model_config.model != config.model:
        raise ExperimentRunError("Requested model does not match its configuration")
    strategy = get_benchmark_strategy(config.strategy, version=config.prompt_version)
    manifest = load_pilot_manifest(config.manifest_path) if config.manifest_path else None
    selected_dataset = dataset or load_all_splits(config.subsets, raw_data_dir=config.raw_data_dir)
    examples = _select_examples(
        selected_dataset,
        config.subsets,
        config.limit,
        manifest=manifest,
        raw_data_dir=config.raw_data_dir,
    )
    run_id = config.resolved_run_id()
    run_dir = config.run_root / run_id
    clock = now or (lambda: datetime.now(timezone.utc))
    research_config = _research_config(config, model_config, run_id, manifest=manifest)
    fingerprint = _fingerprint(research_config)

    if run_dir.exists() and not config.resume:
        raise ExperimentRunError(f"Run directory already exists: {run_dir}")
    if config.resume:
        if not run_dir.exists():
            raise ExperimentRunError(f"Cannot resume missing run directory: {run_dir}")
        _assert_compatible_resume(run_dir, fingerprint)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_config_yaml(run_dir / "config.yaml", research_config)
        _write_metadata(
            run_dir,
            _initial_metadata(
                config=research_config,
                fingerprint=fingerprint,
                expected_examples=len(examples),
                start_time=clock(),
                model_config=model_config,
            ),
        )
        (run_dir / "predictions.jsonl").touch()
        (run_dir / "errors.jsonl").touch()

    completed_keys, failed_keys = _load_checkpoint_keys(run_dir)
    if config.journal_requests and client_factory is None:
        from fyp.models.request_journal import JournaledOllamaClient
        saved_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        legacy = bool(saved_metadata.get("legacy_request_logging_gap")) or (config.resume and not saved_metadata.get("request_logging"))
        client = JournaledOllamaClient(model_config, run_dir / "requests.jsonl", legacy_run=legacy)
        if legacy:
            saved_metadata["legacy_request_logging_gap"] = True
        saved_metadata["request_logging"] = "requests.jsonl: durable assembled-request start and response/error finish events"
        _write_metadata(run_dir, saved_metadata)
    else:
        client = (client_factory or build_model_client)(model_config)
    if config.on_checkpoint:
        config.on_checkpoint(run_dir)
    counters = {
        "completed": 0,
        "failed": 0,
        "skipped_completed": 0,
        "skipped_failed": 0,
    }

    for example in examples:
        if (
            config.max_new_examples is not None
            and counters["completed"] + counters["failed"] >= config.max_new_examples
        ):
            break
        prompt = strategy.build_prompt(example)
        frozen_prompts = (config.provenance or {}).get("rendered_prompts", {}).get(config.strategy)
        if frozen_prompts is not None:
            frozen = next((row for row in frozen_prompts if row["example_id"] == example.example_id), None)
            if frozen is None or frozen["prompt_hash"] != prompt.prompt_hash or frozen["story_hash"] != _sha256_text(example.story):
                raise ExperimentRunError("Frozen prompt/story changed; pending example was not sent")
        stage = prompt.stages[0]
        key_material = _cache_key_material(example, config, model_config, prompt, stage)
        cache_key = _fingerprint(key_material)
        record_base = _record_base(
            example=example,
            config=config,
            run_id=run_id,
            model_config=model_config,
            prompt_build=prompt,
            stage_id=stage.name,
            input_hash=_sha256_text(example.story),
            cache_key=cache_key,
        )

        if cache_key in completed_keys:
            counters["skipped_completed"] += 1
            continue
        if cache_key in failed_keys and not config.retry_failures:
            counters["skipped_failed"] += 1
            continue
        if cache_key in failed_keys and config.retry_failures:
            _remove_prediction_records(run_dir / "predictions.jsonl", {cache_key})
            failed_keys.remove(cache_key)

        latency_started = time.perf_counter()
        if config.journal_requests and hasattr(client, "set_context"):
            identity = (config.provenance or {}).get("models", {}).get(config.model, {}).get("identity", {})
            client.set_context({
                **record_base, "model_digest": identity.get("digest"),
                "prompt_template": strategy.template_path.read_text(encoding="utf-8"),
            })
        try:
            response = client.generate(ModelRequest(prompt=stage.content))
        except (OllamaInterruptedError, KeyboardInterrupt):
            _refresh_metadata(run_dir, clock())
            if config.on_checkpoint:
                config.on_checkpoint(run_dir)
            raise
        except Exception as exc:
            latency_seconds = time.perf_counter() - latency_started
            if config.journal_requests:
                latency_seconds = max(latency_seconds, sum(row.get("latency_seconds") or 0.0 for row in getattr(client, "last_attempt_history", [])))
            available_raw = getattr(client, "last_raw_response", None) or {}
            failure_record = {
                **record_base,
                "target_inference_failure": True,
                "inference_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "raw_response": available_raw.get("response"),
                "thinking": available_raw.get("thinking"),
                "ollama_raw": _compact_ollama_raw(available_raw),
                "attempt_history": list(getattr(client, "last_attempt_history", [])),
                "parsed_prediction": None,
                "latency_seconds": round(latency_seconds, 6),
                "completed_at": _isoformat(clock()),
            }
            _append_jsonl(run_dir / "predictions.jsonl", failure_record)
            _append_jsonl(run_dir / "errors.jsonl", failure_record)
            failed_keys.add(cache_key)
            counters["failed"] += 1
        else:
            latency_seconds = time.perf_counter() - latency_started
            if config.journal_requests:
                latency_seconds = max(latency_seconds, sum(row.get("latency_seconds") or 0.0 for row in getattr(client, "last_attempt_history", [])))
            parsed = parse_official_response(response.response)
            success_record = {
                **record_base,
                "target_inference_failure": False,
                "inference_error": None,
                "raw_response": response.response,
                "thinking": response.thinking,
                "attempt_history": list(getattr(client, "last_attempt_history", [])),
                "ollama_raw": _compact_ollama_raw(response.raw),
                "effective_config": dict(response.effective_config),
                "parsed_prediction": _parsed_to_record(parsed),
                "latency_seconds": round(latency_seconds, 6),
                "completed_at": _isoformat(clock()),
            }
            _append_jsonl(run_dir / "predictions.jsonl", success_record)
            completed_keys.add(cache_key)
            counters["completed"] += 1

        _refresh_metadata(run_dir, clock())
        if config.on_checkpoint:
            config.on_checkpoint(run_dir)

    _refresh_metadata(run_dir, clock(), final=True)
    evaluated = False
    if (run_dir / "predictions.jsonl").exists():
        evaluate_run_directory(run_dir, dataset=selected_dataset)
        evaluated = True

    completed_total, failed_total = _count_prediction_status(run_dir)
    _write_run_summary(run_dir, completed_total, failed_total, len(examples), counters)
    return RunResult(
        run_dir=run_dir,
        run_id=run_id,
        expected_examples=len(examples),
        completed_examples=completed_total,
        failed_examples=failed_total,
        skipped_completed=counters["skipped_completed"],
        skipped_failed=counters["skipped_failed"],
        evaluated=evaluated,
    )


def _apply_run_overrides(model_config: ModelConfig, config: RunConfig) -> ModelConfig:
    if config.think_override is None:
        return model_config
    return replace(
        model_config,
        generation=replace(model_config.generation, think=config.think_override),
    )


def _select_examples(
    dataset: Mapping[FlawedFictionsSplit, Sequence[FlawedFictionsExample]],
    subsets: Sequence[FlawedFictionsSplit],
    limit: int | None,
    *,
    manifest: PilotManifest | None = None,
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
) -> list[FlawedFictionsExample]:
    if manifest is not None:
        if limit is not None:
            raise ExperimentRunError("Do not combine --limit with a fixed pilot manifest")
        return load_pilot_examples(manifest, raw_data_dir=raw_data_dir)
    examples: list[FlawedFictionsExample] = []
    for subset in subsets:
        examples.extend(dataset[subset])
    return examples if limit is None else examples[:limit]


def _research_config(
    config: RunConfig,
    model_config: ModelConfig,
    run_id: str,
    *,
    manifest: PilotManifest | None = None,
) -> dict[str, Any]:
    strategy = get_benchmark_strategy(config.strategy, version=config.prompt_version)
    dataset_config: dict[str, Any] = {
        "identity": DATASET_IDENTITY,
        "version": DATASET_VERSION,
        "subsets": [split.value for split in config.subsets],
        "expected_subset_counts": {
            split.value: SPLIT_SPECS[split].expected_examples for split in config.subsets
        },
        "raw_data_dir": str(config.raw_data_dir),
        "selected_limit": config.limit,
    }
    if manifest is not None:
        dataset_config.update(
            {
                "fixed_manifest": str(manifest.path),
                "fixed_manifest_name": manifest.name,
                "fixed_manifest_hash": manifest.manifest_hash,
                "fixed_manifest_example_count": len(manifest.records),
                "fixed_manifest_role": manifest.role,
            }
        )
        if manifest.role == "pilot":
            dataset_config.update(
                {
                    "pilot_manifest": str(manifest.path),
                    "pilot_name": manifest.name,
                    "pilot_manifest_hash": manifest.manifest_hash,
                    "pilot_example_count": len(manifest.records),
                }
            )
    research = {
        "run_id": run_id,
        "experiment_type": config.experiment_type,
        "model": model_config.model,
        "ollama_model": model_config.model,
        "provider": model_config.provider,
        "model_configuration": {
            "generation": model_config.generation.snapshot(),
            "runtime": asdict(model_config.runtime),
            "source_path": str(model_config.source_path) if model_config.source_path else None,
        },
        "thinking_mode": model_config.generation.think,
        "benchmark_strategy": config.strategy,
        "prompt_version": strategy.prompt_version,
        "prompt_template_hash": _sha256_bytes(strategy.template_path.read_bytes()),
        "temperature": model_config.generation.temperature,
        "seed": model_config.generation.seed,
        "context_configuration": {
            "num_ctx": model_config.generation.num_ctx,
            "num_predict": model_config.generation.num_predict,
        },
        "dataset": dataset_config,
    }
    if config.provenance is not None:
        research["pilot_provenance"] = dict(config.provenance)
    return research


def _initial_metadata(
    *,
    config: Mapping[str, Any],
    fingerprint: str,
    expected_examples: int,
    start_time: datetime,
    model_config: ModelConfig,
) -> dict[str, Any]:
    return {
        **config,
        "configuration_fingerprint": fingerprint,
        "git": _git_metadata(),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "dependencies": _dependency_versions(("nltk", "pyarrow")),
        "ollama_version": _ollama_version(model_config),
        "start_timestamp": _isoformat(start_time),
        "end_timestamp": None,
        "expected_example_count": expected_examples,
        "completed_example_count": 0,
        "failed_example_count": 0,
    }


def _record_base(
    *,
    example: FlawedFictionsExample,
    config: RunConfig,
    run_id: str,
    model_config: ModelConfig,
    prompt_build: Any,
    stage_id: str,
    input_hash: str,
    cache_key: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_type": config.experiment_type,
        "subset": example.split.value,
        "example_id": example.example_id,
        "model": model_config.model,
        "ollama_model": model_config.model,
        "provider": model_config.provider,
        "benchmark_strategy": config.strategy,
        "prompt_version": prompt_build.prompt_version,
        "prompt_template_hash": prompt_build.template_hash,
        "prompt_hash": prompt_build.prompt_hash,
        "input_hash": input_hash,
        "stage_id": stage_id,
        "cache_key": cache_key,
        "thinking_mode": model_config.generation.think,
        "model_configuration": {
            "generation": model_config.generation.snapshot(),
            "runtime": asdict(model_config.runtime),
        },
    }


def _cache_key_material(
    example: FlawedFictionsExample,
    config: RunConfig,
    model_config: ModelConfig,
    prompt_build: Any,
    stage: Any,
) -> dict[str, Any]:
    return {
        "model": model_config.model,
        "provider": model_config.provider,
        "model_configuration": {
            "generation": model_config.generation.snapshot(),
            "runtime": asdict(model_config.runtime),
        },
        "thinking": model_config.generation.think,
        "prompt_strategy": config.strategy,
        "prompt_version": prompt_build.prompt_version,
        "prompt_template_hash": prompt_build.template_hash,
        "prompt_hash": prompt_build.prompt_hash,
        "input_hash": _sha256_text(example.story),
        "stage_id": stage.name,
    }


def _load_checkpoint_keys(run_dir: Path) -> tuple[set[str], set[str]]:
    completed: set[str] = set()
    failed: set[str] = set()
    for record in _iter_jsonl(run_dir / "predictions.jsonl"):
        cache_key = record.get("cache_key")
        if not isinstance(cache_key, str):
            continue
        if record.get("target_inference_failure"):
            failed.add(cache_key)
        else:
            completed.add(cache_key)
    return completed, failed


def _assert_compatible_resume(run_dir: Path, fingerprint: str) -> None:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise ExperimentRunError(f"Cannot resume {run_dir}: missing metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    old = metadata.get("configuration_fingerprint")
    if old != fingerprint:
        raise ExperimentRunError(
            "Refusing to resume run with incompatible configuration "
            f"(existing={old}, requested={fingerprint})"
        )


def _refresh_metadata(run_dir: Path, timestamp: datetime, final: bool = False) -> None:
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    completed, failed = _count_prediction_status(run_dir)
    metadata["completed_example_count"] = completed
    metadata["failed_example_count"] = failed
    if final:
        metadata["end_timestamp"] = _isoformat(timestamp)
    _write_metadata(run_dir, metadata)


def _count_prediction_status(run_dir: Path) -> tuple[int, int]:
    completed = 0
    failed = 0
    seen: set[str] = set()
    for record in _iter_jsonl(run_dir / "predictions.jsonl"):
        cache_key = record.get("cache_key")
        if isinstance(cache_key, str):
            if cache_key in seen:
                raise ExperimentRunError(f"Duplicate prediction cache key in {run_dir}: {cache_key}")
            seen.add(cache_key)
        if record.get("target_inference_failure"):
            failed += 1
        else:
            completed += 1
    return completed, failed


def _remove_prediction_records(path: Path, cache_keys: set[str]) -> None:
    retained = [record for record in _iter_jsonl(path) if record.get("cache_key") not in cache_keys]
    path.write_text("", encoding="utf-8")
    for record in retained:
        _append_jsonl(path, record)


def _write_config_yaml(path: Path, config: Mapping[str, Any]) -> None:
    path.write_text(_to_simple_yaml(config), encoding="utf-8")


def _write_metadata(run_dir: Path, metadata: Mapping[str, Any]) -> None:
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentRunError(f"{path}:{line_no}: invalid JSONL checkpoint") from exc
        if not isinstance(parsed, dict):
            raise ExperimentRunError(f"{path}:{line_no}: checkpoint record must be an object")
        records.append(parsed)
    return records


def _parsed_to_record(parsed: ParsedPrediction) -> dict[str, Any]:
    return {
        **parsed.to_official_result(),
        "parsing_failures": list(parsed.parsing_failures),
    }


def _compact_ollama_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw.items()
        if key not in {"response", "thinking", "context"}
    }


def _write_run_summary(
    run_dir: Path,
    completed: int,
    failed: int,
    expected: int,
    counters: Mapping[str, int],
) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics_note = "Metrics written." if metrics_path.exists() else "Metrics not written."
    artifact_label = "SMOKE TEST -- NOT A RESEARCH RESULT" if expected <= 5 else "Research run artifact."
    if _is_pilot_run(run_dir):
        artifact_label = f"PILOT RESULTS \u2014 N={expected}\n\nEXPLORATORY ONLY\n\nNOT FULL FLAWEDFICTIONS BENCHMARK RESULTS"
    lines = [
        "# Run Summary",
        "",
        artifact_label,
        "",
        f"Expected examples: {expected}",
        f"Completed predictions: {completed}",
        f"Inference failures: {failed}",
        f"Skipped completed on resume: {counters['skipped_completed']}",
        f"Skipped failed on resume: {counters['skipped_failed']}",
        "",
        metrics_note,
        "",
    ]
    (run_dir / "run_summary.md").write_text("\n".join(lines), encoding="utf-8")
    if not (run_dir / "summary.md").exists():
        (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _is_pilot_run(run_dir: Path) -> bool:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    dataset = metadata.get("dataset")
    return isinstance(dataset, Mapping) and (
        dataset.get("fixed_manifest_role") == "pilot" or bool(dataset.get("pilot_name"))
    )


def _git_metadata() -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    status = run_git(["status", "--porcelain"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "dirty": None if status is None else bool(status),
    }


def _dependency_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _ollama_version(model_config: ModelConfig) -> str | None:
    try:
        return OllamaClient(model_config).get_version(timeout_seconds=5)
    except OllamaError:
        return None


def _to_simple_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                lines.append(f"{prefix}{key}:")
                lines.append(_to_simple_yaml(item, indent + 2).rstrip())
            elif isinstance(item, list):
                lines.append(f"{prefix}{key}:")
                for element in item:
                    if isinstance(element, Mapping):
                        lines.append(f"{prefix}  -")
                        lines.append(_to_simple_yaml(element, indent + 4).rstrip())
                    else:
                        lines.append(f"{prefix}  - {_yaml_scalar(element)}")
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{_yaml_scalar(value)}\n"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(char in text for char in ":#[]{}&,*!|>'\"%@`"):
        return json.dumps(text)
    return text


def _fingerprint(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return _sha256_text(payload)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
