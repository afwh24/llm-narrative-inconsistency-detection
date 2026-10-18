"""Environment and local runtime diagnostics for the FYP project."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fyp.data.loader import DEFAULT_RAW_DATA_DIR, load_all_splits
from fyp.data.schema import SPLIT_SPECS
from fyp.data.validation import ValidationError, validate_dataset
from fyp.models.base import ModelConfig
from fyp.models.ollama_client import OllamaClient, OllamaError
from fyp.models.registry import TARGET_MODELS, get_model_config


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str


def collect_doctor_results(project_root: Path | None = None) -> list[CheckResult]:
    """Collect concise diagnostics without running model inference."""

    root = project_root or Path.cwd()
    results: list[CheckResult] = []
    executable = Path(sys.executable)
    expected_python = root / ".venv" / "Scripts" / "python.exe"
    in_expected_venv = _same_path(executable, expected_python)

    results.append(CheckResult("Python executable", "OK", str(executable)))
    results.append(CheckResult("Python version", "OK", platform.python_version()))
    results.append(
        CheckResult(
            "Expected .venv",
            "OK" if in_expected_venv else "WARN",
            "running inside project .venv" if in_expected_venv else f"expected {expected_python}",
        )
    )
    results.append(_project_import_check())
    results.extend(_dependency_checks(("pyarrow",)))
    results.extend(_dataset_checks())

    config_results, configs = _model_config_checks()
    results.extend(config_results)
    results.extend(_ollama_checks(configs))
    return results


def print_doctor_report(results: Iterable[CheckResult]) -> None:
    """Print a compact human-readable doctor report."""

    print("FYP doctor")
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")


def doctor_has_critical_failure(results: Iterable[CheckResult]) -> bool:
    """Return whether non-Ollama project checks failed."""

    critical_prefixes = (
        "Project import",
        "Dependency",
        "Dataset availability",
        "Dataset validation",
        "Model config",
    )
    return any(
        result.status == "FAIL" and result.name.startswith(critical_prefixes)
        for result in results
    )


def _project_import_check() -> CheckResult:
    try:
        import fyp  # noqa: F401
    except Exception as exc:
        return CheckResult("Project import", "FAIL", str(exc))
    return CheckResult("Project import", "OK", "fyp import succeeded")


def _dependency_checks(modules: Iterable[str]) -> list[CheckResult]:
    results = []
    for module in modules:
        status = "OK" if importlib.util.find_spec(module) is not None else "FAIL"
        detail = "available" if status == "OK" else "not importable"
        results.append(CheckResult(f"Dependency {module}", status, detail))
    return results


def _dataset_checks() -> list[CheckResult]:
    missing = [
        spec.filename
        for spec in SPLIT_SPECS.values()
        if not (DEFAULT_RAW_DATA_DIR / spec.filename).exists()
    ]
    if missing:
        return [
            CheckResult(
                "Dataset availability",
                "FAIL",
                f"missing raw files: {', '.join(missing)}",
            )
        ]
    results = [
        CheckResult(
            "Dataset availability",
            "OK",
            f"{len(SPLIT_SPECS)} official raw split files present",
        )
    ]
    try:
        report = validate_dataset(load_all_splits())
    except ValidationError as exc:
        results.append(CheckResult("Dataset validation", "FAIL", str(exc).splitlines()[0]))
    except Exception as exc:
        results.append(CheckResult("Dataset validation", "FAIL", str(exc)))
    else:
        results.append(
            CheckResult(
                "Dataset validation",
                "OK",
                f"{report.total_examples} examples validated",
            )
        )
    return results


def _model_config_checks() -> tuple[list[CheckResult], list[ModelConfig]]:
    results: list[CheckResult] = []
    configs: list[ModelConfig] = []
    for model in TARGET_MODELS:
        try:
            config = get_model_config(model)
        except Exception as exc:
            results.append(CheckResult(f"Model config {model}", "FAIL", str(exc)))
            continue
        configs.append(config)
        effective = config.effective_ollama_request_fields()
        options = ", ".join(sorted(effective["options"])) or "none"
        results.append(
            CheckResult(
                f"Model config {model}",
                "OK",
                f"think={effective['think']!r}; options={options}; timeout={config.runtime.timeout_seconds}s; max_retries={config.runtime.max_retries}",
            )
        )
    return results, configs


def _ollama_checks(configs: list[ModelConfig]) -> list[CheckResult]:
    results: list[CheckResult] = []
    cli_path = shutil.which("ollama")
    if cli_path is None:
        results.append(CheckResult("Ollama CLI", "WARN", "not found on PATH"))
    else:
        results.append(CheckResult("Ollama CLI", "OK", cli_path))
        results.append(_ollama_cli_version(cli_path))

    if not configs:
        results.append(CheckResult("Ollama connectivity", "WARN", "not checked because model configs are invalid"))
        return results

    diagnostic_config = configs[0]
    client = OllamaClient(diagnostic_config)
    try:
        server_version = client.get_version(timeout_seconds=5)
    except OllamaError as exc:
        results.append(CheckResult("Ollama connectivity", "WARN", str(exc)))
        results.append(CheckResult("Ollama models", "WARN", "not checked because server is unavailable"))
        for model in TARGET_MODELS:
            results.append(
                CheckResult(
                    f"Model availability {model}",
                    "WARN",
                    f"not checked; if missing run: ollama pull {model}",
                )
            )
        return results

    results.append(CheckResult("Ollama connectivity", "OK", "server reachable"))
    results.append(CheckResult("Ollama server version", "OK", server_version))
    try:
        models = client.list_models(timeout_seconds=5)
    except OllamaError as exc:
        results.append(CheckResult("Ollama models", "WARN", str(exc)))
        models = []
    else:
        detail = ", ".join(models) if models else "no local models reported"
        results.append(CheckResult("Ollama models", "OK", detail))

    model_set = set(models)
    for model in TARGET_MODELS:
        available = model in model_set
        results.append(
            CheckResult(
                f"Model availability {model}",
                "OK" if available else "WARN",
                "available" if available else f"missing; run: ollama pull {model}",
            )
        )
    return results


def _ollama_cli_version(cli_path: str) -> CheckResult:
    try:
        completed = subprocess.run(
            [cli_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return CheckResult("Ollama CLI version", "WARN", str(exc))
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return CheckResult("Ollama CLI version", "WARN", output or "version command failed")
    return CheckResult("Ollama CLI version", "OK", output)


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False
