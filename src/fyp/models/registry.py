"""Model configuration loading and registry helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fyp.models.base import GenerationConfig, ModelConfig, RuntimeConfig
from fyp.models.ollama_client import OllamaClient


DEFAULT_MODEL_CONFIG_DIR = Path("configs/models")
TARGET_MODELS = ("qwen3:4b", "qwen3:8b")
SUPPORTED_MODELS = (*TARGET_MODELS, "llama3.1:8b", "mistral:7b")
MODEL_CONFIG_FILES = {
    "qwen3:4b": "qwen3_4b.yaml",
    "qwen3:8b": "qwen3_8b.yaml",
    "qwen3_4b": "qwen3_4b.yaml",
    "qwen3_8b": "qwen3_8b.yaml",
}


class ModelConfigError(ValueError):
    """Raised when a model configuration is invalid."""


def get_model_config(
    name: str,
    config_dir: Path | str = DEFAULT_MODEL_CONFIG_DIR,
) -> ModelConfig:
    """Load a named model config from `configs/models/`."""

    filename = MODEL_CONFIG_FILES.get(name)
    if filename is None:
        allowed = ", ".join(sorted(MODEL_CONFIG_FILES))
        raise ModelConfigError(f"Unknown model config {name!r}; expected one of: {allowed}")
    return load_model_config(Path(config_dir) / filename)


def load_model_config(path: Path | str) -> ModelConfig:
    """Load and validate a simple YAML model config."""

    path = Path(path)
    raw = _parse_simple_yaml(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ModelConfigError(f"Model config {path} did not parse as a mapping")

    provider = _require_string(raw, "provider")
    model = _require_string(raw, "model")
    if provider != "ollama":
        raise ModelConfigError(f"{path}: unsupported provider {provider!r}")
    if model not in SUPPORTED_MODELS:
        raise ModelConfigError(f"{path}: unsupported target model {model!r}")

    generation_raw = _optional_mapping(raw, "generation")
    runtime_raw = _optional_mapping(raw, "runtime")
    generation = GenerationConfig(
        temperature=_optional_float(generation_raw, "temperature"),
        seed=_optional_int(generation_raw, "seed"),
        think=_optional_think(generation_raw, "think"),
        num_ctx=_optional_int(generation_raw, "num_ctx"),
        num_predict=_optional_int(generation_raw, "num_predict"),
        top_k=_optional_int(generation_raw, "top_k"),
        top_p=_optional_float(generation_raw, "top_p"),
        repeat_penalty=_optional_float(generation_raw, "repeat_penalty"),
    )
    if model in {"llama3.1:8b", "mistral:7b"} and generation.think is not None:
        raise ModelConfigError(f"{path}: thinking is not applicable; configure think: null")
    runtime = RuntimeConfig(
        timeout_seconds=_positive_int(runtime_raw, "timeout_seconds", default=300),
        max_retries=_nonnegative_int(runtime_raw, "max_retries", default=3),
    )
    return ModelConfig(
        provider=provider,
        model=model,
        generation=generation,
        runtime=runtime,
        source_path=path,
    )


def build_model_client(config: ModelConfig, host: str = "http://localhost:11434") -> OllamaClient:
    """Build the configured provider client."""

    if config.provider != "ollama":
        raise ModelConfigError(f"Unsupported provider {config.provider!r}")
    return OllamaClient(config=config, host=host)


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the limited YAML subset used by `configs/models/*.yaml`.

    Supported shape: top-level key/value pairs and one-level nested mappings
    indented by two spaces. This intentionally rejects lists and complex YAML.
    """

    root: dict[str, Any] = {}
    current_section: str | None = None
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ModelConfigError(f"Line {line_no}: expected key/value pair")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ModelConfigError(f"Line {line_no}: empty key")
        if indent == 0:
            if value == "":
                root[key] = {}
                current_section = key
            else:
                root[key] = _parse_scalar(value)
                current_section = None
        elif indent == 2 and current_section is not None:
            section = root[current_section]
            if not isinstance(section, dict):
                raise ModelConfigError(f"Line {line_no}: parent {current_section!r} is not a mapping")
            section[key] = _parse_scalar(value)
        else:
            raise ModelConfigError(f"Line {line_no}: unsupported indentation")
    return root


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _require_string(raw: Mapping[str, Any], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        raise ModelConfigError(f"Missing or invalid string field {field!r}")
    return value


def _optional_mapping(raw: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = raw.get(field, {})
    if not isinstance(value, Mapping):
        raise ModelConfigError(f"Field {field!r} must be a mapping")
    return value


def _optional_float(raw: Mapping[str, Any], field: str) -> float | None:
    value = raw.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelConfigError(f"Field {field!r} must be a number or null")
    return float(value)


def _optional_int(raw: Mapping[str, Any], field: str) -> int | None:
    value = raw.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelConfigError(f"Field {field!r} must be an integer or null")
    return value


def _optional_think(raw: Mapping[str, Any], field: str) -> bool | str | None:
    value = raw.get(field, False)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value in {"low", "medium", "high", "max"}:
        return value
    raise ModelConfigError(f"Field {field!r} must be null, a boolean or Ollama thinking level")


def _positive_int(raw: Mapping[str, Any], field: str, default: int) -> int:
    value = raw.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelConfigError(f"Field {field!r} must be a positive integer")
    return value


def _nonnegative_int(raw: Mapping[str, Any], field: str, default: int) -> int:
    value = raw.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelConfigError(f"Field {field!r} must be a non-negative integer")
    return value
