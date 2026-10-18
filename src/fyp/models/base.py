"""Shared model abstractions independent of any provider SDK."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class GenerationConfig:
    """Generation settings supported by the local model abstraction."""

    temperature: float | None = 0.0
    seed: int | None = 42
    think: bool | str | None = False
    num_ctx: int | None = None
    num_predict: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None

    def snapshot(self) -> dict[str, Any]:
        """Keep legacy fingerprints stable when optional sampling controls are unset."""

        return {
            key: value
            for key, value in asdict(self).items()
            if key not in {"top_k", "top_p", "repeat_penalty"} or value is not None
        }

    def ollama_options(self) -> dict[str, Any]:
        """Return settings Ollama accepts inside the `options` object."""

        options: dict[str, Any] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.seed is not None:
            options["seed"] = self.seed
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        for name in ("top_k", "top_p", "repeat_penalty"):
            value = getattr(self, name)
            if value is not None:
                options[name] = value
        return options


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime controls for bounded local model calls."""

    timeout_seconds: int = 300
    max_retries: int = 3


@dataclass(frozen=True)
class ModelConfig:
    """Validated model configuration loaded from `configs/models/`."""

    provider: str
    model: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    source_path: Path | None = None

    def effective_ollama_request_fields(self) -> dict[str, Any]:
        """Document fields enforced in Ollama requests."""

        fields = {
            "model": self.model,
            "stream": False,
            "options": self.generation.ollama_options(),
        }
        if self.generation.think is not None:
            fields["think"] = self.generation.think
        return fields


@dataclass(frozen=True)
class ModelRequest:
    """Provider-neutral request object for one model completion."""

    prompt: str


@dataclass(frozen=True)
class ModelResponse:
    """Provider-neutral response with thinking separated from final text."""

    model: str
    response: str
    thinking: str | None
    raw: Mapping[str, Any]
    effective_config: Mapping[str, Any]


class ModelClient(Protocol):
    """Minimal interface future pipeline code should depend on."""

    config: ModelConfig

    def generate(self, request: ModelRequest | str) -> ModelResponse:
        """Generate one response."""
