"""Ollama HTTP client for local model inference."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from socket import timeout as SocketTimeout
from typing import Any, Callable, Mapping

from fyp.models.base import ModelConfig, ModelRequest, ModelResponse


LOGGER = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Base error for Ollama integration failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the local Ollama server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""


class OllamaModelMissingError(OllamaError):
    """Raised when a requested Ollama model is not installed locally."""


class OllamaMalformedResponseError(OllamaError):
    """Raised when Ollama returns an unexpected response shape."""


class OllamaInterruptedError(OllamaError):
    """Raised when an Ollama request is interrupted."""


class OllamaClient:
    """Small, bounded client for Ollama's HTTP API."""

    def __init__(
        self,
        config: ModelConfig,
        host: str = "http://localhost:11434",
        opener: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        if config.provider != "ollama":
            raise ValueError(f"OllamaClient requires provider='ollama', got {config.provider!r}")
        self.config = config
        self.host = host.rstrip("/")
        self._opener = opener or urllib.request.build_opener()
        self._sleep = sleep
        self._logger = logger or LOGGER
        self.last_attempt_history: list[dict[str, Any]] = []
        self.last_raw_response: dict[str, Any] | None = None

    def generate(self, request: ModelRequest | str) -> ModelResponse:
        """Generate one non-streaming response using `/api/generate`."""

        model_request = request if isinstance(request, ModelRequest) else ModelRequest(prompt=request)
        self.last_attempt_history = []
        self.last_raw_response = None
        payload = self.build_generate_payload(model_request)
        raw = self._request_with_retries("/api/generate", payload)
        return self.parse_generate_response(raw)

    def build_generate_payload(self, request: ModelRequest) -> dict[str, Any]:
        """Construct an Ollama request without embedding long prompt logs."""

        return {
            **self.config.effective_ollama_request_fields(),
            "prompt": request.prompt,
        }

    def parse_generate_response(self, raw: Mapping[str, Any]) -> ModelResponse:
        """Validate Ollama response shape and preserve thinking separately."""

        if not isinstance(raw, Mapping):
            raise OllamaMalformedResponseError("Ollama response was not a JSON object")
        response = raw.get("response")
        thinking = raw.get("thinking")
        if not isinstance(response, str):
            raise OllamaMalformedResponseError("Ollama response missing string field 'response'")
        if thinking is not None and not isinstance(thinking, str):
            raise OllamaMalformedResponseError("Ollama response field 'thinking' must be a string")
        done = raw.get("done")
        if done is not True:
            raise OllamaMalformedResponseError("Ollama non-streaming response did not report done=true")
        return ModelResponse(
            model=str(raw.get("model") or self.config.model),
            response=response,
            thinking=thinking,
            raw=dict(raw),
            effective_config=self.config.effective_ollama_request_fields(),
        )

    def get_version(self, timeout_seconds: int = 5) -> str:
        """Return Ollama server version from `/api/version`."""

        raw = self._get_json("/api/version", timeout_seconds=timeout_seconds)
        version = raw.get("version")
        if not isinstance(version, str):
            raise OllamaMalformedResponseError("Ollama version response missing string field 'version'")
        return version

    def list_models(self, timeout_seconds: int = 5) -> list[str]:
        """Return locally installed Ollama model names from `/api/tags`."""

        raw = self._get_json("/api/tags", timeout_seconds=timeout_seconds)
        models = raw.get("models")
        if not isinstance(models, list):
            raise OllamaMalformedResponseError("Ollama tags response missing list field 'models'")
        names = []
        for item in models:
            if not isinstance(item, Mapping):
                raise OllamaMalformedResponseError("Ollama model entry was not a JSON object")
            name = item.get("name") or item.get("model")
            if not isinstance(name, str):
                raise OllamaMalformedResponseError("Ollama model entry missing name/model string")
            names.append(name)
        return sorted(names)

    def list_model_details(self, timeout_seconds: int = 5) -> list[dict[str, Any]]:
        """Inspect installed identities without loading models or generating text."""

        raw = self._get_json("/api/tags", timeout_seconds=timeout_seconds)
        models = raw.get("models")
        if not isinstance(models, list) or not all(isinstance(item, dict) for item in models):
            raise OllamaMalformedResponseError("Ollama tags response missing model objects")
        return models

    def show_model(self, timeout_seconds: int = 5) -> dict[str, Any]:
        """Read the installed model's template, capabilities and GGUF metadata."""

        return self._post_json(
            "/api/show", {"model": self.config.model}, timeout_seconds=timeout_seconds
        )

    def _request_with_retries(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        attempts = max(1, self.config.runtime.max_retries + 1)
        last_error: OllamaError | None = None
        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                raw = self._post_json(
                    endpoint,
                    payload,
                    timeout_seconds=self.config.runtime.timeout_seconds,
                )
                self.last_raw_response = raw
                self._record_attempt(attempt, timestamp, started)
                return raw
            except KeyboardInterrupt as exc:
                self._record_attempt(attempt, timestamp, started, exc)
                raise OllamaInterruptedError("Ollama request interrupted") from exc
            except OllamaModelMissingError as exc:
                self._record_attempt(attempt, timestamp, started, exc)
                raise
            except (OllamaConnectionError, OllamaTimeoutError) as exc:
                self._record_attempt(attempt, timestamp, started, exc)
                last_error = exc
                if attempt >= attempts:
                    raise
                self._logger.warning(
                    "Ollama request failed on attempt %s/%s: %s",
                    attempt,
                    attempts,
                    exc,
                )
                self._sleep(min(2 ** (attempt - 1), 5))
            except OllamaError as exc:
                self._record_attempt(attempt, timestamp, started, exc)
                raise
        if last_error is not None:
            raise last_error
        raise OllamaError("Ollama request failed without a captured error")

    def _record_attempt(
        self, attempt: int, timestamp: str, started: float, error: BaseException | None = None
    ) -> None:
        self.last_attempt_history.append({
            "attempt": attempt,
            "started_at": timestamp,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "status": "response_received" if error is None else "error",
            "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
        })

    def _post_json(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        timeout_seconds: int,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            self.host + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._open_json(request, timeout_seconds=timeout_seconds)

    def _get_json(self, endpoint: str, timeout_seconds: int) -> dict[str, Any]:
        request = urllib.request.Request(self.host + endpoint, method="GET")
        return self._open_json(request, timeout_seconds=timeout_seconds)

    def _open_json(self, request: urllib.request.Request, timeout_seconds: int) -> dict[str, Any]:
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = _read_error_detail(exc)
            if exc.code == 404 and "model" in detail.lower():
                raise OllamaModelMissingError(detail or f"Model not found: {self.config.model}") from exc
            raise OllamaConnectionError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, SocketTimeout) as exc:
            raise OllamaTimeoutError(str(exc)) from exc
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            raise OllamaConnectionError(str(exc)) from exc

        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaMalformedResponseError("Ollama response was not valid JSON") from exc
        if not isinstance(parsed, dict):
            raise OllamaMalformedResponseError("Ollama response JSON was not an object")
        return parsed


def _read_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload.strip()
    error = parsed.get("error") if isinstance(parsed, dict) else None
    return error if isinstance(error, str) else payload.strip()
