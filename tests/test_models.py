from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from urllib.request import Request

import pytest

from fyp.models.base import GenerationConfig, ModelConfig, ModelRequest, RuntimeConfig
from fyp.models.ollama_client import (
    OllamaClient,
    OllamaInterruptedError,
    OllamaMalformedResponseError,
    OllamaModelMissingError,
    OllamaTimeoutError,
)
from fyp.models.registry import get_model_config, load_model_config


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.timeouts = []

    def open(self, request: Request, timeout: int):
        self.requests.append(request)
        self.timeouts.append(timeout)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeHTTPResponse(outcome)


def _config(
    model: str = "qwen3:4b",
    think: bool | str = False,
    max_retries: int = 0,
) -> ModelConfig:
    return ModelConfig(
        provider="ollama",
        model=model,
        generation=GenerationConfig(
            temperature=0.5,
            seed=42,
            think=think,
            num_ctx=None,
            num_predict=4096,
        ),
        runtime=RuntimeConfig(timeout_seconds=11, max_retries=max_retries),
    )


def _request_body(request: Request) -> dict:
    return json.loads(request.data.decode("utf-8"))


def test_configuration_loading_for_qwen3_4b() -> None:
    config = get_model_config("qwen3:4b")

    assert config.provider == "ollama"
    assert config.model == "qwen3:4b"
    assert config.generation.think is False
    assert config.generation.ollama_options() == {
        "temperature": 0.5,
        "seed": 42,
        "num_predict": 4096,
    }


def test_configuration_loading_for_qwen3_8b() -> None:
    config = get_model_config("qwen3:8b")

    assert config.provider == "ollama"
    assert config.model == "qwen3:8b"
    assert config.generation.think is False


def test_request_construction_uses_top_level_think() -> None:
    client = OllamaClient(_config(think=True))

    payload = client.build_generate_payload(ModelRequest(prompt="hello"))

    assert payload["model"] == "qwen3:4b"
    assert payload["prompt"] == "hello"
    assert payload["stream"] is False
    assert payload["think"] is True
    assert payload["options"] == {"temperature": 0.5, "seed": 42, "num_predict": 4096}
    assert "think" not in payload["options"]


@pytest.mark.parametrize("model,filename", [
    ("qwen3:8b", "qwen3_8b.yaml"),
    ("llama3.1:8b", "llama3_1_8b.yaml"),
    ("mistral:7b", "mistral_7b.yaml"),
])
def test_cross_model_pilot_uses_shared_options_and_supported_thinking(model, filename) -> None:
    config = load_model_config(Path("configs/models/three_model_main10") / filename)
    payload = OllamaClient(config).build_generate_payload(ModelRequest(prompt="unchanged story"))
    assert payload["model"] == model
    assert payload["options"] == {
        "temperature": 0.5, "seed": 42, "num_predict": 4096, "num_ctx": 16384,
        "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.0,
    }
    if model == "qwen3:8b":
        assert payload["think"] is False
    else:
        assert "think" not in payload


def test_legacy_generation_snapshot_is_unchanged() -> None:
    assert get_model_config("qwen3:8b").generation.snapshot() == {
        "temperature": 0.5, "seed": 42, "think": False,
        "num_ctx": None, "num_predict": 4096,
    }


def test_three_retries_means_four_attempts() -> None:
    opener = FakeOpener([TimeoutError("slow")] * 4)
    client = OllamaClient(_config(max_retries=3), opener=opener, sleep=lambda _: None)
    with pytest.raises(OllamaTimeoutError):
        client.generate("prompt")
    assert len(opener.requests) == 4
    assert [a["attempt"] for a in client.last_attempt_history] == [1, 2, 3, 4]
    assert all(a["status"] == "error" for a in client.last_attempt_history)
    assert all(a["error"]["type"] == "OllamaTimeoutError" for a in client.last_attempt_history)


def test_ordinary_response_handling() -> None:
    opener = FakeOpener(
        [
            {
                "model": "qwen3:4b",
                "response": "final answer",
                "done": True,
            }
        ]
    )
    client = OllamaClient(_config(), opener=opener)

    response = client.generate("prompt")

    assert response.response == "final answer"
    assert response.thinking is None
    assert response.model == "qwen3:4b"


def test_thinking_response_is_preserved_separately() -> None:
    opener = FakeOpener(
        [
            {
                "model": "qwen3:4b",
                "thinking": "private reasoning",
                "response": "final answer",
                "done": True,
            }
        ]
    )
    client = OllamaClient(_config(think=True), opener=opener)

    response = client.generate("prompt")

    assert response.thinking == "private reasoning"
    assert response.response == "final answer"


def test_missing_model_error_reports_manual_pull_case() -> None:
    error_payload = json.dumps({"error": "model 'qwen3:4b' not found"}).encode("utf-8")
    http_error = urllib.error.HTTPError(
        url="http://localhost:11434/api/generate",
        code=404,
        msg="not found",
        hdrs={},
        fp=io.BytesIO(error_payload),
    )
    opener = FakeOpener([http_error])
    client = OllamaClient(_config(max_retries=2), opener=opener)

    with pytest.raises(OllamaModelMissingError, match="qwen3:4b"):
        client.generate("prompt")

    assert len(opener.requests) == 1


def test_timeout_is_bounded_and_retried() -> None:
    opener = FakeOpener(
        [
            TimeoutError("slow"),
            {
                "model": "qwen3:4b",
                "response": "ok",
                "done": True,
            },
        ]
    )
    client = OllamaClient(_config(max_retries=1), opener=opener, sleep=lambda _: None)

    response = client.generate("prompt")

    assert response.response == "ok"
    assert len(opener.requests) == 2
    assert [a["status"] for a in client.last_attempt_history] == ["error", "response_received"]


def test_timeout_failure_after_retry_budget() -> None:
    opener = FakeOpener([TimeoutError("slow"), TimeoutError("still slow")])
    client = OllamaClient(_config(max_retries=1), opener=opener, sleep=lambda _: None)

    with pytest.raises(OllamaTimeoutError):
        client.generate("prompt")

    assert len(opener.requests) == 2


def test_interrupted_request_is_not_retried() -> None:
    opener = FakeOpener([KeyboardInterrupt()])
    client = OllamaClient(_config(max_retries=3), opener=opener, sleep=lambda _: None)

    with pytest.raises(OllamaInterruptedError):
        client.generate("prompt")

    assert len(opener.requests) == 1


def test_malformed_ollama_response_is_rejected() -> None:
    opener = FakeOpener([{"model": "qwen3:4b", "done": True}])
    client = OllamaClient(_config(), opener=opener)

    with pytest.raises(OllamaMalformedResponseError, match="response"):
        client.generate("prompt")


def test_config_rejects_think_inside_options_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "\n".join(
            [
                "provider: ollama",
                "model: qwen3:4b",
                "generation:",
                "  temperature: 0.0",
                "  seed: 42",
                "  think: false",
                "runtime:",
                "  timeout_seconds: 300",
                "  max_retries: 3",
            ]
        ),
        encoding="utf-8",
    )

    config = load_model_config(path)

    assert config.effective_ollama_request_fields()["think"] is False
    assert "think" not in config.effective_ollama_request_fields()["options"]
