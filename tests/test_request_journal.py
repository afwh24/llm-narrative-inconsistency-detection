from __future__ import annotations

import hashlib
import json

import pytest

from fyp.models.ollama_client import OllamaInterruptedError, OllamaTimeoutError
from fyp.models.registry import load_model_config
from fyp.models.request_journal import JournaledOllamaClient, OllamaPendingError
from test_models import FakeOpener


CONTEXT = {"run_id": "synthetic-run", "cache_key": "synthetic-key", "example_id": "synthetic-story",
           "model": "qwen3:8b", "benchmark_strategy": "evidence_first",
           "prompt_version": "evidence_first_v2", "prompt_template_hash": "synthetic-template-hash",
           "prompt_template": "synthetic template"}
RESPONSE = {"model": "qwen3:8b", "response": "synthetic response", "done": True}


def client(path, opener, **kwargs):
    result = JournaledOllamaClient(
        load_model_config("configs/models/three_model_main10/qwen3_8b.yaml"), path,
        opener=opener, sleep=kwargs.pop("sleep", lambda _: None), **kwargs,
    )
    result.set_context(CONTEXT)
    return result


def events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_actual_assembled_request_is_logged_before_every_attempt(tmp_path):
    path = tmp_path / "requests.jsonl"
    opener = FakeOpener([TimeoutError("synthetic timeout"), RESPONSE])
    response = client(path, opener).generate("synthetic prompt")
    assert response.response == RESPONSE["response"]
    rows = events(path)
    assert [row["event"] for row in rows] == ["started", "finished", "started", "finished"]
    for start, request in zip(rows[::2], opener.requests):
        assert start["application_request"]["body_utf8"] == request.data.decode("utf-8")
        assert start["application_request"]["body_hash"] == hashlib.sha256(request.data).hexdigest().upper()
        assert start["application_request"]["body"] == json.loads(request.data)
        assert start["prompt_template"] == CONTEXT["prompt_template"]
        assert start["application_request"]["body"]["think"] is False


def test_interrupted_story_remains_pending_and_resume_does_not_reset_budget(tmp_path):
    path = tmp_path / "requests.jsonl"
    first = FakeOpener([KeyboardInterrupt()])
    with pytest.raises(OllamaInterruptedError):
        client(path, first).generate("synthetic prompt")
    second = FakeOpener([TimeoutError("synthetic timeout")] * 3)
    resumed = client(path, second)
    with pytest.raises(OllamaTimeoutError):
        resumed.generate("synthetic prompt")
    assert len(first.requests) == 1 and len(second.requests) == 3
    assert [row["attempt"] for row in events(path) if row["event"] == "started"] == [1, 2, 3, 4]
    exhausted = FakeOpener([])
    with pytest.raises(OllamaTimeoutError):
        client(path, exhausted).generate("synthetic prompt")
    assert exhausted.requests == []


def test_received_response_recovers_without_duplicate_generation(tmp_path):
    path = tmp_path / "requests.jsonl"
    client(path, FakeOpener([RESPONSE])).generate("synthetic prompt")
    opener = FakeOpener([])
    resumed = client(path, opener)
    assert resumed.generate("synthetic prompt").response == RESPONSE["response"]
    assert opener.requests == [] and len(resumed.last_attempt_history) == 1


def test_unresolved_exhausted_budget_is_pending_not_a_model_failure(tmp_path):
    path = tmp_path / "requests.jsonl"
    for attempt in range(1, 5):
        with pytest.raises(OllamaInterruptedError):
            client(path, FakeOpener([KeyboardInterrupt()])).generate("synthetic prompt")
    with pytest.raises(OllamaPendingError, match="unresolved/interrupted"):
        client(path, FakeOpener([])).generate("synthetic prompt")


def test_legacy_pending_budget_is_not_invented(tmp_path):
    opener = FakeOpener([])
    with pytest.raises(OllamaPendingError, match="cannot be reconstructed"):
        client(tmp_path / "requests.jsonl", opener, legacy_run=True).generate("synthetic prompt")
    assert opener.requests == []
