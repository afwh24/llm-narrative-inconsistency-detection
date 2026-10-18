"""Durable application-level request logging with bounded resume-aware attempts."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from fyp.models.ollama_client import (
    OllamaClient, OllamaConnectionError, OllamaError, OllamaInterruptedError,
    OllamaMalformedResponseError, OllamaModelMissingError, OllamaTimeoutError,
)


class OllamaPendingError(OllamaInterruptedError):
    """Work is pending and cannot safely consume another inference attempt."""


class JournaledOllamaClient(OllamaClient):
    def __init__(self, config, journal_path: Path, *, legacy_run: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.journal_path = journal_path
        self.legacy_run = legacy_run
        self.context: dict[str, Any] = {}

    def set_context(self, context: Mapping[str, Any]) -> None:
        self.context = dict(context)

    def _append_event(self, event: Mapping[str, Any]) -> None:
        try:
            with self.journal_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise OllamaPendingError("Cannot durably save request journal; story remains pending") from exc

    def _events(self) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        try:
            events = [json.loads(line) for line in self.journal_path.read_text(encoding="utf-8").splitlines() if line]
        except (ValueError, OSError) as exc:
            raise OllamaPendingError("Request journal is incomplete/unreadable; pending work was not failed or retried") from exc
        return [event for event in events if event.get("cache_key") == self.context.get("cache_key")]

    @staticmethod
    def _error(event: dict[str, Any]) -> OllamaError:
        error = event["error"]
        types = {cls.__name__: cls for cls in (
            OllamaTimeoutError, OllamaConnectionError, OllamaModelMissingError, OllamaMalformedResponseError,
        )}
        return types.get(error["type"], OllamaError)(error["message"])

    def _request_with_retries(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.context.get("cache_key"):
            raise OllamaPendingError("Missing run/story context for request journal")
        request = urllib.request.Request(
            self.host + endpoint, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        body_hash = hashlib.sha256(request.data).hexdigest().upper()
        events = self._events()
        starts = {e["attempt"]: e for e in events if e["event"] == "started"}
        finishes = {e["attempt"]: e for e in events if e["event"] == "finished"}
        if any(e["application_request"]["body_hash"] != body_hash for e in starts.values()):
            raise OllamaPendingError("Prior assembled request differs; refusing to resume")
        if self.legacy_run and not starts:
            raise OllamaPendingError("Legacy pending work has no durable request/retry log; its budget cannot be reconstructed")
        if starts and sorted(starts) != list(range(1, max(starts) + 1)):
            raise OllamaPendingError("Request attempt numbers are inconsistent")
        for attempt, start in sorted(starts.items()):
            finish = finishes.get(attempt)
            self.last_attempt_history.append({
                "attempt": attempt, "started_at": start["started_at"],
                "latency_seconds": finish.get("latency_seconds") if finish else None,
                "status": finish["status"] if finish else "unfinished",
                "error": finish.get("error") if finish else None,
                "request_id": start["request_id"],
            })
        if finishes:
            last = finishes[max(finishes)]
            if last["status"] == "response_received":
                self.last_raw_response = last["raw_response"]
                return self.last_raw_response
            if last["status"] == "error":
                error = self._error(last)
                if not isinstance(error, (OllamaConnectionError, OllamaTimeoutError)):
                    raise error
        attempts = self.config.runtime.max_retries + 1
        used = max(starts, default=0)
        if used >= attempts:
            last = finishes.get(used)
            if last and last["status"] == "error":
                raise self._error(last)
            raise OllamaPendingError("Attempt budget consumed by unresolved/interrupted requests; no terminal model failure inferred")
        for attempt in range(used + 1, attempts + 1):
            timestamp = datetime.now(timezone.utc).isoformat()
            request_id = f"{self.context['run_id']}:{self.context['cache_key']}:{attempt}"
            self._append_event({
                **self.context, "event": "started", "attempt": attempt,
                "request_id": request_id, "started_at": timestamp,
                "application_request": {
                    "method": request.get_method(), "url": request.full_url,
                    "headers": dict(request.header_items()), "body": dict(payload),
                    "body_utf8": request.data.decode("utf-8"), "body_hash": body_hash,
                },
                "logging_scope": "Assembled application request before dispatch, not a wire/server receipt trace",
            })
            started = time.perf_counter()
            raw = None
            error = None
            status = "error"
            try:
                raw = self._open_json(request, timeout_seconds=self.config.runtime.timeout_seconds)
                self.last_raw_response = raw
                status = "response_received"
            except KeyboardInterrupt as exc:
                error = {"type": type(exc).__name__, "message": "Execution interrupted"}
                status = "interrupted"
            except OllamaError as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            latency = round(time.perf_counter() - started, 6)
            finish = {
                "event": "finished", "run_id": self.context["run_id"],
                "cache_key": self.context["cache_key"], "example_id": self.context["example_id"],
                "model": self.config.model, "strategy": self.context["benchmark_strategy"],
                "attempt": attempt, "request_id": request_id, "status": status,
                "latency_seconds": latency, "error": error, "raw_response": raw,
            }
            self._append_event(finish)
            self.last_attempt_history.append({
                "attempt": attempt, "started_at": timestamp, "latency_seconds": latency,
                "status": status, "error": error, "request_id": request_id,
            })
            if status == "response_received":
                return raw
            if status == "interrupted":
                raise OllamaInterruptedError("Execution interrupted; story remains pending")
            exc = self._error(finish)
            if not isinstance(exc, (OllamaConnectionError, OllamaTimeoutError)) or attempt >= attempts:
                raise exc
            self._logger.warning("Ollama request failed on attempt %s/%s: %s", attempt, attempts, exc)
            self._sleep(min(2 ** (attempt - 1), 5))
        raise OllamaPendingError("No safe request attempt is available")
