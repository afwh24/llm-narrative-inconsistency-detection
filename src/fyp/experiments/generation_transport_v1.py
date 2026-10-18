"""Opt-in transport corrections for a future, separately frozen generation run.

Reuses CapturingClient and its durable retry loop. No runner, prompt, sampling
override or historical migration is provided here. Existing journals are refused.
"""
import io
import os
import urllib.error

from fyp.experiments.generated_stories import CapturingClient, save
from fyp.models.ollama_client import (
    OllamaConnectionError, OllamaError, OllamaTimeoutError,
)
from fyp.models.request_journal import OllamaPendingError


class OllamaHTTPResponseError(OllamaError):
    """Server returned an HTTP error; not a connection failure or transport retry."""


class OllamaRepetitionAbortError(OllamaHTTPResponseError):
    """Server explicitly aborted generation at its token-repeat limit."""


class GenerationTransportClient(CapturingClient):
    """Inject through execute_stage(client_factory=...) in a new frozen procedure."""
    policy = 'generation_transport_v1'

    def _events(self):
        events = super()._events()
        if any(e.get('client_policy') != self.policy for e in events):
            raise OllamaPendingError('Historical journal policy differs; preserve it and do not migrate retry budgets')
        return events

    @staticmethod
    def _error(event):
        error = event['error']
        types = {cls.__name__: cls for cls in (OllamaHTTPResponseError, OllamaRepetitionAbortError, OllamaPendingError)}
        if error['type'] in types:
            return types[error['type']](error['message'])
        return CapturingClient._error(event)

    def _append_event(self, event):
        super()._append_event({**event, 'client_policy': self.policy})
        if event['event'] == 'finished':
            error = event.get('error')
            log = self._logger.warning if error else self._logger.info
            log(
                'Generation run=%s model=%s pair=%s stage=%s attempt=%s/%s '
                'outcome=%s error_type=%s detail=%s request_id=%s',
                self.context['run_id'], self.config.model, self.context['pair_id'],
                self.context['stage'], event['attempt'], self.config.runtime.max_retries + 1,
                event['status'], error['type'] if error else None,
                error['message'] if error else 'Response received; stage validation still required',
                event['request_id'],
            )

    def _open_json(self, request, timeout_seconds):
        delegate = self._opener
        client = self

        class ErrorCapturingOpener:
            def open(self, req, timeout):
                try:
                    return delegate.open(req, timeout=timeout)
                except urllib.error.HTTPError as exc:
                    try:
                        body = exc.read()
                    except OSError as read_error:
                        # Do not disguise an unreadable server error as a network retry.
                        raise OllamaHTTPResponseError(
                            f'Ollama HTTP {exc.code}; error body unavailable: {read_error}'
                        ) from exc
                    finally:
                        exc.close()
                    start = max((e for e in client._events() if e['event'] == 'started'), key=lambda e: e['attempt'])
                    attempt = start['attempt']
                    path = client.journal_path.parent/'raw_http'/(
                        f"{client.context['pair_id']}_{client.context['stage']}_{attempt}.error.bin")
                    try:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open('xb') as stream:
                            stream.write(body)
                            stream.flush()
                            os.fsync(stream.fileno())
                        save(path.with_suffix('.json'), {
                            'http_status': exc.code, 'body_file': path.name,
                            'run_id': client.context['run_id'], 'pair_id': client.context['pair_id'],
                            'stage': client.context['stage'], 'attempt': attempt,
                            'request_id': start['request_id'],
                        })
                    except OSError as persist_error:
                        raise OllamaPendingError('Cannot preserve HTTP error body; keep attempt pending for investigation') from persist_error
                    # The existing client extracts detail and maintains missing-model semantics.
                    raise urllib.error.HTTPError(exc.url, exc.code, exc.msg, exc.headers, io.BytesIO(body)) from exc

        self._opener = ErrorCapturingOpener()
        try:
            return super()._open_json(request, timeout_seconds)
        except OllamaConnectionError as exc:
            cause = exc.__cause__
            if isinstance(cause, urllib.error.HTTPError):
                error_type = (OllamaRepetitionAbortError
                              if cause.code == 500 and 'prediction aborted, token repeat limit reached' in str(exc).lower()
                              else OllamaHTTPResponseError)
                raise error_type(str(exc)) from exc
            raise
        except OllamaTimeoutError as exc:
            raise OllamaTimeoutError(
                f'Client deadline {timeout_seconds}s exceeded: {exc}; '
                'no complete response received; server completion/cancellation is unknown'
            ) from exc
        finally:
            self._opener = delegate
