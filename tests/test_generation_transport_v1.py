import io
import json
import logging
import urllib.error
from dataclasses import replace

import pytest

from fyp.experiments import generated_stories as g
from fyp.experiments.generation_transport_v1 import GenerationTransportClient
from fyp.models.ollama_client import OllamaInterruptedError
from fyp.models.request_journal import OllamaPendingError
from test_models import FakeOpener


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('External network is forbidden in offline tests')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    settings = g.read(g.CONFIG)
    settings['root'] = str(tmp_path)
    freeze = {'inspection': {'models': {'qwen3:8b': {'identity': {'digest': 'synthetic'}}}}}
    monkeypatch.setattr(g, 'context_budget', lambda *args: {})
    return settings, freeze


def factory(opener):
    def build(config, path):
        config = replace(config, runtime=replace(config.runtime, max_retries=1))
        return GenerationTransportClient(config, path, opener=opener, sleep=lambda _: None)
    return build


def execute(setup, opener, stage='A', original=None):
    s,f = setup
    return g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001',stage,'synthetic '+stage,original,factory(opener))


def http_error(code=500, body=b'{"error":"prediction aborted, token repeat limit reached"}'):
    return urllib.error.HTTPError('http://localhost:11434/api/generate',code,'synthetic',{},io.BytesIO(body))


def response(obj):
    return {'response': json.dumps(obj), 'done': True, 'done_reason': 'stop'}


def test_repeat_abort_is_terminal_server_error_with_raw_capture(setup, caplog):
    caplog.set_level(logging.INFO)
    op=FakeOpener([http_error(),response({'story':'must not dispatch'})])
    cp=execute(setup,op)
    assert cp['status']=='terminal_technical_A_OllamaRepetitionAbortError'
    assert len(op.requests)==1 and len(cp['attempt_history'])==1
    root=__import__('pathlib').Path(setup[0]['root'])
    path=root/'runs/qwen3_8b/raw_http/qwen3_8b_001_A_1.error.bin'
    assert path.read_bytes()==b'{"error":"prediction aborted, token repeat limit reached"}'
    meta=g.read(path.with_suffix('.json'))
    assert meta['http_status']==500 and meta['request_id']==cp['attempt_history'][0]['request_id']
    assert 'pair=qwen3_8b_001 stage=A attempt=1/2' in caplog.text
    assert 'OllamaRepetitionAbortError' in caplog.text
    # Checkpoint and journal-only resume both preserve terminal outcome/budget.
    assert execute(setup,FakeOpener([]))==cp
    g.stage_path(root,'qwen3_8b','qwen3_8b_001','A').unlink()
    again=execute(setup,FakeOpener([]))
    assert again==cp


@pytest.mark.parametrize('code,body,status',[
    (400,b'bad input','OllamaHTTPResponseError'),
    (500,b'{"error":"backend failure"}','OllamaHTTPResponseError'),
    (404,b'{"error":"model not found"}','OllamaModelMissingError'),
])
def test_other_http_errors_preserved_without_transport_retry(setup,code,body,status):
    op=FakeOpener([http_error(code,body)])
    cp=execute(setup,op)
    assert cp['status']=='terminal_technical_A_'+status and len(op.requests)==1


def test_timeouts_exhaust_budget_keep_A_and_export_failed_B(setup):
    a=execute(setup,FakeOpener([response({'story':'Saved original.'})]))
    op=FakeOpener([TimeoutError('timed out')]*2)
    b=execute(setup,op,'B',a['fields']['original_story'])
    assert b['status']=='terminal_technical_B_OllamaTimeoutError' and len(op.requests)==2
    assert 'server completion/cancellation is unknown' in b['error']
    assert execute(setup,FakeOpener([]),'B',a['fields']['original_story'])==b
    assert execute(setup,FakeOpener([]))==a
    s,f=setup
    g.update_progress(s,f)
    from pathlib import Path
    path=Path(s['root'])/'exports/partial/qwen3_8b_generated_pairs.partial.jsonl'
    original_bytes=path.read_bytes()
    g.update_progress(s,f)
    assert path.read_bytes()==original_bytes
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows)==1 and rows[0]['original_story']=='Saved original.'
    assert rows[0]['modified_story'] is None and rows[0]['generation_status']==b['status']


def test_connection_failure_still_retries_and_received_response_resumes(setup):
    op=FakeOpener([urllib.error.URLError('connection refused'),response({'story':'saved'})])
    cp=execute(setup,op)
    assert cp['status']=='ok' and len(op.requests)==2
    from pathlib import Path
    g.stage_path(Path(setup[0]['root']),'qwen3_8b','qwen3_8b_001','A').unlink()
    assert execute(setup,FakeOpener([]))==cp


def test_interruptions_remain_pending_and_never_reset_budget(setup):
    for _ in range(2):
        with pytest.raises(OllamaInterruptedError):
            execute(setup,FakeOpener([KeyboardInterrupt()]))
    with pytest.raises(OllamaPendingError):
        execute(setup,FakeOpener([]))


def test_missing_explanation_is_not_repaired_or_quality_retried(setup):
    a=execute(setup,FakeOpener([response({'story':'Earlier fact.'})]))
    obj={'modified_story':'Earlier fact. Changed fact.','intended_contradiction':{
        'error_sentence':'Changed fact.','contradicted_sentence':'Earlier fact.'}}
    op=FakeOpener([response(obj)])
    b=execute(setup,op,'B',a['fields']['original_story'])
    assert b['status']=='terminal_parsing_B' and len(op.requests)==1
    assert b['fields']['modified_story']==obj['modified_story']
    assert b['fields']['intended_contradiction']['explanation'] is None


@pytest.mark.parametrize('text',['{"story":"unfinished', '{"story":"literal\nnewline"}'])
def test_incomplete_or_malformed_text_remains_failure(setup,text):
    cp=execute(setup,FakeOpener([{'response':text,'done':True,'done_reason':'stop'}]))
    assert cp['status']=='terminal_parsing_A'


def test_legacy_journal_is_not_reinterpreted(setup):
    from pathlib import Path
    cp=execute(setup,FakeOpener([response({'story':'saved'})]))
    root=Path(setup[0]['root'])
    g.stage_path(root,'qwen3_8b','qwen3_8b_001','A').unlink()
    path=root/'runs/qwen3_8b/requests.jsonl'
    events=[json.loads(x) for x in path.read_text().splitlines()]
    for e in events:e.pop('client_policy')
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(OllamaPendingError,match='Historical journal policy'):
        execute(setup,FakeOpener([]))


def test_error_capture_failure_stays_pending_without_more_dispatch(setup,monkeypatch):
    import fyp.experiments.generation_transport_v1 as transport
    def fail(*args):raise OSError('synthetic metadata write failure')
    monkeypatch.setattr(transport,'save',fail)
    op=FakeOpener([http_error()])
    with pytest.raises(OllamaPendingError,match='preserve HTTP error'):
        execute(setup,op)
    assert len(op.requests)==1
    with pytest.raises(OllamaPendingError):execute(setup,FakeOpener([]))


def test_complete_B_with_quality_warning_remains_success_and_skips(setup):
    a=execute(setup,FakeOpener([response({'story':'Saved original.'})]))
    obj={'modified_story':'Saved original.','intended_contradiction':{
        'error_sentence':'Absent.','contradicted_sentence':'Also absent.','explanation':'Model claim.'}}
    b=execute(setup,FakeOpener([response(obj)]),'B',a['fields']['original_story'])
    assert b['status']=='ok' and 'unchanged_story' in b['warnings']
    assert execute(setup,FakeOpener([]),'B',a['fields']['original_story'])==b
    row=g.pair_record('qwen3_8b_001',{'writing_prompt_id':'prompt_001'},a,b)
    assert row['generation_status']=='ok' and row['original_story']=='Saved original.'


def test_preserved_mistral_response_is_missing_data_not_parser_bug():
    path='experiments/generated_stories_recovery_v2/checkpoints/mistral_7b/mistral_7b_001_B.json'
    cp=g.read(path)
    fields,error=g.parse(cp['raw_response']['response'],'B')
    assert fields==cp['fields'] and error=='invalid_required_structure'
    assert fields['intended_contradiction']['explanation'] is None
