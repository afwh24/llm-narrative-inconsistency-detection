import json
import socket
from pathlib import Path

import pytest

from fyp.experiments import generated_stories as g
from fyp.experiments import generated_stories_diagnostics as d
from fyp.models.base import ModelRequest
from fyp.models.ollama_client import OllamaInterruptedError
from fyp.models.request_journal import OllamaPendingError
from test_models import FakeOpener


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No network in diagnostic tests')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    monkeypatch.setattr(socket.socket,'connect_ex',forbidden)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    cfg=g.read(d.CONFIG)
    cfg['root']=str(tmp_path/'diagnostics')
    config=tmp_path/'config.json';g.save(config,cfg)
    monkeypatch.setattr(d,'CONFIG',config)
    d.prepare()
    return cfg


def factory(case,opener):
    return lambda config,path:d.DiagnosticClient(config,path,case=case,
        request_format=case['request_format'],opener=opener,sleep=lambda _:None)


def test_preparation_is_idempotent_and_status_read_only(prepared):
    root=Path(prepared['root'])
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert d.prepare()==d.status()
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert [x['dispatch_starts'] for x in d.status()]==[0,0]


def test_only_declared_factor_changes(prepared):
    _,manifest=d.loaded()
    for name,case in manifest['cases'].items():
        config=g.load_model_config(manifest['source_settings']['models'][case['model']]['config'])
        c=d.DiagnosticClient(config,Path('unused'),case=case,request_format=case['request_format'],opener=FakeOpener([]))
        body=c.build_generate_payload(ModelRequest(case['prompt']))
        different=[k for k in body.keys()|case['source_body'].keys() if body.get(k)!=case['source_body'].get(k)]
        assert different==([] if name=='llama_timeout' else ['format'])
        assert c.config.runtime.max_retries==0
        assert c.config.runtime.timeout_seconds==(1200 if name=='llama_timeout' else 600)
        assert c.config.generation==config.generation


def test_one_dispatch_saved_A_reuse_and_repeat_run_no_duplicates(prepared,monkeypatch):
    monkeypatch.setattr(d,'check',lambda:'metadata mocked')
    monkeypatch.setattr(g,'context_budget',lambda *args:{})
    _,manifest=d.loaded();case=manifest['cases']['mistral_schema']
    original=g.read(case['source_A'])['fields']['original_story']
    response={'response':json.dumps({'modified_story':original,'intended_contradiction':{
        'error_sentence':'Absent','contradicted_sentence':'Absent','explanation':'Claim'}}),'done':True,'done_reason':'stop'}
    op=FakeOpener([response])
    first=d.run('mistral_schema',client_factory=factory(case,op))
    second=d.run('mistral_schema',client_factory=factory(case,FakeOpener([])))
    assert first==second and len(op.requests)==1
    root=Path(prepared['root'])/'mistral_schema'
    rows=[json.loads(x) for x in (root/'exports/diagnostic_pair.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==1 and rows[0]['original_story']==original
    assert not list(root.glob('checkpoints/**/*_A.json'))
    assert rows[0]['generation_status']=='ok' # Quality warnings never cause regeneration.


def test_timeout_single_attempt_terminal_and_no_budget_reset(prepared,monkeypatch):
    monkeypatch.setattr(d,'check',lambda:'metadata mocked')
    monkeypatch.setattr(g,'context_budget',lambda *args:{})
    _,manifest=d.loaded();case=manifest['cases']['llama_timeout']
    op=FakeOpener([TimeoutError('synthetic')])
    rows=d.run('llama_timeout',client_factory=factory(case,op))
    assert rows[0]['status']=='terminal_technical_B_OllamaTimeoutError'
    assert rows[0]['remaining_dispatch_budget']==0 and len(op.requests)==1
    d.run('llama_timeout',client_factory=factory(case,FakeOpener([])))


def test_interrupted_attempt_remains_pending_but_cannot_dispatch_again(prepared,monkeypatch):
    monkeypatch.setattr(d,'check',lambda:'metadata mocked')
    monkeypatch.setattr(g,'context_budget',lambda *args:{})
    _,manifest=d.loaded();case=manifest['cases']['llama_timeout']
    with pytest.raises(OllamaInterruptedError):
        d.run('llama_timeout',client_factory=factory(case,FakeOpener([KeyboardInterrupt()])))
    assert d.status()[0]['remaining_dispatch_budget']==0
    with pytest.raises(OllamaPendingError):
        d.run('llama_timeout',client_factory=factory(case,FakeOpener([])))


def test_config_change_refuses_resume(prepared):
    cfg=g.read(d.CONFIG);cfg['cases']['llama_timeout']['timeout_seconds']=2400;g.save(d.CONFIG,cfg)
    with pytest.raises(RuntimeError,match='configuration changed'):d.loaded()


def test_complete_pair_cannot_be_selected(tmp_path,monkeypatch):
    cfg=g.read(d.CONFIG);cfg['root']=str(tmp_path/'out')
    cfg['cases']['mistral_schema']['pair_id']='mistral_7b_008'
    path=tmp_path/'config.json';g.save(path,cfg);monkeypatch.setattr(d,'CONFIG',path)
    with pytest.raises(RuntimeError,match='failed B'):d.prepare()
