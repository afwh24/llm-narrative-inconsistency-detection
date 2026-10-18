import copy
import json
from pathlib import Path
import pytest
from fyp.experiments import generated_stories as g
from fyp.experiments import generated_stories_diagnostics_v2 as d
from test_models import FakeOpener


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    cfg=g.read(d.CONFIG);cfg['root']=str(tmp_path/'revision')
    path=tmp_path/'config.json';g.save(path,cfg);monkeypatch.setattr(d,'CONFIG',path)
    d.prepare()
    return cfg


def test_only_runtime_revision_and_no_budget_reset(prepared):
    cfg,m=d.loaded();_,old=d.previous.loaded()
    expected=copy.deepcopy(old['inspection']);expected['ollama_version']='0.40.2'
    assert m['inspection']==expected and m['cases']==old['cases']
    assert d.prepare()==d.status()
    assert all(row['remaining_dispatch_budget']==1 for row in d.status())


def test_exact_runtime_pin_still_enforced(prepared,monkeypatch):
    _,m=d.loaded()
    current=copy.deepcopy(m['inspection'])
    monkeypatch.setattr(g,'inspect_models',lambda *args:current)
    assert '0.40.2' in d.check()
    current['ollama_version']='0.40.3'
    with pytest.raises(RuntimeError,match='ollama_version'):d.check()


def test_prior_attempts_block_new_budget(tmp_path):
    p=tmp_path/'case/runs/model/requests.jsonl';p.parent.mkdir(parents=True);p.write_text('')
    with pytest.raises(RuntimeError,match='duplicate budgets'):d.require_unexecuted(tmp_path)


def test_failed_stage_single_dispatch_resume_and_original_retained(prepared,monkeypatch):
    monkeypatch.setattr(d,'check',lambda:'mock metadata')
    monkeypatch.setattr(g,'context_budget',lambda *args:{})
    _,m=d.loaded();case=m['cases']['llama_timeout']
    op=FakeOpener([TimeoutError('synthetic')])
    def factory(config,path):
        return d.previous.DiagnosticClient(config,path,case=case,request_format=case['request_format'],opener=op)
    first=d.run('llama_timeout',client_factory=factory)
    assert first[0]['status']=='terminal_technical_B_OllamaTimeoutError'
    assert d.run('llama_timeout',client_factory=factory)==first and len(op.requests)==1
    row=json.loads((Path(prepared['root'])/'llama_timeout/exports/diagnostic_pair.jsonl').read_text(encoding='utf8'))
    assert row['original_story']==g.read(case['source_A'])['fields']['original_story']
