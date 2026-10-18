import json
from pathlib import Path
import pytest
from fyp.experiments import generated_stories as g
from fyp.experiments import generated_stories_schema_recovery as r
from fyp.models.ollama_client import OllamaInterruptedError
from fyp.models.request_journal import OllamaPendingError
from test_models import FakeOpener


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    cfg=g.read(r.CONFIG);cfg['root']=str(tmp_path/'batch')
    path=tmp_path/'config.json';g.save(path,cfg);monkeypatch.setattr(r,'CONFIG',path)
    r.prepare();monkeypatch.setattr(r,'check',lambda:'mock metadata')
    monkeypatch.setattr(g,'context_budget',lambda *args:{})
    return cfg


def factory(cfg,op):
    return lambda config,path:r.diagnostic.previous.DiagnosticClient(config,path,
        case=cfg,request_format=g.read(cfg['schema_path']),opener=op,sleep=lambda _:None)


def response():
    return {'response':json.dumps({'modified_story':'Earlier. Later.',
        'intended_contradiction':{'error_sentence':'Later.','contradicted_sentence':'Earlier.','explanation':'Claim'}}),
        'done':True,'done_reason':'stop'}


def test_selection_and_fixed_scope(prepared):
    s=r.status();assert s['complete_pairs']==22 and s['pending_pairs']==8 and s['new_dispatch_starts']==0
    _,m,_,_=r.loaded()
    assert [row['pair_id'] for row in m['rows'] if row['B'] is None]==prepared['pairs']
    assert next(row for row in m['rows'] if row['pair_id']=='llama3_1_8b_009')['B_run'].endswith('llama_timeout')
    assert next(row for row in m['rows'] if row['pair_id']=='mistral_7b_001')['B_run'].endswith('mistral_schema')
    assert r.prepare()==r.status()


def test_eight_B_only_no_duplicate_resume(prepared):
    op=FakeOpener([response() for _ in range(8)])
    s=r.run(client_factory=factory(prepared,op))
    assert s['complete_pairs']==30 and s['new_dispatch_starts']==8
    assert r.run(client_factory=factory(prepared,FakeOpener([])))==s
    assert len(op.requests)==8
    for req in op.requests:
        body=json.loads(req.data)
        assert body['model']=='mistral:7b' and isinstance(body['format'],dict)
        assert 'Original story:' in body['prompt']
    root=Path(prepared['root'])
    assert not list(root.glob('checkpoints/**/*_A.json'))
    provenance=g.read(root/'selection_provenance.json');assert len(provenance)==30
    for stem in ('qwen3_8b','llama3_1_8b','mistral_7b'):
        rows=[json.loads(x) for x in (root/'exports'/f'{stem}_generated_pairs.jsonl').read_text(encoding='utf8').splitlines()]
        assert len(rows)==10 and len({x['pair_id'] for x in rows})==10


@pytest.mark.parametrize('outcome',[TimeoutError('synthetic'),{'response':'{"modified_story":"saved"}','done':True}])
def test_first_failure_stops_and_resume_cannot_escalate(prepared,outcome):
    op=FakeOpener([outcome,response()])
    with pytest.raises(RuntimeError,match='Batch stopped'):r.run(client_factory=factory(prepared,op))
    assert len(op.requests)==1 and r.status()['terminal_pairs']==1
    with pytest.raises(RuntimeError,match='previously failed'):r.run(client_factory=factory(prepared,FakeOpener([])))
    assert r.status()['new_dispatch_starts']==1


def test_interrupted_start_blocks_later_pairs(prepared):
    with pytest.raises(OllamaInterruptedError):r.run(client_factory=factory(prepared,FakeOpener([KeyboardInterrupt()])))
    with pytest.raises(OllamaPendingError):r.run(client_factory=factory(prepared,FakeOpener([])))
    assert r.status()['new_dispatch_starts']==1


def test_changed_config_refused(prepared):
    cfg=g.read(r.CONFIG);cfg['max_retries']=1;g.save(r.CONFIG,cfg)
    with pytest.raises(RuntimeError,match='config changed'):r.loaded()
