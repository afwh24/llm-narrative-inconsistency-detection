import json
import socket
from types import SimpleNamespace
import pytest
from fyp.experiments import generated_stories_detection as d


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a,**kw):raise AssertionError('Network forbidden in tests')
    monkeypatch.setattr(socket.socket,'connect',blocked)


def test_actual_frozen_dataset_and_prompt_isolation():
    cfg=d.read(d.CONFIG); audit=d.dataset_audit(cfg)
    assert audit['accepted_pairs']==1 and audit['expected_records']==12
    assert audit['duplicates']==[]
    prompts=d.build_prompts(cfg,d.read(cfg['benchmark_preparation']))
    for group in prompts.values():
        for row in group:
            assert row['story_id'] not in row['prompt']
            assert 'llama3_1_8b_010' not in row['prompt']
            assert 'prompt_010' not in row['prompt']


def test_failure_metrics_do_not_invent_labels():
    metrics=d.scoring.detection_metrics([0,1],[0,None])
    assert metrics['accuracy']==0.5 and metrics['recall']==0
    assert metrics['invalid_gold_positive']==1 and metrics['fn']==0
    assert d.scoring.parse_response('malformed',d.read(d.read(d.CONFIG)['scoring_policy'])).label is None


def test_saved_gold_localization_supported():
    ex=SimpleNamespace(story='Only red. Never red.',cont_error=1,cont_error_lines_raw='Never red.',contradicted_lines_raw='Only red.')
    score,_=d.scoring.score_localization(ex,1,['Never red.'],['Only red.'])
    assert score.full_correct==1
    score,_=d.scoring.score_localization(ex,None,['Never red.'],['Only red.'])
    assert score.full_correct==0


def test_resume_skips_completed_and_preserves_failures(tmp_path,monkeypatch):
    cfg=d.read(d.CONFIG); cfg['root']=str(tmp_path); cfg['batch_size']=10
    c={'id':'qwen_test','model':'qwen3:8b','strategy':'vanilla','model_config':'configs/models/three_model_main10/qwen3_8b.yaml'}
    f={'conditions':[c],'prompts':{'vanilla':[dict(story_id=s,prompt='only text '+s,prompt_sha256=d.hash_text('only text '+s)) for s in ['one','two']]},'inspection':{'models':{'qwen3:8b':{'identity':{'digest':'test'}}}},'readiness':{'expected_records':2},'scoring_policy':d.read(d.read(d.CONFIG)['scoring_policy'])}
    monkeypatch.setattr(d,'loaded',lambda live=False:(cfg,f))
    calls=[]
    class Fake:
        def __init__(self,*args):self.last_raw_response=None;self.last_attempt_history=[]
        def set_context(self,c):self.context=c
        def generate(self,prompt):
            calls.append(prompt)
            if 'two' in prompt:raise d.OllamaError('terminal synthetic')
            text='<response><explanation>None.</explanation><error_lines></error_lines><contradicted_lines></contradicted_lines><decision>No continuity error found.</decision></response>'
            self.last_raw_response={'response':text,'done':True};return SimpleNamespace(response=text)
    out=d.run(client_factory=Fake,live=False)
    assert out['completed']==2 and out['terminal_failures']==1
    out=d.run(client_factory=Fake,live=False)
    assert len(calls)==2 and out['pending']==0
    records=d.record_rows(cfg,f)
    assert next(r for r in records if r['story_id']=='two')['parsed']['label'] is None
    f['changed']=True
    with pytest.raises(RuntimeError,match='checkpoint'):d.record_rows(cfg,f)


def test_batch_ten_limit(tmp_path,monkeypatch):
    cfg=d.read(d.CONFIG);cfg['root']=str(tmp_path);cfg['batch_size']=10
    c={'id':'c','model':'qwen3:8b','strategy':'vanilla','model_config':'configs/models/three_model_main10/qwen3_8b.yaml'}
    f={'conditions':[c],'prompts':{'vanilla':[dict(story_id=str(i),prompt=str(i),prompt_sha256=d.hash_text(str(i))) for i in range(11)]},'inspection':{'models':{'qwen3:8b':{'identity':{'digest':'x'}}}},'readiness':{'expected_records':11},'scoring_policy':d.read(d.read(d.CONFIG)['scoring_policy'])}
    monkeypatch.setattr(d,'loaded',lambda live=False:(cfg,f))
    class Fake:
        def __init__(self,*args):self.last_raw_response=None;self.last_attempt_history=[]
        def set_context(self,c):pass
        def generate(self,prompt):raise d.OllamaError('synthetic')
    assert d.run(client_factory=Fake,live=False)['completed']==10
    assert d.run(client_factory=Fake,live=False)['completed']==11
