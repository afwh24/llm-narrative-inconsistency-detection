import json
from pathlib import Path
import pytest
from fyp.experiments import generated_stories as g
from fyp.models.request_journal import JournaledOllamaClient, OllamaPendingError
from fyp.models.ollama_client import OllamaInterruptedError
from test_models import FakeOpener


@pytest.fixture
def setup(tmp_path,monkeypatch):
    s=g.read(g.CONFIG);s['root']=str(tmp_path)
    f={'inspection':{'models':{'qwen3:8b':{'identity':{'digest':'fake'}}}}}
    monkeypatch.setattr(g,'context_budget',lambda *args:{'test':True})
    return s,f


def factory(opener):
    return lambda config,path:g.CapturingClient(config,path,opener=opener,sleep=lambda _:None)


def response(obj):return {'response':json.dumps(obj),'done':True,'done_reason':'stop'}


def test_original_excludes_editing_targets():
    s=g.read(g.CONFIG)
    ws=g.read(s['writing_prompts']);es=g.read(s['editing_instructions'])
    assert len(ws)==10 and len({w['writing_prompt_id'] for w in ws})==10
    from collections import Counter
    assert sorted(Counter(e['category'] for e in es).values())==[2]*5
    for w,e in zip(ws,es):
        p=g.prompt(s,'A',w,e)
        assert e['editing_instruction'] not in p
        assert '{{' not in p


def test_stage_resume_reuses_original_and_full_pair_export(setup):
    s,f=setup;op=FakeOpener([response({'story':'Original unchanged text.'})])
    a=g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','original instructions',client_factory=factory(op))
    assert a['status']=='ok'
    empty=FakeOpener([])
    assert g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','original instructions',client_factory=factory(empty))==a
    assert not empty.requests
    assert g.pair_record('qwen3_8b_001',{'writing_prompt_id':'prompt_001'},a,None) is None
    b=g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','B','rewrite instructions',a['fields']['original_story'],factory(FakeOpener([response({'modified_story':'Earlier fact. Conflicting fact.','intended_contradiction':{'error_sentence':'Conflicting fact.','contradicted_sentence':'Earlier fact.','explanation':'Conflict'}})])))
    row=g.pair_record('qwen3_8b_001',{'writing_prompt_id':'prompt_001'},a,b)
    assert list(row)==['pair_id','writing_prompt_id','original_story','modified_story','intended_contradiction','generation_status']
    assert row['generation_status']=='ok'
    g.update_progress(s,f)
    p=g.read(Path(s['root'])/'progress.json')
    assert p['completed_pairs']==1 and p['pending_pairs']==29
    assert list((Path(s['root'])/'runs/qwen3_8b/raw_http').glob('*.bin'))


def test_interruption_remains_pending_and_budget_survives(setup):
    s,f=setup
    for _ in range(4):
        with pytest.raises(OllamaInterruptedError):
            g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','same',client_factory=factory(FakeOpener([KeyboardInterrupt()])))
    with pytest.raises(OllamaPendingError):
        g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','same',client_factory=factory(FakeOpener([])))
    assert not g.stage_path(Path(s['root']),'qwen3_8b','qwen3_8b_001','A').exists()


def test_exhausted_failure_final_export_no_extra_retry(setup):
    s,f=setup;op=FakeOpener([TimeoutError('fake')]*4)
    a=g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','same',client_factory=factory(op))
    assert len(op.requests)==4 and a['status'].startswith('terminal_technical_A')
    empty=FakeOpener([])
    g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','same',client_factory=factory(empty))
    assert not empty.requests
    row=g.pair_record('qwen3_8b_001',{'writing_prompt_id':'prompt_001'},a,None)
    assert row['original_story'] is None and row['modified_story'] is None


def test_malformed_response_is_not_quality_regenerated(setup):
    s,f=setup;op=FakeOpener([{'response':'```json\n{"story":"recover me"}\n```','done':True}])
    a=g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','same',client_factory=factory(op))
    assert a['status']=='terminal_parsing_A' and a['fields']['original_story']=='recover me'
    assert len(op.requests)==1


def test_quote_mismatch_is_warning_not_repair():
    s=g.read(g.CONFIG)
    fields,error=g.parse(json.dumps({'modified_story':'Tiny story.','intended_contradiction':{'error_sentence':'Absent.','contradicted_sentence':'Also absent.','explanation':'Claim.'}}),'B')
    assert error is None
    assert 'error_sentence_not_verbatim_in_modified_story' in g.quality(fields,'B',s,'Original.')


def test_context_rejection_does_not_dispatch(setup,monkeypatch):
    s,f=setup
    def reject(*args):raise RuntimeError('too large')
    monkeypatch.setattr(g,'context_budget',reject)
    def forbidden(*args):raise AssertionError('no dispatch')
    a=g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','B','huge',client_factory=forbidden)
    assert a['status']=='terminal_input_capacity_B'


def test_completed_response_recovered_after_checkpoint_loss(setup,monkeypatch):
    s,f=setup;original_save=g.save
    def broken(*args):raise OSError('synthetic crash before checkpoint')
    monkeypatch.setattr(g,'save',broken)
    with pytest.raises(OSError):g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','same',client_factory=factory(FakeOpener([response({'story':'Saved in journal.'})])))
    monkeypatch.setattr(g,'save',original_save)
    op=FakeOpener([])
    a=g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','same',client_factory=factory(op))
    assert a['fields']['original_story']=='Saved in journal.' and not op.requests


def test_malformed_http_bytes_are_preserved(setup):
    s,f=setup;op=FakeOpener([b'not JSON at all'])
    a=g.execute_stage(s,f,'qwen3:8b','qwen3_8b_001','A','same',client_factory=factory(op))
    assert a['status']=='terminal_technical_A_OllamaMalformedResponseError'
    files=list((Path(s['root'])/'runs/qwen3_8b/raw_http').glob('*.bin'))
    assert len(op.requests)==1 and files[0].read_bytes()==b'not JSON at all'


def test_freeze_copies_template_bytes_exactly(tmp_path,monkeypatch):
    s=g.read(g.CONFIG);s['root']=str(tmp_path)
    (tmp_path/'generation_contract.md').write_text('synthetic')
    config=tmp_path/'config.json';g.save(config,s)
    monkeypatch.setattr(g,'CONFIG',config)
    inspection={'models':{m:{'available':True} for m in s['models']}}
    monkeypatch.setattr(g,'inspect_models',lambda *args:inspection)
    g.prepare()
    for path in s['templates'].values():
        assert Path(path).read_bytes()==(tmp_path/'frozen_templates'/Path(path).name).read_bytes()
    g.frozen()
