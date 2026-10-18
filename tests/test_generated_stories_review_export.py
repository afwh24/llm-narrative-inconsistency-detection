"""Offline eligibility/leakage regression tests; never import a model client."""
import copy
import socket
import pytest
from fyp.experiments.generated_stories_review_export import classify, story_exports, JUDGMENTS


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*a, **kw):
        raise AssertionError('Network forbidden')
    monkeypatch.setattr(socket.socket, 'connect', blocked)


@pytest.fixture
def case():
    c = dict(pair_id='x',writing_prompt_id='p',original_story='Only red.',modified_story='Only red. Never red.',generation_status='ok',
        intended_contradiction=dict(error_sentence='Never red.',contradicted_sentence='Only red.',explanation='The claims conflict.'))
    r = dict.fromkeys(JUDGMENTS,'Yes'); r['notes']=''
    return c,r


def test_accept_exact_affirmed_annotations_without_mutating(case):
    c,r=case; before=copy.deepcopy(case)
    out=classify(c,[r],[],{})
    assert out['classification']=='accepted'
    assert out['verified_annotations']==c['intended_contradiction']
    assert case==before


@pytest.mark.parametrize('value', ['', 'Unsure', 'yes', 'Maybe'])
def test_uncertainty_precedes_other_no(case,value):
    c,r=case; r[JUDGMENTS[0]]=value; r[JUDGMENTS[1]]='No'
    assert classify(c,[r],[],{})['classification']=='needs_review'


def test_clear_no_rejected(case):
    c,r=case; r[JUDGMENTS[1]]='No'
    assert classify(c,[r],[],{})['classification']=='rejected'


@pytest.mark.parametrize('quote', ['', 'never red.', 'Not present'])
def test_evidence_conflicts_flag_even_with_other_no(case,quote):
    c,r=case; r[JUDGMENTS[0]]='No'; c['intended_contradiction']['error_sentence']=quote
    assert classify(c,[r],[],{})['classification']=='needs_review'


def test_unresolved_notes_block_acceptance(case):
    c,r=case; r['notes']='Potential original contradiction'
    assert classify(c,[r],[],dict(disposition='unresolved',reason='Requires review'))['classification']=='needs_review'
    assert classify(c,[r],[],{})['classification']=='needs_review'


def test_optional_corrections_not_silently_approved(case):
    c,r=case
    assert classify(c,[r],[{'verified_explanation':'New explanation'}],{})['classification']=='needs_review'


def test_identical_versions_inconsistent_yes(case):
    c,r=case; c['original_story']=c['modified_story']
    assert classify(c,[r],[],{})['classification']=='needs_review'


def test_duplicate_missing_reviews(case):
    c,r=case
    assert classify(c,[],[],{})['classification']=='needs_review'
    assert classify(c,[r,r],[],{})['classification']=='needs_review'


def test_export_two_versions_opaque_and_deterministic(case):
    c,r=case; row=classify(c,[r],[],{})
    row.update(candidate=c,provenance=dict(model='model',category='category',stages=[]))
    result=story_exports([row],'salt')
    assert result==story_exports([row],'salt')
    inputs,gold,mapping=result
    assert len(inputs)==2 and len({i['story_id'] for i in inputs})==2
    assert all(set(i)=={'story_id','story'} for i in inputs)
    assert {g['cont_error'] for g in gold}=={0,1}
    assert {m['version'] for m in mapping}=={'original','modified'}
    assert {m['writing_prompt_id'] for m in mapping}=={'p'}
    negative=next(g for g in gold if g['cont_error']==0)
    assert negative['cont_error_lines']==negative['contradicted_lines']==negative['cont_error_expl']==''
