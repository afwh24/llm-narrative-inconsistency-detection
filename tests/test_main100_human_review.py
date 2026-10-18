"""Offline tests for review selection, human-field separation and preservation."""
from pathlib import Path

import pytest

from fyp.experiments.main100_human_review import (
    HUMAN_FIELDS, candidate, cell, jsonl, outcome, read, require_new_output, save_unannotated_draft, select,
)


def trace(sid, gold, label, affected=False):
    return dict(example_id=sid,gold=gold,corrected_prediction={'label':label},
                label_changed=affected,corrected_changed_any_side=False)


def test_invalid_kept_separate_from_false_negative():
    assert outcome(1,None)=='INVALID'
    c=candidate(trace('a',1,None),trace('a',1,1))
    assert c['stratum']=='baseline_wrong_verify_correct'
    assert 'invalid_output' in c['features'] and 'baseline_fn' not in c['features']


def test_gold_or_story_mismatch_rejected():
    with pytest.raises(ValueError): candidate(trace('a',1,1),trace('b',1,1))
    with pytest.raises(ValueError): candidate(trace('a',1,1),trace('a',0,1))


def test_selection_reproducible_and_covers_real_available_strata():
    config=read('configs/experiments/main100_human_review_v1.json')
    path=Path(config['corrected_analysis'])/'record_traces.jsonl'
    if not path.exists(): pytest.skip('Saved research artifacts are unavailable')
    traces=jsonl(path)
    for model in config['models']:
        b={r['example_id']:r for r in traces if r['condition']==model+'_baseline'}
        v={r['example_id']:r for r in traces if r['condition']==model+'_verify_revise'}
        candidates=[candidate(b[s],v[s]) for s in b]
        chosen,coverage=select(candidates,model,config)
        reversed_chosen,_=select(list(reversed(candidates)),model,config)
        assert [c['example_id'] for c in chosen]==[c['example_id'] for c in reversed_chosen]
        assert len(chosen)==len({c['example_id'] for c in chosen})==10
        assert coverage['selected_strata']==config['stratum_quotas']
        for feature,minimum in config['coverage_minima'].items():
            assert coverage['selected_features'].get(feature,0)>=min(minimum,coverage['available_features'].get(feature,0))
        if model=='llama3_1_8b':
            assert coverage['unavailable_features']==['baseline_fn']
            assert coverage['selected_features']['invalid_output']>=1


def test_duplicate_selection_candidates_rejected():
    c=candidate(trace('a',1,0),trace('a',1,1))
    with pytest.raises(ValueError,match='Duplicate'): select([c,c],'model',{})


def test_missing_stratum_not_silently_substituted():
    config={'stratum_quotas':{'both_wrong':1},'coverage_minima':{}}
    c=candidate(trace('a',1,0),trace('a',1,1))
    with pytest.raises(ValueError,match='Insufficient'): select([c],'model',config)


def test_existing_annotations_cannot_be_overwritten(tmp_path):
    review=tmp_path/'review.csv'
    review.write_text('review_id,baseline_human_detection_correct\na,Yes\n',encoding='utf8')
    before=review.read_bytes()
    with pytest.raises(FileExistsError,match='preserve annotations'): require_new_output(tmp_path)
    assert review.read_bytes()==before


def test_markdown_preserves_visible_tags_and_table_boundaries():
    assert cell('<decision>x | y</decision>\nnext')=='&lt;decision&gt;x &#124; y&lt;/decision&gt;<br>next'


def test_human_fields_complete_and_separate():
    assert len(HUMAN_FIELDS)==12
    assert all('human_' in f for f in HUMAN_FIELDS)
    assert 'human_questionable_gold' in HUMAN_FIELDS


def test_initial_export_completion_preserves_draft_and_rejects_annotations(tmp_path):
    draft=tmp_path/'review.md'
    save_unannotated_draft(draft,'Blank review\n')
    save_unannotated_draft(draft,'Blank review\n')
    draft.write_text('Human annotation: Yes\n',encoding='utf8')
    before=draft.read_bytes()
    with pytest.raises(ValueError,match='preserve possible annotations'):
        save_unannotated_draft(draft,'Blank review\n')
    assert draft.read_bytes()==before
