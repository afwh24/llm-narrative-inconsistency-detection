from pathlib import Path

from fyp.experiments.main100_manual_review_simple import (
    FIELDS, explanation_status, transfer, validate_and_summarize,
)


def original(**kwargs):
    return dict(review_id='m:s',story_id='s',model='m',**kwargs)


def test_blank_transfer_has_only_requested_columns():
    row,issues=transfer(original())
    assert tuple(row)==FIELDS and not issues
    assert all(row[k]=='' for k in FIELDS[3:])


def test_transfer_conservative_and_notes_attributed():
    row,issues=transfer(original(baseline_human_explanation_correct='Yes',
        verify_revise_human_explanation_correct='Unclear',human_gold_notes='Maybe ambiguous'))
    assert row['baseline_explanation_supported']==''
    assert row['verify_revise_explanation_supported']=='Unsure'
    assert len(issues)==1 and row['notes']=='Gold note: Maybe ambiguous'


def test_unambiguous_transfer_and_unmappable_na():
    row,issues=transfer(original(baseline_human_explanation_correct='Yes',baseline_human_detection_correct='Yes',
                                verify_revise_human_explanation_correct='N/A'))
    assert row['baseline_explanation_supported']=='Yes'
    assert row['verify_revise_explanation_supported']=='' and len(issues)==1


def test_missing_explanation_presence_not_quality():
    markers=['','none','n/a','no explanation']
    assert explanation_status('<explanation></explanation><decision>No continuity error found</decision>',markers)=='No explanation'
    assert explanation_status('<response><decision>No continuity error found</decision></response>',markers)=='No explanation'
    assert explanation_status('No continuity error found.',markers)=='No explanation'
    assert explanation_status('<explanation>The story is consistent.</explanation>',markers).startswith('Explanation text present')
    assert explanation_status('The change is explained by the passage of time.',markers).startswith('Review needed')
    assert explanation_status('<explanation>A</explanation><explanation>B</explanation>',markers).startswith('Review needed')


def reference(status='Explanation text present (support not assessed)'):
    return original(baseline_explanation_status=status,verify_revise_explanation_status=status,
                    baseline_gold_agreement='Agrees',verify_revise_gold_agreement='Agrees')


def test_unsure_is_not_dropped_or_called_no():
    row=original(baseline_explanation_supported='Unsure',verify_revise_explanation_supported='Yes',notes='')
    result=validate_and_summarize([row],[reference()],['Yes','No','Unsure'])
    assert result['status']=='READY_TO_SUMMARISE'
    assert result['paired_support_counts']['uncertain_pairs']==1
    assert not result['paired_support_counts']['verify_revise_better_supported']


def test_no_explanation_flag_not_a_judgment_and_separate():
    row=original(baseline_explanation_supported='',verify_revise_explanation_supported='',notes='')
    result=validate_and_summarize([row],[reference('No explanation')],['Yes','No','Unsure'])
    assert len(result['missing_required_judgments'])==2
    assert result['paired_support_counts']['separate_no_explanation_pairs']==1
    assert result['counts_by_model_strategy']['m/baseline']['No']==0


def test_wrong_values_duplicate_and_missing_rows_flagged():
    row=original(baseline_explanation_supported='N/A',verify_revise_explanation_supported='',notes='')
    result=validate_and_summarize([row,row],[reference()],['Yes','No','Unsure'])
    assert len(result['errors'])==2 and len(result['missing_required_judgments'])==1
    missing=validate_and_summarize([],[reference()],['Yes','No','Unsure'])
    assert len(missing['missing_required_judgments'])==2 and missing['errors']


def test_pair_comparison_uses_human_labels_and_categories_are_automated():
    row=original(baseline_explanation_supported='No',verify_revise_explanation_supported='Yes',notes='Wrong premise')
    result=validate_and_summarize([row],[reference()],['Yes','No','Unsure'])
    assert result['paired_support_counts']['verify_revise_better_supported']==1
    assert result['automated_interpretations'][0]['attribution'].startswith('AUTOMATED')
    assert result['automated_interpretations'][0]['reviewer_notes_verbatim']=='Wrong premise'
