from fyp.experiments.main100_review_analysis import validate, counts


def row(b='No',v='Yes'):
    return dict(review_id='m:s',story_id='s',model='m',baseline_explanation_supported=b,verify_revise_explanation_supported=v,notes='Exact note\n preserved')


def test_complete_uncertainty_is_not_missing():
    r=row('No','Unsure');before=r.copy()
    assert validate([r],[r])==([],[])
    c=counts([r],{'m:s':{'baseline':'present','verify_revise':'present'}})['m']
    assert c['pair_counts']['uncertain']==1 and c['pair_counts']['verify_revise_better']==0
    assert r==before


def test_blank_invalid_duplicate_and_omitted_scope():
    r=row('','N/A')
    errors,missing=validate([r,r],[r])
    assert len(errors)==2 and len(missing)==1
    errors,missing=validate([],[r])
    assert len(errors)==1 and len(missing)==2


def test_missing_explanation_does_not_supply_a_human_label():
    r=row('','Yes')
    c=counts([r],{'m:s':{'baseline':'No explanation','verify_revise':'present'}})['m']
    assert c['pair_counts']['no_explanation_separate']==1
    assert c['judgments']['baseline']['No']==0
    assert len(validate([r],[r])[1])==1
