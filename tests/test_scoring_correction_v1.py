"""Regression coverage for the opt-in scorer; no model or network calls."""
import json
from pathlib import Path

import pytest

from fyp.data.loader import normalize_record
from fyp.evaluation.corrected_v1 import aggregate, detection_metrics, expand_evidence, parse_response, score_localization
from fyp.evaluation.localization import score_localization as legacy_score
from fyp.prompting.parser import parse_official_response


@pytest.fixture
def policy():
    return json.loads(Path('configs/experiments/main100_scoring_correction_v1.json').read_text(encoding='utf8'))


@pytest.fixture
def example():
    return normalize_record(dict(example_id='synthetic', story='Ada has green eyes. Ada has blue eyes. The sun sets.',
                                 cont_error=1.0, cont_error_expl='Conflicting eye colour.',
                                 cont_error_lines='<li>Ada has blue eyes.</li>',
                                 contradicted_lines='<li>Ada has green eyes.</li>'), 'flawed_fictions')


def response(decision, error='', contradicted=''):
    return (f'<response><explanation>Explanation.</explanation><error_lines>{error}</error_lines>'
            f'<contradicted_lines>{contradicted}</contradicted_lines><decision>{decision}</decision></response>')


@pytest.mark.parametrize('fragment', ['', ' ', '.', '!!!', '</p>', '<li></li>', '[annotation]', '<br/>'])
def test_empty_normalized_evidence_never_matches(example, fragment):
    assert expand_evidence(example.story, [fragment])['sentences'] == []
    loc, _ = score_localization(example, 1, [fragment], [fragment])
    assert loc.full_correct == loc.cont_error_lines_correct == loc.contradicted_lines_correct == 0


def test_trailing_html_false_credit_is_removed_but_real_evidence_survives(example):
    wrong = {'cont_error': 1, 'cont_error_lines': ['<p>The sun sets. </p>'], 'contradicted_lines': ['<p>The sun sets. </p>']}
    assert legacy_score(example, wrong).full_correct == 1
    fixed, _ = score_localization(example, 1, wrong['cont_error_lines'], wrong['contradicted_lines'])
    assert fixed.full_correct == 0
    good, _ = score_localization(example, 1, ['<p>Ada has blue eyes.</p>'], ['<p>Ada has green eyes.</p>'])
    assert good.full_correct == 1


@pytest.mark.parametrize('error,contradicted', [
    ('ADA HAS BLUE EYES!!!', 'Ada has green eyes.'),
    ('blue eyes', 'green eyes'),
    ('<li>Ada has blue eyes.</li>', '<p>Ada has green eyes.</p>'),
    ('Ada [speaker] has blue eyes.', 'Ada has green eyes.'),
])
def test_legitimate_normalized_substrings_and_html(example, error, contradicted):
    loc, _ = score_localization(example, 1, [error], [contradicted])
    assert loc.full_correct == 1


def test_non_strict_overlap_is_retained_and_sides_not_swapped(example):
    loc, _ = score_localization(example, 1, ['blue eyes', 'The sun sets.'], ['green eyes'])
    assert loc.full_correct == 1
    wrong, _ = score_localization(example, 1, ['green eyes'], ['blue eyes'])
    assert wrong.full_correct == 0


def test_empty_gold_fragment_is_guarded_too(example):
    from dataclasses import replace
    corrupt_gold = replace(example, cont_error_lines_raw='<p></p>')
    loc, trace = score_localization(corrupt_gold, 1, ['blue eyes'], ['green eyes'])
    assert loc.full_correct == 0
    assert trace['gold_error']['sentences'] == []


@pytest.mark.parametrize('raw', [None, '', 'No continuity error found', '<response></response>',
    response(''), response('banana'), response('Maybe a continuity error exists.'),
    '<decision>Continuity error found', '</decision><decision>Continuity error found',
    response('Continuity error found')+'<decision>No continuity error found</decision>',
    response('Continuity error found. No continuity error found.'),
    response('<error_lines>Continuity error found</error_lines>')])
def test_missing_malformed_or_ambiguous_labels_are_explicit_invalids(policy, raw):
    parsed = parse_response(raw, policy)
    assert parsed.label is None
    assert parsed.invalid_reasons


@pytest.mark.parametrize('text,label', [('No continuity error found',0), ('No continuity error was found in the story.',0),
    ('Continuity error found.',1), ('A continuity error exists in the story.',1),
    ('There is a continuity error in the story. It concerns the eyes.',1),
    ('Based on the analysis, I conclude that a continuity error exists in the story.',1)])
def test_explicit_declarations(policy, text, label):
    assert parse_response(response(text), policy).label == label


def test_known_legacy_negative_inversion(policy):
    raw = response('No continuity error was found in the story.')
    assert parse_official_response(raw).cont_error == 1
    assert parse_response(raw, policy).label == 0


def test_wrapper_warning_does_not_destroy_valid_label(policy):
    parsed = parse_response('<decision>No continuity error found</decision>', policy)
    assert parsed.label == 0 and parsed.field_warnings and not parsed.invalid_reasons


def test_inference_failure_overrides_apparent_label(policy):
    parsed = parse_response(response('Continuity error found'), policy, inference_failure=True)
    assert parsed.label is None and 'inference_failure' in parsed.invalid_reasons


def test_missing_close_cannot_import_other_evidence_side(example, policy):
    raw = ('<response><error_lines>The sun sets.\n<contradicted_lines>\nAda has blue eyes. Ada has green eyes.\n'
           '</contradicted_lines><decision>Continuity error found</decision></response>')
    old = parse_official_response(raw)
    assert legacy_score(example, old.to_official_result()).full_correct == 1
    new = parse_response(raw, policy)
    assert new.error_lines == ('The sun sets.',)
    loc, _ = score_localization(example, new.label, new.error_lines, new.contradicted_lines)
    assert loc.full_correct == 0
    assert 'bounded_recovery_error_lines_missing_or_wrong_close' in new.field_warnings


def test_wrong_close_recovery_can_preserve_actual_quote(example, policy):
    raw = response('Continuity error found', 'Ada has blue eyes.', 'Ada has green eyes.').replace('</contradicted_lines>', '</error_lines>')
    new = parse_response(raw, policy)
    assert new.contradicted_lines == ('Ada has green eyes.',)
    loc, _ = score_localization(example, new.label, new.error_lines, new.contradicted_lines)
    assert loc.full_correct == 1 and new.field_warnings


def test_invalids_are_separate_and_all_denominators_conserve_scope():
    gold, pred = [1,0,0,1,1,0], [1,1,0,0,None,None]
    m = detection_metrics(gold,pred)
    assert (m['tp'],m['fp'],m['tn'],m['fn'],m['invalid_gold_positive'],m['invalid_gold_negative']) == (1,1,1,1,1,1)
    assert m['total']==6 and m['valid']==4 and m['invalid']==2
    assert m['accuracy']==m['recall']==m['specificity']==m['false_positive_rate']==m['predicted_positive_rate']==1/3
    assert m['precision']==.5 and m['f1']==m['macro_f1']==.4
    rows = [dict(gold=y,predicted=p,full=int(y==p),error_side=int(y==p),contradicted_side=int(y==p)) for y,p in zip(gold,pred)]
    a = aggregate(rows)
    assert a['ceeval_full']==a['ceeval_pos']==1/3
    assert a['valid_only_diagnostic']['accuracy']==.5
    assert a['valid_only_diagnostic']['total']==4


def test_ceeval_denominators_and_correct_negative_convention(example):
    from dataclasses import replace
    negative = replace(example, raw_label=0.0, has_continuity_error=False)
    loc, _ = score_localization(negative, 0, [], [])
    invalid, _ = score_localization(negative, None, [], [])
    assert loc.full_correct == 1 and invalid.full_correct == 0
    rows=[dict(gold=0,predicted=0,full=1,error_side=1,contradicted_side=1),
          dict(gold=1,predicted=1,full=1,error_side=1,contradicted_side=1),
          dict(gold=1,predicted=0,full=0,error_side=0,contradicted_side=0),
          dict(gold=1,predicted=None,full=0,error_side=0,contradicted_side=0)]
    m=aggregate(rows)
    assert m['ceeval_full']==.5 and m['ceeval_pos']==1/3


def test_controls_and_zero_division():
    gold=[0]*50+[1]*50
    positive=detection_metrics(gold,[1]*100)
    negative=detection_metrics(gold,[0]*100)
    assert positive['f1']==pytest.approx(2/3)
    assert positive['macro_f1']==negative['macro_f1']==pytest.approx(1/3)
    assert positive['balanced_accuracy']==negative['balanced_accuracy']==.5
    assert negative['precision']==negative['f1']==0
    empty=aggregate([])
    assert empty['accuracy']==empty['macro_f1']==empty['ceeval_pos']==0
    invalid=detection_metrics([0,1],[None,None])
    assert invalid['invalid']==2 and invalid['f1']==invalid['macro_f1']==0


def test_mismatched_or_nonbinary_scope_is_rejected():
    with pytest.raises(ValueError): detection_metrics([0,1],[1])
    with pytest.raises(ValueError): detection_metrics([2],[1])
    with pytest.raises(ValueError): detection_metrics([0],[2])
