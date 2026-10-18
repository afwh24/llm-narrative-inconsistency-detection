"""Offline checks for exploratory analysis math and report preservation."""
import pytest
from fyp.experiments.main100_analysis import (
    bootstrap, contributions, family_test, from_counts, holm,
    mcnemar, paired, quantile, save_versioned,
)


def fact(gold, pred, full):
    return dict(gold=gold, predicted=pred, classification=int(gold==pred), full=full)


def test_metrics_include_failed_positives_without_false_positive_penalty():
    facts = [fact(1,1,1),fact(1,None,0),fact(0,None,0),fact(0,0,1),fact(0,1,0)]
    stats = [contributions(x) for x in facts]
    assert from_counts([sum(x[i] for x in stats) for i in range(8)]) == (.4,.5,.5,.5,.4,.5)


def test_family_swap_retains_dependence():
    assert family_test([1,1,1,1], ['a']*4) == 1
    assert family_test([1,1,1,1], list('abcd')) == .125
    assert family_test([1,-1], ['a','a']) == 1
    assert family_test([1,-1,1], list('abc')) == mcnemar(2,1)


def test_paired_counts_and_mcnemar():
    assert paired([0,1,1,0],[1,0,1,0]) == dict(improved=1,regressed=1,both_correct=1,both_wrong=1)
    assert mcnemar(0,0) == 1
    assert mcnemar(5,0) == .0625
    with pytest.raises(ValueError): paired([1],[])


def test_holm_and_quantiles():
    assert holm([.04,.001,.03]) == pytest.approx([.06,.003,.06])
    assert quantile([3,1,2,0], .5) == 1.5


def test_bootstrap_paired_reproducible_and_whole_family():
    c = {'facts':[fact(1,1,1),fact(0,0,1)]}
    draws, zero = bootstrap([c,c], ['same','same'], 20,42)
    assert draws[0] == draws[1]
    assert all(value==1 for metric in draws[0] for value in metric)
    assert (draws,zero) == bootstrap([c,c], ['same','same'],20,42)
    assert zero==0


def test_reports_preserve_previous_bytes(tmp_path):
    path=tmp_path/'report.md'
    save_versioned(path,'first\n')
    save_versioned(path,'second\n')
    save_versioned(path,'second\n')
    old=list((tmp_path/'report_history').iterdir())
    assert len(old)==1 and old[0].read_bytes()==b'first\n'
    assert path.read_bytes()==b'second\n'
