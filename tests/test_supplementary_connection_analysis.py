"""Offline checks for derived comparison semantics; never call model clients."""
import importlib.util
from pathlib import Path

from fyp.evaluation.corrected_v1 import aggregate

spec = importlib.util.spec_from_file_location('connection_analysis', Path(__file__).parents[1] / 'scripts/analyse_supplementary_connection_v1.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_pair_outcomes_partition_binary_and_missing_predictions():
    observed = [analysis.pair_outcome(a, b) for a in (0, 1, None) for b in (0, 1, None)]
    assert observed.count('unusable') == 5
    assert all(observed.count(k) == 1 for k in ('both_correct', 'both_consistent', 'both_contradictory', 'both_incorrect'))


def test_invalid_records_reduce_coverage_without_becoming_false_predictions():
    rows = [dict(gold=y, predicted=p, full=0, error_side=0, contradicted_side=0)
            for y,p in [(1,1),(1,None),(0,0),(0,None)]]
    m = aggregate(rows)
    assert (m['total'], m['invalid'], m['fp'], m['fn']) == (4,2,0,0)
    assert m['accuracy'] == .5 and m['precision'] == 1 and m['recall'] == .5
    assert m['f1'] == 2/3
    assert analysis.pair_outcome(0,None) == 'unusable'


def test_network_guard_rejects_calls():
    import pytest
    with pytest.raises(RuntimeError, match='Network disabled'):
        analysis.blocked('localhost',11434)
