import copy
import pytest
from fyp.experiments import generated_stories_recovery_v2 as r
from fyp.experiments import generated_stories as g


def snapshot():
    return {'ollama_version': '0.40.1', 'models': {'qwen': {'identity': {
        'digest': 'same', 'parameters': 'temperature 0.6\nstop "a"\nstop "b"\ntop_k 20\n'}}}}


def test_parameter_key_order_is_not_identity_change():
    expected = snapshot()
    current = copy.deepcopy(expected)
    current['models']['qwen']['identity']['parameters'] = 'top_k    20\nstop "a"\nstop "b"\ntemperature 0.6\n'
    r.validate_inspection(current, expected)
    assert expected == snapshot()


@pytest.mark.parametrize('kind', ['version', 'digest', 'parameter', 'stop_order'])
def test_real_changes_still_block(kind):
    expected = snapshot()
    current = copy.deepcopy(expected)
    identity = current['models']['qwen']['identity']
    if kind == 'version': current['ollama_version'] = '0.40.2'
    elif kind == 'digest': identity['digest'] = 'other'
    elif kind == 'parameter': identity['parameters'] = identity['parameters'].replace('0.6', '0.7')
    else: identity['parameters'] = 'temperature 0.6\nstop "b"\nstop "a"\ntop_k 20\n'
    with pytest.raises(RuntimeError, match='Live metadata differs at'):
        r.validate_inspection(current, expected)


def test_connection_error_not_reported_as_identity_change():
    with pytest.raises(RuntimeError, match='Cannot inspect local Ollama.*connection refused'):
        r.validate_inspection({'error': 'connection refused'}, snapshot())


def test_original_snapshot_matches_observed_metadata_except_approved_version():
    old = g.read('experiments/generated_stories_v1/preparation.json')['inspection']
    observed = g.read('experiments/generated_stories_recovery_v1/live_inspection_host_diagnostic.json')
    with pytest.raises(RuntimeError, match='/ollama_version'):
        r.validate_inspection(observed, old)
    revised = copy.deepcopy(old)
    revised['ollama_version'] = '0.40.1'
    r.validate_inspection(observed, revised)
