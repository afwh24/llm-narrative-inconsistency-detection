"""Offline runtime-revision checks; no model inference."""
from copy import deepcopy
from pathlib import Path

import pytest

from fyp.experiments import main100_comparison as parent
from fyp.experiments import main100_runtime_revision as revision
from fyp.models.ollama_client import OllamaClient
import test_main100_comparison as inherited


def test_parameter_order_only_preserves_duplicate_values_and_order():
    assert revision.parameter_values('repeat_penalty 1\nstop "a"\nstop "b"') == revision.parameter_values('stop "a"\nstop "b"\nrepeat_penalty 1')
    assert revision.parameter_values('stop "a"\nstop "b"') != revision.parameter_values('stop "b"\nstop "a"')
    assert revision.parameter_values('repeat_penalty 1') != revision.parameter_values('repeat_penalty 2')
    with pytest.raises(parent.ExperimentRunError, match='Malformed'):
        revision.parameter_values('repeat_penalty')


def test_frozen_revision_same_sample_and_isolated_paths():
    old = parent.checked_freeze()
    module = revision.engine()
    new = module.checked_freeze()
    assert parent.CONFIG == Path('configs/experiments/main100_comparison_v1.json')
    assert module.CONFIG == revision.CONFIG
    assert new['ollama_version'] == '0.40.0' and old['ollama_version'] == '0.35.1'
    assert new['manifest_hash'] == old['manifest_hash']
    for field in ('manifest', 'inputs', 'selected_strategy', 'selection_salt', 'template_copy_dir'):
        assert new['settings'][field] == old['settings'][field]
    for field in ('templates', 'rendered_prompts', 'context_budgets', 'dependencies'):
        assert new[field] == old[field]
    assert len(new['conditions']) == 6
    for before, after in zip(old['conditions'], new['conditions']):
        for key in ('id', 'model', 'strategy', 'version', 'model_config'):
            assert before[key] == after[key]
        for key in ('run_dir', 'export', 'partial_export'):
            assert before[key] != after[key]
            assert Path(after[key]).is_relative_to(Path(new['settings']['root']))


def test_runtime_and_model_drift_rejected():
    freeze = revision.engine().checked_freeze()
    live = {'models': deepcopy(freeze['models']), 'ollama_version': '0.40.0'}
    revision.verify_inspection(live, freeze['models'], '0.40.0')
    live['ollama_version'] = '0.35.1'
    with pytest.raises(parent.ExperimentRunError, match='Expected Ollama'):
        revision.verify_inspection(live, freeze['models'], '0.40.0')
    live['ollama_version'] = '0.40.0'
    for field in ('digest', 'template', 'parameters'):
        altered = deepcopy(live)
        altered['models']['qwen3:8b']['identity'][field] += '\nchanged value'
        with pytest.raises(parent.ExperimentRunError, match='changed'):
            revision.verify_inspection(altered, freeze['models'], '0.40.0')


def test_live_check_never_dispatches_inference(monkeypatch):
    module = revision.engine()
    freeze = module.checked_freeze()
    def forbidden(*args, **kwargs):
        pytest.fail('Inference is forbidden during checks')
    monkeypatch.setattr(OllamaClient, 'generate', forbidden)
    monkeypatch.setattr(OllamaClient, '_request_with_retries', forbidden)
    monkeypatch.setattr(revision.baseline_pilot, 'inspect_models', lambda *args: {
        'models': freeze['models'], 'ollama_version': '0.40.0'})
    assert module.checked_freeze(live=True)['ollama_version'] == '0.40.0'


def test_revision_engine_inherits_checkpoint_retry_and_export_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(inherited, 'study', revision.engine())
    inherited.test_end_to_end_interrupt_resume_exhausted_failure_and_no_gold(tmp_path, monkeypatch)


def test_preparation_refuses_to_replace_existing_freeze():
    with pytest.raises(parent.ExperimentRunError, match='already frozen'):
        revision.prepare()
