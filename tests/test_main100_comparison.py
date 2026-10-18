"""Offline selection, export, and recovery checks; no live Ollama requests."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from fyp.data.loader import normalize_record
from fyp.data.pilot import stable_example_id_hash
from fyp.data.schema import FlawedFictionsSplit
from fyp.experiments import main100_comparison as study, runner
from fyp.experiments.main100_audit import build_families
from fyp.models.ollama_client import OllamaInterruptedError
from fyp.models.registry import load_model_config
from fyp.prompting.strategies import get_benchmark_strategy
from test_models import FakeOpener


def examples(count=100):
    return [normalize_record({'example_id': f'synthetic_{i}', 'story': f'Original story {i}.\nUnchanged Unicode: café.',
        'cont_error': float(i % 2), 'cont_error_expl': 'EVALUATOR_SECRET',
        'cont_error_lines': 'GOLD_ERROR_SECRET', 'contradicted_lines': 'GOLD_CONTRADICTION_SECRET'},
        FlawedFictionsSplit.FLAWED_FICTIONS) for i in range(count)]


def test_balanced_deterministic_selection_and_insufficient_pool():
    rows = examples(120)
    eligible = {e.example_id for e in rows[10:]}
    chosen = study.select_examples(rows, eligible, 50, 'fixed')
    assert chosen == study.select_examples(list(reversed(rows)), eligible, 50, 'fixed')
    assert len(chosen) == len({e.example_id for e in chosen}) == 100
    assert sum(e.cont_error for e in chosen) == 50
    assert not {e.example_id for e in rows[:10]} & {e.example_id for e in chosen}
    with pytest.raises(study.ExperimentRunError, match='Insufficient'):
        study.select_examples(rows[:20], eligible, 50, 'fixed')


def test_family_closure_links_variants_without_id_suffix_assumptions():
    stories = ['alpha beta gamma delta epsilon zeta eta theta iota kappa',
               'alpha beta gamma delta epsilon zeta eta theta changed ending',
               'completely distinct narrative with different words in its body and ending']
    rows = [SimpleNamespace(example_id=str(i), story=s) for i, s in enumerate(stories)]
    mapping, groups, edges = build_families(rows, 5, 0.2)
    assert mapping['0'] == mapping['1'] != mapping['2']
    assert groups[mapping['0']] == ['0', '1'] and edges


def test_pilot_selection_uses_full_before_accuracy_or_runtime():
    candidates = [{'model': 'primary', 'strategy': method, 'metrics': {'ceeval_full': full, 'ceeval_pos': pos},
        'failed': 0, 'runtime': {'sum_seconds': time}} for method, full, pos, time in
        [('evidence_first', .2, 0, 1), ('structured_check', .2, 0, 2), ('verify_revise', .3, .2, 100)]]
    sizes = {c['strategy']: 10 for c in candidates}
    assert study.select_strategy(candidates, 'primary', sizes) == 'verify_revise'
    candidates[-1]['metrics']['ceeval_full'] = .2
    assert study.select_strategy(candidates, 'primary', sizes) == 'verify_revise'
    candidates[-1]['metrics']['ceeval_pos'] = 0
    assert study.select_strategy(candidates, 'primary', sizes) == 'evidence_first'
    with pytest.raises(study.ExperimentRunError, match='Missing'):
        study.select_strategy(candidates[:2], 'primary', sizes)


def test_export_complete_text_order_terminal_failure_and_partial_names(tmp_path):
    rows = examples()
    condition = {'model': 'qwen3:8b', 'label': 'baseline', 'export': str(tmp_path / 'complete.json'),
                 'partial_export': str(tmp_path / 'partial/condition.partial.json')}
    records = [{'example_id': e.example_id, 'subset': e.split.value, 'target_inference_failure': False,
        'parsed_prediction': {'cont_error': 0, 'cont_error_expl': 'No error', 'cont_error_lines': ['discard'],
                              'contradicted_lines': ['discard'], 'parsing_failures': ['warning']}} for e in rows]
    records[1].update(target_inference_failure=True, inference_error={'type': 'OllamaTimeoutError'})
    path = study.export_condition(condition, rows, records[:2])
    assert '.partial.' in path.name and not Path(condition['export']).exists()
    path = study.export_condition(condition, rows, records)
    output = study.read(path)
    assert list(output) == ['model', 'prompt', 'total', 'failed', 'metrics', 'results']
    assert output['total'] == 100 and output['failed'] == 1
    assert output['metrics']['accuracy'] == .5
    assert list(output['results'][0]) == ['story_id', 'story', 'status', 'prediction']
    assert [r['story'] for r in output['results']] == [e.story for e in rows]
    assert output['results'][1]['prediction'] is None
    assert output['results'][1]['status'] == 'terminal_OllamaTimeoutError'
    assert output['results'][0]['prediction']['error_sentences'] == []
    assert 'EVALUATOR_SECRET' not in path.read_text(encoding='utf8')


def test_frozen_artifact_drift_is_refused(tmp_path):
    path = tmp_path / 'frozen.txt'
    path.write_text('before')
    hashes = {str(path): study.stable_file_hash(path)}
    study.verify_hashes(hashes)
    path.write_text('after')
    with pytest.raises(study.ExperimentRunError, match='changed'):
        study.verify_hashes(hashes)


def test_end_to_end_interrupt_resume_exhausted_failure_and_no_gold(tmp_path, monkeypatch):
    rows = examples()
    manifest_path = tmp_path / 'manifest.json'
    study._write_json(manifest_path, {'name': 'synthetic', 'scope_role': 'evaluation', 'records': [
        {'example_id': e.example_id, 'subset': e.split.value, 'cont_error': e.cont_error,
         'selection_hash': stable_example_id_hash(e.example_id)} for e in rows]})
    condition = {'id': 'synthetic_baseline', 'model': 'qwen3:8b', 'strategy': 'vanilla', 'label': 'baseline',
        'version': 'vanilla_v1', 'model_config': 'configs/models/three_model_main10/qwen3_8b.yaml',
        'run_id': 'synthetic_run', 'run_dir': str(tmp_path / 'runs/synthetic_run'),
        'export': str(tmp_path / 'export.json'), 'partial_export': str(tmp_path / 'export.partial.json')}
    strategy = get_benchmark_strategy('vanilla')
    freeze = {'settings': {'root': str(tmp_path), 'manifest': str(manifest_path),
                          'run_root': str(tmp_path / 'runs'), 'name': 'synthetic'},
        'conditions': [condition], 'models': {}, 'rendered_prompts': {'vanilla': [
            {'example_id': e.example_id, 'story_hash': study.sha(e.story),
             'prompt_hash': strategy.build_prompt(e).prompt_hash} for e in rows]}}
    monkeypatch.setattr(study, 'checked_freeze', lambda **_: freeze)
    monkeypatch.setattr(study, 'load_pilot_examples', lambda *_: rows)
    monkeypatch.setattr(runner, 'load_pilot_examples', lambda *_, **__: rows)
    monkeypatch.setattr(runner, 'load_all_splits', lambda *_, **__: {rows[0].split: rows})
    monkeypatch.setattr(runner, '_ollama_version', lambda *_: 'mock')
    response = {'response': '<response><decision>No continuity error found</decision></response>', 'done': True}
    first = FakeOpener([response, KeyboardInterrupt()])
    journal = Path(condition['run_dir']) / 'requests.jsonl'
    def factory(opener):
        return lambda config: study.Main100Client(config, journal, opener=opener, sleep=lambda _: None)
    with pytest.raises(OllamaInterruptedError):
        study.run(condition['id'], client_factory=factory(first), live=False)
    assert study.read(tmp_path / 'progress.json')['completed'] == 1
    second = FakeOpener([TimeoutError('synthetic')] * 3)
    study.run(condition['id'], client_factory=factory(second), live=False, max_new_examples=1)
    assert len(second.requests) == 3
    saved = study._read_rows(Path(condition['run_dir']) / 'predictions.jsonl')
    assert len(saved) == 2 and saved[1]['target_inference_failure']
    assert [h['attempt'] for h in saved[1]['attempt_history']] == [1, 2, 3, 4]
    # Compatible resume skips success and exhausted failure and makes one new call.
    third = FakeOpener([response])
    result = study.run(condition['id'], client_factory=factory(third), live=False, max_new_examples=1)
    assert result.skipped_completed == result.skipped_failed == 1 and len(third.requests) == 1
    assert study.read(tmp_path / 'progress.json')['completed'] == 3
    assert study.read(tmp_path / 'progress.json')['terminal_failures'] == 1
    for opener in (first, second, third):
        for request in opener.requests:
            body = request.data.decode('utf8')
            assert 'GOLD_' not in body and 'EVALUATOR_SECRET' not in body
    assert (Path(condition['run_dir']) / 'checkpoint_metadata.json').is_file()
    assert not Path(condition['export']).exists()


def test_preparation_has_no_generated_results_and_scope_is_disjoint():
    freeze = study.checked_freeze()
    exposure = study.read(Path(freeze['settings']['root']) / 'evaluator_only/exposure_audit.json')
    manifest = study.read(freeze['settings']['manifest'])
    ids = {r['example_id'] for r in manifest['records']}
    assert len(ids) == 100 and not ids & set(exposure['excluded_main_ids'])
    assert sum(r['cont_error'] for r in manifest['records']) == 50
    excluded_families = {exposure['family_by_id'][key] for key in exposure['direct_exposure_evidence']}
    assert not {r['source_family_id'] for r in manifest['records']} & excluded_families
    inputs = study._read_rows(Path(freeze['settings']['inputs']))
    assert all(list(row) == ['story_id', 'story'] for row in inputs)
    assert len(freeze['conditions']) == 6 and freeze['expected_records'] == 600
    assert all(c['strategy'] in ('vanilla', 'verify_revise') for c in freeze['conditions'])
