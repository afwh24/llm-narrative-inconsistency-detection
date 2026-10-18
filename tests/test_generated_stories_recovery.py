import json
from pathlib import Path

import pytest

from fyp.experiments import generated_stories as g
from fyp.experiments import generated_stories_recovery as r
from test_models import FakeOpener


def test_malformed_outputs_are_not_parser_bugs():
    for text in ['{"story":"first\nsecond"}', '{"story":"complete"', '{"story":"first" "second"}']:
        assert g.parse(text, 'A')[1] == 'response_not_json'
    assert g.parse(json.dumps({'story': 'first\nsecond'}), 'A')[1] is None


def test_revised_payload_and_finite_budget(tmp_path, monkeypatch):
    settings = g.read(g.CONFIG)
    settings['root'] = str(tmp_path)
    freeze = {'inspection': {'models': {'qwen3:8b': {'identity': {'digest': 'fake'}}}}}
    monkeypatch.setattr(g, 'context_budget', lambda *args: {})
    cfg = {'max_retries': 1, 'request_format': 'json'}
    opener = FakeOpener([TimeoutError('test')] * 2)
    factory = lambda config, path: r.RecoveryClient(config, path, recovery=cfg, opener=opener, sleep=lambda _: None)
    cp = g.execute_stage(settings, freeze, 'qwen3:8b', 'qwen3_8b_002', 'A', 'same', client_factory=factory)
    assert len(opener.requests) == 2
    assert len(cp['attempt_history']) == 2
    assert cp['status'] == 'terminal_technical_A_OllamaTimeoutError'
    assert json.loads(opener.requests[0].data)['format'] == 'json'
    g.execute_stage(settings, freeze, 'qwen3:8b', 'qwen3_8b_002', 'A', 'same', client_factory=factory)
    assert len(opener.requests) == 2


def test_original_success_precedes_recovery_and_reuses_original(tmp_path):
    settings = g.read(g.CONFIG)
    settings['root'] = str(tmp_path/'recovery')
    manifest = {'source_settings': {'root': str(tmp_path/'original')}}
    freeze = {'synthetic': True}
    pair, stem = 'qwen3_8b_001', 'qwen3_8b'
    old = {'status': 'ok', 'fields': {'original_story': 'untouched original'}}
    g.save(g.stage_path(tmp_path/'original', stem, pair, 'A'), old)
    failed = {'status': 'terminal_parsing_B'}
    g.save(g.stage_path(tmp_path/'original', stem, pair, 'B'), failed)
    selected, sources = r.selected_stages(manifest, settings, freeze, 'qwen3:8b', pair)
    assert selected == [old, None] and sources[0]['source'] == 'original'
    new = {'status': 'ok', 'freeze_hash': g.digest(freeze)}
    g.save(g.stage_path(tmp_path/'recovery', stem, pair, 'B'), new)
    assert r.selected_stages(manifest, settings, freeze, 'qwen3:8b', pair)[0] == [old, new]
    g.save(g.stage_path(tmp_path/'recovery', stem, pair, 'A'), new)
    with pytest.raises(RuntimeError, match='duplicates'):
        r.selected_stages(manifest, settings, freeze, 'qwen3:8b', pair)


def test_saved_response_resume_never_dispatches_again(tmp_path, monkeypatch):
    settings = g.read(g.CONFIG)
    settings['root'] = str(tmp_path)
    freeze = {'inspection': {'models': {'qwen3:8b': {'identity': {'digest': 'fake'}}}}}
    monkeypatch.setattr(g, 'context_budget', lambda *args: {})
    opener = FakeOpener([{'response': '{"story":"preserved"}', 'done': True}])
    factory = lambda config, path: r.RecoveryClient(config, path, recovery={'max_retries': 1, 'request_format': 'json'}, opener=opener)
    cp = g.execute_stage(settings, freeze, 'qwen3:8b', 'qwen3_8b_002', 'A', 'same', client_factory=factory)
    g.stage_path(tmp_path, 'qwen3_8b', 'qwen3_8b_002', 'A').unlink()
    again = g.execute_stage(settings, freeze, 'qwen3:8b', 'qwen3_8b_002', 'A', 'same', client_factory=factory)
    assert again['fields'] == cp['fields'] and len(opener.requests) == 1


def test_source_audit_matches_frozen_exports():
    settings, freeze = g.frozen()
    audit = r.audit_source(settings, freeze)
    assert sum(c['pairs_ok'] for c in audit['counts'].values()) == 11
    assert len(audit['plan']) == 19
    assert all(item['execute'] == ['A', 'B'] for item in audit['plan'])


def test_runner_reuses_compatible_A_and_skips_complete_pair(tmp_path, monkeypatch):
    settings = g.read(g.CONFIG)
    settings['root'] = str(tmp_path/'recovery')
    Path(settings['root']).mkdir()
    freeze = {'inspection': {'models': {'qwen3:8b': {'identity': {'digest': 'fake'}}}}}
    manifest = {'source_settings': {'root': str(tmp_path/'original')},
                'source_freeze': {'stage_a_bounds': []},
                'audit': {'plan': [{'model': 'qwen3:8b', 'pair_id': 'qwen3_8b_001', 'index': 0}]}}
    old = {'status': 'ok', 'fields': {'original_story': 'Saved original verbatim.'}}
    g.save(g.stage_path(tmp_path/'original', 'qwen3_8b', 'qwen3_8b_001', 'A'), old)
    cfg = {'max_retries': 1, 'request_format': 'json'}
    monkeypatch.setattr(r, 'loaded', lambda: (cfg, manifest, settings, freeze))
    monkeypatch.setattr(r, 'status', lambda: None)
    monkeypatch.setattr(g, 'inspect_models', lambda *args: freeze['inspection'])
    monkeypatch.setattr(g, 'context_budget', lambda *args: {})
    raw = {'response': json.dumps({'modified_story': 'Saved original changed.', 'intended_contradiction': {
        'error_sentence': 'changed', 'contradicted_sentence': 'original', 'explanation': 'Claim'}}), 'done': True}
    opener = FakeOpener([raw])
    real_client = r.RecoveryClient
    monkeypatch.setattr(r, 'RecoveryClient', lambda config, path, recovery: real_client(config, path, recovery=recovery, opener=opener))
    r.run('qwen3:8b')
    r.run('qwen3:8b')
    assert len(opener.requests) == 1
    assert 'Saved original verbatim.' in json.loads(opener.requests[0].data)['prompt']
    assert not g.stage_path(Path(settings['root']), 'qwen3_8b', 'qwen3_8b_001', 'A').exists()
