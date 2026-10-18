"""Separate, bounded technical recovery; prepare/status are entirely offline."""
import argparse
import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from fyp.experiments import generated_stories as g

CONFIG = Path('configs/experiments/generated_stories_recovery_v2.json')


def hashes(root):
    return {p.as_posix(): g.stable_file_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name != 'execution.lock'}


def verify(values):
    for path, expected in values.items():
        if g.stable_file_hash(Path(path)) != expected:
            raise RuntimeError('Preserved artifact changed: ' + path)


def checkpoint(root, stem, pair, stage):
    path = g.stage_path(root, stem, pair, stage)
    return g.read(path) if path.exists() else None


def diagnosis(cp):
    if cp is None:
        return {'parse': 'not_run', 'reason': 'Stage A failed; B skipped'}
    raw = cp.get('raw_response') or {}
    text = raw.get('response')
    detail = {'parse': cp.get('error') or 'valid', 'reason': cp.get('error'),
              'done': raw.get('done'), 'done_reason': raw.get('done_reason'),
              'eval_count': raw.get('eval_count')}
    if isinstance(text, str):
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            detail.update(json_error=str(exc), error_excerpt=text[max(0, exc.pos-50):exc.pos+50])
            try:
                json.loads(text, strict=False)
                detail['reason'] = 'Unescaped control characters in JSON string; strict parser correctly rejects'
            except json.JSONDecodeError as relaxed:
                detail['secondary_json_error'] = str(relaxed)
                if not text.rstrip().endswith('}'):
                    detail['reason'] = 'Unescaped newlines and missing closing object brace; additional unescaped quotes may exist'
                else:
                    detail['reason'] = 'Adjacent quoted strings without JSON delimiter/structure'
    return detail


def audit_source(settings, freeze):
    root = Path(settings['root'])
    inventory = []
    counts = {}
    plan = []
    for model, entry in settings['models'].items():
        stem = entry['stem']
        events = [json.loads(line) for line in (root/'runs'/stem/'requests.jsonl').read_text(encoding='utf8').splitlines()]
        starts = {e['request_id']: e for e in events if e['event'] == 'started'}
        finishes = {e['request_id']: e for e in events if e['event'] == 'finished'}
        assert len(starts) == len(finishes) and starts.keys() == finishes.keys(), 'Unresolved requests; investigate first'
        records = []
        count = Counter({key: 0 for key in ('pairs_ok', 'pairs_failed', 'A_ok', 'A_failed', 'B_ok', 'B_failed', 'B_skipped')})
        for i, w in enumerate(g.read(settings['writing_prompts']), 1):
            pair = f'{stem}_{i:03d}'
            saved = [checkpoint(root, stem, pair, stage) for stage in ('A', 'B')]
            row = g.pair_record(pair, w, *saved)
            assert row is not None, 'Original run still pending'
            records.append(row)
            count['pairs_ok' if row['generation_status'] == 'ok' else 'pairs_failed'] += 1
            stages = {}
            for stage, cp in zip(('A', 'B'), saved):
                attempts = [e for e in starts.values() if e['pair_id'] == pair and e['stage'] == stage]
                raw_paths = []
                if cp:
                    assert cp['freeze_hash'] == g.digest(freeze)
                    assert len(cp['attempt_history']) == len(attempts)
                    count[stage + ('_ok' if cp['status'] == 'ok' else '_failed')] += 1
                    for start in attempts:
                        end = finishes[start['request_id']]
                        path = root/'runs'/stem/'raw_http'/f'{pair}_{stage}_{start["attempt"]}.bin'
                        if path.exists():
                            assert g.read(path) == end['raw_response'] == cp['raw_response']
                            raw_paths.append(path.as_posix())
                        expected = g.prompt(settings, stage, w, g.read(settings['editing_instructions'])[i-1], saved[0]['fields']['original_story'])
                        assert start['application_request']['body']['prompt'] == expected
                        assert start['application_request']['body'] == g.load_model_config(entry['config']).effective_ollama_request_fields() | {'prompt': expected}
                else:
                    count[stage+'_skipped'] += 1
                stages[stage] = {'status': cp['status'] if cp else 'skipped',
                    'raw_checkpoint': bool(cp and cp.get('raw_response')),
                    'raw_http': raw_paths, 'starts': len(attempts),
                    'unused_dispatch_slots': 4-len(attempts),
                    'executable_original_retries': 0,
                    'warnings': cp['warnings'] if cp else [], **diagnosis(cp)}
            inventory.append({'model': model, 'pair_id': pair, **stages})
            if row['generation_status'] != 'ok':
                plan.append({'model': model, 'stem': stem, 'pair_id': pair,
                             'index': i-1, 'execute': ['B'] if saved[0]['status'] == 'ok' else ['A', 'B'],
                             'B_dependency': 'only after selected A is structurally successful'})
        exported = [json.loads(x) for x in (root/'exports'/f'{stem}_generated_pairs.jsonl').read_text(encoding='utf8').splitlines()]
        assert records == exported, 'Export/checkpoint disagreement'
        count['dispatch_starts'] = len(starts)
        count['received'] = sum(e['status'] == 'response_received' for e in finishes.values())
        counts[model] = dict(count)
    return {'counts': counts, 'inventory': inventory, 'plan': plan}


def prepare():
    cfg = g.read(CONFIG)
    root = Path(cfg['root'])
    if (root/'preparation.json').exists():
        return status()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Unfrozen nonempty recovery directory; inspect before proceeding')
    settings, freeze = g.frozen()
    assert g.read(cfg['source_config']) == settings
    predecessor = Path(cfg['predecessor_root'])
    if any(predecessor.glob('runs/**/requests.jsonl')) or any(predecessor.glob('checkpoints/**/*.json')):
        raise RuntimeError('Prior recovery contains attempts/results; consolidate rather than allocate new attempts')
    audit = audit_source(settings, freeze)
    root.mkdir(parents=True, exist_ok=True)
    source_hashes = hashes(Path(settings['root']))
    sources = {**freeze['hashes'], **source_hashes, **hashes(predecessor),
               CONFIG.as_posix(): g.stable_file_hash(CONFIG),
               Path(__file__).as_posix(): g.stable_file_hash(Path(__file__))}
    g.save(root/'preparation.json', {'config': cfg, 'source_settings': settings,
           'source_freeze': freeze, 'hashes': sources, 'audit': audit})
    return status()


def loaded():
    cfg = g.read(CONFIG)
    manifest = g.read(Path(cfg['root'])/'preparation.json')
    assert cfg == manifest['config']
    verify(manifest['hashes'])
    predecessor = Path(cfg['predecessor_root'])
    if any(predecessor.glob('runs/**/requests.jsonl')) or any(predecessor.glob('checkpoints/**/*.json')):
        raise RuntimeError('Prior recovery now contains attempts/results; refusing duplicate recovery')
    settings = copy.deepcopy(manifest['source_settings'])
    settings.update(root=cfg['root'], name=cfg['name'])
    inspection = copy.deepcopy(manifest['source_freeze']['inspection'])
    inspection['ollama_version'] = cfg['ollama_version']
    freeze = {'inspection': inspection, 'manifest_hash': g.digest(manifest)}
    return cfg, manifest, settings, freeze


def selected_stages(manifest, settings, freeze, model, pair):
    stem = settings['models'][model]['stem']
    original_root = Path(manifest['source_settings']['root'])
    selected = []
    provenance = []
    for stage in ('A', 'B'):
        old = checkpoint(original_root, stem, pair, stage)
        new = checkpoint(Path(settings['root']), stem, pair, stage)
        if old and old['status'] == 'ok':
            value, source = old, 'original'
            if new:
                raise RuntimeError('Recovery unexpectedly duplicates successful stage')
        else:
            value, source = new, 'recovery' if new else 'pending'
            if new and new['freeze_hash'] != g.digest(freeze):
                raise RuntimeError('Incompatible recovery checkpoint')
        selected.append(value)
        provenance.append({'stage': stage, 'source': source,
                           'checkpoint': g.stage_path(original_root if source == 'original' else Path(settings['root']), stem, pair, stage).as_posix() if value else None})
    return selected, provenance


def status():
    cfg, manifest, settings, freeze = loaded()
    root = Path(cfg['root'])
    summaries, provenance = [], []
    with g.experiment_lock(root):
        for model, entry in settings['models'].items():
            records, ready = [], []
            for i, w in enumerate(g.read(settings['writing_prompts']), 1):
                pair = f'{entry["stem"]}_{i:03d}'
                (a, b), sources = selected_stages(manifest, settings, freeze, model, pair)
                row = g.pair_record(pair, w, a, b)
                if row:
                    records.append(row)
                    if row['generation_status'] == 'ok':
                        ready.append(row)
                journal = root/'runs'/entry['stem']/'requests.jsonl'
                events = [json.loads(x) for x in journal.read_text(encoding='utf8').splitlines()] if journal.exists() else []
                for stage, value, source in zip(('A', 'B'), (a,b), sources):
                    starts = [e for e in events if e['event'] == 'started' and e['pair_id'] == pair and e['stage'] == stage]
                    source['recovery_dispatch_starts'] = len(starts)
                    source['remaining_recovery_budget'] = 0 if value else cfg['max_retries']+1-len(starts)
                    source['status'] = value['status'] if value else ('blocked_by_A' if stage == 'B' and a and a['status'] != 'ok' else 'pending')
                provenance.append({'pair_id': pair, 'stages': sources})
            suffix = '' if len(records) == 10 else '.partial'
            g.atomic(root/'exports'/f'{entry["stem"]}_generated_pairs{suffix}.jsonl', ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in records))
            g.atomic(root/'human_review'/f'{entry["stem"]}_complete_pairs.jsonl', ''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in ready))
            summaries.append({'model': model, 'complete_pairs': len(ready),
                              'terminal_recovery_failures': len(records)-len(ready), 'pending_pairs': 10-len(records)})
        g.save(root/'selection_provenance.json', provenance)
        g.save(root/'status.json', summaries)
        g.save(root/'original_inventory.json', manifest['audit'])
        lines = ['# Original per-pair inventory', '',
                 'All original terminal stages are closed: unused dispatch slots are not permission to retry. B slots are unused because A failed. Raw columns show checkpoint/HTTP availability. No saved-response parser-bug recoveries qualified.', '',
                 '| Pair | A | B | Raw A/B | Parse/failure reason | Unused slots A/B | Executable original retries |',
                 '|---|---|---|---|---|---|---|']
        for row in manifest['audit']['inventory']:
            a,b = row['A'],row['B']
            raw = lambda s: 'yes/yes' if s['raw_checkpoint'] and s['raw_http'] else 'none'
            lines.append(f"| {row['pair_id']} | {a['status']} | {b['status']} | {raw(a)}; {raw(b)} | {a['reason'] or b['reason'] or 'valid A/B'} | {a['unused_dispatch_slots']}/{b['unused_dispatch_slots']} | 0 |")
        g.atomic(root/'original_inventory.md', '\n'.join(lines)+'\n')
    return summaries


class RecoveryClient(g.CapturingClient):
    def __init__(self, config, path, *, recovery, **kwargs):
        super().__init__(replace(config, runtime=replace(config.runtime, max_retries=recovery['max_retries'])), path, **kwargs)
        self.recovery = recovery

    def build_generate_payload(self, request):
        return {**super().build_generate_payload(request), 'format': self.recovery['request_format']}


def canonical_inspection(inspection):
    """Ignore parameter-key presentation order; preserve each key's value order."""
    result = copy.deepcopy(inspection)
    for model in result.get('models', {}).values():
        identity = model.get('identity', {})
        parameters = identity.get('parameters')
        if isinstance(parameters, str):
            grouped = {}
            for line in parameters.splitlines():
                if not line.strip():
                    continue
                parts = line.strip().split(None, 1)
                if len(parts) != 2:
                    raise RuntimeError('Unrecognized model parameter metadata: ' + repr(line))
                key, value = parts
                grouped.setdefault(key, []).append(value)
            identity['parameters'] = grouped
    return result


def validate_inspection(current, expected):
    if current.get('error'):
        raise RuntimeError('Cannot inspect local Ollama; no inference sent: ' + current['error'])
    differences = []
    def compare(left, right, path=''):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                compare(left.get(key), right.get(key), path + '/' + key)
        elif left != right:
            differences.append(path)
    compare(canonical_inspection(current), canonical_inspection(expected))
    if differences:
        raise RuntimeError('Live metadata differs at ' + ', '.join(differences) + '; no inference sent')


def run(model):
    cfg, manifest, settings, freeze = loaded()
    if model not in settings['models']:
        raise ValueError('Unscheduled model')
    # Metadata only. Any identity/runtime change stops before dispatch.
    current = g.inspect_models(manifest['source_settings'], manifest['source_freeze']['stage_a_bounds'])
    validate_inspection(current, freeze['inspection'])
    ws, es = g.read(settings['writing_prompts']), g.read(settings['editing_instructions'])
    factory = lambda config, path: RecoveryClient(config, path, recovery=cfg)
    try:
        with g.experiment_lock(Path(settings['root'])):
            for item in manifest['audit']['plan']:
                if item['model'] != model:
                    continue
                pair, i = item['pair_id'], item['index']
                (a,b), _ = selected_stages(manifest, settings, freeze, model, pair)
                if a is None:
                    a = g.execute_stage(settings, freeze, model, pair, 'A', g.prompt(settings, 'A', ws[i], es[i]), client_factory=factory)
                if a['status'] == 'ok' and b is None:
                    original = a['fields']['original_story']
                    g.execute_stage(settings, freeze, model, pair, 'B', g.prompt(settings, 'B', ws[i], es[i], original), original, factory)
    finally:
        status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'status', 'check', 'run'])
    parser.add_argument('--model')
    args = parser.parse_args()
    if args.command == 'check':
        cfg, manifest, settings, freeze = loaded()
        current = g.inspect_models(manifest['source_settings'], manifest['source_freeze']['stage_a_bounds'])
        validate_inspection(current, freeze['inspection'])
        print('Live metadata matches recovery v2 (Ollama ' + cfg['ollama_version'] + '); no inference sent.')
    elif args.command == 'run':
        if not args.model:
            parser.error('run requires --model')
        run(args.model)
    else:
        print(json.dumps(prepare() if args.command == 'prepare' else status(), indent=2))


if __name__ == '__main__':
    main()
