"""Frozen main100 comparison: preparation is offline; only `run` dispatches inference.

Uses the established runner, request journal, prompts, parser and evaluator unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from fyp.data.loader import load_split
from fyp.data.pilot import load_pilot_examples, load_pilot_manifest, stable_example_id_hash, stable_file_hash
from fyp.evaluation.evaluator import _extract_prediction, evaluate_predictions
from fyp.experiments import baseline_pilot, prompting_pilot
from fyp.experiments.prompting_outputs import _read_rows, _write_json
from fyp.experiments.runner import (DATASET_IDENTITY, DATASET_VERSION, ExperimentRunError,
    RunConfig, _cache_key_material, _dependency_versions, _fingerprint, _git_metadata,
    _parsed_to_record, run_experiment)
from fyp.models.registry import load_model_config
from fyp.models.ollama_client import OllamaError, OllamaInterruptedError
from fyp.models.request_journal import JournaledOllamaClient, OllamaPendingError
from fyp.prompting.strategies import get_benchmark_strategy
from fyp.prompting.parser import parse_official_response

CONFIG = Path('configs/experiments/main100_comparison_v1.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest().upper()


def select_examples(examples, eligible_ids, per_class, salt):
    selected = []
    for label in (0, 1):
        eligible = [e for e in examples if e.example_id in eligible_ids and e.cont_error == label]
        if len(eligible) < per_class:
            raise ExperimentRunError(f'Insufficient eligible class {label}: {len(eligible)} < {per_class}')
        selected.extend(sorted(eligible, key=lambda e: (sha(salt + '\n' + e.example_id), e.example_id))[:per_class])
    # Mix labels in final deterministic order; labels never go into detector inputs.
    return sorted(selected, key=lambda e: (sha(salt + '\n' + e.example_id), e.example_id))


def select_strategy(conditions, primary_model, template_sizes):
    candidates = [c for c in conditions if c['model'] == primary_model and c['strategy'] != 'vanilla']
    if {c['strategy'] for c in candidates} != {'evidence_first', 'structured_check', 'verify_revise'}:
        raise ExperimentRunError('Missing or incompatible primary pilot conditions; no winner invented')
    return min(candidates, key=lambda c: (-c['metrics']['ceeval_full'], -c['metrics']['ceeval_pos'],
        c['failed'], c['runtime']['sum_seconds'], template_sizes[c['strategy']], c['strategy']))['strategy']


def audit_pilot():
    """Reuse the existing read-only raw-response/configuration/export audit."""
    spec = importlib.util.spec_from_file_location('main100_pilot_audit', 'scripts/analyse_prompting_main10.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pilot, base = prompting_pilot.load_settings()
    manifest, examples = prompting_pilot.validate_scope(pilot, base)
    prepared = read(Path(pilot['preparation_dir']) / 'metadata.json')
    conditions = [module.audit_condition(model, strategy, pilot, base, manifest, examples, prepared)
                  for model in pilot['models'] for strategy in module.METHODS]
    return [{k: v for k, v in row.items() if k != 'stories'} for row in conditions]


def condition_matrix(settings, pilot, base):
    conditions = []
    for model, entry in base['models'].items():
        stem = entry['export_stem'].removesuffix('_baseline')
        for strategy in ('vanilla', settings['selected_strategy']):
            label = 'baseline' if strategy == 'vanilla' else strategy
            key = f'{stem}_{label}'
            conditions.append({'id': key, 'model': model, 'strategy': strategy, 'label': label,
                'version': pilot['strategies'][strategy]['version'], 'model_config': entry['config'],
                'run_id': settings['name'] + '__' + key,
                'run_dir': f"{settings['run_root']}/{settings['name']}__{key}",
                'export': f"{settings['export_dir']}/{key}.json",
                'partial_export': f"{settings['export_dir']}/partial/{key}.partial.json"})
    return conditions


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if not Path(name).is_file() or stable_file_hash(Path(name)) != expected:
            raise ExperimentRunError(f'Frozen artifact changed or missing: {name}')


def prepare():
    settings = read(CONFIG)
    root = Path(settings['root'])
    if (root / 'preparation.json').exists():
        raise ExperimentRunError('Experiment already frozen; use check, or a new experiment version')
    pilot, base = prompting_pilot.load_settings()
    conditions = audit_pilot()
    selected = select_strategy(conditions, settings['primary_model'],
                              {k: Path(v['path']).stat().st_size for k, v in pilot['strategies'].items()})
    if selected != settings['selected_strategy']:
        raise ExperimentRunError('Configured strategy differs from audited pilot selection')
    audit_path = root / 'evaluator_only/exposure_audit.json'
    exposure = read(audit_path)
    verify_hashes(exposure['source_hashes'])
    examples = select_examples(load_split('flawed_fictions'), set(exposure['eligible_main_ids']),
                               settings['sample_per_class'], settings['selection_salt'])
    manifest_rows = [{'example_id': e.raw_record['example_id'], 'subset': e.split.value,
        'cont_error': e.cont_error, 'selection_hash': stable_example_id_hash(e.example_id),
        'rank_hash': sha(settings['selection_salt'] + '\n' + e.example_id),
        'story_hash': sha(e.story), 'official_id_type': type(e.raw_record['example_id']).__name__,
        'source_family_id': exposure['family_by_id'][e.example_id],
        'official_source_family_id': None} for e in examples]
    manifest = {'name': settings['name'], 'scope_role': 'evaluation',
        'dataset': {'identity': DATASET_IDENTITY, 'version': DATASET_VERSION},
        'raw_data_path': base['raw_data_path'], 'raw_data_hash': stable_file_hash(Path(base['raw_data_path'])),
        'selection_method': 'Within each gold class, choose lowest SHA256(salt + newline + official ID), then globally order by the same hash; ID breaks ties. No model scores used.',
        'selection_salt': settings['selection_salt'], 'class_counts': {'positive': 50, 'negative': 50},
        'exclusions': str(audit_path), 'exclusions_hash': stable_file_hash(audit_path), 'records': manifest_rows}
    _write_json(Path(settings['manifest']), manifest)
    inputs = [{'story_id': e.raw_record['example_id'], 'story': e.raw_record['story']} for e in examples]
    Path(settings['inputs']).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in inputs), encoding='utf8')
    _write_json(root / 'evaluator_only/gold.json', [dict(e.raw_record) for e in examples])
    _write_json(root / 'pilot_selection.json', {'selected': selected, 'conditions': conditions,
        'rule': 'Primary Qwen CEEval-Full, then Pos, fewer terminal failures, lower measured total latency, shorter existing template (simplicity proxy), name for determinism.',
        'caveat': 'Pragmatic development-pilot choice; selected Qwen Full 0.30 is below baseline 0.50. Evidence-first accuracy 0.70 exceeds selected 0.60. No claim of superiority.'})
    rendered, templates = {}, {}
    for name in ('vanilla', selected):
        entry = pilot['strategies'][name]
        source = Path(entry['path'])
        copy = Path(settings['template_copy_dir']) / source.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        if copy.exists() and copy.read_bytes() != source.read_bytes():
            raise ExperimentRunError('Frozen template copy differs')
        copy.write_bytes(source.read_bytes())
        templates[name] = {**entry, 'copy': copy.as_posix()}
        strategy = get_benchmark_strategy(name, version=entry['version'])
        rendered[name] = [{'example_id': e.example_id, 'story_hash': sha(e.story),
            'prompt_hash': strategy.build_prompt(e).prompt_hash,
            'prompt_utf8_bytes': len(strategy.build_prompt(e).prompt.encode('utf8'))} for e in examples]
    inspection = read(root / 'model_inspection.json')
    prior = read(Path(pilot['preparation_dir']) / 'metadata.json')['provenance']
    capacity = [r for rows in rendered.values() for r in rows]
    budgets = {}
    for model in base['models']:
        current = inspection['models'].get(model, {})
        if not current.get('available') or current['identity'] != prior['models'][model]['identity']:
            raise ExperimentRunError('Installed identity differs from verified pilot: ' + model)
        if inspection['ollama_version'] != prior['ollama_version']:
            raise ExperimentRunError('Ollama runtime differs from pilot')
        budgets[model] = baseline_pilot.context_budget(base, capacity, current['identity'])
        config = load_model_config(base['models'][model]['config'])
        generation = config.generation.snapshot()
        if any(generation[k] != v for k, v in base['generation'].items()):
            raise ExperimentRunError('Generation settings differ')
        fields = config.effective_ollama_request_fields()
        if (model == 'qwen3:8b' and fields.get('think') is not False) or (model != 'qwen3:8b' and 'think' in fields):
            raise ExperimentRunError('Thinking configuration mismatch')
    sources = list(Path('src/fyp').rglob('*.py')) + [CONFIG, Path(settings['pilot_configuration']),
        Path(settings['baseline_configuration']), Path(settings['manifest']), Path(settings['inputs']),
        root / 'evaluator_only/gold.json', audit_path, root / 'pilot_selection.json', root / 'model_inspection.json',
        Path('scripts/analyse_prompting_main10.py')]
    sources += list(Path('data/raw/flawed-fictions').glob('*.parquet'))
    sources += [Path(e['config']) for e in base['models'].values()]
    sources += [Path(e[key]) for e in templates.values() for key in ('path', 'copy')]
    import nltk
    tokenizer_dir = Path(str(nltk.data.find('tokenizers/punkt_tab/english/')))
    sources += list(tokenizer_dir.glob('*.txt')) + list(tokenizer_dir.glob('*.tab'))
    freeze = {'status': 'PREPARED_NO_INFERENCE', 'prepared_at': datetime.now(timezone.utc).isoformat(),
        'settings': settings, 'conditions': condition_matrix(settings, pilot, base), 'templates': templates,
        'manifest_hash': stable_file_hash(Path(settings['manifest'])), 'rendered_prompts': rendered,
        'models': inspection['models'], 'ollama_version': inspection['ollama_version'], 'context_budgets': budgets,
        'dependencies': _dependency_versions(('nltk', 'pyarrow')), 'git': _git_metadata(),
        'source_hashes': {p.as_posix(): stable_file_hash(p) for p in sorted(set(sources))},
        'expected_records': 600, 'generations_during_preparation': 0}
    _write_json(root / 'preparation.json', freeze)
    update_progress(freeze)
    return {'status': freeze['status'], 'selected': selected, 'eligibility': exposure['counts'],
            'manifest_hash': freeze['manifest_hash'], 'context_budgets': budgets}


def checked_freeze(live=False):
    settings = read(CONFIG)
    freeze = read(Path(settings['root']) / 'preparation.json')
    if settings != freeze['settings']:
        raise ExperimentRunError('Configuration changed after preparation')
    verify_hashes(freeze['source_hashes'])
    if _dependency_versions(('nltk', 'pyarrow')) != freeze['dependencies']:
        raise ExperimentRunError('Evaluation dependencies changed')
    manifest = load_pilot_manifest(settings['manifest'])
    examples = load_pilot_examples(manifest)
    inputs = _read_rows(Path(settings['inputs']))
    if len(examples) != 100 or len({e.example_id for e in examples}) != 100:
        raise ExperimentRunError('Expected exactly 100 unique examples')
    if [e.raw_record['story'] for e in examples] != [r['story'] for r in inputs] or [e.example_id for e in examples] != [r['story_id'] for r in inputs]:
        raise ExperimentRunError('Full detector story/order mismatch')
    if live:
        capacity = [r for rows in freeze['rendered_prompts'].values() for r in rows]
        inspection = baseline_pilot.inspect_models(read(settings['baseline_configuration']), capacity)
        if inspection.get('models') != freeze['models'] or inspection.get('ollama_version') != freeze['ollama_version']:
            raise ExperimentRunError('Ollama identity/runtime/capacity unavailable or changed; no inference sent')
    return freeze


def validated_records(condition, examples, freeze):
    records = _read_rows(Path(condition['run_dir']) / 'predictions.jsonl')
    expected = {e.example_id: e for e in examples}
    ids = [r['example_id'] for r in records]
    if len(ids) != len(set(ids)) or not set(ids) <= expected.keys():
        raise ExperimentRunError('Duplicate/unexpected checkpoint IDs')
    prompts = {r['example_id']: r for r in freeze['rendered_prompts'][condition['strategy']]}
    model_config = load_model_config(condition['model_config'])
    strategy = get_benchmark_strategy(condition['strategy'], version=condition['version'])
    config = RunConfig(model=condition['model'], strategy=condition['strategy'],
                       subsets=(examples[0].split,), prompt_version=condition['version'])
    journal = _read_rows(Path(condition['run_dir']) / 'requests.jsonl')
    for row in records:
        prompt = strategy.build_prompt(expected[row['example_id']])
        cache_key = _fingerprint(_cache_key_material(expected[row['example_id']], config,
                                                    model_config, prompt, prompt.stages[0]))
        if (row['model'] != condition['model'] or row['benchmark_strategy'] != condition['strategy']
            or row['run_id'] != condition['run_id'] or row['subset'] != 'flawed_fictions'
            or row['input_hash'] != sha(expected[row['example_id']].story)
            or row['prompt_hash'] != prompts[row['example_id']]['prompt_hash']
            or row['prompt_version'] != condition['version'] or row['cache_key'] != cache_key
            or row['prompt_template_hash'] != prompt.template_hash):
            raise ExperimentRunError('Incompatible checkpoint record')
        starts = [event for event in journal if event.get('cache_key') == cache_key and event['event'] == 'started']
        finishes = [event for event in journal if event.get('cache_key') == cache_key and event['event'] == 'finished']
        history = row.get('attempt_history', [])
        if (not starts or len(starts) != len(history) or len(starts) > model_config.runtime.max_retries + 1
            or not finishes or finishes[-1]['status'] not in {'error', 'response_received'}):
            raise ExperimentRunError('Checkpoint lacks a resolved, finite request history')
        if not row['target_inference_failure']:
            if row['parsed_prediction'] != _parsed_to_record(parse_official_response(row['raw_response'])):
                raise ExperimentRunError('Checkpoint parser output differs from frozen parser')
            if finishes[-1]['raw_response']['response'] != row['raw_response']:
                raise ExperimentRunError('Checkpoint response differs from request journal')
    indexed = {r['example_id']: r for r in records}
    return [indexed[e.example_id] for e in examples if e.example_id in indexed]


def export_condition(condition, examples, records):
    complete = len(records) == len(examples)
    selected = {r['example_id'] for r in records}
    present = [e for e in examples if e.example_id in selected]
    report = evaluate_predictions(records, dataset={examples[0].split: present})
    results = []
    for e, row in zip(present, records):
        parsed, flags = _extract_prediction(row)
        if flags['target_inference_failure']:
            kind = (row.get('inference_error') or {}).get('type', 'inference_error')
            status, prediction = 'terminal_' + kind, None
        else:
            if parsed is None:
                raise ExperimentRunError('Successful checkpoint lacks official-style parsed prediction')
            positive = bool(parsed.cont_error)
            status, prediction = 'ok', {'has_error': positive,
                'error_sentences': list(parsed.cont_error_lines) if positive else [],
                'contradicted_sentences': list(parsed.contradicted_lines) if positive else [],
                'explanation': parsed.cont_error_expl}
        results.append({'story_id': e.raw_record['example_id'], 'story': e.raw_record['story'],
                        'status': status, 'prediction': prediction})
    output = {'model': condition['model'], 'prompt': condition['label'], 'total': len(results),
        'failed': report.metrics['failure_counts']['target_inference_failure'],
        'metrics': {**report.metrics['classification_metrics'],
                    **{k: report.metrics['localization_metrics'][k] for k in ('ceeval_full', 'ceeval_pos')}},
        'results': results}
    path = Path(condition['export'] if complete else condition['partial_export'])
    _write_json(path, output)
    return path


def update_progress(freeze):
    settings = freeze['settings']
    examples = load_pilot_examples(load_pilot_manifest(settings['manifest']))
    conditions = []
    for condition in freeze['conditions']:
        records = validated_records(condition, examples, freeze)
        failed = sum(bool(r.get('target_inference_failure')) for r in records)
        output = str(export_condition(condition, examples, records)) if records else None
        conditions.append({'condition': condition['id'], 'pending': 100 - len(records),
            'completed': len(records), 'successful': len(records) - failed, 'terminal_failures': failed,
            'status': 'COMPLETE' if len(records) == 100 else ('PARTIAL' if records else 'PENDING'),
            'output': output})
    _write_json(Path(settings['root']) / 'progress.json', {'expected': 600,
        'completed': sum(c['completed'] for c in conditions), 'pending': sum(c['pending'] for c in conditions),
        'terminal_failures': sum(c['terminal_failures'] for c in conditions), 'conditions': conditions})


class Main100Client(JournaledOllamaClient):
    """Unexpected local errors cannot become invented terminal model failures."""
    def generate(self, request):
        try:
            return super().generate(request)
        except (OllamaInterruptedError, OllamaError):
            raise
        except Exception as exc:
            raise OllamaPendingError(f'Local execution error; story remains pending: {exc}') from exc


@contextmanager
def experiment_lock(root):
    """OS releases this lock on process death; stale file does not block resume."""
    path = Path(root) / 'execution.lock'
    with path.open('a+b') as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ExperimentRunError('Another main100 process holds the execution lock') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def checkpoint(run_dir, freeze):
    for name in ('predictions.jsonl', 'errors.jsonl'):
        path = run_dir / name
        if path.exists():
            with path.open('ab') as handle:
                handle.flush()
                os.fsync(handle.fileno())
    _write_json(run_dir / 'checkpoint_metadata.json', read(run_dir / 'metadata.json'))
    update_progress(freeze)


def run(condition_id, *, client_factory=None, live=True, max_new_examples=None):
    freeze = checked_freeze(live=live)
    settings = freeze['settings']
    condition = next((c for c in freeze['conditions'] if c['id'] == condition_id), None)
    if condition is None:
        raise ExperimentRunError('Unknown condition ID')
    with experiment_lock(settings['root']):
        run_dir = Path(condition['run_dir'])
        # Recover only derived metadata from the last durable checkpoint; raw data is untouched.
        if run_dir.exists() and (run_dir / 'checkpoint_metadata.json').exists():
            try:
                read(run_dir / 'metadata.json')
            except (ValueError, OSError):
                _write_json(run_dir / 'metadata.json', read(run_dir / 'checkpoint_metadata.json'))
        update_progress(freeze)
        try:
            return run_experiment(RunConfig(model=condition['model'], strategy=condition['strategy'],
                prompt_version=condition['version'], subsets=load_pilot_manifest(settings['manifest']).subsets,
                experiment_type=settings['name'], run_id=condition['run_id'], run_root=Path(settings['run_root']),
                manifest_path=Path(settings['manifest']), model_config_path=Path(condition['model_config']),
                resume=run_dir.exists(), provenance=freeze, journal_requests=True,
                on_checkpoint=lambda path: checkpoint(path, freeze), max_new_examples=max_new_examples),
                client_factory=client_factory or (lambda config: Main100Client(config, run_dir / 'requests.jsonl')))
        finally:
            update_progress(freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'status', 'run'))
    parser.add_argument('--condition')
    parser.add_argument('--live', action='store_true', help='Metadata-only Ollama check; run always checks live')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(), indent=2))
    elif args.action == 'run':
        if not args.condition:
            parser.error('run requires --condition')
        print(run(args.condition))
    else:
        freeze = checked_freeze(live=args.live)
        if args.action == 'status':
            with experiment_lock(freeze['settings']['root']):
                update_progress(freeze)
        print(json.dumps({'status': 'VERIFIED', 'conditions': [c['id'] for c in freeze['conditions']]}, indent=2))


if __name__ == '__main__':
    main()
