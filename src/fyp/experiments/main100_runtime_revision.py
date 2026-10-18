"""Ollama 0.40.0 revision using the unchanged main100 execution implementation.

Only `run` dispatches inference. Preparation, check and status never generate.
The v1 source and artifacts remain immutable; an isolated module instance binds
the existing engine to the new configuration without changing v1's globals.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import json

from fyp.experiments import main100_comparison as parent, baseline_pilot
from fyp.experiments.prompting_outputs import _write_json
from fyp.experiments.runner import ExperimentRunError
from fyp.data.pilot import stable_file_hash

CONFIG = Path('configs/experiments/main100_comparison_v2_ollama0400.json')


def parameter_values(text):
    """Ignore order across parameter names; retain all values and repeated-value order."""
    values = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        pair = line.split(None, 1)
        if len(pair) != 2:
            raise ExperimentRunError('Malformed native parameter metadata')
        key, value = pair
        values.setdefault(key, []).append(value.strip())
    return values


def comparable_models(models):
    result = deepcopy(models)
    for model in result.values():
        if model.get('available') and 'identity' in model:
            identity = model['identity']
            identity['parameters'] = parameter_values(identity.get('parameters', ''))
    return result


def verify_inspection(live, expected_models, expected_version):
    if live.get('ollama_version') != expected_version:
        raise ExperimentRunError(f"Expected Ollama {expected_version}; received {live.get('ollama_version')!r}; {live.get('error', '')}")
    if not all(m.get('available') for m in live.get('models', {}).values()):
        raise ExperimentRunError('Required model unavailable')
    if comparable_models(live.get('models', {})) != comparable_models(expected_models):
        raise ExperimentRunError('Model identity, native parameter values or context capacity changed')


def engine():
    spec = importlib.util.spec_from_file_location('fyp.experiments._main100_revision_engine', parent.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CONFIG = CONFIG
    original_check = module.checked_freeze

    def check(live=False):
        freeze = original_check(live=False)
        if live:
            inspection = baseline_pilot.inspect_models(parent.read(freeze['settings']['baseline_configuration']),
                [row for rows in freeze['rendered_prompts'].values() for row in rows])
            verify_inspection(inspection, freeze['models'], freeze['ollama_version'])
        return freeze

    module.checked_freeze = check
    return module


def prepare():
    settings = parent.read(CONFIG)
    root = Path(settings['root'])
    if (root / 'preparation.json').exists() or list((root / 'runs').glob('*/predictions.jsonl')):
        raise ExperimentRunError('Revision already frozen or executed; refusing replacement')
    old = parent.checked_freeze(live=False)
    if parent.read(settings['parent_preparation']) != old:
        raise ExperimentRunError('Parent preparation mismatch')
    # Only namespace and runtime provenance change. No resampling or prompt selection.
    namespace_keys = {'name', 'root', 'run_root', 'export_dir'}
    for key, value in old['settings'].items():
        if key not in namespace_keys and settings.get(key) != value:
            raise ExperimentRunError('Method/sample setting changed: ' + key)
    inspection_path = root / 'model_inspection.json'
    inspection = parent.read(inspection_path)
    verify_inspection(inspection, old['models'], settings['expected_ollama_version'])
    changes = []
    for name, model in inspection['models'].items():
        before = old['models'][name]['identity']['parameters']
        after = model['identity']['parameters']
        if before != after:
            changes.append({'model': name, 'field': 'parameters', 'before': before,
                            'after': after, 'values_equal': parameter_values(before) == parameter_values(after)})
    conditions = []
    for prior in old['conditions']:
        condition = dict(prior)
        condition['run_id'] = settings['name'] + '__' + condition['id']
        condition['run_dir'] = settings['run_root'] + '/' + condition['run_id']
        condition['export'] = settings['export_dir'] + '/' + condition['id'] + '.json'
        condition['partial_export'] = settings['export_dir'] + '/partial/' + condition['id'] + '.partial.json'
        conditions.append(condition)
    old_root = Path(old['settings']['root'])
    prior_records = list(old_root.rglob('predictions.jsonl'))
    prior_requests = list(old_root.rglob('requests.jsonl'))
    revision = {'parent_preparation': settings['parent_preparation'],
        'parent_preparation_hash': stable_file_hash(Path(settings['parent_preparation'])),
        'parent_runtime': old['ollama_version'], 'runtime': settings['expected_ollama_version'],
        'metadata_changes': changes,
        'parent_prediction_records': sum(len(parent._read_rows(p)) for p in prior_records),
        'parent_request_events': sum(len(parent._read_rows(p)) for p in prior_requests),
        'records_reused_from_parent': 0,
        'sample_policy': 'Reference the exact immutable v1 manifest, inputs, gold, exclusions and templates; no resampling.',
        'limitation': 'Pilot selection and 6.19-hour planning estimate are historical Ollama 0.35.1 evidence. Runtime equivalence to 0.40.0 is not claimed or tested with inference.'}
    _write_json(root / 'runtime_revision.json', revision)
    freeze = deepcopy(old)
    freeze.update(settings=settings, conditions=conditions, models=inspection['models'],
        ollama_version=settings['expected_ollama_version'],
        prepared_at=datetime.now(timezone.utc).isoformat(), runtime_revision=revision)
    # Keep every inherited method/data hash; add adapter and revision provenance.
    for path in (CONFIG, Path(__file__), inspection_path, root / 'runtime_revision.json',
                 Path(settings['parent_preparation']), Path(settings['parent_contract'])):
        # Local code uses repository-relative names so the freeze remains readable.
        path = path.resolve().relative_to(Path.cwd()) if path.is_absolute() else path
        freeze['source_hashes'][path.as_posix()] = stable_file_hash(path)
    _write_json(root / 'preparation.json', freeze)
    module = engine()
    module.checked_freeze(live=False)
    module.update_progress(freeze)
    return {'status': freeze['status'], 'runtime': freeze['ollama_version'],
            'manifest_hash': freeze['manifest_hash'], 'expected_records': 600, 'inference_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'check', 'status', 'run'))
    parser.add_argument('--condition')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        print(json.dumps(prepare(), indent=2))
        return
    module = engine()
    if args.action == 'run':
        if not args.condition:
            parser.error('run requires --condition')
        print(module.run(args.condition))
    else:
        freeze = module.checked_freeze(live=args.live)
        if args.action == 'status':
            with module.experiment_lock(freeze['settings']['root']):
                module.update_progress(freeze)
        print(json.dumps({'status': 'VERIFIED', 'runtime': freeze['ollama_version'],
                          'conditions': [c['id'] for c in freeze['conditions']]}, indent=2))


if __name__ == '__main__':
    main()
