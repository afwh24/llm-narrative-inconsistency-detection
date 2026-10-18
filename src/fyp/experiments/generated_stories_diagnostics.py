"""Bounded diagnostic wrapper over the existing stage executor; no full recovery."""
import argparse
import copy
import json
import logging
from dataclasses import replace
from pathlib import Path

from fyp.experiments import generated_stories as g
from fyp.experiments import generated_stories_recovery_v2 as recovery
from fyp.experiments.generation_transport_v1 import GenerationTransportClient

CONFIG = Path('configs/experiments/generated_stories_diagnostics_v1.json')


class DiagnosticClient(GenerationTransportClient):
    def __init__(self, config, path, *, case, request_format, **kwargs):
        super().__init__(replace(config, runtime=replace(
            config.runtime, timeout_seconds=case['timeout_seconds'], max_retries=0)), path, **kwargs)
        self.request_format = request_format

    def build_generate_payload(self, request):
        return {**super().build_generate_payload(request), 'format': self.request_format}


def prepare():
    cfg = g.read(CONFIG)
    root = Path(cfg['root'])
    if (root/'preparation.json').exists():
        return status()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Nonempty diagnostic directory without preparation; preserve and inspect')
    _, source_manifest, settings, source_freeze = recovery.loaded()
    if cfg['source_root'] != settings['root'] or cfg['max_retries'] != 0 or cfg['max_total_dispatch_starts'] != len(cfg['cases']):
        raise RuntimeError('Unsupported diagnostic source or budget')
    cases = {}
    for name, case in cfg['cases'].items():
        stem = settings['models'][case['model']]['stem']
        paths = {stage: g.stage_path(Path(cfg['source_root']), stem, case['pair_id'], stage) for stage in ('A','B')}
        a, b = (g.read(paths[stage]) for stage in ('A','B'))
        if a['status'] != 'ok' or b['status'] != case['expected_source_status']:
            raise RuntimeError('Requires compatible successful A and failed B: ' + name)
        if any(cp['freeze_hash'] != g.digest(source_freeze) for cp in (a,b)):
            raise RuntimeError('Incompatible source checkpoint')
        if name == 'mistral_schema':
            obj = json.loads(b['raw_response']['response'])
            if 'explanation' in obj['intended_contradiction']:
                raise RuntimeError('Representative no longer matches missing-explanation mechanism')
        journal_path = Path(cfg['source_root'])/'runs'/stem/'requests.jsonl'
        starts = [json.loads(line) for line in journal_path.read_text(encoding='utf8').splitlines()]
        starts = [e for e in starts if e['event']=='started' and e['pair_id']==case['pair_id'] and e['stage']=='B']
        bodies = [e['application_request']['body'] for e in starts]
        if not bodies or any(body != bodies[0] for body in bodies):
            raise RuntimeError('Missing or inconsistent source requests')
        i = int(case['pair_id'][-3:])-1
        text = g.prompt(settings,'B',g.read(settings['writing_prompts'])[i],g.read(settings['editing_instructions'])[i],a['fields']['original_story'])
        if bodies[0]['prompt'] != text or b['prompt_hash'] != g.digest(text):
            raise RuntimeError('Source original/prompt mismatch')
        expected = g.load_model_config(settings['models'][case['model']]['config']).effective_ollama_request_fields() | {'prompt':text,'format':'json'}
        if expected != bodies[0]:
            raise RuntimeError('Source request differs from pinned model settings')
        request_format = g.read(cfg['schema_path']) if case['format']=='schema' else case['format']
        cases[name] = {**case, 'stem': stem, 'source_A': paths['A'].as_posix(),
            'source_B': paths['B'].as_posix(), 'source_journal': journal_path.as_posix(),
            'source_requests': [e['request_id'] for e in starts],
            'prompt': text, 'original_story_hash': g.digest(a['fields']['original_story']),
            'source_body': bodies[0], 'request_format': request_format}
    sources = {**source_manifest['hashes'], **recovery.hashes(Path(cfg['source_root']))}
    for path in (CONFIG, Path(cfg['schema_path']), Path(__file__), Path('src/fyp/experiments/generation_transport_v1.py'), recovery.CONFIG, Path(recovery.__file__)):
        sources[path.as_posix()] = g.stable_file_hash(path)
    root.mkdir(parents=True, exist_ok=True)
    g.save(root/'preparation.json', {'config': cfg, 'source_settings': settings,
        'inspection': source_freeze['inspection'], 'cases': cases, 'hashes': sources})
    return status()


def loaded():
    cfg = g.read(CONFIG)
    manifest = g.read(Path(cfg['root'])/'preparation.json')
    if cfg != manifest['config']:
        raise RuntimeError('Diagnostic configuration changed')
    recovery.verify(manifest['hashes'])
    return cfg, manifest


def case_state(cfg, manifest, name):
    case = manifest['cases'][name]
    settings = copy.deepcopy(manifest['source_settings'])
    settings.update(root=str(Path(cfg['root'])/name), name=cfg['name']+'__'+name)
    freeze = {'inspection': manifest['inspection'], 'manifest_hash': g.digest(manifest), 'case': name}
    return case, settings, freeze


def status():
    cfg, manifest = loaded()
    rows = []
    for name in manifest['cases']:
        case, settings, freeze = case_state(cfg, manifest, name)
        root = Path(settings['root'])
        path = g.stage_path(root,case['stem'],case['pair_id'],'B')
        journal = root/'runs'/case['stem']/'requests.jsonl'
        # A read concurrent with an append can see a torn last line; report retry, never repair it.
        try:
            events = [json.loads(x) for x in journal.read_text(encoding='utf8').splitlines()] if journal.exists() else []
        except ValueError:
            raise RuntimeError('Journal incomplete or currently being written; retry read-only status')
        starts = [e for e in events if e['event']=='started']
        if len(starts)>1:
            raise RuntimeError('Diagnostic budget invariant violated')
        cp = g.read(path) if path.exists() else None
        if cp and (cp['freeze_hash'] != g.digest(freeze) or cp['prompt_hash'] != g.digest(case['prompt'])):
            raise RuntimeError('Diagnostic checkpoint mismatch')
        rows.append({'case':name,'pair_id':case['pair_id'],'stage':'B',
            'status':cp['status'] if cp else ('started_without_checkpoint_inspect_journal' if starts else 'prepared'),
            'dispatch_starts':len(starts),'remaining_dispatch_budget':0 if cp else 1-len(starts),
            'error':cp['error'] if cp else None,'checkpoint':path.as_posix()})
    return rows


def check():
    cfg, manifest = loaded()
    source = g.read(Path(cfg['source_root'])/'preparation.json')
    current = g.inspect_models(manifest['source_settings'],source['source_freeze']['stage_a_bounds'])
    recovery.validate_inspection(current,manifest['inspection'])
    return 'Pinned live metadata matches; no inference sent.'


def run(name, *, client_factory=None):
    cfg, manifest = loaded()
    if name not in manifest['cases']:
        raise ValueError('Unknown diagnostic case')
    case, settings, freeze = case_state(cfg,manifest,name)
    # One shared lock for both diagnostic cases. Reuses the existing OS lock.
    with g.experiment_lock(Path(cfg['root'])):
        path = g.stage_path(Path(settings['root']),case['stem'],case['pair_id'],'B')
        if not path.exists():
            check()  # Only metadata. execute_stage/journal enforces remaining one-start budget.
        a = g.read(case['source_A'])
        factory = client_factory or (lambda config,path: DiagnosticClient(
            config,path,case=case,request_format=case['request_format']))
        b = g.execute_stage(settings,freeze,case['model'],case['pair_id'],'B',case['prompt'],a['fields']['original_story'],factory)
        row = g.pair_record(case['pair_id'],{'writing_prompt_id':'prompt_'+case['pair_id'][-3:]},a,b)
        # Same six-field schema; diagnostic-only output, never merged into historical exports.
        g.atomic(Path(settings['root'])/'exports'/'diagnostic_pair.jsonl',json.dumps(row,ensure_ascii=False)+'\n')
        g.save(Path(settings['root'])/'provenance.json',{'diagnostic_case':name,
            'source_A':case['source_A'],'source_B':case['source_B'],
            'original_story_hash':case['original_story_hash'],'manifest_hash':g.digest(manifest),
            'factor':case['factor'],'diagnostic_checkpoint':path.as_posix()})
    return status()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','check','status','run'])
    parser.add_argument('--case')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    if args.command=='run':
        if not args.case:parser.error('run requires --case')
        result=run(args.case)
    else:result={'prepare':prepare,'check':check,'status':status}[args.command]()
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
