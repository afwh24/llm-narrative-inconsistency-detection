"""Runtime revision of the prepared diagnostics; reuses existing stage execution."""
import argparse
import copy
import json
import logging
from pathlib import Path

from fyp.experiments import generated_stories as g
from fyp.experiments import generated_stories_diagnostics as previous
from fyp.experiments import generated_stories_recovery_v2 as recovery

CONFIG = Path('configs/experiments/generated_stories_diagnostics_v2.json')


def require_unexecuted(root):
    if any(root.rglob('requests.jsonl')) or any(root.glob('*/checkpoints/**/*.json')):
        raise RuntimeError('Prior diagnostics contain attempts/results; do not allocate duplicate budgets')


def prepare():
    cfg = g.read(CONFIG)
    root = Path(cfg['root'])
    if (root/'preparation.json').exists():
        return status()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError('Nonempty unprepared directory; preserve and inspect')
    old_cfg, old = previous.loaded()
    predecessor = Path(cfg['predecessor_root'])
    require_unexecuted(predecessor)
    if str(predecessor).replace('\\','/') != old_cfg['root']:
        raise RuntimeError('Unexpected predecessor')
    for key in ('cases','source_root','schema_path','max_retries','max_total_dispatch_starts'):
        if cfg[key] != old_cfg[key]:
            raise RuntimeError('Runtime revision changed non-runtime configuration: '+key)
    manifest = copy.deepcopy(old)
    manifest['config'] = cfg
    manifest['inspection']['ollama_version'] = cfg['ollama_version']
    manifest['predecessor_manifest_hash'] = g.digest(old)
    manifest['hashes'].update(recovery.hashes(predecessor))
    for p in (CONFIG, Path(__file__)):
        manifest['hashes'][p.as_posix()] = g.stable_file_hash(p)
    root.mkdir(parents=True,exist_ok=True)
    g.save(root/'preparation.json',manifest)
    return status()


def loaded():
    cfg = g.read(CONFIG)
    manifest = g.read(Path(cfg['root'])/'preparation.json')
    if cfg != manifest['config']:
        raise RuntimeError('Diagnostic configuration changed')
    recovery.verify(manifest['hashes'])
    require_unexecuted(Path(cfg['predecessor_root']))
    return cfg,manifest


def check():
    cfg,manifest = loaded()
    source = g.read(Path(cfg['source_root'])/'preparation.json')
    current = g.inspect_models(manifest['source_settings'],source['source_freeze']['stage_a_bounds'])
    recovery.validate_inspection(current,manifest['inspection'])
    return 'Diagnostic v2 live metadata matches Ollama '+cfg['ollama_version']+'; no inference sent.'


def status():
    cfg,manifest = loaded()
    rows = []
    for name in manifest['cases']:
        case,settings,freeze = previous.case_state(cfg,manifest,name)
        root = Path(settings['root'])
        path = g.stage_path(root,case['stem'],case['pair_id'],'B')
        journal = root/'runs'/case['stem']/'requests.jsonl'
        try:
            events = [json.loads(x) for x in journal.read_text(encoding='utf8').splitlines()] if journal.exists() else []
        except ValueError as exc:
            raise RuntimeError('Journal incomplete or being written; retry read-only status') from exc
        starts = [e for e in events if e['event']=='started']
        if len(starts)>1:
            raise RuntimeError('Diagnostic budget invariant violated')
        cp = g.read(path) if path.exists() else None
        if cp and (cp['freeze_hash']!=g.digest(freeze) or cp['prompt_hash']!=g.digest(case['prompt'])):
            raise RuntimeError('Checkpoint mismatch')
        rows.append({'case':name,'pair_id':case['pair_id'],'stage':'B',
            'status':cp['status'] if cp else ('started_without_checkpoint_inspect_journal' if starts else 'prepared'),
            'dispatch_starts':len(starts),'remaining_dispatch_budget':0 if cp else 1-len(starts),
            'error':cp['error'] if cp else None,'checkpoint':path.as_posix()})
    return rows


def run(name, *, client_factory=None):
    cfg,manifest = loaded()
    if name not in manifest['cases']:
        raise ValueError('Unknown diagnostic case')
    case,settings,freeze = previous.case_state(cfg,manifest,name)
    with g.experiment_lock(Path(cfg['root'])):
        path = g.stage_path(Path(settings['root']),case['stem'],case['pair_id'],'B')
        if not path.exists():check()
        a = g.read(case['source_A'])
        factory = client_factory or (lambda config,path: previous.DiagnosticClient(
            config,path,case=case,request_format=case['request_format']))
        b = g.execute_stage(settings,freeze,case['model'],case['pair_id'],'B',case['prompt'],a['fields']['original_story'],factory)
        row = g.pair_record(case['pair_id'],{'writing_prompt_id':'prompt_'+case['pair_id'][-3:]},a,b)
        g.atomic(Path(settings['root'])/'exports/diagnostic_pair.jsonl',json.dumps(row,ensure_ascii=False)+'\n')
        g.save(Path(settings['root'])/'provenance.json',{'diagnostic_case':name,'source_A':case['source_A'],
            'source_B':case['source_B'],'original_story_hash':case['original_story_hash'],
            'manifest_hash':g.digest(manifest),'factor':case['factor'],
            'runtime_revision':cfg['procedure_revision'],'diagnostic_checkpoint':path.as_posix()})
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
