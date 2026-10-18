"""Frozen eight-stage schema recovery, reusing the existing executor and client."""
import argparse
import copy
import json
import logging
from pathlib import Path

from fyp.experiments import generated_stories as g
from fyp.experiments import generated_stories_recovery_v2 as recovery
from fyp.experiments import generated_stories_diagnostics_v2 as diagnostic

CONFIG = Path('configs/experiments/generated_stories_schema_recovery_v1.json')


def prepare():
    cfg = g.read(CONFIG);root = Path(cfg['root'])
    if (root/'preparation.json').exists():return status()
    if root.exists() and any(root.iterdir()):raise RuntimeError('Nonempty unprepared output; inspect first')
    dc,dm = diagnostic.loaded()
    _,rm,settings,rf = recovery.loaded()
    original_settings,of = g.frozen()
    if cfg['ollama_version'] != dm['inspection']['ollama_version'] or cfg['max_retries'] != 0:
        raise RuntimeError('Diagnostic runtime/budget mismatch')
    schema = g.read(cfg['schema_path'])
    ref = dm['cases']['mistral_schema']
    if schema != ref['request_format'] or cfg['timeout_seconds'] != ref['timeout_seconds']:
        raise RuntimeError('Batch differs from tested schema/deadline')
    diagnostic_pairs = {}
    for name,case in dm['cases'].items():
        _,ds,df = diagnostic.previous.case_state(dc,dm,name)
        bp = g.stage_path(Path(ds['root']),case['stem'],case['pair_id'],'B')
        b = g.read(bp)
        if b['status']!='ok' or b['freeze_hash']!=g.digest(df):raise RuntimeError('Diagnostic did not succeed')
        if g.parse(b['raw_response']['response'],'B') != (b['fields'],None) or b['raw_response'].get('done_reason')=='length':
            raise RuntimeError('Diagnostic response not complete structurally')
        diagnostic_pairs[case['pair_id']] = (case['source_A'],bp.as_posix(),dc['name']+'/'+name)
    sources = {**dm['hashes'], **recovery.hashes(Path(dc['root'])), **recovery.hashes(Path(settings['root']))}
    for p in (CONFIG,Path(__file__),Path(cfg['schema_path']),diagnostic.CONFIG,Path(diagnostic.__file__)):
        sources[p.as_posix()] = g.stable_file_hash(p)
    registry = {
        original_settings['name']:{'config':g.CONFIG.as_posix(),'preparation':original_settings['root']+'/preparation.json','ollama_version':of['inspection']['ollama_version']},
        settings['name']:{'config':recovery.CONFIG.as_posix(),'preparation':settings['root']+'/preparation.json','ollama_version':rf['inspection']['ollama_version']},
        cfg['name']:{'config':CONFIG.as_posix(),'preparation':cfg['root']+'/preparation.json','ollama_version':cfg['ollama_version']},
    }
    for name in dm['cases']:
        registry[dc['name']+'/'+name]={'config':diagnostic.CONFIG.as_posix(),'preparation':dc['root']+'/preparation.json','case':name,'ollama_version':dc['ollama_version']}
    rows=[];pending=[]
    for model,entry in settings['models'].items():
        stem=entry['stem']
        for i,w in enumerate(g.read(settings['writing_prompts']),1):
            pair=f'{stem}_{i:03d}'
            a0,b0=[g.stage_path(Path(original_settings['root']),stem,pair,stage) for stage in ('A','B')]
            a2,b2=[g.stage_path(Path(settings['root']),stem,pair,stage) for stage in ('A','B')]
            if b0.exists() and g.read(b0)['status']=='ok':ap,bp,ar,br=a0,b0,original_settings['name'],original_settings['name']
            elif b2.exists() and g.read(b2)['status']=='ok':ap,bp,ar,br=a2,b2,settings['name'],settings['name']
            elif pair in diagnostic_pairs:
                ap,bp,br=diagnostic_pairs[pair];ar=settings['name']
            else:
                a,b=g.read(a2),g.read(b2)
                if model!=cfg['model'] or a['status']!='ok' or b['status']!='terminal_parsing_B':raise RuntimeError('Unexpected failure mechanism')
                obj=json.loads(b['raw_response']['response'])
                if 'explanation' in obj['intended_contradiction']:raise RuntimeError('Not the diagnosed missing-field failure')
                ap,bp,ar,br=a2,None,settings['name'],cfg['name'];pending.append(pair)
            rows.append({'pair_id':pair,'model':model,'stem':stem,'writing_prompt_id':w['writing_prompt_id'],
                'A':str(ap).replace('\\','/'),'B':str(bp).replace('\\','/') if bp else None,'A_run':ar,'B_run':br})
    if pending!=cfg['pairs'] or len(pending)!=cfg['max_dispatch_starts']:raise RuntimeError('Scope differs from fixed eight-stage plan')
    for row in rows:
        if row['B'] is None:
            i=int(row['pair_id'][-3:])-1;a=g.read(row['A'])
            row['prompt']=g.prompt(settings,'B',g.read(settings['writing_prompts'])[i],g.read(settings['editing_instructions'])[i],a['fields']['original_story'])
            old=g.read(g.stage_path(Path(settings['root']),row['stem'],row['pair_id'],'B'))
            if old['prompt_hash']!=g.digest(row['prompt']):raise RuntimeError('Saved original/prompt incompatible')
    root.mkdir(parents=True,exist_ok=True)
    g.save(root/'preparation.json',{'config':cfg,'hashes':sources,'inspection':dm['inspection'],
        'settings':settings,'schema':schema,'rows':rows,'source_runs':registry})
    consolidate()
    return status()


def loaded():
    cfg=g.read(CONFIG);m=g.read(Path(cfg['root'])/'preparation.json')
    if cfg!=m['config']:raise RuntimeError('Frozen recovery config changed')
    recovery.verify(m['hashes'])
    s=copy.deepcopy(m['settings']);s.update(root=cfg['root'],name=cfg['name'])
    f={'inspection':m['inspection'],'manifest_hash':g.digest(m)}
    return cfg,m,s,f


def selected(row,s,f):
    a=g.read(row['A']);p=Path(row['B']) if row['B'] else g.stage_path(Path(s['root']),row['stem'],row['pair_id'],'B')
    b=g.read(p) if p.exists() else None
    if not row['B'] and b and (b['freeze_hash']!=g.digest(f) or b['prompt_hash']!=g.digest(row['prompt'])):
        raise RuntimeError('Incompatible new checkpoint')
    return a,b,p


def status():
    cfg,m,s,f=loaded();out=[]
    for row in m['rows']:
        a,b,p=selected(row,s,f)
        journal=Path(s['root'])/'runs'/row['stem']/'requests.jsonl'
        try:events=[json.loads(x) for x in journal.read_text(encoding='utf8').splitlines()] if journal.exists() else []
        except ValueError as exc:raise RuntimeError('Journal incomplete/currently writing; retry read-only status') from exc
        starts=[e for e in events if e['event']=='started' and e['pair_id']==row['pair_id']]
        if len(starts)>1:raise RuntimeError('One-start budget violated')
        out.append({'pair_id':row['pair_id'],'status':b['status'] if b else ('started_without_checkpoint' if starts else 'prepared'),
            'source_run':row['B_run'],'dispatch_starts':len(starts),'remaining_budget':0 if b or row['B'] else 1-len(starts)})
    return {'complete_pairs':sum(x['status']=='ok' for x in out),'terminal_pairs':sum(x['status'].startswith('terminal') for x in out),
        'pending_pairs':sum(x['status'] in ('prepared','started_without_checkpoint') for x in out),
        'new_dispatch_starts':sum(x['dispatch_starts'] for x in out),'pairs':out}


def consolidate():
    cfg,m,s,f=loaded();root=Path(s['root']);records={e['stem']:[] for e in s['models'].values()};provenance=[]
    for row in m['rows']:
        a,b,p=selected(row,s,f)
        if b:
            records[row['stem']].append(g.pair_record(row['pair_id'],row,a,b))
        provenance.append({'pair_id':row['pair_id'],'stages':[
            {'stage':stage,'checkpoint':str(path).replace('\\','/'),'source_run':run,
             'source_configuration':m['source_runs'][run],
             'checkpoint_sha256':g.stable_file_hash(Path(path)) if cp else None,
             'request_ids':[h['request_id'] for h in cp['attempt_history']] if cp else [],
             'status':cp['status'] if cp else 'pending'}
            for stage,path,run,cp in [('A',row['A'],row['A_run'],a),('B',p,row['B_run'],b)]]})
    for stem,values in records.items():
        suffix='' if len(values)==10 else '.partial'
        g.atomic(root/'exports'/f'{stem}_generated_pairs{suffix}.jsonl',''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
    g.save(root/'selection_provenance.json',provenance)


def check():
    cfg,m,s,f=loaded()
    source=g.read(Path(m['settings']['root'])/'preparation.json')
    current=g.inspect_models(m['settings'],source['source_freeze']['stage_a_bounds'])
    recovery.validate_inspection(current,m['inspection'])
    return 'Frozen sources and live metadata match; no inference sent.'


def run(*,client_factory=None):
    cfg,m,s,f=loaded()
    with g.experiment_lock(Path(s['root'])):
        if status()['terminal_pairs']:raise RuntimeError('Batch previously failed; stop for diagnosis, no further dispatch')
        try:
            for row in m['rows']:
                if row['B']:continue
                a,b,p=selected(row,s,f)
                if b:continue
                check()
                factory=client_factory or (lambda config,path:diagnostic.previous.DiagnosticClient(
                    config,path,case=cfg,request_format=m['schema']))
                b=g.execute_stage(s,f,row['model'],row['pair_id'],'B',row['prompt'],a['fields']['original_story'],factory)
                consolidate()
                if b['status']!='ok':raise RuntimeError('Batch stopped at '+row['pair_id']+': '+b['status'])
        finally:consolidate()
    return status()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','check','status','run'])
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(message)s')
    print(json.dumps({'prepare':prepare,'check':check,'status':status,'run':run}[args.command](),indent=2))


if __name__=='__main__':main()
