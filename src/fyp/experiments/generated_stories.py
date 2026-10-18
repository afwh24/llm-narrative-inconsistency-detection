"""Frozen, two-stage candidate generation. Preparation never calls inference."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path

from fyp.data.pilot import stable_file_hash
from fyp.models.registry import load_model_config
from fyp.models.base import ModelRequest
from fyp.models.request_journal import JournaledOllamaClient
from fyp.models.ollama_client import OllamaError, OllamaInterruptedError
from fyp.experiments.baseline_pilot import inspect_models, context_budget
from fyp.experiments.main100_comparison import experiment_lock

CONFIG = Path('configs/experiments/generated_stories_v1.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()


def atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        stream.write(text.encode('utf8'))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def save(path, value):
    atomic(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def prompt(settings, stage, writing, editing, original=None):
    template = Path(settings['templates'][stage]).read_text(encoding='utf8')
    if stage == 'A':
        return template.replace('{{writing_prompt}}', writing['writing_prompt'])
    # Replace the untrusted original last so its literal contents are preserved.
    return template.replace('{{editing_instruction}}', editing['editing_instruction']).replace('{{original_story}}', original)


def parse(text, stage):
    """Strict structural parsing, retaining independently recoverable fields."""
    recovered = {'original_story': None, 'modified_story': None, 'intended_contradiction': None}
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        # Salvage a complete embedded object without accepting the bad format.
        if isinstance(text,str) and '{' in text:
            try:
                embedded,_=json.JSONDecoder().raw_decode(text[text.index('{'):])
                recovered,_=parse(json.dumps(embedded),stage)
            except ValueError:
                pass
        return recovered, 'response_not_json'
    if not isinstance(obj, dict):
        return recovered, 'response_not_object'
    story_key = 'story' if stage == 'A' else 'modified_story'
    if isinstance(obj.get(story_key), str):
        recovered['original_story' if stage == 'A' else story_key] = obj[story_key]
    if stage == 'B' and isinstance(obj.get('intended_contradiction'), dict):
        recovered['intended_contradiction'] = {k: obj['intended_contradiction'].get(k) if isinstance(obj['intended_contradiction'].get(k),str) else None for k in ('error_sentence','contradicted_sentence','explanation')}
    keys = {'story'} if stage == 'A' else {'modified_story','intended_contradiction'}
    valid = set(obj)==keys and isinstance(obj.get(story_key),str) and bool(obj[story_key].strip())
    if stage == 'B':
        evidence = obj.get('intended_contradiction')
        valid = valid and isinstance(evidence,dict) and set(evidence)=={'error_sentence','contradicted_sentence','explanation'} and all(isinstance(v,str) and bool(v.strip()) for v in evidence.values())
    return recovered, None if valid else 'invalid_required_structure'


def quality(fields, stage, settings, original=None):
    story = fields['original_story' if stage=='A' else 'modified_story']
    warnings=[]
    if isinstance(story,str):
        if not settings['target_words'][0] <= len(story.split()) <= settings['target_words'][1]:
            warnings.append('outside_target_word_range')
    if stage=='B' and story is not None:
        evidence = fields.get('intended_contradiction') or {}
        for key in ('error_sentence','contradicted_sentence'):
            if evidence.get(key) and evidence[key] not in story:
                warnings.append(key+'_not_verbatim_in_modified_story')
        if original is not None:
            if story==original: warnings.append('unchanged_story')
            if 1-difflib.SequenceMatcher(None,original,story,autojunk=False).ratio()>settings['max_edit_fraction_warning']:
                warnings.append('large_edit_fraction')
            earlier=evidence.get('contradicted_sentence')
            if earlier and earlier not in original: warnings.append('earlier_quote_not_preserved_from_original')
    return warnings


class CapturingClient(JournaledOllamaClient):
    """Preserve HTTP bytes even when the inherited JSON parser rejects them."""
    def _open_json(self, request, timeout_seconds):
        delegate=self._opener
        starts=[e for e in self._events() if e['event']=='started']
        attempt=max(e['attempt'] for e in starts)
        path=self.journal_path.parent / 'raw_http' / f"{self.context['pair_id']}_{self.context['stage']}_{attempt}.bin"
        path.parent.mkdir(parents=True,exist_ok=True)

        class RecordingOpener:
            def open(self, req, timeout):
                response=delegate.open(req,timeout=timeout)
                class Response:
                    def __enter__(self): response.__enter__();return self
                    def __exit__(self,*args): return response.__exit__(*args)
                    def read(self):
                        data=response.read()
                        with path.open('wb') as out:
                            out.write(data);out.flush();os.fsync(out.fileno())
                        return data
                return Response()
        self._opener=RecordingOpener()
        try:
            return super()._open_json(request,timeout_seconds)
        finally:
            self._opener=delegate


def frozen(live=False):
    settings=read(CONFIG);root=Path(settings['root']);freeze=read(root/'preparation.json')
    for path, expected in freeze['hashes'].items():
        if stable_file_hash(Path(path))!=expected: raise RuntimeError('Frozen artifact changed: '+path)
    if live:
        current=inspect_models(settings,freeze['stage_a_bounds'])
        if current!=freeze['inspection']: raise RuntimeError('Model/runtime/capacity changed or unavailable; no inference sent')
    for path in settings['templates'].values():
        if stable_file_hash(Path(path))!=stable_file_hash(root/'frozen_templates'/Path(path).name):
            raise RuntimeError('Frozen template copy differs')
    return settings,freeze


def prepare():
    settings=read(CONFIG);root=Path(settings['root'])
    if (root/'preparation.json').exists(): raise RuntimeError('Already frozen; use check')
    writing=read(settings['writing_prompts']);editing=read(settings['editing_instructions'])
    assert [p['writing_prompt_id'] for p in writing]==[f'prompt_{i:03d}' for i in range(1,11)]
    assert [p['writing_prompt_id'] for p in editing]==[p['writing_prompt_id'] for p in writing]
    bounds=[{'prompt_utf8_bytes':len(prompt(settings,'A',w,e).encode('utf8'))} for w,e in zip(writing,editing)]
    inspection=inspect_models(settings,bounds)
    if not all(inspection['models'].get(m,{}).get('available') for m in settings['models']):
        save(root/'model_inspection.json',inspection)
        raise RuntimeError('Models unavailable; preparation not frozen')
    paths=[CONFIG,Path(settings['writing_prompts']),Path(settings['editing_instructions']),Path(__file__)]
    paths += [Path(p) for p in settings['templates'].values()]
    paths += [Path(x['config']) for x in settings['models'].values()]
    paths += [Path('src/fyp/models')/p for p in ('base.py','registry.py','ollama_client.py','request_journal.py')]
    paths += [Path('src/fyp/experiments')/p for p in ('baseline_pilot.py','main100_comparison.py')]
    paths += [root/'generation_contract.md']
    freeze={'settings':settings,'inspection':inspection,'stage_a_bounds':bounds,
            'hashes':{p.as_posix():stable_file_hash(p) for p in paths},'generation_calls':0}
    save(root/'preparation.json',freeze)
    for stage,path in settings['templates'].items():
        copy=root/'frozen_templates'/Path(path).name
        copy.parent.mkdir(parents=True,exist_ok=True)
        copy.write_bytes(Path(path).read_bytes())
    update_progress(settings,freeze)
    return freeze


def stage_path(root,stem,pair,stage):
    return root/'checkpoints'/stem/f'{pair}_{stage}.json'


def pair_record(pair,writing,a,b):
    if not a or (a['status']=='ok' and not b): return None
    final = b if a['status']=='ok' else a
    return {'pair_id':pair,'writing_prompt_id':writing['writing_prompt_id'],
        'original_story':a['fields']['original_story'],
        'modified_story':b['fields']['modified_story'] if b else None,
        'intended_contradiction':b['fields']['intended_contradiction'] if b else None,
        'generation_status':final['status']}


def update_progress(settings,freeze):
    root=Path(settings['root']);writing=read(settings['writing_prompts']);progress=[]
    for model,entry in settings['models'].items():
        stem=entry['stem'];records=[];stages=0
        for i,w in enumerate(writing,1):
            pair=f'{stem}_{i:03d}';saved=[]
            for stage in ('A','B'):
                p=stage_path(root,stem,pair,stage)
                value=read(p) if p.exists() else None
                if value:
                    if value['freeze_hash']!=digest(freeze) or value['pair_id']!=pair or value['stage']!=stage:
                        raise RuntimeError('Incompatible checkpoint')
                    stages+=1
                saved.append(value)
            row=pair_record(pair,w,*saved)
            if row:records.append(row)
        failed=sum(r['generation_status']!='ok' for r in records)
        destination=root/'exports'/f'{stem}_generated_pairs.jsonl' if len(records)==10 else root/'exports/partial'/f'{stem}_generated_pairs.partial.jsonl'
        atomic(destination,''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records))
        progress.append({'model':model,'completed_pairs':len(records),'pending_pairs':10-len(records),'terminal_failures':failed,'completed_stages':stages,'export':destination.as_posix()})
    save(root/'progress.json',{'planned_pairs':30,'completed_pairs':sum(x['completed_pairs'] for x in progress),'pending_pairs':sum(x['pending_pairs'] for x in progress),'models':progress})
    atomic(root/'progress.md','\n'.join(f"{x['model']}: {x['completed_pairs']}/10 pairs; {x['pending_pairs']} pending; {x['terminal_failures']} failures; {x['completed_stages']} stages checkpointed" for x in progress)+'\n')


def execute_stage(settings,freeze,model,pair,stage,text,original=None,client_factory=CapturingClient):
    root=Path(settings['root']);entry=settings['models'][model];path=stage_path(root,entry['stem'],pair,stage)
    if path.exists():
        saved=read(path)
        if saved['freeze_hash']!=digest(freeze) or saved['prompt_hash']!=digest(text):raise RuntimeError('Checkpoint mismatch')
        return saved
    fields={'original_story':None,'modified_story':None,'intended_contradiction':None}
    checkpoint={'pair_id':pair,'stage':stage,'freeze_hash':digest(freeze),'prompt_hash':digest(text),'fields':fields,'warnings':[],'attempt_history':[],'raw_response':None,'error':None}
    try:
        budget=context_budget(settings,[{'prompt_utf8_bytes':len(text.encode('utf8'))}],freeze['inspection']['models'][model]['identity'])
    except RuntimeError as exc:
        checkpoint.update(status=f'terminal_input_capacity_{stage}',error=str(exc))
        save(path,checkpoint);return checkpoint
    client=client_factory(load_model_config(entry['config']),root/'runs'/entry['stem']/'requests.jsonl')
    client.journal_path.parent.mkdir(parents=True,exist_ok=True)
    client.set_context({'run_id':settings['name']+'__'+entry['stem'],'cache_key':digest([digest(freeze),pair,stage,text]),'example_id':pair,'pair_id':pair,'stage':stage,'benchmark_strategy':'generation_'+stage,'model':model,'model_digest':freeze['inspection']['models'][model]['identity']['digest']})
    try:
        response=client.generate(ModelRequest(text))
        fields,error=parse(response.response,stage)
        checkpoint.update(fields=fields,status=f'terminal_parsing_{stage}' if error else 'ok',error=error,raw_response=dict(response.raw),warnings=quality(fields,stage,settings,original))
        if response.raw.get('done_reason')=='length':
            checkpoint['status']=f'terminal_output_limit_{stage}'
            checkpoint['warnings'].append('output_allowance_reached_no_quality_retry')
    except OllamaInterruptedError:
        raise  # pending, journal is authoritative; never synthesize a failure
    except OllamaError as exc:
        checkpoint.update(status=f'terminal_technical_{stage}_{type(exc).__name__}',error=str(exc),raw_response=client.last_raw_response)
        if isinstance(client.last_raw_response,dict) and isinstance(client.last_raw_response.get('response'),str):
            checkpoint['fields'],_=parse(client.last_raw_response['response'],stage)
    checkpoint['attempt_history']=client.last_attempt_history
    checkpoint['capacity']=budget
    save(path,checkpoint)
    return checkpoint


def run(model):
    settings,freeze=frozen(live=True)
    if model not in settings['models']:raise ValueError('Unscheduled model')
    root=Path(settings['root']);entry=settings['models'][model]
    with experiment_lock(root):
        try:
            for i,(w,e) in enumerate(zip(read(settings['writing_prompts']),read(settings['editing_instructions'])),1):
                pair=f"{entry['stem']}_{i:03d}"
                a=execute_stage(settings,freeze,model,pair,'A',prompt(settings,'A',w,e))
                update_progress(settings,freeze)
                if a['status']=='ok':
                    execute_stage(settings,freeze,model,pair,'B',prompt(settings,'B',w,e,a['fields']['original_story']),a['fields']['original_story'])
                    update_progress(settings,freeze)
                print(f'{model}: {i}/10 pairs finalized',flush=True)
        finally:
            update_progress(settings,freeze)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','check','status','run'])
    parser.add_argument('--model')
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='check': frozen(live=True);print('Frozen artifacts and live model metadata match; no inference sent.')
    elif args.command=='status':
        settings,freeze=frozen();print(json.dumps(read(Path(settings['root'])/'progress.json'),indent=2))
    elif not args.model: parser.error('run requires --model')
    else:run(args.model)


if __name__=='__main__':
    main()
