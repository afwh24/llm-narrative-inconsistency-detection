"""Supplementary detection on a frozen accepted subset; all source data read-only."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from fyp.experiments import generated_stories as g, baseline_pilot
from fyp.experiments.main100_comparison import Main100Client, experiment_lock
from fyp.experiments.main100_runtime_revision import parameter_values
from fyp.models.registry import load_model_config
from fyp.models.ollama_client import OllamaError, OllamaInterruptedError
from fyp.prompting.strategies import get_benchmark_strategy
from fyp.prompting.parser import parse_official_response
from fyp.evaluation import corrected_v1 as scoring

CONFIG = Path('configs/experiments/generated_stories_detection_v1.json')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def rows(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def hash_text(text):return hashlib.sha256(text.encode('utf8')).hexdigest()
def save(path,value):g.save(Path(path),value)


def verify_hashes(hashes):
    for path,h in hashes.items():
        if sha(path).lower()!=h.lower():raise RuntimeError('Frozen file mismatch: '+path)


def dataset_audit(cfg):
    root=Path(cfg['dataset']); private=root/'evaluator_only'; m=read(private/'manifest.json')
    verify_hashes(m['source_hashes'])
    verify_hashes({str(root/p):h for p,h in m['output_sha256'].items()})
    if sha(m['config_path'])!=m['config_sha256']:raise RuntimeError('Export configuration changed')
    export_cfg=read(m['config_path'])
    if export_cfg['version']!=m['version'] or Path(export_cfg['root'])!=root:raise RuntimeError('Manifest source/version mismatch')
    inputs=rows(root/'detector_inputs.jsonl'); gold=rows(private/'gold_stories.jsonl'); provenance=rows(private/'provenance.jsonl'); pairs=rows(private/'verified_pairs.jsonl')
    ids=[x['story_id'] for x in inputs]
    if ids!=m['presentation_order'] or len(ids)!=len(set(ids)):raise RuntimeError('Presentation order/ID mismatch')
    if any(set(x)!={'story_id','story'} for x in inputs):raise RuntimeError('Detector input leakage')
    if len(gold)!=len(ids) or len(provenance)!=len(ids) or {x['story_id'] for x in gold}!=set(ids) or {x['story_id'] for x in provenance}!=set(ids):raise RuntimeError('Gold/provenance ID mismatch')
    gs={x['story_id']:x for x in gold}; ps={x['story_id']:x for x in provenance}
    decisions={x['candidate']['pair_id']:x for x in rows(private/'all_candidate_dispositions.jsonl')}
    accepted={x['pair_id']:x for x in pairs}
    if len(accepted)!=len(pairs) or set(accepted)!={p for p,r in decisions.items() if r['classification']=='accepted'}:raise RuntimeError('Acceptance scope mismatch')
    if len(ids)!=2*len(pairs):raise RuntimeError('Expected two stories per accepted pair')
    source={x['pair_id']:x for x in rows(export_cfg['candidates'])}
    grouped=defaultdict(list); texts=defaultdict(list)
    for x in inputs:
        sid=x['story_id']; p=ps[sid]; pair=accepted.get(p['pair_id'])
        if pair is None:raise RuntimeError('Rejected/unresolved pair in inputs')
        if p['version'] not in ('original','modified'):raise RuntimeError('Unknown version')
        version=p['version']; key=version+'_story'
        if not x['story'] or x['story']!=pair[key] or x['story']!=source[p['pair_id']][key]:raise RuntimeError('Story text/source mismatch')
        if p['story_sha256']!=hash_text(x['story']):raise RuntimeError('Story hash mismatch')
        if gs[sid]['cont_error']!=int(version=='modified'):raise RuntimeError('Gold label/version mismatch')
        if p['writing_prompt_id']!=pair['writing_prompt_id']:raise RuntimeError('Prompt group mismatch')
        for stage in p['source_stages']:
            if sha(stage['checkpoint'])!=stage['checkpoint_sha256']:raise RuntimeError('Generation checkpoint changed')
        if version=='modified':
            ann=pair['verified_annotations']
            for field,key2 in [('cont_error_lines','error_sentence'),('contradicted_lines','contradicted_sentence'),('cont_error_expl','explanation')]:
                if not ann[key2] or gs[sid][field]!=ann[key2]:raise RuntimeError('Gold annotation mismatch')
                if field!='cont_error_expl' and gs[sid][field] not in x['story']:raise RuntimeError('Gold quotation missing')
        elif any(gs[sid][k] for k in ('cont_error_lines','contradicted_lines','cont_error_expl')):raise RuntimeError('Negative gold contains evidence')
        grouped[p['pair_id']].append(sid); texts[hash_text(x['story'])].append(sid)
    for pair,sids in grouped.items():
        if len(sids)!=2 or {ps[s]['version'] for s in sids}!={'original','modified'}:raise RuntimeError('Pair versions/IDs invalid')
    duplicates=[v for v in texts.values() if len(v)>1]
    if any(len({gs[s]['cont_error'] for s in group})>1 for group in duplicates):raise RuntimeError('Identical text with conflicting labels')
    expected_groups=read(private/'writing_prompt_groups.json')
    actual_groups={p:sorted(s for s,v in ps.items() if v['writing_prompt_id']==p) for p in {v['writing_prompt_id'] for v in ps.values()}}
    if {k:sorted(v) for k,v in expected_groups.items()}!=actual_groups:raise RuntimeError('Prompt grouping differs')
    return dict(version=m['version'],accepted_pairs=len(pairs),story_ids=ids,expected_records=6*len(ids),duplicates=duplicates,
        duplicate_policy=cfg['duplicate_policy'],exposure=read(private/'exposure.json'),authoritative_manifest=(private/'manifest.json').as_posix(),
        dataset_hashes={p.as_posix():sha(p) for p in root.rglob('*') if p.is_file()},checks='All structure, source, gold, grouping and leakage checks passed')


def build_prompts(cfg,benchmark):
    inputs=rows(Path(cfg['dataset'])/'detector_inputs.jsonl'); result={}
    for name,version in cfg['expected_strategies'].items():
        frozen=benchmark['templates'][name]
        if frozen['version']!=version:raise RuntimeError('Unexpected frozen prompt version')
        verify_hashes({frozen['path']:frozen['hash'],frozen['copy']:frozen['hash']})
        strategy=get_benchmark_strategy(name,version)
        result[name]=[]
        for x in inputs:
            built=strategy.build_prompt({'story':x['story']})
            if built.template_hash.lower()!=frozen['hash'].lower() or x['story_id'] in built.prompt or x['story'] not in built.prompt:raise RuntimeError('Prompt identity/input mismatch')
            result[name].append(dict(story_id=x['story_id'],prompt=built.prompt,prompt_sha256=hash_text(built.prompt),prompt_utf8_bytes=len(built.prompt.encode('utf8'))))
    return result


def inspect(cfg,benchmark,prompts):
    settings=read(benchmark['settings']['baseline_configuration'])
    return baseline_pilot.inspect_models(settings,[p for group in prompts.values() for p in group])


def identities(models):
    result={}
    for name,m in models.items():
        if not m.get('available'):raise RuntimeError('Model unavailable: '+name)
        identity=dict(m['identity']); identity['parameters']=parameter_values(identity.get('parameters',''))
        result[name]=identity
    return result


def prepare():
    cfg=read(CONFIG); root=Path(cfg['root']); root.mkdir(parents=True,exist_ok=True)
    if (root/'contract.json').exists():return loaded()[1]['readiness']
    try:
        audit=dataset_audit(cfg)
        if not audit['accepted_pairs']:raise RuntimeError('Zero accepted pairs; no evaluation can run')
        benchmark=read(cfg['benchmark_preparation']); prompts=build_prompts(cfg,benchmark)
        verify_hashes({p:h for p,h in benchmark['source_hashes'].items() if p.startswith(('src/fyp/prompting/','src/fyp/evaluation/','src/fyp/models/'))})
        conditions=benchmark['conditions']
        if len(conditions)!=6 or {(c['model'],c['strategy']) for c in conditions}!={(m,s) for m in cfg['expected_models'] for s in cfg['expected_strategies']}:raise RuntimeError('Frozen six-condition scope mismatch')
        for c in conditions:
            verify_hashes({c['model_config']:benchmark['source_hashes'][c['model_config']]})
            if c['version']!=cfg['expected_strategies'][c['strategy']]:raise RuntimeError('Condition prompt version mismatch')
        live=inspect(cfg,benchmark,prompts)
        if not live.get('ollama_version'):raise RuntimeError('Ollama metadata unavailable: '+str(live))
        if identities(live['models'])!=identities(benchmark['models']):raise RuntimeError('Model identity/native metadata differs from frozen benchmark; inspect before inference')
        # Confirm local-only localization can process this gold before any dispatch.
        inputs={x['story_id']:x for x in rows(Path(cfg['dataset'])/'detector_inputs.jsonl')}
        for gold in rows(Path(cfg['dataset'])/'evaluator_only/gold_stories.jsonl'):
            ex=SimpleNamespace(story=inputs[gold['story_id']]['story'],cont_error=gold['cont_error'],cont_error_lines_raw=gold['cont_error_lines'],contradicted_lines_raw=gold['contradicted_lines'])
            score,_=scoring.score_localization(ex,gold['cont_error'],[gold['cont_error_lines']],[gold['contradicted_lines']])
            if score.full_correct!=1:raise RuntimeError('Gold localization self-check failed')
        files=[CONFIG,Path(__file__),Path(cfg['benchmark_preparation']),Path(cfg['benchmark_configuration']),Path(cfg['scoring_policy'])]
        files += [Path(c['model_config']) for c in conditions]
        files += [Path(t[k]) for t in benchmark['templates'].values() for k in ('path','copy')]
        files += list(Path('src/fyp/models').glob('*.py'))+list(Path('src/fyp/prompting').glob('*.py'))+list(Path('src/fyp/evaluation').glob('*.py'))
        files += [Path('src/fyp/experiments')/n for n in ('main100_comparison.py','baseline_pilot.py','main100_runtime_revision.py','generated_stories.py')]
        hashes={p.as_posix():sha(p) for p in files}; hashes.update(audit['dataset_hashes'])
        models={c['model']:asdict(load_model_config(c['model_config'])) for c in conditions}
        for value in models.values():value['source_path']=str(value['source_path'])
        contract=dict(config=cfg,prepared_at=datetime.now(timezone.utc).isoformat(),readiness=audit,inspection=live,
            historical_runtime=benchmark['ollama_version'],conditions=[{k:c[k] for k in ('id','model','strategy','version','model_config')} for c in conditions],
            templates=benchmark['templates'],models=models,prompts=prompts,hashes=hashes,
            scoring_policy=read(cfg['scoring_policy']),dependencies={n:importlib.metadata.version(n) for n in ('nltk','pyarrow')},
            differences=['Accepted supplementary dataset replaces official main100 sample','Actual Ollama runtime pinned separately from historical 0.40.0','Primary scoring uses existing corrected_v1 explicit decisions and localization; legacy parser retained for audit','New isolated per-story checkpoints reuse frozen Main100Client request journaling; batches capped at ten'],
            failure_scoring='Nulls have zero accuracy and pair credit. TP/FP/TN/FN count valid decisions only; invalid positive gold remains in recall denominator and adds to F1 missed positives. Precision=TP/(TP+FP); no artificial positive prediction for invalids. All zero denominators yield 0. CEEval nulls receive zero; Full divides by all scheduled stories, Pos by all positives. Pending results are provisional, never complete.',
            ceeval='Local corrected_v1 non-strict evidence overlap, no LLM verifier; gold self-check passed. Not full paper verifier parity.')
        save(root/'contract.json',contract)
        save(root/'readiness.json',audit)
        text='# Frozen supplementary detection contract\n\n'+json.dumps({k:v for k,v in contract.items() if k not in ('prompts','hashes')},indent=2)+'\n'
        g.atomic(root/'contract.md',text)
        return audit
    except Exception as exc:
        blocker={'status':'BLOCKED_BEFORE_INFERENCE','error':str(exc),'type':type(exc).__name__,'inference_calls':0}
        save(root/'readiness_blocked.json',blocker)
        g.atomic(root/'readiness_report.md','# Readiness blocked\n\n'+str(exc)+'\n\nNo inference dispatched. Existing sources preserved.\n')
        raise


def loaded(live=False):
    cfg=read(CONFIG); freeze=read(Path(cfg['root'])/'contract.json')
    if cfg!=freeze['config']:raise RuntimeError('Supplementary configuration changed')
    verify_hashes(freeze['hashes'])
    if {n:importlib.metadata.version(n) for n in freeze['dependencies']}!=freeze['dependencies']:raise RuntimeError('Dependency versions changed')
    if live:
        current=inspect(cfg,read(cfg['benchmark_preparation']),freeze['prompts'])
        if current['ollama_version']!=freeze['inspection']['ollama_version'] or identities(current['models'])!=identities(freeze['inspection']['models']):raise RuntimeError('Runtime identity changed; dispatch blocked')
    return cfg,freeze


def cp_path(cfg,cid,sid):return Path(cfg['root'])/'checkpoints'/cid/(sid+'.json')


def record_rows(cfg,freeze):
    records=[]; fp=g.digest(freeze)
    for c in freeze['conditions']:
        for p in freeze['prompts'][c['strategy']]:
            path=cp_path(cfg,c['id'],p['story_id'])
            if path.exists():
                r=read(path)
                if r['contract_hash']!=fp or r['story_id']!=p['story_id'] or r['condition']!=c['id'] or r['prompt_sha256']!=p['prompt_sha256']:raise RuntimeError('Incompatible checkpoint')
                records.append(r)
    return records


def run(condition=None,client_factory=None,live=True):
    cfg,freeze=loaded(live=live); root=Path(cfg['root']); fp=g.digest(freeze)
    with experiment_lock(root):
        record_rows(cfg,freeze)
        for c in freeze['conditions']:
            if condition and c['id']!=condition:continue
            new=0; runroot=root/'runs'/c['id']; runroot.mkdir(parents=True,exist_ok=True)
            for p in freeze['prompts'][c['strategy']]:
                path=cp_path(cfg,c['id'],p['story_id'])
                if path.exists():continue
                if new>=cfg['batch_size']:break
                client=(client_factory or Main100Client)(load_model_config(c['model_config']),runroot/'requests.jsonl')
                key=g.digest(dict(contract=fp,condition=c['id'],story=p['story_id'],prompt=p['prompt_sha256']))
                client.set_context(dict(run_id=cfg['name']+'__'+c['id'],cache_key=key,example_id=p['story_id'],benchmark_strategy=c['strategy'],model=c['model'],model_digest=freeze['inspection']['models'][c['model']]['identity']['digest']))
                print('START '+c['id']+' '+p['story_id'],flush=True)
                error=None; response=None
                try:response=client.generate(p['prompt'])
                except OllamaInterruptedError:raise
                except OllamaError as exc:error=dict(type=type(exc).__name__,message=str(exc))
                raw=client.last_raw_response
                if raw and raw.get('done_reason')=='length':error=dict(type='OutputLimit',message='Output limit termination; null prediction')
                text=response.response if response else (raw.get('response') if isinstance(raw,dict) else None)
                parsed=scoring.parse_response(text,freeze['scoring_policy'],inference_failure=error is not None)
                rec=dict(contract_hash=fp,condition=c['id'],model=c['model'],strategy=c['strategy'],story_id=p['story_id'],prompt_sha256=p['prompt_sha256'],
                    raw_response=raw,response_text=text,error=error,terminal_failure=error is not None,unparseable=error is None and parsed.label is None,
                    parsed=asdict(parsed),legacy_parsed=asdict(parse_official_response(text or '')),attempt_history=client.last_attempt_history,
                    latency_seconds=sum(h.get('latency_seconds') or 0 for h in client.last_attempt_history))
                path.parent.mkdir(parents=True,exist_ok=True); save(path,rec); new+=1
                print('FINISH '+c['id']+' '+p['story_id']+' '+('terminal_failure' if error else 'received'),flush=True)
            print(json.dumps(status()),flush=True)
    return status()


def status():
    cfg,f=loaded(); rs=record_rows(cfg,f)
    return dict(expected=f['readiness']['expected_records'],completed=len(rs),pending=f['readiness']['expected_records']-len(rs),terminal_failures=sum(r['terminal_failure'] for r in rs),unparseable=sum(r['unparseable'] for r in rs),conditions=dict(Counter(r['condition'] for r in rs)))


def analyse():
    cfg,f=loaded(); root=Path(cfg['root']); dataset=Path(cfg['dataset']); records={(r['condition'],r['story_id']):r for r in record_rows(cfg,f)}
    inputs={x['story_id']:x for x in rows(dataset/'detector_inputs.jsonl')}; gold={x['story_id']:x for x in rows(dataset/'evaluator_only/gold_stories.jsonl')}; prov={x['story_id']:x for x in rows(dataset/'evaluator_only/provenance.jsonl')}
    results=[]
    for c in f['conditions']:
        for sid in f['readiness']['story_ids']:
            record=records.get((c['id'],sid)); y=gold[sid]; p=prov[sid]
            parsed=record['parsed'] if record else {}; label=parsed.get('label')
            example=SimpleNamespace(story=inputs[sid]['story'],cont_error=y['cont_error'],cont_error_lines_raw=y['cont_error_lines'],contradicted_lines_raw=y['contradicted_lines'])
            loc,trace=scoring.score_localization(example,label,parsed.get('error_lines',[]),parsed.get('contradicted_lines',[]))
            results.append(dict(condition=c['id'],detector=c['model'],strategy=c['strategy'],story_id=sid,generator=p['generator'],category=p['category'],pair_id=p['pair_id'],writing_prompt_id=p['writing_prompt_id'],
                gold=y['cont_error'],predicted=label,full=loc.full_correct,error_side=loc.cont_error_lines_correct,contradicted_side=loc.contradicted_lines_correct,
                pending=record is None,terminal_failure=record['terminal_failure'] if record else False,unparseable=record['unparseable'] if record else False,
                checkpoint=cp_path(cfg,c['id'],sid).as_posix(),localization_trace=trace))
    def aggregate(group):
        value=scoring.aggregate(group); pairs=defaultdict(list)
        for r in group:pairs[(r['condition'],r['pair_id'])].append(r)
        successes=sum(len(rs)==2 and all(r['predicted']==r['gold'] and not r['pending'] for r in rs) for rs in pairs.values())
        value.update(pair_correct=successes,pair_total=len(pairs),pair_accuracy=successes/len(pairs) if pairs else 0,
            terminal_failures=sum(r['terminal_failure'] for r in group),unparseable=sum(r['unparseable'] for r in group),pending=sum(r['pending'] for r in group),
            unique_writing_prompts=len({r['writing_prompt_id'] for r in group}))
        return value
    tables={}
    for keys in [('strategy',),('detector','strategy'),('generator','strategy'),('category','strategy')]:
        groups=defaultdict(list)
        for r in results:groups[' / '.join(r[k] for k in keys)].append(r)
        tables['_'.join(keys)]={k:aggregate(rs) for k,rs in groups.items()}
    out=root/'analysis'; out.mkdir(exist_ok=True)
    summary=dict(status=status(),tables=tables)
    save(out/'metrics.json',summary); save(out/'story_results.json',results)
    md='# Supplementary generated-story detection results\n\n'+json.dumps(summary['status'])+'\n\n'
    md+='All scheduled records remain in denominators. Null predictions are not classes. Accuracy=(TP+TN)/scheduled; precision=TP/(TP+FP); recall=TP/all gold positives; F1=2TP/(2TP+FP+FN+invalid positive gold). Zero denominators return 0. Invalid negative gold reduces accuracy and pair accuracy but is not a false positive. Pair success requires both versions correct. Pending reports are provisional. CEEval uses the existing local corrected non-strict evidence scorer; no external verifier.\n\n'
    for title,table in tables.items():
        md+='## '+title+'\n\n|Group|Stories|Accuracy|Precision|Recall|F1|Pair correct/total|Pair accuracy|Failures|Unparseable|CEEval-Full|CEEval-Pos|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for k,v in table.items():md+=f"|{k}|{v['total']}|{v['accuracy']:.3f}|{v['precision']:.3f}|{v['recall']:.3f}|{v['f1']:.3f}|{v['pair_correct']}/{v['pair_total']}|{v['pair_accuracy']:.3f}|{v['terminal_failures']}|{v['unparseable']}|{v['ceeval_full']:.3f}|{v['ceeval_pos']:.3f}|\n"
    md+='\n## Errors and private mappings\n\n'
    for r in results:
        if r['predicted']!=r['gold'] or r['full']!=1:
            md+=f"- {r['condition']}, `{r['story_id']}`: gold={r['gold']}, predicted={r['predicted']}, localization={r['full']}; private pair `{r['pair_id']}`. [Saved response](<{Path(r['checkpoint']).resolve().as_posix()}>).\n"
    md+='\n## Scope and limitations\n\nOne accepted pair, two stories, one writing-prompt group. Both stories originate from Llama generation recovery v2. Generator and detector identities are distinct. Qwen/Mistral generators and other categories have zero accepted examples here; no performance estimate exists for those groups. Overall pooling repeats the same two stories across detectors and is descriptive, not independent evidence. The broader candidate collection used mixed generation/recovery procedures and strong human filtering (1 accepted/30 scheduled). Rejected/uncertain cases are not detection failures. Exposure is recorded as unknown; known generation diagnosis and review exposure preclude an untouched-test claim. Runtime 0.40.3 differs from historical main100 0.40.0; exact actual version and model/template hashes are in contract.json. No benchmark scores, detector prompts, dataset or human judgments were modified. No superiority or statistical generalization claim is justified.\n'
    g.atomic(out/'report.md',md)
    return summary['status']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','status','analyse']);p.add_argument('--condition');a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    print(json.dumps(run(a.condition) if a.action=='run' else {'prepare':prepare,'status':status,'analyse':analyse}[a.action](),indent=2))


if __name__=='__main__':main()
