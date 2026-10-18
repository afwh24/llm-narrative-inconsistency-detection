"""Read-only source audit and deterministic aggregates for the integrated FYP report.

No model calls, evaluation-policy changes, annotation changes, or source overwrites.
Run with the bundled artifact Python from the repository root.
"""
from pathlib import Path
import csv, json, hashlib, collections, shutil, zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/integrated_fyp_20261013'
OUT.mkdir(exist_ok=True)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(s) for s in (ROOT / p).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def csvrows(p):
    with (ROOT / p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

# Preserve all existing report sources and rendered reports before authoring a new report.
backup = OUT / 'prior_reports_20261013.zip'
if not backup.exists():
    with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT/'reports').rglob('*')):
            if p.is_file() and OUT not in p.parents:
                z.write(p, p.relative_to(ROOT))

protected = {}
for folder in ['data','experiments','prompts','configs','docs','src','papers','reports']:
    for p in (ROOT/folder).rglob('*'):
        if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts:
            protected[p.relative_to(ROOT).as_posix()] = digest(p)
snapshot = OUT/'source_preservation_before.json'
if not snapshot.exists(): snapshot.write_text(json.dumps(protected,indent=2),encoding='utf-8')
else:
    prior=json.loads(snapshot.read_text(encoding='utf-8'))
    assert all((ROOT/p).exists() and digest(ROOT/p)==h for p,h in prior.items()), 'Protected source changed'

base='experiments/main100_comparison_v2_ollama0400/analysis/scoring_correction_v1/'
audit=read(base+'audit.json')
traces=lines(base+'record_traces.jsonl')
assert len(traces)==600
groups=collections.defaultdict(list)
raws={}
for t in traces:
    path=t['source_path']
    if path not in raws: raws[path]=lines(path)
    r=raws[path][t['source_line']-1]
    assert r['example_id']==t['example_id']
    assert hashlib.sha256(r['raw_response'].encode()).hexdigest()==t['raw_response_sha256']
    groups[t['condition']].append(t)
result={'main':[], 'paired':[], 'main_reliability':[]}
for c in audit['conditions']:
    key=c['condition']; ts=groups[key]; m=c['metrics']['corrected']
    assert len(ts)==100 and len({t['example_id'] for t in ts})==100 and sum(t['gold'] for t in ts)==50
    counts=collections.Counter((t['gold'],t['corrected_prediction']['label']) for t in ts)
    for k,ab in [('tp',(1,1)),('fp',(0,1)),('tn',(0,0)),('fn',(1,0)),('invalid_gold_positive',(1,None)),('invalid_gold_negative',(0,None))]:
        assert m[k]==counts[ab]
    assert abs(m['accuracy']-(counts[1,1]+counts[0,0])/100)<1e-12
    assert abs(m['ceeval_full']-sum(t['corrected_localization']['full_correct'] for t in ts)/100)<1e-12
    assert abs(m['ceeval_pos']-sum(t['corrected_localization']['full_correct'] for t in ts if t['gold'])/50)<1e-12
    result['main'].append({'condition':key,'model':c['model'],'strategy':c['strategy'],**m})
    rs=raws[ts[0]['source_path']]
    result['main_reliability'].append({'condition':key,'records':len(rs),'terminal':sum(r['target_inference_failure'] for r in rs),
      'recorded_attempts':sum(len(r['attempt_history']) for r in rs),'legacy_warning_records':sum(bool(r['parsed_prediction'].get('parsing_failures')) for r in rs),
      'corrected_warning_records':sum(bool(t['corrected_prediction']['field_warnings']) for t in ts),
      'invalid':m['invalid'],'total_client_seconds':sum(r['latency_seconds'] for r in rs)})
for stem in ['qwen3_8b','llama3_1_8b','mistral_7b']:
    a={t['example_id']:t for t in groups[stem+'_baseline']}; b={t['example_id']:t for t in groups[stem+'_verify_revise']}
    ct=collections.Counter()
    ids=collections.defaultdict(list)
    for eid,t in a.items():
        x=t['corrected_prediction']['label']==t['gold']; y=b[eid]['corrected_prediction']['label']==t['gold']
        key={(False,True):'improved',(True,False):'regressed',(True,True):'both_correct',(False,False):'both_incorrect'}[x,y]
        ct[key]+=1; ids[key].append(eid)
    result['paired'].append({'model':stem,**ct,'record_ids':dict(ids)})

review=csvrows('experiments/main100_comparison_v2_ollama0400/analysis/human_review_v1/manual_review_simple.csv')
result['review_columns']=list(review[0]); result['review_rows']=review
assert len(review)==30 and len({r['review_id'] for r in review})==30
result['generation_summary']=read('data/generated_stories_verified_v1/evaluator_only/summary.json')
result['generation_review']=csvrows('reports/generated_stories_manual_review_v1/manual_review.csv')
result['corrections']=csvrows('reports/generated_stories_manual_review_v1/optional_annotation_corrections.csv')
result['supplementary']=read('experiments/generated_stories_detection_v1/analysis/benchmark_connection_v1/comparison.json')
supp_audit=read('experiments/generated_stories_detection_v1/analysis/benchmark_connection_v1/audit.json')
for p,h in supp_audit['source_sha256'].items():
    assert digest(ROOT/p).lower()==h.lower(), p
assert supp_audit['final_records']==12 and supp_audit['benchmark_records']==600
result['supplementary_audit']={k:v for k,v in supp_audit.items() if k not in ['record_results','source_sha256']}
result['historical_runs']=[]
for p in sorted((ROOT/'experiments/runs').glob('*/predictions.jsonl')):
    rs=lines(p.relative_to(ROOT))
    result['historical_runs'].append({'path':p.relative_to(ROOT).as_posix(),'records':len(rs),'unique':len({r['example_id'] for r in rs}),
     'terminal_failures':sum(bool(r.get('target_inference_failure')) for r in rs), 'ids':[r['example_id'] for r in rs]})
pilot=set(next(r['ids'] for r in result['historical_runs'] if 'pilot-qwen3-4b-vanilla-pilot' in r['path']))
main={t['example_id'] for t in traces}
assert not main & pilot
result['main_pilot_overlap']=sorted(main & pilot)
result['protected_file_count']=len(protected)
result['validation']='PASS: counts, raw hashes, pairing, localization aggregates, supplementary pinned sources, and preservation'
(OUT/'verified_aggregates.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'source_preservation_verified.json').write_text(json.dumps({'status':'PASS','files':len(protected),'unchanged':True},indent=2),encoding='utf-8')
print(json.dumps({k:result[k] for k in ['validation','protected_file_count','main_reliability','paired']},indent=2))
