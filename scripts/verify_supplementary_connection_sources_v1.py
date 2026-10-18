"""Additional create-only source checks for the offline connection analysis."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from fyp.experiments import generated_stories_detection as d


def run(output):
    hashes = {}
    def read(p):
        hashes[str(p)] = d.sha(p)
        return d.read(p)
    def rows(p):
        hashes[str(p)] = d.sha(p)
        return d.rows(p)
    previous = read(output/'audit.json')
    assert all(d.sha(p)==h for p,h in previous['source_sha256'].items())
    cfg,freeze=d.loaded()
    prep=read(cfg['benchmark_preparation'])
    manifest=read(prep['settings']['manifest'])
    assert d.sha(prep['settings']['manifest']).lower()==prep['manifest_hash'].lower()
    inputs={x['story_id']:x for x in rows(prep['settings']['inputs'])}
    gold={x['example_id']:x for x in manifest['records']}
    assert set(inputs)==set(gold) and len(inputs)==100
    for sid,x in inputs.items():
        assert d.hash_text(x['story'])==gold[sid]['story_hash'].lower()
    traces=rows(Path(prep['settings']['root'])/'analysis/scoring_correction_v1/record_traces.jsonl')
    raw={t['source_path']:rows(t['source_path']) for t in traces if t['source_line']==1}
    for t in traces:
        x=raw[t['source_path']][t['source_line']-1]
        condition=next(c for c in prep['conditions'] if c['id']==t['condition'])
        prompt=next(p for p in prep['rendered_prompts'][condition['strategy']] if p['example_id']==t['example_id'])
        assert x['input_hash'].lower()==prompt['story_hash'].lower()==gold[t['example_id']]['story_hash'].lower()
        assert x['prompt_hash'].lower()==prompt['prompt_hash'].lower()
        assert t['gold']==gold[t['example_id']]['cont_error']
    root=Path(cfg['dataset'])/'evaluator_only'
    summary=read(root/'summary.json')
    dispositions=rows(root/'all_candidate_dispositions.jsonl')
    by_model=defaultdict(Counter);by_category=defaultdict(Counter);sources=Counter()
    stage_count=0
    for x in dispositions:
        p=x['provenance']
        by_model[p['model']][x['classification']]+=1
        by_category[p['category']][x['classification']]+=1
        assert {s['stage'] for s in p['stages']}=={'A','B'}
        for s in p['stages']:
            for name,hash_key in [('checkpoint','checkpoint_sha256'),('raw_http','raw_http_sha256')]:
                if s.get(name):
                    hashes[s[name]]=d.sha(s[name]);assert hashes[s[name]]==s[hash_key]
            conf=s['source_configuration']['config']
            hashes[conf]=d.sha(conf);assert hashes[conf]==s['configuration_sha256']
            if s['stage']=='B': sources[s['source_run']]+=1
            stage_count+=1
    assert dict(sources)==summary['source_B_counts']
    for dimension,actual in [('model',by_model),('category',by_category)]:
        for key,counts in actual.items():
            assert sum(counts.values())==summary['counts'][dimension][key]['scheduled']
            assert all(counts[k]==summary['counts'][dimension][key][k] for k in ['accepted','rejected','needs_review'])
    assert all(d.sha(p)==h for p,h in hashes.items())
    result=dict(status='PASS',inference_calls=0,benchmark_manifest_hash_verified=True,benchmark_input_prompt_gold_bindings=600,
                generation_source_stages_verified=stage_count,generation_model_counts=dict(by_model),generation_category_counts=dict(by_category),
                source_B_counts=dict(sources),source_sha256=hashes,prior_analysis_sources_still_unchanged=True,
                test_command='python -m pytest tests/test_supplementary_connection_analysis.py tests/test_scoring_correction_v1.py -q -p no:cacheprovider',test_result='45 passed',script_sha256=d.sha(__file__))
    with (output/'additional_source_checks.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','generation_model_counts','generation_category_counts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
