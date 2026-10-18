"""Offline, create-only comparison of frozen saved experiments. No inference calls."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

from fyp.experiments import generated_stories_detection as d
from fyp.evaluation import corrected_v1 as scoring


def blocked(*args, **kwargs):
    raise RuntimeError('Network disabled for offline analysis')


def pair_outcome(original, modified):
    if original is None or modified is None:
        return 'unusable'
    return {(0, 1): 'both_correct', (0, 0): 'both_consistent',
            (1, 1): 'both_contradictory', (1, 0): 'both_incorrect'}[(original, modified)]


def table(headers, rows):
    def clean(x):
        return str(x).replace('|', '\\|').replace('\n', '<br>')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(clean(x) for x in row) + ' |' for row in rows])


def run(out):
    socket.socket = blocked
    socket.create_connection = blocked
    assert not out.exists(), 'Use a new output version; existing analysis is immutable.'
    hashes = {}

    def track(path):
        path = Path(path)
        hashes[path.as_posix()] = d.sha(path)
        return path

    def read(path):
        return d.read(track(path))

    def rows(path):
        return d.rows(track(path))

    cfg, freeze = d.loaded(live=False)
    root = Path(cfg['root'])
    dataset = Path(cfg['dataset'])
    private = dataset / 'evaluator_only'
    for path in freeze['hashes']:
        track(path)
    for path in root.rglob('*'):
        if path.is_file():
            track(path)
    ready = d.dataset_audit(cfg)
    manifest = read(private / 'manifest.json')
    for path in manifest['source_hashes']:
        track(path)
    inputs = {x['story_id']: x for x in rows(dataset / 'detector_inputs.jsonl')}
    gold = {x['story_id']: x for x in rows(private / 'gold_stories.jsonl')}
    prov = {x['story_id']: x for x in rows(private / 'provenance.jsonl')}
    pairs = rows(private / 'verified_pairs.jsonl')
    dispositions = rows(private / 'all_candidate_dispositions.jsonl')
    generation = read(private / 'summary.json')
    assert dict(Counter(x['classification'] for x in dispositions)) == generation['classifications']
    assert len({x['candidate']['pair_id'] for x in dispositions}) == len(dispositions) == 30
    assert all(x['reviewed'] for x in dispositions)
    for x in dispositions:
        assert x['candidate']['original_story'] and x['candidate']['modified_story']
    exposure = read(private / 'exposure.json')
    records = d.record_rows(cfg, freeze)
    expected = {(c['id'], sid) for c in freeze['conditions'] for sid in inputs}
    actual = [(r['condition'], r['story_id']) for r in records]
    assert len(actual) == len(set(actual))
    missing = sorted(expected - set(actual))
    assert not (set(actual) - expected)
    assert len(list((root / 'checkpoints').glob('*/*.json'))) == len(actual)
    assert len(expected) == 12 * len(pairs)
    benchmark = Path(cfg['benchmark_preparation']).parent
    preparation = read(cfg['benchmark_preparation'])
    assert d.build_prompts(cfg, preparation) == freeze['prompts']
    assert d.identities(preparation['models']) == d.identities(freeze['inspection']['models'])
    journal_summary = []
    for c in freeze['conditions']:
        journal = rows(root / 'runs' / c['id'] / 'requests.jsonl')
        starts = [j for j in journal if j['event'] == 'started']
        ends = [j for j in journal if j['event'] == 'finished']
        assert len({j['request_id'] for j in starts}) == len(starts)
        assert len({j['request_id'] for j in ends}) == len(ends)
        start_ids = {j['request_id'] for j in starts}
        end_ids = {j['request_id'] for j in ends}
        assert end_ids <= start_ids
        prompts = {p['story_id']: p for p in freeze['prompts'][c['strategy']]}
        model = freeze['models'][c['model']]
        options = {k: v for k, v in model['generation'].items() if k != 'think'}
        for j in starts:
            body = j['application_request']['body']
            assert body['prompt'] == prompts[j['example_id']]['prompt']
            assert body['options'] == options and body['model'] == c['model'] and body['stream'] is False
            assert body.get('think') == model['generation']['think']
            assert set(body) <= {'model', 'prompt', 'stream', 'options', 'think'}
            assert j['model_digest'] == freeze['inspection']['models'][c['model']]['identity']['digest']
        assert max(Counter(j['example_id'] for j in starts).values(), default=0) <= 4
        for r in [r for r in records if r['condition'] == c['id']]:
            matching = [j for j in ends if j['example_id'] == r['story_id']]
            assert matching and matching[-1]['raw_response'] == r['raw_response']
        journal_summary.append(dict(condition=c['id'], starts=len(starts), finishes=len(ends),
                                    unfinished=sorted(start_ids-end_ids), statuses=dict(Counter(j['status'] for j in ends))))
    results = []
    by_key = {(r['condition'], r['story_id']): r for r in records}
    for c in freeze['conditions']:
        for sid in inputs:
            r = by_key.get((c['id'], sid))
            parsed = scoring.parse_response(r['response_text'], freeze['scoring_policy'], inference_failure=r['terminal_failure']) if r else None
            if r:
                assert json.loads(json.dumps(asdict(parsed))) == r['parsed']
                assert r['raw_response']['response'] == r['response_text']
                assert r['raw_response']['done'] and r['raw_response']['done_reason'] == 'stop'
            y = gold[sid]
            ex = SimpleNamespace(story=inputs[sid]['story'], cont_error=y['cont_error'],
                                 cont_error_lines_raw=y['cont_error_lines'], contradicted_lines_raw=y['contradicted_lines'])
            loc, _ = scoring.score_localization(ex, parsed.label if parsed else None,
                                                  parsed.error_lines if parsed else [], parsed.contradicted_lines if parsed else [])
            results.append(dict(condition=c['id'], model=c['model'], story_id=sid,
                                pair_id=prov[sid]['pair_id'], version=prov[sid]['version'], gold=y['cont_error'],
                                predicted=parsed.label if parsed else None, full=loc.full_correct,
                                error_side=loc.cont_error_lines_correct, contradicted_side=loc.contradicted_lines_correct,
                                pending=r is None, terminal_failure=r['terminal_failure'] if r else False,
                                unparseable=r['unparseable'] if r else False,
                                warnings=parsed.field_warnings if parsed else [],
                                source=d.cp_path(cfg, c['id'], sid).as_posix()))
    metrics = {}
    sensitivity = []
    outcomes = {}
    for c in freeze['conditions']:
        group = [x for x in results if x['condition'] == c['id']]
        metrics[c['id']] = scoring.aggregate(group)
        counter = Counter()
        for pair in pairs:
            p = {x['version']: x['predicted'] for x in group if x['pair_id'] == pair['pair_id']}
            outcome = pair_outcome(p.get('original'), p.get('modified'))
            counter[outcome] += 1
            outcomes[(c['id'], pair['pair_id'])] = outcome
        metrics[c['id']].update(pair_correct=counter['both_correct'], pair_total=len(pairs),
                                pair_accuracy=counter['both_correct']/len(pairs),
                                terminal_failures=sum(x['terminal_failure'] for x in group),
                                unparseable=sum(x['unparseable'] for x in group), pending=sum(x['pending'] for x in group))
        sensitivity.append(dict(condition=c['id'], **{k: counter[k] for k in ['both_correct','both_consistent','both_contradictory','both_incorrect','unusable']}))
    saved_metrics = read(root / 'analysis/metrics.json')
    saved_stories = read(root / 'analysis/story_results.json')
    for x in results:
        old = next(s for s in saved_stories if (s['condition'],s['story_id']) == (x['condition'],x['story_id']))
        assert all(old[k] == x[k] for k in ['gold','predicted','full','error_side','contradicted_side','pending','terminal_failure','unparseable'])
    for c in freeze['conditions']:
        old = saved_metrics['tables']['detector_strategy'][c['model']+' / '+c['strategy']]
        assert all(old[k] == v for k,v in metrics[c['id']].items())

    audit = read(benchmark / 'analysis/scoring_correction_v1/audit.json')
    for path, h in audit['correction_sources'].items():
        assert d.sha(track(path)).lower() == h.lower()
    traces = rows(benchmark / 'analysis/scoring_correction_v1/record_traces.jsonl')
    sources = {t['source_path']: rows(t['source_path']) for t in traces if t['source_line'] == 1}
    bm = {}
    benchmark_failures = Counter()
    reference_ids = None
    for t in traces:
        raw = sources[t['source_path']][t['source_line']-1]
        assert raw['example_id'] == t['example_id'] and d.hash_text(raw['raw_response']) == t['raw_response_sha256']
        assert json.loads(json.dumps(asdict(scoring.parse_response(raw['raw_response'], audit['config'], inference_failure=raw['target_inference_failure'])))) == t['corrected_prediction']
        c = next(c for c in preparation['conditions'] if c['id'] == t['condition'])
        assert raw['prompt_template_hash'].lower() == freeze['templates'][c['strategy']]['hash'].lower()
        assert raw['prompt_version'] == c['version']
        options = {k:v for k,v in freeze['models'][c['model']]['generation'].items() if k!='think'}
        assert raw['effective_config']['options'] == options
        assert raw['effective_config'].get('think') == freeze['models'][c['model']]['generation']['think']
        benchmark_failures[t['condition']] += bool(raw['target_inference_failure'])
    for c in audit['conditions']:
        group = [t for t in traces if t['condition'] == c['condition']]
        ids = [t['example_id'] for t in group]
        assert len(ids) == len(set(ids)) == len(sources[group[0]['source_path']]) == 100
        assert reference_ids is None or reference_ids == ids
        reference_ids = ids
        m = scoring.aggregate([dict(gold=t['gold'], predicted=t['corrected_prediction']['label'],
                                    full=t['corrected_localization']['full_correct'],
                                    error_side=t['corrected_localization']['cont_error_lines_correct'],
                                    contradicted_side=t['corrected_localization']['contradicted_lines_correct']) for t in group])
        assert m == c['metrics']['corrected']
        bm[c['condition']] = dict(m, terminal_failures=benchmark_failures[c['condition']])
    with track(benchmark / 'analysis/scoring_correction_v1/comparison.csv').open(encoding='utf8', newline='') as f:
        for r in csv.DictReader(f):
            if r['evaluator'] == 'corrected' and r['condition'] in bm:
                assert all(abs(float(v)-bm[r['condition']][k]) < 1e-12 for k,v in r.items() if k not in ('condition','evaluator'))
    assert len(traces) == 600
    for name in ['report.md','limitations_issues.md']:
        track(benchmark / 'analysis/benchmark_consolidated_v1' / name).read_text(encoding='utf8')
    # Current annotations are only read and hashed, never rejudged.
    track(benchmark / 'analysis/human_review_v1/manual_review_simple.csv').read_text(encoding='utf-8-sig')
    track(private / 'review_validation.md').read_text(encoding='utf8')

    transitions = []
    selected = set()
    for model in cfg['expected_models']:
        b = next(c['id'] for c in freeze['conditions'] if c['model']==model and c['strategy']=='vanilla')
        v = next(c['id'] for c in freeze['conditions'] if c['model']==model and c['strategy']=='verify_revise')
        counts = Counter()
        for sid in inputs:
            x,y = by_key.get((b,sid)),by_key.get((v,sid))
            xl,yl = x['parsed']['label'] if x else None,y['parsed']['label'] if y else None
            correct = gold[sid]['cont_error']
            counts['improved' if xl!=correct and yl==correct else 'regressed' if xl==correct and yl!=correct else 'unchanged_correct' if xl==yl==correct else 'unchanged_unsuccessful'] += 1
            if xl != yl or xl is None or yl is None:
                selected.update([(b,sid),(v,sid)])
        for p in pairs:
            before,after = outcomes[b,p['pair_id']],outcomes[v,p['pair_id']]
            transitions.append(dict(model=model,pair_id=p['pair_id'],baseline=before,verify_revise=after,
                                    pair_change='improvement' if before!='both_correct' and after=='both_correct' else 'regression' if before=='both_correct' and after!='both_correct' else 'unchanged_success' if before==after=='both_correct' else 'unchanged_failure',story_changes=dict(counts)))
            if before!='both_correct' and after!='both_correct':
                selected.update((cid,sid) for cid in (b,v) for sid in inputs if prov[sid]['pair_id']==p['pair_id'])
    delta_keys = ['accuracy','precision','recall','f1','false_positive_rate','ceeval_full','ceeval_pos']
    deltas = []
    for dataset_name, met in [('supplementary',metrics),('main100_corrected',bm)]:
        for c in freeze['conditions']:
            if c['strategy'] != 'vanilla': continue
            v = next(x['id'] for x in freeze['conditions'] if x['model']==c['model'] and x['strategy']=='verify_revise')
            deltas.append(dict(dataset=dataset_name, model=c['model'], **{k:100*(met[v][k]-met[c['id']][k]) for k in delta_keys}))
    out.mkdir(parents=True)
    def save(name, obj):
        with (out/name).open('x',encoding='utf8') as f:
            f.write(json.dumps(obj,indent=2,ensure_ascii=False)+'\n' if not isinstance(obj,str) else obj+'\n')
    save('comparison.json',dict(supplementary=metrics,main100_corrected=bm,differences_pp=deltas,pair_sensitivity=sensitivity,transitions=transitions,generation_summary=generation))
    blocks = ['# Supporting comparison tables', 'All percentages are descriptive. Baseline=vanilla_v1; verify_revise=verify_revise_v2. Each supplementary condition has two stories (one positive, one negative), one pair and one writing prompt. Each main100 condition has 100 stories (50 positive, 50 negative). Main100 pair accuracy is not defined as a verified original/modified pair measure.']
    for label,met in [('Supplementary',metrics),('Main100 corrected',bm)]:
        blocks += ['## '+label, table(['Condition','TP','FP / negatives','TN','FN / positives','Invalid + / -','Terminal','Accuracy %','Precision %','Recall %','F1 %','Pair correct / pairs','CE Full correct / all','CE Pos correct / positives'],[
            [cid,m['tp'],f"{m['fp']}/{m['gold_negative']}",m['tn'],f"{m['fn']}/{m['gold_positive']}",f"{m['invalid_gold_positive']}/{m['invalid_gold_negative']}",m['terminal_failures'],*[f"{100*m[k]:.2f}" for k in ['accuracy','precision','recall','f1']],f"{m['pair_correct']}/{m['pair_total']}" if 'pair_total' in m else 'N/A',f"{round(m['ceeval_full']*m['total'])}/{m['total']}",f"{round(m['ceeval_pos']*m['gold_positive'])}/{m['gold_positive']}"] for cid,m in met.items()])]
    blocks += ['## Verify_revise minus baseline (percentage points)',table(['Dataset','Detector']+delta_keys,[[x['dataset'],x['model']]+[f"{x[k]:+.2f}" for k in delta_keys] for x in deltas]),'## Pair sensitivity (counts; denominator one per row)',table(list(sensitivity[0]),[list(x.values()) for x in sensitivity]),'## Paired transitions',table(['Detector','Pair','Baseline','Verify_revise','Pair result','Story changes'],[[x['model'],x['pair_id'],x['baseline'],x['verify_revise'],x['pair_change'],json.dumps(x['story_changes'])] for x in transitions])]
    save('tables.md','\n\n'.join(blocks))
    errors=[]
    for cid,sid in sorted(selected):
        r=by_key[cid,sid]
        errors.append(dict(condition=cid,story_id=sid,pair_id=prov[sid]['pair_id'],version=prov[sid]['version'],gold=gold[sid],predicted=r['parsed']['label'],source=d.cp_path(cfg,cid,sid).as_posix(),raw_response_sha256=d.hash_text(r['response_text']),response_text=r['response_text'],parsed=r['parsed']))
    save('selected_error_records.json',errors)
    save('review_context.json',dict(selection_rule='Union of all changed final labels between strategies, both versions of pairs neither strategy classifies fully correctly, and any unusable records. Include both strategy outputs. Sort by condition and story ID. No quality-based representative selection.',selected_records=len(errors),pairs=pairs,provenance=list(prov.values()),exposure=exposure))
    # Recheck every consumed source after computation and output creation.
    assert all(d.sha(p)==h for p,h in hashes.items())
    save('audit.json',dict(version='benchmark_connection_v1',status='PASS' if not missing else 'PARTIAL',inference_calls=0,network_disabled=True,source_preservation_verified=True,
                          expected=12*len(pairs),final_records=len(records),missing=missing,duplicate_final_records=0,
                          readiness=ready,journals=journal_summary,source_sha256=hashes,
                          benchmark_records=len(traces),benchmark_invalid=sum(m['invalid'] for m in bm.values()),
                          supplementary_wrapper_warnings=sum(bool(x['warnings']) for x in results),
                          record_results=results,script_sha256=d.sha(__file__)))
    print(json.dumps(dict(output=str(out),records=len(records),expected=len(expected),selected_error_records=len(errors),source_files_preserved=len(hashes))))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
