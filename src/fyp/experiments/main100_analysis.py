"""Read-only audit/statistical analysis of v2 outputs. No inference imports/calls.

Only reports and analysis artifacts are written. Existing report versions are
preserved by content hash. A shared manifest is read only as pinned by v2;
no v1 predictions, exports, progress, or metrics are read.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics

from fyp.data.loader import load_split
from fyp.data.pilot import stable_file_hash, load_pilot_manifest
from fyp.evaluation.evaluator import evaluate_predictions, _extract_prediction
from fyp.evaluation.localization import preprocess_sent, sentence_tokenize
from fyp.prompting.parser import parse_official_response
from fyp.prompting.strategies import get_benchmark_strategy
from fyp.models.registry import load_model_config
from fyp.experiments.runner import _parsed_to_record, _fingerprint, _to_simple_yaml, _research_config, RunConfig

CONFIG = Path('configs/experiments/main100_analysis_v1.json')
METRICS = ('accuracy', 'precision', 'recall', 'f1', 'ceeval_full', 'ceeval_pos')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest().upper()


def save_versioned(path, text):
    path = Path(path)
    if path.exists():
        previous = path.read_bytes()
        if previous == text.encode('utf8'):
            return
        backup = path.parent / 'report_history' / (path.stem + '_' + hashlib.sha256(previous).hexdigest()[:16] + path.suffix)
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists() and backup.read_bytes() != previous:
            raise ValueError('Version archive collision')
        backup.write_bytes(previous)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf8'))


def save_json(path, value):
    save_versioned(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def paired(before, after):
    if len(before) != len(after):
        raise ValueError('Mismatched paired scope')
    return {'improved': sum(b > a for a, b in zip(before, after)),
            'regressed': sum(b < a for a, b in zip(before, after)),
            'both_correct': sum(a == b == 1 for a, b in zip(before, after)),
            'both_wrong': sum(a == b == 0 for a, b in zip(before, after))}


def mcnemar(improved, regressed):
    n = improved + regressed
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2**n) if n else 1.0


def family_test(differences, families):
    """Exact distribution of sum +/- family score differences under paired swaps."""
    if len(differences) != len(families):
        raise ValueError('Family/pair lengths differ')
    groups = defaultdict(int)
    for diff, family in zip(differences, families):
        groups[family] += int(diff)
    observed = abs(sum(groups.values()))
    distribution = Counter({0: 1})
    active = [v for v in groups.values() if v]
    for difference in active:
        updated = Counter()
        for total, count in distribution.items():
            updated[total + difference] += count
            updated[total - difference] += count
        distribution = updated
    return sum(n for score, n in distribution.items() if abs(score) >= observed) / 2**len(active)


def holm(values):
    result = [0.0] * len(values)
    bound = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        bound = max(bound, (len(values) - rank) * values[index])
        result[index] = min(1.0, bound)
    return result


def contributions(fact):
    gold, pred = fact['gold'], fact['predicted']
    return (1, int(gold == 1), int(gold == 1 and pred == 1),
            int(gold == 0 and pred == 1), int(gold == 1 and pred != 1),
            int(fact['classification']), int(fact['full']), int(gold == 1) * int(fact['full']))


def from_counts(values):
    n, pos, tp, fp, fn, correct, full, positive_full = values
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (correct / n, precision, recall,
            2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            full / n, positive_full / pos if pos else 0.0)


def quantile(values, probability):
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = int(location)
    return ordered[lower] + (location - lower) * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])


def bootstrap(conditions, families, replicates, seed):
    family_names = sorted(set(families))
    blocks = [[i for i, key in enumerate(families) if key == family] for family in family_names]
    sufficient = []
    for condition in conditions:
        stats = [contributions(fact) for fact in condition['facts']]
        sufficient.append([tuple(sum(stats[i][j] for i in block) for j in range(8)) for block in blocks])
    rng = random.Random(seed)
    draws = [[[] for _ in METRICS] for _ in conditions]
    zero_positive = 0
    for _ in range(replicates):
        weights = Counter(rng.randrange(len(blocks)) for _ in blocks)
        for ci, clusters in enumerate(sufficient):
            values = [sum(clusters[i][j] * weight for i, weight in weights.items()) for j in range(8)]
            if ci == 0 and values[1] == 0:
                zero_positive += 1
            for j, value in enumerate(from_counts(values)):
                draws[ci][j].append(value)
    return draws, zero_positive


def audit(root):
    freeze = read(root / 'preparation.json')
    settings = freeze['settings']
    assert Path(settings['root']) == root
    assert read('configs/experiments/' + settings['name'] + '.json') == settings
    manifest_path = Path(settings['manifest'])
    assert stable_file_hash(manifest_path) == freeze['manifest_hash']
    manifest = read(manifest_path)
    ids = [r['example_id'] for r in manifest['records']]
    assert len(ids) == len(set(ids)) == 100
    families = [r['source_family_id'] for r in manifest['records']]
    source_hashes = {}
    for path, digest in freeze['source_hashes'].items():
        if path.startswith(('src/fyp/', 'configs/models/', 'data/raw/')) or 'tokenizers/' in path:
            assert stable_file_hash(Path(path)) == digest, path
            source_hashes[path] = digest
    import nltk
    assert nltk.sent_tokenize('First sentence. Second sentence.') == ['First sentence.', 'Second sentence.']
    index = {e.example_id: e for e in load_split('flawed_fictions')}
    examples = [index[key] for key in ids]
    assert len({sha(e.story) for e in examples}) == 100
    assert sum(e.cont_error for e in examples) == 50
    for e, row in zip(examples, manifest['records']):
        assert e.cont_error == row['cont_error'] and sha(e.story) == row['story_hash']
    assert len(freeze['conditions']) == 6
    assert {(c['model'], c['strategy']) for c in freeze['conditions']} == {
        (model, strategy) for model in ('qwen3:8b', 'llama3.1:8b', 'mistral:7b')
        for strategy in ('vanilla', settings['selected_strategy'])}
    progress = read(root / 'progress.json')
    conditions = []
    for condition in freeze['conditions']:
        directory = Path(condition['run_dir'])
        assert directory.is_relative_to(root)
        records = rows(directory / 'predictions.jsonl')
        actual_ids = [r['example_id'] for r in records]
        if actual_ids != ids:
            raise ValueError(f"INCOMPLETE/INVALID {condition['id']}; missing={set(ids)-set(actual_ids)}; unexpected={set(actual_ids)-set(ids)}; duplicates={len(actual_ids)-len(set(actual_ids))}; order_match={actual_ids==ids}")
        assert len({r['cache_key'] for r in records}) == 100
        model = load_model_config(condition['model_config'])
        strategy = get_benchmark_strategy(condition['strategy'], condition['version'])
        template = freeze['templates'][condition['strategy']]
        assert stable_file_hash(strategy.template_path) == template['hash']
        assert stable_file_hash(Path(template['copy'])) == template['hash']
        metadata = read(directory / 'metadata.json')
        assert metadata['pilot_provenance'] == freeze
        assert metadata['ollama_version'] == freeze['ollama_version']
        assert metadata['end_timestamp'] and metadata['expected_example_count'] == 100
        config_keys = ('run_id','experiment_type','model','ollama_model','provider','model_configuration','thinking_mode','benchmark_strategy','prompt_version','prompt_template_hash','temperature','seed','context_configuration','dataset','pilot_provenance')
        research = {key: metadata[key] for key in config_keys}
        assert _fingerprint(research) == metadata['configuration_fingerprint']
        frozen_manifest = load_pilot_manifest(manifest_path)
        reconstructed = _research_config(RunConfig(model=condition['model'], strategy=condition['strategy'],
            subsets=(examples[0].split,), experiment_type=metadata['experiment_type'],
            manifest_path=manifest_path, provenance=freeze, prompt_version=condition['version']),
            model, condition['run_id'], manifest=frozen_manifest)
        assert reconstructed == research
        research = reconstructed
        assert (directory / 'config.yaml').read_text(encoding='utf8') == _to_simple_yaml(research)
        journal = rows(directory / 'requests.jsonl')
        starts = {r['request_id']: r for r in journal if r['event'] == 'started'}
        ends = {r['request_id']: r for r in journal if r['event'] == 'finished'}
        assert len(starts) + len(ends) == len(journal)
        warning_counts, reasons = Counter(), Counter()
        for e, record in zip(examples, records):
            prompt = strategy.build_prompt(e)
            assert record['run_id'] == condition['run_id'] and record['model'] == condition['model']
            assert record['benchmark_strategy'] == condition['strategy'] and record['subset'] == 'flawed_fictions'
            assert record['input_hash'] == sha(e.story) and record['prompt_hash'] == prompt.prompt_hash
            assert record['prompt_template_hash'] == prompt.template_hash and record['prompt_version'] == condition['version']
            assert record['model_configuration'] == {'generation': model.generation.snapshot(), 'runtime': asdict(model.runtime)}
            history = record['attempt_history']
            assert 1 <= len(history) <= model.runtime.max_retries + 1
            assert [r['attempt'] for r in history] == list(range(1, len(history) + 1))
            assert len([s for s in starts.values() if s['cache_key'] == record['cache_key']]) == len(history)
            for attempt in history:
                start = starts[attempt['request_id']]
                assert start['example_id'] == e.example_id and start['run_id'] == condition['run_id']
                assert start['model_digest'] == freeze['models'][condition['model']]['identity']['digest']
                assert start['application_request']['body'] == {**model.effective_ollama_request_fields(), 'prompt': prompt.prompt}
                assert sha(start['application_request']['body_utf8']) == start['application_request']['body_hash']
                assert json.loads(start['application_request']['body_utf8']) == start['application_request']['body']
                finish = ends.get(attempt['request_id'])
                if finish:
                    assert finish['status'] == attempt['status'] and finish['latency_seconds'] == attempt['latency_seconds']
                else:
                    assert attempt['status'] == 'unfinished'
            if not record['target_inference_failure']:
                parsed = parse_official_response(record['raw_response'])
                assert _parsed_to_record(parsed) == record['parsed_prediction']
                assert record['raw_response'] == ends[history[-1]['request_id']]['raw_response']['response']
                warning_counts.update(parsed.parsing_failures)
            reasons[(record.get('ollama_raw') or {}).get('done_reason', 'unknown')] += 1
        report = evaluate_predictions([{k:v for k,v in r.items() if k != 'parsed_prediction'} for r in records], dataset={examples[0].split: examples})
        assert read(directory / 'metrics.json') == report.metrics
        output_path = Path(condition['export'])
        assert output_path.is_relative_to(root)
        output = read(output_path)
        metrics = {**report.metrics['classification_metrics'], **{k: report.metrics['localization_metrics'][k] for k in ('ceeval_full', 'ceeval_pos')}}
        failed = report.metrics['failure_counts']['target_inference_failure']
        assert list(output) == ['model','prompt','total','failed','metrics','results']
        assert output['model'] == condition['model'] and output['prompt'] == condition['label']
        assert output['metrics'] == metrics and output['total'] == 100 and output['failed'] == failed
        assert metadata['completed_example_count'] == 100 - failed and metadata['failed_example_count'] == failed
        facts = []
        for e, record, evaluated, out, family in zip(examples, records, report.examples, output['results'], families):
            assert list(out) == ['story_id','story','status','prediction']
            assert out['story_id'] == e.raw_record['example_id'] and type(out['story_id']) is type(e.raw_record['example_id'])
            assert out['story'] == e.story
            parsed, flags = _extract_prediction(record)
            if flags['target_inference_failure']:
                assert out['prediction'] is None and out['status'].startswith('terminal_')
            else:
                pos = bool(parsed.cont_error)
                assert out['prediction'] == {'has_error': pos, 'error_sentences': list(parsed.cont_error_lines) if pos else [], 'contradicted_sentences': list(parsed.contradicted_lines) if pos else [], 'explanation': parsed.cont_error_expl}
                assert out['status'] == 'ok'
            empty = {key: [fragment for line in getattr(parsed, key) for fragment in sentence_tokenize(line) if not preprocess_sent(fragment)] if parsed else [] for key in ('cont_error_lines','contradicted_lines')}
            facts.append({'story_id': e.example_id, 'family': family, 'gold': e.cont_error,
                'predicted': evaluated.predicted_cont_error, 'classification': int(evaluated.classification_correct),
                'full': evaluated.ceeval_full, 'error_side': evaluated.cont_error_lines_correct,
                'contradicted_side': evaluated.contradicted_lines_correct, 'failure': flags['target_inference_failure'],
                'warnings': list(evaluated.parsing_failures), 'empty_fragments': empty,
                'gold_explanation': e.cont_error_expl, 'gold_error': e.cont_error_lines_raw,
                'gold_contradicted': e.contradicted_lines_raw, 'prediction': out['prediction'],
                'decision_text': parsed.cont_error_expl_short if parsed else None})
        assert len(output['results']) == 100
        p = next(row for row in progress['conditions'] if row['condition'] == condition['id'])
        assert p['completed'] == 100 and p['pending'] == 0 and p['terminal_failures'] == failed
        confusion = dict(tp=sum(x['gold']==1 and x['predicted']==1 for x in facts),
            fp=sum(x['gold']==0 and x['predicted']==1 for x in facts),
            fn=sum(x['gold']==1 and x['predicted']!=1 for x in facts),
            tn=sum(x['gold']==0 and x['predicted']==0 for x in facts))
        mistakes = Counter()
        for fact in facts:
            if fact['gold']==1 and fact['predicted']==1:
                mistakes['both_sides'] += fact['full'] == 1
                mistakes['error_only'] += fact['error_side']==1 and fact['contradicted_side']==0
                mistakes['contradicted_only'] += fact['error_side']==0 and fact['contradicted_side']==1
                mistakes['neither_side'] += fact['error_side']==0 and fact['contradicted_side']==0
        conditions.append({**condition, 'metrics': metrics, 'failed': failed, 'facts': facts,
            'confusion': confusion, 'positive_localization': dict(mistakes),
            'warning_counts': dict(warning_counts), 'warning_records': sum(bool(x['warnings']) for x in facts),
            'empty_fragment_positive_credits': [x['story_id'] for x in facts if x['gold']==1 and x['full']==1 and any(x['empty_fragments'].values())],
            'attempts': len(starts), 'attempt_statuses': dict(Counter(e['status'] for e in ends.values())),
            'unfinished_attempts': len(set(starts) - set(ends)), 'done_reasons': dict(reasons),
            'client_seconds': sum(r['latency_seconds'] for r in records)})
    assert progress['completed'] == 600 and progress['pending'] == 0
    return freeze, manifest, families, conditions, source_hashes


def analyse():
    config = read(CONFIG)
    root = Path(config['experiment_root'])
    if not root.is_dir():
        raise FileNotFoundError(root)
    protected = {p.as_posix(): stable_file_hash(p) for p in root.rglob('*') if p.is_file() and (p.name in ('experiment_contract.md','preparation.json','progress.json') or 'runs' in p.parts or 'exports' in p.parts)}
    freeze, manifest, families, conditions, sources = audit(root)
    draws, zero_pos = bootstrap(conditions, families, config['bootstrap_replicates'], config['bootstrap_seed'])
    alpha = (1 - config['confidence_level']) / 2
    interval = lambda values: [quantile(values, alpha), quantile(values, 1-alpha)]
    comparisons, tests = [], []
    for i, c in enumerate(conditions):
        c['confidence_intervals'] = {name: interval(draws[i][j]) for j,name in enumerate(METRICS)}
        stats = [contributions(x) for x in c['facts']]
        assert all(math.isclose(a,c['metrics'][name]) for a,name in zip(from_counts([sum(s[j] for s in stats) for j in range(8)]),METRICS))
        if c['strategy'] == 'vanilla':
            continue
        baseline_index = next(k for k,b in enumerate(conditions) if b['model']==c['model'] and b['strategy']=='vanilla')
        b=conditions[baseline_index]
        comparison = {'model':c['model'], 'baseline':b['id'], 'strategy':c['id'],
            'delta_pp':{k:100*(c['metrics'][k]-b['metrics'][k]) for k in METRICS},
            'delta_ci_pp':{name:interval([100*(x-y) for x,y in zip(draws[i][j],draws[baseline_index][j])]) for j,name in enumerate(METRICS)}, 'paired':{}}
        for name,field,positive in [('accuracy','classification',False),('ceeval_full','full',False),('ceeval_pos','full',True)]:
            indices=[j for j,x in enumerate(b['facts']) if not positive or x['gold']==1]
            before=[b['facts'][j][field] for j in indices];after=[c['facts'][j][field] for j in indices]
            table=paired(before,after)
            table['improved_ids']=[b['facts'][j]['story_id'] for j in indices if c['facts'][j][field]>b['facts'][j][field]]
            table['regressed_ids']=[b['facts'][j]['story_id'] for j in indices if c['facts'][j][field]<b['facts'][j][field]]
            table['family_exact_p']=family_test([int(y-x) for x,y in zip(before,after)],[families[j] for j in indices])
            tests.append(table)
            if name=='accuracy':table['mcnemar_reference_p']=mcnemar(table['improved'],table['regressed'])
            comparison['paired'][name]=table
        comparisons.append(comparison)
    for table, adjusted in zip(tests,holm([t['family_exact_p'] for t in tests])):
        table['holm_p_9_tests']=adjusted
    result={'status':'PASS_COMPLETE','config':config,'bootstrap_zero_positive_draws':zero_pos,
        'scope':{'n':100,'positive':50,'negative':50,'families':len(set(families)),
                 'family_size_distribution':dict(Counter(Counter(families).values())),
                 'manifest':freeze['settings']['manifest'],'manifest_hash':freeze['manifest_hash']},
        'conditions':conditions,'comparisons':comparisons,'source_hashes':sources,
        'protected_artifact_hashes':protected,'analysis_code_hash':stable_file_hash(Path(__file__)),
        'analysis_config_hash':stable_file_hash(CONFIG),'templates':freeze['templates'],
        'v1_result_records_used':0,'inference_calls':0}
    assert all(stable_file_hash(Path(path))==digest for path,digest in protected.items())
    save_json(root/'analysis/main100_analysis_v1.json',result)
    stream=io.StringIO(newline='')
    fields=['condition','model','prompt','total','failed',*METRICS,'tp','fp','fn','tn','parser_warning_records','request_attempts']
    fields += [f'{m}_ci95_{side}' for m in METRICS for side in ('lower','upper')]
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for c in conditions:
        row={'condition':c['id'],'model':c['model'],'prompt':c['label'],'total':100,'failed':c['failed'],**c['metrics'],**c['confusion'],'parser_warning_records':c['warning_records'],'request_attempts':c['attempts']}
        row.update({f'{m}_ci95_{side}':c['confidence_intervals'][m][i] for m in METRICS for i,side in enumerate(('lower','upper'))})
        writer.writerow(row)
    save_versioned(root/'main100_metrics.csv',stream.getvalue())
    print(json.dumps({'status':result['status'],'metrics':[{k:c[k] for k in ('id','metrics','confusion','positive_localization','warning_counts','empty_fragment_positive_credits','attempt_statuses','unfinished_attempts','done_reasons')} for c in conditions], 'comparisons':comparisons},indent=2))
    return result


if __name__=='__main__':
    analyse()
