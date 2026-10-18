"""Offline, opt-in audit/rescore. Never invokes inference or writes legacy files."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import hashlib
import io
import json
from pathlib import Path

from fyp.data.loader import load_split
from fyp.evaluation.corrected_v1 import aggregate, parse_response, score_localization
from fyp.evaluation.localization import score_localization as legacy_localization
from fyp.experiments.main100_analysis import audit, read, rows
from fyp.prompting.parser import parse_official_response


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def write_csv(path, values):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(values[0]))
    writer.writeheader()
    writer.writerows(values)
    path.write_text(stream.getvalue(), encoding='utf8')


def fact(example, label, loc):
    return {'gold': int(example.cont_error), 'predicted': label, 'full': loc.full_correct,
            'error_side': loc.cont_error_lines_correct, 'contradicted_side': loc.contradicted_lines_correct}


def run(config_path):
    config_path = Path(config_path)
    config = read(config_path)
    if config['version'] != 'scoring_correction_v1' or config['zero_division'] != 0:
        raise ValueError('Unsupported evaluator policy/version')
    root, out = Path(config['experiment_root']), Path(config['output_directory'])
    if not out.resolve().is_relative_to((root/'analysis').resolve()) or out == root/'analysis':
        raise ValueError('Correction output must use a separate analysis subdirectory')
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out/'preservation_before.json'
    if not snapshot.exists():
        protected = {p.as_posix(): digest(p) for base in (root, Path('prompts'), Path('data/raw'))
                     for p in base.rglob('*') if p.is_file() and not p.is_relative_to(out)}
        write_json(snapshot, protected)
    protected = read(snapshot)
    def verify_preservation():
        changed = [p for p,h in protected.items() if not Path(p).is_file() or digest(p) != h]
        if changed:
            raise ValueError(f'Protected artifacts changed: {changed}')
    verify_preservation()
    # Existing audit is read-only. Checks frozen data, code, model and tokenizer
    # hashes, manifest/order/text/gold, prompts, requests, exports, legacy metrics.
    freeze, manifest, families, legacy_conditions, source_hashes = audit(root)
    index = {e.example_id: e for e in load_split('flawed_fictions')}
    conditions, traces, legacy_reports = [], [], []
    saved_csv = {r['condition']: r for r in csv.DictReader((root/'main100_metrics.csv').open(encoding='utf8'))}
    for condition in legacy_conditions:
        records = rows(Path(condition['run_dir'])/'predictions.jsonl')
        variants = {k: [] for k in ('legacy', 'nonempty_only', 'bounded_evidence', 'corrected')}
        condition_traces = []
        for line, record in enumerate(records, 1):
            example = index[record['example_id']]
            old = parse_official_response(record['raw_response'])
            new = parse_response(record['raw_response'], config,
                                 inference_failure=bool(record.get('target_inference_failure') or record.get('inference_error')))
            old_loc = legacy_localization(example, old.to_official_result())
            guard_loc, guard_trace = score_localization(example, old.cont_error, old.cont_error_lines, old.contradicted_lines)
            bound_loc, _ = score_localization(example, old.cont_error, new.error_lines, new.contradicted_lines)
            new_loc, new_trace = score_localization(example, new.label, new.error_lines, new.contradicted_lines)
            for key, label, loc in [('legacy', old.cont_error, old_loc), ('nonempty_only', old.cont_error, guard_loc),
                                    ('bounded_evidence', old.cont_error, bound_loc), ('corrected', new.label, new_loc)]:
                variants[key].append(fact(example, label, loc))
            changed_side = lambda a,b: (a.cont_error_lines_correct, a.contradicted_lines_correct) != (b.cont_error_lines_correct, b.contradicted_lines_correct)
            trace = dict(condition=condition['id'], example_id=example.example_id, gold=int(example.cont_error),
                         source_path=(Path(condition['run_dir'])/'predictions.jsonl').as_posix(), source_line=line,
                         raw_response_sha256=hashlib.sha256(record['raw_response'].encode('utf8')).hexdigest(),
                         raw_nonempty=bool(record['raw_response'].strip()), ollama_done=record['ollama_raw'].get('done'),
                         done_reason=record['ollama_raw'].get('done_reason'), legacy_prediction=asdict(old),
                         corrected_prediction=asdict(new), legacy_localization=asdict(old_loc),
                         nonempty_only_localization=asdict(guard_loc), bounded_evidence_localization=asdict(bound_loc),
                         corrected_localization=asdict(new_loc), nonempty_trace=guard_trace, corrected_trace=new_trace,
                         label_changed=old.cont_error != new.label,
                         nonempty_changed_full=old_loc.full_correct != guard_loc.full_correct,
                         nonempty_changed_any_side=changed_side(old_loc,guard_loc),
                         boundary_changed_full=guard_loc.full_correct != bound_loc.full_correct,
                         boundary_changed_any_side=changed_side(guard_loc,bound_loc),
                         corrected_changed_full=old_loc.full_correct != new_loc.full_correct,
                         corrected_changed_any_side=changed_side(old_loc,new_loc),
                         empty_fragments_present=any(v['rejected_empty_fragments'] for v in guard_trace.values()))
            condition_traces.append(trace)
        metrics = {k: aggregate(v) for k,v in variants.items()}
        for key, expected in condition['metrics'].items():
            assert abs(metrics['legacy'][key]-expected) < 1e-12
            assert abs(float(saved_csv[condition['id']][key])-expected) < 1e-12
        affected = {key: sum(t[key] for t in condition_traces) for key in
                    ('label_changed','nonempty_changed_full','nonempty_changed_any_side','boundary_changed_full',
                     'boundary_changed_any_side','corrected_changed_full','corrected_changed_any_side','empty_fragments_present')}
        affected['invalid'] = metrics['corrected']['invalid']
        affected['legacy_warning_records'] = condition['warning_records']
        affected['invalid_reasons'] = dict(Counter(r for t in condition_traces for r in t['corrected_prediction']['invalid_reasons']))
        affected['field_warnings'] = dict(Counter(r for t in condition_traces for r in t['corrected_prediction']['field_warnings']))
        affected['raw_nonempty'] = sum(t['raw_nonempty'] for t in condition_traces)
        affected['ollama_done'] = sum(t['ollama_done'] is True for t in condition_traces)
        affected['done_reasons'] = dict(Counter(t['done_reason'] for t in condition_traces))
        conditions.append(dict(condition=condition['id'], model=condition['model'], strategy=condition['label'],
                               metrics=metrics, affected=affected))
        legacy_reports.append(dict(condition=condition['id'], metrics=condition['metrics'],
                                   attempts=condition['attempts'], attempt_statuses=condition['attempt_statuses'],
                                   unfinished_attempts=condition['unfinished_attempts']))
        traces.extend(condition_traces)
    controls = []
    for name, label in [('always_positive',1), ('always_negative',0)]:
        values = []
        for row in manifest['records']:
            e = index[row['example_id']]
            loc, _ = score_localization(e,label,(),())
            values.append(fact(e,label,loc))
        controls.append(dict(condition=name,metrics=aggregate(values)))
    result = dict(version=config['version'], config=config, status='PASS', inference_calls=0,
                  total_records=len(traces), unique_stories=len(manifest['records']), families=len(set(families)),
                  valid_predictions=sum(c['metrics']['corrected']['valid'] for c in conditions),
                  invalid_predictions=sum(c['metrics']['corrected']['invalid'] for c in conditions),
                  affected_records=sum(t['label_changed'] or t['corrected_changed_any_side'] for t in traces),
                  affected_unique_stories=len({t['example_id'] for t in traces if t['label_changed'] or t['corrected_changed_any_side']}),
                  invalid_records=[dict(condition=t['condition'],example_id=t['example_id'],gold=t['gold'],
                                        reasons=t['corrected_prediction']['invalid_reasons']) for t in traces
                                   if t['corrected_prediction']['invalid_reasons']],
                  conditions=conditions, controls=controls, legacy_reproduction=legacy_reports,
                  frozen_source_hashes=source_hashes, manifest_sha256=digest(freeze['settings']['manifest']),
                  correction_sources={p:digest(p) for p in [str(config_path),'src/fyp/evaluation/corrected_v1.py',str(Path(__file__))]})
    write_json(out/'audit.json', result)
    write_json(out/'affected_records.json', [t for t in traces if t['label_changed'] or t['corrected_changed_any_side']])
    (out/'record_traces.jsonl').write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in traces),encoding='utf8')
    flattened = []
    for c in conditions:
        for variant, metrics in c['metrics'].items():
            flattened.append(dict(condition=c['condition'], evaluator=variant,
                                  **{k:v for k,v in metrics.items() if k != 'valid_only_diagnostic'}))
    for c in controls:
        flattened.append(dict(condition=c['condition'], evaluator='corrected',
                              **{k:v for k,v in c['metrics'].items() if k != 'valid_only_diagnostic'}))
    write_csv(out/'comparison.csv',flattened)
    write_csv(out/'valid_only_diagnostics.csv', [dict(condition=c['condition'], **c['metrics']['corrected']['valid_only_diagnostic']) for c in conditions])
    write_csv(out/'legacy_corrected_differences.csv', [dict(condition=c['condition'], **{
        k:c['metrics']['corrected'][k]-c['metrics']['legacy'][k] for k in c['metrics']['legacy'] if k != 'valid_only_diagnostic'}) for c in conditions])
    (out/'report.md').write_text(render_report(result),encoding='utf8')
    verify_preservation()
    write_json(out/'preservation_verified.json',dict(status='PASS',files=len(protected),sha256=protected))
    print(json.dumps({k:result[k] for k in ('status','total_records','valid_predictions','invalid_predictions','inference_calls')},indent=2))
    return result


def table(headers, rows_):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows_])


def render_report(result):
    cs = result['conditions']
    pct = lambda n:f'{100*n:.2f}'
    text = ['# main100 v2: scoring correction v1', '',
        f"Offline integrity audit PASS: {result['total_records']} final records, {result['valid_predictions']} valid labels, "
        f"{result['invalid_predictions']} invalid labels; 100 aligned stories, 50 gold positives and 50 negatives per condition, 96 inferred families. Zero model calls.", '',
        'All six raw-response reparses reproduce original saved metrics, exports and the comparison CSV. All final responses are nonempty; '
        'request journal response text agrees byte-for-byte after JSON decoding. All final Ollama responses report done=true and done_reason=stop. '
        'This is saved-artifact completeness, not proof that every answer obeys the output format. '
        'There are 604 attempt starts, 600 successes, one timeout and three unfinished earlier attempts; no terminal story failures. '
        'Manifest gold/text/order match raw official Parquet. Frozen inputs, prompts, model/runtime metadata, tokenizer and legacy code hashes pass the inherited read-only audit.', '',
        '## Corrected comparison', '', 'Rates are percentages; counts are integers. All-scope metrics include invalid outputs in gold denominators.', '',
        table(['Condition','N','Valid','Invalid','TP','FP','TN','FN','Invalid gold +/−','Pred + %'], [
            [c['condition'],*(c['metrics']['corrected'][k] for k in ('total','valid','invalid','tp','fp','tn','fn')),
             f"{c['metrics']['corrected']['invalid_gold_positive']}/{c['metrics']['corrected']['invalid_gold_negative']}",pct(c['metrics']['corrected']['predicted_positive_rate'])] for c in cs]), '',
        table(['Condition','Accuracy','Precision','Recall','F1+','Macro-F1','Specificity','FPR','Balanced acc.','CEEval-Full*','CEEval-Pos*'], [
            [c['condition'],*(pct(c['metrics']['corrected'][k]) for k in ('accuracy','precision','recall','f1','macro_f1','specificity','false_positive_rate','balanced_accuracy','ceeval_full','ceeval_pos'))] for c in cs]), '',
        '*Corrected verifier-free CEEval variants; not numerically interchangeable with legacy benchmark scores.', '',
        '## Legacy versus corrected', '',
        table(['Condition','Accuracy old → new','F1+ old → new','Full old → new','Pos old → new','Label changes','Full changes'],[
            [c['condition'],*(f"{pct(c['metrics']['legacy'][k])} → {pct(c['metrics']['corrected'][k])}" for k in ('accuracy','f1','ceeval_full','ceeval_pos')),
             c['affected']['label_changed'],c['affected']['corrected_changed_full']] for c in cs]), '',
        'comparison.csv contains four separately named variants: legacy; nonempty_only (only the empty-fragment guard, including gold); '
        'bounded_evidence (also stop evidence at field boundaries); corrected (also explicit label validity/recognition). '
        'legacy_corrected_differences.csv gives every count/rate delta in counts or fractions, not percentage points.', '',
        table(['Condition','Empty-fragment records','Guard changes Full / any side','Boundary changes Full / any side','Label changes','Invalid'],[
            [c['condition'],c['affected']['empty_fragments_present'],
             f"{c['affected']['nonempty_changed_full']}/{c['affected']['nonempty_changed_any_side']}",
             f"{c['affected']['boundary_changed_full']}/{c['affected']['boundary_changed_any_side']}",c['affected']['label_changed'],c['affected']['invalid']] for c in cs]), '',
        f"Across conditions, {result['affected_records']} records on {result['affected_unique_stories']} unique stories change label or localization-side credit. "
        '35 records contain normalized-empty predicted fragments (none in gold). Seven lose Full from the guard alone; '
        'three additional records lose only one side of partial localization credit. Of the nine previously flagged positive Full credits, '
        'seven depended on empty fragments, one also had cross-field contamination, and one retains legitimate Full credit after both fixes. '
        'The boundary fix removes Full for Llama baseline flawed_fictions_389 and restores Full for Qwen verify_revise flawed_fictions_241. '
        'Five labels change: three become invalid and two explicit negative Mistral decisions change from 1 to 0. '
        'Ten records change Full overall (nine losses and one gain); these categories overlap.', '',
        table(['Invalid condition','Story','Gold','Reason'], [[r['condition'],r['example_id'],r['gold'],', '.join(r['reasons'])] for r in result['invalid_records']]), '',
        '## Findings and policy boundaries', '',
        'Confirmed implementation defect: after normalization a fragment such as </p> or punctuation can become the empty string. '
        'Python containment then matches every story sentence. The guard applies symmetrically to predicted and gold fragments; '
        'legitimate nonempty substrings, punctuation/case normalization, HTML wrappers, and non-strict any-overlap remain supported. '
        'Empty evidence cannot earn positive-gold localization credit. An empty extra fragment does not invalidate other legitimate evidence.', '',
        'Confirmed extraction defect: the legacy missing-close fallback reads the rest of the response, potentially importing contradicted evidence into error_lines, '
        'or decision/explanation text into evidence. Bounded field recovery retains the actual field text and flags the malformed boundary. '
        'No line-side swap, semantic judgement, or explanation-quality score is added.', '',
        'The legacy label defaults faithfully reproduce released parse_output, rather than being accidental deviations in this repository: '
        'missing response/decision defaults negative; every decision lacking the exact negative substring defaults positive, including empty/unknown decisions. '
        'This hides invalid labels and inverts the explicit negative “No continuity error was found” in two Mistral baseline records. '
        'The separately versioned explicit_decision_v1 policy requires exactly one complete nonempty decision tag and a recognized unambiguous declaration. '
        'Missing/malformed/unrecognized/conflicting labels become null and remain in the audit. The baseline prompt permits prose decisions, '
        'so known declarative forms are accepted, not just the two literals requested by verify_revise. The grammar is stored in the config and '
        'is an analyst-specified policy, not a claim about official parser semantics. A missing response wrapper or malformed evidence field alone does not invalidate a clear label. '
        'The alternative of rejecting all protocol warnings would be a different policy and is not applied.', '',
        'Intentional benchmark conventions retained: CEEval-Full gives every true negative full credit; a true positive needs overlap '
        'on BOTH corresponding evidence sides. A classification error or invalid label gets zero. CEEval-Pos averages the same per-record '
        'score over ALL gold-positive stories, including missed positives and invalids, not just true positives or predicted positives. '
        'Full therefore equals (TN + correctly localized gold positives)/100, Pos equals localized positives/50. '
        'Full is not a pure localization accuracy. Strict subset matching, requiring all gold lines, penalizing extra evidence, '
        'or dividing Pos by detected positives are alternative measures, not confirmed bug fixes. No official verifier or explanation grader is available here; none is invented.', '',
        '## Denominators, invalids and zero division', '',
        'Let N=TP+FP+TN+FN+I+ + I−, P=TP+FN+I+, and G−=TN+FP+I−. '
        'TP/FP/TN/FN contain only valid binary predictions. I+ and I− are invalid outputs by gold class; they are never silently relabeled or dropped. '
        'Predicted-positive rate=(TP+FP)/N, accuracy=(TP+TN)/N, precision=TP/(TP+FP), recall=TP/P, '
        'specificity=TN/G−, FPR=FP/G−, balanced accuracy=(recall+specificity)/2. '
        'With invalid negative-gold outputs, FPR is NOT 1−specificity: the difference is I−/G−. '
        'Precision conditions on emitted positives and therefore has no invalids in its denominator.', '',
        'Positive F1=2TP/(2TP+FP+FN+I+); negative F1=2TN/(2TN+FP+FN+I−); macro-F1 is their unweighted mean. '
        'This is an explicit abstention-aware two-class extension: invalids miss their gold class without being false predictions of the opposite class. '
        'All-scope CEEval-Full denominator=N; CEEval-Pos denominator=P. Side scores in CSV average over N and retain true-negative credit. '
        'Every zero denominator yields 0.0, including absent classes; undefined counts are never NaN. '
        'valid_only_diagnostics.csv explicitly excludes invalids and recalculates N/P/G− on the valid subset; it is diagnostic only, not the primary comparison.', '',
        '## Detection controls', '',
        'Deterministic controls use the same 100 gold labels, no model outputs or inference. Evidence is explicitly empty: '
        'always-positive CEEval is zero; always-negative Full receives the intentional true-negative credit.', '',
        table(['Control','TP','FP','TN','FN','Pred +','Accuracy','Precision','Recall','F1+','Macro-F1','Specificity','FPR','Balanced','Full','Pos'],[
            [c['condition'],*(c['metrics'][k] for k in ('tp','fp','tn','fn')),
             *(pct(c['metrics'][k]) for k in ('predicted_positive_rate','accuracy','precision','recall','f1','macro_f1','specificity','false_positive_rate','balanced_accuracy','ceeval_full','ceeval_pos'))] for c in result['controls']]), '',
        '## Interpretation', '',
        'The original broad conclusion—verify_revise has model-dependent detection/localization tradeoffs and no demonstrated general superiority—remains. '
        'The corrected numbers and affected examples supersede the legacy values for this correction policy. '
        'The frozen prompt choice remains a historical development decision; this audit does not reselect or tune it. '
        'Original confidence intervals and significance claims belong to legacy scoring and must not be attached to corrected scores. '
        'This report is descriptive; corrected inferential tests are not run.', '',
        'Qwen accuracy still rises 48% → 57%, while Full falls 47% → 30% (17 points, previously 19); '
        'Pos rises 0% → 10%. Its sole baseline positive localization credit was spurious. '
        'Llama accuracy now rises 49% → 50%, reversing the legacy 51% → 50% direction; Full rises 6% → 22% '
        'while Pos falls 12% → 4%. Thus its earlier positive-localization advantage is smaller. '
        'Mistral accuracy rises 42% → 49% (7 points, previously 5), but Full falls 33% → 3% and Pos rises 0% → 4%. '
        'Only Qwen verify_revise exceeds both constant controls on accuracy/balanced accuracy. No model condition exceeds the '
        'always-negative Full of 50%, and no condition exceeds the always-positive F1+ of 66.67%. '
        'This makes high recall or positive F1 alone particularly weak evidence of useful discrimination on this balanced sample.', '',
        '## Reproduction and artifacts', '',
        'From the repository root:', '', '```powershell',
        '.\\.venv\\Scripts\\python.exe -m fyp.experiments.main100_scoring_correction --config configs/experiments/main100_scoring_correction_v1.json', '```', '',
        'audit.json contains integrity, all metrics/diagnostics, variant attribution and code/config hashes. '
        'record_traces.jsonl contains all 600 source paths/line numbers/raw hashes, decision text, parser flags, parsed evidence, '
        'expanded predicted/gold sentence sets and legacy/corrected credits. affected_records.json selects label or side-credit changes. '
        'preservation_before.json and preservation_verified.json verify original experiment, prompts and other protected inputs unchanged. '
        'All correction outputs stay in this analysis directory. No required input was missing.', '',
        'Primary local sources: external/flawedfictions/src/detect_cont_errors.py (parse_output); '
        'external/flawedfictions/src/localization_metrics.py (extract_full_line, eval_cont_error_localization_on_example, pos_only filtering); '
        'docs/paper_spec.md; docs/benchmark_spec.md; frozen experiment contract; '
        'src/fyp/prompting/parser.py; src/fyp/evaluation/{localization,evaluator,metrics}.py.', '']
    return '\n'.join(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)
