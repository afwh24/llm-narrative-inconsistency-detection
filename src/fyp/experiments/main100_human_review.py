"""Prepare a deterministic, unannotated paired review packet from saved results.

No inference imports. Existing deliverables are never overwritten, including
partially edited Markdown and CSV annotations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess

from fyp.data.loader import load_split

STRATEGIES = ('baseline', 'verify_revise')
QUESTIONS = ('detection_correct', 'evidence_correct', 'explanation_correct', 'error_category', 'reviewer_notes')
HUMAN_FIELDS = tuple(f'{s}_human_{q}' for s in STRATEGIES for q in QUESTIONS) + ('human_questionable_gold', 'human_gold_notes')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)


def outcome(gold, label):
    if label is None:
        return 'INVALID'
    return {(1,1):'TP',(1,0):'FN',(0,1):'FP',(0,0):'TN'}[(gold,label)]


def candidate(baseline, verify):
    if baseline['example_id'] != verify['example_id'] or baseline['gold'] != verify['gold']:
        raise ValueError('Pair alignment/gold mismatch')
    gold = baseline['gold']
    labels = [r['corrected_prediction']['label'] for r in (baseline,verify)]
    correct = [p == gold for p in labels]
    stratum = {(False,True):'baseline_wrong_verify_correct',(True,False):'baseline_correct_verify_wrong',
               (False,False):'both_wrong',(True,True):'both_correct'}[tuple(correct)]
    features = {f'{s}_{outcome(gold,p).lower()}' for s,p in zip(STRATEGIES,labels)}
    if any(r['label_changed'] or r['corrected_changed_any_side'] for r in (baseline,verify)):
        features.add('evaluator_affected')
    if None in labels:
        features.add('invalid_output')
    return dict(example_id=baseline['example_id'], gold=gold, stratum=stratum, features=sorted(features),
                baseline=baseline, verify_revise=verify)


def select(candidates, model, config):
    """Stable selection independent of filesystem/record enumeration order."""
    if len({c['example_id'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate story')
    quota = config['stratum_quotas']
    availability = Counter(f for c in candidates for f in c['features'])
    strata = Counter(c['stratum'] for c in candidates)
    if any(strata[s] < n for s,n in quota.items()):
        raise ValueError(f'Insufficient transition strata: {strata}')
    minima = {f:min(n,availability[f]) for f,n in config['coverage_minima'].items()}
    rank = lambda c: hashlib.sha256(f"{config['selection_salt']}:{model}:{c['example_id']}".encode()).hexdigest()
    ordered = sorted(candidates,key=rank)
    selected, used, covered = [], Counter(), Counter()
    while len(selected) < sum(quota.values()):
        available = [c for c in ordered if c not in selected and used[c['stratum']] < quota[c['stratum']]]
        if not available:
            raise ValueError('Cannot fill review quotas')
        score = lambda c: sum(1/availability[f] for f,n in minima.items() if covered[f]<n and f in c['features'])
        chosen = max(available,key=score)  # Stable hash order breaks equal scores.
        selected.append(chosen)
        used[chosen['stratum']] += 1
        covered.update(chosen['features'])
    if any(covered[f]<n for f,n in minima.items()):
        raise ValueError(f'Coverage constraints unresolved: {covered}; required {minima}')
    selected.sort(key=lambda c:(list(quota).index(c['stratum']),rank(c)))
    return selected, dict(available_strata=dict(strata),selected_strata=dict(used),
                           available_features=dict(availability),selected_features=dict(covered),
                           unavailable_features=[f for f in minima if availability[f]==0],
                           selection_hashes={c['example_id']:rank(c) for c in selected})


def review_row(model, item, example, records):
    reason = f"AUTOMATED: {item['stratum']}; " + ', '.join(item['features'])
    row = dict(review_id=f"{model}:{example.example_id}", model=model, story_id=example.example_id,
               selection_reason=reason, automated_stratum=item['stratum'],
               **{field:'' for field in HUMAN_FIELDS},
               story=example.story, gold_label=int(example.cont_error), gold_explanation=example.cont_error_expl,
               gold_error_lines=example.cont_error_lines_raw, gold_contradicted_lines=example.contradicted_lines_raw)
    for strategy in STRATEGIES:
        t=item[strategy]
        record=records[t['condition']][example.example_id]
        raw=record['raw_response']
        if hashlib.sha256(raw.encode('utf8')).hexdigest()!=t['raw_response_sha256']:
            raise ValueError('Raw response changed since correction audit')
        parsed=t['corrected_prediction']
        projection = {k:v for k,v in record['parsed_prediction'].items() if k != 'cont_error_expl_detailed'}
        flags = [f for f in ('label_changed','nonempty_changed_full','nonempty_changed_any_side',
                             'boundary_changed_full','boundary_changed_any_side','corrected_changed_full',
                             'corrected_changed_any_side') if t[f]]
        row.update({f'{strategy}_raw_response':raw, f'{strategy}_legacy_parsed_prediction':dump(projection),
                    f'{strategy}_corrected_parsed_prediction':dump(parsed),
                    f'{strategy}_automated_label': 'INVALID' if parsed['label'] is None else parsed['label'],
                    f'{strategy}_automated_outcome':outcome(item['gold'],parsed['label']),
                    f'{strategy}_automated_ceeval_full':t['corrected_localization']['full_correct'],
                    f'{strategy}_automated_evaluator_changes':', '.join(flags) or 'none',
                    f'{strategy}_source':f"{t['source_path']}:{t['source_line']}",
                    f'{strategy}_raw_sha256':t['raw_response_sha256']})
    return row


RUBRIC = """## Review rubric

Use the editable CSV as the primary annotation file. One row is one model/story pair; both responses share a story and gold annotation. The Markdown is a reading copy, but any handwritten/Markdown annotations will also be preserved. If you annotate both formats, identify which is authoritative. Blank means **unreviewed**, never “No”.

For each response enter **Yes / No / Unclear / N/A** for Detection correct?, Evidence correct?, and Explanation correct?. Use N/A only where the dimension cannot be assessed; explain it in notes. For an invalid or missing decision use Detection=N/A and error category invalid_output. If prose suggests an intended decision, note it without silently replacing the parser result. An unambiguous decision with a malformed outer wrapper need not be semantically invalid.

- **Detection:** Does the response reach the defensible continuity-error decision in the full story? Gold is a reference, not an instruction to agree. Record gold concerns separately.
- **Evidence:** Are both quoted passages faithful to the story, on the correct sides, and sufficient to establish an unresolved contradiction? Correct classification does not guarantee correct evidence. On a negative decision with no evidence requested/provided, use N/A; evaluate an asserted supporting argument under Explanation.
- **Explanation:** Is the stated reasoning supported by the story and does it establish or rule out the contradiction? Respect fictional rules, changes over time, revelations, character beliefs and explicit resolutions. Assess the response actually written, not a better argument one could invent. Use N/A for absent/unusable explanations, with a note.

Choose one or more error categories separated by semicolons. Use **none** only after review when no error applies:

| Category | Meaning |
| --- | --- |
| unsupported_contradiction | Flags a conflict not established by the story: invented premise, real-world assumption, ordinary change, or resolved revelation. |
| missed_contradiction | Misses a supported, unresolved contradiction. |
| ambiguity | Multiple defensible readings or insufficient information prevent a confident judgment; explain the competing readings. |
| wrong_evidence | Misquotes, cites unrelated/insufficient passages, omits a required side, or reverses the error and contradicted passages. |
| invalid_output | Missing, malformed, conflicting or uninterpretable decision/output. Distinguish label failure from a harmless wrapper warning. |
| unsupported_explanation | Explanation introduces false facts or a reasoning gap even if the decision/evidence is otherwise correct. |
| none | Reviewed with no identified error. |
| other | Describe the issue in reviewer notes. |

For each pair separately enter **Yes / No / Unclear** in Questionable gold annotation? and explain concerns in Gold notes. Identify whether the label, quote, quote direction or explanation is questionable. Do not edit the gold fields. Repeat-story disagreements across models should be reconciled later.

Automated labels, TP/FP/TN/FN, transition strata, parsing flags and CEEval values below are **selection aids only, not human judgments**. INVALID is retained separately, not called a false negative. For transition selection, an invalid output is not correct, so it belongs to a wrong/not-correct stratum. Legacy parsed fields may contain malformed-tag spillover; compare them with the raw response and corrected fields. This is an unblinded, deliberately selected diagnostic sample. Do not estimate whole-dataset error percentages from it.

Read the story and make a provisional judgment before consulting the displayed model interpretations. Compare the two responses independently; do not assume verify_revise is better. No human fields have been prefilled.
"""


def cell(value):
    return html.escape(str(value),quote=False).replace('|','&#124;').replace('\r','&#13;').replace('\n','<br>')


def markdown(rows, coverage, config):
    lines=['# main100 v2 paired human review', '',
           'Status: **AWAITING HUMAN REVIEW**. 30 model/story pairs, 60 responses. Ten distinct stories per model; overlap across models is allowed.', '',
           'Full stories, gold annotations and raw responses are copied from saved inputs without truncation. HTML/XML is escaped for display; exact text and line breaks are retained in the CSV. No inference or prompt changes.', '',
           RUBRIC, '', '## Selection and coverage', '',
           'For each model: three baseline-wrong/verify-correct pairs, three baseline-correct/verify-wrong pairs, three both-wrong pairs, and one both-correct reference. '
           'Within these quotas, selection first covers available strategy-specific false positives/negatives, at least two evaluator-affected stories, and an invalid output where available. '
           'Candidate selection uses deterministic rare-feature coverage, with SHA-256 of salt:model:story_id breaking ties. '
           f"Salt: `{config['selection_salt']}`. Exact inputs, hashes, coverage and selected IDs are in selection_manifest.json.", '',
           '| Model | Improved | Regressed | Both wrong | Both correct | Evaluator affected | Invalid output | Unavailable requested coverage |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for model,c in coverage.items():
        st=c['selected_strata'];fe=c['selected_features']
        lines.append('| '+' | '.join([model,*[str(st.get(k,0)) for k in config['stratum_quotas']],str(fe.get('evaluator_affected',0)),str(fe.get('invalid_output',0)),', '.join(c['unavailable_features']) or 'none'])+' |')
    lines += ['', '## Case index', '', '| Case | Model | Story | Automated selection reason |', '| --- | --- | --- | --- |']
    for i,r in enumerate(rows,1):
        lines.append(f"| [Case {i:02d}](#case-{i:02d}) | {r['model']} | {r['story_id']} | {r['selection_reason']} |")
    for i,r in enumerate(rows,1):
        lines += ['', f'<a id="case-{i:02d}"></a>', f"## Case {i:02d}: {r['model']} / {r['story_id']}", '',
                  r['selection_reason'], '', '### Full story', '',cell(r['story']), '', '### Official gold annotation', '',
                  f"Label: **{r['gold_label']}** (1=continuity error; 0=no continuity error).", '',
                  'Explanation: '+cell(r['gold_explanation']), '', 'Error-introducing lines: '+cell(r['gold_error_lines']), '',
                  'Earlier contradicted lines: '+cell(r['gold_contradicted_lines']), '',
                  '| Human gold review | Judgment |', '| --- | --- |', '| Questionable gold annotation? | |', '| Gold notes | |', '',
                  '### Baseline and verify_revise', '', '| Field | Baseline | verify_revise |', '| --- | --- | --- |']
        for title,key in [('Raw response (full)','raw_response'),('Corrected parsed prediction','corrected_parsed_prediction'),
                          ('Legacy parsed prediction (raw repeated field omitted)','legacy_parsed_prediction'),
                          ('AUTOMATED detection outcome','automated_outcome'),('AUTOMATED CEEval-Full credit','automated_ceeval_full'),
                          ('AUTOMATED evaluator changes','automated_evaluator_changes')]:
            lines.append('| '+title+' | '+cell(r['baseline_'+key])+' | '+cell(r['verify_revise_'+key])+' |')
        lines += ['', '### Human response review — leave blank until reviewed', '', '| Human field | Baseline | verify_revise |', '| --- | --- | --- |',
                  '| Detection correct? | | |','| Evidence correct? | | |','| Explanation correct? | | |','| Error category | | |','| Reviewer notes | | |', '',
                  'Source baseline: `'+r['baseline_source']+'`', '', 'Source verify_revise: `'+r['verify_revise_source']+'`', '']
    lines += ['## What remains unresolved', '',
              'All 60 responses require three judgments and an error-category decision (240 entries); all 30 pairs require a gold-questionability decision. '
              'That is 270 pending decisions plus 90 optional notes fields. No completed human annotations were found during the initial repository inspection. '
              'Existing main100_error_analysis.md contains automated/assistant-selected examples, not signed human review.', '',
              'The corrected evaluation supports examining positive detections with failed evidence matching, semantic false alarms, missed contradictions, '
              'and malformed decisions. It does not establish human explanation accuracy or the prevalence of these categories. '
              'An evidence_first strategy could be worth a future controlled test if review confirms that requiring two faithful, contradictory quotes addresses the observed failures. '
              'If quotes are faithful but the claimed conflict is unsupported/resolved, quote-first formatting alone would not address that problem. '
              'Any reviewed stories are now development-exposed and should not be treated as unseen evaluation for a tuned strategy.', '']
    return '\n'.join(lines)


def require_new_output(out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Review directory is nonempty; preserve annotations and use a new version: {out}')


def save_unannotated_draft(path, content):
    """Idempotent completion may read exact drafts, never replace edited files."""
    expected = content.replace('\n', os.linesep).encode('utf8')
    if path.exists():
        if path.read_bytes() != expected:
            raise ValueError(f'Existing review content differs; preserve possible annotations: {path}')
    else:
        path.write_bytes(expected)


def run(config_path, finish_unannotated_export=False):
    config=read(config_path)
    root=Path(config['experiment_root']); out=Path(config['output_directory']); corrected=Path(config['corrected_analysis'])
    if not out.resolve().is_relative_to((root/'analysis').resolve()) or out.resolve()==corrected.resolve():
        raise ValueError('Review output must be an isolated analysis directory')
    if finish_unannotated_export:
        if not out.is_dir() or not {p.name for p in out.iterdir()}.issubset({'review_rows.json','review.md','review.csv'}):
            raise ValueError('Initial-export completion is only for untouched initial drafts without a manifest')
    else:
        require_new_output(out)
    protected={p.as_posix():sha(p) for base in (root,Path('prompts'),Path('data/raw')) for p in base.rglob('*') if p.is_file()}
    for p,h in read(corrected/'preservation_before.json').items():
        if sha(p)!=h: raise ValueError(f'Protected experiment input changed: {p}')
    audit=read(corrected/'audit.json')
    if audit['status']!='PASS' or audit['total_records']!=600: raise ValueError('Corrected audit not complete')
    for p,h in audit['correction_sources'].items():
        if sha(p)!=h: raise ValueError(f'Correction provenance changed: {p}')
    traces=jsonl(corrected/'record_traces.jsonl')
    if len(traces)!=600: raise ValueError('Missing corrected traces')
    freeze=read(root/'preparation.json')
    manifest=read(freeze['settings']['manifest'])
    ids=[r['example_id'] for r in manifest['records']]
    records={};by_condition={}
    for c in freeze['conditions']:
        rs=jsonl(Path(c['run_dir'])/'predictions.jsonl')
        if [r['example_id'] for r in rs]!=ids: raise ValueError('Raw record alignment changed')
        records[c['id']]={r['example_id']:r for r in rs}
        ts=[t for t in traces if t['condition']==c['id']]
        if [t['example_id'] for t in ts]!=ids: raise ValueError('Trace alignment changed')
        by_condition[c['id']]={t['example_id']:t for t in ts}
    examples={e.example_id:e for e in load_split('flawed_fictions')}
    for entry in manifest['records']:
        e=examples[entry['example_id']]
        if e.cont_error!=entry['cont_error'] or hashlib.sha256(e.story.encode()).hexdigest().upper()!=entry['story_hash']:
            raise ValueError('Gold/story mismatch')
    review_rows=[]; coverage={}; selected_ids={}
    for model in config['models']:
        candidates=[candidate(by_condition[model+'_baseline'][sid],by_condition[model+'_verify_revise'][sid]) for sid in ids]
        chosen,coverage[model]=select(candidates,model,config)
        selected_ids[model]=[c['example_id'] for c in chosen]
        review_rows.extend(review_row(model,c,examples[c['example_id']],records) for c in chosen)
    if any(r[f] for r in review_rows for f in HUMAN_FIELDS): raise ValueError('Human judgments must start blank')
    out.mkdir(parents=True,exist_ok=True)
    save_unannotated_draft(out/'review_rows.json',dump(review_rows)+'\n')
    save_unannotated_draft(out/'review.md',markdown(review_rows,coverage,config))
    # Read-only package resolution: no dependency copies, links or mutations.
    exporter=Path(__file__).with_name('export_review_csv.mjs')
    if not (out/'review.csv').exists():
        subprocess.run([config['node_executable'],str(exporter.resolve()),str((out/'review_rows.json').resolve()),
                        str((out/'review.csv').resolve()),str(Path(config['node_modules']).resolve())],check=True)
    with (out/'review.csv').open(encoding='utf-8-sig',newline='') as handle:
        actual=list(csv.DictReader(handle))
    if len(actual)!=len(review_rows): raise ValueError('CSV row count changed')
    for actual_row,expected in zip(actual,review_rows):
        expected_text={k:format(v,'g') if isinstance(v,float) else str(v) for k,v in expected.items()}
        if actual_row!=expected_text: raise ValueError('CSV differs from unannotated source; preserve possible annotations')
    changed=[p for p,h in protected.items() if sha(p)!=h]
    if changed: raise ValueError(f'Protected files changed: {changed}')
    selection=dict(version=config['version'],config=config,selected_ids=selected_ids,coverage=coverage,
                   distinct_stories_across_models=len({r['story_id'] for r in review_rows}),pairs=len(review_rows),responses=2*len(review_rows),
                   source_hashes=protected,code_hashes={str(Path(__file__)):sha(__file__),str(exporter):sha(exporter),str(config_path):sha(config_path)},
                   human_fields=list(HUMAN_FIELDS),human_completed_responses=0,pending_response_decisions=8*len(review_rows),
                   pending_gold_decisions=len(review_rows),inference_calls=0,
                   inspection={'completed_human_reviews_found':False,'searched':['experiments','reports','docs'],
                               'existing_material':'main100 error analysis and scoring correction are automated; interim report review concerns report writing, not human response annotations'},
                   validation={'csv_exact_roundtrip':True,'human_fields_all_blank':True,'protected_files_unchanged':len(protected)})
    (out/'selection_manifest.json').write_text(dump(selection)+'\n',encoding='utf8')
    print(dump({k:selection[k] for k in ('pairs','responses','distinct_stories_across_models','pending_response_decisions','pending_gold_decisions','validation')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--finish-unannotated-export',action='store_true',help='Complete only unchanged initial drafts; refuses any annotations or completed manifest')
    args=parser.parse_args()
    run(args.config,args.finish_unannotated_export)
