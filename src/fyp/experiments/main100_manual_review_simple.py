"""Simplify the existing paired review without selection, inference or scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

STRATEGIES = ('baseline','verify_revise')
FIELDS = ('review_id','story_id','model','baseline_explanation_supported','verify_revise_explanation_supported','notes')
GUIDE = '''Only two human judgments are required per row, each **Yes / No / Unsure**:

- **Yes:** The explanation is supported by the story and supports the correct conclusion.
- **No:** The explanation misses or misidentifies the contradiction, invents facts, or gives unsupported reasoning, even if the predicted label is correct.
- **Unsure:** You cannot confidently judge the explanation, including when the gold annotation appears questionable.

For a “no plot hole” answer, assess whether its justification is consistent with the story. It does not need to quote conflicting evidence. Use the optional shared notes for gold concerns, evidence problems or other observations. There are no separate detection, evidence, error-category or gold judgments to complete.

Automatic label agreement and parser/status flags are reference information, not human judgments. “No explanation” identifies clearly absent explanation text; it does not fill either judgment. Ambiguous or malformed cases remain for human review. Blank CSV cells mean pending; Unsure is an entered uncertainty judgment.

Edit **manual_review_simple.csv**. Match entries by review_id (not physical CSV line number). This is the original deliberately selected diagnostic sample: findings are qualitative and must not be used as a whole-dataset explanation-quality estimate.
'''


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        return reader.fieldnames,list(reader)


def explanation_status(raw, markers):
    """Conservative presence-only flag; never assess support or invent reasoning."""
    openings=list(re.finditer(r'<explanation>',raw))
    if len(openings)>1:
        return 'Review needed: multiple explanation sections'
    if openings:
        tail=raw[openings[0].end():]
        boundary=re.search(r'</?(?:explanation|response|error_lines|contradicted_lines|decision)>|</lines>',tail)
        text=tail[:boundary.start()] if boundary else tail
        plain=re.sub(r'<[^>]+>',' ',text).strip().lower().rstrip('.')
        return 'No explanation' if plain in markers else 'Explanation text present (support not assessed)'
    # A complete decision and evidence fields alone are not explanatory prose.
    rest=re.sub(r'<(decision|error_lines|contradicted_lines)>.*?</\1>','',raw,flags=re.S)
    rest=re.sub(r'</?response>','',rest).strip()
    if not rest or re.fullmatch(r'(?:no )?continuity error(?: was)? found[.!]?',rest,re.I):
        return 'No explanation'
    return 'Review needed: no explanation tag; prose may contain reasoning'


def transfer(row):
    simple={k:row[k] for k in FIELDS[:3]}
    unresolved=[]
    for s in STRATEGIES:
        old=row.get(s+'_human_explanation_correct','').strip()
        detection=row.get(s+'_human_detection_correct','').strip()
        value=''
        if old=='Unclear': value='Unsure'
        elif old=='No' and detection!='N/A': value='No'
        elif old=='Yes' and detection=='Yes': value='Yes'
        elif old: unresolved.append({'review_id':row['review_id'],'field':s+'_human_explanation_correct',
                                     'value':old,'reason':'Old correctness judgment does not unambiguously establish the new support criterion; review manually.'})
        simple[s+'_explanation_supported']=value
    notes=[]
    for label,field in [('Baseline note','baseline_human_reviewer_notes'),('verify_revise note','verify_revise_human_reviewer_notes'),
                        ('Gold note','human_gold_notes'),('Prior gold concern','human_questionable_gold')]:
        if row.get(field,'').strip(): notes.append(label+': '+row[field])
    # Keep old labels available as attributed notes, never present them as new judgments.
    for s in STRATEGIES:
        for field in ('detection_correct','evidence_correct','error_category'):
            value=row.get(s+'_human_'+field,'')
            if value.strip(): notes.append(f'Prior detailed annotation ({s} {field}): {value}')
    simple['notes']='\n'.join(notes)
    return simple,unresolved


def validate_and_summarize(rows, references, allowed):
    lookup={r['review_id']:r for r in references}
    seen=set();errors=[];missing=[];counts={};comparisons=Counter();interpretations=[]
    for row in rows:
        rid=row.get('review_id')
        if rid not in lookup or rid in seen:
            errors.append({'review_id':rid,'error':'Unexpected or duplicate review_id'});continue
        seen.add(rid);ref=lookup[rid]
        if any(row[k]!=ref[k] for k in ('story_id','model')):errors.append({'review_id':rid,'error':'Identifier mismatch'})
        statuses=[]
        for s in STRATEGIES:
            value=row.get(s+'_explanation_supported','').strip()
            status=ref[s+'_explanation_status'];statuses.append(status)
            group=counts.setdefault(row['model']+'/'+s,Counter())
            group['No explanation status']+=status=='No explanation'
            group[value if value in allowed else 'pending' if not value else 'invalid_value']+=1
            if not value: missing.append({'review_id':rid,'field':s+'_explanation_supported'})
            elif value not in allowed: errors.append({'review_id':rid,'field':s+'_explanation_supported','value':value})
            if value=='No':
                interpretations.append({'review_id':rid,'strategy':s,'attribution':'AUTOMATED interpretation, not reviewer-supplied category',
                    'interpretation': 'Absent explanation text' if status=='No explanation' else
                    'Explanation-support concern despite automatic gold-label agreement; subtype unresolved' if ref[s+'_gold_agreement']=='Agrees' else
                    'Explanation-support concern alongside label disagreement or invalid label; subtype unresolved',
                    'reviewer_notes_verbatim':row.get('notes','')})
        pair=tuple(row.get(s+'_explanation_supported','').strip() for s in STRATEGIES)
        if 'No explanation' in statuses: comparisons['separate_no_explanation_pairs']+=1
        elif pair==('Yes','No'): comparisons['baseline_better_supported']+=1
        elif pair==('No','Yes'): comparisons['verify_revise_better_supported']+=1
        elif 'Unsure' in pair: comparisons['uncertain_pairs']+=1
        elif pair==('Yes','Yes'): comparisons['both_supported']+=1
        elif pair==('No','No'): comparisons['neither_supported']+=1
        else: comparisons['incomplete_or_invalid_pairs']+=1
    for rid in lookup.keys()-seen:
        errors.append({'review_id':rid,'error':'Missing CSV row'})
        missing.extend({'review_id':rid,'field':s+'_explanation_supported'} for s in STRATEGIES)
    return dict(status='READY_TO_SUMMARISE' if not errors and not missing else 'PENDING_OR_NEEDS_CHECKING',
                errors=errors,missing_required_judgments=missing,counts_by_model_strategy=counts,
                paired_support_counts=comparisons,automated_interpretations=interpretations,
                scope='Qualitative diagnostic sample only; no dataset-wide explanation-quality proportions.')


def cell(text):
    return html.escape(str(text),quote=False).replace('|','&#124;').replace('\n','<br>')


def render(rows,refs):
    text=['# Simplified explanation review','',GUIDE,'','## Index','','| Entry | review_id |','| --- | --- |']
    text += [f"| [Case {i:02d}](#case-{i:02d}) | {r['review_id']} |" for i,r in enumerate(rows,1)]
    for i,(r,ref) in enumerate(zip(rows,refs),1):
        text += ['',f'<a id="case-{i:02d}"></a>',f"## Case {i:02d} — {r['review_id']}",'',
                 f"**CSV review_id:** `{r['review_id']}` · **Story:** `{r['story_id']}` · **Model:** `{r['model']}`",'',
                 '### Full story','',cell(r['story']),'','### Gold reference','',
                 f"**Label:** {r['gold_label']} (1 = continuity error; 0 = no continuity error).",'',
                 '**Reference explanation:** '+cell(r['gold_explanation']),'',
                 '**Error-introducing evidence:** '+cell(r['gold_error_lines']),'',
                 '**Earlier contradicted evidence:** '+cell(r['gold_contradicted_lines']),'',
                 '| Saved response / automatic reference | Baseline | verify_revise |','| --- | --- | --- |']
        text += ['| Full raw response | '+cell(r['baseline_raw_response'])+' | '+cell(r['verify_revise_raw_response'])+' |',
                 '| Saved corrected parsed prediction | '+cell(r['baseline_corrected_parsed_prediction'])+' | '+cell(r['verify_revise_corrected_parsed_prediction'])+' |']
        for name,key in [('AUTOMATIC gold-label agreement','gold_agreement'),('AUTOMATIC explanation presence status','explanation_status'),('AUTOMATIC parser flags','parser_flags')]:
            text.append('| '+name+' | '+cell(ref['baseline_'+key])+' | '+cell(ref['verify_revise_'+key])+' |')
        text += ['', '### Your two questions — answer in the CSV','',
                 '1. **baseline_explanation_supported:** Yes / No / Unsure','2. **verify_revise_explanation_supported:** Yes / No / Unsure','',
                 '**Yes:** story-supported explanation and correct conclusion. **No:** missed/misidentified contradiction, invented facts or unsupported reasoning—even with the correct label. **Unsure:** cannot judge confidently, including questionable gold.', '',
                 'For a no-plot-hole response, assess the justification’s consistency with the story; conflicting quotes are not required. **notes:** optional shared comments. No other judgments are required.', '']
    return '\n'.join(text)


def run(config_path):
    config=read(config_path);root=Path(config['review_directory']); runtime=read(config['runtime_config'])
    sources={p.as_posix():sha(p) for p in root.iterdir() if p.is_file()}
    rows=read(root/'review_rows.json')
    _,detailed=read_csv(root/'review.csv')
    if [r['review_id'] for r in rows]!=[r['review_id'] for r in detailed]:raise ValueError('Detailed review alignment changed')
    old_validation=read(root/'validation.json')
    if sha(root/'review_rows.json')!=old_validation['output_sha256']['review_rows.json']:raise ValueError('Selected source snapshot changed')
    refs=[];simple=[];unresolved=[]
    for original,annotations in zip(rows,detailed):
        # Verify immutable source columns while allowing human annotations.
        for k,v in original.items():
            if 'human_' not in k and annotations[k]!=(format(v,'g') if isinstance(v,float) else str(v)):
                raise ValueError(f'Source field changed in detailed CSV: {original["review_id"]}/{k}')
        row,issues=transfer(annotations);simple.append(row);unresolved.extend(issues)
        ref={k:original[k] for k in FIELDS[:3]}
        for s in STRATEGIES:
            raw=original[s+'_raw_response']
            if hashlib.sha256(raw.encode()).hexdigest()!=original[s+'_raw_sha256']:raise ValueError('Saved raw response hash mismatch')
            parsed=json.loads(original[s+'_corrected_parsed_prediction']);legacy=json.loads(original[s+'_legacy_parsed_prediction'])
            label=parsed['label']
            ref[s+'_gold_agreement']='Invalid label' if label is None else 'Agrees' if label==original['gold_label'] else 'Disagrees'
            ref[s+'_explanation_status']=explanation_status(raw,config['explicit_no_explanation_markers'])
            ref[s+'_parser_flags']='; '.join(dict.fromkeys(legacy.get('parsing_failures',[])+parsed.get('invalid_reasons',[])+parsed.get('field_warnings',[]))) or 'none'
        refs.append(ref)
    if sha(root/'review.md')!=old_validation['output_sha256']['review.md']:
        unresolved.append({'file':'review.md','reason':'Detailed Markdown has changed; preserve and inspect human annotations manually. No automatic Markdown judgment transfer.'})
    target=root/'manual_review_simple.csv'
    if not target.exists():
        data=root/'manual_review_simple_initial.json'
        if data.exists() and read(data)!=simple:raise ValueError('Initial transfer snapshot differs; preserve it')
        if not data.exists():data.write_text(json.dumps(simple,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        subprocess.run([runtime['node_executable'],str(Path(__file__).with_name('export_review_csv.mjs').resolve()),str(data.resolve()),str(target.resolve()),runtime['node_modules']],check=True)
    fields,actual=read_csv(target)
    if fields!=list(FIELDS):raise ValueError('Simplified CSV must have exactly the six requested columns; existing file preserved')
    doc=root/'manual_review_simple.md'
    if not doc.exists():doc.write_text(render(rows,refs),encoding='utf8')
    result=validate_and_summarize(actual,refs,config['allowed_judgments'])
    result.update(unresolved_transfers=unresolved,references=refs,config=config,
                  note='Existing simplified CSV and Markdown are never rewritten. If present, CSV judgments are authoritative; detailed files remain preserved.')
    payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    digest=hashlib.sha256(payload.encode()).hexdigest()[:12]
    reference=root/f'manual_review_simple_status_{digest}.json'
    if not reference.exists():reference.write_text(payload,encoding='utf8')
    if any(sha(p)!=h for p,h in sources.items()):raise ValueError('An existing review file changed')
    print(json.dumps({'csv':str(target),'document':str(doc),'reference':str(reference),'status':result['status'],
                      'pending_judgments':len(result['missing_required_judgments']),'invalid_entries':len(result['errors']),
                      'unresolved_transfers':unresolved,'no_explanation_responses':sum(r[s+'_explanation_status']=='No explanation' for r in refs for s in STRATEGIES),
                      'preserved_files':len(sources)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    run(p.parse_args().config)
