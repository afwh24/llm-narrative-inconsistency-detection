"""Read-only annotation validation and diagnostic counts; no inference/scoring."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from fyp.experiments.main100_manual_review_simple import FIELDS, STRATEGIES, explanation_status


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def validate(rows, selected):
    expected={r['review_id']:r for r in selected}
    errors=[];missing=[];seen=set()
    for row in rows:
        rid=row.get('review_id')
        if rid not in expected or rid in seen:
            errors.append({'review_id':rid,'issue':'Unexpected or duplicate ID'});continue
        seen.add(rid)
        if any(row.get(k)!=expected[rid][k] for k in ('story_id','model')):
            errors.append({'review_id':rid,'issue':'Identifier mismatch'})
        if set(row)!=set(FIELDS) or None in row.values():
            errors.append({'review_id':rid,'issue':'Malformed CSV row'})
        for s in STRATEGIES:
            field=s+'_explanation_supported';value=row.get(field)
            if value is None or not value.strip():missing.append({'review_id':rid,'field':field})
            elif value not in ('Yes','No','Unsure'):
                errors.append({'review_id':rid,'field':field,'value':value,'issue':'Not exactly Yes/No/Unsure'})
    for rid in expected.keys()-seen:
        errors.append({'review_id':rid,'issue':'Missing row'})
        missing.extend({'review_id':rid,'field':s+'_explanation_supported'} for s in STRATEGIES)
    return errors,missing


def counts(rows, statuses):
    result={}
    names={('No','Yes'):'verify_revise_better',('Yes','No'):'baseline_better',
           ('Yes','Yes'):'both_yes',('No','No'):'both_no'}
    for model in dict.fromkeys(r['model'] for r in rows):
        group=[r for r in rows if r['model']==model]
        state={s:{v:sum(r[s+'_explanation_supported']==v for r in group) for v in ('Yes','No','Unsure')} for s in STRATEGIES}
        pairs={key:[] for key in (*names.values(),'uncertain','no_explanation_separate','incomplete_or_invalid')}
        for r in group:
            rid=r['review_id'];pair=tuple(r[s+'_explanation_supported'] for s in STRATEGIES)
            if any(statuses[rid][s]=='No explanation' for s in STRATEGIES):key='no_explanation_separate'
            elif 'Unsure' in pair:key='uncertain'
            else:key=names.get(pair,'incomplete_or_invalid')
            pairs[key].append(rid)
        result[model]={'judgments':state,'pair_ids':pairs,'pair_counts':{k:len(v) for k,v in pairs.items()}}
    return result


def run(config_path):
    config=read(config_path);root=Path(config['review_directory'])
    protected={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
    path=root/'manual_review_simple.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
        if reader.fieldnames!=list(FIELDS):raise ValueError('Unexpected CSV columns; source preserved')
    selected=read(root/'review_rows.json');manifest=read(root/'selection_manifest.json')
    if sha(root/'review_rows.json')!=read(root/'validation.json')['output_sha256']['review_rows.json']:
        raise ValueError('Original selected source snapshot changed')
    expected_ids=[model+':'+sid for model,ids in manifest['selected_ids'].items() for sid in ids]
    if [r['review_id'] for r in selected]!=expected_ids:raise ValueError('Selected review manifest mismatch')
    errors,missing=validate(rows,selected)
    raw_cache={};statuses={};source_checks=[]
    for r in selected:
        statuses[r['review_id']]={}
        for s in STRATEGIES:
            source,line=r[s+'_source'].rsplit(':',1)
            if source not in raw_cache:
                raw_cache[source]=[json.loads(v) for v in Path(source).read_text(encoding='utf8').splitlines() if v.strip()]
            actual=raw_cache[source][int(line)-1]
            if actual['example_id']!=r['story_id'] or actual['raw_response']!=r[s+'_raw_response']:
                raise ValueError('Saved response/source mismatch')
            if hashlib.sha256(actual['raw_response'].encode()).hexdigest()!=r[s+'_raw_sha256']:
                raise ValueError('Saved response hash mismatch')
            statuses[r['review_id']][s]=explanation_status(actual['raw_response'],config['explicit_no_explanation_markers'])
        source_checks.append(r['review_id'])
    result={'annotation_sha256':sha(path),'annotation_file':str(path),'human_annotations_verbatim':rows,
            'validation_errors':errors,'missing_required_judgments':missing,'form_complete':not errors and not missing,
            'counts':counts(rows,statuses),'explanation_presence_status':statuses,
            'no_explanation_responses':sum(v=='No explanation' for pair in statuses.values() for v in pair.values()),
            'source_verified_review_ids':source_checks,'selected_comparisons':len(selected),
            'distinct_stories':len({r['story_id'] for r in selected}),
            'input_hashes':protected,'raw_input_hashes':{p:sha(p) for p in raw_cache},
            'analysis_source_sha256':sha(__file__),'inference_calls':0,
            'scope':'Selected, unblinded diagnostic sample. Counts are descriptive of entered annotations only; no population estimates.'}
    output=root/('human_review_analysis_'+result['annotation_sha256'][:12]+'.json')
    # First snapshot remains immutable; later revalidation never overwrites it.
    if not output.exists():output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    if any(sha(p)!=h for p,h in protected.items()):raise ValueError('Existing review artifact changed')
    print(json.dumps({'output':str(output),'complete':result['form_complete'],'missing':len(missing),'errors':errors,
                      'no_explanation':result['no_explanation_responses'],'counts':result['counts']},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    run(p.parse_args().config)
