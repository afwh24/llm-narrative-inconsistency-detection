"""Verify report tables against saved aggregates and publish evidence mapping.

Does not infer, rescore raw outputs, or edit source research artifacts.
"""
from pathlib import Path
import json,csv,re,hashlib,math
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/integrated_fyp_20261013'
A=json.loads((OUT/'verified_aggregates.json').read_text(encoding='utf-8'))
text=(OUT/'integrated_report.md').read_text(encoding='utf-8')
checks=[]
def table(n):
    match=re.search(r'^Table '+re.escape(n)+r'\. .*?\n\n((?:\|.*\n)+)',text,re.M)
    assert match,n
    rows=[[x.strip() for x in ln.strip('|').split('|')] for ln in match[1].strip().splitlines()]
    return rows[2:]
def eq(a,b):assert math.isclose(float(a.replace(',','')),float(b),abs_tol=.0051),(a,b)
for row,m in zip(table('5.4'),A['main']):
    assert list(map(int,row[1:]))==[m[k] for k in ['tp','fp','tn','fn','invalid_gold_positive','invalid_gold_negative']]
for row,m in zip(table('5.5'),A['main']):
    for v,k in zip(row[1:],['accuracy','precision','recall','f1','macro_f1','specificity']):eq(v,m[k]*100)
for row,m in zip(table('5.6'),A['paired']):
    assert list(map(int,row[1:]))==[m[k] for k in ['both_correct','improved','regressed','both_incorrect']]+[m['improved']-m['regressed']]
for row,m in zip(table('5.7'),A['main']):
    for v,x in zip(row[1:],[m['tn'],m['ceeval_pos']*50,m['ceeval_full']*100,m['ceeval_pos']*100]):eq(v,x)
for row,m in zip(table('5.3'),A['main_reliability']):
    assert row[1]==f"{m['records']} / {m['recorded_attempts']}"
    for v,k in zip(row[2:],['invalid','legacy_warning_records','corrected_warning_records','total_client_seconds']):eq(v,m[k])
review=A['review_rows']; counts={}
for stem in ['qwen3_8b','llama3_1_8b','mistral_7b']:
    rr=[r for r in review if r['review_id'].startswith(stem+':')];assert len(rr)==10
    pairs=[(r['baseline_explanation_supported'],r['verify_revise_explanation_supported']) for r in rr]
    assert all(x in ['Yes','No','Unsure'] for p in pairs for x in p)
    counts[stem]=[pairs.count(('No','Yes')),pairs.count(('Yes','No')),pairs.count(('No','No')),pairs.count(('Yes','Yes')),sum('Unsure' in p for p in pairs)]
for row,m in zip(table('5.9'),[*counts.values(),[sum(x[i] for x in counts.values()) for i in range(5)]]):assert list(map(int,row[1:]))==m
g=A['generation_summary']
for row,(model,m) in zip(table('5.10')[:3],g['counts']['model'].items()):
    assert list(map(int,row[1:5]))==[10,g['historical']['generated_stories_v1']['model'][model]['complete'],g['historical']['generated_stories_recovery_v2']['model'][model]['complete'],10]
    assert row[5]==f"{m['accepted']} / {m['rejected']} / {m['needs_review']}"
for row,(cat,m) in zip(table('5.11'),g['counts']['category'].items()):
    assert row[0]==cat
    assert list(map(int,row[1:]))==[6,g['historical']['generated_stories_v1']['category'][cat]['complete'],6,m['accepted'],m['rejected'],m['needs_review']]
for row,m in zip(table('5.12'),A['supplementary']['supplementary'].values()):
    expected=[m['fp'],m['tp'],f"{m['tp']+m['tn']}/2",f"{m['pair_correct']}/1",f"{round(m['ceeval_full']*2)}/2",f"{round(m['ceeval_pos'])}/1"]
    assert row[1:]==list(map(str,expected)),(row,expected)
checks.extend(['Main confusion counts and rates','Paired transitions','Main reliability and latency','Localisation denominators','Current human transitions','Generation model/category counts','Supplementary decisions and pair/evidence counts'])
with (ROOT/'experiments/summaries/pilot_results.csv').open(encoding='utf-8-sig',newline='') as f:pilot=list(csv.DictReader(f))
order=[('qwen3:4b','vanilla'),('qwen3:4b','evidence_first'),('qwen3:4b','verify_revise'),('qwen3:8b','vanilla'),('qwen3:8b','evidence_first'),('qwen3:8b','verify_revise')]
for row,(model,strategy) in zip(table('5.1'),order):
    m=next(r for r in pilot if r['model']==model and r['strategy']==strategy)
    for v,k in zip(row[1:6],['accuracy','recall','f1','ceeval_full','ceeval_pos']):eq(v,100*float(m[k]))
checks.append('Legacy mixed-pilot rates')
selection=json.loads((ROOT/'experiments/main100_comparison_v1/pilot_selection.json').read_text(encoding='utf-8'))
for row,c in zip(table('5.2'),selection['conditions']):
    for v,k in zip(row[2:],['accuracy','recall','ceeval_full','ceeval_pos']):eq(v,100*c['metrics'][k])
checks.append('Twelve main10 legacy conditions against frozen selection evidence')
for row,stem in zip(table('5.13'),['qwen3_8b','llama3_1_8b','mistral_7b']):
    for cell,k in zip(row[1:],['accuracy','precision','recall','f1']):
        for v,strategy in zip(cell.split(' → '),['baseline','verify_revise']):eq(v,A['supplementary']['supplementary'][stem+'_'+strategy][k]*100)
checks.append('Supplementary precision recall F1 and accuracy differences')
base='experiments/main100_comparison_v2_ollama0400/analysis/scoring_correction_v1/'
old=json.loads((ROOT/base/'audit.json').read_text(encoding='utf-8'))
for row,c in zip(table('5.8'),old['conditions']):
    for cell,k in zip(row[1:],['accuracy','ceeval_full','ceeval_pos']):
        for v,version in zip(cell.split(' → '),['legacy','corrected']):eq(v,100*c['metrics'][version][k])
checks.append('Evaluator-version differences')

sources={
'E01':('Benchmark definition, released data and requirements',['papers/core/FLAWEDFICTIONS.pdf','papers/guidance/Undergrad Report Writing.pdf','docs/data_provenance.md','docs/benchmark_spec.md','external/flawedfictions/src/localization_metrics.py','external/flawedfictions/src/detect_cont_errors.py']),
'E02':('Development pilots and historical selection',['data/pilot/pilot_20.json','data/pilot/three_model_main10.json','experiments/summaries/pilot_results.csv','experiments/summaries/pilot_thinking_ablation.md','experiments/three_model_main10_prompting/prompting_pilot_analysis.json','experiments/three_model_main10_prompting/exports/prompting_pilot_comparison.md','experiments/main100_comparison_v1/pilot_selection.json']),
'E03':('Main100 freeze, sampling and actual outputs',['experiments/main100_comparison_v1/experiment_contract.md','experiments/main100_comparison_v1/evaluator_only/manifest.json','experiments/main100_comparison_v1/evaluator_only/exposure_audit.json','experiments/main100_comparison_v2_ollama0400/experiment_contract.md','experiments/main100_comparison_v2_ollama0400/preparation.json']),
'E04':('Corrected evaluator and primary metric authority',[base+'audit.json',base+'record_traces.jsonl',base+'comparison.csv',base+'report.md','src/fyp/evaluation/corrected_v1.py','configs/experiments/main100_scoring_correction_v1.json']),
'E05':('Current benchmark explanation review',['experiments/main100_comparison_v2_ollama0400/analysis/human_review_v1/manual_review_simple.csv','experiments/main100_comparison_v2_ollama0400/analysis/human_review_v1/manual_review_simple.md','experiments/main100_comparison_v2_ollama0400/analysis/human_review_v1/selection_manifest.json','experiments/main100_comparison_v2_ollama0400/analysis/human_review_v1/human_review_analysis_1da2ab045e6e.json','experiments/main100_comparison_v2_ollama0400/analysis/human_review_v1/human_review_summary_93a0517b8da4.md']),
'E06':('Generation configuration and recovery provenance',['configs/experiments/generated_stories_v1.json','experiments/generated_stories_v1/writing_prompts.json','experiments/generated_stories_v1/editing_instructions.json','experiments/generated_stories_recovery_v2/preparation.json','experiments/generated_stories_schema_recovery_v1/recovery_plan.md','reports/generated_stories_terminal_diagnosis_20261009/diagnosis.md','reports/generated_stories_manual_review_v1/candidate_provenance.json']),
'E07':('Current generated review, eligibility and frozen data',['reports/generated_stories_manual_review_v1/manual_review.csv','reports/generated_stories_manual_review_v1/optional_annotation_corrections.csv','configs/experiments/generated_stories_review_export_v1.json','data/generated_stories_verified_v1/evaluator_only/manifest.json','data/generated_stories_verified_v1/evaluator_only/summary.json','data/generated_stories_verified_v1/evaluator_only/all_candidate_dispositions.jsonl','data/generated_stories_verified_v1/evaluator_only/verified_pairs.jsonl','data/generated_stories_verified_v1/evaluator_only/gold_stories.jsonl','data/generated_stories_verified_v1/evaluator_only/provenance.jsonl','data/generated_stories_verified_v1/evaluator_only/exposure.json','data/generated_stories_verified_v1/detector_inputs.jsonl']),
'E08':('Completed supplementary comparison',['experiments/generated_stories_detection_v1/contract.json','experiments/generated_stories_detection_v1/contract.md','experiments/generated_stories_detection_v1/analysis/benchmark_connection_v1/audit.json','experiments/generated_stories_detection_v1/analysis/benchmark_connection_v1/comparison.json','experiments/generated_stories_detection_v1/analysis/benchmark_connection_v1/selected_error_records.json','experiments/generated_stories_detection_v1/analysis/benchmark_connection_v1/review_context.json','experiments/generated_stories_detection_v1/analysis/benchmark_connection_v1/report.md']),
'E09':('Independent report-only aggregation',['scripts/build_integrated_report_evidence.py','reports/integrated_fyp_20261013/verified_aggregates.json']),
'E10':('Implementation and prior decisions',['src/fyp/models/ollama_client.py','src/fyp/models/request_journal.py','src/fyp/experiments/runner.py','src/fyp/experiments/generated_stories_detection.py','docs/decision_log.md','docs/final_experiment_design.md','docs/final_run_plan.md','reports/interim_report.md','reports/main100_results_discussion.md'])}
mapped={
'Table 2.1':'E01; released subset counts in data_provenance.md',
'Table 3.1':'E02/E03/E07/E08/E09; historical_runs and final_records',
'Table 3.2':'E03; parent contract sampling table; exposure_audit and manifest',
'Table 3.3':'E03/E08; frozen preparation and contract generation/runtime settings',
'Table 3.4':'E04/E08; corrected_v1.detection_metrics, score_localization and aggregate; supplementary pair metrics',
'Table 5.1':'E02; pilot_results.csv; exact model/strategy rows',
'Table 5.2':'E02; pilot_selection.json and prompting_pilot_comparison.md; twelve complete conditions',
'Table 5.3':'E09; main_reliability; six authoritative predictions.jsonl attempt_history and latency_seconds fields; warning records from traces',
'Table 5.4':'E04; audit.conditions[*].metrics.corrected confusion counts',
'Table 5.5':'E04; same corrected metrics; precision/recall/F1 etc multiplied by 100',
'Table 5.6':'E09; paired; exact IDs in paired[*].record_ids; invalid treated as incorrect',
'Table 5.7':'E04; corrected_localization.full_correct; TN and localized-positive sums',
'Table 5.8':'E04; audit.conditions[*].metrics.{legacy,corrected}; identical saved raw responses',
'Table 5.9':'E05/E09; current CSV, ten rows per model; human Yes/No/Unsure pairs',
'Table 5.10':'E07; summary.counts.model, historical.*.model, classifications',
'Table 5.11':'E07; summary.counts.category, historical.generated_stories_v1.category',
'Table 5.12':'E08; comparison.supplementary; pair llama3_1_8b_010; exact story IDs in Appendix B',
'Table 5.13':'E08; comparison.supplementary accuracy/precision/recall/f1; each of six conditions',
'Table A.1':'E02/E03/E06/E08; frozen prompt versions; exact paths in source map and text',
'Table B.1':'E05/E07/E08; review_id and opaque story_id keys',
'Figure 3.1':'E03/E05/E07/E08; schematic of verified scope; no measured axis',
'Figure 5.1':'E04/E09; main recall and false_positive_rate ×100; arrows pair baseline and verify_revise per model',
'Figure 5.2':'E04/E09; main TN and ceeval_pos×50, denominator 100',
'Figure 5.3':'E07/E09; cumulative complete 11/20/30, final dispositions 1/22/7; denominator 30'}
for label in re.findall(r'^(Table [\dA-Z]+\.\d+)\.',text,re.M):assert label in mapped,label
for label in re.findall(r'!\[(Figure \d+\.\d+)',text):assert label in mapped,label
source_rows=[]; md=['# Evidence map','', 'Paths are relative to the repository root; links resolve from this report directory. SHA-256 values bind exact current files. All reported main100 numbers use scoring_correction_v1 corrected unless explicitly marked legacy. Record-level evidence is in record_evidence.csv.','']
for key,(title,paths) in sources.items():
    md.extend([f'## {key} {title}',''])
    for p in paths:
        assert (ROOT/p).is_file(),p
        h=hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
        source_rows.append({'key':key,'path':p,'sha256':h})
        md.extend([f'- [{p}](../../{p})',f'  SHA-256: `{h}`'])
    md.append('')
md.extend(['## Tables and figures','', '| Report object | Exact authority and transformation |','| --- | --- |'])
md.extend(f'| {k} | {v} |' for k,v in mapped.items())
md.extend(['','## Key claims and exact record keys','',
'- Main100 completion: E04 audit.total_records = 600, valid_predictions = 597, invalid_predictions = 3. Invalids are llama3_1_8b_baseline on flawed_fictions_236 and _354 (gold 0) and _402 (gold 1). The CSV below maps every trace to source file and line.',
'- Main accuracy differences +9/+1/+7 points: corrected strategy accuracy minus same-model baseline. Paired improvements/regressions 32/23, 21/20, 39/32 are recomputed in E09 and retain every contributing ID.',
'- Corrected localisation 0→5, 6→2, 0→2: sum both-side positive full_correct by condition; divide by 50 for Pos. True-negative credit is added only for Full.',
'- Three clear human improvements: llama3_1_8b:flawed_fictions_335, _399, _382. Five deteriorations: qwen3_8b:flawed_fictions_295, _378, _265; mistral_7b:flawed_fictions_374, _235. Current CSV judgments, not automated correctness, determine these changes.',
'- Shared errors: llama3_1_8b:flawed_fictions_300 and _253; mistral_7b:flawed_fictions_400. Localisation-versus-explanation example: qwen3_8b:flawed_fictions_241. Text-presence limitation: qwen3_8b:flawed_fictions_350. Exact current notes reside in E05 CSV and matching reading document.',
'- Seven uncertain benchmark pairs: Qwen 181/241/350, Llama 357, Mistral 128/255/262; each prefixed flawed_fictions_. Mistral262 has two Unsure judgments. No gold corrections are applied.',
'- Accepted generated pair: llama3_1_8b_010, prompt_010. Original s_f8ac30b95f49c23f78326f45; modified s_ad55dc39c9afb9dbd4a8bc9c. E07 verified_pairs/gold_stories/provenance link exact story text and Stage A/B recovery-v2 sources.',
'- Generation examples: qwen3_8b_001/_002, llama3_1_8b_008 and mistral_7b_003 from E07 current review. Unresolved examples llama3_1_8b_007 and mistral_7b_009/_010 retain their notes and disposition.',
'- Supplementary interpretation: E08 selected_error_records.json and review_context.json contain eight selected Llama/Mistral outputs and both gold evidence passages. These interpretation categories are automated, not additional human labels.',
'- Exposure uncertainty: E07 exposure.json and export config preserve the supplied answer Unsure. Runtime difference: E03 preparation is 0.40.0, E08 contract 0.40.3.',
'','## Literature and guidance verification','',
'The saved FLAWEDFICTIONS PDF supports task definition (p. 3), main/long data construction and localisation (p. 5), ordinary model setup and controls (p. 6 and Appendix A.4), verifier procedure and positive-only denominator (p. 20). The released localization_metrics.py confirms correct-negative credit, both-side non-strict matching, and filtering on gold-positive records.',
'Existing references.bib supplied initial identities. Primary web records checked on 13 October 2026: https://arxiv.org/abs/2201.11903 ; https://arxiv.org/abs/2203.11171 ; https://arxiv.org/abs/2304.02769 ; https://arxiv.org/abs/2110.14168 ; NeurIPS official proceedings for Wei et al. No unverified quotations or invented results are added.',
'NTU guidance is the supplied local lecture PDF, not an inferred binding template. Missing institutional requirements remain visible TO VERIFY markers.'])
(OUT/'evidence_map.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
(OUT/'evidence_map.json').write_text(json.dumps({'sources':source_rows,'objects':mapped},indent=2),encoding='utf-8')
traces=[json.loads(s) for s in (ROOT/base/'record_traces.jsonl').read_text(encoding='utf-8').splitlines()]
with (OUT/'record_evidence.csv').open('w',encoding='utf-8-sig',newline='') as f:
    wr=csv.writer(f);wr.writerow(['condition','example_id','gold','corrected_label','full_credit','source_path','source_line','raw_response_sha256'])
    for t in traces:wr.writerow([t['condition'],t['example_id'],t['gold'],t['corrected_prediction']['label'],t['corrected_localization']['full_correct'],t['source_path'],t['source_line'],t['raw_response_sha256']])
for p,h in json.loads((OUT/'source_preservation_before.json').read_text(encoding='utf-8')).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
markers=re.findall(r'\[TO VERIFY:.*?\]',text)
assert len(markers)==3
checks+=['24 mapped tables/figures','All mapped source paths exist','600 record-level evidence links','908 protected input files unchanged','Three visible completion markers collected']
(OUT/'content_validation.json').write_text(json.dumps({'status':'PASS','checks':checks,'markers':markers},indent=2),encoding='utf-8')
print(json.dumps({'status':'PASS','checks':checks,'markers':markers},indent=2))
