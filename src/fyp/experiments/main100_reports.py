"""Render audited v2 analysis artifacts; never dispatch inference."""
from pathlib import Path
from fyp.experiments.main100_analysis import read, save_versioned, save_json, METRICS

ROOT = Path('experiments/main100_comparison_v2_ollama0400')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] + ['| ' + ' | '.join(map(str,row)) + ' |' for row in rows]) + '\n\n'


def render():
    r = read(ROOT/'analysis/main100_analysis_v1.json')
    cs = r['conditions']
    report = '# Main 100 comparison: audited v2 results\n\n'
    report += '''Audit outcome: **PASS, all six conditions complete**. There are 600 final records, 100 per condition, zero pending and zero terminal failures. No inference was run for this analysis. All six exports agree with metrics recomputed from raw responses using the established parser and evaluator.

Only v2 predictions, request journals, metadata and exports were used. Its contract explicitly pins a shared manifest and frozen prompt copies stored under v1; these shared inputs were read solely as v2 provenance, never as v1 results. The pilot is development evidence and contributes no evaluation observations here. The recorded common strategy remains **verify_revise**, template version **verify_revise_v2**; baseline remains **vanilla_v1**. Neither was reselected or edited.

## Integrity and scope

The audit verifies 100 unique expected official IDs per condition, their original types in the simple exports, full original dataset text and fixed manifest ordering. All six use the same 50 positive and 50 negative main-subset stories. The 100 stories form 96 frozen inferred families: 92 singleton families and four pairs. No retry is counted as an extra story. Existing progress agrees with all records.

Run fingerprints, saved YAML configurations, frozen provenance, prompt hashes, model configuration and runtime identity metadata agree. Every recorded application request matches the original full-text rendered prompt and effective model options, including serialized request-body hashes; no evaluator gold is assembled into requests. Raw successful responses reparse identically. Parser/evaluator/model-config/dataset/tokenizer source hashes match the freeze. This establishes saved-record consistency, not independent attestation of the historical server.

There are 604 attempt-start records: 600 completed responses, one timeout and three starts without a recorded outcome. These unfinished attempts remain preserved in retry histories; they are not terminal story failures. Every story ultimately has a successful final record. All final Ollama done reasons are `stop`, not `length`.

Shared settings: Ollama 0.40.0; Q4_K_M models; temperature 0.5, seed 42, context 16384, output allowance 4096, top_k 40, top_p 0.9, repeat_penalty 1.0; timeout 300 seconds; max_retries 3 (four total attempts). Qwen uses think=false; other models omit think. Complete model digests and configuration/source hashes are retained in preparation.json, per-run metadata and analysis/main100_analysis_v1.json. No settings were changed during analysis.

## Recomputed metrics

All values below are percentages; failures are counts. CSV metrics and intervals use fractions from 0 to 1.

'''
    report += table(['Condition','N','Failed','Accuracy','Precision','Recall','F1','CEEval-Full','CEEval-Pos'], [[c['id'],100,c['failed'],*[f"{100*c['metrics'][m]:.2f}" for m in METRICS]] for c in cs])
    report += '## Strategy minus baseline\n\nChanges are percentage points, paired on the same stories.\n\n'
    report += table(['Model',*METRICS],[[c['model'],*[f"{c['delta_pp'][m]:+.2f}" for m in METRICS]] for c in r['comparisons']])
    report += '''Qwen3, the primary model, gains 31 true positives but also 22 false positives. Its 9-point accuracy increase is uncertain, while CEEval-Full falls 19 points (exploratory paired cluster 95% CI −29.00 to −9.09). CEEval-Pos rises from 1/50 to 4/50, a small change with an interval spanning zero. These results do not establish superiority of the selected prompt on the primary endpoint.

Llama's baseline predicts positive for 99/100 stories. The strategy reduces false alarms from 49 to 30 but misses 20 positives previously detected. Full rises through negative-story credit while Pos falls from 12/50 to 2/50. Mistral's strategy predicts positive for 97/100 stories: recall improves but false alarms rise from 17 to 49, and Full drops 31 points. High recall/F1 therefore does not imply good contradiction localisation.

## Paired changes and exploratory tests

Improved means baseline score 0 to strategy score 1; regressed means 1 to 0. Classification and Full use all 100 stories; Pos uses the 50 gold positives. Full includes correct-negative credit and is not a pure positive-evidence endpoint.

'''
    report += table(['Model','Endpoint','Improved','Regressed','Both correct','Both wrong','Family exact p','Holm p (9)'],[[c['model'],m,t['improved'],t['regressed'],t['both_correct'],t['both_wrong'],f"{t['family_exact_p']:.6g}",f"{t['holm_p_9_tests']:.6g}"] for c in r['comparisons'] for m,t in c['paired'].items()])
    report += '''The v2 contract does not contain a detailed inferential plan. `docs/final_experiment_design.md` describes McNemar tests, paired bootstrap intervals and Holm adjustment for a different four-comparison experiment. All inferential results here are **exploratory adaptations**, specified in `configs/experiments/main100_analysis_v1.json`, not preregistered confirmatory tests.

We use 20,000 paired cluster-bootstrap draws, seed 20261008, resampling the 96 frozen inferred families with replacement. All members of each sampled family are retained together, with identical draws across conditions; metrics remain story-weighted. Sampling is unstratified, so replicate class counts and total story counts fluctuate. Intervals are 95% percentile intervals with linear quantiles, not simultaneous intervals. There were no zero-positive draws. Exact family-block sign-randomization tests swap condition labels jointly within families and use two-sided absolute score differences. Holm adjustment covers nine tests (three models × accuracy/Full/Pos). This requires exchangeability under the null; heuristic family assignments may miss dependence. No formal cross-model superiority test is made.

After Holm adjustment, the Full reductions for Qwen and Mistral and the Pos reduction for Llama meet 0.05 in these exploratory tests. Llama's localisation comparison is especially sensitive to the inherited empty-fragment scoring issue described below. Classification differences do not meet this threshold. Unadjusted ordinary exact McNemar p-values below are reference results only: they do not account for related families.

'''
    report += table(['Model','Reference McNemar p'],[[c['model'],f"{c['paired']['accuracy']['mcnemar_reference_p']:.6g}"] for c in r['comparisons']])
    report += '## Exploratory 95% confidence intervals\n\nCondition intervals are percentages.\n\n'
    report += table(['Condition','Metric','Estimate','95% interval'],[[c['id'],m,f"{100*c['metrics'][m]:.2f}",f"[{100*c['confidence_intervals'][m][0]:.2f}, {100*c['confidence_intervals'][m][1]:.2f}]"] for c in cs for m in METRICS])
    report += 'Paired difference intervals are percentage points.\n\n'
    report += table(['Model','Metric','Difference','95% interval'],[[c['model'],m,f"{c['delta_pp'][m]:+.2f}",f"[{c['delta_ci_pp'][m][0]:+.2f}, {c['delta_ci_pp'][m][1]:+.2f}]"] for c in r['comparisons'] for m in METRICS])
    report += '''## Evaluator behavior and limits

Missing response or decision tags default to a negative prediction with warnings; an existing decision is negative only when it contains the parser's expected negative phrase, otherwise positive (including unexpected/empty decisions). Missing close tags can retain the remaining response inside a section. Parser warnings are not terminal failures and receive no new penalty. Missing evidence sections use the existing placeholder behavior. Valid negative exports retain false and empty evidence arrays.

Accuracy includes terminal failures as incorrect; failed positives contribute FN, failed negatives do not add FP. Failures score zero on localisation and stay in denominators. Precision/recall/F1 zero denominators follow the existing zero convention. There are no terminal failures in this dataset, so those policies do not change these results.

Correct negatives get Full=1. For positives, correct classification and overlap on both evidence sides are required, using existing sentence tokenization, normalization and containment expansion, not exact evidence equality. Empty normalized fragments can match all sentences. Nine positive credited records co-occur with this issue (Qwen baseline 1, Llama baseline 7, Mistral baseline 1). These are flags, not a count of counterfactually invalidated scores. No rescoring or penalty was introduced. See the error report for a concrete demonstrably unrelated evidence example.

The sample is small, balanced by design, deterministically selected after development exclusions, and not representative of deployment prevalence. Families are inferred rather than official source IDs. Intervals describe resampling stability of this sample, not variability across generation seeds, runtimes or future datasets; boundary estimates can yield degenerate percentile intervals. Pilot selection and subsequent evaluation are distinct; no pilot observations were pooled. Explanation quality is not directly measured by these six metrics. No detector wording is selected or tuned from this analysis.

## Artifacts and reproduction

`main100_metrics.csv` contains six rows with recomputed metrics, confusion counts, warning/attempt counts and exploratory intervals. `main100_error_analysis.md` gives error categories and traceable cases. `analysis/main100_analysis_v1.json` is evaluator-only: it includes gold annotations, per-story outcomes, paired changed-ID lists, hashes and statistical configuration. It is not a replacement for the gold-free simple exports. All six original completed exports remain under `exports/`; richer raw records and request journals remain under `runs/<run_id>/`.

From the repository root, run only offline analysis:

```powershell
.\\.venv\\Scripts\\python.exe -m fyp.experiments.main100_analysis
if ($LASTEXITCODE -ne 0) { throw 'Analysis audit failed' }
.\\.venv\\Scripts\\python.exe -m fyp.experiments.main100_reports
```

Existing changed reports are preserved by content hash under adjacent `report_history/` directories. Analysis does not write predictions, progress, requests, prompts or the experiment contract.

'''
    report += f"Manifest: `{r['scope']['manifest']}`. SHA256: `{r['scope']['manifest_hash']}`.\n\n"
    report += table(['Detector','Version','Original path','Frozen copy','SHA256'],[[name,t['version'],t['path'],t['copy'],t['hash']] for name,t in r['templates'].items()])
    report += 'These exact existing templates are the detector handoff for upcoming generated-story evaluation. No new experiment or story generation was started.\n'
    save_versioned(ROOT/'main100_comparison.md',report)

    error = '# Main 100 error analysis — v2 only\n\nAll 600 final records are present; zero terminal failures. Categories below follow gold labels and the unchanged evaluator. Qualitative cases are purposive illustrations, not a separately blinded or exhaustive semantic annotation study.\n\n'
    error += table(['Condition','TP','FP (false alarms)','FN (missed)','TN','Both evidence sides / TP','Error only','Contradicted only','Neither'],[[c['id'],*[c['confusion'][k] for k in ('tp','fp','fn','tn')],*[c['positive_localization'].get(k,0) for k in ('both_sides','error_only','contradicted_only','neither_side')]] for c in cs])
    error += '''Qwen baseline misses 49/50 contradictions; strategy misses 18/50 but raises false alarms to 25/50. Among strategy true positives, 28/32 fail full localisation. Llama strategy has 28/30 such failures; Mistral strategy has 46/48. Thus many positive decisions do not identify the benchmark's contradiction pair. These totals include scorer behavior; they do not directly rate explanations.

## Traceable qualitative cases

- **Missed temporal contradiction, `flawed_fictions_357`:** gold marks sunshine immediately after a midnight bell. Qwen baseline declares consistency and misses it. Qwen verify_revise quotes sunshine and midnight and earns both evidence sides, despite a malformed explanation closing tag. Mistral verify_revise predicts positive but instead claims that a sixty-year-old woman having owned a kitten as a child is inconsistent; its evidence does not localise the annotated error.
- **False alarm on a coherent revelation, `flawed_fictions_265`:** this gold-negative story distinguishes a blind man's imagined appearance of his wife from what he sees after his sight returns. Qwen baseline accepts it; Qwen verify_revise flags the difference as contradictory. Both Mistral conditions also flag this premise. A narrative explanation/revelation is being treated as an incompatible fact.
- **Correct decision, wrong supplied evidence, `flawed_fictions_317`:** gold changes a jeweled drinking-horn into a simple wooden cup. Qwen verify_revise's explanation mentions that change, but its error evidence is the Goblin never returning and its earlier quote is abbreviated; Full=0. This illustrates why a plausible explanation does not substitute for the two evidence fields. Qwen baseline misses the positive entirely.
- **Scorer credit despite unrelated evidence, `flawed_fictions_117`:** gold concerns a letter's Washington versus London postmark. Qwen baseline supplies ship-departure and ship-sinking sentences instead. Sentence splitting leaves punctuation/HTML-only fragments (`"</p>`) on both sides, which normalize to empty strings and expand to all story sentences. The inherited evaluator awards Full=1 despite the quoted evidence addressing a different issue. The strategy still discusses the ship, but lacks those empty fragments and gets Full=0. This paired regression is not evidence of worse semantic localisation.
- **Parser-default negative, `flawed_fictions_236`, Llama baseline:** raw prose claims a contradiction, but uses `(decision)` instead of `<decision>`. The parser records `missing_decision_tag` and defaults negative. Because this story is gold-negative, it receives classification and Full credit; it is Llama baseline's only true negative. The analysis does not reinterpret the prose or replace that outcome.

Each case can be located by official story ID in the condition's `predictions.jsonl`, the gold-free export, and the evaluator-only analysis JSON. The raw response and parsed fields remain unchanged. These cases were inspected only after this evaluation; they must not be reused as uninspected development material for a future detector test.

## Technical and parser outcomes

'''
    error += table(['Condition','Warning records / 100','Attempt starts','Unfinished historical attempts','Terminal failures'],[[c['id'],c['warning_records'],c['attempts'],c['unfinished_attempts'],c['failed']] for c in cs])
    for c in cs:
        error += f"- **{c['id']}:** " + (', '.join(f'{k}={v}' for k,v in c['warning_counts'].items()) or 'no parser warnings') + '.\n'
    error += '''
Warning categories overlap. In particular, missing outer close tags in Mistral are retained as warnings, not converted into inference failures. All 100 Mistral strategy records have at least one parser warning. Qwen strategy has 21 malformed/missing explanation closes; retained explanation text may include subsequent tags. This limits export explanation cleanliness, independently of classification.

Mistral baseline story `flawed_fictions_283` has a first-attempt OllamaTimeoutError at 302.065009 seconds followed by a successful retry. Three other historical starts have no finished event (one each in Qwen baseline, Llama baseline and Llama strategy); they remain unfinished rather than being retrospectively labeled failures. In total, 604 starts produce 600 final stories. No retry or interrupted attempt is a seventh condition or extra example. All successful final done reasons are `stop`; missing close tags alone are not proof of output-budget truncation.

## Empty-fragment flags on positive credited records

The following are co-occurrence flags, not corrected scores. A flagged fragment may coexist with otherwise sufficient evidence, so these counts should not all be subtracted from CEEval. The explicit Qwen case above establishes at least one misleading credit. All original metrics are preserved.

'''
    for c in cs:
        if c['empty_fragment_positive_credits']:
            error += f"- **{c['id']}:** " + ', '.join(c['empty_fragment_positive_credits']) + '.\n'
    error += '\nFull per-story evidence fields, gold annotations, warning lists and improved/regressed IDs are in `analysis/main100_analysis_v1.json`. The main comparison report contains all paired counts, family-aware intervals and exploratory statistical tests. No new failure penalties, parser repairs, gold edits or detector changes were made.\n'
    save_versioned(ROOT/'main100_error_analysis.md',error)
    save_json(ROOT/'analysis/detector_handoff.json',{'selected_strategy':'verify_revise','templates':r['templates'],'instruction':'Reuse these exact hash-verified templates; do not tune wording from main100 results.','source_experiment':ROOT.as_posix()})


if __name__ == '__main__':
    render()
