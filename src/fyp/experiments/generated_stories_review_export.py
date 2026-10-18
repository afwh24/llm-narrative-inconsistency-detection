"""Offline human-review validation and create-only dataset export. No model client."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

JUDGMENTS = ('original_is_consistent', 'modified_has_one_clear_intended_contradiction', 'evidence_and_explanation_are_correct')
EVIDENCE = ('error_sentence', 'contradicted_sentence', 'explanation')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError('Malformed CSV row; preserve and inspect source')
    return rows


def classify(candidate, reviews, corrections, notes):
    """Uncertainty/conflicting evidence takes precedence over a definite rejection."""
    result = {'classification': 'needs_review', 'reasons': [], 'verified_annotations': None,
              'reviewed': False, 'evidence_checks': {}}
    reasons = result['reasons']
    if len(reviews) != 1:
        reasons.append('missing_review' if not reviews else 'duplicate_review_rows')
        return result
    row = reviews[0]
    judgments = [row.get(k, '').strip() for k in JUDGMENTS]
    result['reviewed'] = all(x in ('Yes', 'No', 'Unsure') for x in judgments)
    uncertain = False
    for key, value in zip(JUDGMENTS, judgments):
        if value not in ('Yes', 'No'):
            reasons.append(f'{key}: {value or "missing"}')
            uncertain = True
    if notes.get('disposition') == 'unresolved':
        reasons.append('unresolved_notes: ' + notes['reason'])
        uncertain = True
    elif row.get('notes', '').strip() and notes.get('disposition') not in ('rejection_reason', 'resolved'):
        reasons.append('notes_not_assessed')
        uncertain = True
    if len(corrections) > 1:
        reasons.append('duplicate_correction_rows')
        uncertain = True
    corrected = {k: corrections[0].get('verified_' + k, '') for k in EVIDENCE} if len(corrections) == 1 else {}
    proposed = candidate.get('intended_contradiction') or {}
    # The review packet asks judgments about generator annotations. Any changed
    # optional annotations need separate explicit approval, never implicit promotion.
    if any(corrected.values()) and any(corrected.get(k) != proposed.get(k) for k in EVIDENCE):
        reasons.append('optional_corrections_require_explicit_reverification')
        uncertain = True
    for key in EVIDENCE:
        if not isinstance(proposed.get(key), str) or not proposed[key].strip():
            reasons.append('missing_generator_annotation: ' + key)
            uncertain = True
    for key in EVIDENCE[:2]:
        quote = proposed.get(key)
        match = isinstance(quote, str) and bool(quote.strip()) and quote in (candidate.get('modified_story') or '')
        result['evidence_checks'][key] = match
        if row.get(JUDGMENTS[2], '').strip() == 'Yes' and not match:
            reasons.append('conflicts_with_evidence_yes: ' + key + '_not_exact_in_modified')
            uncertain = True
    for key, quote in corrected.items():
        if key != 'explanation' and quote and quote not in (candidate.get('modified_story') or ''):
            reasons.append('corrected_quote_not_exact_in_modified: ' + key)
            uncertain = True
    if candidate.get('original_story') == candidate.get('modified_story') and judgments[:2] == ['Yes', 'Yes']:
        reasons.append('identical_versions_conflict_with_original_yes_and_modified_yes')
        uncertain = True
    if uncertain:
        return result
    failures = [k for k, v in zip(JUDGMENTS, judgments) if v == 'No']
    if failures:
        result['classification'] = 'rejected'
        reasons.extend('human_no: ' + k for k in failures)
        return result
    if candidate.get('generation_status') != 'ok' or not all(isinstance(candidate.get(k), str) and candidate[k].strip() for k in ('original_story', 'modified_story')):
        reasons.append('technical_completeness_conflicts_with_acceptance')
        return result
    result.update(classification='accepted', verified_annotations=dict(proposed))
    reasons.append('three_yes; exact evidence verified; notes resolved; generator explanation human affirmed')
    return result


def story_exports(accepted, salt):
    inputs, gold, mapping = [], [], []
    for row in accepted:
        c, p = row['candidate'], row['provenance']
        for version, label in [('original', 0), ('modified', 1)]:
            sid = 's_' + hashlib.sha256((salt + '\0' + c['pair_id'] + '\0' + version).encode()).hexdigest()[:24]
            inputs.append({'story_id': sid, 'story': c[version + '_story']})
            evidence = row['verified_annotations'] if label else dict.fromkeys(EVIDENCE, '')
            gold.append({'story_id': sid, 'cont_error': label, 'cont_error_lines': evidence['error_sentence'],
                         'contradicted_lines': evidence['contradicted_sentence'], 'cont_error_expl': evidence['explanation']})
            mapping.append({'story_id': sid, 'pair_id': c['pair_id'], 'writing_prompt_id': c['writing_prompt_id'],
                'generator': p['model'], 'version': version, 'category': p['category'], 'source_stages': p['stages'],
                'story_sha256': hashlib.sha256(c[version + '_story'].encode()).hexdigest()})
    order = lambda x: hashlib.sha256((salt + '\0order\0' + x['story_id']).encode()).hexdigest()
    return tuple(sorted(rows, key=order) for rows in (inputs, gold, mapping))


def run(config_path):
    cfg = read(config_path)
    root = Path(cfg['root'])
    if root.exists():
        raise RuntimeError('Version output already exists: never overwrite or silently reclassify')
    for path, h in cfg['source_hashes'].items():
        if digest(path) != h:
            raise RuntimeError('Frozen input changed: ' + path)
    protected = {p.as_posix(): digest(p) for d in Path('experiments').glob('generated_stories*') for p in d.rglob('*') if p.is_file()}
    protected.update({p.as_posix(): digest(p) for p in Path(cfg['review_root']).iterdir() if p.is_file()})
    protected.update({p.as_posix(): digest(p) for p in Path('prompts').rglob('*') if p.is_file()})
    candidates = jsonl(cfg['candidates'])
    provenance = read(cfg['provenance'])
    assert len(candidates) == len({c['pair_id'] for c in candidates}) == 30
    by_id = {p['pair_id']: p for p in provenance}
    assert set(by_id) == {c['pair_id'] for c in candidates}
    reviews, corrections = defaultdict(list), defaultdict(list)
    for source, target in [(cfg['review'], reviews), (cfg['corrections'], corrections)]:
        for r in csv_rows(source):
            if r['pair_id'] not in by_id:
                raise ValueError('Unknown annotation pair ID: ' + r['pair_id'])
            target[r['pair_id']].append(r)
    # Reuse only read-only source integrity checks; never call generation/update paths.
    from fyp.experiments import generated_stories as g
    from fyp.experiments import generated_stories_schema_recovery as recovery
    _, manifest, settings, freeze = recovery.loaded()
    assert len(manifest['rows']) == 30
    selected = {}
    for row in manifest['rows']:
        a, b, _ = recovery.selected(row, settings, freeze)
        assert a['status'] == b['status'] == 'ok'
        for stage, cp in [('A', a), ('B', b)]:
            assert g.parse(cp['raw_response']['response'], stage) == (cp['fields'], None)
            assert cp['raw_response'].get('done') and cp['raw_response'].get('done_reason') != 'length'
        selected[row['pair_id']] = g.pair_record(row['pair_id'], row, a, b)
    results = []
    for c in candidates:
        pid = c['pair_id']; p = by_id[pid]
        assert c == selected[pid]
        for st in p['stages']:
            assert digest(st['checkpoint']) == st['checkpoint_sha256']
        for r in reviews[pid]:
            if r['writing_prompt_id'] != c['writing_prompt_id'] or r['model'] != p['model']:
                raise ValueError('Annotation identifier mismatch: ' + pid)
        result = classify(c, reviews[pid], corrections[pid], cfg['notes_assessment'][pid])
        result.update(candidate=c, provenance=p, source_review_rows=reviews[pid], source_correction_rows=corrections[pid], notes_assessment=cfg['notes_assessment'][pid])
        results.append(result)
    accepted = [r for r in results if r['classification'] == 'accepted']
    inputs, gold, mapping = story_exports(accepted, cfg['presentation_salt'])
    assert len(inputs) == len({x['story_id'] for x in inputs}) == 2 * len(accepted)
    assert all(set(x) == {'story_id', 'story'} for x in inputs)
    counts = {}
    for dimension in ('model', 'category'):
        groups = defaultdict(list)
        for r in results:
            groups[r['provenance'][dimension]].append(r)
        counts[dimension] = {k: {'scheduled': len(rs), 'technically_complete': len(rs), 'reviewed': sum(r['reviewed'] for r in rs),
            **{name: sum(r['classification'] == name for r in rs) for name in ('accepted', 'rejected', 'needs_review')}, 'terminal_technical_failures': 0} for k, rs in groups.items()}
    historical = {}
    for name in ('generated_stories_v1', 'generated_stories_recovery_v2'):
        data = [r for path in (Path('experiments') / name / 'exports').glob('*_generated_pairs.jsonl') for r in jsonl(path)]
        assert len(data) == 30
        historical[name] = {}
        for dimension in ('model', 'category'):
            groups = defaultdict(list)
            for r in data:
                groups[by_id[r['pair_id']][dimension]].append(r)
            historical[name][dimension] = {k:dict(scheduled=len(rs),complete=sum(x['generation_status']=='ok' for x in rs),terminal=sum(x['generation_status'].startswith('terminal') for x in rs),pending=sum(x['generation_status']!='ok' and not x['generation_status'].startswith('terminal') for x in rs)) for k,rs in groups.items()}
    prep = read(cfg['detector_freeze'])
    prompt_checks = {p: {'expected': t['hash'].lower(), 'actual': digest(p), 'matches': digest(p) == t['hash'].lower()} for t in prep['templates'].values() for p in (t['path'], t['copy'])}
    exposure = dict(user_attestation=cfg['exposure_attestation'], detector_freeze=cfg['detector_freeze'],
        detector_prepared_at=prep['prepared_at'], prompt_hash_checks=prompt_checks,
        known_exposure=['generation failure diagnosis', 'recovery schema design', 'human annotation and review', 'assistant review preparation and validation'],
        conclusion='Detector-development exposure is unknown. Existing compared detector templates match their earlier freeze if all checks pass. This does not prove absence of outside development/selection exposure. Do not claim untouched evaluation.',
        scope='Repository freeze/templates plus user attestation; not an exhaustive audit of external conversations or private work.')
    summary = dict(dataset_version=cfg['version'], counts=counts, historical=historical,
        classifications=dict(Counter(r['classification'] for r in results)), accepted_story_count=len(inputs),
        unique_accepted_writing_prompts=len({r['candidate']['writing_prompt_id'] for r in accepted}),
        source_B_counts=dict(Counter(r['provenance']['stages'][1]['source_run'] for r in results)),
        readiness='Offline artifacts ready. Frozen comparison execution not configured for this dataset; exposure unresolved, only one accepted prompt group. No inferential or untouched-test claim.',
        denominators='Scheduled is 30 overall, 10 per model, 6 per category. Reviewed means all three judgments supplied as Yes/No/Unsure (not acceptance). Acceptance yield uses scheduled; acceptance among reviewed uses reviewed. Categories are intended targets. Latest terminal failures are zero; historical failures remain in separate tables.')
    report = '# Generated-story review validation v1\n\n' + summary['denominators'] + '\n\n'
    report += 'User annotations were read without edits. Needs-review takes precedence for missing/Unsure/conflicting judgments or unresolved notes, even when another judgment is No. Clear No rows otherwise reject. Nonempty notes were assessed explicitly in the versioned configuration; no keyword-only inference of resolution. Optional changed annotations require separate approval. Exact quote matching is case- and punctuation-sensitive; no silent normalization.\n\n'
    for dimension, groups in counts.items():
        report += f'## By {dimension}\n\n|Group|Scheduled|Technical complete|Reviewed|Accepted|Rejected|Needs review|Terminal technical failures|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        for k,n in groups.items(): report += '|'+k+'|'+'|'.join(str(v) for v in n.values())+'|\n'
        report += '\n'
    report += '## Historical generation outcomes (not human acceptance)\n\n'
    for run_name, dimensions in historical.items():
        report += f'### {run_name}\n\n'
        for dimension, groups in dimensions.items():
            report += f'By {dimension}:\n\n|Group|Scheduled|Complete|Terminal|Pending|\n|---|---:|---:|---:|---:|\n'
            for k,n in groups.items(): report += '|'+k+'|'+'|'.join(str(v) for v in n.values())+'|\n'
            report += '\n'
    report += 'Recovery-v2 counts are its consolidated view, including preserved original successes, not 30 fresh attempts. Latest selection: 11 original successes, 9 recovery-v2 successes, 2 diagnostic successes and 8 schema-recovery successes. Original run had 19 terminal Stage-A parsing failures; recovery-v2 retained 9 Stage-B parsing failures and one Stage-B timeout. Later recovery does not erase them.\n\n## Pair decisions\n\n|Pair|Classification|Reason|\n|---|---|---|\n'
    for r in results: report += f"|{r['candidate']['pair_id']}|{r['classification']}|{' / '.join(r['reasons']).replace('|','/')}|\n"
    report += '\n## Export and limitations\n\nOne accepted pair produces two full stories (one negative original, one positive modified), grouped privately by writing_prompt_id. No official benchmark dataset or scoring rules were changed. Source generator annotations remain separate from human-affirmed annotations. Blank negative evidence/explanation means no gold contradiction, not fabricated explanatory prose.\n\nOnly detector_inputs.jsonl may be passed to detectors. It contains exactly story_id and story. Never pass evaluator_only files, the manifest, review notes, pair IDs, category or version. Sorting uses salted SHA-256 independent of labels; the salt and ordered IDs are recorded privately. No train/test split is created; any later split must keep writing_prompt_id families together.\n\nThe single accepted pair comes from Llama recovery v2, prompt_010, and the fictional-world category. This tiny, filtered sample cannot support broad generator/strategy comparisons or category coverage claims.\n\nExposure is unknown (researcher answered Unsure). Generation diagnosis, schema design and human review are known exposure; matching pre-existing detector prompt hashes does not establish untouched evaluation. Keep templates frozen and treat any future use as a limited diagnostic comparison. The existing main100 runner remains pinned to its own dataset; no execution configuration or inference command was created for this new dataset.\n'
    root.mkdir(parents=True)
    private = root / 'evaluator_only'; private.mkdir()
    def write(path, value, lines=False):
        text = ''.join(json.dumps(v, ensure_ascii=False)+'\n' for v in value) if lines else (value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        with path.open('x',encoding='utf-8',newline='') as f:f.write(text)
    write(root/'detector_inputs.jsonl', inputs, True)
    write(private/'verified_pairs.jsonl', [dict(**r['candidate'], verified_annotations=r['verified_annotations'], annotation_source='human affirmed generator annotations; exact quotations checked') for r in accepted], True)
    write(private/'gold_stories.jsonl', gold, True)
    write(private/'provenance.jsonl', mapping, True)
    write(private/'all_candidate_dispositions.jsonl', results, True)
    write(private/'summary.json', summary)
    write(private/'exposure.json', exposure)
    write(private/'review_validation.md', report)
    write(private/'writing_prompt_groups.json', {p: [x['story_id'] for x in mapping if x['writing_prompt_id']==p] for p in sorted({x['writing_prompt_id'] for x in mapping})})
    assert all(digest(p)==h for p,h in protected.items()), 'Protected source changed'
    write(private/'manifest.json', dict(version=cfg['version'], source_hashes=cfg['source_hashes'], config_path=str(config_path), config_sha256=digest(config_path), implementation_sha256=digest(__file__),
        output_sha256={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()},
        presentation_salt=cfg['presentation_salt'], presentation_rule='SHA256(salt + NUL + order + NUL + opaque_story_id) ascending', presentation_order=[x['story_id'] for x in inputs],
        protected_source_hashes=protected, source_preservation_verified=True, inference_performed=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiments/generated_stories_review_export_v1.json')
    print(json.dumps(run(parser.parse_args().config),indent=2))


if __name__ == '__main__':main()
