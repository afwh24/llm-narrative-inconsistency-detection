"""Offline exposure/family audit. Never prints story text or invokes inference."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re

from fyp.data.loader import load_all_splits
from fyp.data.pilot import stable_file_hash


ID_PATTERN = re.compile(r"flawed_fictions(?:_error_resolved_negs|_cf_negs|_long)?_\d+\b")


def words(text):
    return re.findall(r"\w+", html.unescape(re.sub(r"<[^>]+>", " ", text)).lower())


def shingles(text, size):
    tokens = words(text)
    return {tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def text_fields(value):
    """Unpack nested JSON logs so escaped story text is compared as text."""
    if isinstance(value, dict):
        for item in value.values():
            yield from text_fields(item)
    elif isinstance(value, list):
        for item in value:
            yield from text_fields(item)
    elif isinstance(value, str):
        yield value
        if value.startswith(('{', '[')):
            try:
                parsed = json.loads(value)
            except ValueError:
                return
            yield from text_fields(parsed)


def build_families(examples, size, threshold):
    """Conservative inferred families, never pretend these are official IDs."""
    sets = [shingles(e.story, size) for e in examples]
    parent = list(range(len(examples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    index = defaultdict(list)
    edges = []
    for i, tokens in enumerate(sets):
        overlaps = Counter(j for token in tokens for j in index[token])
        for j, count in overlaps.items():
            score = count / min(len(tokens), len(sets[j]))
            if score >= threshold:
                parent[find(i)] = find(j)
                edges.append({"a": examples[j].example_id, "b": examples[i].example_id,
                              "shared_shingles": count, "containment": score})
        for token in tokens:
            index[token].append(i)
    groups = defaultdict(list)
    for i, example in enumerate(examples):
        groups[find(i)].append(example.example_id)
    mapping, families = {}, {}
    for members in groups.values():
        members.sort()
        key = "inferred_sha256_" + hashlib.sha256("\n".join(members).encode()).hexdigest()
        families[key] = members
        mapping.update({member: key for member in members})
    return mapping, families, edges


def audit(settings):
    dataset = load_all_splits()
    examples = [e for rows in dataset.values() for e in rows]
    by_id = {e.example_id: e for e in examples}
    family, groups, edges = build_families(examples, settings['family_word_shingle_size'],
                                          settings['family_min_containment'])
    sources, reasons = {}, defaultdict(list)

    def mark(ids, category, source):
        for key in sorted(set(ids) & by_id.keys()):
            row = {"category": category, "source": source}
            if row not in reasons[key]:
                reasons[key].append(row)

    # Inference alone is not proof of human inspection. Exclude it conservatively.
    for p in sorted(Path('experiments').rglob('predictions.jsonl')):
        if Path(settings['root']) in p.parents:
            continue
        rows = [json.loads(line) for line in p.read_text(encoding='utf8').splitlines() if line]
        mark([r.get('example_id') for r in rows], 'prior_inference_inspection_unknown', p.as_posix())
        sources[p.as_posix()] = stable_file_hash(p)
    for p in sorted(Path('data/pilot').glob('*.json')):
        mark(ID_PATTERN.findall(p.read_text(encoding='utf8')), 'documented_development_pilot', p.as_posix())
        sources[p.as_posix()] = stable_file_hash(p)

    main = next(rows for split, rows in dataset.items() if split.value == 'flawed_fictions')
    mark([e.example_id for e in main[:settings['documented_missing_run_prefix_count']]],
         'uncertain_missing_85_record_run_conservative_prefix', 'docs/decision_log.md')

    # Story passage exposure is distinct from mere all-dataset ID inventories.
    passage_index = defaultdict(set)
    for e in examples:
        for token in shingles(e.story, settings['exposure_word_shingle_size']):
            passage_index[token].add(e.example_id)

    def scan(text, source, category, inspect_ids=True):
        ids = set(ID_PATTERN.findall(text))
        if inspect_ids and len(ids) <= 30:
            mark(ids, category + '_id_reference_conservative', source)
        hits = set()
        for token in shingles(text, settings['exposure_word_shingle_size']):
            hits.update(passage_index.get(token, ()))
        mark(hits, category + '_story_passage', source)

    for directory in ('docs', 'reports', 'scripts', 'tests', 'prompts', '.tmp'):
        for p in sorted(Path(directory).rglob('*')):
            if not p.is_file() or p.suffix not in {'.md', '.py', '.txt', '.json'} or 'cache' in p.parts:
                continue
            if 'main100' in p.as_posix():
                continue
            text = p.read_text(encoding='utf8', errors='replace')
            sources[p.as_posix()] = stable_file_hash(p)
            scan(text, p.as_posix(), 'development_artifact')

    sessions = []
    for p in sorted(Path(settings['history_root']).rglob('*.jsonl')):
        if settings['excluded_current_session'] in p.name:
            continue
        text = p.read_text(encoding='utf8')
        if 'flawedfictions_fyp' not in text:
            continue
        sources[p.as_posix()] = stable_file_hash(p)
        sessions.append(p.as_posix())
        for number, line in enumerate(text.splitlines(), 1):
            row = json.loads(line)
            if row.get('type') != 'response_item':
                continue
            payload = row.get('payload', {})
            if payload.get('role') in {'developer', 'system'}:
                continue
            for field in text_fields(payload):
                scan(field, f'{p.as_posix()}:{number}', 'prior_chat_visible',
                     inspect_ids=payload.get('type') not in {'function_call', 'custom_tool_call'})

    excluded_families = {family[key] for key in reasons}
    eligible = [e for e in main if family[e.example_id] not in excluded_families]
    excluded = [e for e in main if family[e.example_id] in excluded_families]
    counts = lambda rows: {"positive": sum(e.cont_error == 1 for e in rows),
                           "negative": sum(e.cont_error == 0 for e in rows), "total": len(rows)}
    return {
        'method': 'Connected components of 5-word shingle containment >= 0.20 across all four released subsets. No official source-story metadata exists. Twelve-word passage matches detect prior text exposure; small ID references conservatively excluded. Selection does not use predictions or scores.',
        'uncertainty': 'Local logs cannot prove absence of human/offline inspection. Prior inference is separately recorded and conservatively excluded. Missing historical 85-record run excluded as original-order prefix plus any recovered history references. Family IDs are inferred, not official; paraphrased/unrecorded relationships may remain undetectable.',
        'sessions_scanned': sessions, 'source_hashes': sources,
        'counts': {'main': counts(main), 'eligible': counts(eligible), 'excluded': counts(excluded)},
        'families': groups, 'family_by_id': family, 'family_edges': edges,
        'direct_exposure_evidence': dict(reasons),
        'excluded_main_ids': [e.example_id for e in excluded],
        'eligible_main_ids': [e.example_id for e in eligible],
        'exclusion_family_evidence': {e.example_id: [key for key in groups[family[e.example_id]] if key in reasons]
                                     for e in excluded},
    }


if __name__ == '__main__':
    from fyp.experiments.prompting_outputs import _write_json
    config = json.loads(Path('configs/experiments/main100_comparison_v1.json').read_text())
    result = audit(config)
    _write_json(Path(config['root']) / 'evaluator_only/exposure_audit.json', result)
    print(json.dumps({'counts': result['counts'], 'sessions': len(result['sessions_scanned'])}, indent=2))
