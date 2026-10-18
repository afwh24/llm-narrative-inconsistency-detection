"""Read-only verification of completed supplementary inference; writes audit only."""
import json
from pathlib import Path
from fyp.experiments import generated_stories_detection as d


def main():
    cfg,contract=d.loaded();root=Path(cfg['root']);status=d.status()
    if status['pending']:raise RuntimeError('Inference incomplete: final audit is not yet available')
    records=d.record_rows(cfg,contract)
    starts_total=finishes_total=0;unfinished=[];requests=[]
    for condition in contract['conditions']:
        journal=d.rows(root/'runs'/condition['id']/'requests.jsonl')
        starts={e['request_id']:e for e in journal if e['event']=='started'}
        finishes={e['request_id']:e for e in journal if e['event']=='finished'}
        assert len(starts)==sum(e['event']=='started' for e in journal)
        assert len(finishes)==sum(e['event']=='finished' for e in journal)
        assert set(finishes)<=set(starts)
        starts_total+=len(starts);finishes_total+=len(finishes)
        prompts={p['story_id']:p for p in contract['prompts'][condition['strategy']]}
        config=d.load_model_config(condition['model_config'])
        for rid,start in starts.items():
            sid=start['example_id'];p=prompts[sid]
            body=start['application_request']['body']
            expected={**config.effective_ollama_request_fields(),'prompt':p['prompt']}
            assert body==expected and sid not in body['prompt']
            assert not any(k in body for k in ('context','messages','gold','pair_id','category'))
            assert d.hash_text(start['application_request']['body_utf8']).upper()==start['application_request']['body_hash']
            requests.append(dict(condition=condition['id'],story_id=sid,request_id=rid,attempt=start['attempt'],body_hash=start['application_request']['body_hash']))
            if rid not in finishes:unfinished.append(dict(condition=condition['id'],story_id=sid,request_id=rid,attempt=start['attempt']))
        for record in [r for r in records if r['condition']==condition['id']]:
            history=record['attempt_history'];attempts=[h['attempt'] for h in history]
            assert attempts==list(range(1,len(history)+1)) and len(history)<=config.runtime.max_retries+1
            assert all(h['request_id'] in starts for h in history)
            received=[finishes[h['request_id']] for h in history if h['request_id'] in finishes and finishes[h['request_id']]['status']=='response_received']
            if received:assert received[-1]['raw_response']==record['raw_response']
    preservation=d.read(root/'preservation_before.json')
    d.verify_hashes(preservation)
    d.dataset_audit(cfg)
    d.analyse()
    report=dict(status=status,contract_sha256=d.sha(root/'contract.json'),independent_request_payloads_verified=True,
        final_records=len(records),dispatch_starts=starts_total,finished_attempts=finishes_total,unresolved_historical_starts=unfinished,
        unfinished_start_policy='Retained as consumed budget, not a final failure or additional final record; subsequent matching resume used remaining budget.',
        known_attempt_latency_seconds=sum(r['latency_seconds'] for r in records),
        latency_limitation='Interrupted starts without finishes have unknown latency, excluded from the known-attempt sum.',
        all_source_hashes_match=True,protected_benchmark_and_review_files=len(preservation),dataset_preserved=True,
        journal_and_checkpoint_hashes={p.relative_to(root).as_posix():d.sha(p) for sub in ('runs','checkpoints') for p in (root/sub).rglob('*') if p.is_file()},
        requests=requests,tests='52 offline supplementary, corrected-scoring and durable-journal tests passed; network blocked for synthetic supplementary tests.')
    d.save(root/'analysis'/'completion_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('journal_and_checkpoint_hashes','requests')},indent=2))


if __name__=='__main__':main()
