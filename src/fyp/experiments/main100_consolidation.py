"""Validate and tabulate saved corrected scores and current review; no rescoring."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from fyp.experiments.main100_review_analysis import validate, counts


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |"] +
                      ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def run(root):
    analysis = root / "analysis"
    review = analysis / "human_review_v1"
    correction = analysis / "scoring_correction_v1"
    csv_path = review / "manual_review_simple.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        annotations = list(csv.DictReader(f))
    selected = read(review / "review_rows.json")
    errors, missing = validate(annotations, selected)
    assert len(selected) == len(annotations) == 30 and not errors and not missing
    snapshot = read(review / ("human_review_analysis_" + sha(csv_path)[:12] + ".json"))
    assert annotations == snapshot["human_annotations_verbatim"]
    summary_path = review / "human_review_summary_93a0517b8da4.md"
    summary = summary_path.read_text(encoding="utf8")
    assert sha(csv_path) in summary
    current_counts = counts(annotations, snapshot["explanation_presence_status"])
    assert current_counts == snapshot["counts"]
    doc = (review / "manual_review_simple.md").read_text(encoding="utf8")
    assert all(r["review_id"] in doc for r in selected)
    manifest = read(review / "selection_manifest.json")
    assert {r["review_id"] for r in selected} == {
        model + ":" + sid for model, ids in manifest["selected_ids"].items() for sid in ids}
    audit = read(correction / "audit.json")
    assert audit["version"] == "scoring_correction_v1" and audit["status"] == "PASS"
    with (correction / "comparison.csv").open(encoding="utf8", newline="") as f:
        saved = {r["condition"]: r for r in csv.DictReader(f) if r["evaluator"] == "corrected"}
    conditions = audit["conditions"]
    assert len(conditions) == 6
    traces = [json.loads(line) for line in (correction / "record_traces.jsonl").read_text(encoding="utf8").splitlines() if line]
    assert len(traces) == 600
    raw = {}
    for t in traces:
        path = t["source_path"]
        if path not in raw:
            raw[path] = [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line]
        record = raw[path][t["source_line"] - 1]
        assert record["example_id"] == t["example_id"]
        assert hashlib.sha256(record["raw_response"].encode()).hexdigest() == t["raw_response_sha256"]
    ids = None
    gold_labels = None
    localisation = []
    for c in conditions:
        name = c["condition"]; m = c["metrics"]["corrected"]
        assert all(abs(float(value) - m[key]) < 1e-12 for key, value in saved[name].items() if key not in ("condition", "evaluator"))
        group = [t for t in traces if t["condition"] == name]
        these = [t["example_id"] for t in group]
        assert len(set(these)) == len(these) == 100
        assert ids is None or ids == these
        ids = these
        gold = [t["gold"] for t in group]
        assert gold_labels is None or gold_labels == gold
        gold_labels = gold
        assert sum(t["gold"] for t in group) == 50
        observed = Counter("invalid_gold_positive" if t["gold"] else "invalid_gold_negative"
                           for t in group if t["corrected_prediction"]["label"] is None)
        for t in group:
            label = t["corrected_prediction"]["label"]
            if label is not None:
                observed[{(1, 1): "tp", (1, 0): "fn", (0, 1): "fp", (0, 0): "tn"}[(t["gold"], label)]] += 1
        assert all(observed[k] == m[k] for k in ("tp", "fp", "tn", "fn", "invalid_gold_positive", "invalid_gold_negative"))
        assert abs(sum(t["corrected_localization"]["full_correct"] for t in group) / 100 - m["ceeval_full"]) < 1e-12
        positive = [t for t in group if t["gold"] == 1 and t["corrected_prediction"]["label"] == 1]
        sides = Counter((int(t["corrected_localization"]["cont_error_lines_correct"]),
                         int(t["corrected_localization"]["contradicted_lines_correct"])) for t in positive)
        both = sides[(1, 1)]
        assert abs(both / 50 - m["ceeval_pos"]) < 1e-12
        localisation.append([name, m["tp"], both, sides[(1, 0)], sides[(0, 1)], sides[(0, 0)]])
    blocks = ["# Saved corrected benchmark tables", "All rate columns are percentages; deltas are percentage points. Evaluator: scoring_correction_v1, corrected variant only."]
    for keys in [
        ["total", "valid", "invalid", "tp", "fp", "tn", "fn", "invalid_gold_positive", "invalid_gold_negative", "predicted_positive_rate"],
        ["accuracy", "precision", "recall", "f1", "macro_f1", "specificity", "false_positive_rate", "balanced_accuracy", "ceeval_full", "ceeval_pos"]]:
        blocks.append(table(["Condition"] + keys, [[c["condition"]] + [
            f"{100*c['metrics']['corrected'][k]:.2f}" if isinstance(c['metrics']['corrected'][k], float)
            else c['metrics']['corrected'][k] for k in keys] for c in conditions]))
    delta_keys = ["accuracy", "precision", "recall", "f1", "macro_f1", "specificity", "false_positive_rate", "predicted_positive_rate", "ceeval_full", "ceeval_pos"]
    deltas = []
    for b in conditions:
        if b["strategy"] != "baseline": continue
        v = next(c for c in conditions if c["model"] == b["model"] and c["strategy"] == "verify_revise")
        deltas.append([b["model"]] + [f"{100*(v['metrics']['corrected'][k]-b['metrics']['corrected'][k]):+.2f}" for k in delta_keys])
    blocks += ["## verify_revise minus baseline", table(["Model"] + delta_keys, deltas),
               "## Localisation within valid true positives (counts, diagnostic only)",
               table(["Condition", "TP", "Both sides", "Error only", "Contradicted only", "Neither"], localisation)]
    inputs = [csv_path, summary_path, review / "manual_review_simple.md", review / "review_rows.json",
              correction / "audit.json", correction / "comparison.csv", correction / "record_traces.jsonl"] + [Path(p) for p in raw]
    output = analysis / "benchmark_consolidated_v1"
    output.mkdir(exist_ok=True)
    result = {"status": "PASS", "evaluator": audit["version"], "records": 600,
              "review_complete": True, "review_counts": current_counts,
              "inputs_sha256": {str(p): sha(p) for p in inputs},
              "inference_calls": 0, "rescoring": False,
              "checks": "CSV/snapshot/manifest/summary hash; audit/CSV metrics; 600 trace/raw hashes; aligned IDs; confusion and localisation counts"}
    for path, data in [(output / "tables.md", "\n\n".join(blocks) + "\n"),
                       (output / "validation.json", json.dumps(result, indent=2) + "\n")]:
        if path.exists() and path.read_text(encoding="utf8") != data:
            history = output / "history"; history.mkdir(exist_ok=True)
            (history / (path.stem + "_" + sha(path)[:12] + path.suffix)).write_bytes(path.read_bytes())
        path.write_text(data, encoding="utf8")
    print(json.dumps({"status": "PASS", "output": str(output), "review_sha256": sha(csv_path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    run(parser.parse_args().experiment_root)
