from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fyp.data.schema import FlawedFictionsExample, FlawedFictionsSplit
from fyp.evaluation.evaluator import evaluate_predictions, evaluate_run_directory
from fyp.evaluation.localization import score_localization


def _example(
    example_id: str,
    split: FlawedFictionsSplit,
    story: str,
    label: float,
    error_line: str = "NA",
    contradicted_line: str = "NA",
) -> FlawedFictionsExample:
    has_error = label == 1.0
    return FlawedFictionsExample(
        split=split,
        example_id=example_id,
        story=story,
        has_continuity_error=has_error,
        cont_error_expl="Error" if has_error else "No error",
        cont_error_lines_raw=error_line,
        contradicted_lines_raw=contradicted_line,
        cont_error_lines=(error_line,) if has_error else (),
        contradicted_lines=(contradicted_line,) if has_error else (),
        inconsistency_type="continuity_error" if has_error else "none",
        raw_label=label,
        raw_record={
            "story": story,
            "cont_error": label,
            "cont_error_expl": "Error" if has_error else "No error",
            "cont_error_lines": error_line,
            "contradicted_lines": contradicted_line,
            "example_id": example_id,
        },
    )


def _dataset():
    split = FlawedFictionsSplit.FLAWED_FICTIONS
    story_pos = "Mara painted the door blue. Later, Mara painted the same door red."
    story_neg = "Ivo packed a map. Ivo followed the map home."
    examples = [
        _example(
            "pos-1",
            split,
            story_pos,
            1.0,
            error_line="Later, Mara painted the same door red.",
            contradicted_line="Mara painted the door blue.",
        ),
        _example("neg-1", split, story_neg, 0.0),
    ]
    return {split: examples}


def _prediction(example_id: str, label: float, lines: bool = True):
    parsed = {
        "cont_error": label,
        "cont_error_expl_detailed": "",
        "cont_error_expl": "",
        "cont_error_expl_short": "",
        "cont_error_lines": ["Later, Mara painted the same door red."] if lines else [],
        "contradicted_lines": ["Mara painted the door blue."] if lines else [],
        "parsing_failures": [],
    }
    return {
        "subset": "flawed_fictions",
        "example_id": example_id,
        "parsed_prediction": parsed,
    }


def test_all_correct_metrics_and_localization() -> None:
    records = [_prediction("pos-1", 1.0), _prediction("neg-1", 0.0)]

    report = evaluate_predictions(records, dataset=_dataset())

    assert report.metrics["classification_metrics"] == {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert report.metrics["localization_metrics"]["ceeval_full"] == 1.0
    assert report.metrics["localization_metrics"]["ceeval_pos"] == 1.0


def test_all_incorrect_metrics() -> None:
    records = [_prediction("pos-1", 0.0), _prediction("neg-1", 1.0)]

    report = evaluate_predictions(records, dataset=_dataset())

    assert report.metrics["classification_metrics"]["accuracy"] == 0.0
    assert report.metrics["classification_metrics"]["precision"] == 0.0
    assert report.metrics["classification_metrics"]["recall"] == 0.0
    assert report.metrics["classification_metrics"]["f1"] == 0.0
    assert report.metrics["localization_metrics"]["ceeval_full"] == 0.0


def test_mixed_predictions_keep_failures_in_denominator() -> None:
    records = [_prediction("pos-1", 1.0), {"subset": "flawed_fictions", "example_id": "neg-1"}]

    report = evaluate_predictions(records, dataset=_dataset())

    assert report.metrics["classification_metrics"]["accuracy"] == 0.5
    assert report.metrics["failure_counts"]["missing_prediction"] == 1
    assert report.examples[1].missing_prediction is True
    assert report.examples[1].classification_correct is False


def test_malformed_prediction_and_unexpected_label_are_failures() -> None:
    records = [_prediction("pos-1", 2.0), _prediction("neg-1", 0.0)]

    report = evaluate_predictions(records, dataset=_dataset())

    assert report.metrics["failure_counts"]["malformed_prediction"] == 1
    assert report.examples[0].predicted_cont_error is None
    assert report.examples[0].classification_correct is False


def test_raw_response_parsing_failure_is_recorded() -> None:
    records = [
        {
            "subset": "flawed_fictions",
            "example_id": "pos-1",
            "raw_response": "",
        },
        _prediction("neg-1", 0.0),
    ]

    report = evaluate_predictions(records, dataset=_dataset())

    assert report.metrics["failure_counts"]["target_parsing_failure"] == 1
    assert report.examples[0].parsing_failures == ("empty_response", "missing_response_tag")


def test_target_inference_failure_is_distinct_from_parsing_failure() -> None:
    records = [
        {
            "subset": "flawed_fictions",
            "example_id": "pos-1",
            "target_inference_failure": True,
            "inference_error": "timeout",
        },
        _prediction("neg-1", 0.0),
    ]

    report = evaluate_predictions(records, dataset=_dataset())

    assert report.metrics["failure_counts"]["target_inference_failure"] == 1
    assert report.metrics["failure_counts"]["target_parsing_failure"] == 0
    assert report.examples[0].predicted_cont_error is None


def test_subset_level_aggregation() -> None:
    records = [_prediction("pos-1", 1.0), _prediction("neg-1", 0.0)]

    report = evaluate_predictions(records, dataset=_dataset())

    subset = report.metrics["subsets"]["flawed_fictions"]
    assert subset["num_examples"] == 2
    assert subset["classification_metrics"]["accuracy"] == 1.0
    assert subset["localization_metrics"]["ceeval_full"] == 1.0


def test_positive_classification_correct_but_bad_lines_gets_zero_ceeval_pos() -> None:
    records = [_prediction("pos-1", 1.0, lines=False), _prediction("neg-1", 0.0)]

    report = evaluate_predictions(records, dataset=_dataset())

    assert report.metrics["classification_metrics"]["accuracy"] == 1.0
    assert report.metrics["localization_metrics"]["ceeval_full"] == 0.5
    assert report.metrics["localization_metrics"]["ceeval_pos"] == 0.0


def test_cli_run_directory_outputs_metrics_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [_prediction("pos-1", 1.0), _prediction("neg-1", 0.0)]
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    report = evaluate_run_directory(run_dir, dataset=_dataset())

    assert report.metrics["num_examples"] == 2
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "evaluation_examples.jsonl").exists()


def test_pilot_run_summary_is_clearly_labeled(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"dataset": {"pilot_name": "pilot_20"}}),
        encoding="utf-8",
    )
    records = [_prediction("pos-1", 1.0), _prediction("neg-1", 0.0)]
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    evaluate_run_directory(run_dir, dataset=_dataset())

    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "PILOT RESULTS \u2014 N=20" in summary
    assert "EXPLORATORY ONLY" in summary
    assert "NOT FULL FLAWEDFICTIONS BENCHMARK RESULTS" in summary


def test_official_row_level_localization_parity() -> None:
    official_func = _load_official_row_eval_function()
    example = _dataset()[FlawedFictionsSplit.FLAWED_FICTIONS][0]
    predicted = {
        "cont_error": 1.0,
        "cont_error_lines": ["Later, Mara painted the same door red."],
        "contradicted_lines": ["Mara painted the door blue."],
    }
    ours = score_localization(example, predicted)
    row = {
        "predicted_cont_error": 1.0,
        "ground_truth_cont_error": 1.0,
        "predicted_cont_error_lines_full": ["Later, Mara painted the same door red."],
        "ground_truth_cont_error_lines_full": ["Later, Mara painted the same door red."],
        "predicted_contradicted_lines_full": ["Mara painted the door blue."],
        "ground_truth_contradicted_lines_full": ["Mara painted the door blue."],
    }

    official = official_func(row)

    assert ours.full_correct == official["full_correct"]
    assert ours.cont_error_lines_correct == official["cont_error_lines_correct"]
    assert ours.contradicted_lines_correct == official["contradicted_lines_correct"]


def _load_official_row_eval_function():
    source = Path("external/flawedfictions/src/localization_metrics.py").read_text(encoding="utf-8")
    match = re.search(
        r"def eval_cont_error_localization_on_example\(row\):\n(?P<body>.*?)(?=\n\ndef eval_cont_error_localization_strict_on_example)",
        source,
        flags=re.S,
    )
    if match is None:
        pytest.fail("could not locate official row-level localization function")
    namespace = {}
    exec(match.group(0), namespace)
    return namespace["eval_cont_error_localization_on_example"]
