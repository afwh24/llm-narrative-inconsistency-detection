"""Evaluate saved FLAWEDFICTIONS inference records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from fyp.data.loader import load_all_splits
from fyp.data.schema import FlawedFictionsExample, FlawedFictionsSplit
from fyp.evaluation.localization import LocalizationScore, score_localization
from fyp.evaluation.metrics import compute_classification_metrics, mean_or_zero
from fyp.prompting.parser import ParsedPrediction, parse_official_response


PREDICTION_FILENAMES = ("predictions.jsonl", "inference_records.jsonl", "results.jsonl")


class EvaluationError(RuntimeError):
    """Raised when evaluation inputs are missing or malformed."""


@dataclass(frozen=True)
class ExampleEvaluation:
    example_id: str
    subset: str
    ground_truth_cont_error: float
    predicted_cont_error: float | None
    classification_correct: bool
    ceeval_full: float
    cont_error_lines_correct: float
    contradicted_lines_correct: float
    target_inference_failure: bool
    target_parsing_failure: bool
    missing_prediction: bool
    malformed_prediction: bool
    evaluator_inference_failure: bool
    evaluator_parsing_failure: bool
    parsing_failures: tuple[str, ...]
    localization: LocalizationScore

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["localization"] = asdict(self.localization)
        return data


@dataclass(frozen=True)
class EvaluationReport:
    metrics: Mapping[str, Any]
    examples: tuple[ExampleEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "examples": [example.to_dict() for example in self.examples],
        }


def evaluate_predictions(
    prediction_records: Sequence[Mapping[str, Any]],
    dataset: Mapping[FlawedFictionsSplit, Sequence[FlawedFictionsExample]] | None = None,
) -> EvaluationReport:
    """Evaluate prediction records against the validated official dataset."""

    dataset = dataset or load_all_splits()
    examples_by_key = _index_examples(dataset)
    evaluated = []
    for record in prediction_records:
        subset = _require_string(record, "subset")
        example_id = _require_string(record, "example_id")
        example = examples_by_key.get((subset, example_id))
        if example is None:
            raise EvaluationError(f"No dataset example found for subset={subset!r}, example_id={example_id!r}")
        prediction, flags = _extract_prediction(record)
        predicted_mapping = prediction.to_official_result() if prediction is not None else None
        loc = score_localization(example, predicted_mapping)
        predicted_label = predicted_mapping["cont_error"] if predicted_mapping is not None else None
        classification_correct = predicted_label is not None and float(predicted_label) == example.cont_error
        evaluated.append(
            ExampleEvaluation(
                example_id=example_id,
                subset=subset,
                ground_truth_cont_error=example.cont_error,
                predicted_cont_error=None if predicted_label is None else float(predicted_label),
                classification_correct=classification_correct,
                ceeval_full=loc.full_correct,
                cont_error_lines_correct=loc.cont_error_lines_correct,
                contradicted_lines_correct=loc.contradicted_lines_correct,
                target_inference_failure=flags["target_inference_failure"],
                target_parsing_failure=flags["target_parsing_failure"],
                missing_prediction=flags["missing_prediction"],
                malformed_prediction=flags["malformed_prediction"],
                evaluator_inference_failure=False,
                evaluator_parsing_failure=False,
                parsing_failures=prediction.parsing_failures if prediction is not None else (),
                localization=loc,
            )
        )
    return EvaluationReport(metrics=_aggregate_metrics(evaluated), examples=tuple(evaluated))


def evaluate_run_directory(
    run_dir: Path | str,
    dataset: Mapping[FlawedFictionsSplit, Sequence[FlawedFictionsExample]] | None = None,
) -> EvaluationReport:
    """Evaluate a run directory and write metrics/summary outputs inside it."""

    run_dir = Path(run_dir)
    prediction_path = _find_prediction_file(run_dir)
    records = _load_prediction_records(prediction_path)
    report = evaluate_predictions(records, dataset=dataset)
    _write_report(run_dir, report)
    return report


def resolve_run_directory(run_id: str) -> Path:
    """Resolve `--run` as a direct path or common local run-id location."""

    direct = Path(run_id)
    if direct.exists():
        return direct
    for base in (Path("runs"), Path("experiments"), Path("data/outputs")):
        candidate = base / run_id
        if candidate.exists():
            return candidate
    return direct


def _aggregate_metrics(examples: Sequence[ExampleEvaluation]) -> dict[str, Any]:
    y_true = [example.ground_truth_cont_error for example in examples]
    y_pred = [example.predicted_cont_error for example in examples]
    classification = compute_classification_metrics(y_true, y_pred)
    metrics = {
        "num_examples": len(examples),
        "classification_metrics": classification.to_dict(),
        "localization_metrics": {
            "ceeval_full": mean_or_zero(example.ceeval_full for example in examples),
            "ceeval_pos": mean_or_zero(
                example.ceeval_full for example in examples if example.ground_truth_cont_error == 1.0
            ),
            "cont_error_lines": mean_or_zero(example.cont_error_lines_correct for example in examples),
            "contradicted_lines": mean_or_zero(example.contradicted_lines_correct for example in examples),
        },
        "failure_counts": {
            "target_inference_failure": sum(example.target_inference_failure for example in examples),
            "target_parsing_failure": sum(example.target_parsing_failure for example in examples),
            "missing_prediction": sum(example.missing_prediction for example in examples),
            "malformed_prediction": sum(example.malformed_prediction for example in examples),
            "evaluator_inference_failure": 0,
            "evaluator_parsing_failure": 0,
        },
        "subsets": {},
    }
    subsets = sorted({example.subset for example in examples})
    for subset in subsets:
        subset_examples = [example for example in examples if example.subset == subset]
        subset_classification = compute_classification_metrics(
            [example.ground_truth_cont_error for example in subset_examples],
            [example.predicted_cont_error for example in subset_examples],
        )
        metrics["subsets"][subset] = {
            "num_examples": len(subset_examples),
            "classification_metrics": subset_classification.to_dict(),
            "localization_metrics": {
                "ceeval_full": mean_or_zero(example.ceeval_full for example in subset_examples),
                "ceeval_pos": mean_or_zero(
                    example.ceeval_full
                    for example in subset_examples
                    if example.ground_truth_cont_error == 1.0
                ),
            },
        }
    return metrics


def _extract_prediction(record: Mapping[str, Any]) -> tuple[ParsedPrediction | None, dict[str, bool]]:
    flags = {
        "target_inference_failure": bool(record.get("target_inference_failure") or record.get("inference_error")),
        "target_parsing_failure": False,
        "missing_prediction": False,
        "malformed_prediction": False,
    }
    if flags["target_inference_failure"]:
        return None, flags

    parsed_raw = record.get("parsed_prediction")
    if isinstance(parsed_raw, Mapping):
        try:
            parsed = ParsedPrediction(
                cont_error=float(parsed_raw["cont_error"]),
                cont_error_expl_detailed=str(parsed_raw.get("cont_error_expl_detailed", "")),
                cont_error_expl=str(parsed_raw.get("cont_error_expl", "")),
                cont_error_expl_short=str(parsed_raw.get("cont_error_expl_short", "")),
                cont_error_lines=tuple(parsed_raw.get("cont_error_lines", [])),
                contradicted_lines=tuple(parsed_raw.get("contradicted_lines", [])),
                parsing_failures=tuple(parsed_raw.get("parsing_failures", ())),
            )
            if parsed.cont_error not in (0.0, 1.0):
                flags["malformed_prediction"] = True
                return None, flags
        except Exception:
            flags["malformed_prediction"] = True
            return None, flags
    elif "raw_response" in record and isinstance(record["raw_response"], str):
        parsed = parse_official_response(record["raw_response"])
    else:
        flags["missing_prediction"] = True
        return None, flags

    flags["target_parsing_failure"] = bool(parsed.parsing_failures)
    return parsed, flags


def _index_examples(
    dataset: Mapping[FlawedFictionsSplit, Sequence[FlawedFictionsExample]],
) -> dict[tuple[str, str], FlawedFictionsExample]:
    indexed = {}
    for split, examples in dataset.items():
        for example in examples:
            indexed[(split.value, example.example_id)] = example
    return indexed


def _find_prediction_file(run_dir: Path) -> Path:
    if not run_dir.exists():
        raise EvaluationError(f"Run directory not found: {run_dir}")
    for filename in PREDICTION_FILENAMES:
        path = run_dir / filename
        if path.exists():
            return path
    raise EvaluationError(
        f"No prediction file found in {run_dir}; expected one of: {', '.join(PREDICTION_FILENAMES)}"
    )


def _load_prediction_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(parsed, dict):
                raise EvaluationError(f"{path}:{line_no}: record must be a JSON object")
            records.append(parsed)
        return records
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(parsed, list):
        return [record for record in parsed if isinstance(record, dict)]
    if isinstance(parsed, dict) and isinstance(parsed.get("records"), list):
        return [record for record in parsed["records"] if isinstance(record, dict)]
    raise EvaluationError(f"Unsupported prediction file shape: {path}")


def _write_report(run_dir: Path, report: EvaluationReport) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(report.metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (run_dir / "evaluation_examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in report.examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    (run_dir / "summary.md").write_text(_render_summary(report, run_dir=run_dir), encoding="utf-8")


def _render_summary(report: EvaluationReport, *, run_dir: Path | None = None) -> str:
    classification = report.metrics["classification_metrics"]
    localization = report.metrics["localization_metrics"]
    failures = report.metrics["failure_counts"]
    heading = ["# Evaluation Summary", ""]
    pilot_labels = _is_pilot_run(run_dir) if run_dir is not None else False
    expected_examples = _expected_example_count(run_dir) if run_dir is not None else None
    if expected_examples is not None and report.metrics["num_examples"] < expected_examples:
        heading.extend(
            [
                "PRE-FLIGHT PARTIAL RUN",
                "",
                "NOT FINAL RESULTS",
                "",
                f"Expected final examples: {expected_examples}",
                "",
            ]
        )
    if pilot_labels:
        heading.extend(
            [
                f"PILOT RESULTS \u2014 N={expected_examples or report.metrics['num_examples']}",
                "",
                "EXPLORATORY ONLY",
                "",
                "NOT FULL FLAWEDFICTIONS BENCHMARK RESULTS",
                "",
            ]
        )
    return "\n".join(
        heading
        + [
            f"Examples: {report.metrics['num_examples']}",
            "",
            "## Classification",
            "",
            f"- Accuracy: {classification['accuracy']:.6f}",
            f"- Precision: {classification['precision']:.6f}",
            f"- Recall: {classification['recall']:.6f}",
            f"- F1: {classification['f1']:.6f}",
            "",
            "## Localization",
            "",
            f"- CEEval-Full: {localization['ceeval_full']:.6f}",
            f"- CEEval-Pos: {localization['ceeval_pos']:.6f}",
            "",
            "## Failures",
            "",
            f"- Target inference failures: {failures['target_inference_failure']}",
            f"- Target parsing failures: {failures['target_parsing_failure']}",
            f"- Missing predictions: {failures['missing_prediction']}",
            f"- Malformed predictions: {failures['malformed_prediction']}",
            f"- Evaluator inference failures: {failures['evaluator_inference_failure']}",
            f"- Evaluator parsing failures: {failures['evaluator_parsing_failure']}",
            "",
        ]
    )


def _is_pilot_run(run_dir: Path) -> bool:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    dataset = metadata.get("dataset")
    return isinstance(dataset, Mapping) and (
        dataset.get("fixed_manifest_role") == "pilot" or bool(dataset.get("pilot_name"))
    )


def _expected_example_count(run_dir: Path | None) -> int | None:
    if run_dir is None:
        return None
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    expected = metadata.get("expected_example_count")
    if isinstance(expected, int) and expected > 0:
        return expected
    dataset = metadata.get("dataset", {})
    if isinstance(dataset, Mapping):
        expected = dataset.get("fixed_manifest_example_count", dataset.get("pilot_example_count"))
        if isinstance(expected, int) and expected > 0:
            return expected
        if dataset.get("pilot_name") == "pilot_20":
            return 20
    return None


def _require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"Prediction record missing string field {field!r}")
    return value
