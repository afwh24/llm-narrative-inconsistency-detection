"""Command line interface for local FYP utilities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fyp.data.loader import DEFAULT_RAW_DATA_DIR, load_all_splits
from fyp.data.pilot import PilotManifestError, load_pilot_manifest, resolve_pilot_manifest
from fyp.data.schema import SPLIT_SPECS, FlawedFictionsSplit
from fyp.data.validation import ValidationError, validate_dataset
from fyp.doctor import (
    collect_doctor_results,
    doctor_has_critical_failure,
    print_doctor_report,
)
from fyp.evaluation.evaluator import (
    EvaluationError,
    evaluate_run_directory,
    resolve_run_directory,
)
from fyp.experiments.runner import DEFAULT_RUN_ROOT, ExperimentRunError, RunConfig, run_experiment
from fyp.models.ollama_client import OllamaError
from fyp.models.registry import TARGET_MODELS
from fyp.prompting.strategies import STRATEGIES


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(prog="python -m fyp.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Load and validate local official FLAWEDFICTIONS raw data.",
    )
    prepare.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing official raw FLAWEDFICTIONS Parquet files.",
    )
    prepare.add_argument(
        "--split",
        action="append",
        choices=[split.value for split in FlawedFictionsSplit],
        help="Official split to validate. May be supplied multiple times.",
    )
    subparsers.add_parser(
        "doctor",
        help="Check Python environment, dataset, configs, and local Ollama status.",
    )
    baseline_pilot = subparsers.add_parser(
        "baseline-pilot",
        help="Prepare or run the separate previously inspected three-model main10 pilot.",
    )
    baseline_pilot.add_argument("action", choices=("prepare", "run", "export"))
    baseline_pilot.add_argument("--model", choices=("qwen3:8b", "llama3.1:8b", "mistral:7b"))
    prompt_pilot = subparsers.add_parser(
        "prompt-pilot",
        help="Prepare or run the isolated three-model main10 prompting matrix.",
    )
    prompt_pilot.add_argument("action", choices=("prepare", "run"))
    prompt_pilot.add_argument("--model", choices=("qwen3:8b", "llama3.1:8b", "mistral:7b"))
    prompt_pilot.add_argument("--strategy", choices=("vanilla", "evidence_first", "structured_check", "verify_revise"))
    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate saved FLAWEDFICTIONS prediction records for one run.",
    )
    evaluate.add_argument(
        "--run",
        required=True,
        help="Run directory path or run id under runs/, experiments/, or data/outputs/.",
    )
    run = subparsers.add_parser(
        "run",
        help="Run or resume a scoped FLAWEDFICTIONS target-model inference job.",
    )
    run.add_argument("--model", required=True, choices=TARGET_MODELS)
    run.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    run.add_argument(
        "--subset",
        action="append",
        choices=[split.value for split in FlawedFictionsSplit],
        help="Official dataset subset to evaluate. May be supplied multiple times.",
    )
    run.add_argument(
        "--pilot",
        help="Named fixed pilot manifest, e.g. pilot_20.",
    )
    run.add_argument(
        "--manifest",
        type=Path,
        help="Explicit fixed pilot manifest path.",
    )
    run.add_argument("--limit", type=int, help="Optional prefix limit for smoke tests only.")
    run.add_argument(
        "--max-new-examples",
        type=int,
        help=(
            "Stop after this many newly attempted examples without changing the run "
            "configuration fingerprint. Use only for pre-flight/resume validation."
        ),
    )
    run.add_argument("--run-id", help="Explicit run id. Required when resuming.")
    run.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Directory that contains run directories.",
    )
    run.add_argument("--resume", action="store_true", help="Resume a compatible existing run.")
    run.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry previously failed records for this exact configuration.",
    )
    run.add_argument(
        "--think",
        choices=("true", "false"),
        help="Override Qwen3 native thinking mode for a controlled ablation.",
    )
    run.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing official raw FLAWEDFICTIONS Parquet files.",
    )

    args = parser.parse_args(argv)
    if args.command == "prepare":
        return prepare_command(args)
    if args.command == "doctor":
        return doctor_command()
    if args.command == "baseline-pilot":
        return baseline_pilot_command(args)
    if args.command == "prompt-pilot":
        return prompt_pilot_command(args)
    if args.command == "evaluate":
        return evaluate_command(args)
    if args.command == "run":
        return run_command(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def prepare_command(args: argparse.Namespace) -> int:
    selected_splits = (
        [FlawedFictionsSplit.parse(split) for split in args.split]
        if args.split
        else list(SPLIT_SPECS)
    )
    print(f"Raw data directory: {args.raw_data_dir}")
    print("Loading official FLAWEDFICTIONS splits:")
    for split in selected_splits:
        spec = SPLIT_SPECS[split]
        print(f"  - {split.value} ({spec.filename})")

    dataset = load_all_splits(selected_splits, raw_data_dir=args.raw_data_dir)
    try:
        report = validate_dataset(dataset)
    except ValidationError as exc:
        print("Validation failed:")
        print(exc)
        return 1

    print("\nValidation summary:")
    for summary in report.splits:
        print(
            f"  - {summary.split.value}: "
            f"{summary.examples} examples, "
            f"{summary.positive} continuity-error, "
            f"{summary.negative} no-error"
        )
        if summary.unexpected_fields:
            print(f"    Unexpected fields preserved in raw_record: {summary.unexpected_fields}")

    print(f"\nTotal examples validated: {report.total_examples}")
    print("Processed files created: none (raw data loaded and validated in memory only)")
    return 0


def doctor_command() -> int:
    results = collect_doctor_results()
    print_doctor_report(results)
    return 1 if doctor_has_critical_failure(results) else 0


def baseline_pilot_command(args: argparse.Namespace) -> int:
    from fyp.experiments.baseline_pilot import export_baseline, prepare_pilot, run_pilot

    if args.action != "prepare" and args.model is None:
        print("--model is required for run/export")
        return 2
    try:
        if args.action == "prepare":
            metadata = prepare_pilot()
            print(metadata["status"])
            print("Previously inspected exploratory pilot: 10 main stories, 5 per class")
            print("Model generations executed: 0")
            for name, inspection in metadata["provenance"]["models"].items():
                print(f"{name}: {'available' if inspection['available'] else 'missing'}")
            return 0 if metadata["status"] == "PREPARED_NO_INFERENCE" else 1
        if args.action == "export":
            print(f"Exported: {export_baseline(args.model)}")
            return 0
        result = run_pilot(args.model)
        print("EXPLORATORY PILOT -- PREVIOUSLY INSPECTED STORIES -- NOT FINAL RESULTS")
        print(f"Run directory: {result.run_dir}")
        print(f"Completed: {result.completed_examples}; failures: {result.failed_examples}")
        print(f"Skipped completed: {result.skipped_completed}; skipped failed: {result.skipped_failed}")
        return 0
    except (ExperimentRunError, OSError, ValueError, KeyError, OllamaError) as exc:
        print(f"Baseline pilot failed: {exc}")
        return 1


def prompt_pilot_command(args: argparse.Namespace) -> int:
    from fyp.experiments.prompting_pilot import prepare_prompting_pilot, run_prompting_pilot

    if args.action == "run" and (args.model is None or args.strategy is None):
        print("--model and --strategy are required for run")
        return 2
    try:
        if args.action == "prepare":
            metadata = prepare_prompting_pilot()
            print(metadata["status"])
            print("Same ten inspected main stories: five positive, five negative")
            print("Matrix: 120 records; reusable baseline: 30; new records planned: 90")
            print("Model generations executed by preparation: 0")
            for model in metadata["provenance"]["baseline_references"]:
                print(f"{model}: baseline REUSABLE_COMPLETE")
            return 0 if metadata["status"] == "PREPARED_NO_INFERENCE" else 1
        result = run_prompting_pilot(args.model, args.strategy)
        if isinstance(result, dict):
            print(f"Reusing baseline: {result['run_dir']}; model generations executed: 0")
        else:
            print(f"EXPLORATORY PILOT -- NOT FINAL RESULTS: {result.run_dir}")
            print(f"Completed: {result.completed_examples}; failures: {result.failed_examples}")
            print(f"Skipped completed: {result.skipped_completed}; skipped failed: {result.skipped_failed}")
        return 0
    except (ExperimentRunError, OSError, ValueError, KeyError, OllamaError) as exc:
        print(f"Prompting pilot failed: {exc}")
        return 1


def evaluate_command(args: argparse.Namespace) -> int:
    run_dir = resolve_run_directory(args.run)
    try:
        report = evaluate_run_directory(run_dir)
    except EvaluationError as exc:
        print("Evaluation failed:")
        print(exc)
        return 1

    metrics = report.metrics
    classification = metrics["classification_metrics"]
    localization = metrics["localization_metrics"]
    print(f"Run directory: {run_dir}")
    print(f"Examples evaluated: {metrics['num_examples']}")
    print(
        "Classification: "
        f"accuracy={classification['accuracy']:.6f}, "
        f"precision={classification['precision']:.6f}, "
        f"recall={classification['recall']:.6f}, "
        f"f1={classification['f1']:.6f}"
    )
    print(
        "Localization: "
        f"ceeval_full={localization['ceeval_full']:.6f}, "
        f"ceeval_pos={localization['ceeval_pos']:.6f}"
    )
    print(f"Saved: {run_dir / 'metrics.json'}")
    print(f"Saved: {run_dir / 'summary.md'}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    if args.limit is not None and args.limit <= 0:
        print("--limit must be a positive integer when supplied")
        return 2
    if args.max_new_examples is not None and args.max_new_examples <= 0:
        print("--max-new-examples must be a positive integer when supplied")
        return 2
    if args.resume and not args.run_id:
        print("--run-id is required with --resume")
        return 2
    if args.pilot and args.manifest:
        print("Use either --pilot or --manifest, not both")
        return 2
    manifest_path: Path | None = None
    manifest_subsets: tuple[FlawedFictionsSplit, ...] | None = None
    if args.pilot or args.manifest:
        if args.limit is not None:
            print("Do not combine --limit with a fixed pilot manifest")
            return 2
        try:
            manifest_path = resolve_pilot_manifest(args.manifest or args.pilot)
            manifest = load_pilot_manifest(manifest_path)
        except PilotManifestError as exc:
            print("Fixed manifest failed:")
            print(exc)
            return 1
        manifest_subsets = manifest.subsets
    if not args.subset and manifest_subsets is None:
        print("Provide at least one --subset, or use --pilot/--manifest")
        return 2
    selected_subsets = (
        manifest_subsets
        if manifest_subsets is not None
        else tuple(FlawedFictionsSplit.parse(split) for split in args.subset)
    )
    config = RunConfig(
        model=args.model,
        strategy=args.strategy,
        subsets=selected_subsets,
        limit=args.limit,
        run_id=args.run_id,
        run_root=args.run_root,
        raw_data_dir=args.raw_data_dir,
        manifest_path=manifest_path,
        resume=args.resume,
        retry_failures=args.retry_failures,
        think_override=_parse_optional_bool(args.think),
        max_new_examples=args.max_new_examples,
    )
    try:
        result = run_experiment(config)
    except ExperimentRunError as exc:
        print("Run failed:")
        print(exc)
        return 1
    except Exception as exc:
        print("Run failed:")
        print(f"{type(exc).__name__}: {exc}")
        return 1

    if args.limit is not None:
        print("SMOKE TEST -- NOT A RESEARCH RESULT")
    elif args.max_new_examples is not None:
        print("PRE-FLIGHT PARTIAL RUN -- NOT FINAL RESULTS")
    else:
        print("Run completed")
    print(f"Run id: {result.run_id}")
    print(f"Run directory: {result.run_dir}")
    print(f"Expected examples: {result.expected_examples}")
    print(f"Completed predictions: {result.completed_examples}")
    print(f"Inference failures: {result.failed_examples}")
    print(f"Skipped completed: {result.skipped_completed}")
    print(f"Skipped failed: {result.skipped_failed}")
    print(f"Saved: {result.run_dir / 'config.yaml'}")
    print(f"Saved: {result.run_dir / 'metadata.json'}")
    print(f"Saved: {result.run_dir / 'predictions.jsonl'}")
    print(f"Saved: {result.run_dir / 'errors.jsonl'}")
    if result.evaluated:
        print(f"Saved: {result.run_dir / 'metrics.json'}")
        print(f"Saved: {result.run_dir / 'summary.md'}")
    return 0


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


if __name__ == "__main__":
    raise SystemExit(main())
