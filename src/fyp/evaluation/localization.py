"""CEEval localization scoring following the released FLAWEDFICTIONS code."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from fyp.data.schema import FlawedFictionsExample


HTML_MARKERS = ("<li>", "</li>", "<br>", "<br/>", "<p>", "</p>", "<m>", "</m>")


@dataclass(frozen=True)
class LocalizationScore:
    """Per-example localization result."""

    full_correct: float
    cont_error_lines_correct: float
    contradicted_lines_correct: float
    predicted_cont_error_lines_full: tuple[str, ...]
    predicted_contradicted_lines_full: tuple[str, ...]
    ground_truth_cont_error_lines_full: tuple[str, ...]
    ground_truth_contradicted_lines_full: tuple[str, ...]


def preprocess_sent(sent: str) -> str:
    """Normalize a sentence with the official evaluator's text cleanup rules."""

    sent = sent.lower()
    sent = sent.replace("â€¢", "")
    sent = sent.replace("“", "").replace("”", "")
    sent = re.sub(r"\[.*?\]", "", sent)
    sent = _replace_html_markers(sent)
    sent = re.sub(r"[^\w\s]", "", sent)
    sent = re.sub(r"\s+", " ", sent)
    return sent.strip()


def listify_lines(lines: Any) -> list[str]:
    """Convert released line fields to lists with official marker stripping."""

    if isinstance(lines, str):
        if "<li>" in lines:
            parts = lines.split("</li>")
        else:
            parts = lines.split("\n")
        return [
            line.strip().replace("-", "").replace("*", "").strip()
            for line in parts
            if line.strip().replace("-", "").replace("*", "").strip() != ""
        ]
    if isinstance(lines, Sequence):
        return [str(line) for line in lines]
    return []


def extract_full_line(story: str, predicted_line: str) -> list[str]:
    """Find story sentences containing normalized predicted sentence snippets."""

    story = re.sub(r"\[.*?\]", "", story)
    predicted_line = re.sub(r"\[.*?\]", "", predicted_line)
    lines = sentence_tokenize(_replace_html_markers(story))
    predicted_lines = sentence_tokenize(predicted_line)
    full_lines: list[str] = []
    for line in lines:
        for pline in predicted_lines:
            processed_line = preprocess_sent(line)
            processed_pline = preprocess_sent(pline)
            if processed_pline in processed_line:
                full_lines.append(line)
    return full_lines


def extract_all_full_lines(story: str, predicted_lines: Iterable[str]) -> list[str]:
    """Expand all predicted snippets to full story sentences."""

    full_lines: list[str] = []
    for line in predicted_lines:
        full_lines += extract_full_line(story, line)
    return full_lines


def score_localization(
    example: FlawedFictionsExample,
    predicted: Mapping[str, Any] | None,
    strict: bool = False,
) -> LocalizationScore:
    """Score one example with official CEEval-Full semantics."""

    predicted_cont_error = _prediction_label(predicted)
    ground_truth_cont_error = bool(example.cont_error)

    if predicted_cont_error != ground_truth_cont_error:
        return LocalizationScore(
            full_correct=0.0,
            cont_error_lines_correct=0.0,
            contradicted_lines_correct=0.0,
            predicted_cont_error_lines_full=(),
            predicted_contradicted_lines_full=(),
            ground_truth_cont_error_lines_full=(),
            ground_truth_contradicted_lines_full=(),
        )

    if predicted_cont_error is False:
        return LocalizationScore(
            full_correct=1.0,
            cont_error_lines_correct=1.0,
            contradicted_lines_correct=1.0,
            predicted_cont_error_lines_full=(),
            predicted_contradicted_lines_full=(),
            ground_truth_cont_error_lines_full=(),
            ground_truth_contradicted_lines_full=(),
        )

    gt_cont_error_lines = listify_lines(example.cont_error_lines_raw)
    gt_contradicted_lines = listify_lines(example.contradicted_lines_raw)
    pred_cont_error_lines = listify_lines(predicted.get("cont_error_lines", []) if predicted else [])
    pred_contradicted_lines = listify_lines(predicted.get("contradicted_lines", []) if predicted else [])

    gt_cont_error_lines_full = tuple(extract_all_full_lines(example.story, gt_cont_error_lines))
    gt_contradicted_lines_full = tuple(extract_all_full_lines(example.story, gt_contradicted_lines))
    pred_cont_error_lines_full = tuple(extract_all_full_lines(example.story, pred_cont_error_lines))
    pred_contradicted_lines_full = tuple(extract_all_full_lines(example.story, pred_contradicted_lines))

    pred_error_set = set(pred_cont_error_lines_full)
    gt_error_set = set(gt_cont_error_lines_full)
    pred_contradicted_set = set(pred_contradicted_lines_full)
    gt_contradicted_set = set(gt_contradicted_lines_full)

    if strict:
        cont_error_detected = bool(pred_error_set) and pred_error_set.issubset(gt_error_set)
        contradicted_detected = bool(pred_contradicted_set) and pred_contradicted_set.issubset(
            gt_contradicted_set
        )
    else:
        cont_error_detected = bool(pred_error_set.intersection(gt_error_set))
        contradicted_detected = bool(pred_contradicted_set.intersection(gt_contradicted_set))

    return LocalizationScore(
        full_correct=float(cont_error_detected and contradicted_detected),
        cont_error_lines_correct=float(cont_error_detected),
        contradicted_lines_correct=float(contradicted_detected),
        predicted_cont_error_lines_full=pred_cont_error_lines_full,
        predicted_contradicted_lines_full=pred_contradicted_lines_full,
        ground_truth_cont_error_lines_full=gt_cont_error_lines_full,
        ground_truth_contradicted_lines_full=gt_contradicted_lines_full,
    )


def sentence_tokenize(text: str) -> list[str]:
    """Tokenize sentences.

    The released code calls `nltk.sent_tokenize`. If NLTK with tokenizer data is
    available, use it. Otherwise use a deterministic fallback for local tests and
    mark exact NLTK-data parity as a documented limitation.
    """

    try:
        from nltk import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        stripped = text.strip()
        if not stripped:
            return []
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped) if part.strip()]


def _replace_html_markers(text: str) -> str:
    for marker in HTML_MARKERS:
        text = text.replace(marker, " ")
    return text


def _prediction_label(predicted: Mapping[str, Any] | None) -> bool | None:
    if predicted is None or "cont_error" not in predicted:
        return None
    value = predicted["cont_error"]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
        return bool(value)
    return None
