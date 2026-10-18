"""Aggregate metrics for FLAWEDFICTIONS evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def compute_classification_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float | None],
) -> ClassificationMetrics:
    """Compute official binary aggregate metrics without dropping failures."""

    true_values = [float(value) for value in y_true]
    pred_values = [None if value is None else float(value) for value in y_pred]
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_values:
        return ClassificationMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0)

    correct = [
        pred is not None and pred == truth
        for truth, pred in zip(true_values, pred_values)
    ]
    tp = sum(1 for truth, pred in zip(true_values, pred_values) if truth == 1.0 and pred == 1.0)
    fp = sum(1 for truth, pred in zip(true_values, pred_values) if truth == 0.0 and pred == 1.0)
    fn = sum(1 for truth, pred in zip(true_values, pred_values) if truth == 1.0 and pred != 1.0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return ClassificationMetrics(
        accuracy=sum(correct) / len(correct),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return mean(values) if values else 0.0
