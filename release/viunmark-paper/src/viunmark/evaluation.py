"""Prediction loading and metric scoring utilities."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from viunmark.config import SIX_CONDITIONS, ViUnMarkContractError


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], *, num_labels: int) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ViUnMarkContractError("y_true and y_pred must be non-empty and aligned")
    scores = []
    for label in range(num_labels):
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / num_labels


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if len(y_true) != len(y_pred) or not y_true:
        raise ViUnMarkContractError("y_true and y_pred must be non-empty and aligned")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def load_prediction_csv(path: Path) -> tuple[list[str], list[str], list[int]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    required = {"sample_id", "condition", "prediction"}
    if not rows or not required <= set(rows[0]):
        raise ViUnMarkContractError(f"prediction CSV must contain {sorted(required)}")
    return [r["sample_id"] for r in rows], [r["condition"] for r in rows], [int(r["prediction"]) for r in rows]


def load_label_csv(path: Path) -> dict[str, int]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows or not {"sample_id", "label"} <= set(rows[0]):
        raise ViUnMarkContractError("label CSV must contain sample_id,label")
    return {r["sample_id"]: int(r["label"]) for r in rows}


def score_prediction_file(prediction_csv: Path, labels_csv: Path, *, num_labels: int = 3) -> dict[str, object]:
    ids, conditions, predictions = load_prediction_csv(prediction_csv)
    labels_by_id = load_label_csv(labels_csv)
    missing = [sample_id for sample_id in ids if sample_id not in labels_by_id]
    if missing:
        raise ViUnMarkContractError(f"labels missing for prediction ids: {missing[:5]}")
    y_true = [labels_by_id[sample_id] for sample_id in ids]
    overall = {"macro_f1": macro_f1(y_true, predictions, num_labels=num_labels), "accuracy": accuracy(y_true, predictions)}
    by_condition = {}
    for condition in SIX_CONDITIONS:
        subset = [i for i, c in enumerate(conditions) if c == condition]
        if subset:
            t = [y_true[i] for i in subset]
            p = [predictions[i] for i in subset]
            by_condition[condition] = {"macro_f1": macro_f1(t, p, num_labels=num_labels), "accuracy": accuracy(t, p), "n": len(subset)}
    counts = Counter(conditions)
    return {"overall": overall, "by_condition": by_condition, "condition_counts": dict(counts)}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
