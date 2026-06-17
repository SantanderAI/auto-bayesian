# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

Labels = Sequence[int]
Scores = Sequence[float]


@dataclass(slots=True)
class BinaryMetrics:
    """Threshold-free and thresholded scores for a binary classifier.

    ``roc_auc`` and ``pr_auc`` describe ranking quality independent of any
    decision threshold; ``precision``, ``recall``, and ``f1`` are measured at
    the stored ``threshold``. ``positive_rate`` is the base rate of the positive
    class, the context needed to read ``pr_auc`` honestly.
    """

    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    positive_rate: float


def roc_auc(labels: Labels, scores: Scores) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) statistic.

    Ties in ``scores`` share their average rank so the result is independent of
    input order. Requires both classes to be present.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC needs both classes present.")

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    position = 0
    while position < len(ranked):
        start = position
        score = ranked[start][0]
        while position < len(ranked) and ranked[position][0] == score:
            position += 1
        average_rank = (start + 1 + position) / 2.0
        positives_in_group = sum(label for _, label in ranked[start:position])
        rank_sum += positives_in_group * average_rank
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: Labels, scores: Scores) -> float:
    """Average precision (the standard "PR AUC"): the precision-weighted sum of
    recall increments as the decision threshold sweeps from high to low scores.

    Records that share a score are grouped so the value does not depend on input
    order. Returns ``0.0`` when there are no positives.
    """
    positives = sum(labels)
    if positives == 0:
        return 0.0

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    cumulative_tp = 0
    seen = 0
    previous_recall = 0.0
    score_total = 0.0
    position = 0
    while position < len(ranked):
        start = position
        score = ranked[start][0]
        while position < len(ranked) and ranked[position][0] == score:
            position += 1
        cumulative_tp += sum(label for _, label in ranked[start:position])
        seen = position
        precision = cumulative_tp / seen
        recall = cumulative_tp / positives
        score_total += precision * (recall - previous_recall)
        previous_recall = recall
    return score_total


def precision_recall_f1(
    labels: Labels, scores: Scores, threshold: float
) -> tuple[float, float, float]:
    """Precision, recall, and F1 when predicting positive iff ``score >= threshold``."""
    tp = fp = fn = 0
    for label, score in zip(labels, scores, strict=True):
        predicted = score >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def best_f1_threshold(labels: Labels, scores: Scores) -> tuple[float, float]:
    """Return the ``score >= threshold`` cutoff that maximizes F1, and that F1.

    Candidate cutoffs are the distinct scores, scanned from high to low. This is
    the right default for imbalanced targets, where a fixed 0.5 cutoff often
    predicts the majority class for every row (F1 = 0). Returns the highest score
    as the threshold when there are no positives.
    """
    positives = sum(labels)
    if positives == 0:
        return (max(scores) if scores else 1.0), 0.0

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    cumulative_tp = 0
    seen = 0
    best_f1 = -1.0
    best_threshold = ranked[0][0]
    position = 0
    while position < len(ranked):
        start = position
        score = ranked[start][0]
        while position < len(ranked) and ranked[position][0] == score:
            position += 1
        cumulative_tp += sum(label for _, label in ranked[start:position])
        seen = position
        precision = cumulative_tp / seen
        recall = cumulative_tp / positives
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = score
    return best_threshold, best_f1


def log_loss(labels: Labels, scores: Scores) -> float:
    """Mean binary cross-entropy, with scores clipped away from 0 and 1."""
    clipped = [min(max(score, 1e-9), 1 - 1e-9) for score in scores]
    total = 0.0
    for label, score in zip(labels, clipped, strict=True):
        total += label * math.log(score) + (1 - label) * math.log(1 - score)
    return -total / len(labels)


def evaluate_binary(
    labels: Labels, scores: Scores, threshold: float | None = None
) -> BinaryMetrics:
    """Compute ranking and thresholded metrics for a binary classifier.

    When ``threshold`` is ``None`` the F1-optimal cutoff is selected with
    :func:`best_f1_threshold`, which is the honest default for imbalanced data.
    """
    labels = [int(label) for label in labels]
    scores = [float(score) for score in scores]
    if not labels:
        raise ValueError("Cannot evaluate an empty set of predictions.")
    chosen = best_f1_threshold(labels, scores)[0] if threshold is None else float(threshold)
    precision, recall, f1 = precision_recall_f1(labels, scores, chosen)
    return BinaryMetrics(
        roc_auc=roc_auc(labels, scores),
        pr_auc=average_precision(labels, scores),
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=chosen,
        positive_rate=sum(labels) / len(labels),
    )
