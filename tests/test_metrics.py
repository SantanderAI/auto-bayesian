# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

import pytest

from auto_bayesian.metrics import (
    BinaryMetrics,
    average_precision,
    best_f1_threshold,
    evaluate_binary,
    log_loss,
    precision_recall_f1,
    roc_auc,
)


def test_roc_auc_perfect_separation() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_roc_auc_random_scores_is_half() -> None:
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_roc_auc_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        roc_auc([1, 1, 1], [0.1, 0.2, 0.3])


def test_average_precision_perfect_ranking() -> None:
    assert average_precision([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_average_precision_no_positives_is_zero() -> None:
    assert average_precision([0, 0, 0], [0.1, 0.2, 0.3]) == 0.0


def test_precision_recall_f1_threshold_semantics() -> None:
    precision, recall, f1 = precision_recall_f1([0, 1, 1], [0.4, 0.6, 0.9], 0.6)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0


def test_best_f1_threshold_recovers_positives() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    threshold, f1 = best_f1_threshold(labels, scores)
    assert f1 == 1.0
    assert [1 if score >= threshold else 0 for score in scores] == labels


def test_best_f1_beats_fixed_half_on_imbalance() -> None:
    # Rare positives whose scores never reach 0.5: a fixed 0.5 cutoff scores F1=0.
    labels = [0] * 90 + [1] * 10
    scores = [0.05] * 90 + [0.2] * 10
    _, _, f1_at_half = precision_recall_f1(labels, scores, 0.5)
    assert f1_at_half == 0.0
    threshold, f1 = best_f1_threshold(labels, scores)
    assert f1 > 0.0
    assert threshold <= 0.2


def test_log_loss_zero_for_confident_correct() -> None:
    assert log_loss([1, 0], [1.0, 0.0]) < 1e-6


def test_evaluate_binary_tunes_threshold_by_default() -> None:
    labels = [0] * 80 + [1] * 20
    scores = [0.1] * 80 + [0.3] * 20
    metrics = evaluate_binary(labels, scores)
    assert isinstance(metrics, BinaryMetrics)
    assert metrics.f1 > 0.0
    assert metrics.positive_rate == 0.2
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_evaluate_binary_respects_explicit_threshold() -> None:
    metrics = evaluate_binary([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], threshold=0.99)
    assert metrics.threshold == 0.99
    assert metrics.precision == 0.0


def test_evaluate_binary_rejects_empty() -> None:
    with pytest.raises(ValueError):
        evaluate_binary([], [])
