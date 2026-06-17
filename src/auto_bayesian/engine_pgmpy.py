# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from auto_bayesian.metrics import average_precision, log_loss, roc_auc


@dataclass(slots=True)
class CandidateResult:
    name: str
    model: Any
    edges: list[tuple[str, str]]
    validation_probabilities: list[float]
    roc_auc: float
    pr_auc: float
    log_loss: float


def train_candidates(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_column: str,
    positive_label: str,
    state_names: dict[str, list[str]],
) -> list[CandidateResult]:
    train_frame = train_frame.astype(object)
    validation_frame = validation_frame.astype(object)
    features = [column for column in train_frame.columns if column != target_column]
    if not features:
        raise ValueError("Training frame needs at least one feature column.")
    candidates = [
        ("naive_bayes", _fit_naive_bayes(train_frame, target_column, state_names)),
        ("tan", _fit_tan(train_frame, target_column, state_names)),
        ("hill_climb", _fit_hill_climb(train_frame, target_column, state_names)),
    ]

    results = []
    labels = (validation_frame[target_column].astype(str) == positive_label).astype(int).tolist()
    for name, model in candidates:
        probabilities = predict_probabilities(
            model, validation_frame[features], target_column, positive_label
        )
        results.append(
            CandidateResult(
                name=name,
                model=model,
                edges=sorted(tuple(edge) for edge in model.edges()),
                validation_probabilities=probabilities,
                roc_auc=roc_auc(labels, probabilities),
                pr_auc=average_precision(labels, probabilities),
                log_loss=log_loss(labels, probabilities),
            )
        )
    return results


def predict_probabilities(
    model: Any,
    frame: pd.DataFrame,
    target_column: str,
    positive_label: str,
) -> list[float]:
    """Return ``P(target == positive_label)`` for each row via exact inference.

    Rows that share the same post-discretization evidence are deduplicated, so
    Variable Elimination runs only once per unique combination. After binning,
    many rows collapse to identical evidence, which keeps inference fast on large
    frames while producing results identical to a naive row-by-row query.
    """
    from pgmpy.inference import VariableElimination

    infer = VariableElimination(model)
    nodes = set(model.nodes())
    evidence_columns = [
        column for column in frame.columns if column != target_column and column in nodes
    ]
    evidence_frame = frame[evidence_columns].astype(str)

    cache: dict[tuple[str, ...], float] = {}
    probabilities: list[float] = []
    for row in evidence_frame.itertuples(index=False, name=None):
        if row not in cache:
            evidence = dict(zip(evidence_columns, row, strict=True))
            factor = infer.query([target_column], evidence=evidence, show_progress=False)
            states = list(factor.state_names[target_column])
            index = states.index(positive_label)
            cache[row] = float(factor.values[index])
        probabilities.append(cache[row])
    return probabilities


def predict_best_action(
    model: Any,
    frame: pd.DataFrame,
    target_column: str,
    positive_label: str,
    action_column: str,
    action_values: list[str],
) -> list[dict[str, object]]:
    """Rank ``action_column`` values by their effect on the positive outcome.

    As in :func:`predict_probabilities`, rows that share the same post-discretization
    evidence are deduplicated so the per-action inference runs once per unique
    evidence combination.
    """
    from pgmpy.inference import VariableElimination

    infer = VariableElimination(model)
    nodes = set(model.nodes())
    evidence_columns = [
        column
        for column in frame.columns
        if column not in (target_column, action_column) and column in nodes
    ]
    evidence_frame = frame[evidence_columns].astype(str)

    cache: dict[tuple[str, ...], dict[str, object]] = {}
    results: list[dict[str, object]] = []
    for row in evidence_frame.itertuples(index=False, name=None):
        if row not in cache:
            base_evidence = dict(zip(evidence_columns, row, strict=True))
            best_action = action_values[0]
            best_prob = -1.0
            action_probs: dict[str, float] = {}
            for action_val in action_values:
                evidence = {**base_evidence, action_column: action_val}
                factor = infer.query([target_column], evidence=evidence, show_progress=False)
                states = list(factor.state_names[target_column])
                idx = states.index(positive_label)
                prob = float(factor.values[idx])
                action_probs[action_val] = prob
                if prob > best_prob:
                    best_prob = prob
                    best_action = action_val
            cache[row] = {
                "recommended_action": best_action,
                "expected_probability": best_prob,
                "action_probabilities": action_probs,
            }
        results.append(cache[row])
    return results


def _fit_naive_bayes(
    train_frame: pd.DataFrame, target_column: str, state_names: dict[str, list[str]]
) -> Any:
    from pgmpy.estimators import BayesianEstimator
    from pgmpy.models import DiscreteBayesianNetwork

    features = [column for column in train_frame.columns if column != target_column]
    model = DiscreteBayesianNetwork((target_column, feature) for feature in features)
    model.add_nodes_from(train_frame.columns)
    model.fit(train_frame, estimator=BayesianEstimator, prior_type="BDeu", state_names=state_names)
    _prune_unfitted_nodes(model, target_column)
    return model


def _fit_tan(
    train_frame: pd.DataFrame, target_column: str, state_names: dict[str, list[str]]
) -> Any:
    from pgmpy.estimators import BayesianEstimator, TreeSearch
    from pgmpy.models import DiscreteBayesianNetwork

    estimator = TreeSearch(train_frame)
    dag = estimator.estimate(estimator_type="tan", class_node=target_column)
    model = DiscreteBayesianNetwork(dag.edges())
    model.add_nodes_from(train_frame.columns)
    model.fit(train_frame, estimator=BayesianEstimator, prior_type="BDeu", state_names=state_names)
    _prune_unfitted_nodes(model, target_column)
    return model


def _fit_hill_climb(
    train_frame: pd.DataFrame,
    target_column: str,
    state_names: dict[str, list[str]],
) -> Any:
    from pgmpy.estimators import BayesianEstimator, HillClimbSearch
    from pgmpy.models import DiscreteBayesianNetwork

    estimator = HillClimbSearch(train_frame)
    dag = estimator.estimate(scoring_method="bic-d", show_progress=False)
    model = DiscreteBayesianNetwork(dag.edges())
    model.add_nodes_from(train_frame.columns)
    model.fit(train_frame, estimator=BayesianEstimator, prior_type="BDeu", state_names=state_names)
    _prune_unfitted_nodes(model, target_column)
    return model


def _prune_unfitted_nodes(model: Any, target_column: str) -> None:
    missing = [node for node in list(model.nodes()) if model.get_cpds(node) is None]
    if target_column in missing:
        raise ValueError("The trained model does not contain a fitted target CPD.")
    if missing:
        model.remove_nodes_from(missing)
