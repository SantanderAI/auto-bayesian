# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from auto_bayesian.engine_pgmpy import CandidateResult, train_candidates
from auto_bayesian.materialize import materialize_project
from auto_bayesian.metrics import evaluate_binary
from auto_bayesian.model import AutoBayesModel, CandidateReport, ModelReport
from auto_bayesian.preprocess import DataPreprocessor, remove_outliers
from auto_bayesian.schema import ProjectSpec


@dataclass(slots=True)
class SplitFrame:
    train: pd.DataFrame
    validation: pd.DataFrame


def fit_project(
    project: ProjectSpec, tables: dict[str, pd.DataFrame] | None = None
) -> AutoBayesModel:
    """Train and select the best Bayesian-network model for a project.

    Materializes the relational tables into a single modeling frame, applies
    deterministic preprocessing, trains the Naive Bayes, TAN, and Hill-Climb
    candidates, and returns the candidate with the best validation ROC-AUC.

    Parameters
    ----------
    project:
        Parsed project configuration describing tables, relations, task, and
        preprocessing options.
    tables:
        Optional in-memory frames keyed by table name. When omitted, tables are
        read from the paths declared in ``project``.

    Returns
    -------
    AutoBayesModel
        The fitted model wrapping the selected network, the fitted
        preprocessor, a report, and the materialized frame.
    """
    materialized = materialize_project(project, tables=tables)
    model_frame = materialized.drop(
        columns=_identifier_columns(project, materialized.columns),
        errors="ignore",
    )
    preprocessor = DataPreprocessor(
        project.preprocess,
        target_column=project.task.target_column,
        positive_label=project.task.positive_label,
    )
    split = stratified_split(
        model_frame,
        target_column=project.task.target_column,
        test_fraction=project.run.test_fraction,
        seed=project.run.random_seed,
    )
    train_raw = split.train
    if project.preprocess.drop_duplicates:
        train_raw = train_raw.drop_duplicates().reset_index(drop=True)
    train_raw = remove_outliers(train_raw, project.preprocess, project.task.target_column)
    train_frame = preprocessor.fit_transform(train_raw)
    validation_frame = preprocessor.transform(split.validation)
    candidates = train_candidates(
        train_frame,
        validation_frame,
        target_column=project.task.target_column,
        positive_label=project.task.positive_label,
        state_names=preprocessor.state_names(),
    )
    metric = project.run.selection_metric
    best = select_best_candidate(candidates, metric=metric)
    validation_labels = (
        (validation_frame[project.task.target_column].astype(str) == project.task.positive_label)
        .astype(int)
        .tolist()
    )
    validation_metrics = evaluate_binary(validation_labels, best.validation_probabilities)
    return AutoBayesModel(
        network=best.model,
        preprocessor=preprocessor,
        report=ModelReport(
            selected_candidate=best.name,
            roc_auc=best.roc_auc,
            log_loss=best.log_loss,
            target_column=project.task.target_column,
            positive_label=project.task.positive_label,
            edges=best.edges,
            pr_auc=validation_metrics.pr_auc,
            threshold=validation_metrics.threshold,
            precision=validation_metrics.precision,
            recall=validation_metrics.recall,
            f1=validation_metrics.f1,
            candidates=[
                CandidateReport(
                    name=candidate.name,
                    roc_auc=candidate.roc_auc,
                    pr_auc=candidate.pr_auc,
                    log_loss=candidate.log_loss,
                    edges=candidate.edges,
                )
                for candidate in sorted(
                    candidates,
                    key=lambda candidate: (-getattr(candidate, metric), candidate.log_loss),
                )
            ],
            task_type=project.task.task_type,
            action_column=project.task.action_column,
        ),
        materialized_frame=materialized,
    )


def fit_tables(project: ProjectSpec, tables: dict[str, pd.DataFrame]) -> AutoBayesModel:
    """Train a model directly from in-memory pandas frames.

    Convenience wrapper around :func:`fit_project` for the common case where the
    tables are already loaded in memory rather than read from disk.
    """
    return fit_project(project, tables=tables)


def select_best_candidate(
    candidates: list[CandidateResult], metric: str = "roc_auc"
) -> CandidateResult:
    """Return the best candidate by ``metric`` (``roc_auc`` or ``pr_auc``).

    Ties are broken by lower log loss. ``pr_auc`` is the better choice for
    imbalanced targets, where ROC-AUC can look healthy while the positive class
    is ranked poorly.
    """
    return max(candidates, key=lambda candidate: (getattr(candidate, metric), -candidate.log_loss))


def stratified_split(
    frame: pd.DataFrame,
    *,
    target_column: str,
    test_fraction: float,
    seed: int,
) -> SplitFrame:
    """Split ``frame`` into train/validation sets, preserving class balance.

    Each target class is shuffled with a seeded RNG and split independently so
    both partitions keep at least one example of every class. The split is fully
    deterministic for a given ``seed``.
    """
    target = frame[target_column].astype(str)
    by_class: dict[str, list[int]] = {}
    for index, value in target.items():
        by_class.setdefault(value, []).append(index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    rng = random.Random(seed)
    for indices in by_class.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * test_fraction)))
        if validation_count >= len(shuffled):
            validation_count = len(shuffled) - 1
        if validation_count <= 0:
            raise ValueError("Each class needs at least two rows for the validation split.")
        validation_indices.extend(shuffled[:validation_count])
        train_indices.extend(shuffled[validation_count:])

    train = frame.loc[sorted(train_indices)].reset_index(drop=True)
    validation = frame.loc[sorted(validation_indices)].reset_index(drop=True)
    return SplitFrame(train=train, validation=validation)


def _identifier_columns(project: ProjectSpec, columns: pd.Index) -> list[str]:
    keys = set()
    for table in project.tables.values():
        keys.update(table.primary_key)
    if project.task.target_column in keys:
        keys.remove(project.task.target_column)
    return [column for column in columns if column in keys]
