# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

"""auto-bayesian: interpretable AutoML for Bayesian-network classifiers.

Declare your relational tables and target, and the library materializes them
into one modeling frame, trains a short list of Bayesian-network candidates
(Naive Bayes, TAN, Hill-Climb), selects the best by ROC-AUC or PR-AUC, tunes a
decision threshold for F1, and returns a model you can persist, score with,
evaluate, and explain.
"""

from auto_bayesian.automl import fit_project, fit_tables
from auto_bayesian.explain import generate_explanation, to_mermaid
from auto_bayesian.materialize import materialize_project
from auto_bayesian.metrics import BinaryMetrics, evaluate_binary
from auto_bayesian.model import AutoBayesModel, CandidateReport, ModelReport
from auto_bayesian.schema import (
    AggregateSpec,
    PreprocessSpec,
    ProjectSpec,
    RelationSpec,
    RunSpec,
    TableSpec,
    TaskSpec,
    build_project,
    load_project,
)

__all__ = [
    "AggregateSpec",
    "AutoBayesModel",
    "BinaryMetrics",
    "CandidateReport",
    "ModelReport",
    "evaluate_binary",
    "PreprocessSpec",
    "ProjectSpec",
    "RelationSpec",
    "RunSpec",
    "TableSpec",
    "TaskSpec",
    "build_project",
    "fit_project",
    "fit_tables",
    "generate_explanation",
    "load_project",
    "materialize_project",
    "to_mermaid",
]
