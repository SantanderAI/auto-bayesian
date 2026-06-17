# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from auto_bayesian.engine_pgmpy import predict_best_action, predict_probabilities
from auto_bayesian.metrics import BinaryMetrics, evaluate_binary
from auto_bayesian.preprocess import DataPreprocessor


@dataclass(slots=True)
class CandidateReport:
    name: str
    roc_auc: float
    log_loss: float
    edges: list[tuple[str, str]]
    pr_auc: float = 0.0


@dataclass(slots=True)
class ModelReport:
    selected_candidate: str
    roc_auc: float
    log_loss: float
    target_column: str
    positive_label: str
    edges: list[tuple[str, str]]
    pr_auc: float = 0.0
    threshold: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    candidates: list[CandidateReport] = field(default_factory=list)
    task_type: str = "classification"
    action_column: str | None = None


class AutoBayesModel:
    """A fitted Bayesian network with preprocessing, scoring, and persistence.

    Wraps the selected ``pgmpy`` network together with the fitted preprocessor
    and the training report, and exposes a small, typed prediction API.
    """

    def __init__(
        self,
        *,
        network: Any,
        preprocessor: DataPreprocessor,
        report: ModelReport,
        materialized_frame: pd.DataFrame | None = None,
    ) -> None:
        self.network = network
        self.preprocessor = preprocessor
        self.report = report
        self.materialized_frame = materialized_frame

    def predict_proba(self, frame: pd.DataFrame) -> pd.Series:
        """Return ``P(target == positive_label)`` for each row in ``frame``."""
        prepared = self.preprocessor.transform(frame)
        probabilities = predict_probabilities(
            self.network,
            prepared[self.preprocessor.feature_columns],
            self.report.target_column,
            self.report.positive_label,
        )
        return pd.Series(probabilities, index=frame.index, name="probability")

    def predict(self, frame: pd.DataFrame, threshold: float | None = None) -> pd.Series:
        """Return binary predictions by thresholding :meth:`predict_proba`.

        When ``threshold`` is ``None`` the model's tuned decision threshold
        (``report.threshold``, chosen to maximize validation F1) is used instead
        of a fixed 0.5, which would predict the majority class on imbalanced data.
        """
        cutoff = self.report.threshold if threshold is None else threshold
        probabilities = self.predict_proba(frame)
        return (probabilities >= cutoff).astype(int).rename("prediction")

    def evaluate(self, frame: pd.DataFrame, threshold: float | None = None) -> BinaryMetrics:
        """Score a labeled ``frame`` and return ranking and thresholded metrics.

        ``frame`` must contain the target column. ROC-AUC and PR-AUC are
        threshold-free; precision, recall, and F1 are measured at the model's
        tuned threshold unless ``threshold`` is given.
        """
        if self.report.target_column not in frame.columns:
            raise ValueError(f"evaluate requires the target column '{self.report.target_column}'.")
        labels = (
            (frame[self.report.target_column].astype(str) == self.report.positive_label)
            .astype(int)
            .tolist()
        )
        scores = self.predict_proba(frame).tolist()
        cutoff = self.report.threshold if threshold is None else threshold
        return evaluate_binary(labels, scores, threshold=cutoff)

    def predict_next_best_action(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Recommend the action that maximizes the positive outcome per row.

        Requires a model trained with ``task_type='next_best_action'`` and an
        ``action_column``. Returns the recommended action and its expected
        probability for each row.
        """
        if not self.report.action_column:
            raise ValueError(
                "predict_next_best_action requires a model trained with "
                "task_type='next_best_action' and an action_column."
            )
        action_col = self.report.action_column
        action_values = self.preprocessor.state_names().get(action_col, [])
        if not action_values:
            raise ValueError(f"No known states for action_column '{action_col}'.")
        prepared = self.preprocessor.transform(frame)
        results = predict_best_action(
            self.network,
            prepared[self.preprocessor.feature_columns],
            self.report.target_column,
            self.report.positive_label,
            action_col,
            action_values,
        )
        return pd.DataFrame(
            [
                {
                    "recommended_action": r["recommended_action"],
                    "expected_probability": r["expected_probability"],
                }
                for r in results
            ],
            index=frame.index,
        )

    def describe(self) -> ModelReport:
        """Return the training report (selected candidate, metrics, edges)."""
        return self.report

    def save(self, output_dir: str | Path) -> None:
        """Persist the model, metrics, network, and materialized frame to disk.

        Writes ``model.pkl``, ``metrics.json``, ``network.json``, and (when
        available) ``materialized.parquet`` into ``output_dir``.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "model.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "network": self.network,
                    "preprocessor": self.preprocessor.to_dict(),
                    "report": {
                        "selected_candidate": self.report.selected_candidate,
                        "roc_auc": self.report.roc_auc,
                        "pr_auc": self.report.pr_auc,
                        "log_loss": self.report.log_loss,
                        "threshold": self.report.threshold,
                        "precision": self.report.precision,
                        "recall": self.report.recall,
                        "f1": self.report.f1,
                        "target_column": self.report.target_column,
                        "positive_label": self.report.positive_label,
                        "edges": self.report.edges,
                        "candidates": [asdict(candidate) for candidate in self.report.candidates],
                        "task_type": self.report.task_type,
                        "action_column": self.report.action_column,
                    },
                },
                handle,
            )
        (output_path / "metrics.json").write_text(
            json.dumps(
                {
                    "selected_candidate": self.report.selected_candidate,
                    "roc_auc": self.report.roc_auc,
                    "pr_auc": self.report.pr_auc,
                    "log_loss": self.report.log_loss,
                    "threshold": self.report.threshold,
                    "precision": self.report.precision,
                    "recall": self.report.recall,
                    "f1": self.report.f1,
                    "candidates": [asdict(candidate) for candidate in self.report.candidates],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_path / "network.json").write_text(
            json.dumps(
                {
                    "target_column": self.report.target_column,
                    "positive_label": self.report.positive_label,
                    "edges": self.report.edges,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if self.materialized_frame is not None:
            self.materialized_frame.to_parquet(output_path / "materialized.parquet", index=False)

    @classmethod
    def load(cls, output_dir: str | Path) -> "AutoBayesModel":
        """Load a model previously written by :meth:`save`."""
        output_path = Path(output_dir)
        with (output_path / "model.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        report_payload = payload["report"].copy()
        report_payload["candidates"] = [
            CandidateReport(**candidate) for candidate in report_payload.get("candidates", [])
        ]
        report = ModelReport(**report_payload)
        materialized_path = output_path / "materialized.parquet"
        materialized = pd.read_parquet(materialized_path) if materialized_path.exists() else None
        return cls(
            network=payload["network"],
            preprocessor=DataPreprocessor.from_dict(payload["preprocessor"]),
            report=report,
            materialized_frame=materialized,
        )
