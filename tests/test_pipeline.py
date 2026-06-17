# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from auto_bayesian.automl import fit_project
from auto_bayesian.model import AutoBayesModel
from auto_bayesian.schema import load_project

EXAMPLE_CONFIG = Path("examples/lead_scoring.toml")


def test_fit_project_trains_and_round_trips(tmp_path: Path) -> None:
    project = load_project(EXAMPLE_CONFIG)
    project.run.output_dir = tmp_path / "artifacts"

    model = fit_project(project)
    model.save(project.run.output_dir)
    restored = AutoBayesModel.load(project.run.output_dir)

    features = restored.materialized_frame.drop(columns=[project.task.target_column])
    probabilities = restored.predict_proba(features)

    assert restored.report.selected_candidate in {"naive_bayes", "tan", "hill_climb"}
    assert 0.0 <= restored.report.roc_auc <= 1.0
    assert restored.report.log_loss >= 0.0
    assert len(restored.report.candidates) == 3
    assert len(probabilities) == len(features)
    assert probabilities.between(0, 1).all()
