# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd

from auto_bayesian.automl import fit_project
from auto_bayesian.schema import load_project

EXAMPLE_CONFIG = Path("examples/lead_scoring.toml")


def _trained_model():
    project = load_project(EXAMPLE_CONFIG)
    return fit_project(project), project.task.target_column


def test_predict_proba_matches_row_by_row() -> None:
    """Evidence deduplication must not change per-row probabilities."""
    model, target_column = _trained_model()
    features = model.materialized_frame.drop(columns=[target_column])

    batch = model.predict_proba(features)
    row_by_row = pd.concat([model.predict_proba(features.iloc[[i]]) for i in range(len(features))])

    pd.testing.assert_series_equal(batch, row_by_row)


def test_duplicated_rows_get_identical_probabilities() -> None:
    """Rows with identical evidence collapse to the same cached result."""
    model, target_column = _trained_model()
    features = model.materialized_frame.drop(columns=[target_column])

    single = features.iloc[[0]]
    repeated = pd.concat([single] * 5, ignore_index=True)

    probabilities = model.predict_proba(repeated)

    assert probabilities.nunique() == 1
