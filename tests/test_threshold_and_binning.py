# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from auto_bayesian import build_project, fit_tables
from auto_bayesian.model import AutoBayesModel
from auto_bayesian.preprocess import DataPreprocessor, _supervised_edges
from auto_bayesian.schema import PreprocessSpec, parse_project, validate_project

# ── Supervised binning ───────────────────────────────────────────────


def test_supervised_edges_splits_separable_data() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0])
    target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    edges = _supervised_edges(values, target, max_bins=4)
    assert edges is not None
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    interior = [edge for edge in edges if np.isfinite(edge)]
    assert any(3.0 < edge < 10.0 for edge in interior)


def test_supervised_edges_none_for_single_bin() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([0, 0, 1, 1])
    assert _supervised_edges(values, target, max_bins=1) is None


def test_supervised_binning_learns_numeric_edges() -> None:
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(0.0, 1.0, 160), rng.normal(6.0, 1.0, 40)])
    frame = pd.DataFrame({"x": values, "target": ["0"] * 160 + ["1"] * 40})
    spec = PreprocessSpec(numeric_bins=4, binning="supervised")
    preprocessor = DataPreprocessor(spec, "target", "1")
    preprocessor.fit_transform(frame)

    transform = preprocessor.transforms["x"]
    assert transform.kind == "numeric"
    assert transform.bins  # a target-aware split was found
    reapplied = preprocessor.transform(frame.drop(columns=["target"]))
    assert reapplied["x"].notna().all()


# ── Tuned threshold + persistence ────────────────────────────────────


def _imbalanced_leads(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.random(n)
    converted = (base > 0.85).astype(int)
    score = base * 10.0 + rng.normal(0.0, 0.5, n)
    age = rng.integers(20, 60, n)
    return pd.DataFrame(
        {
            "lead_id": list(range(n)),
            "score": score,
            "age": age,
            "converted": converted.astype(str),
        }
    )


def test_pipeline_tunes_threshold_and_persists(tmp_path: Path) -> None:
    leads = _imbalanced_leads()
    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        output_dir=tmp_path / "artifacts",
        test_fraction=0.3,
        numeric_bins=4,
        binning="supervised",
        selection_metric="pr_auc",
        tables=[{"name": "leads", "primary_key": "lead_id"}],
        relations=[],
    )

    model = fit_tables(project, {"leads": leads})
    assert 0.0 <= model.report.roc_auc <= 1.0
    assert 0.0 <= model.report.pr_auc <= 1.0
    assert 0.0 <= model.report.threshold <= 1.0

    model.save(project.run.output_dir)
    restored = AutoBayesModel.load(project.run.output_dir)
    assert restored.report.threshold == model.report.threshold
    assert restored.report.f1 == model.report.f1
    assert restored.report.pr_auc == model.report.pr_auc


def test_evaluate_uses_tuned_threshold(tmp_path: Path) -> None:
    leads = _imbalanced_leads()
    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        output_dir=tmp_path / "artifacts",
        test_fraction=0.3,
        selection_metric="pr_auc",
        tables=[{"name": "leads", "primary_key": "lead_id"}],
        relations=[],
    )
    model = fit_tables(project, {"leads": leads})

    metrics = model.evaluate(model.materialized_frame)
    assert metrics.threshold == model.report.threshold
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_evaluate_requires_target_column() -> None:
    leads = _imbalanced_leads()
    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        tables=[{"name": "leads", "primary_key": "lead_id"}],
        relations=[],
    )
    model = fit_tables(project, {"leads": leads})
    features = model.materialized_frame.drop(columns=["converted"])
    with pytest.raises(ValueError, match="requires the target column"):
        model.evaluate(features)


# ── Schema validation for the new knobs ──────────────────────────────


def test_invalid_binning_rejected() -> None:
    raw = {
        "task": {"root_table": "leads", "target_column": "converted"},
        "tables": [{"name": "leads", "path": "root.csv", "primary_key": "id"}],
        "preprocess": {"binning": "magic"},
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="Unsupported binning"):
        validate_project(project)


def test_invalid_selection_metric_rejected() -> None:
    raw = {
        "task": {"root_table": "leads", "target_column": "converted"},
        "tables": [{"name": "leads", "path": "root.csv", "primary_key": "id"}],
        "run": {"selection_metric": "accuracy"},
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="Unsupported selection_metric"):
        validate_project(project)
