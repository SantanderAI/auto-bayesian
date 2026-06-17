# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from auto_bayesian import build_project, fit_tables
from auto_bayesian.preprocess import (
    DataPreprocessor,
    _cramers_v,
    _find_correlated_columns,
    _find_low_variance_columns,
    remove_outliers,
)
from auto_bayesian.schema import PreprocessSpec, parse_project, validate_project


def _make_leads_with_outliers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lead_id": list(range(1, 13)),
            "age": [24, 45, 31, 52, 29, 48, 33, 55, 27, 50, 200, 300],
            "score": [10, 20, 15, 25, 12, 22, 18, 28, 11, 21, 999, 1000],
            "source": [
                "web",
                "referral",
                "email",
                "partner",
                "web",
                "referral",
                "email",
                "partner",
                "web",
                "referral",
                "web",
                "email",
            ],
            "converted": [
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
            ],
        }
    )


# ── Outlier Removal ─────────────────────────────────────────────────


def test_remove_outliers_iqr_drops_extreme_rows() -> None:
    frame = _make_leads_with_outliers()
    spec = PreprocessSpec(outlier_method="iqr", outlier_iqr_factor=1.5)
    cleaned = remove_outliers(frame, spec, target_column="converted")
    assert len(cleaned) < len(frame)
    assert cleaned["age"].max() < 200
    assert cleaned["score"].max() < 999


def test_remove_outliers_noop_when_disabled() -> None:
    frame = _make_leads_with_outliers()
    spec = PreprocessSpec()
    cleaned = remove_outliers(frame, spec, target_column="converted")
    assert len(cleaned) == len(frame)


def test_remove_outliers_preserves_non_numeric() -> None:
    frame = _make_leads_with_outliers()
    spec = PreprocessSpec(outlier_method="iqr", outlier_iqr_factor=1.5)
    cleaned = remove_outliers(frame, spec, target_column="converted")
    assert "source" in cleaned.columns
    assert cleaned["source"].notna().all()


def test_remove_outliers_tight_factor() -> None:
    frame = _make_leads_with_outliers()
    spec_loose = PreprocessSpec(outlier_method="iqr", outlier_iqr_factor=3.0)
    spec_tight = PreprocessSpec(outlier_method="iqr", outlier_iqr_factor=0.5)
    cleaned_loose = remove_outliers(frame, spec_loose, target_column="converted")
    cleaned_tight = remove_outliers(frame, spec_tight, target_column="converted")
    assert len(cleaned_tight) <= len(cleaned_loose)


# ── Low-Variance Filtering ──────────────────────────────────────────


def test_low_variance_drops_near_constant() -> None:
    frame = pd.DataFrame(
        {
            "constant_col": ["A"] * 99 + ["B"],
            "varied_col": list(range(100)),
            "target": ["0", "1"] * 50,
        }
    )
    dropped = _find_low_variance_columns(frame, ["constant_col", "varied_col"], 0.95)
    assert "constant_col" in dropped
    assert "varied_col" not in dropped


def test_low_variance_noop_when_disabled_via_preprocessor() -> None:
    frame = pd.DataFrame(
        {
            "constant_col": ["A"] * 10,
            "varied_col": list(range(10)),
            "converted": ["0", "1"] * 5,
        }
    )
    spec = PreprocessSpec(min_variance_fraction=0.0)
    preprocessor = DataPreprocessor(spec, "converted", "1")
    preprocessor.fit_transform(frame)
    assert "constant_col" in preprocessor.feature_columns
    assert preprocessor.dropped_columns == []


def test_low_variance_integrated_in_fit_transform() -> None:
    frame = pd.DataFrame(
        {
            "constant_col": ["A"] * 10,
            "age": [24, 45, 31, 52, 29, 48, 33, 55, 27, 50],
            "source": [
                "web",
                "referral",
                "email",
                "partner",
                "web",
                "referral",
                "email",
                "partner",
                "web",
                "referral",
            ],
            "converted": ["0", "1", "0", "1", "0", "1", "0", "1", "0", "1"],
        }
    )
    spec = PreprocessSpec(min_variance_fraction=0.95)
    preprocessor = DataPreprocessor(spec, "converted", "1")
    result = preprocessor.fit_transform(frame)
    assert "constant_col" not in result.columns
    assert "constant_col" not in preprocessor.feature_columns
    assert "constant_col" in preprocessor.dropped_columns


# ── Duplicate Removal ────────────────────────────────────────────────


def test_drop_duplicates_in_pipeline(tmp_path: Path) -> None:
    leads = pd.DataFrame(
        {
            "lead_id": list(range(1, 15)),
            "age": [24, 45, 31, 52, 29, 48, 33, 55, 24, 45, 31, 52, 29, 48],
            "source": [
                "web",
                "referral",
                "email",
                "partner",
                "web",
                "referral",
                "email",
                "partner",
                "web",
                "referral",
                "email",
                "partner",
                "web",
                "referral",
            ],
            "converted": [
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
                "0",
                "1",
            ],
        }
    )
    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        output_dir=tmp_path / "artifacts",
        test_fraction=0.3,
        drop_duplicates=True,
        tables=[{"name": "leads", "primary_key": "lead_id"}],
        relations=[],
    )
    model = fit_tables(project, {"leads": leads})
    assert model.report.selected_candidate in {"naive_bayes", "tan", "hill_climb"}


# ── Correlation Pruning ──────────────────────────────────────────────


def test_cramers_v_identical_columns() -> None:
    x = pd.Series(["a", "b", "c", "a", "b", "c"] * 10)
    v = _cramers_v(x, x)
    assert v >= 0.99


def test_cramers_v_independent_columns() -> None:
    import random

    rng = random.Random(42)
    x = pd.Series(rng.choices(["a", "b"], k=200))
    y = pd.Series(rng.choices(["x", "y"], k=200))
    v = _cramers_v(x, y)
    assert v < 0.3


def test_correlated_columns_drops_one() -> None:
    n = 100
    col_a = [str(i % 5) for i in range(n)]
    col_b = list(col_a)
    col_c = [str(i % 3) for i in range(n)]
    target = ["0", "1"] * (n // 2)
    frame = pd.DataFrame({"a": col_a, "b": col_b, "c": col_c, "target": target})
    dropped = _find_correlated_columns(frame, ["a", "b", "c"], "target", 0.95)
    assert len(dropped) == 1
    assert dropped[0] in ("a", "b")


def test_correlation_noop_when_max_is_one() -> None:
    frame = pd.DataFrame({"a": ["x", "y"] * 50, "b": ["x", "y"] * 50, "target": ["0", "1"] * 50})
    dropped = _find_correlated_columns(frame, ["a", "b"], "target", 1.0)
    assert dropped == []


def test_correlation_drops_when_threshold_exceeded() -> None:
    frame = pd.DataFrame({"a": ["x", "y"] * 50, "b": ["x", "y"] * 50, "target": ["0", "1"] * 50})
    dropped = _find_correlated_columns(frame, ["a", "b"], "target", 0.95)
    assert len(dropped) == 1


def test_correlation_integrated_in_fit_transform() -> None:
    n = 100
    frame = pd.DataFrame(
        {
            "a": [str(i % 5) for i in range(n)],
            "b": [str(i % 5) for i in range(n)],
            "c": [str(i % 3) for i in range(n)],
            "converted": ["0", "1"] * (n // 2),
        }
    )
    spec = PreprocessSpec(max_correlation=0.95)
    preprocessor = DataPreprocessor(spec, "converted", "1")
    preprocessor.fit_transform(frame)
    assert len(preprocessor.feature_columns) == 2
    assert len(preprocessor.dropped_columns) == 1


# ── Validation ───────────────────────────────────────────────────────


def test_unsupported_outlier_method_rejected() -> None:
    raw = {
        "task": {"root_table": "leads", "target_column": "converted"},
        "tables": [
            {"name": "leads", "path": "root.csv", "primary_key": "id"},
        ],
        "preprocess": {"outlier_method": "zscore"},
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="Unsupported outlier_method"):
        validate_project(project)


def test_negative_iqr_factor_rejected() -> None:
    raw = {
        "task": {"root_table": "leads", "target_column": "converted"},
        "tables": [
            {"name": "leads", "path": "root.csv", "primary_key": "id"},
        ],
        "preprocess": {"outlier_iqr_factor": -1.0},
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="outlier_iqr_factor must be positive"):
        validate_project(project)


def test_invalid_min_variance_rejected() -> None:
    raw = {
        "task": {"root_table": "leads", "target_column": "converted"},
        "tables": [
            {"name": "leads", "path": "root.csv", "primary_key": "id"},
        ],
        "preprocess": {"min_variance_fraction": 1.0},
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="min_variance_fraction"):
        validate_project(project)


def test_invalid_max_correlation_rejected() -> None:
    raw = {
        "task": {"root_table": "leads", "target_column": "converted"},
        "tables": [
            {"name": "leads", "path": "root.csv", "primary_key": "id"},
        ],
        "preprocess": {"max_correlation": 0.0},
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="max_correlation"):
        validate_project(project)


# ── Full Pipeline Integration ────────────────────────────────────────


def test_full_pipeline_with_all_cleaning(tmp_path: Path) -> None:
    leads = _make_leads_with_outliers()
    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        output_dir=tmp_path / "artifacts",
        test_fraction=0.3,
        outlier_method="iqr",
        outlier_iqr_factor=1.5,
        drop_duplicates=True,
        min_variance_fraction=0.95,
        tables=[{"name": "leads", "primary_key": "lead_id"}],
        relations=[],
    )
    model = fit_tables(project, {"leads": leads})
    assert model.report.selected_candidate in {"naive_bayes", "tan", "hill_climb"}
    assert 0.0 <= model.report.roc_auc <= 1.0

    features = model.materialized_frame.drop(columns=["converted"])
    probs = model.predict_proba(features)
    assert len(probs) == len(leads)
    assert probs.between(0, 1).all()
