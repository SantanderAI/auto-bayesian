# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from auto_bayesian import build_project, fit_tables, materialize_project
from auto_bayesian.schema import parse_project, validate_project

EXAMPLE_ROOT = Path("examples")


def _make_leads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lead_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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
            "region": [
                "north",
                "west",
                "east",
                "south",
                "north",
                "west",
                "east",
                "south",
                "north",
                "west",
            ],
            "converted": ["0", "1", "0", "1", "0", "1", "0", "1", "0", "1"],
        }
    )


def _make_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "interaction_id": list(range(1, 21)),
            "lead_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
            "event_time": [
                "2026-01-02",
                "2026-01-08",
                "2026-01-02",
                "2026-01-05",
                "2026-01-04",
                "2026-01-09",
                "2026-01-03",
                "2026-01-06",
                "2026-01-02",
                "2026-01-07",
                "2026-01-02",
                "2026-01-05",
                "2026-01-03",
                "2026-01-08",
                "2026-01-02",
                "2026-01-04",
                "2026-01-02",
                "2026-01-09",
                "2026-01-02",
                "2026-01-06",
            ],
            "channel": [
                "email",
                "web",
                "phone",
                "meeting",
                "email",
                "email",
                "phone",
                "meeting",
                "web",
                "email",
                "phone",
                "meeting",
                "email",
                "web",
                "meeting",
                "phone",
                "web",
                "email",
                "phone",
                "meeting",
            ],
            "amount": [
                10,
                20,
                30,
                40,
                15,
                25,
                35,
                45,
                12,
                22,
                32,
                42,
                14,
                24,
                34,
                44,
                11,
                21,
                31,
                41,
            ],
        }
    )


# ── Sequence Features ────────────────────────────────────────────────


def test_sequence_features_are_materialized() -> None:
    leads = _make_leads()
    interactions = _make_interactions()

    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        tables=[
            {"name": "leads", "primary_key": "lead_id"},
            {
                "name": "interactions",
                "primary_key": "interaction_id",
                "timestamp_column": "event_time",
            },
        ],
        relations=[
            {
                "parent": "leads",
                "child": "interactions",
                "parent_key": "lead_id",
                "child_key": "lead_id",
                "kind": "one_to_many",
                "aggregations": [{"op": "count", "name": "interaction_count"}],
                "sequence_features": [
                    "recency",
                    "frequency",
                    "time_span",
                    "gap_mean",
                    "gap_std",
                    "acceleration",
                ],
            },
        ],
    )

    frame = materialize_project(project, tables={"leads": leads, "interactions": interactions})

    assert "interactions__recency" in frame.columns
    assert "interactions__frequency" in frame.columns
    assert "interactions__time_span" in frame.columns
    assert "interactions__gap_mean" in frame.columns
    assert "interactions__gap_std" in frame.columns
    assert "interactions__acceleration" in frame.columns
    assert frame.shape[0] == 10
    assert frame["interactions__recency"].notna().all()
    assert (frame["interactions__time_span"] >= 0).all()


def test_sequence_features_only_no_aggregations() -> None:
    leads = _make_leads()
    interactions = _make_interactions()

    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        tables=[
            {"name": "leads", "primary_key": "lead_id"},
            {
                "name": "interactions",
                "primary_key": "interaction_id",
                "timestamp_column": "event_time",
            },
        ],
        relations=[
            {
                "parent": "leads",
                "child": "interactions",
                "parent_key": "lead_id",
                "child_key": "lead_id",
                "kind": "one_to_many",
                "sequence_features": ["recency", "frequency"],
            },
        ],
    )

    frame = materialize_project(project, tables={"leads": leads, "interactions": interactions})

    assert "interactions__recency" in frame.columns
    assert "interactions__frequency" in frame.columns
    assert frame.shape[0] == 10


def test_sequence_features_require_timestamp_column() -> None:
    with pytest.raises(ValueError, match="timestamp_column"):
        build_project(
            root_table="leads",
            target_column="converted",
            tables=[
                {"name": "leads", "primary_key": "lead_id"},
                {"name": "interactions", "primary_key": "interaction_id"},
            ],
            relations=[
                {
                    "parent": "leads",
                    "child": "interactions",
                    "parent_key": "lead_id",
                    "child_key": "lead_id",
                    "kind": "one_to_many",
                    "sequence_features": ["recency"],
                },
            ],
        )


def test_unsupported_sequence_feature_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported sequence feature"):
        build_project(
            root_table="leads",
            target_column="converted",
            tables=[
                {"name": "leads", "primary_key": "lead_id"},
                {
                    "name": "interactions",
                    "primary_key": "interaction_id",
                    "timestamp_column": "event_time",
                },
            ],
            relations=[
                {
                    "parent": "leads",
                    "child": "interactions",
                    "parent_key": "lead_id",
                    "child_key": "lead_id",
                    "kind": "one_to_many",
                    "sequence_features": ["nonexistent_feature"],
                },
            ],
        )


# ── Windowed Aggregations ────────────────────────────────────────────


def test_windowed_aggregation() -> None:
    leads = _make_leads()
    interactions = _make_interactions()

    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        tables=[
            {"name": "leads", "primary_key": "lead_id"},
            {
                "name": "interactions",
                "primary_key": "interaction_id",
                "timestamp_column": "event_time",
            },
        ],
        relations=[
            {
                "parent": "leads",
                "child": "interactions",
                "parent_key": "lead_id",
                "child_key": "lead_id",
                "kind": "one_to_many",
                "aggregations": [
                    {"op": "count", "name": "total_count"},
                    {"op": "count", "window_days": 3, "name": "count_last_3d"},
                    {
                        "column": "amount",
                        "op": "sum",
                        "window_days": 5,
                        "name": "sum_last_5d",
                    },
                ],
            },
        ],
    )

    frame = materialize_project(project, tables={"leads": leads, "interactions": interactions})

    assert "total_count" in frame.columns
    assert "count_last_3d" in frame.columns
    assert "sum_last_5d" in frame.columns
    assert frame.shape[0] == 10
    assert (frame["count_last_3d"] <= frame["total_count"]).all()


def test_windowed_aggregation_requires_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp_column"):
        build_project(
            root_table="leads",
            target_column="converted",
            tables=[
                {"name": "leads", "primary_key": "lead_id"},
                {"name": "interactions", "primary_key": "interaction_id"},
            ],
            relations=[
                {
                    "parent": "leads",
                    "child": "interactions",
                    "parent_key": "lead_id",
                    "child_key": "lead_id",
                    "kind": "one_to_many",
                    "aggregations": [
                        {"op": "count", "window_days": 7, "name": "count_7d"},
                    ],
                },
            ],
        )


def test_negative_window_days_rejected() -> None:
    with pytest.raises(ValueError, match="window_days must be positive"):
        build_project(
            root_table="leads",
            target_column="converted",
            tables=[
                {"name": "leads", "primary_key": "lead_id"},
                {
                    "name": "interactions",
                    "primary_key": "interaction_id",
                    "timestamp_column": "event_time",
                },
            ],
            relations=[
                {
                    "parent": "leads",
                    "child": "interactions",
                    "parent_key": "lead_id",
                    "child_key": "lead_id",
                    "kind": "one_to_many",
                    "aggregations": [
                        {"op": "count", "window_days": -1, "name": "bad"},
                    ],
                },
            ],
        )


# ── Next Best Action ─────────────────────────────────────────────────


def test_next_best_action_schema_validation() -> None:
    raw = {
        "task": {
            "root_table": "leads",
            "target_column": "converted",
            "task_type": "next_best_action",
        },
        "tables": [
            {"name": "leads", "path": "root.csv", "primary_key": "id"},
        ],
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="action_column"):
        validate_project(project)


def test_nba_requires_action_column_in_task() -> None:
    raw = {
        "task": {
            "root_table": "leads",
            "target_column": "converted",
            "task_type": "next_best_action",
        },
        "tables": [
            {"name": "leads", "path": "root.csv", "primary_key": "id"},
        ],
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="action_column"):
        validate_project(project)


def test_unsupported_task_type_rejected() -> None:
    raw = {
        "task": {
            "root_table": "leads",
            "target_column": "converted",
            "task_type": "regression",
        },
        "tables": [
            {"name": "leads", "path": "root.csv", "primary_key": "id"},
        ],
    }
    project = parse_project(Path("tests"), raw)
    with pytest.raises(ValueError, match="Unsupported task_type"):
        validate_project(project)


def test_next_best_action_predict(tmp_path: Path) -> None:
    leads = _make_leads()

    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        task_type="next_best_action",
        action_column="source",
        output_dir=tmp_path / "artifacts",
        test_fraction=0.3,
        tables=[{"name": "leads", "primary_key": "lead_id"}],
        relations=[],
    )

    model = fit_tables(project, {"leads": leads})

    assert model.report.task_type == "next_best_action"
    assert model.report.action_column == "source"

    features = model.materialized_frame.drop(columns=["converted"])
    result = model.predict_next_best_action(features)

    assert "recommended_action" in result.columns
    assert "expected_probability" in result.columns
    assert len(result) == len(features)
    assert result["expected_probability"].between(0, 1).all()

    model.save(tmp_path / "artifacts")
    from auto_bayesian.model import AutoBayesModel

    restored = AutoBayesModel.load(tmp_path / "artifacts")
    assert restored.report.task_type == "next_best_action"
    assert restored.report.action_column == "source"


def test_predict_next_best_action_raises_without_action_column() -> None:
    leads = _make_leads()

    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        tables=[{"name": "leads", "primary_key": "lead_id"}],
        relations=[],
    )

    model = fit_tables(project, {"leads": leads})

    features = model.materialized_frame.drop(columns=["converted"])
    with pytest.raises(ValueError, match="predict_next_best_action"):
        model.predict_next_best_action(features)


# ── Integration: sequence features + training ────────────────────────


def test_full_pipeline_with_sequence_features(tmp_path: Path) -> None:
    leads = _make_leads()
    interactions = _make_interactions()

    project = build_project(
        root_table="leads",
        target_column="converted",
        positive_label="1",
        output_dir=tmp_path / "artifacts",
        test_fraction=0.3,
        tables=[
            {"name": "leads", "primary_key": "lead_id"},
            {
                "name": "interactions",
                "primary_key": "interaction_id",
                "timestamp_column": "event_time",
            },
        ],
        relations=[
            {
                "parent": "leads",
                "child": "interactions",
                "parent_key": "lead_id",
                "child_key": "lead_id",
                "kind": "one_to_many",
                "aggregations": [{"op": "count", "name": "interaction_count"}],
                "sequence_features": ["recency", "frequency"],
            },
        ],
    )

    model = fit_tables(project, {"leads": leads, "interactions": interactions})
    assert model.report.selected_candidate in {"naive_bayes", "tan", "hill_climb"}
    assert 0.0 <= model.report.roc_auc <= 1.0

    features = model.materialized_frame.drop(columns=["converted"])
    probs = model.predict_proba(features)
    assert len(probs) == 10
    assert probs.between(0, 1).all()
