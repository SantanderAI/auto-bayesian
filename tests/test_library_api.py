# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd

from auto_bayesian import build_project, fit_tables, materialize_project
from auto_bayesian.model import AutoBayesModel

EXAMPLE_ROOT = Path("examples")


def test_build_project_and_fit_tables_support_in_memory_data(tmp_path: Path) -> None:
    tables = {
        "leads": pd.read_csv(EXAMPLE_ROOT / "data" / "leads.csv"),
        "customers": pd.read_csv(EXAMPLE_ROOT / "data" / "customers.csv"),
        "interactions": pd.read_csv(EXAMPLE_ROOT / "data" / "interactions.csv"),
    }
    project = build_project(
        root=EXAMPLE_ROOT,
        root_table="leads",
        target_column="converted",
        positive_label="1",
        output_dir=tmp_path / "artifacts",
        test_fraction=0.25,
        numeric_bins=4,
        max_categories=10,
        tables=[
            {"name": "leads", "primary_key": "lead_id"},
            {"name": "customers", "primary_key": "customer_id"},
            {
                "name": "interactions",
                "primary_key": "interaction_id",
                "timestamp_column": "event_time",
            },
        ],
        relations=[
            {
                "parent": "leads",
                "child": "customers",
                "parent_key": "customer_id",
                "child_key": "customer_id",
                "kind": "one_to_one",
            },
            {
                "parent": "leads",
                "child": "interactions",
                "parent_key": "lead_id",
                "child_key": "lead_id",
                "kind": "one_to_many",
                "aggregations": [
                    {"op": "count", "name": "interaction_count"},
                    {"column": "channel", "op": "nunique", "name": "channel_count"},
                    {
                        "column": "days_to_close",
                        "op": "mean",
                        "name": "mean_days_to_close",
                    },
                    {"column": "channel", "op": "latest", "name": "latest_channel"},
                ],
            },
        ],
    )

    materialized = materialize_project(project, tables=tables)
    model = fit_tables(project, tables)
    model.save(project.run.output_dir)
    restored = AutoBayesModel.load(project.run.output_dir)

    assert "interaction_count" in materialized.columns
    assert restored.report.selected_candidate in {
        candidate.name for candidate in restored.report.candidates
    }
    assert len(restored.report.candidates) == 3
