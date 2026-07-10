# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from auto_bayesian.materialize import materialize_project
from auto_bayesian.schema import load_project

EXAMPLE_CONFIG = Path("examples/lead_scoring.toml")


def test_materialize_project_joins_and_aggregates() -> None:
    project = load_project(EXAMPLE_CONFIG)
    frame = materialize_project(project)

    assert frame.shape[0] == 10
    assert "customers__region" in frame.columns
    assert "interaction_count" in frame.columns
    assert "latest_channel" in frame.columns

    lead_2 = frame.loc[frame["lead_id"] == 2].iloc[0]
    assert lead_2["interaction_count"] == 2
    assert lead_2["channel_count"] == 2
    assert lead_2["latest_channel"] == "meeting"


def test_one_to_many_count_is_zero_for_parents_without_children(tmp_path: Path) -> None:
    import pandas as pd

    from auto_bayesian import build_project

    parents = pd.DataFrame(
        {"tender_id": ["T1", "T2", "T3"], "won": ["1", "0", "0"]}
    )
    children = pd.DataFrame(
        {
            "bid_id": ["B1", "B2", "B3"],
            "tender_id": ["T1", "T1", "T2"],
            "bidder": ["a", "b", "a"],
            "price": [0.9, 1.1, 0.95],
        }
    )
    project = build_project(
        root=tmp_path,
        root_table="tenders",
        target_column="won",
        positive_label="1",
        output_dir=tmp_path / "artifacts",
        tables=[
            {"name": "tenders", "primary_key": "tender_id"},
            {"name": "bids", "primary_key": "bid_id"},
        ],
        relations=[
            {
                "parent": "tenders",
                "child": "bids",
                "parent_key": "tender_id",
                "child_key": "tender_id",
                "kind": "one_to_many",
                "aggregations": [
                    {"op": "count", "name": "bid_count"},
                    {"column": "bidder", "op": "nunique", "name": "bidder_count"},
                    {"column": "price", "op": "mean", "name": "mean_price"},
                ],
            }
        ],
    )
    frame = materialize_project(project, tables={"tenders": parents, "bids": children})
    childless = frame.loc[frame["tender_id"] == "T3"].iloc[0]

    assert childless["bid_count"] == 0
    assert childless["bidder_count"] == 0
    assert pd.isna(childless["mean_price"])
    assert frame.loc[frame["tender_id"] == "T1"].iloc[0]["bid_count"] == 2