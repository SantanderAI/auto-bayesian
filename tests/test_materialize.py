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
