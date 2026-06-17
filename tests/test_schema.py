# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from auto_bayesian.schema import load_project, parse_project, validate_project

EXAMPLE_CONFIG = Path("examples/lead_scoring.toml")


def test_load_project_reads_example_config() -> None:
    project = load_project(EXAMPLE_CONFIG)

    assert project.task.root_table == "leads"
    assert set(project.tables) == {"customers", "interactions", "leads"}
    assert len(project.relations) == 2


def test_validate_project_rejects_multiple_parents() -> None:
    raw = {
        "task": {"root_table": "root", "target_column": "target"},
        "tables": [
            {"name": "root", "path": "root.csv", "primary_key": "id"},
            {"name": "left", "path": "left.csv", "primary_key": "id"},
            {"name": "right", "path": "right.csv", "primary_key": "id"},
            {"name": "child", "path": "child.csv", "primary_key": "id"},
        ],
        "relations": [
            {
                "parent": "root",
                "child": "left",
                "parent_key": "id",
                "child_key": "id",
                "kind": "one_to_one",
            },
            {
                "parent": "left",
                "child": "child",
                "parent_key": "id",
                "child_key": "id",
                "kind": "one_to_one",
            },
            {
                "parent": "right",
                "child": "child",
                "parent_key": "id",
                "child_key": "id",
                "kind": "one_to_one",
            },
        ],
    }

    project = parse_project(Path("tests"), raw)
    for table in project.tables.values():
        table.path.write_text("id,target\n1,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="more than one parent"):
        validate_project(project)
