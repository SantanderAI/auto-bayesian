# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

SUPPORTED_RELATION_KINDS = {"one_to_one", "one_to_many"}
SUPPORTED_AGGREGATIONS = {"count", "nunique", "sum", "mean", "min", "max", "latest"}
SUPPORTED_SEQUENCE_FEATURES = {
    "recency",
    "frequency",
    "time_span",
    "gap_mean",
    "gap_std",
    "acceleration",
}
SUPPORTED_TASK_TYPES = {"classification", "next_best_action"}
SUPPORTED_BINNING = {"quantile", "supervised"}
SUPPORTED_SELECTION_METRICS = {"roc_auc", "pr_auc"}


@dataclass(slots=True)
class AggregateSpec:
    op: str
    column: str | None = None
    name: str | None = None
    window_days: int | None = None

    def output_name(self, child: str) -> str:
        if self.name:
            return self.name
        base = self.column or "rows"
        suffix = f"_last_{self.window_days}d" if self.window_days else ""
        return f"{child}__{base}__{self.op}{suffix}"


@dataclass(slots=True)
class TableSpec:
    name: str
    path: Path | None
    primary_key: list[str]
    timestamp_column: str | None = None


@dataclass(slots=True)
class RelationSpec:
    parent: str
    child: str
    parent_key: list[str]
    child_key: list[str]
    kind: str
    aggregations: list[AggregateSpec] = field(default_factory=list)
    sequence_features: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaskSpec:
    root_table: str
    target_column: str
    positive_label: str = "1"
    task_type: str = "classification"
    action_column: str | None = None


SUPPORTED_OUTLIER_METHODS = {"iqr"}


@dataclass(slots=True)
class PreprocessSpec:
    numeric_bins: int = 5
    max_categories: int = 20
    outlier_method: str | None = None
    outlier_iqr_factor: float = 1.5
    drop_duplicates: bool = False
    min_variance_fraction: float = 0.0
    max_correlation: float = 1.0
    binning: str = "quantile"


@dataclass(slots=True)
class RunSpec:
    output_dir: Path
    random_seed: int = 7
    test_fraction: float = 0.2
    selection_metric: str = "roc_auc"


@dataclass(slots=True)
class ProjectSpec:
    root: Path
    tables: dict[str, TableSpec]
    relations: list[RelationSpec]
    task: TaskSpec
    preprocess: PreprocessSpec
    run: RunSpec

    def relation_map(self) -> dict[str, list[RelationSpec]]:
        mapping: dict[str, list[RelationSpec]] = {}
        for relation in self.relations:
            mapping.setdefault(relation.parent, []).append(relation)
        return mapping


def load_project(path: str | Path) -> ProjectSpec:
    """Load and validate a project from a TOML config file.

    Table and output paths in the config are resolved relative to the config
    file's directory.
    """
    config_path = Path(path).resolve()
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    project = parse_project(config_path.parent, data)
    validate_project(project)
    return project


def build_project(
    *,
    root_table: str,
    target_column: str,
    tables: list[TableSpec | Mapping[str, object]],
    relations: list[RelationSpec | Mapping[str, object]] | None = None,
    positive_label: str = "1",
    task_type: str = "classification",
    action_column: str | None = None,
    output_dir: str | Path = "artifacts",
    random_seed: int = 7,
    test_fraction: float = 0.2,
    numeric_bins: int = 5,
    max_categories: int = 20,
    outlier_method: str | None = None,
    outlier_iqr_factor: float = 1.5,
    drop_duplicates: bool = False,
    min_variance_fraction: float = 0.0,
    max_correlation: float = 1.0,
    binning: str = "quantile",
    selection_metric: str = "roc_auc",
    root: str | Path = ".",
) -> ProjectSpec:
    """Build and validate a :class:`ProjectSpec` directly in Python.

    Mirrors the TOML config as keyword arguments so you can describe tables,
    relations, the task, and preprocessing options without writing a file. Tables
    and relations accept either dataclass instances or plain mappings.
    """
    root_path = Path(root).resolve()
    built_table_specs = [_coerce_table_spec(root_path, raw_table) for raw_table in tables]
    built_tables = {table.name: table for table in built_table_specs}
    built_relations = [_coerce_relation_spec(raw_relation) for raw_relation in (relations or [])]
    project = ProjectSpec(
        root=root_path,
        tables=built_tables,
        relations=built_relations,
        task=TaskSpec(
            root_table=root_table,
            task_type=str(task_type),
            action_column=_optional_str(action_column),
            target_column=target_column,
            positive_label=str(positive_label),
        ),
        preprocess=PreprocessSpec(
            numeric_bins=int(numeric_bins),
            max_categories=int(max_categories),
            outlier_method=outlier_method,
            outlier_iqr_factor=float(outlier_iqr_factor),
            drop_duplicates=bool(drop_duplicates),
            min_variance_fraction=float(min_variance_fraction),
            max_correlation=float(max_correlation),
            binning=str(binning),
        ),
        run=RunSpec(
            output_dir=_resolve_path(root_path, output_dir),
            random_seed=int(random_seed),
            test_fraction=float(test_fraction),
            selection_metric=str(selection_metric),
        ),
    )
    available_tables = {name for name, table in project.tables.items() if table.path is None}
    validate_project(project, available_tables=available_tables)
    return project


def parse_project(root: Path, data: dict) -> ProjectSpec:
    tables = {}
    for raw_table in data.get("tables", []):
        table = TableSpec(
            name=raw_table["name"],
            path=_resolve_path(root, raw_table["path"]),
            primary_key=_as_list(raw_table["primary_key"]),
            timestamp_column=raw_table.get("timestamp_column"),
        )
        tables[table.name] = table

    relations = []
    for raw_relation in data.get("relations", []):
        relation = RelationSpec(
            parent=raw_relation["parent"],
            child=raw_relation["child"],
            parent_key=_as_list(raw_relation["parent_key"]),
            child_key=_as_list(raw_relation["child_key"]),
            kind=raw_relation.get("kind", "one_to_many"),
            aggregations=[
                AggregateSpec(
                    op=aggregate["op"],
                    column=aggregate.get("column"),
                    name=aggregate.get("name"),
                    window_days=_optional_int(aggregate.get("window_days")),
                )
                for aggregate in raw_relation.get("aggregations", [])
            ],
            sequence_features=list(raw_relation.get("sequence_features", [])),
        )
        relations.append(relation)

    task_data = data["task"]
    run_data = data.get("run", {})
    preprocess_data = data.get("preprocess", {})
    return ProjectSpec(
        root=root,
        tables=tables,
        relations=relations,
        task=TaskSpec(
            root_table=task_data["root_table"],
            target_column=task_data["target_column"],
            positive_label=str(task_data.get("positive_label", "1")),
            task_type=str(task_data.get("task_type", "classification")),
            action_column=_optional_str(task_data.get("action_column")),
        ),
        preprocess=PreprocessSpec(
            numeric_bins=int(preprocess_data.get("numeric_bins", 5)),
            max_categories=int(preprocess_data.get("max_categories", 20)),
            outlier_method=_optional_str(preprocess_data.get("outlier_method")),
            outlier_iqr_factor=float(preprocess_data.get("outlier_iqr_factor", 1.5)),
            drop_duplicates=bool(preprocess_data.get("drop_duplicates", False)),
            min_variance_fraction=float(preprocess_data.get("min_variance_fraction", 0.0)),
            max_correlation=float(preprocess_data.get("max_correlation", 1.0)),
            binning=str(preprocess_data.get("binning", "quantile")),
        ),
        run=RunSpec(
            output_dir=(root / run_data.get("output_dir", "artifacts")).resolve(),
            random_seed=int(run_data.get("random_seed", 7)),
            test_fraction=float(run_data.get("test_fraction", 0.2)),
            selection_metric=str(run_data.get("selection_metric", "roc_auc")),
        ),
    )


def validate_project(project: ProjectSpec, available_tables: Collection[str] = ()) -> None:
    provided_tables = set(available_tables)
    if not project.tables:
        raise ValueError("Config must declare at least one table.")
    if project.task.root_table not in project.tables:
        raise ValueError(f"Unknown root table: {project.task.root_table}")
    if not 0 < project.run.test_fraction < 0.5:
        raise ValueError("run.test_fraction must be between 0 and 0.5.")
    if project.task.task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {project.task.task_type}")
    if project.task.task_type == "next_best_action" and not project.task.action_column:
        raise ValueError("task_type 'next_best_action' requires an action_column.")

    pp = project.preprocess
    if pp.outlier_method is not None and pp.outlier_method not in SUPPORTED_OUTLIER_METHODS:
        raise ValueError(f"Unsupported outlier_method: {pp.outlier_method}")
    if pp.outlier_iqr_factor <= 0:
        raise ValueError("outlier_iqr_factor must be positive.")
    if not 0.0 <= pp.min_variance_fraction < 1.0:
        raise ValueError("min_variance_fraction must be in [0.0, 1.0).")
    if not 0.0 < pp.max_correlation <= 1.0:
        raise ValueError("max_correlation must be in (0.0, 1.0].")
    if pp.binning not in SUPPORTED_BINNING:
        raise ValueError(f"Unsupported binning: {pp.binning}")
    if project.run.selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unsupported selection_metric: {project.run.selection_metric}")

    parents_by_child: dict[str, str] = {}
    for table in project.tables.values():
        if not table.primary_key:
            raise ValueError(f"Table {table.name} must declare a primary key.")

    for relation in project.relations:
        if relation.kind not in SUPPORTED_RELATION_KINDS:
            raise ValueError(f"Unsupported relation kind: {relation.kind}")
        if relation.parent not in project.tables:
            raise ValueError(f"Unknown relation parent: {relation.parent}")
        if relation.child not in project.tables:
            raise ValueError(f"Unknown relation child: {relation.child}")
        if len(relation.parent_key) != len(relation.child_key):
            raise ValueError(f"Relation {relation.parent}->{relation.child} uses mismatched keys.")
        if relation.child in parents_by_child:
            raise ValueError(f"Table {relation.child} has more than one parent relation.")
        parents_by_child[relation.child] = relation.parent
        if (
            relation.kind == "one_to_many"
            and not relation.aggregations
            and not relation.sequence_features
        ):
            raise ValueError(
                f"Relation {relation.parent}->{relation.child}"
                " needs aggregations or sequence_features."
            )
        for aggregate in relation.aggregations:
            if aggregate.op not in SUPPORTED_AGGREGATIONS:
                raise ValueError(f"Unsupported aggregation: {aggregate.op}")
            if aggregate.op != "count" and not aggregate.column:
                raise ValueError(f"Aggregation {aggregate.op} requires a column.")
            if aggregate.window_days is not None:
                if aggregate.window_days <= 0:
                    raise ValueError(f"window_days must be positive, got {aggregate.window_days}.")
                child_spec = project.tables[relation.child]
                if not child_spec.timestamp_column:
                    raise ValueError(
                        f"Table {relation.child} needs timestamp_column for windowed aggregation."
                    )
        for seq_feat in relation.sequence_features:
            if seq_feat not in SUPPORTED_SEQUENCE_FEATURES:
                raise ValueError(f"Unsupported sequence feature: {seq_feat}")
        if relation.sequence_features:
            child_spec = project.tables[relation.child]
            if not child_spec.timestamp_column:
                raise ValueError(
                    f"Table {relation.child} needs timestamp_column for sequence_features."
                )

    _validate_graph(project)

    for table in project.tables.values():
        if table.name in provided_tables:
            continue
        if table.path is None:
            raise ValueError(f"Table {table.name} needs a file path or an in-memory frame.")
        if not table.path.exists():
            raise ValueError(f"Missing table file: {table.path}")


def _validate_graph(project: ProjectSpec) -> None:
    children = project.relation_map()
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(name: str) -> None:
        if name in visiting:
            raise ValueError("Relation graph contains a cycle.")
        if name in visited:
            return
        visiting.add(name)
        for relation in children.get(name, []):
            walk(relation.child)
        visiting.remove(name)
        visited.add(name)

    walk(project.task.root_table)
    unseen = set(project.tables) - visited
    if unseen:
        names = ", ".join(sorted(unseen))
        raise ValueError(f"Tables are disconnected from the root graph: {names}")


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_table_spec(root: Path, raw_table: TableSpec | Mapping[str, object]) -> TableSpec:
    if isinstance(raw_table, TableSpec):
        path = raw_table.path
        resolved = _resolve_path(root, path) if path is not None else None
        return TableSpec(
            name=raw_table.name,
            path=resolved,
            primary_key=list(raw_table.primary_key),
            timestamp_column=raw_table.timestamp_column,
        )
    return TableSpec(
        name=str(raw_table["name"]),
        path=_resolve_optional_path(root, raw_table.get("path")),
        primary_key=_as_list(raw_table["primary_key"]),
        timestamp_column=_optional_str(raw_table.get("timestamp_column")),
    )


def _coerce_relation_spec(raw_relation: RelationSpec | Mapping[str, object]) -> RelationSpec:
    if isinstance(raw_relation, RelationSpec):
        return RelationSpec(
            parent=raw_relation.parent,
            child=raw_relation.child,
            parent_key=list(raw_relation.parent_key),
            child_key=list(raw_relation.child_key),
            kind=raw_relation.kind,
            aggregations=[
                AggregateSpec(
                    op=aggregate.op,
                    column=aggregate.column,
                    name=aggregate.name,
                    window_days=aggregate.window_days,
                )
                for aggregate in raw_relation.aggregations
            ],
            sequence_features=list(raw_relation.sequence_features),
        )
    return RelationSpec(
        parent=str(raw_relation["parent"]),
        child=str(raw_relation["child"]),
        parent_key=_as_list(raw_relation["parent_key"]),
        child_key=_as_list(raw_relation["child_key"]),
        kind=str(raw_relation.get("kind", "one_to_many")),
        aggregations=[
            AggregateSpec(
                op=str(aggregate["op"]),
                column=_optional_str(aggregate.get("column")),
                name=_optional_str(aggregate.get("name")),
                window_days=_optional_int(aggregate.get("window_days")),
            )
            for aggregate in cast(
                "list[Mapping[str, object]]", raw_relation.get("aggregations", [])
            )
        ],
        sequence_features=[
            str(item) for item in cast("list[object]", raw_relation.get("sequence_features", []))
        ],
    )


def _resolve_optional_path(root: Path, value: object) -> Path | None:
    if value is None:
        return None
    return _resolve_path(root, value)


def _resolve_path(root: Path, value: object) -> Path:
    return (root / Path(str(value))).resolve()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(f"Expected an int-like value, got {type(value).__name__}")
