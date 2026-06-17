# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque

import pandas as pd

from auto_bayesian.preprocess import MISSING_VALUE
from auto_bayesian.schema import ProjectSpec, RelationSpec, TableSpec, validate_project


def load_tables(
    project: ProjectSpec, provided_tables: dict[str, pd.DataFrame] | None = None
) -> dict[str, pd.DataFrame]:
    available_tables = set(provided_tables or {})
    validate_project(project, available_tables=available_tables)
    tables: dict[str, pd.DataFrame] = {}
    for spec in project.tables.values():
        if provided_tables and spec.name in provided_tables:
            frame = provided_tables[spec.name].copy()
        else:
            frame = _read_table(spec)
        missing = set(spec.primary_key) - set(frame.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Table {spec.name} is missing primary key columns: {names}")
        if frame.duplicated(spec.primary_key).any():
            raise ValueError(f"Table {spec.name} primary key is not unique.")
        tables[spec.name] = frame.copy()
    return tables


def materialize_project(
    project: ProjectSpec, tables: dict[str, pd.DataFrame] | None = None
) -> pd.DataFrame:
    prepared = load_tables(project, provided_tables=tables)
    order = _relations_bottom_up(project)

    for relation in order:
        child = prepared[relation.child]
        parent = prepared[relation.parent]
        if relation.kind == "one_to_one":
            prepared[relation.parent] = _join_one_to_one(
                parent, child, relation, project.tables[relation.child]
            )
            continue
        if relation.aggregations:
            aggregated = _aggregate_child(child, relation, project.tables[relation.child])
            parent = parent.merge(
                aggregated,
                how="left",
                left_on=relation.parent_key,
                right_on=relation.parent_key,
            )
        if relation.sequence_features:
            seq = _compute_sequence_features(child, relation, project.tables[relation.child])
            parent = parent.merge(
                seq,
                how="left",
                left_on=relation.parent_key,
                right_on=relation.parent_key,
            )
        prepared[relation.parent] = parent

    root = prepared[project.task.root_table].copy()
    if project.task.target_column not in root.columns:
        raise ValueError(
            f"Target column {project.task.target_column} is missing from root table "
            f"{project.task.root_table}."
        )
    if root[project.task.target_column].isna().any():
        raise ValueError("Target column contains missing values.")
    return root


def _read_table(spec: TableSpec) -> pd.DataFrame:
    if spec.path is None:
        raise ValueError(f"Table {spec.name} needs a file path or an in-memory frame.")
    suffix = spec.path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(spec.path)
    if suffix == ".parquet":
        return pd.read_parquet(spec.path)
    raise ValueError(f"Unsupported file type for {spec.path}")


def _relations_bottom_up(project: ProjectSpec) -> list[RelationSpec]:
    by_parent = project.relation_map()
    queue = deque([(project.task.root_table, 0)])
    depths = {project.task.root_table: 0}
    while queue:
        table, depth = queue.popleft()
        for relation in by_parent.get(table, []):
            depths[relation.child] = depth + 1
            queue.append((relation.child, depth + 1))
    return sorted(project.relations, key=lambda relation: depths[relation.child], reverse=True)


def _join_one_to_one(
    parent: pd.DataFrame,
    child: pd.DataFrame,
    relation: RelationSpec,
    child_spec: TableSpec,
) -> pd.DataFrame:
    if child.duplicated(relation.child_key).any():
        raise ValueError(f"Relation {relation.parent}->{relation.child} is not one-to-one.")

    rename_map = {}
    for column in child.columns:
        if column in relation.child_key:
            continue
        if column in child_spec.primary_key:
            continue
        rename_map[column] = f"{relation.child}__{column}"
    right = child.rename(columns=rename_map)
    merged = parent.merge(
        right,
        how="left",
        left_on=relation.parent_key,
        right_on=relation.child_key,
    )
    duplicate_keys = [
        child_key
        for parent_key, child_key in zip(relation.parent_key, relation.child_key, strict=True)
        if parent_key != child_key and child_key in merged.columns
    ]
    if duplicate_keys:
        merged = merged.drop(columns=duplicate_keys)
    return merged


def _aggregate_child(
    child: pd.DataFrame, relation: RelationSpec, child_spec: TableSpec
) -> pd.DataFrame:
    parts = []
    for aggregate in relation.aggregations:
        name = aggregate.output_name(relation.child)
        source = child
        if aggregate.window_days is not None and child_spec.timestamp_column:
            source = _filter_window(source, child_spec.timestamp_column, aggregate.window_days)
        group = source.groupby(relation.child_key, dropna=False)
        if aggregate.op == "count":
            series = group.size().rename(name)
        elif aggregate.op == "latest":
            if not child_spec.timestamp_column:
                raise ValueError(
                    f"Table {relation.child} needs timestamp_column for latest aggregation."
                )
            if aggregate.column not in source.columns:
                raise ValueError(
                    f"Table {relation.child} is missing aggregation column {aggregate.column}."
                )
            sorted_child = source.sort_values(child_spec.timestamp_column)
            latest = sorted_child.groupby(relation.child_key, dropna=False)[aggregate.column].last()
            series = latest.rename(name)
        else:
            if aggregate.column not in source.columns:
                raise ValueError(
                    f"Table {relation.child} is missing aggregation column {aggregate.column}."
                )
            series = getattr(group[aggregate.column], aggregate.op)().rename(name)
        parts.append(series)

    aggregated = pd.concat(parts, axis=1).reset_index()
    aggregated.columns = relation.parent_key + [
        column for column in aggregated.columns[len(relation.parent_key) :]
    ]

    for aggregate in relation.aggregations:
        column = aggregate.output_name(relation.child)
        if aggregate.op == "latest":
            aggregated[column] = aggregated[column].fillna(MISSING_VALUE)
        else:
            aggregated[column] = aggregated[column].fillna(0)
    return aggregated


def _filter_window(child: pd.DataFrame, timestamp_column: str, window_days: int) -> pd.DataFrame:
    timestamps = pd.to_datetime(child[timestamp_column], errors="coerce")
    ref_date = timestamps.max()
    cutoff = ref_date - pd.Timedelta(days=window_days)
    mask = timestamps > cutoff
    return child.loc[mask].copy()


def _compute_sequence_features(
    child: pd.DataFrame, relation: RelationSpec, child_spec: TableSpec
) -> pd.DataFrame:
    ts_col = child_spec.timestamp_column
    if not ts_col:
        raise ValueError(f"Table {relation.child} needs timestamp_column for sequence_features.")
    child = child.copy()
    child["__ts__"] = pd.to_datetime(child[ts_col], errors="coerce")
    ref_date = child["__ts__"].max()

    sorted_child = child.sort_values(relation.child_key + ["__ts__"])
    grouped = sorted_child.groupby(relation.child_key, dropna=False)["__ts__"]

    results: dict[str, pd.Series] = {}
    features = relation.sequence_features

    if "recency" in features:
        results[f"{relation.child}__recency"] = (
            ref_date - grouped.max()
        ).dt.total_seconds() / 86400.0

    if "time_span" in features:
        results[f"{relation.child}__time_span"] = (
            grouped.max() - grouped.min()
        ).dt.total_seconds() / 86400.0

    if "frequency" in features:
        counts = grouped.count()
        spans = (grouped.max() - grouped.min()).dt.total_seconds() / 86400.0
        results[f"{relation.child}__frequency"] = counts / (spans + 1.0)

    need_gaps = bool({"gap_mean", "gap_std", "acceleration"} & set(features))
    if need_gaps:
        gaps = grouped.apply(_compute_gaps)
        gap_stats = gaps.groupby(level=list(range(len(relation.child_key))))

        if "gap_mean" in features:
            results[f"{relation.child}__gap_mean"] = gap_stats.mean()
        if "gap_std" in features:
            results[f"{relation.child}__gap_std"] = gap_stats.std().fillna(0.0)
        if "acceleration" in features:
            results[f"{relation.child}__acceleration"] = gaps.groupby(
                level=list(range(len(relation.child_key)))
            ).apply(_compute_acceleration)

    if not results:
        empty = child[relation.child_key].drop_duplicates().copy()
        return empty

    out = pd.concat(results, axis=1).reset_index()
    out.columns = relation.parent_key + [col for col in out.columns[len(relation.parent_key) :]]
    for col in out.columns:
        if col not in relation.parent_key:
            out[col] = out[col].fillna(0.0)
    return out


def _compute_gaps(timestamps: pd.Series) -> pd.Series:
    sorted_ts = timestamps.sort_values()
    diffs = sorted_ts.diff().dropna().dt.total_seconds() / 86400.0
    return diffs.reset_index(drop=True)


def _compute_acceleration(gaps: pd.Series) -> float:
    if len(gaps) < 2:
        return 0.0
    mid = len(gaps) // 2
    older_mean = gaps.iloc[:mid].mean()
    recent_mean = gaps.iloc[mid:].mean()
    if older_mean == 0 and recent_mean == 0:
        return 0.0
    denom = older_mean if older_mean != 0 else 1.0
    return float((older_mean - recent_mean) / denom)
