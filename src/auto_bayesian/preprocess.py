# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from auto_bayesian.schema import PreprocessSpec

MISSING_VALUE = "__missing__"
OTHER_VALUE = "__other__"


@dataclass(slots=True)
class ColumnTransform:
    kind: str
    bins: list[float] | None = None
    categories: list[str] | None = None


class DataPreprocessor:
    def __init__(self, spec: PreprocessSpec, target_column: str, positive_label: str) -> None:
        self.spec = spec
        self.target_column = target_column
        self.positive_label = positive_label
        self.transforms: dict[str, ColumnTransform] = {}
        self.feature_columns: list[str] = []
        self.dropped_columns: list[str] = []

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        transformed = frame.copy()
        self.feature_columns = [
            column for column in transformed.columns if column != self.target_column
        ]
        if self.spec.min_variance_fraction > 0:
            low_var = _find_low_variance_columns(
                transformed, self.feature_columns, self.spec.min_variance_fraction
            )
            self.dropped_columns.extend(low_var)
            self.feature_columns = [c for c in self.feature_columns if c not in set(low_var)]
        transformed[self.target_column] = self._fit_target(transformed[self.target_column])
        target_int = (transformed[self.target_column] == self.positive_label).astype(int)
        for column in self.feature_columns:
            transformed[column] = self._fit_feature(column, transformed[column], target_int)
        if self.spec.max_correlation < 1.0:
            corr_drop = _find_correlated_columns(
                transformed,
                self.feature_columns,
                self.target_column,
                self.spec.max_correlation,
            )
            self.dropped_columns.extend(corr_drop)
            self.feature_columns = [c for c in self.feature_columns if c not in set(corr_drop)]
            for col in corr_drop:
                self.transforms.pop(col, None)
        return transformed[[*self.feature_columns, self.target_column]]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Input frame is missing feature columns: {names}")

        transformed = frame.copy()
        if self.target_column in transformed.columns:
            transformed[self.target_column] = self._transform_target(
                transformed[self.target_column]
            )
        for column in self.feature_columns:
            transformed[column] = self._transform_feature(column, transformed[column])

        ordered = self.feature_columns.copy()
        if self.target_column in transformed.columns:
            ordered.append(self.target_column)
        return transformed[ordered]

    def state_names(self) -> dict[str, list[str]]:
        states = {}
        for column in self.feature_columns:
            states[column] = self.transforms[column].categories or []
        states[self.target_column] = self.transforms[self.target_column].categories or []
        return states

    def to_dict(self) -> dict:
        return {
            "spec": asdict(self.spec),
            "target_column": self.target_column,
            "positive_label": self.positive_label,
            "feature_columns": self.feature_columns,
            "dropped_columns": self.dropped_columns,
            "transforms": {
                column: asdict(transform) for column, transform in self.transforms.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DataPreprocessor":
        preprocessor = cls(
            spec=PreprocessSpec(**data["spec"]),
            target_column=data["target_column"],
            positive_label=data["positive_label"],
        )
        preprocessor.feature_columns = list(data["feature_columns"])
        preprocessor.dropped_columns = list(data.get("dropped_columns", []))
        preprocessor.transforms = {
            column: ColumnTransform(**transform) for column, transform in data["transforms"].items()
        }
        return preprocessor

    def _fit_target(self, series: pd.Series) -> pd.Series:
        values = series.astype(str)
        states = sorted(values.dropna().unique().tolist())
        if len(states) != 2:
            raise ValueError("The target column must be binary in v1.")
        if self.positive_label not in states:
            raise ValueError(
                f"Positive label {self.positive_label} is not present in the target column."
            )
        self.transforms[self.target_column] = ColumnTransform(kind="target", categories=states)
        return values

    def _transform_target(self, series: pd.Series) -> pd.Series:
        transformed = series.astype(str)
        allowed = set(self.transforms[self.target_column].categories or [])
        invalid = set(transformed.unique()) - allowed
        if invalid:
            names = ", ".join(sorted(str(item) for item in invalid))
            raise ValueError(f"Input target contains unseen labels: {names}")
        return transformed

    def _fit_feature(self, column: str, series: pd.Series, target_int: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(series):
            transformed, bins = _fit_numeric(
                series, self.spec.numeric_bins, self.spec.binning, target_int
            )
            categories = _categories_from_series(transformed)
            if MISSING_VALUE not in categories:
                categories.append(MISSING_VALUE)
            self.transforms[column] = ColumnTransform(
                kind="numeric", bins=bins, categories=categories
            )
            return transformed

        transformed, categories = _fit_categorical(series, self.spec.max_categories)
        if MISSING_VALUE not in categories:
            categories.append(MISSING_VALUE)
        if OTHER_VALUE not in categories:
            categories.append(OTHER_VALUE)
        self.transforms[column] = ColumnTransform(kind="categorical", categories=categories)
        return transformed

    def _transform_feature(self, column: str, series: pd.Series) -> pd.Series:
        transform = self.transforms[column]
        if transform.kind == "numeric":
            transformed = _transform_numeric(series, transform.bins or [])
        else:
            transformed = _transform_categorical(series, transform.categories or [])
        allowed = set(transform.categories or [])
        if OTHER_VALUE in allowed:
            transformed = transformed.where(transformed.isin(allowed), OTHER_VALUE)
        else:
            transformed = transformed.where(transformed.isin(allowed), MISSING_VALUE)
        return transformed.astype(str)


def _fit_numeric(
    series: pd.Series,
    bins: int,
    binning: str = "quantile",
    target_int: pd.Series | None = None,
) -> tuple[pd.Series, list[float]]:
    cleaned = pd.to_numeric(series, errors="coerce")
    non_missing = cleaned.dropna()
    if non_missing.nunique() <= 1:
        return _constant_numeric(cleaned), []

    edges: list[float] | None = None
    if binning == "supervised" and target_int is not None:
        aligned = target_int.reindex(non_missing.index)
        if aligned.notna().all():
            edges = _supervised_edges(
                non_missing.to_numpy(dtype=float),
                aligned.to_numpy(dtype=int),
                bins,
            )
    if edges is None:
        edges = _quantile_edges(non_missing, bins)
    if edges is None or len(edges) <= 2:
        return _constant_numeric(cleaned), []

    binned = pd.cut(cleaned, bins=edges, include_lowest=True, duplicates="drop")
    transformed = binned.astype(str).replace("nan", MISSING_VALUE).fillna(MISSING_VALUE)
    return transformed.astype(str), edges


def _constant_numeric(cleaned: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(cleaned.isna(), MISSING_VALUE, "constant"), index=cleaned.index
    ).astype(str)


def _quantile_edges(non_missing: pd.Series, bins: int) -> list[float] | None:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(non_missing.quantile(quantiles).to_numpy(dtype=float))
    if len(edges) <= 2:
        return None
    edges = edges.tolist()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _supervised_edges(values: np.ndarray, target: np.ndarray, max_bins: int) -> list[float] | None:
    """Greedy top-down binning that splits ``values`` to separate the target.

    Cut points are chosen one at a time to maximize the decrease in Gini
    impurity of ``target`` (a decision-tree-style 1-D discretization), stopping
    at ``max_bins`` bins or when no split helps. Edges fall between observed,
    distinct values, so the result is deterministic for a given input. Returns
    ``None`` when no informative split exists, letting the caller fall back to
    quantile binning.
    """
    if max_bins <= 1:
        return None
    order = np.argsort(values, kind="mergesort")
    xs = values[order]
    ys = target[order]
    n = xs.shape[0]
    prefix_pos = np.concatenate(([0], np.cumsum(ys))).astype(float)

    def weighted_gini(a: int, b: int) -> float:
        count = b - a
        if count <= 0:
            return 0.0
        positives = prefix_pos[b] - prefix_pos[a]
        negatives = count - positives
        return 2.0 * positives * negatives / count

    boundaries = [0, n]
    while len(boundaries) - 1 < max_bins:
        best_gain = 1e-12
        best_k: int | None = None
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            parent = weighted_gini(a, b)
            for k in range(a + 1, b):
                if xs[k] == xs[k - 1]:
                    continue
                gain = parent - weighted_gini(a, k) - weighted_gini(k, b)
                if gain > best_gain:
                    best_gain = gain
                    best_k = k
        if best_k is None:
            break
        boundaries.append(best_k)
        boundaries.sort()

    internal = boundaries[1:-1]
    if not internal:
        return None
    midpoints = sorted((xs[k - 1] + xs[k]) / 2.0 for k in internal)
    return [-np.inf, *midpoints, np.inf]


def _transform_numeric(series: pd.Series, bins: list[float]) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")
    if not bins:
        transformed = pd.Series(
            np.where(cleaned.isna(), MISSING_VALUE, "constant"), index=series.index
        )
        return transformed.astype(str)
    binned = pd.cut(cleaned, bins=bins, include_lowest=True, duplicates="drop")
    return binned.astype(str).replace("nan", MISSING_VALUE).fillna(MISSING_VALUE).astype(str)


def _fit_categorical(series: pd.Series, max_categories: int) -> tuple[pd.Series, list[str]]:
    cleaned = series.astype("string").fillna(MISSING_VALUE).replace("<NA>", MISSING_VALUE)
    counts = cleaned.value_counts(dropna=False)
    keep = counts.head(max_categories).index.tolist()
    transformed = cleaned.where(cleaned.isin(keep), OTHER_VALUE).astype(str)
    categories = _categories_from_series(transformed)
    if OTHER_VALUE not in categories and len(counts) > len(keep):
        categories.append(OTHER_VALUE)
    return transformed, categories


def _transform_categorical(series: pd.Series, categories: list[str]) -> pd.Series:
    cleaned = (
        series.astype("string").fillna(MISSING_VALUE).replace("<NA>", MISSING_VALUE).astype(str)
    )
    allowed = set(categories)
    if OTHER_VALUE in allowed:
        return cleaned.where(cleaned.isin(allowed), OTHER_VALUE).astype(str)
    return cleaned.where(cleaned.isin(allowed), MISSING_VALUE).astype(str)


def _categories_from_series(series: pd.Series) -> list[str]:
    return sorted(str(value) for value in series.astype(str).dropna().unique().tolist())


def remove_outliers(
    frame: pd.DataFrame,
    spec: PreprocessSpec,
    target_column: str,
) -> pd.DataFrame:
    if spec.outlier_method is None:
        return frame
    if spec.outlier_method == "iqr":
        return _remove_outliers_iqr(frame, spec.outlier_iqr_factor, target_column)
    raise ValueError(f"Unsupported outlier_method: {spec.outlier_method}")


def _remove_outliers_iqr(frame: pd.DataFrame, factor: float, target_column: str) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column in frame.columns:
        if column == target_column:
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = mask & (values.isna() | ((values >= lower) & (values <= upper)))
    return frame.loc[mask].reset_index(drop=True)


def _find_low_variance_columns(
    frame: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
) -> list[str]:
    drop = []
    for column in feature_columns:
        counts = frame[column].value_counts(normalize=True, dropna=False)
        if counts.iloc[0] >= threshold:
            drop.append(column)
    return drop


def _find_correlated_columns(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    max_correlation: float,
) -> list[str]:
    if len(feature_columns) < 2:
        return []
    n = len(frame)
    if n == 0:
        return []

    target_assoc: dict[str, float] = {}
    for col in feature_columns:
        target_assoc[col] = _cramers_v(frame[col], frame[target_column])

    drop: set[str] = set()
    checked = list(feature_columns)
    for i in range(len(checked)):
        if checked[i] in drop:
            continue
        for j in range(i + 1, len(checked)):
            if checked[j] in drop:
                continue
            v = _cramers_v(frame[checked[i]], frame[checked[j]])
            if v > max_correlation:
                weaker = (
                    checked[j]
                    if target_assoc[checked[i]] >= target_assoc[checked[j]]
                    else checked[i]
                )
                drop.add(weaker)
    return sorted(drop)


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x.astype(str), y.astype(str))
    n = confusion.sum().sum()
    if n == 0:
        return 0.0
    chi2 = 0.0
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            expected = row_sums.iloc[i] * col_sums.iloc[j] / n
            if expected > 0:
                chi2 += (confusion.iloc[i, j] - expected) ** 2 / expected
    k = min(confusion.shape[0], confusion.shape[1])
    if k <= 1:
        return 0.0
    return float(np.sqrt(chi2 / (n * (k - 1))))
