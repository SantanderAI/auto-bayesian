# auto-bayesian

> **Open source by [Santander AI Lab](https://github.com/SantanderAI).** An
> **interpretable AI / machine-learning** Python library and CLI for **AutoML**:
> it trains explainable Bayesian-network classifiers from relational tabular
> data. A **trustworthy / responsible-AI** tool whose models are read directly,
> not as a black box.

[![CI](https://github.com/SantanderAI/auto-bayesian/actions/workflows/ci.yml/badge.svg)](https://github.com/SantanderAI/auto-bayesian/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![CodeQL](https://github.com/SantanderAI/auto-bayesian/actions/workflows/codeql.yml/badge.svg)](https://github.com/SantanderAI/auto-bayesian/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://www.conventionalcommits.org)

Part of [Santander AI Open Source](https://github.com/SantanderAI) ·
[santander.com](https://www.santander.com).

`auto-bayesian` trains **interpretable Bayesian-network classifiers** from
relational tabular data. You declare your tables, keys, and allowed aggregates;
the framework joins them into a single modeling table, trains a short list of
Bayesian-network candidates, selects the best one (by ROC-AUC, or PR-AUC for
imbalanced targets), tunes a decision threshold, and gives you a model you can
persist, score with, evaluate, and explain in plain English.

Unlike most AutoML tools, the output is not a black box: every node carries a
conditional probability table you can read directly, e.g.
`P(Converted = 1 | Source = referral, Region = west) = 0.85`.

## Features

- **Relational by design** — declare tables and relations; joins and aggregates are handled for you.
- **Three candidate structures** — Naive Bayes, Tree-Augmented Naive Bayes (TAN), and Hill-Climb search, with automatic selection by ROC-AUC or PR-AUC.
- **Deterministic preprocessing** — quantile or supervised (target-aware) binning, rare-category capping, missing-value handling, optional outlier removal and correlation pruning.
- **Imbalance-aware** — PR-AUC selection and an F1-tuned decision threshold, so rare positives aren't drowned out by a fixed 0.5 cutoff.
- **Interpretable output** — readable CPDs plus a Mermaid diagram of the learned network.
- **Two task types** — binary `classification` and `next_best_action` recommendation.
- **Typed Python API + a thin CLI**, with `pgmpy` as the only modeling engine.

## Scope (v0.1)

- Binary classification (and next-best-action over a binary outcome).
- Local CSV and Parquet inputs.
- `pgmpy` as the only engine.
- Relational support through explicit joins and aggregates.

## Setup

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
```

## CLI

```bash
# Validate a project config
uv run auto-bayesian validate-schema examples/lead_scoring.toml

# Build the flat modeling table
uv run auto-bayesian materialize examples/lead_scoring.toml

# Train, select the best candidate, and persist artifacts
uv run auto-bayesian train examples/lead_scoring.toml

# Score an already materialized table
uv run auto-bayesian predict artifacts/lead_scoring artifacts/lead_scoring/materialized.parquet

# Generate a Mermaid diagram of the network (Markdown)
uv run auto-bayesian explain artifacts/lead_scoring --output artifacts/lead_scoring/explanation.md
```

## Config

Table and output paths are resolved relative to the config file's location.

```toml
[task]
root_table = "leads"
target_column = "converted"
positive_label = "1"

[run]
output_dir = "artifacts/lead_scoring"
random_seed = 7
test_fraction = 0.25

[preprocess]
numeric_bins = 4
max_categories = 10

[[tables]]
name = "leads"
path = "data/leads.csv"
primary_key = "lead_id"

[[tables]]
name = "customers"
path = "data/customers.csv"
primary_key = "customer_id"

[[tables]]
name = "interactions"
path = "data/interactions.csv"
primary_key = "interaction_id"
timestamp_column = "event_time"

[[relations]]
parent = "leads"
child = "customers"
parent_key = "customer_id"
child_key = "customer_id"
kind = "one_to_one"

[[relations]]
parent = "leads"
child = "interactions"
parent_key = "lead_id"
child_key = "lead_id"
kind = "one_to_many"
aggregations = [
  { op = "count", name = "interaction_count" },
  { column = "channel", op = "nunique", name = "channel_count" },
  { column = "days_to_close", op = "mean", name = "mean_days_to_close" },
  { column = "channel", op = "latest", name = "latest_channel" },
]
```

Supported aggregations: `count`, `nunique`, `sum`, `mean`, `min`, `max`, `latest`.
A runnable version of this config lives in `examples/lead_scoring.toml`.

## Python

```python
from auto_bayesian import fit_project, load_project

project = load_project("examples/lead_scoring.toml")
model = fit_project(project)
scores = model.predict_proba(model.materialized_frame.drop(columns=["converted"]))
```

You can also build a project in Python and train directly from `pandas`
DataFrames:

```python
import pandas as pd

from auto_bayesian import build_project, fit_tables

tables = {
    "leads": pd.read_csv("data/leads.csv"),
    "customers": pd.read_csv("data/customers.csv"),
    "interactions": pd.read_csv("data/interactions.csv"),
}

project = build_project(
    root_table="leads",
    target_column="converted",
    positive_label="1",
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
                {"column": "channel", "op": "latest", "name": "latest_channel"},
            ],
        },
    ],
)

model = fit_tables(project, tables)
print(model.describe().selected_candidate)
print([(item.name, item.roc_auc) for item in model.describe().candidates])
```

## Example notebook

A full, relational, end-to-end walkthrough on real Kaggle data lives in
[`examples/notebooks/olist_relational_quickstart.ipynb`](examples/notebooks/olist_relational_quickstart.ipynb).
It uses the [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
dataset to predict late deliveries from four related tables (orders, customers,
order items, payments), then reads the learned CPDs and renders the network as a
Mermaid diagram.

```bash
pip install -e ".[examples]"   # adds kagglehub + JupyterLab
jupyter lab examples/notebooks/olist_relational_quickstart.ipynb
```

The notebook downloads the dataset with `kagglehub`, so you need Kaggle
credentials (`~/.kaggle/kaggle.json`).

## Explainability

Generate a Markdown file with a **Mermaid** diagram of the learned Bayesian
network plus a plain-language list of the relationships. Each arrow `A --> B`
means *A directly influences B*; the target node is highlighted:

```python
from auto_bayesian import generate_explanation, to_mermaid
from auto_bayesian.model import AutoBayesModel

model = AutoBayesModel.load("artifacts/lead_scoring")
generate_explanation(model, output_path="artifacts/lead_scoring/explanation.md")

# Or get just the Mermaid diagram source:
print(to_mermaid(model))
```

## Next best action

Set `task_type = "next_best_action"` with an `action_column` to rank candidate
actions by their predicted effect on the positive outcome. Score with
`model.predict_next_best_action(frame)` (or the `predict` CLI command, which
detects the task type automatically).

## Artifacts

Training writes these files into `run.output_dir`:

- `materialized.parquet`
- `metrics.json`
- `network.json`
- `model.pkl`

> **Security note:** `model.pkl` is a Python
> [pickle](https://docs.python.org/3/library/pickle.html) file. Loading a
> pickle can execute arbitrary code, so call `AutoBayesModel.load()` only on
> model directories you created yourself or obtained from a source you fully
> trust.

## Limits

- Each table can have at most one parent relation.
- The target must live in the root table.
- `one_to_many` relations require explicit aggregates or sequence features.
- CLI `predict` expects an already materialized table.

## Documentation

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full guide: the math behind
Bayesian Networks, the end-to-end pipeline, a configuration and API reference,
and complete tutorials.

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). By
contributing you agree to the [Code of Conduct](CODE_OF_CONDUCT.md) and to sign
the [Contributor License Agreement](https://github.com/SantanderAI/cla) (the CLA
Assistant bot will prompt you on your first pull request).

## Security

Please report vulnerabilities privately as described in
[`.github/SECURITY.md`](.github/SECURITY.md) — do not open public issues for
security reports.

## License

Licensed under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for
attribution.

## Citation

If you use `auto-bayesian` in your work, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).
