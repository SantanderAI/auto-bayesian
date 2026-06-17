# auto-bayesian — Complete Documentation

> **One sentence:** `auto-bayesian` is a Python framework that automatically trains
> interpretable Bayesian Network classifiers from one or more relational tables,
> picks the best structure, and gives you a model you can explain to a business
> stakeholder in plain English.

---

## Table of Contents

1. [Why Does This Library Exist?](#1-why-does-this-library-exist)
2. [What Is a Bayesian Network? (The Intuition)](#2-what-is-a-bayesian-network-the-intuition)
3. [The Mathematics Behind Bayesian Networks](#3-the-mathematics-behind-bayesian-networks)
4. [How the Pipeline Works End-to-End](#4-how-the-pipeline-works-end-to-end)
5. [Step 1 — Schema & Configuration](#5-step-1--schema--configuration)
6. [Step 2 — Multi-Table Materialization](#6-step-2--multi-table-materialization)
7. [Step 3 — Data Preprocessing](#7-step-3--data-preprocessing)
8. [Step 4 — Train/Validation Split](#8-step-4--trainvalidation-split)
9. [Step 5 — Training the Three Candidate Structures](#9-step-5--training-the-three-candidate-structures)
10. [Step 6 — Model Selection](#10-step-6--model-selection)
11. [Step 7 — Prediction (Inference)](#11-step-7--prediction-inference)
12. [Evaluation Metrics — The Math](#12-evaluation-metrics--the-math)
13. [Configuration Reference](#13-configuration-reference)
14. [Python API Reference](#14-python-api-reference)
15. [CLI Reference](#15-cli-reference)
16. [Full Tutorial — Lead Scoring (TOML + CLI)](#16-full-tutorial--lead-scoring-toml--cli)
17. [Full Tutorial — Titanic Survival (Pure Python)](#17-full-tutorial--titanic-survival-pure-python)
18. [Explainability (Mermaid Diagram)](#18-explainability-mermaid-diagram)
19. [Architecture & File Map](#19-architecture--file-map)
20. [Limitations](#20-limitations)
21. [Glossary](#21-glossary)

---

## 1. Why Does This Library Exist?

Most AutoML tools (XGBoost, LightGBM, AutoGluon, etc.) give you great accuracy
but **zero interpretability**. You get a number — "the probability is 0.73" — but
you **cannot** tell a regulator or a business owner *why*.

A **Bayesian Network** is different. It is a small graph where every arrow is a
direct probabilistic dependency, and every node carries a probability table you
can print and read. For example:

```
P(Converted = 1 | Source = referral, Region = west) = 0.85
```

That sentence *is* the model. No black box.

`auto-bayesian` automates the entire process:

1. You describe your tables and how they relate (like a mini-database schema).
2. The library joins them into one flat table.
3. It discretizes numbers, caps rare categories, and handles missing values.
4. It trains **three** different Bayesian Network structures.
5. It picks the best one (by ROC-AUC by default, or PR-AUC for rare positives)
   and tunes a decision threshold for F1.
6. It gives you a model object that can predict, evaluate, save, load, and
   explain itself.

**In short:** you get the convenience of AutoML with the transparency of a
Bayesian Network.

---

## 2. What Is a Bayesian Network? (The Intuition)

Imagine you are trying to predict whether a sales lead will convert. You know
three things about each lead: their **age group**, the **source** (web, email,
referral), and their **region**.

A Bayesian Network says:

> "Let me draw arrows between things that *directly* influence each other,
> and for each arrow, let me store a probability table."

For example:

```
    Source ──────► Converted ◄────── Region
                      ▲
                      │
                   Age Group
```

This diagram (called a **DAG** — Directed Acyclic Graph) says:

- **Source** directly influences whether someone converts.
- **Region** directly influences conversion.
- **Age Group** directly influences conversion.
- Source, Region, and Age Group do *not* directly influence each other (no arrows
  between them).

At each node, there is a **Conditional Probability Distribution (CPD)** — a
table that says "given my parents, here are the probabilities of my different
states."

### Example CPD

| Converted | Source=web | Source=email | Source=referral |
|-----------|-----------|-------------|-----------------|
| 0         | 0.80      | 0.60        | 0.20            |
| 1         | 0.20      | 0.40        | 0.80            |

Reading this table: *"If the source is referral, the probability of converting
is 80%."*

**That is the entire model.** No hidden layers, no ensembles, no gradients.
Just conditional probability tables attached to a graph.

---

## 3. The Mathematics Behind Bayesian Networks

### 3.1 Bayes' Theorem

Everything starts with Bayes' Theorem:

```
                P(Evidence | Hypothesis) × P(Hypothesis)
P(Hypothesis | Evidence) = ──────────────────────────────────────
                                    P(Evidence)
```

In plain English:

> *"The probability of a hypothesis given some evidence equals the likelihood
> of seeing that evidence under the hypothesis, times how likely the hypothesis
> was before we saw the evidence, divided by the overall probability of the
> evidence."*

- **P(Hypothesis)** — the *prior*: what we believed before seeing data.
- **P(Evidence | Hypothesis)** — the *likelihood*: how well the hypothesis
  explains the data.
- **P(Hypothesis | Evidence)** — the *posterior*: our updated belief.

### 3.2 Directed Acyclic Graphs (DAGs)

A Bayesian Network's structure is a **DAG**: a graph where:

- Each node is a random variable (e.g., `Age`, `Source`, `Converted`).
- Each directed edge (arrow) represents a direct probabilistic dependency.
- There are **no cycles** — you can never follow the arrows and loop back.

### 3.3 The Joint Probability Factorization

The power of a DAG is that it lets you factor the joint probability of all
variables into a product of small, manageable conditional probabilities:

```
P(X₁, X₂, ..., Xₙ) = ∏ᵢ P(Xᵢ | Parents(Xᵢ))
```

**Example** with three variables (Source, Region, Converted):

```
P(Source, Region, Converted) = P(Source) × P(Region) × P(Converted | Source, Region)
```

Instead of needing one giant table with every possible combination, you only
need three small tables. This is what makes Bayesian Networks efficient.

### 3.4 Conditional Probability Distributions (CPDs)

Each node stores a **CPD** — a table of probabilities conditioned on the node's
parents in the DAG.

- **Root nodes** (no parents): the CPD is just a simple probability table.
  Example: `P(Source = web) = 0.4, P(Source = email) = 0.35, P(Source = referral) = 0.25`

- **Child nodes** (has parents): the CPD is a table indexed by every
  combination of parent states.
  Example: `P(Converted = 1 | Source = web, Region = north) = 0.15`

### 3.5 Parameter Estimation — Bayesian Estimator with BDeu Prior

Once you have a structure (the DAG), you need to fill in the CPD tables from
data. `auto-bayesian` uses the **Bayesian Estimator** with a **BDeu prior**
(Bayesian Dirichlet equivalent uniform).

**Why not just count frequencies?** With small data, some parent-state
combinations may have zero or very few observations. Raw frequency counts would
give you probabilities of 0 or 1, which are extreme and unreliable.

The BDeu prior adds a small "pseudo-count" (like a smoothing factor) to every
cell in the table, ensuring no probability is ever exactly 0 or 1.

The formula for a single CPD entry is:

```
                    count(Xᵢ = xᵢ, Parents = parents) + αᵢⱼ
P(Xᵢ = xᵢ | Parents = parents) = ─────────────────────────────────────────
                                count(Parents = parents) + αⱼ
```

Where:
- `count(...)` is the number of rows in the training data matching that
  combination.
- `αᵢⱼ` is the prior pseudo-count for this specific cell.
- `αⱼ` is the sum of pseudo-counts for all states of Xᵢ given this parent
  configuration.

Under BDeu, the total prior strength `N'` (called the *equivalent sample size*,
typically 1 or 5) is spread *uniformly* across all possible states:

```
αᵢⱼ = N' / (qᵢ × rᵢ)
```

Where:
- `qᵢ` = number of parent configurations for node i.
- `rᵢ` = number of states of node i.

**Intuition:** BDeu says *"before seeing any data, I believe all combinations
are equally likely, but I'm not very confident."* The data then quickly
overrides this gentle prior.

---

## 4. How the Pipeline Works End-to-End

Here is the complete flow from raw data to predictions:

```
┌─────────────────────┐
│   TOML Config or    │
│   Python build_project()  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  1. Parse & Validate│   schema.py
│     Schema          │   (ProjectSpec, TableSpec, RelationSpec, etc.)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Materialize     │   materialize.py
│     (Join & Agg)    │   Multi-table → single flat DataFrame
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Preprocess      │   preprocess.py
│     (Bin & Encode)  │   Continuous → discrete bins
│                     │   Categorical → top-N + __other__
│                     │   Missing → __missing__
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. Stratified Split│   automl.py
│     Train / Val     │   Preserves class balance
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  5. Train THREE candidate Bayesian Networks     │   engine_pgmpy.py
│                                                 │
│  ┌─────────────┐ ┌─────────┐ ┌──────────────┐  │
│  │ Naive Bayes │ │   TAN   │ │ Hill Climb   │  │
│  │  (simple)   │ │ (tree)  │ │ (greedy DAG) │  │
│  └──────┬──────┘ └────┬────┘ └──────┬───────┘  │
│         │             │             │           │
│         ▼             ▼             ▼           │
│     Evaluate on validation set (ROC-AUC)       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────┐
│  6. Select Best     │   automl.py
│     by ROC/PR-AUC   │   (tie-break: lowest log-loss)
│     + tune threshold│   (F1-optimal cutoff)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  7. AutoBayesModel  │   model.py
│     .predict_proba()│   Uses Variable Elimination
│     .predict()      │   for exact inference
│     .save() / .load()│
└─────────────────────┘
```

---

## 5. Step 1 — Schema & Configuration

**File:** `schema.py`

The schema defines *what* your data looks like and *what* you want to predict.
You can provide it as a TOML file or build it in Python.

### Core Data Classes

| Class | Purpose |
|-------|---------|
| `ProjectSpec` | Top-level container for the entire project |
| `TableSpec` | Describes one table: name, file path, primary key, optional timestamp column |
| `RelationSpec` | Describes how two tables relate: parent, child, keys, kind (`one_to_one` or `one_to_many`), aggregations, sequence features |
| `AggregateSpec` | Describes one aggregation for a `one_to_many` relation: operation, column, output name, optional time window |
| `TaskSpec` | What to predict: root table, target column, positive label, task type, action column |
| `PreprocessSpec` | How to preprocess: numeric bins, binning strategy, max categories, outlier removal, duplicate dropping, low-variance and correlation pruning |
| `RunSpec` | Runtime settings: output directory, random seed, test fraction, candidate selection metric |

### Validation Rules

The schema validator (`validate_project`) enforces:

1. At least one table must be declared.
2. The root table must exist.
3. `test_fraction` must be between 0 and 0.5.
4. Every table must have a primary key.
5. Relations can only use `one_to_one` or `one_to_many`.
6. Parent and child tables in a relation must exist.
7. Parent and child keys must have the same length.
8. **Each table can have at most one parent** (tree-shaped relations only).
9. `one_to_many` relations **must** declare at least one aggregation or sequence feature.
10. Aggregations must use supported operations: `count`, `nunique`, `sum`, `mean`, `min`, `max`, `latest`.
11. All aggregations except `count` require a column name.
12. The relation graph must be **acyclic** and fully connected to the root.
13. All table files must exist on disk (unless provided as in-memory DataFrames).
14. `task_type` must be `classification` or `next_best_action`; the latter requires an `action_column`.
15. Preprocess bounds: `outlier_iqr_factor > 0`, `min_variance_fraction` in `[0.0, 1.0)`, `max_correlation` in `(0.0, 1.0]`.
16. `binning` must be `quantile` or `supervised`; `selection_metric` must be `roc_auc` or `pr_auc`.

---

## 6. Step 2 — Multi-Table Materialization

**File:** `materialize.py`

Most real-world datasets are not a single flat table. You might have a `leads`
table, a `customers` table, and an `interactions` table. Before we can train a
model, we need to combine them into **one flat DataFrame** — this process is
called **materialization**.

### How It Works

1. **Load tables** from CSV/Parquet files or in-memory DataFrames.
2. **Sort relations bottom-up** — process the deepest child tables first.
3. **Join or aggregate** each child into its parent:

#### `one_to_one` Relations

A simple left join. Each row in the parent matches at most one row in the child.
Child columns are prefixed with `{child_name}__` to avoid name collisions.

```
leads (parent)          customers (child)
┌─────────┬─────┐       ┌─────────────┬────────┐
│ lead_id │ ... │       │ customer_id │ region │
├─────────┼─────┤       ├─────────────┼────────┤
│ 1       │ ... │  ──►  │ 101         │ north  │
└─────────┴─────┘       └─────────────┴────────┘

Result: leads table now has a "customers__region" column
```

#### `one_to_many` Relations

You cannot directly join because one parent row matches *many* child rows. So
the child rows are **aggregated** first, then joined.

Supported aggregation operations:

| Operation | What It Computes | Requires Column? |
|-----------|-----------------|-----------------|
| `count` | Number of child rows | No |
| `nunique` | Number of distinct values | Yes |
| `sum` | Sum of a numeric column | Yes |
| `mean` | Average of a numeric column | Yes |
| `min` | Minimum value | Yes |
| `max` | Maximum value | Yes |
| `latest` | Value from the most recent row (by timestamp) | Yes (+ `timestamp_column` on the child table) |

**Example:** If lead #2 has 3 interactions, the `count` aggregation produces:

```
lead_id │ interaction_count
────────┼──────────────────
2       │ 3
```

This aggregated row is then left-joined onto the parent `leads` table.

### Bottom-Up Processing Order

Relations are processed from the **deepest** child first (farthest from the
root table) to the shallowest, using a breadth-first traversal to assign depth.
This ensures that when a parent is joined, its children have already been
aggregated into it.

---

## 7. Step 3 — Data Preprocessing

**File:** `preprocess.py`

Bayesian Networks with pgmpy require **discrete** variables. All columns must
be converted into a finite set of categories. The `DataPreprocessor` handles
this automatically.

### 7.1 Numeric Columns → Quantile Bins

Continuous numbers (age, income, score) are converted into bins using
**quantile-based binning**:

1. Compute `numeric_bins + 1` equally-spaced quantile edges (0%, 20%, 40%,
   60%, 80%, 100% for 5 bins).
2. Remove duplicate edges (happens when many values are the same).
3. Set the first edge to `-∞` and the last to `+∞` (so all values are captured).
4. Use `pd.cut()` to assign each value to an interval like `"(20.0, 35.0]"`.

**Example** with `numeric_bins = 4` on ages [22, 25, 31, 35, 42, 48, 55, 60]:

```
Quantile edges: [-∞, 28.0, 38.5, 51.5, ∞]

22  → "(-inf, 28.0]"
25  → "(-inf, 28.0]"
31  → "(28.0, 38.5]"
35  → "(28.0, 38.5]"
42  → "(38.5, 51.5]"
48  → "(38.5, 51.5]"
55  → "(51.5, inf]"
60  → "(51.5, inf]"
```

**Why quantile bins instead of equal-width bins?** Quantile bins ensure each
bin has roughly the same number of observations, which gives the Bayesian
Network more balanced conditional probability estimates.

**Supervised (target-aware) binning.** Set `binning = "supervised"` to place cut
points where the target rate actually changes instead of at evenly-spaced
quantiles. This is a greedy, decision-tree-style 1-D discretization: cut points
are added one at a time to maximize the decrease in Gini impurity of the target,
up to `numeric_bins` bins. It is deterministic for a given input and falls back
to quantile binning when no informative split exists. Supervised binning often
sharpens signal for skewed or imbalanced targets, where evenly-sized quantile
bins can smear the rare class across every bin.

**Edge case:** If a numeric column has only one unique value (or degenerate
quantiles), it is collapsed to a constant `"constant"`.

### 7.2 Categorical Columns → Top-N Categories

String/categorical columns are limited to the `max_categories` most frequent
values:

1. Count the frequency of each value.
2. Keep the top `max_categories` values.
3. Replace all other values with `"__other__"`.

**Example** with `max_categories = 3` on a column with values
`[web, email, referral, phone, partner]`:

```
web      → "web"       (top 3)
email    → "email"     (top 3)
referral → "referral"  (top 3)
phone    → "__other__" (not in top 3)
partner  → "__other__" (not in top 3)
```

### 7.3 Missing Values

All missing/null values are replaced with the special token `"__missing__"`.
This is treated as just another category — the Bayesian Network naturally
learns `P(Target | Feature = __missing__)`.

### 7.4 Target Column

The target column must be **binary** (exactly 2 distinct values). It is cast to
string. The `positive_label` parameter tells the library which value means "yes"
(e.g., `"1"` or `"True"` or `"converted"`).

### 7.5 State Names

After preprocessing, every column has a known, fixed list of possible states.
These are stored and passed to pgmpy as `state_names`, which ensures the CPD
tables are properly aligned even if some states are missing from the training or
validation split.

---

## 8. Step 4 — Train/Validation Split

**File:** `automl.py` → `stratified_split()`

The data is split into **training** and **validation** sets using a
**stratified** approach:

1. Group all row indices by their target-column value.
2. For each class, shuffle the indices (using the configured `random_seed`).
3. Take `test_fraction` of each class for validation, the rest for training.
4. This guarantees both the training and validation sets have a similar class
   balance.

**Why stratified?** If only 10% of leads convert, a random split might put all
converters in the training set and none in validation (or vice versa). Stratified
splitting prevents this.

**Guard rails:**
- Each class must have at least 2 rows (1 for training, 1 for validation).
- `test_fraction` must be between 0 and 0.5.

---

## 9. Step 5 — Training the Three Candidate Structures

**File:** `engine_pgmpy.py`

This is the heart of the library. Three different Bayesian Network structures
are trained, each with a different philosophy for finding the best DAG.

### 9.1 Candidate 1: Naive Bayes

**Structure:** The simplest possible Bayesian Network for classification.

```
         Target
        /  |  \
       /   |   \
      ▼    ▼    ▼
    F₁    F₂    F₃   ...  Fₙ
```

Every feature is a **direct child of the target**, and there are **no edges
between features**.

**Mathematical assumption — Conditional Independence:**

```
P(F₁, F₂, ..., Fₙ | Target) = P(F₁ | Target) × P(F₂ | Target) × ... × P(Fₙ | Target)
```

This says: *"Given the target, knowing the value of one feature tells you
nothing about any other feature."*

**This is almost never true in practice** (e.g., age and income are correlated),
but Naive Bayes works surprisingly well because:
- It only needs to estimate small 1-parent CPDs (fewer parameters).
- With limited data, simple models often beat complex ones.

**Implementation:**
```python
edges = [(target_column, feature) for feature in features]
model = DiscreteBayesianNetwork(edges)
model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
```

### 9.2 Candidate 2: Tree-Augmented Naive Bayes (TAN)

**Structure:** Like Naive Bayes, but features are also allowed to have **one
edge between them**, forming a tree over the features.

```
         Target
        /  |  \
       /   |   \
      ▼    ▼    ▼
    F₁ ──► F₂   F₃
                  │
                  ▼
                 F₄
```

**Key idea:** TAN relaxes the conditional independence assumption of Naive Bayes.
It allows each feature to depend on **one other feature** (in addition to the
target), forming a tree structure among the features.

**How the tree is found — Chow-Liu Algorithm:**

1. Compute the **conditional mutual information** between every pair of features
   given the target:
   ```
   I(Fᵢ; Fⱼ | Target) = Σ P(fᵢ, fⱼ, t) × log[ P(fᵢ, fⱼ | t) / (P(fᵢ | t) × P(fⱼ | t)) ]
   ```
   This measures how much knowing `Fᵢ` tells you about `Fⱼ` *after* you already
   know the target.

2. Build a complete graph where each edge weight is the conditional mutual
   information.

3. Find the **maximum spanning tree** of this graph (the tree that captures
   the strongest remaining dependencies).

4. Add edges from the target to every feature.

**Why TAN?** It's a great middle ground — more expressive than Naive Bayes
(captures feature correlations) but still simple enough to estimate reliably.

**Implementation:**
```python
estimator = TreeSearch(train_data)
dag = estimator.estimate(estimator_type="tan", class_node=target_column)
model = DiscreteBayesianNetwork(dag.edges())
model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
```

### 9.3 Candidate 3: Hill Climb Search with BIC

**Structure:** A general-purpose DAG learned by greedy search. No structural
constraints — any edge between any pair of variables is possible (as long as it
doesn't create a cycle).

```
    F₁ ──► Target ◄──── F₃
    │         ▲           │
    ▼         │           ▼
    F₂ ───────┘          F₄
```

**How it works — Greedy Hill Climbing:**

1. Start with an empty graph (no edges).
2. At each step, consider three possible actions:
   - **Add** an edge between two unconnected nodes.
   - **Remove** an existing edge.
   - **Reverse** the direction of an existing edge.
3. For each possible action, compute the **BIC score** of the resulting graph.
4. Apply the action that improves the BIC score the most.
5. Repeat until no action improves the score.

**The BIC Scoring Function:**

```
BIC = log P(Data | Structure, θ_MLE) - (d / 2) × log(N)
```

Where:
- `log P(Data | Structure, θ_MLE)` is the log-likelihood of the data given the
  structure and maximum-likelihood parameter estimates.
- `d` is the number of free parameters in the model.
- `N` is the number of training examples.
- `(d / 2) × log(N)` is a **penalty for complexity** — more edges mean more
  parameters, which increases `d`, which decreases the BIC score.

**Intuition:** BIC balances fit (how well does this structure explain the data?)
against complexity (how many parameters does it need?). This prevents
overfitting.

The library uses the `"bic-d"` variant from pgmpy, which is BIC adapted for
discrete variables.

**Implementation:**
```python
estimator = HillClimbSearch(train_data)
dag = estimator.estimate(scoring_method="bic-d")
model = DiscreteBayesianNetwork(dag.edges())
model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu")
```

### 9.4 Post-Training: Pruning Unfitted Nodes

After fitting each candidate, the library runs `_prune_unfitted_nodes()`. This
removes any node from the network that didn't receive a CPD during fitting (e.g.,
isolated nodes that the structure learner ignored). The target node is never
pruned — if it lacks a CPD, an error is raised.

### 9.5 Comparison Summary

| Candidate | Structure | Complexity | Strengths | Weaknesses |
|-----------|-----------|------------|-----------|------------|
| **Naive Bayes** | Star (target → all features) | Very low | Robust with small data, fast | Ignores feature correlations |
| **TAN** | Tree over features + target → all | Low–Medium | Captures pairwise feature dependencies | Only one dependency per feature |
| **Hill Climb** | Arbitrary DAG | Medium–High | Can find complex dependencies | May overfit with few data |

---

## 10. Step 6 — Model Selection

**File:** `automl.py` → `select_best_candidate()`

After training all three candidates, each is evaluated on the **validation set**
(the held-out data). The selection criterion is:

```
Best = max(candidates, key = (selection_metric, -log_loss))
```

1. **Primary criterion:** Highest `selection_metric` — `roc_auc` by default, or
   `pr_auc` (average precision) for imbalanced targets, where ROC-AUC can mask
   poor ranking of the rare positive class.
2. **Tie-breaker:** Lowest log-loss (the candidate whose probability estimates
   are most calibrated).

The selected candidate's validation scores are then scanned for the
**F1-optimal decision threshold**, stored on the report as `report.threshold`
and used by `predict`/`evaluate`.

The winning candidate's network, edges, and metrics are packaged into an
`AutoBayesModel` object. A full `ModelReport` is also generated, listing all
three candidates and their scores, sorted by performance.

---

## 11. Step 7 — Prediction (Inference)

**File:** `engine_pgmpy.py` → `predict_probabilities()`

When you call `model.predict_proba(new_data)`, the following happens:

### 11.1 Preprocessing

The new data is passed through the **same** `DataPreprocessor` that was fitted
during training. This ensures numbers are binned into the same intervals and
categories are mapped to the same set of allowed values.

### 11.2 Variable Elimination

For each row, the library performs **exact inference** using pgmpy's
`VariableElimination` algorithm.

**What is Variable Elimination?**

Given evidence (the observed feature values for one row), we want to compute:

```
P(Target = positive | F₁ = f₁, F₂ = f₂, ..., Fₙ = fₙ)
```

Variable Elimination works by:

1. Start with all the CPD tables (factors) in the network.
2. For each observed feature, **restrict** its factor to the observed value.
3. For any unobserved variable (not in the evidence, not the target), **sum
   it out** (marginalize it away).
4. Multiply the remaining factors together.
5. Normalize so the probabilities sum to 1.

**Why exact inference?** Bayesian Networks trained by `auto-bayesian` are small
(typically < 20 nodes), so exact inference is fast. Approximate methods (like
sampling) are unnecessary.

### 11.3 Output

- `predict_proba(df)` → returns a `pd.Series` of probabilities (floats between
  0 and 1).
- `predict(df, threshold=None)` → returns a `pd.Series` of 0/1 predictions
  (1 if probability ≥ threshold). When `threshold` is `None` the model's **tuned
  decision threshold** (`report.threshold`, chosen to maximize validation F1) is
  used instead of a fixed 0.5 — the right default for a rare positive class,
  where 0.5 would label everything negative.
- `evaluate(df, threshold=None)` → returns a `BinaryMetrics` for a labeled frame
  (must contain the target column): ROC-AUC, PR-AUC, and precision/recall/F1 at
  the tuned (or supplied) threshold, plus the positive base rate.

---

## 12. Evaluation Metrics — The Math

### 12.1 ROC-AUC (Area Under the ROC Curve)

The library computes ROC-AUC using the **Wilcoxon-Mann-Whitney statistic**
(equivalent to AUC but computed from rank sums):

```
         Σ(ranks of positives) - n₊(n₊ + 1) / 2
AUC = ─────────────────────────────────────────────
                      n₊ × n₋
```

Where:
- `n₊` = number of positive examples.
- `n₋` = number of negative examples.
- Examples are sorted by predicted score, and tied scores receive average ranks.

**Interpretation:**
- **AUC = 1.0** → perfect separation (all positives ranked above all negatives).
- **AUC = 0.5** → random chance (no better than coin flip).
- **AUC < 0.5** → worse than random (model is inverted).

**Why AUC?** It is threshold-independent — it measures how well the model
*ranks* positives above negatives, regardless of what cutoff you choose.

**Caveat for imbalanced data.** ROC-AUC can look healthy even when the rare
positive class is ranked poorly, because it is dominated by the abundant
negatives. For skewed targets, prefer PR-AUC (below) and select candidates with
`selection_metric = "pr_auc"`.

### 12.1b PR-AUC (Average Precision)

The library also reports **average precision** — the standard "PR-AUC" — the
precision-weighted sum of recall increments as the threshold sweeps from high to
low scores:

```
AP = Σ (recallₖ - recallₖ₋₁) × precisionₖ
```

Unlike ROC-AUC, PR-AUC focuses on the positive class and should be read against
the **positive base rate**: a random model scores ≈ the base rate, so an AP of
0.30 on a 5%-positive problem is a 6× lift, not a poor result.

### 12.1c Tuned Threshold, Precision, Recall, F1

Ranking metrics do not pick a decision cutoff. After selecting the best
candidate, the library scans the validation scores for the threshold that
maximizes F1 and stores it as `report.threshold`:

```
precision = TP / (TP + FP)      recall = TP / (TP + FN)
F1 = 2 · precision · recall / (precision + recall)
```

`predict` and `evaluate` use this tuned threshold by default, so a model on a
rare positive class still makes useful positive predictions instead of always
choosing the majority class.

### 12.2 Log-Loss (Binary Cross-Entropy)

```
                  1   N
Log-Loss = - ─── Σ [ yᵢ log(pᵢ) + (1 - yᵢ) log(1 - pᵢ) ]
                  N  i=1
```

Where:
- `yᵢ` = true label (0 or 1).
- `pᵢ` = predicted probability, clipped to `[1e-9, 1 - 1e-9]` to avoid
  `log(0)`.

**Interpretation:**
- **Lower is better.**
- Penalizes confident wrong predictions heavily (predicting 0.99 when the true
  label is 0 is very expensive).
- Measures **calibration** — not just ranking, but whether the probabilities
  are accurate.

---

## 13. Configuration Reference

A TOML configuration file has four sections:

### `[task]` — Required

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `root_table` | string | — | Name of the main table containing the target |
| `target_column` | string | — | Column to predict |
| `positive_label` | string | `"1"` | Value that means "yes" / "positive class" |
| `task_type` | string | `"classification"` | `"classification"` or `"next_best_action"` |
| `action_column` | string | `null` | Required when `task_type = "next_best_action"`; the column whose values are ranked as candidate actions |

### `[run]` — Optional

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_dir` | string | `"artifacts"` | Where to save the trained model |
| `random_seed` | int | `7` | Seed for reproducibility |
| `test_fraction` | float | `0.2` | Fraction of data for validation (must be 0 < x < 0.5) |
| `selection_metric` | string | `"roc_auc"` | Metric used to pick the best candidate: `"roc_auc"` or `"pr_auc"` (prefer `"pr_auc"` for imbalanced targets) |

### `[preprocess]` — Optional

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `numeric_bins` | int | `5` | Number of bins for numeric columns |
| `binning` | string | `"quantile"` | Numeric binning strategy: `"quantile"` (equal-frequency) or `"supervised"` (target-aware, decision-tree-style) |
| `max_categories` | int | `20` | Maximum categories to keep for string columns |
| `outlier_method` | string | `null` | Outlier removal method applied to the training split; currently `"iqr"` |
| `outlier_iqr_factor` | float | `1.5` | IQR multiplier used when `outlier_method = "iqr"` (must be > 0) |
| `drop_duplicates` | bool | `false` | Drop duplicate rows from the training split before fitting |
| `min_variance_fraction` | float | `0.0` | Drop features whose most frequent value covers ≥ this fraction of rows (must be in `[0.0, 1.0)`) |
| `max_correlation` | float | `1.0` | Drop one of any two features whose Cramér's V exceeds this value (must be in `(0.0, 1.0]`) |

### `[[tables]]` — At least one required

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name` | string | — | Unique table identifier |
| `path` | string | — | File path (CSV or Parquet), relative to config file |
| `primary_key` | string or list | — | Column(s) that uniquely identify each row |
| `timestamp_column` | string | `null` | Required for `latest` aggregations, windowed aggregations (`window_days`), and `sequence_features` |

### `[[relations]]` — Optional

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `parent` | string | — | Parent table name |
| `child` | string | — | Child table name |
| `parent_key` | string or list | — | Join key(s) on the parent |
| `child_key` | string or list | — | Join key(s) on the child |
| `kind` | string | `"one_to_many"` | `"one_to_one"` or `"one_to_many"` |
| `aggregations` | list of objects | `[]` | `one_to_many` relations require `aggregations` and/or `sequence_features` |
| `sequence_features` | list of strings | `[]` | Time-based features computed from the child's `timestamp_column`: `recency`, `frequency`, `time_span`, `gap_mean`, `gap_std`, `acceleration` |

### Aggregation object

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `op` | string | Yes | `count`, `nunique`, `sum`, `mean`, `min`, `max`, or `latest` |
| `column` | string | Only if op ≠ count | Column to aggregate |
| `name` | string | No | Custom name for the output column (auto-generated if omitted) |
| `window_days` | int | No | Only aggregate child rows within the last N days before the latest event (requires the child's `timestamp_column`; must be > 0) |

> `latest` requires the child table's `timestamp_column`.

---

## 14. Python API Reference

### `load_project(path) → ProjectSpec`

Load and validate a TOML configuration file.

```python
from auto_bayesian import load_project

project = load_project("examples/lead_scoring.toml")
```

### `build_project(...) → ProjectSpec`

Build a project specification in pure Python (no TOML file needed). Tables can
be specified without file paths if you plan to pass DataFrames to `fit_tables`.

```python
from auto_bayesian import build_project

project = build_project(
    root_table="leads",
    target_column="converted",
    positive_label="1",
    tables=[
        {"name": "leads", "primary_key": "lead_id"},
    ],
    numeric_bins=5,
    max_categories=20,
    test_fraction=0.2,
    random_seed=7,
    output_dir="artifacts",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root_table` | str | — | Name of the root table |
| `target_column` | str | — | Column to predict |
| `tables` | list[dict] | — | Table definitions |
| `relations` | list[dict] | `None` | Relation definitions |
| `positive_label` | str | `"1"` | Positive class value |
| `task_type` | str | `"classification"` | `"classification"` or `"next_best_action"` |
| `action_column` | str | `None` | Action column (required for `next_best_action`) |
| `output_dir` | str/Path | `"artifacts"` | Output directory |
| `random_seed` | int | `7` | Random seed |
| `test_fraction` | float | `0.2` | Validation fraction |
| `numeric_bins` | int | `5` | Numeric bins |
| `binning` | str | `"quantile"` | Numeric binning strategy (`"quantile"` or `"supervised"`) |
| `selection_metric` | str | `"roc_auc"` | Candidate selection metric (`"roc_auc"` or `"pr_auc"`) |
| `max_categories` | int | `20` | Max categories |
| `outlier_method` | str | `None` | Outlier removal method (`"iqr"`) |
| `outlier_iqr_factor` | float | `1.5` | IQR multiplier for outlier removal |
| `drop_duplicates` | bool | `False` | Drop duplicate training rows |
| `min_variance_fraction` | float | `0.0` | Low-variance feature drop threshold |
| `max_correlation` | float | `1.0` | Correlated feature drop threshold (Cramér's V) |
| `root` | str/Path | `"."` | Root directory for path resolution |

### `fit_project(project) → AutoBayesModel`

Train a model from a project that has file paths in its table specs.

```python
from auto_bayesian import fit_project, load_project

project = load_project("config.toml")
model = fit_project(project)
```

### `fit_tables(project, tables) → AutoBayesModel`

Train a model from in-memory DataFrames.

```python
from auto_bayesian import build_project, fit_tables

project = build_project(...)
tables = {"leads": pd.read_csv("leads.csv")}
model = fit_tables(project, tables)
```

### `materialize_project(project, tables=None) → pd.DataFrame`

Only do the materialization step (join + aggregate) without training.

```python
from auto_bayesian import materialize_project, load_project

project = load_project("config.toml")
flat_df = materialize_project(project)
```

### `generate_explanation(model, *, output_path, title) → Path`

Write a Markdown file with a Mermaid diagram of the Bayesian network.

```python
from auto_bayesian import generate_explanation

report_path = generate_explanation(
    model,
    output_path="explanation.md",
    title="My Model Network",
)
print(f"Diagram saved to {report_path}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `AutoBayesModel` | — | A trained model |
| `output_path` | str/Path | `"explanation.md"` | Where to write the Markdown file |
| `title` | str | `"Bayesian Network Explanation"` | Heading shown at the top |

The generated Markdown contains:
- A **Mermaid** `flowchart` of the network, with the target node highlighted
- A plain-language list of the learned relationships (which variables directly
  influence which)

### `to_mermaid(model, *, direction="TD") → str`

Return just the Mermaid flowchart source describing the network structure. Each
node is a variable and each edge `A --> B` means *A directly influences B*.

```python
from auto_bayesian import to_mermaid

print(to_mermaid(model))          # top-down layout
print(to_mermaid(model, direction="LR"))  # left-to-right
```

### `AutoBayesModel`

The trained model object.

| Method | Returns | Description |
|--------|---------|-------------|
| `predict_proba(df)` | `pd.Series` | Probability of the positive class |
| `predict(df, threshold=None)` | `pd.Series` | Binary 0/1 predictions (uses the tuned `report.threshold` when `threshold` is `None`) |
| `evaluate(df, threshold=None)` | `BinaryMetrics` | ROC-AUC, PR-AUC, precision/recall/F1 (at the tuned threshold) and positive rate for a labeled frame |
| `predict_next_best_action(df)` | `pd.DataFrame` | `recommended_action` + `expected_probability` per row (requires a `next_best_action` model) |
| `describe()` | `ModelReport` | Model metadata (candidate name, metrics, edges) |
| `save(output_dir)` | `None` | Persist model to disk |
| `AutoBayesModel.load(output_dir)` | `AutoBayesModel` | Load a saved model |

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `network` | pgmpy BayesianNetwork | The trained DAG with CPDs |
| `preprocessor` | `DataPreprocessor` | Fitted preprocessing pipeline |
| `report` | `ModelReport` | Full report with all candidate results |
| `materialized_frame` | `pd.DataFrame` or `None` | The flat table used for training |

### `ModelReport`

| Field | Type | Description |
|-------|------|-------------|
| `selected_candidate` | str | `"naive_bayes"`, `"tan"`, or `"hill_climb"` |
| `roc_auc` | float | Validation ROC-AUC of the selected candidate |
| `pr_auc` | float | Validation PR-AUC (average precision) of the selected candidate |
| `log_loss` | float | Validation log-loss of the selected candidate |
| `threshold` | float | F1-optimal decision threshold (used by `predict`/`evaluate`) |
| `precision` | float | Validation precision at the tuned threshold |
| `recall` | float | Validation recall at the tuned threshold |
| `f1` | float | Validation F1 at the tuned threshold |
| `target_column` | str | Name of the target column |
| `positive_label` | str | The positive class label |
| `edges` | list[tuple[str, str]] | Edges in the selected DAG |
| `candidates` | list[CandidateReport] | All three candidates, sorted by performance |
| `task_type` | str | `"classification"` or `"next_best_action"` |
| `action_column` | str or None | Action column when `task_type = "next_best_action"` |

### `CandidateReport`

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Candidate name |
| `roc_auc` | float | Validation ROC-AUC |
| `pr_auc` | float | Validation PR-AUC (average precision) |
| `log_loss` | float | Validation log-loss |
| `edges` | list[tuple[str, str]] | Edges in this candidate's DAG |

---

## 15. CLI Reference

Install and run with:

```bash
uv sync
uv run auto-bayesian <command> [args]
```

### `validate-schema <config.toml>`

Parse and validate the configuration file. Prints a JSON summary of tables,
relations, root table, and target column.

```bash
uv run auto-bayesian validate-schema examples/lead_scoring.toml
```

### `materialize <config.toml> [--output path]`

Join and aggregate all tables into a single flat file. Outputs a Parquet file by
default to `{output_dir}/materialized.parquet`.

```bash
uv run auto-bayesian materialize examples/lead_scoring.toml
uv run auto-bayesian materialize examples/lead_scoring.toml --output my_table.csv
```

### `train <config.toml>`

Run the full pipeline: materialize → preprocess → train three candidates →
select the best → save to `output_dir`.

```bash
uv run auto-bayesian train examples/lead_scoring.toml
```

**Output files** (in `output_dir`):

| File | Format | Contents |
|------|--------|----------|
| `model.pkl` | Pickle | The full model (network + preprocessor + report) |
| `metrics.json` | JSON | Performance metrics of all candidates |
| `network.json` | JSON | Target, positive label, and edges of the selected DAG |
| `materialized.parquet` | Parquet | The flat training table |

### `predict <model_dir> <input_file> [--output path]`

Score new data using a saved model. The input must be an already-materialized
table (same columns as the training data).

```bash
uv run auto-bayesian predict artifacts/lead_scoring artifacts/lead_scoring/materialized.parquet
uv run auto-bayesian predict artifacts/lead_scoring new_data.csv --output scores.csv
```

### `explain <model_dir> [--output path] [--title text]`

Write a Markdown file with a Mermaid diagram of the network from a trained model.

```bash
uv run auto-bayesian explain artifacts/lead_scoring
uv run auto-bayesian explain artifacts/lead_scoring --output my_network.md --title "Lead Scoring Network"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `<model_dir>/explanation.md` | Where to write the Markdown file |
| `--title` | `"Bayesian Network Explanation"` | Heading shown at the top |

---

## 16. Full Tutorial — Lead Scoring (TOML + CLI)

This tutorial uses the bundled example data in `examples/data/`.

### The Data

**`leads.csv`** — 10 sales leads

| lead_id | customer_id | age | source   | converted |
|---------|-------------|-----|----------|-----------|
| 1       | 101         | 24  | web      | 0         |
| 2       | 102         | 45  | referral | 1         |
| ...     | ...         | ... | ...      | ...       |

**`customers.csv`** — Customer metadata

| customer_id | region | tenure_months | segment    |
|-------------|--------|---------------|------------|
| 101         | north  | 3             | small      |
| 102         | west   | 18            | enterprise |
| ...         | ...    | ...           | ...        |

**`interactions.csv`** — Interaction events (many per lead)

| interaction_id | lead_id | event_time | channel | days_to_close |
|----------------|---------|------------|---------|---------------|
| 1              | 1       | 2026-01-02 | email   | 18            |
| 2              | 1       | 2026-01-08 | web     | 20            |
| ...            | ...     | ...        | ...     | ...           |

### Step 1: Write the Config

```toml
# examples/lead_scoring.toml

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

### Step 2: Validate

```bash
uv run auto-bayesian validate-schema examples/lead_scoring.toml
```

Output:
```json
{
  "tables": ["customers", "interactions", "leads"],
  "relations": ["leads->customers", "leads->interactions"],
  "root_table": "leads",
  "target_column": "converted"
}
```

### Step 3: Materialize (Optional — inspect the flat table)

```bash
uv run auto-bayesian materialize examples/lead_scoring.toml
```

This creates `artifacts/lead_scoring/materialized.parquet` with columns:
`lead_id`, `customer_id`, `age`, `source`, `converted`, `customers__region`,
`customers__tenure_months`, `customers__segment`, `interaction_count`,
`channel_count`, `mean_days_to_close`, `latest_channel`.

### Step 4: Train

```bash
uv run auto-bayesian train examples/lead_scoring.toml
```

Output:
```json
{
  "output_dir": "artifacts/lead_scoring",
  "selected_candidate": "naive_bayes",
  "roc_auc": 0.92,
  "log_loss": 0.45
}
```

### Step 5: Predict

```bash
uv run auto-bayesian predict artifacts/lead_scoring artifacts/lead_scoring/materialized.parquet
```

Output:
```csv
probability,prediction
0.23,0
0.87,1
...
```

### Step 6: Inspect the Model (Python)

```python
from auto_bayesian.model import AutoBayesModel

model = AutoBayesModel.load("artifacts/lead_scoring")

# See which candidate won
print(model.report.selected_candidate)  # e.g., "naive_bayes"

# See the DAG edges
for src, dst in model.report.edges:
    print(f"  {src} → {dst}")

# See all candidate results
for c in model.report.candidates:
    print(f"  {c.name}: AUC={c.roc_auc:.4f}, LogLoss={c.log_loss:.4f}")

# Print the CPD tables (full interpretability!)
for cpd in model.network.get_cpds():
    print(cpd)
```

---

## 17. Full Tutorial — Titanic Survival (Pure Python)

This tutorial shows how to use the library **without any TOML file**, entirely
from Python with in-memory DataFrames.

```python
import pandas as pd
from auto_bayesian import build_project, fit_tables

# --- 1. Load the data -------------------------------------------------------
raw = pd.read_csv("titanic.csv")
passengers = raw[
    ["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp",
     "Parch", "Fare", "Embarked"]
].copy()
passengers["Survived"] = passengers["Survived"].astype(str)

tables = {"passengers": passengers}

# --- 2. Define the project ---------------------------------------------------
project = build_project(
    root_table="passengers",
    target_column="Survived",
    positive_label="1",
    output_dir="artifacts/titanic",
    random_seed=7,
    test_fraction=0.2,
    numeric_bins=5,
    max_categories=10,
    tables=[
        {"name": "passengers", "primary_key": "PassengerId"},
    ],
)

# --- 3. Train the model ------------------------------------------------------
model = fit_tables(project, tables)

# --- 4. Inspect results ------------------------------------------------------
report = model.describe()
print(f"Best candidate: {report.selected_candidate}")
print(f"Validation ROC-AUC: {report.roc_auc:.4f}")
print(f"Validation Log-Loss: {report.log_loss:.4f}")

print("\nAll candidates:")
for c in report.candidates:
    print(f"  {c.name}: AUC={c.roc_auc:.4f}, LogLoss={c.log_loss:.4f}")

print("\nDAG edges:")
for src, dst in report.edges:
    print(f"  {src} → {dst}")

# --- 5. Make predictions -----------------------------------------------------
features = model.materialized_frame.drop(columns=["Survived"])
probabilities = model.predict_proba(features)
predictions = model.predict(features, threshold=0.5)

print(f"\nSample predictions:")
print(pd.DataFrame({
    "PassengerId": features["PassengerId"].values[:5],
    "probability": probabilities.values[:5],
    "prediction": predictions.values[:5],
}))

# --- 6. Save and reload ------------------------------------------------------
model.save("artifacts/titanic")

from auto_bayesian.model import AutoBayesModel
restored = AutoBayesModel.load("artifacts/titanic")
print(f"\nRestored model candidate: {restored.report.selected_candidate}")

# --- 7. Full interpretability — print CPD tables -----------------------------
print("\n=== Conditional Probability Tables ===")
for cpd in restored.network.get_cpds():
    print(cpd)
    print()
```

**Expected output:**

```
Best candidate: tan
Validation ROC-AUC: 0.8234
Validation Log-Loss: 0.5012

All candidates:
  tan: AUC=0.8234, LogLoss=0.5012
  naive_bayes: AUC=0.8102, LogLoss=0.5234
  hill_climb: AUC=0.7891, LogLoss=0.5567

DAG edges:
  Survived → Sex
  Survived → Pclass
  Survived → Fare
  Sex → Age
  ...
```

---

## 18. Explainability (Mermaid Diagram)

**File:** `explain.py`

The `generate_explanation` function writes a single Markdown file that explains
the trained Bayesian Network's structure. It contains a **Mermaid** diagram of
the network plus a plain-language list of the learned relationships. Any Markdown
viewer that supports Mermaid (GitHub, GitLab, Obsidian, JupyterLab, VS Code, …)
renders the diagram automatically.

### What the File Contains

| Part | Description |
|------|-------------|
| **Header** | The target column and its positive label |
| **Network structure** | A Mermaid `flowchart` where each node is a variable and each arrow `A --> B` means *A directly influences B*. The target node is highlighted in red. |
| **Relationships** | A bullet list translating every edge into plain English (e.g. ``customer_state` directly predicts `is_late``). |

### Usage — Python

```python
from auto_bayesian import fit_project, load_project, generate_explanation

project = load_project("examples/lead_scoring.toml")
model = fit_project(project)

report_path = generate_explanation(
    model,
    output_path="artifacts/lead_scoring/explanation.md",
    title="Lead Scoring — Network",
)
print(f"Open {report_path} in any Mermaid-aware Markdown viewer.")
```

To get just the diagram source (e.g. to embed it elsewhere), use `to_mermaid`:

```python
from auto_bayesian import to_mermaid

print(to_mermaid(model))                   # top-down layout
print(to_mermaid(model, direction="LR"))   # left-to-right layout
```

### Usage — CLI

```bash
# Train first
uv run auto-bayesian train examples/lead_scoring.toml

# Generate the diagram
uv run auto-bayesian explain artifacts/lead_scoring

# Custom title and output path
uv run auto-bayesian explain artifacts/lead_scoring \
    --title "Lead Scoring Network" \
    --output network.md
```

### How the Diagram Is Built

The nodes and edges come directly from the selected network
(`model.network.nodes()` and `model.network.edges()`). Each variable becomes a
Mermaid node with a stable id, the target gets a `target` CSS class, and every
directed edge is rendered as an arrow. No external runtime or CDN is required —
the rendering is handled by whichever Markdown viewer you open the file in.

---

## 19. Architecture & File Map

```
auto-bayesian/
├── src/auto_bayesian/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # Entry point: python -m auto_bayesian
│   ├── schema.py            # Configuration parsing, validation, data classes
│   ├── materialize.py       # Multi-table joins and aggregations
│   ├── preprocess.py        # Discretization, binning, category encoding
│   ├── automl.py            # Orchestrator: split, train, select best
│   ├── engine_pgmpy.py      # Bayesian Network training and inference (pgmpy)
│   ├── model.py             # AutoBayesModel: predict, save, load
│   ├── explain.py           # Mermaid diagram of the network (Markdown)
│   └── cli.py               # Command-line interface
├── examples/
│   ├── lead_scoring.toml    # Example TOML configuration
│   └── data/                # Example CSV data files
├── tests/                   # Pytest test suite
├── pyproject.toml           # Dependencies and build config
└── README.md                # Quick-start README
```

### Module Dependency Graph

```
cli.py
  ├──► explain.py ──► model.py
  └──► automl.py
         ├──► materialize.py ──► schema.py
         ├──► preprocess.py  ──► schema.py
         ├──► engine_pgmpy.py
         └──► model.py
                ├──► engine_pgmpy.py
                └──► preprocess.py
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 2.2.0 | DataFrame manipulation |
| `numpy` | ≥ 2.2.0 | Numerical operations (binning, quantiles) |
| `pgmpy` | ≥ 1.0.0 | Bayesian Network modeling, structure learning, inference |
| `pyarrow` | ≥ 18.0.0 | Parquet file read/write |

---

## 20. Limitations

| Limitation | Details |
|-----------|---------|
| **Binary classification only** | The target must have exactly 2 classes |
| **Local files only** | CSV and Parquet from disk (or in-memory DataFrames) |
| **Single engine** | Only pgmpy is supported |
| **Tree-shaped relations** | Each table can have at most one parent |
| **Target in root table** | The target column must exist in the root table |
| **Explicit aggregations** | `one_to_many` relations require you to specify aggregations |
| **CLI predict needs materialized data** | The `predict` CLI command expects a pre-materialized flat table |
| **Row-by-row inference** | Variable Elimination is run per row, which is slow for very large datasets |

---

## 21. Glossary

| Term | Definition |
|------|-----------|
| **Bayesian Network** | A probabilistic graphical model represented as a DAG where nodes are variables and edges represent direct dependencies, with a CPD at each node |
| **DAG** | Directed Acyclic Graph — a graph with directed edges and no cycles |
| **CPD** | Conditional Probability Distribution — a table storing P(node \| parents) |
| **BDeu** | Bayesian Dirichlet equivalent uniform — a prior that spreads pseudo-counts uniformly across all states |
| **Naive Bayes** | A Bayesian Network where all features are conditionally independent given the target |
| **TAN** | Tree-Augmented Naive Bayes — like Naive Bayes but features can form a tree of dependencies |
| **Hill Climb** | A greedy structure-learning algorithm that adds/removes/reverses edges to maximize a score |
| **BIC** | Bayesian Information Criterion — a score that balances data fit against model complexity |
| **ROC-AUC** | Area Under the Receiver Operating Characteristic Curve — measures ranking quality |
| **Log-Loss** | Binary cross-entropy — measures probability calibration quality |
| **Materialization** | The process of joining multiple relational tables into one flat DataFrame |
| **Quantile binning** | Discretizing a continuous variable by splitting at percentile boundaries |
| **Variable Elimination** | An exact inference algorithm that computes posterior probabilities by summing out variables |
| **Stratified split** | A train/test split that preserves the class distribution in both subsets |
| **Positive label** | The target value that represents the "event" or "yes" class |
| **Prior (Bayesian)** | A probability distribution representing beliefs before seeing data |
| **Posterior** | Updated probability distribution after incorporating observed data |
| **Likelihood** | The probability of the observed data given a hypothesis |
| **Equivalent sample size** | The total pseudo-count strength of the BDeu prior |
| **Mutual information** | A measure of the statistical dependence between two variables |
| **Maximum spanning tree** | A tree connecting all nodes that maximizes the sum of edge weights |
| **XAI** | Explainable Artificial Intelligence — techniques that make model decisions interpretable to humans |
| **Confusion matrix** | A 2×2 table showing TP, FP, FN, TN counts for a binary classifier at a given threshold |
| **Precision** | TP / (TP + FP) — of all predicted positives, how many are actually positive |
| **Recall** | TP / (TP + FN) — of all actual positives, how many did the model find |
| **F1 score** | Harmonic mean of precision and recall: 2 × P × R / (P + R) |

---

*Generated for auto-bayesian v0.1.0*
