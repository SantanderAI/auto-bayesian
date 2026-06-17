# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from auto_bayesian.model import AutoBayesModel


def to_mermaid(model: AutoBayesModel, *, direction: str = "TD") -> str:
    """Return a Mermaid flowchart describing the Bayesian network structure.

    Each node is a variable and each directed edge ``A --> B`` means *A directly
    influences B* in the learned network. The target variable is highlighted.

    Parameters
    ----------
    model:
        A trained ``AutoBayesModel``.
    direction:
        Mermaid layout direction (``"TD"``, ``"LR"``, ``"BT"``, ``"RL"``).
    """
    network = model.network
    target = model.report.target_column
    nodes = list(network.nodes())
    edges = sorted(tuple(edge) for edge in network.edges())

    node_ids = {name: f"n{index}" for index, name in enumerate(nodes)}

    lines = [f"flowchart {direction}"]
    for name in nodes:
        label = str(name).replace('"', "'")
        suffix = ":::target" if name == target else ""
        lines.append(f'    {node_ids[name]}["{label}"]{suffix}')
    for parent, child in edges:
        if parent in node_ids and child in node_ids:
            lines.append(f"    {node_ids[parent]} --> {node_ids[child]}")
    lines.append("    classDef target fill:#e74c3c,stroke:#c0392b,color:#fff;")
    return "\n".join(lines)


def generate_explanation(
    model: AutoBayesModel,
    *,
    output_path: str | Path = "explanation.md",
    title: str = "Bayesian Network Explanation",
) -> Path:
    """Write a Markdown file with a Mermaid diagram of the Bayesian network.

    The document contains the network diagram plus a plain-language list of the
    learned relationships (which variables directly influence which).

    Parameters
    ----------
    model:
        A trained ``AutoBayesModel``.
    output_path:
        Where to write the Markdown file.
    title:
        Heading shown at the top of the document.

    Returns
    -------
    Path
        Resolved path to the generated Markdown file.
    """
    output = Path(output_path).resolve()
    diagram = to_mermaid(model)
    relationships = _describe_relationships(model)

    document = (
        f"# {title}\n\n"
        f"Predicting **`{model.report.target_column}`** "
        f"(positive label: `{model.report.positive_label}`).\n\n"
        "## Network structure\n\n"
        "Each arrow `A --> B` means *A directly influences B*. "
        "The highlighted node is the target.\n\n"
        f"```mermaid\n{diagram}\n```\n\n"
        "## Relationships\n\n"
        f"{relationships}\n"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def _describe_relationships(model: AutoBayesModel) -> str:
    target = model.report.target_column
    edges = sorted(tuple(edge) for edge in model.network.edges())
    if not edges:
        return "_No dependencies were learned between variables._"

    lines = []
    for parent, child in edges:
        if child == target:
            lines.append(f"- `{parent}` directly predicts `{child}`")
        elif parent == target:
            lines.append(f"- `{child}` depends on the outcome `{parent}`")
        else:
            lines.append(f"- `{parent}` influences `{child}`")
    return "\n".join(lines)
