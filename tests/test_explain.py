# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from auto_bayesian.explain import generate_explanation, to_mermaid


class _FakeNetwork:
    def __init__(self, nodes: list[str], edges: list[tuple[str, str]]) -> None:
        self._nodes = list(nodes)
        self._edges = list(edges)

    def nodes(self) -> list[str]:
        return self._nodes

    def edges(self) -> list[tuple[str, str]]:
        return self._edges


class _FakeReport:
    def __init__(self, target_column: str, positive_label: str) -> None:
        self.target_column = target_column
        self.positive_label = positive_label


class _FakeModel:
    def __init__(self, network: _FakeNetwork, report: _FakeReport) -> None:
        self.network = network
        self.report = report


def _make_model() -> _FakeModel:
    network = _FakeNetwork(
        nodes=["converted", "age", "region"],
        edges=[("age", "converted"), ("converted", "region")],
    )
    report = _FakeReport(target_column="converted", positive_label="1")
    return _FakeModel(network, report)


def test_to_mermaid_marks_target_and_renders_edges() -> None:
    diagram = to_mermaid(_make_model())

    assert diagram.startswith("flowchart TD")
    assert '"converted"]:::target' in diagram
    assert diagram.count("-->") == 2
    assert "classDef target" in diagram


def test_to_mermaid_direction_is_configurable() -> None:
    assert to_mermaid(_make_model(), direction="LR").startswith("flowchart LR")


def test_generate_explanation_writes_markdown_with_mermaid(tmp_path: Path) -> None:
    output = tmp_path / "explanation.md"
    path = generate_explanation(_make_model(), output_path=output, title="My Network")

    assert path == output.resolve()
    text = path.read_text(encoding="utf-8")
    assert "# My Network" in text
    assert "```mermaid" in text
    assert "flowchart TD" in text
    assert "`age` directly predicts `converted`" in text
    assert "`region` depends on the outcome `converted`" in text
