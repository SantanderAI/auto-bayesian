# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from auto_bayesian.automl import fit_project
from auto_bayesian.explain import generate_explanation
from auto_bayesian.materialize import materialize_project
from auto_bayesian.model import AutoBayesModel
from auto_bayesian.schema import load_project


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auto-bayesian")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-schema", help="Validate the project config.")
    validate_parser.add_argument("config", type=Path)
    validate_parser.set_defaults(handler=_handle_validate)

    materialize_parser = subparsers.add_parser("materialize", help="Build the modeling table.")
    materialize_parser.add_argument("config", type=Path)
    materialize_parser.add_argument("--output", type=Path, default=None)
    materialize_parser.set_defaults(handler=_handle_materialize)

    train_parser = subparsers.add_parser("train", help="Train and persist a model.")
    train_parser.add_argument("config", type=Path)
    train_parser.set_defaults(handler=_handle_train)

    predict_parser = subparsers.add_parser("predict", help="Score a materialized input file.")
    predict_parser.add_argument("model_dir", type=Path)
    predict_parser.add_argument("input_file", type=Path)
    predict_parser.add_argument("--output", type=Path, default=None)
    predict_parser.set_defaults(handler=_handle_predict)

    explain_parser = subparsers.add_parser(
        "explain", help="Generate a Mermaid diagram of the Bayesian network (Markdown)."
    )
    explain_parser.add_argument("model_dir", type=Path)
    explain_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path (default: <model_dir>/explanation.md)",
    )
    explain_parser.add_argument(
        "--title",
        type=str,
        default="Bayesian Network Explanation",
        help="Heading shown at the top of the document.",
    )
    explain_parser.set_defaults(handler=_handle_explain)

    return parser


def _handle_validate(args: argparse.Namespace) -> None:
    project = load_project(args.config)
    print(
        json.dumps(
            {
                "tables": sorted(project.tables),
                "relations": [
                    f"{relation.parent}->{relation.child}" for relation in project.relations
                ],
                "root_table": project.task.root_table,
                "target_column": project.task.target_column,
            },
            indent=2,
        )
    )


def _handle_materialize(args: argparse.Namespace) -> None:
    project = load_project(args.config)
    frame = materialize_project(project)
    output_path = args.output or project.run.output_dir / "materialized.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_frame(frame, output_path)
    print(output_path)


def _handle_train(args: argparse.Namespace) -> None:
    project = load_project(args.config)
    model = fit_project(project)
    model.save(project.run.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(project.run.output_dir),
                "selected_candidate": model.report.selected_candidate,
                "roc_auc": model.report.roc_auc,
                "pr_auc": model.report.pr_auc,
                "log_loss": model.report.log_loss,
                "threshold": model.report.threshold,
                "f1": model.report.f1,
            },
            indent=2,
        )
    )


def _handle_predict(args: argparse.Namespace) -> None:
    model = AutoBayesModel.load(args.model_dir)
    frame = _read_frame(args.input_file)
    if model.report.task_type == "next_best_action" and model.report.action_column:
        output = model.predict_next_best_action(frame)
    else:
        probabilities = model.predict_proba(frame)
        cutoff = model.report.threshold
        output = pd.DataFrame(
            {"probability": probabilities, "prediction": (probabilities >= cutoff).astype(int)}
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_frame(output, args.output)
        print(args.output)
        return
    print(output.to_csv(index=False))


def _handle_explain(args: argparse.Namespace) -> None:
    model = AutoBayesModel.load(args.model_dir)
    output = args.output or args.model_dir / "explanation.md"
    path = generate_explanation(
        model,
        output_path=output,
        title=args.title,
    )
    print(path)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
        return
    frame.to_parquet(path, index=False)
