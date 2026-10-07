#!/usr/bin/env python3
"""Lightweight, dependency-free audit of GNN4TaskPlan prediction artifacts.

This script intentionally avoids the research project's NumPy/HuggingFace
runtime so that supplied prediction files can be inspected independently.
It mirrors the metric definitions used by evaluate.py for the default chain mode
and reports both prediction-record coverage and non-empty-step coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def f1_score(pred: list[str], gt: list[str]) -> float:
    if not pred or not gt:
        return 0.0
    inter = set(pred) & set(gt)
    precision = len(inter) / len(pred)
    recall = len(inter) / len(gt)
    return 2.0 * precision * recall / (precision + recall + 1e-9)


def reformat_nodes(content: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for node in content.get("task_nodes", []):
        if isinstance(node, dict) and "task" in node:
            value = node["task"]
            if isinstance(value, list):
                result.extend(str(v) for v in value)
            elif isinstance(value, str):
                result.append(value)
        elif isinstance(node, dict) and isinstance(node.get("name"), str):
            result.append(node["name"])
        elif isinstance(node, str):
            result.append(node)
    return result


def reformat_links(content: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for link in content.get("task_links", []):
        if isinstance(link, dict):
            source = link.get("source")
            target = link.get("target")
            if isinstance(source, str) and isinstance(target, str) and source and target:
                result.append(f"{source}, {target}")
        # Match the upstream fallback format without using eval().
        elif isinstance(link, list) and len(link) == 2:
            nodes = reformat_nodes(content)
            indices: list[int] = []
            for raw in link:
                token = str(raw).replace("Step ", "").strip()
                if token.isdigit():
                    indices.append(int(token) - 1)
            if len(indices) == 2 and all(0 <= i < len(nodes) for i in indices):
                result.append(f"{nodes[indices[0]]}, {nodes[indices[1]]}")
    return result


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if "id" in record:
                records[str(record["id"])] = record
    return records


def hallucination_rate(solution: list[str], valid_items: set[str]) -> tuple[float, float]:
    if not solution:
        return 0.0, 0.0
    invalid = sum(1 for item in solution if item not in valid_items)
    micro = invalid / len(solution)
    macro = 1.0 if invalid else 0.0
    return micro, macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="huggingface")
    parser.add_argument("--llm", default="CodeLlama-13b")
    parser.add_argument("--method", default="direct")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    split_path = root / "data" / args.dataset / "split_ids.json"
    gt_path = root / "data" / args.dataset / "data.json"
    tool_path = root / "data" / args.dataset / "tool_desc.json"
    graph_path = root / "data" / args.dataset / "graph_desc.json"
    pred_path = root / "prediction" / args.dataset / args.llm / f"{args.method}.json"

    for path in (split_path, gt_path, tool_path, graph_path, pred_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    chain_ids = [str(x) for x in split["test_ids"]["chain"]]
    gt_records = load_jsonl(gt_path)
    pred_records = load_jsonl(pred_path)
    tool_desc = json.loads(tool_path.read_text(encoding="utf-8"))
    graph_desc = json.loads(graph_path.read_text(encoding="utf-8"))

    valid_nodes = {str(node["id"]) for node in tool_desc["nodes"]}
    valid_links = {
        f"{link['source']}, {link['target']}" for link in graph_desc["links"]
    }

    evaluated_ids: list[str] = []
    details: list[dict[str, Any]] = []
    for data_id in chain_ids:
        pred = pred_records.get(data_id)
        gt = gt_records.get(data_id)
        if pred is None or gt is None:
            continue

        pred_nodes = reformat_nodes(pred)
        pred_links = reformat_links(pred)
        gt_nodes = reformat_nodes(gt)
        gt_links = reformat_links(gt)

        node_f1 = f1_score(pred_nodes, gt_nodes)
        link_f1 = f1_score(pred_links, gt_links)
        exact = float(node_f1 >= 0.99)
        node_micro, node_macro = hallucination_rate(pred_nodes, valid_nodes)
        link_micro, link_macro = hallucination_rate(pred_links, valid_links)

        evaluated_ids.append(data_id)
        details.append(
            {
                "id": data_id,
                "node_f1": round(node_f1, 4),
                "link_f1": round(link_f1, 4),
                "accuracy": exact,
                "node_hallucination_micro": round(node_micro, 4),
                "node_hallucination_macro": round(node_macro, 4),
                "link_hallucination_micro": round(link_micro, 4),
                "link_hallucination_macro": round(link_macro, 4),
                "pred_nodes": pred_nodes,
                "gt_nodes": gt_nodes,
                "pred_links": pred_links,
                "gt_links": gt_links,
            }
        )

    if not evaluated_ids:
        raise RuntimeError("No evaluable predictions were found for the selected split.")

    def mean(key: str) -> float:
        return round(sum(float(item[key]) for item in details) / len(details), 4)

    summary = {
        "dataset": args.dataset,
        "llm": args.llm,
        "method": args.method,
        "test_chain_ids": len(chain_ids),
        "prediction_records": len(pred_records),
        "evaluated_predictions": len(evaluated_ids),
        "prediction_record_coverage_over_chain_split": round(len(evaluated_ids) / len(chain_ids), 4),
        "non_empty_step_predictions": sum(1 for data_id in evaluated_ids if pred_records[data_id].get("task_steps")),
        "non_empty_step_coverage_over_chain_split": round(sum(1 for data_id in evaluated_ids if pred_records[data_id].get("task_steps")) / len(chain_ids), 4),
        "metrics": {
            "node_f1": mean("node_f1"),
            "link_f1": mean("link_f1"),
            "accuracy": mean("accuracy"),
            "node_hallucination_micro": mean("node_hallucination_micro"),
            "node_hallucination_macro": mean("node_hallucination_macro"),
            "link_hallucination_micro": mean("link_hallucination_micro"),
            "link_hallucination_macro": mean("link_hallucination_macro"),
        },
        "worst_node_f1": sorted(details, key=lambda x: x["node_f1"])[: args.top_k],
        "worst_link_f1": sorted(details, key=lambda x: x["link_f1"])[: args.top_k],
    }

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Evaluated prediction records: {summary['evaluated_predictions']}/{summary['test_chain_ids']}")
    print(f"Prediction-record coverage: {summary['prediction_record_coverage_over_chain_split']:.4f}")
    print(f"Non-empty step coverage: {summary['non_empty_step_coverage_over_chain_split']:.4f}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved analysis report to {args.save}")


if __name__ == "__main__":
    main()
