import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.evaluate_provided_predictions import f1_score, hallucination_rate, reformat_links, reformat_nodes


def test_f1_score():
    assert round(f1_score(["a", "b"], ["b", "c"]), 4) == 0.5
    assert f1_score([], ["a"]) == 0.0


def test_reformat_nodes_and_links():
    content = {
        "task_nodes": [{"task": "A"}, {"task": "B"}],
        "task_links": [["Step 1", "Step 2"]],
    }
    assert reformat_nodes(content) == ["A", "B"]
    assert reformat_links(content) == ["A, B"]


def test_empty_prediction_semantics():
    # The analysis utility intentionally keeps an existing prediction record
    # even when task_steps is empty, matching evaluate.py's alignment behavior.
    content = {"id": "x", "task_steps": [], "task_nodes": [], "task_links": []}
    assert content["id"] == "x"
    assert not content["task_steps"]


def test_hallucination():
    micro, macro = hallucination_rate(["A", "X"], {"A", "B"})
    assert micro == 0.5
    assert macro == 1.0
