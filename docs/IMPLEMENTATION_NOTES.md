# GNN4TaskPlan — Implementation Notes

This document records implementation-level observations from tracing the released GNN4TaskPlan codebase. It is intended as a technical companion to the main README.

> **Scope:** These notes describe behavior visible in the released implementation. They are not claims of ownership of the original method, code, datasets, or reported results.

## 1. End-to-End Pipeline

The repository supports both training-free and training-based approaches for task/tool retrieval in LLM-based agents.

At a high level:

```text
User Request
    ↓
LLM Task Decomposition
    ↓
Task Steps / Candidate Tools
    ↓
Task Graph
    ↓
Language / Graph Representations
    ↓
Task-Tool Retrieval
    ↓
Predicted Task Nodes + Links
    ↓
Structural Evaluation
```

The training-free path includes Direct, GraphSearch, and SGC. The training-based path includes several GNN variants combined with language-model representations.

---

## 2. Direct Planning

The Direct implementation asks the language model to produce a structured representation of the predicted plan.

The requested fields include:

- `task_steps`
- `task_nodes`
- `arguments`
- `task_links`

The implementation also contains handling for malformed model outputs so that inference can attempt to recover from outputs that do not immediately match the expected structure.

The important distinction is that Direct does not use graph-aware propagation during retrieval; it relies on the LLM's direct plan generation.

---

## 3. GraphSearch

The GraphSearch implementation uses graph structure explicitly during task retrieval.

The released code contains:

- Greedy search
- Adaptive search
- Beam search

The graph is therefore used as a search constraint rather than only as an additional representation.

A practical implication is that graph connectivity can affect which tools remain available as the plan progresses.

---

## 4. SGC Representation Propagation

The training-free SGC path combines language-model representations with information propagated through the task graph.

The graph adjacency is normalized in the form:

```text
A_norm = D^(-1/2) A D^(-1/2)
```

The propagated representation is then combined with the original representation using:

```text
H = αX + (1 − α) A_norm X
```

where:

- `X` is the original representation,
- `A_norm X` is the graph-propagated representation,
- `α` controls the contribution of the original representation.

The training-free implementation evaluates multiple values of `α` in the range `0.50` to `1.00`.

This makes the SGC path a useful place to study the trade-off between language-derived information and graph-derived information.

---

## 5. Sequential and Graph-Constrained Retrieval

The retrieval process is sequential.

For the first task step, the selected tool can be drawn from the full candidate set. After a tool has been selected, the candidate set for the following step is restricted to the outgoing neighbors of the previously selected tool.

Conceptually:

```text
Step 1:
all candidate tools
        ↓
selected tool t1

Step 2:
outgoing_neighbors(t1)
        ↓
selected tool t2

Step 3:
outgoing_neighbors(t2)
        ↓
selected tool t3
```

This means graph structure is not merely a feature used for scoring. It can directly constrain which candidates are considered next.

A consequence visible from the implementation is that a missing or incorrect edge can prevent an otherwise plausible next tool from being selected. When no valid graph-constrained candidate is available, the relevant step can be skipped.

---

## 6. Training-Based Retrieval

The training-based path constructs positive and negative step-tool candidates.

The scoring pipeline combines language-model and graph-based representations, depending on the configured training setup.

The released repository includes GNN variants such as:

- SGC
- GCN
- GAT
- GraphSAGE / SAGE
- GIN
- TransformerConv

The training objective is a pairwise ranking formulation using positive and negative candidates. The implementation uses a softplus form equivalent to:

```text
L = mean(softplus(score_negative − score_positive))
```

The objective therefore encourages positive tool candidates to receive higher scores than negative candidates.

---

## 7. Training Data Structure

The training-data preparation places a structural restriction on the examples used by the training path.

The relevant examples favor:

- single-tool plans
- multi-tool plans forming a single chain

Branching structures are filtered out of this training path.

This matters when interpreting the training-based setup: the training data does not represent the full range of possible branching task graphs.

---

## 8. Evaluation Metrics

The repository evaluates generated plans using structural metrics.

### Node-F1

Measures overlap between predicted and target task nodes.

### Link-F1

Measures overlap between predicted and target task links.

### Plan Accuracy

The evaluation code treats a prediction as correct when:

```text
Node-F1 >= 0.99
```

### Node Hallucination

Measures node predictions that do not correspond to the expected task-node structure.

### Link Hallucination

Measures incorrect or unsupported predicted links.

The evaluation output can include micro- and macro-level hallucination measures.

---

## 9. Prediction Coverage

The original evaluation implementation includes a `remove_non_pred=1` setting.

With that behavior enabled, test examples that do not have a corresponding prediction record are excluded from the evaluation set.

The independent analysis utility added to this repository separates two related quantities:

```text
prediction-record coverage
    = test examples with a prediction record

non-empty-step coverage
    = predictions containing task steps
```

Keeping these separate makes it easier to distinguish missing predictions from predictions that exist but contain an empty task-step sequence.

---

## 10. Original vs. Added Components

### From the original GNN4TaskPlan repository

- Core model implementations
- Direct / GraphSearch / SGC methods
- Training-based GNN implementations
- Dataset and preprocessing code
- Prediction files
- Original evaluation implementation
- Experiment scripts

### Added in this implementation study

- `docs/IMPLEMENTATION_NOTES.md`
- `analysis/evaluate_provided_predictions.py`
- `analysis/test_evaluation.py`
- README documentation describing the implementation study and analysis scope

The original model and dataset files are intentionally left unchanged.

---

## 11. Reproducibility Considerations

The upstream environment uses older pinned versions of several machine-learning dependencies, including PyTorch, PyTorch Geometric, FastChat, and vLLM.

Some experiments also depend on substantially larger GPU resources than a consumer 6 GB GPU.

For that reason, this repository does not present a full hardware-matched reproduction of the original benchmark.

The independent evaluator is deliberately lightweight: it works on prediction artifacts already present in the repository and does not launch new LLM inference.

---

## 12. Implementation-Level Research Questions

Tracing the implementation suggests several questions that could be investigated in future work:

1. How sensitive is plan quality to the balance between graph-propagated and language-derived representations?
2. How robust is graph-constrained retrieval to missing or noisy graph edges?
3. How does retrieval behave as graph depth and branching factor increase?
4. Can graph reasoning be invoked selectively when the underlying language model is uncertain about the next tool?

These are research directions motivated by the implementation. No new empirical claim is made here.
