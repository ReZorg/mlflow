"""
PLN rule: ModelComparisonRule

Infers which of two MLflow runs is probabilistically superior on a given
metric, given their logged metric values and confidence estimates.
"""

from __future__ import annotations

from typing import Iterator

from mlflow.cogmlflow.entities.atom import (
    ConceptNode,
    EvaluationLink,
    ImplicationLink,
    ListLink,
    NumberNode,
    PredicateNode,
    TruthValue,
)
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.entities.ontology import metric_pred_name, run_node_name
from mlflow.cogmlflow.pln.engine import InferenceResult, PLNRule


class ModelComparisonRule(PLNRule):
    """Infers a probabilistic superiority relation between two runs.

    For every pair of runs that share a metric, emits an
    ``ImplicationLink(run_a, run_b)`` with a TV reflecting the probability
    that run_a is better than run_b on the metric.

    Higher metric value is assumed to be better unless ``higher_is_better``
    is set to False.

    The confidence of the result is the geometric mean of the individual run
    confidence values (default 1.0 for observed metrics).
    """

    def __init__(self, metric_key: str, higher_is_better: bool = True) -> None:
        self._metric_key = metric_key
        self._higher_is_better = higher_is_better

    @property
    def name(self) -> str:
        return f"ModelComparisonRule[{self._metric_key}]"

    def apply(self, atomspace: AtomSpace) -> Iterator[InferenceResult]:
        # Collect all (run_id, value) pairs for the target metric
        pred_name = metric_pred_name(self._metric_key)
        run_values: dict[str, float] = {}

        for atom in atomspace:
            if not isinstance(atom, EvaluationLink):
                continue
            if len(atom.outgoing) < 2:
                continue
            pred = atom.outgoing[0]
            args = atom.outgoing[1]
            if not isinstance(pred, PredicateNode) or pred.name != pred_name:
                continue
            if not isinstance(args, ListLink) or len(args.outgoing) < 2:
                continue
            run_node = args.outgoing[0]
            val_node = args.outgoing[1]
            if not isinstance(run_node, ConceptNode) or not run_node.name.startswith("run:"):
                continue
            if not isinstance(val_node, NumberNode):
                continue
            run_id = run_node.name[len("run:") :]
            run_values[run_id] = val_node.value

        run_ids = list(run_values.keys())
        for i, run_a in enumerate(run_ids):
            for run_b in run_ids[i + 1 :]:
                val_a = run_values[run_a]
                val_b = run_values[run_b]

                if self._higher_is_better:
                    a_better = val_a > val_b
                    delta = abs(val_a - val_b) / (abs(val_a) + abs(val_b) + 1e-9)
                else:
                    a_better = val_a < val_b
                    delta = abs(val_a - val_b) / (abs(val_a) + abs(val_b) + 1e-9)

                # Strength: sigmoid of normalised delta, shifted to [0.5, 1]
                strength = 0.5 + 0.5 * (delta / (1.0 + delta))
                if not a_better:
                    strength = 1.0 - strength
                confidence = 0.9  # observed metrics are high confidence

                node_a = atomspace.add(ConceptNode(run_node_name(run_a)))
                node_b = atomspace.add(ConceptNode(run_node_name(run_b)))
                conclusion = ImplicationLink([node_a, node_b])
                tv = TruthValue(strength, confidence)
                explanation = (
                    f"Run {run_a!r} {'>' if a_better else '<'} {run_b!r} on "
                    f"metric '{self._metric_key}' "
                    f"({val_a:.4f} vs {val_b:.4f})"
                )
                yield InferenceResult(
                    rule_name=self.name,
                    conclusion=conclusion,
                    premises=[node_a, node_b],
                    tv=tv,
                    explanation=explanation,
                )
