"""
PLN rule: GeneralizationRule

Infers expected out-of-sample performance by reasoning about the gap between
training and held-out metrics.  Emits an EvaluationLink with predicate
``"expected_generalization"`` and a truth value that penalizes large
train/val gaps.
"""

from __future__ import annotations

from typing import Iterator

from mlflow.cogmlflow.entities.atom import (
    ConceptNode,
    EvaluationLink,
    ListLink,
    NumberNode,
    PredicateNode,
    TruthValue,
)
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.entities.ontology import run_node_name
from mlflow.cogmlflow.pln.engine import InferenceResult, PLNRule
from mlflow.cogmlflow.pln.rules.overfitting_detection import _get_run_metric_value


class GeneralizationRule(PLNRule):
    """Estimates expected generalisation performance for each run.

    The generalisation estimate is:
        expected = val_metric - alpha * max(0, train_metric - val_metric)

    where *alpha* is a penalty factor (default 0.5).  The confidence is
    proportional to the number of evidence points (runs) seen.
    """

    def __init__(
        self,
        train_metric_key: str,
        val_metric_key: str,
        alpha: float = 0.5,
        higher_is_better: bool = True,
    ) -> None:
        self._train_key = train_metric_key
        self._val_key = val_metric_key
        self._alpha = alpha
        self._higher_is_better = higher_is_better

    @property
    def name(self) -> str:
        return "GeneralizationRule"

    def apply(self, atomspace: AtomSpace) -> Iterator[InferenceResult]:
        run_ids: set[str] = set()
        for atom in atomspace:
            if isinstance(atom, ConceptNode) and atom.name.startswith("run:"):
                run_ids.add(atom.name[len("run:") :])

        for run_id in run_ids:
            train_val = _get_run_metric_value(atomspace, run_id, self._train_key)
            val_val = _get_run_metric_value(atomspace, run_id, self._val_key)
            if train_val is None or val_val is None:
                continue

            if self._higher_is_better:
                gap = max(0.0, train_val - val_val)
                expected = val_val - self._alpha * gap
            else:
                gap = max(0.0, val_val - train_val)
                expected = val_val + self._alpha * gap

            strength = min(max(expected, 0.0), 1.0)
            confidence = 0.75

            run_node = atomspace.add(ConceptNode(run_node_name(run_id)))
            conclusion = EvaluationLink([
                PredicateNode("expected_generalization"),
                ListLink([run_node, NumberNode(expected)]),
            ])
            tv = TruthValue(strength, confidence)
            yield InferenceResult(
                rule_name=self.name,
                conclusion=conclusion,
                premises=[run_node],
                tv=tv,
                explanation=(
                    f"Run {run_id!r}: train={train_val:.4f}, val={val_val:.4f}, "
                    f"expected_gen={expected:.4f}"
                ),
            )
