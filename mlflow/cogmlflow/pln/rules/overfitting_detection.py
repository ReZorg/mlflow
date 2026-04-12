"""
PLN rule: OverfittingDetectionRule

Infers whether a run shows signs of overfitting by comparing train and
validation metrics.  Emits an EvaluationLink with predicate
``"overfitting_detected"`` and a truth value reflecting the degree of
train/val divergence.
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
from mlflow.cogmlflow.entities.ontology import metric_pred_name, run_node_name
from mlflow.cogmlflow.pln.engine import InferenceResult, PLNRule


def _get_run_metric_value(atomspace: AtomSpace, run_id: str, metric_key: str) -> float | None:
    pred_name = metric_pred_name(metric_key)
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
        node = args.outgoing[0]
        val = args.outgoing[1]
        if isinstance(node, ConceptNode) and node.name == run_node_name(run_id):
            if isinstance(val, NumberNode):
                return val.value
    return None


class OverfittingDetectionRule(PLNRule):
    """Detects overfitting by comparing train_metric vs val_metric for each run.

    Parameters
    ----------
    train_metric_key:
        Name of the training metric (e.g. ``"train_accuracy"``).
    val_metric_key:
        Name of the validation metric (e.g. ``"val_accuracy"``).
    threshold:
        Minimum relative gap (train - val) / train before overfitting is
        considered likely.  Default 0.05 (5 %).
    higher_is_better:
        Whether higher values are better for this metric.
    """

    def __init__(
        self,
        train_metric_key: str,
        val_metric_key: str,
        threshold: float = 0.05,
        higher_is_better: bool = True,
    ) -> None:
        self._train_key = train_metric_key
        self._val_key = val_metric_key
        self._threshold = threshold
        self._higher_is_better = higher_is_better

    @property
    def name(self) -> str:
        return f"OverfittingDetectionRule[{self._train_key}/{self._val_key}]"

    def apply(self, atomspace: AtomSpace) -> Iterator[InferenceResult]:
        # Collect all run IDs
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
                gap = (train_val - val_val) / (abs(train_val) + 1e-9)
            else:
                gap = (val_val - train_val) / (abs(train_val) + 1e-9)

            if gap > 0:
                strength = min(gap / (self._threshold + 1e-9) * 0.5, 1.0)
                confidence = 0.85
            else:
                strength = 0.1
                confidence = 0.7

            run_node = atomspace.add(ConceptNode(run_node_name(run_id)))
            conclusion = EvaluationLink([
                PredicateNode("overfitting_detected"),
                ListLink([run_node]),
            ])
            tv = TruthValue(strength, confidence)
            yield InferenceResult(
                rule_name=self.name,
                conclusion=conclusion,
                premises=[run_node],
                tv=tv,
                explanation=(
                    f"Run {run_id!r}: train={train_val:.4f}, val={val_val:.4f}, "
                    f"gap={gap:.3f} (threshold={self._threshold})"
                ),
            )
