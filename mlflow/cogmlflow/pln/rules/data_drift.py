"""
PLN rule: DataDriftRule

Infers whether a metric degradation between two time-ordered runs is more
likely caused by data drift (covariate shift) than by a model regression.

Heuristic: if metrics degrade but model hyperparams did not change, data
drift is the more probable explanation.  Emits an EvaluationLink with
predicate ``"data_drift_suspected"``.
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


class DataDriftRule(PLNRule):
    """Infers data drift between a reference run and a comparison run.

    Compares two runs on a held-out metric.  If the metric degrades and the
    key hyperparameters are unchanged, it attributes the degradation to data
    drift with high strength.

    Parameters
    ----------
    reference_run_id:
        The earlier/reference run.
    comparison_run_id:
        The later/comparison run (potentially affected by drift).
    metric_key:
        The held-out metric to compare.
    param_keys:
        Hyperparameters to check for changes.
    higher_is_better:
        Whether higher metric values are better.
    """

    def __init__(
        self,
        reference_run_id: str,
        comparison_run_id: str,
        metric_key: str,
        param_keys: list[str] | None = None,
        higher_is_better: bool = True,
    ) -> None:
        self._ref = reference_run_id
        self._comp = comparison_run_id
        self._metric_key = metric_key
        self._param_keys = param_keys or []
        self._higher_is_better = higher_is_better

    @property
    def name(self) -> str:
        return f"DataDriftRule[{self._metric_key}]"

    def apply(self, atomspace: AtomSpace) -> Iterator[InferenceResult]:
        ref_metric = _get_run_metric_value(atomspace, self._ref, self._metric_key)
        comp_metric = _get_run_metric_value(atomspace, self._comp, self._metric_key)
        if ref_metric is None or comp_metric is None:
            return

        if self._higher_is_better:
            degraded = comp_metric < ref_metric
            delta = (ref_metric - comp_metric) / (abs(ref_metric) + 1e-9)
        else:
            degraded = comp_metric > ref_metric
            delta = (comp_metric - ref_metric) / (abs(ref_metric) + 1e-9)

        if not degraded:
            return

        # Check if hyperparams changed between runs
        params_changed = False
        for param_key in self._param_keys:
            from mlflow.cogmlflow.entities.atom import ListLink as LL
            from mlflow.cogmlflow.entities.ontology import param_pred_name

            ref_val = None
            comp_val = None
            pred_name = param_pred_name(param_key)
            for atom in atomspace:
                if not isinstance(atom, EvaluationLink):
                    continue
                if len(atom.outgoing) < 2:
                    continue
                pred, args = atom.outgoing[0], atom.outgoing[1]
                if not isinstance(pred, PredicateNode) or pred.name != pred_name:
                    continue
                if not isinstance(args, LL) or len(args.outgoing) < 2:
                    continue
                run_node = args.outgoing[0]
                val_node = args.outgoing[1]
                if not isinstance(run_node, ConceptNode):
                    continue
                if run_node.name == run_node_name(self._ref):
                    ref_val = val_node.name
                elif run_node.name == run_node_name(self._comp):
                    comp_val = val_node.name
            if ref_val is not None and comp_val is not None and ref_val != comp_val:
                params_changed = True
                break

        # If params unchanged and metric degraded → data drift
        strength = (
            min(0.5 + delta, 1.0)
            if not params_changed
            else min(0.3 + delta * 0.5, 0.7)
        )

        ref_node = atomspace.add(ConceptNode(run_node_name(self._ref)))
        comp_node = atomspace.add(ConceptNode(run_node_name(self._comp)))
        conclusion = EvaluationLink([
            PredicateNode("data_drift_suspected"),
            ListLink([comp_node, ref_node, NumberNode(delta)]),
        ])
        tv = TruthValue(strength, 0.75)
        yield InferenceResult(
            rule_name=self.name,
            conclusion=conclusion,
            premises=[ref_node, comp_node],
            tv=tv,
            explanation=(
                f"Metric '{self._metric_key}' degraded by {delta:.3f} from run "
                f"{self._ref!r} to {self._comp!r}; "
                f"params_changed={params_changed} → data_drift_strength={strength:.3f}"
            ),
        )
