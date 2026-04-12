"""
PLN rule: HyperparamSensitivityRule

Infers which hyperparameters are most correlated with metric variance
across a collection of runs.  Emits EvaluationLink atoms with predicate
``"hyperparam_sensitivity"`` and a strength proportional to the absolute
Pearson correlation coefficient.
"""

from __future__ import annotations

import math
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
from mlflow.cogmlflow.entities.ontology import metric_pred_name, param_pred_name, run_node_name
from mlflow.cogmlflow.pln.engine import InferenceResult, PLNRule


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return num / (denom_x * denom_y)


class HyperparamSensitivityRule(PLNRule):
    """Correlates hyperparameter values with metric outcomes across runs.

    For each numeric-convertible (param_key, metric_key) pair, computes the
    Pearson correlation and emits a sensitivity atom.
    """

    def __init__(self, metric_key: str) -> None:
        self._metric_key = metric_key

    @property
    def name(self) -> str:
        return f"HyperparamSensitivityRule[{self._metric_key}]"

    def apply(self, atomspace: AtomSpace) -> Iterator[InferenceResult]:
        # Gather {run_id: metric_value} mapping
        metric_map: dict[str, float] = {}
        pred_name = metric_pred_name(self._metric_key)
        for atom in atomspace:
            if not isinstance(atom, EvaluationLink):
                continue
            if len(atom.outgoing) < 2:
                continue
            pred, args = atom.outgoing[0], atom.outgoing[1]
            if not isinstance(pred, PredicateNode) or pred.name != pred_name:
                continue
            if not isinstance(args, ListLink) or len(args.outgoing) < 2:
                continue
            run_node = args.outgoing[0]
            val_node = args.outgoing[1]
            if isinstance(run_node, ConceptNode) and run_node.name.startswith("run:"):
                if isinstance(val_node, NumberNode):
                    metric_map[run_node.name[len("run:") :]] = val_node.value

        if len(metric_map) < 2:
            return

        # Gather {run_id: {param_key: float_value}} — skip non-numeric params
        param_map: dict[str, dict[str, float]] = {}
        for atom in atomspace:
            if not isinstance(atom, EvaluationLink):
                continue
            if len(atom.outgoing) < 2:
                continue
            pred, args = atom.outgoing[0], atom.outgoing[1]
            if not isinstance(pred, PredicateNode) or not pred.name.startswith("param:"):
                continue
            if not isinstance(args, ListLink) or len(args.outgoing) < 2:
                continue
            run_node = args.outgoing[0]
            val_node = args.outgoing[1]
            if not isinstance(run_node, ConceptNode) or not run_node.name.startswith("run:"):
                continue
            run_id = run_node.name[len("run:") :]
            param_key = pred.name[len("param:") :]
            try:
                float_val = float(val_node.name)
            except (ValueError, AttributeError):
                continue
            param_map.setdefault(run_id, {})[param_key] = float_val

        # Collect all param keys that appear in ≥2 runs also present in metric_map
        all_param_keys: set[str] = set()
        for run_id in metric_map:
            if run_id in param_map:
                all_param_keys.update(param_map[run_id].keys())

        for param_key in all_param_keys:
            common_runs = [r for r in metric_map if r in param_map and param_key in param_map[r]]
            if len(common_runs) < 2:
                continue
            xs = [param_map[r][param_key] for r in common_runs]
            ys = [metric_map[r] for r in common_runs]
            corr = abs(_pearson(xs, ys))

            run_nodes = [atomspace.add(ConceptNode(run_node_name(r))) for r in common_runs]
            param_pred = atomspace.add(PredicateNode(param_pred_name(param_key)))
            metric_pred = atomspace.add(PredicateNode(metric_pred_name(self._metric_key)))
            conclusion = EvaluationLink([
                PredicateNode("hyperparam_sensitivity"),
                ListLink([param_pred, metric_pred, NumberNode(corr)]),
            ])
            tv = TruthValue(corr, min(0.5 + len(common_runs) * 0.05, 0.99))
            yield InferenceResult(
                rule_name=self.name,
                conclusion=conclusion,
                premises=run_nodes,
                tv=tv,
                explanation=(
                    f"param '{param_key}' has |Pearson r|={corr:.3f} with "
                    f"metric '{self._metric_key}' across {len(common_runs)} runs"
                ),
            )
