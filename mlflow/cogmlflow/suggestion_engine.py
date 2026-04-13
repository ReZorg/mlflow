"""
CognitiveSuggestionEngine — PLN-powered experiment recommendation.

The suggestion engine analyses the current AtomSpace (which mirrors an MLflow
tracking store) and produces human-readable recommendations for:

* Which hyperparameter values to try next (based on HyperparamSensitivityRule).
* Which runs appear to be overfitting (based on OverfittingDetectionRule).
* Which runs are candidates for model-registry promotion (based on
  ModelComparisonRule + GeneralizationRule).
* Which experiments are structurally similar (for knowledge transfer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mlflow.cogmlflow.entities.atom import (
    ConceptNode,
    EvaluationLink,
    ImplicationLink,
    ListLink,
    NumberNode,
    PredicateNode,
)
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.pln.engine import InferenceResult, PLNEngine
from mlflow.cogmlflow.pln.rules import (
    GeneralizationRule,
    HyperparamSensitivityRule,
    ModelComparisonRule,
    OverfittingDetectionRule,
)


@dataclass
class Suggestion:
    """A single actionable recommendation produced by the suggestion engine."""

    # "promote_model" | "investigate_overfitting" |
    # "tune_hyperparam" | "similar_experiment"
    kind: str
    run_id: str
    confidence: float
    message: str
    supporting_inferences: list[InferenceResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "confidence": self.confidence,
            "message": self.message,
            "supporting_inferences": [i.to_dict() for i in self.supporting_inferences],
        }


class CognitiveSuggestionEngine:
    """Analyses an AtomSpace and generates experiment recommendations.

    Parameters
    ----------
    atomspace:
        The AtomSpace containing MLflow experiment knowledge.
    primary_metric:
        Metric key used for model comparison (higher is better by default).
    train_metric:
        Training metric key for overfitting detection.
    val_metric:
        Validation metric key for overfitting detection.
    higher_is_better:
        Whether higher values of *primary_metric* are better.
    pln_steps:
        Number of PLN forward-chaining steps to run before generating
        suggestions.
    """

    def __init__(
        self,
        atomspace: AtomSpace,
        primary_metric: str = "accuracy",
        train_metric: str = "train_accuracy",
        val_metric: str = "val_accuracy",
        higher_is_better: bool = True,
        pln_steps: int = 1,
    ) -> None:
        self._as = atomspace
        self._primary_metric = primary_metric
        self._train_metric = train_metric
        self._val_metric = val_metric
        self._higher_is_better = higher_is_better
        self._pln_steps = pln_steps

        self._pln = PLNEngine(atomspace)
        self._pln.add_rule(ModelComparisonRule(primary_metric, higher_is_better))
        self._pln.add_rule(OverfittingDetectionRule(train_metric, val_metric))
        self._pln.add_rule(GeneralizationRule(train_metric, val_metric))
        self._pln.add_rule(HyperparamSensitivityRule(primary_metric))

    def run(self) -> list[Suggestion]:
        """Run PLN and return a ranked list of suggestions."""
        inferences = self._pln.run(steps=self._pln_steps)
        suggestions: list[Suggestion] = []

        suggestions.extend(self._model_promotion_suggestions(inferences))
        suggestions.extend(self._overfitting_suggestions(inferences))
        suggestions.extend(self._hyperparam_tune_suggestions(inferences))

        # Sort by confidence descending
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions

    # ------------------------------------------------------------------
    # Suggestion generators
    # ------------------------------------------------------------------

    def _model_promotion_suggestions(self, inferences: list[InferenceResult]) -> list[Suggestion]:
        """Suggest runs that are probabilistically the best performers."""
        # Look for ImplicationLinks with high TV strength (run A > run B)
        win_count: dict[str, int] = {}
        win_inferences: dict[str, list[InferenceResult]] = {}
        for result in inferences:
            if not isinstance(result.conclusion, ImplicationLink):
                continue
            if result.tv.strength < 0.65:
                continue
            if len(result.conclusion.outgoing) < 2:
                continue
            winner = result.conclusion.outgoing[0]
            if isinstance(winner, ConceptNode) and winner.name.startswith("run:"):
                run_id = winner.name[len("run:") :]
                win_count[run_id] = win_count.get(run_id, 0) + 1
                win_inferences.setdefault(run_id, []).append(result)

        suggestions = []
        for run_id, wins in win_count.items():
            if wins < 1:
                continue
            avg_strength = sum(i.tv.strength for i in win_inferences[run_id]) / wins
            suggestions.append(
                Suggestion(
                    kind="promote_model",
                    run_id=run_id,
                    confidence=min(avg_strength * (1.0 + 0.1 * wins), 1.0),
                    message=(
                        f"Run '{run_id}' outperformed {wins} other run(s) on "
                        f"'{self._primary_metric}'. Consider promoting this model."
                    ),
                    supporting_inferences=win_inferences[run_id],
                )
            )
        return suggestions

    def _overfitting_suggestions(self, inferences: list[InferenceResult]) -> list[Suggestion]:
        """Suggest runs that show signs of overfitting."""
        suggestions = []
        for result in inferences:
            if result.rule_name.startswith("OverfittingDetectionRule") and result.tv.strength > 0.5:
                if not isinstance(result.conclusion, EvaluationLink):
                    continue
                args = (
                    result.conclusion.outgoing[1] if len(result.conclusion.outgoing) > 1 else None
                )
                if args is None:
                    continue
                run_node = (
                    args.outgoing[0] if isinstance(args, ListLink) and args.outgoing else None
                )
                if run_node is None or not isinstance(run_node, ConceptNode):
                    continue
                run_id = run_node.name.removeprefix("run:")
                suggestions.append(
                    Suggestion(
                        kind="investigate_overfitting",
                        run_id=run_id,
                        confidence=result.tv.strength * result.tv.confidence,
                        message=(
                            f"Run '{run_id}' shows possible overfitting "
                            f"(strength={result.tv.strength:.2f}). "
                            f"Consider adding regularisation or more training data."
                        ),
                        supporting_inferences=[result],
                    )
                )
        return suggestions

    def _hyperparam_tune_suggestions(self, inferences: list[InferenceResult]) -> list[Suggestion]:
        """Suggest hyperparameter directions based on sensitivity analysis."""
        suggestions = []
        for result in inferences:
            if not result.rule_name.startswith("HyperparamSensitivityRule"):
                continue
            if result.tv.strength < 0.3:
                continue
            # Extract param and metric names from the conclusion
            if not isinstance(result.conclusion, EvaluationLink):
                continue
            args = result.conclusion.outgoing[1] if len(result.conclusion.outgoing) > 1 else None
            if args is None or not isinstance(args, ListLink) or len(args.outgoing) < 3:
                continue
            param_pred = args.outgoing[0]
            metric_pred = args.outgoing[1]
            corr_node = args.outgoing[2]
            if not isinstance(param_pred, PredicateNode) or not isinstance(
                metric_pred, PredicateNode
            ):
                continue
            param_key = param_pred.name.replace("param:", "")
            corr = corr_node.value if isinstance(corr_node, NumberNode) else 0.0
            suggestions.append(
                Suggestion(
                    kind="tune_hyperparam",
                    run_id="",
                    confidence=result.tv.strength * result.tv.confidence,
                    message=(
                        f"Hyperparameter '{param_key}' has a significant correlation "
                        f"(|r|={corr:.3f}) with '{self._primary_metric}'. "
                        f"Focus tuning efforts here."
                    ),
                    supporting_inferences=[result],
                )
            )
        return suggestions

    # ------------------------------------------------------------------
    # Cross-experiment similarity
    # ------------------------------------------------------------------

    def find_similar_experiments(self, run_id: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k most similar run IDs to *run_id*.

        Similarity is computed as the cosine similarity of metric vectors.
        """
        from mlflow.cogmlflow.atomspace.translator import MlflowAtomTranslator

        translator = MlflowAtomTranslator(self._as)
        target_metrics = dict(translator._onto.get_run_metrics(run_id))
        if not target_metrics:
            return []

        # Collect all runs
        all_run_ids: set[str] = set()
        for atom in self._as:
            if isinstance(atom, ConceptNode) and atom.name.startswith("run:"):
                all_run_ids.add(atom.name[len("run:") :])
        all_run_ids.discard(run_id)

        similarities = []
        all_keys = set(target_metrics.keys())
        for other_id in all_run_ids:
            other_metrics = dict(translator._onto.get_run_metrics(other_id))
            common_keys = all_keys & set(other_metrics.keys())
            if not common_keys:
                continue
            dot = sum(target_metrics[k] * other_metrics[k] for k in common_keys)
            norm_a = math.sqrt(sum(v**2 for k, v in target_metrics.items() if k in common_keys))
            norm_b = math.sqrt(sum(v**2 for k, v in other_metrics.items() if k in common_keys))
            if norm_a * norm_b == 0:
                continue
            sim = dot / (norm_a * norm_b)
            similarities.append((other_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
