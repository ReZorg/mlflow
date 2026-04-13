"""
CogMLflow goal system (OpenPsi-inspired).

Goals drive experiment selection by assigning a *desirability* weight to
different objectives.  The goal system is consulted by the CognitiveScheduler
when choosing which experiment to run next.

Goals are represented as named predicates in the AtomSpace with associated
desirability values (which are updated based on feedback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Goal:
    """A single cognitive goal with a desirability weight.

    Parameters
    ----------
    name:
        Human-readable goal name (e.g. ``"maximize_accuracy"``).
    description:
        Optional description.
    desirability:
        Current importance weight in [0, 1].  Higher means the system will
        prioritise experiments that make progress toward this goal.
    predicate:
        Optional callable ``(run_metrics: dict) → float`` that computes how
        much a given run contributes to this goal (returns a value in [0, 1]).
    """

    name: str
    description: str = ""
    desirability: float = 0.5
    predicate: Callable[[dict[str, object]], float] | None = field(default=None, repr=False)

    def evaluate(self, run_metrics: dict[str, object]) -> float:
        """Return the goal-fulfilment score for a run (0 = no progress, 1 = full)."""
        if self.predicate is None:
            return 0.0
        return float(self.predicate(run_metrics))

    def reweight(self, new_desirability: float) -> None:
        self.desirability = max(0.0, min(1.0, new_desirability))


class CogMlflowGoalSystem:
    """Manages a prioritised set of cognitive goals.

    Usage::

        goals = CogMlflowGoalSystem()
        goals.add_goal(
            Goal(
                name="maximize_accuracy",
                desirability=0.9,
                predicate=lambda m: m.get("val_accuracy", 0.0),
            )
        )
        goals.add_goal(
            Goal(
                name="minimize_latency",
                desirability=0.5,
                predicate=lambda m: 1.0 - min(m.get("inference_latency_ms", 1000) / 1000, 1.0),
            )
        )

        # Score a run
        score = goals.score_run({"val_accuracy": 0.95, "inference_latency_ms": 50})
    """

    def __init__(self) -> None:
        self._goals: dict[str, Goal] = {}

    def add_goal(self, goal: Goal) -> "CogMlflowGoalSystem":
        """Register a goal.  Returns self for chaining."""
        self._goals[goal.name] = goal
        return self

    def remove_goal(self, name: str) -> None:
        self._goals.pop(name, None)

    def get_goal(self, name: str) -> Goal | None:
        return self._goals.get(name)

    def score_run(self, run_metrics: dict[str, object]) -> float:
        """Return the weighted sum of goal fulfilments for a run.

        Returns a value in [0, 1] representing how well the run satisfies the
        current goal configuration.
        """
        if not self._goals:
            return 0.0
        total_weight = sum(g.desirability for g in self._goals.values())
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(g.desirability * g.evaluate(run_metrics) for g in self._goals.values())
        return weighted_sum / total_weight

    def update_from_feedback(self, feedback: dict[str, float]) -> None:
        """Adjust goal desirability based on external feedback.

        *feedback* maps goal names to new desirability values.
        """
        for name, desirability in feedback.items():
            goal = self._goals.get(name)
            if goal is not None:
                goal.reweight(desirability)

    @property
    def goals(self) -> list[Goal]:
        return list(self._goals.values())

    def __repr__(self) -> str:
        goal_str = ", ".join(f"{g.name}={g.desirability:.2f}" for g in self._goals.values())
        return f"CogMlflowGoalSystem([{goal_str}])"


# ---------------------------------------------------------------------------
# Preset goals
# ---------------------------------------------------------------------------


def accuracy_goal(metric_key: str = "accuracy", desirability: float = 0.9) -> Goal:
    return Goal(
        name=f"maximize_{metric_key}",
        description=f"Maximise the {metric_key} metric",
        desirability=desirability,
        predicate=lambda m, k=metric_key: min(float(m.get(k, 0.0)), 1.0),
    )


def latency_goal(
    metric_key: str = "inference_latency_ms", max_ms: float = 200.0, desirability: float = 0.5
) -> Goal:
    return Goal(
        name="minimize_latency",
        description=f"Keep {metric_key} below {max_ms} ms",
        desirability=desirability,
        predicate=lambda m, k=metric_key, mx=max_ms: max(0.0, 1.0 - float(m.get(k, mx)) / mx),
    )


def loss_goal(metric_key: str = "loss", desirability: float = 0.8) -> Goal:
    return Goal(
        name=f"minimize_{metric_key}",
        description=f"Minimise the {metric_key} metric",
        desirability=desirability,
        predicate=lambda m, k=metric_key: max(0.0, 1.0 - min(float(m.get(k, 1.0)), 1.0)),
    )
