"""
OODA loop (Observe-Orient-Decide-Act) for autonomous ML research.

The OODA loop implements the full autonomous research cycle:

* **Observe**: ingest new MLflow run results into the AtomSpace.
* **Orient**: run PLN forward-chaining to update beliefs about the landscape.
* **Decide**: ECAN + goal system select the next experiment configuration.
* **Act**: execute the chosen experiment via a user-supplied runner callable.

Usage::

    from mlflow.cogmlflow.orchestrator.ooda_loop import AutoResearcher
    from mlflow.cogmlflow.orchestrator.goal_system import accuracy_goal
    from mlflow.cogmlflow.atomspace.tracking_store import AtomSpaceTrackingStore

    store = AtomSpaceTrackingStore()


    def my_runner(config):
        # Train a model with config, return metrics
        return {"val_accuracy": ...}


    with AutoResearcher(
        store=store,
        runner=my_runner,
        goals=[accuracy_goal()],
        n_iterations=20,
    ) as researcher:
        best = researcher.run()

    print("Best config:", best.config, "Score:", best.fitness)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from mlflow.cogmlflow.atomspace.tracking_store import AtomSpaceTrackingStore
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.orchestrator.goal_system import CogMLflowGoalSystem
from mlflow.cogmlflow.orchestrator.scheduler import CognitiveScheduler, ExperimentCandidate
from mlflow.cogmlflow.pln.engine import PLNEngine
from mlflow.cogmlflow.pln.rules import (
    GeneralizationRule,
    HyperparamSensitivityRule,
    ModelComparisonRule,
    OverfittingDetectionRule,
)

_log = logging.getLogger(__name__)


@dataclass
class ResearchIteration:
    iteration: int
    candidate_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    goal_score: float
    success: bool


class AutoResearcher:
    """Context manager that runs the full OODA autonomous research loop.

    Parameters
    ----------
    store:
        The AtomSpaceTrackingStore (provides both tracking and AtomSpace).
    runner:
        Callable ``(config: dict) → dict`` that trains a model and returns metrics.
    goals:
        List of ``Goal`` objects defining research objectives.
    initial_configs:
        Seed configurations to evaluate before evolution starts.
    n_iterations:
        Maximum number of OODA cycles to execute.
    pln_steps:
        PLN forward-chaining steps per orient phase.
    primary_metric:
        Name of the metric used for model comparison and generalisation rules.
    train_metric:
        Training metric key (used by overfitting and generalisation rules).
    val_metric:
        Validation metric key.
    epsilon:
        Exploration rate for the scheduler.
    """

    def __init__(
        self,
        store: AtomSpaceTrackingStore,
        runner: Callable[[dict], dict],
        goals: list | None = None,
        initial_configs: list[dict] | None = None,
        n_iterations: int = 20,
        pln_steps: int = 2,
        primary_metric: str = "accuracy",
        train_metric: str = "train_accuracy",
        val_metric: str = "val_accuracy",
        epsilon: float = 0.15,
    ) -> None:
        self._store = store
        self._runner = runner
        self._n_iterations = n_iterations
        self._pln_steps = pln_steps
        self._primary_metric = primary_metric
        self._train_metric = train_metric
        self._val_metric = val_metric

        self._as: AtomSpace = store.atomspace

        # Goal system
        self._goal_system = CogMLflowGoalSystem()
        for goal in goals or []:
            self._goal_system.add_goal(goal)

        # PLN engine with default rules
        self._pln = PLNEngine(self._as)
        self._pln.add_rule(ModelComparisonRule(primary_metric))
        self._pln.add_rule(OverfittingDetectionRule(train_metric, val_metric))
        self._pln.add_rule(GeneralizationRule(train_metric, val_metric))
        self._pln.add_rule(HyperparamSensitivityRule(primary_metric))

        # Scheduler
        self._scheduler = CognitiveScheduler(
            atomspace=self._as,
            goal_system=self._goal_system,
            epsilon=epsilon,
        )

        # Pre-load initial candidates
        for cfg in initial_configs or []:
            cid = str(uuid.uuid4())[:8]
            self._scheduler.add_candidate(cid, cfg)

        self._history: list[ResearchIteration] = []
        self._best: ExperimentCandidate | None = None

    # ------------------------------------------------------------------
    # OODA phases
    # ------------------------------------------------------------------

    def _observe(self, candidate_id: str, metrics: dict) -> None:
        """Ingest run results into the AtomSpace."""
        onto = self._store.translator._onto
        for key, value in metrics.items():
            try:
                float_val = float(value)
                onto.log_metric(candidate_id, key, float_val)
            except (TypeError, ValueError):
                pass

    def _orient(self) -> None:
        """Run PLN to update beliefs about the experiment landscape."""
        new_inferences = self._pln.run(steps=self._pln_steps)
        if new_inferences:
            _log.debug("[OODA orient] %d new PLN inferences", len(new_inferences))

    def _decide(self) -> ExperimentCandidate | None:
        """Select the next experiment using ECAN + goal system."""
        return self._scheduler.select_next()

    def _act(self, candidate: ExperimentCandidate) -> dict:
        """Execute the chosen experiment and return its metrics."""
        _log.debug(
            "[OODA act] Running candidate %s with config %s",
            candidate.candidate_id,
            candidate.config,
        )
        return self._runner(candidate.config)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> ExperimentCandidate | None:
        """Execute the full autonomous research loop.

        Returns the best candidate found (by goal score), or None if no
        experiments were run.
        """
        best_score = -1.0
        best_candidate: ExperimentCandidate | None = None

        for iteration in range(self._n_iterations):
            _log.info("[OODA] Iteration %d/%d", iteration + 1, self._n_iterations)

            # Decide
            candidate = self._decide()
            if candidate is None:
                _log.warning("[OODA] No candidates available; stopping.")
                break

            # Act
            try:
                metrics = self._act(candidate)
                success = True
            except Exception as exc:
                _log.warning("[OODA] Runner failed for %s: %s", candidate.candidate_id, exc)
                metrics = {}
                success = False

            # Observe
            self._observe(candidate.candidate_id, metrics)

            # Orient
            self._orient()

            # Record result
            goal_score = self._goal_system.score_run(metrics)
            self._scheduler.record_result(candidate.candidate_id, metrics, success)

            iteration_record = ResearchIteration(
                iteration=iteration,
                candidate_id=candidate.candidate_id,
                config=candidate.config,
                metrics=metrics,
                goal_score=goal_score,
                success=success,
            )
            self._history.append(iteration_record)

            if success and goal_score > best_score:
                best_score = goal_score
                best_candidate = candidate
                _log.info(
                    "[OODA] New best candidate %s (goal_score=%.4f)",
                    candidate.candidate_id,
                    goal_score,
                )

        self._best = best_candidate
        return best_candidate

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "AutoResearcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[ResearchIteration]:
        return list(self._history)

    @property
    def best(self) -> ExperimentCandidate | None:
        return self._best

    @property
    def pln_engine(self) -> PLNEngine:
        return self._pln

    @property
    def scheduler(self) -> CognitiveScheduler:
        return self._scheduler
