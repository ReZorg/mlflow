"""
CogMLflowClient — the public API for CogMLflow.

Extends ``mlflow.tracking.MlflowClient`` with cognitive capabilities:
* Cognitive experiment recommendations
* AtomSpace queries
* ECAN attention priority queue
* Moses hyperparameter search
* Experiment similarity search
* Autonomous research loop

Usage::

    from mlflow.cogmlflow.client import CogMLflowClient

    client = CogMLflowClient()
    exp_id = client.create_experiment("my_experiment")

    # Run cognitive analysis
    suggestions = client.get_cognitive_suggestions(primary_metric="accuracy")

    # Find similar experiments
    similar = client.find_similar_experiments(run_id="abc123")

    # Tune hyperparameters with Moses
    best = client.tune(
        space={"lr": {"type": "float", "low": 1e-5, "high": 1e-1, "log_scale": True}},
        objective=my_train_fn,
        n_generations=10,
    )
"""

from __future__ import annotations

import logging
from typing import Callable

from mlflow.cogmlflow.atomspace.tracking_store import AtomSpaceTrackingStore
from mlflow.cogmlflow.ecan.attention import ExperimentAttentionManager
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.moses.optimizer import HyperparamSpace, Individual, MosesHyperparamOptimizer
from mlflow.cogmlflow.orchestrator.goal_system import CogMLflowGoalSystem, Goal
from mlflow.cogmlflow.orchestrator.ooda_loop import AutoResearcher
from mlflow.cogmlflow.pln.engine import PLNEngine
from mlflow.cogmlflow.suggestion_engine import CognitiveSuggestionEngine, Suggestion

_log = logging.getLogger(__name__)


class CogMLflowClient:
    """High-level client for CogMLflow cognitive capabilities.

    Parameters
    ----------
    tracking_uri:
        Optional MLflow tracking URI.  If provided, an ``AtomSpaceTrackingStore``
        is created in shadow mode wrapping the standard store at this URI.
        If ``None``, a native in-memory AtomSpaceTrackingStore is used.
    store:
        Provide a pre-built ``AtomSpaceTrackingStore`` directly (overrides
        *tracking_uri*).
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        store: AtomSpaceTrackingStore | None = None,
    ) -> None:
        if store is not None:
            self._store = store
        elif tracking_uri is not None:
            # Shadow mode: wrap an existing store
            delegate = self._build_delegate(tracking_uri)
            self._store = AtomSpaceTrackingStore(delegate=delegate)
        else:
            # Native in-memory mode
            self._store = AtomSpaceTrackingStore()

        self._as: AtomSpace = self._store.atomspace
        self._attention_mgr = ExperimentAttentionManager(self._as)
        self._goal_system = CogMLflowGoalSystem()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_delegate(tracking_uri: str):
        """Build a standard MLflow tracking store from a URI."""
        from mlflow.tracking._tracking_service.utils import _get_store

        return _get_store(tracking_uri)

    # ------------------------------------------------------------------
    # Standard MLflow pass-through
    # ------------------------------------------------------------------

    def create_experiment(self, name: str, artifact_location: str = "", tags=None) -> str:
        exp_id = self._store.create_experiment(name, artifact_location, tags or [])
        self._attention_mgr.initialise_experiment(exp_id)
        return exp_id

    def get_experiment(self, experiment_id: str):
        return self._store.get_experiment(experiment_id)

    def create_run(self, experiment_id: str, tags=None, run_name: str = ""):
        import time

        return self._store.create_run(
            experiment_id=experiment_id,
            user_id="cogmlflow",
            start_time=int(time.time() * 1000),
            tags=tags or [],
            run_name=run_name,
        )

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        import time

        from mlflow.entities import Metric

        self._store.log_metric(run_id, Metric(key, value, int(time.time() * 1000), step))

    def log_param(self, run_id: str, key: str, value: str) -> None:
        from mlflow.entities import Param

        self._store.log_param(run_id, Param(key, str(value)))

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        from mlflow.entities import RunTag

        self._store.set_tag(run_id, RunTag(key, str(value)))

    def get_run(self, run_id: str):
        return self._store.get_run(run_id)

    # ------------------------------------------------------------------
    # Cognitive API
    # ------------------------------------------------------------------

    def get_cognitive_suggestions(
        self,
        primary_metric: str = "accuracy",
        train_metric: str = "train_accuracy",
        val_metric: str = "val_accuracy",
        higher_is_better: bool = True,
        pln_steps: int = 1,
    ) -> list[Suggestion]:
        """Run PLN and return a list of actionable experiment suggestions.

        Parameters
        ----------
        primary_metric:
            The key metric to optimise.
        train_metric / val_metric:
            Keys for overfitting detection.
        higher_is_better:
            Whether higher *primary_metric* is better.
        pln_steps:
            PLN forward-chaining steps.

        Returns
        -------
        list[Suggestion]:
            Ranked list of cognitive suggestions.
        """
        engine = CognitiveSuggestionEngine(
            atomspace=self._as,
            primary_metric=primary_metric,
            train_metric=train_metric,
            val_metric=val_metric,
            higher_is_better=higher_is_better,
            pln_steps=pln_steps,
        )
        return engine.run()

    def find_similar_experiments(self, run_id: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return the top-k runs most similar to *run_id* (by metric cosine similarity).

        Parameters
        ----------
        run_id:
            Reference run ID.
        top_k:
            Number of similar runs to return.

        Returns
        -------
        list[tuple[str, float]]:
            ``[(run_id, similarity_score), ...]`` sorted descending.
        """
        engine = CognitiveSuggestionEngine(self._as)
        return engine.find_similar_experiments(run_id, top_k=top_k)

    def get_attention_ranking(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Return the top-k runs by ECAN attention score.

        Returns
        -------
        list[tuple[str, float]]:
            ``[(run_id, score), ...]`` sorted descending.
        """
        return self._attention_mgr.get_priority_queue(top_k=top_k)

    def add_goal(self, goal: Goal) -> "CogMLflowClient":
        """Register a research goal with the client's goal system."""
        self._goal_system.add_goal(goal)
        return self

    def tune(
        self,
        space: dict[str, dict],
        objective: Callable[[dict], float],
        population_size: int = 20,
        n_generations: int = 10,
        seed: int | None = None,
    ) -> Individual | None:
        """Run Moses-style evolutionary hyperparameter optimisation.

        Parameters
        ----------
        space:
            Hyperparameter space specification dict (see ``HyperparamSpace``).
        objective:
            Callable ``(config: dict) → float`` to maximise.
        population_size:
            Number of individuals per generation.
        n_generations:
            Number of evolutionary generations.
        seed:
            Random seed.

        Returns
        -------
        Individual or None:
            The best individual found, or None if the population is empty.
        """
        hpspace = HyperparamSpace(params=space)
        optimizer = MosesHyperparamOptimizer(
            space=hpspace,
            objective=objective,
            population_size=population_size,
            n_generations=n_generations,
            atomspace=self._as,
            seed=seed,
        )
        optimizer.run()
        return optimizer.best

    def auto_research(
        self,
        runner: Callable[[dict], dict],
        initial_configs: list[dict] | None = None,
        n_iterations: int = 20,
        primary_metric: str = "accuracy",
        **kwargs,
    ) -> AutoResearcher:
        """Create an ``AutoResearcher`` context manager for autonomous ML.

        Parameters
        ----------
        runner:
            Callable ``(config: dict) → dict`` that trains a model and returns metrics.
        initial_configs:
            Seed hyperparameter configurations.
        n_iterations:
            Maximum number of OODA cycles.
        primary_metric:
            The primary metric to optimise.
        **kwargs:
            Additional keyword arguments forwarded to ``AutoResearcher``.

        Returns
        -------
        AutoResearcher:
            Use as a context manager::

                with client.auto_research(runner=my_fn, n_iterations=20) as researcher:
                    best = researcher.run()
        """
        return AutoResearcher(
            store=self._store,
            runner=runner,
            goals=list(self._goal_system.goals),
            initial_configs=initial_configs,
            n_iterations=n_iterations,
            primary_metric=primary_metric,
            **kwargs,
        )

    def run_pln(self, steps: int = 1) -> list:
        """Run PLN forward chaining directly on the AtomSpace.

        Returns the list of new ``InferenceResult`` objects derived.
        """
        pln = PLNEngine(self._as)
        return pln.run(steps=steps)

    @property
    def atomspace(self) -> AtomSpace:
        """Direct access to the underlying AtomSpace."""
        return self._as

    @property
    def store(self) -> AtomSpaceTrackingStore:
        """Direct access to the tracking store."""
        return self._store

    @property
    def goal_system(self) -> CogMLflowGoalSystem:
        """The client's goal system."""
        return self._goal_system

    def __repr__(self) -> str:
        return f"CogMLflowClient(atomspace_size={len(self._as)})"
