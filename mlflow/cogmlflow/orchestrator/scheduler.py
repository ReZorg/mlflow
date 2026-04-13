"""
CognitiveScheduler — ECAN + goal-driven experiment scheduler.

The scheduler selects which experiment configuration to run next by combining:
1. **ECAN attention values** (STI/LTI) on run and experiment atoms.
2. **PLN-derived expected performance** (generalisation estimates).
3. **Goal system desirability scores**.

An exploration-exploitation trade-off is controlled via an *epsilon-greedy*
policy (with optional UCB).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from mlflow.cogmlflow.ecan.attention import ExperimentAttentionManager
from mlflow.cogmlflow.entities.atom import (
    ConceptNode,
    EvaluationLink,
    ListLink,
    NumberNode,
    PredicateNode,
)
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.entities.ontology import run_node_name
from mlflow.cogmlflow.orchestrator.goal_system import CogMlflowGoalSystem


@dataclass
class ExperimentCandidate:
    """A candidate experiment configuration awaiting scheduling."""

    candidate_id: str
    config: dict[str, Any]
    experiment_id: str = ""
    expected_score: float = 0.0
    attention_score: float = 0.0
    goal_score: float = 0.0
    combined_score: float = 0.0
    run_count: int = 0
    last_run_time: float = field(default_factory=time.time)


class CognitiveScheduler:
    """Selects the next experiment to run from a pool of candidates.

    Parameters
    ----------
    atomspace:
        The shared AtomSpace used for attention queries.
    goal_system:
        Optional goal system for desirability weighting.
    attention_manager:
        Optional ECAN attention manager.  If None, one is created automatically.
    epsilon:
        Exploration probability for epsilon-greedy policy.
    ucb_c:
        UCB exploration constant (used when epsilon=0).
    """

    def __init__(
        self,
        atomspace: AtomSpace,
        goal_system: CogMlflowGoalSystem | None = None,
        attention_manager: ExperimentAttentionManager | None = None,
        epsilon: float = 0.1,
        ucb_c: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._as = atomspace
        self._goal_system = goal_system or CogMlflowGoalSystem()
        self._attention = attention_manager or ExperimentAttentionManager(atomspace)
        self._epsilon = epsilon
        self._ucb_c = ucb_c
        self._rng = random.Random(seed)
        self._candidates: dict[str, ExperimentCandidate] = {}
        self._total_runs: int = 0

    # ------------------------------------------------------------------
    # Candidate management
    # ------------------------------------------------------------------

    def add_candidate(
        self,
        candidate_id: str,
        config: dict[str, Any],
        experiment_id: str = "",
    ) -> ExperimentCandidate:
        """Register a new experiment candidate for scheduling consideration."""
        cand = ExperimentCandidate(
            candidate_id=candidate_id,
            config=config,
            experiment_id=experiment_id,
        )
        self._candidates[candidate_id] = cand
        self._attention.initialise_run(candidate_id)
        return cand

    def remove_candidate(self, candidate_id: str) -> None:
        self._candidates.pop(candidate_id, None)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _attention_score(self, candidate: ExperimentCandidate) -> float:
        node = self._as.get(ConceptNode(run_node_name(candidate.candidate_id)))
        if node is None:
            return 0.0
        return node.sti + 0.1 * node.lti

    def _expected_performance(self, candidate: ExperimentCandidate) -> float:
        """Read PLN-derived generalisation estimate from AtomSpace."""
        run_id = candidate.candidate_id
        run_node = ConceptNode(run_node_name(run_id))
        for atom in self._as:
            if not isinstance(atom, EvaluationLink):
                continue
            if len(atom.outgoing) < 2:
                continue
            pred, args = atom.outgoing[0], atom.outgoing[1]
            if not isinstance(pred, PredicateNode) or pred.name != "expected_generalization":
                continue
            if not isinstance(args, ListLink) or len(args.outgoing) < 2:
                continue
            if args.outgoing[0] == run_node:
                val = args.outgoing[1]
                if isinstance(val, NumberNode):
                    return val.value * atom.tv.strength * atom.tv.confidence
        return 0.0

    def _goal_score(self, candidate: ExperimentCandidate) -> float:
        """Score the candidate config against the current goal system."""
        return self._goal_system.score_run(candidate.config)

    def _ucb_score(self, candidate: ExperimentCandidate) -> float:
        """Upper Confidence Bound score for exploration-exploitation."""
        if self._total_runs == 0 or candidate.run_count == 0:
            return float("inf")
        return candidate.combined_score + self._ucb_c * math.sqrt(
            math.log(self._total_runs) / candidate.run_count
        )

    def _score_candidate(self, candidate: ExperimentCandidate) -> float:
        attn = self._attention_score(candidate)
        perf = self._expected_performance(candidate)
        goal = self._goal_score(candidate)
        # Weighted combination
        combined = 0.4 * attn / 50.0 + 0.4 * perf + 0.2 * goal
        candidate.attention_score = attn
        candidate.expected_score = perf
        candidate.goal_score = goal
        candidate.combined_score = combined
        return combined

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_next(self) -> ExperimentCandidate | None:
        """Select the next candidate to execute.

        Returns None if there are no candidates.
        """
        if not self._candidates:
            return None

        candidates = list(self._candidates.values())

        # Epsilon-greedy exploration
        if self._rng.random() < self._epsilon:
            return self._rng.choice(candidates)

        # UCB-based selection
        return max(
            candidates,
            key=lambda c: self._ucb_score(c) if self._total_runs > 0 else self._score_candidate(c),
        )

    def record_result(
        self,
        candidate_id: str,
        metrics: dict[str, object],
        success: bool = True,
    ) -> None:
        """Update scheduler state after an experiment completes.

        This feeds the result back into ECAN (reward/penalise) and updates
        run counters for UCB.
        """
        cand = self._candidates.get(candidate_id)
        if cand is None:
            return
        cand.run_count += 1
        cand.last_run_time = time.time()
        self._total_runs += 1

        goal_score = self._goal_system.score_run(metrics)
        if success and goal_score > 0.5:
            self._attention.reward_run(candidate_id, metric_improvement=goal_score)
        else:
            self._attention.penalise_run(candidate_id)

        # Advance one ECAN cycle
        self._attention.step()

    @property
    def candidates(self) -> list[ExperimentCandidate]:
        return list(self._candidates.values())

    @property
    def attention_manager(self) -> ExperimentAttentionManager:
        return self._attention
