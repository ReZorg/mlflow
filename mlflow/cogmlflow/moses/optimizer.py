"""
Moses-style evolutionary hyperparameter optimizer for CogMLflow.

Moses (Meta-Optimizing Semantic Evolutionary Search) evolves *programs* rather
than merely parameter vectors.  In this Python implementation each individual
is a dict of hyperparameter values; the population is evolved using:

* Tournament selection
* Uniform crossover
* Gaussian / categorical mutation
* Optional Bayesian initialisation from AtomSpace history

The optimizer integrates with an AtomSpace to:
1. Warm-start from previously evaluated configurations.
2. Persist every evaluated individual as atoms for PLN reasoning.
3. Report all evaluations to an MLflow tracking store (if provided).
"""

from __future__ import annotations

import copy
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from mlflow.cogmlflow.entities.atom import (
    ConceptNode,
    EvaluationLink,
    ListLink,
    NumberNode,
    PredicateNode,
)
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.entities.ontology import run_node_name

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


@dataclass
class HyperparamSpace:
    """Definition of the hyperparameter search space.

    Attributes:
        params: dict mapping param name → dict with keys:
            - ``type``: ``"float"``, ``"int"``, or ``"categorical"``
            - ``low`` / ``high``: bounds for numeric types
            - ``choices``: list of values for categorical type
            - ``log_scale``: bool, sample in log space (float/int only)
    """

    params: dict[str, dict[str, Any]] = field(default_factory=dict)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        """Draw a random configuration from this space."""
        config: dict[str, Any] = {}
        for name, spec in self.params.items():
            ptype = spec.get("type", "float")
            if ptype == "categorical":
                config[name] = rng.choice(spec["choices"])
            elif ptype == "int":
                low, high = int(spec["low"]), int(spec["high"])
                if spec.get("log_scale", False) and low > 0:
                    config[name] = int(math.exp(rng.uniform(math.log(low), math.log(high))))
                else:
                    config[name] = rng.randint(low, high)
            else:  # float
                low, high = float(spec["low"]), float(spec["high"])
                if spec.get("log_scale", False) and low > 0:
                    config[name] = math.exp(rng.uniform(math.log(low), math.log(high)))
                else:
                    config[name] = rng.uniform(low, high)
        return config

    def mutate(
        self, config: dict[str, Any], rng: random.Random, sigma: float = 0.1
    ) -> dict[str, Any]:
        """Return a mutated copy of *config*."""
        mutated = copy.deepcopy(config)
        for name, spec in self.params.items():
            if rng.random() > 0.3:  # mutate each param with 30% probability
                continue
            ptype = spec.get("type", "float")
            if ptype == "categorical":
                mutated[name] = rng.choice(spec["choices"])
            elif ptype == "int":
                low, high = int(spec["low"]), int(spec["high"])
                delta = max(1, int((high - low) * sigma))
                mutated[name] = max(low, min(high, mutated[name] + rng.randint(-delta, delta)))
            else:
                low, high = float(spec["low"]), float(spec["high"])
                scale = (high - low) * sigma
                mutated[name] = max(low, min(high, mutated[name] + rng.gauss(0, scale)))
        return mutated

    def crossover(
        self, parent_a: dict[str, Any], parent_b: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Uniform crossover between two parents."""
        child: dict[str, Any] = {}
        for name in self.params:
            child[name] = parent_a[name] if rng.random() < 0.5 else parent_b[name]
        return child


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """A candidate hyperparameter configuration with its fitness score."""

    config: dict[str, Any]
    fitness: float = float("-inf")
    individual_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: str | None = None  # MLflow run ID if tracked

    def __lt__(self, other: "Individual") -> bool:
        return self.fitness < other.fitness


# ---------------------------------------------------------------------------
# Moses optimizer
# ---------------------------------------------------------------------------


class MosesHyperparamOptimizer:
    """Moses-style evolutionary hyperparameter search.

    Parameters
    ----------
    space:
        The hyperparameter search space.
    objective:
        A callable ``(config: dict) → float`` returning the fitness score.
        Higher is better.  This is typically a function that trains a model
        with the given config and returns a validation metric.
    population_size:
        Number of individuals per generation.
    n_generations:
        Maximum number of generations to evolve.
    tournament_size:
        Number of individuals competing in each tournament selection.
    mutation_sigma:
        Standard deviation of Gaussian mutations (as a fraction of range).
    atomspace:
        Optional AtomSpace to record evaluated configurations as atoms and
        warm-start from previous evaluations.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        space: HyperparamSpace,
        objective: Callable[[dict[str, Any]], float],
        population_size: int = 20,
        n_generations: int = 10,
        tournament_size: int = 3,
        mutation_sigma: float = 0.15,
        atomspace: AtomSpace | None = None,
        seed: int | None = None,
    ) -> None:
        self._space = space
        self._objective = objective
        self._pop_size = population_size
        self._n_gen = n_generations
        self._tourn_size = tournament_size
        self._sigma = mutation_sigma
        self._as = atomspace
        self._rng = random.Random(seed)
        self._population: list[Individual] = []
        self._hall_of_fame: list[Individual] = []
        self._generation: int = 0

    # ------------------------------------------------------------------
    # Warm start from AtomSpace
    # ------------------------------------------------------------------

    def _warm_start_configs(self) -> list[dict[str, Any]]:
        """Load previously evaluated configs from the AtomSpace."""
        if self._as is None:
            return []
        configs = []
        for atom in self._as:
            if not isinstance(atom, EvaluationLink):
                continue
            if len(atom.outgoing) < 2:
                continue
            pred = atom.outgoing[0]
            if not isinstance(pred, PredicateNode) or pred.name != "moses_individual":
                continue
            args = atom.outgoing[1]
            if not isinstance(args, ListLink):
                continue
            config: dict[str, Any] = {}
            for child in args.outgoing:
                if isinstance(child, EvaluationLink) and len(child.outgoing) == 2:
                    k_node, v_node = child.outgoing
                    if isinstance(k_node, PredicateNode) and isinstance(
                        v_node, (NumberNode, ConceptNode)
                    ):
                        key = k_node.name
                        try:
                            val: Any = float(v_node.name)
                        except (ValueError, AttributeError):
                            val = v_node.name
                        config[key] = val
            if config:
                configs.append(config)
        return configs

    # ------------------------------------------------------------------
    # AtomSpace recording
    # ------------------------------------------------------------------

    def _record_individual(self, individual: Individual) -> None:
        if self._as is None:
            return
        run_id = individual.run_id or individual.individual_id
        param_atoms = []
        for k, v in individual.config.items():
            k_node = self._as.add(PredicateNode(k))
            v_node = self._as.add(
                NumberNode(float(v)) if isinstance(v, (int, float)) else ConceptNode(str(v))
            )
            param_atoms.append(self._as.add(EvaluationLink([k_node, v_node])))
        fitness_node = self._as.add(NumberNode(individual.fitness))
        run_node = self._as.add(ConceptNode(run_node_name(run_id)))
        self._as.add(
            EvaluationLink([
                PredicateNode("moses_individual"),
                ListLink([run_node, fitness_node, *param_atoms]),
            ])
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _tournament_select(self) -> Individual:
        contestants = self._rng.sample(
            self._population, min(self._tourn_size, len(self._population))
        )
        return max(contestants, key=lambda ind: ind.fitness)

    # ------------------------------------------------------------------
    # Main evolution loop
    # ------------------------------------------------------------------

    def run(self) -> list[Individual]:
        """Evolve the population and return the hall of fame (best individuals seen).

        Returns the hall of fame sorted by descending fitness.
        """
        # Initialise population
        warm_configs = self._warm_start_configs()
        initial_configs = warm_configs[: self._pop_size]
        while len(initial_configs) < self._pop_size:
            initial_configs.append(self._space.sample(self._rng))

        self._population = [Individual(config=c) for c in initial_configs]

        for gen in range(self._n_gen):
            self._generation = gen
            # Evaluate
            for ind in self._population:
                if math.isinf(ind.fitness):
                    ind.fitness = self._objective(ind.config)
                    self._record_individual(ind)

            # Update hall of fame
            self._population.sort(key=lambda i: i.fitness, reverse=True)
            for ind in self._population[:3]:
                if not any(abs(ind.fitness - hof.fitness) < 1e-9 for hof in self._hall_of_fame):
                    self._hall_of_fame.append(copy.deepcopy(ind))

            # Breed next generation
            next_gen: list[Individual] = []
            # Elitism: keep top 10%
            elite_count = max(1, self._pop_size // 10)
            next_gen.extend(copy.deepcopy(ind) for ind in self._population[:elite_count])

            while len(next_gen) < self._pop_size:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                child_config = self._space.crossover(p1.config, p2.config, self._rng)
                child_config = self._space.mutate(child_config, self._rng, self._sigma)
                next_gen.append(Individual(config=child_config))

            self._population = next_gen

        self._hall_of_fame.sort(key=lambda i: i.fitness, reverse=True)
        return self._hall_of_fame

    @property
    def best(self) -> Individual | None:
        """Best individual found so far."""
        if not self._hall_of_fame:
            return None
        return self._hall_of_fame[0]

    @property
    def population(self) -> list[Individual]:
        return list(self._population)

    @property
    def generation(self) -> int:
        return self._generation
