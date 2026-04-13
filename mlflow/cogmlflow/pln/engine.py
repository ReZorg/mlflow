"""
PLN (Probabilistic Logic Networks) inference engine for CogMLflow.

This module provides:

* ``PLNRule`` — abstract base class for inference rules.
* ``PLNEngine`` — forward-chaining inference engine that applies rules to an
  AtomSpace and returns newly derived atoms with updated truth values.

PLN truth-value semantics
--------------------------
A TruthValue ``(s, c)`` represents:

* ``s`` (strength): the estimated probability that a proposition is true.
* ``c`` (confidence): the amount of evidence; ``c = n / (n + k)`` where ``n``
  is the number of observations and ``k`` is a sensitivity parameter (default 1).

Reference: Goertzel et al., *Probabilistic Logic Networks*, Springer 2009.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

from mlflow.cogmlflow.entities.atom import Atom, TruthValue
from mlflow.cogmlflow.entities.atomspace import AtomSpace

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PLN Truth-Value arithmetic
# ---------------------------------------------------------------------------


def deduction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """PLN Deduction rule: P(A→C) given P(A→B) and P(B→C)."""
    s = tv1.strength * tv2.strength
    c = tv1.confidence * tv2.confidence
    return TruthValue(min(s, 1.0), min(c, 1.0))


def abduction(tv1: TruthValue, tv2: TruthValue, base_rate: float = 0.5) -> TruthValue:
    """PLN Abduction rule."""
    if base_rate == 0:
        return TruthValue(0.0, 0.0)
    s = tv1.strength * tv2.strength / base_rate
    c = tv1.confidence * tv2.confidence * 0.5
    return TruthValue(min(s, 1.0), min(c, 1.0))


def inversion(tv: TruthValue, base_rate: float = 0.5, target_rate: float = 0.5) -> TruthValue:
    """PLN Inversion (Bayes): P(B|A) → P(A|B)."""
    if tv.strength == 0:
        return TruthValue(0.0, tv.confidence)
    s = (
        tv.strength
        * base_rate
        / (tv.strength * base_rate + (1 - tv.strength) * (1 - target_rate) + 1e-9)
    )
    c = tv.confidence * 0.5
    return TruthValue(min(s, 1.0), min(c, 1.0))


def conjunction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """PLN fuzzy conjunction: min(s1, s2), min(c1, c2)."""
    return TruthValue(min(tv1.strength, tv2.strength), min(tv1.confidence, tv2.confidence))


def disjunction(tv1: TruthValue, tv2: TruthValue) -> TruthValue:
    """PLN fuzzy disjunction: max(s1, s2), min(c1, c2)."""
    return TruthValue(max(tv1.strength, tv2.strength), min(tv1.confidence, tv2.confidence))


def negation(tv: TruthValue) -> TruthValue:
    """PLN negation: 1 - s, same confidence."""
    return TruthValue(1.0 - tv.strength, tv.confidence)


# ---------------------------------------------------------------------------
# PLN Rule base class
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Result produced by a PLN rule firing."""

    rule_name: str
    conclusion: Atom
    premises: list[Atom] = field(default_factory=list)
    tv: TruthValue = field(default_factory=lambda: TruthValue(1.0, 0.0))
    explanation: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule_name,
            "conclusion": repr(self.conclusion),
            "tv": {"strength": self.tv.strength, "confidence": self.tv.confidence},
            "explanation": self.explanation,
        }


class PLNRule(ABC):
    """Abstract base class for PLN inference rules.

    Subclasses implement ``apply`` which receives the current AtomSpace and
    returns zero or more ``InferenceResult`` objects.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this rule."""

    @abstractmethod
    def apply(self, atomspace: AtomSpace) -> Iterator[InferenceResult]:
        """Apply this rule to *atomspace* and yield inference results."""


# ---------------------------------------------------------------------------
# PLN inference engine
# ---------------------------------------------------------------------------


class PLNEngine:
    """Forward-chaining PLN inference engine.

    Rules are registered with ``add_rule``.  Calling ``run`` applies all rules
    against the AtomSpace and inserts newly derived atoms back into the space.
    Multiple forward-chaining cycles can be requested via the ``steps``
    parameter.
    """

    def __init__(self, atomspace: AtomSpace) -> None:
        self._as = atomspace
        self._rules: list[PLNRule] = []
        self._history: list[InferenceResult] = []

    def add_rule(self, rule: PLNRule) -> "PLNEngine":
        """Register a PLN rule.  Returns self for chaining."""
        self._rules.append(rule)
        return self

    def run(self, steps: int = 1) -> list[InferenceResult]:
        """Execute *steps* forward-chaining cycles.

        Each cycle applies every registered rule to the current AtomSpace and
        inserts newly derived atoms.  Returns all newly derived InferenceResults.
        """
        new_results: list[InferenceResult] = []
        for step in range(steps):
            step_results: list[InferenceResult] = []
            for rule in self._rules:
                try:
                    for result in rule.apply(self._as):
                        result.conclusion = self._as.add(result.conclusion, tv=result.tv)
                        step_results.append(result)
                        _log.debug(
                            "[PLN step=%d] %s → %s (tv=%s)",
                            step,
                            rule.name,
                            repr(result.conclusion),
                            result.tv,
                        )
                except Exception as e:
                    _log.warning("PLN rule %r failed: %s", rule.name, e)
            new_results.extend(step_results)
            if not step_results:
                _log.debug("[PLN] No new inferences at step %d; stopping.", step)
                break
        self._history.extend(new_results)
        return new_results

    @property
    def inference_history(self) -> list[InferenceResult]:
        """All inference results produced since the engine was created."""
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    @property
    def atomspace(self) -> AtomSpace:
        return self._as
