"""
Core Atom types for the CogMLflow AtomSpace.

CogMLflow uses a pure-Python hypergraph (AtomSpace) that mirrors the OpenCog
AtomSpace API.  No external OpenCog dependency is required; if the real
``opencog.atomspace`` package is available it will be preferred automatically
via the ``cogmlflow.entities.atomspace`` module.

Atom hierarchy
--------------
Atom
 ├── Node                (has a name)
 │    ├── ConceptNode
 │    ├── PredicateNode
 │    ├── NumberNode
 │    ├── SchemaNode
 │    └── VariableNode
 └── Link                (has an ordered list of outgoing Atoms)
      ├── EvaluationLink
      ├── ListLink
      ├── InheritanceLink
      ├── SimilarityLink
      ├── SequentialAndLink
      ├── ProducedByLink
      └── MemberLink
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class TruthValue:
    """Probabilistic truth value used by PLN.

    Attributes:
        strength: Mean probability estimate in [0, 1].
        confidence: Confidence (amount of evidence) in [0, 1].
    """

    strength: float = 1.0
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength must be in [0, 1], got {self.strength}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def revision(self, other: "TruthValue") -> "TruthValue":
        """PLN revision rule: combine two independent TV estimates."""
        w1 = self.confidence
        w2 = other.confidence
        total = w1 + w2 - w1 * w2
        if total == 0.0:
            return TruthValue(0.0, 0.0)
        strength = (
            (
                self.strength * w1
                + other.strength * w2
                - self.strength * other.strength * (w1 + w2) / (1e-9 + total)
            )
            / (1e-9 + total)
            * total
        )
        # Simplified: weighted average
        strength = (self.strength * w1 + other.strength * w2) / (w1 + w2) if (w1 + w2) > 0 else 0.0
        confidence = w1 + w2 - w1 * w2
        return TruthValue(min(max(strength, 0.0), 1.0), min(confidence, 1.0))

    def __repr__(self) -> str:
        return f"TruthValue(strength={self.strength:.3f}, confidence={self.confidence:.3f})"


# Sentinel default TV
DEFAULT_TV = TruthValue(1.0, 1.0)
UNDEFINED_TV = TruthValue(0.0, 0.0)


class Atom:
    """Base class for all atoms in the AtomSpace.

    Atoms are identified by their *type* and *content* (name for Nodes,
    outgoing set for Links).  Two atoms with the same type and content are
    the same atom — the AtomSpace ensures uniqueness.
    """

    _type: str = "Atom"

    def __init__(self, tv: TruthValue | None = None, attention: float = 0.0) -> None:
        self.tv: TruthValue = tv if tv is not None else DEFAULT_TV
        # Short-term importance (ECAN)
        self.sti: float = attention
        # Long-term importance (ECAN)
        self.lti: float = 0.0
        # Arbitrary metadata dict (not part of atom identity)
        self._meta: dict[str, Any] = {}

    @property
    def type(self) -> str:
        return self._type

    def with_tv(self, tv: TruthValue) -> "Atom":
        """Return a copy of this atom with the given truth value."""
        self.tv = tv
        return self

    # Subclasses must implement these for identity and hashing
    def _identity_key(self) -> tuple:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Atom):
            return False
        return type(self) is type(other) and self._identity_key() == other._identity_key()

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._identity_key()))

    def __repr__(self) -> str:
        return f"{self._type}(...)"


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


class Node(Atom):
    """An atom with a string name (leaf in the hypergraph)."""

    _type = "Node"

    def __init__(self, name: str, tv: TruthValue | None = None) -> None:
        super().__init__(tv=tv)
        if not isinstance(name, str):
            raise TypeError(f"Node name must be str, got {type(name)}")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _identity_key(self) -> tuple:
        return (self._name,)

    def __repr__(self) -> str:
        return f"{self._type}('{self._name}')"


class ConceptNode(Node):
    """A node representing a concept (entity, category)."""

    _type = "ConceptNode"


class PredicateNode(Node):
    """A node representing a predicate (relation name)."""

    _type = "PredicateNode"


class NumberNode(Node):
    """A node whose name encodes a numeric value."""

    _type = "NumberNode"

    def __init__(self, value: float, tv: TruthValue | None = None) -> None:
        super().__init__(str(value), tv=tv)
        self._value = value

    @property
    def value(self) -> float:
        return self._value


class SchemaNode(Node):
    """A node representing a named schema or function."""

    _type = "SchemaNode"


class VariableNode(Node):
    """A node used as a pattern variable in queries."""

    _type = "VariableNode"


class TypeNode(Node):
    """A node representing an atom type."""

    _type = "TypeNode"


# ---------------------------------------------------------------------------
# Link types
# ---------------------------------------------------------------------------


class Link(Atom):
    """An atom that connects an ordered sequence of other atoms (hyperedge)."""

    _type = "Link"

    def __init__(self, outgoing: Sequence[Atom], tv: TruthValue | None = None) -> None:
        super().__init__(tv=tv)
        self._outgoing: tuple[Atom, ...] = tuple(outgoing)

    @property
    def outgoing(self) -> tuple[Atom, ...]:
        return self._outgoing

    def _identity_key(self) -> tuple:
        return tuple(hash(a) for a in self._outgoing)

    def __repr__(self) -> str:
        children = ", ".join(repr(a) for a in self._outgoing)
        return f"{self._type}({children})"


class EvaluationLink(Link):
    """EvaluationLink(predicate, arguments) — represents a ground fact."""

    _type = "EvaluationLink"


class ListLink(Link):
    """Ordered list of atoms."""

    _type = "ListLink"


class InheritanceLink(Link):
    """InheritanceLink(child, parent) — IS-A relationship."""

    _type = "InheritanceLink"


class SimilarityLink(Link):
    """SimilarityLink(a, b) — symmetric similarity."""

    _type = "SimilarityLink"


class SequentialAndLink(Link):
    """SequentialAndLink(a, b, ...) — ordered temporal sequence."""

    _type = "SequentialAndLink"


class ProducedByLink(Link):
    """ProducedByLink(artifact, run) — data provenance edge."""

    _type = "ProducedByLink"


class MemberLink(Link):
    """MemberLink(element, set_node) — set membership."""

    _type = "MemberLink"


class ImplicationLink(Link):
    """ImplicationLink(antecedent, consequent) — probabilistic implication."""

    _type = "ImplicationLink"


class AndLink(Link):
    """AndLink(a, b, ...) — conjunction."""

    _type = "AndLink"


class OrLink(Link):
    """OrLink(a, b, ...) — disjunction."""

    _type = "OrLink"


class NotLink(Link):
    """NotLink(a) — negation."""

    _type = "NotLink"


class ExecutionLink(Link):
    """ExecutionLink(schema, args, result) — function application."""

    _type = "ExecutionLink"


class ContextLink(Link):
    """ContextLink(context, atom) — context-sensitive truth."""

    _type = "ContextLink"


# Registry of all atom types by name (used for serialisation and pattern matching)
ATOM_TYPE_REGISTRY: dict[str, type] = {
    cls._type: cls
    for cls in [
        ConceptNode,
        PredicateNode,
        NumberNode,
        SchemaNode,
        VariableNode,
        TypeNode,
        EvaluationLink,
        ListLink,
        InheritanceLink,
        SimilarityLink,
        SequentialAndLink,
        ProducedByLink,
        MemberLink,
        ImplicationLink,
        AndLink,
        OrLink,
        NotLink,
        ExecutionLink,
        ContextLink,
    ]
}
