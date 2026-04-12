"""
Pure-Python in-memory AtomSpace (hypergraph knowledge store).

The AtomSpace is a *set* of atoms; every atom has a unique identity
determined by its type and content.  Adding the same atom twice returns the
existing atom.  Atoms can be retrieved, queried, and pattern-matched.

This implementation is intentionally compatible with the OpenCog AtomSpace
Python API so that it can be replaced by the real C++ bindings when
available::

    try:
        from opencog.atomspace import AtomSpace, types
    except ImportError:
        from mlflow.cogmlflow.entities.atomspace import AtomSpace

Usage::

    from mlflow.cogmlflow.entities.atom import ConceptNode, PredicateNode, EvaluationLink, ListLink
    from mlflow.cogmlflow.entities.atomspace import AtomSpace

    a = AtomSpace()
    experiment = a.add(ConceptNode("exp_1"))
    metric_pred = a.add(PredicateNode("accuracy"))
    value = a.add(NumberNode(0.95))
    fact = a.add(EvaluationLink([metric_pred, ListLink([experiment, value])]))
"""

from __future__ import annotations

import threading
from typing import Iterator

from mlflow.cogmlflow.entities.atom import (
    Atom,
    Link,
    Node,
    TruthValue,
    VariableNode,
)


class AtomSpace:
    """Thread-safe in-memory hypergraph store.

    Atoms are indexed by their hash for O(1) lookup.  The ``add`` method is
    idempotent — adding an equivalent atom returns the already-stored atom
    and merges truth values via the PLN revision rule.
    """

    def __init__(self, name: str = "default") -> None:
        self._name = name
        self._lock = threading.RLock()
        # Primary store: hash → Atom
        self._atoms: dict[int, Atom] = {}
        # Secondary index: type name → set of hashes
        self._by_type: dict[str, set[int]] = {}
        # Incoming index: atom hash → set of Link hashes that contain it
        self._incoming: dict[int, set[int]] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, atom: Atom, tv: TruthValue | None = None) -> Atom:
        """Add *atom* to this space and return the canonical stored atom.

        If an equivalent atom already exists its truth value is revised with
        the incoming TV (PLN revision rule).  If *tv* is provided it overrides
        the atom's own TV before insertion.
        """
        if tv is not None:
            atom.tv = tv
        key = hash(atom)
        with self._lock:
            if key in self._atoms:
                existing = self._atoms[key]
                existing.tv = existing.tv.revision(atom.tv)
                return existing
            # Register atom
            self._atoms[key] = atom
            type_name = atom.type
            self._by_type.setdefault(type_name, set()).add(key)
            self._incoming.setdefault(key, set())
            # Update incoming index for links
            if isinstance(atom, Link):
                for child in atom.outgoing:
                    child_key = hash(child)
                    self._incoming.setdefault(child_key, set()).add(key)
            return atom

    def remove(self, atom: Atom) -> bool:
        """Remove *atom* from this space.  Returns True if it was present."""
        key = hash(atom)
        with self._lock:
            if key not in self._atoms:
                return False
            stored = self._atoms.pop(key)
            self._by_type.get(stored.type, set()).discard(key)
            # Remove from incoming of children
            if isinstance(stored, Link):
                for child in stored.outgoing:
                    self._incoming.get(hash(child), set()).discard(key)
            self._incoming.pop(key, None)
            return True

    def get(self, atom: Atom) -> Atom | None:
        """Return the stored atom equivalent to *atom*, or None."""
        return self._atoms.get(hash(atom))

    def __contains__(self, atom: Atom) -> bool:
        return hash(atom) in self._atoms

    def __len__(self) -> int:
        return len(self._atoms)

    def __iter__(self) -> Iterator[Atom]:
        with self._lock:
            return iter(list(self._atoms.values()))

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_atoms_by_type(self, atom_type: type | str) -> list[Atom]:
        """Return all atoms whose type matches *atom_type* (or its subclasses)."""
        type_name = atom_type if isinstance(atom_type, str) else atom_type._type
        with self._lock:
            keys = self._by_type.get(type_name, set())
            return [self._atoms[k] for k in keys if k in self._atoms]

    def get_incoming(self, atom: Atom) -> list[Link]:
        """Return all Links that have *atom* in their outgoing set."""
        key = hash(atom)
        with self._lock:
            incoming_keys = self._incoming.get(key, set())
            results = []
            for k in incoming_keys:
                stored = self._atoms.get(k)
                if stored is not None and isinstance(stored, Link):
                    results.append(stored)
            return results

    def get_node(self, node_type: type, name: str) -> Node | None:
        """Convenience: return an existing Node of given type and name."""
        candidate = node_type(name)
        return self.get(candidate)

    # ------------------------------------------------------------------
    # Pattern matching (simplified unification)
    # ------------------------------------------------------------------

    def pattern_match(self, pattern: Atom) -> list[dict[str, Atom]]:
        """Match *pattern* against atoms in this space.

        VariableNode instances in the pattern are bound to concrete atoms.
        Returns a list of variable-binding dicts (possibly empty).

        Only a subset of patterns is supported in this pure-Python
        implementation:

        * Patterns consisting entirely of ground atoms return ``[{}]`` if the
          atom exists, else ``[]``.
        * Patterns with VariableNode children are matched against candidates.
        """
        return self._match(pattern, {})

    def _match(self, pattern: Atom, bindings: dict[str, Atom]) -> list[dict[str, Atom]]:
        """Recursive pattern matching."""
        if isinstance(pattern, VariableNode):
            var_name = pattern.name
            if var_name in bindings:
                return [bindings] if bindings[var_name] == pattern else [bindings]
            # Bind variable to every atom in the space
            solutions = []
            for atom in self:
                new_bindings = {**bindings, var_name: atom}
                solutions.append(new_bindings)
            return solutions

        if isinstance(pattern, Node):
            # Ground node — must exist
            stored = self.get(pattern)
            return [bindings] if stored is not None else []

        if isinstance(pattern, Link):
            has_vars = any(isinstance(c, VariableNode) for c in pattern.outgoing)
            if not has_vars:
                # Fully ground link
                stored = self.get(pattern)
                return [bindings] if stored is not None else []
            # At least one variable — iterate over all links of same type
            solutions = []
            for candidate in self.get_atoms_by_type(pattern.type):
                if not isinstance(candidate, Link):
                    continue
                if len(candidate.outgoing) != len(pattern.outgoing):
                    continue
                current_bindings = [bindings]
                for pat_child, cand_child in zip(pattern.outgoing, candidate.outgoing):
                    next_bindings = []
                    for b in current_bindings:
                        next_bindings.extend(self._match_concrete(pat_child, cand_child, b))
                    current_bindings = next_bindings
                    if not current_bindings:
                        break
                solutions.extend(current_bindings)
            return solutions

        return []

    def _match_concrete(
        self, pattern: Atom, concrete: Atom, bindings: dict[str, Atom]
    ) -> list[dict[str, Atom]]:
        if isinstance(pattern, VariableNode):
            var_name = pattern.name
            if var_name in bindings:
                return [bindings] if bindings[var_name] == concrete else []
            return [{**bindings, var_name: concrete}]
        if pattern == concrete:
            return [bindings]
        return []

    # ------------------------------------------------------------------
    # Attention helpers (ECAN)
    # ------------------------------------------------------------------

    def get_by_sti(self, top_n: int = 10) -> list[Atom]:
        """Return the *top_n* atoms with the highest STI values."""
        with self._lock:
            atoms = list(self._atoms.values())
        return sorted(atoms, key=lambda a: a.sti, reverse=True)[:top_n]

    def decay_sti(self, factor: float = 0.95) -> None:
        """Apply a uniform STI decay to all atoms (simulates forgetting)."""
        with self._lock:
            for atom in self._atoms.values():
                atom.sti *= factor

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the AtomSpace to a JSON-serialisable dict."""

        def _atom_to_dict(a: Atom) -> dict:
            d: dict = {
                "type": a.type,
                "tv": {"strength": a.tv.strength, "confidence": a.tv.confidence},
                "sti": a.sti,
                "lti": a.lti,
            }
            if isinstance(a, Node):
                d["name"] = a.name
            elif isinstance(a, Link):
                d["outgoing"] = [_atom_to_dict(c) for c in a.outgoing]
            return d

        with self._lock:
            return {
                "name": self._name,
                "atoms": [_atom_to_dict(a) for a in self._atoms.values()],
            }

    def __repr__(self) -> str:
        return f"AtomSpace(name='{self._name}', size={len(self)})"
