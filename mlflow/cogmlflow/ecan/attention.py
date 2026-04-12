"""
ECAN (Economic Attention Networks) for CogMLflow.

ECAN assigns *importance* to AtomSpace atoms (experiments, runs) and uses an
economic metaphor — atoms "rent" memory proportional to their importance.
Attention flows via Hebbian spreading: atoms that frequently co-occur in
inferences gain mutual importance.

Key concepts
------------
* **STI** (Short-Term Importance): ephemeral salience; reflects recency and
  current relevance.  Decays over time (forgetting curve).
* **LTI** (Long-Term Importance): stable importance; updated slowly.  Reflects
  structural significance (many dependents, foundational experiments).
* **AF** (Attentional Focus): the set of atoms with STI above a threshold.
  Only AF atoms are considered for most PLN inferences (focused reasoning).
* **Wage** / **Rent**: stimuli into the system increase STI (wages); atoms pay
  rent proportional to their STI.

This module provides:
* ``AttentionValue`` — (STI, LTI) pair for an atom.
* ``ECANAttentionBank`` — manages all attention values and implements spreading.
* ``ExperimentAttentionManager`` — higher-level wrapper that maps MLflow run IDs
  to attention values and provides scheduling primitives.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from mlflow.cogmlflow.entities.atom import Atom, ConceptNode
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.entities.ontology import experiment_node_name, run_node_name

# ---------------------------------------------------------------------------
# Attention Value
# ---------------------------------------------------------------------------


@dataclass
class AttentionValue:
    """Short-Term and Long-Term Importance values for one atom."""

    sti: float = 0.0
    lti: float = 0.0
    vlti: bool = False  # Very Long-Term Importance (never forgotten)

    def decay_sti(self, factor: float = 0.95) -> None:
        if not self.vlti:
            self.sti *= factor

    def decay_lti(self, factor: float = 0.999) -> None:
        if not self.vlti:
            self.lti *= factor

    def __repr__(self) -> str:
        return f"AV(sti={self.sti:.2f}, lti={self.lti:.2f})"


# ---------------------------------------------------------------------------
# ECAN Attention Bank
# ---------------------------------------------------------------------------


class ECANAttentionBank:
    """Manages AttentionValues for atoms in an AtomSpace.

    The bank allocates a fixed total *budget* of STI across all atoms.  When
    an atom is stimulated its STI increases; decay is applied periodically.
    Hebbian spreading propagates STI along link outgoing sets.
    """

    def __init__(
        self,
        atomspace: AtomSpace,
        total_sti_budget: float = 1000.0,
        af_boundary_sti: float = 10.0,
        decay_rate: float = 0.95,
        spread_fraction: float = 0.1,
    ) -> None:
        self._as = atomspace
        self._total_budget = total_sti_budget
        self._af_boundary = af_boundary_sti
        self._decay_rate = decay_rate
        self._spread_fraction = spread_fraction
        self._lock = threading.RLock()
        # atom_hash → AttentionValue
        self._avs: dict[int, AttentionValue] = {}

    # ------------------------------------------------------------------
    # Attention value access
    # ------------------------------------------------------------------

    def get_av(self, atom: Atom) -> AttentionValue:
        key = hash(atom)
        with self._lock:
            if key not in self._avs:
                self._avs[key] = AttentionValue(sti=atom.sti, lti=atom.lti)
            return self._avs[key]

    def set_sti(self, atom: Atom, sti: float) -> None:
        av = self.get_av(atom)
        with self._lock:
            av.sti = sti
            atom.sti = sti

    def set_lti(self, atom: Atom, lti: float) -> None:
        av = self.get_av(atom)
        with self._lock:
            av.lti = lti
            atom.lti = lti

    def stimulate(self, atom: Atom, amount: float = 1.0) -> None:
        """Increase the STI of *atom* by *amount* (wage payment)."""
        av = self.get_av(atom)
        with self._lock:
            av.sti = min(av.sti + amount, self._total_budget)
            atom.sti = av.sti
            # Slowly increase LTI proportional to cumulative stimulation
            av.lti = min(av.lti + amount * 0.01, 100.0)
            atom.lti = av.lti

    def penalise(self, atom: Atom, amount: float = 1.0) -> None:
        """Decrease the STI of *atom* (rent payment)."""
        av = self.get_av(atom)
        with self._lock:
            av.sti = max(av.sti - amount, 0.0)
            atom.sti = av.sti

    # ------------------------------------------------------------------
    # Attentional Focus
    # ------------------------------------------------------------------

    def get_attentional_focus(self) -> list[Atom]:
        """Return atoms whose STI is above the AF boundary."""
        with self._lock:
            return [atom for atom in self._as if self.get_av(atom).sti >= self._af_boundary]

    # ------------------------------------------------------------------
    # Hebbian STI spreading
    # ------------------------------------------------------------------

    def spread_sti(self) -> None:
        """Propagate STI from links to their outgoing atoms.

        A fraction ``spread_fraction`` of each link's STI is distributed
        equally across its outgoing atoms.
        """
        with self._lock:
            atoms = list(self._as)
        for atom in atoms:
            from mlflow.cogmlflow.entities.atom import Link

            if not isinstance(atom, Link):
                continue
            av = self.get_av(atom)
            if av.sti <= 0:
                continue
            if not atom.outgoing:
                continue
            spread = av.sti * self._spread_fraction
            per_child = spread / len(atom.outgoing)
            for child in atom.outgoing:
                child_stored = self._as.get(child)
                if child_stored is not None:
                    self.stimulate(child_stored, per_child)
            self.penalise(atom, spread)

    # ------------------------------------------------------------------
    # Decay cycle
    # ------------------------------------------------------------------

    def decay_cycle(self) -> None:
        """Apply STI/LTI decay to all atoms (simulate time passing)."""
        with self._lock:
            for av in self._avs.values():
                av.decay_sti(self._decay_rate)
                av.decay_lti()
        # Sync back to atoms
        for atom in self._as:
            key = hash(atom)
            if key in self._avs:
                atom.sti = self._avs[key].sti
                atom.lti = self._avs[key].lti

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_atoms(self, top_n: int = 20) -> list[tuple[float, Atom]]:
        """Return the top-N atoms ranked by (STI + 0.1 * LTI)."""
        scores = []
        for atom in self._as:
            av = self.get_av(atom)
            score = av.sti + 0.1 * av.lti
            scores.append((score, atom))
        return sorted(scores, key=lambda x: x[0], reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# High-level experiment attention manager
# ---------------------------------------------------------------------------


class ExperimentAttentionManager:
    """Maps MLflow experiment/run IDs to ECAN attention values.

    Usage::

        mgr = ExperimentAttentionManager(atomspace)

        # Reward a promising run
        mgr.reward_run("abc123", metric_improvement=0.05)

        # Get the next run to execute (highest expected value)
        best_runs = mgr.get_priority_queue(top_k=5)
    """

    def __init__(self, atomspace: AtomSpace, **bank_kwargs) -> None:
        self._as = atomspace
        self._bank = ECANAttentionBank(atomspace, **bank_kwargs)

    @property
    def bank(self) -> ECANAttentionBank:
        return self._bank

    def initialise_run(self, run_id: str, base_sti: float = 5.0) -> None:
        """Add a run to the attention system with a starter STI."""
        node = self._as.add(ConceptNode(run_node_name(run_id)))
        self._bank.stimulate(node, base_sti)

    def initialise_experiment(self, experiment_id: str, base_sti: float = 10.0) -> None:
        node = self._as.add(ConceptNode(experiment_node_name(experiment_id)))
        self._bank.stimulate(node, base_sti)
        self._bank.set_lti(node, 5.0)

    def reward_run(self, run_id: str, metric_improvement: float) -> None:
        """Reward a run for a positive metric improvement.

        The STI stimulus is proportional to the improvement, clipped at 50.
        """
        node = self._as.get(ConceptNode(run_node_name(run_id)))
        if node is None:
            return
        stimulus = min(abs(metric_improvement) * 100.0, 50.0)
        self._bank.stimulate(node, stimulus)

    def penalise_run(self, run_id: str, reason: str = "") -> None:
        """Penalise a failed/stale run."""
        node = self._as.get(ConceptNode(run_node_name(run_id)))
        if node is not None:
            self._bank.penalise(node, 5.0)

    def get_priority_queue(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Return (run_id, score) pairs for the top-K runs by attention score."""
        results = []
        for score, atom in self._bank.rank_atoms(top_n=top_k * 3):
            if isinstance(atom, ConceptNode) and atom.name.startswith("run:"):
                run_id = atom.name[len("run:") :]
                results.append((run_id, score))
                if len(results) >= top_k:
                    break
        return results

    def step(self) -> None:
        """Advance one ECAN cycle: spread STI, then decay."""
        self._bank.spread_sti()
        self._bank.decay_cycle()
