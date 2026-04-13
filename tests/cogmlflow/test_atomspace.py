"""Tests for CogMLflow AtomSpace entity and atom types."""

import pytest

from mlflow.cogmlflow.entities.atom import (
    ConceptNode,
    EvaluationLink,
    InheritanceLink,
    ListLink,
    MemberLink,
    NumberNode,
    PredicateNode,
    TruthValue,
    VariableNode,
)
from mlflow.cogmlflow.entities.atomspace import AtomSpace

# ---------------------------------------------------------------------------
# TruthValue
# ---------------------------------------------------------------------------


class TestTruthValue:
    def test_default_values(self):
        tv = TruthValue()
        assert tv.strength == 1.0
        assert tv.confidence == 1.0

    def test_custom_values(self):
        tv = TruthValue(0.7, 0.9)
        assert tv.strength == 0.7
        assert tv.confidence == 0.9

    def test_out_of_range_strength_raises(self):
        with pytest.raises(ValueError, match="strength"):
            TruthValue(1.5, 0.5)

    def test_out_of_range_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            TruthValue(0.5, -0.1)

    def test_revision_combines_tv(self):
        tv1 = TruthValue(0.8, 0.6)
        tv2 = TruthValue(0.4, 0.6)
        revised = tv1.revision(tv2)
        # Result should be between the two strengths
        assert 0.4 <= revised.strength <= 0.8
        # Confidence should be higher than either input
        assert revised.confidence >= 0.6

    def test_immutability(self):
        tv = TruthValue(0.5, 0.5)
        with pytest.raises(AttributeError, match="cannot|has no setter"):
            tv.strength = 0.9  # frozen dataclass


# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------


class TestNodes:
    def test_concept_node_equality(self):
        a = ConceptNode("hello")
        b = ConceptNode("hello")
        assert a == b
        assert hash(a) == hash(b)

    def test_concept_node_different_names_not_equal(self):
        assert ConceptNode("a") != ConceptNode("b")

    def test_node_requires_str_name(self):
        with pytest.raises(TypeError, match="str"):
            ConceptNode(123)

    def test_number_node_value(self):
        n = NumberNode(3.14)
        assert n.value == 3.14
        assert n.name == "3.14"

    def test_predicate_node_type(self):
        p = PredicateNode("metric:accuracy")
        assert p.type == "PredicateNode"
        assert p.name == "metric:accuracy"

    def test_variable_node(self):
        v = VariableNode("$X")
        assert v.name == "$X"


# ---------------------------------------------------------------------------
# Link types
# ---------------------------------------------------------------------------


class TestLinks:
    def test_evaluation_link_equality(self):
        pred = PredicateNode("p")
        args = ListLink([ConceptNode("a"), NumberNode(1.0)])
        e1 = EvaluationLink([pred, args])
        e2 = EvaluationLink([pred, args])
        assert e1 == e2

    def test_link_outgoing_preserved(self):
        nodes = [ConceptNode("x"), ConceptNode("y")]
        link = InheritanceLink(nodes)
        assert len(link.outgoing) == 2
        assert link.outgoing[0] == ConceptNode("x")

    def test_different_link_types_not_equal(self):
        nodes = [ConceptNode("a"), ConceptNode("b")]
        il = InheritanceLink(nodes)
        ml = MemberLink(nodes)
        assert il != ml


# ---------------------------------------------------------------------------
# AtomSpace
# ---------------------------------------------------------------------------


class TestAtomSpace:
    def test_add_and_retrieve(self):
        space = AtomSpace()
        node = ConceptNode("test")
        stored = space.add(node)
        assert stored == node
        assert node in space

    def test_idempotent_add(self):
        space = AtomSpace()
        node = ConceptNode("dup")
        space.add(node)
        space.add(node)
        assert len(space) == 1

    def test_add_link_indexes_incoming(self):
        space = AtomSpace()
        a = space.add(ConceptNode("a"))
        b = space.add(ConceptNode("b"))
        link = space.add(InheritanceLink([a, b]))
        incoming_b = space.get_incoming(b)
        assert link in incoming_b

    def test_get_atoms_by_type(self):
        space = AtomSpace()
        space.add(ConceptNode("c1"))
        space.add(ConceptNode("c2"))
        space.add(PredicateNode("p1"))
        concepts = space.get_atoms_by_type("ConceptNode")
        assert len(concepts) == 2

    def test_remove(self):
        space = AtomSpace()
        node = space.add(ConceptNode("removeme"))
        assert len(space) == 1
        removed = space.remove(node)
        assert removed is True
        assert len(space) == 0

    def test_remove_nonexistent_returns_false(self):
        space = AtomSpace()
        result = space.remove(ConceptNode("ghost"))
        assert result is False

    def test_decay_sti(self):
        space = AtomSpace()
        node = space.add(ConceptNode("n"))
        node.sti = 100.0
        space.decay_sti(factor=0.5)
        assert node.sti == pytest.approx(50.0)

    def test_get_by_sti_ordering(self):
        space = AtomSpace()
        high = space.add(ConceptNode("high"))
        low = space.add(ConceptNode("low"))
        high.sti = 100.0
        low.sti = 1.0
        ranked = space.get_by_sti(top_n=2)
        assert ranked[0] == high

    def test_tv_revision_on_duplicate_add(self):
        space = AtomSpace()
        node = ConceptNode("tv_test")
        space.add(node, tv=TruthValue(0.8, 0.5))
        space.add(ConceptNode("tv_test"), tv=TruthValue(0.4, 0.5))
        stored = space.get(ConceptNode("tv_test"))
        # Revised TV strength should be between 0.4 and 0.8
        assert 0.4 <= stored.tv.strength <= 0.8

    def test_serialization(self):
        space = AtomSpace(name="test")
        space.add(ConceptNode("x"))
        d = space.to_dict()
        assert d["name"] == "test"
        assert len(d["atoms"]) == 1

    def test_pattern_match_ground_node(self):
        space = AtomSpace()
        space.add(ConceptNode("match_me"))
        results = space.pattern_match(ConceptNode("match_me"))
        assert len(results) == 1

    def test_pattern_match_missing_node(self):
        space = AtomSpace()
        results = space.pattern_match(ConceptNode("not_here"))
        assert results == []

    def test_pattern_match_variable(self):
        space = AtomSpace()
        space.add(ConceptNode("a"))
        space.add(ConceptNode("b"))
        results = space.pattern_match(VariableNode("$X"))
        # Should bind to both atoms
        assert len(results) >= 2
        bound_names = {r.get("$X") for r in results}
        assert ConceptNode("a") in bound_names
        assert ConceptNode("b") in bound_names
