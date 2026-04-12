"""Tests for PLN engine and rules."""

import pytest

from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.entities.ontology import CogMLflowOntology
from mlflow.cogmlflow.pln.engine import PLNEngine, TruthValue, conjunction, deduction, negation
from mlflow.cogmlflow.pln.rules import (
    DataDriftRule,
    GeneralizationRule,
    HyperparamSensitivityRule,
    ModelComparisonRule,
    OverfittingDetectionRule,
)


@pytest.fixture
def populated_space():
    space = AtomSpace()
    onto = CogMLflowOntology(space)
    onto.add_experiment("exp1", "TestExp")
    onto.add_run("run_a", "exp1", status="FINISHED")
    onto.add_run("run_b", "exp1", status="FINISHED")
    onto.log_metric("run_a", "accuracy", 0.90)
    onto.log_metric("run_b", "accuracy", 0.75)
    onto.log_metric("run_a", "train_accuracy", 0.95)
    onto.log_metric("run_a", "val_accuracy", 0.90)
    onto.log_metric("run_b", "train_accuracy", 0.93)
    onto.log_metric("run_b", "val_accuracy", 0.75)
    onto.log_param("run_a", "lr", "0.01")
    onto.log_param("run_b", "lr", "0.1")
    onto.log_param("run_a", "batch_size", "32")
    onto.log_param("run_b", "batch_size", "64")
    return space


class TestTVArithmetic:
    def test_deduction(self):
        tv1 = TruthValue(0.8, 0.9)
        tv2 = TruthValue(0.7, 0.8)
        result = deduction(tv1, tv2)
        assert result.strength == pytest.approx(0.56)

    def test_conjunction(self):
        tv1 = TruthValue(0.6, 0.8)
        tv2 = TruthValue(0.9, 0.7)
        result = conjunction(tv1, tv2)
        assert result.strength == pytest.approx(0.6)
        assert result.confidence == pytest.approx(0.7)

    def test_negation(self):
        tv = TruthValue(0.3, 0.8)
        result = negation(tv)
        assert result.strength == pytest.approx(0.7)
        assert result.confidence == pytest.approx(0.8)


class TestModelComparisonRule:
    def test_fires_when_two_runs_exist(self, populated_space):
        rule = ModelComparisonRule("accuracy")
        results = list(rule.apply(populated_space))
        assert len(results) >= 1

    def test_higher_accuracy_wins(self, populated_space):
        rule = ModelComparisonRule("accuracy", higher_is_better=True)
        results = list(rule.apply(populated_space))
        assert any(r.tv.strength > 0.5 for r in results)


class TestOverfittingDetectionRule:
    def test_detects_overfitting(self, populated_space):
        rule = OverfittingDetectionRule("train_accuracy", "val_accuracy", threshold=0.01)
        results = list(rule.apply(populated_space))
        assert len(results) >= 1
        assert any(r.tv.strength > 0.5 for r in results)


class TestHyperparamSensitivityRule:
    def test_fires_with_numeric_params(self, populated_space):
        rule = HyperparamSensitivityRule("accuracy")
        results = list(rule.apply(populated_space))
        assert len(results) >= 1

    def test_sensitivity_strength_in_range(self, populated_space):
        rule = HyperparamSensitivityRule("accuracy")
        for r in rule.apply(populated_space):
            assert 0.0 <= r.tv.strength <= 1.0


class TestGeneralizationRule:
    def test_emits_generalization_estimate(self, populated_space):
        rule = GeneralizationRule("train_accuracy", "val_accuracy")
        results = list(rule.apply(populated_space))
        assert len(results) >= 1


class TestDataDriftRule:
    def test_detects_drift_when_metric_degrades(self, populated_space):
        rule = DataDriftRule("run_a", "run_b", "accuracy", param_keys=["lr"])
        results = list(rule.apply(populated_space))
        # run_b accuracy (0.75) < run_a accuracy (0.90) → drift suspected
        assert len(results) == 1
        assert results[0].tv.strength > 0.5

    def test_no_drift_when_metric_improves(self, populated_space):
        rule = DataDriftRule("run_b", "run_a", "accuracy")
        results = list(rule.apply(populated_space))
        assert len(results) == 0


class TestPLNEngine:
    def test_engine_runs_all_rules(self, populated_space):
        engine = PLNEngine(populated_space)
        engine.add_rule(ModelComparisonRule("accuracy"))
        engine.add_rule(OverfittingDetectionRule("train_accuracy", "val_accuracy"))
        results = engine.run(steps=1)
        assert len(results) >= 1

    def test_engine_history_accumulated(self, populated_space):
        engine = PLNEngine(populated_space)
        engine.add_rule(ModelComparisonRule("accuracy"))
        engine.run(steps=1)
        engine.run(steps=1)
        assert len(engine.inference_history) >= 1

    def test_engine_clear_history(self, populated_space):
        engine = PLNEngine(populated_space)
        engine.add_rule(ModelComparisonRule("accuracy"))
        engine.run(steps=1)
        engine.clear_history()
        assert engine.inference_history == []
