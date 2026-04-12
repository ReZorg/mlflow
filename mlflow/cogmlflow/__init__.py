"""
CogMLflow — OpenCog Cognitive Orchestration Workbench for MLflow.

CogMLflow extends MLflow with a cognitively-aware ML orchestration layer
powered by a pure-Python AtomSpace (compatible with OpenCog), Probabilistic
Logic Networks (PLN), Economic Attention Networks (ECAN), and Moses-style
evolutionary hyperparameter search.

Quick start::

    from mlflow.cogmlflow import CogMLflowClient
    from mlflow.cogmlflow.orchestrator import accuracy_goal

    client = CogMLflowClient()
    exp_id = client.create_experiment("my_experiment")

    # Log some runs
    run = client.create_run(exp_id, run_name="run_1")
    client.log_metric(run.info.run_id, "accuracy", 0.85)
    client.log_param(run.info.run_id, "learning_rate", "0.01")

    # Get cognitive suggestions
    suggestions = client.get_cognitive_suggestions(primary_metric="accuracy")
    for s in suggestions:
        print(s.message)

    # Tune hyperparameters with Moses
    best = client.tune(
        space={"lr": {"type": "float", "low": 1e-5, "high": 0.1, "log_scale": True}},
        objective=lambda cfg: -abs(cfg["lr"] - 0.001),  # toy objective
        n_generations=5,
    )
    print("Best config:", best.config)

    # Autonomous research loop
    with client.auto_research(
        runner=lambda cfg: {"accuracy": 0.9 - abs(cfg.get("lr", 0.01) - 0.001)},
        initial_configs=[{"lr": 0.1}, {"lr": 0.01}],
        n_iterations=10,
    ) as researcher:
        best_candidate = researcher.run()

Integration modes
-----------------
* **Native mode** (default): All data lives in-memory in the AtomSpace.
* **Shadow mode**: Wrap an existing MLflow tracking store; AtomSpace mirrors it.

  ::

      from mlflow.store.tracking.file_store import FileStore
      from mlflow.cogmlflow.atomspace import AtomSpaceTrackingStore

      delegate = FileStore("/tmp/mlruns")
      shadow_store = AtomSpaceTrackingStore(delegate=delegate)
      client = CogMLflowClient(store=shadow_store)
"""

from mlflow.cogmlflow.atomspace import AtomSpaceTrackingStore, MLflowAtomTranslator
from mlflow.cogmlflow.client import CogMLflowClient
from mlflow.cogmlflow.ecan import AttentionValue, ECANAttentionBank, ExperimentAttentionManager
from mlflow.cogmlflow.entities import (
    AtomSpace,
    CogMLflowOntology,
    ConceptNode,
    EvaluationLink,
    ImplicationLink,
    InheritanceLink,
    ListLink,
    MemberLink,
    NumberNode,
    PredicateNode,
    SequentialAndLink,
    TruthValue,
)
from mlflow.cogmlflow.moses import HyperparamSpace, Individual, MosesHyperparamOptimizer
from mlflow.cogmlflow.orchestrator import (
    AutoResearcher,
    CogMLflowGoalSystem,
    CognitiveScheduler,
    ExperimentCandidate,
    Goal,
    ResearchIteration,
    accuracy_goal,
    latency_goal,
    loss_goal,
)
from mlflow.cogmlflow.pln import (
    DataDriftRule,
    GeneralizationRule,
    HyperparamSensitivityRule,
    InferenceResult,
    ModelComparisonRule,
    OverfittingDetectionRule,
    PLNEngine,
    PLNRule,
)
from mlflow.cogmlflow.suggestion_engine import CognitiveSuggestionEngine, Suggestion

__all__ = [
    # Client
    "CogMLflowClient",
    # Entities
    "AtomSpace",
    "CogMLflowOntology",
    "ConceptNode",
    "PredicateNode",
    "NumberNode",
    "EvaluationLink",
    "ListLink",
    "InheritanceLink",
    "ImplicationLink",
    "MemberLink",
    "SequentialAndLink",
    "TruthValue",
    # AtomSpace store
    "AtomSpaceTrackingStore",
    "MLflowAtomTranslator",
    # PLN
    "PLNEngine",
    "PLNRule",
    "InferenceResult",
    "ModelComparisonRule",
    "OverfittingDetectionRule",
    "GeneralizationRule",
    "HyperparamSensitivityRule",
    "DataDriftRule",
    # ECAN
    "AttentionValue",
    "ECANAttentionBank",
    "ExperimentAttentionManager",
    # Moses
    "HyperparamSpace",
    "Individual",
    "MosesHyperparamOptimizer",
    # Orchestrator
    "AutoResearcher",
    "CogMLflowGoalSystem",
    "CognitiveScheduler",
    "ExperimentCandidate",
    "Goal",
    "ResearchIteration",
    "accuracy_goal",
    "latency_goal",
    "loss_goal",
    # Suggestion engine
    "CognitiveSuggestionEngine",
    "Suggestion",
]
