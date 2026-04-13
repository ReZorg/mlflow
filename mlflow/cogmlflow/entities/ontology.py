"""
CogMLflow ontology: AtomSpace Atom types that represent MLflow entities.

This module defines the semantic layer that maps MLflow's experiment tracking
concepts to OpenCog-compatible Atom types.  The ontology is encoded as
ConceptNode names and InheritanceLinks so that PLN can reason over them.

CogMLflow Atom naming conventions
----------------------------------
* Experiment  →  ConceptNode("experiment:<experiment_id>")
* Run         →  ConceptNode("run:<run_id>")
* Metric      →  EvaluationLink(PredicateNode("metric:<key>"),
                               ListLink(ConceptNode("run:<run_id>"),
                                        NumberNode(<value>)))
* Param       →  EvaluationLink(PredicateNode("param:<key>"),
                               ListLink(ConceptNode("run:<run_id>"),
                                        ConceptNode(<value>)))
* Tag         →  EvaluationLink(PredicateNode("tag:<key>"),
                               ListLink(ConceptNode("run:<run_id>"),
                                        ConceptNode(<value>)))
* Artifact    →  ConceptNode("artifact:<run_id>:<path>")
* ModelVersion →  ConceptNode("model_version:<name>:<version>")
"""

from __future__ import annotations

from mlflow.cogmlflow.entities.atom import (
    ConceptNode,
    EvaluationLink,
    InheritanceLink,
    ListLink,
    MemberLink,
    NumberNode,
    PredicateNode,
    SequentialAndLink,
)
from mlflow.cogmlflow.entities.atomspace import AtomSpace

# ---------------------------------------------------------------------------
# Atom name helpers
# ---------------------------------------------------------------------------


def experiment_node_name(experiment_id: str) -> str:
    return f"experiment:{experiment_id}"


def run_node_name(run_id: str) -> str:
    return f"run:{run_id}"


def metric_pred_name(key: str) -> str:
    return f"metric:{key}"


def param_pred_name(key: str) -> str:
    return f"param:{key}"


def tag_pred_name(key: str) -> str:
    return f"tag:{key}"


def artifact_node_name(run_id: str, path: str) -> str:
    return f"artifact:{run_id}:{path}"


def model_version_node_name(name: str, version: str) -> str:
    return f"model_version:{name}:{version}"


def model_node_name(name: str) -> str:
    return f"model:{name}"


# ---------------------------------------------------------------------------
# Ontology bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ontology(atomspace: AtomSpace) -> None:
    """Add the base CogMLflow type hierarchy to an AtomSpace.

    This creates InheritanceLinks that allow PLN to reason about the
    type hierarchy of CogMLflow entities.
    """
    # Base types
    entity = ConceptNode("CogMLflowEntity")
    experiment_type = ConceptNode("ExperimentType")
    run_type = ConceptNode("RunType")
    metric_type = ConceptNode("MetricType")
    param_type = ConceptNode("ParamType")
    artifact_type = ConceptNode("ArtifactType")
    model_type = ConceptNode("ModelType")

    for node in [
        entity,
        experiment_type,
        run_type,
        metric_type,
        param_type,
        artifact_type,
        model_type,
    ]:
        atomspace.add(node)

    for child in [experiment_type, run_type, metric_type, param_type, artifact_type, model_type]:
        atomspace.add(InheritanceLink([child, entity]))

    # Predicates for lifecycle stages
    for stage in ("active", "deleted"):
        atomspace.add(PredicateNode(f"lifecycle:{stage}"))

    # Predicates for run status
    for status in ("RUNNING", "SCHEDULED", "FINISHED", "FAILED", "KILLED"):
        atomspace.add(PredicateNode(f"run_status:{status}"))


# ---------------------------------------------------------------------------
# Ontology factories — translate MLflow entities to Atoms
# ---------------------------------------------------------------------------


class CogMlflowOntology:
    """Convenience class for building and querying CogMLflow Atoms.

    All methods are stateless helpers; they only require an AtomSpace
    reference so they can ensure atom uniqueness.
    """

    def __init__(self, atomspace: AtomSpace) -> None:
        self._as = atomspace
        bootstrap_ontology(atomspace)

    # -- Experiment ----------------------------------------------------------

    def add_experiment(
        self,
        experiment_id: str,
        name: str,
        lifecycle_stage: str = "active",
    ) -> ConceptNode:
        node = self._as.add(ConceptNode(experiment_node_name(experiment_id)))
        self._as.add(
            EvaluationLink([
                PredicateNode("experiment_name"),
                ListLink([node, ConceptNode(name)]),
            ])
        )
        self._as.add(
            EvaluationLink([
                PredicateNode(f"lifecycle:{lifecycle_stage}"),
                ListLink([node]),
            ])
        )
        self._as.add(InheritanceLink([node, ConceptNode("ExperimentType")]))
        return node

    # -- Run -----------------------------------------------------------------

    def add_run(
        self,
        run_id: str,
        experiment_id: str,
        status: str = "RUNNING",
        start_time: int = 0,
    ) -> ConceptNode:
        run_node = self._as.add(ConceptNode(run_node_name(run_id)))
        exp_node = self._as.add(ConceptNode(experiment_node_name(experiment_id)))
        self._as.add(MemberLink([run_node, exp_node]))
        self._as.add(InheritanceLink([run_node, ConceptNode("RunType")]))
        self._as.add(
            EvaluationLink([
                PredicateNode(f"run_status:{status}"),
                ListLink([run_node]),
            ])
        )
        if start_time:
            self._as.add(
                EvaluationLink([
                    PredicateNode("start_time"),
                    ListLink([run_node, NumberNode(float(start_time))]),
                ])
            )
        return run_node

    # -- Metric --------------------------------------------------------------

    def log_metric(
        self,
        run_id: str,
        key: str,
        value: float,
        step: int = 0,
        timestamp: int = 0,
    ) -> EvaluationLink:
        run_node = self._as.add(ConceptNode(run_node_name(run_id)))
        pred = self._as.add(PredicateNode(metric_pred_name(key)))
        val_node = self._as.add(NumberNode(value))
        step_node = self._as.add(NumberNode(float(step)))
        ts_node = self._as.add(NumberNode(float(timestamp)))
        link = self._as.add(
            EvaluationLink([
                pred,
                ListLink([run_node, val_node, step_node, ts_node]),
            ])
        )
        # Boost STI slightly for recently logged metrics
        link.sti += 1.0
        return link

    # -- Param ---------------------------------------------------------------

    def log_param(self, run_id: str, key: str, value: str) -> EvaluationLink:
        run_node = self._as.add(ConceptNode(run_node_name(run_id)))
        pred = self._as.add(PredicateNode(param_pred_name(key)))
        val_node = self._as.add(ConceptNode(str(value)))
        return self._as.add(EvaluationLink([pred, ListLink([run_node, val_node])]))

    # -- Tag -----------------------------------------------------------------

    def log_tag(self, run_id: str, key: str, value: str) -> EvaluationLink:
        run_node = self._as.add(ConceptNode(run_node_name(run_id)))
        pred = self._as.add(PredicateNode(tag_pred_name(key)))
        val_node = self._as.add(ConceptNode(str(value)))
        return self._as.add(EvaluationLink([pred, ListLink([run_node, val_node])]))

    # -- Artifact provenance -------------------------------------------------

    def log_artifact(self, run_id: str, path: str) -> ConceptNode:
        artifact_node = self._as.add(ConceptNode(artifact_node_name(run_id, path)))
        run_node = self._as.add(ConceptNode(run_node_name(run_id)))
        self._as.add(InheritanceLink([artifact_node, ConceptNode("ArtifactType")]))
        from mlflow.cogmlflow.entities.atom import ProducedByLink

        self._as.add(ProducedByLink([artifact_node, run_node]))
        return artifact_node

    # -- Model Version -------------------------------------------------------

    def add_model_version(self, name: str, version: str, run_id: str | None = None) -> ConceptNode:
        node = self._as.add(ConceptNode(model_version_node_name(name, version)))
        self._as.add(InheritanceLink([node, ConceptNode("ModelType")]))
        if run_id:
            run_node = self._as.add(ConceptNode(run_node_name(run_id)))
            self._as.add(
                EvaluationLink([
                    PredicateNode("model_trained_in"),
                    ListLink([node, run_node]),
                ])
            )
        return node

    # -- Temporal chain of metrics -------------------------------------------

    def add_metric_sequence(
        self,
        run_id: str,
        key: str,
        values: list[tuple[int, float]],  # [(step, value), ...]
    ) -> SequentialAndLink:
        """Encode an ordered metric time series as a SequentialAndLink."""
        run_node = self._as.add(ConceptNode(run_node_name(run_id)))
        atoms = []
        for step, value in values:
            pred = self._as.add(PredicateNode(metric_pred_name(key)))
            val_node = self._as.add(NumberNode(value))
            step_node = self._as.add(NumberNode(float(step)))
            ev = self._as.add(EvaluationLink([pred, ListLink([run_node, val_node, step_node])]))
            atoms.append(ev)
        return self._as.add(SequentialAndLink(atoms))

    # -- Query helpers -------------------------------------------------------

    def get_run_metrics(self, run_id: str) -> list[tuple[str, float]]:
        """Return (metric_key, value) pairs for a run."""
        run_node = ConceptNode(run_node_name(run_id))
        results = []
        for atom in self._as:
            from mlflow.cogmlflow.entities.atom import EvaluationLink as EL
            from mlflow.cogmlflow.entities.atom import ListLink as LL

            if not isinstance(atom, EL):
                continue
            if len(atom.outgoing) < 2:
                continue
            pred, args = atom.outgoing[0], atom.outgoing[1]
            if not isinstance(pred, PredicateNode):
                continue
            if not pred.name.startswith("metric:"):
                continue
            if not isinstance(args, LL):
                continue
            if len(args.outgoing) < 2:
                continue
            if args.outgoing[0] == run_node:
                val = args.outgoing[1]
                if isinstance(val, NumberNode):
                    results.append((pred.name[len("metric:") :], val.value))
        return results

    def get_run_params(self, run_id: str) -> list[tuple[str, str]]:
        """Return (param_key, value) pairs for a run."""
        run_node = ConceptNode(run_node_name(run_id))
        results = []
        for atom in self._as:
            from mlflow.cogmlflow.entities.atom import EvaluationLink as EL
            from mlflow.cogmlflow.entities.atom import ListLink as LL

            if not isinstance(atom, EL):
                continue
            if len(atom.outgoing) < 2:
                continue
            pred, args = atom.outgoing[0], atom.outgoing[1]
            if not isinstance(pred, PredicateNode):
                continue
            if not pred.name.startswith("param:"):
                continue
            if not isinstance(args, LL):
                continue
            if len(args.outgoing) >= 2 and args.outgoing[0] == run_node:
                results.append((pred.name[len("param:") :], args.outgoing[1].name))
        return results
