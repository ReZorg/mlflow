"""
MLflow entity ↔ Atomese bidirectional translator.

Converts MLflow Python entity objects into AtomSpace atoms and vice versa.
"""

from __future__ import annotations

from mlflow.cogmlflow.entities.atom import (
    ConceptNode,
    EvaluationLink,
    ListLink,
    MemberLink,
    NumberNode,
    PredicateNode,
)
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.entities.ontology import (
    CogMLflowOntology,
    experiment_node_name,
    run_node_name,
)


class MLflowAtomTranslator:
    """Translates between MLflow entity objects and AtomSpace atoms.

    This class owns a single AtomSpace and an ontology helper, and provides
    high-level ``ingest_*`` / ``read_*`` methods for each MLflow entity type.
    """

    def __init__(self, atomspace: AtomSpace | None = None) -> None:
        self._as = atomspace if atomspace is not None else AtomSpace("mlflow")
        self._onto = CogMLflowOntology(self._as)

    @property
    def atomspace(self) -> AtomSpace:
        return self._as

    # ------------------------------------------------------------------
    # Write (MLflow entities → AtomSpace)
    # ------------------------------------------------------------------

    def ingest_experiment(self, experiment) -> ConceptNode:
        """Translate an ``mlflow.entities.Experiment`` into atoms."""
        node = self._onto.add_experiment(
            experiment.experiment_id,
            experiment.name,
            lifecycle_stage=experiment.lifecycle_stage or "active",
        )
        if experiment.creation_time:
            self._as.add(
                EvaluationLink([
                    PredicateNode("creation_time"),
                    ListLink([node, NumberNode(float(experiment.creation_time))]),
                ])
            )
        if experiment.tags:
            for tag_key, tag_value in experiment.tags.items():
                self._as.add(
                    EvaluationLink([
                        PredicateNode(f"experiment_tag:{tag_key}"),
                        ListLink([node, ConceptNode(str(tag_value))]),
                    ])
                )
        return node

    def ingest_run_info(self, run_info) -> ConceptNode:
        """Translate an ``mlflow.entities.RunInfo`` into atoms."""
        return self._onto.add_run(
            run_id=run_info.run_id,
            experiment_id=run_info.experiment_id,
            status=run_info.status or "RUNNING",
            start_time=run_info.start_time or 0,
        )

    def ingest_run(self, run) -> ConceptNode:
        """Translate a full ``mlflow.entities.Run`` (info + data) into atoms."""
        node = self.ingest_run_info(run.info)
        if run.data:
            for key, value in (run.data.metrics or {}).items():
                self._onto.log_metric(run.info.run_id, key, value)
            for key, value in (run.data.params or {}).items():
                self._onto.log_param(run.info.run_id, key, str(value))
            for key, value in (run.data.tags or {}).items():
                self._onto.log_tag(run.info.run_id, key, str(value))
        return node

    def ingest_metric(self, run_id: str, metric) -> EvaluationLink:
        """Translate an ``mlflow.entities.Metric`` into an EvaluationLink."""
        return self._onto.log_metric(
            run_id=run_id,
            key=metric.key,
            value=metric.value,
            step=metric.step or 0,
            timestamp=metric.timestamp or 0,
        )

    def ingest_param(self, run_id: str, param) -> EvaluationLink:
        """Translate an ``mlflow.entities.Param`` into an EvaluationLink."""
        return self._onto.log_param(run_id=run_id, key=param.key, value=str(param.value))

    def ingest_run_tag(self, run_id: str, tag) -> EvaluationLink:
        """Translate an ``mlflow.entities.RunTag`` into an EvaluationLink."""
        return self._onto.log_tag(run_id=run_id, key=tag.key, value=str(tag.value))

    # ------------------------------------------------------------------
    # Read (AtomSpace → dicts)
    # ------------------------------------------------------------------

    def read_experiment(self, experiment_id: str) -> dict:
        """Return a dict representation of an experiment from the AtomSpace."""
        node = self._as.get(ConceptNode(experiment_node_name(experiment_id)))
        if node is None:
            return {}
        result: dict = {"experiment_id": experiment_id}
        for link in self._as.get_incoming(node):
            if not isinstance(link, EvaluationLink):
                continue
            if len(link.outgoing) < 2:
                continue
            pred = link.outgoing[0]
            args = link.outgoing[1]
            if not isinstance(pred, PredicateNode):
                continue
            if pred.name == "experiment_name" and isinstance(args, ListLink):
                if len(args.outgoing) >= 2:
                    result["name"] = args.outgoing[1].name
            elif pred.name == "creation_time" and isinstance(args, ListLink):
                if len(args.outgoing) >= 2:
                    result["creation_time"] = args.outgoing[1].value
        return result

    def read_run(self, run_id: str) -> dict:
        """Return a dict representation of a run from the AtomSpace."""
        result: dict = {
            "run_id": run_id,
            "metrics": dict(self._onto.get_run_metrics(run_id)),
            "params": dict(self._onto.get_run_params(run_id)),
        }
        # Read status
        run_node = ConceptNode(run_node_name(run_id))
        for link in self._as.get_incoming(self._as.get(run_node) or run_node):
            if not isinstance(link, EvaluationLink):
                continue
            if len(link.outgoing) < 2:
                continue
            pred = link.outgoing[0]
            if isinstance(pred, PredicateNode) and pred.name.startswith("run_status:"):
                result["status"] = pred.name[len("run_status:") :]
        return result

    def get_runs_for_experiment(self, experiment_id: str) -> list[str]:
        """Return all run_ids belonging to an experiment."""
        exp_node = self._as.get(ConceptNode(experiment_node_name(experiment_id)))
        if exp_node is None:
            return []
        run_ids = []
        for link in self._as.get_incoming(exp_node):
            if isinstance(link, MemberLink) and len(link.outgoing) >= 2:
                member = link.outgoing[0]
                if isinstance(member, ConceptNode) and member.name.startswith("run:"):
                    run_ids.append(member.name[len("run:") :])
        return run_ids
