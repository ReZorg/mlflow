"""
AtomSpaceTrackingStore — an MLflow tracking store backend backed by an
in-process AtomSpace hypergraph.

Integration modes
-----------------
* **Native mode** (``delegate=None``): The AtomSpace IS the tracking store.
  All data lives in the hypergraph.  Suitable for development, testing, and
  in-memory cognitive reasoning sessions.

* **Shadow mode** (``delegate=<AbstractStore>``): All writes go to both the
  delegate store (your existing SQLite / Postgres / etc.) and the AtomSpace.
  Reads are served from the delegate; the AtomSpace is used only for
  cognitive queries (PLN, ECAN, pattern matching).

Usage::

    from mlflow.cogmlflow.atomspace.tracking_store import AtomSpaceTrackingStore

    # Native mode
    store = AtomSpaceTrackingStore()

    # Shadow mode (mirror an existing file store)
    from mlflow.store.tracking.file_store import FileStore

    delegate = FileStore("/tmp/mlruns")
    store = AtomSpaceTrackingStore(delegate=delegate)
"""

from __future__ import annotations

import time
import uuid

from mlflow.cogmlflow.atomspace.translator import MLflowAtomTranslator
from mlflow.cogmlflow.entities.atom import ConceptNode, EvaluationLink, ListLink, PredicateNode
from mlflow.cogmlflow.entities.atomspace import AtomSpace
from mlflow.cogmlflow.entities.ontology import (
    experiment_node_name,
    run_node_name,
)
from mlflow.entities import (
    Experiment,
    ExperimentTag,
    Metric,
    Param,
    Run,
    RunData,
    RunInfo,
    RunStatus,
    RunTag,
    ViewType,
)
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, RESOURCE_DOES_NOT_EXIST
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking.abstract_store import AbstractStore


class AtomSpaceTrackingStore(AbstractStore):
    """MLflow tracking store that uses an in-memory AtomSpace as its backend.

    When *delegate* is provided (shadow mode) all mutating operations are
    forwarded to the delegate store first; the AtomSpace is kept in sync so
    that cognitive queries always have fresh data.

    When *delegate* is ``None`` (native mode) all data lives only in the
    AtomSpace.
    """

    def __init__(
        self,
        delegate: AbstractStore | None = None,
        atomspace: AtomSpace | None = None,
    ) -> None:
        super().__init__()
        self._delegate = delegate
        self._as = atomspace if atomspace is not None else AtomSpace("cogmlflow")
        self._translator = MLflowAtomTranslator(self._as)
        self._onto = self._translator._onto
        # Native-mode in-memory stores
        self._experiments: dict[str, Experiment] = {}
        self._runs: dict[str, Run] = {}
        self._next_experiment_id: int = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_experiment_id(self) -> str:
        eid = str(self._next_experiment_id)
        self._next_experiment_id += 1
        return eid

    def _sync_experiment(self, experiment: Experiment) -> None:
        """Push an experiment into the AtomSpace."""
        self._translator.ingest_experiment(experiment)

    def _sync_run(self, run: Run) -> None:
        """Push a run into the AtomSpace."""
        self._translator.ingest_run(run)

    # ------------------------------------------------------------------
    # AbstractStore — Experiments
    # ------------------------------------------------------------------

    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=1000,
        filter_string=None,
        order_by=None,
        page_token=None,
    ) -> PagedList:
        if self._delegate:
            result = self._delegate.search_experiments(
                view_type=view_type,
                max_results=max_results,
                filter_string=filter_string,
                order_by=order_by,
                page_token=page_token,
            )
            for exp in result:
                self._sync_experiment(exp)
            return result

        exps = list(self._experiments.values())
        if view_type == ViewType.ACTIVE_ONLY:
            exps = [e for e in exps if e.lifecycle_stage == LifecycleStage.ACTIVE]
        elif view_type == ViewType.DELETED_ONLY:
            exps = [e for e in exps if e.lifecycle_stage == LifecycleStage.DELETED]
        return PagedList(exps[:max_results], None)

    def create_experiment(self, name: str, artifact_location: str, tags) -> str:
        if self._delegate:
            exp_id = self._delegate.create_experiment(name, artifact_location, tags)
            exp = self._delegate.get_experiment(exp_id)
            self._sync_experiment(exp)
            return exp_id

        # Check for duplicate names
        for exp in self._experiments.values():
            if exp.name == name and exp.lifecycle_stage == LifecycleStage.ACTIVE:
                raise MlflowException(
                    f"Experiment '{name}' already exists.",
                    RESOURCE_ALREADY_EXISTS,
                )
        exp_id = self._new_experiment_id()
        now = int(time.time() * 1000)
        tag_list = list(tags) if tags else []
        exp = Experiment(
            experiment_id=exp_id,
            name=name,
            artifact_location=artifact_location or f"cogmlflow-artifacts/{exp_id}",
            lifecycle_stage=LifecycleStage.ACTIVE,
            tags=tag_list,
            creation_time=now,
            last_update_time=now,
        )
        self._experiments[exp_id] = exp
        self._sync_experiment(exp)
        return exp_id

    def get_experiment(self, experiment_id: str) -> Experiment:
        if self._delegate:
            exp = self._delegate.get_experiment(experiment_id)
            self._sync_experiment(exp)
            return exp
        exp = self._experiments.get(str(experiment_id))
        if exp is None:
            raise MlflowException(
                f"Experiment {experiment_id!r} does not exist.",
                RESOURCE_DOES_NOT_EXIST,
            )
        return exp

    def get_experiment_by_name(self, experiment_name: str) -> Experiment | None:
        if self._delegate:
            exp = self._delegate.get_experiment_by_name(experiment_name)
            if exp:
                self._sync_experiment(exp)
            return exp
        for exp in self._experiments.values():
            if exp.name == experiment_name and exp.lifecycle_stage == LifecycleStage.ACTIVE:
                return exp
        return None

    def delete_experiment(self, experiment_id: str) -> None:
        if self._delegate:
            self._delegate.delete_experiment(experiment_id)
        exp = self._experiments.get(str(experiment_id))
        if exp:
            exp._lifecycle_stage = LifecycleStage.DELETED
            # Update AtomSpace
            node = self._as.get(ConceptNode(experiment_node_name(str(experiment_id))))
            if node:
                self._as.add(
                    EvaluationLink([
                        PredicateNode("lifecycle:deleted"),
                        ListLink([node]),
                    ])
                )

    def restore_experiment(self, experiment_id: str) -> None:
        if self._delegate:
            self._delegate.restore_experiment(experiment_id)
        exp = self._experiments.get(str(experiment_id))
        if exp:
            exp._lifecycle_stage = LifecycleStage.ACTIVE

    def rename_experiment(self, experiment_id: str, new_name: str) -> None:
        if self._delegate:
            self._delegate.rename_experiment(experiment_id, new_name)
        exp = self._experiments.get(str(experiment_id))
        if exp:
            exp._set_name(new_name)

    def set_experiment_tag(self, experiment_id: str, tag: ExperimentTag) -> None:
        if self._delegate:
            self._delegate.set_experiment_tag(experiment_id, tag)
        exp = self._experiments.get(str(experiment_id))
        if exp:
            exp._add_tag(tag)

    # ------------------------------------------------------------------
    # AbstractStore — Runs
    # ------------------------------------------------------------------

    def create_run(
        self, experiment_id: str, user_id: str, start_time: int, tags, run_name: str
    ) -> Run:
        if self._delegate:
            run = self._delegate.create_run(experiment_id, user_id, start_time, tags, run_name)
            self._sync_run(run)
            return run

        run_id = str(uuid.uuid4()).replace("-", "")
        tag_list = list(tags) if tags else []
        run_info = RunInfo(
            run_id=run_id,
            experiment_id=str(experiment_id),
            user_id=user_id,
            status=RunStatus.to_string(RunStatus.RUNNING),
            start_time=start_time or int(time.time() * 1000),
            end_time=None,
            lifecycle_stage=LifecycleStage.ACTIVE,
            artifact_uri=f"cogmlflow-artifacts/{experiment_id}/{run_id}/artifacts",
            run_name=run_name,
        )
        run_data = RunData(metrics={}, params={}, tags={t.key: t.value for t in tag_list})
        run = Run(run_info=run_info, run_data=run_data)
        self._runs[run_id] = run
        self._sync_run(run)
        return run

    def get_run(self, run_id: str) -> Run:
        if self._delegate:
            run = self._delegate.get_run(run_id)
            self._sync_run(run)
            return run
        run = self._runs.get(run_id)
        if run is None:
            raise MlflowException(
                f"Run {run_id!r} does not exist.",
                RESOURCE_DOES_NOT_EXIST,
            )
        return run

    def update_run_info(self, run_id: str, run_status, end_time, run_name: str) -> RunInfo:
        if self._delegate:
            info = self._delegate.update_run_info(run_id, run_status, end_time, run_name)
            return info
        run = self._runs.get(run_id)
        if run is None:
            raise MlflowException(f"Run {run_id!r} does not exist.", RESOURCE_DOES_NOT_EXIST)
        updated_info = run.info._copy_with_overrides(
            status=run_status, end_time=end_time, run_name=run_name
        )
        self._runs[run_id] = Run(run_info=updated_info, run_data=run.data)
        # Update status atom
        run_node = self._as.get(ConceptNode(run_node_name(run_id)))
        if run_node:
            status_str = (
                RunStatus.to_string(run_status) if not isinstance(run_status, str) else run_status
            )
            self._as.add(
                EvaluationLink([
                    PredicateNode(f"run_status:{status_str}"),
                    ListLink([run_node]),
                ])
            )
        return updated_info

    def delete_run(self, run_id: str) -> None:
        if self._delegate:
            self._delegate.delete_run(run_id)
        run = self._runs.get(run_id)
        if run:
            self._runs[run_id] = Run(
                run_info=run.info._copy_with_overrides(lifecycle_stage=LifecycleStage.DELETED),
                run_data=run.data,
            )

    def restore_run(self, run_id: str) -> None:
        if self._delegate:
            self._delegate.restore_run(run_id)
        run = self._runs.get(run_id)
        if run:
            self._runs[run_id] = Run(
                run_info=run.info._copy_with_overrides(lifecycle_stage=LifecycleStage.ACTIVE),
                run_data=run.data,
            )

    # ------------------------------------------------------------------
    # AbstractStore — Metrics / Params / Tags
    # ------------------------------------------------------------------

    def log_metric(self, run_id: str, metric: Metric) -> None:
        if self._delegate:
            self._delegate.log_metric(run_id, metric)
        run = self._runs.get(run_id)
        if run:
            updated_metrics = dict(run.data.metrics)
            updated_metrics[metric.key] = metric.value
            self._runs[run_id] = Run(
                run_info=run.info,
                run_data=RunData(
                    metrics=updated_metrics,
                    params=run.data.params,
                    tags=run.data.tags,
                ),
            )
        self._onto.log_metric(
            run_id, metric.key, metric.value, metric.step or 0, metric.timestamp or 0
        )

    def log_param(self, run_id: str, param: Param) -> None:
        if self._delegate:
            self._delegate.log_param(run_id, param)
        run = self._runs.get(run_id)
        if run:
            updated_params = dict(run.data.params)
            updated_params[param.key] = param.value
            self._runs[run_id] = Run(
                run_info=run.info,
                run_data=RunData(
                    metrics=run.data.metrics,
                    params=updated_params,
                    tags=run.data.tags,
                ),
            )
        self._onto.log_param(run_id, param.key, str(param.value))

    def set_tag(self, run_id: str, tag: RunTag) -> None:
        if self._delegate:
            self._delegate.set_tag(run_id, tag)
        run = self._runs.get(run_id)
        if run:
            updated_tags = dict(run.data.tags)
            updated_tags[tag.key] = tag.value
            self._runs[run_id] = Run(
                run_info=run.info,
                run_data=RunData(
                    metrics=run.data.metrics,
                    params=run.data.params,
                    tags=updated_tags,
                ),
            )
        self._onto.log_tag(run_id, tag.key, str(tag.value))

    def get_metric_history(self, run_id: str, metric_key: str, max_results=None, page_token=None):
        if self._delegate:
            return self._delegate.get_metric_history(run_id, metric_key, max_results, page_token)
        return PagedList([], None)

    def log_batch(self, run_id: str, metrics, params, tags) -> None:
        if self._delegate:
            self._delegate.log_batch(run_id, metrics, params, tags)
        for m in metrics or []:
            self.log_metric(run_id, m)
        for p in params or []:
            self.log_param(run_id, p)
        for t in tags or []:
            self.set_tag(run_id, t)

    # ------------------------------------------------------------------
    # AbstractStore — Search
    # ------------------------------------------------------------------

    def search_runs(
        self,
        experiment_ids,
        filter_string,
        run_view_type,
        max_results=1000,
        order_by=None,
        page_token=None,
    ) -> PagedList:
        if self._delegate:
            result = self._delegate.search_runs(
                experiment_ids, filter_string, run_view_type, max_results, order_by, page_token
            )
            for run in result:
                self._sync_run(run)
            return result

        runs = [
            r
            for r in self._runs.values()
            if r.info.experiment_id in [str(eid) for eid in experiment_ids]
        ]
        if run_view_type == ViewType.ACTIVE_ONLY:
            runs = [r for r in runs if r.info.lifecycle_stage == LifecycleStage.ACTIVE]
        elif run_view_type == ViewType.DELETED_ONLY:
            runs = [r for r in runs if r.info.lifecycle_stage == LifecycleStage.DELETED]
        return PagedList(runs[:max_results], None)

    # ------------------------------------------------------------------
    # Cognitive accessor
    # ------------------------------------------------------------------

    @property
    def atomspace(self) -> AtomSpace:
        """Direct access to the underlying AtomSpace for cognitive queries."""
        return self._as

    @property
    def translator(self) -> MLflowAtomTranslator:
        """Access the MLflow ↔ AtomSpace translator."""
        return self._translator
