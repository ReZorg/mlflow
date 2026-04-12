"""Tests for AtomSpaceTrackingStore."""

import time

import pytest

from mlflow.cogmlflow.atomspace.tracking_store import AtomSpaceTrackingStore
from mlflow.cogmlflow.entities.atom import ConceptNode
from mlflow.cogmlflow.entities.ontology import experiment_node_name, run_node_name
from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.exceptions import MlflowException


@pytest.fixture
def store():
    return AtomSpaceTrackingStore()


class TestExperimentOps:
    def test_create_and_get_experiment(self, store):
        exp_id = store.create_experiment("test_exp", "", [])
        exp = store.get_experiment(exp_id)
        assert exp.name == "test_exp"
        assert exp.experiment_id == exp_id

    def test_duplicate_experiment_name_raises(self, store):
        store.create_experiment("dup", "", [])
        with pytest.raises(MlflowException):
            store.create_experiment("dup", "", [])

    def test_get_experiment_by_name(self, store):
        store.create_experiment("named_exp", "", [])
        exp = store.get_experiment_by_name("named_exp")
        assert exp is not None
        assert exp.name == "named_exp"

    def test_get_nonexistent_experiment_raises(self, store):
        with pytest.raises(MlflowException):
            store.get_experiment("9999")

    def test_delete_experiment(self, store):
        exp_id = store.create_experiment("to_delete", "", [])
        store.delete_experiment(exp_id)
        exp = store.get_experiment(exp_id)
        assert exp.lifecycle_stage == "deleted"

    def test_search_experiments_active_only(self, store):
        exp_id1 = store.create_experiment("active1", "", [])
        exp_id2 = store.create_experiment("active2", "", [])
        store.delete_experiment(exp_id1)
        results = store.search_experiments(view_type=ViewType.ACTIVE_ONLY)
        ids = [e.experiment_id for e in results]
        assert exp_id2 in ids
        assert exp_id1 not in ids

    def test_experiment_synced_to_atomspace(self, store):
        exp_id = store.create_experiment("atom_test", "", [])
        node = store.atomspace.get(ConceptNode(experiment_node_name(exp_id)))
        assert node is not None


class TestRunOps:
    def test_create_and_get_run(self, store):
        exp_id = store.create_experiment("run_exp", "", [])
        run = store.create_run(exp_id, "user1", int(time.time() * 1000), [], "run1")
        fetched = store.get_run(run.info.run_id)
        assert fetched.info.run_id == run.info.run_id
        assert fetched.info.experiment_id == exp_id

    def test_get_nonexistent_run_raises(self, store):
        with pytest.raises(MlflowException):
            store.get_run("nonexistent_run_id")

    def test_log_metric_stored_in_atomspace(self, store):
        exp_id = store.create_experiment("metric_exp", "", [])
        run = store.create_run(exp_id, "user", int(time.time() * 1000), [], "r")
        run_id = run.info.run_id
        store.log_metric(run_id, Metric("accuracy", 0.95, int(time.time() * 1000), 0))
        # Verify in run data
        fetched = store.get_run(run_id)
        assert fetched.data.metrics.get("accuracy") == pytest.approx(0.95)
        # Verify in AtomSpace
        metrics = store.translator._onto.get_run_metrics(run_id)
        keys = [k for k, _ in metrics]
        assert "accuracy" in keys

    def test_log_param_stored(self, store):
        exp_id = store.create_experiment("param_exp", "", [])
        run = store.create_run(exp_id, "user", int(time.time() * 1000), [], "r")
        run_id = run.info.run_id
        store.log_param(run_id, Param("lr", "0.01"))
        fetched = store.get_run(run_id)
        assert fetched.data.params.get("lr") == "0.01"

    def test_set_tag_stored(self, store):
        exp_id = store.create_experiment("tag_exp", "", [])
        run = store.create_run(exp_id, "user", int(time.time() * 1000), [], "r")
        run_id = run.info.run_id
        store.set_tag(run_id, RunTag("env", "test"))
        fetched = store.get_run(run_id)
        assert fetched.data.tags.get("env") == "test"

    def test_log_batch(self, store):
        exp_id = store.create_experiment("batch_exp", "", [])
        run = store.create_run(exp_id, "u", int(time.time() * 1000), [], "r")
        run_id = run.info.run_id
        now = int(time.time() * 1000)
        store.log_batch(
            run_id,
            metrics=[Metric("f1", 0.88, now, 0), Metric("loss", 0.12, now, 0)],
            params=[Param("bs", "32")],
            tags=[RunTag("source", "batch")],
        )
        fetched = store.get_run(run_id)
        assert fetched.data.metrics["f1"] == pytest.approx(0.88)
        assert fetched.data.params["bs"] == "32"
        assert fetched.data.tags["source"] == "batch"

    def test_search_runs(self, store):
        exp_id = store.create_experiment("search_exp", "", [])
        run1 = store.create_run(exp_id, "u", int(time.time() * 1000), [], "r1")
        run2 = store.create_run(exp_id, "u", int(time.time() * 1000), [], "r2")
        results = store.search_runs([exp_id], "", ViewType.ACTIVE_ONLY)
        ids = [r.info.run_id for r in results]
        assert run1.info.run_id in ids
        assert run2.info.run_id in ids

    def test_delete_and_restore_run(self, store):
        exp_id = store.create_experiment("del_run_exp", "", [])
        run = store.create_run(exp_id, "u", int(time.time() * 1000), [], "r")
        run_id = run.info.run_id
        store.delete_run(run_id)
        fetched = store.get_run(run_id)
        assert fetched.info.lifecycle_stage == "deleted"
        store.restore_run(run_id)
        fetched = store.get_run(run_id)
        assert fetched.info.lifecycle_stage == "active"

    def test_run_synced_to_atomspace(self, store):
        exp_id = store.create_experiment("sync_exp", "", [])
        run = store.create_run(exp_id, "u", int(time.time() * 1000), [], "r")
        run_id = run.info.run_id
        node = store.atomspace.get(ConceptNode(run_node_name(run_id)))
        assert node is not None
