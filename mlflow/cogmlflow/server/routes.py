"""
CogMLflow REST API route extensions.

Provides FastAPI router with ``/cogmlflow/`` endpoints that expose the
cognitive capabilities of CogMLflow over HTTP.  The router is designed to be
mounted onto the existing MLflow FastAPI application.

Endpoints:
    GET  /cogmlflow/health
    GET  /cogmlflow/recommendations/{experiment_id}
    POST /cogmlflow/atomspace/query
    GET  /cogmlflow/attention/ranking
    POST /cogmlflow/schedule
    GET  /cogmlflow/atomspace/stats
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from mlflow.cogmlflow.entities.atomspace import AtomSpace

_log = logging.getLogger(__name__)


if _FASTAPI_AVAILABLE:

    class AtomSpaceQueryRequest(BaseModel):
        """Request body for the AtomSpace pattern-match query endpoint."""

        atom_type: str
        name: str | None = None

    class ScheduleRequest(BaseModel):
        """Request body for the schedule endpoint."""

        candidate_id: str
        config: dict[str, Any]
        experiment_id: str = ""

    def build_router(
        atomspace: AtomSpace, suggestion_engine_factory=None, scheduler=None
    ) -> "APIRouter":
        """Build and return a FastAPI APIRouter for CogMLflow endpoints.

        Parameters
        ----------
        atomspace:
            The shared AtomSpace instance.
        suggestion_engine_factory:
            Optional callable returning a ``CognitiveSuggestionEngine``.
        scheduler:
            Optional ``CognitiveScheduler`` instance.
        """
        router = APIRouter(prefix="/cogmlflow", tags=["CogMLflow"])

        @router.get("/health")
        def health():
            return {"status": "ok", "atomspace_size": len(atomspace)}

        @router.get("/atomspace/stats")
        def atomspace_stats():
            type_counts: dict[str, int] = {}
            for atom in atomspace:
                type_counts[atom.type] = type_counts.get(atom.type, 0) + 1
            return {"total_atoms": len(atomspace), "by_type": type_counts}

        @router.get("/recommendations/{experiment_id}")
        def get_recommendations(experiment_id: str, metric: str = "accuracy"):
            from mlflow.cogmlflow.suggestion_engine import CognitiveSuggestionEngine

            engine = (
                suggestion_engine_factory(experiment_id)
                if suggestion_engine_factory
                else CognitiveSuggestionEngine(atomspace, primary_metric=metric)
            )
            suggestions = engine.run()
            return {
                "experiment_id": experiment_id,
                "suggestions": [s.to_dict() for s in suggestions],
            }

        @router.post("/atomspace/query")
        def atomspace_query(request: AtomSpaceQueryRequest):
            from mlflow.cogmlflow.entities.atom import ATOM_TYPE_REGISTRY

            atom_cls = ATOM_TYPE_REGISTRY.get(request.atom_type)
            if atom_cls is None:
                raise HTTPException(
                    status_code=400, detail=f"Unknown atom type: {request.atom_type!r}"
                )
            atoms = atomspace.get_atoms_by_type(request.atom_type)
            if request.name:
                atoms = [a for a in atoms if hasattr(a, "name") and a.name == request.name]
            return {
                "atom_type": request.atom_type,
                "count": len(atoms),
                "atoms": [repr(a) for a in atoms[:100]],
            }

        @router.get("/attention/ranking")
        def attention_ranking(top_k: int = 10):
            ranked = atomspace.get_by_sti(top_n=top_k)
            return {
                "top_k": top_k,
                "ranking": [{"atom": repr(a), "sti": a.sti, "lti": a.lti} for a in ranked],
            }

        @router.post("/schedule")
        def schedule_experiment(request: ScheduleRequest):
            if scheduler is None:
                raise HTTPException(status_code=503, detail="Scheduler not configured")
            scheduler.add_candidate(
                candidate_id=request.candidate_id,
                config=request.config,
                experiment_id=request.experiment_id,
            )
            next_candidate = scheduler.select_next()
            return {
                "registered": request.candidate_id,
                "next_to_run": next_candidate.candidate_id if next_candidate else None,
            }

        return router

else:

    def build_router(*args, **kwargs):
        _log.warning(
            "FastAPI is not installed; CogMLflow REST API routes are unavailable. "
            "Install with: pip install fastapi"
        )
        return None
