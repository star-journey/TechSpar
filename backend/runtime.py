"""Shared in-memory runtime state for API modules."""

import logging

from backend.models import InterviewMode
from backend.storage.sessions import get_session

logger = logging.getLogger("uvicorn")

# Hot caches for interactive sessions and async task status.
_graphs: dict[str, dict] = {}
_drill_sessions: dict[str, dict] = {}
_job_prep_sessions: dict[str, dict] = {}
_task_status: dict[str, dict] = {}
_copilot_sessions: dict[str, dict] = {}


async def get_or_restore_resume_graph(session_id: str, user_id: str) -> dict | None:
    """Return the cached graph entry, or rebuild it from the checkpoint store."""
    from backend.graphs.resume_interview import compile_resume_interview

    entry = _graphs.get(session_id)
    if entry is not None:
        return entry if entry.get("user_id") == user_id else None

    session = get_session(session_id, user_id=user_id)
    if not session or session.get("mode") != InterviewMode.RESUME.value:
        return None

    graph = compile_resume_interview(user_id)
    config = {"configurable": {"thread_id": session_id}}
    state = await graph.aget_state(config)
    if not state.values:
        return None

    entry = {
        "graph": graph,
        "config": config,
        "mode": InterviewMode.RESUME,
        "topic": session.get("topic"),
        "user_id": user_id,
    }
    _graphs[session_id] = entry
    return entry
