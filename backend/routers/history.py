"""History and task status routes."""

from fastapi import APIRouter, Depends, HTTPException

from backend.auth import get_current_user
from backend.runtime import _task_status
from backend.storage.sessions import (
    STATUS_REVIEW_FAILED,
    STATUS_REVIEWED,
    STATUS_REVIEWING,
    delete_session,
    get_session,
    list_distinct_topics,
    list_sessions,
)

router = APIRouter(prefix="/api")


@router.get("/interview/review/{session_id}")
async def get_review(session_id: str, user_id: str = Depends(get_current_user)):
    """Get review for a completed session."""
    session = get_session(session_id, user_id=user_id)
    if not session:
        raise HTTPException(404, "Session not found.")
    if not session.get("review"):
        raise HTTPException(400, "Interview not yet reviewed.")
    return session


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str, user_id: str = Depends(get_current_user)):
    """Poll async task status."""
    task = _task_status.get(task_id)
    if task and task.get("user_id") in (None, user_id):
        public_task = {key: value for key, value in task.items() if key != "user_id"}
        return {"task_id": task_id, **public_task}

    session = get_session(task_id, user_id=user_id)
    if not session:
        raise HTTPException(404, "Task not found.")

    mode = session.get("mode")
    task_type = "resume_review" if mode == "resume" else "jd_review" if mode == "jd_prep" else "drill_review"
    status = session.get("status")
    if session.get("review") or status == STATUS_REVIEWED:
        return {"task_id": task_id, "status": "done", "type": task_type}
    if status == STATUS_REVIEWING:
        return {"task_id": task_id, "status": "pending", "type": task_type}
    if status == STATUS_REVIEW_FAILED:
        return {
            "task_id": task_id,
            "status": "error",
            "type": task_type,
            "error": session.get("review_error") or "复盘生成失败",
        }
    raise HTTPException(404, "Task not found.")


@router.get("/interview/history")
async def get_history(
    limit: int = 20,
    offset: int = 0,
    mode: str = None,
    topic: str = None,
    user_id: str = Depends(get_current_user),
):
    """List past interview sessions with filtering and pagination."""
    return list_sessions(user_id=user_id, limit=limit, offset=offset, mode=mode, topic=topic)


@router.delete("/interview/session/{session_id}")
async def delete_session_endpoint(session_id: str, user_id: str = Depends(get_current_user)):
    """Delete a session record."""
    deleted = delete_session(session_id, user_id=user_id)
    if not deleted:
        raise HTTPException(404, "Session not found.")
    return {"ok": True}


@router.get("/interview/topics")
async def get_interview_topics(user_id: str = Depends(get_current_user)):
    """List distinct topics from completed sessions (for filter dropdown)."""
    return list_distinct_topics(user_id=user_id)
