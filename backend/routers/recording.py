"""Recording review routes."""

import asyncio
import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage

from backend.auth import get_current_user
from backend.memory import extract_behavior_ops, llm_update_profile
from backend.models import RecordingAnalyzeRequest
from backend.review_formatters import format_drill_review, format_solo_review
from backend.runtime import _task_status
from backend.storage.sessions import (
    STATUS_REVIEW_FAILED,
    STATUS_REVIEWING,
    create_session,
    save_drill_answers,
    save_review,
    update_session_status,
)

logger = logging.getLogger("uvicorn")
router = APIRouter(prefix="/api")


@router.post("/recording/transcribe")
async def recording_transcribe(
    file: UploadFile = File(...),
    mode: str = Form("dual"),
    user_id: str = Depends(get_current_user),
):
    """Transcribe recording audio via the configured STT provider."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file.")

    suffix = "." + (file.filename or "audio.webm").rsplit(".", 1)[-1]

    try:
        from backend.config import settings
        from backend.stt import get_provider

        text = get_provider(settings.stt_provider).transcribe(audio_bytes, suffix=suffix)
        return {"transcript": text, "segments": []}
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}")


def _analyze_recording_background(
    session_id: str,
    req_transcript: str,
    req_recording_mode: str,
    req_company: str | None,
    req_position: str | None,
    user_id: str,
):
    """Background task: analyze recording transcript."""
    try:
        from backend.graphs.topic_drill import _parse_json_response
        from backend.llm_provider import get_langchain_llm, invoke_with_retry
        from backend.memory import get_profile_summary
        from backend.prompts.recording import (
            RECORDING_DUAL_EVAL_PROMPT,
            RECORDING_SOLO_EVAL_PROMPT,
            RECORDING_STRUCTURE_PROMPT,
        )
        from backend.storage.user_settings import load_user_settings

        # Stream (like topic drill) so the gateway never hits its idle read-timeout on
        # long batch-JSON generations, and disable the SDK's own retries so the only
        # retry layer is invoke_with_retry (bounded) — otherwise a 524 fans out into
        # SDK-retries × invoke_with_retry attempts of 120s calls against the provider.
        recording_timeout = load_user_settings(user_id).recording_analysis_timeout_seconds
        llm = get_langchain_llm(user_id, max_retries=0, timeout=recording_timeout)
        profile_summary = get_profile_summary(user_id)

        if req_recording_mode == "dual":
            structure_prompt = RECORDING_STRUCTURE_PROMPT.format(transcript=req_transcript)
            response = invoke_with_retry(llm, [
                SystemMessage(content="你是面试记录分析引擎。只返回 JSON，不要其他内容。"),
                HumanMessage(content=structure_prompt),
            ])
            structured = _parse_json_response(response.content)
            qa_pairs = structured.get("qa_pairs", [])

            questions, answers = [], []
            for pair in qa_pairs:
                question_id = pair.get("id", len(questions) + 1)
                questions.append({
                    "id": question_id,
                    "question": pair["question"],
                    "difficulty": 3,
                    "focus_area": pair.get("focus_area", ""),
                })
                answers.append({"question_id": question_id, "answer": pair.get("answer", "")})

            qa_lines = [
                f"### Q{question['id']} ({question.get('focus_area', '')})\n**题目**: {question['question']}\n**回答**: {answer['answer']}"
                for question, answer in zip(questions, answers)
            ]
            eval_prompt = RECORDING_DUAL_EVAL_PROMPT.format(
                qa_pairs="\n\n".join(qa_lines),
                profile_summary=profile_summary,
            )
            eval_response = invoke_with_retry(llm, [
                SystemMessage(content="你是面试评估引擎。只返回 JSON，不要其他内容。"),
                HumanMessage(content=eval_prompt),
            ])
            eval_result = _parse_json_response(eval_response.content)
            scores = eval_result.get("scores", [])
            overall = eval_result.get("overall", {})
            for score in scores:
                score.setdefault("difficulty", 3)

            review = format_drill_review(questions, answers, scores, overall)
            save_drill_answers(session_id, answers, user_id=user_id)
            save_review(
                session_id,
                review,
                scores,
                overall.get("new_weak_points", []),
                overall,
                user_id=user_id,
            )
        else:
            eval_prompt = RECORDING_SOLO_EVAL_PROMPT.format(
                transcript=req_transcript,
                profile_summary=profile_summary,
            )
            response = invoke_with_retry(llm, [
                SystemMessage(content="你是录音评估引擎。只返回 JSON，不要其他内容。"),
                HumanMessage(content=eval_prompt),
            ])
            eval_result = _parse_json_response(response.content)
            topics_covered = eval_result.get("topics_covered", [])
            overall = eval_result.get("overall", {})
            overall["topics_covered"] = topics_covered
            scores = [
                {"question_id": topic.get("id", index + 1), "score": topic.get("score"), "difficulty": 3}
                for index, topic in enumerate(topics_covered)
            ]

            review = format_solo_review(topics_covered, overall)
            save_review(
                session_id,
                review,
                scores,
                overall.get("new_weak_points", []),
                overall,
                user_id=user_id,
            )

        try:
            asyncio.run(_update_recording_profile(
                overall, scores, max(len(scores), 1), user_id,
                transcript=req_transcript, session_id=session_id,
            ))
        except Exception:
            logger.exception("Post-review profile update failed for recording session %s", session_id)

        _task_status[session_id] = {"status": "done", "type": "recording", "user_id": user_id}
        logger.info("Recording analysis done for session %s", session_id)
    except Exception as exc:
        update_session_status(
            session_id, STATUS_REVIEW_FAILED,
            user_id=user_id, review_error=str(exc)[:500] or "未知错误",
        )
        _task_status[session_id] = {"status": "error", "type": "recording", "user_id": user_id}
        logger.error("Recording analysis failed for session %s: %s", session_id, exc)


@router.post("/recording/analyze")
async def recording_analyze(
    req: RecordingAnalyzeRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
):
    """Analyze a recording transcript — async background processing."""
    session_id = str(uuid.uuid4())
    # Persist the raw inputs so a failed analysis can be regenerated from history
    # (the retry path re-dispatches the background task from these).
    create_session(
        session_id,
        mode="recording",
        meta={
            "transcript": req.transcript,
            "recording_mode": req.recording_mode,
            "company": req.company,
            "position": req.position,
        },
        user_id=user_id,
    )
    update_session_status(session_id, STATUS_REVIEWING, user_id=user_id)

    _task_status[session_id] = {"status": "pending", "type": "recording", "user_id": user_id}
    background_tasks.add_task(
        _analyze_recording_background,
        session_id,
        req.transcript,
        req.recording_mode,
        req.company,
        req.position,
        user_id,
    )
    return {"session_id": session_id, "status": "pending"}


async def _update_recording_profile(overall: dict, scores: list, total_items: int, user_id: str,
                                    transcript: str = "", session_id: str | None = None):
    """Update profile from recording analysis — no single topic, points carry their own topic."""
    valid = []
    for score in scores:
        try:
            valid.append(float(score["score"]))
        except (TypeError, ValueError, KeyError):
            pass

    behavior_ops = await extract_behavior_ops(transcript, user_id, mode="recording", topic=None)

    await llm_update_profile(
        mode="recording",
        topic=None,
        new_weak_points=overall.get("new_weak_points", []),
        new_strong_points=overall.get("new_strong_points", []),
        topic_mastery={},
        communication=overall.get("communication_observations", {}),
        user_id=user_id,
        thinking_patterns=overall.get("thinking_patterns"),
        session_summary=overall.get("summary", ""),
        avg_score=overall.get("avg_score"),
        answer_count=len(valid),
        behavior_ops=behavior_ops,
        session_id=session_id,
    )
