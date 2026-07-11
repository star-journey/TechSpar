"""Resume and speech-to-text routes."""

import asyncio
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.auth import get_current_user
from backend.config import settings
from backend.indexer import invalidate_resume
from backend.utils import safe_child_path

router = APIRouter(prefix="/api")
MAX_RESUME_BYTES = 20 * 1024 * 1024


@router.get("/resume/status")
def resume_status(user_id: str = Depends(get_current_user)):
    """Check if a resume file exists."""
    resume_dir = settings.user_resume_path(user_id)
    if not resume_dir.exists():
        return {"has_resume": False}
    files = [file for file in resume_dir.iterdir() if file.suffix.lower() == ".pdf"]
    if not files:
        return {"has_resume": False}
    resume_file = files[0]
    return {
        "has_resume": True,
        "filename": resume_file.name,
        "size": resume_file.stat().st_size,
    }


@router.post("/resume/upload")
async def upload_resume(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    """Upload a resume PDF. Replaces any existing resume."""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    resume_dir = settings.user_resume_path(user_id)
    resume_dir.mkdir(parents=True, exist_ok=True)
    try:
        dest = safe_child_path(resume_dir, filename)
    except ValueError:
        raise HTTPException(400, "Invalid resume filename.")

    content = await file.read(MAX_RESUME_BYTES + 1)
    if len(content) > MAX_RESUME_BYTES:
        raise HTTPException(413, "Resume PDF is too large (max 20 MB).")
    if b"%PDF-" not in content[:1024]:
        raise HTTPException(400, "Uploaded file is not a valid PDF.")

    temp = resume_dir / f".{uuid.uuid4().hex}.upload"
    try:
        temp.write_bytes(content)
        for old in resume_dir.iterdir():
            if old.is_file() and old.suffix.lower() == ".pdf":
                old.unlink()
        temp.replace(dest)
    finally:
        if temp.exists():
            temp.unlink()

    # Drop stale resume vectors; the next query_resume lazily re-ingests the new PDF.
    invalidate_resume(user_id)

    return {"ok": True, "filename": filename, "size": len(content)}


@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...), user_id: str = Depends(get_current_user)):
    """Transcribe short audio clip to text via DashScope ASR."""
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(400, "Empty audio file.")

    try:
        from backend.transcribe import transcribe_short

        suffix = "." + (file.filename or "audio.webm").rsplit(".", 1)[-1]
        text = await asyncio.to_thread(transcribe_short, audio_bytes, suffix=suffix)
        return {"text": text}
    except Exception as exc:
        raise HTTPException(500, f"Transcription failed: {exc}")
