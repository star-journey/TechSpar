"""Shared in-memory runtime state for API modules."""

# Hot caches for interactive sessions and async task status.
_drill_sessions: dict[str, dict] = {}
_job_prep_sessions: dict[str, dict] = {}
# Async JD-prep jobs (analyze / start). The LLM call routinely outlives upstream
# proxy read timeouts (host nginx / Cloudflare), so those endpoints kick off a
# background task keyed here and the client polls instead of holding the request.
_job_prep_jobs: dict[str, dict] = {}
_task_status: dict[str, dict] = {}
_copilot_sessions: dict[str, dict] = {}
