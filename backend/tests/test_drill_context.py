import json

import backend.memory as memory
from backend.memory import DEFAULT_PROFILE, get_topic_context_for_drill
from backend.routers.interview import _collect_drill_weak_points
from backend.storage import sessions


def _use_temp_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(sessions, "DB_PATH", tmp_path / "interviews.db")
    monkeypatch.setattr(memory, "_profile_path", lambda user_id: tmp_path / "users" / user_id / "profile" / "profile.json")


def _save_profile(tmp_path, user_id, profile):
    profile_dir = tmp_path / "users" / user_id / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.json").write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")


def _reviewed_drill(session_id, topic, user_id, questions=None, scores=None, weak_points=None, overall=None, review=""):
    sessions.create_session(
        session_id,
        "topic_drill",
        topic,
        questions=questions or [],
        user_id=user_id,
    )
    sessions.save_review(
        session_id,
        review,
        scores=scores or [],
        weak_points=weak_points or [],
        overall=overall or {},
        user_id=user_id,
    )


def test_recent_drill_questions_read_from_reviewed_sessions(monkeypatch, tmp_path):
    _use_temp_storage(monkeypatch, tmp_path)
    _reviewed_drill(
        "s1",
        "python",
        "u1",
        questions=[{"id": 1, "question": "解释 GIL 的作用"}],
    )
    _reviewed_drill(
        "s2",
        "python",
        "u2",
        questions=[{"id": 1, "question": "其他用户的问题"}],
    )
    sessions.create_session(
        "s3",
        "topic_drill",
        "python",
        questions=[{"id": 1, "question": "未复盘的问题"}],
        user_id="u1",
    )

    assert sessions.list_recent_drill_questions("python", user_id="u1") == ["解释 GIL 的作用"]


def test_drill_context_uses_session_history_for_missing_prompt_fields(monkeypatch, tmp_path):
    _use_temp_storage(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.vector_memory.search_memory", lambda **_: [])
    _reviewed_drill(
        "s1",
        "python",
        "u1",
        questions=[{"id": 1, "question": "解释 GIL 的作用"}],
        scores=[{"question_id": 1, "score": 5, "weak_point": "对 GIL 调度机制理解不清"}],
        overall={"summary": "本次训练暴露了对线程调度的理解不足"},
        review="## 总结\n需要加强线程模型\n\n## 逐题复盘\n...",
    )

    context = get_topic_context_for_drill("python", "u1")

    assert "对 GIL 调度机制理解不清" in context["weak_points"]
    assert "解释 GIL 的作用" in context["recent_questions"]
    assert context["past_insights"] == ["本次训练暴露了对线程调度的理解不足"]


def test_drill_context_does_not_restore_inactive_profile_weak_point(monkeypatch, tmp_path):
    _use_temp_storage(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.vector_memory.search_memory", lambda **_: [])
    profile = DEFAULT_PROFILE.copy()
    profile["weak_points"] = [{"point": "Redis 过期策略不熟", "topic": "redis", "improved": True}]
    _save_profile(tmp_path, "u1", profile)
    _reviewed_drill(
        "s1",
        "redis",
        "u1",
        scores=[{"question_id": 1, "score": 5, "weak_point": "Redis 过期策略不熟"}],
    )

    context = get_topic_context_for_drill("redis", "u1")

    assert "Redis 过期策略不熟" not in context["weak_points"]


def test_collect_drill_weak_points_merges_overall_and_scores():
    result = _collect_drill_weak_points(
        "redis",
        {"new_weak_points": [{"point": "Redis 持久化不熟", "topic": "redis"}]},
        [
            {"weak_point": "Redis 持久化不熟"},
            {"weak_point": "Redis 过期删除策略不熟"},
            {"weak_point": ""},
        ],
    )

    assert result == [
        {"point": "Redis 持久化不熟", "topic": "redis"},
        {"point": "Redis 过期删除策略不熟", "topic": "redis", "axis": "knowledge"},
    ]


def test_past_insights_fallback_to_review_when_vector_fails(monkeypatch, tmp_path):
    _use_temp_storage(monkeypatch, tmp_path)

    def fail_search(**_):
        raise RuntimeError("vector unavailable")

    monkeypatch.setattr("backend.vector_memory.search_memory", fail_search)
    _reviewed_drill(
        "s1",
        "mysql",
        "u1",
        review="## 总结\n索引选择性理解不足\n\n## 逐题复盘\n...",
    )

    context = get_topic_context_for_drill("mysql", "u1")

    assert context["past_insights"] == ["## 总结\n索引选择性理解不足"]
