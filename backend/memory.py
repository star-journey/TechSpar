"""个性化记忆系统 — 跨面试用户画像。

设计哲学：
- 文件即真相（OpenClaw）：profile.json 可人工编辑
- 两阶段提取（Mem0）：Extract → Update，不无脑追加
- 向量召回（embedding）：语义搜索历史洞察
"""
import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from backend.config import settings
from backend.llm_provider import get_langchain_llm

logger = logging.getLogger("uvicorn")

# Strip "(领域：xxx)" suffix that LLM sometimes copies from format hints
_TOPIC_SUFFIX_RE = re.compile(r'\s*[（(]领域[：:]\s*[^）)]+[）)]\s*$')

# 表现轴的四个固定 namespace。这一层是认知架构分类(怎么表达/怎么想/怎么叙事/对自己怎么评),
# 是封闭集合,LLM 自由度花在 namespace 之下的 behavior_id 涌现,不放在创造新 namespace 上。
BEHAVIOR_NAMESPACES = {"reasoning", "narrative", "communication", "metacognition"}

# behavior_signal ID 格式: <namespace>.<snake_case_name>
_BEHAVIOR_ID_RE = re.compile(r'^([a-z_]+)\.([a-z][a-z0-9_]*)$')


def _clean_point_text(text: str | None) -> str:
    if text is None:
        return ""
    return _TOPIC_SUFFIX_RE.sub('', str(text)).strip()


def _get_canonical_topic_keys(user_id: str) -> set[str]:
    from backend.indexer import load_topics
    return set(load_topics(user_id).keys())


def _normalize_extraction_topics(extraction: dict, canonical: set, fallback_topic: str):
    """Normalize topic for knowledge-axis weak/strong points.

    weak_points 和 strong_points 现在只承载知识轴。topic 必须在 canonical 集合内,
    否则 fallback 到当前面试的 topic。表现轴观察走 behavior_signals,不进这两个数组。
    """
    for item in extraction.get("weak_points", []) + extraction.get("strong_points", []):
        if not isinstance(item, dict):
            continue
        item["point"] = _clean_point_text(item.get("point", ""))
        item.pop("axis", None)  # 旧字段,新数据不带
        topic = item.get("topic", "")
        if topic not in canonical:
            item["topic"] = fallback_topic


# Per-user locks to prevent concurrent read-modify-write on profile.json
_profile_locks: dict[str, asyncio.Lock] = {}


def _get_profile_lock(user_id: str) -> asyncio.Lock:
    if user_id not in _profile_locks:
        _profile_locks[user_id] = asyncio.Lock()
    return _profile_locks[user_id]

# ── Profile Schema ──

DEFAULT_PROFILE = {
    "name": "",
    "target_role": "",
    "updated_at": "",

    # 上次 consolidation 运行时间 (用于节流,避免每次 session 都跑 Stage 3)
    "last_consolidation_at": "",

    # 技术掌握度 (topic → {level: 1-5, notes: str})
    "topic_mastery": {},

    # 知识轴薄弱点 (list of {point, topic, first_seen, last_seen, times_seen, improved})
    "weak_points": [],

    # 知识轴强项 (list of {point, topic, first_seen})
    "strong_points": [],

    # 表现轴: behavior_signals.
    # key 是 emergent ID (格式 <namespace>.<snake_case>),value 是该模式的累积证据.
    # 与 weak_points / strong_points 物理分离,不嵌套. polarity 决定它是负向还是正向.
    # 示例: "reasoning.jump_to_conclusion": {
    #     "namespace": "reasoning",
    #     "polarity": "negative",
    #     "description": "被追问 why 时跳过推导直接给结论",
    #     "first_seen": "...", "last_seen": "...", "times_seen": 3,
    #     "improved": false,
    #     "examples": [{"session_id": "...", "date": "...", "snippet": "..."}]
    # }
    "behavior_signals": {},

    # 表达与沟通特征
    "communication": {
        "style": "",        # e.g. "回答偏短，缺少具体例子"
        "habits": [],       # e.g. ["紧张时语速加快", "喜欢用类比解释"]
        "suggestions": [],  # e.g. ["多用 STAR 法描述项目"]
    },

    # 答题思维模式
    "thinking_patterns": {
        "strengths": [],    # e.g. ["能用类比解释抽象概念", "项目描述有数据支撑"]
        "gaps": [],         # e.g. ["对比类问题缺乏结构", "被追问 why 时容易卡住"]
    },

    # 面试统计
    "stats": {
        "total_sessions": 0,
        "resume_sessions": 0,
        "drill_sessions": 0,
        "job_prep_sessions": 0,
        "avg_score": 0,
        "score_history": [],  # [{date, mode, topic, avg_score}]
    },
}

EXTRACT_PROMPT = """你是一个面试教练的分析引擎。根据面试对话记录，提取关于候选人的结构化洞察。

## 候选人当前画像
{current_profile}

## 候选人已有的 behavior_signals（优先复用这些 ID，不要起新名字除非真的不同）
{existing_behavior_signals}

## 本次面试记录
模式: {mode}
领域: {topic}
{transcript}

## 评分记录（如有）
{scores}

## 合法领域列表
{allowed_topics}

## 画像的两条独立轴（物理分离，不嵌套）

### 知识轴 → weak_points / strong_points
针对具体技术领域的知识掌握情况。每条带 topic，topic 必须从「合法领域列表」选。
观察的是"懂不懂、会不会"，**不**涉及"怎么表达、怎么思考"。
不属于具体领域时，使用本次面试的领域 "{topic}"。

### 表现轴 → behavior_signals（一组 op）
独立于知识轴，描述候选人作为面试者的行为模式。
四个 namespace（**锁定，不可创新**）：
- reasoning：推导/思维方式（被追问 why 时如何应对、能否从底层推导、是否跳步）
- narrative：项目叙事（讲项目的结构、量化指标、技术权衡是否讲清）
- communication：表达特征（节奏、结构信号、清晰度、口头禅）
- metacognition：元认知（自我评估准确性、对自己弱点的觉察、不懂装懂）

每个 behavior_signal 是一个操作 op：
- **ADD**：全新模式。创建新 ID，格式严格为 `<namespace>.<snake_case_name>`。必须给 polarity（negative|positive）+ description（一句话锚定语义，后续不可覆盖）+ snippet（本次具体证据）
- **UPDATE**：复用上面"已有 behavior_signals"中的 ID。只给 snippet（本次新证据）
- **IMPROVE**：已有 negative 模式在本次出现了反向证据。给 evidence_snippet（说明为什么这次是反例）
- **NOOP**：不输出

ID 复用优先级最高：能用已有 ID 就**绝对不要**起新 ID。新 ID 只在所有现有 ID 都无法覆盖时才创建。
namespace 必须在四个里选，不要造新 namespace。
宁可不输出，不要凑数。

## 任务
分析这次面试，返回 JSON：

```json
{{
    "weak_points": [
        {{"point": "对 Python GIL 的理解停留在表面", "topic": "python"}}
    ],
    "strong_points": [
        {{"point": "RAG 架构描述清晰，有实战数据支撑", "topic": "rag"}}
    ],
    "behavior_signals": [
        {{
            "action": "ADD",
            "id": "reasoning.jump_to_conclusion",
            "namespace": "reasoning",
            "polarity": "negative",
            "description": "被追问 why 时跳过推导过程，直接给结论",
            "snippet": "讲为什么用 RAG 而非微调时直接说'更省钱'就停了"
        }},
        {{
            "action": "UPDATE",
            "id": "narrative.lack_metrics",
            "snippet": "讲 RAG 项目时没有任何数字指标"
        }},
        {{
            "action": "IMPROVE",
            "id": "communication.overlong_answer",
            "evidence_snippet": "本次答题平均不超过 90 秒，比之前简洁"
        }}
    ],
    "topic_mastery": {{
        "python": {{"notes": "基础扎实但高级特性（元类、描述符）薄弱"}}
    }},
    "communication_observations": {{
        "style_update": "回答技术题时逻辑清晰，但项目描述缺少量化数据",
        "new_habits": ["遇到不会的题会坦诚说不确定"],
        "new_suggestions": ["项目经历多用数据指标量化成果"]
    }},
    "thinking_patterns": {{
        "new_strengths": ["能用类比解释复杂概念"],
        "new_gaps": ["对比类问题回答缺乏结构"]
    }},
    "session_summary": "本次 Python 专项训练，基础题表现好，但 GIL 和 GC 机制理解不够深入",
    "dimension_scores": {{
        "technical_depth": 6,
        "project_articulation": 7,
        "communication": 5,
        "problem_solving": 6
    }},
    "avg_score": 6.0
}}
```

## dimension_scores 评分说明（仅简历面试模式需要填写，专项训练留空即可）
- technical_depth (1-10): 技术理解的深度，是真懂还是在背？
- project_articulation (1-10): 项目描述能力——设计思路、量化成果、技术权衡是否讲清楚
- communication (1-10): 表达的清晰度、结构化程度、简洁性
- problem_solving (1-10): 被追问时的分析推理能力
- avg_score = 四个维度的均值，保留一位小数

规则：
- 只提取本次面试中明确暴露的信息，不要猜测
- 知识类观察只放 weak_points / strong_points，表达/思维/叙事/元认知只放 behavior_signals，**不要混**
- weak_points / strong_points 的 topic 必须在「合法领域列表」内，禁止自创领域
- topic_mastery 只需提供 notes，score 由算法计算
- 专项训练模式下 dimension_scores 可省略，只需给 avg_score
"""


# ── Per-user path helpers ──

def _profile_path(user_id: str) -> Path:
    return settings.user_profile_dir(user_id) / "profile.json"


def _insights_dir(user_id: str) -> Path:
    return settings.user_profile_dir(user_id) / "insights"


def _load_profile(user_id: str) -> dict:
    path = _profile_path(user_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return DEFAULT_PROFILE.copy()


def _save_profile(profile: dict, user_id: str):
    path = _profile_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile["updated_at"] = datetime.now().isoformat()
    path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_insight(mode: str, topic: str, summary: str, raw_extraction: dict, user_id: str):
    """Append daily insight file (OpenClaw-style daily log)."""
    ins_dir = _insights_dir(user_id)
    ins_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    path = ins_dir / f"{today}.md"

    time_str = datetime.now().strftime("%H:%M")
    entry = f"\n## {time_str} | {mode} | {topic or '综合'}\n\n{summary}\n"

    if raw_extraction.get("weak_points"):
        entry += "\n**薄弱点:**\n"
        for wp in raw_extraction["weak_points"]:
            entry += f"- {wp['point']} ({wp.get('topic', '')})\n"

    if raw_extraction.get("strong_points"):
        entry += "\n**亮点:**\n"
        for sp in raw_extraction["strong_points"]:
            entry += f"- {sp['point']} ({sp.get('topic', '')})\n"

    entry += "\n---\n"

    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def get_profile(user_id: str) -> dict:
    return _load_profile(user_id)


async def update_target_role(user_id: str, target_role: str) -> None:
    """Persist target_role as the sticky default for future sessions."""
    target_role = (target_role or "").strip()
    if not target_role:
        return
    async with _get_profile_lock(user_id):
        profile = _load_profile(user_id)
        if profile.get("target_role") == target_role:
            return
        profile["target_role"] = target_role
        _save_profile(profile, user_id)


def get_topic_context_for_drill(topic: str, user_id: str) -> dict:
    """Get personalized context for drill question generation."""
    from backend.storage.sessions import list_recent_drill_questions, list_sessions_by_topic

    profile = _load_profile(user_id)

    mastery = profile.get("topic_mastery", {}).get(topic, {})
    mastery_score = mastery.get("score", mastery.get("level", 0) * 20)
    mastery_notes = mastery.get("notes", "新领域，暂无历史数据" if mastery_score == 0 else "")
    mastery_info = f"{mastery_score}/100 — {mastery_notes}"

    profile_weak_points = profile.get("weak_points", [])
    inactive_points = {
        _clean_point_text(w.get("point", ""))
        for w in profile_weak_points
        if w.get("topic") == topic and (w.get("improved") or w.get("archived"))
    }
    seen_weak = set()
    topic_weak = []
    for item in profile_weak_points:
        point = _clean_point_text(item.get("point", ""))
        if (
            item.get("topic") == topic
            and point
            and not item.get("improved")
            and not item.get("archived")
            and item.get("axis") != "performance"
        ):
            topic_weak.append(point)
            seen_weak.add(point)

    sessions = list_sessions_by_topic(topic, user_id=user_id, limit=50)
    for session in reversed(sessions):
        candidates = []
        candidates.extend(session.get("weak_points", []))
        candidates.extend(session.get("overall", {}).get("new_weak_points", []))
        candidates.extend({"point": score.get("weak_point"), "topic": topic} for score in session.get("scores", []))
        for item in candidates:
            if isinstance(item, dict):
                point = _clean_point_text(item.get("point", ""))
                item_topic = item.get("topic") or topic
                axis = item.get("axis")
            else:
                point = _clean_point_text(str(item))
                item_topic = topic
                axis = None
            if (
                item_topic == topic
                and point
                and axis != "performance"
                and point not in seen_weak
                and point not in inactive_points
            ):
                topic_weak.append(point)
                seen_weak.add(point)
            if len(topic_weak) >= 15:
                break
        if len(topic_weak) >= 15:
            break

    recent_questions = list_recent_drill_questions(topic, user_id=user_id, limit=20)
    if not recent_questions:
        recent_questions = [
            h.get("question", "")
            for h in profile.get("stats", {}).get("score_history", [])
            if h.get("topic") == topic and h.get("question")
        ][-20:]

    past_insights = []
    try:
        from backend.vector_memory import search_memory
        results = search_memory(
            query=f"{topic} 面试薄弱点 常见错误",
            chunk_types=["session_summary", "insight"],
            topic=topic,
            user_id=user_id,
            top_k=3,
        )
        past_insights = [r["content"] for r in results if r["score"] > 0.3]
    except Exception as exc:
        logger.debug("Vector memory unavailable for drill context: %s", exc)

    if not past_insights:
        for session in reversed(sessions):
            summary = (session.get("overall", {}).get("summary") or "").strip()
            if not summary:
                summary = (session.get("review") or "").split("## 逐题复盘")[0].strip()
            if summary:
                past_insights.append(summary[:500])
            if len(past_insights) >= 3:
                break

    return {
        "mastery_info": mastery_info,
        "mastery_score": mastery_score,
        "weak_points": topic_weak[:15],
        "recent_questions": recent_questions,
        "past_insights": past_insights,
    }


async def update_profile_realtime(
    mode: str,
    topic: str | None,
    user_id: str,
    score_entry: dict | None = None,
    weak_point: str | None = None,
):
    """Lightweight per-answer profile update — no LLM call, just save the data."""
    async with _get_profile_lock(user_id):
        profile = _load_profile(user_id)
        now = datetime.now().isoformat()

        # Record score
        if score_entry and score_entry.get("score") is not None:
            history = profile.setdefault("stats", {}).setdefault("score_history", [])
            history.append({
                "date": now[:10],
                "mode": mode,
                "topic": topic,
                "avg_score": score_entry["score"],
                "question": score_entry.get("question", ""),
                "assessment": score_entry.get("assessment", ""),
            })
            # Rolling average
            recent = [h["avg_score"] for h in history[-30:] if h.get("avg_score")]
            if recent:
                profile["stats"]["avg_score"] = round(sum(recent) / len(recent), 1)

        # Record weak point (semantic matching)
        if weak_point:
            from backend.vector_memory import find_similar_weak_point
            match_idx = find_similar_weak_point(weak_point, profile.get("weak_points", []), user_id=user_id)
            if match_idx is not None:
                matched = profile["weak_points"][match_idx]
                matched["times_seen"] = matched.get("times_seen", 1) + 1
                matched["last_seen"] = now
                if matched.get("archived"):
                    matched["archived"] = False
                    matched.pop("archived_at", None)
                    matched.setdefault("history", []).append({"date": now, "event": "unarchived"})
            else:
                profile.setdefault("weak_points", []).append({
                    "point": weak_point,
                    "topic": topic or "",
                    "source": "observed",
                    "first_seen": now,
                    "last_seen": now,
                    "times_seen": 1,
                    "improved": False,
                })

        # Track that we have activity (for profile page display)
        profile.setdefault("stats", {}).setdefault("total_answers", 0)
        profile["stats"]["total_answers"] = profile["stats"].get("total_answers", 0) + 1

        _save_profile(profile, user_id)


def _active_knowledge_weak_points(profile: dict) -> list[dict]:
    """Knowledge-axis weak points only. Filters out improved, archived, and legacy axis=performance."""
    return [
        w for w in profile.get("weak_points", [])
        if not w.get("improved")
        and not w.get("archived")
        and w.get("axis") != "performance"  # 老数据可能带 axis=performance,排除
    ]


def _top_behavior_signals(profile: dict, polarity: str | None = None, limit: int = 6) -> list[tuple[str, dict]]:
    """Top behavior_signals sorted by recency × times_seen.

    polarity=None returns all (active negatives + improved positives).
    polarity="negative" returns active negative signals only.
    """
    signals = profile.get("behavior_signals", {}) or {}
    items = []
    for sid, data in signals.items():
        if data.get("improved"):
            continue  # 改善了暂不进 summary
        if polarity and data.get("polarity", "negative") != polarity:
            continue
        items.append((sid, data))

    def _sort_key(pair):
        _, data = pair
        return (data.get("times_seen", 0), data.get("last_seen", ""))

    items.sort(key=_sort_key, reverse=True)
    return items[:limit]


def get_profile_summary(user_id: str) -> str:
    """Generate a concise summary for injection into interviewer prompts."""
    profile = _load_profile(user_id)

    parts = []
    active_weak = _active_knowledge_weak_points(profile)
    if active_weak:
        observed = [_clean_point_text(w.get("point")) for w in active_weak if w.get("source", "observed") == "observed"][:6]
        predicted = [_clean_point_text(w.get("point")) for w in active_weak if w.get("source") == "predicted"][:4]
        observed = [p for p in observed if p]
        predicted = [p for p in predicted if p]
        if observed:
            parts.append(f"已知知识薄弱点（训练中暴露）: {', '.join(observed)}")
        if predicted:
            parts.append(f"潜在知识薄弱点（JD分析预测）: {', '.join(predicted)}")

    if profile.get("strong_points"):
        points = ", ".join(
            p for p in (_clean_point_text(s.get("point")) for s in profile["strong_points"][:5]) if p
        )
        if points:
            parts.append(f"知识强项: {points}")

    top_behaviors = _top_behavior_signals(profile, polarity="negative", limit=6)
    if top_behaviors:
        lines = [
            f"{sid} (出现 {data.get('times_seen', 1)} 次): {(data.get('description') or '').strip()}"
            for sid, data in top_behaviors
        ]
        parts.append("行为模式短板:\n  - " + "\n  - ".join(lines))

    if profile.get("communication", {}).get("style"):
        parts.append(f"沟通风格: {profile['communication']['style']}")

    tp = profile.get("thinking_patterns", {})
    if tp.get("gaps"):
        parts.append(f"思维短板: {', '.join(tp['gaps'][:5])}")
    if tp.get("strengths"):
        parts.append(f"思维优势: {', '.join(tp['strengths'][:5])}")

    if profile.get("stats", {}).get("total_sessions"):
        stats = profile["stats"]
        parts.append(f"已完成 {stats['total_sessions']} 次模拟面试")

    if profile.get("topic_mastery"):
        mastery = ", ".join(
            f"{t}: {v.get('score', v.get('level', 0) * 20)}/100"
            for t, v in profile["topic_mastery"].items()
        )
        parts.append(f"掌握度: {mastery}")

    return "\n".join(parts) if parts else "新用户，暂无历史数据"


def get_profile_summary_for_drill(user_id: str) -> str:
    """Concise summary for drill question generation — only cross-topic info."""
    profile = _load_profile(user_id)
    parts = []

    # behavior_signals 是天然跨 topic 的,直接注入 top N
    top_behaviors = _top_behavior_signals(profile, polarity="negative", limit=3)
    if top_behaviors:
        lines = [
            f"{sid}: {(data.get('description') or '').strip()}"
            for sid, data in top_behaviors
        ]
        parts.append("反复出现的行为模式短板:\n  - " + "\n  - ".join(lines))

    if profile.get("communication", {}).get("style"):
        parts.append(f"沟通风格: {profile['communication']['style']}")

    tp = profile.get("thinking_patterns", {})
    if tp.get("gaps"):
        parts.append(f"思维短板: {', '.join(tp['gaps'][:5])}")
    if tp.get("strengths"):
        parts.append(f"思维优势: {', '.join(tp['strengths'][:5])}")

    if profile.get("stats", {}).get("total_sessions"):
        parts.append(f"已完成 {profile['stats']['total_sessions']} 次模拟面试")

    return "\n".join(parts) if parts else "新用户，暂无历史数据"


# ── Mem0-style LLM profile update ──

from backend.utils import parse_json_response as _parse_json_safe  # noqa: E402


def _apply_behavior_ops(profile: dict, ops: list, session_id: str | None, now: str) -> dict:
    """Apply mem0-style ops to behavior_signals dict.

    Supported actions (Stage 2 only, no MERGE here):
    - ADD: create new entry. Requires id / namespace / polarity / description.
           If id already exists, fall through to UPDATE.
    - UPDATE: bump times_seen, append example, refresh last_seen.
              If the signal was marked improved, flip it back and record regression.
    - IMPROVE: mark existing negative signal as improved with evidence.
    - NOOP / unknown / missing existing: silently skipped.

    Validation:
    - id must match <namespace>.<snake_case>
    - namespace must be in BEHAVIOR_NAMESPACES
    - Invalid ops are logged and dropped (no silent default routing)

    Returns a tally dict for logging (added / updated / improved / rejected).
    """
    tally = {"added": 0, "updated": 0, "improved": 0, "rejected": 0, "noop": 0}
    if not ops:
        return tally

    signals = profile.setdefault("behavior_signals", {})

    for op in ops:
        if not isinstance(op, dict):
            tally["rejected"] += 1
            continue

        action = (op.get("action") or "").upper()
        if action == "NOOP":
            tally["noop"] += 1
            continue

        signal_id = (op.get("id") or "").strip()
        m = _BEHAVIOR_ID_RE.match(signal_id)
        if not m:
            logger.warning(f"behavior op rejected: bad id {signal_id!r}")
            tally["rejected"] += 1
            continue

        namespace = m.group(1)
        if namespace not in BEHAVIOR_NAMESPACES:
            logger.warning(
                f"behavior op rejected: namespace {namespace!r} not in {BEHAVIOR_NAMESPACES}"
            )
            tally["rejected"] += 1
            continue

        existing = signals.get(signal_id)

        if action == "ADD" and existing is None:
            polarity = op.get("polarity", "negative")
            if polarity not in ("negative", "positive"):
                polarity = "negative"
            entry = {
                "namespace": namespace,
                "polarity": polarity,
                "description": (op.get("description") or "").strip(),
                "first_seen": now,
                "last_seen": now,
                "times_seen": 1,
                "improved": False,
                "examples": [],
            }
            snippet = (op.get("snippet") or "").strip()
            if snippet:
                entry["examples"].append({
                    "session_id": session_id,
                    "date": now,
                    "snippet": snippet,
                })
            signals[signal_id] = entry
            tally["added"] += 1

        elif action in ("ADD", "UPDATE") and existing is not None:
            # ADD on existing id is degraded to UPDATE
            existing["times_seen"] = existing.get("times_seen", 0) + 1
            existing["last_seen"] = now
            if existing.get("improved"):
                existing["improved"] = False
                existing.setdefault("history", []).append({
                    "date": now,
                    "event": "regressed",
                })
            snippet = (op.get("snippet") or "").strip()
            if snippet:
                examples = existing.setdefault("examples", [])
                examples.append({
                    "session_id": session_id,
                    "date": now,
                    "snippet": snippet,
                })
                if len(examples) > 5:
                    existing["examples"] = examples[-5:]
            tally["updated"] += 1

        elif action == "IMPROVE" and existing is not None:
            existing["improved"] = True
            existing["improved_at"] = now
            existing.setdefault("history", []).append({
                "date": now,
                "event": "improved",
                "evidence": (op.get("evidence_snippet") or "").strip(),
            })
            tally["improved"] += 1

        else:
            # UPDATE/IMPROVE on missing id, or unknown action
            tally["rejected"] += 1

    return tally


def _apply_memory_ops(profile: dict, ops: dict, topic: str | None, now: str, user_id: str = "",
                      new_weak_points: list | None = None, new_strong_points: list | None = None):
    """Execute LLM-decided ADD/UPDATE/NOOP/IMPROVE operations on profile.

    Topic for ADD ops comes from Stage 1 extraction (new_weak_points/new_strong_points),
    not from Stage 2 LLM output, to prevent topic hallucination.
    """
    from backend.vector_memory import upsert_weak_point_vector

    weak_points = profile.setdefault("weak_points", [])

    for i, op in enumerate(ops.get("weak_point_ops", [])):
        action = op.get("action", "NOOP")
        if action == "ADD":
            # Prefer topic from Stage 1 extraction (already normalized)
            add_topic = topic or ""
            if new_weak_points and i < len(new_weak_points):
                nwp = new_weak_points[i]
                add_topic = (nwp.get("topic", topic) if isinstance(nwp, dict) else topic) or ""
            weak_points.append({
                "point": _clean_point_text(op["point"]),
                "topic": add_topic,
                "source": op.get("source", "observed"),
                "first_seen": now, "last_seen": now,
                "times_seen": 1, "improved": False,
            })
        elif action == "UPDATE":
            idx = op.get("index")
            if idx is not None and 0 <= idx < len(weak_points):
                wp = weak_points[idx]
                new_text = _clean_point_text(op.get("new_point", ""))
                if new_text and new_text != wp.get("point"):
                    old_text = wp["point"]
                    history = wp.setdefault("history", [])
                    history.append({"point": old_text, "date": wp.get("last_seen", now)})
                    wp["point"] = new_text
                    if user_id:
                        try:
                            upsert_weak_point_vector(old_text, new_text, wp.get("topic", topic), user_id)
                        except Exception as e:
                            logger.warning(f"Failed to sync vector for updated weak point: {e}")
                wp["times_seen"] = wp.get("times_seen", 1) + 1
                wp["last_seen"] = now
                if wp.get("archived"):
                    wp["archived"] = False
                    wp.pop("archived_at", None)
                    wp.setdefault("history", []).append({"date": now, "event": "unarchived"})

    for imp in ops.get("improvements", []):
        idx = imp.get("weak_index")
        if idx is not None and 0 <= idx < len(weak_points):
            wp = weak_points[idx]
            history = wp.setdefault("history", [])
            history.append({"point": wp["point"], "date": now, "event": "improved"})
            wp["improved"] = True
            wp["improved_at"] = now

    existing_strong = {s["point"] for s in profile.get("strong_points", [])}
    for i, op in enumerate(ops.get("strong_point_ops", [])):
        if op.get("action") == "ADD" and op.get("point") and op["point"] not in existing_strong:
            add_topic = topic or ""
            if new_strong_points and i < len(new_strong_points):
                nsp = new_strong_points[i]
                add_topic = (nsp.get("topic", topic) if isinstance(nsp, dict) else topic) or ""
            profile.setdefault("strong_points", []).append({
                "point": _clean_point_text(op["point"]),
                "topic": add_topic,
                "first_seen": now,
            })


def _deterministic_update(profile: dict, new_weak: list, new_strong: list,
                          topic: str | None, now: str, user_id: str):
    """Fallback: vector cosine dedup when LLM parse fails."""
    from backend.vector_memory import find_similar_weak_point

    for wp in new_weak:
        point = _clean_point_text(wp.get("point", wp) if isinstance(wp, dict) else str(wp))
        match_idx = find_similar_weak_point(point, profile.get("weak_points", []), user_id=user_id)
        if match_idx is not None:
            matched = profile["weak_points"][match_idx]
            matched["times_seen"] = matched.get("times_seen", 1) + 1
            matched["last_seen"] = now
            if matched.get("archived"):
                matched["archived"] = False
                matched.pop("archived_at", None)
                matched.setdefault("history", []).append({"date": now, "event": "unarchived"})
        else:
            profile.setdefault("weak_points", []).append({
                "point": point,
                "topic": wp.get("topic", topic) if isinstance(wp, dict) else (topic or ""),
                "source": wp.get("source", "observed") if isinstance(wp, dict) else "observed",
                "first_seen": now, "last_seen": now,
                "times_seen": 1, "improved": False,
            })

    for sp in new_strong:
        sp_text = sp.get("point", sp) if isinstance(sp, dict) else str(sp)
        sp_topic = sp.get("topic") if isinstance(sp, dict) else topic
        # Use embedding similarity to find the weak point this strong point overcomes
        active_weak = [
            (i, w) for i, w in enumerate(profile.get("weak_points", []))
            if w.get("topic") == sp_topic and not w.get("improved") and not w.get("archived")
        ]
        if active_weak:
            from backend.vector_memory import _embed, _cosine_similarity
            sp_vec = _embed(sp_text)
            weak_texts = [w["point"] for _, w in active_weak]
            weak_vecs = np.stack([_embed(t) for t in weak_texts])
            sims = _cosine_similarity(sp_vec, weak_vecs)
            best_local = int(np.argmax(sims))
            if float(sims[best_local]) >= 0.5:
                _, matched_wp = active_weak[best_local]
                matched_wp["improved"] = True
                matched_wp["improved_at"] = now

        existing = {s["point"] for s in profile.get("strong_points", [])}
        if sp_text not in existing:
            profile.setdefault("strong_points", []).append({
                "point": sp_text,
                "topic": sp_topic or "",
                "first_seen": now,
            })


def _update_mastery(profile: dict, topic: str | None, mastery_data: dict, now: str,
                    min_weight: float = 0.15, user_id: str | None = None):
    """Update topic mastery (0-100 scale). Weight decreases with session count."""
    if not mastery_data:
        return
    # {score, notes} → single topic; {topic_key: {score, notes}} → multi-topic
    if "score" in mastery_data or "level" in mastery_data:
        if not topic:
            return
        entries = {topic: mastery_data}
    else:
        entries = mastery_data

    # Only allow canonical topics from topics.json
    if user_id:
        from backend.indexer import load_topics
        canonical = set(load_topics(user_id).keys())
        if canonical:
            entries = {t: d for t, d in entries.items() if t in canonical}

    for t, data in entries.items():
        if not isinstance(data, dict):
            continue
        existing = profile.setdefault("topic_mastery", {}).setdefault(t, {})
        new_score = data.get("score")
        if new_score is not None:
            old_score = existing.get("score", existing.get("level", 0) * 20)
            n = existing.get("session_count", 0)
            coverage = data.get("coverage", 1.0)
            # Dynamic weight: fast convergence early, stable later
            # Scale down by coverage so partial sessions have less impact
            weight = max(min_weight, 1.0 / (n + 1)) * coverage
            merged = round(old_score * (1 - weight) + new_score * weight, 1)
            existing["score"] = merged
            existing["session_count"] = n + 1
            existing.pop("level", None)
        if data.get("notes"):
            existing["notes"] = data["notes"]
        existing["last_assessed"] = now


_DEDUP_SIMILARITY_THRESHOLD = 0.80


def _append_if_novel(items: list[str], new_item: str, chunk_type: str, user_id: str, limit: int = 8) -> None:
    """Append new_item only if semantically novel. Uses persistent embedding cache."""
    if new_item in items:
        return
    from backend.vector_memory import find_similar_cached, cache_embedding, remove_cached_embedding
    if find_similar_cached(new_item, items, chunk_type, user_id, threshold=_DEDUP_SIMILARITY_THRESHOLD):
        return
    # Evict oldest before adding if at limit
    if len(items) >= limit:
        evicted = items.pop(0)
        remove_cached_embedding(evicted, chunk_type, user_id)
    items.append(new_item)
    # Cache the new item's embedding
    cache_embedding(new_item, chunk_type, user_id)


def _update_communication(profile: dict, comm: dict, user_id: str):
    """Accumulate communication observations, deduplicate via embedding similarity."""
    if not comm:
        return
    c = profile.setdefault("communication", {})
    if comm.get("style_update"):
        observations = c.setdefault("style_observations", [])
        _append_if_novel(observations, comm["style_update"], "comm_style", user_id, limit=5)
        c["style"] = observations[-1]
    for habit in comm.get("new_habits", []):
        _append_if_novel(c.setdefault("habits", []), habit, "comm_habit", user_id)
    for sug in comm.get("new_suggestions", []):
        _append_if_novel(c.setdefault("suggestions", []), sug, "comm_suggestion", user_id)


def _update_thinking_patterns(profile: dict, patterns: dict, user_id: str):
    """Accumulate thinking pattern observations, deduplicate via embedding similarity."""
    if not patterns:
        return
    tp = profile.setdefault("thinking_patterns", {"strengths": [], "gaps": []})
    for s in patterns.get("new_strengths", []):
        _append_if_novel(tp["strengths"], s, "thinking_strength", user_id)
    for g in patterns.get("new_gaps", []):
        _append_if_novel(tp["gaps"], g, "thinking_gap", user_id)


def _archive_stale_weak_points(profile: dict):
    """Archive weak points not seen recently — keeps them in profile but out of active prompts.

    Rules:
    - last_seen > 60 days → archive regardless
    - last_seen > 30 days AND times_seen <= 2 → archive
    - Already improved/archived → skip
    - source == "consolidated" → skip (refreshed by re-running consolidation, not by time)
    """
    now = datetime.now()
    for wp in profile.get("weak_points", []):
        if wp.get("improved") or wp.get("archived"):
            continue
        if wp.get("source") == "consolidated":
            continue
        last_seen_str = wp.get("last_seen", "")
        if not last_seen_str:
            continue
        try:
            last_seen = datetime.fromisoformat(last_seen_str)
        except (ValueError, TypeError):
            continue
        days_since = (now - last_seen).days
        times_seen = wp.get("times_seen", 1)
        if days_since > 60 or (days_since > 30 and times_seen <= 2):
            wp["archived"] = True
            wp["archived_at"] = now.isoformat()
            wp.setdefault("history", []).append({
                "date": now.isoformat(),
                "event": "archived",
                "reason": f"stale: {days_since}d since last seen, seen {times_seen}x",
            })


def _update_stats(
    profile: dict, mode: str, topic: str | None, avg_score: float | None,
    now: str, answer_count: int = 0, dimension_scores: dict | None = None,
):
    """Update session statistics with per-mode averages."""
    stats = profile.setdefault("stats", {})
    stats["total_sessions"] = stats.get("total_sessions", 0) + 1
    if mode == "resume":
        stats["resume_sessions"] = stats.get("resume_sessions", 0) + 1
    elif mode == "topic_drill":
        stats["drill_sessions"] = stats.get("drill_sessions", 0) + 1
    elif mode == "jd_prep":
        stats["job_prep_sessions"] = stats.get("job_prep_sessions", 0) + 1
    elif mode == "recording":
        stats["recording_sessions"] = stats.get("recording_sessions", 0) + 1
    elif mode == "copilot":
        stats["copilot_sessions"] = stats.get("copilot_sessions", 0) + 1

    if answer_count:
        stats["total_answers"] = stats.get("total_answers", 0) + answer_count

    if avg_score:
        history = stats.setdefault("score_history", [])
        entry = {"date": now[:10], "mode": mode, "topic": topic, "avg_score": avg_score}
        if dimension_scores:
            entry["dimension_scores"] = dimension_scores
        history.append(entry)

        # Per-mode rolling averages
        drill_scores = [h["avg_score"] for h in history if h.get("mode") == "topic_drill" and h.get("avg_score")][-20:]
        resume_scores = [h["avg_score"] for h in history if h.get("mode") == "resume" and h.get("avg_score")][-10:]
        job_prep_scores = [h["avg_score"] for h in history if h.get("mode") == "jd_prep" and h.get("avg_score")][-10:]

        if drill_scores:
            stats["drill_avg_score"] = round(sum(drill_scores) / len(drill_scores), 1)
        if resume_scores:
            stats["resume_avg_score"] = round(sum(resume_scores) / len(resume_scores), 1)
        if job_prep_scores:
            stats["job_prep_avg_score"] = round(sum(job_prep_scores) / len(job_prep_scores), 1)

        all_recent = [h["avg_score"] for h in history if h.get("avg_score")][-30:]
        if all_recent:
            stats["avg_score"] = round(sum(all_recent) / len(all_recent), 1)


async def llm_update_profile(
    mode: str,
    topic: str | None,
    new_weak_points: list[dict],
    new_strong_points: list[dict],
    topic_mastery: dict,
    communication: dict,
    user_id: str,
    thinking_patterns: dict | None = None,
    session_summary: str = "",
    avg_score: float | None = None,
    answer_count: int = 0,
    dimension_scores: dict | None = None,
    behavior_ops: list | None = None,
    session_id: str | None = None,
):
    """Mem0-style profile update: LLM decides ADD/UPDATE/NOOP for each fact."""
    from backend.prompts.interviewer import PROFILE_UPDATE_PROMPT

    # LLM calls happen outside the lock (they're slow and don't touch profile)
    profile = _load_profile(user_id)
    has_new_facts = bool(new_weak_points or new_strong_points)
    ops = None
    llm_failed = False

    if has_new_facts:
        # Format existing points with indices for LLM reference
        # Topic deliberately excluded — Stage 2 only compares content, not metadata
        existing_weak_lines = []
        for i, wp in enumerate(profile.get("weak_points", [])):
            status = "已改善" if wp.get("improved") else f"出现{wp.get('times_seen', 1)}次"
            existing_weak_lines.append(f"[{i}] {wp['point']} ({status})")
        existing_strong_lines = []
        for i, sp in enumerate(profile.get("strong_points", [])):
            existing_strong_lines.append(f"[{i}] {sp['point']}")

        new_weak_lines = []
        for wp in new_weak_points:
            point = wp.get("point", wp) if isinstance(wp, dict) else str(wp)
            new_weak_lines.append(f"- {point}")
        new_strong_lines = []
        for sp in new_strong_points:
            point = sp.get("point", sp) if isinstance(sp, dict) else str(sp)
            new_strong_lines.append(f"- {point}")

        prompt = PROFILE_UPDATE_PROMPT.format(
            existing_weak="\n".join(existing_weak_lines) or "暂无",
            existing_strong="\n".join(existing_strong_lines) or "暂无",
            new_weak="\n".join(new_weak_lines) or "暂无",
            new_strong="\n".join(new_strong_lines) or "暂无",
        )

        llm = get_langchain_llm()
        response = llm.invoke([
            SystemMessage(content="你是画像更新引擎。只返回 JSON。"),
            HumanMessage(content=prompt),
        ])

        try:
            ops = _parse_json_safe(response.content)
            if not isinstance(ops, dict):
                raise ValueError(f"Expected dict, got {type(ops)}")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Profile update LLM parse failed ({e}), falling back to deterministic")
            llm_failed = True

    # All profile mutations happen under the lock
    async with _get_profile_lock(user_id):
        # Re-load fresh profile inside the lock
        profile = _load_profile(user_id)
        now = datetime.now().isoformat()

        if has_new_facts:
            if ops and not llm_failed:
                _apply_memory_ops(profile, ops, topic, now, user_id=user_id,
                                  new_weak_points=new_weak_points,
                                  new_strong_points=new_strong_points)
            else:
                _deterministic_update(profile, new_weak_points, new_strong_points, topic, now, user_id)

        # ── Deterministic updates for mastery / communication / thinking / stats ──
        _update_mastery(profile, topic, topic_mastery, now, user_id=user_id)
        _update_communication(profile, communication, user_id)
        _update_thinking_patterns(profile, thinking_patterns, user_id)
        _update_stats(profile, mode, topic, avg_score, now, answer_count, dimension_scores)

        # ── Behavior axis (mem0-style ops) ──
        if behavior_ops:
            tally = _apply_behavior_ops(profile, behavior_ops, session_id, now)
            logger.info(
                f"behavior_signals updated for {user_id}: {tally}"
            )

        _archive_stale_weak_points(profile)

        _save_profile(profile, user_id)

    _save_insight(mode=mode, topic=topic, summary=session_summary, raw_extraction={
        "weak_points": new_weak_points,
        "strong_points": new_strong_points,
    }, user_id=user_id)

    # Index into vector memory for future semantic retrieval
    from backend.vector_memory import index_session_memory
    index_session_memory(
        session_id=None, topic=topic,
        summary=session_summary,
        weak_points=new_weak_points,
        strong_points=new_strong_points,
        insight_text=session_summary,
        user_id=user_id,
    )

    # ── Stage 3: Consolidation (带节流, 失败不阻塞) ──
    # 从 active observed weak_points 里识别跨领域规律, 输出 source="consolidated" 的条目.
    # 内部节流: 24h cooldown + 至少 3 条新 wp + 至少 5 条 active wp 才真的跑 LLM.
    await consolidate_patterns(user_id)


def _format_existing_behavior_signals(profile: dict) -> str:
    """Format existing behavior_signals as prior for the Extract prompt.

    Strong prior pushes the LLM to reuse existing IDs rather than minting near-duplicates.
    Only surfaces a compact summary: id, polarity tag, times_seen, description.
    """
    signals = profile.get("behavior_signals", {}) or {}
    if not signals:
        return "（暂无，本次面试可以从零开始创建。新 ID 必须严格符合 `<namespace>.<snake_case>` 格式。）"

    by_ns: dict[str, list[str]] = {}
    for sid, data in signals.items():
        if data.get("improved"):
            # 还展示,但加 "(已改善)" 提示 LLM 优先用 IMPROVE 而非重复 ADD
            status = "已改善"
        else:
            status = f"出现 {data.get('times_seen', 1)} 次"
        polarity = data.get("polarity", "negative")
        polarity_tag = "+" if polarity == "positive" else "-"
        desc = (data.get("description") or "").strip() or "（无描述）"
        line = f"- [{polarity_tag}] `{sid}` （{status}）: {desc}"
        by_ns.setdefault(data.get("namespace", "other"), []).append(line)

    parts = []
    for ns in ("reasoning", "narrative", "communication", "metacognition"):
        if ns in by_ns:
            parts.append(f"### {ns}\n" + "\n".join(by_ns[ns]))
    # 任何不在四个 namespace 的兜底展示(理论上不会有,但防御一下)
    extras = [ns for ns in by_ns if ns not in BEHAVIOR_NAMESPACES]
    for ns in extras:
        parts.append(f"### {ns} (异常 namespace, 仅展示不复用)\n" + "\n".join(by_ns[ns]))

    return "\n\n".join(parts)


async def update_profile_after_interview(
    mode: str,
    topic: str | None,
    messages: list,
    user_id: str,
    scores: list[dict] | None = None,
    session_id: str | None = None,
) -> dict:
    """Mem0-style two-stage pipeline: Extract → Update."""
    profile = _load_profile(user_id)
    llm = get_langchain_llm()

    canonical = _get_canonical_topic_keys(user_id)
    allowed_topics_str = "、".join(sorted(canonical)) if canonical else "（暂无）"

    # ── Stage 1: Extract insights ──
    transcript_lines = []
    for msg in messages:
        if hasattr(msg, "content"):
            if isinstance(msg, HumanMessage):
                transcript_lines.append(f"候选人: {msg.content}")
            elif hasattr(msg, "content") and not isinstance(msg, SystemMessage):
                transcript_lines.append(f"面试官: {msg.content}")

    score_text = ""
    if scores:
        score_text = "\n".join(
            f"- Q: {s.get('question', '?')} → {s.get('score', '?')}/10 ({s.get('assessment', '')})"
            for s in scores
        )

    extract_msg = EXTRACT_PROMPT.format(
        current_profile=json.dumps(profile, ensure_ascii=False),
        existing_behavior_signals=_format_existing_behavior_signals(profile),
        mode=mode,
        topic=topic or "综合",
        transcript="\n".join(transcript_lines),
        scores=score_text or "无",
        allowed_topics=allowed_topics_str,
    )

    response = llm.invoke([
        SystemMessage(content="你是面试分析引擎。只返回 JSON。"),
        HumanMessage(content=extract_msg),
    ])

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        extraction = json.loads(content)
    except (json.JSONDecodeError, IndexError):
        extraction = {
            "session_summary": "提取失败",
            "weak_points": [],
            "strong_points": [],
            "behavior_signals": [],
        }

    _normalize_extraction_topics(extraction, canonical, fallback_topic=topic or "")

    # ── Stage 2: LLM-based Update (Mem0 style) ──
    await llm_update_profile(
        mode=mode,
        topic=topic,
        new_weak_points=extraction.get("weak_points", []),
        new_strong_points=extraction.get("strong_points", []),
        topic_mastery=extraction.get("topic_mastery", {}),
        communication=extraction.get("communication_observations", {}),
        user_id=user_id,
        thinking_patterns=extraction.get("thinking_patterns"),
        session_summary=extraction.get("session_summary", ""),
        avg_score=extraction.get("avg_score"),
        dimension_scores=extraction.get("dimension_scores"),
        behavior_ops=extraction.get("behavior_signals", []),
        session_id=session_id,
    )

    return extraction


# ── Stage 3: Consolidation ──────────────────────────────────────────────────
# 从扁平的 weak_points 里识别跨领域规律,产出 source="consolidated" 的高层条目。
# 被整合的原始 wp 会被 archive,reason="superseded_by_consolidation"。
# 设计要点:
# - 跨至少 2 个不同 topic 才算合格 pattern (挡住同领域换粒度的假整合)
# - 失败不影响 Stage 1/2 (整个函数 try/except 包裹)
# - 节流: 24h + 至少 3 条新 observed wp 才跑一次

CONSOLIDATE_MIN_ACTIVE_WPS = 5       # 活跃 observed wp 少于这个不跑
CONSOLIDATE_MIN_NEW_WPS = 3          # 距上次 consolidation 新增少于这个不跑
CONSOLIDATE_COOLDOWN_HOURS = 24      # 两次 consolidation 之间的最小间隔
CONSOLIDATE_MIN_SUPPORTING = 2       # 一条 pattern 至少需要引用的 wp 数
CONSOLIDATE_MIN_SPANNING_TOPICS = 2  # 必须跨多少个不同 topic
CONSOLIDATE_MAX_STATEMENT_LEN = 80   # pattern 描述的字符上限

CONSOLIDATE_PROMPT = """你是面试教练的模式识别引擎。你的任务是从用户的薄弱点观察列表里,
识别**用户自己可能没意识到的跨领域规律** (pattern)。

## 合格 pattern 的 4 个必要条件

一条 pattern 必须同时满足以下 4 条, 否则视为不合格:

1. **跨至少 2 个不同的领域 (topic)**
   例: [GIL (python)] + [Transformer 注意力 (llm)] + [B+ 树 (database)]
       → 跨 3 个领域, 可能是一个真规律
   反例: [GIL (python)] + [async (python)] + [描述符 (python)]
       → 全在 python 内, 这只是一个领域的弱点, 不是跨领域规律

2. **比原始观察抽象层次更高**
   例: 5 条"底层机制讲不清"的具体观察 → 1 条"对底层原理偏表面" (思考方式的倾向)
   反例: "GIL 不懂" + "async 不懂" → "Python 并发不懂"
       (这只是换了个粒度, 没有真正抽象, 不合格)

3. **是用户自己不容易察觉的规律**
   例: "被追问 'why' 时倾向跳过推导过程" (思维模式,用户自己难以看到)
   反例: "Python 的很多东西不熟" (用户自己都知道, 没价值)

4. **可证伪**
   pattern 必须是将来能被新观察验证或推翻的具体假设。
   "你可能有点紧张"这种虚话不算。

## 什么时候不要产出

以下任何一种情况, 请返回 {{"patterns": []}}:

- 观察列表里看不到跨领域的规律
- 所有观察都集中在 1-2 个具体技术点
- 你没有高度把握某条 pattern 真的成立
- 观察之间的联系只是表面相似, 不是结构性共性

**宁可产出 0 个 pattern, 不要产出 1 个错的**。
编造的 pattern 会被用户标记为不准, 损害系统可信度。
返回空数组完全不会被惩罚, 乱产出才会被惩罚。

## 输入: 用户当前的活跃薄弱点

{weak_points_formatted}

## 输出格式 (严格 JSON)

{{
  "patterns": [
    {{
      "statement": "一句话规律描述, 不超过 40 字",
      "supporting_wp_indices": [0, 3, 7],
      "topic": "cross_cutting 或 meta",
      "confidence": 0.85,
      "reasoning": "内部用, 为什么这几条指向同一规律 (不展示给用户)"
    }}
  ]
}}

只输出 JSON, 不要任何其他内容。
"""


def _filter_active_observed_wps(profile: dict) -> list[tuple[int, dict]]:
    """返回 (原 index, wp) 对的列表, 只包含活跃的 observed 知识轴条目.

    原 index 用于 consolidation 写回时精确定位 profile["weak_points"] 里的原条目.
    """
    out = []
    for i, wp in enumerate(profile.get("weak_points", [])):
        if wp.get("improved") or wp.get("archived"):
            continue
        # 只对 observed 的条目做 consolidation, 不整合已整合过的或 JD 预测的
        if wp.get("source", "observed") != "observed":
            continue
        # 跳过老数据里的 axis=performance 条目 (这类观察现在走 behavior_signals)
        if wp.get("axis") == "performance":
            continue
        out.append((i, wp))
    return out


def _validate_consolidation_pattern(pattern: dict, active: list[tuple[int, dict]]) -> str | None:
    """验证一条 LLM 产出的 pattern. 返回 None 表示通过, 否则返回拒绝原因."""
    idxs = pattern.get("supporting_wp_indices")
    if not isinstance(idxs, list) or len(idxs) < CONSOLIDATE_MIN_SUPPORTING:
        return "too_few_supporting"

    # idxs 是"输入给 LLM 时的局部 index",引用的是 active 列表的位置
    if any(not isinstance(i, int) or i < 0 or i >= len(active) for i in idxs):
        return "invalid_index"

    # 必须跨至少 2 个 topic
    topics = {active[i][1].get("topic", "") for i in idxs}
    topics.discard("")
    if len(topics) < CONSOLIDATE_MIN_SPANNING_TOPICS:
        return "not_cross_cutting"

    statement = (pattern.get("statement") or "").strip()
    if not statement:
        return "empty_statement"
    if len(statement) > CONSOLIDATE_MAX_STATEMENT_LEN:
        return "statement_too_long"

    return None


def _apply_consolidation_pattern(profile: dict, pattern: dict, active: list[tuple[int, dict]], now: str):
    """把一条 pattern 写入 profile: 追加新 consolidated wp + archive 被 supersede 的原条目."""
    idxs = pattern["supporting_wp_indices"]
    supporting_pairs = [active[i] for i in idxs]
    supporting_wps = [wp for _, wp in supporting_pairs]

    new_wp = {
        "point": pattern["statement"].strip(),
        "topic": pattern.get("topic") or "cross_cutting",
        "source": "consolidated",
        "first_seen": now,
        "last_seen": now,
        "times_seen": sum(w.get("times_seen", 1) for w in supporting_wps),
        "improved": False,
        "archived": False,
        "consolidates": [w.get("point", "") for w in supporting_wps],
        "confidence": float(pattern.get("confidence", 0.7)),
        "user_acknowledged": False,
    }
    profile.setdefault("weak_points", []).append(new_wp)

    # Archive 被 supersede 的原条目 (用原 profile index 精确定位, 防止锁外并发写)
    all_wps = profile.get("weak_points", [])
    for orig_idx, wp in supporting_pairs:
        if orig_idx >= len(all_wps):
            continue
        target = all_wps[orig_idx]
        # 再次确认这条就是我们要改的 (防止锁外并发写把 list 改了)
        if target.get("point") != wp.get("point"):
            continue
        target["archived"] = True
        target["archived_at"] = now
        target["archived_reason"] = "superseded_by_consolidation"
        target.setdefault("history", []).append({
            "date": now,
            "event": "archived",
            "reason": f"superseded by consolidation: {new_wp['point'][:40]}",
        })


def _should_run_consolidation(profile: dict) -> tuple[bool, str]:
    """检查节流条件. 返回 (是否应该跑, 原因)."""
    active = _filter_active_observed_wps(profile)
    if len(active) < CONSOLIDATE_MIN_ACTIVE_WPS:
        return False, f"too_few_active_wps ({len(active)} < {CONSOLIDATE_MIN_ACTIVE_WPS})"

    last_str = profile.get("last_consolidation_at", "")
    if last_str:
        try:
            last_time = datetime.fromisoformat(last_str)
            hours_since = (datetime.now() - last_time).total_seconds() / 3600
            if hours_since < CONSOLIDATE_COOLDOWN_HOURS:
                return False, f"cooldown (last run {hours_since:.1f}h ago)"
        except (ValueError, TypeError):
            pass  # 解析失败就当没跑过

        # 至少 N 条新 observed wp 才值得重跑
        new_count = 0
        for _, wp in active:
            first_seen = wp.get("first_seen", "")
            try:
                if datetime.fromisoformat(first_seen) > last_time:
                    new_count += 1
            except (ValueError, TypeError):
                continue
        if new_count < CONSOLIDATE_MIN_NEW_WPS:
            return False, f"too_few_new_wps ({new_count} < {CONSOLIDATE_MIN_NEW_WPS})"

    return True, "ok"


async def consolidate_patterns(user_id: str) -> dict:
    """Stage 3: 从 active observed weak_points 里识别跨领域规律.

    带节流: 满足 cooldown + 新观察数量 + 活跃数量三个条件才真的跑 LLM.
    失败不影响上游 (所有异常在这里被吞).

    Returns:
        {"ran": bool, "applied": int, "skipped": list, "reason": str}
    """
    try:
        profile = _load_profile(user_id)

        should_run, reason = _should_run_consolidation(profile)
        if not should_run:
            return {"ran": False, "applied": 0, "skipped": [], "reason": reason}

        active = _filter_active_observed_wps(profile)
        formatted = "\n".join(
            f"[{i}] {wp['point']} (领域: {wp.get('topic', '?')}, 观察 {wp.get('times_seen', 1)} 次)"
            for i, (_, wp) in enumerate(active)
        )

        llm = get_langchain_llm()
        response = llm.invoke([
            SystemMessage(content="你是面试教练的模式识别引擎。只返回 JSON。宁可不产出,不要编造。"),
            HumanMessage(content=CONSOLIDATE_PROMPT.format(weak_points_formatted=formatted)),
        ])

        try:
            parsed = _parse_json_safe(response.content)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict, got {type(parsed)}")
            raw_patterns = parsed.get("patterns", []) or []
            if not isinstance(raw_patterns, list):
                raise ValueError("patterns is not a list")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Consolidation parse failed: {e}. Raw: {response.content[:200]}")
            # 解析失败不更新 last_consolidation_at, 下次 session 会重试
            return {"ran": False, "applied": 0, "skipped": [], "reason": "llm_parse_failed"}

        # 验证
        valid_patterns = []
        skipped = []
        for p in raw_patterns:
            if not isinstance(p, dict):
                skipped.append({"reason": "not_a_dict"})
                continue
            rej = _validate_consolidation_pattern(p, active)
            if rej is None:
                valid_patterns.append(p)
            else:
                skipped.append({"statement": p.get("statement", "?"), "reason": rej})

        # 写入 (在锁内)
        applied = 0
        async with _get_profile_lock(user_id):
            profile = _load_profile(user_id)
            # 锁内重新过滤 active, 因为 profile 在 LLM 期间可能被并发写
            active_inside = _filter_active_observed_wps(profile)

            # 重新验证 index 还有效 (active 可能变短了)
            now = datetime.now().isoformat()
            for p in valid_patterns:
                idxs = p["supporting_wp_indices"]
                if any(i >= len(active_inside) for i in idxs):
                    skipped.append({"statement": p.get("statement", "?"), "reason": "stale_index_after_reload"})
                    continue
                # 还要确认 active 列表的顺序没变 (通过比对 point 文本)
                ok = True
                for local_i in idxs:
                    orig_idx_outside = active[local_i][0]
                    if orig_idx_outside >= len(profile.get("weak_points", [])):
                        ok = False
                        break
                    if profile["weak_points"][orig_idx_outside].get("point") != active[local_i][1].get("point"):
                        ok = False
                        break
                if not ok:
                    skipped.append({"statement": p.get("statement", "?"), "reason": "profile_changed_during_llm"})
                    continue

                _apply_consolidation_pattern(profile, p, active, now)
                applied += 1

            profile["last_consolidation_at"] = now
            _save_profile(profile, user_id)

        logger.info(
            f"Consolidation for user {user_id}: applied={applied}, skipped={len(skipped)}, "
            f"candidates={len(raw_patterns)}"
        )
        return {"ran": True, "applied": applied, "skipped": skipped, "reason": "ok"}

    except Exception as e:
        logger.warning(f"Consolidation failed for user {user_id}: {type(e).__name__}: {e}")
        return {"ran": False, "applied": 0, "skipped": [], "reason": f"error: {type(e).__name__}"}
