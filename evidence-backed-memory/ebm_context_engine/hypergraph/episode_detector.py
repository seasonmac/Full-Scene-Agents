"""Episode Detection: segment a session dialogue into coherent episodes.

Each episode is a semantically coherent chunk of conversation about one topic/event.
Uses LLM to detect boundaries, with a robust fallback for when LLM is unavailable.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Sequence

from ebm_context_engine.core.hash import stableId
from ebm_context_engine.text import top_keywords
from ebm_context_engine.types import HmEpisode, TranscriptEntry

logger = logging.getLogger("ebm_context_engine.hypergraph.episode_detector")

InferenceFn = Callable[[str], str]

EPISODE_DETECTION_PROMPT = """You are an episode boundary detector for a conversation memory system.

A conversation may cover multiple topics. Segment it into EPISODES — each episode is a coherent block about one topic, event, or activity.

RULES:
- Each episode should have 2-20 messages (merge very short exchanges, split very long monologues)
- An episode boundary occurs when the topic clearly shifts (e.g., from "weekend plans" to "work project")
- Small tangents within a topic (1-2 messages) should be included in the surrounding episode, not split
- NEVER use relative time expressions. Use absolute dates from the conversation header.
- keywords: 3-6 important nouns/names/dates (not stopwords)
- title: short descriptive label (max 10 words)
- summary: 1-2 sentences capturing the key facts

OUTPUT FORMAT (JSON array only, no markdown):
[
  {
    "turn_start": 0,
    "turn_end": 5,
    "title": "Project Workshop Reflection",
    "summary": "Alex described attending a workshop on 7 May 2023 and explained how it clarified the next steps for a community project.",
    "keywords": ["workshop", "project", "Alex", "7 May 2023", "planning"]
  }
]

CONVERSATION:
"""

FALLBACK_EPISODE_SIZE = 8  # messages per episode in heuristic mode


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[[\s\S]*\]", text.strip())
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _build_dialogue_text(entries: Sequence[TranscriptEntry], date_time: str = "") -> str:
    """Build formatted dialogue text from transcript entries."""
    lines: list[str] = []
    if date_time:
        lines.append(f"[conversation: {date_time}]")
    for entry in entries:
        lines.append(f"[turn {entry.turn_index}] {entry.speaker}: {entry.text}")
    return "\n".join(lines)


def detect_episodes_llm(
    entries: Sequence[TranscriptEntry],
    session_key: str,
    date_time: str,
    inference_fn: InferenceFn,
    max_chars: int = 12000,
) -> list[HmEpisode]:
    """Use LLM to detect episode boundaries in a conversation."""
    logger.info("开始 LLM 片段检测: session_key=%s, 条目数=%d, date_time=%s", session_key, len(entries), date_time)
    if not entries:
        logger.info("片段检测跳过: 无条目")
        return []

    dialogue_text = _build_dialogue_text(entries, date_time)
    prompt = EPISODE_DETECTION_PROMPT + dialogue_text[:max_chars]
    logger.debug("片段检测 prompt 长度=%d, 对话文本长度=%d", len(prompt), len(dialogue_text))

    try:
        response = inference_fn(prompt)
        logger.debug("LLM 片段检测响应长度=%d", len(response) if response else 0)
    except Exception as exc:
        logger.warning("LLM 片段检测调用失败, 回退到启发式方法: %s", exc)
        return detect_episodes_heuristic(entries, session_key, date_time)

    parsed = _parse_json_array(response)
    if not parsed:
        logger.warning("LLM 片段检测响应解析失败, 回退到启发式方法, 响应长度=%d", len(response) if response else 0)
        return detect_episodes_heuristic(entries, session_key, date_time)

    now = int(time.time() * 1000)
    min_turn = min(e.turn_index for e in entries)
    max_turn = max(e.turn_index for e in entries)

    episodes: list[HmEpisode] = []
    for item in parsed:
        turn_start = int(item.get("turn_start", min_turn) or min_turn)
        turn_end = int(item.get("turn_end", max_turn) or max_turn)
        title = str(item.get("title", "") or "").strip()
        summary = str(item.get("summary", "") or "").strip()
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k) for k in keywords if k]

        if not title and not summary:
            continue

        # Clamp to valid range
        turn_start = max(turn_start, min_turn)
        turn_end = min(turn_end, max_turn)

        # Build episode dialogue from entries
        episode_entries = [e for e in entries if turn_start <= e.turn_index <= turn_end]
        dialogue = _build_dialogue_text(episode_entries)

        # Determine timestamps
        ts_start = ""
        ts_end = ""
        if episode_entries:
            ts_start = episode_entries[0].date_time
            ts_end = episode_entries[-1].date_time

        episode_id = stableId("HM_EPISODE", session_key, turn_start, turn_end)
        episodes.append(HmEpisode(
            id=episode_id,
            session_key=session_key,
            title=title[:200],
            summary=summary[:500],
            dialogue=dialogue[:4000],
            keywords=keywords[:8],
            timestamp_start=ts_start,
            timestamp_end=ts_end,
            turn_start=turn_start,
            turn_end=turn_end,
            created_at=now,
        ))

    if episodes:
        logger.info("LLM 片段检测完成: session_key=%s, 检测到 %d 个片段", session_key, len(episodes))
    else:
        logger.warning("LLM 片段检测未产生有效片段, 回退到启发式方法: session_key=%s", session_key)
    return episodes if episodes else detect_episodes_heuristic(entries, session_key, date_time)


def detect_episodes_heuristic(
    entries: Sequence[TranscriptEntry],
    session_key: str,
    date_time: str,
    chunk_size: int = FALLBACK_EPISODE_SIZE,
) -> list[HmEpisode]:
    """Heuristic fallback: split dialogue into fixed-size chunks."""
    logger.info("开始启发式片段检测: session_key=%s, 条目数=%d, chunk_size=%d", session_key, len(entries), chunk_size)
    if not entries:
        logger.info("启发式片段检测跳过: 无条目")
        return []

    now = int(time.time() * 1000)
    sorted_entries = sorted(entries, key=lambda e: e.turn_index)
    episodes: list[HmEpisode] = []

    for i in range(0, len(sorted_entries), chunk_size):
        chunk = sorted_entries[i : i + chunk_size]
        turn_start = chunk[0].turn_index
        turn_end = chunk[-1].turn_index
        dialogue = _build_dialogue_text(chunk)

        keywords = top_keywords([e.text for e in chunk], limit=6)
        title = f"Conversation turns {turn_start}-{turn_end}"
        summary = f"{date_time}: {chunk[0].speaker} and others discuss {', '.join(keywords[:3]) if keywords else 'various topics'}."

        episode_id = stableId("HM_EPISODE", session_key, turn_start, turn_end)
        episodes.append(HmEpisode(
            id=episode_id,
            session_key=session_key,
            title=title[:200],
            summary=summary[:500],
            dialogue=dialogue[:4000],
            keywords=keywords[:8],
            timestamp_start=chunk[0].date_time,
            timestamp_end=chunk[-1].date_time,
            turn_start=turn_start,
            turn_end=turn_end,
            created_at=now,
        ))

    logger.info("启发式片段检测完成: session_key=%s, 生成 %d 个片段", session_key, len(episodes))
    return episodes
