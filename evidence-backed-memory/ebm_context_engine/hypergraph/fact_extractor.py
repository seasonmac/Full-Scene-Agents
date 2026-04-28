"""Atomic Fact Extraction with Query Anticipation (potential field).

Extracts fine-grained atomic facts from episodes. Each fact includes:
- content: the factual assertion
- potential: anticipated queries this fact could answer
- keywords: important terms for BM25 matching
- importance: high/mid/low relevance rating
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Sequence

from ebm_context_engine.core.hash import stableId
from ebm_context_engine.text import top_keywords
from ebm_context_engine.types import HmEpisode, HmFact

logger = logging.getLogger("ebm_context_engine.hypergraph.fact_extractor")

InferenceFn = Callable[[str], str]

# Multilingual greeting/filler phrases for filtering
_GREETINGS = frozenset({
    # English
    "hey", "hi", "hello", "thanks", "thank you", "sure thing", "absolutely",
    "no doubt", "you got this", "you're welcome", "take care", "bye",
    "see you", "cheers", "sounds good", "awesome", "great",
    # Chinese
    "你好", "嗨", "谢谢", "感谢", "没问题", "好的", "再见", "拜拜",
    "不客气", "保重", "加油", "太好了", "棒", "厉害",
})


def _is_greeting(text: str) -> bool:
    """Check if text starts with a greeting/filler phrase (multilingual)."""
    lower = text.lower().strip()
    for g in _GREETINGS:
        if lower.startswith(g):
            return True
    return False

FACT_EXTRACTION_PROMPT = """Extract atomic facts from this conversation episode. Each fact = one third-person declarative statement.

Rules:
- content: third-person statement, NEVER a dialogue quote (e.g. "Jon is searching for a dance studio location")
- Extract: events, dates, opinions, reactions, descriptions, preferences, relationships, plans
- When A comments on B: "A said/thinks [specific content about B]"
- Convert relative dates to absolute using conversation date
- keywords: 3-5 key nouns/names
- importance: high/mid/low
- CRITICAL: Preserve exact nouns, proper names, quantities, and specific details. Do NOT generalize or paraphrase.
  - GOOD: "Caroline is researching adoption agencies" / "Jon signed up for a marathon in April"
  - BAD: "Caroline is doing research" / "Jon is planning to exercise"
- Each fact must be self-contained and answerable: someone reading only this fact should understand WHO did WHAT specifically.

Output JSON array only:
[{{"content":"...","keywords":["..."],"importance":"high"}}]

TITLE: {title}
SUMMARY: {summary}
DIALOGUE:
{dialogue}
"""


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


def extract_facts_from_episode(
    episode: HmEpisode,
    inference_fn: InferenceFn,
    max_chars: int = 8000,
    max_retries: int = 2,
) -> list[HmFact]:
    """Extract atomic facts from a single episode using LLM."""
    logger.info("开始从片段提取原子事实: episode_id=%s, title='%s', max_chars=%d",
                episode.id[:12], episode.title[:50], max_chars)

    prompt = FACT_EXTRACTION_PROMPT.format(
        title=episode.title,
        summary=episode.summary,
        dialogue=episode.dialogue[:max_chars],
    )

    response = None
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = inference_fn(prompt)
            logger.debug("事实提取 LLM 响应: episode_id=%s, 尝试=%d, 响应长度=%d",
                         episode.id[:12], attempt, len(response) if response else 0)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("事实提取 LLM 调用失败: episode_id=%s, 尝试=%d, 错误=%s", episode.id[:12], attempt, exc)
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))  # backoff: 2s, 4s
            continue

    if response is None:
        logger.warning("事实提取 LLM 全部 %d 次尝试失败, 回退到启发式: episode_id=%s, 错误=%s",
                       max_retries + 1, episode.id[:12], last_exc)
        return extract_facts_heuristic(episode)

    parsed = _parse_json_array(response)
    if not parsed:
        logger.warning("事实提取响应解析失败, 回退到启发式: episode_id=%s, 响应长度=%d",
                       episode.id[:12], len(response) if response else 0)
        return extract_facts_heuristic(episode)

    now = int(time.time() * 1000)
    facts: list[HmFact] = []

    for item in parsed:
        content = str(item.get("content", "") or "").strip()
        if not content:
            continue
        potential = str(item.get("potential", "") or "").strip()
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [str(k) for k in keywords if k]
        importance = str(item.get("importance", "mid") or "mid").lower()
        if importance not in {"high", "mid", "low"}:
            importance = "mid"

        fact_id = stableId("HM_FACT", episode.id, content)
        facts.append(HmFact(
            id=fact_id,
            content=content[:500],
            potential=potential[:300],
            keywords=keywords[:8],
            importance=importance,
            episode_id=episode.id,
            session_key=episode.session_key,
            source_turn_start=episode.turn_start,
            source_turn_end=episode.turn_end,
            created_at=now,
        ))

    logger.info("LLM 事实提取完成: episode_id=%s, 提取 %d 个事实", episode.id[:12], len(facts))
    return facts if facts else extract_facts_heuristic(episode)


def extract_facts_heuristic(episode: HmEpisode) -> list[HmFact]:
    """Heuristic fallback: one fact per dialogue turn, filtering greetings."""
    logger.info("开始启发式事实提取: episode_id=%s, title='%s'", episode.id[:12], episode.title[:50])
    now = int(time.time() * 1000)
    facts: list[HmFact] = []

    # Parse dialogue turns: "[turn N] Speaker: content"
    turn_pattern = re.compile(r"\[turn\s+\d+\]\s*(\w+):\s*(.*)")

    lines = [line.strip() for line in episode.dialogue.split("\n") if line.strip()]

    for line in lines:
        m = turn_pattern.match(line)
        if not m:
            continue
        speaker = m.group(1)
        content = m.group(2).strip()
        if not content or len(content) < 20:
            continue
        # Skip pure greetings/filler (multilingual)
        if _is_greeting(content) and len(content) < 80:
            continue

        # Build fact: "Speaker: substantive content"
        fact_content = f"{speaker}: {content}"[:500]
        keywords = top_keywords([fact_content], limit=5)
        fact_id = stableId("HM_FACT", episode.id, fact_content)
        facts.append(HmFact(
            id=fact_id,
            content=fact_content,
            potential="",
            keywords=keywords,
            importance="mid",
            episode_id=episode.id,
            session_key=episode.session_key,
            source_turn_start=episode.turn_start,
            source_turn_end=episode.turn_end,
            created_at=now,
        ))

    # If no parsed turns, fallback to line chunking
    if not facts:
        content_lines = [l for l in lines if not l.startswith("[conversation:")]
        if not content_lines:
            content_lines = lines
        for i in range(0, len(content_lines), 2):
            chunk = content_lines[i : i + 2]
            content = " ".join(chunk)[:500]
            if len(content) < 10:
                continue
            keywords = top_keywords(chunk, limit=5)
            fact_id = stableId("HM_FACT", episode.id, str(i))
            facts.append(HmFact(
                id=fact_id,
                content=content,
                potential="",
                keywords=keywords,
                importance="mid",
                episode_id=episode.id,
                session_key=episode.session_key,
                source_turn_start=episode.turn_start,
                source_turn_end=episode.turn_end,
                created_at=now,
            ))

    logger.info("启发式事实提取完成: episode_id=%s, 提取 %d 个事实", episode.id[:12], len(facts))
    return facts


def extract_facts_batch(
    episodes: Sequence[HmEpisode],
    inference_fn: InferenceFn,
    max_chars: int = 8000,
) -> dict[str, list[HmFact]]:
    """Extract facts from multiple episodes. Returns {episode_id: [facts]}."""
    logger.info("开始批量事实提取: 片段数=%d", len(episodes))
    result: dict[str, list[HmFact]] = {}
    for episode in episodes:
        result[episode.id] = extract_facts_from_episode(episode, inference_fn, max_chars)
    logger.info("批量事实提取完成: 片段数=%d, 总事实数=%d", len(episodes), sum(len(v) for v in result.values()))
    return result
