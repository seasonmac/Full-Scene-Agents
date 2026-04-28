"""Topic Aggregation: group related episodes into topics via hyperedges.

Uses BM25+Vector similarity to find candidate topics for an episode,
then creates/updates topic nodes and hyperedges.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Sequence

import numpy as np

from ebm_context_engine.client import cosine_similarity
from ebm_context_engine.core.hash import stableId
from ebm_context_engine.text import tokenize
from ebm_context_engine.types import HmEpisode, HmTopic

logger = logging.getLogger("ebm_context_engine.hypergraph.topic_aggregator")

InferenceFn = Callable[[str], str]
EmbedFn = Callable[[str], np.ndarray | None]

TOPIC_MATCH_PROMPT = """Given an EXISTING TOPIC and a NEW EPISODE, determine if they belong to the same topic.

Answer with a JSON object:
- "same_topic": true/false
- "updated_title": if same_topic, the improved topic title (max 10 words)
- "updated_summary": if same_topic, a merged summary covering both (max 2 sentences)
- "updated_keywords": if same_topic, merged keywords list (6-8 items)

EXISTING TOPIC:
Title: {topic_title}
Summary: {topic_summary}
Keywords: {topic_keywords}

NEW EPISODE:
Title: {episode_title}
Summary: {episode_summary}
Keywords: {episode_keywords}

OUTPUT FORMAT (JSON only, no markdown):
"""

NEW_TOPIC_PROMPT = """Create a TOPIC node for this episode. A topic groups related episodes about the same subject.

Episode Title: {episode_title}
Episode Summary: {episode_summary}
Episode Keywords: {episode_keywords}

OUTPUT FORMAT (JSON only, no markdown):
{{"title": "short topic title (max 10 words)", "summary": "1-2 sentence topic description", "keywords": ["keyword1", "keyword2", ...]}}
"""

# Similarity thresholds
BM25_MATCH_THRESHOLD = 2  # minimum overlapping tokens to consider
VECTOR_MATCH_THRESHOLD = 0.4  # cosine similarity threshold
MAX_CANDIDATE_TOPICS = 5  # max topics to evaluate with LLM


def _parse_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text.strip())
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _keyword_overlap_score(a_keywords: list[str], b_keywords: list[str]) -> int:
    """Count overlapping keywords (case-insensitive)."""
    a_set = {k.lower().strip() for k in a_keywords if k.strip()}
    b_set = {k.lower().strip() for k in b_keywords if k.strip()}
    return len(a_set & b_set)


def aggregate_episode_to_topic(
    episode: HmEpisode,
    existing_topics: list[HmTopic],
    inference_fn: InferenceFn | None,
    embed_fn: EmbedFn | None,
) -> HmTopic:
    """Assign an episode to a topic: either match an existing one or create new.

    Returns the topic (new or updated) with the episode_id added to episode_ids.
    """
    logger.info("开始聚合片段到主题: episode_id=%s, 已有主题数=%d, LLM可用=%s",
                episode.id[:12], len(existing_topics), inference_fn is not None)
    now = int(time.time() * 1000)

    # 1. Find candidate topics by keyword overlap + vector similarity
    candidates: list[tuple[HmTopic, float]] = []
    episode_text = f"{episode.title} {episode.summary} {' '.join(episode.keywords)}"

    for topic in existing_topics:
        # BM25-like keyword overlap
        kw_score = _keyword_overlap_score(episode.keywords, topic.keywords)
        # Also check title/summary token overlap
        topic_text = f"{topic.title} {topic.summary} {' '.join(topic.keywords)}"
        topic_tokens = set(tokenize(topic_text))
        episode_tokens = set(tokenize(episode_text))
        token_overlap = len(topic_tokens & episode_tokens)

        # Vector similarity (if available)
        vec_score = 0.0
        if embed_fn and episode.vector is not None and topic.vector is not None:
            vec_score = cosine_similarity(episode.vector, topic.vector)

        combined_score = kw_score * 2.0 + token_overlap * 0.5 + vec_score * 5.0
        if kw_score >= BM25_MATCH_THRESHOLD or token_overlap >= 3 or vec_score >= VECTOR_MATCH_THRESHOLD:
            candidates.append((topic, combined_score))

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    candidates = candidates[:MAX_CANDIDATE_TOPICS]
    logger.debug("候选主题数=%d (从 %d 个已有主题中筛选)", len(candidates), len(existing_topics))

    # 2. If LLM available, ask it to confirm match
    if inference_fn and candidates:
        for topic, _score in candidates:
            prompt = TOPIC_MATCH_PROMPT.format(
                topic_title=topic.title,
                topic_summary=topic.summary,
                topic_keywords=", ".join(topic.keywords),
                episode_title=episode.title,
                episode_summary=episode.summary,
                episode_keywords=", ".join(episode.keywords),
            )
            try:
                response = inference_fn(prompt)
                result = _parse_json_object(response)
                if result and result.get("same_topic"):
                    # Update existing topic
                    logger.info("LLM 确认匹配已有主题: topic_id=%s, episode_id=%s", topic.id[:12], episode.id[:12])
                    updated_title = str(result.get("updated_title", topic.title) or topic.title).strip()
                    updated_summary = str(result.get("updated_summary", topic.summary) or topic.summary).strip()
                    updated_keywords = result.get("updated_keywords", topic.keywords)
                    if not isinstance(updated_keywords, list):
                        updated_keywords = topic.keywords

                    topic.title = updated_title[:200]
                    topic.summary = updated_summary[:500]
                    topic.keywords = [str(k) for k in updated_keywords[:10] if k]
                    if episode.id not in topic.episode_ids:
                        topic.episode_ids.append(episode.id)
                    topic.updated_at = now
                    if episode.id not in episode.topic_ids:
                        episode.topic_ids.append(topic.id)
                    return topic
            except Exception:
                logger.debug("LLM 主题匹配调用失败, 尝试下一个候选主题")
                continue

    # 3. If no match or no LLM, check if high-score candidate exists (heuristic)
    if candidates and not inference_fn:
        best_topic, best_score = candidates[0]
        if best_score >= 6.0:  # strong heuristic match
            logger.info("启发式匹配到已有主题: topic_id=%s, score=%.2f, episode_id=%s", best_topic.id[:12], best_score, episode.id[:12])
            if episode.id not in best_topic.episode_ids:
                best_topic.episode_ids.append(episode.id)
            best_topic.updated_at = now
            # Merge keywords
            merged_kw = list(dict.fromkeys(best_topic.keywords + episode.keywords))[:10]
            best_topic.keywords = merged_kw
            if best_topic.id not in episode.topic_ids:
                episode.topic_ids.append(best_topic.id)
            return best_topic

    # 4. Create new topic
    if inference_fn:
        prompt = NEW_TOPIC_PROMPT.format(
            episode_title=episode.title,
            episode_summary=episode.summary,
            episode_keywords=", ".join(episode.keywords),
        )
        try:
            response = inference_fn(prompt)
            result = _parse_json_object(response)
            if result:
                title = str(result.get("title", episode.title) or episode.title).strip()
                summary = str(result.get("summary", episode.summary) or episode.summary).strip()
                keywords = result.get("keywords", episode.keywords)
                if not isinstance(keywords, list):
                    keywords = episode.keywords
                keywords = [str(k) for k in keywords if k]

                topic_id = stableId("HM_TOPIC", title.lower())
                logger.info("LLM 创建新主题: topic_id=%s, title='%s', episode_id=%s", topic_id[:12], title[:50], episode.id[:12])
                new_topic = HmTopic(
                    id=topic_id,
                    title=title[:200],
                    summary=summary[:500],
                    keywords=keywords[:10],
                    episode_ids=[episode.id],
                    created_at=now,
                    updated_at=now,
                )
                episode.topic_ids.append(topic_id)
                return new_topic
        except Exception as exc:
            logger.debug("LLM主题创建失败, 回退到启发式: %s", exc, exc_info=True)

    # Heuristic fallback: create topic from episode metadata
    topic_id = stableId("HM_TOPIC", episode.title.lower(), episode.session_key)
    logger.info("启发式创建新主题: topic_id=%s, title='%s', episode_id=%s", topic_id[:12], episode.title[:50], episode.id[:12])
    new_topic = HmTopic(
        id=topic_id,
        title=episode.title[:200],
        summary=episode.summary[:500],
        keywords=episode.keywords[:10],
        episode_ids=[episode.id],
        created_at=now,
        updated_at=now,
    )
    episode.topic_ids.append(topic_id)
    return new_topic


def aggregate_episodes_to_topics(
    episodes: Sequence[HmEpisode],
    existing_topics: list[HmTopic],
    inference_fn: InferenceFn | None,
    embed_fn: EmbedFn | None,
) -> list[HmTopic]:
    """Process a batch of episodes, assigning each to topics.

    Returns all topics (new + updated).
    """
    logger.info("开始批量聚合片段到主题: 片段数=%d, 已有主题数=%d", len(episodes), len(existing_topics))
    all_topics = list(existing_topics)
    topic_map: dict[str, HmTopic] = {t.id: t for t in all_topics}

    for episode in episodes:
        topic = aggregate_episode_to_topic(episode, all_topics, inference_fn, embed_fn)
        if topic.id not in topic_map:
            topic_map[topic.id] = topic
            all_topics.append(topic)

    logger.info("批量聚合完成: 输入片段=%d, 最终主题数=%d (新增 %d 个)",
                len(episodes), len(topic_map), len(topic_map) - len(existing_topics))
    return list(topic_map.values())
