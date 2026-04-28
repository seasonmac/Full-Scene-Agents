"""Coarse-to-Fine Retrieval: Topic → Episode → Fact.

Three-level retrieval with BM25 + Dense vector + RRF fusion at each level.
Includes temporal decay, intent-adaptive RRF, and relevance gating.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Sequence

import numpy as np

from ebm_context_engine.client import cosine_similarity
from ebm_context_engine.text import tokenize
from ebm_context_engine.types import HmEpisode, HmFact, HmTopic, RecallHit

logger = logging.getLogger("ebm_context_engine.hypergraph.c2f_retriever")


# ── Intent-adaptive RRF k values ──
# Lower k = more emphasis on top ranks; higher k = smoother
_RRF_K_BY_INTENT = {
    "temporal": 40,   # trust BM25 date matching more
    "causal": 60,
    "multi_hop": 60,
    "fact_lookup": 50,
    "opinion": 70,    # broader recall for open questions
    "comparison": 60,
}
_DEFAULT_RRF_K = 60

# Temporal decay: half-life in seconds (90 days)
_TEMPORAL_DECAY_HALF_LIFE = 90 * 24 * 3600


def _bm25_score(query_tokens: set[str], doc_tokens: list[str], k1: float = 1.2, b: float = 0.75, avg_dl: float = 50.0) -> float:
    """Simple BM25 scoring (no IDF — approximated by rarity in doc)."""
    dl = len(doc_tokens)
    if dl == 0:
        return 0.0
    doc_token_set = set(doc_tokens)
    score = 0.0
    for qt in query_tokens:
        tf = doc_tokens.count(qt) if qt in doc_token_set else 0
        if tf == 0:
            continue
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += numerator / denominator
    return score


def _temporal_decay(created_at_ms: int, now_ms: int | None = None) -> float:
    """Exponential decay factor based on age. Returns 0.5..1.0."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    age_s = max(0, (now_ms - created_at_ms) / 1000)
    decay = 0.5 ** (age_s / _TEMPORAL_DECAY_HALF_LIFE)
    return max(0.5, decay)  # floor at 0.5 to avoid zeroing old items


def _rrf_fuse(
    bm25_ranks: dict[str, int],
    vec_ranks: dict[str, int],
    k: int = 60,
    bm25_weight: float = 1.0,
    vec_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Weighted Reciprocal Rank Fusion of two ranked lists."""
    all_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys())
    scores: dict[str, float] = {}
    for doc_id in all_ids:
        bm25_r = bm25_ranks.get(doc_id, len(bm25_ranks) + 100)
        vec_r = vec_ranks.get(doc_id, len(vec_ranks) + 100)
        scores[doc_id] = bm25_weight / (k + bm25_r) + vec_weight / (k + vec_r)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


def _rank_by_bm25(query_tokens: set[str], items: Sequence, text_fn, top_k: int) -> dict[str, int]:
    """Rank items by BM25 score, return {id: rank (0-based)}."""
    scored: list[tuple[str, float]] = []
    for item in items:
        text = text_fn(item)
        tokens = tokenize(text)
        score = _bm25_score(query_tokens, tokens)
        scored.append((item.id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {doc_id: rank for rank, (doc_id, _) in enumerate(scored[:top_k])}


def _rank_by_vector(query_vector: np.ndarray | None, items: Sequence, top_k: int) -> dict[str, int]:
    """Rank items by cosine similarity to query vector, return {id: rank}."""
    if query_vector is None:
        return {}
    scored: list[tuple[str, float]] = []
    for item in items:
        if item.vector is not None:
            score = cosine_similarity(query_vector, item.vector)
            scored.append((item.id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return {doc_id: rank for rank, (doc_id, _) in enumerate(scored[:top_k])}


def retrieve_topics(
    query: str,
    query_vector: np.ndarray | None,
    topics: Sequence[HmTopic],
    top_k: int = 5,
    rrf_k: int = _DEFAULT_RRF_K,
) -> list[tuple[HmTopic, float]]:
    """Level 1: Retrieve top-k topics via BM25+Vector RRF."""
    logger.info("开始检索主题: query='%s', 主题总数=%d, top_k=%d", query[:80], len(topics), top_k)
    if not topics:
        logger.info("主题检索跳过: 无可用主题")
        return []

    query_tokens = set(tokenize(query))
    bm25_ranks = _rank_by_bm25(
        query_tokens, topics,
        lambda t: f"{t.title} {t.summary} {' '.join(t.keywords)}",
        top_k * 2,
    )
    vec_ranks = _rank_by_vector(query_vector, topics, top_k * 2)
    fused = _rrf_fuse(bm25_ranks, vec_ranks, k=rrf_k)

    topic_map = {t.id: t for t in topics}
    results: list[tuple[HmTopic, float]] = []
    for doc_id, score in fused[:top_k]:
        if doc_id in topic_map:
            results.append((topic_map[doc_id], score))
    logger.info("主题检索完成: 返回 %d 个主题", len(results))
    logger.debug("主题检索详情: BM25候选=%d, 向量候选=%d, 融合后=%d", len(bm25_ranks), len(vec_ranks), len(fused))
    return results


def retrieve_episodes(
    query: str,
    query_vector: np.ndarray | None,
    topic_results: list[tuple[HmTopic, float]],
    all_episodes: Sequence[HmEpisode],
    top_k: int = 5,
    rrf_k: int = _DEFAULT_RRF_K,
    bm25_weight: float = 1.0,
    vec_weight: float = 1.0,
) -> list[tuple[HmEpisode, float]]:
    """Level 2: From selected topics, retrieve top-k episodes via BM25+Vector RRF."""
    logger.info("开始检索片段: query='%s', 来源主题数=%d, 全部片段数=%d, top_k=%d", query[:80], len(topic_results), len(all_episodes), top_k)
    candidate_episode_ids: set[str] = set()
    for topic, _ in topic_results:
        candidate_episode_ids.update(topic.episode_ids)
    logger.debug("从主题中收集候选片段 ID 数=%d", len(candidate_episode_ids))

    episode_map = {e.id: e for e in all_episodes}
    candidate_episodes = [episode_map[eid] for eid in candidate_episode_ids if eid in episode_map]

    if not candidate_episodes:
        logger.debug("主题关联片段为空, 使用全部片段作为候选")
        candidate_episodes = list(all_episodes)

    if not candidate_episodes:
        logger.info("片段检索跳过: 无可用片段")
        return []

    query_tokens = set(tokenize(query))
    bm25_ranks = _rank_by_bm25(
        query_tokens, candidate_episodes,
        lambda e: f"{e.title} {e.summary} {' '.join(e.keywords)} {e.dialogue[:300]}",
        top_k * 2,
    )
    vec_ranks = _rank_by_vector(query_vector, candidate_episodes, top_k * 2)
    fused = _rrf_fuse(bm25_ranks, vec_ranks, k=rrf_k, bm25_weight=bm25_weight, vec_weight=vec_weight)

    results: list[tuple[HmEpisode, float]] = []
    for doc_id, score in fused[:top_k]:
        if doc_id in episode_map:
            results.append((episode_map[doc_id], score))
    logger.info("片段检索完成: 候选片段=%d, 返回 %d 个片段", len(candidate_episodes), len(results))
    return results


def retrieve_facts(
    query: str,
    query_vector: np.ndarray | None,
    episode_results: list[tuple[HmEpisode, float]],
    all_facts: Sequence[HmFact],
    top_k: int = 15,
    rrf_k: int = _DEFAULT_RRF_K,
    bm25_weight: float = 1.0,
    vec_weight: float = 1.0,
    apply_temporal_decay: bool = False,
) -> list[tuple[HmFact, float]]:
    """Level 3: From selected episodes, retrieve top-k facts via BM25+Vector RRF."""
    logger.info("开始检索事实: query='%s', 来源片段数=%d, 全部事实数=%d, top_k=%d, 时间衰减=%s",
                query[:80], len(episode_results), len(all_facts), top_k, apply_temporal_decay)
    candidate_fact_episode_ids: set[str] = set()
    for episode, _ in episode_results:
        candidate_fact_episode_ids.add(episode.id)

    fact_map = {f.id: f for f in all_facts}
    candidate_facts = [f for f in all_facts if f.episode_id in candidate_fact_episode_ids]

    if not candidate_facts:
        logger.debug("片段关联事实为空, 使用全部事实作为候选")
        candidate_facts = list(all_facts)

    if not candidate_facts:
        logger.info("事实检索跳过: 无可用事实")
        return []

    query_tokens = set(tokenize(query))
    bm25_ranks = _rank_by_bm25(
        query_tokens, candidate_facts,
        lambda f: f"{f.content} {f.potential} {' '.join(f.keywords)}",
        top_k * 2,
    )
    vec_ranks = _rank_by_vector(query_vector, candidate_facts, top_k * 2)
    fused = _rrf_fuse(bm25_ranks, vec_ranks, k=rrf_k, bm25_weight=bm25_weight, vec_weight=vec_weight)

    # Apply temporal decay if requested (temporal queries benefit from recency)
    if apply_temporal_decay:
        now_ms = int(time.time() * 1000)
        fused = [
            (doc_id, score * _temporal_decay(fact_map[doc_id].created_at, now_ms))
            for doc_id, score in fused
            if doc_id in fact_map
        ]
        fused.sort(key=lambda x: x[1], reverse=True)

    results: list[tuple[HmFact, float]] = []
    for doc_id, score in fused[:top_k]:
        if doc_id in fact_map:
            results.append((fact_map[doc_id], score))
    logger.info("事实检索完成: 候选事实=%d, 返回 %d 个事实", len(candidate_facts), len(results))
    return results


def coarse_to_fine_retrieval(
    query: str,
    query_vector: np.ndarray | None,
    topics: Sequence[HmTopic],
    episodes: Sequence[HmEpisode],
    facts: Sequence[HmFact],
    topic_k: int = 5,
    episode_k: int = 5,
    fact_k: int = 15,
    intent: str = "",
) -> dict[str, list]:
    """Full C2F pipeline: Topic → Episode → Fact.

    Intent-adaptive: adjusts RRF-k and BM25/vec weights per intent.
    """
    logger.info("开始粗到细检索: query='%s', intent=%s, 主题数=%d, 片段数=%d, 事实数=%d",
                query[:80], intent or "default", len(topics), len(episodes), len(facts))
    rrf_k = _RRF_K_BY_INTENT.get(intent, _DEFAULT_RRF_K)

    # For temporal queries, boost BM25 (keyword/date matching)
    if intent == "temporal":
        bm25_w, vec_w = 1.5, 0.8
    elif intent == "opinion":
        bm25_w, vec_w = 0.8, 1.2
    else:
        bm25_w, vec_w = 1.0, 1.0

    topic_results = retrieve_topics(query, query_vector, topics, topic_k, rrf_k)
    episode_results = retrieve_episodes(
        query, query_vector, topic_results, episodes, episode_k,
        rrf_k, bm25_w, vec_w,
    )
    fact_results = retrieve_facts(
        query, query_vector, episode_results, facts, fact_k,
        rrf_k, bm25_w, vec_w,
        apply_temporal_decay=(intent == "temporal"),
    )

    logger.info("粗到细检索完成: 主题=%d, 片段=%d, 事实=%d",
                len(topic_results), len(episode_results), len(fact_results))
    return {
        "topics": topic_results,
        "episodes": episode_results,
        "facts": fact_results,
    }


def c2f_to_recall_hits(
    fact_results: list[tuple[HmFact, float]],
    episode_results: list[tuple[HmEpisode, float]],
) -> list[RecallHit]:
    """Convert C2F results to RecallHit format for context assembly."""
    logger.debug("转换 C2F 结果为 RecallHit: 事实数=%d, 片段数=%d", len(fact_results), len(episode_results))
    hits: list[RecallHit] = []
    for fact, score in fact_results:
        hits.append(RecallHit(
            id=fact.id,
            title=f"[{fact.importance.upper()}] {fact.content[:80]}",
            content=fact.content,
            source="hm_fact",
            score=score,
            reason=f"potential: {fact.potential}" if fact.potential else "atomic fact",
            session_key=fact.session_key,
            turn_index=fact.source_turn_start,
        ))
    return hits
