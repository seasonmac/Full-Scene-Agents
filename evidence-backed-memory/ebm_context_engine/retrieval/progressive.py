"""Progressive Recaller: 3-layer orthogonal retrieval (Topic → Episode → Fact).

Replaces the previous 5-way parallel recall (graph + ledger + summary + c2f + transcript)
with a single progressive pipeline that operates on the unified data model:
  Layer 0: HmTopic (includes community-sourced topics)
  Layer 1: HmEpisode (includes session-summary episodes)
  Layer 2: UnifiedFact (merged LedgerFact + HmFact)
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import numpy as np

from ebm_context_engine.client import cosine_similarity
from ebm_context_engine.hypergraph.c2f_retriever import (
    _bm25_score,
    _rank_by_bm25,
    _rank_by_vector,
    _rrf_fuse,
    _temporal_decay,
    _DEFAULT_RRF_K,
    _RRF_K_BY_INTENT,
)
from ebm_context_engine.text import tokenize
from ebm_context_engine.types import (
    ClassificationResult,
    EvidenceRef,
    HmEpisode,
    HmTopic,
    MemoryState,
    RecallHit,
    UnifiedFact,
)

logger = logging.getLogger("ebm_context_engine.retrieval.progressive")

# Importance multiplier for UnifiedFact scoring
_IMPORTANCE_BOOST = {"high": 1.3, "mid": 1.0, "low": 0.8}


class ProgressiveRecaller:
    """Three-layer progressive retrieval: Topic → Episode → UnifiedFact."""

    def __init__(
        self,
        state: MemoryState,
        embed_fn=None,
        rerank_fn=None,
        *,
        layer0_top_k: int = 5,
        layer1_top_k: int = 8,
        layer2_top_k: int = 6,
        layer0_threshold: float = 0.0,
        layer1_threshold: float = 0.0,
    ) -> None:
        self._state = state
        self._embed_fn = embed_fn
        self._rerank_fn = rerank_fn
        self._layer0_top_k = layer0_top_k
        self._layer1_top_k = layer1_top_k
        self._layer2_top_k = layer2_top_k
        self._layer0_threshold = layer0_threshold
        self._layer1_threshold = layer1_threshold

    def recall(
        self,
        question: str,
        query_vector: Optional[np.ndarray],
        classification: ClassificationResult,
        entity_seed_ids: Optional[set[str]] = None,
    ) -> list[RecallHit]:
        """Run 3-layer progressive retrieval, return final RecallHits."""
        intent = classification.intent
        rrf_k = _RRF_K_BY_INTENT.get(intent, _DEFAULT_RRF_K)
        is_temporal = intent == "temporal"
        bm25_w = 1.5 if is_temporal else 1.0
        vec_w = 0.8 if is_temporal else 1.0

        logger.info("开始三层渐进式检索: intent=%s, rrf_k=%d, is_temporal=%s, question=%s",
                     intent, rrf_k, is_temporal, question[:100])
        logger.debug("检索权重: bm25_w=%.2f, vec_w=%.2f, entity_seed_ids=%s",
                      bm25_w, vec_w, len(entity_seed_ids) if entity_seed_ids else 0)

        # Layer 0: Topic matching
        t0 = time.perf_counter()
        topic_hits = self._recall_topics(question, query_vector, rrf_k)
        t1 = time.perf_counter()

        # Layer 1: Episode matching (scoped to Layer 0 topics)
        episode_hits = self._recall_episodes(
            question, query_vector, topic_hits, rrf_k, bm25_w, vec_w,
        )
        t2 = time.perf_counter()

        # Layer 2: UnifiedFact matching (scoped to Layer 1 episodes)
        fact_hits = self._recall_facts(
            question, query_vector, episode_hits,
            rrf_k, bm25_w, vec_w,
            apply_temporal_decay=is_temporal,
            entity_seed_ids=entity_seed_ids,
        )
        t3 = time.perf_counter()

        # Flat vector supplement: search ALL facts directly by vector similarity
        # to catch facts in topics/episodes that didn't make top-K
        flat_hits = self._flat_vector_recall(question, query_vector, fact_hits)
        t4 = time.perf_counter()

        # Merge: keep the best from both sources
        # Replace the weakest hierarchical hits with flat supplement hits
        if flat_hits:
            seen_ids = {h.id for h in fact_hits}
            new_flat = [h for h in flat_hits if h.id not in seen_ids]
            if new_flat:
                # Insert flat hits, then trim to keep total manageable
                # Keep all hierarchical hits + flat supplement, let contextHitsLimit do final cut
                merged = list(fact_hits) + new_flat
            else:
                merged = list(fact_hits)
        else:
            merged = list(fact_hits)

        logger.info(
            "渐进式检索完成: 主题=%d, 片段=%d, 事实=%d(+%d flat), "
            "耗时=%.1fms/%.1fms/%.1fms/%.1fms",
            len(topic_hits), len(episode_hits), len(fact_hits), len(merged) - len(fact_hits),
            (t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000, (t4 - t3) * 1000,
        )
        return merged

    def _recall_topics(
        self,
        question: str,
        query_vector: Optional[np.ndarray],
        rrf_k: int,
    ) -> list[tuple[HmTopic, float]]:
        """Layer 0: Retrieve top-K topics from state.hm_topics."""
        topics = list(self._state.hm_topics.values())
        logger.info("Layer0-主题检索开始: 候选主题数=%d, top_k=%d, rrf_k=%d", len(topics), self._layer0_top_k, rrf_k)
        if not topics:
            logger.debug("Layer0-主题检索: 无可用主题, 返回空列表")
            return []

        query_tokens = set(tokenize(question))
        bm25_ranks = _rank_by_bm25(
            query_tokens, topics,
            lambda t: f"{t.title} {t.summary} {' '.join(t.keywords)}",
            self._layer0_top_k * 2,
        )
        vec_ranks = _rank_by_vector(query_vector, topics, self._layer0_top_k * 2)
        fused = _rrf_fuse(bm25_ranks, vec_ranks, k=rrf_k)

        topic_map = {t.id: t for t in topics}
        results: list[tuple[HmTopic, float]] = []
        for doc_id, score in fused[:self._layer0_top_k]:
            if score < self._layer0_threshold:
                break
            if doc_id in topic_map:
                results.append((topic_map[doc_id], score))
        logger.debug("Layer0-主题检索结果: 命中=%d, 主题=%s",
                      len(results), [(t.title[:30], f"{s:.4f}") for t, s in results])
        return results

    def _recall_episodes(
        self,
        question: str,
        query_vector: Optional[np.ndarray],
        topic_hits: list[tuple[HmTopic, float]],
        rrf_k: int,
        bm25_w: float,
        vec_w: float,
    ) -> list[tuple[HmEpisode, float]]:
        """Layer 1: Retrieve episodes scoped to Layer 0 topic hits."""
        all_episodes = self._state.hm_episodes
        logger.info("Layer1-片段检索开始: 总片段数=%d, 上层主题命中=%d, top_k=%d",
                     len(all_episodes) if all_episodes else 0, len(topic_hits), self._layer1_top_k)
        if not all_episodes:
            logger.debug("Layer1-片段检索: 无可用片段, 返回空列表")
            return []

        # Collect candidate episode IDs from matched topics
        candidate_ids: set[str] = set()
        for topic, _ in topic_hits:
            candidate_ids.update(topic.episode_ids)

        candidates = [all_episodes[eid] for eid in candidate_ids if eid in all_episodes]

        # Fallback: if no candidates from topics, use all episodes
        if not candidates:
            logger.debug("Layer1-片段检索: 主题未关联片段, 回退使用全部片段")
            candidates = list(all_episodes.values())

        logger.debug("Layer1-片段候选数=%d, 候选ID来源主题数=%d", len(candidates), len(candidate_ids))

        query_tokens = set(tokenize(question))
        bm25_ranks = _rank_by_bm25(
            query_tokens, candidates,
            lambda e: f"{e.title} {e.summary} {' '.join(e.keywords)} {e.dialogue[:300]}",
            self._layer1_top_k * 2,
        )
        vec_ranks = _rank_by_vector(query_vector, candidates, self._layer1_top_k * 2)
        fused = _rrf_fuse(bm25_ranks, vec_ranks, k=rrf_k, bm25_weight=bm25_w, vec_weight=vec_w)

        episode_map = {e.id: e for e in candidates}
        results: list[tuple[HmEpisode, float]] = []
        for doc_id, score in fused[:self._layer1_top_k]:
            if score < self._layer1_threshold:
                break
            if doc_id in episode_map:
                results.append((episode_map[doc_id], score))
        logger.debug("Layer1-片段检索结果: 命中=%d, 片段=%s",
                      len(results), [(e.title[:30], f"{s:.4f}") for e, s in results])
        return results

    def _recall_facts(
        self,
        question: str,
        query_vector: Optional[np.ndarray],
        episode_hits: list[tuple[HmEpisode, float]],
        rrf_k: int,
        bm25_w: float,
        vec_w: float,
        apply_temporal_decay: bool = False,
        entity_seed_ids: Optional[set[str]] = None,
    ) -> list[RecallHit]:
        """Layer 2: Retrieve UnifiedFacts scoped to Layer 1 episode hits."""
        all_facts = self._state.unified_facts
        logger.info("Layer2-事实检索开始: 总事实数=%d, 上层片段命中=%d, top_k=%d, temporal_decay=%s",
                     len(all_facts) if all_facts else 0, len(episode_hits), self._layer2_top_k, apply_temporal_decay)
        if not all_facts:
            logger.debug("Layer2-事实检索: 无可用事实, 返回空列表")
            return []

        # Collect candidate fact IDs from matched episodes
        candidate_episode_ids: set[str] = set()
        for episode, _ in episode_hits:
            candidate_episode_ids.add(episode.id)

        candidates = [f for f in all_facts.values() if f.episode_id in candidate_episode_ids]

        # Fallback: if no candidates from episodes, use all facts
        if not candidates:
            logger.debug("Layer2-事实检索: 片段未关联事实, 回退使用全部事实")
            candidates = list(all_facts.values())

        # Filter out inactive facts
        pre_filter_count = len(candidates)
        candidates = [f for f in candidates if f.status == "active"]
        logger.debug("Layer2-事实候选: 片段关联=%d, 过滤inactive后=%d", pre_filter_count, len(candidates))

        if not candidates:
            return []

        query_tokens = set(tokenize(question))
        bm25_ranks = _rank_by_bm25(
            query_tokens, candidates,
            lambda f: f"{f.content} {f.potential} {' '.join(f.keywords)}",
            self._layer2_top_k * 3,
        )
        vec_ranks = _rank_by_vector(query_vector, candidates, self._layer2_top_k * 3)
        fused = _rrf_fuse(bm25_ranks, vec_ranks, k=rrf_k, bm25_weight=bm25_w, vec_weight=vec_w)

        # Optional rerank with fallback (like OpenViking's hierarchical_retriever)
        if self._rerank_fn and fused:
            try:
                fact_map_tmp = {f.id: f for f in candidates}
                top_ids = [doc_id for doc_id, _ in fused[:self._layer2_top_k * 3]]
                docs = []
                doc_idx_to_top_idx: list[int] = []
                for i, did in enumerate(top_ids):
                    if did in fact_map_tmp:
                        docs.append(fact_map_tmp[did].content)
                        doc_idx_to_top_idx.append(i)
                rerank_results = self._rerank_fn(question, docs, top_n=self._layer2_top_k * 2)
                reranked: list[tuple[str, float]] = []
                for idx, score in rerank_results:
                    if idx < len(doc_idx_to_top_idx):
                        reranked.append((top_ids[doc_idx_to_top_idx[idx]], score))
                if reranked:
                    fused = reranked
                    logger.info("Layer2-Rerank成功: %d条结果", len(reranked))
            except Exception as e:
                logger.warning("Layer2-Rerank失败, 回退RRF: %s", e)

        # Apply confidence, importance, temporal decay, and entity seed boosts
        fact_map = {f.id: f for f in candidates}
        now_ms = int(time.time() * 1000) if apply_temporal_decay else 0
        boosted: list[tuple[str, float]] = []
        for doc_id, score in fused:
            fact = fact_map.get(doc_id)
            if fact is None:
                continue
            s = score
            # Confidence boost
            s *= (0.5 + fact.confidence * 0.5)
            # Importance boost
            s *= _IMPORTANCE_BOOST.get(fact.importance, 1.0)
            # Temporal decay
            if apply_temporal_decay and fact.created_at:
                s *= _temporal_decay(fact.created_at, now_ms)
            # Entity seed boost: facts linked to query entities get a bonus
            if entity_seed_ids and fact.entity_ids:
                if set(fact.entity_ids) & entity_seed_ids:
                    s *= 1.4
            boosted.append((doc_id, s))

        boosted.sort(key=lambda x: x[1], reverse=True)
        logger.debug("Layer2-事实融合排序完成: 融合候选数=%d, 取top_k=%d", len(boosted), self._layer2_top_k)

        # Convert to RecallHit
        results: list[RecallHit] = []
        for doc_id, score in boosted[:self._layer2_top_k]:
            fact = fact_map[doc_id]
            results.append(RecallHit(
                id=fact.id,
                title=f"[{fact.importance.upper()}] {fact.content[:80]}",
                content=fact.content,
                source="progressive",
                score=score,
                reason=f"potential: {fact.potential}" if fact.potential else f"key: {fact.key}" if fact.key else "unified fact",
                evidence=fact.evidence,
                session_key=fact.session_key,
                turn_index=fact.turn_index or fact.source_turn_start,
                verified=fact.source.endswith("-verified") if fact.source else None,
            ))
        logger.debug("Layer2-事实检索结果: 命中=%d, 事实=%s",
                      len(results), [(r.content[:40], f"{r.score:.4f}") for r in results])
        return results

    def _flat_vector_recall(
        self,
        question: str,
        query_vector: np.ndarray | None,
        hierarchical_hits: list[RecallHit],
        *,
        top_k: int = 3,
    ) -> list[RecallHit]:
        """Flat vector search across ALL unified facts to supplement hierarchical recall.

        Catches relevant facts in topics/episodes that didn't make the top-K cut
        in the hierarchical pipeline.
        """
        if query_vector is None:
            return []
        all_facts = self._state.unified_facts
        if not all_facts:
            return []

        # Skip IDs already found by hierarchical search
        seen_ids = {h.id for h in hierarchical_hits}

        # Score all active facts by vector similarity
        scored: list[tuple[UnifiedFact, float]] = []
        for fact in all_facts.values():
            if fact.status != "active" or fact.id in seen_ids:
                continue
            if fact.vector is None:
                continue
            sim = float(cosine_similarity(query_vector, fact.vector))
            scored.append((fact, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[RecallHit] = []
        for fact, sim in scored[:top_k]:
            results.append(RecallHit(
                id=fact.id,
                title=f"[{fact.importance.upper()}] {fact.content[:80]}",
                content=fact.content,
                source="progressive-flat",
                score=sim * 0.8,  # slightly discount flat results vs hierarchical
                reason="flat vector supplement",
                evidence=fact.evidence,
                session_key=fact.session_key,
                turn_index=fact.turn_index or fact.source_turn_start,
                verified=fact.source.endswith("-verified") if fact.source else None,
            ))
        if results:
            logger.info("Flat向量补充: 候选=%d, 命中=%d, top_sim=%.4f",
                        len(scored), len(results), scored[0][1] if scored else 0)
        return results


__all__ = ["ProgressiveRecaller"]
