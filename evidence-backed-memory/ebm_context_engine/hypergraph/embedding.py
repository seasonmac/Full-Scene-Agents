"""Hypergraph Embedding Propagation.

Implements h'_v = h_v + λ·Agg(h_e) where:
- h_v is the node's original embedding
- h_e are embeddings of connected nodes via hyperedges
- Agg is mean aggregation
- λ controls propagation strength
"""
from __future__ import annotations

import logging

import numpy as np

from ebm_context_engine.client import normalize_vector
from ebm_context_engine.types import HmEpisode, HmFact, HmTopic

DEFAULT_LAMBDA = 0.3

logger = logging.getLogger("ebm_context_engine.hypergraph.embedding")


def _aligned_vectors(*vectors: np.ndarray | None) -> list[np.ndarray]:
    """Trim vectors to a shared dimensionality for safe aggregation.

    EBM can mix embeddings from the primary endpoint and a fallback endpoint,
    and those endpoints may legitimately use different dimensions. Retrieval
    already compares vectors on their shared prefix; propagation should do the
    same instead of crashing on shape mismatch.
    """
    arrays = [np.asarray(vector, dtype=np.float32).ravel() for vector in vectors if vector is not None]
    if not arrays:
        return []
    dim = min(array.size for array in arrays)
    if dim <= 0:
        return []
    dims = {array.size for array in arrays}
    if len(dims) > 1:
        logger.warning(
            "HyperMem propagation aligned mixed embedding dimensions: original_dims=%s target_dim=%d",
            sorted(dims),
            dim,
        )
    return [array[:dim] for array in arrays]


def _combine_vectors(base: np.ndarray, neighbor: np.ndarray, weight: float) -> np.ndarray:
    aligned = _aligned_vectors(base, neighbor)
    if len(aligned) != 2:
        return normalize_vector(base)
    base_aligned, neighbor_aligned = aligned
    return normalize_vector(base_aligned + weight * neighbor_aligned)


def _mean_aligned(vectors: list[np.ndarray]) -> np.ndarray | None:
    aligned = _aligned_vectors(*vectors)
    if not aligned:
        return None
    return np.mean(aligned, axis=0)


def propagate_embeddings(
    topics: list[HmTopic],
    episodes: list[HmEpisode],
    facts: list[HmFact],
    lam: float = DEFAULT_LAMBDA,
) -> None:
    """Propagate embeddings through the hypergraph in-place.

    Flow: Topic ←→ Episode ←→ Fact
    Each node gets updated: h'_v = h_v + λ·mean(connected node embeddings)
    """
    episode_map = {e.id: e for e in episodes}
    topic_map = {t.id: t for t in topics}

    # 1. Episode → Fact propagation
    # Each fact gets influence from its parent episode
    for fact in facts:
        ep = episode_map.get(fact.episode_id)
        if fact.vector is not None and ep is not None and ep.vector is not None:
            fact.vector = _combine_vectors(fact.vector, ep.vector, lam)

    # 2. Topic → Episode propagation
    # Each episode gets influence from its parent topic(s)
    for episode in episodes:
        if episode.vector is None:
            continue
        neighbor_vecs: list[np.ndarray] = []
        for tid in episode.topic_ids:
            topic = topic_map.get(tid)
            if topic is not None and topic.vector is not None:
                neighbor_vecs.append(topic.vector)
        if neighbor_vecs:
            agg = _mean_aligned(neighbor_vecs)
            if agg is not None:
                episode.vector = _combine_vectors(episode.vector, agg, lam)

    # 3. Episode → Topic propagation (reverse)
    # Each topic gets influence from its member episodes
    for topic in topics:
        if topic.vector is None:
            continue
        neighbor_vecs: list[np.ndarray] = []
        for eid in topic.episode_ids:
            ep = episode_map.get(eid)
            if ep is not None and ep.vector is not None:
                neighbor_vecs.append(ep.vector)
        if neighbor_vecs:
            agg = _mean_aligned(neighbor_vecs)
            if agg is not None:
                topic.vector = _combine_vectors(topic.vector, agg, lam)

    # 4. Fact → Episode propagation (reverse)
    # Each episode gets influence from its child facts
    ep_fact_vecs: dict[str, list[np.ndarray]] = {}
    for fact in facts:
        if fact.vector is not None and fact.episode_id:
            ep_fact_vecs.setdefault(fact.episode_id, []).append(fact.vector)

    for episode in episodes:
        if episode.vector is None:
            continue
        child_vecs = ep_fact_vecs.get(episode.id)
        if child_vecs:
            agg = _mean_aligned(child_vecs)
            if agg is not None:
                episode.vector = _combine_vectors(episode.vector, agg, lam * 0.5)
