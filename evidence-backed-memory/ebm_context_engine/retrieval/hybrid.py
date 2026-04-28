from __future__ import annotations

import logging
from typing import Callable, Sequence, TypeVar

from ebm_context_engine.client import cosine_similarity
from ebm_context_engine.text import contains_temporal_marker, keyword_overlap, tokenize

logger = logging.getLogger("ebm_context_engine.retrieval.hybrid")


def build_fts_query(query: str) -> str:
    logger.debug("构建全文搜索查询: query=%s", query[:80])
    tokens = tokenize(query)
    if not tokens:
        logger.debug("分词结果为空, 使用默认查询 'memory'")
        return '"memory"'
    fts = " AND ".join(f'"{token.replace(chr(34), "")}"' for token in tokens[:8])
    logger.debug("全文搜索查询构建完成: fts_query=%s", fts[:120])
    return fts


T = TypeVar("T")


def reciprocal_rank_fusion(
    items: Sequence[T],
    score_fns: Sequence[Callable[[T], float]],
    rrf_k: int = 60,
) -> list[tuple[T, float]]:
    logger.info("开始RRF融合排序: 候选数=%d, 评分函数数=%d, rrf_k=%d", len(items), len(score_fns), rrf_k)
    rankings: list[dict[int, int]] = []
    for score_fn in score_fns:
        indexed = sorted(
            ((index, float(score_fn(item))) for index, item in enumerate(items)),
            key=lambda item: item[1],
            reverse=True,
        )
        rankings.append({index: rank + 1 for rank, (index, _) in enumerate(indexed)})

    fused: list[tuple[T, float]] = []
    for index, item in enumerate(items):
        rrf_score = 0.0
        for ranking in rankings:
            rank = ranking.get(index, len(items))
            rrf_score += 1.0 / (rrf_k + rank)
        fused.append((item, rrf_score))
    fused.sort(key=lambda item: item[1], reverse=True)
    logger.debug("RRF融合排序完成: 结果数=%d, Top3分数=%s",
                  len(fused), [f"{s:.6f}" for _, s in fused[:3]])
    return fused


def temporal_bonus(text: str, intent_weights: dict[str, float] | None) -> float:
    if not intent_weights:
        return 1.0
    bonus = float(intent_weights.get("temporalFactBonus", 1.0))
    if bonus <= 1.0:
        return 1.0
    return bonus if contains_temporal_marker(text) else 1.0


def rank_graph_nodes(
    query: str,
    nodes: Sequence[object],
    *,
    query_vector,
    weights: object = None,
    intent_weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> list[tuple[object, float]]:
    logger.info("开始图节点排序: 节点数=%d, 使用意图权重=%s, rrf_k=%d", len(nodes), intent_weights is not None, rrf_k)
    if not nodes:
        logger.debug("图节点排序: 无候选节点, 返回空列表")
        return []
    if intent_weights:
        fused = reciprocal_rank_fusion(
            nodes,
            [
                lambda node: cosine_similarity(query_vector, getattr(node, "vector", None)),
                lambda node: keyword_overlap(tokenize(query), tokenize(f"{getattr(node, 'label', '')} {getattr(node, 'content', '')} {' '.join(getattr(node, 'keywords', []))}")),
            ],
            rrf_k=rrf_k,
        )
        return [(node, score * temporal_bonus(getattr(node, "content", ""), intent_weights)) for node, score in fused]

    # Use config weights if provided, otherwise fallback defaults
    vector_weight = getattr(weights, "vectorWeight", None) or getattr(weights, "vector_weight", None) or 0.7 if weights else 0.7
    bm25_weight = getattr(weights, "bm25Weight", None) or getattr(weights, "bm25_weight", None) or 0.3 if weights else 0.3
    logger.debug("图节点排序使用加权模式: vector_weight=%.2f, bm25_weight=%.2f", vector_weight, bm25_weight)
    ranked = []
    query_tokens = tokenize(query)
    for node in nodes:
        lexical = keyword_overlap(query_tokens, tokenize(f"{getattr(node, 'label', '')} {getattr(node, 'content', '')} {' '.join(getattr(node, 'keywords', []))}"))
        semantic = cosine_similarity(query_vector, getattr(node, "vector", None))
        ranked.append((node, vector_weight * semantic + bm25_weight * lexical))
    ranked.sort(key=lambda item: item[1], reverse=True)
    logger.debug("图节点加权排序完成: 结果数=%d, Top3分数=%s",
                  len(ranked), [f"{s:.4f}" for _, s in ranked[:3]])
    return ranked


def rank_text_records(
    query: str,
    records: Sequence[object],
    *,
    query_vector,
    get_text: Callable[[object], str],
    get_vector: Callable[[object], object],
    weights: object = None,
    intent_weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> list[tuple[object, float]]:
    logger.info("开始文本记录排序: 记录数=%d, 使用意图权重=%s, rrf_k=%d", len(records), intent_weights is not None, rrf_k)
    if not records:
        logger.debug("文本记录排序: 无候选记录, 返回空列表")
        return []
    if intent_weights:
        fused = reciprocal_rank_fusion(
            records,
            [
                lambda record: cosine_similarity(query_vector, get_vector(record)),
                lambda record: keyword_overlap(tokenize(query), tokenize(get_text(record))),
            ],
            rrf_k=rrf_k,
        )
        return [(record, score * temporal_bonus(get_text(record), intent_weights)) for record, score in fused]

    # Use config weights if provided, otherwise fallback defaults
    vector_weight = getattr(weights, "vectorWeight", None) or getattr(weights, "vector_weight", None) or 0.7 if weights else 0.7
    bm25_weight = getattr(weights, "bm25Weight", None) or getattr(weights, "bm25_weight", None) or 0.3 if weights else 0.3
    logger.debug("文本记录排序使用加权模式: vector_weight=%.2f, bm25_weight=%.2f", vector_weight, bm25_weight)
    ranked = []
    query_tokens = tokenize(query)
    for record in records:
        lexical = keyword_overlap(query_tokens, tokenize(get_text(record)))
        semantic = cosine_similarity(query_vector, get_vector(record))
        ranked.append((record, vector_weight * semantic + bm25_weight * lexical))
    ranked.sort(key=lambda item: item[1], reverse=True)
    logger.debug("文本记录加权排序完成: 结果数=%d, Top3分数=%s",
                  len(ranked), [f"{s:.4f}" for _, s in ranked[:3]])
    return ranked


__all__ = [
    "build_fts_query",
    "rank_graph_nodes",
    "rank_text_records",
    "reciprocal_rank_fusion",
    "temporal_bonus",
]
