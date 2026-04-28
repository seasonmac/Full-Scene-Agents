from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

from ebm_context_engine.types import GraphEdgeRecord

logger = logging.getLogger("ebm_context_engine.retrieval.ppr")

CAUSAL_EDGE_TYPES = {"causes", "prevents", "enables", "triggers"}
TEMPORAL_EDGE_TYPES = {"temporal"}
ENTITY_EDGE_TYPES = {"related_to", "has_attribute", "participates_in"}


def _edge_boost(edge_type: str, intent_weights: dict[str, float] | None) -> float:
    if not intent_weights:
        return 1.0
    if edge_type in CAUSAL_EDGE_TYPES:
        return float(intent_weights.get("causalEdgeBoost", 1.0))
    if edge_type in TEMPORAL_EDGE_TYPES:
        return float(intent_weights.get("temporalEdgeBoost", 1.0))
    if edge_type in ENTITY_EDGE_TYPES:
        return float(intent_weights.get("entityEdgeBoost", 1.0))
    return 1.0


def personalized_page_rank(
    seed_weights: dict[str, float],
    edges: Iterable[GraphEdgeRecord],
    *,
    iterations: int | None = None,
    damping: float = 0.85,
    intent_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    logger.info("开始个性化PageRank: seed数量=%d, damping=%.2f", len(seed_weights), damping)
    logger.debug("种子节点权重: %s", {k: f"{v:.4f}" for k, v in list(seed_weights.items())[:10]})
    nodes: set[str] = set(seed_weights)
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    edge_count = 0
    for edge in edges:
        nodes.add(edge.from_id)
        nodes.add(edge.to_id)
        boosted = float(edge.weight) * _edge_boost(edge.edge_type, intent_weights)
        adjacency[edge.from_id].append((edge.to_id, boosted))
        adjacency[edge.to_id].append((edge.from_id, boosted))
        edge_count += 1

    logger.debug("图构建完成: 节点数=%d, 边数=%d", len(nodes), edge_count)

    restart_total = sum(seed_weights.values()) or 1.0
    scores = {node: float(seed_weights.get(node, 0.0)) for node in nodes}
    total_iterations = int(intent_weights.get("pprIterations", iterations or 8)) if intent_weights else int(iterations or 8)

    logger.debug("PPR迭代参数: 总迭代次数=%d, restart_total=%.4f", total_iterations, restart_total)

    for _ in range(max(1, total_iterations)):
        next_scores = {
            node: ((1.0 - damping) * float(seed_weights.get(node, 0.0))) / restart_total
            for node in nodes
        }
        for node in nodes:
            outgoing = adjacency.get(node, [])
            if not outgoing:
                continue
            current = scores.get(node, 0.0)
            total_weight = sum(weight for _, weight in outgoing) or 1.0
            for target, weight in outgoing:
                next_scores[target] = next_scores.get(target, 0.0) + damping * current * (weight / total_weight)
        # Dangling node handling: redistribute scores from nodes with no outgoing edges back to seeds
        dangling_sum = sum(
            scores.get(node, 0.0) for node in nodes
            if not adjacency.get(node)
        )
        if dangling_sum > 0:
            for seed_node, seed_weight in seed_weights.items():
                next_scores[seed_node] = next_scores.get(seed_node, 0.0) + damping * dangling_sum * (seed_weight / restart_total)
        scores = next_scores
    top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.debug("PPR收敛后Top5节点: %s", [(nid, f"{sc:.6f}") for nid, sc in top_scores])
    logger.info("个性化PageRank完成: 节点数=%d, 迭代次数=%d, 非零得分节点=%d",
                len(nodes), total_iterations, sum(1 for v in scores.values() if v > 0))
    return scores


__all__ = ["personalized_page_rank"]
