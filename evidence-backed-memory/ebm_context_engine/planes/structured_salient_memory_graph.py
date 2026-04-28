from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

logger = logging.getLogger("ebm_context_engine.planes.structured_salient_memory_graph")

from ebm_context_engine.core.hash import stableId
from ebm_context_engine.client import cosine_similarity
from ebm_context_engine.core.text import topKeywords
from ebm_context_engine.core.vector import embedText
from ebm_context_engine.retrieval.hybrid import rank_graph_nodes, rank_text_records
from ebm_context_engine.retrieval.ppr import personalized_page_rank
from ebm_context_engine.text import tokenize, top_keywords
from ebm_context_engine.types import RecallHit


@dataclass
class _GraphNodeView:
    id: str
    nodeType: str
    label: str
    content: str
    keywords: list[str]
    vector: object
    confidence: float
    evidence: object = None
    mention_count: int = 0


# ── Tuning constants (matching TS StructuredSalientMemoryGraph) ──────────────────
DEFAULT_MAX_HOPS = 2
DEFAULT_ENTITY_SEED_SIM_THRESHOLD = 0.6
DEFAULT_GENERALIZED_VEC_SIM_THRESHOLD = 0.4
DEFAULT_MULTI_HOP_BONUS = 1.5
DEFAULT_REBUILD_CHANGE_THRESHOLD = 7
DEFAULT_RRF_K = 60
DEFAULT_PPR_ITERATIONS = 8
DEFAULT_PPR_DAMPING = 0.85


def _salient_memory_label_from_error(text: str) -> str:
    return f"Resolve {text[:72]}"


def _dedupe_nodes(nodes: list[_GraphNodeView]) -> list[_GraphNodeView]:
    by_id: dict[str, _GraphNodeView] = {}
    for node in nodes:
        if node.id not in by_id:
            by_id[node.id] = node
    return list(by_id.values())


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


class StructuredSalientMemoryGraph:
    def __init__(
        self,
        state_getter: Callable[[], object],
        *,
        generalized_recall_discount: float = 0.8,
        embed_fn: Optional[Callable] = None,
        max_hops: int = DEFAULT_MAX_HOPS,
        entity_seed_sim_threshold: float = DEFAULT_ENTITY_SEED_SIM_THRESHOLD,
        generalized_vec_sim_threshold: float = DEFAULT_GENERALIZED_VEC_SIM_THRESHOLD,
        multi_hop_bonus: float = DEFAULT_MULTI_HOP_BONUS,
        rebuild_change_threshold: int = DEFAULT_REBUILD_CHANGE_THRESHOLD,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self._state_getter = state_getter
        self.generalized_recall_discount = generalized_recall_discount
        self._embed_fn = embed_fn
        self._max_hops = max_hops
        self._entity_seed_sim_threshold = entity_seed_sim_threshold
        self._generalized_vec_sim_threshold = generalized_vec_sim_threshold
        self._multi_hop_bonus = multi_hop_bonus
        self._rebuild_change_threshold = rebuild_change_threshold
        self._rrf_k = rrf_k

        # Concurrent rebuild protection (matching TS mutex pattern)
        self._rebuild_lock = threading.Lock()
        self._rebuild_in_progress = False
        self._rebuild_needed = False
        self._pending_graph_changes = 0

    def _state(self):
        return self._state_getter()

    # ── distillTurn (matching TS plane-b.ts:308-474) ─────────────────────
    def distillTurn(self, params: dict) -> dict:
        """Extract TASK/EVENT/SALIENT_MEMORY/FACT graph nodes and edges from a conversation turn.
        Matches TS StructuredSalientMemoryGraph.distillTurn."""
        state = self._state()
        session_id = str(params.get("sessionId", "") or "")
        session_file = str(params.get("sessionFile", "") or "")
        query = str(params.get("query", "") or "")
        turn_texts: list[str] = list(params.get("turnMessagesText", []) or [])
        turn_indexes: list[int] = list(params.get("turnMessageIndexes", []) or [])
        start_index = int(params.get("startIndex", 0) or 0)
        logger.info("开始蒸馏对话轮次: session_id=%s, 轮次文本数=%d, query长度=%d", session_id, len(turn_texts), len(query))

        nodes: list[dict] = []
        edges: list[dict] = []

        task_message_index = turn_indexes[0] if turn_indexes else start_index
        task_node_id = stableId("TASK", session_id, query)
        task_vector = embedText(query, self._embed_fn)
        task_node = {
            "id": task_node_id,
            "nodeType": "TASK",
            "label": query[:120],
            "content": query,
            "keywords": top_keywords([query]),
            "vector": task_vector,
            "confidence": 0.9,
            "evidence": {
                "sessionFile": session_file,
                "messageIndex": task_message_index,
                "snippet": query,
            },
        }
        nodes.append(task_node)

        previous_event: Optional[dict] = None
        for index, text in enumerate(turn_texts):
            if not text:
                continue
            message_index = turn_indexes[index] if index < len(turn_indexes) else start_index + index
            evidence = {
                "sessionFile": session_file,
                "messageIndex": message_index,
                "snippet": text,
            }
            is_error = bool(re.search(
                r"error|failed|exception|429|denied|timeout|报错|失败|冲突",
                text, re.IGNORECASE,
            ))
            event_node_id = stableId("EVENT", session_id, message_index, text)
            event_vector = embedText(text, self._embed_fn)
            event_node = {
                "id": event_node_id,
                "nodeType": "EVENT",
                "label": text[:100],
                "content": text,
                "keywords": top_keywords([text]),
                "vector": event_vector,
                "confidence": 0.92 if is_error else 0.75,
                "evidence": evidence,
            }
            nodes.append(event_node)

            # TASK → EVENT edge
            edges.append({
                "id": stableId(task_node_id, event_node_id, "temporal"),
                "fromId": task_node_id,
                "toId": event_node_id,
                "edgeType": "triggers" if is_error else "supports",
                "weight": 1.0 if is_error else 0.7,
                "evidence": evidence,
            })

            # Temporal chain between sequential events
            if previous_event is not None:
                edges.append({
                    "id": stableId(previous_event["id"], event_node_id, "temporal"),
                    "fromId": previous_event["id"],
                    "toId": event_node_id,
                    "edgeType": "temporal",
                    "weight": 0.8,
                    "evidence": evidence,
                })
            previous_event = event_node

            # Error → SALIENT_MEMORY node (error-resolution pattern)
            if is_error:
                trigger_text = text[:200]
                action_text = turn_texts[index + 1][:200] if index + 1 < len(turn_texts) else "pending resolution"
                outcome_text = turn_texts[index + 2][:200] if index + 2 < len(turn_texts) else "unknown"
                salient_memory_content = f"trigger: {trigger_text}\naction: {action_text}\noutcome: {outcome_text}"
                salient_memory_label = _salient_memory_label_from_error(text)
                salient_memory_node_id = stableId("SALIENT_MEMORY", salient_memory_label)
                salient_memory_vector = embedText(salient_memory_content, self._embed_fn)
                salient_memory_node = {
                    "id": salient_memory_node_id,
                    "nodeType": "SALIENT_MEMORY",
                    "label": salient_memory_label,
                    "content": salient_memory_content,
                    "keywords": top_keywords([text, action_text]),
                    "vector": salient_memory_vector,
                    "confidence": 0.81,
                    "evidence": evidence,
                }
                nodes.append(salient_memory_node)
                edges.append({
                    "id": stableId(event_node_id, salient_memory_node_id, "solves"),
                    "fromId": event_node_id,
                    "toId": salient_memory_node_id,
                    "edgeType": "solves",
                    "weight": 1.0,
                    "evidence": evidence,
                })

            # Pattern-driven dependency/conflict/fact synthesis is intentionally disabled in
            # the generic memory path. These edges/nodes must come from structured extraction,
            # not query-specific or phrase-specific regex rules.

        # Apply to state graph
        self._upsert_distilled_nodes(nodes)
        self._upsert_distilled_edges(edges)
        self._pending_graph_changes += len(nodes) + len(edges)

        logger.info("蒸馏轮次完成: 生成节点数=%d, 生成边数=%d, 累计图变更=%d", len(nodes), len(edges), self._pending_graph_changes)
        return {"nodes": nodes, "edges": edges}

    def _upsert_distilled_nodes(self, nodes: list[dict]) -> None:
        """Store distilled graph nodes into the in-memory state."""
        logger.debug("写入蒸馏节点到状态: 节点数=%d", len(nodes))
        state = self._state()
        from ebm_context_engine.types import EntityNode, EventNode, LedgerFact, EvidenceRef
        for node in nodes:
            node_id = node["id"]
            evidence_raw = node.get("evidence")
            evidence = EvidenceRef(
                sessionFile=str(evidence_raw.get("sessionFile", "") or "") if evidence_raw else "",
                messageIndex=evidence_raw.get("messageIndex") if evidence_raw else None,
                snippet=str(evidence_raw.get("snippet", "") or "") if evidence_raw else "",
            ) if evidence_raw else None
            # Store as generic graph node attributes in a unified way
            # The distilled nodes are TASK/EVENT/SALIENT_MEMORY/FACT types from distillTurn
            # Store them in the appropriate state dict based on type
            node_type = node.get("nodeType", "")
            if node_type == "ENTITY":
                if node_id not in state.entities:
                    state.entities[node_id] = EntityNode(
                        id=node_id,
                        name=node.get("label", ""),
                        tokens=list(node.get("keywords", [])),
                        description=node.get("content", ""),
                        vector=node.get("vector"),
                    )
            elif node_type == "EVENT":
                if node_id not in state.events:
                    session_file = evidence.sessionFile if evidence else ""
                    state.events[node_id] = EventNode(
                        id=node_id,
                        session_key=session_file,
                        date_time="",
                        turn_index=evidence.messageIndex if evidence and evidence.messageIndex is not None else 0,
                        speaker="",
                        text=node.get("content", ""),
                        content=node.get("content", ""),
                        tokens=list(node.get("keywords", [])),
                        evidence=evidence,
                        vector=node.get("vector"),
                    )
            # For TASK, SALIENT_MEMORY, FACT distilled nodes: store as facts in the graph
            # since the Python MemoryState doesn't have separate TASK/SALIENT_MEMORY stores
            elif node_type in ("TASK", "SALIENT_MEMORY", "FACT"):
                if node_id not in state.facts:
                    from ebm_context_engine.types import LedgerFact as LF
                    state.facts[node_id] = LF(
                        id=node_id,
                        subject="graph",
                        key=f"distilled.{node_type.lower()}",
                        scope="experience" if node_type == "SALIENT_MEMORY" else "project" if node_type == "TASK" else "environment",
                        value=node.get("content", ""),
                        text=f"graph: {node.get('content', '')}",
                        session_key=evidence.sessionFile if evidence else "",
                        turn_index=evidence.messageIndex if evidence and evidence.messageIndex is not None else 0,
                        tokens=list(node.get("keywords", [])),
                        evidence=evidence,
                        confidence=node.get("confidence", 0.8),
                        source="distillTurn",
                        status="active",
                        vector=node.get("vector"),
                    )

    def _upsert_distilled_edges(self, edges: list[dict]) -> None:
        """Store distilled graph edges into the in-memory state."""
        logger.debug("写入蒸馏边到状态: 边数=%d", len(edges))
        state = self._state()
        from ebm_context_engine.types import GraphEdgeRecord, EvidenceRef
        for edge in edges:
            edge_id = edge["id"]
            evidence_raw = edge.get("evidence")
            evidence = EvidenceRef(
                sessionFile=str(evidence_raw.get("sessionFile", "") or "") if evidence_raw else "",
                messageIndex=evidence_raw.get("messageIndex") if evidence_raw else None,
                snippet=str(evidence_raw.get("snippet", "") or "") if evidence_raw else "",
            ) if evidence_raw else None
            state.graph_edges[edge_id] = GraphEdgeRecord(
                id=edge_id,
                from_id=edge.get("fromId", ""),
                to_id=edge.get("toId", ""),
                edge_type=edge.get("edgeType", ""),
                weight=float(edge.get("weight", 0.5)),
                relation_label=edge.get("relationLabel", ""),
                evidence=evidence,
            )
            # Update adjacency
            from_id = edge.get("fromId", "")
            to_id = edge.get("toId", "")
            state.adjacency.setdefault(from_id, set()).add(to_id)
            state.adjacency.setdefault(to_id, set()).add(from_id)

    # ── Multi-hop expansion (matching TS plane-b.ts:824-882) ─────────────
    def _expand_multi_hop(
        self,
        seeds: dict[str, float],
        edges,
        node_by_id: dict[str, _GraphNodeView],
        max_hops: int,
        intent_weights: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """BFS expansion from seed nodes with per-hop score decay and intent-aware edge boosting."""
        logger.debug("开始多跳扩展: 种子数=%d, 最大跳数=%d", len(seeds), max_hops)
        adjacency: dict[str, list[tuple[str, float, str]]] = {}
        for edge in edges:
            from_id = edge.from_id
            to_id = edge.to_id
            weight = edge.weight
            edge_type = edge.edge_type
            adjacency.setdefault(from_id, []).append((to_id, weight, edge_type))
            adjacency.setdefault(to_id, []).append((from_id, weight, edge_type))

        visited: dict[str, float] = {}
        frontier = dict(seeds)

        CAUSAL_TYPES = {"causes", "prevents", "enables", "triggers"}
        TEMPORAL_TYPES = {"temporal"}
        ENTITY_TYPES = {"related_to", "has_attribute", "participates_in"}

        for hop in range(max_hops):
            next_frontier: dict[str, float] = {}
            decay_factor = 1.0 / (hop + 1)
            for node_id, score in frontier.items():
                existing = visited.get(node_id, 0.0)
                if score > existing:
                    visited[node_id] = score
                for target, weight, edge_type in adjacency.get(node_id, []):
                    if intent_weights:
                        if edge_type in CAUSAL_TYPES:
                            edge_boost = float(intent_weights.get("causalEdgeBoost", 1.0))
                        elif edge_type in TEMPORAL_TYPES:
                            edge_boost = float(intent_weights.get("temporalEdgeBoost", 1.0))
                        elif edge_type in ENTITY_TYPES:
                            edge_boost = float(intent_weights.get("entityEdgeBoost", 1.0))
                        else:
                            edge_boost = 1.0
                    else:
                        edge_boost = 1.2 if edge_type in ENTITY_TYPES else 0.8
                    propagated = score * weight * decay_factor * edge_boost
                    if propagated > next_frontier.get(target, 0.0):
                        next_frontier[target] = propagated
            frontier = next_frontier

        # Add remaining frontier
        for node_id, score in frontier.items():
            existing = visited.get(node_id, 0.0)
            if score > existing:
                visited[node_id] = score
        logger.debug("多跳扩展完成: 访问节点数=%d", len(visited))
        return visited

    # ── Entity summary building (matching TS plane-b.ts:890-919) ─────────
    def _build_entity_summary(self, entity: _GraphNodeView, edges, node_by_id: dict[str, _GraphNodeView]) -> str:
        """Build consolidated entity summary from attributes and relationships."""
        parts: list[str] = [entity.content]
        seen: set[str] = set()
        # 自适应过滤策略：
        # has_attribute 边承载语义属性（偏好、框架、事实），信息密度高，优先展示。
        # participates_in 边连接 event 节点，对高频实体（如 user）会产生大量
        # 低价值 turn 引用（每个 turn 都建边），淹没真正有用的属性行。
        # 因此：先收集 attribute，再收集 relation/event，event 按 mention 频率
        # 自适应限制条数——高频实体（mention_count > 10）最多 2 条 event，
        # 低频实体保留最多 6 条以支持时间线定位和多跳推理。
        attr_parts: list[str] = []
        rel_parts: list[str] = []
        event_parts: list[str] = []
        max_events = 2 if entity.mention_count > 10 else 6
        for edge in edges:
            neighbor_id: Optional[str] = None
            if edge.from_id == entity.id:
                neighbor_id = edge.to_id
            elif edge.to_id == entity.id:
                neighbor_id = edge.from_id
            if not neighbor_id:
                continue
            neighbor = node_by_id.get(neighbor_id)
            if not neighbor or neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            if edge.edge_type == "has_attribute":
                attr_parts.append(f"- {edge.relation_label or 'attr'}: {neighbor.content[:100]}")
            elif edge.edge_type == "related_to":
                label = edge.relation_label or "related to"
                rel_parts.append(f"- {label} → {neighbor.label}")
            elif edge.edge_type == "participates_in":
                if len(event_parts) < max_events:
                    event_parts.append(f"- event: {neighbor.label}")
        combined = attr_parts + rel_parts + event_parts
        return "\n".join((parts + combined)[:12])

    # ── 2-hop relationship chains (matching TS plane-b.ts:927-996) ───────
    def _build_relationship_chains(
        self,
        entity_seeds: dict[str, float],
        edges,
        node_by_id: dict[str, _GraphNodeView],
        max_chains: int,
    ) -> list[RecallHit]:
        """Build 1-hop AND 2-hop relationship chain items for multi-hop reasoning."""
        logger.debug("开始构建关系链: 实体种子数=%d, 最大链数=%d", len(entity_seeds), max_chains)
        if not entity_seeds:
            return []

        adjacency: dict[str, list[tuple[str, object]]] = {}
        for edge in edges:
            adjacency.setdefault(edge.from_id, []).append((edge.to_id, edge))
            adjacency.setdefault(edge.to_id, []).append((edge.from_id, edge))

        chains: list[RecallHit] = []
        seed_ids = list(entity_seeds.keys())

        for seed_id in seed_ids:
            seed_node = node_by_id.get(seed_id)
            if not seed_node:
                continue
            neighbors = adjacency.get(seed_id, [])
            for mid_id, edge1 in neighbors:
                mid_node = node_by_id.get(mid_id)
                if not mid_node:
                    continue
                label1 = getattr(edge1, "relation_label", "") or edge1.edge_type

                # 1-hop paths
                if mid_node.nodeType in ("ENTITY", "FACT"):
                    chain_text = f"{seed_node.label} → [{label1}] → {mid_node.label}: {mid_node.content[:100]}"
                    chains.append(RecallHit(
                        id=stableId("CHAIN", seed_id, mid_id),
                        title=f"Chain: {seed_node.label} → {mid_node.label}",
                        content=chain_text,
                        source="graph",
                        score=(entity_seeds.get(seed_id, 0.0)) * edge1.weight,
                        reason="entity relationship chain (1-hop)",
                    ))

                # 2-hop paths
                mid_neighbors = adjacency.get(mid_id, [])
                for end_id, edge2 in mid_neighbors:
                    if end_id == seed_id:
                        continue
                    end_node = node_by_id.get(end_id)
                    if not end_node:
                        continue
                    if end_node.nodeType not in ("ENTITY", "FACT"):
                        continue
                    label2 = getattr(edge2, "relation_label", "") or edge2.edge_type
                    chain_text = f"{seed_node.label} → [{label1}] → {mid_node.label} → [{label2}] → {end_node.label}: {end_node.content[:100]}"
                    chains.append(RecallHit(
                        id=stableId("CHAIN", seed_id, mid_id, end_id),
                        title=f"Chain: {seed_node.label} → {mid_node.label} → {end_node.label}",
                        content=chain_text,
                        source="graph",
                        score=(entity_seeds.get(seed_id, 0.0)) * edge1.weight * edge2.weight * 0.8,
                        reason="entity relationship chain (2-hop)",
                    ))

        # Deduplicate and sort
        seen: set[str] = set()
        deduped: list[RecallHit] = []
        for chain in chains:
            if chain.id in seen:
                continue
            seen.add(chain.id)
            deduped.append(chain)
        deduped.sort(key=lambda c: c.score, reverse=True)
        logger.debug("关系链构建完成: 去重后链数=%d", len(deduped[:max_chains]))
        return deduped[:max_chains]

    # ── Generalized recall (matching TS plane-b.ts:731-793) ──────────────
    def _recall_generalized(
        self,
        query: str,
        query_vector,
        nodes: list[_GraphNodeView],
        edges,
        node_by_id: dict[str, _GraphNodeView],
        limit: int,
        exclude_ids: set[str],
        communities,
        classification,
    ) -> list[RecallHit]:
        """Community-level search when entity seeds are narrow."""
        logger.debug("开始泛化召回: limit=%d, 排除节点数=%d, 社区数=%d", limit, len(exclude_ids), len(communities))
        if limit <= 0 or not communities:
            return []

        ranked_communities = rank_text_records(
            query,
            list(communities),
            query_vector=query_vector,
            get_text=lambda c: f"{c.title} {c.summary} {' '.join(c.keywords)}",
            get_vector=lambda c: c.vector,
            intent_weights=classification.weights if classification else None,
            rrf_k=self._rrf_k,
        )
        if not ranked_communities:
            return []

        # Pick top-2 representative nodes from each top community (by confidence)
        community_members: dict[str, list[_GraphNodeView]] = {}
        for node in nodes:
            if node.id in exclude_ids:
                continue
            for community, _score in ranked_communities[:3]:
                kw_overlap = any(kw in community.keywords for kw in node.keywords)
                vec_sim = cosine_similarity(node.vector, community.vector)
                if kw_overlap or vec_sim > self._generalized_vec_sim_threshold:
                    community_members.setdefault(community.id, []).append(node)
                    break

        # 1-hop PPR from representatives
        rep_seeds: dict[str, float] = {}
        for members in community_members.values():
            sorted_members = sorted(members, key=lambda m: m.confidence, reverse=True)
            for m in sorted_members[:2]:
                rep_seeds[m.id] = 0.5 + cosine_similarity(m.vector, query_vector) * 0.5

        if not rep_seeds:
            logger.debug("泛化召回: 无代表种子，返回空结果")
            return []

        ppr_scores = personalized_page_rank(rep_seeds, edges, damping=DEFAULT_PPR_DAMPING)
        results = [
            (node_id, score)
            for node_id, score in sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
            if node_id not in exclude_ids and node_id in node_by_id
        ][:limit]

        return [
            RecallHit(
                id=node_by_id[node_id].id,
                title=f"{node_by_id[node_id].nodeType}: {node_by_id[node_id].label}",
                content=node_by_id[node_id].content,
                source="graph",
                score=score * self.generalized_recall_discount,
                reason="generalized community recall fallback",
                evidence=node_by_id[node_id].evidence,
            )
            for node_id, score in results
        ]

    # ── Community rebuild with mutex (matching TS plane-b.ts:494-512) ────
    def rebuildCommunities(self) -> dict[str, object]:
        logger.info("开始重建社区结构")
        with self._rebuild_lock:
            if self._rebuild_in_progress:
                self._rebuild_needed = True
                return self._state().communities
            self._rebuild_in_progress = True

        try:
            while True:
                self._rebuild_needed = False
                result = self._do_rebuild_communities()
                with self._rebuild_lock:
                    if not self._rebuild_needed:
                        self._pending_graph_changes = 0
                        break
            return result
        finally:
            with self._rebuild_lock:
                self._rebuild_in_progress = False
                logger.info("社区重建流程结束")

    def rebuildCommunitiesIfNeeded(self, force: bool = False) -> dict[str, object]:
        logger.debug("检查是否需要重建社区: force=%s, 待处理变更=%d, 阈值=%d", force, self._pending_graph_changes, self._rebuild_change_threshold)
        if not force and self._pending_graph_changes < self._rebuild_change_threshold:
            return self._state().communities
        return self.rebuildCommunities()

    def _do_rebuild_communities(self) -> dict[str, object]:
        logger.debug("执行社区重建")
        state = self._state()
        nodes = self._graph_nodes()
        edges = list(state.graph_edges.values())
        if not nodes:
            state.communities = {}
            return state.communities
        node_by_id = {node.id: node for node in nodes}
        total_edge_weight = sum(edge.weight for edge in edges) or 1.0
        members: dict[str, set[str]] = {node.id: {node.id} for node in nodes}
        community_of: dict[str, str] = {node.id: node.id for node in nodes}
        incident_weight: dict[str, float] = {node.id: 0.0 for node in nodes}
        for edge in edges:
            incident_weight[community_of[edge.from_id]] = incident_weight.get(community_of[edge.from_id], 0.0) + edge.weight
            incident_weight[community_of[edge.to_id]] = incident_weight.get(community_of[edge.to_id], 0.0) + edge.weight

        while True:
            pair_weights: dict[tuple[str, str], float] = {}
            for edge in edges:
                left = community_of.get(edge.from_id)
                right = community_of.get(edge.to_id)
                if not left or not right or left == right:
                    continue
                pair = _pair_key(left, right)
                pair_weights[pair] = pair_weights.get(pair, 0.0) + edge.weight
            best_pair = None
            best_gain = 0.0
            for (left, right), inter_weight in pair_weights.items():
                gain = inter_weight / total_edge_weight - (
                    incident_weight.get(left, 0.0) * incident_weight.get(right, 0.0)
                ) / (2 * total_edge_weight * total_edge_weight)
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (left, right)
            if not best_pair:
                break
            left, right = best_pair
            merged_id = f"community:{stableId(*sorted(members[left] | members[right]))}"
            merged_members = members[left] | members[right]
            members[merged_id] = merged_members
            incident_weight[merged_id] = incident_weight.get(left, 0.0) + incident_weight.get(right, 0.0)
            for node_id in merged_members:
                community_of[node_id] = merged_id
            del members[left]
            del members[right]
            incident_weight.pop(left, None)
            incident_weight.pop(right, None)

        communities: dict[str, object] = {}
        for community_id, node_ids in members.items():
            community_nodes = [node_by_id[node_id] for node_id in node_ids if node_id in node_by_id]
            if not community_nodes:
                continue
            summary_nodes = sorted(community_nodes, key=lambda node: node.confidence, reverse=True)[:8]
            title_nodes = summary_nodes[:2]
            summary_text = self._buildLocalCommunitySummary(summary_nodes)
            # Embed community vector (fixing P3: community vectors were None)
            community_vector = embedText(summary_text, self._embed_fn)
            communities[community_id] = {
                "id": community_id,
                "title": " / ".join(node.label for node in title_nodes) if title_nodes else community_id,
                "summary": summary_text,
                "keywords": [keyword for node in community_nodes for keyword in node.keywords][:12],
                "member_ids": sorted(node_ids),
                "vector": community_vector,
            }
        state.communities = communities
        logger.debug("社区重建完成: 社区数=%d", len(communities))
        return communities

    def _buildLocalCommunitySummary(self, nodes: list[_GraphNodeView]) -> str:
        top_nodes = sorted(nodes, key=lambda node: node.confidence, reverse=True)[:5]
        parts: list[str] = []
        for node in top_nodes:
            if node.nodeType == "SALIENT_MEMORY" and "trigger:" in node.content:
                parts.append(node.content.split("\n")[0] + "; " + (node.content.split("\n")[1] if len(node.content.split("\n")) > 1 else ""))
            elif node.nodeType == "TASK":
                parts.append(f"Task: {node.label}")
            elif node.nodeType == "FACT":
                parts.append(f"Fact: {node.label}")
            elif node.nodeType == "ENTITY":
                parts.append(f"Entity: {node.label}")
            else:
                parts.append(f"{node.nodeType}: {node.label}")
        return ". ".join(parts)

    def _graph_nodes(self) -> list[_GraphNodeView]:
        state = self._state()
        nodes: list[_GraphNodeView] = []
        for entity in state.entities.values():
            nodes.append(_GraphNodeView(entity.id, "ENTITY", entity.name, entity.description or entity.name, list(entity.tokens), entity.vector, min(0.99, 0.6 + entity.mention_count * 0.05), None, entity.mention_count))
        for event in state.events.values():
            nodes.append(_GraphNodeView(event.id, "EVENT", f"{event.session_key} / turn {event.turn_index + 1} / {event.speaker}", event.content, list(event.tokens), event.vector, 0.9, event.evidence))
        for fact in state.facts.values():
            nodes.append(_GraphNodeView(fact.id, "FACT", f"{fact.scope} / {fact.key}", fact.text, list(fact.tokens), fact.vector, fact.confidence, fact.evidence))
        return nodes

    def _find_entity_seeds(self, query: str, query_vector) -> dict[str, float]:
        logger.debug("开始查找实体种子: query长度=%d", len(query))
        state = self._state()
        lower = query.lower()
        seeds: dict[str, float] = {}
        for entity in state.entities.values():
            name_lower = entity.name.lower()
            if name_lower and len(name_lower) >= 2 and name_lower in lower:
                seeds[entity.id] = 2.0
                continue
            sim = cosine_similarity(query_vector, entity.vector)
            if sim > self._entity_seed_sim_threshold:
                seeds[entity.id] = sim
        logger.debug("实体种子查找完成: 候选实体数=%d, 匹配种子数=%d", len(state.entities), len(seeds))
        return seeds

    def recall(
        self,
        query: str,
        *,
        query_vector,
        graph_top_k: int,
        community_top_k: int,
        matched_entity_ids: list[str],
        classification,
    ) -> tuple[list[RecallHit], list[RecallHit]]:
        logger.info("开始图召回: query长度=%d, graph_top_k=%d, community_top_k=%d, 匹配实体数=%d", len(query), graph_top_k, community_top_k, len(matched_entity_ids))
        state = self._state()
        nodes = self._graph_nodes()
        if not nodes:
            logger.debug("图召回: 无图节点，返回空结果")
            return [], []

        edges = list(state.graph_edges.values())
        node_by_id = {node.id: node for node in nodes}

        # Phase 1: Entity-seeded retrieval
        entity_seeds = self._find_entity_seeds(query, query_vector)
        logger.debug("Phase1 实体种子: 种子数=%d", len(entity_seeds))

        # Phase 2: Multi-hop expansion from entity seeds
        multi_hop_nodes = self._expand_multi_hop(
            entity_seeds, edges, node_by_id, self._max_hops,
            classification.weights if classification else None,
        )

        # Phase 3: Standard PPR
        ranked = rank_graph_nodes(query, nodes, query_vector=query_vector, intent_weights=classification.weights, rrf_k=self._rrf_k)
        seed_weights: dict[str, float] = {node_id: score * 2.0 for node_id, score in entity_seeds.items()}
        for node, score in ranked[: max(graph_top_k, 1)]:
            seed_weights.setdefault(node.id, max(0.001, score))

        ppr_scores = personalized_page_rank(seed_weights, edges, damping=DEFAULT_PPR_DAMPING, intent_weights=classification.weights)
        logger.debug("Phase3 PPR完成: PPR节点数=%d", len(ppr_scores))

        # Phase 4: Merge multi-hop + PPR scores
        merged_scores: dict[str, float] = {}
        for node_id, score in ppr_scores.items():
            merged_scores[node_id] = score
        for node_id, hop_score in multi_hop_nodes.items():
            existing = merged_scores.get(node_id, 0.0)
            merged_scores[node_id] = existing + hop_score * self._multi_hop_bonus

        # Phase 5: Build graph hits with entity-centric formatting
        precise_half_k = max(1, (graph_top_k + 1) // 2)
        sorted_merged = sorted(
            ((node_by_id[nid], score) for nid, score in merged_scores.items() if nid in node_by_id),
            key=lambda x: x[1], reverse=True,
        )

        graph_hits: list[RecallHit] = []
        for node, score in sorted_merged[:precise_half_k]:
            is_entity = node.nodeType == "ENTITY"
            is_multi_hop = node.id in multi_hop_nodes
            content = self._build_entity_summary(node, edges, node_by_id) if is_entity else node.content
            reason = "multi-hop graph traversal" if is_multi_hop else "entity seed match" if is_entity else "personalized PageRank graph walk"
            graph_hits.append(RecallHit(
                id=node.id,
                title=f"{node.nodeType}: {node.label}",
                content=content,
                source="graph",
                score=score,
                reason=reason,
                evidence=node.evidence,
                session_key=getattr(node.evidence, "sessionFile", "") if node.evidence else "",
                turn_index=getattr(node.evidence, "messageIndex", None) if node.evidence else None,
            ))

        # Phase 5b: Generalized recall fallback
        generalized_half_k = graph_top_k - precise_half_k
        precise_node_ids = {h.id for h in graph_hits}
        generalized_hits = self._recall_generalized(
            query, query_vector, nodes, edges, node_by_id,
            generalized_half_k, precise_node_ids,
            list(state.communities.values()), classification,
        )

        # Phase 6: Relationship chains (2-hop)
        relationship_chains = self._build_relationship_chains(
            entity_seeds, edges, node_by_id, graph_top_k,
        )

        all_graph_hits = [*relationship_chains, *graph_hits, *generalized_hits]
        all_graph_hits.sort(key=lambda item: item.score, reverse=True)
        logger.debug("图命中合并: 关系链=%d, 精确命中=%d, 泛化命中=%d, 合计=%d", len(relationship_chains), len(graph_hits), len(generalized_hits), len(all_graph_hits))

        # Community hits
        community_hits: list[RecallHit] = []
        communities = list(state.communities.values())
        ranked_communities = rank_text_records(
            query,
            communities,
            query_vector=query_vector,
            get_text=lambda community: f"{community.title} {community.summary} {' '.join(community.keywords)}",
            get_vector=lambda community: community.vector,
            intent_weights=classification.weights,
            rrf_k=self._rrf_k,
        )
        for community, score in ranked_communities[:community_top_k]:
            community_hits.append(RecallHit(id=community.id, title=community.title, content=community.summary, source="community", score=score * self.generalized_recall_discount, reason="community summary semantic match"))

        logger.info("图召回完成: 图命中数=%d, 社区命中数=%d", len(all_graph_hits[:graph_top_k]), len(community_hits))
        return all_graph_hits[:graph_top_k], community_hits

    def getAllGraphNodes(self) -> list[dict]:
        return [{"id": node.id, "nodeType": node.nodeType, "label": node.label, "content": node.content, "keywords": node.keywords, "confidence": node.confidence} for node in self._graph_nodes()]

    def getAllGraphEdges(self) -> list[dict]:
        return [{"id": edge.id, "fromId": edge.from_id, "toId": edge.to_id, "edgeType": edge.edge_type, "weight": edge.weight, "relationLabel": edge.relation_label} for edge in self._state().graph_edges.values()]

    def getAllCommunities(self) -> list[dict]:
        return [{"id": item.id, "title": item.title, "summary": item.summary, "keywords": list(item.keywords)} for item in self._state().communities.values()]
