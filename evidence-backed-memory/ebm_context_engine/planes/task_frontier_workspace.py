from __future__ import annotations

import logging
import time
from typing import Iterable

logger = logging.getLogger("ebm_context_engine.planes.task_frontier_workspace")

from ebm_context_engine.core.hash import _stable_id
from ebm_context_engine.core.messages import summarizeMessage
from ebm_context_engine.core.text import topKeywords
from ebm_context_engine.core.token import estimateTokens, truncateToTokens
from ebm_context_engine.types import MemoryPacket, MemoryPacketSection, PinnedEntry, ScratchpadEntry, TopicEntry, RecallHit


def _clone_evidence(evidence, snippet: str):
    if evidence is None:
        return None
    return type(evidence)(
        sessionFile=evidence.sessionFile,
        messageIndex=evidence.messageIndex,
        startLine=evidence.startLine,
        endLine=evidence.endLine,
        snippet=evidence.snippet or snippet[:200],
        dateTime=evidence.dateTime,
        speaker=evidence.speaker,
    )


class TaskFrontierWorkspace:
    def __init__(
        self,
        store=None,
        pinned_budget_ratio: float = 0.12,
        scratchpad_window: int = 6,
        *,
        graph_items_limit: int = 5,
        ledger_items_limit: int = 4,
        recall_content_max_chars: int = 180,
        demoter=None,
    ):
        self.store = store
        self.pinned_budget_ratio = pinned_budget_ratio
        self.scratchpad_window = scratchpad_window
        self.graph_items_limit = graph_items_limit
        self.ledger_items_limit = ledger_items_limit
        self.recall_content_max_chars = recall_content_max_chars
        self.demoter = demoter

    def refreshPinnedContext(self, session_id: str, session_file: str | None, query: str, messages: list[dict], promoted_facts: list, message_evidence_by_index=None) -> list[PinnedEntry]:
        logger.info("开始刷新固定上下文: session_id=%s, 消息数=%d, 提升事实数=%d", session_id, len(messages), len(promoted_facts))
        message_evidence_by_index = message_evidence_by_index or {}
        candidates: list[PinnedEntry] = []
        policy_texts = [
            (index, summarizeMessage(message))
            for index, message in enumerate(messages)
            if any(token in summarizeMessage(message).lower() for token in ("must", "never", "constraint", "strictly", "不要", "必须"))
        ][-4:]
        logger.debug("策略文本匹配数=%d", len(policy_texts))
        for index, text in policy_texts:
            import re as _re
            scope = "safety" if _re.search(r"safety|合规", text, _re.IGNORECASE) else "task"
            candidates.append(
                PinnedEntry(
                    id=_stable_id("PIN", session_id, index, text),
                    sessionId=session_id,
                    scope=scope,
                    label="conversation constraint",
                    content=truncateToTokens(text, 60),
                    priority=1.0,
                    tokenCost=estimateTokens(text),
                    evidence=_clone_evidence(message_evidence_by_index.get(index), text),
                )
            )
        for fact in promoted_facts:
            candidates.append(
                PinnedEntry(
                    id=_stable_id("PIN", session_id, fact.id),
                    sessionId=session_id,
                    scope="task" if "constraint" in fact.title else "user",
                    label=fact.title,
                    content=truncateToTokens(fact.content, 50),
                    priority=fact.score,
                    tokenCost=estimateTokens(fact.content),
                    evidence=fact.evidence,
                )
            )
        # NOTE: active goal removed — it duplicates the Question section
        if self.store is not None:
            self.store.upsert_pinned_entries(candidates)
        logger.info("固定上下文刷新完成: 候选条目数=%d", len(candidates))
        return candidates

    def refreshTopicIndex(self, session_id: str, query: str, messages: list[dict], retrieval_items: list) -> list[TopicEntry]:
        logger.info("开始刷新主题索引: session_id=%s, 消息数=%d, 检索项数=%d", session_id, len(messages), len(retrieval_items))
        keywords = topKeywords(
            [query, *[summarizeMessage(message) for message in messages], *[f"{item.title} {item.content}" for item in retrieval_items]],
            16,
        )
        entries = [
            TopicEntry(sessionId=session_id, topic=topic, score=max(0.1, 1 - index * 0.05), source="fast-path")
            for index, topic in enumerate(keywords)
        ]
        if self.store is not None:
            self.store.replace_topic_entries(session_id, entries)
        logger.info("主题索引刷新完成: 主题数=%d", len(entries))
        return entries

    def refreshScratchpad(self, session_id: str, session_file: str | None, messages: list[dict], retrieval_items: list, message_evidence_by_index=None) -> None:
        logger.info("开始刷新草稿板: session_id=%s, 消息数=%d, 检索项数=%d", session_id, len(messages), len(retrieval_items))
        message_evidence_by_index = message_evidence_by_index or {}
        recent_messages = messages[-3:]
        recent_start_index = max(0, len(messages) - len(recent_messages))
        now = int(time.time() * 1000)
        entries: list[ScratchpadEntry] = []
        for index, message in enumerate(recent_messages):
            content = summarizeMessage(message)
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", "")
            entries.append(
                ScratchpadEntry(
                    id=_stable_id("SCRATCH", session_id, content, index),
                    sessionId=session_id,
                    kind="result" if role == "toolResult" else "summary",
                    content=truncateToTokens(content, 64),
                    tokenCost=estimateTokens(content),
                    createdAt=now + index,
                    evidence=_clone_evidence(message_evidence_by_index.get(recent_start_index + index), content),
                )
            )
        for index, item in enumerate(retrieval_items[:3]):
            entries.append(
                ScratchpadEntry(
                    id=_stable_id("SCRATCH", session_id, item.id),
                    sessionId=session_id,
                    kind="evidence",
                    content=truncateToTokens(f"{item.title}: {item.content}", 64),
                    tokenCost=estimateTokens(item.content),
                    createdAt=now + len(entries) + index,
                    evidence=item.evidence,
                )
            )
        if self.store is not None:
            self.store.append_scratchpad_entries(entries, self.scratchpad_window)
        logger.info("草稿板刷新完成: 条目数=%d", len(entries))

    def buildPacket(self, *args, **kwargs) -> MemoryPacket:
        logger.info("开始构建记忆数据包")
        if args and isinstance(args[0], dict):
            params = dict(args[0])
        else:
            question, _classification, plan, graph_hits, ledger_hits, summary_hits, community_hits = args
            params = {
                "sessionId": "",
                "workspaceId": "",
                "query": question,
                "tokenBudget": max(2048, sum(getattr(item, "tokenCost", 0) for item in [])),
                "graphItems": graph_hits,
                "ledgerItems": ledger_hits,
                "summaryItems": summary_hits,
                "communityItems": community_hits,
                "graphItemsLimit": getattr(plan, "graph_item_limit", self.graph_items_limit),
                "ledgerItemsLimit": getattr(plan, "ledger_item_limit", self.ledger_items_limit),
            }
        session_id = str(params.get("sessionId", "") or "")
        workspace_id = str(params.get("workspaceId", "") or session_id)
        query = str(params.get("query", "") or "")
        token_budget = int(params.get("tokenBudget", 16000) or 16000)
        graph_items = list(params.get("graphItems", []) or [])
        ledger_items = list(params.get("ledgerItems", []) or [])
        logger.debug("数据包参数: session_id=%s, query长度=%d, token预算=%d, 图项数=%d, 账本项数=%d", session_id, len(query), token_budget, len(graph_items), len(ledger_items))
        pinned_budget = max(64, int(token_budget * self.pinned_budget_ratio))
        pinned_result = self._build_pinned_section(workspace_id, pinned_budget)
        graph_section = self._build_recall_section(
            "Structured Memory Graph",
            graph_items[: int(params.get("graphItemsLimit", self.graph_items_limit) or self.graph_items_limit)],
        )
        ledger_section = self._build_recall_section(
            "Memory Facts",
            ledger_items[: int(params.get("ledgerItemsLimit", self.ledger_items_limit) or self.ledger_items_limit)],
        )
        sections = [pinned_result["section"], graph_section, ledger_section]
        sections = [section for section in sections if section.lines]
        warnings: list[str] = []
        if pinned_result["skippedCount"] > 0:
            warnings.append(
                f"[EBM WARN] Pinned context exceeded budget ({pinned_budget} tokens): {pinned_result['skippedCount']} entries skipped from prompt"
            )
        if pinned_result["evictedToLedger"] > 0:
            warnings.append(
                f"[EBM WARN] {pinned_result['evictedToLedger']} pinned entries evicted from DB due to budget overflow"
            )
        if warnings:
            sections.insert(
                0,
                MemoryPacketSection(
                    title="EBM Warnings",
                    lines=warnings,
                    tokenCost=estimateTokens("\n".join(warnings)),
                ),
            )
        total = sum(section.tokenCost for section in sections)
        logger.info("记忆数据包构建完成: 段落数=%d, 总token估算=%d", len(sections), total)
        return MemoryPacket(query=query, totalEstimatedTokens=total, sections=sections, traceId=f"packet:{_stable_id('TRACE', session_id, query, int(time.time() * 1000))}")

    def renderSystemPrompt(self, packet: MemoryPacket, question: str, classification, *, episode_lines: list[str] | None = None) -> str:
        logger.info("开始渲染系统提示: 意图=%s, 复杂度=%s, 段落数=%d", classification.intent, classification.complexity, len(packet.sections))
        parts = [
            "Answer using the supplied memory context. Give the best direct answer grounded in evidence. "
            "For opinion or open-domain questions, synthesize an answer from what people said and did — report attitudes, praise, or concerns as opinions. "
            "Only say Information not found if nothing relevant is present.",
            f"Intent={classification.intent}; Complexity={classification.complexity}.",
        ]
        for section in packet.sections:
            parts.append(f"{section.title}:\n" + "\n".join(section.lines))
        if episode_lines:
            parts.append("Episode Context:\n" + "\n".join(episode_lines))
        parts.append(f"Question: {question}")
        logger.info("系统提示渲染完成: 总部分数=%d", len(parts))
        return "\n\n".join(parts)

    def _build_pinned_section(self, session_id: str, pinned_budget: int) -> dict[str, object]:
        logger.debug("构建固定上下文段落: session_id=%s, 预算=%d tokens", session_id, pinned_budget)
        if self.store is None or not session_id:
            return {"section": MemoryPacketSection(title="Pinned Context", lines=[], tokenCost=0), "skippedCount": 0, "evictedToLedger": 0}
        all_entries = self.store.list_pinned_entries(session_id, 256)
        logger.debug("固定条目总数=%d", len(all_entries))
        admitted: list[PinnedEntry] = []
        skipped: list[PinnedEntry] = []
        spent = 0
        for entry in all_entries:
            if spent + entry.tokenCost <= pinned_budget:
                admitted.append(entry)
                spent += entry.tokenCost
            else:
                skipped.append(entry)
        evicted_to_ledger = 0
        if skipped and self.demoter is not None:
            self.demoter(skipped)
            evicted_to_ledger = len(skipped)
            self.store.evict_pinned_entries(session_id, [entry.id for entry in skipped])
        lines = [f"- [{entry.scope}] {entry.label}: {entry.content}" for entry in admitted]
        logger.debug("固定段落构建完成: 准入=%d, 跳过=%d, 降级到账本=%d", len(admitted), len(skipped), evicted_to_ledger)
        return {
            "section": MemoryPacketSection(title="Pinned Context", lines=lines, tokenCost=spent),
            "skippedCount": len(skipped),
            "evictedToLedger": evicted_to_ledger,
        }

    def _build_recall_section(self, title: str, items: Iterable) -> MemoryPacketSection:
        logger.debug("构建召回段落: title=%s", title)
        lines = [f"- {item.title}: {truncateToTokens(item.content, self.recall_content_max_chars)}" for item in items]
        logger.debug("召回段落完成: title=%s, 行数=%d", title, len(lines))
        return MemoryPacketSection(title=title, lines=lines, tokenCost=estimateTokens("\n".join(lines)))
