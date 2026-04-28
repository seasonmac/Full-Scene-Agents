from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import functools
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .client import ChatResult, OpenAICompatClient, cosine_similarity
from .core.config import EbmConfig
from .core.hash import stableId as _stable_id
from .core.messages import messageToText
from .db.store import EbmStore
from .retrieval.intent_router import classify_query
from .planes import TaskFrontierWorkspace, StructuredSalientMemoryGraph, TemporalSemanticLedger
from .hypergraph.episode_detector import detect_episodes_llm, detect_episodes_heuristic
from .hypergraph.fact_extractor import extract_facts_from_episode
from .hypergraph.aaak_encoder import encode_facts_aaak
from .hypergraph.topic_aggregator import aggregate_episodes_to_topics
from .hypergraph.c2f_retriever import coarse_to_fine_retrieval, c2f_to_recall_hits
from .retrieval.progressive import ProgressiveRecaller
from .hypergraph.embedding import propagate_embeddings
from .retrieval.hybrid import rank_text_records
from .slowpath.llm_extractor import applyExtractedEntityGraph, buildSessionSummary, extractAllWithLlm, extractEntityGraph, extractFactsWithLlm, extractHighValueFactsWithLlm, extractProfileFactsWithLlm, summarizeSession
from .slowpath_processor import SlowPathProcessor, buildSlowPathTurnFingerprint
from .core.state_ops import (
    addEdge,
    clipText,
    embedState,
    findEntityIdByName,
    matchEntities,
    payloadToEntries,
    rebuildIndices,
    registerSessionEntries,
    resolveNodeLabel,
    selectFocusTokens,
    upsertEntity,
    upsertFact,
)
from .text import (
    COMMON_CAPITALIZED,
    contains_temporal_marker,
    keyword_overlap,
    normalize_whitespace,
    pick_sentences,
    tokenize,
    top_keywords,
    unique_preserve_order,
)
from .types import (
    ClassificationResult,
    CommunitySummaryRecord,
    EntityNode,
    EventNode,
    EvidenceRef,
    GraphEdgeRecord,
    HmEpisode,
    HmFact,
    HmTopic,
    LedgerFact,
    MemoryState,
    PythonEbmQueryResult,
    RecallHit,
    SessionSummary,
    TranscriptEntry,
    UnifiedFact,
)

logger = logging.getLogger("ebm_context_engine.engine")

ARTIFACT_VERSION = 2
VALID_FACT_SCOPES = {"preference", "constraint", "environment", "project", "experience"}
VALID_EDGE_TYPES = {
    "depends_on", "triggers", "solves", "conflicts", "temporal", "supports",
    "has_attribute", "related_to", "participates_in", "causes", "prevents", "enables",
}
ANSWER_SYSTEM = (
    "Answer using memory evidence. Be specific with names, dates, quantities. "
    "For list questions, include ALL matching items from evidence. "
    "For 'would' questions, infer from character traits and behaviors. "
    "Convert relative time expressions to dates using timestamps in evidence. "
    "If evidence is partial, still give the best answer you can — infer from clues. "
    "ONLY say 'Information not found.' when evidence is completely unrelated to the question."
)
AAAK_ANSWER_SYSTEM = (
    "You will receive memory context in AAAK V2 shorthand format. "
    "Format key: V2|wing|room|date|source header, then N:ENTITY|\"quote\"|Weight|emotion|flag lines. "
    "ENTITY = 3-letter speaker code (JON=Jon, GIN=Gina). Quote = core assertion. W1-W5 = importance. EP = episode summary. "
    "YOUR ANSWER MUST BE IN PLAIN NATURAL ENGLISH — DO NOT output AAAK format. "
    "Be CONCISE. For 'when' → date. For 'who' → name. For 'what' → specific item(s). "
    "Be specific — include names, dates, and details from the context. "
    "Use dates from context exactly. List ALL items for list questions. "
    "Infer from clues when the exact answer is not stated explicitly. "
    "ONLY say 'Information not found.' if the evidence is completely unrelated."
)
EMPTY_ANSWER_MAX_RETRIES = 5  # default; overridden by config.emptyAnswerMaxRetries
EMPTY_ANSWER_RETRY_DELAY_S = 5.0  # default; overridden by config.emptyAnswerRetryDelayS
WEEKDAY_TO_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


# ── UnifiedFact conversion helpers ──

def _ledger_fact_to_unified(fact: LedgerFact, episode_id: str = "") -> UnifiedFact:
    """Convert a LedgerFact to UnifiedFact."""
    return UnifiedFact(
        id=fact.id,
        content=fact.text or fact.value,
        subject=fact.subject,
        key=fact.key,
        scope=fact.scope,
        keywords=list(fact.tokens) if fact.tokens else [],
        confidence=fact.confidence,
        evidence=fact.evidence,
        episode_id=episode_id,
        entity_ids=[fact.subject_entity_id] if fact.subject_entity_id else [],
        session_key=fact.session_key,
        turn_index=fact.turn_index,
        source_turn_start=fact.turn_index,
        source_turn_end=fact.turn_index,
        validFrom=fact.validFrom,
        validTo=fact.validTo,
        invalidAt=fact.invalidAt,
        expiresAt=fact.expiresAt,
        source=fact.source,
        status=fact.status,
        vector=fact.vector,
        created_at=fact.validFrom or 0,
    )


def _hm_fact_to_unified(fact: HmFact, evidence: Optional[EvidenceRef] = None, confidence: float = 0.7) -> UnifiedFact:
    """Convert an HmFact to UnifiedFact."""
    return UnifiedFact(
        id=fact.id,
        content=fact.content,
        potential=fact.potential,
        keywords=list(fact.keywords) if fact.keywords else [],
        importance=fact.importance,
        confidence=confidence,
        evidence=evidence,
        episode_id=fact.episode_id,
        session_key=fact.session_key,
        source_turn_start=fact.source_turn_start,
        source_turn_end=fact.source_turn_end,
        turn_index=fact.source_turn_start,
        source="hm-extraction",
        vector=fact.vector,
        created_at=fact.created_at,
    )


def _merge_unified_facts(existing: UnifiedFact, incoming: UnifiedFact) -> UnifiedFact:
    """Merge two UnifiedFacts covering the same content. Keeps the richer metadata."""
    merged = UnifiedFact(
        id=existing.id,
        content=existing.content if len(existing.content) >= len(incoming.content) else incoming.content,
        subject=existing.subject or incoming.subject,
        key=existing.key or incoming.key,
        scope=existing.scope or incoming.scope,
        potential=existing.potential or incoming.potential,
        keywords=list(set(existing.keywords) | set(incoming.keywords)),
        importance=existing.importance if existing.importance != "mid" else incoming.importance,
        confidence=max(existing.confidence, incoming.confidence),
        evidence=existing.evidence if existing.confidence >= incoming.confidence else (incoming.evidence or existing.evidence),
        episode_id=existing.episode_id or incoming.episode_id,
        entity_ids=list(set(existing.entity_ids) | set(incoming.entity_ids)),
        session_key=existing.session_key or incoming.session_key,
        turn_index=min(existing.turn_index, incoming.turn_index) if existing.turn_index and incoming.turn_index else existing.turn_index or incoming.turn_index,
        source_turn_start=min(existing.source_turn_start, incoming.source_turn_start),
        source_turn_end=max(existing.source_turn_end, incoming.source_turn_end),
        validFrom=existing.validFrom or incoming.validFrom,
        validTo=existing.validTo or incoming.validTo,
        invalidAt=existing.invalidAt or incoming.invalidAt,
        expiresAt=existing.expiresAt or incoming.expiresAt,
        source=existing.source if existing.confidence >= incoming.confidence else incoming.source,
        status=existing.status,
        vector=existing.vector if existing.vector is not None else incoming.vector,
        created_at=existing.created_at or incoming.created_at,
    )
    return merged


def _upsert_unified_fact(state: MemoryState, fact: UnifiedFact, embed_fn=None, similarity_threshold: float = 0.9) -> None:
    """Insert or merge a UnifiedFact into state.unified_facts with dedup."""
    # Check for duplicate by content similarity within same episode
    if embed_fn and fact.vector is None:
        fact.vector = embed_fn(fact.content)
    if fact.vector is not None:
        for existing_id, existing in state.unified_facts.items():
            if existing.episode_id and existing.episode_id == fact.episode_id and existing.vector is not None:
                sim = cosine_similarity(fact.vector, existing.vector)
                if sim >= similarity_threshold:
                    state.unified_facts[existing_id] = _merge_unified_facts(existing, fact)
                    return
    state.unified_facts[fact.id] = fact


def _clip(text: str, max_chars: int) -> str:
    return clipText(text, max_chars)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_priority_structured_source(source: str) -> bool:
    """Check if a fact source indicates verified structured extraction."""
    tag = str(source or "").strip().lower()
    return tag.endswith("-verified")


class EvidenceBackedMemoryEngine:
    def __init__(
        self,
        artifact_path: str,
        *,
        config: Optional[EbmConfig] = None,
        memllm_endpoint: Any = None,
        llm_endpoint: Any = None,
        embedding_endpoint: Any = None,
        rerank_endpoint: Any = None,
        slowpath_llm_enabled: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.db_path = artifact_path
        self.artifact_path = artifact_path
        self.meta_path = f"{artifact_path}.meta.json"
        self.logger = logger or logging.getLogger("benchmark.ebm_context_engine")
        self.config = config or EbmConfig(storagePath=artifact_path)
        llm_timeout_s = max(120.0, float(self.config.sdkTimeoutMs) / 1000.0)
        embedding_timeout_s = max(12.0, float(self.config.sdkTimeoutMs) / 1000.0)
        self._llm = OpenAICompatClient(memllm_endpoint, timeout_s=llm_timeout_s, max_retries=self.config.sdkMaxRetries)
        self._answer_llm = OpenAICompatClient(llm_endpoint, timeout_s=llm_timeout_s, max_retries=self.config.sdkMaxRetries) if llm_endpoint is not None else self._llm
        self._embedder = OpenAICompatClient(embedding_endpoint, timeout_s=embedding_timeout_s, max_retries=self.config.sdkMaxRetries)
        self._reranker: OpenAICompatClient | None = OpenAICompatClient(rerank_endpoint, timeout_s=10.0, max_retries=1) if rerank_endpoint is not None else None
        self.logger.info(
            "EBM Python endpoints: memllm_model=%s memllm_base=%s answer_model=%s answer_base=%s embedding_model=%s embedding_base=%s rerank_model=%s rerank_base=%s",
            getattr(memllm_endpoint, "model", "") if memllm_endpoint is not None else "",
            getattr(memllm_endpoint, "base_url", None) or getattr(memllm_endpoint, "baseUrl", None) or "",
            getattr(llm_endpoint, "model", "") if llm_endpoint is not None else getattr(memllm_endpoint, "model", "") if memllm_endpoint is not None else "",
            getattr(llm_endpoint, "base_url", None) or getattr(llm_endpoint, "baseUrl", None) or getattr(memllm_endpoint, "base_url", None) or getattr(memllm_endpoint, "baseUrl", None) or "",
            getattr(embedding_endpoint, "model", "") if embedding_endpoint is not None else "",
            getattr(embedding_endpoint, "base_url", None) or getattr(embedding_endpoint, "baseUrl", None) or "",
            getattr(rerank_endpoint, "model", "") if rerank_endpoint is not None else "",
            getattr(rerank_endpoint, "base_url", None) or getattr(rerank_endpoint, "baseUrl", None) or "",
        )
        self._store = EbmStore(self.db_path)
        self._state: Optional[MemoryState] = None
        self._intent_cache: dict[str, tuple[ClassificationResult, Any, dict[str, int | str | float | bool]]] = {}
        self._slowpath_llm_failures = 0
        self._slowpath_llm_disabled = not slowpath_llm_enabled
        self._answer_llm_failures = 0
        self._answer_llm_disabled = False
        self.workspace = TaskFrontierWorkspace(
            store=self._store,
            pinned_budget_ratio=self.config.pinnedBudgetRatio,
            scratchpad_window=self.config.scratchpadWindow,
            graph_items_limit=self.config.graphItemsLimit,
            ledger_items_limit=self.config.ledgerItemsLimit,
            recall_content_max_chars=self.config.recallContentMaxTokens,
            demoter=lambda entries: [self.ledger.demoteFromPinned(entry) for entry in entries],
        )
        self.salientMemoryGraph = StructuredSalientMemoryGraph(
            lambda: self._require_state(),
            generalized_recall_discount=self.config.generalizedRecallDiscount,
            embed_fn=self._embedder.embed_text,
        )
        self.ledger = TemporalSemanticLedger(
            lambda: self._require_state(),
            confidence_threshold=self.config.confidenceThreshold,
            fact_ttl_days=self.config.factTTLDays,
            forgetting_half_life_days=self.config.forgettingHalfLifeDays,
        )
        self.slowPath = SlowPathProcessor(
            store=self._store,
            concurrency=self.config.slowPathConcurrency,
            retry_delay_ms=200,
            job_timeout_ms=self.config.slowPathJobTimeoutMs,
        )
        self._runtime_lock = threading.RLock()
        self._flush_state_lock = threading.RLock()
        self._flush_thread: threading.Thread | None = None
        self._flush_active = False
        self._flush_last_error = ""

    def close(self) -> None:
        self._llm.close()
        if self._answer_llm is not self._llm:
            self._answer_llm.close()
        self._embedder.close()
        if self._reranker is not None:
            self._reranker.close()
        self._store.close()

    def reset(self) -> None:
        logger.info("开始 reset: 重置引擎状态")
        self._state = MemoryState(version=ARTIFACT_VERSION)
        self._store.reset()
        logger.info("reset 完成: 状态已重置")

    def artifact_status(self) -> tuple[bool, str]:
        if not os.path.exists(self.artifact_path):
            return False, f"missing artifact: {self.artifact_path}"
        if not os.path.exists(self.meta_path):
            return False, f"missing metadata: {self.meta_path}"
        try:
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return False, f"invalid metadata: {self.meta_path}"
        if int(meta.get("version", 0) or 0) != ARTIFACT_VERSION:
            return False, f"artifact version mismatch: {self.meta_path}"
        return True, "artifact present"

    def load(self) -> bool:
        ok, _ = self.artifact_status()
        if not ok:
            return False
        self._state = MemoryState(version=ARTIFACT_VERSION)
        self._load_from_store()
        self.slowPath.resume()
        return True

    def save(self) -> None:
        state = self._require_state()
        state.artifact_created_at = time.time()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._persist_to_store()
        meta = {
            "version": ARTIFACT_VERSION,
            "created_at": state.artifact_created_at,
            "speaker_names": list(state.speaker_names),
            "counts": {
                "transcripts": len(state.transcripts),
                "events": len(state.events),
                "entities": len(state.entities),
                "facts": len(state.facts),
                "session_summaries": len(state.session_summaries),
                "edges": len(state.graph_edges),
                "communities": len(state.communities),
                "traces": len(state.traces),
                "slow_path_jobs": len(state.slow_path_jobs),
                "hm_topics": len(state.hm_topics),
                "hm_episodes": len(state.hm_episodes),
                "hm_facts": len(state.hm_facts),
            },
        }
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)

    def ensure_loaded(self) -> None:
        logger.info("开始 ensure_loaded: 检查引擎状态是否已加载")
        if self._state is not None:
            logger.debug("ensure_loaded: 状态已存在，跳过加载")
            return
        if not self.load():
            logger.error("ensure_loaded 失败: artifact 未加载，请先执行 ingest")
            raise RuntimeError("EBM Python artifact is not loaded. Run ingest first.")
        logger.info("ensure_loaded 完成: 状态加载成功")

    def _ensure_mutable_state(self) -> MemoryState:
        logger.info("开始 _ensure_mutable_state: 确保可变状态可用")
        if self._state is not None:
            logger.debug("_ensure_mutable_state: 状态已存在")
            return self._state
        if self.load():
            logger.debug("_ensure_mutable_state: 从存储加载状态成功")
            return self._require_state()
        logger.warning("_ensure_mutable_state: 无法加载已有状态，执行 reset 创建新状态")
        self.reset()
        logger.info("_ensure_mutable_state 完成: 新状态已创建")
        return self._require_state()

    def _persist_to_store(self) -> None:
        state = self._require_state()
        self._store.reset()
        self._store.upsert_transcripts(state.transcripts)
        self._store.upsert_entities(list(state.entities.values()))
        self._store.upsert_events(list(state.events.values()))
        self._store.upsert_facts(list(state.facts.values()))
        self._store.upsert_session_summaries(list(state.session_summaries.values()))
        self._store.upsert_edges(list(state.graph_edges.values()))
        self._store.upsert_communities(list(state.communities.values()))
        for trace in state.traces:
            trace_id = str(trace.get("trace_id", "") or trace.get("id", "") or _stable_id("trace", trace))
            self._store.append_trace(trace_id, float(trace.get("created_at", time.time()) or time.time()), trace)
        for job in state.slow_path_jobs:
            self._store.upsert_slow_path_job(
                str(job.get("id", "")),
                float(job.get("created_at", time.time()) or time.time()),
                str(job.get("status", "pending")),
                int(job.get("attempts", 0) or 0),
                str(job.get("last_error", "") or ""),
                str(job.get("query", "") or ""),
                job,
            )
        # HyperMem three-level hierarchy
        if state.hm_topics:
            self._store.upsert_hm_topics(list(state.hm_topics.values()))
        if state.hm_episodes:
            self._store.upsert_hm_episodes(list(state.hm_episodes.values()))
        if state.hm_facts:
            self._store.upsert_hm_facts(list(state.hm_facts.values()))
        if state.unified_facts:
            self._store.upsert_unified_facts(list(state.unified_facts.values()))

    def _load_from_store(self) -> None:
        state = self._require_state()
        try:
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, json.JSONDecodeError):
            meta = {}
        state.artifact_created_at = float(meta.get("created_at", 0.0) or 0.0)
        state.speaker_names = [str(name) for name in meta.get("speaker_names", []) or []]
        state.transcripts = self._store.list_transcripts()
        state.entities = {item.id: item for item in self._store.list_entities()}
        state.events = {item.id: item for item in self._store.list_events()}
        state.facts = {item.id: item for item in self._store.list_facts()}
        state.session_summaries = {item.session_key: item for item in self._store.list_session_summaries()}
        state.graph_edges = {item.id: item for item in self._store.list_edges()}
        state.communities = {item.id: item for item in self._store.list_communities()}
        state.traces = self._store.list_traces(200)
        state.slow_path_jobs = self._store.list_slow_path_jobs(500)
        # HyperMem three-level hierarchy
        state.hm_topics = {item.id: item for item in self._store.list_hm_topics()}
        state.hm_episodes = {item.id: item for item in self._store.list_hm_episodes()}
        state.hm_facts = {item.id: item for item in self._store.list_hm_facts()}
        state.unified_facts = {item.id: item for item in self._store.list_unified_facts()}
        self._rebuild_runtime_links()

    def _backfill_transcript_vectors(self) -> None:
        state = self._require_state()
        missing_indexes = [
            index for index, entry in enumerate(state.transcripts)
            if getattr(entry, "vector", None) is None
        ]
        if not missing_indexes:
            return
        vectors = self._embedder.embed_texts([state.transcripts[index].content for index in missing_indexes])
        for index, vector in zip(missing_indexes, vectors, strict=True):
            state.transcripts[index].vector = vector
        self._store.upsert_transcripts([state.transcripts[index] for index in missing_indexes])

    def _ensure_transcript_vectors_loaded(self) -> None:
        state = self._require_state()
        missing_count = sum(1 for entry in state.transcripts if getattr(entry, "vector", None) is None)
        if missing_count <= 0:
            return
        logger.info("检测到缺失 transcript 向量，按需补齐: 数量=%d", missing_count)
        self._backfill_transcript_vectors()

    def _rebuild_runtime_links(self) -> None:
        state = self._require_state()
        state.adjacency = {}
        state.entity_to_events = {}
        for edge in state.graph_edges.values():
            state.adjacency.setdefault(edge.from_id, set()).add(edge.to_id)
            state.adjacency.setdefault(edge.to_id, set()).add(edge.from_id)
            if edge.from_id.startswith("entity:") and edge.to_id.startswith("event:"):
                state.entity_to_events.setdefault(edge.from_id, set()).add(edge.to_id)
            if edge.to_id.startswith("entity:") and edge.from_id.startswith("event:"):
                state.entity_to_events.setdefault(edge.to_id, set()).add(edge.from_id)
        rebuildIndices(state)

    def _resolve_session_file(self, session_id: str, session_key: str = "", explicit_session_file: str = "") -> str:
        if explicit_session_file:
            return explicit_session_file
        state = self._state
        if state is not None:
            for entry in reversed(state.transcripts):
                if entry.session_id == session_id and entry.session_file:
                    return entry.session_file
        return self._store.lookup_session_file(session_id) or ""

    def _next_transcript_message_index(self, session_id: str) -> int:
        state = self._state
        if state is not None:
            indexes = [entry.turn_index for entry in state.transcripts if entry.session_id == session_id]
            if indexes:
                return max(indexes) + 1
        return self._store.get_next_transcript_message_index(session_id)

    def _normalize_message_record(self, message: Any, *, fallback_timestamp: int, message_index: int, start_line: int | None = None, end_line: int | None = None) -> dict[str, Any]:
        role = str(message.get("role", "") or getattr(message, "role", "") or "user")
        text = messageToText(message) if isinstance(message, dict) else normalize_whitespace(str(getattr(message, "content", "") or ""))
        timestamp = int(message.get("timestamp", fallback_timestamp) if isinstance(message, dict) else getattr(message, "timestamp", fallback_timestamp) or fallback_timestamp)
        return {
            "role": role,
            "text": text,
            "timestamp": timestamp,
            "message_index": message_index,
            "start_line": start_line,
            "end_line": end_line,
        }

    def _store_transcript_rows(
        self,
        *,
        session_id: str,
        session_key: str,
        session_file: str,
        rows: Sequence[dict[str, Any]],
        date_time: str = "",
    ) -> list[TranscriptEntry]:
        state = self._ensure_mutable_state()
        entries: list[TranscriptEntry] = []
        for row in rows:
            text = normalize_whitespace(str(row.get("text", "") or ""))
            if not text:
                continue
            message_index = int(row.get("message_index", 0) or 0)
            role = str(row.get("role", "") or "user")
            start_line = row.get("start_line")
            end_line = row.get("end_line")
            evidence = EvidenceRef(
                sessionFile=session_file,
                messageIndex=message_index,
                startLine=int(start_line) if start_line is not None else None,
                endLine=int(end_line) if end_line is not None else None,
                snippet=text[:200],
                dateTime=date_time,
                speaker=role,
            )
            entry = TranscriptEntry(
                id=f"transcript:{_stable_id(session_id, row.get('timestamp', 0), message_index, text)}",
                session_key=session_key,
                date_time=date_time,
                turn_index=message_index,
                speaker=role,
                text=text,
                content=text,
                tokens=tokenize(text),
                entity_ids=[],
                evidence=evidence,
                session_id=session_id,
                session_file=session_file,
                created_at=int(row.get("timestamp", 0) or 0),
            )
            entries.append(entry)
        if not entries:
            return []
        existing: dict[str, TranscriptEntry] = {entry.id: entry for entry in state.transcripts}
        for entry in entries:
            existing[entry.id] = entry
        state.transcripts = sorted(existing.values(), key=lambda item: (item.session_id, item.turn_index, item.created_at))
        self._store.upsert_transcripts(entries)
        return entries

    def ingest_sessions(self, sessions: Sequence[Any], speakers: list[str] | None = None, speaker_a: str = "", speaker_b: str = "") -> dict[str, int]:
        self.reset()
        state = self._require_state()
        if speakers:
            state.speaker_names = unique_preserve_order(speakers)
        elif speaker_a or speaker_b:
            state.speaker_names = unique_preserve_order([s for s in [speaker_a, speaker_b] if s])
        else:
            state.speaker_names = []

        for session in sessions:
            entries, event_ids = self._register_session_entries(session)
            turn_input = self._build_turn_input(
                session_id=session.session_key,
                session_key=session.session_key,
                session_file=session.session_key,
                entries=entries,
                query_fallback=session.session_key,
            )
            job_id, payload = self._build_slow_path_job_payload(
                turn_input=turn_input,
                date_time=session.date_time,
                entries=entries,
                event_ids=event_ids,
            )
            self.slowPath.enqueue(job_id, payload)

        self.slowPath.drain(self._execute_slow_path_job)
        state.slow_path_jobs = self.slowPath.jobs()

        # Embed the primary state before graph-promotion so promoted facts can reuse
        # their source event vectors instead of issuing duplicate embedding requests.
        self._embed_state()

        if not self.config.benchmarkFastIngest:
            # Run session-end refinement once after all sessions (not per-session)
            self._run_session_end_refinement("all")
            self._rebuild_communities()
            self._embed_state()
            # HyperMem embedding propagation (after all nodes are embedded)
            self._propagate_hm_embeddings()
        elif self._llm.is_enabled and not self._slowpath_llm_disabled:
            self._extract_profile_facts()
        self._rebuild_indices()
        self.save()
        return {
            "sessions_ingested": len(sessions),
            "events": len(state.events),
            "entities": len(state.entities),
            "facts": len(state.facts),
            "summaries": len(state.session_summaries),
            "edges": len(state.graph_edges),
            "communities": len(state.communities),
            "hm_topics": len(state.hm_topics),
            "hm_episodes": len(state.hm_episodes),
            "hm_facts": len(state.hm_facts),
            "unified_facts": len(state.unified_facts),
        }

    def bootstrap(self, params: dict[str, Any]) -> dict[str, Any]:
        logger.info("开始 bootstrap: params_keys=%s", list(params.keys()))
        self._ensure_mutable_state()
        session_id = str(params.get("sessionId", "") or params.get("sessionKey", "") or "")
        session_key = str(params.get("sessionKey", "") or session_id)
        session_file = str(params.get("sessionFile", "") or "")
        logger.debug("bootstrap: session_id=%s session_key=%s session_file=%s", session_id, session_key, session_file)
        if not session_file or not Path(session_file).exists():
            logger.warning("bootstrap: session_file 不存在或为空，跳过导入")
            return {"bootstrapped": True, "importedMessages": 0}
        imported = 0
        try:
            lines = Path(session_file).read_text(encoding="utf-8").splitlines()
            logger.debug("bootstrap: 读取文件行数=%d", len(lines))
            rows: list[dict[str, Any]] = []
            next_message_index = 0
            for line_idx, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict) or record.get("type") != "message":
                    continue
                message = record.get("message")
                if not isinstance(message, dict):
                    continue
                normalized = self._normalize_message_record(
                    message,
                    fallback_timestamp=int(time.time() * 1000),
                    message_index=next_message_index,
                    start_line=line_idx + 1,
                    end_line=line_idx + 1,
                )
                if not normalized["text"]:
                    continue
                rows.append(normalized)
                next_message_index += 1
            if rows:
                imported = len(self._store_transcript_rows(session_id=session_id, session_key=session_key, session_file=session_file, rows=rows, date_time=session_key or session_id))
                logger.debug("bootstrap: 导入 transcript 行数=%d", imported)
                self.save()
        except Exception:
            logger.error("bootstrap 失败: session_file=%s", session_file, exc_info=True)
            raise
        logger.info("bootstrap 完成: importedMessages=%d", imported)
        return {"bootstrapped": True, "importedMessages": imported}

    def ingest(self, params: dict[str, Any]) -> dict[str, Any]:
        logger.info("开始 ingest: session_id=%s", params.get("sessionId", ""))
        self._ensure_mutable_state()
        message = params.get("message")
        if not isinstance(message, dict):
            logger.warning("ingest: message 不是 dict 类型，跳过")
            return {"ingested": False}
        self.ingestBatch(
            {
                "sessionId": params.get("sessionId", ""),
                "sessionKey": params.get("sessionKey"),
                "sessionFile": params.get("sessionFile", ""),
                "messages": [message],
            }
        )
        logger.info("ingest 完成: session_id=%s", params.get("sessionId", ""))
        return {"ingested": True}

    def ingestBatch(self, params: dict[str, Any]) -> dict[str, Any]:
        logger.info("开始 ingestBatch: session_id=%s message_count=%d", params.get("sessionId", ""), len(params.get("messages", []) or []))
        self._ensure_mutable_state()
        session_id = str(params.get("sessionId", "") or params.get("sessionKey", "") or "")
        session_key = str(params.get("sessionKey", "") or session_id)
        session_file = self._resolve_session_file(session_id, session_key, str(params.get("sessionFile", "") or ""))
        messages = list(params.get("messages", []) or [])
        logger.debug("ingestBatch: session_id=%s session_key=%s messages=%d", session_id, session_key, len(messages))
        base_message_index = self._next_transcript_message_index(session_id)
        rows = [
            self._normalize_message_record(
                message,
                fallback_timestamp=int(time.time() * 1000),
                message_index=base_message_index + index,
            )
            for index, message in enumerate(messages)
        ]
        entries = self._store_transcript_rows(
            session_id=session_id,
            session_key=session_key,
            session_file=session_file,
            rows=rows,
            date_time=session_key or session_id,
        )
        self.save()
        logger.info("ingestBatch 完成: session_id=%s ingestedCount=%d", session_id, len(entries))
        return {"ingestedCount": len(entries)}

    def after_turn(
        self,
        *,
        session_id: str,
        session_file: str,
        messages: Sequence[dict[str, Any]],
        pre_prompt_message_count: int = 0,
        date_time: str = "",
        session_key: str = "",
    ) -> None:
        logger.info("开始 afterTurn: session_id=%s message_count=%d pre_prompt=%d", session_id, len(messages), pre_prompt_message_count)
        self._ensure_mutable_state()
        state = self._require_state()
        session_key = str(session_key or session_file or session_id)
        line_map = self._build_line_map(session_file, len(messages))
        transcript_rows = [
            self._normalize_message_record(
                message,
                fallback_timestamp=int(time.time() * 1000),
                message_index=index,
                start_line=line_map.get(index),
                end_line=line_map.get(index),
            )
            for index, message in enumerate(messages)
        ]
        self._store_transcript_rows(
            session_id=session_id,
            session_key=str(session_key or session_id),
            session_file=session_file,
            rows=transcript_rows,
            date_time=date_time or session_file or session_id,
        )
        turn_rows = transcript_rows[pre_prompt_message_count:]
        entries = payloadToEntries(
            [
                {
                    "id": f"turn:{_stable_id(session_id, row['message_index'], row['text'])}",
                    "session_id": session_id,
                    "session_key": session_key,
                    "session_file": session_file,
                    "date_time": date_time or session_file or session_id,
                    "turn_index": row["message_index"],
                    "speaker": row["role"],
                    "text": row["text"],
                    "content": row["text"],
                    "tokens": tokenize(row["text"]),
                    "entity_ids": [],
                    "created_at": row["timestamp"],
                    "evidence": {
                        "sessionFile": session_file,
                        "messageIndex": row["message_index"],
                        "startLine": row.get("start_line"),
                        "endLine": row.get("end_line"),
                        "speaker": row["role"],
                        "dateTime": date_time or session_file or session_id,
                        "snippet": row["text"][:200],
                    },
                }
                for row in turn_rows
                if row.get("text")
            ]
        )
        if not entries:
            logger.warning("afterTurn: 无有效 entries，跳过 slow path")
            self.save()
            return
        turn_input = self._build_turn_input(
            session_id=session_id,
            session_key=session_key,
            session_file=session_file,
            entries=entries,
            query_fallback=session_key,
        )
        job_id, payload = self._build_slow_path_job_payload(
            turn_input=turn_input,
            date_time=date_time or session_file or session_id,
            entries=entries,
            event_ids=[],
        )
        self.slowPath.enqueue(job_id, payload)
        state.slow_path_jobs = self.slowPath.jobs()
        self.save()
        logger.info("afterTurn 完成: session_id=%s entries=%d job_id=%s", session_id, len(entries), job_id)

    def afterTurn(self, params: dict[str, Any]) -> None:
        self.after_turn(
            session_id=str(params.get("sessionId", "") or ""),
            session_key=str(params.get("sessionKey", "") or ""),
            session_file=str(params.get("sessionFile", "") or ""),
            messages=list(params.get("messages", []) or []),
            pre_prompt_message_count=int(params.get("prePromptMessageCount", 0) or 0),
            date_time=str(params.get("sessionFile", "") or ""),
        )

    def _flush_slow_path_foreground(self) -> None:
        logger.info("开始 flush_slow_path: 排空慢路径队列")
        with self._runtime_lock:
            self.ensure_loaded()
        status_before = self.slowPath.status()
        total_jobs = int(status_before.get("pending", 0) or 0) + int(status_before.get("running", 0) or 0)
        if total_jobs > 0:
            from ebm_context_engine.client import _EmbeddingProgressBar
            job_progress = _EmbeddingProgressBar("slow_path_jobs", total_jobs)
            original_executor = self._execute_slow_path_job_with_runtime_lock

            def _tracked_executor(payload: dict[str, Any]) -> None:
                original_executor(payload)
                job_progress.update(1)

            self.slowPath.drain(_tracked_executor)
            job_progress.close()
        else:
            self.slowPath.drain(self._execute_slow_path_job_with_runtime_lock)
        with self._runtime_lock:
            self._require_state().slow_path_jobs = self.slowPath.jobs()
            if not self.config.benchmarkFastIngest:
                logger.debug("flush_slow_path: 重建社区和嵌入")
                self._rebuild_communities()
                self._embed_state()
            self._rebuild_indices()
            self.save()
            remaining_jobs = len(self._require_state().slow_path_jobs)
        logger.info("flush_slow_path 完成: 剩余 jobs=%d", remaining_jobs)

    def _background_flush_worker(self) -> None:
        try:
            self._flush_slow_path_foreground()
        except Exception as error:
            logger.error("后台 flush_slow_path 失败: %s", error, exc_info=True)
            with self._flush_state_lock:
                self._flush_last_error = str(error)
        finally:
            with self._flush_state_lock:
                self._flush_active = False
                self._flush_thread = None

    def _execute_slow_path_job_with_runtime_lock(self, payload: dict[str, Any]) -> None:
        with self._runtime_lock:
            self._execute_slow_path_job(payload)

    def flush_slow_path(self) -> dict[str, int]:
        logger.info("开始 flush_slow_path: 请求后台排空 slow path")
        with self._flush_state_lock:
            existing = self._flush_thread
            if existing is not None and existing.is_alive():
                return self.get_slow_path_status()
            self._flush_last_error = ""
            self._flush_active = True
            worker = threading.Thread(
                target=self._background_flush_worker,
                name="ebm-slowpath-flush",
                daemon=True,
            )
            self._flush_thread = worker
            worker.start()
        return self.get_slow_path_status()

    def flushSlowPath(self) -> dict[str, int]:
        return self.flush_slow_path()

    def retry_failed(self) -> int:
        logger.info("开始 retry_failed: 重试失败的 slow path 任务")
        with self._runtime_lock:
            self.ensure_loaded()
        retried = self.slowPath.retryFailed(self._execute_slow_path_job_with_runtime_lock)
        with self._runtime_lock:
            state = self._require_state()
            state.slow_path_jobs = self.slowPath.jobs()
            if retried:
                logger.debug("retry_failed: 成功重试 %d 个任务，重建索引", retried)
                if not self.config.benchmarkFastIngest:
                    self._rebuild_communities()
                    self._embed_state()
                self._rebuild_indices()
                self.save()
            else:
                logger.debug("retry_failed: 无需重试的任务")
        logger.info("retry_failed 完成: retried=%d", retried)
        return retried

    def retryFailed(self) -> int:
        return self.retry_failed()


    def _log_query_phase(
        self,
        phase: str,
        payload: dict[str, Any],
        *,
        level: int = logging.DEBUG,
    ) -> None:
        try:
            rendered = json.dumps(payload, ensure_ascii=False)
        except Exception:
            rendered = str(payload)
        self.logger.log(level, "EBM query phase: %s %s", phase, rendered)
    def _filtered_context_hits(
        self,
        question: str,
        classification: ClassificationResult,
        hits: Sequence[RecallHit],
    ) -> list[RecallHit]:
        filtered: list[RecallHit] = []
        for hit in hits:
            title_lower = hit.title.lower()
            if title_lower.startswith("chain:") or "relationship chain" in title_lower:
                continue
            if classification.intent == "temporal":
                if hit.source == "summary":
                    continue
                if hit.source == "graph" and not title_lower.startswith("event:"):
                    continue
            filtered.append(hit)

        filtered.sort(key=lambda item: item.score, reverse=True)
        return filtered[:self.config.contextHitsLimit]

    def _retrieve_structured_slot_hits(
        self,
        question: str,
        classification: ClassificationResult,
        query_vector: np.ndarray | None,
        *,
        limit: int = 6,
    ) -> list[RecallHit]:
        state = self._require_state()
        threshold = getattr(self.config, "structuredVerificationConfidence", 0.85)
        candidates = [
            fact for fact in state.facts.values()
            if fact.status == "active" and fact.confidence >= threshold and str(fact.source or "").endswith("-verified")
        ]
        if not candidates:
            return []
        lowered_entities = {entity.strip().lower() for entity in classification.entities if entity.strip()}
        if lowered_entities:
            prioritized = [
                fact for fact in candidates
                if fact.subject.strip().lower() in lowered_entities
            ]
            if prioritized:
                candidates = prioritized
        if classification.target_slots:
            slot_prioritized = [
                fact for fact in candidates
                if any(fact.key == slot or fact.key.startswith(slot + ".") or slot in fact.key for slot in classification.target_slots)
            ]
            if slot_prioritized:
                candidates = slot_prioritized
        ranked = rank_text_records(
            question,
            candidates,
            query_vector=query_vector,
            get_text=lambda fact: f"{fact.subject} {fact.key} {fact.value}",
            get_vector=lambda fact: fact.vector,
            rrf_k=40,
        )
        hits: list[RecallHit] = []
        for fact, score in ranked[: max(limit * 2, 8)]:
            hits.append(
                RecallHit(
                    id=fact.id,
                    title=f"{fact.scope} / {fact.key}",
                    content=f"{fact.subject}: {fact.value}",
                    source="ledger",
                    score=float(score),
                    reason="structured slot candidate",
                    evidence=fact.evidence,
                    session_key=fact.session_key,
                    turn_index=fact.turn_index,
                    verificationNote=fact.source,
                )
            )
        reranked, _debug = self._rerank_hits(
            question,
            ClassificationResult(
                intent="generic",
                complexity="standard",
                confidence=1.0,
                source="structured",
                weights={},
                entities=[],
                focus_terms=[],
                reasoning_modes=["fact_lookup"],
                time_scope="",
            ),
            hits,
        )
        return reranked[:limit]

    def _build_structured_slot_context(self, question: str, hits: Sequence[RecallHit]) -> str:
        lines = [
            "Answer only from the structured facts below.",
            "If the facts are sufficient, answer directly and concisely.",
            "If they are insufficient, say Information not found.",
            "",
            "Structured Facts:",
        ]
        for index, hit in enumerate(hits[:4], start=1):
            snippet = hit.evidence.snippet if hit.evidence and hit.evidence.snippet else ""
            lines.append(f"{index}. {hit.title} => {hit.content}" + (f" | evidence={_clip(snippet, 140)}" if snippet else ""))
        lines.extend(["", f"Question: {question}"])
        return "\n".join(lines)

    def _should_attempt_structured_answer(
        self,
        question: str,
        classification: ClassificationResult,
        hits: Sequence[RecallHit],
    ) -> tuple[bool, str]:
        if not hits:
            return False, "no_hits"
        if classification.intent == "temporal":
            return False, "skip_temporal"
        if classification.intent in {"multi_hop", "causal"}:
            return False, "skip_complex_reasoning"
        query_tokens = tokenize(question)
        lowered_entities = {
            entity.strip().lower()
            for entity in classification.entities
            if entity.strip()
        }
        for hit in hits[:3]:
            title_overlap = keyword_overlap(query_tokens, tokenize(hit.title))
            content_overlap = keyword_overlap(query_tokens, tokenize(hit.content))
            overlap = max(title_overlap, content_overlap)
            entity_match = not lowered_entities or any(
                entity in hit.content.lower() for entity in lowered_entities
            )
            if overlap >= 0.34 and entity_match:
                return True, f"overlap={overlap:.2f}"
        return False, "low_alignment"

    def _answer_from_structured_slots(
        self,
        question: str,
        classification: ClassificationResult,
        hits: Sequence[RecallHit],
    ) -> tuple[str, dict[str, Any]] | None:
        if not hits:
            return None
        context = self._build_structured_slot_context(question, hits)
        try:
            result = self._answer_llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Decide whether the structured facts are sufficient to answer the question. "
                            "Return JSON only with keys {\"sufficient\": boolean, \"answer\": string, "
                            "\"normalized_answer\": string, \"confidence\": number}."
                        ),
                    },
                    {"role": "user", "content": context},
                ],
                temperature=0.0,
                max_tokens=120,
                response_format={"type": "json_object"},
            )
            payload = json.loads(result.content)
        except Exception as error:  # noqa: BLE001
            self.logger.warning("structured slot answer failed: question=%s error=%s", question, error)
            return None
        sufficient = bool(payload.get("sufficient", False))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        answer = str(payload.get("normalized_answer", "") or payload.get("answer", "") or "").strip()
        answer = self._normalize_answer_text(answer)
        if not sufficient or confidence < 0.72 or not answer or "information not found" in answer.lower():
            return None
        return answer, payload

    def _parse_reference_datetime(self, raw: str) -> datetime | None:
        cleaned = normalize_whitespace(str(raw or ""))
        if not cleaned:
            return None
        formats = (
            # LoCoMo-style
            "%I:%M %p on %d %B, %Y",
            "%I:%M %p on %d %b, %Y",
            # Common date formats
            "%d %B %Y",
            "%d %b %Y",
            "%B %d, %Y",
            "%b %d, %Y",
            # ISO 8601 and standard formats
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
        )
        for fmt in formats:
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None

    def _format_calendar_date(self, value: datetime) -> str:
        return f"{value.day} {value.strftime('%B %Y')}"

    def _shift_months(self, value: datetime, delta_months: int) -> datetime:
        month_index = (value.month - 1) + delta_months
        year = value.year + (month_index // 12)
        month = (month_index % 12) + 1
        day = min(
            value.day,
            [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1],
        )
        return value.replace(year=year, month=month, day=day)

    def _normalize_temporal_expression(
        self,
        text: str,
        reference: datetime,
    ) -> str | None:
        lowered = normalize_whitespace(text).lower()
        if not lowered:
            return None
        explicit_match = re.search(
            r"\b(\d{1,2})\s+"
            r"(january|february|march|april|may|june|july|august|september|october|november|december|"
            r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
            r"\s*,?\s*(\d{4})\b",
            lowered,
            flags=re.IGNORECASE,
        )
        if explicit_match:
            try:
                parsed = datetime.strptime(
                    f"{explicit_match.group(1)} {explicit_match.group(2)} {explicit_match.group(3)}",
                    "%d %B %Y",
                )
            except ValueError:
                try:
                    parsed = datetime.strptime(
                        f"{explicit_match.group(1)} {explicit_match.group(2)} {explicit_match.group(3)}",
                        "%d %b %Y",
                    )
                except ValueError:
                    parsed = None
            if parsed is not None:
                return self._format_calendar_date(parsed)

        for phrase, delta_days in (
            ("day before yesterday", -2),
            ("yesterday", -1),
            ("today", 0),
            ("tomorrow", 1),
            ("day after tomorrow", 2),
            ("last week", -7),
            ("next week", 7),
        ):
            if phrase in lowered:
                return self._format_calendar_date(reference + timedelta(days=delta_days))

        if "last month" in lowered:
            return self._format_calendar_date(self._shift_months(reference, -1))
        if "next month" in lowered:
            return self._format_calendar_date(self._shift_months(reference, 1))
        if "last year" in lowered:
            try:
                shifted = reference.replace(year=reference.year - 1)
            except ValueError:
                shifted = reference.replace(year=reference.year - 1, day=28)
            return self._format_calendar_date(shifted)
        if "next year" in lowered:
            try:
                shifted = reference.replace(year=reference.year + 1)
            except ValueError:
                shifted = reference.replace(year=reference.year + 1, day=28)
            return self._format_calendar_date(shifted)

        weekday_match = re.search(
            r"\b(last|next|this)\s+"
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            lowered,
            flags=re.IGNORECASE,
        )
        if weekday_match:
            direction = weekday_match.group(1).lower()
            target_weekday = WEEKDAY_TO_INDEX[weekday_match.group(2).lower()]
            current_weekday = reference.weekday()
            if direction == "last":
                delta = (current_weekday - target_weekday) % 7 or 7
                return self._format_calendar_date(reference - timedelta(days=delta))
            if direction == "next":
                delta = (target_weekday - current_weekday) % 7 or 7
                return self._format_calendar_date(reference + timedelta(days=delta))
            delta = target_weekday - current_weekday
            return self._format_calendar_date(reference + timedelta(days=delta))

        return None

    def _fast_temporal_grounding(
        self,
        question: str,
        classification: ClassificationResult,
        hits: Sequence[RecallHit],
        *,
        transcript_context: str = "",
    ) -> tuple[str, dict[str, Any]] | None:
        lowered_entities = {
            entity.strip().lower()
            for entity in classification.entities
            if entity.strip()
        }
        query_tokens = tokenize(question)
        candidates: list[tuple[float, str, str, str]] = []
        for raw_line in transcript_context.splitlines():
            match = re.match(r"^\[(?P<date>[^\]]+)\]\s+\[(?P<speaker>[^\]]+)\]:\s*(?P<text>.+)$", raw_line.strip())
            if not match:
                continue
            text = match.group("text").strip()
            if not contains_temporal_marker(text):
                continue
            speaker = match.group("speaker").strip()
            combined = f"{speaker} {text}"
            overlap = keyword_overlap(query_tokens, tokenize(combined))
            if lowered_entities and speaker.lower() in lowered_entities:
                overlap += 0.12
            if overlap <= 0:
                continue
            candidates.append((overlap, match.group("date"), speaker, text))

        for hit in hits[:4]:
            if not hit.evidence or not hit.evidence.snippet:
                continue
            snippet = hit.evidence.snippet.strip()
            if not contains_temporal_marker(snippet):
                continue
            combined = f"{hit.evidence.speaker or ''} {snippet}".strip()
            overlap = keyword_overlap(query_tokens, tokenize(combined))
            if lowered_entities and (hit.evidence.speaker or "").strip().lower() in lowered_entities:
                overlap += 0.12
            if overlap <= 0:
                continue
            candidates.append((overlap, hit.evidence.dateTime or "", hit.evidence.speaker or "", snippet))

        for score, raw_date, speaker, text in sorted(candidates, key=lambda item: item[0], reverse=True):
            reference = self._parse_reference_datetime(raw_date)
            if reference is None:
                continue
            normalized = self._normalize_temporal_expression(text, reference)
            if not normalized:
                continue
            return normalized, {
                "mode": "fast",
                "score": round(score, 4),
                "speaker": speaker,
                "reference_datetime": raw_date,
                "evidence_text": text,
            }
        return None

    def _build_temporal_grounding_context(
        self,
        question: str,
        hits: Sequence[RecallHit],
        *,
        transcript_context: str = "",
    ) -> str:
        state = self._require_state()
        lines = [
            "Resolve the question using only the temporal evidence below.",
            "Use dated transcript lines as the reference datetime for any relative expression.",
            "Convert to the most concrete supported answer when the evidence allows it.",
            "If the evidence is insufficient, answer Information not found.",
            "",
            f"Question: {question}",
            "",
            "Temporal Evidence:",
        ]
        seen_entries: set[tuple[str, int]] = set()
        evidence_blocks = 0
        for hit in hits[:4]:
            block_lines = [f"- Hit: {hit.title} => {hit.content}"]
            if hit.evidence and hit.evidence.snippet:
                block_lines.append(f"  Snippet: {_clip(hit.evidence.snippet, 180)}")
            session_key = hit.session_key or (hit.evidence.sessionFile if hit.evidence else "")
            if session_key and hit.turn_index is not None:
                neighbors = [
                    entry
                    for entry in state.transcripts
                    if entry.session_key == session_key and abs(entry.turn_index - hit.turn_index) <= 1
                ]
                neighbors.sort(key=lambda item: item.turn_index)
                for neighbor in neighbors[:3]:
                    neighbor_key = (neighbor.session_key, neighbor.turn_index)
                    if neighbor_key in seen_entries:
                        continue
                    seen_entries.add(neighbor_key)
                    date_part = f"[{neighbor.date_time}] " if neighbor.date_time else ""
                    block_lines.append(f"  Transcript: {date_part}[{neighbor.speaker}] {neighbor.text[:320]}")
            if len(block_lines) == 1 and not (hit.evidence and hit.evidence.snippet):
                continue
            lines.extend(block_lines)
            lines.append("")
            evidence_blocks += 1
        if transcript_context.strip():
            lines.append("Transcript Windows:")
            lines.append(transcript_context[:1400])
            lines.append("")
        return "\n".join(lines) if evidence_blocks or transcript_context.strip() else ""

    def _answer_from_temporal_evidence(
        self,
        question: str,
        hits: Sequence[RecallHit],
        *,
        transcript_context: str = "",
    ) -> tuple[ChatResult, dict[str, Any]] | None:
        """Resolve temporal questions using LLM only (no regex shortcuts)."""
        context = self._build_temporal_grounding_context(
            question,
            hits,
            transcript_context=transcript_context,
        )
        if not context:
            return None
        try:
            try:
                result = self._answer_llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Resolve temporal questions from evidence only. "
                                "Return JSON only with keys {\"sufficient\": boolean, \"answer\": string, "
                                "\"normalized_answer\": string, \"confidence\": number}. "
                                "Use transcript timestamps as reference datetimes for relative expressions. "
                                "Convert relative time expressions (e.g. 'last year', '3 months ago') to absolute dates using the timestamps in evidence. "
                                "Do not infer beyond the evidence."
                            ),
                        },
                        {"role": "user", "content": context},
                    ],
                    temperature=0.0,
                    max_tokens=120,
                    response_format={"type": "json_object"},
                )
            except TypeError:
                result = self._answer_llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Resolve temporal questions from evidence only. "
                                "Use transcript timestamps as reference datetimes for relative expressions. "
                                "Convert relative time expressions to absolute dates using the timestamps. "
                                "Do not infer beyond the evidence."
                            ),
                        },
                        {"role": "user", "content": context},
                    ],
                    temperature=0.0,
                    max_tokens=120,
                )
        except Exception as error:  # noqa: BLE001
            self.logger.warning("temporal grounding failed: question=%s error=%s", question, error)
            return None
        from ebm_context_engine.retrieval.intent_router import _robust_json_parse
        payload = _robust_json_parse(result.content)
        if not isinstance(payload, dict):
            self.logger.warning("temporal grounding JSON parse failed: content=%s", result.content[:200])
            return None
        sufficient = bool(payload.get("sufficient", False))
        try:
            confidence = float(payload.get("confidence", 0.0) or 0.0)
        except (ValueError, TypeError):
            confidence = 0.0
        answer = str(payload.get("normalized_answer", "") or payload.get("answer", "") or "").strip()
        answer = self._normalize_answer_text(answer)
        if not sufficient or confidence < 0.5 or not answer or "information not found" in answer.lower():
            return None
        payload["context"] = context
        return (
            ChatResult(
                content=answer,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            ),
            payload,
        )

    def query(self, question: str, use_aaak: bool = False) -> PythonEbmQueryResult:
        logger.info("开始 query: question='%s' use_aaak=%s", question[:80], use_aaak)
        started_at = int(time.time() * 1000)
        query_started_at = time.perf_counter()
        stage_timings_ms: dict[str, float] = {}

        stage_started_at = time.perf_counter()
        was_loaded = self._state is not None
        self.ensure_loaded()
        stage_timings_ms["ensure_loaded"] = round((time.perf_counter() - stage_started_at) * 1000, 2)

        state = self._require_state()
        normalized_question = normalize_whitespace(question).lower()

        # Embed query first so we can use the vector for intent classification
        stage_started_at = time.perf_counter()
        query_tokens = tokenize(question)
        query_vector = self._embedder.embed_text(question)
        stage_timings_ms["embed_query"] = round((time.perf_counter() - stage_started_at) * 1000, 2)
        self._log_query_phase(
            "embed_query",
            {
                "question": question,
                "query_token_count": len(query_tokens),
                "query_vector_dim": int(len(query_vector)) if query_vector is not None else 0,
            },
        )

        stage_started_at = time.perf_counter()
        cached = self._intent_cache.get(normalized_question)
        if cached is None:
            embed_fn = self._embedder.embed_text if self._embedder.is_enabled else None
            # Use LLM for intent classification (like OpenViking), with embedding fallback
            intent_llm = self._answer_llm if self._answer_llm.is_enabled and not self._answer_llm_disabled else None
            cached = classify_query(
                question,
                llm_client=intent_llm,
                query_vector=query_vector,
                embed_fn=embed_fn,
                state=state,
            )
            self._intent_cache[normalized_question] = cached
            classification, plan, classify_debug = cached
            logger.debug("query: 意图分类完成: intent=%s complexity=%s source=%s answer_type=%s",
                         classification.intent, classification.complexity, classification.source,
                         getattr(classification, "answer_type", ""))
        else:
            classification, plan, _cached_debug = cached
            logger.debug("query: 意图分类命中缓存: intent=%s complexity=%s", classification.intent, classification.complexity)
            classify_debug = {
                "source": "cache",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "used_llm": False,
                "fallback_used": False,
            }
        stage_timings_ms["classify"] = round((time.perf_counter() - stage_started_at) * 1000, 2)

        stage_started_at = time.perf_counter()
        structured_hits = self._retrieve_structured_slot_hits(question, classification, query_vector, limit=6)
        stage_timings_ms["structured_path"] = round((time.perf_counter() - stage_started_at) * 1000, 2)
        self._log_query_phase(
            "structured_path",
            {
                "candidate_count": len(structured_hits),
                "attempted": False,
                "reason": "local_only",
                "used": False,
                "answer": "",
                "confidence": 0.0,
            },
            level=logging.INFO,
        )

        stage_timings_ms["fast_path"] = 0.0

        use_hypermem = (not self.config.benchmarkFastIngest) and bool(state.hm_topics and state.hm_episodes and state.hm_facts)
        use_progressive = bool(state.unified_facts)
        logger.debug("query: use_progressive=%s use_hypermem=%s", use_progressive, use_hypermem)
        verify_reasons: list[str] = []
        direct_answer = None
        direct_confidence = 0.0
        used_direct = False
        packet = None

        stage_started_at = time.perf_counter()
        matched_entity_ids = self._match_entities(question, classification.entities)
        logger.debug("query: 匹配实体数=%d", len(matched_entity_ids))

        # ── Progressive Recall (unified 3-layer path) ──
        graph_hits: list[RecallHit] = []
        community_hits: list[RecallHit] = []
        ledger_hits: list[RecallHit] = []
        summary_hits: list[RecallHit] = []
        hypermem_hits: list[RecallHit] = []
        c2f_fact_results: list = []
        c2f_episode_results: list = []
        c2f_topic_results: list = []
        progressive_hits: list[RecallHit] = []

        if use_progressive:
            recaller = ProgressiveRecaller(
                state,
                self._embedder.embed_text if self._embedder.is_enabled else None,
                self._reranker.rerank if self._reranker is not None and self._reranker.is_enabled else None,
                layer0_top_k=self.config.layer0TopK,
                layer1_top_k=self.config.layer1TopK,
                layer2_top_k=self.config.layer2TopK,
            )
            progressive_hits = recaller.recall(
                question, query_vector, classification,
                entity_seed_ids=set(matched_entity_ids) if matched_entity_ids else None,
            )
            stage_timings_ms["progressive_recall"] = round((time.perf_counter() - stage_started_at) * 1000, 2)
            self._log_query_phase(
                "progressive_recall",
                {
                    "progressive_hit_count": len(progressive_hits),
                    "top_hits": [self._hit_payload(hit) for hit in progressive_hits[:3]],
                },
                level=logging.INFO,
            )
        else:
            # ── Legacy 5-way parallel recall ──
            self._log_query_phase(
                "graph_recall_input",
                {
                    "question": question,
                    "matched_entity_ids": matched_entity_ids,
                    "graph_top_k": plan.graph_top_k,
                    "community_top_k": plan.community_top_k,
                    "intent": classification.intent,
                    "complexity": classification.complexity,
                },
            )
            graph_hits, community_hits = self.salientMemoryGraph.recall(
                question,
                query_vector=query_vector,
                graph_top_k=plan.graph_top_k,
                community_top_k=plan.community_top_k,
                matched_entity_ids=matched_entity_ids,
                classification=classification,
            )
            stage_timings_ms["graph_recall"] = round((time.perf_counter() - stage_started_at) * 1000, 2)
            self._log_query_phase(
                "graph_recall_output",
                {
                    "graph_hit_count": len(graph_hits),
                    "community_hit_count": len(community_hits),
                    "top_graph_hits": [self._hit_payload(hit) for hit in graph_hits[:3]],
                    "top_community_hits": [self._hit_payload(hit) for hit in community_hits[:2]],
                },
            )

        risk_level = self._detect_risk_level(question)

        if not use_progressive:
            stage_started_at = time.perf_counter()
            ledger_subjects = list({"user", *self._resolve_entity_subjects(matched_entity_ids), *self._derive_ledger_subjects_from_query(question)})
            self._log_query_phase(
                "ledger_recall_input",
                {
                    "question": question,
                    "subjects": ledger_subjects,
                    "limit": plan.ledger_top_k,
                    "risk_level": risk_level,
                    "intent": classification.intent,
                },
            )
            ledger_hits, verify_reasons = self.ledger.recall(
                question,
                query_vector=query_vector,
                limit=plan.ledger_top_k,
                risk_level=risk_level,
                subjects=ledger_subjects,
                classification=classification,
            )
            stage_timings_ms["ledger_recall"] = round((time.perf_counter() - stage_started_at) * 1000, 2)
            self._log_query_phase(
                "ledger_recall_output",
                {
                    "ledger_hit_count": len(ledger_hits),
                    "verify_reason_count": len(verify_reasons),
                    "verify_reasons": verify_reasons[:6],
                    "top_ledger_hits": [self._hit_payload(hit) for hit in ledger_hits[:3]],
                },
            )

            stage_started_at = time.perf_counter()
            summary_hits = self._recall_summaries(query_tokens, query_vector, graph_hits, ledger_hits, plan)
            stage_timings_ms["summary_recall"] = round((time.perf_counter() - stage_started_at) * 1000, 2)

            if use_hypermem:
                stage_started_at = time.perf_counter()
                c2f_result = coarse_to_fine_retrieval(
                    question,
                    query_vector,
                    list(state.hm_topics.values()),
                    list(state.hm_episodes.values()),
                    list(state.hm_facts.values()),
                    topic_k=self.config.c2fTopicK,
                    episode_k=self.config.c2fEpisodeK,
                    fact_k=self.config.c2fFactK,
                    intent=classification.intent,
                )
                c2f_fact_results = c2f_result["facts"]
                c2f_episode_results = c2f_result["episodes"]
                c2f_topic_results = c2f_result["topics"]
                hypermem_hits = c2f_to_recall_hits(c2f_fact_results, c2f_episode_results)
                hypermem_hits.extend(self._episode_results_to_hits(c2f_episode_results))
                stage_timings_ms["c2f_retrieval"] = round((time.perf_counter() - stage_started_at) * 1000, 2)

        stage_started_at = time.perf_counter()
        transcript_context = self._recall_transcript_context(
            question,
            query_vector,
            limit=self.config.transcriptRecallLimit if classification.complexity != "simple" else self.config.transcriptRecallLimitSimple,
        )
        stage_timings_ms["transcript_recall"] = round((time.perf_counter() - stage_started_at) * 1000, 2)
        self._log_query_phase(
            "transcript_recall_output",
            {
                "transcript_chars": len(transcript_context),
                "transcript_preview": _clip(transcript_context, 500),
            },
        )

        stage_started_at = time.perf_counter()
        if use_progressive:
            # Progressive path: hits are already ranked by ProgressiveRecaller
            # Allow a few extra hits if flat vector supplement added diverse results
            progressive_limit = self.config.contextHitsLimit + 2
            context_hits = progressive_hits[:progressive_limit]
            reranked_hits = progressive_hits
            ranked_hits = progressive_hits
            rerank_debug = {"enabled": False, "mode": "progressive", "candidate_count": len(progressive_hits), "used": False}
        else:
            # Legacy path: merge 5 sources
            ranked_hits = self._rank_combined_hits(
                graph_hits,
                ledger_hits,
                summary_hits,
                community_hits,
                extra_hits=list(hypermem_hits) + list(structured_hits),
            )
            reranked_hits, rerank_debug = self._rerank_hits(question, classification, ranked_hits)
            context_hits = self._filtered_context_hits(question, classification, reranked_hits)
        context = self._build_ranked_context(
            question,
            classification,
            context_hits,
            transcript_context=transcript_context,
            use_aaak=use_aaak,
        )
        stage_timings_ms["render_context"] = round((time.perf_counter() - stage_started_at) * 1000, 2)
        self._log_query_phase(
            "render_context",
            {
                "use_progressive": use_progressive,
                "ranked_hit_count_before_rerank": len(ranked_hits),
                "reranked_hit_count": len(reranked_hits),
                "rerank_debug": rerank_debug,
                "summary_hit_count": len(summary_hits),
                "c2f_fact_count": len(c2f_fact_results),
                "c2f_episode_count": len(c2f_episode_results),
                "context_chars": len(context),
                "context_preview": _clip(context, 1000),
            },
            level=logging.INFO,
        )

        answer_source = "llm"
        temporal_payload: dict[str, Any] | None = None

        stage_started_at = time.perf_counter()
        temporal_answer = None
        # Use temporal grounding when intent is temporal OR expected answer is a date/time/duration
        answer_type = getattr(classification, "answer_type", "")
        use_temporal_path = (
            classification.intent == "temporal"
            or answer_type in ("date", "time", "duration")
        )
        if use_temporal_path:
            temporal_answer = self._answer_from_temporal_evidence(
                question,
                context_hits,
                transcript_context=transcript_context,
            )
        stage_timings_ms["temporal_grounding"] = round((time.perf_counter() - stage_started_at) * 1000, 2)

        if temporal_answer is not None:
            answer, returned_temporal_payload = temporal_answer
            if temporal_payload is None:
                temporal_payload = returned_temporal_payload
            answer_source = "temporal-grounded"
            logger.debug("query: 使用时间推理回答 answer_source=%s", answer_source)
            stage_timings_ms["answer_generation"] = 0.0
        else:
            stage_started_at = time.perf_counter()
            answer = self._answer(question, context, use_aaak=use_aaak)
            stage_timings_ms["answer_generation"] = round((time.perf_counter() - stage_started_at) * 1000, 2)
        stage_timings_ms["total"] = round((time.perf_counter() - query_started_at) * 1000, 2)
        self.logger.info(
            "EBM query: use_progressive=%s use_hypermem=%s timings=%s",
            use_progressive,
            use_hypermem,
            json.dumps(stage_timings_ms),
        )
        trace = {
            "trace_id": _stable_id("trace", question, time.time()),
            "query": question,
            "classification": asdict(classification),
            "used_direct_answer": used_direct,
            "direct_confidence": direct_confidence,
            "verify_reasons": verify_reasons,
            "use_progressive": use_progressive,
            "counts": {
                "progressive_hits": len(progressive_hits),
                "graph_hits": len(graph_hits),
                "ledger_hits": len(ledger_hits),
                "summary_hits": len(summary_hits),
                "community_hits": len(community_hits),
                "c2f_facts": len(c2f_fact_results),
                "c2f_episodes": len(c2f_episode_results),
                "c2f_topics": len(c2f_topic_results),
                "reranked_hits": len(reranked_hits),
                "context_hits": len(context_hits),
            },
            "graph_hits": [self._hit_payload(hit) for hit in graph_hits[:8]],
            "ledger_hits": [self._hit_payload(hit) for hit in ledger_hits[:8]],
            "summary_hits": [self._hit_payload(hit) for hit in summary_hits[:4]],
            "community_hits": [self._hit_payload(hit) for hit in community_hits[:4]],
            "reranked_hits": [self._hit_payload(hit) for hit in reranked_hits[:8]],
            "context_hits": [self._hit_payload(hit) for hit in context_hits[:6]],
            "created_at": time.time(),
        }
        state.traces.append(trace)
        state.traces = state.traces[-200:]
        self._store.append_trace(
            str(trace["trace_id"]),
            float(trace["created_at"]),
            {
                "traceId": trace["trace_id"],
                "sessionId": "",
                "query": question,
                "packet": asdict(packet) if packet is not None else {},
                "graphHits": trace["graph_hits"],
                "ledgerHits": trace["ledger_hits"],
                "latencyMs": int(time.time() * 1000) - started_at,
                "createdAt": int(trace["created_at"] * 1000),
            },
        )

        total_prompt = int(classify_debug.get("prompt_tokens", 0) or 0) + answer.prompt_tokens
        total_completion = int(classify_debug.get("completion_tokens", 0) or 0) + answer.completion_tokens
        total_tokens = int(classify_debug.get("total_tokens", 0) or 0) + answer.total_tokens

        logger.info("query 完成: question='%s' answer_source=%s total_tokens=%d context_hits=%d timings=%s", question[:80], answer_source, total_tokens, len(context_hits), json.dumps(stage_timings_ms))
        return PythonEbmQueryResult(
            answer=answer.content,
            context=context,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_tokens,
            answer_source=answer_source,
            debug={
                "answer_source": answer_source,
                "classification": {
                    **asdict(classification),
                    **classify_debug,
                },
                "used_direct_answer": used_direct,
                "direct_confidence": direct_confidence,
                "verify_reasons": verify_reasons,
                "counts": trace["counts"],
                "rerank": rerank_debug,
                "context_chars": len(context),
                "artifact_created_at": state.artifact_created_at,
                "timings_ms": stage_timings_ms,
                "temporal_grounding": temporal_payload or {},
            },
        )

    def assemble(self, params: dict[str, Any]) -> dict[str, Any]:
        logger.info("开始 assemble: session_id=%s", params.get("sessionId", ""))
        self.ensure_loaded()
        started_at = int(time.time() * 1000)
        session_id = str(params.get("sessionId", "") or "")
        session_key = str(params.get("sessionKey", "") or "")
        messages = list(params.get("messages", []) or [])
        fallback_message = messages[-1] if messages else {"role": "user", "content": "", "timestamp": int(time.time() * 1000)}
        query = str(params.get("prompt") or messageToText(fallback_message))
        budget = max(2048, int(params.get("tokenBudget", 16000) or 16000))
        risk_level = self._detect_risk_level(query)
        session_scope = self._resolve_session_scope(session_id, session_key)
        transcript_file = self._resolve_session_file(session_id, session_key)
        message_evidence_by_index = self._build_transcript_evidence_map(session_id, transcript_file, messages)
        query_vector = self._embedder.embed_text(query)
        embed_fn = self._embedder.embed_text if self._embedder.is_enabled else None
        classification, _plan, _debug = classify_query(query, query_vector=query_vector, embed_fn=embed_fn, state=self._require_state())
        logger.debug("assemble: query='%s' budget=%d risk_level=%s intent=%s", query[:80], budget, risk_level, classification.intent)
        matched_entity_ids = self._match_entities(query, classification.entities)
        graph_hits, community_hits = self.salientMemoryGraph.recall(
            query,
            query_vector=query_vector,
            graph_top_k=self.config.graphRecallTopK,
            community_top_k=self.config.communityRecallTopK,
            matched_entity_ids=matched_entity_ids,
            classification=classification,
        )
        degraded_paths: list[str] = []
        effective_graph_hits = graph_hits
        if not effective_graph_hits:
            degraded_paths.append("graph-empty-fallback")
            logger.warning("assemble: 图谱召回为空，使用文本回退检索")
            effective_graph_hits = self._fallback_text_retrieval(session_id, query, self.config.graphRecallTopK)
        ledger_recall_items, verify_reasons = self.ledger.recall(
            query,
            query_vector=query_vector,
            limit=self.config.ledgerRecallTopK,
            risk_level=risk_level,
            subjects=self._resolve_ledger_subjects_for_session(session_id, session_key, query),
            classification=classification,
        )
        promoted_facts = self.ledger.derivePinnedFacts(self.config.pinnedFactsLimit, self._resolve_ledger_subjects_for_session(session_id, session_key, query))
        self.workspace.refreshPinnedContext(session_scope, transcript_file, query, messages, promoted_facts, message_evidence_by_index)
        packet = self.workspace.buildPacket(
            {
                "sessionId": session_id,
                "workspaceId": session_scope,
                "query": query,
                "tokenBudget": budget,
                "graphItems": effective_graph_hits,
                "ledgerItems": ledger_recall_items,
                "graphItemsLimit": self.config.graphItemsLimit,
                "ledgerItemsLimit": self.config.ledgerItemsLimit,
            }
        )
        session_summary_lines = self._build_session_summary_lines(session_id, query, budget - packet.totalEstimatedTokens)
        episodic_lines = self._build_episodic_evidence_lines(effective_graph_hits, ledger_recall_items, self.config.episodicMaxTokens)
        summary_tokens = self._estimate_text_tokens(session_summary_lines)
        episodic_tokens = self._estimate_text_tokens(episodic_lines)
        recent_message_budget = max(256, budget - packet.totalEstimatedTokens - summary_tokens - episodic_tokens)
        recent_messages = self._select_recent_messages(messages, recent_message_budget)
        trace = {
            "traceId": packet.traceId,
            "sessionId": session_id,
            "query": query,
            "budget": budget,
            "packet": asdict(packet),
            "graphHits": [self._hit_payload(hit) for hit in effective_graph_hits],
            "ledgerHits": [self._hit_payload(hit) for hit in ledger_recall_items],
            "degradedPaths": degraded_paths,
            "verifyReasons": verify_reasons,
            "latencyMs": int(time.time() * 1000) - started_at,
            "createdAt": int(time.time() * 1000),
        }
        self._store.write_trace(trace)
        if self._state is not None:
            self._state.traces.append(
                {
                    "trace_id": trace["traceId"],
                    "sessionId": session_id,
                    "query": query,
                    "packet": trace["packet"],
                    "graph_hits": trace["graphHits"],
                    "ledger_hits": trace["ledgerHits"],
                    "created_at": trace["createdAt"] / 1000.0,
                }
            )
            self._state.traces = self._state.traces[-200:]
        logger.debug("assemble: graph_hits=%d ledger_hits=%d community_hits=%d degraded=%s", len(effective_graph_hits), len(ledger_recall_items), len(community_hits), degraded_paths)
        logger.info("assemble 完成: session_id=%s estimatedTokens=%d latencyMs=%d", session_id, packet.totalEstimatedTokens, int(time.time() * 1000) - started_at)
        return {
            "messages": recent_messages,
            "estimatedTokens": packet.totalEstimatedTokens + summary_tokens + episodic_tokens + self._estimate_message_tokens(recent_messages),
            "systemPromptAddition": self._build_system_prompt_addition(trace, session_summary_lines, episodic_lines),
        }

    def compact(self, params: dict[str, Any]) -> dict[str, Any]:
        logger.info("开始 compact: params_keys=%s", list(params.keys()))
        logger.warning("compact: delegateCompactionToRuntime 在 Python 版本中未实现")
        tokens_before_raw = params.get("currentTokenCount", 0)
        try:
            tokens_before = int(tokens_before_raw or 0)
        except (TypeError, ValueError):
            tokens_before = 0
        return {
            "ok": True,
            "compacted": False,
            "reason": "delegateCompactionToRuntime not implemented in Python translation",
            "result": {
                "summary": "",
                "firstKeptEntryId": "",
                "tokensBefore": max(0, tokens_before),
                "details": {
                    "sessionId": str(params.get("sessionId", "") or ""),
                    "sessionKey": str(params.get("sessionKey", "") or ""),
                    "sessionFile": str(params.get("sessionFile", "") or ""),
                    "force": bool(params.get("force", False)),
                    "compactionTarget": params.get("compactionTarget"),
                    "runtimeContextPresent": isinstance(params.get("runtimeContext"), dict),
                },
            },
        }

    def dispose(self) -> None:
        logger.info("开始 dispose: 释放引擎资源")
        logger.info("dispose 完成")
        return None

    def memory_search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        logger.info("开始 memory_search: query='%s' limit=%d", query[:80], limit)
        self.ensure_loaded()
        query_tokens = tokenize(query)
        query_vector = self._embedder.embed_text(query)
        embed_fn = self._embedder.embed_text if self._embedder.is_enabled else None
        classification, plan, _ = classify_query(query, query_vector=query_vector, embed_fn=embed_fn, state=self._require_state())
        logger.debug("memory_search: intent=%s complexity=%s", classification.intent, classification.complexity)
        matched_entity_ids = self._match_entities(query, classification.entities)
        logger.debug("memory_search: 匹配实体数=%d", len(matched_entity_ids))
        graph_hits, community_hits = self.salientMemoryGraph.recall(
            query,
            query_vector=query_vector,
            graph_top_k=max(1, limit // 2),
            community_top_k=max(1, limit // 4),
            matched_entity_ids=matched_entity_ids,
            classification=classification,
        )
        ledger_hits, _ = self.ledger.recall(
            query,
            query_vector=query_vector,
            limit=max(1, limit // 2),
            risk_level=self._detect_risk_level(query),
            subjects=self._resolve_entity_subjects(matched_entity_ids),
            classification=classification,
        )
        hits = [*graph_hits, *ledger_hits, *community_hits]
        hits.sort(key=lambda item: item.score, reverse=True)
        logger.info("memory_search 完成: query='%s' 返回 %d 条结果 (graph=%d ledger=%d community=%d)", query[:80], min(len(hits), limit), len(graph_hits), len(ledger_hits), len(community_hits))
        return [self._hit_payload(hit) for hit in hits[:limit]]

    def memorySearch(self, query: str, sessionId: str = "", limit: int = 10) -> list[dict[str, Any]]:
        return self.memory_search(query, limit)

    def memory_get(self, item_id: str) -> Optional[dict[str, Any]]:
        self.ensure_loaded()
        state = self._require_state()
        for entry in state.transcripts:
            if entry.id == item_id:
                return {"node_type": "TRANSCRIPT", **self._transcript_payload(entry)}
        if item_id in state.entities:
            return self._entity_payload(state.entities[item_id])
        if item_id in state.events:
            return self._event_payload(state.events[item_id])
        if item_id in state.facts:
            return self._fact_payload(state.facts[item_id])
        if item_id in state.hm_topics:
            return self._hm_topic_payload(state.hm_topics[item_id])
        if item_id in state.hm_episodes:
            return self._hm_episode_payload(state.hm_episodes[item_id])
        if item_id in state.hm_facts:
            return self._hm_fact_payload(state.hm_facts[item_id])
        if item_id in state.unified_facts:
            return self._unified_fact_payload(state.unified_facts[item_id])
        for summary in state.session_summaries.values():
            if item_id in {summary.id, summary.session_key, summary.session_id, summary.session_file}:
                return self._session_summary_payload(summary)
        if item_id in state.communities:
            return self._community_payload(state.communities[item_id])
        return None

    def memoryGet(self, itemId: str) -> Optional[dict[str, Any]]:
        return self.memory_get(itemId)

    def memory_forget(self, item_id: str) -> dict[str, Any]:
        """Soft-delete memory facts while preserving transcript evidence chains."""
        item_id = str(item_id or "").strip()
        if not item_id:
            return {"forgotten": False, "id": item_id, "reason": "missing_id"}
        self.ensure_loaded()
        state = self._require_state()
        now_ms = int(time.time() * 1000)

        if item_id in state.facts:
            fact = state.facts[item_id]
            if fact.status != "active":
                return {"forgotten": False, "id": item_id, "reason": "already_inactive"}
            fact.status = "deleted"
            fact.invalidAt = now_ms
            self._rebuild_indices()
            self.save()
            return {"forgotten": True, "id": item_id, "type": "FACT"}

        if item_id in state.unified_facts:
            fact = state.unified_facts[item_id]
            if fact.status != "active":
                return {"forgotten": False, "id": item_id, "reason": "already_inactive"}
            fact.status = "deleted"
            fact.invalidAt = now_ms
            self._rebuild_indices()
            self.save()
            return {"forgotten": True, "id": item_id, "type": "UNIFIED_FACT"}

        # HmFact has no status field; remove the derived fact and references from episodes.
        if item_id in state.hm_facts:
            del state.hm_facts[item_id]
            for episode in state.hm_episodes.values():
                if item_id in episode.fact_ids:
                    episode.fact_ids = [fid for fid in episode.fact_ids if fid != item_id]
            self._rebuild_indices()
            self.save()
            return {"forgotten": True, "id": item_id, "type": "HM_FACT"}

        existing = self.memory_get(item_id)
        if existing:
            return {
                "forgotten": False,
                "id": item_id,
                "reason": "unsupported_type",
                "type": existing.get("node_type") or existing.get("source") or "unknown",
            }
        return {"forgotten": False, "id": item_id, "reason": "not_found"}

    def memoryForget(self, itemId: str) -> dict[str, Any]:
        return self.memory_forget(itemId)

    def archive_expand(
        self,
        archive_id: str,
        session_id: str = "",
        session_key: str = "",
        limit: int = 200,
    ) -> dict[str, Any]:
        archive_id = str(archive_id or "").strip()
        session_id = str(session_id or "").strip()
        session_key = str(session_key or "").strip()
        if not archive_id and not session_id and not session_key:
            return {"archiveId": archive_id, "summary": "", "messages": [], "source": "missing_ref"}
        self.ensure_loaded()
        state = self._require_state()
        max_messages = max(1, min(1000, int(limit or 200)))

        summary = self._resolve_session_summary_ref(archive_id, session_id, session_key)
        resolved_session_id = session_id or (summary.session_id if summary else "")
        resolved_session_key = session_key or (summary.session_key if summary else archive_id)
        resolved_session_file = summary.session_file if summary else ""

        messages = self._resolve_archive_transcript_messages(
            archive_id=archive_id,
            session_id=resolved_session_id,
            session_key=resolved_session_key,
            session_file=resolved_session_file,
            limit=max_messages,
        )
        source = "summary+transcript" if summary and messages else "summary" if summary else "transcript" if messages else "not_found"
        return {
            "archiveId": archive_id,
            "sessionId": resolved_session_id,
            "sessionKey": resolved_session_key,
            "summary": self._session_summary_payload(summary) if summary else None,
            "messages": messages,
            "source": source,
        }

    def archiveExpand(self, archiveId: str, sessionId: str = "", sessionKey: str = "", limit: int = 200) -> dict[str, Any]:
        return self.archive_expand(archiveId, sessionId, sessionKey, limit)

    def _resolve_session_summary_ref(
        self,
        archive_id: str,
        session_id: str = "",
        session_key: str = "",
    ) -> SessionSummary | None:
        state = self._require_state()
        refs = {ref for ref in [archive_id, session_id, session_key] if ref}
        for summary in state.session_summaries.values():
            if refs & {summary.id, summary.session_key, summary.session_id, summary.session_file}:
                return summary
        return None

    def _resolve_archive_transcript_messages(
        self,
        *,
        archive_id: str,
        session_id: str,
        session_key: str,
        session_file: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        state = self._require_state()
        refs = {ref for ref in [archive_id, session_id, session_key, session_file] if ref}
        matched = [
            entry for entry in state.transcripts
            if entry.session_id in refs or entry.session_key in refs or entry.session_file in refs
        ]
        matched.sort(key=lambda entry: (entry.turn_index, entry.created_at))
        return [
            {
                "role": entry.speaker,
                "text": entry.text,
                "messageIndex": entry.turn_index,
                "sessionId": entry.session_id,
                "sessionKey": entry.session_key,
                "sessionFile": entry.session_file,
                "evidence": asdict(entry.evidence) if entry.evidence else None,
            }
            for entry in matched[:limit]
        ]

    def clear_transient_workspace(self, session_id: str) -> None:
        self._store.clear_workspace_state(session_id)

    def clearTransientWorkspace(self, sessionId: str) -> None:
        self.clear_transient_workspace(sessionId)

    def get_recent_traces(self, session_id: str = "", limit: int = 5) -> list[dict[str, Any]]:
        self.ensure_loaded()
        if session_id:
            traces = self._store.list_recent_traces(session_id, limit)
            if traces:
                return traces
        return list(self._require_state().traces[-limit:])

    def getRecentTraces(self, sessionId: str = "", limit: int = 5) -> list[dict[str, Any]]:
        return self.get_recent_traces(sessionId, limit)

    def get_all_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        self.ensure_loaded()
        return list(self._require_state().traces[-limit:])

    def getAllTraces(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.get_all_traces(limit)

    def get_all_slow_path_jobs(self, limit: int = 200) -> list[dict[str, Any]]:
        self.ensure_loaded()
        return list(self._require_state().slow_path_jobs[-limit:])

    def getAllSlowPathJobs(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.get_all_slow_path_jobs(limit)

    def get_all_transcripts(self, limit: int = 200) -> list[dict[str, Any]]:
        self.ensure_loaded()
        return [self._transcript_payload(entry) for entry in self._require_state().transcripts[-limit:]]

    def getAllTranscripts(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.get_all_transcripts(limit)

    def get_all_graph_nodes(self) -> list[dict[str, Any]]:
        self.ensure_loaded()
        return self.salientMemoryGraph.getAllGraphNodes()

    def getAllGraphNodes(self) -> list[dict[str, Any]]:
        return self.get_all_graph_nodes()

    def get_all_graph_edges(self) -> list[dict[str, Any]]:
        self.ensure_loaded()
        return self.salientMemoryGraph.getAllGraphEdges()

    def getAllGraphEdges(self) -> list[dict[str, Any]]:
        return self.get_all_graph_edges()

    def get_all_communities(self) -> list[dict[str, Any]]:
        self.ensure_loaded()
        return self.salientMemoryGraph.getAllCommunities()

    def getAllCommunities(self) -> list[dict[str, Any]]:
        return self.get_all_communities()

    def get_all_facts(self, limit: int = 200) -> list[dict[str, Any]]:
        self.ensure_loaded()
        return self.ledger.getAllFacts(limit)

    def getAllFacts(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.get_all_facts(limit)

    def get_slow_path_status(self) -> dict[str, int]:
        logger.info("开始 get_slow_path_status")
        status = dict(self.slowPath.status())
        with self._flush_state_lock:
            flush_active = self._flush_active
        if flush_active and status.get("running", 0) <= 0:
            status["running"] = 1
        logger.info("get_slow_path_status 完成: %s", status)
        return status

    def getSlowPathStatus(self) -> dict[str, int]:
        return self.get_slow_path_status()

    def get_slow_path_status_detailed(self) -> dict[str, Any]:
        status = self.get_slow_path_status()
        jobs = self.slowPath.jobs()
        pending_jobs = [
            {
                "id": job.get("id", "")[:16],
                "attempts": int(job.get("attempts", 0) or 0),
                "query": str(job.get("query", "") or "")[:120],
                "last_error": job.get("last_error", ""),
            }
            for job in jobs
            if job.get("status") in {"pending", "running"}
        ][:20]
        failed_jobs = [
            {
                "id": job.get("id", "")[:16],
                "attempts": int(job.get("attempts", 0) or 0),
                "query": str(job.get("query", "") or "")[:120],
                "last_error": job.get("last_error", ""),
            }
            for job in jobs
            if job.get("status") == "failed"
        ][:20]
        with self._flush_state_lock:
            flush_active = self._flush_active
            flush_last_error = self._flush_last_error
        return {
            **status,
            "pending_jobs": pending_jobs,
            "failed_jobs": failed_jobs,
            "flush_active": flush_active,
            "flush_last_error": flush_last_error,
        }

    def getSlowPathStatusDetailed(self) -> dict[str, Any]:
        return self.get_slow_path_status_detailed()

    def _build_hypermem_context(
        self,
        question: str,
        classification: ClassificationResult,
        fact_results: list[tuple[HmFact, float]],
        episode_results: list[tuple[HmEpisode, float]],
        use_aaak: bool = False,
    ) -> str:
        """Build compact context from HyperMem C2F retrieval results.

        Token budget target: ≤800 tokens (~3200 chars).
        When use_aaak=True, renders facts in AAAK v2 pipe-delimited notation.
        """
        if use_aaak:
            return encode_facts_aaak(
                fact_results, episode_results, question,
                intent=classification.intent,
                max_facts=self.config.c2fMaxFacts,
                max_episodes=self.config.c2fMaxEpisodes,
            )

        lines: list[str] = []

        # System instruction (1 line)
        lines.append(
            f"Answer from memory context. Intent={classification.intent}."
        )
        lines.append("")

        # ── Memory Facts ──
        max_facts = self.config.c2fMaxFacts
        fact_max_chars = self.config.c2fFactContentMaxChars
        if fact_results:
            lines.append("Facts:")
            count = 0
            for i, (fact, score) in enumerate(fact_results):
                if count >= max_facts:
                    break
                # Skip LOW importance facts unless top-3 by score
                if fact.importance == "low" and i >= 3:
                    continue
                count += 1
                # Compress: strip dialogue markers, truncate
                content = self._compress_fact_content(fact.content, max_chars=fact_max_chars)
                if not content or len(content) < 10:
                    count -= 1
                    continue
                lines.append(f"{count}. {content}")
            lines.append("")

        # ── Episode Context ──
        max_episodes = self.config.c2fMaxEpisodes
        ep_summary_max = self.config.c2fEpisodeSummaryMaxChars
        if episode_results:
            lines.append("Episodes:")
            for episode, score in episode_results[:max_episodes]:
                ts = episode.timestamp_start or episode.session_key
                summary = episode.summary[:ep_summary_max] if episode.summary else episode.title
                lines.append(f"- [{ts}] {summary}")
            lines.append("")

        # Question
        lines.append(question)

        return "\n".join(lines)

    @staticmethod
    def _compress_fact_content(content: str, max_chars: int = 200) -> str:
        """Strip turn markers and compress fact text to ≤max_chars, preserving speaker attribution."""
        # Remove [turn N] markers only — keep "Speaker: text" for attribution
        text = re.sub(r"\[turn \d+\]\s*", "", content)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            # Try to cut at sentence boundary
            cut = text[:max_chars].rfind(". ")
            if cut > max_chars * 0.6:
                text = text[:cut + 1]
            else:
                text = text[:max_chars - 3] + "..."
        return text

    def _detect_risk_level(self, query: str) -> str:
        return "high" if re.search(r"(delete|drop|truncate|permission|auth|支付|核心|生产|database|schema|rm -rf|权限)", query, re.IGNORECASE) else "normal"

    def _resolve_entity_subjects(self, matched_entity_ids: list[str]) -> list[str]:
        if not matched_entity_ids:
            return []
        state = self._require_state()
        subjects = []
        for entity_id in matched_entity_ids:
            entity = state.entities.get(entity_id)
            if entity:
                subjects.append(entity.name)
        return subjects

    def _resolve_session_scope(self, session_id: str, session_key: str | None = None) -> str:
        normalized = (session_key or "").strip()
        return normalized if normalized else session_id

    def _resolve_scoped_ledger_subject(self, session_id: str, session_key: str | None = None) -> str:
        return f"context:{self._resolve_session_scope(session_id, session_key)}"

    def _derive_ledger_subjects_from_query(self, query: str) -> list[str]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return []
        matches: list[str] = []
        for subject in self._store.list_distinct_active_fact_subjects(200):
            if subject == "user" or subject.startswith("context:"):
                continue
            subject_tokens = set(tokenize(subject))
            if subject_tokens & query_tokens:
                matches.append(subject)
        return matches

    def _resolve_ledger_subjects_for_session(self, session_id: str, session_key: str | None = None, query: str | None = None) -> list[str]:
        subjects = {"user", self._resolve_scoped_ledger_subject(session_id, session_key)}
        if query:
            subjects.update(self._derive_ledger_subjects_from_query(query))
        return list(subjects)

    def _execute_slow_path_job(self, payload: dict[str, Any]) -> None:
        entries = payloadToEntries(payload.get("entries", []))
        event_ids = list(payload.get("event_ids", []) or [])
        if not event_ids and entries:
            session = type(
                "Session",
                (),
                {
                    "session_id": entries[0].session_id,
                    "session_key": str(payload.get("session_key", "") or entries[0].session_key),
                    "session_file": entries[0].session_file,
                    "date_time": str(payload.get("date_time", "") or entries[0].date_time),
                    "turns": [
                        type(
                            "Turn",
                            (),
                            {
                                "speaker": entry.speaker,
                                "text": entry.text,
                                "blip_caption": "",
                                "message_index": entry.turn_index,
                                "created_at": entry.created_at,
                            },
                        )
                        for entry in entries
                    ],
                    "base_message_index": entries[0].turn_index,
                    "line_map": {
                        entry.turn_index: entry.evidence.startLine
                        for entry in entries
                        if entry.evidence is not None and entry.evidence.startLine is not None
                    },
                },
            )
            _registered_entries, event_ids = self._register_session_entries(session, append=True, record_transcripts=False)
        self._apply_slow_path_payload(
            str(payload.get("session_key", "") or ""),
            str(payload.get("date_time", "") or ""),
            entries,
            event_ids,
        )

    def _build_turn_input(
        self,
        *,
        session_id: str,
        session_key: str,
        session_file: str,
        entries: Sequence[TranscriptEntry],
        query_fallback: str,
    ) -> dict[str, Any]:
        return {
            "sessionId": session_id,
            "sessionKey": session_key,
            "sessionFile": session_file,
            "query": entries[0].text[:120] if entries else query_fallback,
            "turnMessagesText": [entry.text for entry in entries],
            "turnMessageIndexes": [entry.turn_index for entry in entries],
            "turnStartIndex": entries[0].turn_index if entries else 0,
        }

    def _extract_profile_facts(self) -> None:
        state = self._require_state()
        if not state.speaker_names:
            return

        def inference_fn(prompt: str) -> str:
            return self._llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1400,
            ).content

        for subject in state.speaker_names:
            related_entries = [
                entry for entry in state.transcripts
                if entry.speaker == subject or subject.lower() in (entry.text or "").lower()
            ]
            if not related_entries:
                continue
            texts = [
                f"[{entry.date_time}] [{entry.speaker}] {entry.text}"
                for entry in related_entries[:80]
            ]
            facts = extractProfileFactsWithLlm(
                {
                    "inferenceFn": inference_fn,
                    "subject": subject,
                    "texts": texts,
                    "factTtlDays": 90,
                    "llmTruncationChars": 12000,
                }
            )
            verified_facts = self._verify_structured_facts(facts, related_entries, inference_fn)
            for fact in verified_facts:
                fact.tokens = tokenize(f"{fact.subject} {fact.key} {fact.value}")
                upsertFact(
                    state,
                    fact,
                    add_edge_fn=lambda state, from_id, to_id, edge_type, weight, evidence, relation_label: addEdge(
                        state, from_id, to_id, edge_type, weight, evidence, relation_label, VALID_EDGE_TYPES
                    ),
                    find_entity_id_fn=findEntityIdByName,
                )

    def _candidate_entries_for_fact(
        self,
        fact: LedgerFact,
        entries: Sequence[TranscriptEntry],
        *,
        limit: int = 4,
        precomputed_vector: Any = None,
    ) -> list[TranscriptEntry]:
        if not entries:
            return []
        query_text = f"{fact.subject} {fact.key} {fact.value}"
        query_vector = precomputed_vector if precomputed_vector is not None else self._embedder.embed_text(query_text)
        ranked = rank_text_records(
            query_text,
            entries,
            query_vector=query_vector,
            get_text=lambda entry: f"{entry.speaker} {entry.text}",
            get_vector=lambda entry: getattr(entry, "vector", None),
            rrf_k=40,
        )
        return [entry for entry, _score in ranked[:limit]]

    def _verify_structured_fact(
        self,
        fact: LedgerFact,
        candidate_entries: Sequence[TranscriptEntry],
        inference_fn,
    ) -> LedgerFact | None:
        if not candidate_entries:
            return None
        evidence_lines = []
        for index, entry in enumerate(candidate_entries):
            evidence_lines.append(
                f"{index}. [{entry.date_time}] [{entry.speaker}] {entry.text}"
            )
        prompt = (
            "Verify whether the candidate fact is supported by the evidence snippets.\n"
            "Return JSON only with keys "
            "{\"supported\": boolean, \"best_index\": number, \"normalized_value\": string, \"confidence\": number}.\n\n"
            f"FACT:\nsubject={fact.subject}\nkey={fact.key}\nvalue={fact.value}\n\n"
            "EVIDENCE:\n" + "\n".join(evidence_lines)
        )
        try:
            payload = json.loads(inference_fn(prompt))
        except Exception:
            return None
        if not bool(payload.get("supported", False)):
            return None
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        if confidence < 0.72:
            return None
        best_index = int(payload.get("best_index", 0) or 0)
        if best_index < 0 or best_index >= len(candidate_entries):
            best_index = 0
        entry = candidate_entries[best_index]
        normalized_value = str(payload.get("normalized_value", "") or fact.value).strip()
        return LedgerFact(
            id=fact.id,
            subject=fact.subject,
            key=fact.key,
            value=normalized_value,
            scope=fact.scope,
            text=f"{fact.subject}: {normalized_value}",
            session_key=entry.session_key,
            turn_index=entry.turn_index,
            tokens=[],
            evidence=entry.evidence,
            subject_entity_id=fact.subject_entity_id,
            confidence=max(fact.confidence, confidence),
            validFrom=fact.validFrom,
            validTo=fact.validTo,
            invalidAt=fact.invalidAt,
            expiresAt=fact.expiresAt,
            source=f"{fact.source}-verified",
            status=fact.status,
            vector=self._embedder.embed_text(f"{fact.subject} {fact.key} {normalized_value}"),
        )

    def _should_verify_structured_fact(self, fact: LedgerFact) -> bool:
        """Verify facts with high confidence that represent durable profile information."""
        key = str(fact.key or "").strip().lower()
        if not key:
            return False
        threshold = getattr(self.config, "structuredVerificationConfidence", 0.85)
        return fact.confidence >= threshold

    def _verify_batch(
        self,
        to_verify: list[tuple[LedgerFact, list[TranscriptEntry]]],
        inference_fn,
    ) -> list[tuple[LedgerFact, TranscriptEntry, str, float]]:
        """Run one batched LLM verify call; return (fact, entry, normalized_value, confidence) tuples."""
        if not to_verify:
            return []
        if len(to_verify) == 1:
            fact, candidate_entries = to_verify[0]
            result = self._verify_structured_fact(fact, candidate_entries, inference_fn)
            if result is None:
                return []
            best_entry = candidate_entries[0]
            return [(fact, best_entry, result.value, result.confidence)]

        fact_blocks: list[str] = []
        for idx, (fact, candidate_entries) in enumerate(to_verify):
            evidence_lines = [
                f"  {j}. [{e.date_time}] [{e.speaker}] {e.text}"
                for j, e in enumerate(candidate_entries)
            ]
            fact_blocks.append(
                f"FACT_{idx}:\n"
                f"  subject={fact.subject}\n  key={fact.key}\n  value={fact.value}\n"
                f"  EVIDENCE:\n" + "\n".join(evidence_lines)
            )
        prompt = (
            "Verify whether each candidate fact is supported by its evidence snippets.\n"
            "Return a JSON array (one object per FACT_N, in order) with keys:\n"
            '  {"supported": boolean, "best_index": number, "normalized_value": string, "confidence": number}\n\n'
            + "\n\n".join(fact_blocks)
        )
        try:
            results = json.loads(inference_fn(prompt))
            if isinstance(results, dict):
                results = [results]
            if not isinstance(results, list):
                results = []
        except Exception:
            results = []

        out: list[tuple[LedgerFact, TranscriptEntry, str, float]] = []
        for idx, (fact, candidate_entries) in enumerate(to_verify):
            if idx >= len(results):
                break
            payload = results[idx]
            if not isinstance(payload, dict) or not bool(payload.get("supported", False)):
                continue
            confidence = float(payload.get("confidence", 0.0) or 0.0)
            if confidence < 0.72:
                continue
            best_index = int(payload.get("best_index", 0) or 0)
            if best_index < 0 or best_index >= len(candidate_entries):
                best_index = 0
            normalized_value = str(payload.get("normalized_value", "") or fact.value).strip()
            out.append((fact, candidate_entries[best_index], normalized_value, confidence))
        return out

    def _verify_structured_facts(
        self,
        facts: Sequence[LedgerFact],
        entries: Sequence[TranscriptEntry],
        inference_fn,
        max_batch: int = 20,
    ) -> list[LedgerFact]:
        pending: list[LedgerFact] = []
        seen: set[tuple[str, str, str]] = set()
        for fact in facts:
            if not self._should_verify_structured_fact(fact):
                continue
            signature = (
                str(fact.subject or "").strip().lower(),
                str(fact.key or "").strip().lower(),
                str(fact.value or "").strip().lower(),
            )
            if signature in seen:
                continue
            seen.add(signature)
            pending.append(fact)

        if not pending:
            return []

        query_texts = [f"{f.subject} {f.key} {f.value}" for f in pending]
        query_vectors = self._embedder.embed_texts(query_texts)

        to_verify: list[tuple[LedgerFact, list[TranscriptEntry]]] = []
        for fact, qvec in zip(pending, query_vectors):
            candidates = self._candidate_entries_for_fact(fact, entries, precomputed_vector=qvec)
            if candidates:
                to_verify.append((fact, candidates))

        if not to_verify:
            return []

        verified_meta: list[tuple[LedgerFact, TranscriptEntry, str, float]] = []
        for i in range(0, len(to_verify), max_batch):
            verified_meta.extend(self._verify_batch(to_verify[i : i + max_batch], inference_fn))

        if not verified_meta:
            return []

        vectors = self._embedder.embed_texts([f"{f.subject} {f.key} {nv}" for f, _, nv, _ in verified_meta])
        verified: list[LedgerFact] = []
        for (fact, entry, normalized_value, confidence), vector in zip(verified_meta, vectors):
            verified.append(LedgerFact(
                id=fact.id,
                subject=fact.subject,
                key=fact.key,
                value=normalized_value,
                scope=fact.scope,
                text=f"{fact.subject}: {normalized_value}",
                session_key=entry.session_key,
                turn_index=entry.turn_index,
                tokens=[],
                evidence=entry.evidence,
                subject_entity_id=fact.subject_entity_id,
                confidence=max(fact.confidence, confidence),
                validFrom=fact.validFrom,
                validTo=fact.validTo,
                invalidAt=fact.invalidAt,
                expiresAt=fact.expiresAt,
                source=f"{fact.source}-verified",
                status=fact.status,
                vector=vector,
            ))
        return verified

    def _build_slow_path_job_payload(
        self,
        *,
        turn_input: dict[str, Any],
        date_time: str,
        entries: Sequence[TranscriptEntry],
        event_ids: Sequence[str],
    ) -> tuple[str, dict[str, Any]]:
        fingerprint = buildSlowPathTurnFingerprint(turn_input)
        job_id = _stable_id("SLOWPATH", fingerprint)
        payload = {
            "session_key": turn_input["sessionKey"],
            "date_time": date_time,
            "entries": [self._transcript_payload(entry) for entry in entries],
            "event_ids": list(event_ids),
            "query": turn_input["query"],
            "fingerprint": fingerprint,
            "turn_input": turn_input,
        }
        return job_id, payload

    def _require_state(self) -> MemoryState:
        if self._state is None:
            raise RuntimeError("engine state is not initialized")
        return self._state

    def _register_session_entries(self, session: Any, append: bool = False, record_transcripts: bool = True) -> tuple[list[TranscriptEntry], list[str]]:
        return registerSessionEntries(
            self._require_state(),
            session,
            self._require_state().speaker_names,
            set(COMMON_CAPITALIZED),
            append=append,
            record_transcripts=record_transcripts,
            add_edge_fn=lambda state, from_id, to_id, edge_type, weight, evidence, relation_label: addEdge(
                state, from_id, to_id, edge_type, weight, evidence, relation_label, VALID_EDGE_TYPES
            ),
            valid_edge_types=VALID_EDGE_TYPES,
        )

    def _apply_slow_path_payload(
        self,
        session_key: str,
        date_time: str,
        entries: Sequence[TranscriptEntry],
        event_ids: Optional[Sequence[str]] = None,
    ) -> None:
        session_file = entries[0].session_file if entries else session_key
        scoped_subject = self._resolve_scoped_ledger_subject(entries[0].session_id, session_key) if entries else self._resolve_scoped_ledger_subject(session_key, session_key)
        regex_facts = self.ledger.ingestTexts(
            {
                "globalSubject": "user",
                "scopedSubject": scoped_subject,
                "texts": [entry.text for entry in entries],
                "source": "turn-distillation",
                "evidenceBase": {
                    "sessionFile": session_file,
                    "startIndex": entries[0].turn_index if entries else 0,
                    "messageIndexes": [entry.turn_index for entry in entries],
                },
            }
        )
        for fact in regex_facts:
            fact.tokens = tokenize(f"{fact.key} {fact.subject} {fact.value}")
            upsertFact(
                self._require_state(),
                fact,
                add_edge_fn=lambda state, from_id, to_id, edge_type, weight, evidence, relation_label: addEdge(
                    state, from_id, to_id, edge_type, weight, evidence, relation_label, VALID_EDGE_TYPES
                ),
                find_entity_id_fn=findEntityIdByName,
            )
            _upsert_unified_fact(self._require_state(), _ledger_fact_to_unified(fact))

        extracted_summary = None
        if self._llm.is_enabled and not self._slowpath_llm_disabled and entries:
            import sys as _sys
            import time as _time_mod
            _label = session_key[:20]
            _t_job = _time_mod.monotonic()

            _llm_call_count = [0]

            def inference_fn(prompt: str) -> str:
                _llm_call_count[0] += 1
                _sys.stderr.write(f"\r[ebm llm ] {_label:<20} LLM call #{_llm_call_count[0]} ...")
                _sys.stderr.flush()
                _t0 = _time_mod.monotonic()
                result = self._llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1600,
                ).content
                _sys.stderr.write(f"\r[ebm llm ] {_label:<20} LLM call #{_llm_call_count[0]} done ({_time_mod.monotonic()-_t0:.1f}s)\n")
                _sys.stderr.flush()
                return result

            _embed_count = [0]
            _embed_label = f"embed:{session_key[:16]}"

            def _tracked_embed_texts(texts: list[str]) -> list[Any]:
                _sys.stderr.write(f"\r[ebm embed] {_embed_label:<18} embedding {len(texts)} texts ...")
                _sys.stderr.flush()
                results = self._embedder.embed_texts(texts)
                _embed_count[0] += len(texts)
                _sys.stderr.write(f"\r[ebm embed] {_embed_label:<18} {_embed_count[0]} vectors done\n")
                _sys.stderr.flush()
                return results

            extraction_params = {
                "inferenceFn": inference_fn,
                "embedFn": None,
                "embedTextsFn": _tracked_embed_texts,
                "texts": [f"[conversation: {date_time}]"] + [entry.text for entry in entries] if date_time else [entry.text for entry in entries],
                "sessionFile": session_file,
                "sessionId": session_key,
                "startIndex": entries[0].turn_index,
                "messageIndexes": [entry.turn_index for entry in entries],
                "globalSubject": scoped_subject,
                "factTtlDays": 90,
                "llmTruncationChars": 12000,
                "confidenceCeiling": 0.97,
            }
            _sys.stderr.write(f"[ebm slow] {_label:<20} start  msgs={len(entries)}\n")
            _sys.stderr.flush()
            if self.config.benchmarkFastIngest:
                combined = extractHighValueFactsWithLlm(extraction_params)
                llm_facts = self._verify_structured_facts(combined["facts"], entries, inference_fn)
                for fact in llm_facts:
                    fact.tokens = tokenize(f"{fact.key} {fact.subject} {fact.value}")
                    fact.session_key = session_key
                    fact.turn_index = fact.evidence.messageIndex if fact.evidence and fact.evidence.messageIndex is not None else entries[0].turn_index
                    upsertFact(
                        self._require_state(),
                        fact,
                        add_edge_fn=lambda state, from_id, to_id, edge_type, weight, evidence, relation_label: addEdge(
                            state, from_id, to_id, edge_type, weight, evidence, relation_label, VALID_EDGE_TYPES
                        ),
                        find_entity_id_fn=findEntityIdByName,
                    )
                    _upsert_unified_fact(self._require_state(), _ledger_fact_to_unified(fact))
                extracted_summary = combined["summary"]
            else:
                combined = extractAllWithLlm(extraction_params)
                llm_facts = combined["facts"]
                for fact in llm_facts:
                    fact.tokens = tokenize(f"{fact.key} {fact.subject} {fact.value}")
                    fact.session_key = session_key
                    fact.turn_index = fact.evidence.messageIndex if fact.evidence and fact.evidence.messageIndex is not None else entries[0].turn_index
                    upsertFact(
                        self._require_state(),
                        fact,
                        add_edge_fn=lambda state, from_id, to_id, edge_type, weight, evidence, relation_label: addEdge(
                            state, from_id, to_id, edge_type, weight, evidence, relation_label, VALID_EDGE_TYPES
                        ),
                        find_entity_id_fn=findEntityIdByName,
                    )
                    _upsert_unified_fact(self._require_state(), _ledger_fact_to_unified(fact))

                extracted_summary = combined["summary"]
                entity_graph = combined["entity_graph"]
                applyExtractedEntityGraph(self._require_state(), entity_graph)

            _sys.stderr.write(f"[ebm slow] {_label:<20} done   facts={len(llm_facts)} elapsed={_time_mod.monotonic()-_t_job:.1f}s\n")
            _sys.stderr.flush()

        # Phase 5: Graph distillation (TASK/EVENT/SALIENT_MEMORY/FACT nodes + edges)
        if entries:
            session_id = entries[0].session_id if entries else session_key
            distill_result = self.salientMemoryGraph.distillTurn({
                "sessionId": session_id,
                "sessionFile": session_file,
                "query": entries[0].text[:120] if entries else session_key,
                "turnMessagesText": [entry.text for entry in entries],
                "turnMessageIndexes": [entry.turn_index for entry in entries],
                "startIndex": entries[0].turn_index if entries else 0,
            })
            self.logger.info(
                "EBM slow path graph distill: session=%s nodes=%d edges=%d",
                session_key, len(distill_result.get("nodes", [])), len(distill_result.get("edges", [])),
            )

        # Phase 6: Semantic dedup (matching TS engine.ts Phase 6)
        dedup_merged = self._run_semantic_dedup(entries[0].session_id if entries else session_key)

        # Phase 7: Session-end refinement is deferred to post-ingest (run once after all sessions)

        summary = buildSessionSummary(session_key, date_time, list(entries), list(event_ids or []))
        summary.session_id = entries[0].session_id if entries else session_key
        summary.session_file = session_file
        summary.id = _stable_id("SUMMARY", summary.session_id, session_file, entries[0].turn_index if entries else 0)
        summary.message_count = len(entries)
        summary.created_at = int(entries[-1].created_at if entries and entries[-1].created_at else int(time.time() * 1000))
        summary.tokens = tokenize(f"{summary.abstract} {summary.overview}")
        if extracted_summary:
            abstract = normalize_whitespace(str(extracted_summary.get("abstract", "") or ""))
            overview = normalize_whitespace(str(extracted_summary.get("overview", "") or ""))
            if abstract or overview:
                summary.abstract = abstract or summary.abstract
                summary.overview = overview or summary.overview
                summary.tokens = tokenize(f"{summary.abstract} {summary.overview}")
        self._require_state().session_summaries[session_key] = summary

        # Also create a session-level HmEpisode for unified Layer 1
        session_episode_id = _stable_id("SESSION_EPISODE", session_key, session_file)
        session_episode = HmEpisode(
            id=session_episode_id,
            session_key=session_key,
            title=f"Session: {session_key}",
            summary=f"{summary.abstract} {summary.overview}".strip(),
            dialogue="",
            keywords=list(summary.tokens) if summary.tokens else [],
            timestamp_start=date_time,
            turn_start=entries[0].turn_index if entries else 0,
            turn_end=entries[-1].turn_index if entries else 0,
            source_event_ids=list(summary.source_event_ids),
            is_session_summary=True,
            vector=summary.vector,
            created_at=summary.created_at,
        )
        self._require_state().hm_episodes[session_episode_id] = session_episode

        # ── HyperMem Pipeline: Episode Detection → Fact Extraction → Topic Aggregation ──
        self._apply_hypermem_pipeline(session_key, date_time, entries)

    def _apply_hypermem_pipeline(
        self,
        session_key: str,
        date_time: str,
        entries: Sequence[TranscriptEntry],
    ) -> None:
        """HyperMem three-level ingest: Episode Detection → Fact Extraction → Topic Aggregation."""
        if not entries:
            return
        state = self._require_state()

        # Build LLM inference function (same as existing slow path)
        inference_fn = None
        if self.config.benchmarkFastIngest:
            inference_fn = None
        elif self._llm.is_enabled and not self._slowpath_llm_disabled:
            def inference_fn(prompt: str) -> str:
                return self._llm.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4000,
                ).content

        # Stage 1: Episode Detection
        if inference_fn:
            episodes = detect_episodes_llm(entries, session_key, date_time, inference_fn)
        else:
            episodes = detect_episodes_heuristic(entries, session_key, date_time)
        self.logger.info(
            "HyperMem episode detection: session=%s episodes=%d entries=%d",
            session_key, len(episodes), len(entries),
        )

        # Stage 2: Fact Extraction (per episode)
        all_facts: list[HmFact] = []
        for episode in episodes:
            if inference_fn:
                ep_facts = extract_facts_from_episode(episode, inference_fn, max_chars=4000)
            else:
                from .hypergraph.fact_extractor import extract_facts_heuristic
                ep_facts = extract_facts_heuristic(episode)
            episode.fact_ids = [f.id for f in ep_facts]
            all_facts.extend(ep_facts)
        self.logger.info(
            "HyperMem fact extraction: session=%s facts=%d across %d episodes",
            session_key, len(all_facts), len(episodes),
        )

        # Stage 3: Topic Aggregation
        existing_topics = list(state.hm_topics.values())
        updated_topics = aggregate_episodes_to_topics(
            episodes, existing_topics, inference_fn, self._embedder.embed_text if self._embedder.is_enabled else None,
        )
        self.logger.info(
            "HyperMem topic aggregation: session=%s topics_total=%d (new_or_updated=%d)",
            session_key, len(updated_topics), len(updated_topics) - len(existing_topics),
        )

        # Persist to state
        for topic in updated_topics:
            state.hm_topics[topic.id] = topic
        for episode in episodes:
            state.hm_episodes[episode.id] = episode
        for fact in all_facts:
            state.hm_facts[fact.id] = fact
            # Also upsert into unified_facts with dedup
            embed_fn = self._embedder.embed_text if self._embedder.is_enabled else None
            _upsert_unified_fact(state, _hm_fact_to_unified(fact), embed_fn=embed_fn)

    def _rebuild_communities(self) -> None:
        communities = self.salientMemoryGraph.rebuildCommunities()
        state = self._require_state()
        state.communities = {
            community_id: CommunitySummaryRecord(
                id=item["id"],
                title=item["title"],
                summary=item["summary"],
                keywords=list(item["keywords"]),
                member_ids=list(item["member_ids"]),
                vector=item.get("vector"),
            )
            for community_id, item in communities.items()
        }
        # Also write communities as HmTopics for unified Layer 0
        for community_id, item in communities.items():
            topic_id = f"community-{community_id}"
            state.hm_topics[topic_id] = HmTopic(
                id=topic_id,
                title=item["title"],
                summary=item["summary"],
                keywords=list(item["keywords"]),
                member_entity_ids=list(item["member_ids"]),
                source="community_detection",
                vector=item.get("vector"),
            )

    def _embed_state(self) -> None:
        embedState(self._require_state(), self._embedder)

    def _propagate_hm_embeddings(self) -> None:
        """Run hypergraph embedding propagation on HyperMem nodes."""
        state = self._require_state()
        if not state.hm_topics and not state.hm_episodes and not state.hm_facts:
            return
        propagate_embeddings(
            list(state.hm_topics.values()),
            list(state.hm_episodes.values()),
            list(state.hm_facts.values()),
        )

    def _rebuild_indices(self) -> None:
        rebuildIndices(self._require_state())

    # ── Phase 6: Semantic dedup (matching TS engine.ts:731-767) ──────────
    def _run_semantic_dedup(self, session_id: str) -> int:
        """Merge near-duplicate graph nodes by cosine similarity with prefix guard."""
        from .client import cosine_similarity as cos_sim
        state = self._require_state()
        dedup_threshold = getattr(self.config, "dedupSimilarityThreshold", 0.92)
        merged_count = 0

        # Group nodes by type for same-type dedup
        nodes_by_type: dict[str, list[tuple[str, str, str, object]]] = {}  # type -> [(id, label, content, vector)]
        for entity in state.entities.values():
            nodes_by_type.setdefault("ENTITY", []).append((entity.id, entity.name, entity.description or entity.name, entity.vector))
        for fact in state.facts.values():
            key_label = f"{fact.scope}/{fact.key}"
            nodes_by_type.setdefault("FACT", []).append((fact.id, key_label, fact.value, fact.vector))

        for node_type, type_nodes in nodes_by_type.items():
            if len(type_nodes) < 2:
                continue
            consumed: set[str] = set()
            for i in range(len(type_nodes)):
                if type_nodes[i][0] in consumed:
                    continue
                for j in range(i + 1, len(type_nodes)):
                    if type_nodes[j][0] in consumed:
                        continue
                    # Prefix guard: don't merge nodes with different name prefixes
                    prefix_a = type_nodes[i][1].split(".")[0].lower().strip()
                    prefix_b = type_nodes[j][1].split(".")[0].lower().strip()
                    if prefix_a != prefix_b and prefix_a not in prefix_b and prefix_b not in prefix_a:
                        continue
                    sim = cos_sim(type_nodes[i][3], type_nodes[j][3])
                    if sim >= dedup_threshold:
                        # Keep the one with longer content
                        keep_idx, merge_idx = (i, j) if len(type_nodes[i][2]) >= len(type_nodes[j][2]) else (j, i)
                        keep_id = type_nodes[keep_idx][0]
                        merge_id = type_nodes[merge_idx][0]
                        # Migrate edges from merge → keep
                        for edge_id, edge in list(state.graph_edges.items()):
                            if edge.from_id == merge_id:
                                edge.from_id = keep_id
                            if edge.to_id == merge_id:
                                edge.to_id = keep_id
                        # Remove merged node
                        if node_type == "ENTITY":
                            state.entities.pop(merge_id, None)
                        elif node_type == "FACT":
                            state.facts.pop(merge_id, None)
                        consumed.add(merge_id)
                        merged_count += 1
                        self.logger.info(
                            "EBM semantic dedup: merged '%s' into '%s' sim=%.3f",
                            type_nodes[merge_idx][1][:40], type_nodes[keep_idx][1][:40], sim,
                        )
        return merged_count

    # ── Phase 7: Session-end refinement (matching TS engine.ts:1141-1199) ─
    def _run_session_end_refinement(self, session_id: str) -> None:
        """Rule-based graph refinement: stale confidence decay only.

        EVENT→SALIENT_MEMORY promotion is disabled because:
        1. The low indegree threshold (3) promoted 412 raw conversation turns
           as facts, flooding retrieval with noisy, unstructured dialogue.
        2. These promoted facts diluted the signal-to-noise ratio, causing the
           retrieval engine to return irrelevant content for many queries.
        3. The LLM-extracted facts and entity graph already capture the
           important information from high-frequency events.
        """
        # Promotion disabled — no-op.  The LLM slow-path already extracts
        # structured facts and entity relations from conversation turns.
        pass

    def _rank_combined_hits(
        self,
        graph_hits: list[RecallHit],
        ledger_hits: list[RecallHit],
        summary_hits: list[RecallHit],
        community_hits: list[RecallHit],
        *,
        extra_hits: Sequence[RecallHit] = (),
    ) -> list[RecallHit]:
        seen: set[str] = set()
        ranked: list[RecallHit] = []
        def _priority(hit: RecallHit) -> tuple[int, float]:
            source_tag = (hit.verificationNote or "").lower()
            if hit.source == "ledger" and _is_priority_structured_source(source_tag):
                return (0, -float(hit.score))
            if hit.source == "ledger":
                return (1, -float(hit.score))
            if hit.source == "graph":
                return (2, -float(hit.score))
            if hit.source == "summary":
                return (3, -float(hit.score))
            return (4, -float(hit.score))

        for hit in sorted([*graph_hits, *ledger_hits, *summary_hits, *community_hits, *extra_hits], key=_priority):
            key = (hit.session_key or "") + "|" + str(hit.turn_index) + "|" + hit.content[:120]
            if key in seen:
                continue
            seen.add(key)
            ranked.append(hit)
        return ranked

    def _episode_results_to_hits(self, episode_results: Sequence[tuple[HmEpisode, float]]) -> list[RecallHit]:
        hits: list[RecallHit] = []
        for episode, score in episode_results:
            hits.append(
                RecallHit(
                    id=f"hm-episode:{episode.id}",
                    title=episode.title or f"{episode.session_key}:{episode.turn_start}-{episode.turn_end}",
                    content=episode.summary or episode.dialogue[:300],
                    source="hm_episode",
                    score=float(score),
                    reason="hypermem episode summary",
                    session_key=episode.session_key,
                    turn_index=episode.turn_start,
                )
            )
        return hits

    def _rerank_hits(
        self,
        question: str,
        classification: ClassificationResult,
        hits: Sequence[RecallHit],
    ) -> tuple[list[RecallHit], dict[str, Any]]:
        """Rerank by existing PPR/confidence scores — no LLM call."""
        candidate_limit = 8
        final_limit = 5
        candidates = list(hits[:candidate_limit])
        if not candidates:
            return [], {"enabled": False, "candidate_count": 0, "used": False}
        candidates.sort(key=lambda h: h.score, reverse=True)
        return candidates[:final_limit], {
            "enabled": False,
            "candidate_count": len(candidates),
            "used": False,
            "mode": "ppr_passthrough",
        }

    def _build_ranked_context(
        self,
        question: str,
        classification: ClassificationResult,
        hits: Sequence[RecallHit],
        *,
        transcript_context: str = "",
        use_aaak: bool = False,
    ) -> str:
        if use_aaak:
            lines = [
                "AAAK-style memory context",
                *[
                    f"N:{hit.source.upper()}|\"{self._compress_fact_content(hit.content, max_chars=220)}\"|W3|neutral|"
                    for hit in hits[: min(len(hits), self.config.c2fMaxFacts)]
                ],
                question,
            ]
            return "\n".join(lines)

        lines = [
            f"Answer from memory context. Intent={classification.intent}.",
            "",
        ]
        if hits:
            lines.append("Evidence:")
            for index, hit in enumerate(hits, start=1):
                # Use content directly — avoid duplicating title which often contains the same text
                content = self._compress_fact_content(hit.content, max_chars=200)
                line = f"{index}. {content}"
                lines.append(line)
            lines.append("")
        if transcript_context:
            lines.append("Transcript:")
            lines.append(transcript_context[:min(self.config.transcriptContextMaxChars, 1750)])
            lines.append("")
        lines.append(question)
        return "\n".join(lines)

    def _match_entities(self, question: str, classified_entities: Sequence[str]) -> list[str]:
        return matchEntities(self._require_state(), question, classified_entities)

    def _select_focus_tokens(self, tokens: Sequence[str], extra: Sequence[str] = ()) -> list[str]:
        return selectFocusTokens(self._require_state(), tokens, extra)

    def _recall_summaries(
        self,
        query_tokens: list[str],
        query_vector: np.ndarray,
        graph_hits: list[RecallHit],
        ledger_hits: list[RecallHit],
        plan: Any,
    ) -> list[RecallHit]:
        if not plan.include_summaries or plan.summary_top_k <= 0:
            return []
        state = self._require_state()
        candidate_keys: set[str] = set()
        candidate_keys.update(self._store.search_summary_keys(self._select_focus_tokens(query_tokens), max(plan.summary_top_k * 4, 4)))
        for token in self._select_focus_tokens(query_tokens):
            candidate_keys.update(state.summary_index.get(token, set()))
        for hit in graph_hits[:3]:
            if hit.session_key:
                candidate_keys.add(hit.session_key)
        for hit in ledger_hits[:2]:
            if hit.session_key:
                candidate_keys.add(hit.session_key)
        if not candidate_keys:
            candidate_keys.update(list(state.session_summaries.keys())[-2:])

        scored: list[tuple[str, float]] = []
        for session_key in candidate_keys:
            summary = state.session_summaries.get(session_key)
            if summary is None:
                continue
            lexical = keyword_overlap(query_tokens, summary.tokens)
            semantic = cosine_similarity(query_vector, summary.vector)
            score = lexical * 1.8 + semantic * 0.8
            scored.append((session_key, score))

        hits: list[RecallHit] = []
        for session_key, score in sorted(scored, key=lambda item: item[1], reverse=True)[: plan.summary_top_k]:
            summary = state.session_summaries[session_key]
            content = summary.overview if plan.complexity == "deep" else summary.abstract
            hits.append(
                RecallHit(
                    id=f"summary:{session_key}",
                    title=f"{session_key} / {summary.date_time}",
                    content=content,
                    source="summary",
                    score=score,
                    reason="session summary match",
                    session_key=session_key,
                )
            )
        return hits

    def _fallback_text_retrieval(self, session_id: str, query: str, limit: int) -> list[RecallHit]:
        session_file = self._resolve_session_file(session_id)
        if not session_file:
            return []
        entries = self._store.lookup_transcript_by_session_file(session_file)
        if not entries:
            return []
        hits: list[RecallHit] = []
        for entry in entries:
            score = keyword_overlap(tokenize(query), tokenize(str(entry.get("text", "") or "")))
            if score <= 0:
                continue
            message_index = int(entry.get("messageIndex", 0) or 0)
            hits.append(
                RecallHit(
                    id=_stable_id("FALLBACK", session_id, message_index),
                    title=f"transcript:{entry.get('role', 'user')}#{message_index}",
                    content=str(entry.get("text", "") or "")[:300],
                    source="graph",
                    score=score,
                    reason="graph-empty text fallback",
                    evidence=EvidenceRef(sessionFile=session_file, messageIndex=message_index, snippet=str(entry.get("text", "") or "")[:200]),
                    session_key=session_file,
                    turn_index=message_index,
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def _recall_transcript_context(self, question: str, query_vector: np.ndarray, limit: int = 5) -> str:
        """Retrieve original transcript text matching the query for use as context fallback."""
        state = self._require_state()
        if not state.transcripts:
            return ""
        self._ensure_transcript_vectors_loaded()
        query_tokens = tokenize(question)
        scored: list[tuple[TranscriptEntry, float]] = []
        for entry in state.transcripts:
            # Include speaker in the searchable text so queries mentioning a person's name
            # can match entries where that person is speaking.
            search_tokens = list(entry.tokens)
            if entry.speaker:
                search_tokens.extend(tokenize(entry.speaker))
            lexical = keyword_overlap(query_tokens, search_tokens)
            entry_vector = getattr(entry, "vector", None)
            semantic = cosine_similarity(query_vector, entry_vector) if entry_vector is not None else 0.0
            score = lexical * 1.5 + semantic * 1.0
            if score > 0:
                scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        selected_windows: list[str] = []
        seen_centers: set[tuple[str, int]] = set()
        for entry, _score in scored:
            center_key = (entry.session_key, entry.turn_index)
            if center_key in seen_centers:
                continue
            seen_centers.add(center_key)
            neighbors = [
                candidate
                for candidate in state.transcripts
                if candidate.session_key == entry.session_key and abs(candidate.turn_index - entry.turn_index) <= self.config.transcriptWindowRadius
            ]
            if not neighbors:
                continue
            neighbors.sort(key=lambda item: item.turn_index)
            window_lines = []
            for neighbor in neighbors:
                date_part = f"[{neighbor.date_time}] " if neighbor.date_time else ""
                window_lines.append(f"{date_part}[{neighbor.speaker}]: {neighbor.text[:self.config.transcriptSnippetMaxChars]}")
            selected_windows.append("\n".join(window_lines))
            if len(selected_windows) >= limit:
                break
        return "\n\n".join(selected_windows)

    def _build_transcript_evidence_map(self, session_id: str, session_file: str | None, messages: Sequence[dict[str, Any]]) -> dict[int, EvidenceRef]:
        evidence_by_index: dict[int, EvidenceRef] = {}
        if not session_file:
            return evidence_by_index
        transcript_entries = self._store.list_transcript_evidence_entries(session_id, session_file)
        if not transcript_entries:
            return evidence_by_index
        comparable_messages = [
            {
                "index": index,
                "role": str(message.get("role", "") or ""),
                "text": normalize_whitespace(messageToText(message)),
            }
            for index, message in enumerate(messages)
            if normalize_whitespace(messageToText(message))
        ]
        transcript_cursor = len(transcript_entries) - 1
        for message_cursor in range(len(comparable_messages) - 1, -1, -1):
            current = comparable_messages[message_cursor]
            saved_cursor = transcript_cursor
            found = False
            while transcript_cursor >= 0:
                candidate = transcript_entries[transcript_cursor]
                if self._transcript_messages_match(
                    current["role"],
                    current["text"],
                    str(candidate.get("role", "") or ""),
                    normalize_whitespace(str(candidate.get("text", "") or "")),
                ):
                    candidate_evidence = candidate.get("evidence")
                    evidence_by_index[current["index"]] = EvidenceRef(
                        sessionFile=session_file,
                        messageIndex=int(candidate.get("messageIndex", 0) or 0),
                        startLine=getattr(candidate_evidence, "startLine", None) if candidate_evidence is not None else None,
                        endLine=getattr(candidate_evidence, "endLine", None) if candidate_evidence is not None else None,
                        snippet=(getattr(candidate_evidence, "snippet", None) if candidate_evidence is not None else None) or str(candidate.get("text", "") or "")[:200],
                    )
                    transcript_cursor -= 1
                    found = True
                    break
                transcript_cursor -= 1
            if not found:
                transcript_cursor = saved_cursor
        return evidence_by_index

    def _transcript_messages_match(self, left_role: str, left_text: str, right_role: str, right_text: str) -> bool:
        return left_role == right_role and normalize_whitespace(left_text) == normalize_whitespace(right_text)

    def _select_recent_messages(self, messages: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        spent = 0
        for index in range(len(messages) - 1, -1, -1):
            message = dict(messages[index])
            cost = len(tokenize(messageToText(message)))
            if selected and spent + cost > budget:
                break
            selected.insert(0, message)
            spent += cost
            if len(selected) >= self.config.recentMessageWindow:
                break
        return selected

    def _build_session_summary_lines(self, session_id: str, query: str, token_budget: int) -> list[str]:
        relevant = self._store.search_session_summaries(query, 10)
        recent = self._store.list_session_summaries_by_session(session_id, 20)
        seen_ids: set[str] = set()
        summaries: list[SessionSummary] = []
        for item in [*relevant, *recent]:
            summary_id = item.id or item.session_key
            if summary_id in seen_ids:
                continue
            seen_ids.add(summary_id)
            summaries.append(item)
        if not summaries:
            return []
        summaries.sort(key=lambda item: item.created_at)
        lines: list[str] = []
        spent = 0
        max_budget = min(int(token_budget * self.config.sessionSummaryBudgetRatio), self.config.sessionSummaryMaxTokens)
        for index, summary in enumerate(summaries):
            is_newest = index == len(summaries) - 1
            text = summary.overview if is_newest and summary.overview else summary.abstract
            cost = len(tokenize(text))
            if spent + cost > max_budget:
                break
            sid = summary.session_key or summary.session_id or summary.id
            lines.append(f"- [session:{sid}] {text}")
            spent += cost
        return lines

    def _estimate_text_tokens(self, lines: Sequence[str]) -> int:
        return sum(len(tokenize(line)) for line in lines)

    def _estimate_message_tokens(self, messages: Sequence[dict[str, Any]]) -> int:
        return sum(len(tokenize(messageToText(message))) for message in messages)

    def _build_episodic_evidence_lines(self, graph_hits: list[RecallHit], ledger_hits: list[RecallHit], max_tokens: int) -> list[str]:
        all_hits = [
            hit for hit in [*graph_hits, *ledger_hits]
            if hit.evidence is not None and hit.evidence.sessionFile and hit.evidence.messageIndex is not None
        ]
        all_hits.sort(key=lambda item: item.score, reverse=True)
        lines: list[str] = []
        spent = 0
        seen_windows: dict[tuple[str, int], list[str]] = {}
        window_size = self.config.episodicMessageWindow
        for hit in all_hits[: self.config.episodicHitsLimit]:
            if hit.evidence is None or hit.evidence.messageIndex is None:
                continue
            window_start = max(0, hit.evidence.messageIndex - window_size)
            window_key = (hit.evidence.sessionFile, window_start)
            if window_key in seen_windows:
                seen_windows[window_key].append(hit.title)
                continue
            seen_windows[window_key] = [hit.title]
            messages = self._store.get_episodic_messages(hit.evidence.sessionFile, hit.evidence.messageIndex, window_size)
            if not messages:
                continue
            snippets = [f"[{str(message.get('role', '')).upper()}] {str(message.get('text', '') or '')[:200]}" for message in messages]
            trace_text = f"[{hit.title}]\n" + "\n".join(snippets)
            cost = len(tokenize(trace_text))
            if spent + cost > max_tokens:
                break
            lines.append(trace_text)
            spent += cost
        for line_idx, line in enumerate(lines):
            first_title = line.split("\n", 1)[0]
            for window_key, titles in seen_windows.items():
                if len(titles) > 1 and f"[{titles[0]}]" == first_title:
                    extra_labels = ", ".join(titles[1:])
                    lines[line_idx] = f"[{titles[0]}] (also: {extra_labels})\n" + line.split("\n", 1)[1]
                    break
        return lines

    def _build_system_prompt_addition(self, trace: dict[str, Any], session_summary_lines: list[str], episodic_lines: list[str]) -> str:
        graph_hits = trace.get("graphHits", []) or []
        entity_chains: list[str] = []
        entity_summaries: list[str] = []
        other_hits: list[str] = []
        for hit in graph_hits:
            reason = str(hit.get("reason", "") or "")
            content = str(hit.get("content", "") or "")
            title = str(hit.get("title", "") or "")
            if "relationship chain" in reason:
                entity_chains.append(content)
            elif "entity" in reason:
                # 防御层：即使 graph plane 未过滤，此处也裁剪 event 噪声。
                # entity summary 格式为 "label\n- attr: ...\n- event: ..."，
                # 若 event 行占比 > 60% 且总行数 > 4，说明该实体缺少语义属性，
                # 截断 event 行只保留前 2 条，避免浪费 token 预算。
                lines = content.split("\n")
                event_lines = [l for l in lines if l.strip().startswith("- event:")]
                non_event_lines = [l for l in lines if not l.strip().startswith("- event:")]
                if len(event_lines) > 2 and len(lines) > 4 and len(event_lines) / max(len(lines) - 1, 1) > 0.6:
                    content = "\n".join(non_event_lines + event_lines[:2])
                entity_summaries.append(content)
            else:
                other_hits.append(f"{title}: {content}")
        sections = []
        packet = trace.get("packet", {}) or {}
        for section in packet.get("sections", []) or []:
            sections.append(f"{section.get('title', '')}:\n" + "\n".join(section.get("lines", []) or []))
        parts = [
            "Memory context for this query. Use these facts to answer directly and confidently.",
            "If the memory contains relevant information, use it to answer even if not perfectly verified.",
        ]
        if session_summary_lines:
            parts.append("Session History:\n" + "\n".join(session_summary_lines))
        if entity_summaries or entity_chains:
            graph_parts: list[str] = []
            if entity_summaries:
                graph_parts.append("Known Entities:\n" + "\n\n".join(entity_summaries))
            if entity_chains:
                graph_parts.append("Relationship Chains:\n" + "\n".join(entity_chains))
            parts.append("\n\n".join(graph_parts))
        if sections:
            parts.append("\n\n".join(sections))
        if episodic_lines:
            parts.append("Evidence Traces:\n" + "\n".join(episodic_lines))
        return "\n\n".join(part for part in parts if part)

    def _build_line_map(self, session_file: str, message_count: int) -> dict[int, int]:
        line_map: dict[int, int] = {}
        if not session_file or not os.path.exists(session_file):
            return line_map
        try:
            lines = Path(session_file).read_text(encoding="utf-8").splitlines()
        except OSError:
            return line_map
        message_index = 0
        for line_number, line in enumerate(lines, start=1):
            if message_index >= message_count or not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("type") != "message":
                continue
            line_map[message_index] = line_number
            message_index += 1
        return line_map

    def _build_episodic_lines(self, graph_hits: list[RecallHit], ledger_hits: list[RecallHit], limit: int) -> list[str]:
        if limit <= 0:
            return []
        state = self._require_state()
        selected = [hit for hit in (graph_hits + ledger_hits) if hit.turn_index is not None][:limit]
        lines: list[str] = []
        for hit in selected:
            if hit.turn_index is None:
                continue
            neighbors = [
                entry
                for entry in state.transcripts
                if entry.session_key == hit.session_key and abs(entry.turn_index - hit.turn_index) <= 1
            ]
            if not neighbors:
                continue
            snippets = [f"[{entry.speaker}] {_clip(entry.text or entry.content, 160)}" for entry in neighbors]
            lines.append(f"{hit.title}\n" + "\n".join(snippets))
        return lines

    def _build_context(
        self,
        question: str,
        classification: ClassificationResult,
        plan: Any,
        graph_hits: list[RecallHit],
        ledger_hits: list[RecallHit],
        summary_hits: list[RecallHit],
        community_hits: list[RecallHit],
    ) -> str:
        parts = [
            "Answer from the ranked memory evidence below. Include specific names, dates, and details.",
            f"Intent={classification.intent}; Complexity={classification.complexity}.",
        ]
        ranked_hits = self._rank_combined_hits(graph_hits, ledger_hits, summary_hits, community_hits)
        evidence_lines = [
            f"- {hit.title}: {_clip(hit.content, 110 if plan.complexity == 'simple' else 130)}"
            for hit in ranked_hits[: max(3, plan.graph_item_limit + plan.ledger_item_limit)]
        ]
        if evidence_lines:
            parts.append("Ranked Evidence:\n" + "\n".join(evidence_lines))
        episodic_lines = self._build_episodic_lines(graph_hits, ledger_hits, plan.episodic_limit if plan.include_episodic else 0)
        if episodic_lines:
            compact_episodic = [ _clip(line.replace("\n", " | "), 150) for line in episodic_lines[:1] ]
            parts.append("Trace:\n" + "\n".join(f"- {line}" for line in compact_episodic))
        parts.append(f"Question: {question}")
        return "\n\n".join(parts)

    def _answer(self, question: str, context: str, use_aaak: bool = False) -> ChatResult:
        if not self._answer_llm.is_enabled or self._answer_llm_disabled:
            raise RuntimeError("EBM answer LLM is unavailable")

        system_prompt = AAAK_ANSWER_SYSTEM if use_aaak else ANSWER_SYSTEM
        last_empty_result: ChatResult | None = None
        for attempt in range(EMPTY_ANSWER_MAX_RETRIES + 1):
            try:
                try:
                    result = self._answer_llm.chat(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": context + "\n\nReturn JSON only with keys {\"answer\": string, \"normalized_answer\": string, \"confidence\": number, \"kind\": string}. If you can normalize a relative time using the dated evidence, put the final form in normalized_answer."},
                        ],
                        temperature=0.0,
                        max_tokens=220,
                        response_format={"type": "json_object"},
                    )
                except TypeError:
                    result = self._answer_llm.chat(
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": context},
                        ],
                        temperature=0.0,
                        max_tokens=220,
                    )
            except Exception as error:
                self._answer_llm_failures += 1
                if self._answer_llm_failures >= 3:
                    self._answer_llm_disabled = True
                raise RuntimeError(f"EBM answer generation failed: {error}") from error

            try:
                from ebm_context_engine.retrieval.intent_router import _robust_json_parse
                payload = _robust_json_parse(result.content)
                if payload:
                    normalized_candidate = str(payload.get("normalized_answer", "") or "").strip()
                    if normalized_candidate:
                        normalized = self._normalize_answer_text(normalized_candidate)
                    else:
                        normalized = self._normalize_answer_text(str(payload.get("answer", "") or ""))
                else:
                    normalized = self._normalize_answer_text(result.content)
            except Exception:
                normalized = self._normalize_answer_text(result.content)
            if normalized:
                self._answer_llm_failures = 0
                if self._looks_like_question_echo(question, normalized):
                    raise RuntimeError("EBM answer generation repeated the question")
                return ChatResult(
                    content=normalized,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    total_tokens=result.total_tokens,
                )

            last_empty_result = result
            if attempt < EMPTY_ANSWER_MAX_RETRIES:
                self.logger.warning(
                    "EBM answer generation returned empty content, retrying (%d/%d)",
                    attempt + 1,
                    EMPTY_ANSWER_MAX_RETRIES,
                )

        self.logger.warning(
            "EBM answer generation returned empty content after %d retries, "
            "returning fallback answer",
            EMPTY_ANSWER_MAX_RETRIES,
        )
        tok = last_empty_result.total_tokens if last_empty_result else 0
        return ChatResult(
            content="Information not found.",
            prompt_tokens=last_empty_result.prompt_tokens if last_empty_result else 0,
            completion_tokens=0,
            total_tokens=tok,
        )

    def _normalize_answer_text(self, text: str) -> str:
        normalized = normalize_whitespace(text or "")
        if not normalized:
            return ""
        normalized = re.sub(r"(?is)<think>.*?</think>", " ", normalized).strip()
        normalized = re.sub(r"^(answer|final answer)\s*:\s*", "", normalized, flags=re.IGNORECASE).strip()
        normalized = normalized.strip("\"'` ")
        return normalize_whitespace(normalized)

    def _looks_like_question_echo(self, question: str, answer: str) -> bool:
        normalized_question = normalize_whitespace(question).rstrip("?.! ").lower()
        normalized_answer = normalize_whitespace(answer).rstrip("?.! ").lower()
        return bool(normalized_answer) and normalized_answer == normalized_question

    def _candidate_answer_lines(self, context: str) -> list[str]:
        lines: list[str] = []
        for raw_line in context.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.endswith(":"):
                continue
            if stripped.startswith("Answer using the supplied memory packet"):
                continue
            if stripped.startswith("Intent="):
                continue
            if stripped.startswith("Question:"):
                continue
            if stripped.startswith("- "):
                candidate = stripped[2:].strip()
                if candidate.startswith("[task] active goal:"):
                    continue
                if candidate.endswith("?"):
                    continue
                lines.append(candidate)
                continue
            if re.match(r"^\[[^\]]+\]\s+\[[^\]]+\]:\s+", stripped):
                if stripped.endswith("?"):
                    continue
                lines.append(stripped)
                continue
            if re.match(r"^\[[A-Z]+\]\s+", stripped):
                if stripped.endswith("?"):
                    continue
                lines.append(stripped)
        return lines

    def _clean_candidate_line(self, line: str) -> str:
        cleaned = line.strip()
        cleaned = re.sub(r"^\[[^\]]+\]\s+\[[^\]]+\]:\s*", "", cleaned).strip()
        cleaned = re.sub(r"^\[[A-Z]+\]\s*", "", cleaned).strip()
        cleaned = re.sub(r"^[^:]+:\s*", "", cleaned).strip()
        return normalize_whitespace(cleaned)

    def _similarity_units(self, text: str) -> list[str]:
        normalized = normalize_whitespace(text or "").lower()
        if not normalized:
            return []
        ascii_tokens = tokenize(normalized)
        if ascii_tokens:
            return ascii_tokens
        return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", normalized, flags=re.IGNORECASE)

    def _candidate_similarity(self, question: str, candidate: str) -> float:
        query_units = self._similarity_units(question)
        candidate_units = self._similarity_units(candidate)
        return keyword_overlap(query_units, candidate_units)

    def _score_candidates(self, question: str, lines: Sequence[str]) -> list[tuple[str, float]]:
        scored = sorted(
            ((line, self._candidate_similarity(question, line)) for line in lines),
            key=lambda item: item[1],
            reverse=True,
        )
        return scored

    def _payload_to_entries(self, payload: Sequence[dict[str, Any]]) -> list[TranscriptEntry]:
        return payloadToEntries(payload)

    def _resolve_node_label(self, node_id: str) -> str:
        return resolveNodeLabel(self._require_state(), node_id)

    def _hit_payload(self, hit: RecallHit) -> dict[str, Any]:
        return {
            "id": hit.id,
            "title": hit.title,
            "content": hit.content,
            "source": hit.source,
            "score": hit.score,
            "reason": hit.reason,
            "session_key": hit.session_key,
            "turn_index": hit.turn_index,
            "evidence": asdict(hit.evidence) if hit.evidence else None,
        }

    def _transcript_payload(self, entry: TranscriptEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "session_id": entry.session_id,
            "session_key": entry.session_key,
            "session_file": entry.session_file,
            "date_time": entry.date_time,
            "turn_index": entry.turn_index,
            "speaker": entry.speaker,
            "text": entry.text,
            "content": entry.content,
            "tokens": list(entry.tokens),
            "entity_ids": list(entry.entity_ids),
            "created_at": entry.created_at,
            "vector": entry.vector.tolist() if entry.vector is not None else None,
            "evidence": asdict(entry.evidence),
        }

    def _entity_payload(self, node: EntityNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "node_type": "ENTITY",
            "label": node.name,
            "content": node.description,
            "keywords": list(node.tokens),
            "confidence": min(0.99, 0.6 + node.mention_count * 0.05),
            "session_keys": sorted(node.session_keys),
        }

    def _event_payload(self, node: EventNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "node_type": "EVENT",
            "label": f"{node.session_key} / turn {node.turn_index + 1} / {node.speaker}",
            "content": node.content,
            "keywords": list(node.tokens),
            "confidence": 0.9,
            "evidence": asdict(node.evidence) if node.evidence else None,
        }

    def _fact_payload(self, node: LedgerFact) -> dict[str, Any]:
        return {
            "id": node.id,
            "node_type": "FACT",
            "label": f"{node.scope} / {node.key}",
            "content": node.text,
            "keywords": list(node.tokens),
            "confidence": node.confidence,
            "evidence": asdict(node.evidence) if node.evidence else None,
        }

    def _session_summary_payload(self, item: SessionSummary) -> dict[str, Any]:
        return {
            "id": item.id or item.session_key,
            "node_type": "SESSION_SUMMARY",
            "session_key": item.session_key,
            "session_id": item.session_id,
            "session_file": item.session_file,
            "date_time": item.date_time,
            "abstract": item.abstract,
            "overview": item.overview,
            "message_count": item.message_count,
            "created_at": item.created_at,
            "source_event_ids": list(item.source_event_ids),
        }

    def _hm_topic_payload(self, item: HmTopic) -> dict[str, Any]:
        return {
            "id": item.id,
            "node_type": "HM_TOPIC",
            "title": item.title,
            "content": item.summary,
            "keywords": list(item.keywords),
            "episode_ids": list(item.episode_ids),
            "created_at": item.created_at,
            "updated_at": item.updated_at,
        }

    def _hm_episode_payload(self, item: HmEpisode) -> dict[str, Any]:
        return {
            "id": item.id,
            "node_type": "HM_EPISODE",
            "session_key": item.session_key,
            "title": item.title,
            "summary": item.summary,
            "content": item.dialogue,
            "keywords": list(item.keywords),
            "turn_start": item.turn_start,
            "turn_end": item.turn_end,
            "topic_ids": list(item.topic_ids),
            "fact_ids": list(item.fact_ids),
            "created_at": item.created_at,
        }

    def _hm_fact_payload(self, item: HmFact) -> dict[str, Any]:
        return {
            "id": item.id,
            "node_type": "HM_FACT",
            "content": item.content,
            "potential": item.potential,
            "keywords": list(item.keywords),
            "importance": item.importance,
            "episode_id": item.episode_id,
            "session_key": item.session_key,
            "source_turn_start": item.source_turn_start,
            "source_turn_end": item.source_turn_end,
            "created_at": item.created_at,
        }

    def _unified_fact_payload(self, item: UnifiedFact) -> dict[str, Any]:
        return {
            "id": item.id,
            "node_type": "UNIFIED_FACT",
            "content": item.content,
            "subject": item.subject,
            "key": item.key,
            "scope": item.scope,
            "potential": item.potential,
            "keywords": list(item.keywords),
            "importance": item.importance,
            "confidence": item.confidence,
            "evidence": asdict(item.evidence) if item.evidence else None,
            "episode_id": item.episode_id,
            "entity_ids": list(item.entity_ids),
            "session_key": item.session_key,
            "turn_index": item.turn_index,
            "source_turn_start": item.source_turn_start,
            "source_turn_end": item.source_turn_end,
            "status": item.status,
            "created_at": item.created_at,
        }

    def _edge_payload(self, edge: GraphEdgeRecord) -> dict[str, Any]:
        return {
            "id": edge.id,
            "from_id": edge.from_id,
            "to_id": edge.to_id,
            "edge_type": edge.edge_type,
            "weight": edge.weight,
            "relation_label": edge.relation_label,
            "evidence": asdict(edge.evidence) if edge.evidence else None,
        }

    def _community_payload(self, item: CommunitySummaryRecord) -> dict[str, Any]:
        return {
            "id": item.id,
            "title": item.title,
            "summary": item.summary,
            "keywords": list(item.keywords),
            "member_ids": list(item.member_ids),
        }


def _with_runtime_lock(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        lock = getattr(self, "_runtime_lock", None)
        if lock is None:
            return method(self, *args, **kwargs)
        with lock:
            return method(self, *args, **kwargs)
    return wrapper


for _name in (
    "close",
    "reset",
    "load",
    "save",
    "ensure_loaded",
    "_ensure_mutable_state",
    "bootstrap",
    "ingest",
    "ingestBatch",
    "after_turn",
    "afterTurn",
    "query",
    "assemble",
    "compact",
    "dispose",
    "memory_search",
    "memory_get",
    "memory_forget",
    "archive_expand",
):
    setattr(
        EvidenceBackedMemoryEngine,
        _name,
        _with_runtime_lock(getattr(EvidenceBackedMemoryEngine, _name)),
    )


PythonEbmEngine = EvidenceBackedMemoryEngine
