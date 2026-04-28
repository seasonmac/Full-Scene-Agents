from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class EvidenceRef:
    sessionFile: str
    messageIndex: Optional[int] = None
    startLine: Optional[int] = None
    endLine: Optional[int] = None
    snippet: Optional[str] = None
    dateTime: str = ""
    speaker: str = ""

    @property
    def session_key(self) -> str:
        return self.sessionFile

    @property
    def date_time(self) -> str:
        return self.dateTime

    @property
    def turn_index(self) -> int:
        return int(self.messageIndex or 0)


@dataclass
class TranscriptEntry:
    id: str
    session_key: str
    date_time: str
    turn_index: int
    speaker: str
    text: str
    content: str
    tokens: list[str]
    entity_ids: list[str]
    evidence: EvidenceRef
    session_id: str = ""
    session_file: str = ""
    created_at: int = 0
    vector: Optional[np.ndarray] = None

    @property
    def messageIndex(self) -> int:
        return self.turn_index

    @property
    def role(self) -> str:
        return self.speaker


@dataclass
class PinnedEntry:
    id: str
    sessionId: str
    scope: str
    label: str
    content: str
    priority: float
    tokenCost: int
    evidence: Optional[EvidenceRef] = None


@dataclass
class TopicEntry:
    sessionId: str
    topic: str
    score: float
    source: str


@dataclass
class ScratchpadEntry:
    id: str
    sessionId: str
    kind: str
    content: str
    tokenCost: int
    createdAt: int
    evidence: Optional[EvidenceRef] = None


@dataclass
class EntityNode:
    id: str
    name: str
    tokens: list[str] = field(default_factory=list)
    description: str = ""
    snippets: list[str] = field(default_factory=list)
    session_keys: set[str] = field(default_factory=set)
    mention_count: int = 0
    vector: Optional[np.ndarray] = None


@dataclass
class EventNode:
    id: str
    session_key: str
    date_time: str
    turn_index: int
    speaker: str
    text: str
    content: str
    tokens: list[str]
    entity_ids: list[str] = field(default_factory=list)
    evidence: Optional[EvidenceRef] = None
    vector: Optional[np.ndarray] = None


@dataclass
class LedgerFact:
    id: str
    subject: str
    key: str
    scope: str
    value: str
    text: str
    session_key: str
    turn_index: int
    tokens: list[str]
    evidence: Optional[EvidenceRef] = None
    subject_entity_id: str = ""
    confidence: float = 0.7
    validFrom: int = 0
    validTo: Optional[int] = None
    invalidAt: Optional[int] = None
    expiresAt: Optional[int] = None
    source: str = "local-distillation"
    status: str = "active"
    vector: Optional[np.ndarray] = None


@dataclass
class SessionSummary:
    session_key: str
    date_time: str
    abstract: str
    overview: str
    tokens: list[str]
    source_event_ids: list[str] = field(default_factory=list)
    vector: Optional[np.ndarray] = None
    id: str = ""
    session_id: str = ""
    session_file: str = ""
    message_count: int = 0
    created_at: int = 0


@dataclass
class GraphEdgeRecord:
    id: str
    from_id: str
    to_id: str
    edge_type: str
    weight: float
    relation_label: str = ""
    evidence: Optional[EvidenceRef] = None


@dataclass
class CommunitySummaryRecord:
    id: str
    title: str
    summary: str
    keywords: list[str]
    member_ids: list[str] = field(default_factory=list)
    vector: Optional[np.ndarray] = None


@dataclass
class RecallHit:
    id: str
    title: str
    content: str
    source: str
    score: float
    reason: str
    evidence: Optional[EvidenceRef] = None
    session_key: str = ""
    turn_index: Optional[int] = None
    verified: Optional[bool] = None
    verificationNote: Optional[str] = None


@dataclass
class ClassificationResult:
    intent: str
    complexity: str
    confidence: float
    source: str
    weights: dict[str, float] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    target_slots: list[str] = field(default_factory=list)
    focus_terms: list[str] = field(default_factory=list)
    reasoning_modes: list[str] = field(default_factory=list)
    time_scope: str = ""
    answer_type: str = ""


@dataclass
class QueryPlan:
    intent: str
    complexity: str
    graph_top_k: int
    ledger_top_k: int
    summary_top_k: int
    community_top_k: int
    graph_item_limit: int
    ledger_item_limit: int
    summary_item_limit: int
    community_item_limit: int
    episodic_limit: int
    max_hops: int
    max_seed_events: int
    include_summaries: bool = True
    include_communities: bool = True
    include_episodic: bool = True
    prefer_entity_expansion: bool = False
    prefer_temporal: bool = False


@dataclass
class PythonEbmQueryResult:
    answer: str
    context: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    answer_source: str = ""
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class HmTopic:
    """Layer 0 — Topic 层节点（合并原 CommunitySummaryRecord 功能）"""
    id: str
    title: str
    summary: str
    keywords: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    member_entity_ids: list[str] = field(default_factory=list)
    source: str = "llm_aggregation"  # "llm_aggregation" | "community_detection"
    vector: Optional[np.ndarray] = None
    created_at: int = 0
    updated_at: int = 0


@dataclass
class HmEpisode:
    """Layer 1 — Episode 层节点（合并原 SessionSummary 功能）"""
    id: str
    session_key: str
    title: str
    summary: str
    dialogue: str
    keywords: list[str] = field(default_factory=list)
    timestamp_start: str = ""
    timestamp_end: str = ""
    turn_start: int = 0
    turn_end: int = 0
    topic_ids: list[str] = field(default_factory=list)
    fact_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    source_event_ids: list[str] = field(default_factory=list)
    is_session_summary: bool = False
    vector: Optional[np.ndarray] = None
    created_at: int = 0


@dataclass
class HmFact:
    """三级超图 — Fact 层节点（原子事实 + query anticipation）— deprecated, use UnifiedFact"""
    id: str
    content: str
    potential: str
    keywords: list[str] = field(default_factory=list)
    importance: str = "mid"  # high / mid / low
    episode_id: str = ""
    session_key: str = ""
    source_turn_start: int = 0
    source_turn_end: int = 0
    vector: Optional[np.ndarray] = None
    created_at: int = 0


@dataclass
class UnifiedFact:
    """Layer 2 — 统一事实节点（合并 LedgerFact + HmFact）"""
    id: str
    content: str
    subject: str = ""
    key: str = ""
    scope: str = ""
    potential: str = ""
    keywords: list[str] = field(default_factory=list)
    importance: str = "mid"  # high / mid / low
    confidence: float = 0.7
    evidence: Optional[EvidenceRef] = None
    episode_id: str = ""
    entity_ids: list[str] = field(default_factory=list)
    session_key: str = ""
    turn_index: int = 0
    source_turn_start: int = 0
    source_turn_end: int = 0
    validFrom: int = 0
    validTo: Optional[int] = None
    invalidAt: Optional[int] = None
    expiresAt: Optional[int] = None
    source: str = "local-distillation"
    status: str = "active"
    vector: Optional[np.ndarray] = None
    created_at: int = 0


@dataclass
class MemoryPacketSection:
    title: str
    lines: list[str]
    tokenCost: int


@dataclass
class MemoryPacket:
    query: str
    totalEstimatedTokens: int
    sections: list[MemoryPacketSection]
    traceId: str


@dataclass
class AssembleTrace:
    traceId: str
    sessionId: str
    query: str
    budget: int
    packet: MemoryPacket
    graphHits: list[RecallHit]
    ledgerHits: list[RecallHit]
    degradedPaths: list[str]
    verifyReasons: list[str]
    latencyMs: int
    createdAt: int


@dataclass
class DistillTurnInput:
    sessionId: str
    sessionKey: Optional[str]
    sessionFile: str
    query: str
    turnMessagesText: list[str]
    turnMessageIndexes: list[int]
    turnStartIndex: int
    turnFingerprint: str
    turnTimestamp: int


@dataclass
class MemoryState:
    version: int
    speaker_names: list[str] = field(default_factory=list)
    transcripts: list[TranscriptEntry] = field(default_factory=list)
    events: dict[str, EventNode] = field(default_factory=dict)
    entities: dict[str, EntityNode] = field(default_factory=dict)
    facts: dict[str, LedgerFact] = field(default_factory=dict)
    session_summaries: dict[str, SessionSummary] = field(default_factory=dict)
    graph_edges: dict[str, GraphEdgeRecord] = field(default_factory=dict)
    communities: dict[str, CommunitySummaryRecord] = field(default_factory=dict)
    hm_topics: dict[str, HmTopic] = field(default_factory=dict)
    hm_episodes: dict[str, HmEpisode] = field(default_factory=dict)
    hm_facts: dict[str, HmFact] = field(default_factory=dict)
    unified_facts: dict[str, UnifiedFact] = field(default_factory=dict)
    traces: list[dict[str, Any]] = field(default_factory=list)
    slow_path_jobs: list[dict[str, Any]] = field(default_factory=list)
    adjacency: dict[str, set[str]] = field(default_factory=dict)
    entity_to_events: dict[str, set[str]] = field(default_factory=dict)
    event_index: dict[str, set[str]] = field(default_factory=dict)
    fact_index: dict[str, set[str]] = field(default_factory=dict)
    summary_index: dict[str, set[str]] = field(default_factory=dict)
    entity_index: dict[str, set[str]] = field(default_factory=dict)
    community_index: dict[str, set[str]] = field(default_factory=dict)
    token_idf: dict[str, float] = field(default_factory=dict)
    artifact_created_at: float = 0.0
