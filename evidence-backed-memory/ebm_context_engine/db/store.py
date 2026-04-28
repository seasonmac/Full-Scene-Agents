from __future__ import annotations

import json
import os
import sqlite3
import time
import threading
import functools
from pathlib import Path
from typing import Any, Iterable

import logging

import numpy as np

from ebm_context_engine.types import (
    PinnedEntry,
    ScratchpadEntry,
    TopicEntry,
    CommunitySummaryRecord,
    EntityNode,
    EventNode,
    EvidenceRef,
    GraphEdgeRecord,
    HmEpisode,
    HmFact,
    HmTopic,
    LedgerFact,
    SessionSummary,
    TranscriptEntry,
    UnifiedFact,
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Any, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _vec_to_json(vector) -> str | None:
    if vector is None:
        return None
    if isinstance(vector, list):
        return _json_dumps(vector)
    return _json_dumps(vector.tolist())


def _vec_from_json(raw: Any) -> np.ndarray | None:
    values = _json_loads(raw, None)
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def _evidence_from_json(raw: Any) -> EvidenceRef | None:
    payload = _json_loads(raw, None)
    if not isinstance(payload, dict):
        return None
    try:
        return EvidenceRef(
            sessionFile=str(payload.get("sessionFile", "") or payload.get("session_file", "") or payload.get("session_key", "") or ""),
            messageIndex=int(payload.get("messageIndex", payload.get("message_index", payload.get("turn_index", 0))) or 0),
            startLine=payload.get("startLine", payload.get("start_line")),
            endLine=payload.get("endLine", payload.get("end_line")),
            snippet=str(payload.get("snippet", "") or ""),
            dateTime=str(payload.get("dateTime", payload.get("date_time", "")) or ""),
            speaker=str(payload.get("speaker", "") or ""),
        )
    except Exception:
        return None


def _evidence_to_json(evidence: EvidenceRef | None) -> str | None:
    if evidence is None:
        return None
    return _json_dumps(
        {
            "sessionFile": evidence.sessionFile,
            "messageIndex": evidence.messageIndex,
            "startLine": evidence.startLine,
            "endLine": evidence.endLine,
            "speaker": evidence.speaker,
            "dateTime": evidence.dateTime,
            "snippet": evidence.snippet,
        }
    )


def _build_fts_query(tokens: Iterable[str]) -> str:
    cleaned = [token.replace('"', "").strip() for token in tokens if token.strip()]
    if not cleaned:
        return '"memory"'
    return " AND ".join(f'"{token}"' for token in cleaned[:8])


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS transcript_entries (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL DEFAULT '',
  session_key TEXT NOT NULL,
  session_file TEXT NOT NULL DEFAULT '',
  date_time TEXT NOT NULL,
  message_index INTEGER NOT NULL DEFAULT 0,
  turn_index INTEGER NOT NULL,
  role TEXT NOT NULL DEFAULT '',
  speaker TEXT NOT NULL,
  text TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at INTEGER NOT NULL DEFAULT 0,
  tokens_json TEXT NOT NULL,
  entity_ids_json TEXT NOT NULL,
  evidence_json TEXT,
  vector_json TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS transcript_entries_fts USING fts5(id UNINDEXED, speaker, text, content, session_key);

CREATE TABLE IF NOT EXISTS pinned_entries (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  scope TEXT NOT NULL,
  label TEXT NOT NULL,
  content TEXT NOT NULL,
  priority REAL NOT NULL,
  token_cost INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS topic_entries (
  session_id TEXT NOT NULL,
  topic TEXT NOT NULL,
  score REAL NOT NULL,
  source TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS scratchpad_entries (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  content TEXT NOT NULL,
  token_cost INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  tokens_json TEXT NOT NULL,
  snippets_json TEXT NOT NULL,
  session_keys_json TEXT NOT NULL,
  mention_count INTEGER NOT NULL,
  vector_json TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(id UNINDEXED, name, description);

CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  session_key TEXT NOT NULL,
  date_time TEXT NOT NULL,
  turn_index INTEGER NOT NULL,
  speaker TEXT NOT NULL,
  text TEXT NOT NULL,
  content TEXT NOT NULL,
  tokens_json TEXT NOT NULL,
  entity_ids_json TEXT NOT NULL,
  evidence_json TEXT,
  vector_json TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(id UNINDEXED, speaker, text, content, session_key);

CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY,
  subject TEXT NOT NULL,
  key TEXT NOT NULL,
  scope TEXT NOT NULL,
  value TEXT NOT NULL,
  text TEXT NOT NULL,
  session_key TEXT NOT NULL,
  turn_index INTEGER NOT NULL,
  tokens_json TEXT NOT NULL,
  evidence_json TEXT,
  subject_entity_id TEXT NOT NULL,
  confidence REAL NOT NULL,
  valid_from INTEGER NOT NULL DEFAULT 0,
  valid_to INTEGER,
  invalid_at INTEGER,
  expires_at INTEGER,
  source TEXT NOT NULL DEFAULT 'local-distillation',
  status TEXT NOT NULL DEFAULT 'active',
  vector_json TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(id UNINDEXED, subject, key, value, text);

CREATE TABLE IF NOT EXISTS session_summaries (
  session_key TEXT PRIMARY KEY,
  id TEXT NOT NULL DEFAULT '',
  session_id TEXT NOT NULL DEFAULT '',
  session_file TEXT NOT NULL DEFAULT '',
  date_time TEXT NOT NULL,
  abstract TEXT NOT NULL,
  overview TEXT NOT NULL,
  message_count INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL DEFAULT 0,
  tokens_json TEXT NOT NULL,
  source_event_ids_json TEXT NOT NULL,
  vector_json TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS session_summaries_fts USING fts5(session_key UNINDEXED, abstract, overview);

CREATE TABLE IF NOT EXISTS graph_edges (
  id TEXT PRIMARY KEY,
  from_id TEXT NOT NULL,
  to_id TEXT NOT NULL,
  edge_type TEXT NOT NULL,
  weight REAL NOT NULL,
  relation_label TEXT NOT NULL,
  evidence_json TEXT
);

CREATE TABLE IF NOT EXISTS communities (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  keywords_json TEXT NOT NULL,
  member_ids_json TEXT NOT NULL,
  vector_json TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS communities_fts USING fts5(id UNINDEXED, title, summary, keywords);

CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL DEFAULT '',
  query TEXT NOT NULL DEFAULT '',
  latency_ms INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  trace_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slow_path_jobs (
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  status TEXT NOT NULL,
  attempts INTEGER NOT NULL,
  last_error TEXT NOT NULL,
  query TEXT NOT NULL,
  job_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hm_topics (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  keywords_json TEXT NOT NULL DEFAULT '[]',
  episode_ids_json TEXT NOT NULL DEFAULT '[]',
  vector_json TEXT,
  created_at INTEGER NOT NULL DEFAULT 0,
  updated_at INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS hm_topics_fts USING fts5(id UNINDEXED, title, summary, keywords);

CREATE TABLE IF NOT EXISTS hm_episodes (
  id TEXT PRIMARY KEY,
  session_key TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  dialogue TEXT NOT NULL DEFAULT '',
  keywords_json TEXT NOT NULL DEFAULT '[]',
  timestamp_start TEXT NOT NULL DEFAULT '',
  timestamp_end TEXT NOT NULL DEFAULT '',
  turn_start INTEGER NOT NULL DEFAULT 0,
  turn_end INTEGER NOT NULL DEFAULT 0,
  topic_ids_json TEXT NOT NULL DEFAULT '[]',
  fact_ids_json TEXT NOT NULL DEFAULT '[]',
  vector_json TEXT,
  created_at INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS hm_episodes_fts USING fts5(id UNINDEXED, title, summary, dialogue, keywords);

CREATE TABLE IF NOT EXISTS hm_facts (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  potential TEXT NOT NULL DEFAULT '',
  keywords_json TEXT NOT NULL DEFAULT '[]',
  importance TEXT NOT NULL DEFAULT 'mid',
  episode_id TEXT NOT NULL DEFAULT '',
  session_key TEXT NOT NULL DEFAULT '',
  source_turn_start INTEGER NOT NULL DEFAULT 0,
  source_turn_end INTEGER NOT NULL DEFAULT 0,
  vector_json TEXT,
  created_at INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS hm_facts_fts USING fts5(id UNINDEXED, content, potential, keywords);

CREATE TABLE IF NOT EXISTS unified_facts (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  subject TEXT NOT NULL DEFAULT '',
  key TEXT NOT NULL DEFAULT '',
  scope TEXT NOT NULL DEFAULT '',
  potential TEXT NOT NULL DEFAULT '',
  keywords_json TEXT NOT NULL DEFAULT '[]',
  importance TEXT NOT NULL DEFAULT 'mid',
  confidence REAL NOT NULL DEFAULT 0.7,
  evidence_json TEXT,
  episode_id TEXT NOT NULL DEFAULT '',
  entity_ids_json TEXT NOT NULL DEFAULT '[]',
  session_key TEXT NOT NULL DEFAULT '',
  turn_index INTEGER NOT NULL DEFAULT 0,
  source_turn_start INTEGER NOT NULL DEFAULT 0,
  source_turn_end INTEGER NOT NULL DEFAULT 0,
  valid_from INTEGER NOT NULL DEFAULT 0,
  valid_to INTEGER,
  invalid_at INTEGER,
  expires_at INTEGER,
  source TEXT NOT NULL DEFAULT 'local-distillation',
  status TEXT NOT NULL DEFAULT 'active',
  vector_json TEXT,
  created_at INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS unified_facts_fts USING fts5(id UNINDEXED, content, potential, keywords);
"""


logger = logging.getLogger("ebm_context_engine.db.store")


class EbmStore:
    def __init__(self, db_path: str):
        logger.info("初始化 EbmStore, 数据库路径: %s", db_path)
        self.db_path = db_path
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self._ensure_schema_migrations()
        logger.info("EbmStore 初始化完成, 数据库已就绪")

    def _ensure_schema_migrations(self) -> None:
        logger.info("开始执行数据库 schema 迁移检查")
        wanted = {
            "valid_from": "ALTER TABLE facts ADD COLUMN valid_from INTEGER NOT NULL DEFAULT 0",
            "valid_to": "ALTER TABLE facts ADD COLUMN valid_to INTEGER",
            "invalid_at": "ALTER TABLE facts ADD COLUMN invalid_at INTEGER",
            "expires_at": "ALTER TABLE facts ADD COLUMN expires_at INTEGER",
            "source": "ALTER TABLE facts ADD COLUMN source TEXT NOT NULL DEFAULT 'local-distillation'",
            "status": "ALTER TABLE facts ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
        }
        existing = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        with self.conn:
            for column, stmt in wanted.items():
                if column not in existing:
                    logger.debug("迁移 facts 表: 添加列 %s", column)
                    self.conn.execute(stmt)
        self._ensure_table_columns(
            "transcript_entries",
            {
                "session_id": "ALTER TABLE transcript_entries ADD COLUMN session_id TEXT NOT NULL DEFAULT ''",
                "session_file": "ALTER TABLE transcript_entries ADD COLUMN session_file TEXT NOT NULL DEFAULT ''",
                "message_index": "ALTER TABLE transcript_entries ADD COLUMN message_index INTEGER NOT NULL DEFAULT 0",
                "role": "ALTER TABLE transcript_entries ADD COLUMN role TEXT NOT NULL DEFAULT ''",
                "created_at": "ALTER TABLE transcript_entries ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0",
                "vector_json": "ALTER TABLE transcript_entries ADD COLUMN vector_json TEXT",
            },
        )
        self._ensure_table_columns(
            "session_summaries",
            {
                "id": "ALTER TABLE session_summaries ADD COLUMN id TEXT NOT NULL DEFAULT ''",
                "session_id": "ALTER TABLE session_summaries ADD COLUMN session_id TEXT NOT NULL DEFAULT ''",
                "session_file": "ALTER TABLE session_summaries ADD COLUMN session_file TEXT NOT NULL DEFAULT ''",
                "message_count": "ALTER TABLE session_summaries ADD COLUMN message_count INTEGER NOT NULL DEFAULT 0",
                "created_at": "ALTER TABLE session_summaries ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0",
            },
        )
        self._ensure_table_columns(
            "traces",
            {
                "session_id": "ALTER TABLE traces ADD COLUMN session_id TEXT NOT NULL DEFAULT ''",
                "query": "ALTER TABLE traces ADD COLUMN query TEXT NOT NULL DEFAULT ''",
                "latency_ms": "ALTER TABLE traces ADD COLUMN latency_ms INTEGER NOT NULL DEFAULT 0",
            },
        )

    def _ensure_table_columns(self, table: str, wanted: dict[str, str]) -> None:
        logger.debug("检查表 %s 的列迁移", table)
        existing = {
            row["name"]
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        with self.conn:
            for column, stmt in wanted.items():
                if column not in existing:
                    self.conn.execute(stmt)

    def close(self) -> None:
        logger.info("关闭数据库连接: %s", self.db_path)
        self.conn.close()

    def reset(self) -> None:
        logger.info("重置数据库: 清空所有表")
        self.conn.executescript(
            """
            DELETE FROM transcript_entries;
            DELETE FROM transcript_entries_fts;
            DELETE FROM pinned_entries;
            DELETE FROM topic_entries;
            DELETE FROM scratchpad_entries;
            DELETE FROM entities;
            DELETE FROM entities_fts;
            DELETE FROM events;
            DELETE FROM events_fts;
            DELETE FROM facts;
            DELETE FROM facts_fts;
            DELETE FROM session_summaries;
            DELETE FROM session_summaries_fts;
            DELETE FROM graph_edges;
            DELETE FROM communities;
            DELETE FROM communities_fts;
            DELETE FROM traces;
            DELETE FROM slow_path_jobs;
            DELETE FROM hm_topics;
            DELETE FROM hm_topics_fts;
            DELETE FROM hm_episodes;
            DELETE FROM hm_episodes_fts;
            DELETE FROM hm_facts;
            DELETE FROM hm_facts_fts;
            DELETE FROM unified_facts;
            DELETE FROM unified_facts_fts;
            """
        )
        self.conn.commit()
        logger.info("数据库重置完成")

    def upsert_pinned_entries(self, entries: list[PinnedEntry]) -> None:
        logger.info("写入置顶条目, 数量: %d", len(entries))
        with self.conn:
            updated_at = int(time.time() * 1000)
            for entry in entries:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO pinned_entries
                    (id, session_id, scope, label, content, priority, token_cost, updated_at, evidence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.id,
                        entry.sessionId,
                        entry.scope,
                        entry.label,
                        entry.content,
                        entry.priority,
                        entry.tokenCost,
                        updated_at,
                        _evidence_to_json(entry.evidence),
                    ),
                )

    def list_pinned_entries(self, session_id: str, limit: int) -> list[PinnedEntry]:
        logger.info("查询置顶条目, session_id=%s, limit=%d", session_id, limit)
        rows = self.conn.execute(
            """
            SELECT * FROM pinned_entries
            WHERE session_id = ?
            ORDER BY priority DESC, updated_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        logger.debug("查询置顶条目完成, 返回 %d 条", len(rows))
        return [
            PinnedEntry(
                id=row["id"],
                sessionId=row["session_id"],
                scope=row["scope"],
                label=row["label"],
                content=row["content"],
                priority=row["priority"],
                tokenCost=row["token_cost"],
                evidence=_evidence_from_json(row["evidence_json"]),
            )
            for row in rows
        ]

    def replace_topic_entries(self, session_id: str, entries: list[TopicEntry]) -> None:
        logger.info("替换主题条目, session_id=%s, 数量: %d", session_id, len(entries))
        with self.conn:
            self.conn.execute("DELETE FROM topic_entries WHERE session_id = ?", (session_id,))
            for entry in entries:
                self.conn.execute(
                    """
                    INSERT INTO topic_entries (session_id, topic, score, source, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, entry.topic, entry.score, entry.source, int(time.time() * 1000)),
                )

    def list_topic_entries(self, session_id: str, limit: int) -> list[TopicEntry]:
        logger.info("查询主题条目, session_id=%s, limit=%d", session_id, limit)
        rows = self.conn.execute(
            """
            SELECT session_id, topic, score, source
            FROM topic_entries
            WHERE session_id = ?
            ORDER BY score DESC, topic ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [
            TopicEntry(
                sessionId=row["session_id"],
                topic=row["topic"],
                score=row["score"],
                source=row["source"],
            )
            for row in rows
        ]

    def append_scratchpad_entries(self, entries: list[ScratchpadEntry], keep_window: int) -> None:
        logger.info("追加草稿条目, 数量: %d, 保留窗口: %d", len(entries), keep_window)
        with self.conn:
            for entry in entries:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO scratchpad_entries
                    (id, session_id, kind, content, token_cost, created_at, evidence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.id,
                        entry.sessionId,
                        entry.kind,
                        entry.content,
                        entry.tokenCost,
                        entry.createdAt,
                        _evidence_to_json(entry.evidence),
                    ),
                )
            session_ids = sorted({entry.sessionId for entry in entries})
            for session_id in session_ids:
                self.conn.execute(
                    """
                    DELETE FROM scratchpad_entries
                    WHERE session_id = ?
                      AND id NOT IN (
                        SELECT id FROM scratchpad_entries
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                      )
                    """,
                    (session_id, session_id, keep_window),
                )

    def list_scratchpad_entries(self, session_id: str, limit: int) -> list[ScratchpadEntry]:
        logger.info("查询草稿条目, session_id=%s, limit=%d", session_id, limit)
        rows = self.conn.execute(
            """
            SELECT * FROM scratchpad_entries
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [
            ScratchpadEntry(
                id=row["id"],
                sessionId=row["session_id"],
                kind=row["kind"],
                content=row["content"],
                tokenCost=row["token_cost"],
                createdAt=row["created_at"],
                evidence=_evidence_from_json(row["evidence_json"]),
            )
            for row in rows
        ]

    def clear_workspace_state(self, session_id: str) -> None:
        logger.info("清空工作区状态, session_id=%s", session_id)
        with self.conn:
            self.conn.execute("DELETE FROM pinned_entries WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM topic_entries WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM scratchpad_entries WHERE session_id = ?", (session_id,))

    def upsert_transcripts(self, entries: list[TranscriptEntry]) -> None:
        logger.info("写入对话记录, 数量: %d", len(entries))
        with self.conn:
            for entry in entries:
                session_id = entry.session_id or entry.session_key
                session_file = entry.session_file or (entry.evidence.sessionFile if entry.evidence else entry.session_key)
                message_index = entry.turn_index
                role = entry.speaker
                created_at = int(entry.created_at or 0)
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO transcript_entries
                    (id, session_id, session_key, session_file, date_time, message_index, turn_index, role, speaker, text, content, created_at, tokens_json, entity_ids_json, evidence_json, vector_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.id,
                        session_id,
                        entry.session_key,
                        session_file,
                        entry.date_time,
                        message_index,
                        entry.turn_index,
                        role,
                        entry.speaker,
                        entry.text,
                        entry.content,
                        created_at,
                        _json_dumps(entry.tokens),
                        _json_dumps(entry.entity_ids),
                        _evidence_to_json(entry.evidence),
                        _vec_to_json(entry.vector),
                    ),
                )
                self.conn.execute("DELETE FROM transcript_entries_fts WHERE id = ?", (entry.id,))
                self.conn.execute(
                    "INSERT INTO transcript_entries_fts (id, speaker, text, content, session_key) VALUES (?, ?, ?, ?, ?)",
                    (entry.id, entry.speaker, entry.text, entry.content, session_file or entry.session_key),
                )

    def upsert_entities(self, nodes: list[EntityNode]) -> None:
        logger.info("写入实体节点, 数量: %d", len(nodes))
        with self.conn:
            for node in nodes:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO entities
                    (id, name, description, tokens_json, snippets_json, session_keys_json, mention_count, vector_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.id,
                        node.name,
                        node.description,
                        _json_dumps(node.tokens),
                        _json_dumps(node.snippets),
                        _json_dumps(sorted(node.session_keys)),
                        node.mention_count,
                        _vec_to_json(node.vector),
                    ),
                )
                self.conn.execute("DELETE FROM entities_fts WHERE id = ?", (node.id,))
                self.conn.execute(
                    "INSERT INTO entities_fts (id, name, description) VALUES (?, ?, ?)",
                    (node.id, node.name, node.description),
                )

    def upsert_events(self, nodes: list[EventNode]) -> None:
        logger.info("写入事件节点, 数量: %d", len(nodes))
        with self.conn:
            for node in nodes:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO events
                    (id, session_key, date_time, turn_index, speaker, text, content, tokens_json, entity_ids_json, evidence_json, vector_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.id,
                        node.session_key,
                        node.date_time,
                        node.turn_index,
                        node.speaker,
                        node.text,
                        node.content,
                        _json_dumps(node.tokens),
                        _json_dumps(node.entity_ids),
                        _evidence_to_json(node.evidence),
                        _vec_to_json(node.vector),
                    ),
                )
                self.conn.execute("DELETE FROM events_fts WHERE id = ?", (node.id,))
                self.conn.execute(
                    "INSERT INTO events_fts (id, speaker, text, content, session_key) VALUES (?, ?, ?, ?, ?)",
                    (node.id, node.speaker, node.text, node.content, node.session_key),
                )

    def upsert_facts(self, nodes: list[LedgerFact]) -> None:
        logger.info("写入事实记录, 数量: %d", len(nodes))
        with self.conn:
            ordered_nodes = sorted(
                nodes,
                key=lambda item: (int(item.validFrom or 0), item.session_key, int(item.turn_index or 0), item.id),
            )
            for node in ordered_nodes:
                invalid_at = int(node.validFrom or int(time.time() * 1000))
                logger.debug("写入事实: id=%s, subject=%s, key=%s, status=%s", node.id, node.subject, node.key, node.status)
                self.conn.execute(
                    """
                    UPDATE facts
                    SET invalid_at = ?, status = 'invalidated'
                    WHERE subject = ? AND key = ? AND status = 'active' AND id <> ?
                    """,
                    (invalid_at, node.subject, node.key, node.id),
                )
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO facts
                    (id, subject, key, scope, value, text, session_key, turn_index, tokens_json, evidence_json, subject_entity_id, confidence, valid_from, valid_to, invalid_at, expires_at, source, status, vector_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node.id,
                        node.subject,
                        node.key,
                        node.scope,
                        node.value,
                        node.text,
                        node.session_key,
                        node.turn_index,
                        _json_dumps(node.tokens),
                        _evidence_to_json(node.evidence),
                        node.subject_entity_id,
                        node.confidence,
                        node.validFrom,
                        node.validTo,
                        node.invalidAt,
                        node.expiresAt,
                        node.source,
                        node.status,
                        _vec_to_json(node.vector),
                    ),
                )
                self.conn.execute("DELETE FROM facts_fts WHERE id = ?", (node.id,))
                self.conn.execute(
                    "INSERT INTO facts_fts (id, subject, key, value, text) VALUES (?, ?, ?, ?, ?)",
                    (node.id, node.subject, node.key, node.value, node.text),
                )

    def upsert_session_summaries(self, items: list[SessionSummary]) -> None:
        logger.info("写入会话摘要, 数量: %d", len(items))
        with self.conn:
            for item in items:
                summary_id = item.id or item.session_key
                session_id = item.session_id or item.session_key
                session_file = item.session_file or item.session_key
                created_at = int(item.created_at or 0)
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO session_summaries
                    (session_key, id, session_id, session_file, date_time, abstract, overview, message_count, created_at, tokens_json, source_event_ids_json, vector_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.session_key,
                        summary_id,
                        session_id,
                        session_file,
                        item.date_time,
                        item.abstract,
                        item.overview,
                        int(item.message_count or 0),
                        created_at,
                        _json_dumps(item.tokens),
                        _json_dumps(item.source_event_ids),
                        _vec_to_json(item.vector),
                    ),
                )
                self.conn.execute("DELETE FROM session_summaries_fts WHERE session_key = ?", (item.session_key,))
                self.conn.execute(
                    "INSERT INTO session_summaries_fts (session_key, abstract, overview) VALUES (?, ?, ?)",
                    (item.session_key, item.abstract, item.overview),
                )

    def upsert_edges(self, items: list[GraphEdgeRecord]) -> None:
        logger.info("写入图边, 数量: %d", len(items))
        with self.conn:
            for item in items:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO graph_edges
                    (id, from_id, to_id, edge_type, weight, relation_label, evidence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        item.from_id,
                        item.to_id,
                        item.edge_type,
                        item.weight,
                        item.relation_label,
                        _evidence_to_json(item.evidence),
                    ),
                )

    def upsert_communities(self, items: list[CommunitySummaryRecord]) -> None:
        logger.info("写入社区摘要, 数量: %d", len(items))
        with self.conn:
            for item in items:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO communities
                    (id, title, summary, keywords_json, member_ids_json, vector_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        item.title,
                        item.summary,
                        _json_dumps(item.keywords),
                        _json_dumps(item.member_ids),
                        _vec_to_json(item.vector),
                    ),
                )
                self.conn.execute("DELETE FROM communities_fts WHERE id = ?", (item.id,))
                self.conn.execute(
                    "INSERT INTO communities_fts (id, title, summary, keywords) VALUES (?, ?, ?, ?)",
                    (item.id, item.title, item.summary, " ".join(item.keywords)),
                )

    def append_trace(self, trace_id: str, created_at: float, payload: dict[str, Any]) -> None:
        logger.info("追加追踪记录, trace_id=%s", trace_id)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO traces (id, session_id, query, latency_ms, created_at, trace_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    trace_id,
                    str(payload.get("sessionId", "") or payload.get("session_id", "") or ""),
                    str(payload.get("query", "") or ""),
                    int(payload.get("latencyMs", payload.get("latency_ms", 0)) or 0),
                    created_at,
                    _json_dumps(payload),
                ),
            )

    def upsert_slow_path_job(self, job_id: str, created_at: float, status: str, attempts: int, last_error: str, query: str, payload: dict[str, Any]) -> None:
        logger.info("写入慢路径任务, job_id=%s, status=%s, attempts=%d", job_id, status, attempts)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO slow_path_jobs (id, created_at, status, attempts, last_error, query, job_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, created_at, status, attempts, last_error, query, _json_dumps(payload)),
            )

    def list_transcripts(self) -> list[TranscriptEntry]:
        logger.info("查询所有对话记录")
        rows = self.conn.execute("SELECT * FROM transcript_entries ORDER BY session_id, message_index, created_at").fetchall()
        return [
            TranscriptEntry(
                id=row["id"],
                session_key=row["session_key"],
                date_time=row["date_time"],
                turn_index=row["message_index"] if row["message_index"] is not None else row["turn_index"],
                speaker=row["role"] or row["speaker"],
                text=row["text"],
                content=row["content"],
                tokens=_json_loads(row["tokens_json"], []),
                entity_ids=_json_loads(row["entity_ids_json"], []),
                evidence=_evidence_from_json(row["evidence_json"]) or EvidenceRef(
                    sessionFile=row["session_file"] or row["session_key"],
                    messageIndex=row["message_index"] if row["message_index"] is not None else row["turn_index"],
                    startLine=(row["message_index"] if row["message_index"] is not None else row["turn_index"]) + 1,
                    endLine=(row["message_index"] if row["message_index"] is not None else row["turn_index"]) + 1,
                    snippet=row["text"][:200],
                    dateTime=row["date_time"],
                    speaker=row["role"] or row["speaker"],
                ),
                session_id=row["session_id"] or row["session_key"],
                session_file=row["session_file"] or row["session_key"],
                created_at=row["created_at"] or 0,
                vector=_vec_from_json(row["vector_json"]) if "vector_json" in row.keys() else None,
            )
            for row in rows
        ]

    def lookup_transcript_entry(self, session_file: str, message_index: int) -> dict[str, str] | None:
        logger.debug("查找对话条目, session_file=%s, message_index=%d", session_file, message_index)
        row = self.conn.execute(
            """
            SELECT text, role, speaker FROM transcript_entries
            WHERE session_file = ? AND message_index = ?
            LIMIT 1
            """,
            (session_file, message_index),
        ).fetchone()
        if row is None:
            return None
        return {"text": row["text"], "role": row["role"] or row["speaker"]}

    def lookup_transcript_by_session_file(self, session_file: str) -> list[dict[str, object]]:
        logger.debug("按 session_file 查找对话记录, session_file=%s", session_file)
        rows = self.conn.execute(
            """
            SELECT message_index, text, role, speaker FROM transcript_entries
            WHERE session_file = ?
            ORDER BY message_index ASC
            """,
            (session_file,),
        ).fetchall()
        return [{"messageIndex": row["message_index"], "text": row["text"], "role": row["role"] or row["speaker"]} for row in rows]

    def list_transcript_evidence_entries(self, session_id: str, session_file: str) -> list[dict[str, object]]:
        logger.debug("查询对话证据条目, session_id=%s, session_file=%s", session_id, session_file)
        rows = self.conn.execute(
            """
            SELECT message_index, text, role, speaker, evidence_json
            FROM transcript_entries
            WHERE session_id = ?
              AND session_file = ?
              AND evidence_json IS NOT NULL
              AND evidence_json <> ''
              AND evidence_json <> 'null'
            ORDER BY message_index ASC, created_at ASC
            """,
            (session_id, session_file),
        ).fetchall()
        return [
            {
                "messageIndex": row["message_index"],
                "text": row["text"],
                "role": row["role"] or row["speaker"],
                "evidence": _evidence_from_json(row["evidence_json"]),
            }
            for row in rows
        ]

    def lookup_session_file(self, session_id: str) -> str | None:
        logger.debug("查找 session_file, session_id=%s", session_id)
        row = self.conn.execute(
            """
            SELECT session_file
            FROM transcript_entries
            WHERE session_file <> ''
              AND session_id = ?
            ORDER BY created_at DESC, message_index DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if not row or not row["session_file"]:
            return None
        return str(row["session_file"])

    def get_next_transcript_message_index(self, session_id: str) -> int:
        logger.debug("获取下一条消息索引, session_id=%s", session_id)
        row = self.conn.execute(
            """
            SELECT MAX(message_index) AS max_message_index
            FROM transcript_entries
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        max_message_index = row["max_message_index"] if row else -1
        try:
            return int(max_message_index if max_message_index is not None else -1) + 1
        except (TypeError, ValueError):
            return 0

    def list_entities(self) -> list[EntityNode]:
        logger.info("查询所有实体节点")
        rows = self.conn.execute("SELECT * FROM entities ORDER BY mention_count DESC, name ASC").fetchall()
        result: list[EntityNode] = []
        for row in rows:
            result.append(
                EntityNode(
                    id=row["id"],
                    name=row["name"],
                    tokens=_json_loads(row["tokens_json"], []),
                    description=row["description"],
                    snippets=_json_loads(row["snippets_json"], []),
                    session_keys=set(_json_loads(row["session_keys_json"], [])),
                    mention_count=row["mention_count"],
                    vector=_vec_from_json(row["vector_json"]),
                )
            )
        return result

    def list_events(self) -> list[EventNode]:
        logger.info("查询所有事件节点")
        rows = self.conn.execute("SELECT * FROM events ORDER BY session_key, turn_index").fetchall()
        return [
            EventNode(
                id=row["id"],
                session_key=row["session_key"],
                date_time=row["date_time"],
                turn_index=row["turn_index"],
                speaker=row["speaker"],
                text=row["text"],
                content=row["content"],
                tokens=_json_loads(row["tokens_json"], []),
                entity_ids=_json_loads(row["entity_ids_json"], []),
                evidence=_evidence_from_json(row["evidence_json"]),
                vector=_vec_from_json(row["vector_json"]),
            )
            for row in rows
        ]

    def list_facts(self) -> list[LedgerFact]:
        logger.info("查询所有事实记录")
        rows = self.conn.execute("SELECT * FROM facts ORDER BY confidence DESC, session_key, turn_index").fetchall()
        return [
            LedgerFact(
                id=row["id"],
                subject=row["subject"],
                key=row["key"],
                scope=row["scope"],
                value=row["value"],
                text=row["text"],
                session_key=row["session_key"],
                turn_index=row["turn_index"],
                tokens=_json_loads(row["tokens_json"], []),
                evidence=_evidence_from_json(row["evidence_json"]),
                subject_entity_id=row["subject_entity_id"],
                confidence=row["confidence"],
                validFrom=row["valid_from"],
                validTo=row["valid_to"],
                invalidAt=row["invalid_at"],
                expiresAt=row["expires_at"],
                source=row["source"],
                status=row["status"],
                vector=_vec_from_json(row["vector_json"]),
            )
            for row in rows
        ]

    def list_session_summaries(self) -> list[SessionSummary]:
        logger.info("查询所有会话摘要")
        rows = self.conn.execute("SELECT * FROM session_summaries ORDER BY created_at, session_key").fetchall()
        return [
            SessionSummary(
                session_key=row["session_key"],
                date_time=row["date_time"],
                abstract=row["abstract"],
                overview=row["overview"],
                tokens=_json_loads(row["tokens_json"], []),
                source_event_ids=_json_loads(row["source_event_ids_json"], []),
                vector=_vec_from_json(row["vector_json"]),
                id=row["id"] or row["session_key"],
                session_id=row["session_id"] or row["session_key"],
                session_file=row["session_file"] or row["session_key"],
                message_count=row["message_count"] or 0,
                created_at=row["created_at"] or 0,
            )
            for row in rows
        ]

    def upsert_session_summary(self, item: SessionSummary) -> None:
        logger.debug("写入单条会话摘要, session_key=%s", item.session_key)
        self.upsert_session_summaries([item])

    def list_session_summaries_by_session(self, session_id: str, limit: int) -> list[SessionSummary]:
        logger.debug("按会话查询摘要, session_id=%s, limit=%d", session_id, limit)
        rows = self.conn.execute(
            """
            SELECT * FROM session_summaries
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        return [
            SessionSummary(
                session_key=row["session_key"],
                date_time=row["date_time"],
                abstract=row["abstract"],
                overview=row["overview"],
                tokens=_json_loads(row["tokens_json"], []),
                source_event_ids=_json_loads(row["source_event_ids_json"], []),
                vector=_vec_from_json(row["vector_json"]),
                id=row["id"] or row["session_key"],
                session_id=row["session_id"] or row["session_key"],
                session_file=row["session_file"] or row["session_key"],
                message_count=row["message_count"] or 0,
                created_at=row["created_at"] or 0,
            )
            for row in rows
        ]

    def search_session_summaries(self, query: str, limit: int) -> list[SessionSummary]:
        logger.info("全文搜索会话摘要, query=%s, limit=%d", query, limit)
        rows = self.conn.execute(
            """
            SELECT s.* FROM session_summaries_fts f
            JOIN session_summaries s ON s.session_key = f.session_key
            WHERE session_summaries_fts MATCH ?
            LIMIT ?
            """,
            (_build_fts_query(query.split()), limit),
        ).fetchall()
        return [
            SessionSummary(
                session_key=row["session_key"],
                date_time=row["date_time"],
                abstract=row["abstract"],
                overview=row["overview"],
                tokens=_json_loads(row["tokens_json"], []),
                source_event_ids=_json_loads(row["source_event_ids_json"], []),
                vector=_vec_from_json(row["vector_json"]),
                id=row["id"] or row["session_key"],
                session_id=row["session_id"] or row["session_key"],
                session_file=row["session_file"] or row["session_key"],
                message_count=row["message_count"] or 0,
                created_at=row["created_at"] or 0,
            )
            for row in rows
        ]

    def list_edges(self) -> list[GraphEdgeRecord]:
        logger.info("查询所有图边")
        rows = self.conn.execute("SELECT * FROM graph_edges").fetchall()
        return [
            GraphEdgeRecord(
                id=row["id"],
                from_id=row["from_id"],
                to_id=row["to_id"],
                edge_type=row["edge_type"],
                weight=row["weight"],
                relation_label=row["relation_label"],
                evidence=_evidence_from_json(row["evidence_json"]),
            )
            for row in rows
        ]

    def get_episodic_messages(self, session_file: str, message_index: int, window_size: int) -> list[dict[str, object]]:
        logger.debug("获取情景消息, session_file=%s, message_index=%d, window_size=%d", session_file, message_index, window_size)
        rows = self.conn.execute(
            """
            SELECT role, speaker, text, message_index FROM transcript_entries
            WHERE session_file = ?
              AND message_index BETWEEN ? AND ?
            ORDER BY message_index ASC
            """,
            (session_file, max(0, message_index - window_size), message_index + window_size),
        ).fetchall()
        return [{"role": row["role"] or row["speaker"], "text": row["text"], "messageIndex": row["message_index"]} for row in rows]

    def list_communities(self) -> list[CommunitySummaryRecord]:
        logger.info("查询所有社区摘要")
        rows = self.conn.execute("SELECT * FROM communities ORDER BY id").fetchall()
        return [
            CommunitySummaryRecord(
                id=row["id"],
                title=row["title"],
                summary=row["summary"],
                keywords=_json_loads(row["keywords_json"], []),
                member_ids=_json_loads(row["member_ids_json"], []),
                vector=_vec_from_json(row["vector_json"]),
            )
            for row in rows
        ]

    def list_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        logger.debug("查询追踪记录, limit=%d", limit)
        rows = self.conn.execute("SELECT trace_json FROM traces ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_json_loads(row["trace_json"], {}) for row in rows]

    def list_recent_traces(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        logger.debug("查询最近追踪记录, session_id=%s, limit=%d", session_id, limit)
        rows = self.conn.execute(
            "SELECT trace_json FROM traces WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [_json_loads(row["trace_json"], {}) for row in rows]

    def list_slow_path_jobs(self, limit: int = 200) -> list[dict[str, Any]]:
        logger.debug("查询慢路径任务列表, limit=%d", limit)
        rows = self.conn.execute("SELECT job_json FROM slow_path_jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_json_loads(row["job_json"], {}) for row in rows]

    def get_slow_path_job(self, job_id: str) -> dict[str, Any] | None:
        logger.debug("获取慢路径任务, job_id=%s", job_id)
        row = self.conn.execute("SELECT job_json FROM slow_path_jobs WHERE id = ? LIMIT 1", (job_id,)).fetchone()
        return _json_loads(row["job_json"], {}) if row else None

    def list_resumable_slow_path_jobs(self) -> list[dict[str, Any]]:
        logger.debug("查询可恢复的慢路径任务")
        rows = self.conn.execute(
            "SELECT job_json FROM slow_path_jobs WHERE status IN ('pending', 'running') ORDER BY created_at ASC"
        ).fetchall()
        return [_json_loads(row["job_json"], {}) for row in rows]

    def list_slow_path_jobs_by_status(self, statuses: list[str]) -> list[dict[str, Any]]:
        logger.debug("按状态查询慢路径任务, statuses=%s", statuses)
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        rows = self.conn.execute(
            f"SELECT job_json FROM slow_path_jobs WHERE status IN ({placeholders}) ORDER BY created_at DESC",
            tuple(statuses),
        ).fetchall()
        return [_json_loads(row["job_json"], {}) for row in rows]

    def count_slow_path_jobs_by_status(self) -> dict[str, int]:
        logger.debug("统计慢路径任务状态分布")
        result = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS count FROM slow_path_jobs GROUP BY status"
        ).fetchall()
        for row in rows:
            status = row["status"]
            count = row["count"]
            if status == "pending":
                result["pending"] += count
            elif status == "running":
                result["running"] += count
            elif status == "completed":
                result["done"] += count
            elif status == "failed":
                result["failed"] += count
        return result

    def evict_pinned_entries(self, session_id: str, entry_ids: list[str]) -> None:
        logger.info("驱逐置顶条目, session_id=%s, 数量: %d", session_id, len(entry_ids))
        if not entry_ids:
            return
        placeholders = ",".join("?" for _ in entry_ids)
        with self.conn:
            self.conn.execute(
                f"DELETE FROM pinned_entries WHERE session_id = ? AND id IN ({placeholders})",
                (session_id, *entry_ids),
            )

    def write_trace(self, trace: dict[str, Any]) -> None:
        logger.debug("写入追踪记录")
        trace_id = str(trace.get("traceId", "") or trace.get("trace_id", "") or "")
        created_at = float(trace.get("createdAt", trace.get("created_at", time.time())) or time.time())
        self.append_trace(trace_id, created_at, trace)

    def list_distinct_active_fact_subjects(self, limit: int = 200) -> list[str]:
        logger.debug("查询活跃事实主题, limit=%d", limit)
        rows = self.conn.execute(
            """
            SELECT subject
            FROM facts
            WHERE status = 'active'
              AND invalid_at IS NULL
            GROUP BY subject
            ORDER BY COUNT(*) DESC, subject ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [str(row["subject"]) for row in rows]

    def search_event_ids(self, tokens: list[str], limit: int) -> list[str]:
        logger.debug("全文搜索事件ID, tokens=%s, limit=%d", tokens, limit)
        rows = self.conn.execute(
            "SELECT id FROM events_fts WHERE events_fts MATCH ? LIMIT ?",
            (_build_fts_query(tokens), limit),
        ).fetchall()
        return [row["id"] for row in rows]

    def search_fact_ids(self, tokens: list[str], limit: int) -> list[str]:
        logger.debug("全文搜索事实ID, tokens=%s, limit=%d", tokens, limit)
        rows = self.conn.execute(
            "SELECT id FROM facts_fts WHERE facts_fts MATCH ? LIMIT ?",
            (_build_fts_query(tokens), limit),
        ).fetchall()
        return [row["id"] for row in rows]

    def search_summary_keys(self, tokens: list[str], limit: int) -> list[str]:
        logger.debug("全文搜索摘要键, tokens=%s, limit=%d", tokens, limit)
        rows = self.conn.execute(
            "SELECT session_key FROM session_summaries_fts WHERE session_summaries_fts MATCH ? LIMIT ?",
            (_build_fts_query(tokens), limit),
        ).fetchall()
        return [row["session_key"] for row in rows]

    def search_community_ids(self, tokens: list[str], limit: int) -> list[str]:
        logger.debug("全文搜索社区ID, tokens=%s, limit=%d", tokens, limit)
        rows = self.conn.execute(
            "SELECT id FROM communities_fts WHERE communities_fts MATCH ? LIMIT ?",
            (_build_fts_query(tokens), limit),
        ).fetchall()
        return [row["id"] for row in rows]

    # ── HyperMem CRUD ──────────────────────────────────────────────

    def upsert_hm_topics(self, items: list[HmTopic]) -> None:
        logger.info("写入 HyperMem 主题, 数量: %d", len(items))
        with self.conn:
            for item in items:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO hm_topics
                    (id, title, summary, keywords_json, episode_ids_json, vector_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        item.title,
                        item.summary,
                        _json_dumps(item.keywords),
                        _json_dumps(item.episode_ids),
                        _vec_to_json(item.vector),
                        item.created_at,
                        item.updated_at or int(time.time() * 1000),
                    ),
                )
                self.conn.execute("DELETE FROM hm_topics_fts WHERE id = ?", (item.id,))
                self.conn.execute(
                    "INSERT INTO hm_topics_fts (id, title, summary, keywords) VALUES (?, ?, ?, ?)",
                    (item.id, item.title, item.summary, " ".join(item.keywords)),
                )

    def list_hm_topics(self) -> list[HmTopic]:
        logger.debug("查询所有 HyperMem 主题")
        rows = self.conn.execute("SELECT * FROM hm_topics ORDER BY updated_at DESC, created_at DESC").fetchall()
        return [
            HmTopic(
                id=row["id"],
                title=row["title"],
                summary=row["summary"],
                keywords=_json_loads(row["keywords_json"], []),
                episode_ids=_json_loads(row["episode_ids_json"], []),
                vector=_vec_from_json(row["vector_json"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def search_hm_topic_ids(self, tokens: list[str], limit: int) -> list[str]:
        logger.debug("全文搜索 HyperMem 主题ID, tokens=%s, limit=%d", tokens, limit)
        rows = self.conn.execute(
            "SELECT id FROM hm_topics_fts WHERE hm_topics_fts MATCH ? LIMIT ?",
            (_build_fts_query(tokens), limit),
        ).fetchall()
        return [row["id"] for row in rows]

    def upsert_hm_episodes(self, items: list[HmEpisode]) -> None:
        logger.info("写入 HyperMem 情节, 数量: %d", len(items))
        with self.conn:
            for item in items:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO hm_episodes
                    (id, session_key, title, summary, dialogue, keywords_json, timestamp_start, timestamp_end,
                     turn_start, turn_end, topic_ids_json, fact_ids_json, vector_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        item.session_key,
                        item.title,
                        item.summary,
                        item.dialogue,
                        _json_dumps(item.keywords),
                        item.timestamp_start,
                        item.timestamp_end,
                        item.turn_start,
                        item.turn_end,
                        _json_dumps(item.topic_ids),
                        _json_dumps(item.fact_ids),
                        _vec_to_json(item.vector),
                        item.created_at or int(time.time() * 1000),
                    ),
                )
                self.conn.execute("DELETE FROM hm_episodes_fts WHERE id = ?", (item.id,))
                self.conn.execute(
                    "INSERT INTO hm_episodes_fts (id, title, summary, dialogue, keywords) VALUES (?, ?, ?, ?, ?)",
                    (item.id, item.title, item.summary, item.dialogue, " ".join(item.keywords)),
                )

    def list_hm_episodes(self) -> list[HmEpisode]:
        logger.debug("查询所有 HyperMem 情节")
        rows = self.conn.execute("SELECT * FROM hm_episodes ORDER BY session_key, turn_start").fetchall()
        return [
            HmEpisode(
                id=row["id"],
                session_key=row["session_key"],
                title=row["title"],
                summary=row["summary"],
                dialogue=row["dialogue"],
                keywords=_json_loads(row["keywords_json"], []),
                timestamp_start=row["timestamp_start"],
                timestamp_end=row["timestamp_end"],
                turn_start=row["turn_start"],
                turn_end=row["turn_end"],
                topic_ids=_json_loads(row["topic_ids_json"], []),
                fact_ids=_json_loads(row["fact_ids_json"], []),
                vector=_vec_from_json(row["vector_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def list_hm_episodes_by_ids(self, ids: list[str]) -> list[HmEpisode]:
        logger.debug("按ID查询 HyperMem 情节, 数量: %d", len(ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT * FROM hm_episodes WHERE id IN ({placeholders}) ORDER BY session_key, turn_start",
            tuple(ids),
        ).fetchall()
        return [
            HmEpisode(
                id=row["id"],
                session_key=row["session_key"],
                title=row["title"],
                summary=row["summary"],
                dialogue=row["dialogue"],
                keywords=_json_loads(row["keywords_json"], []),
                timestamp_start=row["timestamp_start"],
                timestamp_end=row["timestamp_end"],
                turn_start=row["turn_start"],
                turn_end=row["turn_end"],
                topic_ids=_json_loads(row["topic_ids_json"], []),
                fact_ids=_json_loads(row["fact_ids_json"], []),
                vector=_vec_from_json(row["vector_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def search_hm_episode_ids(self, tokens: list[str], limit: int) -> list[str]:
        logger.debug("全文搜索 HyperMem 情节ID, tokens=%s, limit=%d", tokens, limit)
        rows = self.conn.execute(
            "SELECT id FROM hm_episodes_fts WHERE hm_episodes_fts MATCH ? LIMIT ?",
            (_build_fts_query(tokens), limit),
        ).fetchall()
        return [row["id"] for row in rows]

    def upsert_hm_facts(self, items: list[HmFact]) -> None:
        logger.info("写入 HyperMem 事实, 数量: %d", len(items))
        with self.conn:
            for item in items:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO hm_facts
                    (id, content, potential, keywords_json, importance, episode_id, session_key,
                     source_turn_start, source_turn_end, vector_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        item.content,
                        item.potential,
                        _json_dumps(item.keywords),
                        item.importance,
                        item.episode_id,
                        item.session_key,
                        item.source_turn_start,
                        item.source_turn_end,
                        _vec_to_json(item.vector),
                        item.created_at or int(time.time() * 1000),
                    ),
                )
                self.conn.execute("DELETE FROM hm_facts_fts WHERE id = ?", (item.id,))
                self.conn.execute(
                    "INSERT INTO hm_facts_fts (id, content, potential, keywords) VALUES (?, ?, ?, ?)",
                    (item.id, item.content, item.potential, " ".join(item.keywords)),
                )

    def list_hm_facts(self) -> list[HmFact]:
        logger.debug("查询所有 HyperMem 事实")
        rows = self.conn.execute("SELECT * FROM hm_facts ORDER BY created_at DESC").fetchall()
        return [
            HmFact(
                id=row["id"],
                content=row["content"],
                potential=row["potential"],
                keywords=_json_loads(row["keywords_json"], []),
                importance=row["importance"],
                episode_id=row["episode_id"],
                session_key=row["session_key"],
                source_turn_start=row["source_turn_start"],
                source_turn_end=row["source_turn_end"],
                vector=_vec_from_json(row["vector_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def list_hm_facts_by_ids(self, ids: list[str]) -> list[HmFact]:
        logger.debug("按ID查询 HyperMem 事实, 数量: %d", len(ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.conn.execute(
            f"SELECT * FROM hm_facts WHERE id IN ({placeholders})",
            tuple(ids),
        ).fetchall()
        return [
            HmFact(
                id=row["id"],
                content=row["content"],
                potential=row["potential"],
                keywords=_json_loads(row["keywords_json"], []),
                importance=row["importance"],
                episode_id=row["episode_id"],
                session_key=row["session_key"],
                source_turn_start=row["source_turn_start"],
                source_turn_end=row["source_turn_end"],
                vector=_vec_from_json(row["vector_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def list_hm_facts_by_episode_ids(self, episode_ids: list[str]) -> list[HmFact]:
        logger.debug("按情节ID查询 HyperMem 事实, episode_ids 数量: %d", len(episode_ids))
        if not episode_ids:
            return []
        placeholders = ",".join("?" for _ in episode_ids)
        rows = self.conn.execute(
            f"SELECT * FROM hm_facts WHERE episode_id IN ({placeholders}) ORDER BY created_at",
            tuple(episode_ids),
        ).fetchall()
        return [
            HmFact(
                id=row["id"],
                content=row["content"],
                potential=row["potential"],
                keywords=_json_loads(row["keywords_json"], []),
                importance=row["importance"],
                episode_id=row["episode_id"],
                session_key=row["session_key"],
                source_turn_start=row["source_turn_start"],
                source_turn_end=row["source_turn_end"],
                vector=_vec_from_json(row["vector_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def search_hm_fact_ids(self, tokens: list[str], limit: int) -> list[str]:
        logger.debug("全文搜索 HyperMem 事实ID, tokens=%s, limit=%d", tokens, limit)
        rows = self.conn.execute(
            "SELECT id FROM hm_facts_fts WHERE hm_facts_fts MATCH ? LIMIT ?",
            (_build_fts_query(tokens), limit),
        ).fetchall()
        return [row["id"] for row in rows]

    # ── UnifiedFact persistence ──

    def upsert_unified_facts(self, items: list[UnifiedFact]) -> None:
        logger.info("写入统一事实, 数量: %d", len(items))
        with self.conn:
            for item in items:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO unified_facts
                    (id, content, subject, key, scope, potential, keywords_json, importance,
                     confidence, evidence_json, episode_id, entity_ids_json, session_key,
                     turn_index, source_turn_start, source_turn_end, valid_from, valid_to,
                     invalid_at, expires_at, source, status, vector_json, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item.id, item.content, item.subject, item.key, item.scope,
                        item.potential, _json_dumps(item.keywords), item.importance,
                        item.confidence, _json_dumps(vars(item.evidence)) if item.evidence else None,
                        item.episode_id, _json_dumps(item.entity_ids), item.session_key,
                        item.turn_index, item.source_turn_start, item.source_turn_end,
                        item.validFrom, item.validTo, item.invalidAt, item.expiresAt,
                        item.source, item.status, _vec_to_json(item.vector), item.created_at,
                    ),
                )
                self.conn.execute("DELETE FROM unified_facts_fts WHERE id = ?", (item.id,))
                self.conn.execute(
                    "INSERT INTO unified_facts_fts (id, content, potential, keywords) VALUES (?, ?, ?, ?)",
                    (item.id, item.content, item.potential, " ".join(item.keywords)),
                )

    def list_unified_facts(self) -> list[UnifiedFact]:
        logger.debug("查询所有统一事实")
        rows = self.conn.execute("SELECT * FROM unified_facts ORDER BY created_at DESC").fetchall()
        return [self._row_to_unified_fact(row) for row in rows]

    def _row_to_unified_fact(self, row) -> UnifiedFact:
        return UnifiedFact(
            id=row["id"],
            content=row["content"],
            subject=row["subject"],
            key=row["key"],
            scope=row["scope"],
            potential=row["potential"],
            keywords=_json_loads(row["keywords_json"], []),
            importance=row["importance"],
            confidence=row["confidence"],
            evidence=_evidence_from_json(row["evidence_json"]),
            episode_id=row["episode_id"],
            entity_ids=_json_loads(row["entity_ids_json"], []),
            session_key=row["session_key"],
            turn_index=row["turn_index"],
            source_turn_start=row["source_turn_start"],
            source_turn_end=row["source_turn_end"],
            validFrom=row["valid_from"],
            validTo=row["valid_to"],
            invalidAt=row["invalid_at"],
            expiresAt=row["expires_at"],
            source=row["source"],
            status=row["status"],
            vector=_vec_from_json(row["vector_json"]),
            created_at=row["created_at"],
        )

    # TS-style aliases
    upsertPinnedEntries = upsert_pinned_entries
    listPinnedEntries = list_pinned_entries
    replaceTopicEntries = replace_topic_entries
    listTopicEntries = list_topic_entries
    appendScratchpadEntries = append_scratchpad_entries
    listScratchpadEntries = list_scratchpad_entries
    clearWorkspaceState = clear_workspace_state
    upsertTranscriptEntries = upsert_transcripts
    upsertGraphNodes = upsert_entities
    upsertGraphEdges = upsert_edges
    insertFact = lambda self, fact: self.upsert_facts([fact])
    upsertSessionSummary = upsert_session_summary
    listSessionSummaries = list_session_summaries_by_session
    searchSessionSummaries = search_session_summaries
    lookupTranscriptEntry = lookup_transcript_entry
    lookupTranscriptBySessionFile = lookup_transcript_by_session_file
    listTranscriptEvidenceEntries = list_transcript_evidence_entries
    lookupSessionFile = lookup_session_file
    getNextTranscriptMessageIndex = get_next_transcript_message_index
    getEpisodicMessages = get_episodic_messages
    evictPinnedEntries = evict_pinned_entries
    writeTrace = write_trace
    listRecentTraces = list_recent_traces
    listDistinctActiveFactSubjects = list_distinct_active_fact_subjects
    getSlowPathJob = get_slow_path_job
    listResumableSlowPathJobs = list_resumable_slow_path_jobs
    listSlowPathJobsByStatus = list_slow_path_jobs_by_status
    countSlowPathJobsByStatus = count_slow_path_jobs_by_status


def _with_store_lock(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


_wrapped_store_callables: dict[int, Any] = {}

for _name, _value in list(EbmStore.__dict__.items()):
    if _name.startswith("_"):
        continue
    if not callable(_value):
        continue
    wrapped = _wrapped_store_callables.get(id(_value))
    if wrapped is None:
        wrapped = _with_store_lock(_value)
        _wrapped_store_callables[id(_value)] = wrapped
    setattr(EbmStore, _name, wrapped)
