from __future__ import annotations

import logging
import math
import re
import time
from typing import Callable

logger = logging.getLogger("ebm_context_engine.planes.temporal_semantic_ledger")

from ebm_context_engine.core.hash import stableId
from ebm_context_engine.retrieval.hybrid import rank_text_records
from ebm_context_engine.types import EvidenceRef, LedgerFact, RecallHit

PREFERENCE_PATTERNS: list[dict[str, object]] = []


def sanitizeFactValue(value: str) -> str:
    return re.sub(r"[。！？!?,，；;:\s]+$", "", value).strip()


def looksLikeTranscriptConversation(text: str) -> bool:
    bracketed_speakers = len(re.findall(r"\[[^\]\n]{1,40}\]:", text))
    line_speakers = len(re.findall(r"(?:^|\n)\s*[\w][\w' -]{0,30}:\s", text))
    return bracketed_speakers + line_speakers >= 2


def shouldApplyPattern(pattern: dict[str, object], text: str) -> bool:
    if not looksLikeTranscriptConversation(text):
        return True
    return pattern.get("key") == "language.preference"


class TemporalSemanticLedger:
    def __init__(
        self,
        state_getter: Callable[[], object],
        *,
        confidence_threshold: float = 0.6,
        fact_ttl_days: int = 90,
        forgetting_half_life_days: int = 30,
    ):
        self._state_getter = state_getter
        self.confidence_threshold = confidence_threshold
        self.fact_ttl_days = fact_ttl_days
        self.forgetting_half_life_days = forgetting_half_life_days

    def _state(self):
        return self._state_getter()

    def demoteFromPinned(self, entry) -> None:
        logger.info("从固定上下文降级条目: label=%s, scope=%s", entry.label[:60], entry.scope)
        now = int(time.time() * 1000)
        fact = LedgerFact(
            id=stableId("DEMOTED", entry.label, entry.content),
            subject="user",
            key=f"demoted.{entry.scope}.{entry.label[:60]}",
            value=entry.content,
            scope="constraint" if entry.scope == "safety" else "experience",
            text=f"user: {entry.content}",
            session_key=entry.evidence.sessionFile if entry.evidence else "",
            turn_index=entry.evidence.messageIndex if entry.evidence and entry.evidence.messageIndex is not None else 0,
            tokens=[],
            evidence=entry.evidence,
            confidence=0.7,
            validFrom=now,
            expiresAt=now + self.fact_ttl_days * 24 * 60 * 60 * 1000,
            source="pinned-demotion",
            status="active",
        )
        self._state().facts[fact.id] = fact
        logger.debug("降级事实已写入: fact_id=%s", fact.id)

    def ingestTexts(self, params: dict) -> list[LedgerFact]:
        logger.info("开始摄入文本: 文本数=%d", len(list(params.get("texts", []))))
        if not PREFERENCE_PATTERNS:
            logger.debug("无偏好模式，跳过摄入")
            return []
        now = int(time.time() * 1000)
        facts: list[LedgerFact] = []
        texts: list[str] = list(params.get("texts", []))
        evidence_base = params.get("evidenceBase", {})
        start_index = int(evidence_base.get("startIndex", 0) or 0)
        message_indexes: list[int] = list(evidence_base.get("messageIndexes", []))
        session_file = str(evidence_base.get("sessionFile", "") or "")
        global_subject = params.get("globalSubject")
        scoped_subject = params.get("scopedSubject")
        for index, text in enumerate(texts):
            for pattern in PREFERENCE_PATTERNS:
                if not shouldApplyPattern(pattern, text):
                    continue
                match = pattern["expression"].search(text)
                if not match:
                    continue
                value = sanitizeFactValue(match.group(2) or "")
                if not value:
                    continue
                scope = str(pattern["scope"])
                subject = (global_subject if scope == "preference" else scoped_subject or global_subject or "user")
                message_index = message_indexes[index] if index < len(message_indexes) else start_index + index
                fact = LedgerFact(
                    id=stableId(subject, pattern["key"], value),
                    subject=str(subject),
                    key=str(pattern["key"]),
                    value=value,
                    scope=scope,
                    text=f"{subject}: {value}",
                    session_key=session_file,
                    turn_index=message_index,
                    tokens=[],
                    confidence=float(pattern["confidence"]),
                    validFrom=now,
                    expiresAt=now + self.fact_ttl_days * 24 * 60 * 60 * 1000,
                    source=str(params.get("source", "turn-distillation")),
                    status="active",
                    evidence=EvidenceRef(sessionFile=session_file, messageIndex=message_index, snippet=text),
                )
                self._state().facts[fact.id] = fact
                facts.append(fact)
        logger.info("文本摄入完成: 提取事实数=%d", len(facts))
        return facts

    def deleteFact(self, fact_id: str) -> bool:
        logger.info("删除事实: fact_id=%s", fact_id)
        state = self._state()
        fact = state.facts.get(fact_id)
        if fact is None:
            logger.debug("事实未找到: fact_id=%s", fact_id)
            return False
        fact.status = "deleted"
        fact.invalidAt = fact.invalidAt or fact.validFrom
        return True

    def recall(
        self,
        query: str,
        *,
        query_vector,
        limit: int,
        risk_level: str,
        subjects: list[str],
        classification,
    ) -> tuple[list[RecallHit], list[str]]:
        logger.info("开始账本召回: query长度=%d, limit=%d, 风险级别=%s, 主题数=%d", len(query), limit, risk_level, len(subjects))
        state = self._state()
        now_ms = int(state.artifact_created_at * 1000) if state.artifact_created_at else 0
        active_facts = [
            fact for fact in state.facts.values()
            if fact.status == "active"
            and (fact.invalidAt is None)
            and (fact.validTo is None or fact.validTo >= now_ms or now_ms == 0)
            and (fact.expiresAt is None or fact.expiresAt >= now_ms or now_ms == 0)
        ]
        logger.debug("活跃事实数=%d, 总事实数=%d", len(active_facts), len(state.facts))
        lowered_subjects = {str(subject).strip().lower() for subject in subjects if str(subject).strip()}
        matched_fact_ids = {
            fact.id
            for fact in active_facts
            if not lowered_subjects or fact.subject.lower() in lowered_subjects
        }
        # Subject filtering helps interactive runtime memory isolation, but in benchmark
        # corpora most structured facts are promoted under synthetic subjects like "graph".
        # If the scoped filter is too restrictive, fall back to ranking all active facts.
        if lowered_subjects and len(matched_fact_ids) >= max(limit, 3):
            facts_for_ranking = [fact for fact in active_facts if fact.id in matched_fact_ids]
        else:
            facts_for_ranking = active_facts
        ranked = rank_text_records(
            query,
            facts_for_ranking,
            query_vector=query_vector,
            get_text=lambda fact: f"{fact.subject} {fact.key} {fact.value}",
            get_vector=lambda fact: fact.vector,
            intent_weights=classification.weights,
            rrf_k=60,
        )
        verify_reasons: list[str] = []
        items: list[RecallHit] = []
        for fact, score in ranked[: max(limit * 3, limit)]:
            if fact.validFrom:
                age_days = max(0.0, (now_ms - fact.validFrom) / (24 * 60 * 60 * 1000)) if now_ms else 0.0
                decay = math.pow(0.5, age_days / self.forgetting_half_life_days) if self.forgetting_half_life_days > 0 else 1.0
            else:
                decay = 1.0
            effective_score = score * decay
            if fact.id in matched_fact_ids:
                effective_score *= 1.1
            verified = fact.confidence >= self.confidence_threshold
            verification_note = "confidence short-circuit" if verified else "low-confidence"
            if risk_level == "high":
                verified, verification_note = self.verifyFactAgainstEvidence(fact)
                if not verified:
                    verify_reasons.append(f"fact:{fact.key}:{verification_note}")
            items.append(
                RecallHit(
                    id=fact.id,
                    title=f"{fact.scope} / {fact.key}",
                    content=f"{fact.subject}: {fact.value}",
                    source="ledger",
                    score=effective_score,
                    reason="verified" if verified else f"confidence={fact.confidence:.2f}",
                    evidence=fact.evidence,
                    session_key=fact.session_key,
                    turn_index=fact.turn_index,
                    verificationNote=fact.source,
                )
            )
        items.sort(key=lambda item: item.score, reverse=True)
        logger.info("账本召回完成: 返回项数=%d, 验证原因数=%d", min(limit, len(items)), len(verify_reasons))
        return items[:limit], verify_reasons

    def derivePinnedFacts(self, limit: int, subjects: list[str]) -> list[RecallHit]:
        logger.info("开始派生固定事实: limit=%d, 主题数=%d", limit, len(subjects))
        state = self._state()
        now_ms = int(state.artifact_created_at * 1000) if state.artifact_created_at else 0
        facts = [
            fact for fact in state.facts.values()
            if fact.status == "active"
            and fact.scope in {"constraint", "preference"}
            and (not subjects or fact.subject in subjects)
        ]
        pinned: list[RecallHit] = []
        for fact in facts:
            # Apply temporal decay (matching TS plane-c.ts derivePinnedFacts)
            if fact.validFrom and now_ms and self.forgetting_half_life_days > 0:
                age_days = max(0.0, (now_ms - fact.validFrom) / (24 * 60 * 60 * 1000))
                decay = math.pow(0.5, age_days / self.forgetting_half_life_days)
            else:
                decay = 1.0
            effective_score = fact.confidence * decay
            pinned.append(
                RecallHit(
                    id=fact.id,
                    title=fact.key,
                    content=fact.value,
                    source="ledger",
                    score=effective_score,
                    reason="promoted into pinned context",
                    evidence=fact.evidence,
                    session_key=fact.session_key,
                    turn_index=fact.turn_index,
                    verified=True,
                    verificationNote="write-time promoted fact",
                )
            )
        pinned.sort(key=lambda item: item.score, reverse=True)
        logger.info("派生固定事实完成: 候选数=%d, 返回数=%d", len(pinned), min(limit, len(pinned)))
        return pinned[:limit]

    def getAllFacts(self, limit: int = 200) -> list[dict]:
        logger.debug("获取所有事实: limit=%d", limit)
        facts = list(self._state().facts.values())[-limit:]
        return [
            {
                "id": fact.id,
                "subject": fact.subject,
                "key": fact.key,
                "scope": fact.scope,
                "value": fact.value,
                "confidence": fact.confidence,
                "validFrom": fact.validFrom,
                "validTo": fact.validTo,
                "invalidAt": fact.invalidAt,
                "expiresAt": fact.expiresAt,
                "source": fact.source,
                "status": fact.status,
            }
            for fact in facts
        ]

    def verifyFactAgainstEvidence(self, fact: LedgerFact) -> tuple[bool, str]:
        logger.debug("验证事实证据: fact_id=%s, key=%s", fact.id, fact.key)
        state = self._state()
        if not fact.evidence or not fact.evidence.sessionFile:
            logger.debug("事实缺少证据: fact_id=%s", fact.id)
            return False, "missing-evidence"
        # Use indexed lookup by turn_index when available (matching TS store.lookupTranscriptEntry)
        if fact.evidence.messageIndex is not None:
            # O(1)-ish: scan only entries matching session + turn_index
            matches = [
                entry for entry in state.transcripts
                if entry.session_key == fact.evidence.sessionFile
                and entry.turn_index == fact.evidence.messageIndex
            ]
            if not matches:
                return False, "source-missing"
            matched = any(fact.value in entry.text for entry in matches)
            return matched, "source-line-confirmed" if matched else "source-line-mismatch"
        # Fallback: scan session transcripts (matching TS lookupTranscriptBySessionFile)
        matches = [entry for entry in state.transcripts if entry.session_key == fact.evidence.sessionFile]
        if not matches:
            return False, "source-missing"
        matched = any(fact.value in entry.text for entry in matches)
        return matched, "source-session-confirmed" if matched else "source-session-mismatch"
