from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable

from ebm_context_engine.core.hash import stableId
from ebm_context_engine.text import top_keywords
from ebm_context_engine.types import EntityNode, EventNode, EvidenceRef, GraphEdgeRecord, LedgerFact, SessionSummary, TranscriptEntry

logger = logging.getLogger("ebm_context_engine.slowpath.llm_extractor")

InferenceFn = Callable[[str], str]
EmbedFn = Callable[[str], object]
EmbedTextsFn = Callable[[list[str]], list[object]]


def _batch_embed_nodes(nodes: list[dict[str, Any]], embed_texts_fn: EmbedTextsFn) -> None:
    """Fill node['vector'] in-place using a single batched embed call."""
    indices = [i for i, n in enumerate(nodes) if n.get("vector") is None and n.get("content")]
    if not indices:
        return
    texts = [nodes[i]["content"] for i in indices]
    vectors = embed_texts_fn(texts)
    for i, vec in zip(indices, vectors):
        nodes[i]["vector"] = vec

EXTRACTION_PROMPT = """You are a memory extraction engine. Extract ALL factual information from the following conversation. Focus on accurate, specific details.

RULES:
- NEVER use relative time expressions ("today", "recently", "last week"). Convert to absolute dates.
- If the conversation includes a date/time header, use it as the reference datetime for resolving relative time expressions.
- Extract specific facts, not vague summaries.
- Each fact must be atomic (one piece of information).
- Include dates, names, places, activities, preferences, relationships, status changes.
- Confidence: 0.95 for explicitly stated facts, 0.8 for strongly implied, 0.6 for inferred.
- CRITICAL: Preserve exact nouns, proper names, quantities, and specific details from the conversation. Do NOT generalize or paraphrase. For example:
  - GOOD: "researching adoption agencies" / "bought a blue Toyota Camry" / "enrolled in a Python bootcamp"
  - BAD: "doing research" / "bought a car" / "taking a course"
- The value field must contain enough detail that someone who never read the conversation can answer specific questions about it.

OUTPUT FORMAT (JSON array only, no markdown):
[
  {"category":"event","subject":"PersonName","key":"event.description","value":"attended team meeting","confidence":0.95,"date":"2024-03-15"}
]

CONVERSATION:
"""

SUMMARY_PROMPT = """Summarize this conversation session in two parts. Be specific, include names, dates, key facts.

If the conversation includes a date/time header, use it as the reference datetime for resolving relative time expressions (e.g. "yesterday" → absolute date).

PART 1 - ABSTRACT (one sentence, max 30 words): Ultra-compressed summary capturing the most important fact.
PART 2 - OVERVIEW (one paragraph, max 100 words): Key points including who, what, when, where.

OUTPUT FORMAT (JSON only, no markdown):
{"abstract":"One sentence summary here","overview":"Detailed paragraph summary here"}

CONVERSATION:
"""

HIGH_VALUE_SLOT_PROMPT = """You are a memory extraction engine. Extract only high-value, retrieval-grade facts from the conversation.

Focus on durable, query-worthy information such as:
- identity, self-description, or role
- relationships and social connections
- profession, career, or expertise
- origin, location, or relocation history
- interests, hobbies, or recurring activities
- named entities: people, places, organizations, projects
- explicit plans with time anchors
- preferences, opinions, or stated goals
- analysis frameworks, decision criteria, or evaluation methods the user explicitly states
- explicit instructions for future behavior ("以后请...", "之后你...", "please always...", "from now on...")

Use descriptive, stable keys that reflect the content (e.g. "profession", "hobby", "location.current", "project.name").
For analysis frameworks use keys like "analysis.default_framework.<domain>" (e.g. "analysis.default_framework.tech_stocks").
For behavioral instructions use keys like "instruction.<domain>" (e.g. "instruction.response_style").
If multiple distinct items are mentioned for the same category, emit one fact per value.
Infer conservatively from context when information is implied but not stated directly.
CRITICAL: Preserve exact nouns, proper names, quantities, and specific details. Do NOT generalize.
  - GOOD: "researching adoption agencies" / "signed up for salsa dancing classes"
  - BAD: "doing research" / "taking classes"

Do not extract generic chatter, filler, encouragement, or vague feelings.
If a time expression is relative, normalize it using any date/time header in the conversation.

OUTPUT FORMAT (JSON only, no markdown):
{
  "facts": [
    {"subject":"PersonA","key":"profession","value":"software engineer","confidence":0.95,"date":"2024-03-15"},
    {"subject":"PersonA","key":"project.current","value":"migration to microservices","confidence":0.90,"date":"2024-03-15"},
    {"subject":"PersonB","key":"hobby","value":"trail running","confidence":0.93,"date":"2024-03-15"},
    {"subject":"User","key":"analysis.default_framework.tech_stocks","value":"研发投入占比、用户增长率、现金流","confidence":0.95,"date":"2024-03-15"}
  ],
  "summary": {
    "abstract":"One sentence summary here",
    "overview":"Short paragraph summary here"
  }
}

CONVERSATION:
"""

PROFILE_FACT_PROMPT = """You are a memory extraction engine. Extract durable profile facts about the target person from the multi-session conversation archive.

Target person: {subject}

Focus on stable, high-value fields:
- identity, self-description, or role
- relationships and social connections
- location, origin, or relocation history
- profession, career, or expertise
- long-running goals or projects
- interests, hobbies, or recurring activities
- preferences, opinions, or values

Do not extract generic chat, encouragement, or one-off chatter.
If a date is relative, normalize it using any date/time header in the conversation.

Use descriptive, stable keys that reflect the content (e.g. "profession", "hobby", "location.current", "goal").
If multiple distinct items are mentioned for the same category, emit one fact per value.
Infer conservatively from context when information is implied but not stated directly.

OUTPUT FORMAT (JSON only, no markdown):
{{
  "facts": [
    {{"subject":"PersonA","key":"profession","value":"data scientist","confidence":0.95,"date":"2024-03-15"}},
    {{"subject":"PersonA","key":"location.current","value":"Berlin","confidence":0.9,"date":"2024-03-15"}},
    {{"subject":"PersonA","key":"hobby","value":"rock climbing","confidence":0.92,"date":"2024-03-15"}}
  ]
}}

ARCHIVE:
"""

ENTITY_EXTRACTION_PROMPT = """You are a knowledge graph extraction engine. Extract all ENTITIES, their ATTRIBUTES, RELATIONSHIPS, EVENTS, and CAUSAL LINKS from this conversation.

ENTITY: A named person, place, organization, group, pet, project, or significant concept.
ATTRIBUTE: A property of an entity (hobby, job, age, preference, status, trait, opinion).
RELATIONSHIP: A connection between two entities (friend, colleague, family, member of, attends).
EVENT: A dated occurrence involving one or more entities.
CAUSAL: A cause-effect link between two events or between an entity action and an outcome. Types: "causes", "prevents", "enables".

OUTPUT FORMAT (JSON only, no markdown):
{
  "entities": [{"name":"PersonA","category":"person","description":"Backend engineer on the payments team"}],
  "attributes": [{"entity":"PersonA","key":"expertise","value":"distributed systems","confidence":0.95,"date":"2024-03-15"}],
  "relationships": [{"from":"PersonA","to":"PersonB","type":"colleague","description":"work together on the payments service","confidence":0.9}],
  "events": [{"entities":["PersonA"],"description":"presented at the architecture review","date":"2024-03-15","confidence":0.95}],
  "causal": [{"source":"PersonA proposed caching layer","target":"API latency reduced by 40%","type":"enables","confidence":0.8}]
}

CONVERSATION:
"""


def _parse_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*\}", text.strip())
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    match = re.search(r"\[[\s\S]*\]", text.strip())
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _fact_category_to_scope(category: str) -> str:
    if category == "preference":
        return "preference"
    if category in {"entity", "relationship"}:
        return "environment"
    if category in {"event", "activity"}:
        return "experience"
    return "project"


def _graph_fact_scope(label: str, content: str) -> str:
    """Map graph FACT nodes to a scope category.

    Scope is derived from the attribute key supplied by the LLM extractor
    (e.g. "preference", "constraint").  Anything unrecognised falls back to
    "environment" to avoid language-specific keyword matching.
    """
    text = f"{label} {content}".lower()
    if "preference" in text or "constraint" in text:
        # Only match the exact categories that are in VALID_FACT_SCOPES
        if "preference" in text:
            return "preference"
        return "constraint"
    return "environment"


def extractFactsWithLlm(params: dict[str, Any]) -> list[LedgerFact]:
    logger.info("开始 LLM 事实提取: sessionFile=%s, 文本段数=%d", params.get("sessionFile", ""), len(params.get("texts", [])))
    inference_fn: InferenceFn = params["inferenceFn"]
    texts: list[str] = params["texts"]
    conversation_text = "\n\n".join(texts)
    prompt = EXTRACTION_PROMPT + conversation_text[: params.get("llmTruncationChars", 12000)]
    try:
        response = inference_fn(prompt)
        logger.debug("LLM 事实提取响应长度=%d", len(response) if response else 0)
    except Exception as exc:
        logger.warning("LLM 事实提取调用失败: %s", exc)
        return []
    extracted = _parse_json_array(response)
    if not extracted:
        logger.warning("LLM 事实提取响应解析失败")
        return []
    now = int(time.time() * 1000)
    facts: list[LedgerFact] = []
    for item in extracted:
        subject = str(item.get("subject", "") or "")
        key = str(item.get("key", "") or "")
        value = str(item.get("value", "") or "")
        if not subject or not key or not value:
            continue
        date = str(item.get("date", "") or "")
        value_with_date = f"{value} ({date})" if date else value
        facts.append(
            LedgerFact(
                id=stableId(subject, key, value),
                subject=subject,
                key=key,
                value=value_with_date,
                scope=_fact_category_to_scope(str(item.get("category", "status") or "status")),
                text=f"{subject}: {value_with_date}",
                session_key=params["sessionFile"],
                turn_index=int((params.get("messageIndexes") or [params["startIndex"]])[0]),
                tokens=[],
                evidence=EvidenceRef(
                    sessionFile=params["sessionFile"],
                    messageIndex=int((params.get("messageIndexes") or [params["startIndex"]])[0]),
                    snippet=conversation_text[:200],
                ),
                confidence=min(float(params.get("confidenceCeiling", 0.97)), float(item.get("confidence", 0.7) or 0.7)),
                validFrom=now,
                expiresAt=now + int(params["factTtlDays"]) * 24 * 60 * 60 * 1000,
                source="llm-extraction",
                status="active",
            )
        )
    logger.info("LLM 事实提取完成: 提取 %d 个事实", len(facts))
    return facts


def summarizeSession(params: dict[str, Any]) -> dict[str, str] | None:
    logger.info("开始 LLM 会话摘要: sessionFile=%s", params.get("sessionFile", ""))
    inference_fn: InferenceFn = params["inferenceFn"]
    texts: list[str] = params["texts"]
    prompt = SUMMARY_PROMPT + "\n\n".join(texts)[: params.get("llmTruncationChars", 12000)]
    try:
        response = inference_fn(prompt)
        logger.debug("LLM 会话摘要响应长度=%d", len(response) if response else 0)
    except Exception as exc:
        logger.warning("LLM 会话摘要调用失败: %s", exc)
        return None
    payload = _parse_json_object(response)
    if not payload:
        logger.warning("LLM 会话摘要响应解析失败")
        return None
    abstract = str(payload.get("abstract", "") or "").strip()
    overview = str(payload.get("overview", "") or "").strip()
    if not abstract and not overview:
        logger.warning("LLM 会话摘要结果为空")
        return None
    logger.info("LLM 会话摘要完成: abstract长度=%d, overview长度=%d", len(abstract), len(overview))
    return {"abstract": abstract, "overview": overview}


def extractHighValueFactsWithLlm(params: dict[str, Any]) -> dict[str, Any]:
    logger.info("开始高价值事实提取: sessionFile=%s", params.get("sessionFile", ""))
    inference_fn: InferenceFn = params["inferenceFn"]
    texts: list[str] = params["texts"]
    conversation_text = "\n\n".join(texts)
    prompt = HIGH_VALUE_SLOT_PROMPT + conversation_text[: params.get("llmTruncationChars", 8000)]
    try:
        response = inference_fn(prompt)
        logger.debug("高价值事实提取响应长度=%d", len(response) if response else 0)
    except Exception as exc:
        logger.warning("高价值事实提取调用失败: %s", exc)
        return {"facts": [], "summary": None}
    payload = _parse_json_object(response)
    if not payload:
        logger.warning("高价值事实提取响应解析失败")
        return {"facts": [], "summary": None}

    now = int(time.time() * 1000)
    facts: list[LedgerFact] = []
    for item in (payload.get("facts", []) if isinstance(payload.get("facts"), list) else []):
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "") or "").strip()
        key = str(item.get("key", "") or "").strip()
        value = str(item.get("value", "") or "").strip()
        if not subject or not key or not value:
            continue
        date = str(item.get("date", "") or "").strip()
        value_with_date = f"{value} ({date})" if date else value
        fact = LedgerFact(
            id=stableId("slot", subject, key, value),
            subject=subject,
            key=key,
            value=value_with_date,
            scope="environment",
            text=f"{subject}: {value_with_date}",
            session_key=params["sessionFile"],
            turn_index=int((params.get("messageIndexes") or [params["startIndex"]])[0]),
            tokens=[],
            evidence=EvidenceRef(
                sessionFile=params["sessionFile"],
                messageIndex=int((params.get("messageIndexes") or [params["startIndex"]])[0]),
                snippet=conversation_text[:200],
            ),
            confidence=min(float(params.get("confidenceCeiling", 0.97)), float(item.get("confidence", 0.75) or 0.75)),
            validFrom=now,
            expiresAt=now + int(params["factTtlDays"]) * 24 * 60 * 60 * 1000,
            source="llm-slot-extraction",
            status="active",
        )
        facts.append(fact)

    summary = None
    summary_data = payload.get("summary")
    if isinstance(summary_data, dict):
        abstract = str(summary_data.get("abstract", "") or "").strip()
        overview = str(summary_data.get("overview", "") or "").strip()
        if abstract or overview:
            summary = {"abstract": abstract, "overview": overview}

    logger.info("高价值事实提取完成: 事实数=%d, 有摘要=%s", len(facts), summary is not None)
    return {"facts": facts, "summary": summary}


def extractProfileFactsWithLlm(params: dict[str, Any]) -> list[LedgerFact]:
    logger.info("开始档案事实提取: subject=%s, 文本段数=%d", params.get("subject", ""), len(params.get("texts", [])))
    inference_fn: InferenceFn = params["inferenceFn"]
    subject = str(params["subject"])
    texts: list[str] = params["texts"]
    archive_text = "\n\n".join(texts)
    prompt = PROFILE_FACT_PROMPT.format(subject=subject) + archive_text[: params.get("llmTruncationChars", 12000)]
    try:
        response = inference_fn(prompt)
        logger.debug("档案事实提取响应长度=%d", len(response) if response else 0)
    except Exception as exc:
        logger.warning("档案事实提取调用失败: %s", exc)
        return []
    payload = _parse_json_object(response)
    if not payload:
        logger.warning("档案事实提取响应解析失败")
        return []
    now = int(time.time() * 1000)
    facts: list[LedgerFact] = []
    for item in (payload.get("facts", []) if isinstance(payload.get("facts"), list) else []):
        if not isinstance(item, dict):
            continue
        fact_subject = str(item.get("subject", "") or "").strip()
        key = str(item.get("key", "") or "").strip()
        value = str(item.get("value", "") or "").strip()
        if not fact_subject or not key or not value:
            continue
        date = str(item.get("date", "") or "").strip()
        value_with_date = f"{value} ({date})" if date else value
        facts.append(
            LedgerFact(
                id=stableId("profile", fact_subject, key, value),
                subject=fact_subject,
                key=key,
                value=value_with_date,
                scope="environment",
                text=f"{fact_subject}: {value_with_date}",
                session_key=f"profile:{fact_subject}",
                turn_index=0,
                tokens=[],
                evidence=EvidenceRef(
                    sessionFile=f"profile:{fact_subject}",
                    messageIndex=0,
                    snippet=archive_text[:200],
                ),
                confidence=float(item.get("confidence", 0.8) or 0.8),
                validFrom=now,
                expiresAt=now + int(params.get("factTtlDays", 90)) * 24 * 60 * 60 * 1000,
                source="llm-profile-extraction",
                status="active",
            )
        )
    logger.info("档案事实提取完成: subject=%s, 提取 %d 个事实", subject, len(facts))
    return facts


def extractEntityGraph(params: dict[str, Any]) -> dict[str, Any]:
    logger.info("开始实体图谱提取: sessionId=%s, sessionFile=%s", params.get("sessionId", ""), params.get("sessionFile", ""))
    inference_fn: InferenceFn = params["inferenceFn"]
    embed_fn: EmbedFn | None = params.get("embedFn")
    embed_texts_fn: EmbedTextsFn | None = params.get("embedTextsFn")
    texts: list[str] = params["texts"]
    session_id = params["sessionId"]
    session_file = params["sessionFile"]
    start_index = params["startIndex"]
    message_indexes = params.get("messageIndexes") or [start_index]
    conversation_text = "\n\n".join(texts)
    prompt = ENTITY_EXTRACTION_PROMPT + conversation_text[: params.get("llmTruncationChars", 12000)]
    try:
        response = inference_fn(prompt)
        logger.debug("实体图谱提取响应长度=%d", len(response) if response else 0)
    except Exception as exc:
        logger.warning("实体图谱提取调用失败: %s", exc)
        return {"nodes": [], "edges": []}
    payload = _parse_json_object(response)
    if not payload:
        logger.warning("实体图谱提取响应解析失败")
        return {"nodes": [], "edges": []}

    evidence = EvidenceRef(
        sessionFile=session_file,
        messageIndex=int(message_indexes[0]),
        snippet=conversation_text[:200],
    )
    nodes: list[dict[str, Any]] = []
    edges: list[GraphEdgeRecord] = []
    entity_node_ids: dict[str, str] = {}

    for entity in payload.get("entities", []) if isinstance(payload.get("entities"), list) else []:
        if not isinstance(entity, dict) or not entity.get("name"):
            continue
        name = str(entity["name"])
        node_id = stableId("ENTITY", name.lower().strip())
        entity_node_ids[name.lower().strip()] = node_id
        content = str(entity.get("description", "") or name)
        nodes.append(
            {
                "id": node_id,
                "nodeType": "ENTITY",
                "label": name,
                "content": content,
                "keywords": top_keywords([name, content, str(entity.get("category", ""))], 6),
                "vector": None,
                "confidence": 0.95,
                "evidence": evidence,
                "entityCategory": str(entity.get("category", "unknown") or "unknown"),
            }
        )

    for attr in payload.get("attributes", []) if isinstance(payload.get("attributes"), list) else []:
        if not isinstance(attr, dict):
            continue
        entity_name = str(attr.get("entity", "") or "").lower().strip()
        entity_node_id = entity_node_ids.get(entity_name)
        value = str(attr.get("value", "") or "")
        key = str(attr.get("key", "attribute") or "attribute")
        if not entity_node_id or not value:
            continue
        date = str(attr.get("date", "") or "")
        value_with_date = f"{value} ({date})" if date else value
        fact_content = f"{attr.get('entity')}: {key} = {value_with_date}"
        fact_node_id = stableId("FACT", entity_name, key, value)
        nodes.append(
            {
                "id": fact_node_id,
                "nodeType": "FACT",
                "label": f"{attr.get('entity')}.{key}",
                "content": fact_content,
                "keywords": top_keywords([str(attr.get("entity", "")), key, value], 6),
                "vector": None,
                "confidence": float(attr.get("confidence", 0.7) or 0.7),
                "evidence": evidence,
            }
        )
        edges.append(
            GraphEdgeRecord(
                id=stableId(entity_node_id, fact_node_id, "has_attribute"),
                from_id=entity_node_id,
                to_id=fact_node_id,
                edge_type="has_attribute",
                weight=float(attr.get("confidence", 0.7) or 0.7),
                relation_label=key,
                evidence=evidence,
            )
        )

    for rel in payload.get("relationships", []) if isinstance(payload.get("relationships"), list) else []:
        if not isinstance(rel, dict):
            continue
        from_id = entity_node_ids.get(str(rel.get("from", "") or "").lower().strip())
        to_id = entity_node_ids.get(str(rel.get("to", "") or "").lower().strip())
        if not from_id or not to_id:
            continue
        rel_type = str(rel.get("type", "related_to") or "related_to")
        if rel_type not in {"related_to", "has_attribute", "participates_in", "causes", "prevents", "enables", "supports", "temporal"}:
            rel_type = "related_to"
        edges.append(
            GraphEdgeRecord(
                id=stableId(from_id, to_id, rel_type),
                from_id=from_id,
                to_id=to_id,
                edge_type=rel_type,
                weight=float(rel.get("confidence", 0.7) or 0.7),
                relation_label=f"{rel_type}: {str(rel.get('description', '') or '')}",
                evidence=evidence,
            )
        )

    for event in payload.get("events", []) if isinstance(payload.get("events"), list) else []:
        if not isinstance(event, dict) or not event.get("description"):
            continue
        description = str(event["description"])
        date = str(event.get("date", "") or "")
        event_label = f"{description} ({date})" if date else description
        event_node_id = stableId("EVENT", session_id, event_label)
        nodes.append(
            {
                "id": event_node_id,
                "nodeType": "EVENT",
                "label": event_label[:120],
                "content": event_label,
                "keywords": top_keywords([description, date], 6),
                "vector": None,
                "confidence": float(event.get("confidence", 0.7) or 0.7),
                "evidence": evidence,
            }
        )
        for entity_name in event.get("entities", []) if isinstance(event.get("entities"), list) else []:
            entity_id = entity_node_ids.get(str(entity_name).lower().strip())
            if not entity_id:
                continue
            edges.append(
                GraphEdgeRecord(
                    id=stableId(entity_id, event_node_id, "participates_in"),
                    from_id=entity_id,
                    to_id=event_node_id,
                    edge_type="participates_in",
                    weight=float(event.get("confidence", 0.7) or 0.7),
                    relation_label="participates in",
                    evidence=evidence,
                )
            )

    for causal in payload.get("causal", []) if isinstance(payload.get("causal"), list) else []:
        if not isinstance(causal, dict):
            continue
        source_content = str(causal.get("source", "") or "")
        target_content = str(causal.get("target", "") or "")
        edge_type = str(causal.get("type", "causes") or "causes")
        if not source_content or not target_content or edge_type not in {"causes", "prevents", "enables"}:
            continue
        source_node_id = stableId("EVENT", session_id, source_content)
        target_node_id = stableId("EVENT", session_id, target_content)
        if not any(node["id"] == source_node_id for node in nodes):
            nodes.append(
                {
                    "id": source_node_id,
                    "nodeType": "EVENT",
                    "label": source_content[:120],
                    "content": source_content,
                    "keywords": top_keywords([source_content], 6),
                    "vector": None,
                    "confidence": float(causal.get("confidence", 0.7) or 0.7),
                    "evidence": evidence,
                }
            )
        if not any(node["id"] == target_node_id for node in nodes):
            nodes.append(
                {
                    "id": target_node_id,
                    "nodeType": "EVENT",
                    "label": target_content[:120],
                    "content": target_content,
                    "keywords": top_keywords([target_content], 6),
                    "vector": None,
                    "confidence": float(causal.get("confidence", 0.7) or 0.7),
                    "evidence": evidence,
                }
            )
        edges.append(
            GraphEdgeRecord(
                id=stableId(source_node_id, target_node_id, edge_type),
                from_id=source_node_id,
                to_id=target_node_id,
                edge_type=edge_type,
                weight=float(causal.get("confidence", 0.7) or 0.7),
                relation_label=edge_type,
                evidence=evidence,
            )
        )

    if embed_texts_fn and nodes:
        _batch_embed_nodes(nodes, embed_texts_fn)
    elif embed_fn:
        for node in nodes:
            if node.get("vector") is None and node.get("content"):
                node["vector"] = embed_fn(node["content"])

    logger.info("实体图谱提取完成: 节点数=%d, 边数=%d", len(nodes), len(edges))
    return {"nodes": nodes, "edges": edges}


def buildSessionSummary(session_key: str, date_time: str, entries: list[TranscriptEntry], source_event_ids: list[str]) -> SessionSummary:
    logger.info("构建会话摘要: session_key=%s, 条目数=%d", session_key, len(entries))
    keyword_list = top_keywords([entry.content for entry in entries], limit=4)
    lead_lines: list[str] = []
    for entry in entries:
        if len(lead_lines) >= 2:
            break
        lead_lines.append(f"{entry.speaker}: {' '.join((entry.text or entry.content).split())[:120].rstrip()}")
    keyword_part = ", ".join(keyword_list) if keyword_list else "recent life updates"
    abstract = f"{date_time}: conversation about {keyword_part}."
    overview = " ".join(lead_lines) if lead_lines else abstract
    return SessionSummary(
        session_key=session_key,
        date_time=date_time,
        abstract=abstract,
        overview=overview,
        tokens=[],
        source_event_ids=list(source_event_ids),
    )


def applyExtractedEntityGraph(state: Any, entity_graph: dict[str, Any]) -> None:
    logger.info("开始应用实体图谱: 节点数=%d, 边数=%d", len(entity_graph.get("nodes", [])), len(entity_graph.get("edges", [])))
    for node in entity_graph.get("nodes", []):
        if node.get("nodeType") == "ENTITY":
            entity = EntityNode(
                id=node["id"],
                name=node["label"],
                tokens=list(node.get("keywords", [])),
                description=node.get("content", ""),
                mention_count=max(1, int(node.get("confidence", 0.95) * 10)),
                vector=node.get("vector"),
            )
            existing = state.entities.get(entity.id)
            if existing is None:
                state.entities[entity.id] = entity
            else:
                existing.description = existing.description or entity.description
                existing.tokens = existing.tokens or entity.tokens
                if existing.vector is None:
                    existing.vector = entity.vector
        elif node.get("nodeType") == "EVENT":
            if node["id"] not in state.events:
                state.events[node["id"]] = EventNode(
                    id=node["id"],
                    session_key=getattr(node.get("evidence"), "sessionFile", "") if node.get("evidence") else "",
                    date_time=getattr(node.get("evidence"), "dateTime", "") if node.get("evidence") else "",
                    turn_index=getattr(node.get("evidence"), "messageIndex", 0) if node.get("evidence") else 0,
                    speaker=getattr(node.get("evidence"), "speaker", "memory") if node.get("evidence") else "memory",
                    text=node.get("content", ""),
                    content=node.get("content", ""),
                    tokens=list(node.get("keywords", [])),
                    entity_ids=[],
                    evidence=node.get("evidence"),
                    vector=node.get("vector"),
                )
        elif node.get("nodeType") == "FACT":
            if node["id"] not in state.facts:
                content = node.get("content", "")
                state.facts[node["id"]] = LedgerFact(
                    id=node["id"],
                    subject=content.split(":", 1)[0] if ":" in content else "user",
                    key=node.get("label", "fact"),
                    value=content.split("=", 1)[-1].strip() if "=" in content else content,
                    scope=_graph_fact_scope(str(node.get("label", "fact")), content),
                    text=content,
                    session_key=getattr(node.get("evidence"), "sessionFile", "") if node.get("evidence") else "",
                    turn_index=getattr(node.get("evidence"), "messageIndex", 0) if node.get("evidence") else 0,
                    tokens=list(node.get("keywords", [])),
                    evidence=node.get("evidence"),
                    confidence=float(node.get("confidence", 0.7) or 0.7),
                )
    for edge in entity_graph.get("edges", []):
        if edge.id not in state.graph_edges:
            state.graph_edges[edge.id] = edge
    logger.info("实体图谱应用完成")


COMBINED_EXTRACTION_PROMPT = """You are a memory extraction engine. Perform THREE tasks on the following conversation in a single pass:

TASK 1 - FACTS: Extract ALL factual information as atomic facts.
TASK 2 - SUMMARY: Summarize the session in two parts (abstract + overview).
TASK 3 - ENTITY GRAPH: Extract entities, attributes, relationships, events, and causal links.

RULES:
- NEVER use relative time expressions ("today", "recently", "last week"). Convert to absolute dates.
- If the conversation includes a date/time header, use it as the reference datetime for resolving relative time expressions.
- Extract specific facts, not vague summaries. Each fact must be atomic.
- Include dates, names, places, activities, preferences, relationships, status changes.
- Confidence: 0.95 for explicitly stated facts, 0.8 for strongly implied, 0.6 for inferred.
- Entity names must be consistent (use the same canonical name throughout).
- For causal links, identify explicit or strongly implied cause→effect chains.
- CROSS-SPEAKER OPINIONS: When person A comments on person B's action/plan/trait, extract as a fact with subject=A (e.g. "PersonA thinks PersonB's proposal is promising").
- SPECIFIC IDENTITY: Extract precise identity details only when explicitly stated or strongly supported by multiple pieces of evidence.
- IDENTITY INFERENCE: If identity is inferred rather than explicitly stated, mark it with lower confidence and keep the wording conservative.
- CRITICAL: Preserve exact nouns, proper names, quantities, and specific details from the conversation. Do NOT generalize or paraphrase. For example:
  - GOOD: "researching adoption agencies" / "bought a blue Toyota Camry" / "enrolled in a Python bootcamp"
  - BAD: "doing research" / "bought a car" / "taking a course"
- The value field must contain enough detail that someone who never read the conversation can answer specific questions about it.

OUTPUT FORMAT (JSON only, no markdown):
{
  "facts": [
    {"category":"event","subject":"PersonA","key":"event.description","value":"attended a project planning workshop","confidence":0.95,"date":"2024-03-15"},
    {"category":"opinion","subject":"PersonB","key":"opinion.about_PersonA_project","value":"thinks PersonA's plan is practical","confidence":0.9,"date":"2024-03-15"}
  ],
  "summary": {
    "abstract":"One sentence summary here (max 30 words)",
    "overview":"Detailed paragraph summary here (max 100 words)"
  },
  "entities": [{"name":"PersonA","category":"person","description":"Engineer planning a new project"}],
  "attributes": [{"entity":"PersonA","key":"expertise","value":"backend systems","confidence":0.9,"date":"2024-03-15"}],
  "relationships": [{"from":"PersonA","to":"PersonB","type":"colleague","description":"work together on the same team","confidence":0.9}],
  "events": [{"entities":["PersonA"],"description":"attended a project planning workshop","date":"2024-03-15","confidence":0.95}],
  "causal": [{"source":"PersonB suggested a new approach","target":"PersonA adopted the revised architecture","type":"enables","confidence":0.8}]
}

CONVERSATION:
"""


def extractAllWithLlm(params: dict[str, Any]) -> dict[str, Any]:
    """Combined extraction: facts + summary + entity graph in a single LLM call."""
    logger.info("开始联合提取(事实+摘要+图谱): sessionFile=%s", params.get("sessionFile", ""))
    inference_fn: InferenceFn = params["inferenceFn"]
    texts: list[str] = params["texts"]
    conversation_text = "\n\n".join(texts)
    truncated = conversation_text[: params.get("llmTruncationChars", 12000)]
    prompt = COMBINED_EXTRACTION_PROMPT + truncated
    try:
        response = inference_fn(prompt)
        logger.debug("联合提取响应长度=%d", len(response) if response else 0)
    except Exception as exc:
        logger.warning("联合提取调用失败: %s", exc)
        return {"facts": [], "summary": None, "entity_graph": {"nodes": [], "edges": []}}
    payload = _parse_json_object(response)
    if not payload:
        logger.warning("联合提取响应解析失败")
        return {"facts": [], "summary": None, "entity_graph": {"nodes": [], "edges": []}}

    # Parse facts from combined response
    now = int(time.time() * 1000)
    facts: list[LedgerFact] = []
    for item in (payload.get("facts", []) if isinstance(payload.get("facts"), list) else []):
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "") or "")
        key = str(item.get("key", "") or "")
        value = str(item.get("value", "") or "")
        if not subject or not key or not value:
            continue
        date = str(item.get("date", "") or "")
        value_with_date = f"{value} ({date})" if date else value
        facts.append(
            LedgerFact(
                id=stableId(subject, key, value),
                subject=subject,
                key=key,
                value=value_with_date,
                scope=_fact_category_to_scope(str(item.get("category", "status") or "status")),
                text=f"{subject}: {value_with_date}",
                session_key=params["sessionFile"],
                turn_index=int((params.get("messageIndexes") or [params["startIndex"]])[0]),
                tokens=[],
                evidence=EvidenceRef(
                    sessionFile=params["sessionFile"],
                    messageIndex=int((params.get("messageIndexes") or [params["startIndex"]])[0]),
                    snippet=conversation_text[:200],
                ),
                confidence=min(float(params.get("confidenceCeiling", 0.97)), float(item.get("confidence", 0.7) or 0.7)),
                validFrom=now,
                expiresAt=now + int(params["factTtlDays"]) * 24 * 60 * 60 * 1000,
                source="llm-extraction",
                status="active",
            )
        )

    # Parse summary from combined response
    summary_data = payload.get("summary")
    summary = None
    if isinstance(summary_data, dict):
        abstract = str(summary_data.get("abstract", "") or "").strip()
        overview = str(summary_data.get("overview", "") or "").strip()
        if abstract or overview:
            summary = {"abstract": abstract, "overview": overview}

    # Build entity graph from combined response (reuse existing parsing logic)
    embed_fn = params.get("embedFn")
    embed_texts_fn: EmbedTextsFn | None = params.get("embedTextsFn")
    session_id = params.get("sessionId", "")
    session_file = params["sessionFile"]
    start_index = params["startIndex"]
    message_indexes = params.get("messageIndexes") or [start_index]
    evidence = EvidenceRef(
        sessionFile=session_file,
        messageIndex=int(message_indexes[0]),
        snippet=conversation_text[:200],
    )
    nodes: list[dict[str, Any]] = []
    edges: list[GraphEdgeRecord] = []
    entity_node_ids: dict[str, str] = {}

    for entity in payload.get("entities", []) if isinstance(payload.get("entities"), list) else []:
        if not isinstance(entity, dict) or not entity.get("name"):
            continue
        name = str(entity["name"])
        node_id = stableId("ENTITY", name.lower().strip())
        entity_node_ids[name.lower().strip()] = node_id
        content = str(entity.get("description", "") or name)
        nodes.append({
            "id": node_id, "nodeType": "ENTITY", "label": name, "content": content,
            "keywords": top_keywords([name, content, str(entity.get("category", ""))], 6),
            "vector": None, "confidence": 0.95, "evidence": evidence,
            "entityCategory": str(entity.get("category", "unknown") or "unknown"),
        })

    for attr in payload.get("attributes", []) if isinstance(payload.get("attributes"), list) else []:
        if not isinstance(attr, dict):
            continue
        entity_name = str(attr.get("entity", "") or "").lower().strip()
        entity_node_id = entity_node_ids.get(entity_name)
        value = str(attr.get("value", "") or "")
        key = str(attr.get("key", "attribute") or "attribute")
        if not entity_node_id or not value:
            continue
        date = str(attr.get("date", "") or "")
        value_with_date = f"{value} ({date})" if date else value
        fact_content = f"{attr.get('entity')}: {key} = {value_with_date}"
        fact_node_id = stableId("FACT", entity_name, key, value)
        nodes.append({
            "id": fact_node_id, "nodeType": "FACT", "label": f"{attr.get('entity')}.{key}",
            "content": fact_content, "keywords": top_keywords([str(attr.get("entity", "")), key, value], 6),
            "vector": None, "confidence": float(attr.get("confidence", 0.7) or 0.7), "evidence": evidence,
        })
        edges.append(GraphEdgeRecord(
            id=stableId(entity_node_id, fact_node_id, "has_attribute"),
            from_id=entity_node_id, to_id=fact_node_id, edge_type="has_attribute",
            weight=float(attr.get("confidence", 0.7) or 0.7), relation_label=key, evidence=evidence,
        ))

    for rel in payload.get("relationships", []) if isinstance(payload.get("relationships"), list) else []:
        if not isinstance(rel, dict):
            continue
        from_id = entity_node_ids.get(str(rel.get("from", "") or "").lower().strip())
        to_id = entity_node_ids.get(str(rel.get("to", "") or "").lower().strip())
        if not from_id or not to_id:
            continue
        rel_type = str(rel.get("type", "related_to") or "related_to")
        if rel_type not in {"related_to", "has_attribute", "participates_in", "causes", "prevents", "enables", "supports", "temporal"}:
            rel_type = "related_to"
        edges.append(GraphEdgeRecord(
            id=stableId(from_id, to_id, rel_type), from_id=from_id, to_id=to_id,
            edge_type=rel_type, weight=float(rel.get("confidence", 0.7) or 0.7),
            relation_label=f"{rel_type}: {str(rel.get('description', '') or '')}", evidence=evidence,
        ))

    for event in payload.get("events", []) if isinstance(payload.get("events"), list) else []:
        if not isinstance(event, dict) or not event.get("description"):
            continue
        description = str(event["description"])
        date = str(event.get("date", "") or "")
        event_label = f"{description} ({date})" if date else description
        event_node_id = stableId("EVENT", session_id, event_label)
        nodes.append({
            "id": event_node_id, "nodeType": "EVENT", "label": event_label[:120],
            "content": event_label, "keywords": top_keywords([description, date], 6),
            "vector": None, "confidence": float(event.get("confidence", 0.7) or 0.7), "evidence": evidence,
        })
        for entity_name in event.get("entities", []) if isinstance(event.get("entities"), list) else []:
            entity_id = entity_node_ids.get(str(entity_name).lower().strip())
            if not entity_id:
                continue
            edges.append(GraphEdgeRecord(
                id=stableId(entity_id, event_node_id, "participates_in"),
                from_id=entity_id, to_id=event_node_id, edge_type="participates_in",
                weight=float(event.get("confidence", 0.7) or 0.7), relation_label="participates in", evidence=evidence,
            ))

    for causal in payload.get("causal", []) if isinstance(payload.get("causal"), list) else []:
        if not isinstance(causal, dict):
            continue
        source_content = str(causal.get("source", "") or "")
        target_content = str(causal.get("target", "") or "")
        edge_type = str(causal.get("type", "causes") or "causes")
        if not source_content or not target_content or edge_type not in {"causes", "prevents", "enables"}:
            continue
        source_node_id = stableId("EVENT", session_id, source_content)
        target_node_id = stableId("EVENT", session_id, target_content)
        if not any(node["id"] == source_node_id for node in nodes):
            nodes.append({
                "id": source_node_id, "nodeType": "EVENT", "label": source_content[:120],
                "content": source_content, "keywords": top_keywords([source_content], 6),
                "vector": None, "confidence": float(causal.get("confidence", 0.7) or 0.7), "evidence": evidence,
            })
        if not any(node["id"] == target_node_id for node in nodes):
            nodes.append({
                "id": target_node_id, "nodeType": "EVENT", "label": target_content[:120],
                "content": target_content, "keywords": top_keywords([target_content], 6),
                "vector": None, "confidence": float(causal.get("confidence", 0.7) or 0.7), "evidence": evidence,
            })
        edges.append(GraphEdgeRecord(
            id=stableId(source_node_id, target_node_id, edge_type),
            from_id=source_node_id, to_id=target_node_id, edge_type=edge_type,
            weight=float(causal.get("confidence", 0.7) or 0.7), relation_label=edge_type, evidence=evidence,
        ))

    if embed_texts_fn and nodes:
        _batch_embed_nodes(nodes, embed_texts_fn)
    elif embed_fn:
        for node in nodes:
            if node.get("vector") is None and node.get("content"):
                node["vector"] = embed_fn(node["content"])

    entity_graph = {"nodes": nodes, "edges": edges}
    logger.info("联合提取完成: 事实数=%d, 有摘要=%s, 图谱节点=%d, 图谱边=%d",
                len(facts), summary is not None, len(nodes), len(edges))
    return {"facts": facts, "summary": summary, "entity_graph": entity_graph}


__all__ = ["extractFactsWithLlm", "summarizeSession", "extractEntityGraph", "extractAllWithLlm", "extractHighValueFactsWithLlm", "extractProfileFactsWithLlm", "buildSessionSummary", "applyExtractedEntityGraph"]
