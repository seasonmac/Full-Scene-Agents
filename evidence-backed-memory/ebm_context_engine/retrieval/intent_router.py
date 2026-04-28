"""LLM + embedding intent classification for query routing.

Primary: Embedding-based classification (fast, zero LLM calls).
LLM verification: Only for temporally ambiguous cases (embedding temporal hit or near-tie).
No hardcoded keywords, regex, or rule-based temporal detection.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

import numpy as np

from ebm_context_engine.client import OpenAICompatClient, cosine_similarity
from ebm_context_engine.types import ClassificationResult, MemoryState, QueryPlan
from ebm_context_engine.text import tokenize

logger = logging.getLogger("ebm_context_engine.retrieval.intent_router")


INTENT_WEIGHT_MAP: dict[str, dict[str, float]] = {
    "generic": {"vectorWeight": 0.5, "bm25Weight": 0.3, "causalEdgeBoost": 1.2, "temporalEdgeBoost": 1.2, "entityEdgeBoost": 1.2, "pprIterations": 8.0, "temporalFactBonus": 1.0},
    "temporal": {"vectorWeight": 0.4, "bm25Weight": 0.3, "causalEdgeBoost": 1.0, "temporalEdgeBoost": 1.8, "entityEdgeBoost": 1.0, "pprIterations": 8.0, "temporalFactBonus": 1.5},
    "entity": {"vectorWeight": 0.4, "bm25Weight": 0.3, "causalEdgeBoost": 1.0, "temporalEdgeBoost": 1.0, "entityEdgeBoost": 1.6, "pprIterations": 8.0, "temporalFactBonus": 1.0},
    "multi_hop": {"vectorWeight": 0.3, "bm25Weight": 0.2, "causalEdgeBoost": 1.4, "temporalEdgeBoost": 1.1, "entityEdgeBoost": 1.6, "pprIterations": 10.0, "temporalFactBonus": 1.0},
    "causal": {"vectorWeight": 0.3, "bm25Weight": 0.2, "causalEdgeBoost": 1.8, "temporalEdgeBoost": 1.0, "entityEdgeBoost": 1.0, "pprIterations": 10.0, "temporalFactBonus": 1.0},
}

# Intent prototype descriptions for embedding-based classification
INTENT_PROTOTYPES: dict[str, str] = {
    "temporal": "When did something happen? What time, date, year, month? Questions about timing, schedule, deadlines, duration, before, after, during, since, ago.",
    "causal": "Why did something happen? What caused it? What is the reason? What are the effects, consequences, and impacts? Cause and effect relationships.",
    "multi_hop": "Compare two things. What is the relationship between A and B? What do they have in common? How are they different? Questions requiring multiple sources.",
    "entity": "Tell me about a specific person, place, thing, or concept. What are their attributes, properties, or characteristics?",
    "generic": "General factual question. What happened? What is something? Describe an event or situation.",
}

_intent_prototype_cache: dict[int, dict[str, np.ndarray]] = {}


def _infer_reasoning_modes(intent: str) -> list[str]:
    return {
        "temporal": ["timeline", "fact_lookup"],
        "causal": ["causal_chain", "fact_lookup"],
        "multi_hop": ["relationship", "comparison", "entity_lookup"],
        "entity": ["entity_lookup", "fact_lookup"],
        "generic": ["fact_lookup"],
    }.get(intent, ["fact_lookup"])


def _ensure_intent_prototypes(embed_fn: Callable[[str], np.ndarray]) -> dict[str, np.ndarray]:
    """Lazily compute and cache intent prototype embeddings, keyed by embed_fn identity."""
    fn_id = id(embed_fn)
    cached = _intent_prototype_cache.get(fn_id)
    if cached and len(cached) == len(INTENT_PROTOTYPES):
        return cached
    local = {intent: embed_fn(desc) for intent, desc in INTENT_PROTOTYPES.items()}
    _intent_prototype_cache[fn_id] = local
    logger.debug("意图原型向量已缓存: fn_id=%d, keys=%s", fn_id, list(local.keys()))
    return local


def _embedding_classification(
    query: str,
    query_vector: np.ndarray,
    embed_fn: Callable[[str], np.ndarray],
    state: Optional[MemoryState] = None,
) -> tuple[ClassificationResult, dict[str, float]]:
    """Classify query intent using embedding similarity -- zero LLM calls.

    Pure semantic similarity, no keyword/regex overrides.
    Returns (ClassificationResult, all_scores_dict).
    """
    t0 = time.perf_counter()
    tokens = tokenize(query)[:8]

    entities = _extract_entities_from_state(query, state) if state else []

    prototypes = _ensure_intent_prototypes(embed_fn)
    best_intent = "generic"
    best_score = -1.0
    all_scores: dict[str, float] = {}

    for intent, proto_vec in prototypes.items():
        score = float(cosine_similarity(query_vector, proto_vec))
        all_scores[intent] = score
        if score > best_score:
            best_score = score
            best_intent = intent

    word_count = len(query.split())
    if word_count > 25 or len(entities) > 2:
        complexity = "deep"
    elif word_count > 12 or len(entities) > 1:
        complexity = "standard"
    else:
        complexity = "simple"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "嵌入式意图分类完成: intent=%s, score=%.4f, complexity=%s, entities=%s, scores=%s, 耗时=%.1fms",
        best_intent, best_score, complexity, entities,
        {k: round(v, 4) for k, v in all_scores.items()}, elapsed_ms,
    )
    return ClassificationResult(
        intent=best_intent,
        complexity=complexity,
        confidence=min(0.95, max(0.6, best_score)),
        source="embedding",
        weights=INTENT_WEIGHT_MAP[best_intent].copy(),
        entities=entities,
        target_slots=[],
        focus_terms=tokens,
        reasoning_modes=_infer_reasoning_modes(best_intent),
        time_scope="",
    ), all_scores


_LLM_INTENT_SYSTEM = (
    "Classify the question's intent. Return JSON: "
    '{"intent":"temporal|entity|causal|multi_hop|generic",'
    '"answer_type":"date|time|duration|person|place|thing|description|yes_no|list|number",'
    '"search_query":"core search terms",'
    '"confidence":0.0-1.0}. '
    "temporal=expected answer IS a date/time/duration. "
    "entity=asks about person/place/thing/attribute. "
    "CRITICAL: time words (ago,before,after) do NOT make it temporal. "
    "Only temporal if EXPECTED ANSWER is date/time/duration. "
    '"Where did she live 4 years ago?"->entity,place. '
    '"When did she move?"->temporal,date.'
)


def _robust_json_parse(text: str) -> dict:
    """Parse JSON from LLM response with robust fallback.

    Handles cases where LLM wraps JSON in markdown code blocks or
    includes extra text before/after the JSON object.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Try stripping markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                try:
                    return json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    pass
    # Try finding first { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _llm_classification(
    query: str,
    llm_client: OpenAICompatClient,
    state: Optional[MemoryState] = None,
) -> tuple[ClassificationResult, dict[str, Any]] | None:
    """LLM-based intent classification (like OpenViking's intent_analysis).

    Returns ClassificationResult + debug dict, or None if LLM call fails.
    """
    entities = _extract_entities_from_state(query, state) if state else []
    t0 = time.perf_counter()

    try:
        try:
            result = llm_client.chat(
                [
                    {"role": "system", "content": _LLM_INTENT_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=80,
                response_format={"type": "json_object"},
            )
        except TypeError:
            # Fallback for providers that don't support response_format
            result = llm_client.chat(
                [
                    {"role": "system", "content": _LLM_INTENT_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=80,
            )
    except Exception as exc:
        logger.warning("LLM意图分类调用失败: %s", exc)
        return None

    payload = _robust_json_parse(result.content)
    if not isinstance(payload, dict):
        logger.warning("LLM意图分类返回非dict结构: type=%s content=%s", type(payload).__name__, result.content[:200])
        return None
    intent = str(payload.get("intent", "generic") or "generic").lower().strip()
    if intent not in INTENT_WEIGHT_MAP:
        intent = "generic"

    answer_type = str(payload.get("answer_type", "description") or "description").lower().strip()
    search_query = str(payload.get("search_query", "") or "").strip()
    try:
        llm_confidence = float(payload.get("confidence", 0.8) or 0.8)
    except (ValueError, TypeError):
        llm_confidence = 0.8

    tokens = tokenize(search_query or query)[:8]
    word_count = len(query.split())
    if word_count > 25 or len(entities) > 2:
        complexity = "deep"
    elif word_count > 12 or len(entities) > 1:
        complexity = "standard"
    else:
        complexity = "simple"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "LLM意图分类完成: intent=%s, answer_type=%s, confidence=%.2f, complexity=%s, 耗时=%.1fms",
        intent, answer_type, llm_confidence, complexity, elapsed_ms,
    )

    debug = {
        "source": "llm",
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "used_llm": True,
        "fallback_used": False,
        "answer_type": answer_type,
        "search_query": search_query,
    }

    classification = ClassificationResult(
        intent=intent,
        complexity=complexity,
        confidence=min(0.95, max(0.6, llm_confidence)),
        source="llm",
        weights=INTENT_WEIGHT_MAP[intent].copy(),
        entities=entities,
        target_slots=[],
        focus_terms=tokens,
        reasoning_modes=_infer_reasoning_modes(intent),
        time_scope="",
        answer_type=answer_type,
    )
    return classification, debug


def _extract_entities_from_state(query: str, state: MemoryState) -> list[str]:
    """Extract entities by matching known entity names from state against the query."""
    if not state or not state.entities:
        return []

    query_lower = query.lower()
    found: list[str] = []
    seen: set[str] = set()

    known_names = sorted(state.entities.keys(), key=len, reverse=True)
    for name in known_names:
        name_lower = name.lower()
        if len(name_lower) < 2:
            continue
        if name_lower in query_lower and name_lower not in seen:
            seen.add(name_lower)
            found.append(name)
    return found


def _build_plan(classification: ClassificationResult) -> QueryPlan:
    intent = classification.intent
    complexity = classification.complexity
    logger.info("构建查询计划: intent=%s, complexity=%s", intent, complexity)
    if complexity == "deep":
        plan = QueryPlan(intent, complexity, 6, 5, 1, 1, 5, 5, 1, 1, 1, 2, 8, True, True, False)
    elif complexity == "simple":
        plan = QueryPlan(intent, complexity, 5, 4, 0, 0, 4, 4, 0, 0, 0, 1, 4, False, False, False)
    else:
        plan = QueryPlan(intent, complexity, 5, 4, 1, 0, 4, 4, 1, 0, 0, 1, 6, True, False, False)

    reasoning_modes = set(classification.reasoning_modes)
    if "timeline" in reasoning_modes or intent == "temporal":
        plan.prefer_temporal = True
        plan.include_summaries = True
        plan.summary_top_k = max(plan.summary_top_k, 1)
    if "entity_lookup" in reasoning_modes or "relationship" in reasoning_modes or intent in {"entity", "multi_hop", "causal"}:
        plan.prefer_entity_expansion = True
    if "comparison" in reasoning_modes or "causal_chain" in reasoning_modes or intent in {"multi_hop", "causal"}:
        plan.include_communities = True
        plan.community_top_k = max(plan.community_top_k, 1)
        plan.max_hops = max(plan.max_hops, 2)
    logger.debug("查询计划构建完成: prefer_temporal=%s, prefer_entity_expansion=%s, include_communities=%s, max_hops=%d",
                 plan.prefer_temporal, plan.prefer_entity_expansion, plan.include_communities, plan.max_hops)
    return plan


def classify_query(
    query: str,
    llm_client: Optional[OpenAICompatClient] = None,
    *,
    query_vector: Optional[np.ndarray] = None,
    embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    state: Optional[MemoryState] = None,
):
    """Classify query intent.

    Priority: embedding-based (fast) → LLM verification for ambiguous cases → heuristic.
    No regex or hardcoded keyword matching.
    """
    logger.info("开始查询分类: query=%s", query[:120])

    classification = None
    debug: dict[str, int | str | float | bool] = {}
    all_scores: dict[str, float] = {}

    # Primary: fast embedding-based classification
    if query_vector is not None and embed_fn is not None:
        classification, all_scores = _embedding_classification(query, query_vector, embed_fn, state)

    # LLM verification for ambiguous cases:
    # 1. When embedding classifies as temporal (may be false positive)
    # 2. When temporal score is close to best score (may be false negative)
    need_llm_verify = False
    if classification is not None and llm_client is not None and llm_client.is_enabled:
        if classification.intent == "temporal":
            need_llm_verify = True
        elif all_scores:
            temporal_score = all_scores.get("temporal", 0.0)
            best_score = max(all_scores.values()) if all_scores else 0.0
            # If temporal score is within 0.03 of best, it's ambiguous
            if best_score - temporal_score < 0.03:
                need_llm_verify = True
                logger.info("时间意图得分接近最高分: temporal=%.4f, best=%.4f, 触发LLM验证",
                            temporal_score, best_score)

    if need_llm_verify:
        llm_result = _llm_classification(query, llm_client, state)
        if llm_result is not None:
            llm_cls, debug = llm_result
            if classification.intent == "temporal" and llm_cls.intent != "temporal":
                logger.info("LLM纠正意图: embedding=%s → llm=%s (answer_type=%s)",
                            classification.intent, llm_cls.intent, llm_cls.answer_type)
                classification = llm_cls
            elif classification.intent != "temporal" and llm_cls.intent == "temporal":
                logger.info("LLM补充时间意图: embedding=%s → llm=%s (answer_type=%s)",
                            classification.intent, llm_cls.intent, llm_cls.answer_type)
                classification = llm_cls
            elif classification.intent == "temporal" and llm_cls.intent == "temporal":
                # LLM confirms temporal, use LLM's richer classification
                classification = llm_cls

    # Last resort: heuristic
    if classification is None:
        classification = _heuristic_fallback(query, state)

    if not debug:
        debug = {
            "source": classification.source,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "used_llm": classification.source == "llm",
            "fallback_used": classification.source in ("heuristic", "embedding"),
        }
    logger.info("查询分类完成: intent=%s, complexity=%s, source=%s, answer_type=%s",
                classification.intent, classification.complexity, classification.source,
                getattr(classification, "answer_type", ""))
    return classification, _build_plan(classification), debug


def _heuristic_fallback(query: str, state: Optional[MemoryState] = None) -> ClassificationResult:
    """Minimal heuristic using only state entity matching (no regex/keyword detection)."""
    tokens = tokenize(query)[:8]
    entities = _extract_entities_from_state(query, state) if state else []

    if entities:
        intent = "entity"
    else:
        intent = "generic"

    word_count = len(query.split())
    if word_count > 25 or len(entities) > 2:
        complexity = "deep"
    elif word_count > 12 or len(entities) > 1:
        complexity = "standard"
    else:
        complexity = "simple"

    logger.info("启发式分类完成: intent=%s, complexity=%s, entities=%s", intent, complexity, entities)
    return ClassificationResult(
        intent=intent,
        complexity=complexity,
        confidence=0.6,
        source="heuristic",
        weights=INTENT_WEIGHT_MAP[intent].copy(),
        entities=entities,
        target_slots=[],
        focus_terms=tokens,
        reasoning_modes=_infer_reasoning_modes(intent),
        time_scope="",
    )


IntentClassification = ClassificationResult


def classifyIntent(
    query: str,
    llm_client: Optional[OpenAICompatClient] = None,
    **kwargs: Any,
):
    return classify_query(query, llm_client, **kwargs)


__all__ = ["IntentClassification", "INTENT_WEIGHT_MAP", "classifyIntent", "classify_query"]
