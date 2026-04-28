"""AAAK encoder for HyperMem context rendering.

Converts retrieved facts and episodes into compact AAAK v2 notation to reduce
token consumption while preserving key information:
- Entity markers (WHO: 3-letter uppercase)
- Weight (importance)
- Core assertion (stripped of framing boilerplate)
"""
from __future__ import annotations

import re
from typing import Sequence, Tuple

from ebm_context_engine.types import HmFact, HmEpisode

# Speaker name → 3-letter entity code
_ENTITY_CACHE: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Multilingual fact parsing patterns
# ---------------------------------------------------------------------------

# Pattern 1: "On YYYY-MM-DD, Speaker verb ..." (English)
_FACT_PARSE_EN = re.compile(
    r"^(?:On\s+(\d{4}-\d{2}-\d{2}),?\s+)?"     # optional date prefix
    r"(\w+)\s+"                                   # speaker name
    r"(?:said\s+(?:that\s+)?|told\s+\w+\s+(?:that\s+)?|described\s+|encouraged\s+)"
    r"(.+)$",
    re.IGNORECASE,
)

# Pattern 2: "Speaker: content" (heuristic fallback facts)
_SPEAKER_PREFIX = re.compile(r"^(\w+):\s+(.+)$")

# Pattern 3: "YYYY-MM-DD Speaker content" or "YYYY年MM月DD日 Speaker content"
_DATE_PREFIX = re.compile(
    r"^(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?),?\s*"
)


def _entity_code(name: str) -> str:
    """Convert a speaker name to 3-letter uppercase entity code."""
    if name in _ENTITY_CACHE:
        return _ENTITY_CACHE[name]
    # For CJK names, use first 1-2 chars; for Latin, first 3
    if any('\u4e00' <= ch <= '\u9fff' for ch in name):
        code = name[:2].upper()
    else:
        code = name[:3].upper()
    _ENTITY_CACHE[name] = code
    return code


def _importance_to_weight(importance: str) -> int:
    return {"high": 4, "mid": 3, "low": 2}.get(importance, 3)


def _strip_fact(content: str) -> Tuple[str, str, str]:
    """Parse a fact into (date, entity_code, core_assertion).

    Tries multiple patterns for language-agnostic parsing.
    Returns ('', 'UNK', content) if parsing fails.
    """
    # Try English verb-frame pattern
    m = _FACT_PARSE_EN.match(content)
    if m:
        date = m.group(1) or ""
        speaker = m.group(2)
        assertion = m.group(3).strip().rstrip(".")
        return date, _entity_code(speaker), assertion

    # Try "Speaker: content" pattern
    m2 = _SPEAKER_PREFIX.match(content)
    if m2:
        speaker = m2.group(1)
        assertion = m2.group(2).strip().rstrip(".")
        return "", _entity_code(speaker), assertion

    # Try extracting date prefix from any language
    m3 = _DATE_PREFIX.match(content)
    if m3:
        date = m3.group(1)
        rest = content[m3.end():].strip()
        return date, "UNK", rest.rstrip(".")

    return "", "UNK", content.strip().rstrip(".")


def encode_facts_aaak(
    fact_results: Sequence[Tuple[HmFact, float]],
    episode_results: Sequence[Tuple[HmEpisode, float]],
    question: str,
    intent: str,
    max_facts: int = 8,
    max_episodes: int = 3,
) -> str:
    """Render retrieved facts/episodes as a compact AAAK v2 document string."""
    lines: list[str] = []

    dates: set[str] = set()
    zettel_lines: list[str] = []
    idx = 0

    for i, (fact, score) in enumerate(fact_results):
        if idx >= max_facts:
            break
        if fact.importance == "low" and i >= 3:
            continue

        date, entity, assertion = _strip_fact(fact.content)
        if not assertion or len(assertion) < 8:
            continue

        if len(assertion) > 150:
            cut = assertion[:150].rfind(" ")
            if cut < 80:
                # For CJK text, space-based cut may fail; just truncate
                assertion = assertion[:147] + "..."
            else:
                assertion = assertion[:cut]

        if date:
            dates.add(date)

        w = _importance_to_weight(fact.importance)
        zettel_lines.append(f'{idx}:{entity}|"{assertion}"|W{w}|dt|C')
        idx += 1

    if not zettel_lines:
        return _fallback_plain(fact_results, episode_results, question, intent, max_facts, max_episodes)

    date_str = sorted(dates)[0] if dates else "?"
    lines.append(f"V2|mem|{intent}|{date_str}|hm")
    lines.extend(zettel_lines)

    for episode, score in episode_results[:max_episodes]:
        ts = episode.timestamp_start or episode.session_key
        summary = episode.summary[:120] if episode.summary else episode.title
        lines.append(f"EP:[{ts}] {summary}")

    lines.append(f"Q:{question}")
    return "\n".join(lines)


def _fallback_plain(
    fact_results: Sequence[Tuple[HmFact, float]],
    episode_results: Sequence[Tuple[HmEpisode, float]],
    question: str,
    intent: str,
    max_facts: int,
    max_episodes: int,
) -> str:
    """Plain text fallback when AAAK parsing fails for all facts."""
    lines = [f"Answer from memory context. Intent={intent}.", ""]
    if fact_results:
        lines.append("Facts:")
        count = 0
        for i, (fact, score) in enumerate(fact_results):
            if count >= max_facts:
                break
            if fact.importance == "low" and i >= 3:
                continue
            content = fact.content.strip()
            if len(content) < 10:
                continue
            count += 1
            if len(content) > 200:
                content = content[:197] + "..."
            lines.append(f"{count}. {content}")
        lines.append("")

    if episode_results:
        lines.append("Episodes:")
        for episode, score in episode_results[:max_episodes]:
            ts = episode.timestamp_start or episode.session_key
            summary = episode.summary[:160] if episode.summary else episode.title
            lines.append(f"- [{ts}] {summary}")
        lines.append("")

    lines.append(question)
    return "\n".join(lines)
