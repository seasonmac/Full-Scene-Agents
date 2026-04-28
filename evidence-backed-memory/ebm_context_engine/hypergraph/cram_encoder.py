"""CRAM encoder for EBM HyperMem context rendering.

CRAM = Compact Representation for AI Memory

Bridges EBM types (HmFact, HmEpisode, UnifiedFact, RecallHit)
to the CRAM compact format.

Key design decisions:
- Evidence (fact.content) is NEVER compressed — kept verbatim
- Metadata (keywords, potential, summary) is aggressively compressed
- Language-agnostic: no NLP dependency
- Output is LLM-readable without special instructions
"""
from __future__ import annotations

import sys
import os
from typing import Sequence, Tuple

# Add cram to path so we can import the library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cram"))

from src import CramCodec, CompactFact, CompactEpisode

from ebm_context_engine.types import HmFact, HmEpisode, UnifiedFact, RecallHit


def encode_facts_cram(
    fact_results: Sequence[Tuple[HmFact, float]],
    episode_results: Sequence[Tuple[HmEpisode, float]],
    question: str,
    intent: str,
    max_facts: int = 8,
    max_episodes: int = 3,
) -> str:
    """Encode HmFact/HmEpisode results into compact CRAM string.

    Evidence (fact.content) is preserved verbatim — never compressed.
    """
    codec = CramCodec(max_facts=max_facts, max_episodes=max_episodes)

    compact_facts: list[CompactFact] = []
    dates: set[str] = set()
    count = 0
    for i, (fact, score) in enumerate(fact_results):
        if count >= max_facts:
            break
        if fact.importance == "low" and i >= 3:
            continue
        if not fact.content or len(fact.content.strip()) < 8:
            continue

        cf = codec.encode_fact(
            content=fact.content,  # VERBATIM
            importance=fact.importance,
            keywords=fact.keywords,
            potential=fact.potential,
            fact_id=fact.id,
            episode_id=fact.episode_id,
            session_key=fact.session_key,
            source_turn_start=fact.source_turn_start,
            source_turn_end=fact.source_turn_end,
        )
        compact_facts.append(cf)
        count += 1

    if not compact_facts:
        return _fallback_plain(fact_results, episode_results, question, intent,
                               max_facts, max_episodes)

    compact_episodes: list[CompactEpisode] = []
    for episode, score in episode_results[:max_episodes]:
        ce = codec.encode_episode(
            episode_id=episode.id,
            session_key=episode.session_key,
            title=episode.title,
            summary=episode.summary,
            keywords=episode.keywords,
            timestamp_start=episode.timestamp_start,
            timestamp_end=episode.timestamp_end,
            turn_start=episode.turn_start,
            turn_end=episode.turn_end,
        )
        compact_episodes.append(ce)
        if episode.timestamp_start:
            dates.add(episode.timestamp_start[:10])

    date_str = sorted(dates)[0] if dates else "?"

    packet = codec.encode_packet(
        facts=compact_facts,
        episodes=compact_episodes,
        question=question,
        intent=intent,
        date=date_str,
    )
    return packet.encode()


def encode_unified_facts_cram(
    fact_results: Sequence[Tuple[UnifiedFact, float]],
    episode_results: Sequence[Tuple[HmEpisode, float]],
    question: str,
    intent: str,
    max_facts: int = 8,
    max_episodes: int = 3,
) -> str:
    """Encode UnifiedFact results into compact CRAM string."""
    codec = CramCodec(max_facts=max_facts, max_episodes=max_episodes)

    compact_facts: list[CompactFact] = []
    count = 0
    for i, (fact, score) in enumerate(fact_results):
        if count >= max_facts:
            break
        if fact.importance == "low" and i >= 3:
            continue
        if not fact.content or len(fact.content.strip()) < 8:
            continue

        cf = codec.encode_fact(
            content=fact.content,  # VERBATIM
            importance=fact.importance,
            keywords=fact.keywords,
            potential=fact.potential,
            fact_id=fact.id,
            episode_id=fact.episode_id,
            session_key=fact.session_key,
            source_turn_start=fact.source_turn_start,
            source_turn_end=fact.source_turn_end,
            subject=fact.subject,
            scope=fact.scope,
        )
        compact_facts.append(cf)
        count += 1

    if not compact_facts:
        return _fallback_plain_unified(fact_results, episode_results, question,
                                       intent, max_facts, max_episodes)

    compact_episodes = _encode_episodes(codec, episode_results, max_episodes)
    dates = {e.t_range.split("~")[0][:10] for e in compact_episodes if e.t_range}
    date_str = sorted(dates)[0] if dates else "?"

    packet = codec.encode_packet(
        facts=compact_facts,
        episodes=compact_episodes,
        question=question,
        intent=intent,
        date=date_str,
    )
    return packet.encode()


def encode_recall_hits_cram(
    hits: Sequence[RecallHit],
    question: str,
    intent: str = "",
    max_hits: int = 8,
) -> str:
    """Encode RecallHit list into compact CRAM string.

    For RecallHit, content is the evidence — kept verbatim.
    """
    codec = CramCodec(max_facts=max_hits)
    compact_facts: list[CompactFact] = []

    for hit in hits[:max_hits]:
        cf = CompactFact(
            fid=hit.id[:8] if hit.id else "",
            evidence=hit.content,  # VERBATIM
            imp="m",
            kw="",
            pot="",
            eid="",
            sk=hit.session_key or "",
            subj=hit.source or "",
        )
        compact_facts.append(cf)

    if not compact_facts:
        return f"No memory context found.\nQ|{question}"

    packet = codec.encode_packet(
        facts=compact_facts,
        question=question,
        intent=intent,
    )
    return packet.encode()


def _encode_episodes(
    codec: CramCodec,
    episode_results: Sequence[Tuple[HmEpisode, float]],
    max_episodes: int,
) -> list[CompactEpisode]:
    result = []
    for episode, score in episode_results[:max_episodes]:
        ce = codec.encode_episode(
            episode_id=episode.id,
            session_key=episode.session_key,
            title=episode.title,
            summary=episode.summary,
            keywords=episode.keywords,
            timestamp_start=episode.timestamp_start,
            timestamp_end=episode.timestamp_end,
            turn_start=episode.turn_start,
            turn_end=episode.turn_end,
        )
        result.append(ce)
    return result


def _fallback_plain(
    fact_results: Sequence[Tuple[HmFact, float]],
    episode_results: Sequence[Tuple[HmEpisode, float]],
    question: str,
    intent: str,
    max_facts: int,
    max_episodes: int,
) -> str:
    """Plain text fallback when no facts are parseable."""
    lines = [f"Answer from memory context. Intent={intent}.", ""]
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
        lines.append(f"{count}. {content}")  # evidence verbatim
    if count:
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


def _fallback_plain_unified(
    fact_results: Sequence[Tuple[UnifiedFact, float]],
    episode_results: Sequence[Tuple[HmEpisode, float]],
    question: str,
    intent: str,
    max_facts: int,
    max_episodes: int,
) -> str:
    lines = [f"Answer from memory context. Intent={intent}.", ""]
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
        lines.append(f"{count}. {content}")
    if count:
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
