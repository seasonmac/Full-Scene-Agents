from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Multilingual stop words (English + Chinese common function words)
# ---------------------------------------------------------------------------
STOP_WORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "both", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "me", "more", "most", "my", "of", "on", "or", "our", "she", "so", "than", "that",
    "the", "their", "them", "then", "there", "these", "they", "this", "those", "to",
    "too", "up", "us", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "with", "would", "you", "your",
    # Chinese function words / particles
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "他", "她", "它", "们", "那", "些", "什么", "吗", "吧", "呢",
    "啊", "哦", "嗯", "把", "被", "让", "给", "从", "向", "对", "与", "及",
}

COMMON_CAPITALIZED = {
    "A", "An", "And", "As", "At", "Before", "But", "By", "Can", "Could", "Did", "Do",
    "Does", "For", "From", "Good", "Great", "Had", "Has", "Have", "He", "Hey", "Hi",
    "How", "I", "If", "Image", "It", "Last", "Maybe", "My", "No", "Now", "Of", "Okay",
    "Is", "On", "Or", "Please", "She", "Thanks", "That", "The", "Their", "Then", "There",
    "These", "They", "This", "Those", "Today", "Tomorrow", "Was", "We", "Well", "Were",
    "What", "When", "Where", "Who", "Why", "Would", "Yes", "Yesterday", "You",
}

# ---------------------------------------------------------------------------
# Unicode-aware tokenizer
# ---------------------------------------------------------------------------
# Matches: Latin/digit words (including apostrophes), OR individual CJK characters
_TOKEN_RE = re.compile(
    r"[a-z0-9]+(?:'[a-z0-9]+)?"   # Latin/digit words
    r"|[\u4e00-\u9fff]"            # CJK Unified Ideographs
    r"|[\u3400-\u4dbf]"            # CJK Extension A
    r"|[\uf900-\ufaff]"            # CJK Compatibility Ideographs
    r"|[\U00020000-\U0002a6df]"    # CJK Extension B
    r"|[\u3040-\u309f]"            # Hiragana
    r"|[\u30a0-\u30ff]"            # Katakana
    r"|[\uac00-\ud7af]"            # Korean Hangul
    r"|[\u0400-\u04ff]+"           # Cyrillic words
    r"|[\u0600-\u06ff]+"           # Arabic words
    , re.IGNORECASE
)

# Keep old name for backward compat (used by client.py build_hash_vector)
TOKEN_RE = _TOKEN_RE

WHITESPACE_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Multilingual temporal markers
# ---------------------------------------------------------------------------
_TEMPORAL_MARKERS_EN = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "today", "yesterday", "tomorrow", "before", "after", "during", "since", "until", "ago",
)
_TEMPORAL_MARKERS_ZH = (
    "年", "月", "日", "号", "时", "分", "秒",
    "今天", "昨天", "明天", "前天", "后天",
    "上周", "下周", "本周", "上个月", "下个月", "本月",
    "去年", "今年", "明年", "前年",
    "之前", "之后", "以前", "以后", "期间", "以来", "直到",
    "最近", "刚才", "刚刚", "早上", "下午", "晚上", "凌晨",
    "春天", "夏天", "秋天", "冬天",
    "周一", "周二", "周三", "周四", "周五", "周六", "周日",
    "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期天", "星期日",
)

# Keep old name for backward compat
DATE_MARKERS = _TEMPORAL_MARKERS_EN


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def tokenize(text: str, *, keep_stop_words: bool = False) -> list[str]:
    """Unicode-aware tokenizer: handles Latin, CJK, Cyrillic, Arabic, etc."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
    if keep_stop_words:
        return tokens
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def keyword_overlap(query_tokens: Sequence[str], candidate_tokens: Sequence[str]) -> float:
    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    if not query_set or not candidate_set:
        return 0.0
    # Exact match
    overlap = query_set & candidate_set
    # Prefix match for Latin tokens (weight 0.5) + CJK character containment
    partial = 0.0
    unmatched_query = query_set - overlap
    unmatched_candidate = candidate_set - overlap
    for qt in unmatched_query:
        if _is_cjk_char(qt):
            # CJK single char: check if it appears in any multi-char candidate
            for ct in unmatched_candidate:
                if len(ct) > 1 and qt in ct:
                    partial += 0.5
                    break
        elif len(qt) >= 4:
            prefix = qt[:4]
            for ct in unmatched_candidate:
                if len(ct) >= 4 and ct[:4] == prefix:
                    partial += 0.5
                    break
    effective_overlap = len(overlap) + partial
    if effective_overlap == 0:
        return 0.0
    precision = effective_overlap / len(candidate_set)
    recall = effective_overlap / len(query_set)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _is_cjk_char(ch: str) -> bool:
    """Check if a single character is CJK."""
    if len(ch) != 1:
        return False
    cp = ord(ch)
    return (0x4e00 <= cp <= 0x9fff or 0x3400 <= cp <= 0x4dbf
            or 0xf900 <= cp <= 0xfaff or 0x20000 <= cp <= 0x2a6df)


def top_keywords(texts: Iterable[str], limit: int = 6) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return [token for token, _ in counter.most_common(limit)]


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def contains_temporal_marker(text: str) -> bool:
    """Multilingual temporal marker detection."""
    lower = (text or "").lower()
    # Date patterns: YYYY, YYYY-MM-DD, MM/DD, etc.
    if re.search(r"\b\d{4}\b", lower):
        return True
    if re.search(r"\b\d{1,2}[:/.-]\d{1,2}(?:[:/.-]\d{2,4})?\b", lower):
        return True
    # English markers
    if any(marker in lower for marker in _TEMPORAL_MARKERS_EN):
        return True
    # Chinese markers (check original text, not lowered)
    original = text or ""
    if any(marker in original for marker in _TEMPORAL_MARKERS_ZH):
        return True
    return False


def pick_sentences(text: str, *, limit: int = 2, max_chars: int = 240) -> list[str]:
    raw_parts = re.split(r"(?<=[.!?。！？])\s*", normalize_whitespace(text))
    picked: list[str] = []
    for part in raw_parts:
        cleaned = normalize_whitespace(part)
        if len(cleaned) < 12:
            continue
        picked.append(cleaned[:max_chars])
        if len(picked) >= limit:
            break
    return picked
