from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from time import perf_counter as _time
from typing import Any, Optional

import httpx
import numpy as np

from .text import tokenize
from .core.service_endpoint_utils import endpointLooksLocal, resolveEmbeddingApiKey, resolveServiceApiKey

logger = logging.getLogger("ebm_context_engine.client")


class _EmptyChatContentError(RuntimeError):
    """Raised when an OpenAI-compatible chat response contains no answer text."""


def normalize_vector(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        return array
    return array / norm


def build_hash_vector(text: str, dimension: int = 256) -> np.ndarray:
    values = np.zeros(dimension, dtype=np.float32)
    for token in tokenize(text, keep_stop_words=True):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = -1.0 if digest[4] % 2 else 1.0
        values[bucket] += sign
    norm = float(np.linalg.norm(values))
    if norm <= 0:
        return values
    return values / norm


def cosine_similarity(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> float:
    """Proper cosine similarity: dot(a,b) / (||a|| * ||b||).
    Matches TS vector.ts cosineSimilarity."""
    if left is None or right is None:
        return 0.0
    a = np.asarray(left, dtype=np.float32).ravel()
    b = np.asarray(right, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    dim = min(a.size, b.size)
    if dim <= 0:
        return 0.0
    a = a[:dim]
    b = b[:dim]
    numerator = float(np.dot(a, b))
    left_norm = float(np.dot(a, a))
    right_norm = float(np.dot(b, b))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (np.sqrt(left_norm) * np.sqrt(right_norm))


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _embed_progress_enabled() -> bool:
    raw = str(os.getenv("EBM_EMBED_PROGRESS", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


class _EmbeddingProgressBar:
    def __init__(self, label: str, total: int) -> None:
        self.label = label
        self.total = max(int(total), 1)
        self.current = 0
        self.enabled = _embed_progress_enabled()
        if self.enabled:
            self._render()

    def update(self, step: int) -> None:
        if not self.enabled:
            return
        self.current = min(self.total, self.current + max(int(step), 0))
        self._render()

    def close(self) -> None:
        if not self.enabled:
            return
        self.current = self.total
        self._render()
        sys.stderr.write("\n")
        sys.stderr.flush()

    def _render(self) -> None:
        width = 28
        ratio = 0.0 if self.total <= 0 else self.current / self.total
        filled = min(width, int(width * ratio))
        bar = "#" * filled + "-" * (width - filled)
        sys.stderr.write(
            f"\r[ebm embed] {self.label:<18} [{bar}] {self.current}/{self.total}"
        )
        sys.stderr.flush()


def _extract_chat_content(data: dict) -> str:
    """Extract text content from an OpenAI-compatible chat completion response dict.

    Handles content-parts format and reasoning-model responses where the
    answer may appear in reasoning_content instead of content.
    """
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text") or ""
                if isinstance(text_value, dict):
                    text_value = text_value.get("value") or ""
                elif isinstance(text_value, list):
                    text_value = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in text_value
                    )
                elif not isinstance(text_value, str):
                    text_value = str(text_value)
                if not text_value and isinstance(part.get("content"), list):
                    text_value = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in part.get("content", [])
                    )
                parts.append(text_value)
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    text = str(content).strip()
    if text:
        return text
    # Fallback: reasoning models may put the answer in reasoning_content
    return str(message.get("reasoning_content") or "").strip()


def _extract_usage(data: dict) -> tuple[int, int, int]:
    usage = data.get("usage") or {}
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(usage.get("total_tokens") or 0),
    )


def _message_content(message: Any) -> str:
    """Legacy helper kept for compatibility — delegates to dict-based extraction."""
    if message is None:
        return ""
    if not isinstance(message, dict):
        message = {
            "content": getattr(message, "content", None),
            "reasoning_content": getattr(message, "reasoning_content", None),
        }
    return _extract_chat_content({"choices": [{"message": message}]})


class OpenAICompatClient:
    def __init__(self, endpoint: Any, *, timeout_s: float = 60.0, hash_dimension: int = 256, max_retries: int = 1, _allow_fallback: bool = True):
        self._endpoint = endpoint
        self._max_retries = max(int(max_retries), 0)
        endpoint_dimension = getattr(endpoint, "dimension", None) or getattr(endpoint, "Dimension", None) or None
        self._hash_dimension = int(endpoint_dimension or hash_dimension)
        self._cache: dict[str, np.ndarray] = {}

        base_url = str(
            getattr(endpoint, "base_url", None)
            or getattr(endpoint, "baseUrl", None)
            or ""
        ).rstrip("/")
        model = str(getattr(endpoint, "model", None) or getattr(endpoint, "Model", None) or "")
        service_api_key = resolveServiceApiKey(endpoint) if endpoint is not None else {"apiKey": None}
        embedding_api_key = resolveEmbeddingApiKey(endpoint) if endpoint is not None else {"apiKey": None, "isDummy": False}
        api_key = service_api_key.get("apiKey") or None
        embedding_api_key_value = embedding_api_key.get("apiKey") or None
        raw_headers = getattr(endpoint, "headers", None) or getattr(endpoint, "Headers", None) or None
        extra_headers = {str(k): str(v) for k, v in raw_headers.items()} if isinstance(raw_headers, dict) else {}

        self.model = model
        self.dimension = endpoint_dimension
        self.is_enabled = bool(base_url and model)
        self._base_url = base_url
        self._client: Optional[httpx.Client] = None
        self._embedding_failures = 0
        self._embedding_disabled = False
        self._chat_failures = 0
        self._chat_disabled = False
        fallback_endpoint = getattr(endpoint, "fallback", None) if endpoint is not None else None
        self._fallback = OpenAICompatClient(
            fallback_endpoint,
            timeout_s=timeout_s,
            hash_dimension=hash_dimension,
            max_retries=max_retries,
            _allow_fallback=False,
        ) if _allow_fallback and fallback_endpoint is not None else None

        if self.is_enabled:
            # Use the embedding API key for embedding-only endpoints (dimension set),
            # otherwise use the service (chat) key first.
            effective_key = (
                embedding_api_key_value if endpoint_dimension is not None
                else api_key or embedding_api_key_value or (
                    "sk-local-no-key-required" if endpointLooksLocal(base_url) else "EMPTY"
                )
            )
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if effective_key:
                headers["Authorization"] = f"Bearer {effective_key}"
            headers.update(extra_headers)
            # Store connection params so we can rebuild on stale-connection errors.
            self._client_headers = headers
            self._client_timeout = timeout_s
            self._client = httpx.Client(
                base_url=base_url + "/",
                headers=headers,
                timeout=timeout_s,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
        if self._fallback is not None:
            self._fallback.close()

    def _reconnect(self) -> None:
        """Rebuild the httpx.Client after a stale-connection error.

        HTTP proxies may silently drop idle keep-alive connections.  When we
        detect a RemoteProtocolError we tear down the old client and open a
        fresh one so the next attempt succeeds.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        logger.debug(
            "Rebuilding httpx.Client after stale-connection error: model=%s base_url=%s",
            self.model,
            self._base_url,
        )
        self._client = httpx.Client(
            base_url=self._base_url + "/",
            headers=self._client_headers,
            timeout=self._client_timeout,
        )

    def embed_text(self, text: str, *, progress_label: str | None = None) -> np.ndarray:
        return self.embed_texts([text], progress_label=progress_label)[0]

    def embed_texts(self, texts: list[str], *, progress_label: str | None = None) -> list[np.ndarray]:
        if not texts:
            return []

        cached: dict[int, np.ndarray] = {}
        pending: list[tuple[int, str]] = []
        for index, text in enumerate(texts):
            if text in self._cache:
                cached[index] = self._cache[text]
            else:
                pending.append((index, text))

        progress = _EmbeddingProgressBar(progress_label, len(pending)) if progress_label and pending else None

        if pending and self.is_enabled and self._client is not None and not self._embedding_disabled:
            max_embed_retries = max(1, self._max_retries)
            last_exc: Exception | None = None
            for attempt in range(max_embed_retries):
              try:
                batch_size = 128
                for offset in range(0, len(pending), batch_size):
                    chunk = pending[offset: offset + batch_size]
                    payload: dict[str, Any] = {
                        "model": self.model,
                        "input": [text for _, text in chunk],
                    }
                    if self.dimension:
                        payload["dimensions"] = self.dimension
                    _t0 = _time()
                    logger.debug(
                        "embed request: model=%s base_url=%s n_texts=%d attempt=%d",
                        self.model, self._base_url, len(chunk), attempt + 1,
                    )
                    resp = self._client.post("embeddings", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    logger.debug(
                        "embed response: model=%s n_texts=%d latency=%.0fms",
                        self.model, len(chunk), (_time() - _t0) * 1000,
                    )
                    items = data.get("data") or []
                    if len(items) != len(chunk):
                        raise ValueError("embedding response shape mismatch")
                    for (index, text), item in zip(chunk, items, strict=True):
                        embedding = item.get("embedding") if isinstance(item, dict) else None
                        vector = normalize_vector(list(embedding or []))
                        self._cache[text] = vector
                        cached[index] = vector
                    if progress is not None:
                        progress.update(len(chunk))
                self._embedding_failures = 0
                last_exc = None
                break
              except Exception as exc:
                last_exc = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                # Don't retry on auth errors or schema errors
                if status_code in (401, 403, 422):
                    logger.warning("embed permanent error (status=%s), skipping retries: %s", status_code, exc, exc_info=True)
                    break
                if attempt < max_embed_retries - 1:
                    logger.warning("embed transient error (attempt=%d/%d): %s", attempt + 1, max_embed_retries, exc, exc_info=True)
                else:
                    logger.warning("embed failed after %d attempts: %s", max_embed_retries, exc, exc_info=True)
            if last_exc is not None:
                if self._fallback is not None and self._fallback.is_enabled:
                    fallback_vectors = self._fallback.embed_texts([text for _, text in pending])
                    for (index, text), vector in zip(pending, fallback_vectors, strict=True):
                        self._cache[text] = vector
                        cached[index] = vector
                    return [cached[index] for index in range(len(texts))]
                self._embedding_failures += 1
                if self._embedding_failures >= 3:
                    logger.warning("embed disabled after %d consecutive failures", self._embedding_failures)
                    self._embedding_disabled = True
                for index, text in pending:
                    vector = build_hash_vector(text, self._hash_dimension)
                    self._cache[text] = vector
                    cached[index] = vector
                if progress is not None:
                    progress.update(len(pending))
        else:
            for index, text in pending:
                vector = build_hash_vector(text, self._hash_dimension)
                self._cache[text] = vector
                cached[index] = vector
            if progress is not None:
                progress.update(len(pending))

        if progress is not None:
            progress.close()

        return [cached[index] for index in range(len(texts))]

    def rerank(self, query: str, documents: list[str], top_n: Optional[int] = None) -> list[tuple[int, float]]:
        """Rerank documents using the rerank API (OpenAI-compatible /rerank endpoint).

        Returns list of (original_index, relevance_score) sorted by score descending.
        Raises RuntimeError if rerank is not available.
        """
        if not self.is_enabled or self._client is None:
            raise RuntimeError("rerank endpoint is not configured")
        if not documents:
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        try:
            resp = self._client.post("rerank", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("rerank request failed: %s", exc)
            raise

        results: list[tuple[int, float]] = []
        for item in data.get("results", []):
            idx = int(item.get("index", 0))
            score = float(item.get("relevance_score", 0.0))
            results.append((idx, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> ChatResult:
        if not self.is_enabled or self._client is None or self._chat_disabled:
            raise RuntimeError("chat endpoint is not configured")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        # max_tokens is universally supported by all OpenAI-compatible APIs.
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        primary_attempts = max(self._max_retries + 1, 1)
        last_exc: Optional[Exception] = None
        for attempt in range(primary_attempts):
            try:
                _t0 = _time()
                logger.debug(
                    "chat request: model=%s base_url=%s messages=%d max_tokens=%s attempt=%d/%d",
                    self.model, self._base_url, len(messages), max_tokens, attempt + 1, primary_attempts,
                )
                resp = self._client.post("chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = _extract_chat_content(data)
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(data)
                logger.debug(
                    "chat response: model=%s base_url=%s latency=%.0fms prompt=%d completion=%d content_len=%d",
                    self.model, self._base_url, (_time() - _t0) * 1000,
                    prompt_tokens, completion_tokens, len(content),
                )
                if not content:
                    detail = json.dumps(data, ensure_ascii=False)[:2000]
                    logger.warning(
                        "OpenAICompatClient received empty chat content: model=%s base_url=%s data=%s",
                        self.model,
                        self._base_url,
                        detail,
                    )
                    raise _EmptyChatContentError(
                        f"empty chat content from {self.model} @ {self._base_url}: {detail}"
                    )
                self._chat_failures = 0
                return ChatResult(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            except httpx.RemoteProtocolError as exc:
                last_exc = exc
                logger.warning(
                    "chat stale-connection on attempt %d/%d, reconnecting: model=%s base_url=%s error=%s",
                    attempt + 1, primary_attempts, self.model, self._base_url, exc,
                )
                self._reconnect()
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "chat attempt failed: model=%s base_url=%s attempt=%d/%d error=%s",
                    self.model,
                    self._base_url,
                    attempt + 1,
                    primary_attempts,
                    exc,
                )

        if self._fallback is not None and self._fallback.is_enabled:
            logger.info(
                "Primary chat exhausted retries, switching to fallback: model=%s base_url=%s",
                self._fallback.model,
                self._fallback._base_url,
            )
            try:
                result = self._fallback.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                self._chat_failures = 0
                return result
            except Exception as fb_exc:
                logger.warning("Chat fallback failed: primary=%s fallback=%s", last_exc, fb_exc)
                last_exc = fb_exc

        self._chat_failures += 1
        if self._chat_failures >= 5:
            self._chat_disabled = True
        raise last_exc  # type: ignore[misc]
