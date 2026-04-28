from __future__ import annotations

from typing import Any

from .core.benchmark_utils import formatBenchmarkSessionText
from .core.config import resolveConfig
from .core.embedding_client import buildEmbedFnFromConfig
from .client import OpenAICompatClient
from .engine import EvidenceBackedMemoryEngine


def register(api: Any) -> dict[str, Any]:
    config = resolveConfig(getattr(api, "pluginConfig", None), getattr(api, "resolvePath", lambda x: x))
    service = config.serviceConfig or {}
    embed_fn = buildEmbedFnFromConfig(service.get("embedding"), config.sdkTimeoutMs, config.sdkMaxRetries) if service.get("embedding") else None
    inference_client = OpenAICompatClient(service.get("llm"), timeout_s=max(1, config.sdkTimeoutMs / 1000), max_retries=config.sdkMaxRetries) if service.get("llm") else None

    def inference(prompt: str) -> str:
        if inference_client is None:
            raise RuntimeError("LLM inference endpoint is not configured")
        endpoint = service.get("llm")
        temperature = getattr(endpoint, "temperature", None) if endpoint is not None else None
        max_tokens = getattr(endpoint, "maxTokens", None) if endpoint is not None else None
        return inference_client.chat(
            [{"role": "user", "content": prompt}],
            temperature=float(temperature if temperature is not None else 0.0),
            max_tokens=int(max_tokens) if isinstance(max_tokens, int) else None,
        ).content
    engine_ref: dict[str, EvidenceBackedMemoryEngine | None] = {"engine": None}

    def ensureEngine() -> EvidenceBackedMemoryEngine:
        if engine_ref["engine"] is None:
            engine_ref["engine"] = EvidenceBackedMemoryEngine(
                config.storagePath,
                config=config,
                memllm_endpoint=service.get("memllm"),
                llm_endpoint=service.get("llm"),
                embedding_endpoint=service.get("embedding"),
                slowpath_llm_enabled=config.slowPathEnabled,
            )
        return engine_ref["engine"]

    def contextEngineFactory() -> EvidenceBackedMemoryEngine:
        return ensureEngine()

    def gateway_traces(params: dict[str, Any] | None = None):
        params = params or {}
        engine = ensureEngine()
        limit = int(params.get("limit", 50) or 50)
        return {
            "traces": engine.getAllTraces(limit),
            "slowPathJobs": engine.getAllSlowPathJobs(limit),
            "slowPathStatus": engine.getSlowPathStatus(),
        }

    def gateway_data(params: dict[str, Any] | None = None):
        params = params or {}
        engine = ensureEngine()
        limit = int(params.get("limit", 200) or 200)
        return {
            "transcripts": engine.getAllTranscripts(limit),
            "graphNodes": engine.getAllGraphNodes(),
            "graphEdges": engine.getAllGraphEdges(),
            "communities": engine.getAllCommunities(),
            "facts": engine.getAllFacts(limit),
        }

    def gateway_flush():
        engine = ensureEngine()
        return {"status": engine.flushSlowPath()}

    def http_status(detail: bool = False):
        engine = ensureEngine()
        return {"ok": True, "status": engine.getSlowPathStatusDetailed() if detail else engine.getSlowPathStatus()}

    def http_retry_failed():
        engine = ensureEngine()
        retried = engine.retryFailed()
        return {"ok": True, "retried": retried}

    return {
        "config": config,
        "embedFn": embed_fn,
        "inferenceFn": inference if inference_client is not None else None,
        "ensureEngine": ensureEngine,
        "contextEngineFactory": contextEngineFactory,
        "contextEngineId": "ebm-context-engine",
        "pluginInfo": {
            "id": "ebm-context-engine",
            "name": "Evidence-Backed Memory",
            "description": "Three-plane memory architecture with evidence backlinks and dual-path retrieval.",
            "kind": "context-engine",
        },
        "gatewayMethods": {
            "ebm.traces": gateway_traces,
            "ebm.data": gateway_data,
            "ebm.flush": gateway_flush,
        },
        "httpRoutes": {
            "/v1/extensions/ebm/status": http_status,
            "/v1/extensions/ebm/retry-failed": http_retry_failed,
        },
        "benchmarkUtils": {
            "formatBenchmarkSessionText": formatBenchmarkSessionText,
        },
    }


__all__ = ["register"]
