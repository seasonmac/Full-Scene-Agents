"""
EBM Python HTTP Sidecar Server
==============================
为 OpenClaw TS 插件提供 HTTP 接口，代理所有 context engine 调用到 Python EvidenceBackedMemoryEngine。

启动方式:
    python -m ebm_context_engine.server --port 18790 --config /path/to/ebm/config.json --db /path/to/ebm.sqlite

接口列表:
    POST /bootstrap     — 初始化 session
    POST /ingest        — 写入单条消息
    POST /ingest-batch  — 批量写入消息
    POST /assemble      — 组装上下文（召回 + 构建 systemPrompt）
    POST /after-turn    — 轮次结束后处理（slow path 入队）
    POST /compact       — 压缩（当前委托给 runtime）
    POST /dispose       — 释放资源
    POST /query         — 独立查询接口（用于测试）
    POST /memory-search — 记忆搜索
    POST /memory-get    — 读取单条记忆对象
    POST /memory-forget — 删除/失效可安全删除的记忆对象
    POST /archive-expand — 展开 EBM session summary / transcript
    GET  /status        — slow path 状态
    POST /flush         — 排空 slow path 队列
    POST /retry-failed  — 重试失败的 slow path 作业
    GET  /health        — 健康检查
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ebm_context_engine.core.config import EbmConfig, ServiceEndpointConfig, loadServiceConfig, _parse_service_endpoint
from ebm_context_engine.engine import EvidenceBackedMemoryEngine

logger = logging.getLogger("ebm_context_engine.server")


def _build_engine(
    db_path: str,
    config_path: str | None = None,
    slow_path_enabled_override: bool | None = None,
) -> EvidenceBackedMemoryEngine:
    """根据 config.json 构建 EvidenceBackedMemoryEngine 实例。"""
    logger.info("正在初始化 EBM 引擎: db_path=%s config_path=%s", db_path, config_path)

    service_config: dict[str, Any] = {}
    tuning: dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        embedding_ep = _parse_service_endpoint(raw.get("embedding"))
        llm_ep = _parse_service_endpoint(raw.get("llm"))
        memllm_ep = _parse_service_endpoint(raw.get("memllm"))
        rerank_ep = _parse_service_endpoint(raw.get("rerank"))
        service_config = {"embedding": embedding_ep, "llm": llm_ep, "memllm": memllm_ep, "rerank": rerank_ep}
        for group in ["tuning_ingest", "tuning_qa", "tuning_rrf", "tuning_ppr", "tuning_graph", "tuning_slowpath"]:
            if isinstance(raw.get(group), dict):
                tuning.update(raw[group])
        logger.info("已加载配置文件: embedding=%s llm=%s memllm=%s rerank=%s",
                     embedding_ep.model if embedding_ep else "无",
                     llm_ep.model if llm_ep else "无",
                     memllm_ep.model if memllm_ep else "无",
                     rerank_ep.model if rerank_ep else "无")
    else:
        logger.warning("未找到 config.json，使用默认配置")

    config = EbmConfig(
        storagePath=db_path,
        sdkTimeoutMs=int(tuning.get("sdkTimeoutMs", 45000)),
        sdkMaxRetries=int(tuning.get("sdkMaxRetries", 1)),
        slowPathConcurrency=int(tuning.get("slowPathConcurrency", 3)),
        slowPathJobTimeoutMs=int(tuning.get("slowPathJobTimeoutMs", 9000000)),
        dedupSimilarityThreshold=float(tuning.get("dedupSimilarityThreshold", 0.95)),
        eventPromotionIndegreeThreshold=int(tuning.get("eventPromotionIndegreeThreshold", 3)),
        rebuildChangeThreshold=int(tuning.get("rebuildChangeThreshold", 7)),
        llmTruncationChars=int(tuning.get("llmTruncationChars", 12000)),
        confidenceCeiling=float(tuning.get("confidenceCeiling", 0.97)),
        graphRecallTopK=int(tuning.get("graphRecallTopK", 5)),
        ledgerRecallTopK=int(tuning.get("ledgerRecallTopK", 6)),
        communityRecallTopK=int(tuning.get("communityRecallTopK", 6)),
        recallContentMaxTokens=int(tuning.get("recallContentMaxTokens", 100)),
        episodicMaxTokens=int(tuning.get("episodicMaxTokens", 300)),
        episodicHitsLimit=int(tuning.get("episodicHitsLimit", 3)),
        episodicMessageWindow=int(tuning.get("episodicMessageWindow", 2)),
        sessionSummaryBudgetRatio=float(tuning.get("sessionSummaryBudgetRatio", 0.4)),
        sessionSummaryMaxTokens=int(tuning.get("sessionSummaryMaxTokens", 500)),
        pinnedFactsLimit=int(tuning.get("pinnedFactsLimit", 2)),
        generalizedRecallDiscount=float(tuning.get("generalizedRecallDiscount", 0.8)),
        c2fTopicK=int(tuning.get("c2fTopicK", 10)),
        c2fEpisodeK=int(tuning.get("c2fEpisodeK", 10)),
        c2fFactK=int(tuning.get("c2fFactK", 30)),
        c2fMaxFacts=int(tuning.get("c2fMaxFacts", 20)),
        c2fMaxEpisodes=int(tuning.get("c2fMaxEpisodes", 8)),
        c2fEpisodeSummaryMaxChars=int(tuning.get("c2fEpisodeSummaryMaxChars", 400)),
        c2fFactContentMaxChars=int(tuning.get("c2fFactContentMaxChars", 400)),
        rrfK=int(tuning.get("rrfK", 60)),
        pprDefaultIterations=int(tuning.get("pprDefaultIterations", 8)),
        pprDamping=float(tuning.get("pprDamping", 0.85)),
        layer0TopK=int(tuning.get("layer0TopK", 5)),
        layer1TopK=int(tuning.get("layer1TopK", 8)),
        layer2TopK=int(tuning.get("layer2TopK", 6)),
        contextHitsLimit=int(tuning.get("contextHitsLimit", 6)),
        transcriptRecallLimit=int(tuning.get("transcriptRecallLimit", 3)),
        transcriptRecallLimitSimple=int(tuning.get("transcriptRecallLimitSimple", 2)),
        transcriptContextMaxChars=int(tuning.get("transcriptContextMaxChars", 2200)),
        transcriptSnippetMaxChars=int(tuning.get("transcriptSnippetMaxChars", 300)),
        transcriptWindowRadius=int(tuning.get("transcriptWindowRadius", 1)),
    )
    if slow_path_enabled_override is not None:
        config.slowPathEnabled = slow_path_enabled_override

    engine = EvidenceBackedMemoryEngine(
        artifact_path=db_path,
        config=config,
        memllm_endpoint=service_config.get("memllm") or service_config.get("llm"),
        llm_endpoint=service_config.get("llm"),
        embedding_endpoint=service_config.get("embedding"),
        rerank_endpoint=service_config.get("rerank"),
        slowpath_llm_enabled=config.slowPathEnabled,
    )
    # 使用 _ensure_mutable_state 而非 ensure_loaded，空数据库时自动初始化空状态
    engine._ensure_mutable_state()
    logger.info("EBM 引擎初始化完成: db_path=%s 状态已加载", db_path)
    return engine


class EbmRequestHandler(BaseHTTPRequestHandler):
    """处理 EBM HTTP 请求的 handler。"""

    engine: EvidenceBackedMemoryEngine  # 由 server 工厂设置

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug(format, *args)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_error(self, exc: Exception) -> None:
        logger.error("请求处理失败: %s\n%s", exc, traceback.format_exc())
        self._send_json(500, {"ok": False, "error": str(exc)})

    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                self._send_json(200, {"ok": True, "engine": "ebm_context_engine"})
            elif self.path.startswith("/status"):
                status = self.engine.get_slow_path_status()
                self._send_json(200, {"ok": True, "status": status})
            else:
                self._send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self._handle_error(exc)

    def do_POST(self) -> None:
        try:
            params = self._read_json_body()
            path = self.path.rstrip("/")

            if path == "/bootstrap":
                result = self.engine.bootstrap(params)
                self._send_json(200, {"ok": True, **result})

            elif path == "/ingest":
                result = self.engine.ingest(params)
                self._send_json(200, {"ok": True, **result})

            elif path == "/ingest-batch":
                result = self.engine.ingestBatch(params)
                self._send_json(200, {"ok": True, **result})

            elif path == "/assemble":
                result = self.engine.assemble(params)
                self._send_json(200, {"ok": True, **result})

            elif path == "/after-turn":
                self.engine.afterTurn(params)
                self._send_json(200, {"ok": True})

            elif path == "/compact":
                result = self.engine.compact(params)
                self._send_json(200, {"ok": True, **result})

            elif path == "/dispose":
                self.engine.dispose()
                self._send_json(200, {"ok": True})

            elif path == "/query":
                question = params.get("question", "")
                use_aaak = bool(params.get("useAaak", False))
                result = self.engine.query(question, use_aaak=use_aaak)
                self._send_json(200, {
                    "ok": True,
                    "answer": result.answer,
                    "context": result.context,
                    "debug": result.debug,
                })

            elif path == "/memory-search":
                query = params.get("query", "")
                limit = int(params.get("limit", 10))
                results = self.engine.memory_search(query, limit=limit)
                self._send_json(200, {"ok": True, "results": results})

            elif path == "/memory-get":
                item_id = params.get("id") or params.get("itemId") or ""
                item = self.engine.memory_get(str(item_id))
                self._send_json(200, {"ok": True, "item": item})

            elif path == "/memory-forget":
                item_id = params.get("id") or params.get("itemId") or ""
                result = self.engine.memory_forget(str(item_id))
                self._send_json(200, {"ok": True, **result})

            elif path == "/archive-expand":
                result = self.engine.archive_expand(
                    str(params.get("archiveId") or params.get("id") or ""),
                    session_id=str(params.get("sessionId") or ""),
                    session_key=str(params.get("sessionKey") or ""),
                    limit=int(params.get("limit", 200) or 200),
                )
                self._send_json(200, {"ok": True, **result})

            elif path == "/flush":
                self.engine.flush_slow_path()
                status = self.engine.get_slow_path_status()
                self._send_json(200, {"ok": True, "status": status})

            elif path == "/retry-failed":
                retried = self.engine.retry_failed()
                self._send_json(200, {"ok": True, "retried": retried})

            else:
                self._send_json(404, {"ok": False, "error": "not found"})

        except Exception as exc:
            self._handle_error(exc)


def create_server(port: int, engine: EvidenceBackedMemoryEngine) -> ThreadingHTTPServer:
    """创建并返回 HTTP 服务器实例。"""

    class Handler(EbmRequestHandler):
        pass

    Handler.engine = engine  # type: ignore[attr-defined]
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    logger.info("EBM Python HTTP 服务器已创建: 端口=%d", port)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="EBM Python HTTP Sidecar Server")
    parser.add_argument("--port", type=int, default=18790, help="监听端口 (默认 18790)")
    parser.add_argument("--config", type=str, default=None, help="config.json 路径")
    parser.add_argument("--db", type=str, default=None, help="SQLite 数据库路径")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 默认路径
    db_path = args.db or str(Path.home() / ".openclaw" / "memory" / "ebm_context_engine.sqlite")
    config_path = args.config
    if not config_path:
        candidates = [
            Path(__file__).resolve().parents[1] / "ebm" / "config.json",
            Path.cwd() / "ebm" / "config.json",
        ]
        for c in candidates:
            if c.exists():
                config_path = str(c)
                break

    # 确保数据库目录存在
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    slow_path_enabled_override: bool | None = None
    raw_override = os.environ.get("EBM_PY_SLOWPATH_ENABLED")
    if raw_override is not None:
        normalized = raw_override.strip().lower()
        slow_path_enabled_override = normalized in {"1", "true", "yes", "on"}

    engine = _build_engine(db_path, config_path, slow_path_enabled_override=slow_path_enabled_override)
    server = create_server(args.port, engine)

    def shutdown_handler(signum, frame):
        logger.info("收到信号 %s，正在关闭服务器...", signum)
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    logger.info("EBM Python HTTP 服务器启动: http://127.0.0.1:%d", args.port)
    # 输出 READY 标记，供 TS 插件检测启动完成
    print(f"EBM_PY_READY port={args.port}", flush=True)
    server.serve_forever()
    logger.info("EBM Python HTTP 服务器已停止")


if __name__ == "__main__":
    main()
