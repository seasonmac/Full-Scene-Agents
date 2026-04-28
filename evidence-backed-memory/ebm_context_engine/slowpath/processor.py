from __future__ import annotations

import logging
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ebm_context_engine.core.hash import stableId

logger = logging.getLogger("ebm_context_engine.slowpath.processor")


def buildSlowPathTurnFingerprint(input: dict[str, Any]) -> str:
    fingerprint_parts: list[Any] = [
        "TURN",
        input.get("sessionId"),
        input.get("sessionKey"),
        input.get("sessionFile"),
        input.get("turnStartIndex"),
        input.get("query"),
    ]
    texts = list(input.get("turnMessagesText", []))
    indexes = list(input.get("turnMessageIndexes", []))
    start_index = int(input.get("turnStartIndex", 0) or 0)
    for index, text in enumerate(texts):
        fingerprint_parts.append(indexes[index] if index < len(indexes) else start_index + index)
        fingerprint_parts.append(text)
    return stableId(*fingerprint_parts)


@dataclass
class SlowPathJob:
    id: str
    payload: dict
    status: str = "pending"
    attempts: int = 0
    lastError: str = ""
    createdAt: float = 0.0
    completedAt: float | None = None


class SlowPathProcessor:
    def __init__(
        self,
        store: Optional[Any] = None,
        *,
        max_retries: int = 3,
        retry_delay_ms: int = 200,
        concurrency: int = 1,
        job_timeout_ms: int = 0,
    ):
        self.store = store
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self.concurrency = max(1, concurrency)
        self.job_timeout_ms = max(0, int(job_timeout_ms))
        self._jobs: dict[str, SlowPathJob] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.RLock()

    def _persist_job(self, job: SlowPathJob, query: str = "") -> None:
        if self.store is None:
            return
        fingerprint = str(job.payload.get("fingerprint", "") or "")
        with self._lock:
            self.store.upsert_slow_path_job(
                job.id,
                job.createdAt,
                job.status,
                job.attempts,
                job.lastError,
                query or str(job.payload.get("query", "") or ""),
                {
                    "id": job.id,
                    "status": job.status,
                    "attempts": job.attempts,
                    "last_error": job.lastError,
                    "created_at": job.createdAt,
                    "completed_at": job.completedAt,
                    "payload": job.payload,
                    "query": query or str(job.payload.get("query", "") or ""),
                    "fingerprint": fingerprint,
                },
            )

    def _materialize_job(self, payload: dict[str, Any]) -> SlowPathJob:
        return SlowPathJob(
            id=str(payload.get("id", "")),
            payload=payload.get("payload", {}),
            status=str(payload.get("status", "pending")),
            attempts=int(payload.get("attempts", 0) or 0),
            lastError=str(payload.get("last_error", "") or ""),
            createdAt=float(payload.get("created_at", time.time()) or time.time()),
            completedAt=float(payload["completed_at"]) if payload.get("completed_at") is not None else None,
        )

    def enqueue(self, job_id: str, payload: dict) -> None:
        logger.info("入队慢路径任务: job_id=%s", job_id)
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is None and self.store is not None:
                stored = self.store.get_slow_path_job(job_id)
                if stored:
                    existing = self._materialize_job(stored)
                    self._jobs[job_id] = existing
                    logger.debug("从存储恢复已有任务: job_id=%s, status=%s", job_id, existing.status)
            if existing and existing.status in {"pending", "running", "completed"}:
                logger.debug("任务已存在且状态为 %s, 跳过入队: job_id=%s", existing.status, job_id)
                return
            job = existing or SlowPathJob(id=job_id, payload=payload, createdAt=time.time())
            job.status = "pending"
            job.attempts = existing.attempts if existing else 0
            self._jobs[job_id] = job
            self._queue.append(job_id)
        self._persist_job(job)
        with self._lock:
            queue_len = len(self._queue)
        logger.info("任务入队完成: job_id=%s, 当前队列长度=%d", job_id, queue_len)

    def resume(self) -> int:
        logger.info("开始恢复未完成的慢路径任务")
        if self.store is None:
            logger.debug("无存储后端, 跳过恢复")
            return 0
        resumable = self.store.list_resumable_slow_path_jobs()
        with self._lock:
            for payload in resumable:
                job = self._materialize_job(payload)
                self._jobs[job.id] = job
                if job.id not in self._queue:
                    self._queue.append(job.id)
        logger.info("恢复完成: 恢复了 %d 个任务", len(resumable))
        return len(resumable)

    def drain(self, executor_fn: Callable[[dict], None]) -> None:
        with self._lock:
            queue_len = len(self._queue)
        logger.info("开始排空慢路径队列: 队列长度=%d, 并发数=%d", queue_len, self.concurrency)
        while True:
            batch: list[str] = []
            with self._lock:
                while self._queue and len(batch) < self.concurrency:
                    batch.append(self._queue.popleft())
                remaining = len(self._queue)
            if not batch:
                break
            logger.debug("处理批次: 批次大小=%d, 剩余队列=%d", len(batch), remaining)
            if self.concurrency == 1:
                for job_id in batch:
                    with self._lock:
                        job = self._jobs[job_id]
                    self._run_job(job, executor_fn)
                continue
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                with self._lock:
                    jobs = {job_id: self._jobs[job_id] for job_id in batch}
                futures = {pool.submit(self._run_job, jobs[job_id], executor_fn): job_id for job_id in batch}
                for future in as_completed(futures):
                    future.result()

    def _run_job(self, job: SlowPathJob, executor_fn: Callable[[dict], None]) -> None:
        logger.info("开始执行任务: job_id=%s, 已尝试次数=%d, 最大重试=%d", job.id, job.attempts, self.max_retries)
        for attempt in range(job.attempts, self.max_retries):
            with self._lock:
                job.attempts = attempt + 1
                job.status = "running"
                job.lastError = ""
            self._persist_job(job)
            try:
                if self.job_timeout_ms > 0:
                    # Run in a daemon thread so timeout doesn't block on hung tasks.
                    # We do NOT join/wait after timeout — the thread is abandoned.
                    pool = ThreadPoolExecutor(max_workers=1)
                    future = pool.submit(executor_fn, job.payload)
                    pool.shutdown(wait=False)
                    try:
                        future.result(timeout=self.job_timeout_ms / 1000.0)
                    except TimeoutError:
                        future.cancel()
                        raise
                else:
                    executor_fn(job.payload)
                with self._lock:
                    job.status = "completed"
                    job.completedAt = time.time()
                self._persist_job(job)
                logger.info("任务执行成功: job_id=%s, 尝试次数=%d", job.id, job.attempts)
                return
            except Exception as error:
                with self._lock:
                    job.lastError = str(error)
                logger.warning("任务执行失败: job_id=%s, 尝试次数=%d, 错误=%s", job.id, attempt + 1, error)
                if attempt >= self.max_retries - 1:
                    with self._lock:
                        job.status = "failed"
                    self._persist_job(job)
                    return
                with self._lock:
                    job.status = "pending"
                self._persist_job(job)
                time.sleep(self.retry_delay_ms * (attempt + 1) / 1000.0)
        with self._lock:
            job.status = "failed"
        self._persist_job(job)

    def status(self) -> dict[str, int]:
        if self.store is not None:
            return self.store.count_slow_path_jobs_by_status()
        with self._lock:
            return {
                "pending": sum(1 for job in self._jobs.values() if job.status == "pending"),
                "running": sum(1 for job in self._jobs.values() if job.status == "running"),
                "done": sum(1 for job in self._jobs.values() if job.status == "completed"),
                "failed": sum(1 for job in self._jobs.values() if job.status == "failed"),
            }

    def retryFailed(self, executor_fn: Callable[[dict], None]) -> int:
        logger.info("开始重试失败的任务")
        with self._lock:
            failed = [job for job in self._jobs.values() if job.status == "failed"]
        if self.store is not None:
            for payload in self.store.list_slow_path_jobs_by_status(["failed"]):
                job = self._materialize_job(payload)
                with self._lock:
                    self._jobs[job.id] = job
                if job not in failed:
                    failed.append(job)
        with self._lock:
            for job in failed:
                job.status = "pending"
                job.lastError = ""
                self._queue.append(job.id)
                self._persist_job(job)
        if failed:
            self.drain(executor_fn)
        logger.info("重试失败任务完成: 重试了 %d 个任务", len(failed))
        return len(failed)

    def jobs(self) -> list[dict]:
        if self.store is not None:
            return self.store.list_slow_path_jobs(500)
        with self._lock:
            return [
                {
                    "id": job.id,
                    "status": job.status,
                    "attempts": job.attempts,
                    "last_error": job.lastError,
                    "created_at": job.createdAt,
                    "completed_at": job.completedAt,
                    "payload": job.payload,
                }
                for job in self._jobs.values()
            ]


__all__ = ["SlowPathJob", "SlowPathProcessor", "buildSlowPathTurnFingerprint"]
