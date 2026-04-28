#!/usr/bin/env python3
"""
Ingest JSONL chat sessions into OpenClaw session transcripts.

Default source dataset:
    news_report/dataset/*.jsonl

Default behavior:
    - replay each JSONL message into transcript storage
    - one OpenClaw session per file
    - write transcript messages without triggering model generation
    - write manifest.json and ingest.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = "./dataset"
DEFAULT_OUTPUT_DIR = "./openclaw_ingest"
OPENCLAW_BASE_URL = "http://127.0.0.1:18789"
DEFAULT_GLOB = "*.jsonl"
DEFAULT_MODE = "replay"
DEFAULT_SESSION_LAYOUT = "per-file"
DEFAULT_TIMEOUT = 300
DEFAULT_EBM_BRIDGE = "bootstrap"


class IngestRunError(RuntimeError):
    """Raised when --stop-on-error should terminate the run."""


@dataclass
class MessageRecord:
    role: str
    content: str
    timestamp: int
    line_number: int


@dataclass
class SessionPlan:
    source_file: Path
    file_stem: str
    user: str
    session_key: str
    messages: list[MessageRecord]
    errors: list[str]
    send_inputs: list[dict[str, Any]]
    session_id: str
    transcript_path: Path

    @property
    def first_timestamp(self) -> int | None:
        return self.messages[0].timestamp if self.messages else None

    @property
    def last_timestamp(self) -> int | None:
        return self.messages[-1].timestamp if self.messages else None


@dataclass
class EbmRuntimeConfig:
    mode: str
    base_url: str
    db_path: Path


class RunLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str = "") -> None:
        self.lines.append(message)
        print(message, file=sys.stderr)

    def write_to(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def sanitize_key_part(value: str, fallback: str) -> str:
    chars = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("-")
    cleaned = "".join(chars).strip("-")
    return cleaned or fallback


def default_dataset_name(dataset_dir: Path) -> str:
    return sanitize_key_part(dataset_dir.name, "dataset")


def derive_user(
    dataset_name: str,
    file_stem: str,
    session_layout: str,
    user_prefix: str | None,
) -> str:
    suffix = "all-in-one" if session_layout == "all-in-one" else file_stem
    if user_prefix:
        prefix = user_prefix.rstrip(":")
    else:
        prefix = f"news-ingest:{dataset_name}"
    return f"{prefix}:{suffix}"


def derive_session_key(
    dataset_name: str,
    file_stem: str,
    session_layout: str,
    session_prefix: str | None,
) -> str:
    suffix = "all-in-one" if session_layout == "all-in-one" else file_stem
    if session_prefix:
        prefix = session_prefix.rstrip("/")
    else:
        prefix = f"agent:main:news/{dataset_name}"
    return f"{prefix}/{suffix}"


def derive_session_id(session_key: str) -> str:
    digest = hashlib.sha1(session_key.encode("utf-8")).hexdigest()
    return f"ingest-{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def extract_response_text(response_json: dict[str, Any]) -> str:
    """Extract assistant text from a /v1/responses payload."""
    try:
        for item in response_json.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text = content.get("text")
                        if isinstance(text, str):
                            return text
        for item in response_json.get("output", []):
            text = item.get("text")
            if isinstance(text, str):
                return text
            for content in item.get("content", []):
                text = content.get("text")
                if isinstance(text, str):
                    return text
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


def read_message_records(path: Path) -> tuple[list[MessageRecord], list[str]]:
    records: list[MessageRecord] = []
    errors: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: invalid json ({exc.msg})")
            continue

        if payload.get("type") != "message":
            continue

        message = payload.get("message")
        if not isinstance(message, dict):
            errors.append(f"line {line_number}: missing message object")
            continue

        role = message.get("role")
        content = message.get("content")
        timestamp_raw = message.get("timestamp", payload.get("timestamp"))

        if not isinstance(role, str) or not role.strip():
            errors.append(f"line {line_number}: missing/invalid message.role")
            continue
        if not isinstance(content, str) or not content.strip():
            errors.append(f"line {line_number}: missing/invalid message.content")
            continue
        try:
            timestamp = int(timestamp_raw)
        except (TypeError, ValueError):
            errors.append(f"line {line_number}: missing/invalid timestamp")
            continue

        records.append(
            MessageRecord(
                role=role.strip(),
                content=content,
                timestamp=timestamp,
                line_number=line_number,
            )
        )
    return records, errors


def build_bundle_input(file_name: str, records: list[MessageRecord]) -> str:
    parts = [f"[session transcript: {file_name}]"]
    for record in records:
        parts.append(f"{record.role}: {record.content}")
    return "\n\n".join(parts)


def expand_path(value: str) -> str:
    return str(Path(value).expanduser())


def resolve_ebm_runtime_config() -> EbmRuntimeConfig | None:
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    if not config_path.exists():
        return None
    data = json.loads(config_path.read_text(encoding="utf-8"))
    entry = (((data.get("plugins") or {}).get("entries") or {}).get("ebm-context-engine") or {})
    plugin_cfg = entry.get("config") or {}
    if not isinstance(plugin_cfg, dict):
        return None
    mode = str(plugin_cfg.get("mode") or "local")
    if mode == "remote":
        base_url = str(plugin_cfg.get("baseUrl") or "").rstrip("/")
    else:
        port = int(plugin_cfg.get("port") or 18790)
        base_url = f"http://127.0.0.1:{port}"
    db_path = Path(expand_path(str(plugin_cfg.get("dbPath") or "~/.openclaw/memory/ebm_context_engine.sqlite"))).resolve()
    return EbmRuntimeConfig(mode=mode, base_url=base_url, db_path=db_path)


def ebm_post_json(base_url: str, path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ebm_health(base_url: str, timeout: int = 10) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return bool(payload.get("ok"))
    except Exception:
        return False


def build_ebm_messages_for_session(plan: SessionPlan, mode: str) -> list[dict[str, Any]]:
    if mode == "bundle":
        return [
            {
                "role": "user",
                "content": build_bundle_input(plan.source_file.name, plan.messages),
                "timestamp": plan.first_timestamp or int(time.time() * 1000),
            }
        ]
    return [
        {
            "role": record.role,
            "content": record.content,
            "timestamp": record.timestamp,
        }
        for record in plan.messages
    ]


def bridge_session_to_ebm(
    plan: SessionPlan,
    ebm_cfg: EbmRuntimeConfig,
    mode: str,
    bridge_mode: str,
    timeout: int,
) -> dict[str, Any]:
    if bridge_mode == "off":
        return {
            "bridge_mode": bridge_mode,
            "bootstrap": None,
            "after_turn": None,
            "messages_sent": 0,
        }

    if not ebm_health(ebm_cfg.base_url, timeout=min(timeout, 10)):
        raise OSError(f"EBM sidecar unavailable at {ebm_cfg.base_url}")

    bootstrap_res = None
    after_turn_res = None
    messages_sent = 0
    if bridge_mode in {"bootstrap", "both"}:
        bootstrap_res = ebm_post_json(
            ebm_cfg.base_url,
            "/bootstrap",
            {
                "sessionId": plan.session_id,
                "sessionKey": plan.session_key,
                "sessionFile": str(plan.transcript_path),
            },
            timeout=timeout,
        )

    if bridge_mode in {"after-turn", "both"}:
        after_turn_messages = build_ebm_messages_for_session(plan, mode)
        after_turn_res = ebm_post_json(
            ebm_cfg.base_url,
            "/after-turn",
            {
                "sessionId": plan.session_id,
                "sessionKey": plan.session_key,
                "sessionFile": str(plan.transcript_path),
                "messages": after_turn_messages,
                "prePromptMessageCount": 0,
            },
            timeout=timeout,
        )
        messages_sent = len(after_turn_messages)

    return {
        "bridge_mode": bridge_mode,
        "bootstrap": bootstrap_res,
        "after_turn": after_turn_res,
        "messages_sent": messages_sent,
    }


def flush_ebm(ebm_cfg: EbmRuntimeConfig, timeout: int) -> dict[str, Any]:
    return ebm_post_json(ebm_cfg.base_url, "/flush", {}, timeout=timeout)


def query_ebm_db_stats(db_path: Path, plan: SessionPlan) -> dict[str, int]:
    if not db_path.exists():
        return {
            "transcript_entries": 0,
            "session_summaries": 0,
            "facts": 0,
        }
    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        transcript_entries = cursor.execute(
            """
            select count(*)
            from transcript_entries
            where session_id = ?
               or session_key = ?
               or session_file = ?
            """,
            (plan.session_id, plan.session_key, str(plan.transcript_path)),
        ).fetchone()[0]
        session_summaries = cursor.execute(
            """
            select count(*)
            from session_summaries
            where session_id = ?
               or session_key = ?
               or session_file = ?
            """,
            (plan.session_id, plan.session_key, str(plan.transcript_path)),
        ).fetchone()[0]
        facts = cursor.execute(
            """
            select count(*)
            from facts
            where session_key = ?
               or json_extract(evidence_json, '$.sessionFile') = ?
            """,
            (plan.session_key, str(plan.transcript_path)),
        ).fetchone()[0]
        return {
            "transcript_entries": int(transcript_entries),
            "session_summaries": int(session_summaries),
            "facts": int(facts),
        }
    finally:
        conn.close()


def collect_session_plans(
    *,
    dataset_dir: Path,
    glob_pattern: str,
    limit_files: int | None,
    session_layout: str,
    mode: str,
    user_prefix: str | None,
    session_prefix: str | None,
) -> list[SessionPlan]:
    dataset_name = default_dataset_name(dataset_dir)
    source_files = sorted(dataset_dir.glob(glob_pattern))
    if limit_files is not None:
        source_files = source_files[:limit_files]

    plans: list[SessionPlan] = []
    for source_file in source_files:
        file_stem = sanitize_key_part(source_file.stem, "session")
        messages, errors = read_message_records(source_file)
        session_key = derive_session_key(dataset_name, file_stem, session_layout, session_prefix)
        user = derive_user(dataset_name, file_stem, session_layout, user_prefix)
        session_id = derive_session_id(session_key)
        transcript_path = (
            Path.home() / ".openclaw" / "agents" / "main" / "sessions" / f"{session_id}.jsonl"
        ).resolve()
        if mode == "bundle":
            send_inputs = [
                {
                    "role": "user",
                    "content": build_bundle_input(source_file.name, messages),
                }
            ] if messages else []
        else:
            send_inputs = [
                {
                    "role": record.role,
                    "content": record.content,
                    "timestamp": record.timestamp,
                }
                for record in messages
            ]
        plans.append(
            SessionPlan(
                source_file=source_file,
                file_stem=file_stem,
                user=user,
                session_key=session_key,
                messages=messages,
                errors=list(errors),
                send_inputs=send_inputs,
                session_id=session_id,
                transcript_path=transcript_path,
            )
        )
    return plans


def send_openclaw_transcript_messages(
    *,
    session_key: str,
    session_id: str,
    transcript_path: Path,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    helper_source = """
import fs from "node:fs/promises";
import path from "node:path";
import { CURRENT_SESSION_VERSION, SessionManager } from "@mariozechner/pi-coding-agent";
import { loadSessionStore, updateSessionStore } from "../src/config/sessions/store.ts";

async function main() {
  const [, , payloadPath] = process.argv;
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf-8"));

  await fs.mkdir(path.dirname(payload.transcriptPath), { recursive: true });

  try {
    await fs.access(payload.transcriptPath);
  } catch {
    const header = {
      type: "session",
      version: CURRENT_SESSION_VERSION,
      id: payload.sessionId,
      timestamp: new Date().toISOString(),
      cwd: process.cwd(),
    };
    await fs.writeFile(payload.transcriptPath, `${JSON.stringify(header)}\\n`, { encoding: "utf-8", mode: 0o600 });
  }

  const store = loadSessionStore(payload.storePath);
  await updateSessionStore(payload.storePath, (draft) => {
    draft[payload.sessionKey] = {
      ...(draft[payload.sessionKey] ?? {}),
      sessionId: payload.sessionId,
      updatedAt: Date.now(),
      sessionFile: payload.transcriptPath,
    };
  });

  const sessionManager = SessionManager.open(payload.transcriptPath);
  const appended = [];
  for (const msg of payload.messages) {
    const body = {
      role: msg.role,
      content: [{ type: "text", text: msg.content }],
      timestamp: Number.isFinite(msg.timestamp) ? msg.timestamp : Date.now(),
    };
    const messageId = sessionManager.appendMessage(body);
    appended.push({ messageId, role: msg.role });
  }

  console.log(JSON.stringify({ ok: true, appended, sessionFile: payload.transcriptPath }));
}

main().catch((err) => {
  console.error(err instanceof Error ? err.stack || err.message : String(err));
  process.exitCode = 1;
});
"""
    with tempfile.TemporaryDirectory(
        prefix="openclaw-transcript-import-",
        dir=str((REPO_ROOT / "openclaw").resolve()),
    ) as tmpdir:
        tmp_path = Path(tmpdir)
        helper_path = tmp_path / "inject-session.ts"
        payload_path = tmp_path / "payload.json"
        helper_path.write_text(helper_source, encoding="utf-8")
        payload_path.write_text(
            json.dumps(
                {
                    "storePath": str((Path.home() / ".openclaw" / "agents" / "main" / "sessions" / "sessions.json").resolve()),
                    "sessionKey": session_key,
                    "sessionId": session_id,
                    "transcriptPath": str(transcript_path),
                    "messages": messages,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result = subprocess.run(
            ["node", "--import", "tsx", str(helper_path), str(payload_path)],
            cwd=str((REPO_ROOT / "openclaw").resolve()),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise OSError(result.stderr.strip() or result.stdout.strip() or f"helper exited with code {result.returncode}")
        return json.loads(result.stdout.strip() or "{}")


def build_manifest(
    *,
    dataset_dir: Path,
    session_layout: str,
    mode: str,
    started_at: str,
    finished_at: str,
    totals: dict[str, int],
    sessions: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "session_layout": session_layout,
        "mode": mode,
        "dry_run": dry_run,
        "started_at": started_at,
        "finished_at": finished_at,
        "totals": totals,
        "sessions": sessions,
    }


def run_ingest(
    args: argparse.Namespace,
    *,
    sender: Callable[..., dict[str, Any]] = send_openclaw_transcript_messages,
    logger: RunLogger | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    logger = logger or RunLogger()
    started_at = datetime.now().astimezone().isoformat()
    run_dir = output_root / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ebm_cfg = None if args.skip_ebm_bridge else resolve_ebm_runtime_config()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    plans = collect_session_plans(
        dataset_dir=dataset_dir,
        glob_pattern=args.glob,
        limit_files=args.limit_files,
        session_layout=args.session_layout,
        mode=args.mode,
        user_prefix=args.user_prefix,
        session_prefix=args.session_prefix,
    )

    logger.log("=== OpenClaw Ingest Run ===")
    logger.log(f"dataset_dir: {dataset_dir}")
    logger.log(f"mode: {args.mode}")
    logger.log(f"session_layout: {args.session_layout}")
    logger.log(f"dry_run: {args.dry_run}")
    logger.log(f"glob: {args.glob}")
    logger.log(f"limit_files: {args.limit_files if args.limit_files is not None else 'all'}")
    logger.log(f"request_timeout: {args.request_timeout}s")
    logger.log(f"output_dir: {run_dir}")
    if ebm_cfg:
        logger.log(f"ebm_bridge_mode: {args.ebm_bridge_mode}")
        logger.log(f"ebm_flush: {args.ebm_flush}")
        logger.log(f"ebm_base_url: {ebm_cfg.base_url}")
        logger.log(f"ebm_db_path: {ebm_cfg.db_path}")
    else:
        logger.log("ebm_bridge: disabled")
    logger.log("")

    totals = {
        "files_total": len(plans),
        "files_succeeded": 0,
        "files_failed": 0,
        "messages_total": 0,
        "messages_sent": 0,
        "messages_failed": 0,
    }
    session_results: list[dict[str, Any]] = []

    for index, plan in enumerate(plans, start=1):
        totals["messages_total"] += len(plan.send_inputs)
        source_file = str(plan.source_file.resolve())
        logger.log(f"[{index}/{len(plans)}] {plan.source_file.name}")
        logger.log(f"  user={plan.user}")
        logger.log(f"  session_key={plan.session_key}")
        logger.log(f"  session_id={plan.session_id}")
        logger.log(f"  transcript_path={plan.transcript_path}")
        logger.log(f"  message_count={len(plan.messages)} send_count={len(plan.send_inputs)}")

        session_errors = list(plan.errors)
        request_summaries: list[dict[str, Any]] = []
        session_status = "dry_run" if args.dry_run else "success"
        ebm_result: dict[str, Any] | None = None
        ebm_db_stats: dict[str, int] | None = None

        if not plan.send_inputs:
            session_status = "failed"
            if not session_errors:
                session_errors.append("no valid message records to send")

        if args.dry_run:
            for send_index, _send_payload in enumerate(plan.send_inputs, start=1):
                logger.log(f"  dry-run send {send_index}/{len(plan.send_inputs)}")
        else:
            try:
                response_json = sender(
                    session_key=plan.session_key,
                    session_id=plan.session_id,
                    transcript_path=plan.transcript_path,
                    messages=plan.send_inputs,
                )
                totals["messages_sent"] += len(plan.send_inputs)
                request_summaries.append(
                    {
                        "index": 1,
                        "reply_preview": "",
                        "appended": response_json.get("appended", []),
                    }
                )
                logger.log(
                    f"  send 1/{1} ok: appended {len(response_json.get('appended', []))} transcript messages"
                )
                if ebm_cfg:
                    ebm_result = bridge_session_to_ebm(
                        plan,
                        ebm_cfg,
                        args.mode,
                        args.ebm_bridge_mode,
                        args.request_timeout,
                    )
                    bootstrap_imported = (
                        ebm_result["bootstrap"].get("importedMessages", 0)
                        if isinstance(ebm_result.get("bootstrap"), dict)
                        else 0
                    )
                    logger.log(
                        "  ebm bridge ok: "
                        f"mode={ebm_result['bridge_mode']} "
                        f"bootstrap_imported={bootstrap_imported} "
                        f"after_turn_messages={ebm_result['messages_sent']}"
                    )
            except (json.JSONDecodeError, OSError) as exc:
                totals["messages_failed"] += len(plan.send_inputs)
                session_status = "failed"
                error_text = f"send 1/1 failed: {exc}"
                session_errors.append(error_text)
                logger.log(f"  {error_text}")
                if args.stop_on_error:
                    raise IngestRunError(error_text)

        if not args.dry_run and session_status == "success" and session_errors:
            session_status = "partial_failure"

        if args.dry_run and session_errors:
            session_status = "failed"

        if session_status in {"success", "dry_run"}:
            totals["files_succeeded"] += 1
        else:
            totals["files_failed"] += 1

        session_results.append(
            {
                "source_file": source_file,
                "file_stem": plan.file_stem,
                "user": plan.user,
                "session_key": plan.session_key,
                "session_id": plan.session_id,
                "transcript_path": str(plan.transcript_path),
                "message_count": len(plan.messages),
                "send_count": len(plan.send_inputs),
                "first_timestamp": plan.first_timestamp,
                "last_timestamp": plan.last_timestamp,
                "status": session_status,
                "errors": session_errors,
                "requests": request_summaries,
                "ebm": ebm_result,
                "ebm_db_stats": ebm_db_stats,
            }
        )
        logger.log("")

    if ebm_cfg and not args.dry_run and args.ebm_flush:
        try:
            flush_result = flush_ebm(ebm_cfg, args.request_timeout)
            logger.log(f"ebm flush ok: {json.dumps(flush_result, ensure_ascii=False)}")
            for session_result, plan in zip(session_results, plans, strict=False):
                session_result["ebm_db_stats"] = query_ebm_db_stats(ebm_cfg.db_path, plan)
        except Exception as exc:  # noqa: BLE001
            logger.log(f"ebm flush failed: {exc}")

    finished_at = datetime.now().astimezone().isoformat()
    manifest = build_manifest(
        dataset_dir=dataset_dir,
        session_layout=args.session_layout,
        mode=args.mode,
        started_at=started_at,
        finished_at=finished_at,
        totals=totals,
        sessions=session_results,
        dry_run=args.dry_run,
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.log("=== Summary ===")
    logger.log(
        "files: total={files_total} succeeded={files_succeeded} failed={files_failed}".format(**totals)
    )
    logger.log(
        "messages: total={messages_total} sent={messages_sent} failed={messages_failed}".format(**totals)
    )
    logger.write_to(run_dir / "ingest.log")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest JSONL chat sessions into OpenClaw.")
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help=f"JSONL dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--base-url",
        default=OPENCLAW_BASE_URL,
        help=f"Unused legacy option kept for CLI compatibility (default: {OPENCLAW_BASE_URL})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("OPENCLAW_GATEWAY_TOKEN", ""),
        help="Unused legacy option kept for CLI compatibility (default: OPENCLAW_GATEWAY_TOKEN)",
    )
    parser.add_argument(
        "--session-layout",
        choices=["all-in-one", "per-file"],
        default=DEFAULT_SESSION_LAYOUT,
        help="Session layout strategy (default: per-file)",
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "bundle"],
        default=DEFAULT_MODE,
        help="Send mode (default: replay)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Run output directory root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help=f"File glob inside dataset dir (default: {DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Only ingest the first N sorted files.",
    )
    parser.add_argument(
        "--user-prefix",
        default=None,
        help="Override the default user prefix.",
    )
    parser.add_argument(
        "--session-prefix",
        default=None,
        help="Override the default session key prefix.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=False,
        help="Stop the run on the first parsing or request error.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse and plan the ingest without sending requests.",
    )
    parser.add_argument(
        "--skip-ebm-bridge",
        action="store_true",
        default=False,
        help="Only write OpenClaw transcript files; do not bridge into ebm-context-engine.",
    )
    parser.add_argument(
        "--ebm-bridge-mode",
        choices=["bootstrap", "after-turn", "both", "off"],
        default=DEFAULT_EBM_BRIDGE,
        help=(
            "How to bridge transcript data into ebm-context-engine "
            "(default: bootstrap). "
            "'bootstrap' only imports transcript rows; "
            "'after-turn' only enqueues slow-path extraction; "
            "'both' does both; 'off' skips EBM HTTP calls."
        ),
    )
    parser.add_argument(
        "--ebm-flush",
        action="store_true",
        default=False,
        help="Flush the EBM slow-path queue after ingest. Off by default to avoid blocking OpenClaw.",
    )
    args = parser.parse_args(argv)

    if args.limit_files is not None and args.limit_files <= 0:
        parser.error("--limit-files must be greater than 0")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than 0")
    if args.skip_ebm_bridge:
        args.ebm_bridge_mode = "off"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = run_ingest(args)
    except (FileNotFoundError, IngestRunError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_status = "dry-run" if manifest.get("dry_run") else "completed"
    print(
        f"{run_status}: files={manifest['totals']['files_total']} "
        f"messages={manifest['totals']['messages_total']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
