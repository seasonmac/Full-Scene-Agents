#!/usr/bin/env python3
"""Dump EBM ingest coverage for injectDataset2Openclaw datasets as Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_DB_PATH = Path.home() / ".openclaw" / "memory" / "ebm_context_engine.sqlite"
DEFAULT_GLOB = "*.jsonl"
DEFAULT_SESSION_PREFIX = "agent:main:news/dataset"


def sanitize_key_part(value: str, fallback: str) -> str:
    chars = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            chars.append(ch)
        else:
            chars.append("-")
    cleaned = "".join(chars).strip("-")
    return cleaned or fallback


def derive_session_id(session_key: str) -> str:
    digest = hashlib.sha1(session_key.encode("utf-8")).hexdigest()
    return f"ingest-{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def count_source_messages(path: Path) -> int:
    count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        if payload.get("type") != "message":
            continue
        message = payload.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str) and isinstance(content, str) and content.strip():
            count += 1
    return count


def query_scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] if row else 0)


def db_counts_for_session(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    session_key: str,
    transcript_path: Path,
) -> dict[str, int]:
    transcript_file = str(transcript_path)
    transcripts = query_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM transcript_entries
        WHERE session_id = ?
           OR session_key = ?
           OR session_file = ?
        """,
        (session_id, session_key, transcript_file),
    )
    summaries = query_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM session_summaries
        WHERE session_id = ?
           OR session_key = ?
           OR session_file = ?
        """,
        (session_id, session_key, transcript_file),
    )
    facts = query_scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM facts
        WHERE session_key = ?
           OR json_extract(evidence_json, '$.sessionFile') = ?
        """,
        (session_key, transcript_file),
    )
    return {
        "transcripts": transcripts,
        "summaries": summaries,
        "facts": facts,
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| 文件 | 源消息数 | DB transcript | summaries | facts |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {file} | {source_messages} | {transcripts} | {summaries} | {facts} |".format(**row)
        )
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> str:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()
    source_files = sorted(dataset_dir.glob(args.glob))

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
    if not db_path.exists():
        raise FileNotFoundError(f"EBM database not found: {db_path}")

    rows: list[dict[str, Any]] = []
    totals = {
        "source_messages": 0,
        "transcripts": 0,
        "summaries": 0,
        "facts": 0,
    }

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for source_file in source_files:
            file_stem = sanitize_key_part(source_file.stem, "session")
            session_key = f"{args.session_prefix.rstrip('/')}/{file_stem}"
            session_id = derive_session_id(session_key)
            transcript_path = (
                Path.home() / ".openclaw" / "agents" / "main" / "sessions" / f"{session_id}.jsonl"
            ).resolve()
            source_messages = count_source_messages(source_file)
            counts = db_counts_for_session(
                conn,
                session_id=session_id,
                session_key=session_key,
                transcript_path=transcript_path,
            )
            row = {
                "file": source_file.name,
                "source_messages": source_messages,
                **counts,
            }
            rows.append(row)
            totals["source_messages"] += source_messages
            totals["transcripts"] += counts["transcripts"]
            totals["summaries"] += counts["summaries"]
            totals["facts"] += counts["facts"]
    finally:
        conn.close()

    lines = [
        "# EBM Dataset Coverage",
        "",
        markdown_table(rows),
        "",
        "## 整体状态",
        "",
        f"dataset 源消息总数: {totals['source_messages']}",
        f"DB 中 {args.session_prefix.rstrip('/')}/* transcript 总数: {totals['transcripts']}",
        f"session_summaries: {totals['summaries']}",
        f"facts: {totals['facts']}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump EBM DB coverage for imported JSONL sessions.")
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help=f"Dataset directory containing JSONL files (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"EBM SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help=f"File glob inside dataset dir (default: {DEFAULT_GLOB})",
    )
    parser.add_argument(
        "--session-prefix",
        default=DEFAULT_SESSION_PREFIX,
        help=f"Session key prefix used during ingest (default: {DEFAULT_SESSION_PREFIX})",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional file path to write the Markdown report. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
