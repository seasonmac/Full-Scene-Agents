from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ebm_context_engine.engine import EvidenceBackedMemoryEngine
from ebm_context_engine.server import _build_engine


class OpenClawPluginContractTests(unittest.TestCase):
    def test_compact_returns_structured_stub(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ebm.sqlite")
            engine = _build_engine(db_path)
            result = engine.compact(
                {
                    "sessionId": "session-1",
                    "sessionKey": "agent:main:main",
                    "sessionFile": "/tmp/session.jsonl",
                    "currentTokenCount": 321,
                    "runtimeContext": {"source": "test"},
                }
            )
            self.assertTrue(result["ok"])
            self.assertFalse(result["compacted"])
            self.assertEqual(result["result"]["tokensBefore"], 321)
            self.assertTrue(result["result"]["details"]["runtimeContextPresent"])

    def test_build_engine_applies_slow_path_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ebm.sqlite")
            engine = _build_engine(db_path, slow_path_enabled_override=False)
            self.assertFalse(engine.config.slowPathEnabled)

    def test_build_engine_does_not_backfill_transcript_vectors_during_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ebm.sqlite")
            session_file = Path(tmpdir) / "session.jsonl"
            session_file.write_text("", encoding="utf-8")

            seed = EvidenceBackedMemoryEngine(db_path)
            seed.bootstrap(
                {
                    "sessionId": "session-1",
                    "sessionKey": "agent:main:main",
                    "sessionFile": str(session_file),
                }
            )
            seed.ingest(
                {
                    "sessionId": "session-1",
                    "sessionKey": "agent:main:main",
                    "message": {"role": "user", "content": "Remember that I prefer jasmine tea."},
                }
            )
            seed.close()

            with patch.object(
                EvidenceBackedMemoryEngine,
                "_backfill_transcript_vectors",
                autospec=True,
            ) as backfill:
                engine = _build_engine(db_path)
                try:
                    self.assertTrue(engine._require_state().transcripts)
                    backfill.assert_not_called()
                finally:
                    engine.close()

    def test_transcript_recall_backfills_vectors_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ebm.sqlite")
            session_file = Path(tmpdir) / "session.jsonl"
            session_file.write_text("", encoding="utf-8")

            seed = EvidenceBackedMemoryEngine(db_path)
            seed.bootstrap(
                {
                    "sessionId": "session-1",
                    "sessionKey": "agent:main:main",
                    "sessionFile": str(session_file),
                }
            )
            seed.ingest(
                {
                    "sessionId": "session-1",
                    "sessionKey": "agent:main:main",
                    "message": {"role": "user", "content": "Remember that I prefer jasmine tea."},
                }
            )
            seed.close()

            engine = _build_engine(db_path)
            try:
                self.assertIsNone(engine._require_state().transcripts[0].vector)
                with patch.object(engine, "_backfill_transcript_vectors", autospec=True) as backfill:
                    context = engine._recall_transcript_context(
                        "What tea do I prefer?",
                        np.zeros(256, dtype=np.float32),
                        limit=1,
                    )
                    backfill.assert_called_once_with()
                self.assertIn("jasmine tea", context)
            finally:
                engine.close()

    def test_flush_schedules_background_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ebm.sqlite")
            engine = _build_engine(db_path)
            started = threading.Event()
            release = threading.Event()

            def fake_flush() -> None:
                started.set()
                release.wait(timeout=2)

            with patch.object(engine, "_flush_slow_path_foreground", side_effect=fake_flush) as flush:
                status = engine.flush_slow_path()
                self.assertTrue(started.wait(timeout=1))
                self.assertGreaterEqual(int(status.get("running", 0) or 0), 1)
                self.assertTrue(engine._runtime_lock.acquire(timeout=0.2))
                engine._runtime_lock.release()
                engine.flush_slow_path()
                self.assertEqual(flush.call_count, 1)
                release.set()
                for _ in range(50):
                    if not engine.get_slow_path_status_detailed()["flush_active"]:
                        break
                    threading.Event().wait(0.02)
                self.assertFalse(engine.get_slow_path_status_detailed()["flush_active"])
            engine.close()

    def test_session_key_and_file_passthrough_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ebm.sqlite")
            session_file = Path(tmpdir) / "session.jsonl"
            session_file.write_text("", encoding="utf-8")
            engine = _build_engine(db_path)

            bootstrap = engine.bootstrap(
                {
                    "sessionId": "session-1",
                    "sessionKey": "agent:main:main",
                    "sessionFile": str(session_file),
                }
            )
            self.assertTrue(bootstrap["bootstrapped"])

            assemble = engine.assemble(
                {
                    "sessionId": "session-1",
                    "sessionKey": "agent:main:main",
                    "sessionFile": str(session_file),
                    "messages": [{"role": "user", "content": "remember that I like tea"}],
                    "tokenBudget": 2048,
                    "runtimeContext": {"source": "test"},
                }
            )
            self.assertIn("estimatedTokens", assemble)

    def test_memory_get_forget_and_archive_expand(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "ebm.sqlite")
            engine = _build_engine(db_path)

            ingest = engine.ingest(
                {
                    "sessionId": "session-1",
                    "sessionKey": "agent:main:session-1",
                    "message": {"role": "user", "content": "Remember that I prefer jasmine tea."},
                }
            )
            self.assertTrue(ingest["ingested"])

            expanded = engine.archive_expand("session-1", session_id="session-1")
            self.assertEqual(expanded["source"], "transcript")
            self.assertEqual(len(expanded["messages"]), 1)
            self.assertIn("jasmine tea", expanded["messages"][0]["text"])

            transcript_id = engine._require_state().transcripts[0].id
            item = engine.memory_get(transcript_id)
            self.assertIsNotNone(item)
            self.assertEqual(item["node_type"], "TRANSCRIPT")

            unsupported = engine.memory_forget(transcript_id)
            self.assertFalse(unsupported["forgotten"])
            self.assertEqual(unsupported["reason"], "unsupported_type")


if __name__ == "__main__":
    unittest.main()
