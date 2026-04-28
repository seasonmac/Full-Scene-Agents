from __future__ import annotations

import unittest

import numpy as np

from ebm_context_engine.hypergraph.embedding import propagate_embeddings
from ebm_context_engine.types import HmEpisode, HmFact, HmTopic


class HypergraphEmbeddingTests(unittest.TestCase):
    def test_propagation_handles_mixed_embedding_dimensions(self) -> None:
        topic = HmTopic(
            id="topic-1",
            title="topic",
            summary="summary",
            episode_ids=["episode-1"],
            vector=np.ones(1024, dtype=np.float32),
        )
        episode = HmEpisode(
            id="episode-1",
            session_key="session-1",
            title="episode",
            summary="summary",
            dialogue="dialogue",
            topic_ids=["topic-1"],
            fact_ids=["fact-1"],
            vector=np.ones(1024, dtype=np.float32),
        )
        fact = HmFact(
            id="fact-1",
            content="content",
            potential="potential",
            episode_id="episode-1",
            session_key="session-1",
            vector=np.ones(2560, dtype=np.float32),
        )

        propagate_embeddings([topic], [episode], [fact])

        self.assertEqual(topic.vector.shape, (1024,))
        self.assertEqual(episode.vector.shape, (1024,))
        self.assertEqual(fact.vector.shape, (1024,))
        self.assertAlmostEqual(float(np.linalg.norm(topic.vector)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(episode.vector)), 1.0, places=5)
        self.assertAlmostEqual(float(np.linalg.norm(fact.vector)), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
