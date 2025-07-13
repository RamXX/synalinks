# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import Literal
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import Entity
from synalinks.src.backend import Relation
from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.knowledge_bases import KnowledgeBase


class Document(Entity):
    label: Literal["Document"]
    text: str


class Chunk(Entity):
    label: Literal["Chunk"]
    text: str


class IsPartOf(Relation):
    subj: Chunk
    label: Literal["IsPartOf"]
    obj: Document


class KnowledgeBaseTest(testing.TestCase):
    @patch("litellm.aembedding")
    async def test_knowledge_base(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        knowledge_base = KnowledgeBase(
            uri="neo4j://localhost:7687",
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            embedding_model=embedding_model,
            metric="cosine",
            wipe_on_start=False,
        )

        _ = await knowledge_base.query("RETURN 1")

    @patch("litellm.aembedding")
    def test_knowledge_base_serialization(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        knowledge_base = KnowledgeBase(
            uri="neo4j://localhost:7687",
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            embedding_model=embedding_model,
            metric="cosine",
            wipe_on_start=False,
        )

        config = knowledge_base.get_config()
        cloned_knowledge_base = KnowledgeBase.from_config(config)
        self.assertEqual(
            cloned_knowledge_base.get_config(),
            knowledge_base.get_config(),
        )
