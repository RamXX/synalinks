# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Literal
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import Entities
from synalinks.src.backend import Entity
from synalinks.src.backend import KnowledgeGraph
from synalinks.src.backend import Relation
from synalinks.src.backend import Relations
from synalinks.src.backend import is_embedded_entity
from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.modules import Input
from synalinks.src.modules.knowledge.embedding import Embedding
from synalinks.src.programs import Program


class Document(Entity):
    label: Literal["Document"]
    text: str


class Documents(Entities):
    entities: List[Document]


class IsPartOf(Relation):
    subj: Document
    label: Literal["IsPartOf"]
    obj: Document


class DocumentRelations(Relations):
    relations: List[IsPartOf]


class DocumentGraph(KnowledgeGraph):
    entities: List[Document]
    relations: List[IsPartOf]


class EmbeddingTest(testing.TestCase):
    @patch("litellm.aembedding")
    async def test_embedding_single_entity(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        i0 = Input(data_model=Document)
        x0 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
            name="test_embedding",
            description="test_embedding",
        )

        input_doc = Document(
            label="Document",
            text="test document",
        )

        result = await program(input_doc)
        self.assertTrue(len(result.get("embedding")) > 0)
        self.assertTrue(is_embedded_entity(result))

    @patch("litellm.aembedding")
    async def test_embedding_single_relation(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        i0 = Input(data_model=IsPartOf)
        x0 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
            name="test_embedding",
            description="test_embedding",
        )

        rel = IsPartOf(
            subj=Document(
                label="Document",
                text="test document 1",
            ),
            label="IsPartOf",
            obj=Document(
                label="Document",
                text="test document 2",
            ),
        )

        result = await program(rel)
        self.assertTrue(len(result.get("subj").get("embedding")) > 0)
        self.assertTrue(len(result.get("obj").get("embedding")) > 0)
        subj = result.get_nested_entity("subj")
        self.assertTrue(is_embedded_entity(subj))
        obj = result.get_nested_entity("obj")
        self.assertTrue(is_embedded_entity(obj))

    @patch("litellm.aembedding")
    async def test_embedding_entities(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        i0 = Input(data_model=Documents)

        x0 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
            name="test_embedding",
            description="test_embedding",
        )

        inputs = Documents(
            entities=[
                Document(
                    label="Document",
                    text="test document 1",
                ),
                Document(
                    label="Document",
                    text="test document 2",
                ),
            ]
        )

        result = await program(inputs)
        self.assertTrue(len(result.get_json().get("entities")[0].get("embedding")) > 0)
        self.assertTrue(len(result.get_json().get("entities")[1].get("embedding")) > 0)
        for entity in result.get_nested_entity_list("entities"):
            self.assertTrue(is_embedded_entity(entity))

    @patch("litellm.aembedding")
    async def test_embedding_relations(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        i0 = Input(data_model=DocumentRelations)

        x0 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
            name="test_embedding",
            description="test_embedding",
        )

        rel1 = IsPartOf(
            subj=Document(
                label="Document",
                text="test document 1",
            ),
            label="IsPartOf",
            obj=Document(
                label="Document",
                text="test document 2",
            ),
        )

        rel2 = IsPartOf(
            subj=Document(
                label="Document",
                text="test document 3",
            ),
            label="IsPartOf",
            obj=Document(
                label="Document",
                text="test document 4",
            ),
        )

        inputs = DocumentRelations(relations=[rel1, rel2])

        result = await program(inputs)
        self.assertTrue(
            len(result.get_json().get("relations")[0].get("subj").get("embedding")) > 0
        )
        self.assertTrue(
            len(result.get_json().get("relations")[0].get("obj").get("embedding")) > 0
        )
        self.assertTrue(
            len(result.get_json().get("relations")[1].get("subj").get("embedding")) > 0
        )
        self.assertTrue(
            len(result.get_json().get("relations")[1].get("obj").get("embedding")) > 0
        )
        for relation in result.get_nested_entity_list("relations"):
            subj = relation.get_nested_entity("subj")
            obj = relation.get_nested_entity("obj")
            self.assertTrue(is_embedded_entity(subj))
            self.assertTrue(is_embedded_entity(obj))

    @patch("litellm.aembedding")
    async def test_embedding_knowledge_graph(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        i0 = Input(data_model=Documents)

        x0 = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
            name="test_embedding",
            description="test_embedding",
        )

        rel1 = IsPartOf(
            subj=Document(
                label="Document",
                text="test document 1",
            ),
            label="IsPartOf",
            obj=Document(
                label="Document",
                text="test document 2",
            ),
        )

        rel2 = IsPartOf(
            subj=Document(
                label="Document",
                text="test document 3",
            ),
            label="IsPartOf",
            obj=Document(
                label="Document",
                text="test document 4",
            ),
        )

        inputs = DocumentGraph(
            entities=[
                Document(
                    label="Document",
                    text="test document 1",
                ),
                Document(
                    label="Document",
                    text="test document 2",
                ),
            ],
            relations=[rel1, rel2],
        )

        result = await program(inputs)
        self.assertTrue(len(result.get_json().get("entities")[0].get("embedding")) > 0)
        self.assertTrue(len(result.get_json().get("entities")[1].get("embedding")) > 0)
        self.assertTrue(
            len(result.get_json().get("relations")[0].get("subj").get("embedding")) > 0
        )
        self.assertTrue(
            len(result.get_json().get("relations")[0].get("obj").get("embedding")) > 0
        )
        self.assertTrue(
            len(result.get_json().get("relations")[1].get("subj").get("embedding")) > 0
        )
        self.assertTrue(
            len(result.get_json().get("relations")[1].get("obj").get("embedding")) > 0
        )
        for entity in result.get_nested_entity_list("entities"):
            self.assertTrue(is_embedded_entity(entity))
        for relation in result.get_nested_entity_list("relations"):
            subj = relation.get_nested_entity("subj")
            obj = relation.get_nested_entity("obj")
            self.assertTrue(is_embedded_entity(subj))
            self.assertTrue(is_embedded_entity(obj))
