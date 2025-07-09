# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import Literal
from typing import Union
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import Entity
from synalinks.src.backend import Relation
from synalinks.src.backend import SimilaritySearch
from synalinks.src.embedding_models import EmbeddingModel
from synalinks.src.knowledge_bases.database_adapters.memgraph_adapter import (
    MemGraphAdapter,
)
from synalinks.src.modules import Embedding
from synalinks.src.modules import Input
from synalinks.src.programs import Program


class Document(Entity):
    label: Literal["Document"]
    text: str


class Chunk(Document):
    label: Literal["Chunk"]
    text: str


class IsPartOf(Relation):
    obj: Union[Chunk, Document]
    label: Literal["IsPartOf"]
    subj: Document


class MemGraphAdapterTest(testing.TestCase):
    @patch("litellm.aembedding")
    async def test_adapter(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        adapter = MemGraphAdapter(
            uri="memgraph://localhost:7688",
            embedding_model=embedding_model,
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            wipe_on_start=True,
        )

        _ = await adapter.query("RETURN 1")

    @patch("litellm.aembedding")
    async def test_adapter_update_entity(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")

        adapter = MemGraphAdapter(
            uri="memgraph://localhost:7688",
            embedding_model=embedding_model,
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            wipe_on_start=True,
        )

        doc1 = Document(
            label="Document",
            text="test document 1",
        )

        inputs = Input(data_model=Document)
        outputs = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        embedded_doc = await program(doc1)

        await adapter.update(embedded_doc)

    @patch("litellm.aembedding")
    async def test_adapter_update_relation(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(
            model="ollama/mxbai-embed-large",
        )

        adapter = MemGraphAdapter(
            uri="memgraph://localhost:7688",
            embedding_model=embedding_model,
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            wipe_on_start=True,
        )

        doc1 = Document(
            label="Document",
            text="test document 1",
        )
        doc2 = Document(
            label="Document",
            text="test document 2",
        )

        rel1 = IsPartOf(
            subj=doc1,
            label="IsPartOf",
            obj=doc2,
        )

        inputs = Input(data_model=Document)
        outputs = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        embedded_rel = await program(rel1)

        await adapter.update(embedded_rel)

    @patch("litellm.aembedding")
    async def test_adapter_similarity_search(self, mock_embedding):
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")

        adapter = MemGraphAdapter(
            uri="memgraph://localhost:7688",
            embedding_model=embedding_model,
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            wipe_on_start=True,
        )

        inputs = Input(data_model=Document)

        outputs = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        doc1 = Document(
            label="Document",
            text="test document 1",
        )
        doc2 = Document(
            label="Document",
            text="test document 2",
        )
        doc3 = Document(
            label="Document",
            text="test document 3",
        )

        batch = np.array([doc1, doc2, doc3], dtype=object)

        embedded_docs = await program.predict(batch)

        await adapter.update(embedded_docs[0])
        await adapter.update(embedded_docs[1])
        await adapter.update(embedded_docs[2])

        search = SimilaritySearch(
            entity_label="Document",
            similarity_search="test document",
        ).to_json_data_model()
        result = await adapter.similarity_search(search, threshold=0.0)
        self.assertTrue(len(result) > 0)

    @patch("litellm.aembedding")
    async def test_adapter_triplet_search_basic(self, mock_embedding):
        """Test basic triplet search functionality"""
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")

        adapter = MemGraphAdapter(
            uri="memgraph://localhost:7688",
            embedding_model=embedding_model,
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            wipe_on_start=True,
        )

        # Create test documents
        doc1 = Document(label="Document", text="test document 1")
        doc2 = Document(label="Document", text="test document 2")
        chunk1 = Chunk(label="Chunk", text="test chunk 1")

        # Create embeddings
        inputs = Input(data_model=Document)
        outputs = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        # Embed and store entities
        embedded_doc1 = await program(doc1)
        embedded_doc2 = await program(doc2)
        embedded_chunk1 = await program(chunk1)

        await adapter.update(embedded_doc1)
        await adapter.update(embedded_doc2)
        await adapter.update(embedded_chunk1)

        # Create and store relation
        rel1 = IsPartOf(subj=chunk1, label="IsPartOf", obj=doc1)
        embedded_rel = await program(rel1)
        await adapter.update(embedded_rel)

        # Test triplet search
        from synalinks.src.backend import TripletSearch

        triplet_search = TripletSearch(
            subject_label="Chunk",
            subject_similarity_search="",
            relation_label="IsPartOf",
            object_label="Document",
            object_similarity_search="",
        ).to_json_data_model()

        result = await adapter.triplet_search(triplet_search, threshold=0.0)
        self.assertTrue(len(result) > 0)
        self.assertIn("subj", result[0])
        self.assertIn("relation", result[0])
        self.assertIn("obj", result[0])

    @patch("litellm.aembedding")
    async def test_adapter_triplet_search_with_similarity(self, mock_embedding):
        """Test triplet search with similarity queries"""
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")

        adapter = MemGraphAdapter(
            uri="memgraph://localhost:7688",
            embedding_model=embedding_model,
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            wipe_on_start=True,
        )

        # Create and embed test data
        doc1 = Document(label="Document", text="artificial intelligence document")
        chunk1 = Chunk(label="Chunk", text="machine learning chapter")

        inputs = Input(data_model=Document)
        outputs = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        embedded_doc1 = await program(doc1)
        embedded_chunk1 = await program(chunk1)

        await adapter.update(embedded_doc1)
        await adapter.update(embedded_chunk1)

        # Create relation
        rel1 = IsPartOf(subj=chunk1, label="IsPartOf", obj=doc1)
        embedded_rel = await program(rel1)
        await adapter.update(embedded_rel)

        # Test triplet search with subject similarity
        from synalinks.src.backend import TripletSearch

        triplet_search = TripletSearch(
            subject_label="Chunk",
            subject_similarity_search="machine learning",
            relation_label="IsPartOf",
            object_label="Document",
            object_similarity_search="?",
        ).to_json_data_model()

        result = await adapter.triplet_search(triplet_search, threshold=0.0)
        self.assertTrue(len(result) > 0)

    @patch("litellm.aembedding")
    async def test_adapter_triplet_search_with_object_similarity(self, mock_embedding):
        """Test triplet search with object similarity query"""
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")

        adapter = MemGraphAdapter(
            uri="memgraph://localhost:7688",
            embedding_model=embedding_model,
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            wipe_on_start=True,
        )

        # Create and embed test data
        doc1 = Document(label="Document", text="programming tutorial")
        chunk1 = Chunk(label="Chunk", text="python basics")

        inputs = Input(data_model=Document)
        outputs = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        embedded_doc1 = await program(doc1)
        embedded_chunk1 = await program(chunk1)

        await adapter.update(embedded_doc1)
        await adapter.update(embedded_chunk1)

        # Create relation
        rel1 = IsPartOf(subj=chunk1, label="IsPartOf", obj=doc1)
        embedded_rel = await program(rel1)
        await adapter.update(embedded_rel)

        # Test triplet search with object similarity
        from synalinks.src.backend import TripletSearch

        triplet_search = TripletSearch(
            subject_label="Chunk",
            subject_similarity_search="?",
            relation_label="IsPartOf",
            object_label="Document",
            object_similarity_search="programming tutorial",
        ).to_json_data_model()

        result = await adapter.triplet_search(triplet_search, threshold=0.0)
        self.assertTrue(len(result) > 0)

    @patch("litellm.aembedding")
    async def test_adapter_triplet_search_with_both_similarities(self, mock_embedding):
        """Test triplet search with both subject and object similarity queries"""
        expected_value = np.random.rand(1024)
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")

        adapter = MemGraphAdapter(
            uri="memgraph://localhost:7688",
            embedding_model=embedding_model,
            entity_models=[Document, Chunk],
            relation_models=[IsPartOf],
            wipe_on_start=True,
        )

        # Create test data
        doc1 = Document(label="Document", text="data science handbook")
        doc2 = Document(label="Document", text="machine learning guide")
        chunk1 = Chunk(label="Chunk", text="statistics chapter")
        chunk2 = Chunk(label="Chunk", text="neural networks section")

        inputs = Input(data_model=Document)
        outputs = await Embedding(
            embedding_model=embedding_model,
            in_mask=["text"],
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        # Embed and store all entities
        entities = [doc1, doc2, chunk1, chunk2]
        for entity in entities:
            embedded_entity = await program(entity)
            await adapter.update(embedded_entity)

        # Create relations
        rel1 = IsPartOf(subj=chunk1, label="IsPartOf", obj=doc1)
        rel2 = IsPartOf(subj=chunk2, label="IsPartOf", obj=doc2)

        for rel in [rel1, rel2]:
            embedded_rel = await program(rel)
            await adapter.update(embedded_rel)

        # Test triplet search with both similarities
        from synalinks.src.backend import TripletSearch

        triplet_search = TripletSearch(
            subject_label="Chunk",
            subject_similarity_search="statistics",
            relation_label="IsPartOf",
            object_label="Document",
            object_similarity_search="data science",
        ).to_json_data_model()

        result = await adapter.triplet_search(triplet_search, threshold=0.0)
        self.assertTrue(len(result) > 0)
