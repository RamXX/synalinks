# # License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

# from typing import Literal
# from typing import Union
# from unittest.mock import patch

# import numpy as np

# from synalinks.src import testing
# from synalinks.src.backend import Entity
# from synalinks.src.backend import Relation
# from synalinks.src.backend import SimilaritySearch
# from synalinks.src.embedding_models import EmbeddingModel
# from synalinks.src.knowledge_bases.database_adapters.kuzu_adapter import KuzuAdapter
# from synalinks.src.modules import Embedding
# from synalinks.src.modules import Input
# from synalinks.src.programs import Program


# class Document(Entity):
#     label: Literal["Document"]
#     text: str


# class Chunk(Document):
#     label: Literal["Chunk"]
#     text: str


# class IsPartOf(Relation):
#     obj: Union[Chunk, Document]
#     label: Literal["IsPartOf"]
#     subj: Document


# class KuzuAdapterTest(testing.TestCase):

#     @patch("litellm.aembedding")
#     async def test_adapter(self, mock_embedding):
#         expected_value = np.random.rand(1024)
#         mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

#         embedding_model = EmbeddingModel(
#             model="ollama/mxbai-embed-large",
#         )

#         adapter = KuzuAdapter(
#             uri="kuzu:///tmp/test",
#             embedding_model=embedding_model,
#             entity_models=[Document, Chunk],
#             relation_models=[IsPartOf],
#             wipe_on_start=True,
#         )

#         _ = await adapter.query("RETURN 1")

#         doc1 = Document(
#             label="Document",
#             text="test document 1",
#         )

#         inputs = Input(data_model=Document)
#         outputs = await Embedding(
#             embedding_model=embedding_model,
#             in_mask=["text"],
#         )(inputs)

#         program = Program(
#             inputs=inputs,
#             outputs=outputs,
#             name="embedd_data_program"
#         )

#         embedded_doc = await program(doc1)

#         await adapter.update(embedded_doc)

#         doc1 = Document(
#             label="Document",
#             text="test document 1",
#         )
#         doc2 = Document(
#             label="Document",
#             text="test document 2",
#         )

#         rel1 = IsPartOf(
#             subj=doc1,
#             label="IsPartOf",
#             obj=doc2,
#         )

#         embedded_rel = await program(rel1)

#         await adapter.update(embedded_rel)

#         doc1 = Document(
#             label="Document",
#             text="test document 1",
#         )
#         doc2 = Document(
#             label="Document",
#             text="test document 2",
#         )

#         rel1 = IsPartOf(
#             subj=doc1,
#             label="IsPartOf",
#             obj=doc2,
#         )

#         await adapter.update(rel1.to_json_data_model())

#         doc1 = Document(
#             label="Document",
#             text="test document 1",
#         )
#         doc2 = Document(
#             label="Document",
#             text="test document 2",
#         )
#         doc3 = Document(
#             label="Document",
#             text="test document 3",
#         )

#         batch = np.array([doc1, doc2, doc3], dtype=object)

#         embedded_docs = await program.predict(batch)

#         await adapter.update(embedded_docs[0])
#         await adapter.update(embedded_docs[1])
#         await adapter.update(embedded_docs[2])

#         search = SimilaritySearch(
#             entity_label="Document",
#             similarity_search="test document",
#         ).to_json_data_model()
#         result = await adapter.similarity_search(search, threshold=0.0)
#         self.assertTrue(len(result) > 0)
