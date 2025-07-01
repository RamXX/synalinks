# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import Embeddings
from synalinks.src.embedding_models.embedding_model import EmbeddingModel


class EmbeddingModelTest(testing.TestCase):
    @patch("litellm.aembedding")
    async def test_call_api(self, mock_embedding):
        embedding_model = EmbeddingModel(model="ollama/all-minilm")

        expected_value = [0.0, 0.1, 0.2, 0.3]
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        result = await embedding_model(["What is the capital of France?"])
        self.assertEqual(result, Embeddings(**result).get_json())
        self.assertEqual(result, {"embeddings": [expected_value]})
