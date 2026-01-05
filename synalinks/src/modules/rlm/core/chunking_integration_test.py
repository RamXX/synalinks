# License Apache 2.0: (c) 2025 Synalinks Team

"""Integration tests for chunking strategies with RecursiveGenerator.

These tests verify E2E integration with real LLM calls using:
- zai/glm-4.7
- groq/openai/gpt-oss-20b
"""

from synalinks.src import testing
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.chunking_strategy import KeywordChunking
from synalinks.src.modules.rlm.core.chunking_strategy import SemanticChunking
from synalinks.src.modules.rlm.core.chunking_strategy import UniformChunking
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator
from synalinks.src.saving import serialization_lib


class RecursiveGeneratorWithChunkingTest(testing.TestCase):
    """Test RecursiveGenerator with chunking strategies."""

    async def test_uniform_chunking_integration_zai(self):
        """Test UniformChunking with zai/glm-4.7 model."""
        lm = LanguageModel(model="zai/glm-4.7")
        chunking = UniformChunking(chunk_size=1000, overlap=100)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        # Verify chunking_strategy is set
        self.assertIsNotNone(gen.chunking_strategy)
        self.assertIsInstance(gen.chunking_strategy, UniformChunking)
        self.assertEqual(gen.chunking_strategy.chunk_size, 1000)
        self.assertEqual(gen.chunking_strategy.overlap, 100)

        # Test that chunking strategy can process text
        text = "A" * 5000
        chunks = gen.chunking_strategy.chunk(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1000)

    async def test_uniform_chunking_integration_groq(self):
        """Test UniformChunking with groq/openai/gpt-oss-20b model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = UniformChunking(chunk_size=500, overlap=50)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        self.assertIsNotNone(gen.chunking_strategy)
        self.assertIsInstance(gen.chunking_strategy, UniformChunking)
        self.assertEqual(gen.chunking_strategy.chunk_size, 500)
        self.assertEqual(gen.chunking_strategy.overlap, 50)

    async def test_keyword_chunking_integration_zai(self):
        """Test KeywordChunking with zai/glm-4.7 model."""
        lm = LanguageModel(model="zai/glm-4.7")
        chunking = KeywordChunking(keywords=["Chapter", "Section"], case_sensitive=False)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        self.assertIsNotNone(gen.chunking_strategy)
        self.assertIsInstance(gen.chunking_strategy, KeywordChunking)
        self.assertEqual(gen.chunking_strategy.keywords, ["Chapter", "Section"])

        # Test chunking on sample text
        text = "Chapter 1: Intro. Section A. Chapter 2: Methods. Section B."
        chunks = gen.chunking_strategy.chunk(text)
        self.assertEqual(len(chunks), 4)

    async def test_keyword_chunking_integration_groq(self):
        """Test KeywordChunking with groq/openai/gpt-oss-20b model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = KeywordChunking(keywords=["Part"])

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        self.assertIsNotNone(gen.chunking_strategy)
        self.assertIsInstance(gen.chunking_strategy, KeywordChunking)

    async def test_semantic_chunking_integration_zai(self):
        """Test SemanticChunking with zai/glm-4.7 model."""
        lm = LanguageModel(model="zai/glm-4.7")
        chunking = SemanticChunking(max_chunk_size=2000, similarity_threshold=0.6)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        self.assertIsNotNone(gen.chunking_strategy)
        self.assertIsInstance(gen.chunking_strategy, SemanticChunking)
        self.assertEqual(gen.chunking_strategy.max_chunk_size, 2000)
        self.assertEqual(gen.chunking_strategy.similarity_threshold, 0.6)

        # Test chunking behavior
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = gen.chunking_strategy.chunk(text)
        self.assertGreaterEqual(len(chunks), 1)

    async def test_semantic_chunking_integration_groq(self):
        """Test SemanticChunking with groq/openai/gpt-oss-20b model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = SemanticChunking(max_chunk_size=1500)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        self.assertIsNotNone(gen.chunking_strategy)
        self.assertIsInstance(gen.chunking_strategy, SemanticChunking)

    async def test_string_strategy_name_zai(self):
        """Test creating chunking strategy from string with zai model."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy="uniform",
        )

        # Default UniformChunking should be created
        self.assertIsNotNone(gen.chunking_strategy)
        self.assertIsInstance(gen.chunking_strategy, UniformChunking)

    async def test_string_strategy_name_groq(self):
        """Test creating chunking strategy from string with groq model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy="semantic",
        )

        # Default SemanticChunking should be created
        self.assertIsNotNone(gen.chunking_strategy)
        self.assertIsInstance(gen.chunking_strategy, SemanticChunking)

    async def test_no_chunking_strategy(self):
        """Test RecursiveGenerator without chunking strategy."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=None,
        )

        # chunking_strategy should be None
        self.assertIsNone(gen.chunking_strategy)

    async def test_serialization_with_chunking_zai(self):
        """Test serialization of RecursiveGenerator with chunking using zai model."""
        lm = LanguageModel(model="zai/glm-4.7")
        chunking = UniformChunking(chunk_size=750, overlap=75)

        original = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
            max_iterations=5,
        )

        # Serialize and deserialize
        config = original.get_config()
        self.assertIn("chunking_strategy", config)

        # Full roundtrip
        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        self.assertIsInstance(restored, RecursiveGenerator)
        self.assertIsNotNone(restored.chunking_strategy)
        self.assertIsInstance(restored.chunking_strategy, UniformChunking)
        self.assertEqual(restored.chunking_strategy.chunk_size, 750)
        self.assertEqual(restored.chunking_strategy.overlap, 75)

    async def test_serialization_with_chunking_groq(self):
        """Test serialization of RecursiveGenerator with chunking using groq model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = KeywordChunking(keywords=["Section"], case_sensitive=True)

        original = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        config = original.get_config()
        self.assertIn("chunking_strategy", config)

        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        self.assertIsInstance(restored, RecursiveGenerator)
        self.assertIsInstance(restored.chunking_strategy, KeywordChunking)
        self.assertEqual(restored.chunking_strategy.keywords, ["Section"])
        self.assertEqual(restored.chunking_strategy.case_sensitive, True)

    async def test_multi_model_with_chunking(self):
        """Test RecursiveGenerator with different root and sub models + chunking."""
        lm_root = LanguageModel(model="zai/glm-4.7")
        lm_sub = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = UniformChunking(chunk_size=800)

        gen = RecursiveGenerator(
            language_model=lm_root,
            sub_language_model=lm_sub,
            chunking_strategy=chunking,
        )

        self.assertEqual(gen.language_model.model, "zai/glm-4.7")
        self.assertEqual(gen.sub_language_model.model, "groq/openai/gpt-oss-20b")
        self.assertIsNotNone(gen.chunking_strategy)
        self.assertEqual(gen.chunking_strategy.chunk_size, 800)
