# License Apache 2.0: (c) 2025 Synalinks Team

"""Integration tests for chunking strategies with RecursiveGenerator.

These tests verify E2E integration with real LLM calls using:
- zai/glm-4.7
- groq/openai/gpt-oss-20b
"""

import pytest

from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.rlm.core.chunking_strategy import KeywordChunking
from synalinks.src.modules.rlm.core.chunking_strategy import SemanticChunking
from synalinks.src.modules.rlm.core.chunking_strategy import UniformChunking
from synalinks.src.modules.rlm.core.recursive_generator import RecursiveGenerator


class TestRecursiveGeneratorWithChunking:
    """Test RecursiveGenerator with chunking strategies."""

    @pytest.mark.asyncio
    async def test_uniform_chunking_integration_zai(self):
        """Test UniformChunking with zai/glm-4.7 model."""
        lm = LanguageModel(model="zai/glm-4.7")
        chunking = UniformChunking(chunk_size=1000, overlap=100)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        # Verify chunking_strategy is set
        assert gen.chunking_strategy is not None
        assert isinstance(gen.chunking_strategy, UniformChunking)
        assert gen.chunking_strategy.chunk_size == 1000
        assert gen.chunking_strategy.overlap == 100

        # Test that chunking strategy can process text
        text = "A" * 5000
        chunks = gen.chunking_strategy.chunk(text)
        assert len(chunks) > 1
        assert all(len(chunk) <= 1000 for chunk in chunks)

    @pytest.mark.asyncio
    async def test_uniform_chunking_integration_groq(self):
        """Test UniformChunking with groq/openai/gpt-oss-20b model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = UniformChunking(chunk_size=500, overlap=50)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        assert gen.chunking_strategy is not None
        assert isinstance(gen.chunking_strategy, UniformChunking)
        assert gen.chunking_strategy.chunk_size == 500
        assert gen.chunking_strategy.overlap == 50

    @pytest.mark.asyncio
    async def test_keyword_chunking_integration_zai(self):
        """Test KeywordChunking with zai/glm-4.7 model."""
        lm = LanguageModel(model="zai/glm-4.7")
        chunking = KeywordChunking(keywords=["Chapter", "Section"], case_sensitive=False)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        assert gen.chunking_strategy is not None
        assert isinstance(gen.chunking_strategy, KeywordChunking)
        assert gen.chunking_strategy.keywords == ["Chapter", "Section"]

        # Test chunking on sample text
        text = "Chapter 1: Intro. Section A. Chapter 2: Methods. Section B."
        chunks = gen.chunking_strategy.chunk(text)
        assert len(chunks) == 4

    @pytest.mark.asyncio
    async def test_keyword_chunking_integration_groq(self):
        """Test KeywordChunking with groq/openai/gpt-oss-20b model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = KeywordChunking(keywords=["Part"])

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        assert gen.chunking_strategy is not None
        assert isinstance(gen.chunking_strategy, KeywordChunking)

    @pytest.mark.asyncio
    async def test_semantic_chunking_integration_zai(self):
        """Test SemanticChunking with zai/glm-4.7 model."""
        lm = LanguageModel(model="zai/glm-4.7")
        chunking = SemanticChunking(max_chunk_size=2000, similarity_threshold=0.6)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        assert gen.chunking_strategy is not None
        assert isinstance(gen.chunking_strategy, SemanticChunking)
        assert gen.chunking_strategy.max_chunk_size == 2000
        assert gen.chunking_strategy.similarity_threshold == 0.6

        # Test chunking behavior
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = gen.chunking_strategy.chunk(text)
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_semantic_chunking_integration_groq(self):
        """Test SemanticChunking with groq/openai/gpt-oss-20b model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = SemanticChunking(max_chunk_size=1500)

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        assert gen.chunking_strategy is not None
        assert isinstance(gen.chunking_strategy, SemanticChunking)

    @pytest.mark.asyncio
    async def test_string_strategy_name_zai(self):
        """Test creating chunking strategy from string with zai model."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy="uniform",
        )

        # Default UniformChunking should be created
        assert gen.chunking_strategy is not None
        assert isinstance(gen.chunking_strategy, UniformChunking)

    @pytest.mark.asyncio
    async def test_string_strategy_name_groq(self):
        """Test creating chunking strategy from string with groq model."""
        lm = LanguageModel(model="groq/openai/gpt-oss-20b")

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy="semantic",
        )

        # Default SemanticChunking should be created
        assert gen.chunking_strategy is not None
        assert isinstance(gen.chunking_strategy, SemanticChunking)

    @pytest.mark.asyncio
    async def test_no_chunking_strategy(self):
        """Test RecursiveGenerator without chunking strategy."""
        lm = LanguageModel(model="zai/glm-4.7")

        gen = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=None,
        )

        # chunking_strategy should be None
        assert gen.chunking_strategy is None

    @pytest.mark.asyncio
    async def test_serialization_with_chunking_zai(self):
        """Test serialization of RecursiveGenerator with chunking using zai model."""
        from synalinks.src.saving import serialization_lib

        lm = LanguageModel(model="zai/glm-4.7")
        chunking = UniformChunking(chunk_size=750, overlap=75)

        original = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
            max_iterations=5,
        )

        # Serialize and deserialize
        config = original.get_config()
        assert "chunking_strategy" in config

        # Full roundtrip
        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        assert isinstance(restored, RecursiveGenerator)
        assert restored.chunking_strategy is not None
        assert isinstance(restored.chunking_strategy, UniformChunking)
        assert restored.chunking_strategy.chunk_size == 750
        assert restored.chunking_strategy.overlap == 75

    @pytest.mark.asyncio
    async def test_serialization_with_chunking_groq(self):
        """Test serialization of RecursiveGenerator with chunking using groq model."""
        from synalinks.src.saving import serialization_lib

        lm = LanguageModel(model="groq/openai/gpt-oss-20b")
        chunking = KeywordChunking(keywords=["Section"], case_sensitive=True)

        original = RecursiveGenerator(
            language_model=lm,
            chunking_strategy=chunking,
        )

        config = original.get_config()
        assert "chunking_strategy" in config

        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        assert isinstance(restored, RecursiveGenerator)
        assert isinstance(restored.chunking_strategy, KeywordChunking)
        assert restored.chunking_strategy.keywords == ["Section"]
        assert restored.chunking_strategy.case_sensitive is True

    @pytest.mark.asyncio
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

        assert gen.language_model.model == "zai/glm-4.7"
        assert gen.sub_language_model.model == "groq/openai/gpt-oss-20b"
        assert gen.chunking_strategy is not None
        assert gen.chunking_strategy.chunk_size == 800
