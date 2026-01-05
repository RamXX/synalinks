# License Apache 2.0: (c) 2025 Synalinks Team

"""Unit tests for chunking strategies."""

import pytest

from synalinks.src.modules.rlm.core.chunking_strategy import KeywordChunking
from synalinks.src.modules.rlm.core.chunking_strategy import SemanticChunking
from synalinks.src.modules.rlm.core.chunking_strategy import UniformChunking
from synalinks.src.modules.rlm.core.chunking_strategy import get_chunking_strategy
from synalinks.src.saving import serialization_lib


class TestUniformChunking:
    """Test UniformChunking strategy."""

    def test_basic_chunking(self):
        """Test basic uniform chunking without overlap."""
        chunking = UniformChunking(chunk_size=10, overlap=0)
        text = "0123456789" * 3  # 30 chars
        chunks = chunking.chunk(text)

        assert len(chunks) == 3
        assert chunks[0] == "0123456789"
        assert chunks[1] == "0123456789"
        assert chunks[2] == "0123456789"

    def test_chunking_with_overlap(self):
        """Test uniform chunking with overlap."""
        chunking = UniformChunking(chunk_size=10, overlap=3)
        text = "0123456789ABCDEFGHIJ"  # 20 chars

        chunks = chunking.chunk(text)

        # First chunk: 0-10
        assert chunks[0] == "0123456789"
        # Second chunk: 7-17 (10-3=7)
        assert chunks[1] == "789ABCDEFG"
        # Third chunk: 14-20 (17-3=14)
        assert chunks[2] == "EFGHIJ"

    def test_empty_text(self):
        """Test chunking empty text."""
        chunking = UniformChunking(chunk_size=10)
        chunks = chunking.chunk("")

        assert chunks == []

    def test_text_shorter_than_chunk_size(self):
        """Test text shorter than chunk size."""
        chunking = UniformChunking(chunk_size=100)
        text = "Short text"
        chunks = chunking.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_invalid_chunk_size(self):
        """Test invalid chunk_size parameter."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            UniformChunking(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            UniformChunking(chunk_size=-1)

    def test_invalid_overlap(self):
        """Test invalid overlap parameter."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            UniformChunking(chunk_size=10, overlap=-1)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            UniformChunking(chunk_size=10, overlap=10)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            UniformChunking(chunk_size=10, overlap=15)

    def test_serialization(self):
        """Test serialization and deserialization."""
        original = UniformChunking(chunk_size=500, overlap=50)
        config = original.get_config()

        assert config["chunk_size"] == 500
        assert config["overlap"] == 50
        assert config["class_name"] == "UniformChunking"

        restored = UniformChunking.from_config(config)
        assert restored.chunk_size == 500
        assert restored.overlap == 50

    def test_full_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        original = UniformChunking(chunk_size=200, overlap=20)
        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        assert isinstance(restored, UniformChunking)
        assert restored.chunk_size == 200
        assert restored.overlap == 20

        # Test that chunking behavior is preserved
        text = "A" * 500
        assert original.chunk(text) == restored.chunk(text)


class TestKeywordChunking:
    """Test KeywordChunking strategy."""

    def test_basic_keyword_chunking(self):
        """Test basic keyword chunking."""
        chunking = KeywordChunking(keywords=["Chapter"])
        text = "Chapter 1: Introduction. Chapter 2: Methods. Chapter 3: Results."

        chunks = chunking.chunk(text)

        assert len(chunks) == 3
        assert chunks[0] == "Chapter 1: Introduction. "
        assert chunks[1] == "Chapter 2: Methods. "
        assert chunks[2] == "Chapter 3: Results."

    def test_case_insensitive(self):
        """Test case-insensitive keyword matching."""
        chunking = KeywordChunking(keywords=["chapter"], case_sensitive=False)
        text = "Chapter 1. CHAPTER 2. chapter 3."

        chunks = chunking.chunk(text)

        assert len(chunks) == 3
        assert "Chapter 1" in chunks[0]
        assert "CHAPTER 2" in chunks[1]
        assert "chapter 3" in chunks[2]

    def test_case_sensitive(self):
        """Test case-sensitive keyword matching."""
        chunking = KeywordChunking(keywords=["Chapter"], case_sensitive=True)
        text = "Chapter 1. CHAPTER 2. chapter 3."

        chunks = chunking.chunk(text)

        # Only "Chapter" should match, not "CHAPTER" or "chapter"
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_keywords(self):
        """Test chunking with multiple keywords."""
        chunking = KeywordChunking(keywords=["Chapter", "Section"])
        text = "Chapter 1. Section A. Chapter 2. Section B."

        chunks = chunking.chunk(text)

        assert len(chunks) == 4
        assert "Chapter 1" in chunks[0]
        assert "Section A" in chunks[1]
        assert "Chapter 2" in chunks[2]
        assert "Section B" in chunks[3]

    def test_empty_text(self):
        """Test chunking empty text."""
        chunking = KeywordChunking(keywords=["Chapter"])
        chunks = chunking.chunk("")

        assert chunks == []

    def test_no_keywords_found(self):
        """Test text with no keyword matches."""
        chunking = KeywordChunking(keywords=["Chapter"])
        text = "This is some text without the keyword."

        chunks = chunking.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_keywords_list(self):
        """Test initialization with empty keywords list."""
        with pytest.raises(ValueError, match="keywords list cannot be empty"):
            KeywordChunking(keywords=[])

    def test_serialization(self):
        """Test serialization and deserialization."""
        original = KeywordChunking(keywords=["Chapter", "Section"], case_sensitive=True)
        config = original.get_config()

        assert config["keywords"] == ["Chapter", "Section"]
        assert config["case_sensitive"] is True
        assert config["class_name"] == "KeywordChunking"

        restored = KeywordChunking.from_config(config)
        assert restored.keywords == ["Chapter", "Section"]
        assert restored.case_sensitive is True

    def test_full_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        original = KeywordChunking(keywords=["Part"], case_sensitive=False)
        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        assert isinstance(restored, KeywordChunking)
        assert restored.keywords == ["Part"]
        assert restored.case_sensitive is False

        # Test that chunking behavior is preserved
        text = "Part 1. Part 2. Part 3."
        assert original.chunk(text) == restored.chunk(text)


class TestSemanticChunking:
    """Test SemanticChunking strategy."""

    def test_fallback_chunking_no_model(self):
        """Test fallback chunking when no embedding model is provided."""
        chunking = SemanticChunking(embedding_model=None, max_chunk_size=50)
        text = "First sentence. Second sentence. Third sentence."

        chunks = chunking.chunk(text)

        # Should use sentence-based fallback
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_sentence_splitting(self):
        """Test sentence-based splitting in fallback mode."""
        chunking = SemanticChunking(max_chunk_size=100)
        text = "This is sentence one. This is sentence two! And sentence three?"

        chunks = chunking.chunk(text)

        # Should split on sentence boundaries
        assert len(chunks) >= 1
        # Verify all text is preserved
        assert "".join(chunks).replace(" ", "") == text.replace(" ", "")

    def test_max_chunk_size_respected(self):
        """Test that max_chunk_size is respected."""
        chunking = SemanticChunking(max_chunk_size=20)
        # Create a long sentence
        text = "This is a very long sentence that exceeds the maximum chunk size."

        chunks = chunking.chunk(text)

        # Each chunk should be <= max_chunk_size
        for chunk in chunks:
            assert len(chunk) <= 20

    def test_empty_text(self):
        """Test chunking empty text."""
        chunking = SemanticChunking()
        chunks = chunking.chunk("")

        assert chunks == []

    def test_invalid_similarity_threshold(self):
        """Test invalid similarity_threshold parameter."""
        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            SemanticChunking(similarity_threshold=-0.1)

        with pytest.raises(ValueError, match="similarity_threshold must be between"):
            SemanticChunking(similarity_threshold=1.5)

    def test_invalid_max_chunk_size(self):
        """Test invalid max_chunk_size parameter."""
        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            SemanticChunking(max_chunk_size=0)

        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            SemanticChunking(max_chunk_size=-100)

    def test_serialization_without_model(self):
        """Test serialization without embedding model."""
        original = SemanticChunking(
            similarity_threshold=0.7,
            max_chunk_size=1500,
        )
        config = original.get_config()

        assert config["similarity_threshold"] == 0.7
        assert config["max_chunk_size"] == 1500
        assert "embedding_model" not in config
        assert config["class_name"] == "SemanticChunking"

        restored = SemanticChunking.from_config(config)
        assert restored.similarity_threshold == 0.7
        assert restored.max_chunk_size == 1500
        assert restored.embedding_model is None

    def test_full_serialization_roundtrip_no_model(self):
        """Test full serialization roundtrip without embedding model."""
        original = SemanticChunking(similarity_threshold=0.6, max_chunk_size=1000)
        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        assert isinstance(restored, SemanticChunking)
        assert restored.similarity_threshold == 0.6
        assert restored.max_chunk_size == 1000
        assert restored.embedding_model is None

        # Test that chunking behavior is preserved
        text = "Sentence one. Sentence two. Sentence three."
        assert original.chunk(text) == restored.chunk(text)


class TestGetChunkingStrategy:
    """Test get_chunking_strategy factory function."""

    def test_get_uniform_from_string(self):
        """Test creating UniformChunking from string."""
        strategy = get_chunking_strategy("uniform", chunk_size=500, overlap=50)

        assert isinstance(strategy, UniformChunking)
        assert strategy.chunk_size == 500
        assert strategy.overlap == 50

    def test_get_keyword_from_string(self):
        """Test creating KeywordChunking from string."""
        strategy = get_chunking_strategy(
            "keyword", keywords=["Part"], case_sensitive=True
        )

        assert isinstance(strategy, KeywordChunking)
        assert strategy.keywords == ["Part"]
        assert strategy.case_sensitive is True

    def test_get_semantic_from_string(self):
        """Test creating SemanticChunking from string."""
        strategy = get_chunking_strategy("semantic", max_chunk_size=1000)

        assert isinstance(strategy, SemanticChunking)
        assert strategy.max_chunk_size == 1000

    def test_case_insensitive_string(self):
        """Test that string names are case-insensitive."""
        strategy1 = get_chunking_strategy("UNIFORM", chunk_size=100)
        strategy2 = get_chunking_strategy("Uniform", chunk_size=100)
        strategy3 = get_chunking_strategy("uniform", chunk_size=100)

        assert isinstance(strategy1, UniformChunking)
        assert isinstance(strategy2, UniformChunking)
        assert isinstance(strategy3, UniformChunking)

    def test_pass_through_instance(self):
        """Test that passing an instance returns it unchanged."""
        original = UniformChunking(chunk_size=200)
        result = get_chunking_strategy(original)

        assert result is original

    def test_unknown_strategy_name(self):
        """Test error on unknown strategy name."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunking_strategy("unknown_strategy")

    def test_available_strategies_in_error(self):
        """Test that error message lists available strategies."""
        with pytest.raises(ValueError, match="uniform.*keyword.*semantic"):
            get_chunking_strategy("invalid")
