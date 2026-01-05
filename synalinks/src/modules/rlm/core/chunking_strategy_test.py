# License Apache 2.0: (c) 2025 Synalinks Team

"""Unit tests for chunking strategies."""

from synalinks.src import testing
from synalinks.src.modules.rlm.core.chunking_strategy import KeywordChunking
from synalinks.src.modules.rlm.core.chunking_strategy import SemanticChunking
from synalinks.src.modules.rlm.core.chunking_strategy import UniformChunking
from synalinks.src.modules.rlm.core.chunking_strategy import get_chunking_strategy
from synalinks.src.saving import serialization_lib


class UniformChunkingTest(testing.TestCase):
    """Test UniformChunking strategy."""

    def test_basic_chunking(self):
        """Test basic uniform chunking without overlap."""
        chunking = UniformChunking(chunk_size=10, overlap=0)
        text = "0123456789" * 3  # 30 chars
        chunks = chunking.chunk(text)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "0123456789")
        self.assertEqual(chunks[1], "0123456789")
        self.assertEqual(chunks[2], "0123456789")

    def test_chunking_with_overlap(self):
        """Test uniform chunking with overlap."""
        chunking = UniformChunking(chunk_size=10, overlap=3)
        text = "0123456789ABCDEFGHIJ"  # 20 chars

        chunks = chunking.chunk(text)

        # First chunk: 0-10
        self.assertEqual(chunks[0], "0123456789")
        # Second chunk: 7-17 (10-3=7)
        self.assertEqual(chunks[1], "789ABCDEFG")
        # Third chunk: 14-20 (17-3=14)
        self.assertEqual(chunks[2], "EFGHIJ")

    def test_empty_text(self):
        """Test chunking empty text."""
        chunking = UniformChunking(chunk_size=10)
        chunks = chunking.chunk("")

        self.assertEqual(chunks, [])

    def test_text_shorter_than_chunk_size(self):
        """Test text shorter than chunk size."""
        chunking = UniformChunking(chunk_size=100)
        text = "Short text"
        chunks = chunking.chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_invalid_chunk_size(self):
        """Test invalid chunk_size parameter."""
        with self.assertRaisesRegex(ValueError, "chunk_size must be positive"):
            UniformChunking(chunk_size=0)

        with self.assertRaisesRegex(ValueError, "chunk_size must be positive"):
            UniformChunking(chunk_size=-1)

    def test_invalid_overlap(self):
        """Test invalid overlap parameter."""
        with self.assertRaisesRegex(ValueError, "overlap must be non-negative"):
            UniformChunking(chunk_size=10, overlap=-1)

        with self.assertRaisesRegex(ValueError, "overlap must be less than chunk_size"):
            UniformChunking(chunk_size=10, overlap=10)

        with self.assertRaisesRegex(ValueError, "overlap must be less than chunk_size"):
            UniformChunking(chunk_size=10, overlap=15)

    def test_serialization(self):
        """Test serialization and deserialization."""
        original = UniformChunking(chunk_size=500, overlap=50)
        config = original.get_config()

        self.assertEqual(config["chunk_size"], 500)
        self.assertEqual(config["overlap"], 50)
        self.assertEqual(config["class_name"], "UniformChunking")

        restored = UniformChunking.from_config(config)
        self.assertEqual(restored.chunk_size, 500)
        self.assertEqual(restored.overlap, 50)

    def test_full_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        original = UniformChunking(chunk_size=200, overlap=20)
        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        self.assertIsInstance(restored, UniformChunking)
        self.assertEqual(restored.chunk_size, 200)
        self.assertEqual(restored.overlap, 20)

        # Test that chunking behavior is preserved
        text = "A" * 500
        self.assertEqual(original.chunk(text), restored.chunk(text))


class KeywordChunkingTest(testing.TestCase):
    """Test KeywordChunking strategy."""

    def test_basic_keyword_chunking(self):
        """Test basic keyword chunking."""
        chunking = KeywordChunking(keywords=["Chapter"])
        text = "Chapter 1: Introduction. Chapter 2: Methods. Chapter 3: Results."

        chunks = chunking.chunk(text)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], "Chapter 1: Introduction. ")
        self.assertEqual(chunks[1], "Chapter 2: Methods. ")
        self.assertEqual(chunks[2], "Chapter 3: Results.")

    def test_case_insensitive(self):
        """Test case-insensitive keyword matching."""
        chunking = KeywordChunking(keywords=["chapter"], case_sensitive=False)
        text = "Chapter 1. CHAPTER 2. chapter 3."

        chunks = chunking.chunk(text)

        self.assertEqual(len(chunks), 3)
        self.assertIn("Chapter 1", chunks[0])
        self.assertIn("CHAPTER 2", chunks[1])
        self.assertIn("chapter 3", chunks[2])

    def test_case_sensitive(self):
        """Test case-sensitive keyword matching."""
        chunking = KeywordChunking(keywords=["Chapter"], case_sensitive=True)
        text = "Chapter 1. CHAPTER 2. chapter 3."

        chunks = chunking.chunk(text)

        # Only "Chapter" should match, not "CHAPTER" or "chapter"
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_multiple_keywords(self):
        """Test chunking with multiple keywords."""
        chunking = KeywordChunking(keywords=["Chapter", "Section"])
        text = "Chapter 1. Section A. Chapter 2. Section B."

        chunks = chunking.chunk(text)

        self.assertEqual(len(chunks), 4)
        self.assertIn("Chapter 1", chunks[0])
        self.assertIn("Section A", chunks[1])
        self.assertIn("Chapter 2", chunks[2])
        self.assertIn("Section B", chunks[3])

    def test_empty_text(self):
        """Test chunking empty text."""
        chunking = KeywordChunking(keywords=["Chapter"])
        chunks = chunking.chunk("")

        self.assertEqual(chunks, [])

    def test_no_keywords_found(self):
        """Test text with no keyword matches."""
        chunking = KeywordChunking(keywords=["Chapter"])
        text = "This is some text without the keyword."

        chunks = chunking.chunk(text)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_empty_keywords_list(self):
        """Test initialization with empty keywords list."""
        with self.assertRaisesRegex(ValueError, "keywords list cannot be empty"):
            KeywordChunking(keywords=[])

    def test_serialization(self):
        """Test serialization and deserialization."""
        original = KeywordChunking(keywords=["Chapter", "Section"], case_sensitive=True)
        config = original.get_config()

        self.assertEqual(config["keywords"], ["Chapter", "Section"])
        self.assertEqual(config["case_sensitive"], True)
        self.assertEqual(config["class_name"], "KeywordChunking")

        restored = KeywordChunking.from_config(config)
        self.assertEqual(restored.keywords, ["Chapter", "Section"])
        self.assertEqual(restored.case_sensitive, True)

    def test_full_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        original = KeywordChunking(keywords=["Part"], case_sensitive=False)
        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        self.assertIsInstance(restored, KeywordChunking)
        self.assertEqual(restored.keywords, ["Part"])
        self.assertEqual(restored.case_sensitive, False)

        # Test that chunking behavior is preserved
        text = "Part 1. Part 2. Part 3."
        self.assertEqual(original.chunk(text), restored.chunk(text))


class SemanticChunkingTest(testing.TestCase):
    """Test SemanticChunking strategy."""

    def test_fallback_chunking_no_model(self):
        """Test fallback chunking when no embedding model is provided."""
        chunking = SemanticChunking(embedding_model=None, max_chunk_size=50)
        text = "First sentence. Second sentence. Third sentence."

        chunks = chunking.chunk(text)

        # Should use sentence-based fallback
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 50)

    def test_sentence_splitting(self):
        """Test sentence-based splitting in fallback mode."""
        chunking = SemanticChunking(max_chunk_size=100)
        text = "This is sentence one. This is sentence two! And sentence three?"

        chunks = chunking.chunk(text)

        # Should split on sentence boundaries
        self.assertGreaterEqual(len(chunks), 1)
        # Verify all text is preserved
        self.assertEqual("".join(chunks).replace(" ", ""), text.replace(" ", ""))

    def test_max_chunk_size_respected(self):
        """Test that max_chunk_size is respected for multiple sentences."""
        chunking = SemanticChunking(max_chunk_size=30)
        # Create multiple sentences that can be chunked
        text = "First sentence here. Second sentence. Third one. Fourth sentence."

        chunks = chunking.chunk(text)

        # Each chunk should be <= max_chunk_size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)

        # Should create multiple chunks
        self.assertGreater(len(chunks), 1)

    def test_empty_text(self):
        """Test chunking empty text."""
        chunking = SemanticChunking()
        chunks = chunking.chunk("")

        self.assertEqual(chunks, [])

    def test_invalid_similarity_threshold(self):
        """Test invalid similarity_threshold parameter."""
        with self.assertRaisesRegex(ValueError, "similarity_threshold must be between"):
            SemanticChunking(similarity_threshold=-0.1)

        with self.assertRaisesRegex(ValueError, "similarity_threshold must be between"):
            SemanticChunking(similarity_threshold=1.5)

    def test_invalid_max_chunk_size(self):
        """Test invalid max_chunk_size parameter."""
        with self.assertRaisesRegex(ValueError, "max_chunk_size must be positive"):
            SemanticChunking(max_chunk_size=0)

        with self.assertRaisesRegex(ValueError, "max_chunk_size must be positive"):
            SemanticChunking(max_chunk_size=-100)

    def test_serialization_without_model(self):
        """Test serialization without embedding model."""
        original = SemanticChunking(
            similarity_threshold=0.7,
            max_chunk_size=1500,
        )
        config = original.get_config()

        self.assertEqual(config["similarity_threshold"], 0.7)
        self.assertEqual(config["max_chunk_size"], 1500)
        self.assertNotIn("embedding_model", config)
        self.assertEqual(config["class_name"], "SemanticChunking")

        restored = SemanticChunking.from_config(config)
        self.assertEqual(restored.similarity_threshold, 0.7)
        self.assertEqual(restored.max_chunk_size, 1500)
        self.assertIsNone(restored.embedding_model)

    def test_full_serialization_roundtrip_no_model(self):
        """Test full serialization roundtrip without embedding model."""
        original = SemanticChunking(similarity_threshold=0.6, max_chunk_size=1000)
        serialized = serialization_lib.serialize_synalinks_object(original)
        restored = serialization_lib.deserialize_synalinks_object(serialized)

        self.assertIsInstance(restored, SemanticChunking)
        self.assertEqual(restored.similarity_threshold, 0.6)
        self.assertEqual(restored.max_chunk_size, 1000)
        self.assertIsNone(restored.embedding_model)

        # Test that chunking behavior is preserved
        text = "Sentence one. Sentence two. Sentence three."
        self.assertEqual(original.chunk(text), restored.chunk(text))


class GetChunkingStrategyTest(testing.TestCase):
    """Test get_chunking_strategy factory function."""

    def test_get_uniform_from_string(self):
        """Test creating UniformChunking from string."""
        strategy = get_chunking_strategy("uniform", chunk_size=500, overlap=50)

        self.assertIsInstance(strategy, UniformChunking)
        self.assertEqual(strategy.chunk_size, 500)
        self.assertEqual(strategy.overlap, 50)

    def test_get_keyword_from_string(self):
        """Test creating KeywordChunking from string."""
        strategy = get_chunking_strategy(
            "keyword", keywords=["Part"], case_sensitive=True
        )

        self.assertIsInstance(strategy, KeywordChunking)
        self.assertEqual(strategy.keywords, ["Part"])
        self.assertEqual(strategy.case_sensitive, True)

    def test_get_semantic_from_string(self):
        """Test creating SemanticChunking from string."""
        strategy = get_chunking_strategy("semantic", max_chunk_size=1000)

        self.assertIsInstance(strategy, SemanticChunking)
        self.assertEqual(strategy.max_chunk_size, 1000)

    def test_case_insensitive_string(self):
        """Test that string names are case-insensitive."""
        strategy1 = get_chunking_strategy("UNIFORM", chunk_size=100)
        strategy2 = get_chunking_strategy("Uniform", chunk_size=100)
        strategy3 = get_chunking_strategy("uniform", chunk_size=100)

        self.assertIsInstance(strategy1, UniformChunking)
        self.assertIsInstance(strategy2, UniformChunking)
        self.assertIsInstance(strategy3, UniformChunking)

    def test_pass_through_instance(self):
        """Test that passing an instance returns it unchanged."""
        original = UniformChunking(chunk_size=200)
        result = get_chunking_strategy(original)

        self.assertIs(result, original)

    def test_unknown_strategy_name(self):
        """Test error on unknown strategy name."""
        with self.assertRaisesRegex(ValueError, "Unknown chunking strategy"):
            get_chunking_strategy("unknown_strategy")

    def test_available_strategies_in_error(self):
        """Test that error message lists available strategies."""
        with self.assertRaisesRegex(ValueError, "uniform.*keyword.*semantic"):
            get_chunking_strategy("invalid")
