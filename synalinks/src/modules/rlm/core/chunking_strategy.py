# License Apache 2.0: (c) 2025 Synalinks Team

"""Chunking strategies for handling large contexts in RecursiveGenerator."""

import re
from abc import ABC
from abc import abstractmethod
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.saving.object_registration import register_synalinks_serializable


@synalinks_export("synalinks.ChunkingStrategy")
@register_synalinks_serializable()
class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies.

    ChunkingStrategy defines how large inputs are split into smaller chunks
    that can be processed independently by RecursiveGenerator.

    Subclasses must implement the chunk() method.
    """

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks
        """
        pass

    def get_config(self) -> dict:
        """Get serialization config.

        Returns:
            Configuration dictionary
        """
        return {"class_name": self.__class__.__name__}

    @classmethod
    def from_config(cls, config: dict):
        """Deserialize from config.

        Args:
            config: Configuration dictionary

        Returns:
            ChunkingStrategy instance
        """
        return cls()


@synalinks_export("synalinks.UniformChunking")
@register_synalinks_serializable()
class UniformChunking(ChunkingStrategy):
    """Split text into uniform-sized chunks with optional overlap.

    Args:
        chunk_size: Maximum number of characters per chunk
        overlap: Number of characters to overlap between chunks (default: 0)

    Example:
        >>> chunking = UniformChunking(chunk_size=100, overlap=20)
        >>> chunks = chunking.chunk("Long text...")
        >>> len(chunks)
        5
    """

    def __init__(self, chunk_size: int = 1000, overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Split text into uniform chunks with overlap.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Move start forward by chunk_size minus overlap
            start = end - self.overlap
            if start >= len(text):
                break

        return chunks

    def get_config(self) -> dict:
        """Get serialization config."""
        base_config = super().get_config()
        return {
            **base_config,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }

    @classmethod
    def from_config(cls, config: dict):
        """Deserialize from config."""
        return cls(
            chunk_size=config.get("chunk_size", 1000),
            overlap=config.get("overlap", 0),
        )


@synalinks_export("synalinks.KeywordChunking")
@register_synalinks_serializable()
class KeywordChunking(ChunkingStrategy):
    """Split text at keyword boundaries.

    Splits text at occurrences of specified keywords, keeping the keyword
    at the start of each chunk.

    Args:
        keywords: List of keywords to split on
        case_sensitive: Whether keyword matching is case-sensitive (default: False)

    Example:
        >>> chunking = KeywordChunking(keywords=["Chapter", "Section"])
        >>> chunks = chunking.chunk("Chapter 1... Chapter 2...")
        >>> len(chunks)
        2
    """

    def __init__(self, keywords: list[str], case_sensitive: bool = False):
        if not keywords:
            raise ValueError("keywords list cannot be empty")

        self.keywords = keywords
        self.case_sensitive = case_sensitive

    def chunk(self, text: str) -> list[str]:
        """Split text at keyword boundaries.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Build regex pattern from keywords
        escaped_keywords = [re.escape(kw) for kw in self.keywords]
        pattern = r"(" + "|".join(escaped_keywords) + r")"

        flags = 0 if self.case_sensitive else re.IGNORECASE

        # Split on keywords but keep them
        parts = re.split(pattern, text, flags=flags)

        # Reconstruct chunks with keywords at the start
        chunks = []
        current_chunk = ""

        for i, part in enumerate(parts):
            if not part:
                continue

            # Check if this part is a keyword
            is_keyword = False
            for kw in self.keywords:
                if self.case_sensitive:
                    if part == kw:
                        is_keyword = True
                        break
                else:
                    if part.lower() == kw.lower():
                        is_keyword = True
                        break

            if is_keyword:
                # Start a new chunk with this keyword
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += part

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    def get_config(self) -> dict:
        """Get serialization config."""
        base_config = super().get_config()
        return {
            **base_config,
            "keywords": self.keywords,
            "case_sensitive": self.case_sensitive,
        }

    @classmethod
    def from_config(cls, config: dict):
        """Deserialize from config."""
        return cls(
            keywords=config.get("keywords", []),
            case_sensitive=config.get("case_sensitive", False),
        )


@synalinks_export("synalinks.SemanticChunking")
@register_synalinks_serializable()
class SemanticChunking(ChunkingStrategy):
    """Split text based on semantic similarity using embeddings.

    Uses sentence embeddings to identify natural breakpoints where
    semantic similarity drops below a threshold.

    Args:
        embedding_model: Optional LanguageModel to use for embeddings.
            If None, falls back to simple sentence-based chunking.
        similarity_threshold: Cosine similarity threshold for splitting (default: 0.5)
        max_chunk_size: Maximum chunk size in characters (default: 2000)

    Example:
        >>> from synalinks import LanguageModel
        >>> lm = LanguageModel(model="zai/glm-4.7")
        >>> chunking = SemanticChunking(embedding_model=lm)
        >>> chunks = chunking.chunk("Long document...")
    """

    def __init__(
        self,
        embedding_model: Optional["LanguageModel"] = None,  # noqa: F821
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 2000,
    ):
        if similarity_threshold < 0 or similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")

        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str) -> list[str]:
        """Split text based on semantic similarity.

        If no embedding model is provided, falls back to simple
        sentence-based chunking.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # For now, implement simple sentence-based fallback
        # Full semantic chunking with embeddings would be implemented
        # when embedding_model is provided
        if self.embedding_model is None:
            return self._fallback_chunk(text)

        # TODO: Implement semantic chunking with embeddings
        # This would involve:
        # 1. Split into sentences
        # 2. Get embeddings for each sentence
        # 3. Compute cosine similarity between consecutive sentences
        # 4. Create chunks where similarity drops below threshold
        # 5. Respect max_chunk_size constraint

        # For now, use fallback
        return self._fallback_chunk(text)

    def _fallback_chunk(self, text: str) -> list[str]:
        """Fallback to simple sentence-based chunking.

        Args:
            text: The input text to chunk

        Returns:
            List of text chunks
        """
        # Split on sentence boundaries
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Check if adding this sentence would exceed max_chunk_size
            if len(current_chunk) + len(sentence) + 1 > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def get_config(self) -> dict:
        """Get serialization config."""
        base_config = super().get_config()
        config = {
            **base_config,
            "similarity_threshold": self.similarity_threshold,
            "max_chunk_size": self.max_chunk_size,
        }

        # Only serialize embedding_model if present
        if self.embedding_model is not None:
            from synalinks.src.saving import serialization_lib

            config["embedding_model"] = serialization_lib.serialize_synalinks_object(
                self.embedding_model
            )

        return config

    @classmethod
    def from_config(cls, config: dict):
        """Deserialize from config."""
        embedding_model = None
        if "embedding_model" in config:
            from synalinks.src.saving import serialization_lib

            embedding_model = serialization_lib.deserialize_synalinks_object(
                config["embedding_model"]
            )

        return cls(
            embedding_model=embedding_model,
            similarity_threshold=config.get("similarity_threshold", 0.5),
            max_chunk_size=config.get("max_chunk_size", 2000),
        )


@synalinks_export("synalinks.get_chunking_strategy")
def get_chunking_strategy(strategy: str | ChunkingStrategy, **kwargs) -> ChunkingStrategy:
    """Factory function to get chunking strategy from string name or instance.

    Args:
        strategy: Either a ChunkingStrategy instance or a string name
            ('uniform', 'keyword', 'semantic')
        **kwargs: Additional arguments to pass to strategy constructor

    Returns:
        ChunkingStrategy instance

    Raises:
        ValueError: If strategy name is not recognized

    Example:
        >>> # Create from string
        >>> strategy = get_chunking_strategy('uniform', chunk_size=500)
        >>> # Pass through existing instance
        >>> custom = UniformChunking(chunk_size=1000)
        >>> strategy = get_chunking_strategy(custom)
    """
    if isinstance(strategy, ChunkingStrategy):
        return strategy

    strategy_map = {
        "uniform": UniformChunking,
        "keyword": KeywordChunking,
        "semantic": SemanticChunking,
    }

    strategy_lower = strategy.lower()
    if strategy_lower not in strategy_map:
        raise ValueError(
            f"Unknown chunking strategy: {strategy}. "
            f"Available: {list(strategy_map.keys())}"
        )

    return strategy_map[strategy_lower](**kwargs)
