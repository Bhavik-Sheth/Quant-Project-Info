"""
Embedding Module

Provides free, open-source embedding generation using HuggingFace sentence-transformers.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """
    Free embedding generator using HuggingFace sentence-transformers.
    
    Default model: all-MiniLM-L6-v2 (384 dimensions, fast and efficient)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array of embeddings
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Cosine similarity
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim


# Default global embedder instance
_default_embedder: SentenceTransformerEmbedder = None


def get_embedder(model_name: str = None) -> SentenceTransformerEmbedder:
    """
    Get or create the default embedder instance.
    
    Args:
        model_name: Optional model name (uses config default if None)
        
    Returns:
        SentenceTransformerEmbedder instance
    """
    global _default_embedder
    
    if _default_embedder is None:
        from RAG_Mentor.config import Config
        model = model_name or Config.EMBEDDING_MODEL
        _default_embedder = SentenceTransformerEmbedder(model)
    
    return _default_embedder
