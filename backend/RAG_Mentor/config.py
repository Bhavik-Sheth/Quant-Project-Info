"""
Configuration Module for RAG Trading Mentor

Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for RAG Trading Mentor"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    RAG_DIR = BASE_DIR / "RAG_Mentor"
    
    # LLM API Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
    
    # ChromaDB Configuration
    CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", str(RAG_DIR / "chroma_db"))
    
    # Embedding Model (Free HuggingFace)
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # LLM Models
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-pro")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
    # RAG Configuration
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # ChromaDB Collections
    COLLECTION_PRINCIPLES = "trading_principles"
    COLLECTION_NEWS = "news_articles"
    COLLECTION_BACKTEST = "backtest_results"
    
    # News API (Optional)
    NEWS_API_KEY: Optional[str] = os.getenv("NEWS_API_KEY")
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required API keys are present.
        
        Returns:
            bool: True if configuration is valid
        """
        if not cls.GEMINI_API_KEY and not cls.GROQ_API_KEY:
            raise ValueError(
                "At least one LLM API key (GEMINI_API_KEY or GROQ_API_KEY) must be set"
            )
        
        # Create ChromaDB directory if it doesn't exist
        Path(cls.CHROMADB_PATH).mkdir(parents=True, exist_ok=True)
        
        return True
    
    @classmethod
    def get_available_llm(cls) -> str:
        """
        Get the primary available LLM.
        
        Returns:
            str: 'gemini' or 'groq'
        """
        if cls.GEMINI_API_KEY:
            return "gemini"
        elif cls.GROQ_API_KEY:
            return "groq"
        else:
            raise ValueError("No LLM API key configured")


# Validate configuration on import
Config.validate()
