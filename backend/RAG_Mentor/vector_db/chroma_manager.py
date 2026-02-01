"""
ChromaDB Manager

Manages vector database operations using ChromaDB for semantic search.
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

from RAG_Mentor.config import Config
from RAG_Mentor.vector_db.embeddings import get_embedder

logger = logging.getLogger(__name__)


class ChromaManager:
    """
    ChromaDB vector database manager for the RAG system.
    
    Manages three collections:
    - trading_principles: Expert trading wisdom
    - news_articles: Market news with timestamps
    - backtest_results: Historical backtest data
    """
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Path for persistent storage
        """
        self.persist_directory = persist_directory or Config.CHROMADB_PATH
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedder
        self.embedder = get_embedder()
        
        # Create or get collections
        self.principles_collection = self._get_or_create_collection(
            Config.COLLECTION_PRINCIPLES
        )
        self.news_collection = self._get_or_create_collection(
            Config.COLLECTION_NEWS
        )
        self.backtest_collection = self._get_or_create_collection(
            Config.COLLECTION_BACKTEST
        )
        
        logger.info(f"ChromaDB initialized at {self.persist_directory}")
    
    def _get_or_create_collection(self, name: str):
        """
        Get or create a ChromaDB collection.
        
        Args:
            name: Collection name
            
        Returns:
            ChromaDB collection
        """
        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{name}' ready (count: {collection.count()})")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            raise
    
    def add_trading_principle(
        self,
        principle: str,
        explanation: str,
        author: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a trading principle to the vector database.
        
        Args:
            principle: The principle statement
            explanation: Detailed explanation
            author: Original author (e.g., "Jesse Livermore")
            category: Category (e.g., "risk_management", "trend_following")
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        
        # Combine principle and explanation for embedding
        full_text = f"{principle}\n\n{explanation}"
        
        # Prepare metadata
        meta = {
            "author": author,
            "category": category,
            "principle": principle,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            meta.update(metadata)
        
        # Generate embedding
        embedding = self.embedder.embed_text(full_text).tolist()
        
        # Add to collection
        self.principles_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[full_text],
            metadatas=[meta]
        )
        
        logger.debug(f"Added principle: {principle[:50]}...")
        return doc_id
    
    def add_news_article(
        self,
        headline: str,
        summary: str,
        timestamp: datetime,
        symbols: List[str],
        source: str = "Unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a news article to the vector database.
        
        Args:
            headline: Article headline
            summary: Article summary
            timestamp: Publication timestamp
            symbols: Related stock symbols
            source: News source
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        
        # Combine headline and summary
        full_text = f"{headline}\n\n{summary}"
        
        # Prepare metadata
        meta = {
            "headline": headline,
            "timestamp": timestamp.isoformat(),
            "symbols": ",".join(symbols),
            "source": source
        }
        if metadata:
            meta.update(metadata)
        
        # Generate embedding
        embedding = self.embedder.embed_text(full_text).tolist()
        
        # Add to collection
        self.news_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[full_text],
            metadatas=[meta]
        )
        
        logger.debug(f"Added news: {headline[:50]}...")
        return doc_id
    
    def add_backtest_result(
        self,
        strategy_name: str,
        description: str,
        performance_summary: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add backtest results to the vector database.
        
        Args:
            strategy_name: Name of strategy
            description: Strategy description
            performance_summary: Performance metrics
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        doc_id = str(uuid.uuid4())
        
        # Create searchable text
        full_text = f"{strategy_name}\n\n{description}\n\nPerformance: {str(performance_summary)}"
        
        # Prepare metadata
        meta = {
            "strategy_name": strategy_name,
            "timestamp": datetime.now().isoformat(),
            **performance_summary
        }
        if metadata:
            meta.update(metadata)
        
        # Generate embedding
        embedding = self.embedder.embed_text(full_text).tolist()
        
        # Add to collection
        self.backtest_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[full_text],
            metadatas=[meta]
        )
        
        logger.debug(f"Added backtest: {strategy_name}")
        return doc_id
    
    def search_principles(
        self,
        query: str,
        top_k: int = None,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant trading principles.
        
        Args:
            query: Search query
            top_k: Number of results
            category_filter: Optional category filter
            
        Returns:
            Search results with documents and metadata
        """
        top_k = top_k or Config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query).tolist()
        
        # Build filter
        where_filter = {"category": category_filter} if category_filter else None
        
        # Search
        results = self.principles_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        
        return results
    
    def search_news(
        self,
        query: str,
        top_k: int = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Search for relevant news articles.
        
        Args:
            query: Search query
            top_k: Number of results
            start_date: Filter start date
            end_date: Filter end date
            symbols: Filter by symbols
            
        Returns:
            Search results
        """
        top_k = top_k or Config.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query).tolist()
        
        # Search
        results = self.news_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Post-filter by date and symbols if needed
        # (ChromaDB's where clause has limitations, so we filter after retrieval)
        if start_date or end_date or symbols:
            results = self._filter_news_results(results, start_date, end_date, symbols)
        
        return results
    
    def _filter_news_results(
        self,
        results: Dict[str, Any],
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        symbols: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Filter news results by date and symbols."""
        if not results.get('metadatas'):
            return results
        
        filtered_indices = []
        for i, meta in enumerate(results['metadatas'][0]):
            # Date filter
            if start_date or end_date:
                timestamp = datetime.fromisoformat(meta['timestamp'])
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
            
            # Symbol filter
            if symbols:
                article_symbols = meta.get('symbols', '').split(',')
                if not any(s in article_symbols for s in symbols):
                    continue
            
            filtered_indices.append(i)
        
        # Rebuild results with filtered indices
        return {
            'ids': [[results['ids'][0][i] for i in filtered_indices]],
            'documents': [[results['documents'][0][i] for i in filtered_indices]],
            'metadatas': [[results['metadatas'][0][i] for i in filtered_indices]],
            'distances': [[results['distances'][0][i] for i in filtered_indices]]
        }
    
    def reset_collection(self, collection_name: str) -> None:
        """
        Delete and recreate a collection.
        
        Args:
            collection_name: Name of collection to reset
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            
            # Recreate
            if collection_name == Config.COLLECTION_PRINCIPLES:
                self.principles_collection = self._get_or_create_collection(collection_name)
            elif collection_name == Config.COLLECTION_NEWS:
                self.news_collection = self._get_or_create_collection(collection_name)
            elif collection_name == Config.COLLECTION_BACKTEST:
                self.backtest_collection = self._get_or_create_collection(collection_name)
                
        except Exception as e:
            logger.error(f"Error resetting collection {collection_name}: {e}")
    
    def get_collection_stats(self) -> Dict[str, int]:
        """
        Get statistics about all collections.
        
        Returns:
            Dictionary with collection counts
        """
        return {
            "trading_principles": self.principles_collection.count(),
            "news_articles": self.news_collection.count(),
            "backtest_results": self.backtest_collection.count()
        }
